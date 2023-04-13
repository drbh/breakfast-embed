//! This program provides a web server for updating and searching an
//! HNSW (Hierarchical Navigable Small World) map using Actix-web.
//! The map stores sentence embeddings as points in a high-dimensional space
//! and allows efficient nearest-neighbor search for similar sentences.

use instant_distance::{Builder, HnswMap, Search};
use parking_lot::Mutex;
use reqwest::header;
use reqwest::redirect::Policy;
use reqwest::Client;
use rusqlite::Result;
use serde_json::json;
use std::env;
use std::sync::Arc;

use crate::Request;
use crate::Point;
use crate::EmbedResponse;

/// Fetch the sentence embeddings for a list of sentences using OpenAI's
pub async fn create_openai_embedding(
    text_to_embed: &str,
) -> Result<EmbedResponse, Box<dyn std::error::Error>> {
    println!("Creating OpenAI embedding for: {}", text_to_embed);
    let mut headers = header::HeaderMap::new();
    headers.insert("Content-Type", "application/json".parse().unwrap());
    headers.insert(
        "Authorization",
        [
            "Bearer ",
            env::var("OPENAI_API_KEY")
                .unwrap_or("".to_string())
                .as_str(),
        ]
        .concat()
        .parse()
        .unwrap(),
    );

    let client = Client::builder().redirect(Policy::none()).build().unwrap();

    let body = json!({
    "input": text_to_embed,
    "model": "text-embedding-ada-002"
    })
    .to_string();

    let res = client
        .post("https://api.openai.com/v1/embeddings")
        .headers(headers)
        .body(body)
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let response_json: EmbedResponse = serde_json::from_str(&res).unwrap();

    Ok(response_json)
}

pub fn search_closest_points(
    arc_mutex_map: &Arc<Mutex<HnswMap<Point, String>>>,
    vector: &[f32],
    structured_request: &Request,
) -> Result<Vec<(String, f32)>, Box<dyn std::error::Error>> {
    let point = Point::from_slice(vector);
    let mut map = arc_mutex_map.lock();
    let mut _search = Search::default();
    if map.values.len() == 0 {
        println!(
            "Initializing map with {} points...",
            structured_request.sentences.len()
        );
        *map = Builder::default().build(
            structured_request
                .vectors
                .iter()
                .map(|vector| Point::from_slice(vector))
                .collect(),
            structured_request.sentences.clone(),
        );
    }

    let closest_points = {
        let mut closest_points_vec = Vec::new();
        for closest_point in map.search(&point, &mut _search).take(15) {
            closest_points_vec.push((closest_point.value.clone(), closest_point.distance));
        }
        closest_points_vec
    };

    Ok(closest_points)
}

pub fn insert_if_needed(
    arc_map: &Arc<Mutex<HnswMap<Point, String>>>,
    vector: &[f32],
    sentence: &str,
    closest_points: &[(String, f32)],
) -> String {
    let mut map = arc_map.lock();
    if closest_points[0].1 > 0.002 {
        map.insert(Point::from_slice(vector), sentence.to_string())
            .expect("insertion failed");
        "success".to_string()
    } else {
        println!("Closest point is too close, not inserting.");
        println!("Closest point: {:?}", closest_points[0].0);
        "no insert".to_string()
    }
}
