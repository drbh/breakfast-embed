//! This program provides a web server for updating and searching an
//! HNSW (Hierarchical Navigable Small World) map using Actix-web.
//! The map stores sentence embeddings as points in a high-dimensional space
//! and allows efficient nearest-neighbor search for similar sentences.

use actix_web::web;
use instant_distance::{Builder, HnswMap, Search};
use parking_lot::Mutex;
use pretty_good_embeddings::Client as EmbeddingsClient;
use reqwest::{header, redirect::Policy, Client};
use rusqlite::{Connection, Result};
use serde_json::json;
use std::env;
use std::sync::Arc;

use crate::{AppState, EmbedResponse, MyLabelledResponse, MyResponse, Point, Request};

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
) -> String {
    let mut map = arc_map.lock();
    map.insert(Point::from_slice(vector), sentence.to_string())
        .expect("insertion failed");
    "success".to_string()
}

pub async fn process_sentence_with_label(
    sentence: &str,
    label: &str,
    data: web::Data<AppState>,
    should_insert: bool,
) -> Result<MyLabelledResponse, Box<dyn std::error::Error>> {
    let conn = data.arc_conn.lock();

    // Check if the sentence is already in the database and if so, return it.
    if let Some(result) = try_find_label_in_sqlite(&conn, sentence)? {
        let structured_request = Request {
            vectors: vec![result.search_distance.clone()],
            sentences: vec![sentence.to_string()],
        };
        let closest_points = search_closest_points(
            &data.arc_mutex_map,
            &result.search_distance,
            &structured_request,
        )
        .unwrap();
        let to_send = MyLabelledResponse {
            search_result: closest_points
                .iter()
                .map(|(value, _)| value.clone())
                .collect::<Vec<_>>(),
            search_distance: closest_points
                .iter()
                .map(|(_, distance)| *distance)
                .collect::<Vec<_>>(),
            insertion: "already exists".to_string(),
            labels: result.labels.clone(),
        };
        return Ok(to_send);
    }

    let _client = EmbeddingsClient::new();
    let mut client = _client.init("/Users/drbh/Projects/pretty-good-embeddings/onnx".to_string());

    // If the sentence is not in the database, create an embedding for it.
    let embedding = client.embedding(sentence).unwrap();
    let vectors = vec![embedding];

    // Only insert if configured to do so.
    if should_insert {
        conn.execute(
            "INSERT INTO key_label_store (key, label) VALUES (?1, ?2)",
            &[sentence, label],
        )?;

        conn.execute(
            "INSERT INTO key_value_store (key, value) VALUES (?1, ?2)",
            &[sentence, json!(vectors).to_string().as_str()],
        )?;
    }

    // Search for the closest points to the embedding.
    let structured_request = Request {
        vectors: vectors.clone(),
        sentences: vec![sentence.to_string()],
    };

    let closest_points = search_closest_points(
        &data.arc_mutex_map,
        structured_request.vectors[0].as_slice(),
        &structured_request,
    )
    .unwrap();

    // Add the embedding to the hnsw map.
    if should_insert {
        insert_if_needed(
            &data.arc_mutex_map,
            structured_request.vectors[0].as_slice(),
            sentence,
        );
    }

    // Get labels for keys from closest points
    let mut labels = Vec::new();
    for closest_point in &closest_points {
        if let Some(result) = try_find_label_in_sqlite(&conn, &closest_point.0)? {
            labels.extend(result.labels.clone());
        }
    }

    let to_send = MyLabelledResponse {
        search_result: closest_points
            .iter()
            .map(|(value, _)| value.clone())
            .collect::<Vec<_>>(),
        search_distance: closest_points
            .iter()
            .map(|(_, distance)| *distance)
            .collect::<Vec<_>>(),
        insertion: "inserted".to_string(),
        labels: labels,
    };

    Ok(to_send)
}

pub async fn process_sentence(
    sentence: &str,
    data: web::Data<AppState>,
    should_insert: bool,
) -> Result<MyResponse, Box<dyn std::error::Error>> {
    println!("Embedding sentence: {}", sentence);

    let model_path = data.model_path.clone();

    let conn = data.arc_conn.lock();
    if let Some(result) = try_find_in_sqlite(&conn, sentence)? {
        // TODO: if we find it we should use the stored vectors to search for the closest point

        let structured_request = Request {
            vectors: vec![result.search_distance.clone()],
            sentences: vec![sentence.to_string()],
        };

        let closest_points = search_closest_points(
            //
            &data.arc_mutex_map,
            &result.search_distance,
            &structured_request,
        )
        .unwrap();

        println!("Closest points: {:?}", closest_points);

        let to_send = MyResponse {
            search_result: closest_points
                .iter()
                .map(|(value, _)| value.clone())
                .collect::<Vec<_>>(),
            search_distance: closest_points
                .iter()
                .map(|(_, distance)| *distance)
                .collect::<Vec<_>>(),
            insertion: "already exists".to_string(),
        };

        // TODO: should return the closest point
        return Ok(to_send);
    }

    let client = EmbeddingsClient::new();
    let mut session = client.init(model_path);

    let embedding = session.embedding(sentence).unwrap();
    let vectors = vec![embedding];

    if should_insert {
        conn.execute(
            "INSERT INTO key_value_store (key, value) VALUES (?1, ?2)",
            &[sentence, json!(vectors).to_string().as_str()],
        )?;
    }

    let structured_request = Request {
        vectors: vectors.clone(),
        sentences: vec![sentence.to_string()],
    };

    let closest_points = search_closest_points(
        &data.arc_mutex_map,
        structured_request.vectors[0].as_slice(),
        &structured_request,
    )
    .unwrap();

    if should_insert {
        insert_if_needed(
            &data.arc_mutex_map,
            structured_request.vectors[0].as_slice(),
            sentence,
        );
    }

    let to_send = MyResponse {
        search_result: closest_points
            .iter()
            .map(|(value, _)| value.clone())
            .collect::<Vec<_>>(),
        search_distance: closest_points
            .iter()
            .map(|(_, distance)| *distance)
            .collect::<Vec<_>>(),
        insertion: "inserted".to_string(),
    };

    Ok(to_send)
}

pub fn try_find_in_sqlite(
    conn: &Connection,
    sentence: &str,
) -> Result<Option<MyResponse>, rusqlite::Error> {
    let mut stmt = conn.prepare("SELECT value FROM key_value_store WHERE key = ?")?;

    let mut rows = stmt.query_map(
        &[sentence],
        |row: &rusqlite::Row| -> rusqlite::Result<String> { row.get(0) },
    )?;

    if let Some(_row) = rows.next() {
        println!("Sentence already in sqlite.");

        // parse the row into a vector
        let vector: Vec<Vec<f32>> = serde_json::from_str(&_row.unwrap()).unwrap();

        let result = MyResponse {
            search_result: vec![],
            search_distance: vector[0].clone(),
            insertion: "found in sqlite".to_string(),
        };

        Ok(Some(result))
    } else {
        Ok(None)
    }
}

// try to get label for a key in sqlite
pub fn try_find_label_in_sqlite(
    conn: &Connection,
    sentence: &str,
) -> Result<Option<MyLabelledResponse>, rusqlite::Error> {
    let mut stmt = conn.prepare("SELECT label FROM key_label_store WHERE key = ?")?;

    let mut rows = stmt.query_map(
        &[sentence],
        |row: &rusqlite::Row| -> rusqlite::Result<String> { row.get(0) },
    )?;

    let mut result = MyLabelledResponse {
        search_result: vec![],
        search_distance: vec![],
        insertion: "found in sqlite".to_string(),
        labels: vec![],
    };

    if let Some(_row) = rows.next() {
        let label: String = _row.unwrap();
        result.labels.push(label);
    }

    let mut stmt = conn.prepare("SELECT value FROM key_value_store WHERE key = ?")?;

    let mut rows = stmt.query_map(
        &[sentence],
        |row: &rusqlite::Row| -> rusqlite::Result<String> { row.get(0) },
    )?;

    if let Some(_row) = rows.next() {
        let vector: Vec<Vec<f32>> = serde_json::from_str(&_row.unwrap()).unwrap();
        result.search_distance = vector[0].clone();
    }

    // if len of labels is 0, then we didn't find anything
    if result.labels.len() == 0 {
        return Ok(None);
    }

    Ok(Some(result))
}
