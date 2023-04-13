//! This program provides a web server for updating and searching an
//! HNSW (Hierarchical Navigable Small World) map using Actix-web.
//! The map stores sentence embeddings as points in a high-dimensional space
//! and allows efficient nearest-neighbor search for similar sentences.

use actix_web::{patch, post, web, App, HttpResponse, HttpServer, Responder};
use fastbloom_rs::Membership;
use fastbloom_rs::{BloomFilter, FilterBuilder};
use instant_distance::{Builder, HnswMap, Search};
use parking_lot::Mutex;
use reqwest::header;
use reqwest::redirect::Policy;
use reqwest::Client;
use rusqlite::{Connection, Result};
use serde_derive::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::sync::Arc;

// use std::time::Duration;
// use async_std::task;

/// Application state containing a shared HNSW map.
struct AppState {
    arc_mutex_map: Arc<Mutex<HnswMap<Point, String>>>,
    arc_mutex_bloom: Arc<Mutex<BloomFilter>>,
    arc_conn: Arc<Mutex<Connection>>,
}

/// Fetch the sentence embeddings for a list of sentences using OpenAI's
async fn create_openai_embedding(
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

/// Flushes the HNSW map to disk.
#[patch("/flush")]
async fn flush(req_body: String, data: web::Data<AppState>) -> impl Responder {
    // serialize the map
    let map = data.arc_mutex_map.lock();
    let serialized = serde_json::to_string(&*map).unwrap();
    std::fs::write("map.json", serialized.clone()).unwrap();

    // serialize the bloom filter
    let bloom = data.arc_mutex_bloom.lock();
    let mut file = std::fs::File::create("bloom.json").unwrap();
    serde_json::to_writer(&mut file, bloom.get_u64_array()).unwrap();

    HttpResponse::Ok().body("Flushed map to disk.")
}

/// Loads the HNSW map from disk.
#[patch("/load")]
async fn load(req_body: String, data: web::Data<AppState>) -> impl Responder {
    let mut file = std::fs::File::open("map.json").unwrap();
    let map: HnswMap<Point, String> = serde_json::from_reader(&mut file).unwrap();
    let mut map_mutex = data.arc_mutex_map.lock();
    *map_mutex = map;
    HttpResponse::Ok().body("Loaded map from disk.")
}

/// Search for the nearest sentence embedding to the provided point.
#[post("/search")]
async fn search(req_body: String, data: web::Data<AppState>) -> impl Responder {
    let floats: Result<Vec<f32>, _> = serde_json::from_str(&req_body);

    floats.map_or_else(
        |_| HttpResponse::BadRequest().body("Invalid JSON format."),
        |floats| {
            let point = Point::from_slice(&floats);
            let map = data.arc_mutex_map.lock();
            let mut search = Search::default();
            let closest_point = map.search(&point, &mut search).next().unwrap();

            HttpResponse::Ok().body(format!("{}\n", closest_point.value))
        },
    )
}

/// Initalize the HNSW map with new sentence embeddings.
#[post("/init")]
async fn init(req_body: String, data: web::Data<AppState>) -> impl Responder {
    let req: Result<Request, _> = serde_json::from_str(&req_body);

    req.map_or_else(
        |_| HttpResponse::BadRequest().body("Invalid JSON format."),
        |req| {
            let mut map = data.arc_mutex_map.lock();

            let points = req
                .vectors
                .iter()
                .map(|vector| Point::from_slice(vector))
                .collect::<Vec<_>>();

            println!("Initializing map with {} points...", req.vectors.len());

            *map = Builder::default().build(points, req.sentences);

            // print the size of the map
            println!("Map size: {}", map.values.len());

            HttpResponse::Ok().body(req_body)
        },
    )
}

/// Update the HNSW map with new sentence embeddings.
#[post("/update")]
async fn update(req_body: String, data: web::Data<AppState>) -> impl Responder {
    let req: Result<Request, _> = serde_json::from_str(&req_body);

    req.map_or_else(
        |_| HttpResponse::BadRequest().body("Invalid JSON format."),
        |req| {
            let mut map = data.arc_mutex_map.lock();

            println!("Updating map with {} points...", req.vectors.len());

            for (vector, sentence) in req.vectors.iter().zip(req.sentences.iter()) {
                map.insert(Point::from_slice(vector), sentence.clone())
                    .expect("insertion failed");
            }

            // print the size of the map
            println!("Map size: {}", map.values.len());

            HttpResponse::Ok().body(req_body)
        },
    )
}

/// Embed a sentence using OpenAI's text embedding API.
#[post("/embed")]
async fn embed(req_body: String, data: web::Data<AppState>) -> impl Responder {
    let req: Result<EmbedRequest, _> = serde_json::from_str(&req_body);

    match req {
        Ok(req) => {
            let mut vectors: Vec<Vec<f32>> = Vec::new();

            for sentence in &req.sentences {
                println!("Embedding sentence: {}", sentence);
                match create_openai_embedding(&sentence).await {
                    Ok(open_ai_response) => {
                        let vector: Vec<f32> = open_ai_response
                            .data
                            .iter()
                            .map(|x| x.embedding.iter().map(|y| *y as f32).collect())
                            .collect::<Vec<Vec<f32>>>()
                            .into_iter()
                            .flatten()
                            .collect();

                        vectors.push(vector);
                    }
                    Err(err) => {
                        // Handle the error and return an appropriate error response.
                        eprintln!("Error creating OpenAI embedding: {:?}", err);
                        return HttpResponse::InternalServerError()
                            .body("Error creating OpenAI embedding.");
                    }
                }
            }

            let structured = Request {
                vectors,
                sentences: req.sentences,
            };
            HttpResponse::Ok().json(structured)
        }
        Err(_) => HttpResponse::BadRequest().body("Invalid JSON format."),
    }
}

/// Embed search and insert a sentence using OpenAI's text embedding API.
#[post("/embed_search_insert")]
async fn embed_search_insert(req_body: String, data: web::Data<AppState>) -> impl Responder {
    let req: Result<EmbedRequest, _> = serde_json::from_str(&req_body);

    match req {
        Ok(req) => {
            let mut results = Vec::new();

            for sentence in &req.sentences {
                println!("Embedding sentence: {}", sentence);

                // TODO: improve how we search sqlite
                //
                let mut conn = data.arc_conn.lock();
                let mut stmt = conn
                    .prepare("SELECT value FROM key_value_store WHERE key = ?")
                    .unwrap();

                let mut rows = stmt
                    .query_map(
                        &[sentence],
                        |row: &rusqlite::Row| -> rusqlite::Result<String> { row.get(0) },
                    )
                    .unwrap();

                if let Some(row) = rows.next() {
                    println!("Sentence already in sqlite.");

                    // TODO: return the result here
                    let result = MyResponse {
                        search_result: vec![],
                        search_distance: vec![],
                        insertion: "found in sqlite".to_string(),
                    };

                    results.push(result);

                    println!("Continuing...");
                    continue;
                }

                // TODO: decide if we want to check the bloom filter here or not
                //
                // check if contained in bloom filter
                let mut bloom_filter = data.arc_mutex_bloom.lock();
                if bloom_filter.contains(sentence.as_bytes()) {
                    println!("Sentence already in map.");
                    // results.push(sentence.clone());
                    // continue;
                } else {
                    bloom_filter.add(sentence.as_bytes());
                }

                match create_openai_embedding(&sentence).await {
                    Ok(open_ai_response) => {
                        let vectors: Vec<Vec<f32>> = open_ai_response
                            .data
                            .iter()
                            .map(|x| x.embedding.iter().map(|y| *y as f32).collect())
                            .collect();

                        conn.execute(
                            "INSERT INTO key_value_store (key, value) VALUES (?1, ?2)",
                            &[sentence, json!(vectors).to_string().as_str()],
                        )
                        .unwrap();

                        let structured = Request {
                            vectors: vectors.clone(),
                            sentences: req.sentences.clone(),
                        };

                        // Search for the nearest sentence embedding.
                        let point = Point::from_slice(&vectors[0]);
                        let mut map = data.arc_mutex_map.lock();
                        let mut _search = Search::default();

                        // if map is none init it
                        if map.values.len() == 0 {
                            println!("Initializing map with {} points...", req.sentences.len());
                            *map = Builder::default().build(
                                vectors
                                    .iter()
                                    .map(|vector| Point::from_slice(vector))
                                    .collect(),
                                req.sentences.clone(),
                            );

                            let result = MyResponse {
                                search_result: vec![],
                                search_distance: vec![],
                                insertion: "success".to_string(),
                            };

                            results.push(result);
                            continue;
                        }

                        // Perform the search and store the result
                        let closest_points = {
                            let mut closest_points_vec = Vec::new();
                            for closest_point in map.search(&point, &mut _search).take(5) {
                                closest_points_vec
                                    .push((closest_point.value.clone(), closest_point.distance));
                            }
                            closest_points_vec
                        };
                        // Explicitly drop the lock, allowing mutable access to the map.
                        drop(map);

                        let mut insertion = "no insert".to_string();

                        // if the closest point is too far away, insert the new sentence embedding
                        if closest_points[0].1 > 0.002 {
                            // Re-acquire the lock and insert the new sentence embeddings into the HNSW map.
                            let mut map = data.arc_mutex_map.lock();
                            map.insert(Point::from_slice(&vectors[0]), sentence.clone())
                                .expect("insertion failed");
                            insertion = "success".to_string();
                        } else {
                            println!("Closest point is too close, not inserting.");
                            println!("Closest point: {:?}", closest_points[0].0)
                        }

                        // Return the results of all three operations.
                        let result = MyResponse {
                            search_result: closest_points
                                .iter()
                                .map(|(value, _)| value.clone())
                                .collect::<Vec<_>>(),
                            search_distance: closest_points
                                .iter()
                                .map(|(_, distance)| *distance)
                                .collect::<Vec<_>>(),
                            insertion,
                        };

                        results.push(result);
                    }
                    Err(err) => {
                        // Handle the error and return an appropriate error response.
                        eprintln!("Error creating OpenAI embedding: {:?}", err);
                        return HttpResponse::InternalServerError()
                            .body("Error creating OpenAI embedding.");
                    }
                }
            }

            println!("Results: {:?}", results);
            HttpResponse::Ok().json(results)
        }
        Err(_) => HttpResponse::BadRequest().body("Invalid JSON format."),
    }
}

// // TODO: revist
// async fn flush_periodically(data: web::Data<AppState>) {
//     loop {
//         task::sleep(Duration::from_secs(10)).await;
//         // println!("Flushing map to disk...");
//         let mut file = std::fs::File::open("map.json").unwrap();
//         let map: HnswMap<Point, String> = serde_json::from_reader(&mut file).unwrap();
//         let mut map_mutex = data.arc_mutex_map.lock();
//         *map_mutex = map;
//         // println!("Map flushed to disk.");
//     }
// }

/// Main entry point for the web server.
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // open file and if it doesn't exist create it
    let mut file = std::fs::File::open("map.json").unwrap_or_else(|_| {
        println!("No map found on disk, creating a new one...");
        std::fs::File::create("map.json").unwrap()
    });

    // try to load the map from disk if it exists otherwise create a new one
    let map: HnswMap<Point, String> = serde_json::from_reader(&mut file).unwrap_or_else(|_| {
        println!("No map found on disk, creating a new one...");
        Builder::default().build(Vec::new(), Vec::new())
    });

    // Create an Arc<Mutex<HnswMap>> to share between the web server and the background task.
    let arc_mutex_map = Arc::new(Mutex::new(map));
    let host = std::env::var("HOST").unwrap_or_else(|_| "[::0]:8080".to_string());

    let conn = Connection::open("vectors.db").unwrap();

    conn.execute(
        "CREATE TABLE IF NOT EXISTS key_value_store (
            key TEXT PRIMARY KEY,
            value TEXT
        );",
        [],
    )
    .unwrap();

    let arc_conn = Arc::new(Mutex::new(conn));

    // TODO: add insert command
    //
    // INSERT INTO key_value_store (key, value) VALUES ('some_key', 'some_value')
    // ON CONFLICT (key) DO UPDATE SET value = excluded.value;

    // SELECT value FROM key_value_store WHERE key = 'some_key';

    // open file and if it doesn't exist create it
    let mut bloom_file = std::fs::File::open("bloom.json").unwrap_or_else(|_| {
        println!("No bloom filter found on disk, creating a new one...");
        std::fs::File::create("bloom.json").unwrap()
    });

    // try to load the bloom filter from disk if it exists otherwise create a new one
    let bloom: BloomFilter = match serde_json::from_reader::<_, Vec<u64>>(&mut bloom_file) {
        Ok(u64_array) => {
            println!("Loading bloom filter from disk...");
            BloomFilter::from_u64_array(u64_array.as_slice(), 1_000_000)
        }
        Err(_) => {
            println!("No bloom filter found on disk, creating in memory...");
            // Create a new Bloom filter with 1 million slots and a false positive rate of 1%.
            BloomFilter::new(FilterBuilder::new(1_000_000, 0.01))
        }
    };

    // let builder = FilterBuilder::new(1_000_000, 0.01);
    // let bloom = BloomFilter::new(builder);
    let arc_mutex_bloom = Arc::new(Mutex::new(bloom));

    // Create a copy of the Arc<Mutex<HnswMap>> to pass to the web server.
    let app_state = web::Data::new(AppState {
        arc_mutex_map: arc_mutex_map.clone(),
        arc_mutex_bloom: arc_mutex_bloom.clone(),
        arc_conn: arc_conn.clone(),
    });

    // let mut file = std::fs::File::create("bloom.json").unwrap();
    // serde_json::to_writer(&mut file, &bloom).unwrap();

    // // TODO: revist
    // // Spawn the background task
    // // task::spawn(flush_periodically(web::Data::new(AppState {
    // //     arc_mutex_map: arc_mutex_map.clone(),
    // // })));

    println!("Starting server at {}...", host);
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(search)
            .service(init)
            .service(update)
            .service(embed)
            .service(embed_search_insert)
            .service(flush)
            .service(load)
    })
    .bind(host)?
    .run()
    .await
}

/// Represents a point in a high-dimensional space.
#[derive(Clone, Copy, Debug)]
struct Point([f32; 1536]);

impl Point {
    /// Create a `Point2` from a slice of f32 values.
    fn from_slice(slice: &[f32]) -> Self {
        let mut point = Point::default();
        point.0.copy_from_slice(slice);
        point
    }
}

impl Default for Point {
    fn default() -> Self {
        Point([0.0; 1536])
    }
}

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MyResponse {
    pub search_result: Vec<String>,
    pub search_distance: Vec<f32>,
    pub insertion: String,
}

/// Request structure for updating the HNSW map.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub sentences: Vec<String>,
    pub vectors: Vec<Vec<f32>>,
}

/// Request structure for embedding a sentence.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub sentences: Vec<String>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbedResponse {
    pub object: String,
    pub data: Vec<Daum>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Daum {
    pub object: String,
    pub index: i64,
    pub embedding: Vec<f64>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    #[serde(rename = "prompt_tokens")]
    pub prompt_tokens: i64,
    #[serde(rename = "total_tokens")]
    pub total_tokens: i64,
}

// implement serde::Serialize for Point
impl serde::Serialize for Point {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut vec = vec![];
        for i in 0..self.0.len() {
            vec.push(self.0[i]);
        }
        vec.serialize(serializer)
    }
}

// impl serde::Deserialize for Point {
//     fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
//     where
//         D: serde::Deserializer<'_>,
//     {
//         let vec = Vec::<f32>::deserialize(deserializer)?;
//         let mut point = Point::default();
//         point.0.copy_from_slice(&vec);
//         Ok(point)
//     }
// }
use serde_big_array::BigArray;

impl<'de> serde::Deserialize<'de> for Point {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let arr = <[f32; 1536]>::deserialize(deserializer)?;
        Ok(Point(arr))
    }
}
