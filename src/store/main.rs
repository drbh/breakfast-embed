//! This program provides a web server for updating and searching an
//! HNSW (Hierarchical Navigable Small World) map using Actix-web.
//! The map stores sentence embeddings as points in a high-dimensional space
//! and allows efficient nearest-neighbor search for similar sentences.

use actix_web::web::JsonConfig;
use actix_web::{patch, post, web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use instant_distance::{Builder, HnswMap, Search};
use parking_lot::Mutex;
use rusqlite::{Connection, Result};
use std::sync::Arc;

mod utils;
use utils::*;

mod types;
use types::*;

// use sqlite as a queue for storing new embeddings

/// Application state containing a shared HNSW map.
pub struct AppState {
    arc_mutex_map: Arc<Mutex<HnswMap<Point, String>>>,
    arc_conn: Arc<Mutex<Connection>>,
}

/// Flushes the HNSW map to disk.
#[patch("/flush")]
async fn flush(_req_body: String, data: web::Data<AppState>) -> impl Responder {
    // serialize the map
    let map = data.arc_mutex_map.lock();
    let serialized = serde_json::to_string(&*map).unwrap();

    let hnws_path: String =
        std::env::var("HNSW_PATH").unwrap_or_else(|_| "data/hnsw.json".to_string());

    std::fs::write(hnws_path, serialized.clone()).unwrap();

    HttpResponse::Ok().body("Flushed map to disk.")
}

/// Loads the HNSW map from disk.
#[patch("/load")]
async fn load(_req_body: String, data: web::Data<AppState>) -> impl Responder {
    let hnws_path: String =
        std::env::var("HNSW_PATH").unwrap_or_else(|_| "data/hnsw.json".to_string());

    let mut file = std::fs::File::open(hnws_path).unwrap();
    let map: HnswMap<Point, String> = serde_json::from_reader(&mut file).unwrap();
    let mut map_mutex = data.arc_mutex_map.lock();
    *map_mutex = map;
    HttpResponse::Ok().body("Loaded map from disk.")
}

/// Wipes the data from the HNSW map and the SQLite database.
#[patch("/wipe")]
async fn wipe(_req_body: String, data: web::Data<AppState>) -> impl Responder {
    println!("Wiping map and database.");

    let mut map_mutex = data.arc_mutex_map.lock();
    map_mutex.values.clear();
    let hnws_path: String =
        std::env::var("HNSW_PATH").unwrap_or_else(|_| "data/hnsw.json".to_string());

    // save the map to disk
    let serialized = serde_json::to_string(&*map_mutex).unwrap();
    std::fs::write(&hnws_path, serialized.clone()).unwrap();

    // re read the empty map from disk
    let mut file = std::fs::File::open(hnws_path).unwrap();
    let map: HnswMap<Point, String> = serde_json::from_reader(&mut file).unwrap();
    *map_mutex = map;

    let conn = data.arc_conn.lock();
    // drop the key_value_store and key_label_store tables
    conn.execute("DROP TABLE IF EXISTS key_value_store", [])
        .unwrap();
    conn.execute("DROP TABLE IF EXISTS key_label_store", [])
        .unwrap();

    // create the key_value_store and key_label_store tables
    conn.execute(
        "CREATE TABLE IF NOT EXISTS key_value_store (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )",
        [],
    )
    .unwrap();

    conn.execute(
        "CREATE TABLE IF NOT EXISTS key_label_store (
            key TEXT PRIMARY KEY,
            label TEXT NOT NULL
        )",
        [],
    )
    .unwrap();

    HttpResponse::Ok().body("Wiped map and database.")
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

/// Embed a sentence using ONNX Runtime.
#[post("/embed")]
async fn embed(req_body: String, _data: web::Data<AppState>) -> impl Responder {
    let req: Result<EmbedRequest, _> = serde_json::from_str(&req_body);

    match req {
        Ok(req) => {
            let mut vectors: Vec<Vec<f32>> = Vec::new();

            for sentence in &req.sentences {
                println!("Embedding sentence: {}", sentence);
                match get_embedding(&sentence).await {
                    Ok(vector) => {
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

#[post("/embed_search_insert")]
async fn embed_search_insert(
    _req: HttpRequest,
    req_body: String,
    data: web::Data<AppState>,
) -> impl Responder {
    let req: Result<EmbedRequest, _> = serde_json::from_str(&req_body);

    // Only insert the query params if the query string starts with "should_insert"
    let query_str = _req.query_string();
    let should_insert_query_params = query_str.starts_with("should_insert");

    match req {
        Ok(req) => {
            let mut results = Vec::new();

            for sentence in &req.sentences {
                match process_sentence(sentence, data.clone(), should_insert_query_params).await {
                    Ok(result) => results.push(result),
                    Err(err) => {
                        eprintln!("Error processing sentence: {:?}", err);
                        return HttpResponse::InternalServerError().body(err.to_string());
                    }
                }
            }

            HttpResponse::Ok().json(results)
        }
        Err(_) => HttpResponse::BadRequest().body("Invalid JSON format."),
    }
}

#[post("/embed_label_search_insert")]
async fn embed_label_search_insert(
    _req: HttpRequest,
    req_body: String,
    data: web::Data<AppState>,
) -> impl Responder {
    let req: Result<EmbedLabelRequest, _> = serde_json::from_str(&req_body);

    // Only insert the query params if the query string starts with "should_insert"
    let query_str = _req.query_string();
    let should_insert_query_params = query_str.starts_with("should_insert");

    match req {
        Ok(req) => {
            let mut results = Vec::new();

            // iterate over the sentences and labels at the same time
            for (sentence, label) in req.sentences.iter().zip(req.labels.iter()) {
                match process_sentence_with_label(
                    sentence,
                    label,
                    data.clone(),
                    should_insert_query_params,
                )
                .await
                {
                    Ok(result) => results.push(result),
                    Err(err) => {
                        eprintln!("Error processing sentence: {:?}", err);
                        return HttpResponse::InternalServerError().body(err.to_string());
                    }
                }
            }

            HttpResponse::Ok().json(results)
        }
        Err(_) => HttpResponse::BadRequest().body("Invalid JSON format."),
    }
}

/// Main entry point for the web server.
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Starting web server...");

    // add data folder if it doesn't exist
    std::fs::create_dir_all("data").unwrap();

    let hnws_path: String =
        std::env::var("HNSW_PATH").unwrap_or_else(|_| "data/hnsw.json".to_string());
    let sqlite_path =
        std::env::var("SQLITE_PATH").unwrap_or_else(|_| "data/vectors.db".to_string());

    // open file and if it doesn't exist create it
    let mut file = std::fs::File::open(hnws_path.clone()).unwrap_or_else(|_| {
        println!("No map found on disk, creating a new one...");
        std::fs::File::create(hnws_path).unwrap()
    });

    // try to load the map from disk if it exists otherwise create a new one
    let map: HnswMap<Point, String> = serde_json::from_reader(&mut file).unwrap_or_else(|_| {
        println!("Could not load map from disk, creating a new one...");
        Builder::default().build(Vec::new(), Vec::new())
    });

    // Create an Arc<Mutex<HnswMap>> to share between the web server and the background task.
    let arc_mutex_map = Arc::new(Mutex::new(map));
    let host = std::env::var("HOST").unwrap_or_else(|_| "[::0]:8080".to_string());

    let conn = Connection::open(sqlite_path).unwrap();

    // Create a KV store for the vectors.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS key_value_store (
            key TEXT PRIMARY KEY,
            value TEXT
        );",
        [],
    )
    .unwrap();

    // Create a KV store for the labels.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS key_label_store (
            key TEXT PRIMARY KEY,
            label TEXT
        );",
        [],
    )
    .unwrap();

    let arc_conn = Arc::new(Mutex::new(conn));

    // Create a copy of the Arc<Mutex<HnswMap>> to pass to the web server.
    let app_state = web::Data::new(AppState {
        arc_mutex_map: arc_mutex_map.clone(),
        arc_conn: arc_conn.clone(),
    });

    println!("Starting server at {}...", host);
    HttpServer::new(move || {
        App::new()
            .app_data(JsonConfig::default().limit(2 * 1024 * 1024)) // 2MB limit
            .app_data(app_state.clone())
            .service(search)
            .service(init)
            .service(update)
            .service(embed)
            .service(embed_search_insert)
            .service(embed_label_search_insert)
            .service(flush)
            .service(load)
            .service(wipe)
    })
    .bind(host)?
    .run()
    .await
}
