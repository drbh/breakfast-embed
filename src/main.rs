//! This program provides a web server for updating and searching an
//! HNSW (Hierarchical Navigable Small World) map using Actix-web.
//! The map stores sentence embeddings as points in a high-dimensional space
//! and allows efficient nearest-neighbor search for similar sentences.

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
    std::fs::write("map.json", serialized.clone()).unwrap();

    HttpResponse::Ok().body("Flushed map to disk.")
}

/// Loads the HNSW map from disk.
#[patch("/load")]
async fn load(_req_body: String, data: web::Data<AppState>) -> impl Responder {
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
async fn embed(req_body: String, _data: web::Data<AppState>) -> impl Responder {
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
            .app_data(app_state.clone())
            .service(search)
            .service(init)
            .service(update)
            .service(embed)
            .service(embed_search_insert)
            .service(embed_label_search_insert)
            .service(flush)
            .service(load)
    })
    .bind(host)?
    .run()
    .await
}
