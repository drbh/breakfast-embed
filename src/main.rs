//! This program provides a web server for updating and searching an
//! HNSW (Hierarchical Navigable Small World) map using Actix-web.
//! The map stores sentence embeddings as points in a high-dimensional space
//! and allows efficient nearest-neighbor search for similar sentences.

use actix_web::{patch, post, web, App, HttpResponse, HttpServer, Responder};
use instant_distance::{Builder, HnswMap, Search};
use parking_lot::Mutex;
use rusqlite::{Connection, Result};
use serde_json::json;
use std::sync::Arc;

mod utils;
use utils::*;

mod types;
use types::*;

// use std::time::Duration;
// use async_std::task;

// use sqlite as a queue for storing new embeddings

/// Application state containing a shared HNSW map.
struct AppState {
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
async fn embed_search_insert(req_body: String, data: web::Data<AppState>) -> impl Responder {
    let req: Result<EmbedRequest, _> = serde_json::from_str(&req_body);

    match req {
        Ok(req) => {
            let mut results = Vec::new();

            for sentence in &req.sentences {
                match process_sentence(sentence, data.clone()).await {
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

async fn process_sentence(
    sentence: &str,
    data: web::Data<AppState>,
) -> Result<MyResponse, Box<dyn std::error::Error>> {
    println!("Embedding sentence: {}", sentence);

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

    // TODO: if we don't find it we should create the vectors and insert them into the map

    let open_ai_response = create_openai_embedding(sentence).await?;
    let vectors: Vec<Vec<f32>> = open_ai_response
        .data
        .iter()
        .map(|x| x.embedding.iter().map(|y| *y as f32).collect())
        .collect();

    conn.execute(
        "INSERT INTO key_value_store (key, value) VALUES (?1, ?2)",
        &[sentence, json!(vectors).to_string().as_str()],
    )?;

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

    insert_if_needed(
        &data.arc_mutex_map,
        structured_request.vectors[0].as_slice(),
        sentence,
        &closest_points,
    );

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

fn try_find_in_sqlite(
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

    // Create a copy of the Arc<Mutex<HnswMap>> to pass to the web server.
    let app_state = web::Data::new(AppState {
        arc_mutex_map: arc_mutex_map.clone(),
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
