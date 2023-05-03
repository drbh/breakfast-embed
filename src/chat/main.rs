use breakfast_embed::common::chat_api_client::APIRequestClient;
use breakfast_embed::common::embedding_api_client::EmbeddingAPIClient;
use serde_derive::Deserialize;
use serde_derive::Serialize;
use std::io::{self, Write};

pub type Root = Vec<EmbeddingResp>;

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingResp {
    pub search_result: Vec<String>,
    pub search_distance: Vec<f64>,
    pub insertion: String,
    pub labels: Vec<String>,
}

#[actix_web::main]
async fn main() {
    loop {
        // init a new client for each interaction - this avoids
        // the client from being dropped and the connection closed
        let embedding_client = EmbeddingAPIClient::new("http://localhost:8080");

        print!("> ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        // remove the newline from the input
        input = input.trim().to_string();

        // handle commands like !clear, !exit, and !store
        match input.as_str() {
            "!help" => {
                // print the help menu
                println!("\nThe following commands are available:\n");
                println!("!clear - clear the screen");
                println!("!drop - drop the database");
                println!("!exit - exit the program");
                println!("!help - print this help menu");
                println!("!store - upload the sentences.txt file to the database");
                println!("[sentence] - search for similar sentences\n");
                continue;
            }
            "!clear" => {
                // clear the screen
                print!("{}[2J", 27 as char);
                // move the cursor to the top left
                print!("{}[1;1H", 27 as char);
                continue;
            }
            "!drop" => {
                // Implement the logic to clear the database here
                println!("Database cleared.");

                // call wipe on the embedding_client
                let raw_response = embedding_client.wipe().await;

                match raw_response {
                    Ok(_raw_response) => {}
                    Err(e) => {
                        eprintln!("Error: {:?}", e);
                    }
                }
            }
            "!exit" => {
                println!("Exiting the program.");
                break;
            }
            "!store" => {
                // Implement the logic to upload a file line by line to the database here
                println!("File uploaded to the database.");

                let file_contents = std::fs::read_to_string("sentences.txt").unwrap();

                // split the file_contents into sentences
                let mut lines = file_contents
                    // split by .
                    .split(|c| c == '.')
                    .map(|s| s.to_string())
                    .collect::<Vec<String>>();

                // filter out empty strings or sentence that are < 3 characters
                lines = lines
                    .into_iter()
                    // remove close to empty strings
                    .filter(|s| s.len() > 3)
                    // remove new lines
                    .map(|s| s.replace("\n", " "))
                    // remove trailing spaces
                    .map(|s| s.trim().to_string())
                    .collect::<Vec<String>>();

                let labels = vec![String::from("ABC"); lines.len()];

                // call embed_label_search_insert on the embedding_client
                let raw_response = embedding_client
                    .embed_label_search_insert(lines, labels, true)
                    .await;

                match raw_response {
                    Ok(_raw_response) => {
                        // now lets make sure to persist the data to disk
                        let raw_response = embedding_client.flush().await;

                        match raw_response {
                            Ok(_raw_response) => {
                                println!("Data persisted to disk.");
                            }
                            Err(e) => {
                                eprintln!("Error: {:?}", e);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error: {:?}", e);
                    }
                }
            }
            _ => {
                let mut start_time = std::time::Instant::now();

                let raw_response = embedding_client
                    .embed_label_search_insert(
                        vec![input.trim().to_string()],
                        vec!["".to_string()],
                        false,
                    )
                    .await;

                let mut end_time = std::time::Instant::now();

                let mut response_sentence = String::new();

                match raw_response {
                    Ok(raw_response) => {
                        let responses: Result<Vec<EmbeddingResp>, _> =
                            serde_json::from_str(&raw_response);

                        let embedding_response: EmbeddingResp = responses.unwrap()[0].clone();

                        for ((_distance, _label), sentence) in embedding_response
                            .search_distance
                            .iter()
                            .zip(embedding_response.labels.iter())
                            .zip(embedding_response.search_result.iter())
                        {
                            response_sentence
                                .push_str(&format!("{}\n", sentence.trim().to_string()));
                        }
                    }
                    Err(e) => {
                        eprintln!("Error: {:?}", e);
                    }
                }

                println!("• Search took {}ms", (end_time - start_time).as_millis());

                start_time = std::time::Instant::now();
                println!("• Waiting for chatbot response...");
                let client = APIRequestClient::new("http://localhost:5000");

                let input_prompt = input.trim().to_string();
                let context = response_sentence;

                match client.send_request(&input_prompt, &context).await {
                    Ok(response) => {
                        end_time = std::time::Instant::now();
                        println!(
                            "• Chatbot response took {}ms",
                            (end_time - start_time).as_millis()
                        );

                        let response = &response[13..response.len() - 3];
                        println!(">> {}\n", response);
                    }
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
        }
    }
}
