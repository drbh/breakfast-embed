use breakfast_embed::common::embedding_api_client::EmbeddingAPIClient;
use breakfast_embed::common::text_generation::TextGenerator;
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
    println!("🦩 We are loading the model, please wait a few seconds...");

    let model_path = "../lamini-lm/LaMini-Flan-T5-783M/rust_model.ot";
    let config_path = "../lamini-lm/LaMini-Flan-T5-783M/config.json";
    let vocab_path = "../lamini-lm/LaMini-Flan-T5-783M/spiece.model";

    let start_time = std::time::Instant::now();
    let text_generator = TextGenerator::new(model_path, config_path, vocab_path).unwrap();
    let end_time = std::time::Instant::now();

    println!(
        "Model loaded in {} seconds.",
        end_time.duration_since(start_time).as_secs()
    );

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

                println!("• Waiting for chatbot response...");
                // let client = APIRequestClient::new("http://localhost:5000");
                let input_prompt = input.trim().to_string();
                let context = response_sentence;
                let full_prompt = format!(
                    r#"
Answer {input_prompt} with the following context in mind {context}
"#,
                );
                println!(
                    "• Generating text from a prompt of {} characters",
                    full_prompt.len()
                );
                start_time = std::time::Instant::now();
                let output = text_generator
                    .generate_text(&full_prompt)
                    .unwrap_or(vec!["Failed to generate text".to_string()]);
                end_time = std::time::Instant::now();
                println!(
                    "• Generated in {} seconds.",
                    end_time.duration_since(start_time).as_secs()
                );

                // combine the output into a single string
                let final_output = output.join(" ");

                // count number of words in output
                let num_words = final_output.split_whitespace().count();

                let words_generated_per_minute = (num_words as f64
                    / end_time.duration_since(start_time).as_secs() as f64)
                    * 60.0;

                println!(
                    "• Generated {} words per minute",
                    words_generated_per_minute
                );

                println!("=> {}\n", final_output);
            }
        }
    }
}
