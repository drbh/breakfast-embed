use reqwest::{Client, Error};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize)]
struct RequestPayload {
    input_prompt: String,
    context: String,
}

pub struct APIRequestClient {
    api_url: String,
    client: Client,
}

impl APIRequestClient {
    pub fn new(base_url: &str) -> Self {
        let client = Client::new();
        Self {
            api_url: base_url.to_string(),
            client,
        }
    }

    async fn post_data<T: Serialize>(&self, endpoint: &str, data: &T) -> Result<String, Error> {
        let body = json!(data).to_string();

        let response = self
            .client
            .post(&format!("{}/{}", self.api_url, endpoint))
            .header("Content-Type", "application/json")
            .body(body)
            .send()
            .await?;

        let response_text = response.text().await?;
        Ok(response_text)
    }

    pub async fn send_request(&self, input_prompt: &str, context: &str) -> Result<String, Error> {
        let payload = RequestPayload {
            input_prompt: input_prompt.to_string(),
            context: context.to_string(),
        };

        self.post_data("api", &payload).await
    }
}
