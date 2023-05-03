use reqwest::{Client, Error};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Duration;

#[derive(Serialize, Deserialize)]
struct Sentences {
    sentences: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct LabelledSentences {
    sentences: Vec<String>,
    labels: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct UpdatePayload {
    sentences: Vec<String>,
    vectors: Vec<Vec<f64>>,
}

pub struct EmbeddingAPIClient {
    api_url: String,
    client: Client,
}

impl EmbeddingAPIClient {
    pub fn new(base_url: &str) -> Self {
        let client = reqwest::ClientBuilder::new()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build Reqwest client");

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
            .body(body)
            .send()
            .await?;

        let response_text = response.text().await?;
        Ok(response_text)
    }

    pub async fn embed_label_search_insert(
        &self,
        sentences: Vec<String>,
        labels: Vec<String>,
        save: bool,
    ) -> Result<String, Error> {
        let endpoint = if save {
            "embed_label_search_insert?should_insert=true"
        } else {
            "embed_label_search_insert"
        };

        self.post_data(&endpoint, &LabelledSentences { sentences, labels })
            .await
    }

    pub async fn embed_search_insert(&self, sentences: Vec<String>) -> Result<String, Error> {
        self.post_data("embed_search_insert", &Sentences { sentences })
            .await
    }

    pub async fn embed(&self, sentences: Vec<String>) -> Result<String, Error> {
        self.post_data("embed", &Sentences { sentences }).await
    }

    pub async fn search(&self, embedding: Vec<f64>) -> Result<String, Error> {
        self.post_data("search", &embedding).await
    }

    pub async fn wipe(&self) -> Result<String, Error> {
        let response = self
            .client
            .patch(&format!("{}/wipe", self.api_url))
            .send()
            .await?;

        let response_text = response.text().await?;
        Ok(response_text)
    }

    pub async fn update(
        &self,
        sentences: Vec<String>,
        vectors: Vec<Vec<f64>>,
    ) -> Result<String, Error> {
        self.post_data("update", &UpdatePayload { sentences, vectors })
            .await
    }

    pub async fn init(
        &self,
        sentences: Vec<String>,
        vectors: Vec<Vec<f64>>,
    ) -> Result<String, Error> {
        self.post_data("init", &UpdatePayload { sentences, vectors })
            .await
    }

    pub async fn flush(&self) -> Result<String, Error> {
        let response = self
            .client
            .patch(&format!("{}/flush", self.api_url))
            .send()
            .await?;

        let response_text = response.text().await?;
        Ok(response_text)
    }

    pub async fn load(&self) -> Result<String, Error> {
        let response = self
            .client
            .patch(&format!("{}/load", self.api_url))
            .send()
            .await?;

        let response_text = response.text().await?;
        Ok(response_text)
    }
}
