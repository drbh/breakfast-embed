use serde_big_array::BigArray;
use serde_derive::{Deserialize, Serialize};

/// Represents a point in a high-dimensional space.
#[derive(Clone, Copy, Debug)]
pub struct Point(pub [f32; 1536]);

impl Point {
    /// Create a `Point2` from a slice of f32 values.
    pub fn from_slice(slice: &[f32]) -> Self {
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
pub struct MyResponse {
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

impl<'de> serde::Deserialize<'de> for Point {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let arr = <[f32; 1536]>::deserialize(deserializer)?;
        Ok(Point(arr))
    }
}
