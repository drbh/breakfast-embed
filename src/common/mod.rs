// src/common/mod.rs
pub mod embedding_api_client;
pub mod chat_api_client;

// only include if the chat feature is enabled
#[cfg(feature = "chat")]
pub mod text_generation;