use anyhow::Result;
use rust_bert::pipelines::{
    common::ModelType,
    text_generation::{TextGenerationConfig, TextGenerationModel},
};
use rust_bert::resources::LocalResource;
use std::path::PathBuf;

pub struct TextGenerator {
    model: TextGenerationModel,
}

impl TextGenerator {
    pub fn new(model_path: &str, config_path: &str, vocab_path: &str) -> Result<Self> {
        let config_resource = Box::new(LocalResource::from(PathBuf::from(config_path)));
        let vocab_resource = Box::new(LocalResource::from(PathBuf::from(vocab_path)));
        let model_resource = Box::new(LocalResource::from(PathBuf::from(model_path)));

        let generate_config = TextGenerationConfig {
            model_type: ModelType::T5,
            model_resource,
            config_resource,
            vocab_resource,
            max_length: Some(512),
            do_sample: true,
            num_beams: 1,
            temperature: 1.0,
            num_return_sequences: 1,
            ..Default::default()
        };

        let model = TextGenerationModel::new(generate_config)?;

        Ok(Self { model })
    }

    pub fn generate_text(&self, input_context: &str) -> Result<Vec<String>> {
        let output = self
            .model
            .generate(&[input_context], None)
            .into_iter()
            .map(|sentence| sentence.to_string())
            .collect();
        Ok(output)
    }
}
