//! Local LLM Interface - Llama 3.2 Integration
//!
//! Provides an interface to run Llama 3.2 locally on the device.
//! Uses llama.cpp for efficient inference on mobile hardware.
//!
//! Supports:
//! - Model loading and management
//! - Text generation with streaming
//! - Chat-style interactions
//! - Neural context injection

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// LLM errors
#[derive(Error, Debug)]
pub enum LlmError {
    #[error("Model not loaded")]
    ModelNotLoaded,
    #[error("Model file not found: {0}")]
    ModelNotFound(PathBuf),
    #[error("Loading error: {0}")]
    LoadingError(String),
    #[error("Generation error: {0}")]
    GenerationError(String),
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("Runtime error: {0}")]
    RuntimeError(String),
}

/// LLM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Path to model file (GGUF format)
    pub model_path: PathBuf,
    /// Number of threads for inference
    pub n_threads: usize,
    /// Context window size
    pub context_size: usize,
    /// Number of GPU layers (0 for CPU-only)
    pub n_gpu_layers: usize,
    /// Enable memory mapping
    pub use_mmap: bool,
    /// Batch size for prompt processing
    pub batch_size: usize,
    /// Use flash attention
    pub flash_attention: bool,
    /// Model type hint
    pub model_type: ModelType,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("/data/local/tmp/llama-3.2-1b-q4_k_m.gguf"),
            n_threads: 4,
            context_size: 2048,
            n_gpu_layers: 0, // CPU for mobile
            use_mmap: true,
            batch_size: 512,
            flash_attention: false,
            model_type: ModelType::Llama3_2_1B,
        }
    }
}

/// Supported model types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelType {
    Llama3_2_1B,
    Llama3_2_3B,
    Phi3Mini,
    Gemma2B,
    Custom,
}

impl ModelType {
    /// Get the chat template for this model type
    pub fn chat_template(&self) -> &'static str {
        match self {
            ModelType::Llama3_2_1B | ModelType::Llama3_2_3B => {
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            }
            ModelType::Phi3Mini => {
                "<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"
            }
            ModelType::Gemma2B => {
                "<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n"
            }
            ModelType::Custom => "{system}\n\nUser: {user}\nAssistant: ",
        }
    }

    /// Get stop tokens for this model
    pub fn stop_tokens(&self) -> Vec<&'static str> {
        match self {
            ModelType::Llama3_2_1B | ModelType::Llama3_2_3B => {
                vec!["<|eot_id|>", "<|end_of_text|>"]
            }
            ModelType::Phi3Mini => vec!["<|end|>", "<|endoftext|>"],
            ModelType::Gemma2B => vec!["<end_of_turn>"],
            ModelType::Custom => vec!["\n\nUser:", "\n\nHuman:"],
        }
    }
}

/// Chat message role
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    /// Optional neural context to inject
    pub neural_context: Option<String>,
}

/// Generation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature (0.0-2.0)
    pub temperature: f32,
    /// Top-p (nucleus) sampling
    pub top_p: f32,
    /// Top-k sampling
    pub top_k: usize,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Frequency penalty
    pub frequency_penalty: f32,
    /// Presence penalty
    pub presence_penalty: f32,
    /// Stop sequences
    pub stop: Vec<String>,
    /// Enable streaming
    pub stream: bool,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop: Vec::new(),
            stream: false,
        }
    }
}

/// Generation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    /// Generated text
    pub text: String,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Number of tokens in prompt
    pub tokens_prompt: usize,
    /// Generation time (ms)
    pub generation_time_ms: f64,
    /// Tokens per second
    pub tokens_per_second: f32,
    /// Finish reason
    pub finish_reason: FinishReason,
}

/// Why generation finished
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    Error,
    Cancelled,
}

/// Streaming token callback
pub type TokenCallback = Box<dyn Fn(&str) + Send>;

/// Local LLM wrapper
///
/// Note: This is an interface abstraction. The actual llama.cpp integration
/// would be done via FFI bindings in a real implementation.
pub struct LocalLlm {
    config: LlmConfig,
    loaded: AtomicBool,
    cancel_flag: Arc<AtomicBool>,

    // Simulated model state (in real impl, this would be llama.cpp context)
    system_prompt: String,
    conversation_history: Vec<ChatMessage>,
}

impl LocalLlm {
    /// Create new LLM instance
    pub fn new(config: LlmConfig) -> Self {
        Self {
            config,
            loaded: AtomicBool::new(false),
            cancel_flag: Arc::new(AtomicBool::new(false)),
            system_prompt: String::new(),
            conversation_history: Vec::new(),
        }
    }

    /// Load the model
    pub async fn load(&mut self) -> Result<(), LlmError> {
        info!("Loading model from {:?}", self.config.model_path);

        // In real implementation, this would:
        // 1. Load the GGUF file
        // 2. Initialize llama.cpp context
        // 3. Allocate KV cache
        // 4. Warm up the model

        // Simulate loading time
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Check if file exists (simulated for now)
        if !self.config.model_path.to_string_lossy().contains("llama") {
            // Accept any llama path for now
        }

        self.loaded.store(true, Ordering::SeqCst);
        info!("Model loaded successfully");

        Ok(())
    }

    /// Unload the model
    pub fn unload(&mut self) {
        info!("Unloading model");
        self.loaded.store(false, Ordering::SeqCst);
        self.conversation_history.clear();
    }

    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    /// Set system prompt
    pub fn set_system_prompt(&mut self, prompt: &str) {
        self.system_prompt = prompt.to_string();
    }

    /// Generate text completion
    pub async fn generate(
        &mut self,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<GenerationResult, LlmError> {
        if !self.is_loaded() {
            return Err(LlmError::ModelNotLoaded);
        }

        let start = std::time::Instant::now();
        self.cancel_flag.store(false, Ordering::SeqCst);

        debug!("Generating with prompt length: {}", prompt.len());

        // Simulate generation (in real impl, this calls llama.cpp)
        let generated_text = self.simulate_generation(prompt, &params).await?;

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let tokens = generated_text.split_whitespace().count();

        Ok(GenerationResult {
            text: generated_text,
            tokens_generated: tokens,
            tokens_prompt: prompt.split_whitespace().count(),
            generation_time_ms: elapsed,
            tokens_per_second: tokens as f32 / (elapsed / 1000.0) as f32,
            finish_reason: FinishReason::Stop,
        })
    }

    /// Chat with the model
    pub async fn chat(
        &mut self,
        message: ChatMessage,
        params: GenerationParams,
    ) -> Result<GenerationResult, LlmError> {
        if !self.is_loaded() {
            return Err(LlmError::ModelNotLoaded);
        }

        // Build prompt from history
        let mut prompt = self.build_chat_prompt(&message);

        // Inject neural context if present
        if let Some(ref context) = message.neural_context {
            prompt = format!("{}\n\n{}", context, prompt);
        }

        // Add to history
        self.conversation_history.push(message);

        // Generate response
        let result = self.generate(&prompt, params).await?;

        // Add assistant response to history
        self.conversation_history.push(ChatMessage {
            role: Role::Assistant,
            content: result.text.clone(),
            neural_context: None,
        });

        Ok(result)
    }

    /// Stream generation with callback
    pub async fn generate_stream(
        &mut self,
        prompt: &str,
        params: GenerationParams,
        callback: TokenCallback,
    ) -> Result<GenerationResult, LlmError> {
        if !self.is_loaded() {
            return Err(LlmError::ModelNotLoaded);
        }

        let start = std::time::Instant::now();
        self.cancel_flag.store(false, Ordering::SeqCst);

        // Simulate streaming generation
        let words = vec!["The", "neural", "state", "indicates", "moderate", "activity."];
        let mut generated = String::new();

        for word in &words {
            if self.cancel_flag.load(Ordering::SeqCst) {
                let tokens_generated = generated.split_whitespace().count();
                return Ok(GenerationResult {
                    text: generated,
                    tokens_generated,
                    tokens_prompt: prompt.split_whitespace().count(),
                    generation_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                    tokens_per_second: 0.0,
                    finish_reason: FinishReason::Cancelled,
                });
            }

            generated.push_str(word);
            generated.push(' ');
            callback(word);

            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let tokens = generated.split_whitespace().count();

        Ok(GenerationResult {
            text: generated.trim().to_string(),
            tokens_generated: tokens,
            tokens_prompt: prompt.split_whitespace().count(),
            generation_time_ms: elapsed,
            tokens_per_second: tokens as f32 / (elapsed / 1000.0) as f32,
            finish_reason: FinishReason::Stop,
        })
    }

    /// Cancel ongoing generation
    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
    }

    /// Clear conversation history
    pub fn clear_history(&mut self) {
        self.conversation_history.clear();
    }

    /// Get conversation history
    pub fn get_history(&self) -> &[ChatMessage] {
        &self.conversation_history
    }

    /// Build chat prompt using model's template
    fn build_chat_prompt(&self, message: &ChatMessage) -> String {
        let template = self.config.model_type.chat_template();

        let system = if self.system_prompt.is_empty() {
            "You are a helpful AI assistant integrated with a neurosymbolic system. \
             You have access to neural state information that provides context about \
             the user's environment and activity patterns."
        } else {
            &self.system_prompt
        };

        template
            .replace("{system}", system)
            .replace("{user}", &message.content)
    }

    /// Simulate text generation (placeholder for real llama.cpp integration)
    async fn simulate_generation(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<String, LlmError> {
        // This is a placeholder - real implementation would call llama.cpp

        // Simulate processing time based on prompt length
        let delay_ms = (prompt.len() / 10).min(100) as u64;
        tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;

        // Generate a contextual response based on prompt content
        let response = if prompt.contains("NEURAL_STATE") {
            if prompt.contains("high") || prompt.contains("active") {
                "Based on the current neural state showing high activity, \
                 I detect increased engagement. The sensory patterns suggest \
                 movement or interaction with the environment."
            } else if prompt.contains("low") || prompt.contains("quiet") {
                "The neural state indicates a calm, stable condition. \
                 Sensor readings are within normal ranges with minimal variation."
            } else {
                "The neural state is being processed. The integrated sensory \
                 information shows typical activity patterns."
            }
        } else if prompt.contains("?") {
            "I understand your question. Based on the available information \
             and current system state, I can provide relevant assistance."
        } else {
            "I've processed your input through the neurosymbolic pipeline. \
             The system is functioning normally."
        };

        // Truncate to max tokens (approximated by words)
        let words: Vec<&str> = response.split_whitespace().collect();
        let max_words = params.max_tokens.min(words.len());
        let truncated = words[..max_words].join(" ");

        Ok(truncated)
    }

    /// Get model info
    pub fn get_model_info(&self) -> ModelInfo {
        ModelInfo {
            model_type: self.config.model_type,
            context_size: self.config.context_size,
            loaded: self.is_loaded(),
            n_threads: self.config.n_threads,
            n_gpu_layers: self.config.n_gpu_layers,
        }
    }
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_type: ModelType,
    pub context_size: usize,
    pub loaded: bool,
    pub n_threads: usize,
    pub n_gpu_layers: usize,
}

/// Builder for creating LLM instances with fluent API
pub struct LlmBuilder {
    config: LlmConfig,
}

impl LlmBuilder {
    pub fn new() -> Self {
        Self {
            config: LlmConfig::default(),
        }
    }

    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.model_path = path.into();
        self
    }

    pub fn model_type(mut self, model_type: ModelType) -> Self {
        self.config.model_type = model_type;
        self
    }

    pub fn threads(mut self, n: usize) -> Self {
        self.config.n_threads = n;
        self
    }

    pub fn context_size(mut self, size: usize) -> Self {
        self.config.context_size = size;
        self
    }

    pub fn gpu_layers(mut self, n: usize) -> Self {
        self.config.n_gpu_layers = n;
        self
    }

    pub fn build(self) -> LocalLlm {
        LocalLlm::new(self.config)
    }
}

impl Default for LlmBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_llm_creation() {
        let llm = LlmBuilder::new()
            .model_type(ModelType::Llama3_2_1B)
            .threads(4)
            .build();

        assert!(!llm.is_loaded());
    }

    #[tokio::test]
    async fn test_llm_load() {
        let mut llm = LlmBuilder::new().build();
        llm.load().await.unwrap();
        assert!(llm.is_loaded());
    }

    #[tokio::test]
    async fn test_generation() {
        let mut llm = LlmBuilder::new().build();
        llm.load().await.unwrap();

        let result = llm.generate("Hello", GenerationParams::default()).await.unwrap();
        assert!(!result.text.is_empty());
    }

    #[tokio::test]
    async fn test_chat() {
        let mut llm = LlmBuilder::new().build();
        llm.load().await.unwrap();

        let message = ChatMessage {
            role: Role::User,
            content: "What is the current state?".to_string(),
            neural_context: Some("[NEURAL_STATE]\nActivity: high\n[/NEURAL_STATE]".to_string()),
        };

        let result = llm.chat(message, GenerationParams::default()).await.unwrap();
        assert!(!result.text.is_empty());
        assert_eq!(llm.get_history().len(), 2); // User + Assistant
    }
}
