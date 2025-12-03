//! Claude API Client - Cloud Connection
//!
//! Connects to Claude (Anthropic's AI) for advanced reasoning
//! when local LLM is insufficient or for complex queries.
//!
//! Features:
//! - Message API integration
//! - Streaming support
//! - Neural context injection
//! - Hybrid local/cloud inference
//! - Rate limiting and retry logic

use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Claude API errors
#[derive(Error, Debug)]
pub enum ClaudeError {
    #[error("API key not configured")]
    NoApiKey,
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
    #[error("API error: {status} - {message}")]
    ApiError { status: u16, message: String },
    #[error("Rate limited: retry after {retry_after_secs}s")]
    RateLimited { retry_after_secs: u64 },
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
    #[error("Timeout")]
    Timeout,
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Claude model variants
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ClaudeModel {
    /// Claude 3.5 Sonnet - Best balance of intelligence and speed
    #[serde(rename = "claude-sonnet-4-20250514")]
    Claude35Sonnet,
    /// Claude 3.5 Haiku - Fast and efficient
    #[serde(rename = "claude-3-5-haiku-20241022")]
    Claude35Haiku,
    /// Claude 3 Opus - Most capable
    #[serde(rename = "claude-3-opus-20240229")]
    Claude3Opus,
}

impl Default for ClaudeModel {
    fn default() -> Self {
        ClaudeModel::Claude35Sonnet
    }
}

impl ClaudeModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            ClaudeModel::Claude35Sonnet => "claude-sonnet-4-20250514",
            ClaudeModel::Claude35Haiku => "claude-3-5-haiku-20241022",
            ClaudeModel::Claude3Opus => "claude-3-opus-20240229",
        }
    }

    /// Get max context window for this model
    pub fn max_context(&self) -> usize {
        match self {
            ClaudeModel::Claude35Sonnet => 200_000,
            ClaudeModel::Claude35Haiku => 200_000,
            ClaudeModel::Claude3Opus => 200_000,
        }
    }
}

/// Client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeConfig {
    /// API key (from environment or config)
    pub api_key: Option<String>,
    /// Base URL for API
    pub base_url: String,
    /// Default model to use
    pub model: ClaudeModel,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Maximum retries on failure
    pub max_retries: usize,
    /// System prompt
    pub system_prompt: Option<String>,
    /// Include neural context in requests
    pub include_neural_context: bool,
}

impl Default for ClaudeConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            model: ClaudeModel::default(),
            timeout_secs: 60,
            max_retries: 3,
            system_prompt: None,
            include_neural_context: true,
        }
    }
}

/// Message role
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
}

/// Content block in a message
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
}

/// A message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn user(text: &str) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![ContentBlock::Text { text: text.to_string() }],
        }
    }

    pub fn assistant(text: &str) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::Text { text: text.to_string() }],
        }
    }
}

/// Request to Claude API
#[derive(Debug, Clone, Serialize)]
struct CreateMessageRequest {
    model: String,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

/// Response from Claude API
#[derive(Debug, Clone, Deserialize)]
pub struct MessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<ContentBlockResponse>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContentBlockResponse {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Error response from API
#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ErrorDetail {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

/// Claude API client
pub struct ClaudeClient {
    config: ClaudeConfig,
    client: Client,
    conversation_history: Vec<Message>,
}

impl ClaudeClient {
    /// Create new client with configuration
    pub fn new(config: ClaudeConfig) -> Result<Self, ClaudeError> {
        if config.api_key.is_none() {
            return Err(ClaudeError::NoApiKey);
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(ClaudeError::NetworkError)?;

        Ok(Self {
            config,
            client,
            conversation_history: Vec::new(),
        })
    }

    /// Create client from environment
    pub fn from_env() -> Result<Self, ClaudeError> {
        Self::new(ClaudeConfig::default())
    }

    /// Send a simple message and get response
    pub async fn send_message(&mut self, content: &str) -> Result<String, ClaudeError> {
        let messages = vec![Message::user(content)];
        let response = self.create_message(messages, None).await?;

        // Extract text from response
        let text = response.content
            .into_iter()
            .filter_map(|c| c.text)
            .collect::<Vec<_>>()
            .join("");

        // Save to history
        self.conversation_history.push(Message::user(content));
        self.conversation_history.push(Message::assistant(&text));

        Ok(text)
    }

    /// Send message with neural context
    pub async fn send_message_with_context(
        &mut self,
        content: &str,
        neural_context: &str,
    ) -> Result<String, ClaudeError> {
        // Prepend neural context to system prompt
        let system = if self.config.include_neural_context {
            let base_system = self.config.system_prompt.as_deref().unwrap_or(
                "You are a helpful AI assistant integrated with a neurosymbolic system on the user's phone. \
                 You have access to neural state information derived from phone sensors and reservoir computing."
            );
            Some(format!("{}\n\n{}", neural_context, base_system))
        } else {
            self.config.system_prompt.clone()
        };

        let messages = vec![Message::user(content)];
        let response = self.create_message(messages, system).await?;

        let text = response.content
            .into_iter()
            .filter_map(|c| c.text)
            .collect::<Vec<_>>()
            .join("");

        Ok(text)
    }

    /// Continue conversation with history
    pub async fn chat(&mut self, content: &str) -> Result<String, ClaudeError> {
        self.conversation_history.push(Message::user(content));

        let response = self.create_message(
            self.conversation_history.clone(),
            self.config.system_prompt.clone(),
        ).await?;

        let text = response.content
            .into_iter()
            .filter_map(|c| c.text)
            .collect::<Vec<_>>()
            .join("");

        self.conversation_history.push(Message::assistant(&text));

        Ok(text)
    }

    /// Create a message with full control
    pub async fn create_message(
        &self,
        messages: Vec<Message>,
        system: Option<String>,
    ) -> Result<MessageResponse, ClaudeError> {
        let api_key = self.config.api_key.as_ref().ok_or(ClaudeError::NoApiKey)?;

        let request = CreateMessageRequest {
            model: self.config.model.as_str().to_string(),
            max_tokens: 1024,
            system,
            messages,
            temperature: Some(0.7),
            top_p: None,
            stream: None,
        };

        let url = format!("{}/messages", self.config.base_url);

        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = Duration::from_millis(1000 * 2u64.pow(attempt as u32));
                debug!("Retry attempt {} after {:?}", attempt, delay);
                tokio::time::sleep(delay).await;
            }

            let response = self.client
                .post(&url)
                .header("x-api-key", api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&request)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    let status = resp.status().as_u16();

                    if status == 429 {
                        let retry_after = resp
                            .headers()
                            .get("retry-after")
                            .and_then(|v| v.to_str().ok())
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(60);

                        warn!("Rate limited, retry after {}s", retry_after);
                        last_error = Some(ClaudeError::RateLimited { retry_after_secs: retry_after });
                        continue;
                    }

                    if !resp.status().is_success() {
                        let error_body = resp.text().await.unwrap_or_default();
                        if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(&error_body) {
                            last_error = Some(ClaudeError::ApiError {
                                status,
                                message: error_resp.error.message,
                            });
                        } else {
                            last_error = Some(ClaudeError::ApiError {
                                status,
                                message: error_body,
                            });
                        }
                        continue;
                    }

                    let message_response: MessageResponse = resp.json().await
                        .map_err(|e| ClaudeError::InvalidResponse(e.to_string()))?;

                    return Ok(message_response);
                }
                Err(e) => {
                    warn!("Request failed: {}", e);
                    last_error = Some(ClaudeError::NetworkError(e));
                }
            }
        }

        Err(last_error.unwrap_or(ClaudeError::Timeout))
    }

    /// Clear conversation history
    pub fn clear_history(&mut self) {
        self.conversation_history.clear();
    }

    /// Get conversation history
    pub fn get_history(&self) -> &[Message] {
        &self.conversation_history
    }

    /// Set model
    pub fn set_model(&mut self, model: ClaudeModel) {
        self.config.model = model;
    }

    /// Set system prompt
    pub fn set_system_prompt(&mut self, prompt: Option<String>) {
        self.config.system_prompt = prompt;
    }
}

/// Hybrid inference manager - chooses between local and cloud
pub struct HybridInference {
    claude: Option<ClaudeClient>,
    /// Complexity threshold for using cloud (0-1)
    cloud_threshold: f32,
    /// Network connectivity status
    is_online: bool,
    /// Prefer local when possible
    prefer_local: bool,
}

impl HybridInference {
    /// Create new hybrid inference manager
    pub fn new(claude_config: Option<ClaudeConfig>) -> Self {
        let claude = claude_config.and_then(|c| ClaudeClient::new(c).ok());

        Self {
            claude,
            cloud_threshold: 0.6,
            is_online: true,
            prefer_local: true,
        }
    }

    /// Check if Claude is available
    pub fn has_claude(&self) -> bool {
        self.claude.is_some()
    }

    /// Get Claude client if available
    pub fn claude(&mut self) -> Option<&mut ClaudeClient> {
        self.claude.as_mut()
    }

    /// Decide whether to use cloud for given complexity
    pub fn should_use_cloud(&self, complexity: f32, local_available: bool) -> bool {
        if !self.is_online || !self.has_claude() {
            return false;
        }

        if !local_available {
            return true;
        }

        if self.prefer_local && complexity < self.cloud_threshold {
            return false;
        }

        complexity >= self.cloud_threshold
    }

    /// Set online status
    pub fn set_online(&mut self, online: bool) {
        self.is_online = online;
    }

    /// Set cloud complexity threshold
    pub fn set_threshold(&mut self, threshold: f32) {
        self.cloud_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Set local preference
    pub fn set_prefer_local(&mut self, prefer: bool) {
        self.prefer_local = prefer;
    }

    /// Estimate query complexity (simple heuristic)
    pub fn estimate_complexity(&self, query: &str) -> f32 {
        let mut score = 0.0;

        // Length factor
        let len = query.len() as f32;
        score += (len / 1000.0).min(0.3);

        // Complexity indicators
        let complex_words = ["analyze", "explain", "compare", "synthesize",
            "evaluate", "reason", "complex", "detailed"];
        for word in &complex_words {
            if query.to_lowercase().contains(word) {
                score += 0.1;
            }
        }

        // Question depth
        if query.matches('?').count() > 1 {
            score += 0.1;
        }

        // Code or technical content
        if query.contains("```") || query.contains("function") || query.contains("class") {
            score += 0.2;
        }

        score.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_strings() {
        assert_eq!(ClaudeModel::Claude35Sonnet.as_str(), "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_message_creation() {
        let user_msg = Message::user("Hello");
        assert!(matches!(user_msg.role, MessageRole::User));

        let assistant_msg = Message::assistant("Hi there!");
        assert!(matches!(assistant_msg.role, MessageRole::Assistant));
    }

    #[test]
    fn test_hybrid_inference() {
        let hybrid = HybridInference::new(None);
        assert!(!hybrid.has_claude());
        assert!(!hybrid.should_use_cloud(0.9, true));
    }

    #[test]
    fn test_complexity_estimation() {
        let hybrid = HybridInference::new(None);

        let simple = "What time is it?";
        let complex = "Please analyze and compare the different approaches to machine learning, evaluate their strengths and weaknesses, and synthesize a detailed explanation.";

        let simple_score = hybrid.estimate_complexity(simple);
        let complex_score = hybrid.estimate_complexity(complex);

        assert!(complex_score > simple_score);
    }
}
