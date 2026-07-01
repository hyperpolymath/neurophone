// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>
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

#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, warn};

pub mod egress_gate;
pub use egress_gate::{EgressClass, EgressGate, EgressGateError};

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
    /// Obligation 3.1: the egress GO/NO-GO veto refused this payload. The
    /// network call is guaranteed to NOT have been attempted when this
    /// variant is returned -- see `ClaudeClient::create_message`.
    #[error("egress denied by policy veto: {0}")]
    EgressDenied(#[from] EgressGateError),
}

/// Claude model variants
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Default)]
pub enum ClaudeModel {
    /// Claude 3.5 Sonnet - Best balance of intelligence and speed
    #[default]
    #[serde(rename = "claude-sonnet-4-20250514")]
    Claude35Sonnet,
    /// Claude 3.5 Haiku - Fast and efficient
    #[serde(rename = "claude-3-5-haiku-20241022")]
    Claude35Haiku,
    /// Claude 3 Opus - Most capable
    #[serde(rename = "claude-3-opus-20240229")]
    Claude3Opus,
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
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    pub fn assistant(text: &str) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
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
    #[allow(dead_code)]
    error_type: String,
    message: String,
}

/// Claude API client
pub struct ClaudeClient {
    config: ClaudeConfig,
    client: Client,
    conversation_history: Vec<Message>,
    /// Obligation 3.1: every outbound call goes through this veto first --
    /// see `create_message`, the single choke point that actually performs
    /// the network request.
    egress: EgressGate,
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
            egress: EgressGate::new(),
        })
    }

    /// Create client from environment
    pub fn from_env() -> Result<Self, ClaudeError> {
        Self::new(ClaudeConfig::default())
    }

    /// Send a simple message and get response. Classified as plain user text
    /// for the egress veto (obligation 3.1) -- it carries no sensor/neural
    /// provenance.
    pub async fn send_message(&mut self, content: &str) -> Result<String, ClaudeError> {
        let messages = vec![Message::user(content)];
        let response = self
            .create_message(messages, None, EgressClass::UserText)
            .await?;

        // Extract text from response
        let text = response
            .content
            .into_iter()
            .filter_map(|c| c.text)
            .collect::<Vec<_>>()
            .join("");

        // Save to history
        self.conversation_history.push(Message::user(content));
        self.conversation_history.push(Message::assistant(&text));

        Ok(text)
    }

    /// Send message with neural context. `class` is the caller's declared
    /// classification of `neural_context` for the egress veto (obligation
    /// 3.1) -- e.g. `EgressClass::RawNeuralState` if it is raw reservoir
    /// state, `EgressClass::DerivedInference` if it has already been
    /// summarised/aggregated. The veto enforces policy on whatever is
    /// declared; it cannot independently prove the declaration is honest
    /// (see `egress_gate` module docs).
    pub async fn send_message_with_context(
        &mut self,
        content: &str,
        neural_context: &str,
        class: EgressClass,
    ) -> Result<String, ClaudeError> {
        // Prepend neural context to the system prompt (only when enabled).
        let system = self.build_system(neural_context);

        let messages = vec![Message::user(content)];
        let response = self.create_message(messages, system, class).await?;

        let text = response
            .content
            .into_iter()
            .filter_map(|c| c.text)
            .collect::<Vec<_>>()
            .join("");

        Ok(text)
    }

    /// Continue conversation with history. Classified as plain user text for
    /// the egress veto (obligation 3.1); this method has no neural-context
    /// parameter to begin with.
    pub async fn chat(&mut self, content: &str) -> Result<String, ClaudeError> {
        self.conversation_history.push(Message::user(content));

        let response = self
            .create_message(
                self.conversation_history.clone(),
                self.config.system_prompt.clone(),
                EgressClass::UserText,
            )
            .await?;

        let text = response
            .content
            .into_iter()
            .filter_map(|c| c.text)
            .collect::<Vec<_>>()
            .join("");

        self.conversation_history.push(Message::assistant(&text));

        Ok(text)
    }

    /// Create a message with full control.
    ///
    /// This is the single choke point that performs the actual outbound
    /// HTTP request to the Claude API (every other method funnels through
    /// here). Obligation 3.1: the egress veto runs first, against the exact
    /// text that is about to be serialized onto the wire; on `Block`/
    /// `Escalate` this returns `Err(ClaudeError::EgressDenied(_))` and the
    /// network call is never made.
    pub async fn create_message(
        &self,
        messages: Vec<Message>,
        system: Option<String>,
        class: EgressClass,
    ) -> Result<MessageResponse, ClaudeError> {
        let api_key = self.config.api_key.as_ref().ok_or(ClaudeError::NoApiKey)?;

        let outbound_text = Self::render_outbound_text(&system, &messages);
        self.egress
            .check(class, &self.config.base_url, &outbound_text)?;

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
                let delay = Self::backoff_delay(attempt as u32);
                debug!("Retry attempt {} after {:?}", attempt, delay);
                tokio::time::sleep(delay).await;
            }

            let response = self
                .client
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
                        last_error = Some(ClaudeError::RateLimited {
                            retry_after_secs: retry_after,
                        });
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

                    let message_response: MessageResponse = resp
                        .json()
                        .await
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

    /// Build the system prompt, prepending neural context ONLY when enabled.
    /// When `include_neural_context` is false, the (sensor-derived) neural
    /// context is never placed on the wire (obligation 3.1).
    fn build_system(&self, neural_context: &str) -> Option<String> {
        if self.config.include_neural_context {
            let base = self.config.system_prompt.as_deref().unwrap_or(
                "You are a helpful AI assistant integrated with a neurosymbolic system on the user's phone. \
                 You have access to neural state information derived from phone sensors and reservoir computing."
            );
            Some(format!("{}\n\n{}", neural_context, base))
        } else {
            self.config.system_prompt.clone()
        }
    }

    /// Render exactly the text that `create_message` is about to serialize
    /// onto the wire (system prompt + all message bodies), for the egress
    /// veto (obligation 3.1) to scan. Kept separate from JSON construction so
    /// the veto sees plain text, not the JSON envelope.
    fn render_outbound_text(system: &Option<String>, messages: &[Message]) -> String {
        let mut text = String::new();
        if let Some(sys) = system {
            text.push_str(sys);
            text.push('\n');
        }
        for message in messages {
            for block in &message.content {
                let ContentBlock::Text { text: block_text } = block;
                text.push_str(block_text);
                text.push('\n');
            }
        }
        text
    }

    /// Exponential backoff for retry `attempt`, saturating and capped at 60s so
    /// it can never overflow or wait unboundedly (obligation 3.2).
    fn backoff_delay(attempt: u32) -> Duration {
        let factor = 2u64.saturating_pow(attempt.min(16));
        let ms = 1000u64.saturating_mul(factor).min(60_000);
        Duration::from_millis(ms)
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
        let complex_words = [
            "analyze",
            "explain",
            "compare",
            "synthesize",
            "evaluate",
            "reason",
            "complex",
            "detailed",
        ];
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
        assert_eq!(
            ClaudeModel::Claude35Sonnet.as_str(),
            "claude-sonnet-4-20250514"
        );
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

    // ===== Tier 3: trust boundary / egress (3.1, 3.2) =====

    #[test]
    fn egress_request_body_has_only_allowed_fields() {
        // 3.1: the serialised request must carry ONLY declared fields — never the
        // API key (it is a header), device identifiers, or raw sensor data.
        let req = CreateMessageRequest {
            model: "m".into(),
            max_tokens: 16,
            system: Some("sys".into()),
            messages: vec![Message::user("hi")],
            temperature: Some(0.7),
            top_p: None,
            stream: None,
        };
        let v = serde_json::to_value(&req).unwrap();
        let obj = v.as_object().unwrap();
        const ALLOWED: &[&str] = &[
            "model",
            "max_tokens",
            "system",
            "messages",
            "temperature",
            "top_p",
            "stream",
        ];
        for k in obj.keys() {
            assert!(
                ALLOWED.contains(&k.as_str()),
                "unexpected egress field: {k}"
            );
        }
        for forbidden in ["api_key", "x-api-key", "key", "device", "sensor", "secret"] {
            assert!(
                !obj.contains_key(forbidden),
                "sensitive field leaked: {forbidden}"
            );
        }
    }

    #[test]
    fn neural_context_not_sent_when_disabled() {
        // 3.1: include_neural_context = false => sensor-derived context never goes out.
        let off = ClaudeClient::new(ClaudeConfig {
            include_neural_context: false,
            system_prompt: Some("BASE".into()),
            api_key: Some("k".into()),
            ..Default::default()
        })
        .unwrap();
        let sys = off.build_system("SENSITIVE_NEURAL_STATE");
        assert_eq!(sys.as_deref(), Some("BASE"));

        let on = ClaudeClient::new(ClaudeConfig {
            include_neural_context: true,
            system_prompt: Some("BASE".into()),
            api_key: Some("k".into()),
            ..Default::default()
        })
        .unwrap();
        assert!(on
            .build_system("SENSITIVE_NEURAL_STATE")
            .unwrap()
            .contains("SENSITIVE_NEURAL_STATE"));
    }

    #[test]
    fn backoff_is_bounded_and_monotonic() {
        // 3.2: backoff never overflows and is capped, so the retry loop terminates.
        let mut prev = Duration::ZERO;
        for a in 0..40u32 {
            let d = ClaudeClient::backoff_delay(a);
            assert!(d >= prev, "backoff must be non-decreasing");
            assert!(d <= Duration::from_millis(60_000), "backoff must be capped");
            prev = d;
        }
    }

    #[test]
    fn total_retry_budget_is_finite() {
        // 3.2 (bounded external interaction): the retry loop runs a *bounded*
        // number of attempts (`for attempt in 0..=max_retries`), and each wait is
        // the capped backoff — so the worst-case total wait across the whole
        // interaction is finite and bounded, not just each individual delay.
        let max_retries = ClaudeConfig::default().max_retries;
        let mut total = Duration::ZERO;
        for a in 0..=(max_retries as u32) {
            total = total
                .checked_add(ClaudeClient::backoff_delay(a))
                .expect("total backoff must not overflow");
        }
        // (max_retries + 1) attempts, each capped at 60s.
        let ceiling = Duration::from_secs(60) * (max_retries as u32 + 1);
        assert!(
            total <= ceiling,
            "total retry budget {total:?} exceeds bound {ceiling:?}"
        );
    }

    #[test]
    fn user_content_cannot_inject_into_request_json() {
        // 3.2: user content is typed JSON, so quotes/braces are escaped and cannot
        // break out of the message structure.
        let nasty = "\"}],\"injected\":true,\"x\":[{\"";
        let req = CreateMessageRequest {
            model: "m".into(),
            max_tokens: 16,
            system: None,
            messages: vec![Message::user(nasty)],
            temperature: None,
            top_p: None,
            stream: None,
        };
        let v = serde_json::to_value(&req).unwrap();
        assert!(
            v.get("injected").is_none(),
            "content broke out of its field"
        );
        assert_eq!(
            v["messages"][0]["content"][0]["text"].as_str().unwrap(),
            nasty
        );
    }
}

/// Obligation 3.1: end-to-end proof that the egress veto actually gates the
/// real network call inside `create_message`, not just the standalone
/// `EgressGate` unit (see `egress_gate::tests`).
///
/// Rather than adding a mocking framework/transport-trait abstraction, these
/// tests spin up a tiny real HTTP/1.1 server on loopback that counts
/// accepted connections. `ClaudeClient` is pointed at it via `base_url` and
/// makes real `reqwest` calls against it -- so "zero calls" here means
/// literally zero TCP connections were made to the (fake) API, not "a mock
/// was not invoked".
#[cfg(test)]
mod egress_integration_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A minimal, valid `MessageResponse` JSON body.
    const FAKE_RESPONSE_BODY: &str = r#"{"id":"msg_test","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"model":"m","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}"#;

    /// Spawn a fake Claude API endpoint on `127.0.0.1` that counts every
    /// accepted connection and replies with `FAKE_RESPONSE_BODY`. Returns the
    /// `http://host:port` base URL and a shared call counter.
    async fn spawn_fake_claude_server() -> (String, Arc<AtomicUsize>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback listener");
        let addr = listener.local_addr().expect("read local addr");
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_task = calls.clone();

        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                calls_for_task.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 8192];
                    // Best-effort read of the request; we don't need to parse
                    // it, just drain enough that reqwest sees a response.
                    let _ = socket.read(&mut buf).await;
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        FAKE_RESPONSE_BODY.len(),
                        FAKE_RESPONSE_BODY
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.shutdown().await;
                });
            }
        });

        (format!("http://{addr}"), calls)
    }

    fn test_config(base_url: String) -> ClaudeConfig {
        ClaudeConfig {
            api_key: Some("test-key".to_string()),
            base_url,
            max_retries: 0,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn blocked_egress_never_reaches_the_network() {
        let (base_url, calls) = spawn_fake_claude_server().await;
        let client = ClaudeClient::new(test_config(base_url)).expect("client config is valid");

        let result = client
            .create_message(
                vec![Message::user("raw sensor payload")],
                None,
                EgressClass::RawSensor,
            )
            .await;

        assert!(
            matches!(result, Err(ClaudeError::EgressDenied(_))),
            "expected EgressDenied, got {result:?}"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "a blocked request must never open a connection to the transport"
        );
    }

    #[tokio::test]
    async fn allowed_egress_reaches_the_network_exactly_once() {
        let (base_url, calls) = spawn_fake_claude_server().await;
        let client = ClaudeClient::new(test_config(base_url)).expect("client config is valid");

        let result = client
            .create_message(vec![Message::user("hello")], None, EgressClass::UserText)
            .await;

        assert!(result.is_ok(), "expected Ok, got {result:?}");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "an allowed request must reach the transport exactly once"
        );
    }
}
