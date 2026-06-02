// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Local LLM Interface — abstraction for on-device inference.
//!
//! Real production wiring uses `llama.cpp` via a separate native crate.
//! This module defines the trait + a deterministic `MockBackend` so the rest
//! of the workspace can be tested end-to-end without a 700 MB model file.

#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum LlmError {
    #[error("model not loaded")]
    NotLoaded,
    #[error("invalid prompt: {0}")]
    InvalidPrompt(String),
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("inference failed: {0}")]
    InferenceFailed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub model_path: String,
    pub n_threads: u32,
    pub context_size: u32,
    pub max_tokens: u32,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model_path: "/data/local/tmp/llama-3.2-1b-q4_k_m.gguf".to_string(),
            n_threads: 4,
            context_size: 2048,
            max_tokens: 256,
        }
    }
}

impl LlmConfig {
    pub fn validate(&self) -> Result<(), LlmError> {
        if self.n_threads == 0 {
            return Err(LlmError::InvalidConfig("n_threads must be > 0".into()));
        }
        if self.context_size == 0 || self.max_tokens == 0 {
            return Err(LlmError::InvalidConfig(
                "context_size/max_tokens must be > 0".into(),
            ));
        }
        if self.max_tokens > self.context_size {
            return Err(LlmError::InvalidConfig(
                "max_tokens cannot exceed context_size".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub text: String,
    pub tokens_emitted: u32,
    pub elapsed_ms: u64,
}

/// Backend trait — real impl wraps `llama.cpp`, mock impl is deterministic.
pub trait LlmBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn is_loaded(&self) -> bool;
    fn load(&mut self) -> Result<(), LlmError>;
    fn generate(&mut self, prompt: &str, max_tokens: u32) -> Result<LlmResponse, LlmError>;
    fn unload(&mut self);
}

/// Deterministic mock backend for tests + offline development.
#[derive(Debug)]
pub struct MockBackend {
    config: LlmConfig,
    loaded: bool,
    calls: AtomicU64,
}

impl MockBackend {
    pub fn new(config: LlmConfig) -> Result<Self, LlmError> {
        config.validate()?;
        Ok(Self {
            config,
            loaded: false,
            calls: AtomicU64::new(0),
        })
    }

    pub fn config(&self) -> &LlmConfig {
        &self.config
    }
    pub fn call_count(&self) -> u64 {
        self.calls.load(Ordering::SeqCst)
    }
}

impl LlmBackend for MockBackend {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    fn load(&mut self) -> Result<(), LlmError> {
        self.loaded = true;
        Ok(())
    }

    fn generate(&mut self, prompt: &str, max_tokens: u32) -> Result<LlmResponse, LlmError> {
        if !self.loaded {
            return Err(LlmError::NotLoaded);
        }
        if prompt.trim().is_empty() {
            return Err(LlmError::InvalidPrompt("empty prompt".into()));
        }
        if max_tokens == 0 {
            return Err(LlmError::InvalidConfig("max_tokens=0".into()));
        }
        self.calls.fetch_add(1, Ordering::SeqCst);
        let words: Vec<&str> = prompt
            .split_whitespace()
            .take(max_tokens as usize)
            .collect();
        let text = format!("(local-llama-mock) echo: {}", words.join(" "));
        let tokens_emitted = words.len() as u32;
        Ok(LlmResponse {
            text,
            tokens_emitted,
            elapsed_ms: 1,
        })
    }

    fn unload(&mut self) {
        self.loaded = false;
    }
}

pub fn hello() -> &'static str {
    "llm"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation() {
        assert!(LlmConfig::default().validate().is_ok());
        let bad = LlmConfig {
            n_threads: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
        let bad2 = LlmConfig {
            max_tokens: 9999,
            context_size: 100,
            ..Default::default()
        };
        assert!(bad2.validate().is_err());
    }

    #[test]
    fn mock_lifecycle() {
        let mut b = MockBackend::new(LlmConfig::default()).unwrap();
        assert!(!b.is_loaded());
        b.load().unwrap();
        assert!(b.is_loaded());
        let r = b.generate("hello there friend", 8).unwrap();
        assert!(r.text.contains("hello"));
        assert_eq!(r.tokens_emitted, 3);
        b.unload();
        assert!(!b.is_loaded());
    }

    #[test]
    fn generate_rejects_unloaded() {
        let mut b = MockBackend::new(LlmConfig::default()).unwrap();
        assert!(matches!(b.generate("x", 1), Err(LlmError::NotLoaded)));
    }

    #[test]
    fn generate_rejects_empty_prompt() {
        let mut b = MockBackend::new(LlmConfig::default()).unwrap();
        b.load().unwrap();
        assert!(matches!(
            b.generate("   ", 5),
            Err(LlmError::InvalidPrompt(_))
        ));
    }

    #[test]
    fn generate_rejects_zero_tokens() {
        let mut b = MockBackend::new(LlmConfig::default()).unwrap();
        b.load().unwrap();
        assert!(matches!(
            b.generate("ok", 0),
            Err(LlmError::InvalidConfig(_))
        ));
    }

    #[test]
    fn call_count_increments() {
        let mut b = MockBackend::new(LlmConfig::default()).unwrap();
        b.load().unwrap();
        for _ in 0..3 {
            b.generate("a b c", 5).unwrap();
        }
        assert_eq!(b.call_count(), 3);
    }
}
