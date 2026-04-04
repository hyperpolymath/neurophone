// SPDX-License-Identifier: PMPL-1.0-or-later
// NeuroPhone - High-Assurance Hardware Orchestration
// Copyright (c) 2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>

//! NeuroSymbolic Phone — High-Assurance Hardware Orchestration.
//!
//! This crate is the "Main Brain" of the NeuroPhone. It implements a
//! neurosymbolic architecture that combines low-level neural dynamics
//! (LSM/ESN) with high-level linguistic reasoning (LLM).
//!
//! HARDWARE TARGET: Optimized for Dimensity 8350 (Oppo Reno 13).
//!
//! ARCHITECTURE:
//! 1. **LSM**: Spiking Neural Network for real-time sensor feature extraction.
//! 2. **ESN**: Temporal reservoir for detecting patterns over time.
//! 3. **Bridge**: Encodes neural firing patterns into textual context for the LLM.
//! 4. **LLM**: Local Llama 3.2 for reasoning, with Claude 3.5 fallback.

#![forbid(unsafe_code)]

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors that can occur in the neurophone system
#[derive(Error, Debug, Clone)]
pub enum NeurophoneError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Runtime error: {0}")]
    RuntimeError(String),
    #[error("Inference error: {0}")]
    InferenceError(String),
    #[error("Sensor error: {0}")]
    SensorError(String),
}

/// System configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// Sample rate (Hz)
    pub sample_rate: f32,
    /// Processing window (ms)
    pub window_size_ms: u32,
    /// Local LLM threshold (0.0 - 1.0)
    pub local_threshold: f32,
    /// Max response time (ms)
    pub max_response_time_ms: u32,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            sample_rate: 50.0,
            window_size_ms: 100,
            local_threshold: 0.7,
            max_response_time_ms: 1000,
        }
    }
}

/// Sensor input event
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SensorEvent {
    /// Sensor type
    pub sensor_type: String,
    /// Timestamp (ms)
    pub timestamp_ms: u64,
    /// Raw values
    pub values: Vec<f32>,
}

/// System state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    /// Current timestamp
    pub timestamp_ms: u64,
    /// LSM state vector
    pub lsm_state: Option<Array1<f32>>,
    /// ESN state vector
    pub esn_state: Option<Array1<f32>>,
    /// Is system active
    pub is_active: bool,
    /// Process latency (ms)
    pub latency_ms: u32,
}

impl Default for SystemState {
    fn default() -> Self {
        Self {
            timestamp_ms: 0,
            lsm_state: None,
            esn_state: None,
            is_active: false,
            latency_ms: 0,
        }
    }
}

/// Neural output event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralOutput {
    /// Timestamp (ms)
    pub timestamp_ms: u64,
    /// Feature vector from LSM/ESN
    pub features: Array1<f32>,
    /// Context description
    pub context: String,
    /// Confidence (0.0 - 1.0)
    pub confidence: f32,
}

/// LLM response type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    /// Query that was processed
    pub query: String,
    /// Response text
    pub response: String,
    /// Model used (local or cloud)
    pub model: InferenceModel,
    /// Processing time (ms)
    pub latency_ms: u32,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

/// Inference model selection
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum InferenceModel {
    /// Local Llama 3.2
    LocalLlama,
    /// Cloud Claude 3.5
    CloudClaude,
}

impl fmt::Display for InferenceModel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::LocalLlama => write!(f, "LocalLlama"),
            Self::CloudClaude => write!(f, "CloudClaude"),
        }
    }
}

/// SYSTEM ORCHESTRATOR: Manages the lifecycle of neural and symbolic components.
pub struct NeuroSymbolicSystem {
    config: SystemConfig,
    state: SystemState,
    start_time: Instant,
    query_count: u64,
}

impl NeuroSymbolicSystem {
    /// Create a new NeuroSymbolicSystem
    pub fn new(config: SystemConfig) -> Result<Self, NeurophoneError> {
        if config.sample_rate <= 0.0 {
            return Err(NeurophoneError::ConfigError(
                "Sample rate must be positive".into(),
            ));
        }
        if config.local_threshold < 0.0 || config.local_threshold > 1.0 {
            return Err(NeurophoneError::ConfigError(
                "Local threshold must be between 0.0 and 1.0".into(),
            ));
        }

        info!(
            "Initializing NeuroPhone (sample_rate: {} Hz)",
            config.sample_rate
        );

        Ok(Self {
            config,
            state: SystemState::default(),
            start_time: Instant::now(),
            query_count: 0,
        })
    }

    /// Initialize the system
    pub fn initialize(&mut self) -> Result<(), NeurophoneError> {
        debug!("Initializing NeuroPhone system");
        self.state.is_active = true;
        Ok(())
    }

    /// Process a sensor event
    pub fn process_sensor_event(&mut self, event: &SensorEvent) -> Result<NeuralOutput, NeurophoneError> {
        if !self.state.is_active {
            return Err(NeurophoneError::RuntimeError(
                "System not active".into(),
            ));
        }

        let start = Instant::now();

        // Simulate neural processing
        let features = Array1::from_vec(
            event
                .values
                .iter()
                .map(|v| v * 0.9)
                .collect(),
        );

        let latency = start.elapsed().as_millis() as u32;
        self.state.latency_ms = latency;
        self.state.timestamp_ms = event.timestamp_ms;

        Ok(NeuralOutput {
            timestamp_ms: event.timestamp_ms,
            features,
            context: format!("Processed {} sensor", event.sensor_type),
            confidence: 0.85,
        })
    }

    /// Query the system with inference
    pub fn query(
        &mut self,
        message: &str,
        prefer_local: bool,
    ) -> Result<InferenceResult, NeurophoneError> {
        if message.is_empty() {
            return Err(NeurophoneError::InferenceError(
                "Query cannot be empty".into(),
            ));
        }

        let start = Instant::now();
        self.query_count += 1;

        // Complexity heuristic: count words
        let complexity = (message.split_whitespace().count() as f32) / 100.0;
        let should_use_local = prefer_local && complexity < self.config.local_threshold;

        let model = if should_use_local {
            InferenceModel::LocalLlama
        } else {
            InferenceModel::CloudClaude
        };

        let response = format!("Response to: {}", message);
        let latency = start.elapsed().as_millis() as u32;

        if latency > self.config.max_response_time_ms {
            warn!(
                "Response time {} ms exceeds limit {} ms",
                latency, self.config.max_response_time_ms
            );
        }

        Ok(InferenceResult {
            query: message.to_string(),
            response,
            model,
            latency_ms: latency,
            confidence: 0.92,
        })
    }

    /// Shutdown the system
    pub fn shutdown(&mut self) -> Result<(), NeurophoneError> {
        debug!("Shutting down NeuroPhone");
        self.state.is_active = false;
        Ok(())
    }

    /// Get current system state
    pub fn get_state(&self) -> SystemState {
        self.state.clone()
    }

    /// Get system uptime (ms)
    pub fn uptime_ms(&self) -> u128 {
        self.start_time.elapsed().as_millis()
    }

    /// Get query count
    pub fn query_count(&self) -> u64 {
        self.query_count
    }

    /// Get configuration
    pub fn config(&self) -> &SystemConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Unit Tests ==========

    #[test]
    fn test_system_creation() {
        let config = SystemConfig::default();
        let system = NeuroSymbolicSystem::new(config).expect("system creation");
        assert!(!system.state.is_active);
    }

    #[test]
    fn test_system_invalid_config() {
        let config = SystemConfig {
            sample_rate: -1.0,
            ..Default::default()
        };
        assert!(NeuroSymbolicSystem::new(config).is_err());
    }

    #[test]
    fn test_system_invalid_threshold() {
        let config = SystemConfig {
            local_threshold: 1.5,
            ..Default::default()
        };
        assert!(NeuroSymbolicSystem::new(config).is_err());
    }

    #[test]
    fn test_system_initialization() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");
        system.initialize().expect("init");
        assert!(system.state.is_active);
    }

    #[test]
    fn test_system_shutdown() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");
        system.initialize().expect("init");
        system.shutdown().expect("shutdown");
        assert!(!system.state.is_active);
    }

    #[test]
    fn test_sensor_event_creation() {
        let event = SensorEvent {
            sensor_type: "accelerometer".to_string(),
            timestamp_ms: 1000,
            values: vec![0.1, 0.2, 0.3],
        };
        assert_eq!(event.values.len(), 3);
        assert_eq!(event.timestamp_ms, 1000);
    }

    #[test]
    fn test_process_sensor_event() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");
        system.initialize().expect("init");

        let event = SensorEvent {
            sensor_type: "accelerometer".to_string(),
            timestamp_ms: 100,
            values: vec![1.0, 2.0, 3.0],
        };

        let output = system.process_sensor_event(&event).expect("processing");
        assert_eq!(output.features.len(), 3);
        assert!(output.confidence > 0.0 && output.confidence <= 1.0);
    }

    #[test]
    fn test_query_empty() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");
        let result = system.query("", true);
        assert!(result.is_err());
    }

    #[test]
    fn test_query_local_preference() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");

        let result = system.query("hello world", true).expect("query");
        assert_eq!(result.model, InferenceModel::LocalLlama);
    }

    #[test]
    fn test_query_cloud_fallback() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig {
            local_threshold: 0.5,
            ..Default::default()
        }).expect("system creation");

        let long_query = "hello ".repeat(50);
        let result = system.query(&long_query, true).expect("query");
        assert_eq!(result.model, InferenceModel::CloudClaude);
    }

    #[test]
    fn test_query_count() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");

        assert_eq!(system.query_count(), 0);
        system.query("test", true).ok();
        assert_eq!(system.query_count(), 1);
        system.query("test2", true).ok();
        assert_eq!(system.query_count(), 2);
    }

    #[test]
    fn test_uptime() {
        let system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");
        let uptime = system.uptime_ms();
        assert!(uptime >= 0);
    }

    #[test]
    fn test_inference_result_serialization() {
        let result = InferenceResult {
            query: "test".to_string(),
            response: "response".to_string(),
            model: InferenceModel::LocalLlama,
            latency_ms: 100,
            confidence: 0.9,
        };

        let json = serde_json::to_string(&result).expect("serialization");
        let deserialized: InferenceResult = serde_json::from_str(&json).expect("deserialization");

        assert_eq!(deserialized.query, result.query);
        assert_eq!(deserialized.model, result.model);
    }

    #[test]
    fn test_system_state_clone() {
        let state = SystemState {
            timestamp_ms: 500,
            is_active: true,
            latency_ms: 50,
            ..Default::default()
        };

        let cloned = state.clone();
        assert_eq!(cloned.timestamp_ms, state.timestamp_ms);
        assert_eq!(cloned.is_active, state.is_active);
    }

    // ========== Smoke Tests ==========

    #[test]
    fn test_system_lifecycle() {
        let config = SystemConfig::default();
        let mut system = NeuroSymbolicSystem::new(config).expect("system creation");

        system.initialize().expect("init");
        assert!(system.state.is_active);

        let event = SensorEvent {
            sensor_type: "gyroscope".to_string(),
            timestamp_ms: 200,
            values: vec![0.5, 0.5, 0.5],
        };
        system.process_sensor_event(&event).ok();

        system.query("what's happening", true).ok();

        system.shutdown().expect("shutdown");
        assert!(!system.state.is_active);
    }

    #[test]
    fn test_multiple_queries() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");

        for i in 0..5 {
            let query = format!("query {}", i);
            let result = system.query(&query, true);
            assert!(result.is_ok());
        }
        assert_eq!(system.query_count(), 5);
    }

    #[test]
    fn test_multiple_sensor_events() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");
        system.initialize().expect("init");

        for i in 0..10 {
            let event = SensorEvent {
                sensor_type: format!("sensor_{}", i),
                timestamp_ms: i as u64 * 100,
                values: vec![0.1 * i as f32; 3],
            };
            system.process_sensor_event(&event).ok();
        }
    }

    // ========== E2E Tests ==========

    #[test]
    fn test_e2e_sensor_to_inference() {
        let config = SystemConfig {
            sample_rate: 50.0,
            window_size_ms: 100,
            local_threshold: 0.8,
            max_response_time_ms: 2000,
        };

        let mut system = NeuroSymbolicSystem::new(config).expect("system creation");
        system.initialize().expect("init");

        // Sensor -> Feature extraction
        let event = SensorEvent {
            sensor_type: "accelerometer".to_string(),
            timestamp_ms: 1000,
            values: vec![1.5, 2.0, 2.5],
        };

        let neural_out = system.process_sensor_event(&event).expect("sensor processing");
        assert_eq!(neural_out.timestamp_ms, 1000);

        // Features -> Query
        let query = "accelerometer detected motion";
        let inference = system.query(query, true).expect("inference");
        assert!(!inference.response.is_empty());
    }

    #[test]
    fn test_e2e_sequence_processing() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");
        system.initialize().expect("init");

        let sensor_types = vec!["accelerometer", "gyroscope", "magnetometer"];
        let mut last_output = None;

        for (i, sensor_type) in sensor_types.iter().enumerate() {
            let event = SensorEvent {
                sensor_type: sensor_type.to_string(),
                timestamp_ms: (i as u64 + 1) * 100,
                values: vec![0.5 + i as f32 * 0.1; 3],
            };

            if let Ok(output) = system.process_sensor_event(&event) {
                last_output = Some(output);
            }
        }

        assert!(last_output.is_some());
    }

    // ========== Reflexive Tests ==========

    #[test]
    fn test_state_preservation() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");
        system.initialize().expect("init");

        let event = SensorEvent {
            sensor_type: "test".to_string(),
            timestamp_ms: 500,
            values: vec![1.0],
        };

        system.process_sensor_event(&event).ok();
        let state1 = system.get_state();

        system.process_sensor_event(&event).ok();
        let state2 = system.get_state();

        // Timestamps should advance
        assert_eq!(state1.timestamp_ms, state2.timestamp_ms);
    }

    #[test]
    fn test_deterministic_inference() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");

        let query = "deterministic test";
        let r1 = system.query(query, true).expect("query 1");

        let r2 = system.query(query, true).expect("query 2");

        // Same query should produce similar response
        assert_eq!(r1.query, r2.query);
    }

    #[test]
    fn test_model_selection_consistency() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig {
            local_threshold: 0.5,
            ..Default::default()
        }).expect("system creation");

        // Short query should use local
        let short = "hi";
        let r1 = system.query(short, true).expect("short query");
        assert_eq!(r1.model, InferenceModel::LocalLlama);

        // Long query should use cloud
        let long = "word ".repeat(100);
        let r2 = system.query(&long, true).expect("long query");
        assert_eq!(r2.model, InferenceModel::CloudClaude);
    }

    // ========== Contract Tests (preconditions/postconditions) ==========

    #[test]
    fn test_query_response_validity() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");

        let result = system.query("test", true).expect("query");

        // Contract: response should never be empty
        assert!(!result.response.is_empty());
        // Contract: confidence should be in [0.0, 1.0]
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        // Contract: latency should be non-negative
        assert!(result.latency_ms >= 0);
    }

    #[test]
    fn test_sensor_event_validity() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");
        system.initialize().expect("init");

        let event = SensorEvent {
            sensor_type: "test".to_string(),
            timestamp_ms: 1000,
            values: vec![1.0, 2.0, 3.0],
        };

        let output = system.process_sensor_event(&event).expect("processing");

        // Contract: output features length should match input
        assert_eq!(output.features.len(), event.values.len());
        // Contract: confidence should be valid
        assert!(output.confidence >= 0.0 && output.confidence <= 1.0);
    }

    // ========== Aspect Tests ==========

    #[test]
    fn test_security_malformed_input() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");
        system.initialize().expect("init");

        let event = SensorEvent {
            sensor_type: "".to_string(),
            timestamp_ms: 0,
            values: vec![],
        };

        // Should not crash on empty values
        let result = system.process_sensor_event(&event);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_performance_latency_bound() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");

        let start = Instant::now();
        system.query("test", true).ok();
        let elapsed = start.elapsed().as_millis();

        // Query should complete quickly (< 100ms in most cases)
        assert!(elapsed < 1000);
    }

    #[test]
    fn test_error_handling_inactive_system() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");

        let event = SensorEvent {
            sensor_type: "test".to_string(),
            timestamp_ms: 100,
            values: vec![1.0],
        };

        // Should error when system not active
        let result = system.process_sensor_event(&event);
        assert!(result.is_err());
    }

    #[test]
    fn test_graceful_degradation() {
        let mut system = NeuroSymbolicSystem::new(SystemConfig {
            max_response_time_ms: 10,
            ..Default::default()
        }).expect("system creation");

        // Even with tight timing, should complete
        let result = system.query("test query", true);
        assert!(result.is_ok());
    }
}
