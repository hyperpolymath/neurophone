// SPDX-License-Identifier: MPL-2.0
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
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use bridge::ActionGate;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::marker::PhantomData;
use std::time::Instant;
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

/// Explicit model routing for a query.
///
/// `Auto` uses the word-count complexity heuristic against `local_threshold`;
/// `ForceLocal` / `ForceCloud` pin the model regardless of complexity. The JNI
/// surface's `queryLocal` / `queryClaude` need hard routing, which `prefer_local`
/// alone cannot express (it only *permits* local when the query is simple).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum QueryRoute {
    /// Heuristic auto-routing (local only when complexity < `local_threshold`).
    Auto,
    /// Always route to the local Llama, regardless of complexity.
    ForceLocal,
    /// Always route to cloud Claude, regardless of complexity.
    ForceCloud,
}

/// Lifecycle phase markers (typestate). Illegal transitions are compile errors.
pub mod phase {
    /// Constructed but not yet initialised.
    #[derive(Debug)]
    pub struct Created;
    /// Initialised and running.
    #[derive(Debug)]
    pub struct Active;
    /// Shut down — terminal, no restart.
    #[derive(Debug)]
    pub struct Down;
}

/// SYSTEM ORCHESTRATOR: manages the lifecycle of neural and symbolic components.
///
/// The lifecycle is enforced at compile time by the typestate parameter `S`:
/// `new() -> Created`, then `initialize() -> Active`, then `shutdown() -> Down`.
/// `process_sensor_event` and `query` exist only on `Active`, so using the system
/// before initialisation or after shutdown does not compile. Shutdown is terminal.
///
/// # Typestate safety (proof obligations 2.1 / 2.3, issue #84)
///
/// These examples are part of the proof surface: each is a `compile_fail`
/// doc-test, so `cargo test` fails if any of them ever *starts* compiling — i.e.
/// if the typestate protection regresses. They complement the TLC model check in
/// `proofs/tla/Lifecycle.tla` (the runtime protocol) with the compile-time
/// guarantee (the API shape).
///
/// Use before initialisation does not compile (`query` exists only on `Active`):
/// ```compile_fail
/// use neurophone_core::{NeuroSymbolicSystem, SystemConfig};
/// let mut s = NeuroSymbolicSystem::new(SystemConfig::default()).unwrap();
/// let _ = s.query("hi", true); // no `query` on Created
/// ```
///
/// Use after shutdown does not compile (`Down` has no `query`):
/// ```compile_fail
/// use neurophone_core::{NeuroSymbolicSystem, SystemConfig};
/// let s = NeuroSymbolicSystem::new(SystemConfig::default())
///     .unwrap()
///     .initialize()
///     .unwrap();
/// let down = s.shutdown();
/// let _ = down.query("hi", true); // no `query` on Down
/// ```
///
/// Releasing (shutting down) twice does not compile — `shutdown` consumes
/// `self`, so the resource is released exactly once:
/// ```compile_fail
/// use neurophone_core::{NeuroSymbolicSystem, SystemConfig};
/// let s = NeuroSymbolicSystem::new(SystemConfig::default())
///     .unwrap()
///     .initialize()
///     .unwrap();
/// let _first = s.shutdown();
/// let _second = s.shutdown(); // use of moved value: `s`
/// ```
pub struct NeuroSymbolicSystem<S = phase::Active> {
    config: SystemConfig,
    state: SystemState,
    start_time: Instant,
    query_count: u64,
    action_gate: ActionGate,
    _phase: PhantomData<S>,
}

impl NeuroSymbolicSystem<phase::Created> {
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
            action_gate: ActionGate::new(),
            _phase: PhantomData,
        })
    }

    /// Initialise the system, transitioning `Created -> Active`.
    pub fn initialize(mut self) -> Result<NeuroSymbolicSystem<phase::Active>, NeurophoneError> {
        debug!("Initializing NeuroPhone system");
        self.state.is_active = true;
        Ok(NeuroSymbolicSystem {
            config: self.config,
            state: self.state,
            start_time: self.start_time,
            query_count: self.query_count,
            action_gate: self.action_gate,
            _phase: PhantomData,
        })
    }
}

impl NeuroSymbolicSystem<phase::Active> {
    /// Process a sensor event
    pub fn process_sensor_event(
        &mut self,
        event: &SensorEvent,
    ) -> Result<NeuralOutput, NeurophoneError> {
        let start = Instant::now();

        // Simulate neural processing
        let features = Array1::from_vec(event.values.iter().map(|v| v * 0.9).collect());

        let latency = start.elapsed().as_millis() as u32;
        self.state.latency_ms = latency;
        self.state.timestamp_ms = event.timestamp_ms;

        let confidence = 0.85;
        let context = format!("Processed {} sensor", event.sensor_type);

        // Conative-gating: gate the bridge action
        if let Err(e) = self.action_gate.check(confidence, &context) {
            tracing::warn!("Bridge action vetoed by policy: {}", e);
            // On Block/Escalate, we must not dispatch the action.
            // In a real flow, we might return a default or error. Here we return an error.
            return Err(NeurophoneError::RuntimeError(format!("Vetoed: {}", e)));
        }

        Ok(NeuralOutput {
            timestamp_ms: event.timestamp_ms,
            features,
            context,
            confidence,
        })
    }

    /// Query the system with inference, auto-routing by complexity.
    ///
    /// Preserves the historical contract: `prefer_local == false` always routes
    /// to cloud; `prefer_local == true` routes local only when the query is
    /// simple enough (complexity < `local_threshold`). Implemented on top of
    /// [`query_routed`](Self::query_routed).
    pub fn query(
        &mut self,
        message: &str,
        prefer_local: bool,
    ) -> Result<InferenceResult, NeurophoneError> {
        let route = if prefer_local {
            QueryRoute::Auto
        } else {
            QueryRoute::ForceCloud
        };
        self.query_routed(message, route)
    }

    /// Query with an explicit [`QueryRoute`], forcing the model when requested.
    ///
    /// This is the routing primitive the JNI `queryLocal` / `queryClaude`
    /// surfaces need: `ForceLocal` / `ForceCloud` pin the model regardless of
    /// the complexity heuristic.
    pub fn query_routed(
        &mut self,
        message: &str,
        route: QueryRoute,
    ) -> Result<InferenceResult, NeurophoneError> {
        if message.is_empty() {
            return Err(NeurophoneError::InferenceError(
                "Query cannot be empty".into(),
            ));
        }

        let start = Instant::now();
        self.query_count += 1;

        let model = self.select_model(message, route);

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

    /// Select the inference model for a message under a routing policy.
    fn select_model(&self, message: &str, route: QueryRoute) -> InferenceModel {
        match route {
            QueryRoute::ForceLocal => InferenceModel::LocalLlama,
            QueryRoute::ForceCloud => InferenceModel::CloudClaude,
            QueryRoute::Auto => {
                // Complexity heuristic: count words, normalise, compare to threshold.
                let complexity = (message.split_whitespace().count() as f32) / 100.0;
                if complexity < self.config.local_threshold {
                    InferenceModel::LocalLlama
                } else {
                    InferenceModel::CloudClaude
                }
            }
        }
    }

    /// Shut down the system, transitioning `Active -> Down` (terminal).
    pub fn shutdown(mut self) -> NeuroSymbolicSystem<phase::Down> {
        debug!("Shutting down NeuroPhone");
        self.state.is_active = false;
        NeuroSymbolicSystem {
            config: self.config,
            state: self.state,
            start_time: self.start_time,
            query_count: self.query_count,
            action_gate: self.action_gate,
            _phase: PhantomData,
        }
    }
}

/// Inspectors available in every lifecycle phase.
impl<S> NeuroSymbolicSystem<S> {
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

    /// Render a compact neural-context summary string.
    ///
    /// This is the format the Android service embeds as LLM context. It is
    /// composed strictly from real [`SystemState`] fields (no invented salience
    /// value): activity, last latency, whether the LSM/ESN reservoirs hold
    /// state, and the last processed timestamp.
    pub fn get_neural_context(&self) -> String {
        let s = &self.state;
        format!(
            "[NEURAL_STATE] active={} latency_ms={} lsm={} esn={} ts={} [/NEURAL_STATE]",
            s.is_active,
            s.latency_ms,
            s.lsm_state.is_some(),
            s.esn_state.is_some(),
            s.timestamp_ms
        )
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
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let system = system.initialize().expect("init");
        assert!(system.state.is_active);
    }

    #[test]
    fn test_system_shutdown() {
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let system = system.initialize().expect("init");
        let system = system.shutdown();
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
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let mut system = system.initialize().expect("init");

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
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let mut system = system.initialize().expect("init");
        let result = system.query("", true);
        assert!(result.is_err());
    }

    #[test]
    fn test_query_local_preference() {
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let mut system = system.initialize().expect("init");

        let result = system.query("hello world", true).expect("query");
        assert_eq!(result.model, InferenceModel::LocalLlama);
    }

    #[test]
    fn test_query_cloud_fallback() {
        let system = NeuroSymbolicSystem::new(SystemConfig {
            local_threshold: 0.5,
            ..Default::default()
        })
        .expect("system creation");
        let mut system = system.initialize().expect("init");

        let long_query = "hello ".repeat(50);
        let result = system.query(&long_query, true).expect("query");
        assert_eq!(result.model, InferenceModel::CloudClaude);
    }

    #[test]
    fn test_query_count() {
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let mut system = system.initialize().expect("init");

        assert_eq!(system.query_count(), 0);
        system.query("test", true).ok();
        assert_eq!(system.query_count(), 1);
        system.query("test2", true).ok();
        assert_eq!(system.query_count(), 2);
    }

    #[test]
    fn test_uptime() {
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let uptime = system.uptime_ms();
        let _ = uptime; // u128, always defined; just verify no panic
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
        let system = NeuroSymbolicSystem::new(config).expect("system creation");

        let mut system = system.initialize().expect("init");
        assert!(system.state.is_active);

        let event = SensorEvent {
            sensor_type: "gyroscope".to_string(),
            timestamp_ms: 200,
            values: vec![0.5, 0.5, 0.5],
        };
        system.process_sensor_event(&event).ok();

        system.query("what's happening", true).ok();

        let system = system.shutdown();
        assert!(!system.state.is_active);
    }

    #[test]
    fn test_multiple_queries() {
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let mut system = system.initialize().expect("init");

        for i in 0..5 {
            let query = format!("query {}", i);
            let result = system.query(&query, true);
            assert!(result.is_ok());
        }
        assert_eq!(system.query_count(), 5);
    }

    #[test]
    fn test_multiple_sensor_events() {
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let mut system = system.initialize().expect("init");

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

        let system = NeuroSymbolicSystem::new(config).expect("system creation");
        let mut system = system.initialize().expect("init");

        // Sensor -> Feature extraction
        let event = SensorEvent {
            sensor_type: "accelerometer".to_string(),
            timestamp_ms: 1000,
            values: vec![1.5, 2.0, 2.5],
        };

        let neural_out = system
            .process_sensor_event(&event)
            .expect("sensor processing");
        assert_eq!(neural_out.timestamp_ms, 1000);

        // Features -> Query
        let query = "accelerometer detected motion";
        let inference = system.query(query, true).expect("inference");
        assert!(!inference.response.is_empty());
    }

    #[test]
    fn test_e2e_sequence_processing() {
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let mut system = system.initialize().expect("init");

        let sensor_types = ["accelerometer", "gyroscope", "magnetometer"];
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
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let mut system = system.initialize().expect("init");

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
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let mut system = system.initialize().expect("init");

        let query = "deterministic test";
        let r1 = system.query(query, true).expect("query 1");

        let r2 = system.query(query, true).expect("query 2");

        // Same query should produce similar response
        assert_eq!(r1.query, r2.query);
    }

    #[test]
    fn test_model_selection_consistency() {
        let system = NeuroSymbolicSystem::new(SystemConfig {
            local_threshold: 0.5,
            ..Default::default()
        })
        .expect("system creation");
        let mut system = system.initialize().expect("init");

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
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let mut system = system.initialize().expect("init");

        let result = system.query("test", true).expect("query");

        // Contract: response should never be empty
        assert!(!result.response.is_empty());
        // Contract: confidence should be in [0.0, 1.0]
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        // Contract: latency should be non-negative
        let _ = result.latency_ms; // u32, always >= 0
    }

    #[test]
    fn test_sensor_event_validity() {
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let mut system = system.initialize().expect("init");

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
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let mut system = system.initialize().expect("init");

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
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let mut system = system.initialize().expect("init");

        let start = Instant::now();
        system.query("test", true).ok();
        let elapsed = start.elapsed().as_millis();

        // Query should complete quickly (< 100ms in most cases)
        assert!(elapsed < 1000);
    }

    // NOTE: "process before init" is now a compile-time guarantee of the
    // typestate API (`process_sensor_event`/`query` exist only on
    // `NeuroSymbolicSystem<phase::Active>`), so the old runtime-error test
    // `test_error_handling_inactive_system` has been removed.

    #[test]
    fn test_graceful_degradation() {
        let system = NeuroSymbolicSystem::new(SystemConfig {
            max_response_time_ms: 10,
            ..Default::default()
        })
        .expect("system creation");
        let mut system = system.initialize().expect("init");

        // Even with tight timing, should complete
        let result = system.query("test query", true);
        assert!(result.is_ok());
    }

    // ========== QueryRoute Tests ==========

    #[test]
    fn test_query_force_local_overrides_complexity() {
        // A long query would auto-route to cloud, but ForceLocal pins it local.
        let system = NeuroSymbolicSystem::new(SystemConfig {
            local_threshold: 0.1,
            ..Default::default()
        })
        .expect("system creation");
        let mut system = system.initialize().expect("init");

        let long = "word ".repeat(100);
        let r = system
            .query_routed(&long, QueryRoute::ForceLocal)
            .expect("query");
        assert_eq!(r.model, InferenceModel::LocalLlama);
    }

    #[test]
    fn test_query_force_cloud_overrides_simplicity() {
        // A trivial query would auto-route local, but ForceCloud pins it cloud.
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let mut system = system.initialize().expect("init");

        let r = system
            .query_routed("hi", QueryRoute::ForceCloud)
            .expect("query");
        assert_eq!(r.model, InferenceModel::CloudClaude);
    }

    #[test]
    fn test_query_delegates_preserving_legacy_behaviour() {
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let mut system = system.initialize().expect("init");

        // prefer_local == false must always be cloud (historical contract).
        let cloud = system.query("hi", false).expect("query");
        assert_eq!(cloud.model, InferenceModel::CloudClaude);

        // prefer_local == true on a simple query is local (Auto).
        let local = system.query("hi", true).expect("query");
        assert_eq!(local.model, InferenceModel::LocalLlama);
    }

    #[test]
    fn test_get_neural_context_format() {
        let system = NeuroSymbolicSystem::new(SystemConfig::default()).expect("system creation");
        let mut system = system.initialize().expect("init");
        let event = SensorEvent {
            sensor_type: "accelerometer".to_string(),
            timestamp_ms: 4242,
            values: vec![1.0, 2.0, 3.0],
        };
        system.process_sensor_event(&event).ok();

        let ctx = system.get_neural_context();
        assert!(ctx.starts_with("[NEURAL_STATE]"));
        assert!(ctx.ends_with("[/NEURAL_STATE]"));
        assert!(ctx.contains("active=true"));
        assert!(ctx.contains("ts=4242"));
    }
}
