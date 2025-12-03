//! Bridge - State Encoding Between Neural Components and LLM
//!
//! The Bridge is responsible for:
//! 1. Integrating states from LSM and ESN into a unified representation
//! 2. Encoding neural states into formats suitable for LLM context
//! 3. Decoding LLM outputs back to control signals
//! 4. Maintaining temporal context and attention mechanisms

use chrono::{DateTime, Utc};
use ndarray::{s, Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use thiserror::Error;
use tracing::{debug, trace};

/// Bridge errors
#[derive(Error, Debug)]
pub enum BridgeError {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("Encoding error: {0}")]
    EncodingError(String),
    #[error("No state available")]
    NoState,
}

/// Bridge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// LSM state dimension
    pub lsm_dim: usize,
    /// ESN state dimension
    pub esn_dim: usize,
    /// LLM embedding dimension
    pub llm_dim: usize,
    /// History size for temporal context
    pub history_size: usize,
    /// Sparse encoding threshold
    pub sparse_threshold: f32,
    /// Temporal window for context (ms)
    pub temporal_window_ms: f32,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            lsm_dim: 512,
            esn_dim: 300,
            llm_dim: 2048,
            history_size: 100,
            sparse_threshold: 0.1,
            temporal_window_ms: 1000.0,
        }
    }
}

/// Integrated state from both reservoirs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedState {
    /// Combined state vector
    pub vector: Array1<f32>,
    /// Timestamp
    pub timestamp: f64,
    /// Salience score (how significant is this state)
    pub salience: f32,
    /// Dominant features (sparse representation)
    pub dominant_features: Vec<(usize, f32)>,
    /// State category (if classified)
    pub category: Option<String>,
}

/// Temporal pattern detected in state history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPattern {
    /// Pattern name/type
    pub name: String,
    /// Confidence score
    pub confidence: f32,
    /// Duration (ms)
    pub duration_ms: f32,
    /// Associated state indices
    pub state_indices: Vec<usize>,
}

/// Encoding of neural state for LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateEncoding {
    /// Textual description of current state
    pub description: String,
    /// Key-value features
    pub features: Vec<(String, f32)>,
    /// Detected patterns
    pub patterns: Vec<String>,
    /// Suggested context for LLM
    pub context: String,
    /// Urgency level (0-1)
    pub urgency: f32,
}

/// The Bridge component
pub struct Bridge {
    config: BridgeConfig,

    /// State history
    history: VecDeque<IntegratedState>,

    /// LSM projection matrix (lsm_dim -> internal_dim)
    lsm_projection: Array2<f32>,

    /// ESN projection matrix (esn_dim -> internal_dim)
    esn_projection: Array2<f32>,

    /// LLM projection matrix (internal_dim -> llm_dim)
    llm_projection: Array2<f32>,

    /// Internal representation dimension
    internal_dim: usize,

    /// Running statistics for normalization
    mean: Array1<f32>,
    variance: Array1<f32>,
    count: usize,

    /// Feature names for interpretability
    feature_names: Vec<String>,

    /// Detected patterns cache
    recent_patterns: Vec<TemporalPattern>,

    /// Current simulation time
    current_time: f64,
}

impl Bridge {
    /// Create a new Bridge
    pub fn new(config: BridgeConfig) -> Self {
        let internal_dim = 128; // Fixed internal representation

        // Initialize random projections (would be trained in production)
        let mut rng = rand::thread_rng();
        use rand::Rng;

        let lsm_projection = Array2::from_shape_fn((config.lsm_dim, internal_dim), |_| {
            rng.gen::<f32>() * 0.1 - 0.05
        });

        let esn_projection = Array2::from_shape_fn((config.esn_dim, internal_dim), |_| {
            rng.gen::<f32>() * 0.1 - 0.05
        });

        let llm_projection = Array2::from_shape_fn((internal_dim * 2, config.llm_dim), |_| {
            rng.gen::<f32>() * 0.1 - 0.05
        });

        // Initialize feature names
        let feature_names: Vec<String> = (0..internal_dim * 2)
            .map(|i| {
                if i < internal_dim {
                    format!("lsm_f{}", i)
                } else {
                    format!("esn_f{}", i - internal_dim)
                }
            })
            .collect();

        Self {
            config,
            history: VecDeque::with_capacity(100),
            lsm_projection,
            esn_projection,
            llm_projection,
            internal_dim,
            mean: Array1::zeros(internal_dim * 2),
            variance: Array1::ones(internal_dim * 2),
            count: 0,
            feature_names,
            recent_patterns: Vec::new(),
            current_time: 0.0,
        }
    }

    /// Integrate states from LSM and ESN
    pub fn integrate_states(
        &mut self,
        lsm_state: &Array1<f32>,
        esn_state: &Array1<f32>,
        time_ms: f32,
    ) -> Result<IntegratedState, BridgeError> {
        // Handle dimension mismatches with padding/truncation
        let lsm_processed = self.process_input(lsm_state, self.config.lsm_dim);
        let esn_processed = self.process_input(esn_state, self.config.esn_dim);

        // Project to internal dimensions
        let lsm_internal = self.lsm_projection.t().dot(&lsm_processed);
        let esn_internal = self.esn_projection.t().dot(&esn_processed);

        // Combine
        let mut combined = Array1::zeros(self.internal_dim * 2);
        combined.slice_mut(s![..self.internal_dim]).assign(&lsm_internal);
        combined.slice_mut(s![self.internal_dim..]).assign(&esn_internal);

        // Update running statistics
        self.update_statistics(&combined);

        // Normalize
        let normalized = self.normalize(&combined);

        // Calculate salience (how different from recent history)
        let salience = self.calculate_salience(&normalized);

        // Extract dominant features
        let dominant_features = self.extract_dominant_features(&normalized);

        // Create integrated state
        let state = IntegratedState {
            vector: normalized,
            timestamp: time_ms as f64,
            salience,
            dominant_features,
            category: None,
        };

        // Add to history
        self.history.push_back(state.clone());
        if self.history.len() > self.config.history_size {
            self.history.pop_front();
        }

        self.current_time = time_ms as f64;

        // Detect patterns periodically
        if self.history.len() % 10 == 0 {
            self.detect_patterns();
        }

        trace!("Integrated state: salience={:.3}, features={}", salience, state.dominant_features.len());

        Ok(state)
    }

    /// Process input with dimension handling
    fn process_input(&self, input: &Array1<f32>, expected_dim: usize) -> Array1<f32> {
        if input.len() == expected_dim {
            input.clone()
        } else if input.len() > expected_dim {
            input.slice(s![..expected_dim]).to_owned()
        } else {
            let mut padded = Array1::zeros(expected_dim);
            padded.slice_mut(s![..input.len()]).assign(input);
            padded
        }
    }

    /// Update running mean and variance
    fn update_statistics(&mut self, x: &Array1<f32>) {
        self.count += 1;
        let n = self.count as f32;

        // Welford's online algorithm
        let delta = x - &self.mean;
        self.mean = &self.mean + &delta / n;
        let delta2 = x - &self.mean;
        self.variance = &self.variance + &(delta * delta2);
    }

    /// Normalize using running statistics
    fn normalize(&self, x: &Array1<f32>) -> Array1<f32> {
        if self.count < 2 {
            return x.clone();
        }

        let std = (&self.variance / (self.count as f32 - 1.0)).mapv(|v| v.sqrt().max(1e-8));
        (x - &self.mean) / std
    }

    /// Calculate salience (novelty) of current state
    fn calculate_salience(&self, state: &Array1<f32>) -> f32 {
        if self.history.is_empty() {
            return 1.0;
        }

        // Compare to recent states
        let n_compare = self.history.len().min(10);
        let recent: Vec<_> = self.history.iter().rev().take(n_compare).collect();

        let mut total_dist = 0.0;
        for hist in &recent {
            let diff = state - &hist.vector;
            let dist = diff.mapv(|x| x * x).sum().sqrt();
            total_dist += dist;
        }

        let avg_dist = total_dist / n_compare as f32;

        // Sigmoid normalization
        1.0 / (1.0 + (-avg_dist + 1.0).exp())
    }

    /// Extract features above threshold
    fn extract_dominant_features(&self, state: &Array1<f32>) -> Vec<(usize, f32)> {
        let mut features: Vec<_> = state
            .iter()
            .enumerate()
            .filter(|(_, &v)| v.abs() > self.config.sparse_threshold)
            .map(|(i, &v)| (i, v))
            .collect();

        // Sort by absolute value
        features.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        // Keep top 20
        features.truncate(20);
        features
    }

    /// Detect temporal patterns in history
    fn detect_patterns(&mut self) {
        self.recent_patterns.clear();

        if self.history.len() < 10 {
            return;
        }

        // Simple pattern detection: look for repeated high-salience events
        let high_salience: Vec<_> = self.history
            .iter()
            .enumerate()
            .filter(|(_, s)| s.salience > 0.7)
            .collect();

        if high_salience.len() >= 2 {
            self.recent_patterns.push(TemporalPattern {
                name: "high_activity_burst".to_string(),
                confidence: 0.8,
                duration_ms: self.config.temporal_window_ms,
                state_indices: high_salience.iter().map(|(i, _)| *i).collect(),
            });
        }

        // Detect rhythmic patterns (simplified)
        if self.history.len() >= 20 {
            let recent: Vec<f32> = self.history
                .iter()
                .rev()
                .take(20)
                .map(|s| s.salience)
                .collect();

            // Check for oscillation
            let mut sign_changes = 0;
            let mean: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
            let deviations: Vec<f32> = recent.iter().map(|x| x - mean).collect();

            for i in 1..deviations.len() {
                if deviations[i] * deviations[i - 1] < 0.0 {
                    sign_changes += 1;
                }
            }

            if sign_changes >= 6 {
                self.recent_patterns.push(TemporalPattern {
                    name: "oscillation".to_string(),
                    confidence: sign_changes as f32 / 19.0,
                    duration_ms: self.config.temporal_window_ms,
                    state_indices: (0..20).collect(),
                });
            }
        }
    }

    /// Encode current state for LLM consumption
    pub fn encode_for_llm(&self) -> Result<StateEncoding, BridgeError> {
        let current = self.history.back().ok_or(BridgeError::NoState)?;

        // Generate description
        let description = self.generate_description(current);

        // Extract key features
        let features: Vec<(String, f32)> = current
            .dominant_features
            .iter()
            .take(10)
            .map(|(idx, val)| {
                let name = self.feature_names.get(*idx)
                    .cloned()
                    .unwrap_or_else(|| format!("f{}", idx));
                (name, *val)
            })
            .collect();

        // Pattern descriptions
        let patterns: Vec<String> = self.recent_patterns
            .iter()
            .map(|p| format!("{} (conf: {:.2})", p.name, p.confidence))
            .collect();

        // Generate context
        let context = self.generate_context();

        // Calculate urgency
        let urgency = self.calculate_urgency(current);

        Ok(StateEncoding {
            description,
            features,
            patterns,
            context,
            urgency,
        })
    }

    /// Generate human-readable description of state
    fn generate_description(&self, state: &IntegratedState) -> String {
        let activity_level = if state.salience > 0.8 {
            "highly active"
        } else if state.salience > 0.5 {
            "moderately active"
        } else if state.salience > 0.2 {
            "low activity"
        } else {
            "quiescent"
        };

        let feature_summary = if state.dominant_features.is_empty() {
            "no dominant features".to_string()
        } else {
            let top: Vec<_> = state.dominant_features
                .iter()
                .take(3)
                .map(|(i, v)| format!("f{}={:.2}", i, v))
                .collect();
            top.join(", ")
        };

        format!(
            "Neural state: {} (salience: {:.2}). Top features: {}",
            activity_level, state.salience, feature_summary
        )
    }

    /// Generate LLM context string
    fn generate_context(&self) -> String {
        let mut parts = Vec::new();

        // Recent activity summary
        if self.history.len() >= 5 {
            let recent_salience: f32 = self.history
                .iter()
                .rev()
                .take(5)
                .map(|s| s.salience)
                .sum::<f32>() / 5.0;

            parts.push(format!("Recent activity level: {:.2}", recent_salience));
        }

        // Pattern summary
        if !self.recent_patterns.is_empty() {
            let pattern_names: Vec<_> = self.recent_patterns
                .iter()
                .map(|p| p.name.as_str())
                .collect();
            parts.push(format!("Detected patterns: {}", pattern_names.join(", ")));
        }

        // Temporal context
        parts.push(format!("Time window: {:.0}ms", self.config.temporal_window_ms));

        parts.join(". ")
    }

    /// Calculate urgency based on state
    fn calculate_urgency(&self, state: &IntegratedState) -> f32 {
        let mut urgency = state.salience * 0.5;

        // High activity patterns increase urgency
        for pattern in &self.recent_patterns {
            if pattern.name == "high_activity_burst" {
                urgency += 0.3 * pattern.confidence;
            }
        }

        urgency.min(1.0)
    }

    /// Generate context string for LLM prompting
    pub fn generate_llm_context(&self) -> String {
        match self.encode_for_llm() {
            Ok(encoding) => {
                let mut context = String::new();
                context.push_str("[NEURAL_STATE]\n");
                context.push_str(&format!("Description: {}\n", encoding.description));
                context.push_str(&format!("Context: {}\n", encoding.context));
                context.push_str(&format!("Urgency: {:.2}\n", encoding.urgency));

                if !encoding.patterns.is_empty() {
                    context.push_str(&format!("Patterns: {}\n", encoding.patterns.join(", ")));
                }

                if !encoding.features.is_empty() {
                    context.push_str("Features:\n");
                    for (name, value) in encoding.features.iter().take(5) {
                        context.push_str(&format!("  - {}: {:.3}\n", name, value));
                    }
                }

                context.push_str("[/NEURAL_STATE]\n");
                context
            }
            Err(_) => "[NEURAL_STATE]\nNo state available\n[/NEURAL_STATE]\n".to_string()
        }
    }

    /// Get projection to LLM embedding space
    pub fn project_to_llm(&self, state: &Array1<f32>) -> Array1<f32> {
        // Ensure correct dimension
        let processed = self.process_input(state, self.internal_dim * 2);
        self.llm_projection.t().dot(&processed)
    }

    /// Reset the bridge
    pub fn reset(&mut self) {
        self.history.clear();
        self.mean = Array1::zeros(self.internal_dim * 2);
        self.variance = Array1::ones(self.internal_dim * 2);
        self.count = 0;
        self.recent_patterns.clear();
        self.current_time = 0.0;
    }

    /// Get history length
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Get recent states
    pub fn get_recent_states(&self, n: usize) -> Vec<&IntegratedState> {
        self.history.iter().rev().take(n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let config = BridgeConfig::default();
        let bridge = Bridge::new(config);
        assert_eq!(bridge.history_len(), 0);
    }

    #[test]
    fn test_state_integration() {
        let config = BridgeConfig {
            lsm_dim: 100,
            esn_dim: 50,
            ..Default::default()
        };
        let mut bridge = Bridge::new(config);

        let lsm_state = Array1::from_vec(vec![0.5; 100]);
        let esn_state = Array1::from_vec(vec![0.3; 50]);

        let result = bridge.integrate_states(&lsm_state, &esn_state, 0.0);
        assert!(result.is_ok());

        let state = result.unwrap();
        assert!(state.salience >= 0.0 && state.salience <= 1.0);
    }

    #[test]
    fn test_encoding() {
        let config = BridgeConfig {
            lsm_dim: 100,
            esn_dim: 50,
            ..Default::default()
        };
        let mut bridge = Bridge::new(config);

        // Add some states
        for i in 0..10 {
            let lsm = Array1::from_vec(vec![i as f32 * 0.1; 100]);
            let esn = Array1::from_vec(vec![i as f32 * 0.05; 50]);
            bridge.integrate_states(&lsm, &esn, i as f32 * 10.0).unwrap();
        }

        let encoding = bridge.encode_for_llm().unwrap();
        assert!(!encoding.description.is_empty());
        assert!(encoding.urgency >= 0.0 && encoding.urgency <= 1.0);
    }

    #[test]
    fn test_llm_context_generation() {
        let config = BridgeConfig {
            lsm_dim: 100,
            esn_dim: 50,
            ..Default::default()
        };
        let mut bridge = Bridge::new(config);

        let lsm = Array1::from_vec(vec![0.5; 100]);
        let esn = Array1::from_vec(vec![0.3; 50]);
        bridge.integrate_states(&lsm, &esn, 0.0).unwrap();

        let context = bridge.generate_llm_context();
        assert!(context.contains("[NEURAL_STATE]"));
        assert!(context.contains("[/NEURAL_STATE]"));
    }
}
