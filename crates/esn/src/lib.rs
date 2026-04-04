// SPDX-License-Identifier: PMPL-1.0-or-later
// NeuroPhone - High-Assurance Hardware Orchestration
// Copyright (c) 2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>

//! Echo State Network (ESN) - Reservoir Computing
//!
//! Implements a recurrent neural network with fixed random weights
//! for temporal pattern recognition and sequence processing.
//! The ESN serves as a secondary reservoir in the neurosymbolic pipeline,
//! operating on transformed LSM outputs for higher-level temporal features.

#![allow(unsafe_code)]

use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand::Rng;
use rand_distr::{Normal, Uniform};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use thiserror::Error;
use tracing::{debug, trace};

/// Errors that can occur in ESN operations
#[derive(Error, Debug)]
pub enum EsnError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("Processing error: {0}")]
    ProcessingError(String),
}

/// Configuration for the Echo State Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EsnConfig {
    /// Reservoir size (number of internal neurons)
    pub reservoir_size: usize,
    /// Input dimension
    pub input_dim: usize,
    /// Spectral radius of the recurrent weight matrix
    pub spectral_radius: f32,
    /// Input scaling factor
    pub input_scale: f32,
    /// Connection sparsity (0.0 to 1.0)
    pub sparsity: f32,
    /// Leaking rate (0.0 to 1.0)
    pub leaking_rate: f32,
}

impl Default for EsnConfig {
    fn default() -> Self {
        Self {
            reservoir_size: 256,
            input_dim: 100,
            spectral_radius: 0.9,
            input_scale: 1.0,
            sparsity: 0.9,
            leaking_rate: 0.3,
        }
    }
}

/// Echo State Network implementation
pub struct EchoStateNetwork {
    config: EsnConfig,

    /// Recurrent weight matrix (sparse)
    recurrent_weights: Array2<f32>,

    /// Input weight matrix
    input_weights: Array2<f32>,

    /// Current reservoir state
    state: Array1<f32>,

    /// State history (ring buffer)
    state_history: VecDeque<Array1<f32>>,

    /// Maximum history window
    max_history: usize,
}

impl EchoStateNetwork {
    /// Create a new ESN with given configuration
    pub fn new(config: EsnConfig) -> Result<Self, EsnError> {
        if config.reservoir_size == 0 {
            return Err(EsnError::InvalidConfig(
                "Reservoir size must be positive".into(),
            ));
        }
        if config.input_dim == 0 {
            return Err(EsnError::InvalidConfig(
                "Input dimension must be positive".into(),
            ));
        }
        if !(0.0..=1.0).contains(&config.sparsity) {
            return Err(EsnError::InvalidConfig(
                "Sparsity must be between 0.0 and 1.0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&config.leaking_rate) {
            return Err(EsnError::InvalidConfig(
                "Leaking rate must be between 0.0 and 1.0".into(),
            ));
        }

        debug!(
            "Creating ESN with {} neurons, {} input dims",
            config.reservoir_size, config.input_dim
        );

        let mut rng = rand::thread_rng();

        // Create recurrent weight matrix with sparsity
        let recurrent_weights =
            Self::create_recurrent_weights(config.reservoir_size, config.sparsity, &mut rng);

        // Scale to achieve target spectral radius
        let recurrent_weights = Self::scale_to_spectral_radius(
            &recurrent_weights,
            config.spectral_radius,
        );

        // Create input weight matrix
        let input_weights =
            Self::create_input_weights(config.reservoir_size, config.input_dim, config.input_scale, &mut rng);

        // Initialize state
        let state = Array1::zeros(config.reservoir_size);

        // State history with capacity for recent states
        let state_history = VecDeque::with_capacity(1000);
        let max_history = 1000;

        Ok(Self {
            config,
            recurrent_weights,
            input_weights,
            state,
            state_history,
            max_history,
        })
    }

    /// Create sparse recurrent weight matrix
    fn create_recurrent_weights(
        size: usize,
        sparsity: f32,
        rng: &mut impl Rng,
    ) -> Array2<f32> {
        let mut weights = Array2::<f32>::zeros((size, size));
        let dist = Normal::new(0.0, 1.0).expect("valid normal distribution");

        let connection_prob = 1.0 - sparsity;
        for i in 0..size {
            for j in 0..size {
                if rng.gen::<f32>() < connection_prob {
                    let w: f32 = rng.sample(dist);
                    weights[[i, j]] = w;
                }
            }
        }

        weights
    }

    /// Create input weight matrix
    fn create_input_weights(
        reservoir_size: usize,
        input_dim: usize,
        scale: f32,
        rng: &mut impl Rng,
    ) -> Array2<f32> {
        let dist = Uniform::new(-1.0, 1.0).expect("valid uniform");
        let weights = Array2::random_using((reservoir_size, input_dim), dist, rng);
        weights * scale
    }

    /// Scale matrix to achieve target spectral radius
    fn scale_to_spectral_radius(weights: &Array2<f32>, target_radius: f32) -> Array2<f32> {
        // Simplified: compute maximum absolute row sum as approximation
        let max_row_sum = weights
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|x| x.abs()).sum::<f32>())
            .fold(0.0f32, f32::max);

        if max_row_sum > 0.0 {
            weights * (target_radius / max_row_sum)
        } else {
            weights.clone()
        }
    }

    /// Process a single input step
    pub fn step(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let _n = self.config.reservoir_size;

        // Check input dimension
        if input.len() != self.config.input_dim {
            let msg = format!(
                "Expected input dim {}, got {}",
                self.config.input_dim,
                input.len()
            );
            trace!("{}", msg);
            // Pad or trim input
            let mut padded = Array1::zeros(self.config.input_dim);
            let copy_len = input.len().min(self.config.input_dim);
            padded
                .slice_mut(ndarray::s![..copy_len])
                .assign(&input.slice(ndarray::s![..copy_len]));
            return self.step(&padded);
        }

        // Input activation
        let input_activation = self.input_weights.dot(input);

        // Recurrent activation
        let recurrent_activation = self.recurrent_weights.dot(&self.state);

        // Total activation
        let total_activation = input_activation + recurrent_activation;

        // Apply tanh nonlinearity with leaking rate
        let new_state: Array1<f32> = (1.0 - self.config.leaking_rate) * &self.state
            + self.config.leaking_rate * total_activation.mapv(|x| x.tanh());

        // Update state
        self.state = new_state.clone();

        // Store in history
        if self.state_history.len() >= self.max_history {
            self.state_history.pop_front();
        }
        self.state_history.push_back(self.state.clone());

        let norm = self.state.iter().map(|x| x * x).sum::<f32>().sqrt();
        trace!("ESN step completed, state norm: {:.4}", norm);

        new_state
    }

    /// Get current reservoir state
    pub fn get_state(&self) -> Array1<f32> {
        self.state.clone()
    }

    /// Get state history
    pub fn get_state_history(&self) -> Vec<Array1<f32>> {
        self.state_history.iter().cloned().collect()
    }

    /// Reset ESN to initial state
    pub fn reset(&mut self) {
        self.state = Array1::zeros(self.config.reservoir_size);
        self.state_history.clear();
        trace!("ESN reset to initial state");
    }

    /// Get reservoir size
    pub fn reservoir_size(&self) -> usize {
        self.config.reservoir_size
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    /// Process a sequence of inputs
    pub fn process_sequence(&mut self, inputs: &[Array1<f32>]) -> Result<Vec<Array1<f32>>, EsnError> {
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            outputs.push(self.step(input));
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_esn_creation() {
        let config = EsnConfig {
            reservoir_size: 100,
            input_dim: 50,
            ..Default::default()
        };
        let esn = EchoStateNetwork::new(config).expect("ESN creation");
        assert_eq!(esn.reservoir_size(), 100);
        assert_eq!(esn.input_dim(), 50);
    }

    #[test]
    fn test_esn_invalid_config() {
        let config = EsnConfig {
            reservoir_size: 0,
            ..Default::default()
        };
        assert!(EchoStateNetwork::new(config).is_err());
    }

    #[test]
    fn test_esn_step() {
        let config = EsnConfig {
            reservoir_size: 50,
            input_dim: 20,
            ..Default::default()
        };
        let mut esn = EchoStateNetwork::new(config).expect("ESN creation");

        let input = Array1::from_vec(vec![0.5; 20]);
        let output = esn.step(&input);

        assert_eq!(output.len(), 50);
    }

    #[test]
    fn test_esn_reset() {
        let config = EsnConfig {
            reservoir_size: 50,
            input_dim: 20,
            ..Default::default()
        };
        let mut esn = EchoStateNetwork::new(config).expect("ESN creation");

        let input = Array1::from_vec(vec![0.5; 20]);
        for _ in 0..10 {
            esn.step(&input);
        }

        esn.reset();
        let state = esn.get_state();
        assert!(state.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_esn_sequence_processing() {
        let config = EsnConfig {
            reservoir_size: 50,
            input_dim: 20,
            ..Default::default()
        };
        let mut esn = EchoStateNetwork::new(config).expect("ESN creation");

        let inputs: Vec<_> = (0..5)
            .map(|i| Array1::from_vec(vec![i as f32 * 0.1; 20]))
            .collect();

        let outputs = esn.process_sequence(&inputs).expect("sequence processing");
        assert_eq!(outputs.len(), 5);
    }

    #[test]
    fn test_esn_sparsity_config() {
        let config = EsnConfig {
            sparsity: 0.99,
            ..Default::default()
        };
        let esn = EchoStateNetwork::new(config).expect("ESN creation");
        assert_eq!(esn.config.sparsity, 0.99);
    }

    #[test]
    fn test_esn_invalid_sparsity() {
        let config = EsnConfig {
            sparsity: 1.5,
            ..Default::default()
        };
        assert!(EchoStateNetwork::new(config).is_err());
    }

    #[test]
    fn test_esn_leaking_rate() {
        let config = EsnConfig {
            leaking_rate: 0.5,
            ..Default::default()
        };
        let esn = EchoStateNetwork::new(config).expect("ESN creation");
        assert_eq!(esn.config.leaking_rate, 0.5);
    }

    #[test]
    fn test_esn_state_history() {
        let config = EsnConfig {
            reservoir_size: 30,
            input_dim: 10,
            ..Default::default()
        };
        let mut esn = EchoStateNetwork::new(config).expect("ESN creation");

        let input = Array1::from_vec(vec![0.1; 10]);
        for _ in 0..5 {
            esn.step(&input);
        }

        let history = esn.get_state_history();
        assert_eq!(history.len(), 5);
    }
}
