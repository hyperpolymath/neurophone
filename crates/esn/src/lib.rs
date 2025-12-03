//! Echo State Network (ESN) - Reservoir Computing
//!
//! Implements a classic echo state network with:
//! - Sparse randomly connected reservoir
//! - Leaky integrator neurons
//! - Linear readout with ridge regression training
//! - Optional feedback connections
//!
//! The ESN receives processed state from the LSM and produces
//! outputs that can drive actions or influence LLM prompts.

use ndarray::{s, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::Rng;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors in ESN operations
#[derive(Error, Debug)]
pub enum EsnError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("Training error: {0}")]
    TrainingError(String),
    #[error("Not trained: readout weights not set")]
    NotTrained,
}

/// ESN configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EsnConfig {
    /// Number of reservoir neurons
    pub reservoir_size: usize,
    /// Spectral radius of reservoir weights
    pub spectral_radius: f32,
    /// Leaking rate for neuron dynamics (0-1)
    pub leaking_rate: f32,
    /// Input scaling factor
    pub input_scale: f32,
    /// Feedback scaling factor (0 for no feedback)
    pub feedback_scale: f32,
    /// Sparsity of reservoir (0-1, higher = more sparse)
    pub sparsity: f32,
    /// Ridge regression regularization parameter
    pub ridge_param: f64,
    /// Include bias in readout
    pub use_bias: bool,
    /// Noise level for reservoir state
    pub noise_level: f32,
}

impl Default for EsnConfig {
    fn default() -> Self {
        Self {
            reservoir_size: 500,
            spectral_radius: 0.95,
            leaking_rate: 0.3,
            input_scale: 0.5,
            feedback_scale: 0.0,
            sparsity: 0.9,
            ridge_param: 1e-6,
            use_bias: true,
            noise_level: 1e-4,
        }
    }
}

/// Activation function types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    Tanh,
    Sigmoid,
    ReLU,
    LeakyReLU(f32),
    Identity,
}

impl Activation {
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU(alpha) => if x > 0.0 { x } else { alpha * x },
            Activation::Identity => x,
        }
    }

    pub fn apply_array(&self, arr: &Array1<f32>) -> Array1<f32> {
        arr.mapv(|x| self.apply(x))
    }
}

/// Echo State Network
pub struct EchoStateNetwork {
    config: EsnConfig,
    activation: Activation,

    /// Input dimension
    input_dim: usize,

    /// Output dimension (set during training)
    output_dim: Option<usize>,

    /// Reservoir weight matrix (sparse)
    w_reservoir: Array2<f32>,

    /// Input weight matrix
    w_input: Array2<f32>,

    /// Feedback weight matrix (optional)
    w_feedback: Option<Array2<f32>>,

    /// Readout weight matrix (trained)
    w_out: Option<Array2<f32>>,

    /// Current reservoir state
    state: Array1<f32>,

    /// Previous output (for feedback)
    prev_output: Option<Array1<f32>>,

    /// State history for context
    state_history: VecDeque<Array1<f32>>,

    /// Maximum history length
    max_history: usize,
}

impl EchoStateNetwork {
    /// Create a new ESN
    pub fn new(config: EsnConfig, input_dim: usize, activation: Activation) -> Result<Self, EsnError> {
        if config.reservoir_size == 0 {
            return Err(EsnError::InvalidConfig("Reservoir size must be positive".into()));
        }

        if config.spectral_radius <= 0.0 {
            return Err(EsnError::InvalidConfig("Spectral radius must be positive".into()));
        }

        debug!("Creating ESN: {} neurons, input_dim={}", config.reservoir_size, input_dim);

        let mut rng = rand::thread_rng();

        // Create sparse reservoir matrix
        let w_reservoir = Self::create_reservoir_matrix(&config, &mut rng);

        // Create input weight matrix
        let w_input = Self::create_input_matrix(&config, input_dim, &mut rng);

        // Initialize state
        let state = Array1::zeros(config.reservoir_size);

        Ok(Self {
            config,
            activation,
            input_dim,
            output_dim: None,
            w_reservoir,
            w_input,
            w_feedback: None,
            w_out: None,
            state,
            prev_output: None,
            state_history: VecDeque::with_capacity(100),
            max_history: 100,
        })
    }

    /// Create sparse reservoir weight matrix with target spectral radius
    fn create_reservoir_matrix(config: &EsnConfig, rng: &mut impl Rng) -> Array2<f32> {
        let n = config.reservoir_size;
        let dist = Uniform::new(-1.0f32, 1.0f32);

        let mut w = Array2::zeros((n, n));

        // Fill with sparse random values
        for i in 0..n {
            for j in 0..n {
                if rng.gen::<f32>() > config.sparsity {
                    w[[i, j]] = rng.sample(dist);
                }
            }
        }

        // Estimate spectral radius using power iteration
        let mut v = Array1::random_using(n, dist, rng);
        for _ in 0..20 {
            let new_v = w.dot(&v);
            let norm = new_v.mapv(|x| x * x).sum().sqrt();
            if norm > 0.0 {
                v = new_v / norm;
            }
        }

        let current_radius = w.dot(&v).mapv(|x| x * x).sum().sqrt();

        // Scale to target spectral radius
        if current_radius > 0.0 {
            w *= config.spectral_radius / current_radius;
        }

        w
    }

    /// Create input weight matrix
    fn create_input_matrix(config: &EsnConfig, input_dim: usize, rng: &mut impl Rng) -> Array2<f32> {
        let dist = Uniform::new(-1.0f32, 1.0f32);
        let w = Array2::random_using((config.reservoir_size, input_dim), dist, rng);
        w * config.input_scale
    }

    /// Process one input and update reservoir state
    pub fn step(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let mut rng = rand::thread_rng();

        // Handle input dimension mismatch
        let processed_input = if input.len() != self.input_dim {
            let mut padded = Array1::zeros(self.input_dim);
            let copy_len = input.len().min(self.input_dim);
            padded.slice_mut(s![..copy_len]).assign(&input.slice(s![..copy_len]));
            padded
        } else {
            input.clone()
        };

        // Compute pre-activation
        let mut pre_activation = self.w_reservoir.dot(&self.state) + self.w_input.dot(&processed_input);

        // Add feedback if available
        if let (Some(ref w_fb), Some(ref prev_out)) = (&self.w_feedback, &self.prev_output) {
            pre_activation = pre_activation + w_fb.dot(prev_out);
        }

        // Add noise
        let noise_dist = Normal::new(0.0, self.config.noise_level as f64).unwrap();
        let noise: Array1<f32> = Array1::from_iter(
            (0..self.config.reservoir_size).map(|_| rng.sample(&noise_dist) as f32)
        );
        pre_activation = pre_activation + noise;

        // Apply activation and leaky integration
        let activated = self.activation.apply_array(&pre_activation);
        self.state = (1.0 - self.config.leaking_rate) * &self.state
            + self.config.leaking_rate * activated;

        // Save to history
        self.state_history.push_back(self.state.clone());
        if self.state_history.len() > self.max_history {
            self.state_history.pop_front();
        }

        self.state.clone()
    }

    /// Get output using trained readout weights
    pub fn predict(&self) -> Result<Array1<f32>, EsnError> {
        let w_out = self.w_out.as_ref().ok_or(EsnError::NotTrained)?;

        // Add bias if configured
        let state_with_bias = if self.config.use_bias {
            let mut extended = Array1::zeros(self.state.len() + 1);
            extended.slice_mut(s![..self.state.len()]).assign(&self.state);
            extended[self.state.len()] = 1.0; // Bias term
            extended
        } else {
            self.state.clone()
        };

        Ok(w_out.dot(&state_with_bias))
    }

    /// Train readout weights using ridge regression
    pub fn train(
        &mut self,
        inputs: &[Array1<f32>],
        targets: &[Array1<f32>],
        washout: usize,
    ) -> Result<f64, EsnError> {
        if inputs.len() != targets.len() {
            return Err(EsnError::TrainingError("Input/target length mismatch".into()));
        }

        if inputs.len() <= washout {
            return Err(EsnError::TrainingError("Not enough samples after washout".into()));
        }

        let output_dim = targets[0].len();
        self.output_dim = Some(output_dim);

        info!("Training ESN: {} samples, {} washout, {} outputs",
            inputs.len(), washout, output_dim);

        // Collect reservoir states
        self.reset();
        let mut states = Vec::with_capacity(inputs.len() - washout);

        for (i, input) in inputs.iter().enumerate() {
            let state = self.step(input);
            if i >= washout {
                if self.config.use_bias {
                    let mut extended = Array1::zeros(state.len() + 1);
                    extended.slice_mut(s![..state.len()]).assign(&state);
                    extended[state.len()] = 1.0;
                    states.push(extended);
                } else {
                    states.push(state);
                }
            }
        }

        let state_dim = states[0].len();
        let n_samples = states.len();

        // Build state matrix X (n_samples x state_dim)
        let mut x = Array2::zeros((n_samples, state_dim));
        for (i, state) in states.iter().enumerate() {
            x.row_mut(i).assign(state);
        }

        // Build target matrix Y (n_samples x output_dim)
        let mut y = Array2::zeros((n_samples, output_dim));
        for (i, target) in targets.iter().skip(washout).enumerate() {
            y.row_mut(i).assign(target);
        }

        // Ridge regression: W = (X^T X + lambda I)^-1 X^T Y
        // Using normal equations with regularization
        let xtx = x.t().dot(&x);
        let mut xtx_reg = xtx.clone();

        // Add regularization
        for i in 0..state_dim {
            xtx_reg[[i, i]] += self.config.ridge_param as f32;
        }

        let xty = x.t().dot(&y);

        // Solve using simple Gaussian elimination (for embedded use)
        // In production, use a proper linear algebra library
        match Self::solve_linear_system(&xtx_reg, &xty) {
            Ok(w_out) => {
                self.w_out = Some(w_out.t().to_owned());

                // Calculate training error
                let predictions = x.dot(&w_out);
                let error = &y - &predictions;
                let mse = error.mapv(|x| x * x).mean().unwrap_or(0.0) as f64;

                info!("Training complete. MSE: {:.6}", mse);
                Ok(mse)
            }
            Err(e) => Err(EsnError::TrainingError(format!("Linear solve failed: {}", e)))
        }
    }

    /// Simple linear system solver (Gaussian elimination with partial pivoting)
    fn solve_linear_system(a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>, String> {
        let n = a.nrows();
        let m = b.ncols();

        // Create augmented matrix
        let mut aug = Array2::zeros((n, n + m));
        aug.slice_mut(s![.., ..n]).assign(a);
        aug.slice_mut(s![.., n..]).assign(b);

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[[col, col]].abs();
            for row in (col + 1)..n {
                if aug[[row, col]].abs() > max_val {
                    max_val = aug[[row, col]].abs();
                    max_row = row;
                }
            }

            if max_val < 1e-10 {
                return Err("Matrix is singular or nearly singular".into());
            }

            // Swap rows
            if max_row != col {
                for j in 0..(n + m) {
                    let tmp = aug[[col, j]];
                    aug[[col, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = tmp;
                }
            }

            // Eliminate
            for row in (col + 1)..n {
                let factor = aug[[row, col]] / aug[[col, col]];
                for j in col..(n + m) {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array2::zeros((n, m));
        for row in (0..n).rev() {
            for k in 0..m {
                let mut sum = aug[[row, n + k]];
                for col in (row + 1)..n {
                    sum -= aug[[row, col]] * x[[col, k]];
                }
                x[[row, k]] = sum / aug[[row, row]];
            }
        }

        Ok(x)
    }

    /// Set feedback weights
    pub fn set_feedback(&mut self, output_dim: usize) {
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(-1.0f32, 1.0f32);
        let w = Array2::random_using((self.config.reservoir_size, output_dim), dist, &mut rng);
        self.w_feedback = Some(w * self.config.feedback_scale);
        self.output_dim = Some(output_dim);
    }

    /// Reset reservoir state
    pub fn reset(&mut self) {
        self.state = Array1::zeros(self.config.reservoir_size);
        self.prev_output = None;
        self.state_history.clear();
    }

    /// Get current reservoir state
    pub fn get_state(&self) -> &Array1<f32> {
        &self.state
    }

    /// Get reservoir state history
    pub fn get_history(&self) -> &VecDeque<Array1<f32>> {
        &self.state_history
    }

    /// Get number of neurons
    pub fn size(&self) -> usize {
        self.config.reservoir_size
    }

    /// Check if trained
    pub fn is_trained(&self) -> bool {
        self.w_out.is_some()
    }
}

/// Hierarchical ESN with multiple reservoirs
pub struct HierarchicalEsn {
    layers: Vec<EchoStateNetwork>,
}

impl HierarchicalEsn {
    /// Create hierarchical ESN with multiple layers
    pub fn new(
        configs: Vec<EsnConfig>,
        input_dim: usize,
        activations: Vec<Activation>,
    ) -> Result<Self, EsnError> {
        if configs.len() != activations.len() {
            return Err(EsnError::InvalidConfig("Config and activation count mismatch".into()));
        }

        let mut layers = Vec::with_capacity(configs.len());
        let mut current_dim = input_dim;

        for (config, activation) in configs.into_iter().zip(activations) {
            let layer = EchoStateNetwork::new(config.clone(), current_dim, activation)?;
            current_dim = config.reservoir_size;
            layers.push(layer);
        }

        Ok(Self { layers })
    }

    /// Process input through all layers
    pub fn step(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let mut current = input.clone();
        for layer in &mut self.layers {
            current = layer.step(&current);
        }
        current
    }

    /// Reset all layers
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Get combined state from all layers
    pub fn get_combined_state(&self) -> Array1<f32> {
        let total_size: usize = self.layers.iter().map(|l| l.size()).sum();
        let mut combined = Array1::zeros(total_size);

        let mut offset = 0;
        for layer in &self.layers {
            let state = layer.get_state();
            combined.slice_mut(s![offset..offset + state.len()]).assign(state);
            offset += state.len();
        }

        combined
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_esn_creation() {
        let config = EsnConfig {
            reservoir_size: 100,
            ..Default::default()
        };
        let esn = EchoStateNetwork::new(config, 10, Activation::Tanh).unwrap();
        assert_eq!(esn.size(), 100);
    }

    #[test]
    fn test_esn_step() {
        let config = EsnConfig {
            reservoir_size: 100,
            ..Default::default()
        };
        let mut esn = EchoStateNetwork::new(config, 10, Activation::Tanh).unwrap();

        let input = Array1::from_vec(vec![0.5; 10]);
        let state = esn.step(&input);

        assert_eq!(state.len(), 100);
    }

    #[test]
    fn test_esn_echo_property() {
        // ESN should forget inputs after ~1/leaking_rate steps
        let config = EsnConfig {
            reservoir_size: 50,
            leaking_rate: 0.3,
            spectral_radius: 0.9,
            ..Default::default()
        };
        let mut esn = EchoStateNetwork::new(config, 5, Activation::Tanh).unwrap();

        // Drive with input
        let input = Array1::from_vec(vec![1.0; 5]);
        for _ in 0..100 {
            esn.step(&input);
        }
        let state_after_input = esn.get_state().clone();

        // Let it decay
        let zero_input = Array1::zeros(5);
        for _ in 0..100 {
            esn.step(&zero_input);
        }
        let state_after_decay = esn.get_state();

        // State should have decayed significantly
        let input_energy: f32 = state_after_input.mapv(|x| x * x).sum();
        let decay_energy: f32 = state_after_decay.mapv(|x| x * x).sum();

        assert!(decay_energy < input_energy * 0.1);
    }

    #[test]
    fn test_hierarchical_esn() {
        let configs = vec![
            EsnConfig { reservoir_size: 50, ..Default::default() },
            EsnConfig { reservoir_size: 30, ..Default::default() },
        ];
        let activations = vec![Activation::Tanh, Activation::Tanh];

        let mut h_esn = HierarchicalEsn::new(configs, 10, activations).unwrap();

        let input = Array1::from_vec(vec![0.5; 10]);
        let output = h_esn.step(&input);

        assert_eq!(output.len(), 30); // Last layer size
        assert_eq!(h_esn.get_combined_state().len(), 80); // 50 + 30
    }
}
