//! Liquid State Machine (LSM) - Spiking Neural Network Reservoir
//!
//! Implements a biologically-inspired spiking neural network using
//! Leaky Integrate-and-Fire (LIF) neurons arranged in a 3D grid.
//! This serves as the first reservoir in our neurosymbolic pipeline,
//! processing temporal sensor data with spike-timing dynamics.

use ndarray::{Array1, Array2, Array3, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::{Bernoulli, Uniform};
use rand::Rng;
use rand_distr::Normal;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use thiserror::Error;
use tracing::{debug, trace};

/// Errors that can occur in LSM operations
#[derive(Error, Debug)]
pub enum LsmError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("Simulation error: {0}")]
    SimulationError(String),
}

/// Configuration for the Liquid State Machine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LsmConfig {
    /// Grid dimensions (x, y, z)
    pub dimensions: (usize, usize, usize),
    /// Connection probability for excitatory neurons
    pub p_exc: f32,
    /// Connection probability for inhibitory neurons
    pub p_inh: f32,
    /// Fraction of inhibitory neurons
    pub frac_inh: f32,
    /// Spectral radius for weight scaling
    pub spectral_radius: f32,
    /// Input scaling factor
    pub input_scale: f32,
    /// Simulation timestep (ms)
    pub dt: f32,
}

impl Default for LsmConfig {
    fn default() -> Self {
        Self {
            dimensions: (10, 10, 10),
            p_exc: 0.3,
            p_inh: 0.2,
            frac_inh: 0.2,
            spectral_radius: 0.9,
            input_scale: 1.0,
            dt: 1.0,
        }
    }
}

/// Parameters for Leaky Integrate-and-Fire neurons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifParameters {
    /// Membrane time constant (ms)
    pub tau_m: f32,
    /// Resting potential (mV)
    pub v_rest: f32,
    /// Threshold potential (mV)
    pub v_thresh: f32,
    /// Reset potential (mV)
    pub v_reset: f32,
    /// Refractory period (ms)
    pub t_refrac: f32,
    /// Membrane resistance (MOhm)
    pub r_m: f32,
}

impl Default for LifParameters {
    fn default() -> Self {
        Self {
            tau_m: 20.0,
            v_rest: -65.0,
            v_thresh: -50.0,
            v_reset: -70.0,
            t_refrac: 2.0,
            r_m: 10.0,
        }
    }
}

/// Single LIF neuron state
#[derive(Debug, Clone, Default)]
struct NeuronState {
    /// Membrane potential (mV)
    v: f32,
    /// Remaining refractory time (ms)
    refrac_remaining: f32,
    /// Is this an inhibitory neuron?
    is_inhibitory: bool,
    /// Position in grid
    position: (usize, usize, usize),
}

/// Spike event with timing information
#[derive(Debug, Clone, Copy)]
pub struct SpikeEvent {
    pub neuron_idx: usize,
    pub time: f64,
}

/// Liquid State Machine implementation
pub struct LiquidStateMachine {
    config: LsmConfig,
    lif_params: LifParameters,

    /// Neuron states (flattened 3D grid)
    neurons: Vec<NeuronState>,

    /// Recurrent weight matrix (sparse representation)
    weights: Array2<f32>,

    /// Input weight matrix
    input_weights: Array2<f32>,

    /// Spike history for each neuron (ring buffer)
    spike_history: Vec<VecDeque<f64>>,

    /// Current simulation time (ms)
    current_time: f64,

    /// Recent spikes for output
    recent_spikes: VecDeque<SpikeEvent>,

    /// History window for spike collection (ms)
    history_window: f64,
}

impl LiquidStateMachine {
    /// Create a new LSM with given configuration
    pub fn new(config: LsmConfig, input_dim: usize) -> Result<Self, LsmError> {
        let (nx, ny, nz) = config.dimensions;
        let n_neurons = nx * ny * nz;

        if n_neurons == 0 {
            return Err(LsmError::InvalidConfig("Grid dimensions must be positive".into()));
        }

        debug!("Creating LSM with {} neurons ({} x {} x {})", n_neurons, nx, ny, nz);

        let mut rng = rand::thread_rng();
        let lif_params = LifParameters::default();

        // Initialize neurons
        let mut neurons = Vec::with_capacity(n_neurons);
        let n_inhibitory = (n_neurons as f32 * config.frac_inh) as usize;

        for idx in 0..n_neurons {
            let x = idx % nx;
            let y = (idx / nx) % ny;
            let z = idx / (nx * ny);

            neurons.push(NeuronState {
                v: lif_params.v_rest + rng.gen::<f32>() * 5.0, // Small random offset
                refrac_remaining: 0.0,
                is_inhibitory: idx < n_inhibitory,
                position: (x, y, z),
            });
        }

        // Shuffle to distribute inhibitory neurons
        use rand::seq::SliceRandom;
        neurons.shuffle(&mut rng);

        // Restore positions
        for (idx, neuron) in neurons.iter_mut().enumerate() {
            let x = idx % nx;
            let y = (idx / nx) % ny;
            let z = idx / (nx * ny);
            neuron.position = (x, y, z);
        }

        // Create weight matrix with distance-dependent connectivity
        let weights = Self::create_recurrent_weights(&config, &neurons, &mut rng);

        // Create input weights
        let input_weights = Self::create_input_weights(n_neurons, input_dim, config.input_scale, &mut rng);

        // Initialize spike history
        let spike_history = vec![VecDeque::with_capacity(100); n_neurons];

        Ok(Self {
            config,
            lif_params,
            neurons,
            weights,
            input_weights,
            spike_history,
            current_time: 0.0,
            recent_spikes: VecDeque::with_capacity(1000),
            history_window: 100.0,
        })
    }

    /// Create recurrent weight matrix with distance-dependent connectivity
    fn create_recurrent_weights(
        config: &LsmConfig,
        neurons: &[NeuronState],
        rng: &mut impl Rng,
    ) -> Array2<f32> {
        let n = neurons.len();
        let (nx, ny, nz) = config.dimensions;
        let max_dist = ((nx * nx + ny * ny + nz * nz) as f32).sqrt();

        let mut weights = Array2::<f32>::zeros((n, n));
        let weight_dist = Normal::new(0.5, 0.2).unwrap();

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                // Calculate Euclidean distance
                let (xi, yi, zi) = neurons[i].position;
                let (xj, yj, zj) = neurons[j].position;
                let dist = (((xi as i32 - xj as i32).pow(2)
                    + (yi as i32 - yj as i32).pow(2)
                    + (zi as i32 - zj as i32).pow(2)) as f32).sqrt();

                // Distance-dependent connection probability
                let p_connect = if neurons[i].is_inhibitory {
                    config.p_inh * (-dist / (max_dist * 0.3)).exp()
                } else {
                    config.p_exc * (-dist / (max_dist * 0.5)).exp()
                };

                if rng.gen::<f32>() < p_connect {
                    let w: f32 = rng.sample(weight_dist);
                    let w = w.abs().max(0.01); // Ensure positive base weight

                    // Inhibitory neurons have negative weights
                    weights[[i, j]] = if neurons[i].is_inhibitory { -w } else { w };
                }
            }
        }

        // Scale weights to achieve target spectral radius
        let eigenvalue_estimate = weights.mapv(|x| x.abs()).sum_axis(Axis(1)).into_iter()
            .fold(0.0f32, |a, b| a.max(b));

        if eigenvalue_estimate > 0.0 {
            weights *= config.spectral_radius / eigenvalue_estimate;
        }

        weights
    }

    /// Create input weight matrix
    fn create_input_weights(
        n_neurons: usize,
        input_dim: usize,
        scale: f32,
        rng: &mut impl Rng,
    ) -> Array2<f32> {
        let dist = Uniform::new(-1.0, 1.0);
        let mut weights = Array2::random_using((n_neurons, input_dim), dist, rng);

        // Sparse connectivity: only ~30% of neurons receive each input
        let bernoulli = Bernoulli::new(0.3).unwrap();
        for i in 0..n_neurons {
            for j in 0..input_dim {
                if !rng.sample(bernoulli) {
                    weights[[i, j]] = 0.0;
                }
            }
        }

        weights * scale
    }

    /// Perform one simulation step
    pub fn step(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let n = self.neurons.len();
        let dt = self.config.dt;

        // Check input dimension
        if input.len() != self.input_weights.ncols() {
            // Resize or pad input if necessary
            let expected = self.input_weights.ncols();
            let mut padded = Array1::zeros(expected);
            let copy_len = input.len().min(expected);
            padded.slice_mut(ndarray::s![..copy_len]).assign(&input.slice(ndarray::s![..copy_len]));
            return self.step(&padded);
        }

        // Calculate input currents
        let input_current = self.input_weights.dot(input);

        // Get previous spike indicators for recurrent input
        let spike_indicators: Array1<f32> = Array1::from_iter(
            self.neurons.iter().map(|n| {
                if let Some(&last_spike) = self.spike_history[0].back() {
                    if self.current_time - last_spike < dt as f64 { 1.0 } else { 0.0 }
                } else {
                    0.0
                }
            })
        );

        // Calculate recurrent currents
        let recurrent_current = self.weights.t().dot(&spike_indicators);

        // Total current
        let total_current = &input_current + &recurrent_current;

        // Update each neuron
        let mut spikes = Vec::new();

        for (idx, neuron) in self.neurons.iter_mut().enumerate() {
            // Check refractory period
            if neuron.refrac_remaining > 0.0 {
                neuron.refrac_remaining -= dt;
                continue;
            }

            // LIF dynamics: tau_m * dV/dt = -(V - V_rest) + R_m * I
            let dv = dt / self.lif_params.tau_m
                * (-(neuron.v - self.lif_params.v_rest)
                   + self.lif_params.r_m * total_current[idx]);

            neuron.v += dv;

            // Check for spike
            if neuron.v >= self.lif_params.v_thresh {
                neuron.v = self.lif_params.v_reset;
                neuron.refrac_remaining = self.lif_params.t_refrac;

                // Record spike
                self.spike_history[idx].push_back(self.current_time);

                // Trim old spikes from history
                while let Some(&t) = self.spike_history[idx].front() {
                    if self.current_time - t > self.history_window {
                        self.spike_history[idx].pop_front();
                    } else {
                        break;
                    }
                }

                spikes.push(SpikeEvent {
                    neuron_idx: idx,
                    time: self.current_time,
                });

                trace!("Neuron {} spiked at t={:.2}ms", idx, self.current_time);
            }
        }

        // Update recent spikes
        for spike in spikes {
            self.recent_spikes.push_back(spike);
        }

        // Trim old spikes
        while let Some(spike) = self.recent_spikes.front() {
            if self.current_time - spike.time > self.history_window {
                self.recent_spikes.pop_front();
            } else {
                break;
            }
        }

        self.current_time += dt as f64;

        // Return current membrane potentials as state
        Array1::from_iter(self.neurons.iter().map(|n| n.v))
    }

    /// Get the current firing rates (spikes per window)
    pub fn get_firing_rates(&self, window_ms: f64) -> Array1<f32> {
        Array1::from_iter(
            self.spike_history.iter().map(|history| {
                let count = history.iter()
                    .filter(|&&t| self.current_time - t <= window_ms)
                    .count();
                count as f32 / (window_ms / 1000.0) as f32 // Convert to Hz
            })
        )
    }

    /// Get state vector (normalized membrane potentials)
    pub fn get_state(&self, window_ms: f64) -> Array1<f32> {
        // Combine membrane potentials and firing rates
        let potentials = Array1::from_iter(
            self.neurons.iter().map(|n| {
                (n.v - self.lif_params.v_rest) /
                (self.lif_params.v_thresh - self.lif_params.v_rest)
            })
        );

        let rates = self.get_firing_rates(window_ms);
        let max_rate = rates.iter().cloned().fold(1.0f32, f32::max);
        let normalized_rates = rates / max_rate;

        // Concatenate: [potentials, normalized_rates]
        let n = self.neurons.len();
        let mut state = Array1::zeros(n * 2);
        state.slice_mut(ndarray::s![..n]).assign(&potentials);
        state.slice_mut(ndarray::s![n..]).assign(&normalized_rates);

        state
    }

    /// Get recent spike events
    pub fn get_recent_spikes(&self) -> Vec<SpikeEvent> {
        self.recent_spikes.iter().cloned().collect()
    }

    /// Reset the LSM to initial state
    pub fn reset(&mut self) {
        let mut rng = rand::thread_rng();

        for neuron in &mut self.neurons {
            neuron.v = self.lif_params.v_rest + rng.gen::<f32>() * 5.0;
            neuron.refrac_remaining = 0.0;
        }

        for history in &mut self.spike_history {
            history.clear();
        }

        self.recent_spikes.clear();
        self.current_time = 0.0;
    }

    /// Get number of neurons
    pub fn size(&self) -> usize {
        self.neurons.len()
    }

    /// Get grid dimensions
    pub fn dimensions(&self) -> (usize, usize, usize) {
        self.config.dimensions
    }

    /// Get current simulation time
    pub fn current_time(&self) -> f64 {
        self.current_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lsm_creation() {
        let config = LsmConfig {
            dimensions: (5, 5, 5),
            ..Default::default()
        };
        let lsm = LiquidStateMachine::new(config, 10).unwrap();
        assert_eq!(lsm.size(), 125);
    }

    #[test]
    fn test_lsm_step() {
        let config = LsmConfig {
            dimensions: (5, 5, 5),
            ..Default::default()
        };
        let mut lsm = LiquidStateMachine::new(config, 10).unwrap();

        let input = Array1::from_vec(vec![1.0; 10]);
        let output = lsm.step(&input);

        assert_eq!(output.len(), 125);
    }

    #[test]
    fn test_lsm_reset() {
        let config = LsmConfig::default();
        let mut lsm = LiquidStateMachine::new(config, 10).unwrap();

        // Run some steps
        let input = Array1::from_vec(vec![1.0; 10]);
        for _ in 0..100 {
            lsm.step(&input);
        }

        // Reset
        lsm.reset();
        assert_eq!(lsm.current_time(), 0.0);
        assert!(lsm.get_recent_spikes().is_empty());
    }

    #[test]
    fn test_firing_rates() {
        let config = LsmConfig {
            dimensions: (3, 3, 3),
            input_scale: 2.0,
            ..Default::default()
        };
        let mut lsm = LiquidStateMachine::new(config, 5).unwrap();

        // Strong input should produce spikes
        let input = Array1::from_vec(vec![5.0; 5]);
        for _ in 0..100 {
            lsm.step(&input);
        }

        let rates = lsm.get_firing_rates(100.0);
        // Should have some activity
        assert!(rates.sum() > 0.0);
    }
}
