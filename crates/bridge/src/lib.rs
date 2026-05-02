// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Neural ↔ Symbolic Bridge.
//!
//! Encodes LSM + ESN state vectors into salience/urgency scores and a short
//! natural-language description suitable for prompt injection into an LLM.

#![forbid(unsafe_code)]

use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum BridgeError {
    #[error("dimension mismatch: lsm={lsm} esn={esn}")]
    DimensionMismatch { lsm: usize, esn: usize },
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

/// Bridge configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Threshold above which a feature is considered "active" (0..=1).
    pub activation_threshold: f32,
    /// Weighting of LSM vs ESN in salience (0..=1, 1 = LSM-only).
    pub lsm_weight: f32,
}

impl Default for BridgeConfig {
    fn default() -> Self { Self { activation_threshold: 0.3, lsm_weight: 0.5 } }
}

impl BridgeConfig {
    pub fn validate(&self) -> Result<(), BridgeError> {
        if !(0.0..=1.0).contains(&self.activation_threshold) {
            return Err(BridgeError::InvalidConfig("activation_threshold ∉ [0,1]".into()));
        }
        if !(0.0..=1.0).contains(&self.lsm_weight) {
            return Err(BridgeError::InvalidConfig("lsm_weight ∉ [0,1]".into()));
        }
        Ok(())
    }
}

/// Result of bridging neural state to symbolic form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuralContext {
    /// Salience score (0..=1): how much "stuff" is happening.
    pub salience: f32,
    /// Urgency score (0..=1): how *fast* things are changing.
    pub urgency: f32,
    /// Number of LSM neurons above threshold.
    pub lsm_active: usize,
    /// Number of ESN neurons above threshold.
    pub esn_active: usize,
    /// Natural-language description (for LLM prompt).
    pub description: String,
}

/// Bridge encoder.
#[derive(Debug, Clone)]
pub struct Bridge {
    config: BridgeConfig,
    last_lsm: Option<Array1<f32>>,
}

impl Bridge {
    pub fn new(config: BridgeConfig) -> Result<Self, BridgeError> {
        config.validate()?;
        Ok(Self { config, last_lsm: None })
    }

    pub fn config(&self) -> &BridgeConfig { &self.config }

    fn count_active(view: ArrayView1<f32>, thresh: f32) -> usize {
        view.iter().filter(|v| v.abs() >= thresh).count()
    }

    fn rms_delta(prev: &Array1<f32>, curr: ArrayView1<f32>) -> f32 {
        if prev.len() != curr.len() { return 0.0; }
        let n = prev.len() as f32;
        if n == 0.0 { return 0.0; }
        let sum_sq: f32 = prev.iter().zip(curr.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        (sum_sq / n).sqrt().min(1.0)
    }

    /// Encode a paired LSM/ESN state into a `NeuralContext`.
    pub fn encode(&mut self, lsm: ArrayView1<f32>, esn: ArrayView1<f32>) -> NeuralContext {
        let t = self.config.activation_threshold;
        let lsm_active = Self::count_active(lsm, t);
        let esn_active = Self::count_active(esn, t);
        let lsm_frac = if lsm.len() == 0 { 0.0 } else { lsm_active as f32 / lsm.len() as f32 };
        let esn_frac = if esn.len() == 0 { 0.0 } else { esn_active as f32 / esn.len() as f32 };
        let w = self.config.lsm_weight;
        let salience = (w * lsm_frac + (1.0 - w) * esn_frac).clamp(0.0, 1.0);

        let urgency = match &self.last_lsm {
            Some(prev) => Self::rms_delta(prev, lsm),
            None => 0.0,
        };

        let description = describe(salience, urgency, lsm_active, esn_active);
        self.last_lsm = Some(lsm.to_owned());

        NeuralContext { salience, urgency, lsm_active, esn_active, description }
    }

    pub fn reset(&mut self) { self.last_lsm = None; }
}

fn describe(salience: f32, urgency: f32, lsm_active: usize, esn_active: usize) -> String {
    let s_label = match salience {
        x if x < 0.2 => "quiet",
        x if x < 0.5 => "moderate",
        _ => "high",
    };
    let u_label = match urgency {
        x if x < 0.1 => "stable",
        x if x < 0.3 => "shifting",
        _ => "rapidly changing",
    };
    format!(
        "[NEURAL_STATE] salience={s_label} ({salience:.2}); dynamics={u_label} ({urgency:.2}); \
         lsm_active={lsm_active}; esn_active={esn_active} [/NEURAL_STATE]"
    )
}

pub fn hello() -> &'static str { "bridge" }

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test] fn config_validates() {
        assert!(BridgeConfig { activation_threshold: -0.1, lsm_weight: 0.5 }.validate().is_err());
        assert!(BridgeConfig { activation_threshold: 0.5, lsm_weight: 1.5 }.validate().is_err());
        assert!(BridgeConfig::default().validate().is_ok());
    }

    #[test] fn quiet_state_low_salience() {
        let mut b = Bridge::new(BridgeConfig::default()).unwrap();
        let lsm = Array1::zeros(100);
        let esn = Array1::zeros(50);
        let ctx = b.encode(lsm.view(), esn.view());
        assert!(ctx.salience < 0.05);
        assert_eq!(ctx.lsm_active, 0);
        assert_eq!(ctx.esn_active, 0);
        assert!(ctx.description.contains("quiet"));
    }

    #[test] fn high_state_high_salience() {
        let mut b = Bridge::new(BridgeConfig::default()).unwrap();
        let lsm = Array1::from_elem(100, 1.0);
        let esn = Array1::from_elem(50, 1.0);
        let ctx = b.encode(lsm.view(), esn.view());
        assert!(ctx.salience > 0.95);
        assert!(ctx.description.contains("high"));
    }

    #[test] fn urgency_zero_on_first_call() {
        let mut b = Bridge::new(BridgeConfig::default()).unwrap();
        let lsm = Array1::from_elem(10, 0.5);
        let ctx = b.encode(lsm.view(), Array1::zeros(10).view());
        assert!(ctx.urgency.abs() < 1e-6);
    }

    #[test] fn urgency_rises_with_change() {
        let mut b = Bridge::new(BridgeConfig::default()).unwrap();
        let _ = b.encode(Array1::zeros(10).view(), Array1::zeros(10).view());
        let ctx = b.encode(Array1::from_elem(10, 1.0).view(), Array1::zeros(10).view());
        assert!(ctx.urgency > 0.5);
    }

    #[test] fn reset_clears_history() {
        let mut b = Bridge::new(BridgeConfig::default()).unwrap();
        let _ = b.encode(Array1::from_elem(10, 1.0).view(), Array1::zeros(10).view());
        b.reset();
        let ctx = b.encode(Array1::zeros(10).view(), Array1::zeros(10).view());
        assert!(ctx.urgency.abs() < 1e-6);
    }

    #[test] fn description_contains_markers() {
        let mut b = Bridge::new(BridgeConfig::default()).unwrap();
        let ctx = b.encode(Array1::zeros(10).view(), Array1::zeros(10).view());
        assert!(ctx.description.starts_with("[NEURAL_STATE]"));
        assert!(ctx.description.ends_with("[/NEURAL_STATE]"));
    }
}
