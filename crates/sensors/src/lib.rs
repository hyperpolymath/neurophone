// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Phone Sensor Processing — Temporal Feature Extraction
//!
//! Acquisition + filtering + windowed feature extraction for the standard
//! Android sensor set (accelerometer, gyroscope, magnetometer, light,
//! proximity). Output feeds the LSM at 50 Hz.

#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum SensorError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("empty buffer")]
    EmptyBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensorKind {
    Accelerometer,
    Gyroscope,
    Magnetometer,
    Light,
    Proximity,
}

impl SensorKind {
    pub const fn arity(self) -> usize {
        match self {
            Self::Accelerometer | Self::Gyroscope | Self::Magnetometer => 3,
            Self::Light | Self::Proximity => 1,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Accelerometer => "accelerometer",
            Self::Gyroscope => "gyroscope",
            Self::Magnetometer => "magnetometer",
            Self::Light => "light",
            Self::Proximity => "proximity",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorReading {
    pub kind: SensorKind,
    pub timestamp_ms: u64,
    pub values: Vec<f32>,
}

impl SensorReading {
    pub fn new(kind: SensorKind, timestamp_ms: u64, values: Vec<f32>) -> Result<Self, SensorError> {
        if values.len() != kind.arity() {
            return Err(SensorError::DimensionMismatch {
                expected: kind.arity(),
                got: values.len(),
            });
        }
        Ok(Self {
            kind,
            timestamp_ms,
            values,
        })
    }
}

/// First-order IIR low-pass / high-pass filter (per channel).
#[derive(Debug, Clone)]
pub struct IirFilter {
    alpha: f32,
    /// y[n-1] (low-pass) or y[n-1] (high-pass).
    prev_out: Vec<f32>,
    /// Only used by high-pass: x[n-1].
    prev_in: Vec<f32>,
    high_pass: bool,
}

impl IirFilter {
    pub fn new(
        channels: usize,
        cutoff_hz: f32,
        sample_hz: f32,
        high_pass: bool,
    ) -> Result<Self, SensorError> {
        if channels == 0 || sample_hz <= 0.0 || cutoff_hz <= 0.0 {
            return Err(SensorError::InvalidConfig("non-positive params".into()));
        }
        let dt = 1.0 / sample_hz;
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
        let alpha = if high_pass {
            rc / (rc + dt)
        } else {
            dt / (rc + dt)
        };
        Ok(Self {
            alpha,
            prev_out: vec![0.0; channels],
            prev_in: vec![0.0; channels],
            high_pass,
        })
    }

    pub fn step(&mut self, input: &[f32]) -> Result<Vec<f32>, SensorError> {
        if input.len() != self.prev_out.len() {
            return Err(SensorError::DimensionMismatch {
                expected: self.prev_out.len(),
                got: input.len(),
            });
        }
        let mut out = vec![0.0; self.prev_out.len()];
        for (i, &x) in input.iter().enumerate() {
            if self.high_pass {
                // y[n] = alpha * (y[n-1] + x[n] - x[n-1])
                let y = self.alpha * (self.prev_out[i] + x - self.prev_in[i]);
                self.prev_in[i] = x;
                self.prev_out[i] = y;
                out[i] = y;
            } else {
                // y[n] = y[n-1] + alpha * (x[n] - y[n-1])
                self.prev_out[i] += self.alpha * (x - self.prev_out[i]);
                out[i] = self.prev_out[i];
            }
        }
        Ok(out)
    }

    pub fn reset(&mut self) {
        self.prev_out.iter_mut().for_each(|v| *v = 0.0);
        self.prev_in.iter_mut().for_each(|v| *v = 0.0);
    }
}

/// Sliding-window feature extractor: per-channel mean+variance + L2 magnitude.
#[derive(Debug, Clone)]
pub struct WindowedFeatures {
    capacity: usize,
    buffers: Vec<VecDeque<f32>>,
}

impl WindowedFeatures {
    pub fn new(channels: usize, capacity: usize) -> Result<Self, SensorError> {
        if channels == 0 || capacity == 0 {
            return Err(SensorError::InvalidConfig(
                "zero channels or capacity".into(),
            ));
        }
        Ok(Self {
            capacity,
            buffers: (0..channels)
                .map(|_| VecDeque::with_capacity(capacity))
                .collect(),
        })
    }

    pub fn push(&mut self, sample: &[f32]) -> Result<(), SensorError> {
        if sample.len() != self.buffers.len() {
            return Err(SensorError::DimensionMismatch {
                expected: self.buffers.len(),
                got: sample.len(),
            });
        }
        for (buf, &v) in self.buffers.iter_mut().zip(sample) {
            if buf.len() == self.capacity {
                buf.pop_front();
            }
            buf.push_back(v);
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.buffers.first().map_or(0, |b| b.len())
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn features(&self) -> Result<Array1<f32>, SensorError> {
        if self.is_empty() {
            return Err(SensorError::EmptyBuffer);
        }
        let mut out = Vec::with_capacity(self.buffers.len() * 2 + 1);
        let mut last_sq_sum = 0.0f32;
        for buf in &self.buffers {
            let n = buf.len() as f32;
            let mean = buf.iter().sum::<f32>() / n;
            let var = buf.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
            out.push(mean);
            out.push(var);
            if let Some(&last) = buf.back() {
                last_sq_sum += last * last;
            }
        }
        out.push(last_sq_sum.sqrt());
        Ok(Array1::from_vec(out))
    }
}

#[derive(Debug, Clone)]
pub struct SensorPipeline {
    kind: SensorKind,
    low_pass: IirFilter,
    high_pass: IirFilter,
    window: WindowedFeatures,
    last_ts: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub sample_hz: f32,
    pub low_pass_hz: f32,
    pub high_pass_hz: f32,
    pub window_size: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            sample_hz: 50.0,
            low_pass_hz: 10.0,
            high_pass_hz: 0.5,
            window_size: 25,
        }
    }
}

impl SensorPipeline {
    pub fn new(kind: SensorKind, config: PipelineConfig) -> Result<Self, SensorError> {
        let arity = kind.arity();
        Ok(Self {
            kind,
            low_pass: IirFilter::new(arity, config.low_pass_hz, config.sample_hz, false)?,
            high_pass: IirFilter::new(arity, config.high_pass_hz, config.sample_hz, true)?,
            window: WindowedFeatures::new(arity, config.window_size)?,
            last_ts: None,
        })
    }

    pub fn kind(&self) -> SensorKind {
        self.kind
    }

    pub fn ingest(&mut self, reading: &SensorReading) -> Result<(), SensorError> {
        if reading.kind != self.kind {
            return Err(SensorError::InvalidConfig(format!(
                "kind mismatch {:?} vs {:?}",
                reading.kind, self.kind
            )));
        }
        let lp = self.low_pass.step(&reading.values)?;
        let hp = self.high_pass.step(&lp)?;
        self.window.push(&hp)?;
        self.last_ts = Some(reading.timestamp_ms);
        Ok(())
    }

    pub fn features(&self) -> Result<Array1<f32>, SensorError> {
        self.window.features()
    }
    pub fn last_timestamp_ms(&self) -> Option<u64> {
        self.last_ts
    }

    pub fn reset(&mut self) {
        self.low_pass.reset();
        self.high_pass.reset();
        // arity()/capacity are the same valid values used to build the window in
        // new(), so reconstruction cannot fail; retain the window rather than panic.
        if let Ok(w) = WindowedFeatures::new(self.kind.arity(), self.window.capacity) {
            self.window = w;
        }
        self.last_ts = None;
    }
}

pub fn hello() -> &'static str {
    "sensors"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arity_for_each_kind() {
        assert_eq!(SensorKind::Accelerometer.arity(), 3);
        assert_eq!(SensorKind::Gyroscope.arity(), 3);
        assert_eq!(SensorKind::Magnetometer.arity(), 3);
        assert_eq!(SensorKind::Light.arity(), 1);
        assert_eq!(SensorKind::Proximity.arity(), 1);
    }

    #[test]
    fn reading_validates_arity() {
        assert!(SensorReading::new(SensorKind::Light, 0, vec![1.0]).is_ok());
        assert!(SensorReading::new(SensorKind::Light, 0, vec![1.0, 2.0]).is_err());
    }

    #[test]
    fn lowpass_smooths_step() {
        let mut f = IirFilter::new(1, 1.0, 50.0, false).unwrap();
        for _ in 0..200 {
            let _ = f.step(&[1.0]);
        }
        let out = f.step(&[1.0]).unwrap()[0];
        assert!(
            (out - 1.0).abs() < 1e-2,
            "lowpass should converge to 1.0, got {}",
            out
        );
    }

    #[test]
    fn highpass_zeroes_constant() {
        let mut f = IirFilter::new(1, 1.0, 50.0, true).unwrap();
        for _ in 0..200 {
            let _ = f.step(&[1.0]);
        }
        let out = f.step(&[1.0]).unwrap()[0];
        assert!(out.abs() < 1e-3, "highpass should zero DC, got {}", out);
    }

    #[test]
    fn window_features_dimension() {
        let mut w = WindowedFeatures::new(3, 5).unwrap();
        for i in 0..5 {
            w.push(&[i as f32, 0.0, 1.0]).unwrap();
        }
        let f = w.features().unwrap();
        assert_eq!(f.len(), 3 * 2 + 1);
    }

    #[test]
    fn pipeline_kind_mismatch_rejected() {
        let mut p = SensorPipeline::new(SensorKind::Light, PipelineConfig::default()).unwrap();
        let r = SensorReading::new(SensorKind::Accelerometer, 0, vec![0.0, 0.0, 0.0]).unwrap();
        assert!(p.ingest(&r).is_err());
    }

    #[test]
    fn pipeline_ingest_then_features() {
        let mut p =
            SensorPipeline::new(SensorKind::Accelerometer, PipelineConfig::default()).unwrap();
        for i in 0..30 {
            let r = SensorReading::new(
                SensorKind::Accelerometer,
                i as u64 * 20,
                vec![i as f32 * 0.01, 0.0, 9.81],
            )
            .unwrap();
            p.ingest(&r).unwrap();
        }
        assert_eq!(p.last_timestamp_ms(), Some(580));
        let f = p.features().unwrap();
        assert_eq!(f.len(), 7);
    }

    #[test]
    fn pipeline_reset_clears_state() {
        let mut p = SensorPipeline::new(SensorKind::Light, PipelineConfig::default()).unwrap();
        let r = SensorReading::new(SensorKind::Light, 0, vec![100.0]).unwrap();
        p.ingest(&r).unwrap();
        p.reset();
        assert!(p.features().is_err());
        assert_eq!(p.last_timestamp_ms(), None);
    }
}
