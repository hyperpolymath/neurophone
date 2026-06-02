// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Integration tests for the ESN (Echo State Network).

use esn::{EchoStateNetwork, EsnConfig};
use ndarray::Array1;

fn small_cfg() -> EsnConfig {
    EsnConfig {
        reservoir_size: 50,
        input_dim: 4,
        spectral_radius: 0.9,
        input_scale: 1.0,
        sparsity: 0.9,
        leaking_rate: 0.3,
    }
}

#[test]
fn point_to_point_step_returns_state() {
    let mut e = EchoStateNetwork::new(small_cfg()).unwrap();
    let s = e.step(&Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]));
    assert_eq!(s.len(), 50);
    assert!(s.iter().all(|v| v.is_finite()));
}

#[test]
fn lifecycle_step_reset_step() {
    let mut e = EchoStateNetwork::new(small_cfg()).unwrap();
    for _ in 0..100 {
        let _ = e.step(&Array1::from_vec(vec![1.0, 0.5, 0.0, -0.5]));
    }
    let s_before = e.get_state();
    e.reset();
    let s_after = e.get_state();
    assert!(s_after.iter().all(|v| v.abs() < 1e-9));
    let _ = (s_before, s_after);
}

#[test]
fn aspect_invalid_config_zero_size() {
    let cfg = EsnConfig {
        reservoir_size: 0,
        ..small_cfg()
    };
    assert!(EchoStateNetwork::new(cfg).is_err());
}

#[test]
fn aspect_invalid_leaking_rate() {
    let cfg = EsnConfig {
        leaking_rate: 1.5,
        ..small_cfg()
    };
    assert!(EchoStateNetwork::new(cfg).is_err());
}

#[test]
fn process_sequence_matches_step_count() {
    let mut e = EchoStateNetwork::new(small_cfg()).unwrap();
    let inputs: Vec<Array1<f32>> = (0..10)
        .map(|i| Array1::from_vec(vec![i as f32 * 0.1; 4]))
        .collect();
    let outputs = e.process_sequence(&inputs).unwrap();
    assert_eq!(outputs.len(), 10);
    assert!(outputs.iter().all(|s| s.len() == 50));
}

#[test]
fn aspect_state_history_grows_then_caps() {
    let mut e = EchoStateNetwork::new(small_cfg()).unwrap();
    for _ in 0..2000 {
        let _ = e.step(&Array1::from_vec(vec![0.1; 4]));
    }
    assert!(e.get_state_history().len() <= 1000);
}
