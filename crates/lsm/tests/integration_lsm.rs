// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Integration tests for the LSM (Liquid State Machine).

use lsm::{LiquidStateMachine, LsmConfig};
use ndarray::Array1;

#[test]
fn small_lsm_constructs_and_steps() {
    let cfg = LsmConfig {
        dimensions: (4, 4, 4),
        ..Default::default()
    };
    let mut m = LiquidStateMachine::new(cfg, 3).expect("lsm");
    let state = m.step(&Array1::from_vec(vec![0.1, 0.2, 0.3]));
    assert_eq!(state.len(), 4 * 4 * 4);
}

#[test]
fn lifecycle_step_then_reset() {
    let cfg = LsmConfig {
        dimensions: (3, 3, 3),
        ..Default::default()
    };
    let mut m = LiquidStateMachine::new(cfg, 2).expect("lsm");
    for _ in 0..50 {
        let _ = m.step(&Array1::from_vec(vec![0.1, 0.2]));
    }
    m.reset();
    let state = m.step(&Array1::from_vec(vec![0.0, 0.0]));
    assert_eq!(state.len(), 27);
}

#[test]
fn aspect_input_padding_no_panic() {
    let cfg = LsmConfig {
        dimensions: (2, 2, 2),
        ..Default::default()
    };
    let mut m = LiquidStateMachine::new(cfg, 3).expect("lsm");
    // input shorter than expected — should pad and not panic
    let s = m.step(&Array1::from_vec(vec![0.1, 0.2]));
    assert_eq!(s.len(), 8);
}

#[test]
fn aspect_long_run_state_finite() {
    let cfg = LsmConfig {
        dimensions: (4, 4, 4),
        ..Default::default()
    };
    let mut m = LiquidStateMachine::new(cfg, 3).expect("lsm");
    for i in 0..500 {
        let phase = (i as f32) * 0.05;
        let s = m.step(&Array1::from_vec(vec![phase.sin(), phase.cos(), 0.5]));
        assert!(s.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn lifecycle_firing_rate_stable_after_warmup() {
    let cfg = LsmConfig {
        dimensions: (3, 3, 3),
        ..Default::default()
    };
    let mut m = LiquidStateMachine::new(cfg, 2).expect("lsm");
    for _ in 0..200 {
        let _ = m.step(&Array1::from_vec(vec![0.5, 0.5]));
    }
    let rates = m.get_firing_rates(50.0);
    assert_eq!(rates.len(), 27);
    assert!(rates.iter().all(|v| v.is_finite() && *v >= 0.0));
}
