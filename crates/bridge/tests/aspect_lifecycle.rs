// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Aspect + lifecycle tests for the bridge.

use bridge::{Bridge, BridgeConfig};
use ndarray::Array1;

#[test]
fn aspect_invalid_threshold_rejected() {
    let cfg = BridgeConfig {
        activation_threshold: 1.5,
        lsm_weight: 0.5,
    };
    assert!(Bridge::new(cfg).is_err());
}

#[test]
fn aspect_invalid_weight_rejected() {
    let cfg = BridgeConfig {
        activation_threshold: 0.5,
        lsm_weight: -0.1,
    };
    assert!(Bridge::new(cfg).is_err());
}

#[test]
fn lifecycle_reset_isolates_urgency_episodes() {
    let mut b = Bridge::new(BridgeConfig::default()).unwrap();
    let _ = b.encode(Array1::zeros(10).view(), Array1::zeros(10).view());
    let _ = b.encode(Array1::from_elem(10, 1.0).view(), Array1::zeros(10).view());
    b.reset();
    let ctx = b.encode(Array1::zeros(10).view(), Array1::zeros(10).view());
    assert!(ctx.urgency.abs() < 1e-6, "urgency should reset");
}

#[test]
fn lifecycle_thousand_encodings_stable_memory() {
    let mut b = Bridge::new(BridgeConfig::default()).unwrap();
    let lsm = Array1::from_elem(512, 0.4);
    let esn = Array1::from_elem(300, 0.4);
    for _ in 0..1000 {
        let _ = b.encode(lsm.view(), esn.view());
    }
}
