// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Integration tests for the bridge crate (point-to-point: state -> context).

use bridge::{Bridge, BridgeConfig};
use ndarray::Array1;

#[test]
fn empty_states_produce_quiet_context() {
    let mut b = Bridge::new(BridgeConfig::default()).unwrap();
    let ctx = b.encode(Array1::zeros(0).view(), Array1::zeros(0).view());
    assert_eq!(ctx.salience, 0.0);
}

#[test]
fn salience_monotonic_in_active_fraction() {
    let mut b = Bridge::new(BridgeConfig::default()).unwrap();
    let lsm_low = Array1::from_iter((0..100).map(|i| if i < 10 { 1.0 } else { 0.0 }));
    let lsm_high = Array1::from_iter((0..100).map(|i| if i < 90 { 1.0 } else { 0.0 }));
    let esn_zero = Array1::zeros(50);

    let ctx_low = b.encode(lsm_low.view(), esn_zero.view());
    b.reset();
    let ctx_high = b.encode(lsm_high.view(), esn_zero.view());

    assert!(ctx_high.salience > ctx_low.salience);
}

#[test]
fn lsm_weight_zero_uses_only_esn() {
    let cfg = BridgeConfig { lsm_weight: 0.0, ..Default::default() };
    let mut b = Bridge::new(cfg).unwrap();
    let lsm = Array1::from_elem(10, 1.0);
    let esn = Array1::zeros(10);
    let ctx = b.encode(lsm.view(), esn.view());
    assert_eq!(ctx.salience, 0.0);
}

#[test]
fn lsm_weight_one_uses_only_lsm() {
    let cfg = BridgeConfig { lsm_weight: 1.0, ..Default::default() };
    let mut b = Bridge::new(cfg).unwrap();
    let lsm = Array1::zeros(10);
    let esn = Array1::from_elem(10, 1.0);
    let ctx = b.encode(lsm.view(), esn.view());
    assert_eq!(ctx.salience, 0.0);
}

#[test]
fn description_roundtrips_through_serde() {
    let mut b = Bridge::new(BridgeConfig::default()).unwrap();
    let ctx = b.encode(Array1::from_elem(20, 0.6).view(), Array1::zeros(10).view());
    let json = serde_json::to_string(&ctx).unwrap();
    let de: bridge::NeuralContext = serde_json::from_str(&json).unwrap();
    assert_eq!(de, ctx);
}
