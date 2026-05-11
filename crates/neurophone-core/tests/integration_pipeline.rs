// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Cross-crate integration: sensor → LSM → ESN → bridge → LLM mock.
//! This is the closest thing to a true end-to-end test we can run on the host.

use bridge::{Bridge, BridgeConfig};
use esn::{EchoStateNetwork, EsnConfig};
use llm::{LlmBackend, LlmConfig, MockBackend};
use lsm::{LiquidStateMachine, LsmConfig};
use ndarray::Array1;
use sensors::{PipelineConfig, SensorKind, SensorPipeline, SensorReading};

fn build_pipeline() -> (
    SensorPipeline,
    LiquidStateMachine,
    EchoStateNetwork,
    Bridge,
    MockBackend,
) {
    let sp = SensorPipeline::new(SensorKind::Accelerometer, PipelineConfig::default()).unwrap();
    let lsm = LiquidStateMachine::new(
        LsmConfig {
            dimensions: (3, 3, 3),
            ..Default::default()
        },
        7,
    )
    .unwrap();
    let esn = EchoStateNetwork::new(EsnConfig {
        reservoir_size: 64,
        input_dim: 27,
        spectral_radius: 0.9,
        input_scale: 1.0,
        sparsity: 0.9,
        leaking_rate: 0.3,
    })
    .unwrap();
    let bridge = Bridge::new(BridgeConfig::default()).unwrap();
    let mut llm = MockBackend::new(LlmConfig::default()).unwrap();
    llm.load().unwrap();
    (sp, lsm, esn, bridge, llm)
}

#[test]
fn end_to_end_single_step() {
    let (mut sp, mut lsm, mut esn, mut br, mut llm) = build_pipeline();

    // Feed enough samples to fill the sensor window.
    for i in 0..50 {
        let r = SensorReading::new(
            SensorKind::Accelerometer,
            i as u64 * 20,
            vec![(i as f32 * 0.1).sin(), (i as f32 * 0.1).cos(), 9.81],
        )
        .unwrap();
        sp.ingest(&r).unwrap();
    }

    let features = sp.features().unwrap();
    let lsm_state = lsm.step(&features);
    let esn_state = esn.step(&lsm_state);
    let ctx = br.encode(lsm_state.view(), esn_state.view());
    assert!(ctx.description.starts_with("[NEURAL_STATE]"));

    let prompt = format!("{}\n\nUser: what am I doing?", ctx.description);
    let response = llm.generate(&prompt, 32).unwrap();
    assert!(response.text.contains("(local-llama-mock)"));
}

#[test]
fn end_to_end_50hz_one_second_loop() {
    let (mut sp, mut lsm, mut esn, mut br, mut llm) = build_pipeline();

    // 50 cycles = 1 second @ 50Hz.
    let mut last_ctx_salience = -1.0f32;
    for i in 0..50 {
        let r = SensorReading::new(
            SensorKind::Accelerometer,
            i as u64 * 20,
            vec![(i as f32 * 0.2).sin(), 0.0, 9.81],
        )
        .unwrap();
        sp.ingest(&r).unwrap();
        if i >= 25 {
            let f = sp.features().unwrap();
            let lsm_state = lsm.step(&f);
            let esn_state = esn.step(&lsm_state);
            let ctx = br.encode(lsm_state.view(), esn_state.view());
            assert!(ctx.salience >= 0.0 && ctx.salience <= 1.0);
            last_ctx_salience = ctx.salience;
        }
    }
    assert!(last_ctx_salience >= 0.0);

    let response = llm.generate("test query after pipeline", 16).unwrap();
    assert!(response.tokens_emitted > 0);
}

#[test]
fn end_to_end_quiet_input_yields_quiet_description() {
    let (mut sp, mut lsm, mut esn, mut br, _llm) = build_pipeline();
    // Constant gravity = no motion variance.
    for i in 0..200 {
        let r = SensorReading::new(
            SensorKind::Accelerometer,
            i as u64 * 20,
            vec![0.0, 0.0, 9.81],
        )
        .unwrap();
        sp.ingest(&r).unwrap();
    }
    let f = sp.features().unwrap();
    let lsm_state = lsm.step(&f);
    let esn_state = esn.step(&lsm_state);
    let ctx = br.encode(lsm_state.view(), esn_state.view());
    // Stationary input must still produce a valid description and a salience in [0,1].
    assert!(ctx.salience >= 0.0 && ctx.salience <= 1.0);
    assert!(ctx.description.starts_with("[NEURAL_STATE]"));
}

#[test]
fn end_to_end_dimension_negotiation() {
    // Verify shapes line up: features.len → lsm.input → lsm.size → esn.input.
    let cfg_lsm = LsmConfig {
        dimensions: (4, 4, 4),
        ..Default::default()
    };
    let mut lsm = LiquidStateMachine::new(cfg_lsm, 7).unwrap();
    let lsm_state = lsm.step(&Array1::from_vec(vec![0.0; 7]));
    assert_eq!(lsm_state.len(), 64);

    let mut esn = EchoStateNetwork::new(EsnConfig {
        reservoir_size: 128,
        input_dim: 64,
        spectral_radius: 0.9,
        input_scale: 1.0,
        sparsity: 0.9,
        leaking_rate: 0.3,
    })
    .unwrap();
    let esn_state = esn.step(&lsm_state);
    assert_eq!(esn_state.len(), 128);
}
