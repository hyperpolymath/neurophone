// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! End-to-end pipeline bench: sensor → LSM → ESN → bridge → LLM mock.

use bridge::{Bridge, BridgeConfig};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use esn::{EchoStateNetwork, EsnConfig};
use llm::{LlmBackend, LlmConfig, MockBackend};
use lsm::{LiquidStateMachine, LsmConfig};
use sensors::{PipelineConfig, SensorKind, SensorPipeline, SensorReading};
use std::hint::black_box;

fn warmed_pipeline() -> (
    SensorPipeline,
    LiquidStateMachine,
    EchoStateNetwork,
    Bridge,
    MockBackend,
) {
    let mut sp = SensorPipeline::new(SensorKind::Accelerometer, PipelineConfig::default()).unwrap();
    for i in 0..50u64 {
        sp.ingest(
            &SensorReading::new(SensorKind::Accelerometer, i * 20, vec![0.1, 0.2, 9.81]).unwrap(),
        )
        .unwrap();
    }
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
    let br = Bridge::new(BridgeConfig::default()).unwrap();
    let mut llm = MockBackend::new(LlmConfig::default()).unwrap();
    llm.load().unwrap();
    (sp, lsm, esn, br, llm)
}

fn bench_pipeline_step(c: &mut Criterion) {
    let mut g = c.benchmark_group("e2e_pipeline");
    g.throughput(Throughput::Elements(1));
    let (mut sp, mut lsm, mut esn, mut br, mut llm) = warmed_pipeline();
    let mut t = 1000u64;
    g.bench_function("sensor_to_llm_step", |b| {
        b.iter(|| {
            t += 20;
            let r = SensorReading::new(SensorKind::Accelerometer, t, vec![0.1, 0.2, 9.81]).unwrap();
            sp.ingest(&r).unwrap();
            let f = sp.features().unwrap();
            let lsm_state = lsm.step(&f);
            let esn_state = esn.step(&lsm_state);
            let ctx = br.encode(lsm_state.view(), esn_state.view());
            let _ = black_box(llm.generate(&ctx.description, 8).unwrap());
        })
    });
    g.finish();
}

fn bench_bridge_only_step(c: &mut Criterion) {
    let (mut sp, mut lsm, mut esn, mut br, _) = warmed_pipeline();
    let mut t = 1000u64;
    c.bench_function("e2e_to_bridge_step", |b| {
        b.iter(|| {
            t += 20;
            let r = SensorReading::new(SensorKind::Accelerometer, t, vec![0.1, 0.2, 9.81]).unwrap();
            sp.ingest(&r).unwrap();
            let f = sp.features().unwrap();
            let lsm_state = lsm.step(&f);
            let esn_state = esn.step(&lsm_state);
            let _ = black_box(br.encode(lsm_state.view(), esn_state.view()));
        })
    });
}

criterion_group!(benches, bench_pipeline_step, bench_bridge_only_step);
criterion_main!(benches);
