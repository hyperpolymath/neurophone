// SPDX-License-Identifier: PMPL-1.0-or-later
// NeuroPhone - High-Assurance Hardware Orchestration
// Copyright (c) 2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>

//! Benchmarks for neurophone-core

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lsm::{LiquidStateMachine, LsmConfig};
use esn::{EchoStateNetwork, EsnConfig};
use ndarray::Array1;
use neurophone_core::*;

// ========== LSM Benchmarks ==========

fn bench_lsm_creation(c: &mut Criterion) {
    c.bench_function("lsm_creation_10x10x10", |b| {
        b.iter(|| {
            let config = LsmConfig {
                dimensions: (10, 10, 10),
                ..Default::default()
            };
            LiquidStateMachine::new(config, 10)
        })
    });

    c.bench_function("lsm_creation_20x20x20", |b| {
        b.iter(|| {
            let config = LsmConfig {
                dimensions: (20, 20, 20),
                ..Default::default()
            };
            LiquidStateMachine::new(config, 20)
        })
    });
}

fn bench_lsm_step(c: &mut Criterion) {
    c.bench_function("lsm_step_10x10x10", |b| {
        let config = LsmConfig {
            dimensions: (10, 10, 10),
            ..Default::default()
        };
        let mut lsm = LiquidStateMachine::new(config, 10).expect("LSM creation");
        let input = Array1::from_vec(vec![0.5; 10]);

        b.iter(|| {
            lsm.step(black_box(&input))
        })
    });

    c.bench_function("lsm_step_20x20x20", |b| {
        let config = LsmConfig {
            dimensions: (20, 20, 20),
            ..Default::default()
        };
        let mut lsm = LiquidStateMachine::new(config, 20).expect("LSM creation");
        let input = Array1::from_vec(vec![0.5; 20]);

        b.iter(|| {
            lsm.step(black_box(&input))
        })
    });
}

fn bench_lsm_reset(c: &mut Criterion) {
    c.bench_function("lsm_reset_10x10x10", |b| {
        let config = LsmConfig {
            dimensions: (10, 10, 10),
            ..Default::default()
        };
        let mut lsm = LiquidStateMachine::new(config, 10).expect("LSM creation");

        b.iter(|| {
            lsm.reset();
        })
    });
}

fn bench_lsm_get_state(c: &mut Criterion) {
    c.bench_function("lsm_get_state_10x10x10", |b| {
        let config = LsmConfig {
            dimensions: (10, 10, 10),
            ..Default::default()
        };
        let mut lsm = LiquidStateMachine::new(config, 10).expect("LSM creation");
        let input = Array1::from_vec(vec![0.5; 10]);

        // Warm up
        for _ in 0..10 {
            lsm.step(&input);
        }

        b.iter(|| {
            lsm.get_state(100.0)
        })
    });
}

// ========== ESN Benchmarks ==========

fn bench_esn_creation(c: &mut Criterion) {
    c.bench_function("esn_creation_256", |b| {
        b.iter(|| {
            let config = EsnConfig {
                reservoir_size: 256,
                input_dim: 50,
                ..Default::default()
            };
            EchoStateNetwork::new(config)
        })
    });

    c.bench_function("esn_creation_512", |b| {
        b.iter(|| {
            let config = EsnConfig {
                reservoir_size: 512,
                input_dim: 50,
                ..Default::default()
            };
            EchoStateNetwork::new(config)
        })
    });
}

fn bench_esn_step(c: &mut Criterion) {
    c.bench_function("esn_step_256", |b| {
        let config = EsnConfig {
            reservoir_size: 256,
            input_dim: 50,
            ..Default::default()
        };
        let mut esn = EchoStateNetwork::new(config).expect("ESN creation");
        let input = Array1::from_vec(vec![0.1; 50]);

        b.iter(|| {
            esn.step(black_box(&input))
        })
    });

    c.bench_function("esn_step_512", |b| {
        let config = EsnConfig {
            reservoir_size: 512,
            input_dim: 50,
            ..Default::default()
        };
        let mut esn = EchoStateNetwork::new(config).expect("ESN creation");
        let input = Array1::from_vec(vec![0.1; 50]);

        b.iter(|| {
            esn.step(black_box(&input))
        })
    });
}

fn bench_esn_reset(c: &mut Criterion) {
    c.bench_function("esn_reset_256", |b| {
        let config = EsnConfig {
            reservoir_size: 256,
            input_dim: 50,
            ..Default::default()
        };
        let mut esn = EchoStateNetwork::new(config).expect("ESN creation");

        b.iter(|| {
            esn.reset();
        })
    });
}

fn bench_esn_process_sequence(c: &mut Criterion) {
    c.bench_function("esn_sequence_100_steps", |b| {
        let config = EsnConfig {
            reservoir_size: 256,
            input_dim: 50,
            ..Default::default()
        };
        let mut esn = EchoStateNetwork::new(config).expect("ESN creation");

        let inputs: Vec<Array1<f32>> = (0..100)
            .map(|i| Array1::from_vec(vec![i as f32 * 0.01; 50]))
            .collect();

        b.iter(|| {
            esn.process_sequence(black_box(&inputs)).ok();
        })
    });
}

// ========== NeuroPhone Core Benchmarks ==========

fn bench_system_query(c: &mut Criterion) {
    c.bench_function("system_query_short", |b| {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");

        b.iter(|| {
            system.query(black_box("hello world"), true).ok()
        })
    });

    c.bench_function("system_query_long", |b| {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");

        let long_query = "word ".repeat(50);

        b.iter(|| {
            system.query(black_box(&long_query), true).ok()
        })
    });
}

fn bench_system_sensor_processing(c: &mut Criterion) {
    c.bench_function("system_sensor_processing", |b| {
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");
        system.initialize().expect("init");

        let event = SensorEvent {
            sensor_type: "accelerometer".to_string(),
            timestamp_ms: 1000,
            values: vec![1.0, 2.0, 3.0],
        };

        b.iter(|| {
            system.process_sensor_event(black_box(&event)).ok()
        })
    });
}

fn bench_system_lifecycle(c: &mut Criterion) {
    c.bench_function("system_create_init_shutdown", |b| {
        b.iter(|| {
            let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
                .expect("system creation");
            system.initialize().expect("init");
            system.shutdown().expect("shutdown");
        })
    });
}

fn bench_system_state_access(c: &mut Criterion) {
    c.bench_function("system_state_access", |b| {
        let system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");

        b.iter(|| {
            system.get_state()
        })
    });
}

// ========== Serialization Benchmarks ==========

fn bench_serialization(c: &mut Criterion) {
    c.bench_function("serialize_inference_result", |b| {
        let result = InferenceResult {
            query: "test query".to_string(),
            response: "test response".to_string(),
            model: InferenceModel::LocalLlama,
            latency_ms: 50,
            confidence: 0.92,
        };

        b.iter(|| {
            serde_json::to_string(black_box(&result)).ok()
        })
    });

    c.bench_function("serialize_system_state", |b| {
        let state = SystemState {
            timestamp_ms: 1000,
            is_active: true,
            latency_ms: 50,
            ..Default::default()
        };

        b.iter(|| {
            serde_json::to_string(black_box(&state)).ok()
        })
    });

    c.bench_function("deserialize_inference_result", |b| {
        let json = r#"{"query":"test","response":"resp","model":"LocalLlama","latency_ms":50,"confidence":0.92}"#;

        b.iter(|| {
            serde_json::from_str::<InferenceResult>(black_box(json)).ok()
        })
    });
}

// ========== Integration Benchmarks ==========

fn bench_end_to_end(c: &mut Criterion) {
    c.bench_function("e2e_sensor_to_query", |b| {
        b.iter(|| {
            let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
                .expect("system creation");
            system.initialize().expect("init");

            let event = SensorEvent {
                sensor_type: "accelerometer".to_string(),
                timestamp_ms: 1000,
                values: vec![1.0, 2.0, 3.0],
            };

            system.process_sensor_event(&event).ok();
            system.query("what happened", true).ok();
        })
    });
}

criterion_group!(
    benches,
    bench_lsm_creation,
    bench_lsm_step,
    bench_lsm_reset,
    bench_lsm_get_state,
    bench_esn_creation,
    bench_esn_step,
    bench_esn_reset,
    bench_esn_process_sequence,
    bench_system_query,
    bench_system_sensor_processing,
    bench_system_lifecycle,
    bench_system_state_access,
    bench_serialization,
    bench_end_to_end
);

criterion_main!(benches);
