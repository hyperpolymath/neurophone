// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell

//! Liquid State Machine (LSM) benchmarks — reservoir step latency,
//! state extraction, and reset throughput.
//!
//! The LSM is the real-time hot path in the NeuroPhone pipeline:
//! sensor data arrives at up to 100Hz and must be processed within
//! the sample window. These benchmarks track:
//! - `LiquidStateMachine::step` — single timestep forward pass
//! - `LiquidStateMachine::get_firing_rates` — rate computation over a time window
//! - `LiquidStateMachine::reset` — reservoir state clear between sessions
//! - Full pipeline: N steps + firing rate extraction

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lsm::{LiquidStateMachine, LsmConfig};
use ndarray::Array1;

// ============================================================================
// Helpers
// ============================================================================

/// Build a minimal LSM config for benchmarking (5x5x5 = 125 neurons).
fn make_small_config() -> LsmConfig {
    LsmConfig {
        dimensions: (5, 5, 5),
        p_exc: 0.3,
        p_inh: 0.2,
        frac_inh: 0.2,
        spectral_radius: 0.9,
        input_scale: 1.0,
        dt: 1.0,
    }
}

/// Build a medium LSM config for benchmarking (10x10x10 = 1000 neurons).
fn make_medium_config() -> LsmConfig {
    LsmConfig {
        dimensions: (10, 10, 10),
        ..LsmConfig::default()
    }
}

/// Build a zero-mean unit-variance input signal of the given dimension.
fn make_input(dim: usize) -> Array1<f32> {
    Array1::linspace(0.0f32, 1.0, dim)
}

/// Build a pre-initialized LSM with a given config and input dimension.
fn make_lsm(config: LsmConfig, input_dim: usize) -> LiquidStateMachine {
    LiquidStateMachine::new(config, input_dim)
        .expect("Failed to create LiquidStateMachine for benchmark")
}

// ============================================================================
// Step benchmarks
// ============================================================================

/// Benchmark a single LSM step on a small reservoir (125 neurons, 8-dim input).
fn bench_step_small(c: &mut Criterion) {
    let mut lsm = make_lsm(make_small_config(), 8);
    let input = make_input(8);

    c.bench_function("lsm_step_small_125neurons", |b| {
        b.iter(|| black_box(lsm.step(black_box(&input))))
    });
}

/// Benchmark a single LSM step on the default reservoir (1000 neurons, 16-dim input).
fn bench_step_medium(c: &mut Criterion) {
    let mut lsm = make_lsm(make_medium_config(), 16);
    let input = make_input(16);

    c.bench_function("lsm_step_medium_1000neurons", |b| {
        b.iter(|| black_box(lsm.step(black_box(&input))))
    });
}

/// Benchmark LSM step latency at varying reservoir sizes.
///
/// Shows how the O(n²) connectivity cost grows with reservoir dimension.
fn bench_step_scaling(c: &mut Criterion) {
    let input_dim = 8;
    let mut group = c.benchmark_group("lsm_step_reservoir_size");

    for (dx, dy, dz) in [(3usize, 3, 3), (5, 5, 5), (8, 8, 8)] {
        let n_neurons = dx * dy * dz;
        let config = LsmConfig {
            dimensions: (dx, dy, dz),
            ..LsmConfig::default()
        };
        let mut lsm = make_lsm(config, input_dim);
        let input = make_input(input_dim);

        group.bench_with_input(
            BenchmarkId::from_parameter(n_neurons),
            &n_neurons,
            |b, _| {
                b.iter(|| black_box(lsm.step(black_box(&input))))
            },
        );
    }
    group.finish();
}

// ============================================================================
// Firing rate extraction benchmarks
// ============================================================================

/// Benchmark extracting firing rates over a 100ms window after 50 steps.
fn bench_get_firing_rates(c: &mut Criterion) {
    let mut lsm = make_lsm(make_small_config(), 8);
    let input = make_input(8);

    // Prime the reservoir with 50 steps so there is actual spike history.
    for _ in 0..50 {
        lsm.step(&input);
    }

    c.bench_function("lsm_get_firing_rates_100ms", |b| {
        b.iter(|| black_box(lsm.get_firing_rates(black_box(100.0f64))))
    });
}

/// Benchmark extracting state vector over varying window sizes.
fn bench_get_state_window_scaling(c: &mut Criterion) {
    let mut lsm = make_lsm(make_small_config(), 8);
    let input = make_input(8);

    for _ in 0..100 {
        lsm.step(&input);
    }

    let mut group = c.benchmark_group("lsm_get_state_window_ms");
    for window_ms in [10.0f64, 50.0, 100.0, 500.0] {
        group.bench_with_input(
            BenchmarkId::from_parameter(window_ms as u64),
            &window_ms,
            |b, &w| {
                b.iter(|| black_box(lsm.get_state(black_box(w))))
            },
        );
    }
    group.finish();
}

// ============================================================================
// Reset benchmark
// ============================================================================

/// Benchmark resetting the reservoir between processing sessions.
///
/// Called at the start of each new recording session — should be fast
/// regardless of reservoir history depth.
fn bench_reset(c: &mut Criterion) {
    let mut lsm = make_lsm(make_small_config(), 8);
    let input = make_input(8);

    // Build up 200 steps of history first.
    for _ in 0..200 {
        lsm.step(&input);
    }

    c.bench_function("lsm_reset_after_200_steps", |b| {
        b.iter(|| {
            lsm.reset();
            // Immediately do one step to prevent dead-code elimination.
            black_box(lsm.step(black_box(&input)));
        })
    });
}

// ============================================================================
// Full pipeline benchmark
// ============================================================================

/// Benchmark a complete 100-step processing window (1 second at 100Hz).
///
/// This represents the real workload: process a sensor burst, then extract
/// a state vector for the ESN layer.
fn bench_full_pipeline_100hz(c: &mut Criterion) {
    let mut lsm = make_lsm(make_small_config(), 8);
    let inputs: Vec<Array1<f32>> = (0..100).map(|i| make_input(i % 8 + 1)).collect();

    c.bench_function("lsm_full_pipeline_100_steps", |b| {
        b.iter(|| {
            lsm.reset();
            for inp in &inputs {
                black_box(lsm.step(black_box(inp)));
            }
            black_box(lsm.get_firing_rates(black_box(100.0f64)))
        })
    });
}

criterion_group!(
    benches,
    bench_step_small,
    bench_step_medium,
    bench_step_scaling,
    bench_get_firing_rates,
    bench_get_state_window_scaling,
    bench_reset,
    bench_full_pipeline_100hz,
);
criterion_main!(benches);
