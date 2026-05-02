// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Benches for the ESN reservoir.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use esn::{EchoStateNetwork, EsnConfig};
use ndarray::Array1;

fn cfg(size: usize, input: usize) -> EsnConfig {
    EsnConfig {
        reservoir_size: size,
        input_dim: input,
        spectral_radius: 0.9,
        input_scale: 1.0,
        sparsity: 0.9,
        leaking_rate: 0.3,
    }
}

fn bench_step(c: &mut Criterion) {
    let mut g = c.benchmark_group("esn_step");
    for &size in &[50usize, 300, 1000] {
        let mut e = EchoStateNetwork::new(cfg(size, 8)).unwrap();
        let inp = Array1::from_vec(vec![0.1; 8]);
        g.bench_with_input(BenchmarkId::from_parameter(size), &inp, |b, inp| {
            b.iter(|| black_box(e.step(black_box(inp))))
        });
    }
    g.finish();
}

fn bench_construct(c: &mut Criterion) {
    c.bench_function("esn_construct_300x100", |b| {
        b.iter(|| black_box(EchoStateNetwork::new(cfg(300, 100)).unwrap()))
    });
}

criterion_group!(benches, bench_step, bench_construct);
criterion_main!(benches);
