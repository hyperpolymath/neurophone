// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Benches for the bridge crate.

use bridge::{Bridge, BridgeConfig};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;
use std::hint::black_box;

fn bench_encode_sizes(c: &mut Criterion) {
    let mut g = c.benchmark_group("bridge_encode");
    for &(n_lsm, n_esn) in &[(100usize, 50usize), (512, 300), (2048, 1024)] {
        let lsm = Array1::from_elem(n_lsm, 0.4);
        let esn = Array1::from_elem(n_esn, 0.4);
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("lsm{n_lsm}_esn{n_esn}")),
            &(lsm, esn),
            |b, (lsm, esn)| {
                let mut br = Bridge::new(BridgeConfig::default()).unwrap();
                b.iter(|| black_box(br.encode(lsm.view(), esn.view())))
            },
        );
    }
    g.finish();
}

fn bench_encode_with_dynamics(c: &mut Criterion) {
    c.bench_function("bridge_encode_with_dynamics_512_300", |b| {
        let mut br = Bridge::new(BridgeConfig::default()).unwrap();
        let mut t = 0u32;
        let mut lsm = Array1::zeros(512);
        let esn = Array1::from_elem(300, 0.4);
        b.iter(|| {
            t = t.wrapping_add(1);
            for v in lsm.iter_mut() {
                *v = ((t % 100) as f32) * 0.01;
            }
            black_box(br.encode(lsm.view(), esn.view()));
        })
    });
}

criterion_group!(benches, bench_encode_sizes, bench_encode_with_dynamics);
criterion_main!(benches);
