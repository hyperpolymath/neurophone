// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Benches for the LLM mock backend (real backend benched out-of-tree).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use llm::{LlmBackend, LlmConfig, MockBackend};
use std::hint::black_box;

fn bench_generate(c: &mut Criterion) {
    let mut g = c.benchmark_group("llm_mock_generate");
    let prompt_short = "hello world".to_string();
    let prompt_long: String = "word ".repeat(256);
    for &(name, prompt) in &[("short", &prompt_short), ("long_256", &prompt_long)] {
        g.bench_with_input(BenchmarkId::from_parameter(name), prompt, |b, p| {
            let mut be = MockBackend::new(LlmConfig::default()).unwrap();
            be.load().unwrap();
            b.iter(|| black_box(be.generate(p, 64).unwrap()));
        });
    }
    g.finish();
}

fn bench_load_unload_cycle(c: &mut Criterion) {
    c.bench_function("llm_mock_load_unload", |b| {
        b.iter(|| {
            let mut be = MockBackend::new(LlmConfig::default()).unwrap();
            be.load().unwrap();
            be.unload();
            black_box(())
        })
    });
}

criterion_group!(benches, bench_generate, bench_load_unload_cycle);
criterion_main!(benches);
