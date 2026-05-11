// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Benches for the Claude client (offline / routing only — no network).

use claude_client::{HybridInference, Message};
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn bench_complexity_estimation(c: &mut Criterion) {
    let h = HybridInference::new(None);
    let query = "Please analyze and explain this complex topic in detail; \
                 compare it with prior approaches and synthesize a recommendation. \
                 ```rust fn main() {} ```";
    c.bench_function("complexity_estimation", |b| {
        b.iter(|| black_box(h.estimate_complexity(black_box(query))))
    });
}

fn bench_should_use_cloud(c: &mut Criterion) {
    let h = HybridInference::new(None);
    c.bench_function("should_use_cloud_decision", |b| {
        b.iter(|| black_box(h.should_use_cloud(black_box(0.5), black_box(true))))
    });
}

fn bench_message_construction(c: &mut Criterion) {
    c.bench_function("message_user_construction", |b| {
        b.iter(|| black_box(Message::user(black_box("hello world"))))
    });
}

criterion_group!(
    benches,
    bench_complexity_estimation,
    bench_should_use_cloud,
    bench_message_construction
);
criterion_main!(benches);
