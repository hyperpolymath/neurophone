// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Benches for the sensors pipeline.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use sensors::{
    IirFilter, PipelineConfig, SensorKind, SensorPipeline, SensorReading, WindowedFeatures,
};
use std::hint::black_box;

fn bench_iir_lowpass_step(c: &mut Criterion) {
    let mut f = IirFilter::new(3, 5.0, 50.0, false).unwrap();
    c.bench_function("iir_lowpass_step_3ch", |b| {
        b.iter(|| {
            let _ = black_box(f.step(black_box(&[1.0, 2.0, 3.0])));
        })
    });
}

fn bench_iir_highpass_step(c: &mut Criterion) {
    let mut f = IirFilter::new(3, 0.5, 50.0, true).unwrap();
    c.bench_function("iir_highpass_step_3ch", |b| {
        b.iter(|| {
            let _ = black_box(f.step(black_box(&[1.0, 2.0, 3.0])));
        })
    });
}

fn bench_windowed_features(c: &mut Criterion) {
    let mut g = c.benchmark_group("windowed_features");
    for &cap in &[10usize, 50, 200] {
        let mut w = WindowedFeatures::new(3, cap).unwrap();
        for _ in 0..cap {
            w.push(&[0.1, 0.2, 0.3]).unwrap();
        }
        g.bench_function(format!("features_cap{cap}"), |b| {
            b.iter(|| black_box(w.features().unwrap()))
        });
    }
    g.finish();
}

fn bench_pipeline_ingest_50hz(c: &mut Criterion) {
    let mut g = c.benchmark_group("pipeline_ingest");
    g.throughput(Throughput::Elements(1));
    let mut p = SensorPipeline::new(SensorKind::Accelerometer, PipelineConfig::default()).unwrap();
    let mut t = 0u64;
    g.bench_function("accelerometer_3ch", |b| {
        b.iter(|| {
            t += 20;
            let r = SensorReading::new(SensorKind::Accelerometer, t, vec![0.1, 0.2, 9.81]).unwrap();
            p.ingest(black_box(&r)).unwrap();
        })
    });
    g.finish();
}

fn bench_pipeline_full_second(c: &mut Criterion) {
    c.bench_function("pipeline_50hz_full_second", |b| {
        b.iter(|| {
            let mut p =
                SensorPipeline::new(SensorKind::Accelerometer, PipelineConfig::default()).unwrap();
            for i in 0..50u64 {
                let r = SensorReading::new(SensorKind::Accelerometer, i * 20, vec![0.1, 0.2, 9.81])
                    .unwrap();
                p.ingest(&r).unwrap();
            }
            black_box(p.features().unwrap());
        })
    });
}

criterion_group!(
    benches,
    bench_iir_lowpass_step,
    bench_iir_highpass_step,
    bench_windowed_features,
    bench_pipeline_ingest_50hz,
    bench_pipeline_full_second
);
criterion_main!(benches);
