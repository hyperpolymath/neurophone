// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Integration tests for the sensors crate (point-to-point: reading -> features).

use sensors::{
    IirFilter, PipelineConfig, SensorKind, SensorPipeline, SensorReading, WindowedFeatures,
};

#[test]
fn pipeline_50hz_one_second_yields_features() {
    let mut p = SensorPipeline::new(SensorKind::Accelerometer, PipelineConfig::default()).unwrap();
    for i in 0..50 {
        let r = SensorReading::new(
            SensorKind::Accelerometer,
            i as u64 * 20,
            vec![0.1, 0.2, 9.81],
        )
        .unwrap();
        p.ingest(&r).unwrap();
    }
    let f = p.features().unwrap();
    assert_eq!(f.len(), 7);
    assert!(f.iter().all(|v| v.is_finite()));
}

#[test]
fn light_sensor_singlechannel_pipeline() {
    let mut p = SensorPipeline::new(SensorKind::Light, PipelineConfig::default()).unwrap();
    for i in 0..30 {
        let r =
            SensorReading::new(SensorKind::Light, i as u64 * 20, vec![100.0 + i as f32]).unwrap();
        p.ingest(&r).unwrap();
    }
    let f = p.features().unwrap();
    assert_eq!(f.len(), 3);
}

#[test]
fn filter_chain_preserves_dimensionality() {
    let mut lp = IirFilter::new(3, 5.0, 50.0, false).unwrap();
    let mut hp = IirFilter::new(3, 0.5, 50.0, true).unwrap();
    let lp_out = lp.step(&[1.0, 2.0, 3.0]).unwrap();
    let hp_out = hp.step(&lp_out).unwrap();
    assert_eq!(hp_out.len(), 3);
}

#[test]
fn windowed_features_after_overflow_stays_bounded() {
    let mut w = WindowedFeatures::new(2, 5).unwrap();
    for i in 0..100 {
        w.push(&[i as f32, -(i as f32)]).unwrap();
    }
    assert_eq!(w.len(), 5);
    let f = w.features().unwrap();
    assert!(f.iter().all(|v| v.is_finite()));
}

#[test]
fn timestamps_propagate_in_order() {
    let mut p = SensorPipeline::new(SensorKind::Gyroscope, PipelineConfig::default()).unwrap();
    let mut last = 0u64;
    for i in 0..20 {
        let ts = i as u64 * 25;
        let r = SensorReading::new(SensorKind::Gyroscope, ts, vec![0.0; 3]).unwrap();
        p.ingest(&r).unwrap();
        assert_eq!(p.last_timestamp_ms(), Some(ts));
        assert!(ts >= last);
        last = ts;
    }
}
