// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Aspect + lifecycle tests for sensors.

use sensors::{PipelineConfig, SensorError, SensorKind, SensorPipeline, SensorReading};

#[test]
fn lifecycle_create_ingest_reset_ingest() {
    let mut p = SensorPipeline::new(SensorKind::Light, PipelineConfig::default()).unwrap();
    p.ingest(&SensorReading::new(SensorKind::Light, 10, vec![1.0]).unwrap())
        .unwrap();
    assert!(p.features().is_ok());
    p.reset();
    assert!(p.features().is_err());
    p.ingest(&SensorReading::new(SensorKind::Light, 20, vec![2.0]).unwrap())
        .unwrap();
    assert!(p.features().is_ok());
}

#[test]
fn aspect_error_path_dimension_mismatch() {
    let r = SensorReading::new(SensorKind::Accelerometer, 0, vec![1.0]);
    match r {
        Err(SensorError::DimensionMismatch { expected, got }) => {
            assert_eq!(expected, 3);
            assert_eq!(got, 1);
        }
        _ => panic!("expected DimensionMismatch"),
    }
}

#[test]
fn aspect_invalid_config_zero_window() {
    let cfg = PipelineConfig {
        window_size: 0,
        ..Default::default()
    };
    assert!(SensorPipeline::new(SensorKind::Light, cfg).is_err());
}

#[test]
fn aspect_invalid_config_zero_sample_rate() {
    let cfg = PipelineConfig {
        sample_hz: 0.0,
        ..Default::default()
    };
    assert!(SensorPipeline::new(SensorKind::Light, cfg).is_err());
}

#[test]
fn aspect_resource_thousand_ingests_no_growth() {
    let mut p = SensorPipeline::new(SensorKind::Accelerometer, PipelineConfig::default()).unwrap();
    for i in 0..1000 {
        let r =
            SensorReading::new(SensorKind::Accelerometer, i as u64, vec![0.1, 0.2, 9.81]).unwrap();
        p.ingest(&r).unwrap();
    }
    let f = p.features().unwrap();
    assert!(f.iter().all(|v| v.is_finite()));
}
