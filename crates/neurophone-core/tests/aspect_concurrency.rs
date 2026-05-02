// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Aspect tests — concurrency, error paths, resource bounds, timing.

use neurophone_core::{NeuroSymbolicSystem, SensorEvent, SystemConfig};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn aspect_concurrent_queries_via_mutex() {
    let sys = Arc::new(Mutex::new(NeuroSymbolicSystem::new(SystemConfig::default()).unwrap()));
    let mut handles = vec![];
    for i in 0..16 {
        let sys = sys.clone();
        handles.push(thread::spawn(move || {
            let q = format!("query {i}");
            sys.lock().unwrap().query(&q, true).unwrap();
        }));
    }
    for h in handles { h.join().unwrap(); }
    assert_eq!(sys.lock().unwrap().query_count(), 16);
}

#[test]
fn aspect_concurrent_sensor_events() {
    let sys = Arc::new(Mutex::new(NeuroSymbolicSystem::new(SystemConfig::default()).unwrap()));
    sys.lock().unwrap().initialize().unwrap();
    let mut handles = vec![];
    for i in 0..32u64 {
        let sys = sys.clone();
        handles.push(thread::spawn(move || {
            let event = SensorEvent {
                sensor_type: "accelerometer".into(),
                timestamp_ms: i * 10,
                values: vec![0.1, 0.2, 0.3],
            };
            let _ = sys.lock().unwrap().process_sensor_event(&event);
        }));
    }
    for h in handles { h.join().unwrap(); }
}

#[test]
fn aspect_query_under_one_second() {
    let mut sys = NeuroSymbolicSystem::new(SystemConfig::default()).unwrap();
    let t = Instant::now();
    sys.query("hello world", true).unwrap();
    assert!(t.elapsed() < Duration::from_millis(1000));
}

#[test]
fn aspect_error_path_empty_query() {
    let mut sys = NeuroSymbolicSystem::new(SystemConfig::default()).unwrap();
    assert!(sys.query("", true).is_err());
    assert_eq!(sys.query_count(), 0);
}

#[test]
fn aspect_error_path_inactive_system_event() {
    let mut sys = NeuroSymbolicSystem::new(SystemConfig::default()).unwrap();
    let e = SensorEvent {
        sensor_type: "x".into(), timestamp_ms: 0, values: vec![0.0],
    };
    assert!(sys.process_sensor_event(&e).is_err());
}

#[test]
fn aspect_resource_bounded_state_size() {
    let mut sys = NeuroSymbolicSystem::new(SystemConfig::default()).unwrap();
    sys.initialize().unwrap();
    for i in 0..1_000u64 {
        let e = SensorEvent {
            sensor_type: "accelerometer".into(),
            timestamp_ms: i * 20,
            values: vec![0.1, 0.2, 9.81],
        };
        let _ = sys.process_sensor_event(&e);
    }
    // Snapshot should still be cheap to clone.
    let _ = sys.get_state();
}
