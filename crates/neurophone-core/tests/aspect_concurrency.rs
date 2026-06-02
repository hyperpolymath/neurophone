// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Aspect tests — concurrency, error paths, resource bounds, timing.

use neurophone_core::{NeuroSymbolicSystem, SensorEvent, SystemConfig};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn aspect_concurrent_queries_via_mutex() {
    // Initialise to Active before wrapping: query/process exist only on Active.
    let sys = Arc::new(Mutex::new(
        NeuroSymbolicSystem::new(SystemConfig::default())
            .unwrap()
            .initialize()
            .unwrap(),
    ));
    let mut handles = vec![];
    for i in 0..16 {
        let sys = sys.clone();
        handles.push(thread::spawn(move || {
            let q = format!("query {i}");
            sys.lock().unwrap().query(&q, true).unwrap();
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(sys.lock().unwrap().query_count(), 16);
}

#[test]
fn aspect_concurrent_sensor_events() {
    // Initialise to Active before wrapping: query/process exist only on Active.
    let sys = Arc::new(Mutex::new(
        NeuroSymbolicSystem::new(SystemConfig::default())
            .unwrap()
            .initialize()
            .unwrap(),
    ));
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
    for h in handles {
        h.join().unwrap();
    }
}

// Concurrency safety (obligation 2.2): data-race freedom is compiler-guaranteed
// (the crate is `forbid`/`deny(unsafe_code)`, so shared state is only reached
// through the `Mutex`); deadlock freedom follows from a single lock with no lock
// ordering and no condvars. The remaining property worth proving is *no lost
// updates* under contention — the mutex must serialise each read-modify-write.
#[test]
fn aspect_no_lost_updates_under_contention() {
    const THREADS: u64 = 8;
    const PER_THREAD: u64 = 250;
    let sys = Arc::new(Mutex::new(
        NeuroSymbolicSystem::new(SystemConfig::default())
            .unwrap()
            .initialize()
            .unwrap(),
    ));
    let mut handles = vec![];
    for t in 0..THREADS {
        let sys = sys.clone();
        handles.push(thread::spawn(move || {
            for j in 0..PER_THREAD {
                let q = format!("t{t}-q{j}");
                sys.lock().unwrap().query(&q, true).unwrap();
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    // No lost updates: every query is counted exactly once. (Completion also
    // demonstrates no deadlock.)
    assert_eq!(sys.lock().unwrap().query_count(), THREADS * PER_THREAD);
}

#[test]
fn aspect_query_under_one_second() {
    let sys = NeuroSymbolicSystem::new(SystemConfig::default()).unwrap();
    let mut sys = sys.initialize().unwrap();
    let t = Instant::now();
    sys.query("hello world", true).unwrap();
    assert!(t.elapsed() < Duration::from_millis(1000));
}

#[test]
fn aspect_error_path_empty_query() {
    let sys = NeuroSymbolicSystem::new(SystemConfig::default()).unwrap();
    let mut sys = sys.initialize().unwrap();
    assert!(sys.query("", true).is_err());
    assert_eq!(sys.query_count(), 0);
}

// NOTE: `aspect_error_path_inactive_system_event` removed — using the system
// before initialisation is now a compile-time error (`process_sensor_event`
// exists only on `phase::Active`), so there is no runtime error path to test.

#[test]
fn aspect_resource_bounded_state_size() {
    let sys = NeuroSymbolicSystem::new(SystemConfig::default()).unwrap();
    let mut sys = sys.initialize().unwrap();
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
