// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Lifecycle tests: init → start → run → shutdown → restart.

use neurophone_core::{NeuroSymbolicSystem, SensorEvent, SystemConfig};

fn make_event(ts: u64) -> SensorEvent {
    SensorEvent {
        sensor_type: "accelerometer".into(),
        timestamp_ms: ts,
        values: vec![0.1, 0.2, 9.81],
    }
}

#[test]
fn full_lifecycle_init_run_shutdown() {
    let mut sys = NeuroSymbolicSystem::new(SystemConfig::default()).unwrap();
    sys.initialize().unwrap();
    for i in 0..10 {
        sys.process_sensor_event(&make_event(i * 20)).unwrap();
    }
    sys.query("status?", true).unwrap();
    sys.shutdown().unwrap();
    assert!(!sys.get_state().is_active);
}

#[test]
fn restart_after_shutdown_works() {
    let mut sys = NeuroSymbolicSystem::new(SystemConfig::default()).unwrap();
    sys.initialize().unwrap();
    sys.shutdown().unwrap();
    sys.initialize().unwrap();
    assert!(sys.get_state().is_active);
}

#[test]
fn process_after_shutdown_errors() {
    let mut sys = NeuroSymbolicSystem::new(SystemConfig::default()).unwrap();
    sys.initialize().unwrap();
    sys.shutdown().unwrap();
    assert!(sys.process_sensor_event(&make_event(100)).is_err());
}

#[test]
fn query_count_persists_across_state_transitions() {
    let mut sys = NeuroSymbolicSystem::new(SystemConfig::default()).unwrap();
    sys.query("a", true).unwrap();
    sys.initialize().unwrap();
    sys.query("b", true).unwrap();
    sys.shutdown().unwrap();
    sys.query("c", true).unwrap();
    assert_eq!(sys.query_count(), 3);
}

#[test]
fn shutdown_idempotent() {
    let mut sys = NeuroSymbolicSystem::new(SystemConfig::default()).unwrap();
    sys.initialize().unwrap();
    sys.shutdown().unwrap();
    sys.shutdown().unwrap();
    assert!(!sys.get_state().is_active);
}

#[test]
fn uptime_monotonically_nondecreasing() {
    let sys = NeuroSymbolicSystem::new(SystemConfig::default()).unwrap();
    let a = sys.uptime_ms();
    std::thread::sleep(std::time::Duration::from_millis(2));
    let b = sys.uptime_ms();
    assert!(b >= a);
}
