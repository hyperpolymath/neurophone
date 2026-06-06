// SPDX-License-Identifier: MPL-2.0
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
    let sys = NeuroSymbolicSystem::new(SystemConfig::default()).unwrap();
    let mut sys = sys.initialize().unwrap();
    for i in 0..10 {
        sys.process_sensor_event(&make_event(i * 20)).unwrap();
    }
    sys.query("status?", true).unwrap();
    let sys = sys.shutdown();
    assert!(!sys.get_state().is_active);
}

// NOTE: `restart_after_shutdown_works` removed — shutdown is terminal in the
// typestate API. A `Down` system has no `initialize`, so restart cannot be
// expressed (compile-time enforced).

// NOTE: `process_after_shutdown_errors` removed — using the system after
// shutdown is now a compile-time error (`process_sensor_event` exists only on
// `phase::Active`), so there is no runtime error path left to test.

#[test]
fn query_count_counts_queries_while_active() {
    let sys = NeuroSymbolicSystem::new(SystemConfig::default()).unwrap();
    let mut sys = sys.initialize().unwrap();
    sys.query("a", true).unwrap();
    sys.query("b", true).unwrap();
    sys.query("c", true).unwrap();
    let sys = sys.shutdown();
    // Count persists into the Down phase (getter available in every phase).
    assert_eq!(sys.query_count(), 3);
}

// NOTE: `shutdown_idempotent` removed — double shutdown is impossible in the
// typestate API (`shutdown` consumes the `Active` system and returns `Down`,
// which has no `shutdown`); compile-time enforced.

#[test]
fn uptime_monotonically_nondecreasing() {
    let sys = NeuroSymbolicSystem::new(SystemConfig::default()).unwrap();
    let a = sys.uptime_ms();
    std::thread::sleep(std::time::Duration::from_millis(2));
    let b = sys.uptime_ms();
    assert!(b >= a);
}
