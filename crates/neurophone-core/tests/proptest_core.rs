// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>
//! Property tests for the core orchestrator — panic-freedom (obligation 0.1)
//! and numeric containment (obligation 0.2), issue #84, on the operational
//! entrypoints `process_sensor_event` and `query_routed`.
//!
//! For arbitrary finite sensor inputs and arbitrary query text, the operational
//! paths must never panic and must return contained values: finite feature
//! vectors whose length matches the input, and confidences within `[0, 1]`.

use neurophone_core::{NeuroSymbolicSystem, QueryRoute, SensorEvent, SystemConfig};
use proptest::prelude::*;

fn active() -> NeuroSymbolicSystem {
    NeuroSymbolicSystem::new(SystemConfig::default())
        .expect("valid default config")
        .initialize()
        .expect("initialise")
}

proptest! {
    /// 0.1 + 0.2 on the sensor path: never panics; features stay finite and
    /// length-preserving; confidence stays in range.
    #[test]
    fn process_sensor_event_is_panic_free_and_contained(
        sensor_type in "[a-zA-Z_]{0,16}",
        ts in any::<u64>(),
        values in proptest::collection::vec(-1.0e9f32..1.0e9f32, 0..64),
    ) {
        let mut sys = active();
        let event = SensorEvent {
            sensor_type,
            timestamp_ms: ts,
            values: values.clone(),
        };
        let out = sys
            .process_sensor_event(&event)
            .expect("process_sensor_event must succeed on finite input");
        prop_assert!(out.features.iter().all(|x| x.is_finite()));
        prop_assert_eq!(out.features.len(), values.len());
        prop_assert!((0.0..=1.0).contains(&out.confidence));
    }

    /// 0.1 + 0.2 on the query path, across every routing mode: never panics; the
    /// only permitted error is the documented empty-message case; confidence
    /// stays in range and the response is non-empty.
    #[test]
    fn query_is_panic_free_and_contained(
        message in ".{0,256}",
        route in prop_oneof![
            Just(QueryRoute::Auto),
            Just(QueryRoute::ForceLocal),
            Just(QueryRoute::ForceCloud),
        ],
    ) {
        let mut sys = active();
        match sys.query_routed(&message, route) {
            Ok(r) => {
                prop_assert!((0.0..=1.0).contains(&r.confidence));
                prop_assert!(!r.response.is_empty());
            }
            Err(_) => {
                // The sole documented failure mode is an empty query.
                prop_assert!(message.is_empty());
            }
        }
    }
}
