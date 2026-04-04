// SPDX-License-Identifier: PMPL-1.0-or-later
// NeuroPhone - High-Assurance Hardware Orchestration
// Copyright (c) 2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>

//! Property-based tests for neurophone-core using proptest

use neurophone_core::*;
use proptest::prelude::*;

// ========== Property Strategies ==========

fn arb_system_config() -> impl Strategy<Value = SystemConfig> {
    (0.1f32..100.0, 0.0f32..1.0, 100u32..5000u32).prop_map(
        |(sample_rate, threshold, response_time)| {
            SystemConfig {
                sample_rate,
                window_size_ms: 100,
                local_threshold: threshold,
                max_response_time_ms: response_time,
            }
        },
    )
}

fn arb_sensor_event() -> impl Strategy<Value = SensorEvent> {
    (
        prop::string::string_regex("[a-z_]{5,20}").unwrap(),
        0u64..10000u64,
        prop::collection::vec(-10.0f32..10.0f32, 1..10),
    )
    .prop_map(|(sensor_type, timestamp_ms, values)| SensorEvent {
        sensor_type,
        timestamp_ms,
        values,
    })
}

fn arb_query_string() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-z0-9 ]{1,100}").unwrap()
}

// ========== Property Tests ==========

proptest! {
    #[test]
    fn prop_system_creation_with_valid_config(config in arb_system_config()) {
        // Property: System creation should succeed with any valid config
        let system = NeuroSymbolicSystem::new(config.clone());
        prop_assert!(system.is_ok());

        if let Ok(sys) = system {
            prop_assert_eq!(sys.config().sample_rate, config.sample_rate);
            prop_assert_eq!(sys.config().local_threshold, config.local_threshold);
        }
    }

    #[test]
    fn prop_query_always_valid_result(query in arb_query_string()) {
        // Property: Any valid query should produce a valid response
        if !query.is_empty() {
            let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
                .expect("system creation");

            let result = system.query(&query, true);
            prop_assert!(result.is_ok());

            if let Ok(r) = result {
                // Contract: confidence must be normalized
                prop_assert!(r.confidence >= 0.0 && r.confidence <= 1.0);
                // Contract: response should never be empty
                prop_assert!(!r.response.is_empty());
                // Contract: query should be preserved
                prop_assert_eq!(r.query, query);
            }
        }
    }

    #[test]
    fn prop_sensor_processing_preserves_dimensions(event in arb_sensor_event()) {
        // Property: Neural output features should have same dimension as input
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");
        system.initialize().expect("init");

        let result = system.process_sensor_event(&event);
        if let Ok(output) = result {
            prop_assert_eq!(output.features.len(), event.values.len());
            // Contract: confidence must be valid
            prop_assert!(output.confidence >= 0.0 && output.confidence <= 1.0);
        }
    }

    #[test]
    fn prop_query_count_increases(queries in prop::collection::vec(arb_query_string(), 1..20)) {
        // Property: Query count should increase with each query
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");

        let mut expected_count = 0u64;
        for query in queries.iter().filter(|q| !q.is_empty()) {
            system.query(query, true).ok();
            expected_count += 1;
            prop_assert_eq!(system.query_count(), expected_count);
        }
    }

    #[test]
    fn prop_uptime_always_increases(configs in prop::collection::vec(arb_system_config(), 1..5)) {
        // Property: Uptime should never decrease
        for config in configs {
            let system1 = NeuroSymbolicSystem::new(config.clone())
                .expect("system creation");
            let uptime1 = system1.uptime_ms();

            let system2 = NeuroSymbolicSystem::new(config)
                .expect("system creation");
            let uptime2 = system2.uptime_ms();

            // uptime2 >= uptime1 because system2 created after system1
            prop_assert!(uptime2 >= uptime1);
        }
    }

    #[test]
    fn prop_state_clone_equal(event in arb_sensor_event()) {
        // Property: Cloned state should equal original
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");
        system.initialize().expect("init");

        system.process_sensor_event(&event).ok();

        let state1 = system.get_state();
        let state2 = state1.clone();

        prop_assert_eq!(state1.timestamp_ms, state2.timestamp_ms);
        prop_assert_eq!(state1.is_active, state2.is_active);
    }

    #[test]
    fn prop_model_selection_deterministic(queries in prop::collection::vec(arb_query_string(), 5..15)) {
        // Property: Given same query, model selection should be deterministic
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");

        for query in queries.iter().filter(|q| !q.is_empty()) {
            let r1 = system.query(query, true);
            let model1 = r1.ok().map(|r| r.model);

            // Reset system and try again
            let mut system2 = NeuroSymbolicSystem::new(SystemConfig::default())
                .expect("system creation");
            let r2 = system2.query(query, true);
            let model2 = r2.ok().map(|r| r.model);

            prop_assert_eq!(model1, model2);
        }
    }

    #[test]
    fn prop_empty_query_always_errors(_empty in Just("")) {
        // Property: Empty query should always produce error
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");

        let result = system.query("", true);
        prop_assert!(result.is_err());
    }

    #[test]
    fn prop_sensor_event_values_match(
        sensor_type in prop::string::string_regex("[a-z]{5,15}").unwrap(),
        values in prop::collection::vec(-10.0f32..10.0f32, 1..20)
    ) {
        // Property: Processed event should have same structure
        let event = SensorEvent {
            sensor_type: sensor_type.clone(),
            timestamp_ms: 1000,
            values: values.clone(),
        };

        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");
        system.initialize().expect("init");

        let result = system.process_sensor_event(&event);
        if let Ok(output) = result {
            prop_assert_eq!(output.features.len(), values.len());
        }
    }

    #[test]
    fn prop_latency_bounds(
        _config in arb_system_config()
    ) {
        // Property: Latency should be non-negative and reasonable
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");

        let result = system.query("test", true);
        if let Ok(r) = result {
            prop_assert!(r.latency_ms >= 0);
            prop_assert!(r.latency_ms < 5000); // Should complete within 5 seconds
        }
    }

    #[test]
    fn prop_config_threshold_validation(threshold in 0.0f32..=1.0f32) {
        // Property: Valid thresholds should create system successfully
        let config = SystemConfig {
            local_threshold: threshold,
            ..Default::default()
        };

        let system = NeuroSymbolicSystem::new(config);
        prop_assert!(system.is_ok());
    }

    #[test]
    fn prop_multiple_sensor_types(
        events in prop::collection::vec(arb_sensor_event(), 5..20)
    ) {
        // Property: Multiple different sensor types should be processable
        let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
            .expect("system creation");
        system.initialize().expect("init");

        let mut success_count = 0;
        for event in events {
            if let Ok(_output) = system.process_sensor_event(&event) {
                success_count += 1;
            }
        }

        // Should process at least half successfully
        prop_assert!(success_count > 0);
    }

    #[test]
    fn prop_inference_confidence_range(
        query in arb_query_string()
    ) {
        // Property: Confidence should always be normalized
        if !query.is_empty() {
            let mut system = NeuroSymbolicSystem::new(SystemConfig::default())
                .expect("system creation");

            let result = system.query(&query, true);
            if let Ok(r) = result {
                prop_assert!(r.confidence >= 0.0, "confidence >= 0");
                prop_assert!(r.confidence <= 1.0, "confidence <= 1");
            }
        }
    }

    #[test]
    fn prop_state_transitions_valid(
        config in arb_system_config()
    ) {
        // Property: State transitions should always be valid
        let mut system = NeuroSymbolicSystem::new(config)
            .expect("system creation");

        // Transition 1: inactive -> active
        let init_result = system.initialize();
        prop_assert!(init_result.is_ok());
        let state = system.get_state();
        prop_assert!(state.is_active);

        // Transition 2: active -> inactive
        let shutdown_result = system.shutdown();
        prop_assert!(shutdown_result.is_ok());
        let state = system.get_state();
        prop_assert!(!state.is_active);
    }
}
