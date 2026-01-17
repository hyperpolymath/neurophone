;; SPDX-License-Identifier: AGPL-3.0-or-later
;; Neurophone Testing Report - Guile Scheme Format
;; Generated: 2025-12-29

(testing-report
  (metadata
    (version "1.0.0")
    (schema-version "1.0.0")
    (generated "2025-12-29T00:00:00Z")
    (generator "Claude Code (Automated Testing)")
    (project "neurophone")
    (repo "hyperpolymath/neurophone"))

  (summary
    (build-status 'pass)
    (test-count 28)
    (tests-passed 28)
    (tests-failed 0)
    (tests-ignored 0)
    (clippy-errors 0)
    (clippy-warnings 18)
    (total-test-duration-seconds 156))

  (issues-found
    (compilation-errors
      ((id "ERR-001")
       (severity 'error)
       (category 'api-change)
       (description "rand crate API changes in v0.9 - distributions renamed to distr")
       (files-affected
         ("crates/lsm/src/lib.rs"
          "crates/esn/src/lib.rs"
          "crates/bridge/src/lib.rs"))
       (fix-applied "Changed 'use rand::distributions' to 'use rand::distr'")
       (status 'fixed))

      ((id "ERR-002")
       (severity 'error)
       (category 'api-change)
       (description "Uniform::new() now returns Result in rand 0.9+")
       (files-affected
         ("crates/lsm/src/lib.rs"
          "crates/esn/src/lib.rs"))
       (fix-applied "Added .unwrap() to Uniform::new() calls")
       (status 'fixed))

      ((id "ERR-003")
       (severity 'error)
       (category 'type-inference)
       (description "Type annotation required for closure parameters in mapv()")
       (files-affected ("crates/esn/src/lib.rs"))
       (fix-applied "Added explicit type annotation |x: f32|")
       (status 'fixed))

      ((id "ERR-004")
       (severity 'error)
       (category 'api-change)
       (description "DateTime::from_timestamp_nanos does not exist in chrono")
       (files-affected ("crates/neurophone-android/src/lib.rs"))
       (fix-applied "Used DateTime::from_timestamp with manual conversion")
       (status 'fixed))

      ((id "ERR-005")
       (severity 'warning)
       (category 'deprecated)
       (description "rand::thread_rng() renamed to rand::rng()")
       (files-affected
         ("crates/lsm/src/lib.rs"
          "crates/esn/src/lib.rs"
          "crates/bridge/src/lib.rs"))
       (fix-applied "Replaced thread_rng() with rng()")
       (status 'fixed))

      ((id "ERR-006")
       (severity 'warning)
       (category 'deprecated)
       (description "rng.gen() renamed to rng.random()")
       (files-affected
         ("crates/lsm/src/lib.rs"
          "crates/esn/src/lib.rs"
          "crates/bridge/src/lib.rs"))
       (fix-applied "Replaced gen::<f32>() with random::<f32>()")
       (status 'fixed)))

    (clippy-issues
      ((id "CLIP-001")
       (lint "not_unsafe_ptr_arg_deref")
       (severity 'error)
       (file "crates/neurophone-android/src/lib.rs")
       (line 173)
       (fix-applied "Added #[allow(clippy::not_unsafe_ptr_arg_deref)] - valid JNI pattern")
       (status 'fixed))

      ((id "CLIP-002")
       (lint "await_holding_lock")
       (severity 'warning)
       (count 10)
       (file "crates/neurophone-android/src/lib.rs")
       (recommendation "Replace std::sync::Mutex with tokio::sync::Mutex")
       (status 'deferred))

      ((id "CLIP-003")
       (lint "static_mut_refs")
       (severity 'warning)
       (count 2)
       (file "crates/neurophone-android/src/lib.rs")
       (recommendation "Replace static mut with OnceLock for Rust 2024 compatibility")
       (status 'deferred))

      ((id "CLIP-004")
       (lint "unused_imports")
       (severity 'warning)
       (count 12)
       (fix-applied "Removed via cargo fix")
       (status 'fixed))))

  (test-results
    (crates
      ((name "bridge")
       (tests 4)
       (passed 4)
       (failed 0)
       (duration-seconds 4.96)
       (test-names
         ("test_bridge_creation"
          "test_encoding"
          "test_llm_context_generation"
          "test_state_integration")))

      ((name "claude-client")
       (tests 4)
       (passed 4)
       (failed 0)
       (duration-seconds 0.01)
       (test-names
         ("test_complexity_estimation"
          "test_hybrid_inference"
          "test_message_creation"
          "test_model_strings")))

      ((name "esn")
       (tests 4)
       (passed 4)
       (failed 0)
       (duration-seconds 0.63)
       (test-names
         ("test_esn_creation"
          "test_esn_echo_property"
          "test_esn_step"
          "test_hierarchical_esn")))

      ((name "llm")
       (tests 4)
       (passed 4)
       (failed 0)
       (duration-seconds 0.15)
       (test-names
         ("test_chat"
          "test_generation"
          "test_llm_creation"
          "test_llm_load")))

      ((name "lsm")
       (tests 4)
       (passed 4)
       (failed 0)
       (duration-seconds 140.53)
       (notes "test_lsm_reset is slow (~140s) due to large network size")
       (test-names
         ("test_firing_rates"
          "test_lsm_creation"
          "test_lsm_reset"
          "test_lsm_step")))

      ((name "neurophone-android")
       (tests 1)
       (passed 1)
       (failed 0)
       (duration-seconds 0.00)
       (test-names ("test_compile")))

      ((name "neurophone-core")
       (tests 2)
       (passed 2)
       (failed 0)
       (duration-seconds 5.44)
       (test-names
         ("test_system_creation"
          "test_system_start_stop")))

      ((name "sensors")
       (tests 5)
       (passed 5)
       (failed 0)
       (duration-seconds 0.01)
       (test-names
         ("test_activity_detection"
          "test_add_reading"
          "test_filter"
          "test_process_features"
          "test_sensor_processor_creation")))))

  (recommendations
    (high-priority
      ((id "REC-001")
       (category 'rust-2024-compat)
       (description "Replace static mut with OnceLock pattern")
       (affected-files ("crates/neurophone-android/src/lib.rs")))

      ((id "REC-002")
       (category 'async-safety)
       (description "Use tokio::sync::Mutex instead of std::sync::Mutex in async contexts")
       (affected-files ("crates/neurophone-android/src/lib.rs"))))

    (medium-priority
      ((id "REC-003")
       (category 'performance)
       (description "Optimize or mark slow LSM tests as ignored for CI"))

      ((id "REC-004")
       (category 'documentation)
       (description "Add doc tests for public API")))

    (low-priority
      ((id "REC-005")
       (category 'cleanup)
       (description "Apply remaining clippy suggestions (derivable_impls, manual_is_multiple_of)"))

      ((id "REC-006")
       (category 'dependencies)
       (description "Remove unused rayon dependency from lsm crate"))))

  (files-modified
    ("crates/lsm/src/lib.rs"
     "crates/esn/src/lib.rs"
     "crates/bridge/src/lib.rs"
     "crates/llm/src/lib.rs"
     "crates/sensors/src/lib.rs"
     "crates/claude-client/src/lib.rs"
     "crates/neurophone-core/src/lib.rs"
     "crates/neurophone-android/src/lib.rs"))

  (conclusion
    (overall-status 'pass)
    (build-works #t)
    (tests-pass #t)
    (production-ready #f)
    (notes "Project compiles and passes all tests after fixing rand 0.9 API migration issues. Architectural warnings in Android JNI bindings should be addressed for Rust 2024 compatibility. The neurosymbolic pipeline (LSM -> ESN -> Bridge -> LLM) is functional.")))
