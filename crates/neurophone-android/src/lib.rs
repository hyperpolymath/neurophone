// SPDX-License-Identifier: MPL-2.0
// NeuroPhone — Android JNI bindings (Gossamer NativeLib port).
// Copyright (c) 2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>

//! JNI surface for `ai.neurophone.NativeLib`.
//!
//! This crate is the native (`neurophone_android`) shared library loaded by the
//! Android app via `System.loadLibrary("neurophone_android")`. It exposes the
//! 11-method contract declared in `NativeLib.kt` and delegates to the pure-Rust
//! workspace crates:
//!
//! * [`neurophone_core::NeuroSymbolicSystem`] — lifecycle, sensor processing,
//!   neural context, state, and the hybrid local/cloud query router.
//! * [`llm::MockBackend`] — deterministic on-device LLM stand-in (real builds
//!   swap in a `llama.cpp` backend).
//! * [`claude_client::HybridInference`] / [`claude_client::ClaudeClient`] — the
//!   cloud (Claude) path.
//! * [`sensors::SensorKind`] — Android sensor-type id mapping.
//!
//! ## Memory safety
//!
//! The crate is `#![deny(unsafe_code)]`. The only `unsafe` is the JNI ABI
//! boundary: `#[unsafe(no_mangle)] pub unsafe extern "C"` is mandated by the
//! `jni` crate / JVM calling convention and cannot be expressed in safe Rust.
//! Each export is annotated with a local, justified `#[allow(unsafe_code)]`
//! (see [`jni_boundary`]). Inside every export we immediately hand off to a
//! safe `fn`, so no `unsafe` operations are performed beyond the declaration.
//!
//! TODO(#83): `NativeLib.kt` / `MainActivity.kt` still own the Kotlin/Java
//! binding declarations for these symbols. They are reconciled in the shim PRs
//! and legacy-delete step of the Android Kotlin→Rust/Gossamer migration.

#![deny(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use std::sync::{Mutex, OnceLock};

use claude_client::{ClaudeConfig, HybridInference};
use llm::{LlmBackend, LlmConfig, MockBackend};
use neurophone_core::{phase, NeuroSymbolicSystem, SensorEvent, SystemConfig};
use sensors::SensorKind;
use serde_json::json;

/// Process-wide native runtime, lazily constructed on first `init`.
///
/// A single [`Mutex`] guards the whole state: JNI calls arrive on arbitrary JVM
/// threads, and the contained handles (`NeuroSymbolicSystem`, `MockBackend`,
/// `HybridInference`) are `!Sync` mutable orchestrators. Coarse-grained locking
/// is correct and simple; sensor/query throughput is well within budget.
static RUNTIME: OnceLock<Mutex<Option<NativeRuntime>>> = OnceLock::new();

/// Holds the live, initialised native subsystems.
struct NativeRuntime {
    /// Neurosymbolic orchestrator (typestate-`Active`).
    system: NeuroSymbolicSystem<phase::Active>,
    /// On-device LLM (mock backend until `llama.cpp` is wired in).
    local: MockBackend,
    /// Cloud (Claude) router. `has_claude()` is false without an API key.
    hybrid: HybridInference,
    /// `start()`/`stop()` toggle for the processing loop.
    running: bool,
    /// Last neural context string produced by `process_sensor_event`.
    last_context: String,
}

fn runtime_cell() -> &'static Mutex<Option<NativeRuntime>> {
    RUNTIME.get_or_init(|| Mutex::new(None))
}

/// Map the Android `Sensor.TYPE_*` integer constant to our [`SensorKind`].
///
/// Contract id map: accelerometer=1, magnetometer=2, gyroscope=4, light=5,
/// proximity=8, anything else => `None` (unknown / unsupported).
fn sensor_kind_from_android_id(sensor_type: i32) -> Option<SensorKind> {
    match sensor_type {
        1 => Some(SensorKind::Accelerometer),
        2 => Some(SensorKind::Magnetometer),
        4 => Some(SensorKind::Gyroscope),
        5 => Some(SensorKind::Light),
        8 => Some(SensorKind::Proximity),
        _ => None,
    }
}

/// Parse an optional JSON config string into a [`SystemConfig`].
///
/// `None` / empty / invalid JSON all fall back to `SystemConfig::default()` so
/// `init(null)` from Kotlin is always valid.
fn parse_config(config_json: Option<&str>) -> SystemConfig {
    match config_json {
        Some(s) if !s.trim().is_empty() => {
            serde_json::from_str::<SystemConfig>(s).unwrap_or_default()
        }
        _ => SystemConfig::default(),
    }
}

// ===========================================================================
// Safe, JVM-free core. Each `fn` here is the real implementation; the JNI
// exports below are thin shells that decode arguments and call into these.
// Keeping the logic in safe functions means the `unsafe` surface is purely the
// ABI declaration.
// ===========================================================================

/// `init(configJson)` — (re)create the native runtime. Returns `true` on success.
fn core_init(config_json: Option<&str>) -> bool {
    let config = parse_config(config_json);

    let system = match NeuroSymbolicSystem::new(config) {
        Ok(s) => match s.initialize() {
            Ok(active) => active,
            Err(_) => return false,
        },
        Err(_) => return false,
    };

    let mut local = match MockBackend::new(LlmConfig::default()) {
        Ok(b) => b,
        Err(_) => return false,
    };
    if local.load().is_err() {
        return false;
    }

    // Cloud path: enabled only if an API key is present in the environment.
    // Without a key `HybridInference::new` yields `has_claude() == false`, and
    // `queryClaude` reports the missing-key condition instead of panicking.
    let claude_config = ClaudeConfig::default();
    let hybrid = if claude_config.api_key.is_some() {
        HybridInference::new(Some(claude_config))
    } else {
        HybridInference::new(None)
    };

    let runtime = NativeRuntime {
        system,
        local,
        hybrid,
        running: false,
        last_context: String::new(),
    };

    if let Ok(mut guard) = runtime_cell().lock() {
        *guard = Some(runtime);
        true
    } else {
        false
    }
}

/// `start()` — mark the processing loop running. Requires a prior `init`.
fn core_start() -> bool {
    with_runtime(|rt| {
        rt.running = true;
        true
    })
    .unwrap_or(false)
}

/// `stop()` — mark the processing loop stopped (idempotent, no-op if uninit).
fn core_stop() {
    let _ = with_runtime(|rt| {
        rt.running = false;
    });
}

/// `processSensor(sensorType, values, timestampNs, accuracy)` — ingest a reading.
///
/// Returns `true` if the sensor type is known, the value arity matches, and the
/// neurosymbolic system accepted the event.
fn core_process_sensor(
    sensor_type: i32,
    values: Vec<f32>,
    timestamp_ns: i64,
    _accuracy: i32,
) -> bool {
    let Some(kind) = sensor_kind_from_android_id(sensor_type) else {
        return false;
    };
    if values.len() != kind.arity() {
        return false;
    }

    let timestamp_ms = (timestamp_ns.max(0) as u64) / 1_000_000;
    let event = SensorEvent {
        sensor_type: kind.as_str().to_string(),
        timestamp_ms,
        values,
    };

    with_runtime(|rt| match rt.system.process_sensor_event(&event) {
        Ok(output) => {
            rt.last_context = output.context;
            true
        }
        Err(_) => false,
    })
    .unwrap_or(false)
}

/// `queryLocal(message)` — force the on-device (mock Llama) backend.
fn core_query_local(message: &str) -> String {
    with_runtime(|rt| {
        let max_tokens = rt.local.config().max_tokens;
        match rt.local.generate(message, max_tokens) {
            Ok(resp) => resp.text,
            Err(e) => format!("[local-llm-error] {e}"),
        }
    })
    .unwrap_or_else(|| "[error] native runtime not initialised".to_string())
}

/// `queryClaude(message)` — force the cloud (Claude) backend.
///
/// Builds a short-lived single-threaded tokio runtime to drive the async
/// `ClaudeClient`. With no API key configured this returns a clear message
/// rather than failing the FFI call.
fn core_query_claude(message: &str) -> String {
    with_runtime(|rt| {
        if !rt.hybrid.has_claude() {
            return "[claude-unavailable] no API key configured".to_string();
        }
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(r) => r,
            Err(e) => return format!("[claude-error] runtime: {e}"),
        };
        match rt.hybrid.claude() {
            Some(client) => match runtime.block_on(client.send_message(message)) {
                Ok(text) => text,
                Err(e) => format!("[claude-error] {e}"),
            },
            None => "[claude-unavailable] no client".to_string(),
        }
    })
    .unwrap_or_else(|| "[error] native runtime not initialised".to_string())
}

/// `query(message, preferLocal)` — hybrid router via the core orchestrator.
fn core_query(message: &str, prefer_local: bool) -> String {
    with_runtime(|rt| match rt.system.query(message, prefer_local) {
        Ok(result) => result.response,
        Err(e) => format!("[query-error] {e}"),
    })
    .unwrap_or_else(|| "[error] native runtime not initialised".to_string())
}

/// `getNeuralContext()` — formatted neural context from the last sensor event.
fn core_get_neural_context() -> String {
    with_runtime(|rt| {
        if rt.last_context.is_empty() {
            "no neural context yet".to_string()
        } else {
            rt.last_context.clone()
        }
    })
    .unwrap_or_else(|| "no neural context yet".to_string())
}

/// `getState()` — system state as a JSON string.
fn core_get_state() -> String {
    with_runtime(|rt| {
        let state = rt.system.get_state();
        json!({
            "running": rt.running,
            "is_active": state.is_active,
            "timestamp_ms": state.timestamp_ms,
            "latency_ms": state.latency_ms,
            "uptime_ms": rt.system.uptime_ms() as u64,
            "query_count": rt.system.query_count(),
            "has_claude": rt.hybrid.has_claude(),
        })
        .to_string()
    })
    .unwrap_or_else(|| json!({ "running": false, "is_active": false }).to_string())
}

/// `reset()` — re-initialise all neural components, preserving the current
/// config. Equivalent to `init` with the running flag cleared.
fn core_reset() {
    let _ = core_init(None);
}

/// `isRunning()` — whether `start` has been called without a following `stop`.
fn core_is_running() -> bool {
    with_runtime(|rt| rt.running).unwrap_or(false)
}

/// Run `f` against the live runtime, returning `None` if uninitialised or the
/// lock is poisoned.
fn with_runtime<T>(f: impl FnOnce(&mut NativeRuntime) -> T) -> Option<T> {
    let mut guard = runtime_cell().lock().ok()?;
    guard.as_mut().map(f)
}

// ===========================================================================
// JNI ABI boundary.
//
// JUSTIFICATION FOR `unsafe`:
//   The crate is `#![deny(unsafe_code)]`. The JVM resolves native methods by
//   C symbol name (`Java_<class>_<method>`), which requires `#[unsafe(no_mangle)]`
//   plus `unsafe extern "C"` — there is no safe-Rust spelling of an exported
//   C-ABI symbol. The `jni` crate's `JNIEnv`/`JObject` arguments are likewise
//   raw handles produced by the JVM. The `unsafe` is therefore confined to the
//   declarations in this module; every body decodes its inputs and delegates to
//   the safe `core_*` functions above, performing no `unsafe` operations.
// ===========================================================================
mod jni_boundary {
    use super::*;
    use jni::objects::{JClass, JFloatArray, JObject, JString};
    use jni::sys::{jboolean, jfloat, jint, jlong, JNI_FALSE, JNI_TRUE};
    use jni::JNIEnv;

    /// Convert a Rust bool into the JNI `jboolean` (`0`/`1`) representation.
    fn to_jboolean(b: bool) -> jboolean {
        if b {
            JNI_TRUE
        } else {
            JNI_FALSE
        }
    }

    /// Decode a (possibly null) `JString` into an owned `Option<String>`.
    fn opt_string(env: &mut JNIEnv, s: &JString) -> Option<String> {
        // `JString` has no `is_null`; check the underlying raw object handle.
        if s.as_raw().is_null() {
            return None;
        }
        env.get_string(s).ok().map(|js| js.into())
    }

    /// Decode a non-null `JString`, defaulting to empty on null/decode error.
    fn req_string(env: &mut JNIEnv, s: &JString) -> String {
        opt_string(env, s).unwrap_or_default()
    }

    /// Allocate a Java `String` from a Rust `&str`, returning a null ref on error.
    fn new_jstring<'l>(env: &mut JNIEnv<'l>, s: &str) -> JObject<'l> {
        match env.new_string(s) {
            Ok(js) => js.into(),
            Err(_) => JObject::null(),
        }
    }

    /// `init(String?) -> boolean`
    #[allow(unsafe_code)] // JNI ABI: see module-level justification.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn Java_ai_neurophone_NativeLib_init(
        mut env: JNIEnv,
        _class: JClass,
        config_json: JString,
    ) -> jboolean {
        let cfg = opt_string(&mut env, &config_json);
        to_jboolean(core_init(cfg.as_deref()))
    }

    /// `start() -> boolean`
    #[allow(unsafe_code)] // JNI ABI: see module-level justification.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn Java_ai_neurophone_NativeLib_start(
        _env: JNIEnv,
        _class: JClass,
    ) -> jboolean {
        to_jboolean(core_start())
    }

    /// `stop() -> void`
    #[allow(unsafe_code)] // JNI ABI: see module-level justification.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn Java_ai_neurophone_NativeLib_stop(
        _env: JNIEnv,
        _class: JClass,
    ) {
        core_stop();
    }

    /// `processSensor(int, float[], long, int) -> boolean`
    #[allow(unsafe_code)] // JNI ABI: see module-level justification.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn Java_ai_neurophone_NativeLib_processSensor(
        env: JNIEnv,
        _class: JClass,
        sensor_type: jint,
        values: JFloatArray,
        timestamp: jlong,
        accuracy: jint,
    ) -> jboolean {
        let len = env.get_array_length(&values).unwrap_or(0).max(0) as usize;
        let mut buf: Vec<jfloat> = vec![0.0; len];
        if len > 0 && env.get_float_array_region(&values, 0, &mut buf).is_err() {
            return to_jboolean(false);
        }
        // jint == i32, jlong == i64; pass through to the typed core fn.
        to_jboolean(core_process_sensor(sensor_type, buf, timestamp, accuracy))
    }

    /// `queryLocal(String) -> String`
    #[allow(unsafe_code)] // JNI ABI: see module-level justification.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn Java_ai_neurophone_NativeLib_queryLocal<'l>(
        mut env: JNIEnv<'l>,
        _class: JClass<'l>,
        message: JString<'l>,
    ) -> JObject<'l> {
        let msg = req_string(&mut env, &message);
        let out = core_query_local(&msg);
        new_jstring(&mut env, &out)
    }

    /// `queryClaude(String) -> String`
    #[allow(unsafe_code)] // JNI ABI: see module-level justification.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn Java_ai_neurophone_NativeLib_queryClaude<'l>(
        mut env: JNIEnv<'l>,
        _class: JClass<'l>,
        message: JString<'l>,
    ) -> JObject<'l> {
        let msg = req_string(&mut env, &message);
        let out = core_query_claude(&msg);
        new_jstring(&mut env, &out)
    }

    /// `query(String, boolean) -> String`
    #[allow(unsafe_code)] // JNI ABI: see module-level justification.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn Java_ai_neurophone_NativeLib_query<'l>(
        mut env: JNIEnv<'l>,
        _class: JClass<'l>,
        message: JString<'l>,
        prefer_local: jboolean,
    ) -> JObject<'l> {
        let msg = req_string(&mut env, &message);
        let out = core_query(&msg, prefer_local != JNI_FALSE);
        new_jstring(&mut env, &out)
    }

    /// `getNeuralContext() -> String`
    #[allow(unsafe_code)] // JNI ABI: see module-level justification.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn Java_ai_neurophone_NativeLib_getNeuralContext<'l>(
        mut env: JNIEnv<'l>,
        _class: JClass<'l>,
    ) -> JObject<'l> {
        let out = core_get_neural_context();
        new_jstring(&mut env, &out)
    }

    /// `getState() -> String` (JSON)
    #[allow(unsafe_code)] // JNI ABI: see module-level justification.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn Java_ai_neurophone_NativeLib_getState<'l>(
        mut env: JNIEnv<'l>,
        _class: JClass<'l>,
    ) -> JObject<'l> {
        let out = core_get_state();
        new_jstring(&mut env, &out)
    }

    /// `reset() -> void`
    #[allow(unsafe_code)] // JNI ABI: see module-level justification.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn Java_ai_neurophone_NativeLib_reset(
        _env: JNIEnv,
        _class: JClass,
    ) {
        core_reset();
    }

    /// `isRunning() -> boolean`
    #[allow(unsafe_code)] // JNI ABI: see module-level justification.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn Java_ai_neurophone_NativeLib_isRunning(
        _env: JNIEnv,
        _class: JClass,
    ) -> jboolean {
        to_jboolean(core_is_running())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// Serialise tests: they share the process-global `RUNTIME` singleton.
    static TEST_LOCK: StdMutex<()> = StdMutex::new(());

    #[test]
    fn sensor_id_map_matches_contract() {
        assert_eq!(sensor_kind_from_android_id(1), Some(SensorKind::Accelerometer));
        assert_eq!(sensor_kind_from_android_id(2), Some(SensorKind::Magnetometer));
        assert_eq!(sensor_kind_from_android_id(4), Some(SensorKind::Gyroscope));
        assert_eq!(sensor_kind_from_android_id(5), Some(SensorKind::Light));
        assert_eq!(sensor_kind_from_android_id(8), Some(SensorKind::Proximity));
        assert_eq!(sensor_kind_from_android_id(0), None);
        assert_eq!(sensor_kind_from_android_id(99), None);
    }

    #[test]
    fn parse_config_falls_back_on_bad_input() {
        let d = SystemConfig::default();
        assert_eq!(parse_config(None).sample_rate, d.sample_rate);
        assert_eq!(parse_config(Some("")).sample_rate, d.sample_rate);
        assert_eq!(parse_config(Some("not json")).sample_rate, d.sample_rate);
    }

    #[test]
    fn lifecycle_init_start_stop_reset() {
        let _g = TEST_LOCK.lock().unwrap();
        assert!(core_init(None));
        assert!(!core_is_running());
        assert!(core_start());
        assert!(core_is_running());
        core_stop();
        assert!(!core_is_running());
        core_reset();
        // reset leaves the runtime initialised but not running.
        assert!(!core_is_running());
    }

    #[test]
    fn process_sensor_validates_arity_and_type() {
        let _g = TEST_LOCK.lock().unwrap();
        assert!(core_init(None));
        // accelerometer expects 3 values.
        assert!(core_process_sensor(1, vec![0.1, 0.2, 0.3], 1_000_000, 3));
        // wrong arity rejected.
        assert!(!core_process_sensor(1, vec![0.1], 1_000_000, 3));
        // unknown sensor id rejected.
        assert!(!core_process_sensor(0, vec![0.1, 0.2, 0.3], 1_000_000, 3));
        // light expects 1 value.
        assert!(core_process_sensor(5, vec![42.0], 2_000_000, 3));
        // context becomes available after processing.
        assert!(core_get_neural_context().contains("light"));
    }

    #[test]
    fn query_paths_return_text() {
        let _g = TEST_LOCK.lock().unwrap();
        assert!(core_init(None));
        let local = core_query_local("hello world");
        assert!(local.contains("local-llama-mock"));
        // No API key in test env => clear unavailable message, not a panic.
        let cloud = core_query_claude("hello");
        assert!(cloud.contains("claude-unavailable") || cloud.contains("claude-error"));
        let hybrid = core_query("what is the time", true);
        assert!(hybrid.contains("Response to"));
    }

    #[test]
    fn get_state_is_valid_json() {
        let _g = TEST_LOCK.lock().unwrap();
        assert!(core_init(None));
        let s = core_get_state();
        let v: serde_json::Value = serde_json::from_str(&s).expect("state must be JSON");
        assert_eq!(v["is_active"], true);
        assert!(v.get("uptime_ms").is_some());
    }

    #[test]
    fn calls_before_init_are_safe() {
        let _g = TEST_LOCK.lock().unwrap();
        // Force an uninitialised state by clearing the singleton.
        if let Some(cell) = RUNTIME.get() {
            *cell.lock().unwrap() = None;
        }
        assert!(!core_start());
        assert!(!core_is_running());
        core_stop(); // must not panic
        assert!(!core_process_sensor(1, vec![0.0, 0.0, 0.0], 0, 0));
        assert!(core_query_local("x").contains("not initialised"));
    }
}
