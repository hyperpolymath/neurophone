// SPDX-License-Identifier: MPL-2.0
//! Global runtime-state holder for the JNI surface, plus the pure bridge
//! operations the JNI exports marshal into.
//!
//! All logic here is JVM-free and host-testable: the JNI layer in `lib.rs` only
//! does string/array marshalling and then calls these functions. The held
//! `Option<RuntimeState>` is the boundary's affine resource — `init` acquires it
//! (`None -> Some`) and `reset` consumes the running system via the core
//! `shutdown(self)` typestate (`Active -> Down`, dropped exactly once) before
//! reinstating a fresh one.

use crate::error::JniBridgeError;
use neurophone_core::{
    phase, NeuroSymbolicSystem, NeurophoneError, QueryRoute, SensorEvent, SystemConfig,
};
use std::sync::{Mutex, OnceLock};

/// The live system plus the metadata needed for `start`/`stop`/`reset`.
pub struct RuntimeState {
    system: NeuroSymbolicSystem<phase::Active>,
    config: SystemConfig,
    running: bool,
}

static HOLDER: OnceLock<Mutex<Option<RuntimeState>>> = OnceLock::new();

fn holder() -> &'static Mutex<Option<RuntimeState>> {
    HOLDER.get_or_init(|| Mutex::new(None))
}

/// Acquire the lock, recovering (rather than panicking) if a previous holder
/// panicked while holding it.
fn guard() -> std::sync::MutexGuard<'static, Option<RuntimeState>> {
    holder()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn parse_config(config_json: Option<&str>) -> Result<SystemConfig, JniBridgeError> {
    match config_json {
        Some(s) if !s.trim().is_empty() => serde_json::from_str::<SystemConfig>(s)
            .map_err(|e| JniBridgeError::Msg(format!("invalid config JSON: {e}"))),
        _ => Ok(SystemConfig::default()),
    }
}

/// `init(configJson)` — construct + initialise the system (acquire). Idempotent:
/// re-initialising replaces any prior instance, consuming it via `shutdown`.
pub fn init(config_json: Option<&str>) -> Result<(), JniBridgeError> {
    let config = parse_config(config_json)?;
    let system = NeuroSymbolicSystem::new(config.clone())?.initialize()?;
    let mut g = guard();
    if let Some(prev) = g.take() {
        // Consume the previous Active system exactly once (Active -> Down).
        let _down = prev.system.shutdown();
    }
    *g = Some(RuntimeState {
        system,
        config,
        running: false,
    });
    Ok(())
}

/// `start()` — mark the system running. Errors if not initialised.
pub fn start() -> Result<(), JniBridgeError> {
    let mut g = guard();
    let st = g
        .as_mut()
        .ok_or_else(|| JniBridgeError::Msg("start() before init()".into()))?;
    st.running = true;
    Ok(())
}

/// `stop()` — mark the system not running. No-op if not initialised.
pub fn stop() {
    if let Some(st) = guard().as_mut() {
        st.running = false;
    }
}

/// `isRunning()` — true only if initialised and started.
pub fn is_running() -> bool {
    guard().as_ref().map(|st| st.running).unwrap_or(false)
}

fn require_running(g: &mut Option<RuntimeState>) -> Result<&mut RuntimeState, JniBridgeError> {
    let st = g
        .as_mut()
        .ok_or_else(|| JniBridgeError::Msg("operation before init()".into()))?;
    if !st.running {
        return Err(JniBridgeError::Msg("operation before start()".into()));
    }
    Ok(st)
}

/// `processSensor(...)` — feed one sensor sample through the system.
pub fn process_sensor(
    sensor_name: &str,
    values: Vec<f32>,
    timestamp_ms: u64,
) -> Result<(), JniBridgeError> {
    let mut g = guard();
    let st = require_running(&mut g)?;
    let event = SensorEvent {
        sensor_type: sensor_name.to_string(),
        timestamp_ms,
        values,
    };
    st.system.process_sensor_event(&event)?;
    Ok(())
}

/// `query(...)` / `queryLocal` / `queryClaude` — run inference, return the text.
pub fn query(message: &str, route: QueryRoute) -> Result<String, JniBridgeError> {
    let mut g = guard();
    let st = require_running(&mut g)?;
    let result = st.system.query_routed(message, route)?;
    Ok(result.response)
}

/// `getNeuralContext()` — the LLM-context summary string.
pub fn neural_context() -> Result<String, JniBridgeError> {
    let g = guard();
    let st = g
        .as_ref()
        .ok_or_else(|| JniBridgeError::Msg("getNeuralContext() before init()".into()))?;
    Ok(st.system.get_neural_context())
}

/// `getState()` — the `SystemState` as JSON.
pub fn state_json() -> Result<String, JniBridgeError> {
    let g = guard();
    let st = g
        .as_ref()
        .ok_or_else(|| JniBridgeError::Msg("getState() before init()".into()))?;
    serde_json::to_string(&st.system.get_state())
        .map_err(|e| JniBridgeError::Msg(format!("state serialisation failed: {e}")))
}

/// `reset()` — consume the running system (`Active -> Down`, dropped once) and
/// reinstate a fresh one from the saved config, preserving the running flag.
pub fn reset() -> Result<(), JniBridgeError> {
    let mut g = guard();
    let prev = g
        .take()
        .ok_or_else(|| JniBridgeError::Msg("reset() before init()".into()))?;
    let running = prev.running;
    let config = prev.config.clone();
    let _down = prev.system.shutdown(); // consume-once: Active -> Down
    let system = rebuild(&config)?;
    *g = Some(RuntimeState {
        system,
        config,
        running,
    });
    Ok(())
}

fn rebuild(config: &SystemConfig) -> Result<NeuroSymbolicSystem<phase::Active>, NeurophoneError> {
    NeuroSymbolicSystem::new(config.clone())?.initialize()
}

/// Test-only reset of the global holder so unit tests don't share state.
#[cfg(test)]
pub(crate) fn clear_for_test() {
    *guard() = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    // The holder is process-global; serialise the tests that touch it.
    static TEST_LOCK: StdMutex<()> = StdMutex::new(());

    fn with_clean<T>(f: impl FnOnce() -> T) -> T {
        let _l = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        clear_for_test();
        let out = f();
        clear_for_test();
        out
    }

    #[test]
    fn full_lifecycle() {
        with_clean(|| {
            assert!(!is_running());
            // ops before init are rejected
            assert!(query("hi", QueryRoute::Auto).is_err());

            init(None).expect("init");
            assert!(!is_running());
            // ops before start are rejected
            assert!(process_sensor("accelerometer", vec![1.0], 1).is_err());

            start().expect("start");
            assert!(is_running());

            process_sensor("accelerometer", vec![1.0, 2.0, 3.0], 100).expect("process");
            let resp = query("what is happening", QueryRoute::ForceLocal).expect("query");
            assert!(!resp.is_empty());

            let ctx = neural_context().expect("ctx");
            assert!(ctx.contains("[NEURAL_STATE]"));

            let json = state_json().expect("state json");
            assert!(json.contains("is_active"));

            stop();
            assert!(!is_running());
        });
    }

    #[test]
    fn reset_preserves_running_and_config() {
        with_clean(|| {
            let cfg = r#"{"sample_rate":25.0,"window_size_ms":80,"local_threshold":0.4,"max_response_time_ms":500}"#;
            init(Some(cfg)).expect("init");
            start().expect("start");
            assert!(is_running());

            reset().expect("reset");
            // running flag preserved through the Active->Down->fresh cycle
            assert!(is_running());
            // fresh system is usable
            query("post reset", QueryRoute::Auto).expect("query after reset");
        });
    }

    #[test]
    fn bad_config_json_is_rejected() {
        with_clean(|| {
            assert!(init(Some("{not valid json")).is_err());
        });
    }

    #[test]
    fn every_boundary_op_before_init_is_rejected() {
        with_clean(|| {
            assert!(!is_running());
            // All read/act operations must refuse to run before the holder is
            // acquired — the affine resource is `None`, so there is nothing to
            // borrow. (This covers the guard error paths that the happy-path
            // lifecycle test does not exercise for every entrypoint.)
            assert!(start().is_err(), "start before init");
            assert!(
                process_sensor("accelerometer", vec![1.0], 1).is_err(),
                "process_sensor before init"
            );
            assert!(query("hi", QueryRoute::Auto).is_err(), "query before init");
            assert!(neural_context().is_err(), "neural_context before init");
            assert!(state_json().is_err(), "state_json before init");
            assert!(reset().is_err(), "reset before init");
            // stop() is defined as a safe no-op before init — it must neither
            // error nor panic, and must leave the system not-running.
            stop();
            assert!(!is_running());
        });
    }

    #[test]
    fn empty_or_whitespace_config_falls_back_to_default() {
        with_clean(|| {
            // Empty and whitespace-only config JSON are treated as "no config",
            // not as a parse error — parse_config trims before deciding.
            init(Some("")).expect("empty config -> default");
            start().expect("start");
            assert!(is_running());
        });
        with_clean(|| {
            init(Some("   ")).expect("whitespace config -> default");
            assert!(!is_running());
        });
    }

    #[test]
    fn reset_preserves_not_running_and_stays_usable() {
        with_clean(|| {
            init(None).expect("init");
            // Never started: the running flag is false and reset must preserve it
            // (complements reset_preserves_running_and_config, which covers true).
            assert!(!is_running());
            reset().expect("reset");
            assert!(!is_running());
            // The reinstated system is fresh and usable once started.
            start().expect("start");
            query("after reset", QueryRoute::ForceLocal).expect("query after reset");
        });
    }
}
