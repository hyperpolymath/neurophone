// SPDX-License-Identifier: MPL-2.0
//! NeuroPhone Android JNI bindings — class `ai.neurophone.NativeLib`.
//!
//! This is a pure ABI boundary: every `#[no_mangle]` export marshals JNI
//! arguments to/from Rust and delegates to [`state`], which holds the live
//! [`neurophone_core::NeuroSymbolicSystem`] and contains all bridge logic. Under
//! jni 0.22, native methods receive an [`EnvUnowned`] and call `with_env`, which
//! wraps the body in `catch_unwind` so a panic can never unwind across FFI; the
//! [`ThrowRuntimeExAndDefault`] policy converts any error into a Java
//! `RuntimeException`.
//!
//! There is no unsafe *code* here — every JVM interaction goes through jni's safe
//! `with_env`/`resolve` API. The crate cannot use `#![forbid(unsafe_code)]` only
//! because the JNI ABI export attribute is spelled `#[unsafe(no_mangle)]`; this is
//! the estate-sanctioned FFI boundary crate (Trust `[FFI]` policy), and the
//! exports are the sole reason the attribute appears.
//!
//! The JNI surface mirrors the contract in `docs/migrations/JNI-SURFACE-AUDIT.adoc`
//! (11 methods). It is gossamer-independent: gossamer changes *who* loads and
//! calls this cdylib, not this ABI.

mod error;
mod sensor_map;
mod state;

use error::JniBridgeError;
use jni::errors::ThrowRuntimeExAndDefault;
use jni::objects::{JClass, JFloatArray, JString};
use jni::refs::Reference;
use jni::sys::{jboolean, jint, jlong};
use jni::EnvUnowned;
use neurophone_core::QueryRoute;

// jni 0.22 models `jboolean` as a real Rust `bool`.
const JNI_TRUE: jboolean = true;

/// `init(configJson: String?): Boolean`
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_neurophone_NativeLib_init<'c>(
    mut unowned_env: EnvUnowned<'c>,
    _class: JClass<'c>,
    config_json: JString<'c>,
) -> jboolean {
    unowned_env
        .with_env(|_env| -> Result<jboolean, JniBridgeError> {
            let json = if config_json.is_null() {
                None
            } else {
                Some(config_json.to_string())
            };
            state::init(json.as_deref())?;
            Ok(JNI_TRUE)
        })
        .resolve::<ThrowRuntimeExAndDefault>()
}

/// `start(): Boolean`
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_neurophone_NativeLib_start<'c>(
    mut unowned_env: EnvUnowned<'c>,
    _class: JClass<'c>,
) -> jboolean {
    unowned_env
        .with_env(|_env| -> Result<jboolean, JniBridgeError> {
            state::start()?;
            Ok(JNI_TRUE)
        })
        .resolve::<ThrowRuntimeExAndDefault>()
}

/// `stop()`
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_neurophone_NativeLib_stop<'c>(
    mut unowned_env: EnvUnowned<'c>,
    _class: JClass<'c>,
) {
    unowned_env
        .with_env(|_env| -> Result<(), JniBridgeError> {
            state::stop();
            Ok(())
        })
        .resolve::<ThrowRuntimeExAndDefault>()
}

/// `isRunning(): Boolean`
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_neurophone_NativeLib_isRunning<'c>(
    mut unowned_env: EnvUnowned<'c>,
    _class: JClass<'c>,
) -> jboolean {
    unowned_env
        .with_env(|_env| -> Result<jboolean, JniBridgeError> { Ok(state::is_running()) })
        .resolve::<ThrowRuntimeExAndDefault>()
}

/// `processSensor(sensorType: Int, values: FloatArray, timestamp: Long, accuracy: Int): Boolean`
///
/// `accuracy` is accepted for ABI stability and intentionally ignored (the core
/// does not consume it).
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_neurophone_NativeLib_processSensor<'c>(
    mut unowned_env: EnvUnowned<'c>,
    _class: JClass<'c>,
    sensor_type: jint,
    values: JFloatArray<'c>,
    timestamp: jlong,
    _accuracy: jint,
) -> jboolean {
    unowned_env
        .with_env(|env| -> Result<jboolean, JniBridgeError> {
            let name = sensor_map::name_from_id(sensor_type)
                .ok_or_else(|| JniBridgeError::Msg(format!("unknown sensor id {sensor_type}")))?;
            let len = values.len(env)?;
            let mut buf = vec![0f32; len];
            values.get_region(env, 0, &mut buf)?;
            state::process_sensor(name, buf, timestamp.max(0) as u64)?;
            Ok(JNI_TRUE)
        })
        .resolve::<ThrowRuntimeExAndDefault>()
}

/// `query(message: String, preferLocal: Boolean): String`
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_neurophone_NativeLib_query<'c>(
    mut unowned_env: EnvUnowned<'c>,
    _class: JClass<'c>,
    message: JString<'c>,
    prefer_local: jboolean,
) -> JString<'c> {
    unowned_env
        .with_env(|env| -> Result<_, JniBridgeError> {
            let msg = message.to_string();
            let route = if prefer_local {
                QueryRoute::Auto
            } else {
                QueryRoute::ForceCloud
            };
            let resp = state::query(&msg, route)?;
            Ok(JString::from_str(env, resp)?)
        })
        .resolve::<ThrowRuntimeExAndDefault>()
}

/// `queryLocal(message: String): String` — forces the local model.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_neurophone_NativeLib_queryLocal<'c>(
    mut unowned_env: EnvUnowned<'c>,
    _class: JClass<'c>,
    message: JString<'c>,
) -> JString<'c> {
    unowned_env
        .with_env(|env| -> Result<_, JniBridgeError> {
            let resp = state::query(&message.to_string(), QueryRoute::ForceLocal)?;
            Ok(JString::from_str(env, resp)?)
        })
        .resolve::<ThrowRuntimeExAndDefault>()
}

/// `queryClaude(message: String): String` — forces the cloud model.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_neurophone_NativeLib_queryClaude<'c>(
    mut unowned_env: EnvUnowned<'c>,
    _class: JClass<'c>,
    message: JString<'c>,
) -> JString<'c> {
    unowned_env
        .with_env(|env| -> Result<_, JniBridgeError> {
            let resp = state::query(&message.to_string(), QueryRoute::ForceCloud)?;
            Ok(JString::from_str(env, resp)?)
        })
        .resolve::<ThrowRuntimeExAndDefault>()
}

/// `getNeuralContext(): String`
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_neurophone_NativeLib_getNeuralContext<'c>(
    mut unowned_env: EnvUnowned<'c>,
    _class: JClass<'c>,
) -> JString<'c> {
    unowned_env
        .with_env(|env| -> Result<_, JniBridgeError> {
            let ctx = state::neural_context()?;
            Ok(JString::from_str(env, ctx)?)
        })
        .resolve::<ThrowRuntimeExAndDefault>()
}

/// `getState(): String` — the `SystemState` as JSON.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_neurophone_NativeLib_getState<'c>(
    mut unowned_env: EnvUnowned<'c>,
    _class: JClass<'c>,
) -> JString<'c> {
    unowned_env
        .with_env(|env| -> Result<_, JniBridgeError> {
            let json = state::state_json()?;
            Ok(JString::from_str(env, json)?)
        })
        .resolve::<ThrowRuntimeExAndDefault>()
}

/// `reset()`
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_neurophone_NativeLib_reset<'c>(
    mut unowned_env: EnvUnowned<'c>,
    _class: JClass<'c>,
) {
    unowned_env
        .with_env(|_env| -> Result<(), JniBridgeError> {
            state::reset()?;
            Ok(())
        })
        .resolve::<ThrowRuntimeExAndDefault>()
}
