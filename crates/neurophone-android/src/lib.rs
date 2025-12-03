//! NeuroPhone Android JNI Bindings
//!
//! Provides native interface for Android application to interact
//! with the neurosymbolic system.

use jni::JNIEnv;
use jni::objects::{JClass, JObject, JString, JValue, JFloatArray};
use jni::sys::{jboolean, jfloat, jfloatArray, jint, jlong, jstring, JNI_TRUE, JNI_FALSE};
use neurophone_core::{
    NeuroSymbolicSystem, SystemConfig, SystemState,
    SensorReading, SensorType, GenerationParams,
};
use sensors::SensorAccuracy;
use std::sync::{Arc, Mutex, Once};
use tracing::{info, warn, error};

// Global system instance
static mut SYSTEM: Option<Arc<Mutex<NeuroSymbolicSystem>>> = None;
static INIT: Once = Once::new();

// Tokio runtime for async operations
static mut RUNTIME: Option<tokio::runtime::Runtime> = None;

/// Initialize the system
fn get_system() -> Option<Arc<Mutex<NeuroSymbolicSystem>>> {
    unsafe { SYSTEM.clone() }
}

fn get_runtime() -> &'static tokio::runtime::Runtime {
    unsafe {
        RUNTIME.as_ref().expect("Runtime not initialized")
    }
}

/// Initialize the native library
#[no_mangle]
pub extern "system" fn Java_ai_neurophone_NativeLib_init(
    mut env: JNIEnv,
    _class: JClass,
    config_json: JString,
) -> jboolean {
    // Initialize logging
    #[cfg(target_os = "android")]
    {
        android_logger::init_once(
            android_logger::Config::default()
                .with_max_level(log::LevelFilter::Debug)
                .with_tag("neurophone")
        );
    }

    #[cfg(not(target_os = "android"))]
    {
        tracing_subscriber::fmt::init();
    }

    info!("Initializing NeuroPhone native library");

    // Parse config if provided
    let config: SystemConfig = if !config_json.is_null() {
        match env.get_string(&config_json) {
            Ok(s) => {
                let config_str: String = s.into();
                serde_json::from_str(&config_str).unwrap_or_default()
            }
            Err(_) => SystemConfig::default(),
        }
    } else {
        SystemConfig::default()
    };

    // Initialize runtime and system
    INIT.call_once(|| {
        unsafe {
            RUNTIME = Some(
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .expect("Failed to create runtime")
            );

            match NeuroSymbolicSystem::with_config(config) {
                Ok(sys) => {
                    SYSTEM = Some(Arc::new(Mutex::new(sys)));
                    info!("System initialized successfully");
                }
                Err(e) => {
                    error!("Failed to initialize system: {}", e);
                }
            }
        }
    });

    if get_system().is_some() {
        JNI_TRUE
    } else {
        JNI_FALSE
    }
}

/// Start the system
#[no_mangle]
pub extern "system" fn Java_ai_neurophone_NativeLib_start(
    _env: JNIEnv,
    _class: JClass,
) -> jboolean {
    let system = match get_system() {
        Some(s) => s,
        None => return JNI_FALSE,
    };

    let runtime = get_runtime();

    runtime.block_on(async {
        let mut sys = system.lock().unwrap();
        match sys.start().await {
            Ok(_) => {
                info!("System started");
                JNI_TRUE
            }
            Err(e) => {
                error!("Failed to start system: {}", e);
                JNI_FALSE
            }
        }
    })
}

/// Stop the system
#[no_mangle]
pub extern "system" fn Java_ai_neurophone_NativeLib_stop(
    _env: JNIEnv,
    _class: JClass,
) {
    if let Some(system) = get_system() {
        let runtime = get_runtime();
        runtime.block_on(async {
            let mut sys = system.lock().unwrap();
            sys.stop().await;
            info!("System stopped");
        });
    }
}

/// Process sensor data
#[no_mangle]
pub extern "system" fn Java_ai_neurophone_NativeLib_processSensor(
    mut env: JNIEnv,
    _class: JClass,
    sensor_type: jint,
    values: jfloatArray,
    timestamp: jlong,
    accuracy: jint,
) -> jboolean {
    let system = match get_system() {
        Some(s) => s,
        None => return JNI_FALSE,
    };

    // Convert sensor type
    let sensor_type = match sensor_type {
        1 => SensorType::Accelerometer,
        4 => SensorType::Gyroscope,
        2 => SensorType::Magnetometer,
        5 => SensorType::Light,
        8 => SensorType::Proximity,
        6 => SensorType::Barometer,
        _ => return JNI_FALSE,
    };

    // Convert raw jfloatArray to JFloatArray
    let values_arr = unsafe { JFloatArray::from_raw(values) };

    // Get float array values
    let len = match env.get_array_length(&values_arr) {
        Ok(l) => l as usize,
        Err(_) => return JNI_FALSE,
    };

    let mut rust_values = vec![0.0f32; len];
    if env.get_float_array_region(&values_arr, 0, &mut rust_values).is_err() {
        return JNI_FALSE;
    }

    // Convert accuracy
    let accuracy = match accuracy {
        3 => SensorAccuracy::High,
        2 => SensorAccuracy::Medium,
        1 => SensorAccuracy::Low,
        _ => SensorAccuracy::Unreliable,
    };

    // Create reading
    let reading = SensorReading {
        sensor_type,
        timestamp: chrono::DateTime::from_timestamp_nanos(timestamp),
        values: rust_values,
        accuracy,
    };

    // Send to system
    let runtime = get_runtime();
    runtime.block_on(async {
        let sys = system.lock().unwrap();
        match sys.send_sensor(reading).await {
            Ok(_) => JNI_TRUE,
            Err(e) => {
                warn!("Failed to send sensor data: {}", e);
                JNI_FALSE
            }
        }
    })
}

/// Query the system (local LLM)
#[no_mangle]
pub extern "system" fn Java_ai_neurophone_NativeLib_queryLocal<'a>(
    mut env: JNIEnv<'a>,
    _class: JClass,
    message: JString,
) -> jstring {
    let empty = env.new_string("").unwrap();

    let system = match get_system() {
        Some(s) => s,
        None => return empty.into_raw(),
    };

    let msg: String = match env.get_string(&message) {
        Ok(s) => s.into(),
        Err(_) => return empty.into_raw(),
    };

    let runtime = get_runtime();
    let result = runtime.block_on(async {
        let sys = system.lock().unwrap();
        sys.query_local(&msg, None).await
    });

    match result {
        Ok(response) => {
            match env.new_string(&response) {
                Ok(s) => s.into_raw(),
                Err(_) => empty.into_raw(),
            }
        }
        Err(e) => {
            warn!("Query error: {}", e);
            empty.into_raw()
        }
    }
}

/// Query Claude (cloud)
#[no_mangle]
pub extern "system" fn Java_ai_neurophone_NativeLib_queryClaude<'a>(
    mut env: JNIEnv<'a>,
    _class: JClass,
    message: JString,
) -> jstring {
    let empty = env.new_string("").unwrap();

    let system = match get_system() {
        Some(s) => s,
        None => return empty.into_raw(),
    };

    let msg: String = match env.get_string(&message) {
        Ok(s) => s.into(),
        Err(_) => return empty.into_raw(),
    };

    let runtime = get_runtime();
    let result = runtime.block_on(async {
        let sys = system.lock().unwrap();
        sys.query_claude(&msg).await
    });

    match result {
        Ok(response) => {
            match env.new_string(&response) {
                Ok(s) => s.into_raw(),
                Err(_) => empty.into_raw(),
            }
        }
        Err(e) => {
            warn!("Claude query error: {}", e);
            empty.into_raw()
        }
    }
}

/// Smart query (auto-selects local or cloud)
#[no_mangle]
pub extern "system" fn Java_ai_neurophone_NativeLib_query<'a>(
    mut env: JNIEnv<'a>,
    _class: JClass,
    message: JString,
    prefer_local: jboolean,
) -> jstring {
    let empty = env.new_string("").unwrap();

    let system = match get_system() {
        Some(s) => s,
        None => return empty.into_raw(),
    };

    let msg: String = match env.get_string(&message) {
        Ok(s) => s.into(),
        Err(_) => return empty.into_raw(),
    };

    let runtime = get_runtime();
    let result = runtime.block_on(async {
        let sys = system.lock().unwrap();
        sys.query(&msg, prefer_local == JNI_TRUE).await
    });

    match result {
        Ok(response) => {
            match env.new_string(&response) {
                Ok(s) => s.into_raw(),
                Err(_) => empty.into_raw(),
            }
        }
        Err(e) => {
            warn!("Query error: {}", e);
            empty.into_raw()
        }
    }
}

/// Get neural context
#[no_mangle]
pub extern "system" fn Java_ai_neurophone_NativeLib_getNeuralContext<'a>(
    mut env: JNIEnv<'a>,
    _class: JClass,
) -> jstring {
    let empty = env.new_string("").unwrap();

    let system = match get_system() {
        Some(s) => s,
        None => return empty.into_raw(),
    };

    let runtime = get_runtime();
    let context = runtime.block_on(async {
        let sys = system.lock().unwrap();
        sys.get_neural_context().await
    });

    match env.new_string(&context) {
        Ok(s) => s.into_raw(),
        Err(_) => empty.into_raw(),
    }
}

/// Get system state as JSON
#[no_mangle]
pub extern "system" fn Java_ai_neurophone_NativeLib_getState<'a>(
    mut env: JNIEnv<'a>,
    _class: JClass,
) -> jstring {
    let empty = env.new_string("{}").unwrap();

    let system = match get_system() {
        Some(s) => s,
        None => return empty.into_raw(),
    };

    let runtime = get_runtime();
    let state = runtime.block_on(async {
        let sys = system.lock().unwrap();
        sys.get_state().await
    });

    let json = serde_json::to_string(&state).unwrap_or_else(|_| "{}".to_string());

    match env.new_string(&json) {
        Ok(s) => s.into_raw(),
        Err(_) => empty.into_raw(),
    }
}

/// Reset the system
#[no_mangle]
pub extern "system" fn Java_ai_neurophone_NativeLib_reset(
    _env: JNIEnv,
    _class: JClass,
) {
    if let Some(system) = get_system() {
        let runtime = get_runtime();
        runtime.block_on(async {
            let mut sys = system.lock().unwrap();
            sys.reset().await;
            info!("System reset");
        });
    }
}

/// Check if system is running
#[no_mangle]
pub extern "system" fn Java_ai_neurophone_NativeLib_isRunning(
    _env: JNIEnv,
    _class: JClass,
) -> jboolean {
    match get_system() {
        Some(system) => {
            let runtime = get_runtime();
            let state = runtime.block_on(async {
                let sys = system.lock().unwrap();
                sys.get_state().await
            });
            if state.running { JNI_TRUE } else { JNI_FALSE }
        }
        None => JNI_FALSE,
    }
}

// Tests (run on host, not Android)
#[cfg(test)]
mod tests {
    #[test]
    fn test_compile() {
        // Just verify compilation works
        assert!(true);
    }
}
