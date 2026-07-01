// SPDX-License-Identifier: MPL-2.0
//! Android `Sensor.TYPE_*` id → canonical neurophone-core sensor name.
//!
//! This lookup used to live in the Kotlin `NativeLib` wrapper; per the JNI
//! surface audit it now lives in the Rust boundary so the name contract is
//! owned by the core stack, not the JVM shim.

use jni::sys::jint;

/// The five sensors neurophone consumes, keyed by the Android framework's
/// `Sensor.TYPE_*` integer constants.
const ID_TO_NAME: &[(jint, &str)] = &[
    (1, "accelerometer"), // Sensor.TYPE_ACCELEROMETER
    (4, "gyroscope"),     // Sensor.TYPE_GYROSCOPE
    (2, "magnetometer"),  // Sensor.TYPE_MAGNETIC_FIELD
    (5, "light"),         // Sensor.TYPE_LIGHT
    (8, "proximity"),     // Sensor.TYPE_PROXIMITY
];

/// Map an Android sensor type id to a canonical name, or `None` if unknown.
pub fn name_from_id(id: jint) -> Option<&'static str> {
    ID_TO_NAME
        .iter()
        .find_map(|(k, v)| if *k == id { Some(*v) } else { None })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_ids_map() {
        assert_eq!(name_from_id(1), Some("accelerometer"));
        assert_eq!(name_from_id(4), Some("gyroscope"));
        assert_eq!(name_from_id(2), Some("magnetometer"));
        assert_eq!(name_from_id(5), Some("light"));
        assert_eq!(name_from_id(8), Some("proximity"));
    }

    #[test]
    fn unknown_id_is_none() {
        assert_eq!(name_from_id(99), None);
        assert_eq!(name_from_id(-1), None);
        assert_eq!(name_from_id(0), None);
    }
}
