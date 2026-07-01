// SPDX-License-Identifier: MPL-2.0
//! Error type bridging `neurophone-core` and `jni` failures across the boundary.

use thiserror::Error;

/// Unified error for the JNI bridge.
///
/// The `jni` `with_env`/`resolve` machinery only requires the closure error to
/// implement [`std::error::Error`]; this enum lets a single `?` propagate both
/// JNI failures and core failures, which `ThrowRuntimeExAndDefault` then turns
/// into a Java `RuntimeException` (never an unwind across FFI).
#[derive(Error, Debug)]
pub enum JniBridgeError {
    /// A failure raised by the `jni` crate itself.
    #[error(transparent)]
    Jni(#[from] jni::errors::Error),
    /// A failure raised by `neurophone-core`.
    #[error(transparent)]
    Core(#[from] neurophone_core::NeurophoneError),
    /// A bridge-layer precondition failure (e.g. "not initialised").
    #[error("{0}")]
    Msg(String),
}
