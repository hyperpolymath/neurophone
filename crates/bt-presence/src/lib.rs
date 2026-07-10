// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell
//! Bluetooth Low-Energy *nearby-presence* sensor — read-only consumer of
//! burble's frozen presence beacon (wire v1, burble ADR-0015).
//!
//! # What this is
//!
//! neurophone treats a co-located burble contact's rotating BLE beacon as one
//! more **presence sensor**: a single-channel signal (a decayed presence score)
//! that feeds the existing `sensor → LSM → ESN → bridge → LLM` pipeline exactly
//! like light or proximity. It is *sensor-class only* — no microphone, no
//! outbound advertising, read-only on the wire (see `docs/BT-PRESENCE-PLAN.adoc`).
//!
//! # What this is not
//!
//! This crate does not scan the radio and does not emit anything. Acquiring the
//! raw 24-byte advertisement bytes is the platform's job (the Android/gossamer
//! surface, per #83); this crate is the pure, host-testable **decode + contact
//! resolution + presence decay** core that turns those bytes into a
//! [`BtPresenceReading`]. The upstream *emitter* is burble Phase 1, which does
//! not exist yet — so at runtime there is nothing to receive until burble ships
//! its Android client. The wire format it will emit is, however, frozen (v1),
//! which is what makes this consumer buildable and testable now.
//!
//! # Protocol provenance
//!
//! The wire constants are code-generated at build time from the byte-exact
//! vendored copy of burble's frozen spec (`vendor/nearby-presence.a2ml`) — see
//! [`wire`] and `vendor/PROVENANCE.adoc`. The decoder is proven wire-compatible
//! against burble's frozen conformance vectors in `tests/vectors.rs`.

#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

/// Frozen BLE presence wire constants, generated from `vendor/nearby-presence.a2ml`
/// by `build.rs` (burble ADR-0015, wire v1). Regenerated whenever the vendored
/// spec changes; the build fails loudly if that spec is not a frozen v1.
pub mod wire {
    include!(concat!(env!("OUT_DIR"), "/wire.rs"));
}

pub mod decay;
pub mod decode;

pub use decay::PresenceDecay;
pub use decode::{beacon_id, epoch_for, Contact, DecodeError, PresenceFrame};

use sensors::{SensorError, SensorKind, SensorReading};

/// A resolved presence observation, ready to enter the sensor pipeline.
///
/// `presence_score` is the decayed 0.0..=1.0 confidence that a *known* contact
/// is nearby right now — high just after a beacon from a held contact secret is
/// seen, decaying toward 0 as epochs pass without a re-sighting.
#[derive(Debug, Clone, PartialEq)]
pub struct BtPresenceReading {
    /// The resolved contact id, if the beacon matched a held secret. `None`
    /// means "a valid, fresh beacon was seen but it belongs to no known
    /// contact" — presence of a stranger, which the score still reflects as
    /// unresolved co-location if the caller chooses to feed it.
    pub contact_id: Option<String>,
    /// Decayed presence confidence in `0.0..=1.0`.
    pub presence_score: f32,
    /// Sensor timestamp (ms) for pipeline alignment.
    pub timestamp_ms: u64,
}

impl BtPresenceReading {
    /// Convert to a standard single-channel [`SensorReading`] so the presence
    /// score flows through the existing `SensorPipeline` into the LSM.
    pub fn to_sensor_reading(&self) -> Result<SensorReading, SensorError> {
        SensorReading::new(
            SensorKind::Presence,
            self.timestamp_ms,
            vec![self.presence_score],
        )
    }
}
