// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell
//! Presence score with exponential decay.
//!
//! A resolved contact's beacon arrives intermittently (it rotates every epoch,
//! and BLE scans are duty-cycled). The LSM wants a smooth single-channel signal,
//! not a sparse train of sightings, so we hold a `0.0..=1.0` presence score that
//! is *reinforced* by each sighting (scaled by signal strength) and *decays*
//! exponentially between them. A contact that walks away fades toward 0 with a
//! configurable half-life instead of vanishing abruptly.

use sensors::SensorError;

/// RSSI (dBm) mapped to a `0.0..=1.0` instantaneous proximity target. Typical
/// BLE RSSI spans roughly -100 dBm (far / weak) to -30 dBm (very close).
const RSSI_FLOOR_DBM: f32 = -100.0;
const RSSI_CEIL_DBM: f32 = -30.0;

/// Exponentially-decaying presence score for a single contact channel.
#[derive(Debug, Clone)]
pub struct PresenceDecay {
    /// Time for the score to halve with no sightings, in seconds.
    half_life_s: f32,
    score: f32,
    last_update_ms: Option<u64>,
}

impl PresenceDecay {
    /// Create a decayer with the given half-life (seconds). Errors on a
    /// non-positive half-life.
    pub fn new(half_life_s: f32) -> Result<Self, SensorError> {
        // Reject non-positive and NaN half-lives (a NaN would poison the decay).
        if half_life_s <= 0.0 || half_life_s.is_nan() {
            return Err(SensorError::InvalidConfig(
                "half_life_s must be positive".into(),
            ));
        }
        Ok(Self {
            half_life_s,
            score: 0.0,
            last_update_ms: None,
        })
    }

    /// Map an RSSI reading to a `0.0..=1.0` proximity target.
    pub fn rssi_to_target(rssi_dbm: i16) -> f32 {
        let r = rssi_dbm as f32;
        ((r - RSSI_FLOOR_DBM) / (RSSI_CEIL_DBM - RSSI_FLOOR_DBM)).clamp(0.0, 1.0)
    }

    /// Decay the held score forward to `now_ms` (idempotent for repeated calls
    /// at the same instant). Returns the decayed score.
    pub fn score_at(&mut self, now_ms: u64) -> f32 {
        if let Some(last) = self.last_update_ms {
            let dt_s = now_ms.saturating_sub(last) as f32 / 1000.0;
            if dt_s > 0.0 {
                // score *= 0.5 ^ (dt / half_life)
                let factor = 0.5f32.powf(dt_s / self.half_life_s);
                self.score *= factor;
                self.last_update_ms = Some(now_ms);
            }
        } else {
            self.last_update_ms = Some(now_ms);
        }
        self.score
    }

    /// Register a sighting at `now_ms` with the given RSSI, reinforcing the
    /// score toward the RSSI-derived target. Returns the updated score.
    pub fn observe(&mut self, rssi_dbm: i16, now_ms: u64) -> f32 {
        // Decay to now first, then take the stronger of the decayed score and
        // the fresh proximity target — a closer/stronger sighting can only raise
        // presence, never lower it below what we already believe.
        let decayed = self.score_at(now_ms);
        let target = Self::rssi_to_target(rssi_dbm);
        self.score = decayed.max(target);
        self.last_update_ms = Some(now_ms);
        self.score
    }

    /// Current score without advancing time.
    pub fn score(&self) -> f32 {
        self.score
    }

    /// Reset to the unseen state.
    pub fn reset(&mut self) {
        self.score = 0.0;
        self.last_update_ms = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_nonpositive_half_life() {
        assert!(PresenceDecay::new(0.0).is_err());
        assert!(PresenceDecay::new(-1.0).is_err());
        assert!(PresenceDecay::new(30.0).is_ok());
    }

    #[test]
    fn rssi_mapping_clamps() {
        assert_eq!(PresenceDecay::rssi_to_target(-30), 1.0);
        assert_eq!(PresenceDecay::rssi_to_target(-10), 1.0); // above ceiling clamps
        assert_eq!(PresenceDecay::rssi_to_target(-100), 0.0);
        assert_eq!(PresenceDecay::rssi_to_target(-120), 0.0); // below floor clamps
        let mid = PresenceDecay::rssi_to_target(-65);
        assert!((mid - 0.5).abs() < 0.01, "midpoint ~0.5, got {mid}");
    }

    #[test]
    fn observe_then_halves_after_one_half_life() {
        let mut d = PresenceDecay::new(30.0).unwrap();
        d.observe(-30, 0); // target 1.0
        assert!((d.score() - 1.0).abs() < 1e-6);
        let s = d.score_at(30_000); // +30s = one half-life
        assert!((s - 0.5).abs() < 1e-3, "after one half-life ~0.5, got {s}");
        let s2 = d.score_at(60_000); // +another half-life
        assert!(
            (s2 - 0.25).abs() < 1e-3,
            "after two half-lives ~0.25, got {s2}"
        );
    }

    #[test]
    fn unseen_contact_decays_toward_zero() {
        let mut d = PresenceDecay::new(10.0).unwrap();
        d.observe(-40, 0);
        let s = d.score_at(200_000); // 20 half-lives later
        assert!(s < 1e-3, "should fade to ~0, got {s}");
    }

    #[test]
    fn resighting_reinforces() {
        let mut d = PresenceDecay::new(30.0).unwrap();
        d.observe(-50, 0);
        let faded = d.score_at(60_000); // decayed
        let renewed = d.observe(-30, 60_000); // strong re-sighting
        assert!(renewed > faded, "re-sighting should raise the score");
        assert!((renewed - 1.0).abs() < 1e-6);
    }
}
