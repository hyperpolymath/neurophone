// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell
//! Byte-exact decoder for burble's frozen BLE presence beacon (wire v1).
//!
//! Frame (24 bytes), all offsets/sizes from the generated [`crate::wire`]:
//!
//! ```text
//! magic(1)=0x42 │ ver_type(1)=0x12 │ epoch(4, u32 BE) │ beacon_id(18)
//! ```
//!
//! `epoch = unix_seconds / 900`. The contact-resolvable beacon id is
//! `HMAC-SHA256(presence_secret, "BRBL-PRES-v1" ‖ magic ‖ ver_type ‖ epoch(u32 BE))[0..18]`
//! (burble `Burble.Presence.BleSpa.beacon_id/2`). Resolution is: reject frames
//! whose carried epoch is more than `ACCEPT_EPOCH_WINDOW` from now, then
//! constant-time-compare the beacon against `beacon_id(secret, carried_epoch)`
//! for each held contact.

use crate::wire;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use subtle::ConstantTimeEq;
use thiserror::Error;

type HmacSha256 = Hmac<Sha256>;

/// Structural decode failures (mirrors burble's wire checks). "Valid frame but
/// unknown/stale contact" is *not* an error — it is `resolve` returning `None`.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DecodeError {
    #[error("wrong length: expected {expected}, got {got}")]
    BadLength { expected: usize, got: usize },
    #[error("bad magic byte")]
    BadMagic,
    #[error("unsupported wire version")]
    BadVersion,
    #[error("not a presence frame")]
    BadFrameType,
}

/// A held contact and the presence secret shared with them (burble ADR-0010;
/// secret distribution is out of scope here — it arrives via burble's handshake).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contact {
    pub id: String,
    /// `presence_secret` — the HMAC key. 32 bytes in practice; any length valid.
    pub secret: Vec<u8>,
}

impl Contact {
    pub fn new(id: impl Into<String>, secret: impl Into<Vec<u8>>) -> Self {
        Self {
            id: id.into(),
            secret: secret.into(),
        }
    }
}

/// The 15-minute epoch index for a unix-seconds timestamp (`unix / 900`).
#[inline]
pub fn epoch_for(unix_s: u64) -> u64 {
    unix_s / wire::EPOCH_SECONDS
}

/// The 18-byte contact-resolvable beacon id for `(secret, epoch)`.
///
/// `HMAC-SHA256(secret, "BRBL-PRES-v1" ‖ MAGIC ‖ VER_TYPE ‖ epoch(u32 BE))[0..18]`.
pub fn beacon_id(secret: &[u8], epoch: u64) -> [u8; wire::BEACON_ID_BYTES] {
    // HMAC accepts a key of any length, so this never actually errors; the
    // unreachable arm degrades to an all-zero id (which cannot match a real
    // beacon) rather than unwrapping/panicking.
    let mut mac = match HmacSha256::new_from_slice(secret) {
        Ok(m) => m,
        Err(_) => return [0u8; wire::BEACON_ID_BYTES],
    };
    mac.update(wire::LABEL_PRES);
    mac.update(&[wire::MAGIC, wire::VER_TYPE]);
    mac.update(&(epoch as u32).to_be_bytes());
    let full = mac.finalize().into_bytes();
    let mut out = [0u8; wire::BEACON_ID_BYTES];
    out.copy_from_slice(&full[..wire::BEACON_ID_BYTES]);
    out
}

/// A parsed presence frame: the carried epoch and the raw beacon id bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresenceFrame {
    /// Epoch carried in the clear (leaks nothing; makes vectors self-contained).
    pub epoch: u64,
    /// The 18-byte beacon id to resolve against held contact secrets.
    pub beacon_id: [u8; wire::BEACON_ID_BYTES],
}

impl PresenceFrame {
    /// Parse and structurally validate a raw advertisement payload.
    pub fn parse(bytes: &[u8]) -> Result<Self, DecodeError> {
        if bytes.len() != wire::FRAME_BYTES {
            return Err(DecodeError::BadLength {
                expected: wire::FRAME_BYTES,
                got: bytes.len(),
            });
        }
        if bytes[wire::MAGIC_OFFSET] != wire::MAGIC {
            return Err(DecodeError::BadMagic);
        }
        let vt = bytes[wire::VER_TYPE_OFFSET];
        if vt >> 4 != wire::WIRE_VERSION {
            return Err(DecodeError::BadVersion);
        }
        if vt != wire::VER_TYPE {
            // High nibble matched the version but the low nibble is not the
            // presence frame type (e.g. this is a knock, 0x11).
            return Err(DecodeError::BadFrameType);
        }
        let mut epoch_be = [0u8; wire::EPOCH_BYTES];
        epoch_be
            .copy_from_slice(&bytes[wire::EPOCH_OFFSET..wire::EPOCH_OFFSET + wire::EPOCH_BYTES]);
        let epoch = u32::from_be_bytes(epoch_be) as u64;

        let mut beacon = [0u8; wire::BEACON_ID_BYTES];
        beacon.copy_from_slice(
            &bytes[wire::BEACON_ID_OFFSET..wire::BEACON_ID_OFFSET + wire::BEACON_ID_BYTES],
        );
        Ok(Self {
            epoch,
            beacon_id: beacon,
        })
    }

    /// Is the carried epoch within `ACCEPT_EPOCH_WINDOW` of `now`? Bounds replay
    /// of stale beacons while tolerating modest clock skew.
    pub fn is_fresh(&self, now_s: u64) -> bool {
        epoch_for(now_s).abs_diff(self.epoch) <= wire::ACCEPT_EPOCH_WINDOW
    }

    /// Resolve this frame against held `contacts`, returning the first whose
    /// secret reproduces the beacon id (constant-time), or `None` if the frame
    /// is stale or belongs to no known contact.
    pub fn resolve<'a>(&self, contacts: &'a [Contact], now_s: u64) -> Option<&'a Contact> {
        if !self.is_fresh(now_s) {
            return None;
        }
        contacts.iter().find(|c| {
            let expected = beacon_id(&c.secret, self.epoch);
            expected[..].ct_eq(&self.beacon_id[..]).into()
        })
    }
}

/// Parse `bytes` and resolve in one step. `Ok(Some(id))` = a fresh beacon from a
/// known contact; `Ok(None)` = a structurally-valid beacon that is stale or from
/// a stranger; `Err(_)` = not a well-formed v1 presence frame.
pub fn decode_and_resolve(
    bytes: &[u8],
    contacts: &[Contact],
    now_s: u64,
) -> Result<Option<String>, DecodeError> {
    let frame = PresenceFrame::parse(bytes)?;
    Ok(frame.resolve(contacts, now_s).map(|c| c.id.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Round-trip: an encoded beacon resolves back to its contact within window.
    fn encode(secret: &[u8], epoch: u64) -> Vec<u8> {
        let mut v = Vec::with_capacity(wire::FRAME_BYTES);
        v.push(wire::MAGIC);
        v.push(wire::VER_TYPE);
        v.extend_from_slice(&(epoch as u32).to_be_bytes());
        v.extend_from_slice(&beacon_id(secret, epoch));
        v
    }

    #[test]
    fn epoch_is_floor_div_900() {
        assert_eq!(epoch_for(1767225600), 1963584);
        assert_eq!(epoch_for(1767225600 + 899), 1963584);
        assert_eq!(epoch_for(1767225600 + 900), 1963585);
    }

    #[test]
    fn roundtrip_resolves_owner() {
        let secret = vec![7u8; 32];
        let epoch = epoch_for(1767225600);
        let bytes = encode(&secret, epoch);
        let contacts = vec![Contact::new("me", secret.clone())];
        let got = decode_and_resolve(&bytes, &contacts, 1767225600).unwrap();
        assert_eq!(got.as_deref(), Some("me"));
    }

    #[test]
    fn wrong_secret_does_not_resolve() {
        let epoch = epoch_for(1767225600);
        let bytes = encode(&[7u8; 32], epoch);
        let contacts = vec![Contact::new("other", vec![8u8; 32])];
        assert_eq!(
            decode_and_resolve(&bytes, &contacts, 1767225600).unwrap(),
            None
        );
    }

    #[test]
    fn stale_epoch_beyond_window_rejected() {
        let secret = vec![7u8; 32];
        let epoch = epoch_for(1767225600);
        let bytes = encode(&secret, epoch);
        let contacts = vec![Contact::new("me", secret)];
        // now is +2 epochs (1800s) later — beyond ACCEPT_EPOCH_WINDOW (1).
        let now = 1767225600 + 2 * wire::EPOCH_SECONDS;
        assert_eq!(decode_and_resolve(&bytes, &contacts, now).unwrap(), None);
    }

    #[test]
    fn within_window_still_resolves() {
        let secret = vec![7u8; 32];
        let epoch = epoch_for(1767225600);
        let bytes = encode(&secret, epoch);
        let contacts = vec![Contact::new("me", secret)];
        // now is +1 epoch later — within ACCEPT_EPOCH_WINDOW.
        let now = 1767225600 + wire::EPOCH_SECONDS;
        assert_eq!(
            decode_and_resolve(&bytes, &contacts, now)
                .unwrap()
                .as_deref(),
            Some("me")
        );
    }

    #[test]
    fn structural_errors() {
        let secret = vec![7u8; 32];
        let mut bytes = encode(&secret, epoch_for(1767225600));
        assert_eq!(
            PresenceFrame::parse(&bytes[..23]).unwrap_err(),
            DecodeError::BadLength {
                expected: 24,
                got: 23
            }
        );
        let mut bad_magic = bytes.clone();
        bad_magic[0] = 0x43;
        assert_eq!(
            PresenceFrame::parse(&bad_magic).unwrap_err(),
            DecodeError::BadMagic
        );
        let mut bad_ver = bytes.clone();
        bad_ver[1] = 0x22; // version nibble 2
        assert_eq!(
            PresenceFrame::parse(&bad_ver).unwrap_err(),
            DecodeError::BadVersion
        );
        let mut knock = bytes.clone();
        knock[1] = 0x11; // v1 but knock frame-type
        assert_eq!(
            PresenceFrame::parse(&knock).unwrap_err(),
            DecodeError::BadFrameType
        );
        // tamper a beacon byte → parses fine, resolves to nobody
        bytes[10] ^= 0xff;
        let contacts = vec![Contact::new("me", vec![7u8; 32])];
        assert_eq!(
            decode_and_resolve(&bytes, &contacts, 1767225600).unwrap(),
            None
        );
    }
}
