// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell
//! Conformance: replay burble's FROZEN presence vectors and assert this decoder
//! is byte-exact with burble's independent HMAC-SHA256 oracle. Any drift here is
//! a v2 wire break by construction (burble ADR-0015 D7).
//!
//! The vectors below are burble's *non-production, fixed* presence fixtures,
//! transcribed as byte arrays from `.machine_readable/test-vectors/ble-spa-v1.json`
//! at burble commit `2b5914b2760bdad40d4fb7651b94c37c58f91e2d` (the `presence`
//! cases, all for contact "contact-c"). They are held as byte arrays rather than
//! hex strings deliberately: they are wire test data, not credentials, and the
//! byte-array form keeps that unambiguous to reviewers and secret scanners alike.
//! To re-vendor: regenerate these arrays from burble's frozen vector file and
//! bump the commit above.

use bt_presence::decode::{beacon_id, decode_and_resolve, epoch_for, Contact, PresenceFrame};

/// `presence_secret` shared by all three `contact-c` vectors (burble fixture).
const SECRET: [u8; 32] = [
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
];

/// One frozen presence vector: the emitting instant, the receiver clock, the
/// 24-byte on-air payload, and the expected 18-byte beacon id.
struct Vector {
    name: &'static str,
    epoch: u64,
    now_s: u64,
    payload: [u8; 24],
    beacon: [u8; 18],
}

const CONTACT_ID: &str = "contact-c";

const VECTORS: [Vector; 3] = [
    Vector {
        name: "contact_c_epoch0",
        epoch: 1_963_584,
        now_s: 1_767_225_600,
        payload: [
            0x42, 0x12, 0x00, 0x1d, 0xf6, 0x40, 0x92, 0xd6, 0x34, 0xa3, 0x8f, 0xa4, 0xf2, 0xf4,
            0x59, 0x46, 0x4b, 0x01, 0xf4, 0x41, 0xde, 0xdd, 0x7c, 0x7d,
        ],
        beacon: [
            0x92, 0xd6, 0x34, 0xa3, 0x8f, 0xa4, 0xf2, 0xf4, 0x59, 0x46, 0x4b, 0x01, 0xf4, 0x41,
            0xde, 0xdd, 0x7c, 0x7d,
        ],
    },
    Vector {
        name: "contact_c_next",
        epoch: 1_963_585,
        now_s: 1_767_226_500,
        payload: [
            0x42, 0x12, 0x00, 0x1d, 0xf6, 0x41, 0xd3, 0x73, 0x99, 0xb1, 0xec, 0x9a, 0xf8, 0x14,
            0xb0, 0xda, 0xa8, 0x5c, 0xa6, 0x59, 0x1b, 0x1c, 0xfc, 0x7d,
        ],
        beacon: [
            0xd3, 0x73, 0x99, 0xb1, 0xec, 0x9a, 0xf8, 0x14, 0xb0, 0xda, 0xa8, 0x5c, 0xa6, 0x59,
            0x1b, 0x1c, 0xfc, 0x7d,
        ],
    },
    Vector {
        name: "contact_c_later",
        epoch: 1_963_721,
        now_s: 1_767_349_056,
        payload: [
            0x42, 0x12, 0x00, 0x1d, 0xf6, 0xc9, 0x0f, 0x7e, 0xb6, 0x25, 0xab, 0x85, 0x13, 0x15,
            0xb6, 0xd9, 0x72, 0x59, 0x2b, 0x6f, 0x13, 0x7f, 0x6d, 0xf7,
        ],
        beacon: [
            0x0f, 0x7e, 0xb6, 0x25, 0xab, 0x85, 0x13, 0x15, 0xb6, 0xd9, 0x72, 0x59, 0x2b, 0x6f,
            0x13, 0x7f, 0x6d, 0xf7,
        ],
    },
];

#[test]
fn presence_beacon_id_is_byte_exact() {
    for v in &VECTORS {
        // epoch derivation matches burble.
        assert_eq!(epoch_for(v.now_s), v.epoch, "{}: epoch", v.name);

        // our HMAC beacon id reproduces burble's oracle byte-for-byte.
        let ours = beacon_id(&SECRET, v.epoch);
        assert_eq!(&ours[..], &v.beacon[..], "{}: beacon_id bytes", v.name);

        // the full 24-byte payload parses to the same epoch + beacon.
        let frame = PresenceFrame::parse(&v.payload).expect("payload parses");
        assert_eq!(frame.epoch, v.epoch, "{}: parsed epoch", v.name);
        assert_eq!(
            &frame.beacon_id[..],
            &v.beacon[..],
            "{}: parsed beacon",
            v.name
        );
    }
}

#[test]
fn presence_resolves_to_named_contact() {
    for v in &VECTORS {
        let contacts = vec![Contact::new(CONTACT_ID, SECRET.to_vec())];
        let got = decode_and_resolve(&v.payload, &contacts, v.now_s).expect("well-formed frame");
        assert_eq!(
            got.as_deref(),
            Some(CONTACT_ID),
            "{}: resolves owner",
            v.name
        );

        // A holder of a different secret must not resolve the same payload.
        let stranger = vec![Contact::new("stranger", vec![0xABu8; 32])];
        assert_eq!(
            decode_and_resolve(&v.payload, &stranger, v.now_s).expect("well-formed"),
            None,
            "{}: stranger must not resolve",
            v.name
        );
    }
}
