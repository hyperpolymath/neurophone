// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell
//! Conformance: replay burble's FROZEN presence vectors (`vendor/ble-spa-v1.json`,
//! ADR-0015) and assert this decoder is byte-exact with burble's independent
//! HMAC-SHA256 oracle. Any drift here is a wire break, by construction.

use bt_presence::decode::{beacon_id, decode_and_resolve, epoch_for, Contact, PresenceFrame};
use serde_json::Value;

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "odd-length hex: {s}");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("valid hex"))
        .collect()
}

fn vectors() -> Value {
    let raw = include_str!("../vendor/ble-spa-v1.json");
    serde_json::from_str(raw).expect("vendor/ble-spa-v1.json is valid JSON")
}

#[test]
fn vector_file_is_frozen_v1() {
    let v = vectors();
    assert_eq!(v["wire_version"], 1);
    assert_eq!(v["spec_version"], "1.0.0");
}

#[test]
fn presence_beacon_id_is_byte_exact() {
    let v = vectors();
    let cases = v["presence"].as_array().expect("presence[] present");
    assert!(!cases.is_empty(), "expected presence vectors");
    for c in cases {
        let name = c["name"].as_str().unwrap_or("?");
        let secret = unhex(c["presence_secret_hex"].as_str().expect("secret"));
        let unix_s = c["unix_s"].as_u64().expect("unix_s");
        let expected_epoch = c["epoch"].as_u64().expect("epoch");
        let expected_beacon = unhex(c["beacon_id_hex"].as_str().expect("beacon_id_hex"));

        // 1. epoch derivation matches burble.
        assert_eq!(epoch_for(unix_s), expected_epoch, "{name}: epoch");

        // 2. our HMAC beacon id reproduces burble's oracle byte-for-byte.
        let ours = beacon_id(&secret, expected_epoch);
        assert_eq!(&ours[..], &expected_beacon[..], "{name}: beacon_id bytes");

        // 3. the full 24-byte payload parses to the same epoch + beacon.
        let payload = unhex(c["payload_hex"].as_str().expect("payload_hex"));
        let frame = PresenceFrame::parse(&payload).expect("payload parses");
        assert_eq!(frame.epoch, expected_epoch, "{name}: parsed epoch");
        assert_eq!(
            &frame.beacon_id[..],
            &expected_beacon[..],
            "{name}: parsed beacon"
        );
    }
}

#[test]
fn presence_resolves_to_named_contact() {
    let v = vectors();
    for c in v["presence"].as_array().expect("presence[]") {
        let name = c["name"].as_str().unwrap_or("?");
        let secret = unhex(c["presence_secret_hex"].as_str().expect("secret"));
        let payload = unhex(c["payload_hex"].as_str().expect("payload_hex"));
        let resolve = &c["resolve"];
        let now = resolve["now"].as_u64().expect("now");
        let contact_id = resolve["contact_id"].as_str().expect("contact_id");

        let contacts = vec![Contact::new(contact_id, secret)];
        let got = decode_and_resolve(&payload, &contacts, now).expect("well-formed frame");
        assert_eq!(got.as_deref(), Some(contact_id), "{name}: resolves owner");

        // A holder of a *different* secret must not resolve the same payload.
        let stranger = vec![Contact::new("stranger", vec![0xABu8; 32])];
        assert_eq!(
            decode_and_resolve(&payload, &stranger, now).expect("well-formed"),
            None,
            "{name}: stranger must not resolve"
        );
    }
}
