// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Property tests — bridge soundness (obligation 1.3, #90).
//!
//! The neural->symbolic encoder must be:
//!   * total — defined on all finite inputs (any/empty/mismatched lengths), no panic;
//!   * deterministic — identical input sequences yield identical outputs (incl. the
//!     history-dependent urgency term);
//!   * well-formed — salience, urgency in [0,1] and finite; active counts never
//!     exceed the input lengths; description never empty.

use bridge::{Bridge, BridgeConfig};
use ndarray::Array1;
use proptest::prelude::*;

proptest! {
    #[test]
    fn bridge_encode_deterministic_and_bounded(
        seq in proptest::collection::vec(
            (
                proptest::collection::vec(-1.0e3f32..1.0e3f32, 0..48),
                proptest::collection::vec(-1.0e3f32..1.0e3f32, 0..48),
            ),
            1..40)
    ) {
        let mut b1 = Bridge::new(BridgeConfig::default()).unwrap();
        let mut b2 = Bridge::new(BridgeConfig::default()).unwrap();

        for (lv, ev) in seq {
            let lsm = Array1::from_vec(lv);
            let esn = Array1::from_vec(ev);

            let c1 = b1.encode(lsm.view(), esn.view());
            let c2 = b2.encode(lsm.view(), esn.view());

            // Determinism (including the last_lsm-dependent urgency).
            prop_assert_eq!(&c1, &c2);

            // Well-formed, bounded outputs.
            prop_assert!(c1.salience.is_finite() && (0.0..=1.0).contains(&c1.salience));
            prop_assert!(c1.urgency.is_finite() && (0.0..=1.0).contains(&c1.urgency));
            prop_assert!(c1.lsm_active <= lsm.len());
            prop_assert!(c1.esn_active <= esn.len());
            prop_assert!(!c1.description.is_empty());
        }
    }
}
