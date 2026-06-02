// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Property tests — numeric containment (obligation 0.2, #87) for sensor filtering.
//!
//! Cascaded IIR low-pass/high-pass filtering of finite inputs (including large
//! magnitudes) must never produce NaN/Inf.

use proptest::prelude::*;
use sensors::IirFilter;

proptest! {
    #[test]
    fn iir_filter_contains_no_nan_inf(
        seq in proptest::collection::vec(
            proptest::collection::vec(-1.0e6f32..1.0e6f32, 3), 1..60)
    ) {
        let mut lp = IirFilter::new(3, 5.0, 50.0, false).unwrap();
        let mut hp = IirFilter::new(3, 0.5, 50.0, true).unwrap();
        for v in seq {
            let a = lp.step(&v).unwrap();
            let b = hp.step(&a).unwrap();
            prop_assert!(
                b.iter().all(|x| x.is_finite()),
                "non-finite filter output: {b:?}"
            );
        }
    }
}
