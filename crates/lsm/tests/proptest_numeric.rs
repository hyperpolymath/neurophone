// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Property tests — numeric containment (obligation 0.2, #87) for the LSM.
//!
//! Finite inputs (including large magnitudes) must never produce NaN/Inf in the
//! membrane/state output, across a long run.

use lsm::{LiquidStateMachine, LsmConfig};
use ndarray::Array1;
use proptest::prelude::*;

proptest! {
    #[test]
    fn lsm_step_contains_no_nan_inf(
        seq in proptest::collection::vec(
            proptest::collection::vec(-1.0e6f32..1.0e6f32, 3), 1..40)
    ) {
        let cfg = LsmConfig { dimensions: (3, 3, 3), ..Default::default() };
        let mut m = LiquidStateMachine::new(cfg, 3).unwrap();
        for v in seq {
            let out = m.step(&Array1::from_vec(v));
            prop_assert!(
                out.iter().all(|x| x.is_finite()),
                "non-finite LSM state produced: {out:?}"
            );
        }
    }
}
