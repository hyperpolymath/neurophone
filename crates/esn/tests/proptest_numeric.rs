// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Property tests — numeric containment (obligation 0.2, #87) for the ESN.
//!
//! Finite inputs (including large magnitudes) must never produce NaN/Inf in the
//! reservoir state, across a long run.

use esn::{EchoStateNetwork, EsnConfig};
use ndarray::Array1;
use proptest::prelude::*;

fn cfg() -> EsnConfig {
    EsnConfig {
        reservoir_size: 32,
        input_dim: 4,
        spectral_radius: 0.9,
        input_scale: 1.0,
        sparsity: 0.8,
        leaking_rate: 0.3,
    }
}

proptest! {
    #[test]
    fn esn_step_contains_no_nan_inf(
        seq in proptest::collection::vec(
            proptest::collection::vec(-1.0e6f32..1.0e6f32, 4), 1..50)
    ) {
        let mut e = EchoStateNetwork::new(cfg()).unwrap();
        for v in seq {
            let out = e.step(&Array1::from_vec(v));
            prop_assert!(
                out.iter().all(|x| x.is_finite()),
                "non-finite ESN state produced: {out:?}"
            );
        }
    }
}
