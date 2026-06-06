// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Property tests — LSM bounded dynamics (obligation 1.2, #89).
//!
//! Across long runs of arbitrary (bounded) input:
//!   * every membrane potential stays finite and at or below threshold (a spike
//!     resets it within the same step, so it can never end a step above θ);
//!   * each neuron's retained spike history stays within the window/refractory bound.

use lsm::{LiquidStateMachine, LsmConfig};
use ndarray::Array1;
use proptest::prelude::*;

proptest! {
    #[test]
    fn lsm_membrane_and_spike_history_bounded(
        seq in proptest::collection::vec(
            proptest::collection::vec(-10.0f32..10.0f32, 3), 1..200)
    ) {
        let cfg = LsmConfig { dimensions: (3, 3, 3), ..Default::default() };
        let mut m = LiquidStateMachine::new(cfg, 3).unwrap();

        let v_thresh = m.lif_params().v_thresh;
        let t_refrac = m.lif_params().t_refrac as f64;
        let window = m.history_window_ms();
        // A neuron can spike at most once per refractory period, and history only
        // retains spikes within `window`, so the retained count is bounded.
        let max_hist = (window / t_refrac).ceil() as usize + 2;

        for v in seq {
            let _ = m.step(&Array1::from_vec(v));
            for mv in m.membrane_potentials() {
                prop_assert!(mv.is_finite(), "non-finite membrane potential: {mv}");
                prop_assert!(
                    mv <= v_thresh + 1e-3,
                    "membrane {mv} ended a step above threshold {v_thresh}"
                );
            }
            prop_assert!(
                m.max_spike_history_len() <= max_hist,
                "spike history {} exceeded bound {}",
                m.max_spike_history_len(),
                max_hist
            );
        }
    }
}
