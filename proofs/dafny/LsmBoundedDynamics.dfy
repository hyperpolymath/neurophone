// SPDX-License-Identifier: MPL-2.0
// NeuroPhone - High-Assurance Hardware Orchestration
// Copyright (c) 2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>
//
// Obligation 1.2 (LSM bounded dynamics), issue #84 / #89.
//
// Models exactly the per-neuron Leaky Integrate-and-Fire (LIF) update in
// `crates/lsm/src/lib.rs::LiquidStateMachine::step`:
//
//   if refrac_remaining > 0.0 {
//       refrac_remaining -= dt;                      // v unchanged
//   } else {
//       dv = dt / tau_m * (-(v - v_rest) + r_m * total_current);
//       v  = v + dv;
//       if v >= v_thresh {
//           v = v_reset;
//           refrac_remaining = t_refrac;
//       }
//   }
//
// `LifStep` below is that same two-branch update (as a pure Dafny
// `function`, so its defining equations are transparent to the verifier),
// and `LifRun` folds it over an arbitrary finite sequence of per-step input
// currents (modelling an arbitrary-length simulation run, matching the
// property test's "long runs of arbitrary input" framing in
// `crates/lsm/tests/proptest_bounds.rs`).
//
// ## What is proven (the half of 1.2 that is actually true of the code)
//
// `LifRunKeepsBelowThreshold` proves, by induction over the input-current
// sequence (a loop-invariant argument mirrored by the recursive structure
// of `LifRun`), that a membrane potential which starts `<= v_thresh` *never*
// exceeds `v_thresh` at the end of any subsequent step, for *any* sequence
// of (unrestricted, not merely bounded) input currents. This holds because
// `LifStep` is a two-branch invariant-preserving map: the refractory branch
// passes `v` through unchanged (so the invariant is inherited), and the
// non-refractory branch either resets to `v_reset < v_thresh` or leaves the
// updated potential strictly below `v_thresh` (that's exactly the negation
// of the reset condition) — regardless of how large `total_current` is.
// Because Dafny's `real` is exact, this is a genuine `<=`, not the `+ 1e-3`
// slack the Rust proptest needs to accommodate `f32` rounding.
//
// ## Honest scope note — the lower bound is NOT provable (corrects an
// overclaim in the *original* `proofs/dafny/README.adoc`)
//
// The original `proofs/dafny/README.adoc` text (written before this file
// existed) claimed the membrane potential stays within `[reset, threshold +
// ε]` — a *two*-sided bound. That is false in general for the code as
// written: `total_current` (`input_weights.dot(input) + recurrent
// weights^T . spike_indicators`) is not clamped anywhere, so a sufficiently
// negative `total_current` drives `dv` (and hence the updated `v`)
// arbitrarily far *below* `v_reset` in a single step — there is a ceiling
// (the threshold-crossing reset) but no floor. `LowerBoundFails` below is a
// concrete Dafny counterexample: valid LIF parameters, a valid pre-state,
// and one step with a large negative current produce a post-step potential
// strictly below `v_reset`, refuting the general two-sided claim. This file
// (and the corrected `proofs/README.adoc` / `proofs/dafny/README.adoc`
// entries) now honestly state a *one-sided* (ceiling-only) bound, which is
// also all that `crates/lsm/tests/proptest_bounds.rs` actually asserts (it
// checks finiteness and the upper bound; it never asserts a lower bound).
//
// Similarly, the refractory countdown (`refrac_remaining -= dt`, taken
// verbatim from the Rust code) is *not* clamped at `0` either — it can dip
// slightly negative for exactly one step (harmless, since the next step's
// guard is `> 0.0`, which a small negative value also fails, so it is
// treated identically to `0`). This file does not claim
// `refracNext >= 0.0` for that reason: it would be a false, unrequested
// invariant, not something `LiquidStateMachine::step` actually maintains.
//
// ## Honest scope note — exact reals vs. `f32`
//
// Dafny's `real` is an exact, unbounded-precision numeric domain: it has no
// NaN, no `Inf`, no overflow, no rounding. This file therefore proves the
// bound for the *idealised* exact-arithmetic recurrence. It does not, by
// itself, model IEEE-754 `f32` rounding, subnormal behaviour, or
// overflow-to-infinity — that empirical side (over the real `f32`
// execution) is exactly what `crates/lsm/tests/proptest_bounds.rs` and
// `crates/lsm/tests/proptest_numeric.rs` check. A Dafny/SMT
// floating-point-theory model (or a Kani harness) bridging exact reals to
// `f32` remains open — tracked under issue #84.
//
// ## Honest scope note — spike-history bound not attempted here
//
// `crates/lsm/tests/proptest_bounds.rs` also property-tests a *third*
// invariant: retained spike-history length per neuron stays within
// `ceil(history_window / t_refrac) + 2`. This file does not attempt that
// (it is a separate combinatorial argument over spike *timing*, not the
// membrane-potential recurrence) — left open, see `proofs/README.adoc`.

module LsmBoundedDynamics {

  // LIF parameters, named exactly as the fields of
  // `crates/lsm/src/lib.rs::LifParameters` (`tau_m`, `v_rest`, `v_thresh`,
  // `v_reset`, `t_refrac`, `r_m`).
  datatype LifParams = LifParams(
    tauM: real,
    vRest: real,
    vThresh: real,
    vReset: real,
    tRefrac: real,
    rM: real)

  // The configuration invariants the Rust code relies on (positive time
  // constant; the reset potential is strictly below the firing threshold —
  // true of `LifParameters::default()`: -70.0 < -50.0).
  predicate ValidParams(p: LifParams)
  {
    p.tauM > 0.0 && p.vReset < p.vThresh
  }

  /// One LIF step for a single neuron, exactly mirroring the per-neuron
  /// branch of `LiquidStateMachine::step`. A transparent `function` (not an
  /// opaque `method`) so its defining equations are directly available to
  /// the verifier at call sites. Returns `(nextV, nextRefracRemaining)`.
  function LifStep(p: LifParams, dt: real, v: real, refracRemaining: real, totalCurrent: real)
      : (real, real)
    requires ValidParams(p)
  {
    if refracRemaining > 0.0 then
      (v, refracRemaining - dt)
    else
      var dv := dt / p.tauM * (-(v - p.vRest) + p.rM * totalCurrent);
      var vUpdated := v + dv;
      if vUpdated >= p.vThresh then (p.vReset, p.tRefrac) else (vUpdated, 0.0)
  }

  /// **Obligation 1.2 (single-step ceiling).** If the potential entering
  /// the step already respects the threshold, it still does after the
  /// step — regardless of the magnitude of `totalCurrent`.
  lemma LifStepKeepsBelowThreshold(p: LifParams, dt: real, v: real, refracRemaining: real, totalCurrent: real)
    requires ValidParams(p)
    requires v <= p.vThresh
    ensures LifStep(p, dt, v, refracRemaining, totalCurrent).0 <= p.vThresh
  {
    // `LifStep` is a transparent function, so Dafny unfolds it here: in the
    // refractory branch the result is literally `v`, which is `<= vThresh`
    // by hypothesis; in the other branch it is either `vReset` (`< vThresh`
    // by `ValidParams`) or `vUpdated` (`< vThresh` by the negation of the
    // reset guard). No further hints are needed.
  }

  /// **Obligation 1.2 (membrane-potential ceiling, whole-run version).**
  /// Folding `LifStep` over *any* finite sequence of per-step input
  /// currents (arbitrary reals — a fortiori covering the property test's
  /// bounded `-10.0..10.0` input range) never lets the membrane potential
  /// exceed `v_thresh`, given a valid starting state. Proved by induction
  /// over the sequence length via a loop invariant.
  lemma LifRunKeepsBelowThreshold(p: LifParams, dt: real, v0: real, refrac0: real, currents: seq<real>)
    requires ValidParams(p)
    requires v0 <= p.vThresh
    ensures LifRun(p, dt, v0, refrac0, currents).0 <= p.vThresh
  {
    var v := v0;
    var refrac := refrac0;
    var i := 0;
    while i < |currents|
      invariant 0 <= i <= |currents|
      invariant v <= p.vThresh
      invariant (v, refrac) == LifRunUpTo(p, dt, v0, refrac0, currents, i)
    {
      LifStepKeepsBelowThreshold(p, dt, v, refrac, currents[i]);
      var next := LifStep(p, dt, v, refrac, currents[i]);
      v, refrac := next.0, next.1;
      i := i + 1;
    }
  }

  /// Fold `LifStep` over the first `i` elements of `currents`.
  function LifRunUpTo(p: LifParams, dt: real, v0: real, refrac0: real, currents: seq<real>, i: nat)
      : (real, real)
    requires ValidParams(p)
    requires i <= |currents|
  {
    if i == 0 then (v0, refrac0)
    else
      var (v, refrac) := LifRunUpTo(p, dt, v0, refrac0, currents, i - 1);
      LifStep(p, dt, v, refrac, currents[i - 1])
  }

  /// Fold `LifStep` over an entire (arbitrary-length) input sequence.
  function LifRun(p: LifParams, dt: real, v0: real, refrac0: real, currents: seq<real>): (real, real)
    requires ValidParams(p)
  {
    LifRunUpTo(p, dt, v0, refrac0, currents, |currents|)
  }

  /// **Honest counterexample.** Refutes the two-sided `[v_reset, v_thresh]`
  /// bound the *original* `proofs/dafny/README.adoc` claimed: with
  /// `LifParameters::default()`-shaped parameters, starting at `v_rest`,
  /// one step with a large negative `total_current` produces a post-step
  /// potential strictly below `v_reset`. (Arithmetic: `dv = 1/20 * (-(−65 −
  /// (−65)) + 10 * (−1000)) = -500`, so `v' = -65 + (-500) = -565 <
  /// v_reset = -70`.)
  lemma LowerBoundFails()
    ensures
      var p := LifParams(20.0, -65.0, -50.0, -70.0, 2.0, 10.0);
      LifStep(p, 1.0, -65.0, 0.0, -1000.0).0 < p.vReset
  {
  }
}
