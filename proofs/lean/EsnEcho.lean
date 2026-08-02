-- SPDX-License-Identifier: MPL-2.0
-- NeuroPhone - High-Assurance Hardware Orchestration
-- Copyright (c) 2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>
--
-- Obligation 1.1 (Echo State Property), issue #84 / #88.
--
-- This file formalises and *proves* (no `sorry`/`admit`) that the Echo State
-- Network reservoir update implemented in `crates/esn/src/lib.rs`
-- (`EchoStateNetwork::step`) is a contraction in the state variable whenever
-- the recurrent weight matrix's infinity-operator-norm (max absolute row
-- sum) is below `1`, and derives the classical Echo State (fading-memory)
-- Property as a corollary: iterating the update under an arbitrary shared
-- input sequence forgets the initial reservoir state exponentially fast.
--
-- ## Exact correspondence with the Rust implementation
--
-- `crates/esn/src/lib.rs::EchoStateNetwork::step` computes
-- ```
-- total_activation   = input_weights.dot(input) + recurrent_weights.dot(state)
-- new_state          = (1 - leaking_rate) * state
--                       + leaking_rate * total_activation.mapv(tanh)
-- ```
-- i.e. (writing `a` for `leaking_rate`, `W` for `recurrent_weights`, and
-- collecting `input_weights.dot(input)` into a single vector `b` that is
-- *constant* across the two trajectories being compared — it depends only on
-- the shared input at that step, not on the reservoir state):
-- `x' = (1 - a) • x + a • tanh(W *ᵥ x + b)`.
-- `update` below is exactly this map.
--
-- ## Honest scope note — spectral radius vs. operator norm (the actual gap)
--
-- The theorem's hypothesis is `‖W‖∞ < 1` (`rowAbsSum W i ≤ ρ < 1` for every
-- row `i`), matching the classical sufficient condition for the Echo State
-- Property (tanh is 1-Lipschitz; a convex combination with a `< 1`-operator-norm
-- linear map is a strict contraction). This is also what the *original*
-- `proofs/lean/README.adoc` text described before this file existed.
--
-- `crates/esn/src/lib.rs::scale_to_spectral_radius` scales the reservoir by
-- the matrix's *true spectral radius* (via power iteration,
-- `estimate_spectral_radius`), **not** by the infinity norm. For a general
-- (non-normal) real matrix the spectral radius can be *strictly smaller* than
-- every induced operator norm — e.g. a nilpotent matrix has spectral radius
-- `0` but can have an arbitrarily large infinity norm — so
-- "`spectral_radius(W) < 1`" does **not**, in general, imply "`‖W‖∞ < 1`".
-- This is a known subtlety in the reservoir-computing literature (spectral
-- radius `< 1` is the classical *heuristic* used in practice, but is not by
-- itself a sufficient condition for the fading-memory property for
-- non-normal reservoirs — see e.g. Yildiz, Jaeger & Kiebel 2012,
-- "Re-visiting the echo state property").
--
-- So: **this file fully and honestly discharges the contraction theorem
-- under the classical `‖W‖∞ < 1` hypothesis** (a complete, `sorry`-free proof
-- — see `update_isContraction` / `update_iterate_isContraction` below). What
-- remains genuinely open (tracked in `proofs/README.adoc` / issue #84) is the
-- *separate* bridge lemma "`spectral_radius(W) < 1` (what the Rust code
-- actually enforces) implies `‖W‖∞ < ρ'` for some `ρ' < 1`" — this is false in
-- full generality for non-normal matrices (as above), and the true general
-- fact (Gelfand's formula: for every `ε > 0` there is *some* submultiplicative
-- norm with `‖W‖ ≤ spectralRadius W + ε`) is a substantially harder theorem
-- (needs a real Schur/Jordan-form construction) that has not been attempted
-- here. Do not read this file as closing that gap — it does not.

import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Data.Complex.Trigonometric
import Mathlib.Data.Matrix.Mul
import Mathlib.Analysis.Normed.Group.Constructions

open scoped Matrix
open Finset

namespace NeuroPhone.EsnEcho

/-! ### Vectors and matrices

Reservoir states live in `Fin n → ℝ`; Mathlib's `Pi.normedAddCommGroup`
instance equips this with the **sup norm** `‖x‖ = ⨆ i, |x i|`
(`Finset.univ.sup` on `Fin n`, `0` when `n = 0`), i.e. exactly `‖x‖∞`. -/

variable {n : ℕ}

/-- The max-absolute-row-sum ("infinity operator norm") of a real square
matrix, row `i`. -/
def rowAbsSum (W : Matrix (Fin n) (Fin n) ℝ) (i : Fin n) : ℝ :=
  ∑ j, |W i j|

theorem rowAbsSum_def (W : Matrix (Fin n) (Fin n) ℝ) (i : Fin n) :
    rowAbsSum W i = ∑ j, |W i j| :=
  rfl

/-- `Matrix.mulVec` unfolds to the expected finite sum. -/
theorem mulVec_apply (W : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) (i : Fin n) :
    (W *ᵥ x) i = ∑ j, W i j * x j :=
  rfl

/-- Each component of `W *ᵥ x` is bounded by `(rowAbsSum W i) * ‖x‖`. -/
theorem abs_mulVec_apply_le (W : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) (i : Fin n) :
    |(W *ᵥ x) i| ≤ rowAbsSum W i * ‖x‖ := by
  rw [mulVec_apply]
  calc
    |∑ j, W i j * x j| ≤ ∑ j, |W i j * x j| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ j, |W i j| * |x j| := by simp [abs_mul]
    _ ≤ ∑ j, |W i j| * ‖x‖ := by
        apply Finset.sum_le_sum
        intro j _
        have hxj : |x j| ≤ ‖x‖ := by
          have := norm_le_pi_norm x j
          rwa [Real.norm_eq_abs] at this
        exact mul_le_mul_of_nonneg_left hxj (abs_nonneg _)
    _ = rowAbsSum W i * ‖x‖ := by rw [rowAbsSum_def, Finset.sum_mul]

/-- If every row of `W` has absolute-sum `≤ ρ`, then `W` is `ρ`-Lipschitz
(as a linear map) with respect to the sup norm: `‖W *ᵥ x‖ ≤ ρ * ‖x‖`. -/
theorem norm_mulVec_le {W : Matrix (Fin n) (Fin n) ℝ} {ρ : ℝ} (hρ0 : 0 ≤ ρ)
    (hW : ∀ i, rowAbsSum W i ≤ ρ) (x : Fin n → ℝ) : ‖W *ᵥ x‖ ≤ ρ * ‖x‖ := by
  have hb : 0 ≤ ρ * ‖x‖ := mul_nonneg hρ0 (norm_nonneg _)
  refine (pi_norm_le_iff_of_nonneg hb).2 fun i => ?_
  rw [Real.norm_eq_abs]
  exact (abs_mulVec_apply_le W x i).trans (mul_le_mul_of_nonneg_right (hW i) (norm_nonneg _))

/-! ### `tanh` is `1`-Lipschitz

Classical fact: `tanh' x = 1 / (cosh x)^2 ∈ (0, 1]` since `cosh x ≥ 1`
everywhere, so the mean-value theorem gives a global `1`-Lipschitz bound. -/

theorem hasDerivAt_tanh (x : ℝ) : HasDerivAt Real.tanh (1 / Real.cosh x ^ 2) x := by
  have hc : Real.cosh x ≠ 0 := (Real.cosh_pos x).ne'
  have h := (Real.hasDerivAt_sinh x).div (Real.hasDerivAt_cosh x) hc
  have hnum : Real.cosh x * Real.cosh x - Real.sinh x * Real.sinh x = 1 := by
    linear_combination Real.cosh_sq_sub_sinh_sq x
  rw [hnum] at h
  have hfun : (fun y => Real.sinh y / Real.cosh y) = Real.tanh :=
    funext fun y => (Real.tanh_eq_sinh_div_cosh y).symm
  rwa [hfun] at h

theorem deriv_tanh_le_one (x : ℝ) : |deriv Real.tanh x| ≤ 1 := by
  rw [(hasDerivAt_tanh x).deriv]
  have hc1 : (1 : ℝ) ≤ Real.cosh x := Real.one_le_cosh x
  have hsq : (1 : ℝ) ≤ Real.cosh x ^ 2 := by nlinarith
  rw [abs_of_pos (by positivity), div_le_one (by positivity)]
  linarith

/-- `Real.tanh` is `1`-Lipschitz: `|tanh a - tanh b| ≤ |a - b|`. -/
theorem tanh_lipschitz (a b : ℝ) : |Real.tanh a - Real.tanh b| ≤ |a - b| := by
  have hdiff : ∀ x ∈ (Set.univ : Set ℝ), DifferentiableAt ℝ Real.tanh x :=
    fun x _ => (hasDerivAt_tanh x).differentiableAt
  have hbound : ∀ x ∈ (Set.univ : Set ℝ), ‖deriv Real.tanh x‖ ≤ 1 := by
    intro x _
    simpa [Real.norm_eq_abs] using deriv_tanh_le_one x
  have hmvt := convex_univ.norm_image_sub_le_of_norm_deriv_le hdiff hbound
    (Set.mem_univ b) (Set.mem_univ a)
  simpa [Real.norm_eq_abs] using hmvt

/-- Elementwise `tanh` on reservoir-sized vectors. -/
noncomputable def vtanh (x : Fin n → ℝ) : Fin n → ℝ := fun i => Real.tanh (x i)

/-- `vtanh` is `1`-Lipschitz for the sup norm: `‖vtanh x - vtanh y‖ ≤ ‖x - y‖`. -/
theorem norm_vtanh_sub_le (x y : Fin n → ℝ) : ‖vtanh x - vtanh y‖ ≤ ‖x - y‖ := by
  refine (pi_norm_le_iff_of_nonneg (norm_nonneg _)).2 fun i => ?_
  have hi : ‖(vtanh x - vtanh y) i‖ = |Real.tanh (x i) - Real.tanh (y i)| := by
    simp [vtanh, Real.norm_eq_abs]
  rw [hi]
  refine (tanh_lipschitz (x i) (y i)).trans ?_
  have : |x i - y i| = ‖(x - y) i‖ := by simp [Real.norm_eq_abs]
  rw [this]
  exact norm_le_pi_norm (x - y) i

/-! ### The reservoir update map and its contraction constant -/

/-- The Echo State Network reservoir update, exactly as implemented by
`EchoStateNetwork::step`: `x' = (1 - a) • x + a • tanh(W *ᵥ x + b)`, where `a`
is the leaking rate, `W` the recurrent weight matrix, and `b` the (per-step,
trajectory-independent) driven-input contribution `input_weights.dot(input)`. -/
noncomputable def update (a : ℝ) (W : Matrix (Fin n) (Fin n) ℝ) (b x : Fin n → ℝ) : Fin n → ℝ :=
  (1 - a) • x + a • vtanh (W *ᵥ x + b)

/-- **Obligation 1.1 (single step).** If the leaking rate `a ∈ [0, 1]` and the
recurrent matrix satisfies `‖W‖∞ ≤ ρ`, then `update a W b` is Lipschitz in the
state with constant `(1 - a) + a * ρ`, uniformly in the (shared, per-step)
input contribution `b`. -/
theorem update_lipschitz {a ρ : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (hρ0 : 0 ≤ ρ)
    {W : Matrix (Fin n) (Fin n) ℝ} (hW : ∀ i, rowAbsSum W i ≤ ρ) (b x y : Fin n → ℝ) :
    ‖update a W b x - update a W b y‖ ≤ ((1 - a) + a * ρ) * ‖x - y‖ := by
  have hstep :
      update a W b x - update a W b y
        = (1 - a) • (x - y) + a • (vtanh (W *ᵥ x + b) - vtanh (W *ᵥ y + b)) := by
    funext i
    simp only [update, Pi.sub_apply, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    ring
  rw [hstep]
  refine (norm_add_le _ _).trans ?_
  have h1 : ‖(1 - a) • (x - y)‖ ≤ (1 - a) * ‖x - y‖ := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (by linarith)]
  have h2 : ‖a • (vtanh (W *ᵥ x + b) - vtanh (W *ᵥ y + b))‖ ≤ a * (ρ * ‖x - y‖) := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg ha0]
    have hlin : W *ᵥ x + b - (W *ᵥ y + b) = W *ᵥ (x - y) := by
      rw [Matrix.mulVec_sub]; abel
    have := norm_vtanh_sub_le (W *ᵥ x + b) (W *ᵥ y + b)
    rw [hlin] at this
    have hWxy := norm_mulVec_le hρ0 hW (x - y)
    calc
      a * ‖vtanh (W *ᵥ x + b) - vtanh (W *ᵥ y + b)‖ ≤ a * ‖W *ᵥ (x - y)‖ :=
        mul_le_mul_of_nonneg_left this ha0
      _ ≤ a * (ρ * ‖x - y‖) := mul_le_mul_of_nonneg_left hWxy ha0
  calc
    ‖(1 - a) • (x - y)‖ + ‖a • (vtanh (W *ᵥ x + b) - vtanh (W *ᵥ y + b))‖
        ≤ (1 - a) * ‖x - y‖ + a * (ρ * ‖x - y‖) := add_le_add h1 h2
    _ = ((1 - a) + a * ρ) * ‖x - y‖ := by ring

/-- The single-step contraction constant is strictly below `1` exactly when
the leaking rate is strictly positive and `ρ < 1` — i.e. `update` is a
genuine (strict) contraction, not merely non-expansive. A leaking rate of
exactly `0` is a real, if degenerate, edge case allowed by
`EsnConfig`'s validation (`leaking_rate ∈ [0.0, 1.0]`): with `a = 0` the
update never changes the state (`x' = x`) regardless of input, so the Echo
State Property genuinely fails at that boundary — this is not a proof gap,
it is a correct fact about the map. -/
theorem contraction_constant_lt_one {a ρ : ℝ} (ha0 : 0 < a) (hρ1 : ρ < 1) :
    (1 - a) + a * ρ < 1 := by nlinarith

/-- Iterating `update` over a (finite, arbitrary-length) shared input
sequence `bs`, one additive contribution per step, oldest first. This models
running the same driven input sequence into two different initial reservoir
states. -/
noncomputable def iterateUpdate (a : ℝ) (W : Matrix (Fin n) (Fin n) ℝ)
    (bs : List (Fin n → ℝ)) (x : Fin n → ℝ) : Fin n → ℝ :=
  bs.foldl (fun acc b => update a W b acc) x

/-- **Obligation 1.1 (Echo State / fading-memory property).** Driving two
different reservoir states `x0`, `y0` with the *same* input sequence `bs`
(any finite length) shrinks their distance by the single-step contraction
constant raised to the `bs.length`-th power. Since `(1 - a) + a * ρ < 1`
whenever `0 < a` and `ρ < 1` (`contraction_constant_lt_one`), this distance
→ 0 as the sequence length → ∞ — the reservoir *forgets its initial state*
independently of what that initial state was, which is precisely the Echo
State (fading-memory) Property. -/
theorem iterateUpdate_lipschitz {a ρ : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (hρ0 : 0 ≤ ρ)
    {W : Matrix (Fin n) (Fin n) ℝ} (hW : ∀ i, rowAbsSum W i ≤ ρ) :
    ∀ (bs : List (Fin n → ℝ)) (x0 y0 : Fin n → ℝ),
      ‖iterateUpdate a W bs x0 - iterateUpdate a W bs y0‖
        ≤ ((1 - a) + a * ρ) ^ bs.length * ‖x0 - y0‖ := by
  have hL0 : 0 ≤ (1 - a) + a * ρ := by nlinarith
  intro bs
  induction bs with
  | nil =>
      intro x0 y0
      simp [iterateUpdate]
  | cons b bs' ih =>
      intro x0 y0
      have hunfold : ∀ z : Fin n → ℝ,
          iterateUpdate a W (b :: bs') z = iterateUpdate a W bs' (update a W b z) := fun z => rfl
      rw [hunfold x0, hunfold y0]
      calc
        ‖iterateUpdate a W bs' (update a W b x0) - iterateUpdate a W bs' (update a W b y0)‖
            ≤ ((1 - a) + a * ρ) ^ bs'.length * ‖update a W b x0 - update a W b y0‖ :=
          ih (update a W b x0) (update a W b y0)
        _ ≤ ((1 - a) + a * ρ) ^ bs'.length * (((1 - a) + a * ρ) * ‖x0 - y0‖) := by
              apply mul_le_mul_of_nonneg_left (update_lipschitz ha0 ha1 hρ0 hW b x0 y0)
              exact pow_nonneg hL0 _
        _ = ((1 - a) + a * ρ) ^ (b :: bs').length * ‖x0 - y0‖ := by
              rw [List.length_cons]; ring

/-- **Obligation 1.1 (closing the loop).** Corollary of `iterateUpdate_lipschitz`:
for a strict contraction (`0 < a`, `ρ < 1`) and *any* way of extending the
shared input sequence over time (`bs : ℕ → List (Fin n → ℝ)` with
`(bs k).length = k`), two reservoirs started from different initial states
`x0`, `y0` and driven by that same growing input sequence converge to each
other — `iterateUpdate a W (bs k) x0 - iterateUpdate a W (bs k) y0 → 0` as
`k → ∞`. This is the formal statement of "fading memory" / "echo state"
independence from initial conditions: the reservoir state is asymptotically
determined entirely by the (shared) input history, not by where it started. -/
theorem iterateUpdate_forgets_initial_state {a ρ : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) {W : Matrix (Fin n) (Fin n) ℝ}
    (hW : ∀ i, rowAbsSum W i ≤ ρ) (bs : ℕ → List (Fin n → ℝ))
    (hbs : ∀ k, (bs k).length = k) (x0 y0 : Fin n → ℝ) :
    Filter.Tendsto
      (fun k => iterateUpdate a W (bs k) x0 - iterateUpdate a W (bs k) y0)
      Filter.atTop (nhds 0) := by
  have hL0 : 0 ≤ (1 - a) + a * ρ := by nlinarith
  have hL1 : (1 - a) + a * ρ < 1 := contraction_constant_lt_one ha0 hρ1
  have hbound : ∀ k, ‖iterateUpdate a W (bs k) x0 - iterateUpdate a W (bs k) y0‖
      ≤ ((1 - a) + a * ρ) ^ k * ‖x0 - y0‖ := by
    intro k
    have hik := iterateUpdate_lipschitz ha0.le ha1 hρ0 hW (bs k) x0 y0
    rwa [hbs k] at hik
  have htends :
      Filter.Tendsto (fun k => ((1 - a) + a * ρ) ^ k * ‖x0 - y0‖) Filter.atTop (nhds 0) := by
    simpa using (tendsto_pow_atTop_nhds_zero_of_lt_one hL0 hL1).mul_const (‖x0 - y0‖)
  exact squeeze_zero_norm hbound htends

end NeuroPhone.EsnEcho
