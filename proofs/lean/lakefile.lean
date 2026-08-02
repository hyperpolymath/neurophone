-- SPDX-License-Identifier: MPL-2.0
-- Obligation 1.1 (Echo State Property) — see issue #84 / #88 and
-- proofs/README.adoc. Toolchain pinned in `lean-toolchain`
-- (`leanprover/lean4:v4.16.0`, matching the mathlib revision below).
import Lake
open Lake DSL

package neurophone_proofs where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.16.0"

/-- Obligation 1.1: the ESN reservoir state-update map is a contraction
    (Echo State / fading-memory Property) whenever the recurrent weight
    matrix's operator norm is `< 1`. See `EsnEcho/Contraction.lean` for the
    exact statement and honest scope notes. -/
@[default_target]
lean_lib EsnEcho where
  srcDir := "."
  roots := #[`EsnEcho]
