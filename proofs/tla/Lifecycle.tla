---------------------------- MODULE Lifecycle ----------------------------
(* SPDX-License-Identifier: MPL-2.0                              *)
(* Obligation 2.1 (issue #84): lifecycle safety for                       *)
(* neurophone-core::NeuroSymbolicSystem.                                  *)
(*                                                                        *)
(* Models the intended protocol:                                         *)
(*   new -> initialize -> { process_sensor_event | query }* -> shutdown   *)
(*                                                                        *)
(* Safety claims (checked by TLC against this spec):                      *)
(*   - no process/query before initialize                                 *)
(*   - no action after shutdown (shutdown is terminal & idempotent)       *)
(*========================================================================*)
EXTENDS Naturals

VARIABLES
    phase,      \* "created" | "initialized" | "down"
    work        \* count of process/query operations performed

vars == <<phase, work>>

TypeOK == /\ phase \in {"created", "initialized", "down"}
          /\ work \in Nat

Init == /\ phase = "created"
        /\ work  = 0

Initialize == /\ phase = "created"
              /\ phase' = "initialized"
              /\ UNCHANGED work

ProcessOrQuery == /\ phase = "initialized"
                  /\ phase' = "initialized"
                  /\ work' = work + 1

Shutdown == /\ phase = "initialized"
            /\ phase' = "down"
            /\ UNCHANGED work

\* No transition is enabled from "down": shutdown is terminal & idempotent.
Next == \/ Initialize
        \/ ProcessOrQuery
        \/ Shutdown

Spec == Init /\ [][Next]_vars

(* ---- Safety invariants ---- *)

\* Any work that happened implies we are at or past initialization.
NoUseBeforeInit == (work > 0) => (phase \in {"initialized", "down"})

\* Once down, we never perform more work and never leave "down".
NoUseAfterShutdown == [][ (phase = "down") => (phase' = "down" /\ work' = work) ]_vars

THEOREM Spec => [](TypeOK /\ NoUseBeforeInit)
==========================================================================
