// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>
//! Cloud-egress GO/NO-GO veto (issue #103, proof obligation 3.1).
//!
//! Wraps `hyperpolymath/conative-gating`'s `gating-contract` crate as the
//! policy layer that must approve any payload before it is allowed to leave
//! the device over the network. The veto is synchronous, deterministic and
//! fast (the underlying oracle is a plain rule/regex scan, no ML), so it can
//! sit directly in front of every outbound network call without meaningfully
//! affecting latency.
//!
//! ## Adapting `gating-contract` to network egress
//!
//! `gating-contract`'s `GatingRequest`/`Proposal` types are shaped for
//! *code-proposal* gating (a candidate file edit/command, not a network
//! payload) and there is no `ActionType::NetworkEgress` variant upstream.
//! Rather than force a fictitious fit or fork conative-gating (explicitly
//! out of scope per issue #103), this module makes a deliberate, documented
//! mapping:
//!
//! - `ActionType::ExecuteCommand { command }` represents "attempt to send
//!   this payload to `destination`" (`command = "network-egress:<destination>"`).
//!   This is the closest existing variant to "perform an external-effect
//!   action"; it is not a perfect fit, but it is honest about what it means
//!   (no new upstream variant was invented for this).
//! - `Proposal.content` carries the *literal outbound payload text* (system
//!   prompt + message bodies), so the oracle's generic content-scanning
//!   rules (e.g. hardcoded-secret patterns) really do run against what would
//!   go over the wire.
//! - `Proposal.files_affected` carries a synthetic `egress://<destination>`
//!   locator, purely so the audit trail records *where* the payload was
//!   headed (there is no real file).
//! - The caller-supplied [`EgressClass`] (raw sensor / raw neural state /
//!   derived inference / plain user text) is encoded as a literal
//!   `[[EGRESS_CLASS:...]]` marker prepended to `content`. This is a real use
//!   of the oracle's actual mechanism (a configured regex over `content`),
//!   not a bypass of it: `egress_policy()` below is a bespoke
//!   `Policy` (NOT `Policy::rsr_default()`, which is a source-code-hygiene
//!   policy about forbidden languages/toolchains and has nothing useful to
//!   say about a network payload) whose only forbidden patterns are (a) the
//!   raw-sensor/raw-neural-state marker and (b) the same hardcoded-secret
//!   regex `rsr_default()` uses, reused here as defense-in-depth against a
//!   credential accidentally ending up in outbound message text.
//!
//! ## Honesty about the classification boundary
//!
//! [`EgressClass`] is **caller-declared**: this module cannot itself prove
//! that a payload labelled `DerivedInference` really is an aggregate rather
//! than raw sensor data relabelled by mistake -- that would require real
//! sensor-provenance tracking, which does not exist yet in this repo. What
//! this gate *does* guarantee is: whatever class is declared, `RawSensor`
//! and `RawNeuralState` are always blocked, and the hardcoded-secret check
//! always runs regardless of the declared class.
use gating_contract::{ContractError, ContractRunner, GatingDecision, GatingRequest, Verdict};
use policy_oracle::{
    ActionType, EnforcementConfig, ForbiddenPattern, LanguagePolicy, PatternPolicy, Policy,
    Proposal, ToolchainPolicy,
};
use uuid::Uuid;

/// Coarse, caller-declared sensitivity classification for an outbound
/// payload. See the module-level docs for the honesty caveat: this is a
/// declaration, not a proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EgressClass {
    /// Unprocessed sensor readings (accelerometer/mic/etc.) -- must never
    /// leave the device.
    RawSensor,
    /// Reservoir/neural internal state that directly reflects raw sensor
    /// input 1:1 (not yet aggregated/summarised) -- must never leave the
    /// device.
    RawNeuralState,
    /// A derived or aggregated inference over neural/sensor state (e.g. a
    /// summarised description) -- may leave the device.
    DerivedInference,
    /// Plain user-authored text with no sensor provenance -- may leave the
    /// device.
    UserText,
}

impl EgressClass {
    fn marker(self) -> &'static str {
        match self {
            EgressClass::RawSensor => "RAW_SENSOR",
            EgressClass::RawNeuralState => "RAW_NEURAL_STATE",
            EgressClass::DerivedInference => "DERIVED_INFERENCE",
            EgressClass::UserText => "USER_TEXT",
        }
    }
}

/// Errors from the egress veto stage. Distinguished from [`crate::ClaudeError`]
/// so callers can tell "the network call itself failed" apart from "the
/// network call was never attempted because policy said no".
#[derive(Debug, thiserror::Error)]
pub enum EgressGateError {
    /// Hard NO-GO: the payload must not be sent. Not overridable from here.
    #[error("egress blocked by policy: {reason}")]
    Blocked { reason: String },
    /// The policy engine wants a human decision before this can proceed.
    /// `ContractRunner::evaluate` does not currently produce this verdict
    /// (no arbiter/escalation stage is wired up yet upstream), but it is
    /// handled here so this gate fails closed rather than silently allowing
    /// a future `Escalate` verdict through.
    #[error("egress escalated for human review: {reason}")]
    Escalated { reason: String },
    /// The gating contract itself errored (e.g. an invalid regex in the
    /// policy) -- fails closed, never falls through to "allowed".
    #[error("gating contract error: {0}")]
    Contract(#[from] ContractError),
}

/// Policy tailored for network-egress veto decisions (obligation 3.1).
/// Deliberately not `Policy::rsr_default()` -- see the module docs.
fn egress_policy() -> Policy {
    Policy {
        name: "neurophone-egress-veto".to_string(),
        languages: LanguagePolicy::default(),
        toolchain: ToolchainPolicy::default(),
        patterns: PatternPolicy {
            forbidden_patterns: vec![
                ForbiddenPattern {
                    name: "raw_sensor_egress".to_string(),
                    regex: r"\[\[EGRESS_CLASS:(RAW_SENSOR|RAW_NEURAL_STATE)\]\]".to_string(),
                    file_types: vec!["*".to_string()],
                    reason: "Raw/unaggregated sensor-derived data must never leave the device (obligation 3.1)"
                        .to_string(),
                },
                ForbiddenPattern {
                    // Same rule `Policy::rsr_default()` uses for source code;
                    // reused here as defense-in-depth against a credential
                    // accidentally ending up in outbound message text.
                    name: "hardcoded_secrets".to_string(),
                    regex: r#"(?i)(password|secret|api_key)\s*=\s*["'][^"']{8,}["']"#.to_string(),
                    file_types: vec!["*".to_string()],
                    reason: "Hardcoded secret detected in outbound payload".to_string(),
                },
            ],
        },
        enforcement: EnforcementConfig::default(),
    }
}

/// GO/NO-GO veto for outbound network calls, backed by `conative-gating`'s
/// `gating-contract` crate.
pub struct EgressGate {
    runner: ContractRunner,
}

impl EgressGate {
    /// Build the gate with the bespoke `egress_policy()`.
    pub fn new() -> Self {
        Self {
            runner: ContractRunner::with_policy(egress_policy()),
        }
    }

    /// Evaluate whether `content`, classified as `class`, may be sent to
    /// `destination`. Returns `Ok(GatingDecision)` when the verdict is
    /// `Allow`/`Warn` (i.e. `Verdict::is_allowed()`); `Err` otherwise. Callers
    /// MUST NOT perform the network call unless this returns `Ok`.
    pub fn check(
        &self,
        class: EgressClass,
        destination: &str,
        content: &str,
    ) -> Result<GatingDecision, EgressGateError> {
        let tagged = format!("[[EGRESS_CLASS:{}]]\n{}", class.marker(), content);
        let proposal = Proposal {
            id: Uuid::new_v4(),
            action_type: ActionType::ExecuteCommand {
                command: format!("network-egress:{destination}"),
            },
            content: tagged,
            files_affected: vec![format!("egress://{destination}")],
            // Unused by `ContractRunner::evaluate` today (only the oracle
            // stage runs; the SLM stage that would consume this confidence
            // score is an unimplemented stub upstream). Set to 1.0 as an
            // honest "not applicable" placeholder rather than a fabricated
            // confidence value.
            llm_confidence: 1.0,
        };
        let request = GatingRequest::new(proposal);
        let decision = self.runner.evaluate(&request)?;
        let audit = self.runner.audit(&request, &decision);

        match decision.verdict {
            Verdict::Allow => {
                tracing::debug!(request_id = %audit.request_id, "egress allowed");
                Ok(decision)
            }
            Verdict::Warn => {
                let reason = decision
                    .refusal
                    .as_ref()
                    .map(|r| r.message.clone())
                    .unwrap_or_default();
                tracing::warn!(request_id = %audit.request_id, %reason, "egress allowed with policy warning");
                Ok(decision)
            }
            Verdict::Block => {
                let reason = decision
                    .refusal
                    .as_ref()
                    .map(|r| r.message.clone())
                    .unwrap_or_else(|| "blocked by policy".to_string());
                tracing::warn!(request_id = %audit.request_id, %reason, "egress BLOCKED by policy veto");
                Err(EgressGateError::Blocked { reason })
            }
            Verdict::Escalate => {
                let reason = decision
                    .refusal
                    .as_ref()
                    .map(|r| r.message.clone())
                    .unwrap_or_else(|| "escalation required".to_string());
                tracing::warn!(request_id = %audit.request_id, %reason, "egress ESCALATED for human review");
                Err(EgressGateError::Escalated { reason })
            }
        }
    }
}

impl Default for EgressGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_sensor_is_blocked() {
        let gate = EgressGate::new();
        let err = gate
            .check(
                EgressClass::RawSensor,
                "api.anthropic.com",
                "accel=[0.1,0.2,9.8]",
            )
            .expect_err("raw sensor data must be blocked");
        assert!(matches!(err, EgressGateError::Blocked { .. }));
    }

    #[test]
    fn raw_neural_state_is_blocked() {
        let gate = EgressGate::new();
        let err = gate
            .check(
                EgressClass::RawNeuralState,
                "api.anthropic.com",
                "reservoir_state=[0.02, -0.87, 0.4, ...]",
            )
            .expect_err("raw neural state must be blocked");
        assert!(matches!(err, EgressGateError::Blocked { .. }));
    }

    #[test]
    fn derived_inference_is_allowed() {
        let gate = EgressGate::new();
        let decision = gate
            .check(
                EgressClass::DerivedInference,
                "api.anthropic.com",
                "the user appears calm and focused",
            )
            .expect("derived/aggregated inference should be allowed");
        assert_eq!(decision.verdict, Verdict::Allow);
    }

    #[test]
    fn user_text_is_allowed() {
        let gate = EgressGate::new();
        let decision = gate
            .check(
                EgressClass::UserText,
                "api.anthropic.com",
                "what time is it?",
            )
            .expect("plain user text should be allowed");
        assert_eq!(decision.verdict, Verdict::Allow);
    }

    #[test]
    fn hardcoded_secret_is_blocked_even_in_an_otherwise_allowed_class() {
        let gate = EgressGate::new();
        let err = gate
            .check(
                EgressClass::UserText,
                "api.anthropic.com",
                r#"api_key = "sk-not-a-real-secret-value""#,
            )
            .expect_err("hardcoded-secret pattern must block regardless of declared class");
        assert!(matches!(err, EgressGateError::Blocked { .. }));
    }

    #[test]
    fn decision_carries_a_stable_request_id_for_audit_correlation() {
        let gate = EgressGate::new();
        let decision = gate
            .check(EgressClass::UserText, "api.anthropic.com", "hi")
            .unwrap();
        // request_id round-trips through GatingRequest -> GatingDecision, which
        // is what lets an AuditEntry correlate back to the original request.
        assert_ne!(decision.request_id, Uuid::nil());
    }
}
