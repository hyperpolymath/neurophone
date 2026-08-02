// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell
//! Neural-to-symbolic Action Gate (policy enforcement).

use gating_contract::{ContractRunner, GatingDecision, GatingRequest, Verdict};
use policy_oracle::{EnforcementConfig, ForbiddenPattern, LanguagePolicy, PatternPolicy, Policy, Proposal, ToolchainPolicy, ActionType};
use gating_contract::ContractError;
use thiserror::Error;
use uuid::Uuid;

/// Errors emitted by the bridge action gate.
#[derive(Error, Debug)]
pub enum ActionGateError {
    /// Hard NO-GO: the action must not be dispatched.
    #[error("action blocked by policy: {reason}")]
    Blocked { reason: String },
    /// Escalate for human review.
    #[error("action escalated for human review: {reason}")]
    Escalated { reason: String },
    /// Gating contract failure.
    #[error("gating contract error: {0}")]
    Contract(#[from] ContractError),
}

/// Policy tailored for neural-to-symbolic bridge actions (veto).
fn action_policy() -> Policy {
    Policy {
        name: "neurophone-action-veto".to_string(),
        languages: LanguagePolicy::default(),
        toolchain: ToolchainPolicy::default(),
        patterns: PatternPolicy {
            forbidden_patterns: vec![
                ForbiddenPattern {
                    name: "low_confidence_action".to_string(),
                    regex: r"\[\[ACTION_CONFIDENCE:LOW\]\]".to_string(),
                    file_types: vec!["*".to_string()],
                    reason: "Low-confidence neural inferences cannot trigger symbolic effectors".to_string(),
                },
            ],
        },
        enforcement: EnforcementConfig::default(),
    }
}

/// GO/NO-GO veto for bridge action dispatch.
pub struct ActionGate {
    runner: ContractRunner,
}

impl ActionGate {
    pub fn new() -> Self {
        Self {
            runner: ContractRunner::with_policy(action_policy()),
        }
    }

    /// Check if a proposed action is permitted.
    pub fn check(
        &self,
        confidence: f32,
        action_desc: &str,
    ) -> Result<GatingDecision, ActionGateError> {
        let tag = if confidence < 0.6 { "LOW" } else { "HIGH" };
        let content = format!("[[ACTION_CONFIDENCE:{}]]\n{}", tag, action_desc);
        
        let proposal = Proposal {
            id: Uuid::new_v4(),
            action_type: ActionType::ExecuteCommand {
                command: "neural-bridge:dispatch".to_string(),
            },
            content,
            files_affected: vec!["effector://bridge".to_string()],
            llm_confidence: confidence,
        };
        
        let request = GatingRequest::new(proposal);
        let decision = self.runner.evaluate(&request)?;
        let audit = self.runner.audit(&request, &decision);

        match decision.verdict {
            Verdict::Allow => {
                tracing::debug!(request_id = %audit.request_id, "action allowed");
                Ok(decision)
            }
            Verdict::Warn => {
                let reason = decision.refusal.as_ref().map(|r| r.message.clone()).unwrap_or_default();
                tracing::warn!(request_id = %audit.request_id, %reason, "action allowed with warning");
                Ok(decision)
            }
            Verdict::Block => {
                let reason = decision.refusal.as_ref().map(|r| r.message.clone()).unwrap_or_else(|| "blocked by policy".to_string());
                tracing::warn!(request_id = %audit.request_id, %reason, "action BLOCKED by policy veto");
                Err(ActionGateError::Blocked { reason })
            }
            Verdict::Escalate => {
                let reason = decision.refusal.as_ref().map(|r| r.message.clone()).unwrap_or_else(|| "escalation required".to_string());
                tracing::warn!(request_id = %audit.request_id, %reason, "action ESCALATED for human review");
                Err(ActionGateError::Escalated { reason })
            }
        }
    }
}

impl Default for ActionGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_low_confidence() {
        let gate = ActionGate::new();
        let err = gate.check(0.5, "turn on lights").expect_err("should block");
        assert!(matches!(err, ActionGateError::Blocked { .. }));
    }

    #[test]
    fn allow_high_confidence() {
        let gate = ActionGate::new();
        let res = gate.check(0.8, "turn on lights").expect("should allow");
        assert_eq!(res.verdict, Verdict::Allow);
    }
}
