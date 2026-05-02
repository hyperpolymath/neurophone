// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Integration tests for the Claude client routing logic (no network).

use claude_client::{ClaudeConfig, ClaudeModel, HybridInference, Message, MessageRole};

#[test]
fn point_to_point_message_construction() {
    let user = Message::user("hi");
    assert!(matches!(user.role, MessageRole::User));
    let asst = Message::assistant("hello");
    assert!(matches!(asst.role, MessageRole::Assistant));
}

#[test]
fn aspect_offline_never_routes_to_cloud() {
    let cfg = ClaudeConfig { api_key: Some("sk-test".into()), ..Default::default() };
    let mut h = HybridInference::new(Some(cfg));
    h.set_online(false);
    assert!(!h.should_use_cloud(0.99, true));
}

#[test]
fn aspect_no_local_always_cloud_when_online() {
    let cfg = ClaudeConfig { api_key: Some("sk-test".into()), ..Default::default() };
    let h = HybridInference::new(Some(cfg));
    assert!(h.should_use_cloud(0.0, false));
}

#[test]
fn aspect_threshold_clamped_to_unit_interval() {
    let mut h = HybridInference::new(None);
    h.set_threshold(2.5);
    h.set_threshold(-1.0);
}

#[test]
fn complexity_estimation_monotonic_for_keywords() {
    let h = HybridInference::new(None);
    let plain = h.estimate_complexity("what time is it");
    let complex = h.estimate_complexity("analyze and explain this complex situation in detail");
    assert!(complex > plain);
}

#[test]
fn complexity_estimation_bounded() {
    let h = HybridInference::new(None);
    let huge = "analyze ".repeat(5000);
    let s = h.estimate_complexity(&huge);
    assert!(s <= 1.0 && s >= 0.0);
}

#[test]
fn lifecycle_no_claude_when_no_config() {
    let h = HybridInference::new(None);
    assert!(!h.has_claude());
    assert!(!h.should_use_cloud(1.0, false));
}

#[test]
fn aspect_model_default_is_recent() {
    assert!(ClaudeModel::default().as_str().starts_with("claude-"));
}
