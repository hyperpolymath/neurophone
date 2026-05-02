// SPDX-License-Identifier: PMPL-1.0-or-later
// SPDX-FileCopyrightText: 2025 Jonathan D.A. Jewell
//! Integration tests for the LLM backend abstraction.

use llm::{LlmBackend, LlmConfig, LlmError, MockBackend};

fn backend() -> MockBackend {
    let mut b = MockBackend::new(LlmConfig::default()).unwrap();
    b.load().unwrap();
    b
}

#[test]
fn point_to_point_prompt_to_response() {
    let mut b = backend();
    let r = b.generate("the quick brown fox", 4).unwrap();
    assert_eq!(r.tokens_emitted, 4);
    assert!(r.text.contains("the"));
}

#[test]
fn lifecycle_load_generate_unload_reload() {
    let mut b = MockBackend::new(LlmConfig::default()).unwrap();
    assert!(matches!(b.generate("x", 1), Err(LlmError::NotLoaded)));
    b.load().unwrap();
    b.generate("hello", 2).unwrap();
    b.unload();
    assert!(matches!(b.generate("x", 1), Err(LlmError::NotLoaded)));
    b.load().unwrap();
    assert!(b.generate("again", 2).is_ok());
}

#[test]
fn aspect_concurrent_calls_via_arc_mutex() {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let backend = Arc::new(Mutex::new(backend()));
    let mut handles = vec![];
    for i in 0..8 {
        let b = backend.clone();
        handles.push(thread::spawn(move || {
            let prompt = format!("hello world {i}");
            let mut guard = b.lock().unwrap();
            guard.generate(&prompt, 5).unwrap();
        }));
    }
    for h in handles { h.join().unwrap(); }
    assert_eq!(backend.lock().unwrap().call_count(), 8);
}

#[test]
fn config_max_tokens_capped_by_prompt_words() {
    let mut b = backend();
    let r = b.generate("two words", 100).unwrap();
    assert_eq!(r.tokens_emitted, 2);
}

#[test]
fn aspect_dyn_dispatch_through_trait_object() {
    let mut backend: Box<dyn LlmBackend> = Box::new(backend());
    assert_eq!(backend.name(), "mock");
    let r = backend.generate("trait dispatch", 4).unwrap();
    assert!(!r.text.is_empty());
}
