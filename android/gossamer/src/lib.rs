// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>

//! NeuroPhone Android — Gossamer backend (scaffold).
//!
//! Epic #83, RFC PR #97, sub-issue #109, sub-PR #3. Scaffolding ONLY.
//!
//! This crate is the eventual entry point for the Gossamer webview shell on
//! Android, replacing the legacy Kotlin Activity/Service/Receiver/widget stack.
//! It is currently inert: the `gossamer-rs` dependency is commented out in
//! `Cargo.toml` because it is not yet resolvable in a neurophone checkout
//! (`gossamer-rs` is unpublished; the estate consumes it as a path dependency
//! from the sibling `hyperpolymath/gossamer` repo — see `Cargo.toml`).
//!
//! TODO(#83 sub-PR #8): add `use gossamer_rs::App;`, build the webview app, and
//! register IPC command handlers.
//! TODO(#83 sub-PR #4): dispatch the 11 `NativeLib` JNI methods into IPC
//! handlers that call the Rust core in `crates/neurophone-core`.

/// Placeholder so the standalone scaffold crate compiles on its own.
pub fn scaffold_marker() -> &'static str {
    "neurophone-gossamer: scaffold (sub-PR #3)"
}

#[cfg(test)]
mod tests {
    #[test]
    fn marker_is_present() {
        assert!(super::scaffold_marker().contains("scaffold"));
    }
}
