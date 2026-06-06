<!--
SPDX-License-Identifier: MPL-2.0
SPDX-FileCopyrightText: 2026 Jonathan D.A. Jewell (hyperpolymath)
-->

# Changelog

All notable changes to `neurophone` will be documented in this file.

This file is generated from conventional commits by the
[`changelog-reusable.yml`](https://github.com/hyperpolymath/standards/blob/main/.github/workflows/changelog-reusable.yml)
workflow (`hyperpolymath/standards#206`). Adopt the workflow in this repo's CI to keep this file in sync automatically — see
[`templates/cliff.toml`](https://github.com/hyperpolymath/standards/blob/main/templates/cliff.toml)
for the canonical config.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- fix(deps): pin rand 0.9 / rand_distr 0.5 to match ndarray-rand 0.16 — un-break workspace build (#129)
- fix(esn): scale by true spectral radius via power iteration, not infinity-norm (obligation 1.1) (#101)
- fix(claude-client): cap backoff at 60s with saturating_pow (obligation 3.2) (#106)
- fix(license): revert PMPL → MPL-2.0 sweep + sweep residue (#102, #118)
- fix(ci): carve out android/** from banned-language scan pending gossamer migration (#105)
- fix(ci): bump a2ml/k9-validate-action pins to canonical (#62)
- fix(ci): sync hypatia-scan.yml to canonical (#61)
- fix(ci): build Hypatia escript from repo root (estate dogfood drift)
- fix(ci): Phase-2 fleet submission must not fail the security gate (#58)

### Added

- proof: discharge proof-obligation map Tiers 0–3 (obligations 0.1–3.2) (#84 via #106)
- feat(core): typestate lifecycle refactor — NeuroSymbolicSystem<phase::Created|Active|Down> (#106)
- feat(android): port NativeLib JNI surface (11 methods) to Rust/Spark — crates/neurophone-android (PR #130 draft)
- docs: comprehensive README/EXPLAINME/6a2/contractiles documentation pass to Gold standard
- chore: dependabot ignore rules for rand/rand_distr (prevent re-bump of ndarray-rand-incompatible versions)
- chore: session HANDOVER.adoc for post-compaction continuity (PR #125)

### Android Migration (epic #83, RFC #97)

- chore: decompose Kotlin→gossamer migration into sub-issues #108–#115
- chore: NativeLib JNI surface ported to Rust/Spark (PR #130 draft)
- chore: gossamer scaffold PRs #121, #126 (draft — pick one; blocked on gossamer-rs licence)
- chore: gossamer service/bootreceiver/widgets/ui/delete-legacy PRs (#122–#124, #127–#128 draft)

### CI

- ci(language-policy): DRY local Java/Kotlin check via estate reusable (#117)
- ci(codeql): cron weekly→monthly (#116)
- ci(rust): convert rust-ci.yml to thin wrapper (standards#174) (#68)
- ci: redistribute concurrency-cancel guard to read-only check workflows (#64)
- ci: fix remaining external-action failures (a2ml, hypatia, fuzz) (#60)

## Pre-history

Prior commits to this file's introduction are recorded in git history but not formally classified into Keep-a-Changelog sections. To backfill, run `git cliff -o CHANGELOG.md` locally using the canonical [`cliff.toml`](https://github.com/hyperpolymath/standards/blob/main/templates/cliff.toml) — this is one-shot mechanical work.

---

<!-- This file was seeded by the 2026-05-26 estate tech-debt audit follow-up (Row-2 Phase 3); see [`hyperpolymath/standards/docs/audits/2026-05-26-estate-documentation-debt.md`](https://github.com/hyperpolymath/standards/blob/main/docs/audits/2026-05-26-estate-documentation-debt.md). -->
