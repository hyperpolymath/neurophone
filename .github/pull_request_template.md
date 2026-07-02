<!--
SPDX-License-Identifier: CC-BY-SA-4.0
Copyright (c) Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>
-->
## Summary

<!-- Briefly describe what this PR does and why. Link to related issues with "Closes #N". -->

## Changes

<!-- List the key changes introduced by this PR. -->

-

## RSR Quality Checklist

<!-- Check all that apply. PRs that fail required checks will not be merged. -->

### Required

- [ ] Tests pass (`just test` / `cargo test`)
- [ ] Code is formatted (`just fmt` / `cargo fmt --check`)
- [ ] Linter is clean (`cargo clippy --all-targets -- -D warnings`)
- [ ] No banned language patterns (no TypeScript, no npm/bun, no Go/Python)
- [ ] No `unsafe` blocks without `// SAFETY:` comments (`esn`/`lsm` use `deny`, all others `forbid`)
- [ ] No proof escape hatches (`believe_me`, `unsafeCoerce`, `Obj.magic`, `Admitted`, `sorry`, `assert_total`)
- [ ] SPDX license headers present on all new/modified source files
- [ ] No secrets, credentials, or `.env` files included

### As Applicable

- [ ] `.machine_readable/6a2/STATE.a2ml` updated (if project state changed)
- [ ] `.machine_readable/6a2/ECOSYSTEM.a2ml` updated (if integrations changed)
- [ ] `.machine_readable/6a2/META.a2ml` updated (if architectural decisions changed)
- [ ] `proofs/README.adoc` obligation table updated (if a proof obligation changed state)
- [ ] Documentation updated for user-facing changes
- [ ] `TOPOLOGY.adoc` updated (if architecture changed)
- [ ] `CHANGELOG` or release notes updated
- [ ] New dependencies reviewed for license compatibility (MPL-2.0)
- [ ] JNI surface changes validated (`crates/neurophone-android` host-testable + lifecycle typestate preserved)

## Testing

<!-- Describe how you tested these changes. Record the actual command output, not a claim. -->

## Screenshots

<!-- If applicable, add screenshots or terminal output demonstrating the change. -->
