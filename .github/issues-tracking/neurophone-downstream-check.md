# GitHub Issue: Verify neurophone downstream compatibility

**Repository**: hyperpolymath/neurophone  
**Title**: Verify downstream (burble) compatibility after neurophone changes  
**Labels**: `maintenance`, `compatibility`, `downstream`  
**Priority**: Low  
**Status**: TODO  

---

## Description

After resolving neurophone build issues and deleting stale branches, we need to verify that downstream dependencies are not affected.

**Changes Made to neurophone**:
- Deleted 7 stale branches
- Fixed dependency versions (rand, rand_distr, toml)
- Fixed import issues (KeyInit, ActionGate)
- No breaking changes to public API

**Known Downstream**:
- `meta-repos/burble` - Has experimental `neurophone_bridge` (ADR-0015)

## Verification Tasks

- [ ] Check burble's `server/lib/burble/experimental/neurophone_bridge.ex` still compiles
- [ ] Verify burble's test suite still passes
- [ ] Confirm neurophone_bridge compatibility with updated neurophone

## Assessment

**Expected Impact**: **MINIMAL/ZERO**

Rationale:
1. All changes are internal to neurophone
2. No public API changes
3. neurophone_bridge is experimental only (ADR-0015)
4. Dependency fixes are internal implementation details
5. Branch cleanup doesn't affect runtime

From burble documentation:
> The frozen presence-beacon wire spec. `build.rs` parses its `[presence-frame]` offsets/sizes into `wire.rs` constants and *fails the build loudly* if `[metadata].wire-version != 1` — so a v2 re-vendor cannot silently miscompile a v1 decoder.

The neurophone changes do not affect the wire protocol (v1 remains frozen).

## Verification Commands

```bash
# In burble repo:
cd meta-repos/burble
cargo check --package burble  # Verify main burble compiles
# Check experimental bridge:
grep -r "neurophone_bridge" server/lib/burble/experimental/
```

## Related Work

- Audit: `hyper-repos/_HARDWARE _SET/neurophone/NEUROPHONE-BRANCH-AUDIT-2026-08-12.md`
- Tracking: `dev-notes/ISSUE-TRACKING-2026-08-12.md`

## Notes

This is low priority. The changes are internal and non-breaking. Verification is recommended but not urgent.
