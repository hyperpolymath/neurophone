# GitHub Issues - Created 2026-08-12

This directory contains GitHub issue templates that need to be opened on GitHub to track the remaining verification work.

---

## Issue Summary

Following the comprehensive estate cleanup work, **3 GitHub issues** need to be opened to track remaining verification tasks that require web access.

### Overall Progress: 95% Complete (20/21 tasks)

All local work is complete. The remaining tasks require GitHub web access for verification.

---

## Issues to Open

### 1. 🔴 **HIGH PRIORITY**

**File**: [laniakea-verify-pr-status.md](./laniakea-verify-pr-status.md)  
**Repository**: hyperpolymath/laniakea  
**Title**: Verify PR #58, #50, #59 status after branch deletions  
**Priority**: High  

**Why**: PR #59 should auto-close after branch deletion. PRs #58 and #50 should show as merged. Need to verify all are properly resolved.

---

### 2. 🟡 **MEDIUM PRIORITY**

**File**: [neurophone-verify-github-status.md](./neurophone-verify-github-status.md)  
**Repository**: hyperpolymath/neurophone  
**Title**: Verify GitHub issue/PR status after branch cleanup and build fixes  
**Priority**: Medium  

**Why**: Need to verify no neurophone issues are blocked by deleted branches and CI/CD is smooth.

---

### 3. 🟢 **LOW PRIORITY**

**File**: [neurophone-downstream-check.md](./neurophone-downstream-check.md)  
**Repository**: hyperpolymath/neurophone  
**Title**: Verify downstream (burble) compatibility after neurophone changes  
**Priority**: Low  

**Why**: Verify burble's experimental neurophone_bridge still works. Expected impact: minimal/zero.

---

## Quick Reference

| # | Repo | Title | Priority | File |
|---|------|-------|----------|------|
| 1 | laniakea | Verify PR #58, #50, #59 status | High | [link](./laniakea-verify-pr-status.md) |
| 2 | neurophone | Verify issue/PR status | Medium | [link](./neurophone-verify-github-status.md) |
| 3 | neurophone | Verify downstream compatibility | Low | [link](./neurophone-downstream-check.md) |

---

## What's Already Done ✅

All actionable work is complete:

1. ✅ Standards: Priority order updated (Bun > Deno > pnpm > npm)
2. ✅ Deno audit complete (34 repos identified)
3. ✅ Laniakea: 4 stale branches deleted
4. ✅ Neurophone: 7 stale branches deleted
5. ✅ Neurophone: All build issues resolved
6. ✅ Neurophone: Commit pushed (70a6550)

---

## How to Use

Copy the content of each `.md` file in this directory and create corresponding issues on GitHub. The files contain:
- Issue title
- Labels
- Priority
- Description
- Acceptance criteria
- Related work references

---

## Tracking

See `../ISSUE-TRACKING-2026-08-12.md` for complete progress tracking.
