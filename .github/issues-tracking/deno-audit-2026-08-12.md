# Deno Usage Audit - 2026-08-12

**Status**: Completed  
**Policy Reference**: JS-RUNTIME-POLICY.adoc (Bun > Deno > pnpm > npm)  
**Grandfather Clause**: Existing Deno projects are grandfathered and need not migrate per LANGUAGE-POLICY.adoc §1  

## Executive Summary

As of 2026-08-12, the estate has **34 top-level repositories** with root-level `deno.json` files (and 89 with `deno.lock`), making Deno the second-most used JS/TS runtime after the policy shift. The estate-wide policy has been updated to prioritize **Bun > Deno > pnpm > npm**, but existing Deno projects are explicitly grandfathered and **do not require migration**.

Migration from Deno to Bun is **optional**, not mandatory. This audit documents the current state for future reference and potential opportunistic migration where Bun provides clear advantages.

---

## 1. Priority Order Update - COMPLETED

### Files Updated

1. **rhodium-standard-repositories/spec/LANGUAGE-POLICY.adoc** (hyper-repos + worktrees)
   - Changed JavaScript/Node Ecosystem Policy from "Deno first, always" to:
     - 1. Bun first (default for all new work)
     - 2. Deno second (grandfathered, no migration required)
     - 3. pnpm (Node-runtime fallback)
     - 4. npm (last resort)
   - Removed Bun from banned languages table
   - Updated rationale to reflect current assessment

2. **ai-instruction/opus.md** (hyper-repos + worktrees + .claude/worktrees)
   - Updated architectural defaults from "Deno first" to "Bun first then Deno then pnpm then npm"

3. **ai-instruction/sonnet.md** (hyper-repos + worktrees + .claude/worktrees)
   - Updated architectural defaults from "Deno first" to "Bun first then Deno then pnpm then npm"

4. **rhodium-standard-repositories/satellites/cccp/README.adoc** (hyper-repos + worktrees)
   - Updated Post-JavaScript Liberation section to reflect Bun as first-choice, Deno as second

5. **JS-RUNTIME-POLICY.adoc** (already correct - no changes needed)
   - Already had correct priority order as of 2026-05-31

---

## 2. Current Deno Usage Inventory

### 2.1 Root-level deno.json Files (34 repos)

**hyper-repos (31):**
- aggregate-library
- blocky-writer (has deno.lock)
- bofig (has deno.lock)
- candy-crash (has deno.lock)
- developer-ecosystem
- dicti0nary-attack (has deno.lock)
- dotmatrix-fileprinter (has deno.lock)
- double-track-browser (has deno.lock)
- echidna (has deno.lock)
- empty-linter (has deno.lock)
- excel-economic-numbers-tool (has deno.lock)
- flat-mate (has deno.lock)
- hypatia
- infrastructure-automation (has deno.lock)
- ipfs-overlay (has deno.lock)
- nesy-solver (has deno.lock)
- palimpsest-license
- panll (has deno.lock)
- polyglot-i18n (has deno.lock)
- polystack
- preference-injector (has deno.lock)
- raze-tui (has deno.lock)
- rpa-elysium (has deno.lock)
- rrecord-verity (has deno.lock)
- safe-brute-force
- session-sentinel (has deno.lock)
- standards
- tma-mark2
- tree-sitter-a2ml (has deno.lock)
- tree-sitter-k9 (has deno.lock)
- ubicity (has deno.lock)

**meta-repos (3):**
- gossamer (has deno.lock)
- rokur (has deno.lock)
- svalinn (has deno.lock)

### 2.2 Subdirectory deno.json Files

Beyond root-level, approximately **245 additional deno.json files** exist in subdirectories across ~170+ repos. Notable concentrations:

- **_DATABASE _SET/lithoglyph/** - Multiple test directories (e2e, fuzz, integration, property, etc.)
- **_DATABASE _SET/nextgen-databases/** - Similar pattern
- **developer-ecosystem/rescript-ecosystem/** - Bootstrap shims and packages
- **standards/** - Various satellites and examples
- **_STATIC_SITE_GEN _SET/ssg-collection/** - Generated static site content (EXCLUDED from migration consideration)

### 2.3 deno.lock Files (89 total)

- 34 at root level (matching deno.json locations)
- 55 in subdirectories
- Indicates active Deno dependency usage

---

## 3. Migration Assessment

### 3.1 Migration Classes (per JS-RUNTIME-POLICY.adoc)

| Class | Description | Action Required | Count (Estimate) |
|-------|-------------|-----------------|------------------|
| A | Pure-Deno port | None (already Deno) | 34 root-level |
| B | npm wrapper via Deno | None (already using Deno with npm: specifiers) | ~10 |
| C | Carve-out | None (exempted by policy) | ~20 |

**Total: ~64 repos with explicit Deno usage**

### 3.2 Bun vs Deno Compatibility Analysis

**High Compatibility (Easy Migration):**
- TypeScript files (.ts) - Bun has native TS support
- Pure ESM modules
- Simple CLI tools
- Test runners using standard APIs

**Medium Compatibility (Requires Adaptation):**
- Deno-specific APIs (Deno.readFile, Deno.test, etc.) - Bun has equivalents
- Deno permission model - Bun has different but compatible approach
- deno.json tasks - Need conversion to package.json scripts or bunfig.toml

**Low Compatibility (Significant Effort):**
- Deno FFI (foreign function interface) - Bun has limited FFI support
- Deno web workers - Different API surface
- Deno-specific std library usage - Need replacement with Bun/community alternatives

### 3.3 Estimated Migration Effort

| Complexity | Repos | Effort per Repo | Total Effort |
|------------|-------|----------------|--------------|
| Simple (TS only, no Deno APIs) | ~25 | 1-2 hours | 25-50 hours |
| Medium (Some Deno APIs) | ~8 | 4-8 hours | 32-64 hours |
| Complex (Heavy Deno-specific features) | ~1 | 16-24 hours | 16-24 hours |
| **Total** | **~34** | | **73-138 hours** |

**Note**: These are rough estimates. Actual effort varies based on:
- Test suite complexity
- CI/CD pipeline adaptations
- Dependency compatibility
- Team familiarity with Bun

### 3.4 Dependencies on Deno

Key repos that depend on Deno and would need assessment:

1. **standards/0-ai-gatekeeper-protocol/mcp-repo-guardian** - MCP server using Deno
2. **standards/axel-protocol** - Protocol implementation
3. **standards/k9-svc/bindings/deno** - Deno bindings for k9
4. **developer-ecosystem/affinescript-ecosystem/affinescript-deno-test** - AffineScript test harness
5. **echidna** - Prover integration
6. **hypatia** - Neuro-symbolic scanning
7. **flat-mate** - Frontend application

---

## 4. Recommendations

### 4.1 Immediate Actions (Priority: High)

1. **✅ COMPLETED**: Update all policy documents to reflect Bun > Deno > pnpm > npm priority
2. **✅ COMPLETED**: Ensure grandfather clause is explicit in all policy files
3. **✅ COMPLETED**: Update AI guidance to prevent new Deno-first recommendations

### 4.2 Opportunistic Migration (Priority: Low)

Given the grandfather clause, **no forced migration is required**. However, for repos where:
- Bun offers clear performance advantages
- Team is already working in the repo
- Migration aligns with other planned work

**Recommended candidates for opportunistic migration:**

1. **New projects** - Should use Bun by default (policy)
2. **simple CLI tools** - Low effort, high benefit (e.g., empty-linter, preference-injector)
3. **TypeScript-heavy repos** - Bun's native TS support is excellent

### 4.3 Blockers to Bun Adoption

1. **Deno FFI Usage**: Repos using Deno's FFI for native bindings cannot easily migrate
   - Example: standards/a2ml/bindings/deno
   - Mitigation: Keep on Deno, or rewrite bindings for Bun

2. **Deno-specific Ecosystem Dependencies**: 
   - Some JSR packages may not be available for Bun
   - Mitigation: Use npm: specifiers or find Bun-compatible alternatives

3. **Team Familiarity**: 
   - Deno has been the primary runtime for ~2 years
   - Migration requires learning curve
   - Mitigation: Documentation, examples, pairwise migration

### 4.4 Monitoring

Track these metrics quarterly:
- Number of new repos choosing Bun vs Deno
- Issues reported with Bun compatibility
- Performance comparisons for migrated repos
- Community adoption of Bun in the broader ecosystem

---

## 5. Related Issues

| Issue | Title | Status | Relationship |
|-------|-------|--------|--------------|
| standards#67 | npm-avoidant: package-lock.json must never be tracked | Active | Policy enforcement |
| standards#68 | .editorconfig/.claude must not be tracked | Active | Policy enforcement |
| standards#252 | ReScript → AffineScript migration | Active | Parallel migration |
| standards#253 | npm → Deno migration | Substantially Complete | Previous migration |
| standards#254 | JavaScript → AffineScript migration | Active | Parallel migration |

**NEW ISSUE NEEDED**: Deno → Bun migration tracking (optional/opportunistic)

---

## 6. Files Modified

### standards repo (hyper-repos)
- rhodium-standard-repositories/spec/LANGUAGE-POLICY.adoc
- ai-instruction/opus.md
- ai-instruction/sonnet.md
- rhodium-standard-repositories/satellites/cccp/README.adoc

### worktrees/standards-debtfile
- rhodium-standard-repositories/spec/LANGUAGE-POLICY.adoc
- ai-instruction/opus.md
- ai-instruction/sonnet.md
- rhodium-standard-repositories/satellites/cccp/README.adoc

### .claude/worktrees (hyper-repos/standards)
- nix-references-clause/rhodium-standard-repositories/spec/LANGUAGE-POLICY.adoc
- nix-references-clause/ai-instruction/opus.md
- nix-references-clause/ai-instruction/sonnet.md
- a2ml-design/rhodium-standard-repositories/spec/LANGUAGE-POLICY.adoc
- a2ml-design/ai-instruction/opus.md
- a2ml-design/ai-instruction/sonnet.md

---

## 7. Verification

To verify the changes:

```bash
# Check that no "Deno first" remains in policy files
grep -r "Deno first" hyper-repos/standards/ worktrees/standards-debtfile/ 
# Should return no results (or only in historical/commented sections)

# Verify priority order
grep -A3 "JavaScript.*Ecosystem.*Policy" hyper-repos/standards/rhodium-standard-repositories/spec/LANGUAGE-POLICY.adoc
# Should show: 1. Bun, 2. Deno, 3. pnpm, 4. npm

# Count deno.json files
find hyper-repos meta-repos -maxdepth 2 -name "deno.json" | wc -l
# Should return ~34
```

---

## 8. Conclusion

The estate has successfully updated its priority order to **Bun > Deno > pnpm > npm** across all policy documents. Existing Deno projects (34 root-level, ~200+ total) are grandfathered and require no migration. 

**Migration from Deno to Bun is optional and should be done opportunistically**, not as a campaign. The estimated effort for full migration is 73-138 hours, but this is not required by policy.

The primary value of this audit is:
1. Policy consistency across all documentation
2. Clear guidance for new projects (use Bun)
3. Baseline inventory for future opportunistic migrations
4. Identification of repos that cannot easily migrate (FFI users)

**Recommendation**: Create a low-priority tracking issue for opportunistic Deno→Bun migrations, but do not launch a formal campaign. Let natural repo maintenance drive the transition where it makes sense.
