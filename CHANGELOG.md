# Changelog

All notable changes to wsp-kit. The format is "Added / Changed / Fixed /
Security" per release. No tag is made by the agent — the founder tags.

## [Unreleased]

### Changed
- **The eval gate pass bar is now +2 points (was +5).** An edit ships only if the failing metric improves by at least 2 points, with no more than 2 points regression on guards and neighbours, at serving temperature and greedy. The ±2 regression tolerance and the 1% weight cap are unchanged.
- **The 3B model cap is removed.** `wsp run`, `wsp merge`, and the
  weight loader no longer refuse models over 3 billion parameters.
  The eval gate (+2 primary, ≤2 regression on guards and neighbours)
  and the 1% weight cap still apply. Guidance:
  use small models for better performance.
- The refusal registry is five rows now, not six (the `over_3b`
  refusal is gone). Both READMEs render their refusal text from it.

## [0.3.0] - 2026-11-06

### Added
- Protocol v2 is the default for `run`, `watch`, `loop`, and `merge`
  (holdout splits, three samples at non-greedy temperatures, a
  confidence bound on the gate, causal placebos, the default guard
  bundle). See the kit README "Flags" table.
- `wsp/refusals.py`: a registry of the six hard refusals, with the
  truth as the code has it — two enforced in code, one partial, three
  SKILL.md only. Both READMEs render their refusal text from it.
- MCP tool `wsp_capabilities`: reports that the tools drive the
  built-in toy model only, and points at `wsp run` for real models.
- `install.sh --ref <tag>` / `WSP_REF` (clone at a tag or branch, and
  print the resolved commit) and `--dry-run` (print the ref, install
  nothing). SKILL.md is now installed from the same checkout as the
  kit, never a second download.
- `SECURITY.md` (threat model) and this CHANGELOG.

### Changed
- **The default protocol changed: `wsp run` now runs protocol v2 and
  only SHIPs on standard evidence. `--protocol v1` restores the old
  measurement (frozen, labelled QUICK, never evidence).**
- SKILL.md has YAML frontmatter (`name`, `description`) and a
  provenance note that separates the MIT background (GPT-6 Astra,
  blog post, not peer reviewed) from the editor of each edit (the
  kit's numerical estimator, no LLM). The report note and the merge
  note use the same wording for new reports.
- Star nudges go through one helper (`maybe_nudge`): only after a
  standard-evidence SHIP, only on a TTY, silenced by `WSP_NO_NUDGE=1`.
  The installer no longer asks for a star (no result exists yet);
  `wsp demo` keeps its star line (toy pipeline) and the status-page
  star button stays.
- Status page and `wsp status` use verdict wording: SHIP is "Approved
  (standard evidence)", QUICK_PASS is "Quick check passed (not
  evidence)".
- Docs truthfulness sweep: MCP is labelled demo-only (toy model);
  "originals never overwritten" now says "on disk" and states that
  serve-model changes only the in-memory copy; REAL-RUNS.md describes
  the v1 causal rule (`abs()`) versus v2 (expected sign plus
  placebos) accurately, and both passing runs are labelled quick
  evidence (3 failing prompts, not reproducible from this repo yet).
- `mcp` extra pinned to `mcp>=2,<3`.

### Fixed
- The MCP import path is verified against mcp 2.3.0
  (`mcp.server.mcpserver.MCPServer` is correct; the 1.x name
  `mcp.server.fastmcp.FastMCP` is kept only as a fallback).
- REAL-RUNS.md, README.md, and the site table: the SmolLM2 causal
  proof (−30.00) passed the v1 bar (`|−30| >= 5`) even though the
  ablation made the failure worse — the old "FAILED" label
  contradicted the code. The +29.44 scorer caveat now sits in the
  table rows, not only in the section text.
- DEMO-PLAN.md status lines now match REAL-RUNS.md (7 runs, 2 passes,
  3 models); the README roadmap names all three models.

### Security
- Refusal claims now match the code (registry): safety-removal is
  detected by the v2 guard bundle after an edit, not prevented —
  stated in the docs instead of implied by the old "enforced in
  code" list.
- No new network surface in this release; the local servers keep the
  Phase 1 Host / Origin / Content-Type / size / clamp checks.

## [0.2.0] - 2026-11-06

### Added
- Protocol v2 (evaluation integrity): splits, confidence-bound gate,
  causal placebos, default guard bundle, cheap-fix parity, v2
  scorers, dry-run cost plan. `wsp auto` stays on v1 (QUICK).

### Fixed
- Live path and file safety (Phase 1): atomic writes, recipe
  validation + provenance, safe output dirs, pre-load 3B size check,
  rollback that sticks, relative `run_dir`, replace-not-stack,
  local-server hardening, merge honesty, `cmd_apply_edit` rewrite.
