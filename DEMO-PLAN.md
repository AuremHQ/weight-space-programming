# Demo Plan — How This Repo Earns Its Stars (Updated: Real Runs Live)

Stars do not come from a good README. They come from real runs with real numbers that someone else can reproduce. The real runs are now live — see [REAL-RUNS.md](REAL-RUNS.md). This file is the updated plan: what the runs did, what they proved, and the bar the next run must clear.

## The real runs (done)

**Qwen3-0.6B (run 1, run 2, and a clean re-run) and GPT2-124M, local CPU, numerical editor, no LLM.**

1. Picked small open-weight models (under 3B parameters) with a visible, repeatable failure — list-looping on 60-item lists.
2. Recorded the failure on fixed prompt sets with fixed sampling settings, with baselines.
3. Ran the Weight-Space Programming skill (SKILL.md): diagnosed the direction, tested causality, and — where the proof passed — attempted the edit.
4. Re-ran the identical prompt sets with identical settings. Published the before/after tables in [REAL-RUNS.md](REAL-RUNS.md).
5. No edit passed the gate, so **no edited weights ship**. The reports are the artifact.

## Acceptance criteria — status

- [x] Before/after numbers on the SAME prompts with the SAME settings — no cherry-picked prompts.
- [x] Edit touches under 1% of weights, with layers and counts reported. (No edit shipped — 0 weights changed, 0.77% on the pipeline test.)
- [x] At least one guard metric reported (good + neighbours sets, not just the headline metric).
- [x] Editor named in every report (numerical estimator, no LLM; results are editor-dependent).
- [ ] Edit script reproduces the edit from the original weights on a fresh run. — **Not yet: no real-model edit has passed the gate.** The pipeline test's edit is reproducible; a real-model passing edit is the next milestone.
- [x] A short video of the run report (GPT-2 report video, MIT-style before/after bars + gate verdict).
- [x] The honest-limits section of the README stays in place. No claim the runs do not support.

## What NOT to do

- Do not re-publish MIT's numbers as your own. Their Qwen result is theirs; our numbers are ours, in [REAL-RUNS.md](REAL-RUNS.md).
- Do not claim the runs prove this works on large models, closed models, or all models. They prove the pipeline runs and the gate works on two small models.
- Do not claim a real model was fixed. **None was.** The gate refused all four real runs.
- Do not include any private code, secrets, or customer data. This repo is standalone: research summary + skill + real runs only.

## Publishing checklist

- [x] Repo name: `weight-space-programming`
- [x] License: MIT (see LICENSE)
- [x] Topics/tags for discovery: `weight-space-programming`, `machine-learning`, `llm`, `open-source`, `model-editing`, `small-language-models`, `eval-gate`, `ai-engineering`, `python`, `mcp`
- [x] README status badge flipped from "Demo in progress" to "Real runs live"
- [ ] Launch post: GPT-2 report graphic + video + honest caption (numbers first, gate verdict included), linking the repo — not a thread about the skill in the abstract (graphic + video made; post not yet published — awaits Teji's OK)
- [x] Link back to https://auremcto.com (the team behind this skill builds Ora by Aurem; the skill is public, ORA itself stays private)

## Next

1. First passing weight edit on a real model — the milestone that changes the README's headline.
2. Frontier-LLM editor variant (MIT-style editor) — only if Teji approves it.
3. First community test: your model, your failure, your numbers.
