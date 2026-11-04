# Demo Plan — How This Repo Earns Its Stars

Stars do not come from a good README. They come from a demo with real numbers that someone else can reproduce. This file is the build plan for that demo, and the bar it must clear before the repo is promoted anywhere.

## The demo (one thing, done well)

**Reproduce the loop-fix on a small open model.**

1. Pick a small open-weight model (under 3B parameters) with a visible, repeatable failure — ideally a reasoning loop like Qwen's, or another single narrow failure.
2. Record the failure on a fixed prompt set with fixed sampling settings: how often it loops, how often it finishes, baseline scores.
3. Run the Weight-Space Programming skill (SKILL.md): diagnose the direction, subtract it from the identified layers, record layers touched and percentage of weights changed.
4. Re-run the identical prompt set with identical settings. Publish the before/after table.
5. Publish the edit script so anyone can reproduce the edit from the original weights, plus the edited weights as a separate artifact.

## Acceptance criteria (all must hold before promotion)

- [ ] Before/after numbers on the SAME prompts with the SAME settings — no cherry-picked prompts.
- [ ] Edit touches under 1% of weights, with layers and counts reported.
- [ ] At least one guard metric reported (something that must not get worse), not just the headline metric.
- [ ] Editor model named in the report (results are editor-dependent).
- [ ] Edit script reproduces the edit from the original weights on a fresh run.
- [ ] A short screen recording or GIF of the before/after behaviour (loops stopping is visible — show it).
- [ ] The honest-limits section of the README stays in place. No claim the demo does not support.

## What NOT to do

- Do not re-publish MIT's numbers as your own. Their Qwen result is theirs; the demo must produce its own numbers.
- Do not claim the demo proves this works on large models, closed models, or all models. It proves one thing on one small model.
- Do not promote the repo (posts, launch lists, social) before the demo criteria are met. A repo with a great README and no demo earns skepticism, not stars.
- Do not include any private code, secrets, or customer data. This repo is standalone: research summary + skill + demo only.

## Publishing checklist (when the demo is ready)

- [ ] Repo name: `weight-space-programming`
- [ ] License: MIT (see LICENSE)
- [ ] Topics/tags for discovery: `weight-space`, `llm`, `interpretability`, `model-editing`, `open-weights`, `ai-safety`, `machine-learning`
- [ ] README status badge flipped from "Demo in progress" to "Demo live"
- [ ] Launch post: one short post showing the before/after (video/GIF first, numbers second), linking the repo and the demo report — not a thread about the skill in the abstract
- [ ] Link back to https://auremcto.com (the team behind this skill builds Ora by Aurem; the skill is public, ORA itself stays private)

## Order of work

1. Scaffold + README (done — this repo).
2. Demo build (coding agent, per SKILL.md + this plan).
3. Demo numbers + video.
4. Publish + post. That is when the hype window pays.
