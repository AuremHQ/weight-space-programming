# Weight-Space Programming

![wsp-kit — real runs](assets/wsp-kit-banner-v0.2.0.png)

> **wsp-kit by Aurem** — an engineering tool for small models. Try to fix small AI models by editing their weights directly — no retraining, no fine-tuning, no gradients — with an eval gate deciding whether any edit ships.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research: MIT CSAIL](https://img.shields.io/badge/Research-MIT%20CSAIL-blue.svg)](RESEARCH.md)
[![GitHub Stars](https://img.shields.io/github/stars/AuremHQ/weight-space-programming?style=social)](https://github.com/AuremHQ/weight-space-programming)
![Status: Real runs live](https://img.shields.io/badge/Status-Real%20runs%20live-green.svg)

> From the team building **Ora by Aurem — https://auremcto.com** — an AI engineer for non-technical founders. ORA itself stays private; this skill is public so anyone can use it on their own models.

> ⭐ **Star this repo** — the real runs are live below, with honest numbers.
> 🔁 **Share it** with anyone running a small open model with one stubborn failure.

---

## The 30-second version

- **MIT CSAIL showed** a frontier model can read and write neural network weights directly — no training, no gradients.
- **One tiny edit** (0.016% of weights) fixed a model's reasoning loop: MATH **7.9% → 48.1%**.
- **This repo turns that idea into a skill** you can run on your own small open-weight model — with evaluation, not enthusiasm, deciding whether any edit ships.

**This repo is not affiliated with MIT.** It is an independent implementation inspired by their public research, with full credit below.

---

## Real runs — our numbers (not MIT's)

We ran the kit on **real models** — Qwen3-0.6B and GPT2-124M, local CPU, no LLM editor. Here is what actually happened. Full numbers: [REAL-RUNS.md](REAL-RUNS.md).

| Run | Model | Failure | Diagnosis | Causal proof | Weights changed | Gate |
|---|---|---|---|---|---|---|
| **Pipeline test** | Kit weights (smoke test) | Failing 64.14 vs good 75.98 | Row 0 | **+18.00 — PASSED** | **128 (0.77%)** | **PASSED — failing 64.14 → 75.43 (+11.29)** |
| **Qwen run 1** | Qwen3-0.6B | 60-item lists loop (60 animals → only 16 distinct) | Layers [27, 26] | −3.33 — FAILED | **0** | **FAILED — nothing shipped** |
| **Qwen run 2** | Qwen3-0.6B | Same failure | Layers [27, 26] | **+32.22 — PASSED** | **0** | **FAILED — nothing shipped** (attempt numbers contaminated by a restore bug — see clean re-run) |
| **Qwen clean re-run** | Qwen3-0.6B | Same failure, restore bug fixed | Layers [27, 26] | **+32.22 — PASSED** | **0** | **FAILED — nothing shipped** (all 3 clean attempts made failing worse: −11.11 / −11.11 / −12.22) |
| **GPT-2 run** | GPT2-124M | 60-item lists loop (animals 1.7, countries 5.0, fruits 1.7) | Layers [10, 11] | −1.11 — FAILED | **0** | **FAILED — nothing shipped** |

**What this proves:** the kit runs end-to-end on real models, finds real failures, diagnoses real layers — and **the gate works**: when the proof fails, nothing ships.
**What it does not prove yet:** a passing weight edit on a real model. That is the next milestone, not a claim.

---

## Where it started — MIT's numbers

```
MATH-500           7.9%  ████░░░░░░░░░░░░░░░░  →  48.1% ████████████████████
GSM8K             13.8%  ██████░░░░░░░░░░░░░░  →  67.1% ██████████████████████████
IFEval            21.0%  ████████░░░░░░░░░░░░  →  62.3% █████████████████████████
Answers hitting the token limit    93%  →  33%
Weights changed: 131,072 — just 0.016% of the model
```

How: the model (Qwen3.5-0.8B) kept getting stuck repeating itself until it ran out of tokens. The researchers traced the loop to one direction written by the MLPs of layers 15 and 19, and subtracted it. The model didn't learn new math — **it already knew the math; the loop was hiding it.**

Full details, all 8 experiments, and the honest limits: see [RESEARCH.md](RESEARCH.md).

---

## Founder's vision

> **Big AI should not belong only to people who can pay for big models and big retraining bills.**
> Small, local, cheap models are already in everyone's hands — they just need to perform at their best. This kit exists to make a small local model give **better results**, one tiny proven edit at a time, without retraining and without sending your data anywhere. If the numbers don't improve, nothing ships. That honesty is the product.
>
> **This kit is free for everyone, and it stays free.** A small model should not mean bad results — everyone has the right to the best experience and the best results. The team's vision is simple: everyone should benefit from this kit.

**— Tejinder Sandhu**, founder, Ora by Aurem ([auremcto.com](https://auremcto.com)) · [Tejinder Sandhu on LinkedIn](https://www.linkedin.com/in/tejinder-sandhu-156a3a181)

---

## How the skill works

```mermaid
flowchart LR
    A[Detect: find weights + failure] --> B[Diagnose: find the direction]
    B --> C[Fix: subtract it — under 1% of weights]
    C --> D{Eval gate: same prompts, same settings}
    D -->|Improved +5, no guard regression| E[Ship as reviewable change]
    D -->|Not improved| F[Discard — ship nothing]
```

1. **Detect** — Find the weights in your repo, run your model on your own test prompts, diagnose the failure, and measure it first (how often, how bad) — that baseline is what the eval gate compares against. No weights, no test prompts, or a model that's too large → the skill refuses, clearly, and does nothing.
2. **Fix** — Subtract the offending direction from the identified layers only. Edits are tiny by design: anything over 1% of weights is rejected as "too broad."
3. **Eval-gate** — Re-run the *same* prompts with the *same* settings. The fix ships only if the primary metric improves by at least 5 points with no guard-metric regression worse than 2 points. Otherwise the edit is discarded. **No gate, no ship.**

Full rules, refusals, and safety gates: see [SKILL.md](SKILL.md).

## How this differs from gradient descent

Gradient descent trains: data + loss → gradients → many small updates → a retrained model. This skill is a different path: one direct edit, no data, no gradients, no training loop — and the eval gate decides whether it ships.

```mermaid
flowchart LR
    G[Data + loss] --> H[Gradients] --> I[Many small updates] --> J[Retrained model]
    K[Diagnosed direction] --> L[One subtraction] --> M[Re-test + eval gate] --> N[Reviewable fix — or nothing]
```

If you want training, use training. If you want to try removing one stubborn failure without retraining, use this.

---

## What's inside

- **[RESEARCH.md](RESEARCH.md)** — the MIT research, explained simply: what weight-space programming is, all 8 experiments with numbers, and where it fails.
- **[SKILL.md](SKILL.md)** — the Weight-Space Programming skill: the 3-step workflow, hard refusal rules, and safety gates.
- **[REAL-RUNS.md](REAL-RUNS.md)** — our real-model runs (Qwen + GPT-2) with full numbers, honest gate verdicts, and what they prove.
- **[DEMO-PLAN.md](DEMO-PLAN.md)** — how the real runs were built, and the exact bar the next run must clear.
- **[kit/](kit/)** — the runnable kit: engine, CLI, MCP server, task pack, and the universal real-model pipeline (`wsp run`).

---

## Run it

The skill is runnable as `wsp-kit`. From a clone:

```
pip install './kit[real]'
wsp demo
```

`wsp demo` runs the quick pipeline check end to end and writes `report.json`. For a real model:

```
wsp run --model Qwen/Qwen3-0.6B --eval kit/evals/loop-fix-qwen3-0.6b.json --out-dir out
```

That runs the full gated pipeline — temperature sweep, baselines, diagnosis, causal proof, up to 3 edit attempts — and writes `report.json`. If the gate fails, the out-dir holds only the report: **nothing ships**.

For agents and harnesses, the same pipeline is an MCP server (requires `pip install 'wsp-kit[mcp]'` first):

```json
{
  "mcpServers": {
    "wsp-kit": {
      "command": "python",
      "args": ["-m", "wsp.mcp_server"]
    }
  }
}
```

---

## Who this is for

| ✅ This is for you if… | ❌ This is NOT for you if… |
|---|---|
| You have a **small open-weight model** (Qwen, Gemma, Llama class) in your own app or software | Your model is served by an API (ChatGPT, Claude…) — its weights are never in your hands |
| It has **one specific, repeatable failure** — looping, a stuck behaviour, a narrow quirk | You want to "make my model smarter" — editing removes obstacles, it does not add abilities |
| You have test prompts that show the failure | Your model is large — this is practical today only for small models |

---

## Try it on your model

1. Point the skill at your weights and your failing prompts.
2. Get a diagnosis and a tiny, reviewable edit.
3. Let the eval gate decide — if the numbers don't improve, nothing ships.

That's the whole deal. No retraining bill, no blind trust.

---

## Honest limits — why you can trust this repo

- The MIT work is a **blog post, not a peer-reviewed paper**, and has not yet been independently replicated.
- Demonstrated so far mainly with **one editor model (GPT-6 Astra)**. Whether other models do this equally well is **unproven** — our reports name the editor behind every run (ours is a numerical estimator, no LLM).
- This is a **batch job, not real-time**: load weights, run evaluations, reason, edit, re-evaluate. It never runs live inside a chat reply.
- **No real-model edit has passed the gate yet.** The pipeline test passed; Qwen run 1, Qwen run 2, and the GPT-2 run all ended with the gate refusing and nothing shipping. That is published above, not hidden.

---

## Roadmap

- [x] Research summary + skill spec
- [x] **Real runs live: Qwen3-0.6B + GPT2-124M, honest numbers, gate enforced** — see [REAL-RUNS.md](REAL-RUNS.md)
- [ ] First passing weight edit on a real model
- [ ] Demo video/GIF + example report
- [ ] First community test: your model, your failure, your numbers

---

## ⭐ Star & share

- **Star** = you follow the next run. GitHub notifies you when new numbers land here.
- **Share** = someone with a looping small model finds a fix instead of paying for a full retraining run.
- **Built by the team behind https://auremcto.com** — Ora by Aurem, an AI engineer for non-technical founders.

---

## Credits

- Research: *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"* — Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab, MIT CSAIL, 2 Nov 2026. Original blog: https://rikiyat.github.io/blog/weight-space-programming/ · Full experiment logs: https://github.com/RikiyaT/lm_understand-logs
- This repo's skill design and packaging: the team building [Ora by Aurem](https://auremcto.com), founded by **Tejinder Sandhu**. ORA itself stays private; this skill is public so anyone can use it on their own models.

## License

MIT — see [LICENSE](LICENSE). Use it in your own model, app, or software. If it fixes something for you, a star and a note about your numbers would mean a lot.
