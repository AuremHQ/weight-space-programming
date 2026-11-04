# Weight-Space Doctor

> Fix small AI models by editing their weights directly — no retraining, no fine-tuning, no gradients.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research: MIT CSAIL](https://img.shields.io/badge/Research-MIT%20CSAIL-blue.svg)](RESEARCH.md)
![Status: Skill spec complete · Runnable demo in progress](https://img.shields.io/badge/Status-Demo%20in%20progress-orange.svg)

**The one-line idea:** a neural network's weights are numbers. If a small open model has one specific failure — like getting stuck repeating itself — you may not need to retrain it. You can find the direction in the weights that causes the failure, remove it, and verify the fix with before/after tests. This repo turns that idea, demonstrated by MIT CSAIL in November 2026, into a usable skill for your own models, apps, and software.

> From the team building **Ora by Aurem — https://auremcto.com** — an AI engineer for non-technical founders. ORA itself stays private; this skill is public so anyone can use it on their own models.

**This repo is not affiliated with MIT.** It is an independent implementation inspired by their public research, with full credit below.

---

## The MIT result that started this

MIT CSAIL's *"Sparks of Weight-Space Programming"* (2 Nov 2026, by Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab) showed that a frontier model, OpenAI's GPT-6 Astra, can natively read and write neural network weights — no training loops, no gradients.

Their headline experiment:

| | Before | After |
|---|---|---|
| MATH-500 (Qwen3.5-0.8B) | 7.9% | **48.1%** |
| GSM8K | 13.8% | **67.1%** |
| IFEval (instruction following) | 21.0% | **62.3%** |
| Answers hitting the token limit | 93% | **33%** |
| Weights changed | — | **131,072 (0.016% of the model)** |

How: Qwen kept getting stuck in reasoning loops, repeating itself until it ran out of tokens. Astra traced the loop to one direction written by the MLPs of layers 15 and 19, and subtracted that direction from 64 output columns in each layer. The model didn't learn new math — it already knew the math; the loop was hiding it.

Full details, all 8 experiments, and the honest limits: see [RESEARCH.md](RESEARCH.md).

---

## What this repo gives you

- **[RESEARCH.md](RESEARCH.md)** — the MIT research, explained simply: what weight-space programming is, all 8 experiments with numbers, and where it fails.
- **[SKILL.md](SKILL.md)** — the Weight-Space Doctor skill: a 3-step workflow (Detect → Fix → Eval-gate) you can run on your own open-weight model, with hard refusal rules and safety gates.
- **[DEMO-PLAN.md](DEMO-PLAN.md)** — how the runnable demo is being built, and the exact numbers it must show before anyone should trust it.

## Who this is for

You, if you have a **small open-weight model** (Qwen, Gemma, Llama class, a few billion parameters at most) inside your own app or software, and it has **one specific, repeatable failure** — looping, a stuck behaviour, a narrow quirk.

## Who this is NOT for

- **Closed API models** (ChatGPT, Claude, etc. served over an API). Their weights are never in your hands, so this cannot touch them. Any tool claiming otherwise is lying.
- **"Make my model smarter" requests.** Editing removes an obstacle hiding an ability the model already has. It does not add abilities the model lacks — MIT's own control experiment on Gemma 3 270M proved exactly that: no improvement.
- **Large models.** This is practical today only for small models. Anything above a few billion parameters is out of scope for this skill.

## How the skill works (3 steps)

1. **Detect** — Find the weights in your repo, run your model on your own test prompts, and diagnose the failure. No weights, no test prompts, or a model that's too large → the skill refuses, clearly, and does nothing.
2. **Fix** — Subtract the offending direction from the identified layers only. Edits are tiny by design: if a fix touches more than 1% of weights, it is rejected as "too broad."
3. **Eval-gate** — Re-run the *same* prompts with the *same* settings. The fix ships only if the primary metric improves by at least 5 points with no guard-metric regression worse than 2 points. Otherwise the edit is discarded and no change is proposed. No gate, no ship.

Read the full rules, refusals, and safety gates in [SKILL.md](SKILL.md).

## Honest limits (read before you trust this)

- Demonstrated so far mainly with **one model (GPT-6 Astra)**. Whether other models can do this equally well is **unproven** — this repo's demo will say which model produced each edit.
- The MIT work is a **blog post, not a peer-reviewed paper**, and has not yet been independently replicated.
- This is a **batch job, not real-time**: load weights, run evaluations, reason, edit, re-evaluate. It never runs live inside a chat reply.
- A failed evaluation scan does not mean confirmed vulnerabilities, and a weight edit can bypass safety behaviour — which is why this skill refuses safety-removal requests and gates every edit behind evaluation.

## Roadmap

- [x] Research summary + skill spec
- [ ] Runnable demo: loop-fix on a small open model, with reproducible before/after numbers
- [ ] Demo video/GIF + example report
- [ ] First community test: your model, your failure, your numbers

## Credits

- Research: *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"* — Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab, MIT CSAIL, 2 Nov 2026. Original blog: https://rikiyat.github.io/blog/weight-space-programming/ · Full experiment logs: https://github.com/RikiyaT/lm_understand-logs
- This repo's skill design and packaging: the team building [Ora by Aurem](https://auremcto.com) — an AI engineer for non-technical founders. ORA itself stays private; this skill is public so anyone can use it on their own models.

## License

MIT — see [LICENSE](LICENSE). Use it in your own model, app, or software. If it fixes something for you, a star and a note about your numbers would mean a lot.
