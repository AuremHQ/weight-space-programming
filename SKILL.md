---
name: weight-space-programming
description: Fix ONE narrow repeatable failure of YOUR OWN open-weight model by a reasoned edit to under 1% of weights, gated by your own eval — use small models for better performance. Refuses API-only models and runs without your test prompts.
---

# Weight-Space Programming — The Skill

Use this skill on **your own small open-weight model** — one that lives in your repo or on your server, in a model, app, or piece of software you control.

It does not work on closed API models, it does not train anything, and it never runs in real time. It is a batch job: load weights, run tests, reason, edit, re-test.

## How this differs from gradient descent

Gradient descent trains a model: data plus a loss function produce gradients, and many small updates (w ← w − η∇L) slowly move the weights. This skill does none of that. It makes one reasoned, direct edit — subtracting a diagnosed direction from a small set of layers — with no data, no loss function, no gradients, and no training loop. Then the eval gate proves whether that single edit helped, on your own prompts. If you want training, use training. If you want one stubborn failure removed without retraining, use this.

## The steps

### Step 0 — Try the cheap fix first

Before touching any weight, try the cheap fixes: a prompt change, a decoding or settings change (for example, thinking mode off, temperature, stop rules). If the failure goes away, that fix belongs in your baseline — not in a weight edit. Edit weights only for a failure that survives the cheap fixes. A settings change that matches an edit's gain is a better result than the edit, because it costs nothing and risks nothing. Temperature is the first knob to try: record the serving temperature, try one lower and one higher setting, and measure each on its own. Never change temperature and weights in the same run — if both change at once, you will not know which one helped.

### Step 1 — Detect

1. Find the weights: `safetensors` files, PyTorch `.bin` files, plus `config.json`. Read the model card for known issues (e.g. "prone to thinking loops").
2. Run the model on **your own test prompts** — the prompts that show the failure. Record what actually goes wrong: repetition loops (same text until the token limit), stuck behaviour, degenerate outputs. Before editing anything, measure the failure as your baseline: on how many of your prompts it happens, and how bad it is (for example, loops to the token limit vs an occasional slip). The eval gate compares against these numbers, so a baseline is not optional.
3. Diagnose the way the MIT experiment did: compare the model's internal signals on failing outputs vs succeeding outputs, and identify the layers/components whose direction tracks the failure. This diagnosis is heuristic + reasoned — never present it as exact science.
4. Prove the suspect causes the failure before editing it: temporarily disable or swap the suspect layers/columns (an ablation/patch test) and show the failure changes. A suspicious pattern that does not move the failure when removed is not a diagnosis — do not edit it.

**Hard refusals — if any of these is true, stop and say so plainly. Do nothing else:**

1. No weights found (the model is served by an API) → "This model has no weights I can read."
2. No test prompts from you and no eval set in the repo → never invent test prompts.
3. "Add capability X" with no diagnosed obstacle → refuse, and explain: editing removes obstacles, it does not add abilities (MIT's Gemma control experiment is the proof).
4. Edit a production branch in place, or remove safety/refusal behaviour → refuse, always. Edits go to a new branch and a reviewable change only.

### Step 2 — Fix

- Subtract the offending direction from the identified layers/columns only — the same shape as the MIT Qwen fix (one direction, a small set of layers, a small set of output columns).
- Report exactly: layers touched, columns touched, weight count changed, percentage of total weights changed.
- **If a fix touches more than 1% of weights, reject it as "too broad."** Small targeted edits are the entire point.
- Never run a training loop, never compute gradients, never touch a dataset. Direct edits only.
- Never write over the original model files. Work on a copy, and keep the edit script that reproduces the edit from the original weights — a fix nobody can reproduce is not a fix.
- Budget: at most 3 edit attempts per failure. If the third attempt still fails the gate, discard and stop — do not keep iterating until the eval set is overfit.

### Step 3 — Eval-gate (no gate, no ship)

- Re-run the **same prompts with the same sampling settings** before and after.
- Check the gate at the serving temperature and at greedy (temperature 0) as well. An edit that passes at one temperature and fails at another is not ready to ship.
- Ship only if: primary metric improves by **at least 2 points** AND no guard metric regresses by **more than 2 points** AND no neighbouring behaviour regresses by **more than 2 points**. Neighbours are tasks or concepts close to the edited behaviour (for example, nearby facts or related skills) — edits disturb neighbours, so test them, do not assume they are untouched. (Starting values — tune them for your task and say what you used.)
- If the gate fails → discard the edit, publish the before/after table anyway, and ship nothing.
- If the gate passes → propose the change as a reviewable package containing, no exceptions:
  1. The edit script (reproducible from the original weights).
  2. A before/after metrics table.
  3. Layers touched, weight count, percentage changed.
  4. The name of the model that produced the edit, the tools it used (for example, a Python sandbox, forward passes), and how many eval passes were spent. Tools and pass counts are part of the result — an edit made with heavy tooling must never be reported as if it came from weights alone. Also record the sampling settings used, including temperature — a result without its temperature is not reproducible.
  5. This note: "Background: the LLM-editor method was demonstrated by MIT with GPT-6 Astra; blog post, not peer reviewed. Editor of THIS edit: the kit's numerical estimator (no LLM). This edit passed the eval gate below; verify independently before use."
  6. The edited weights as a separate artifact — never a silent overwrite of the original files.

## Merge mode — separate from surgical editing (R6)

Merging is a different mode from the surgical editing in Steps 0–3. Never mix them: a merge blends two whole models; a surgical edit removes one diagnosed direction from a few layers. The rules below are the merge rules; the surgical rules do not apply to merges, and the merge rules do not apply to surgical edits.

- **Methods:** Task Arithmetic, TIES, DARE, SLERP, Model Soups (MergeKit-style tools). The kit ships linear averaging and SLERP first; the others are future work, not silent defaults.
- **Same base required, enforced not assumed:** the two models must share the same architecture AND the same tokenizer — same `architectures`, same `model_type`, same `vocab_size`, same hidden/intermediate sizes and layer count, same tokenizer files. The kit checks this in code and refuses the merge on any mismatch. Never merge models from different bases and hope the shapes line up.
- **NOT under the 1% weight cap:** a merge changes most weights by definition (it blends two whole models). The 1% cap is a surgical-edit rule and does not apply here — say so plainly in every merge report instead of pretending the merge was small.
- **Own full eval gate:** the merged model is evaluated on your own eval file, at the serving temperature AND at greedy, against the first base model as the baseline. It ships only if the primary metric improves by at least 2 points AND no guard or neighbour metric regresses by more than 2 points — the same bar as the surgical gate. If the gate fails, discard the merge and ship nothing.
- **No model-size cap.** Merges work on any size model. Use small models for better performance.
- **Never merge on a production branch, and never overwrite the originals.** The merged model goes to a new directory only, with a merge recipe (method, blend weight, base models) that reproduces it.
- **Never invent evals for a merge.** A merge without your own eval file is refused, exactly as a surgical edit without one is.
- **Merges need no editor model** — they are arithmetic on weights, not reasoning.
- **Every merge report states:** the method and blend weight, the two base models, the same-base evidence, and the eval passes spent.

## Safety rules

- Weight edits can bypass safety behaviour. That is why safety-removal requests are refused and why every edit goes through evaluation and human review.
- Never claim an edit "adds knowledge" or "makes the model smarter." Describe only what was measured: which failure was diagnosed, which direction was removed, and what the numbers show.
- Log every attempt (success or discarded): who asked, which model, layers touched, tools used, eval passes spent, before/after metrics, and whether anything shipped.

## What you need to run this

- Open weights you legally control, in a repo or on a server you own.
- Your own test prompts showing the failure, plus at least one guard metric (something that must not get worse) and a few neighbour probes (nearby tasks that should not change).
- CPU or GPU to load the model and run evaluations. The bigger the model, the more you need. Use small models for better performance.
- An editor to do the reasoning: either a frontier-class model (agent-reasoned path — say which one you used; results are editor-dependent until proven otherwise) or the kit's numerical estimator (`wsp run` — no LLM involved).

## What success looks like

Not a bigger number on one benchmark. Success is: a specific failure, diagnosed in weights, fixed with a tiny edit, proven on your own tests, with the edit script in hand so anyone can reproduce it. If you get that, share your numbers — that is how this field stops being black magic.
