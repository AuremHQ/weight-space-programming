# Weight-Space Doctor — The Skill

Use this skill on **your own small open-weight model** — one that lives in your repo or on your server, in a model, app, or piece of software you control.

It does not work on closed API models, it does not train anything, and it never runs in real time. It is a batch job: load weights, run tests, reason, edit, re-test.

## The 3 steps

### Step 1 — Detect

1. Find the weights: `safetensors` files, PyTorch `.bin` files, plus `config.json`. Read the model card for known issues (e.g. "prone to thinking loops").
2. Run the model on **your own test prompts** — the prompts that show the failure. Record what actually goes wrong: repetition loops (same text until the token limit), stuck behaviour, degenerate outputs.
3. Diagnose the way the MIT experiment did: compare the model's internal signals on failing outputs vs succeeding outputs, and identify the layers/components whose direction tracks the failure. This diagnosis is heuristic + reasoned — never present it as exact science.

**Hard refusals — if any of these is true, stop and say so plainly. Do nothing else:**

1. No weights found (the model is served by an API) → "This model has no weights I can read."
2. Model larger than 3 billion parameters → "Too large for weight-space editing with current tooling." (This cap is a starting value, not a law of nature.)
3. No test prompts from you and no eval set in the repo → never invent test prompts.
4. "Add capability X" with no diagnosed obstacle → refuse, and explain: editing removes obstacles, it does not add abilities (MIT's Gemma control experiment is the proof).
5. Edit a production branch in place, or remove safety/refusal behaviour → refuse, always. Edits go to a new branch and a reviewable change only.

### Step 2 — Fix

- Subtract the offending direction from the identified layers/columns only — the same shape as the MIT Qwen fix (one direction, a small set of layers, a small set of output columns).
- Report exactly: layers touched, columns touched, weight count changed, percentage of total weights changed.
- **If a fix touches more than 1% of weights, reject it as "too broad."** Small targeted edits are the entire point.
- Never run a training loop, never compute gradients, never touch a dataset. Direct edits only.
- Never write over the original model files. Work on a copy, and keep the edit script that reproduces the edit from the original weights — a fix nobody can reproduce is not a fix.

### Step 3 — Eval-gate (no gate, no ship)

- Re-run the **same prompts with the same sampling settings** before and after.
- Ship only if: primary metric improves by **at least 5 points** AND no guard metric regresses by **more than 2 points**. (Starting values — tune them for your task and say what you used.)
- If the gate fails → discard the edit, publish the before/after table anyway, and ship nothing.
- If the gate passes → propose the change as a reviewable package containing, no exceptions:
  1. The edit script (reproducible from the original weights).
  2. A before/after metrics table.
  3. Layers touched, weight count, percentage changed.
  4. The name of the model that produced the edit.
  5. This note: "Weight-space editing was demonstrated with GPT-6 Astra and is unproven for other models. This edit passed the eval gate below; verify independently before use."
  6. The edited weights as a separate artifact — never a silent overwrite of the original files.

## Safety rules

- Weight edits can bypass safety behaviour. That is why safety-removal requests are refused and why every edit goes through evaluation and human review.
- Never claim an edit "adds knowledge" or "makes the model smarter." Describe only what was measured: which failure was diagnosed, which direction was removed, and what the numbers show.
- Log every attempt (success or discarded): who asked, which model, layers touched, before/after metrics, and whether anything shipped.

## What you need to run this

- Open weights you legally control, in a repo or on a server you own.
- Your own test prompts showing the failure, plus at least one guard metric (something that must not get worse).
- CPU or GPU to load the model and run evaluations. The bigger the model, the more you need — which is why the 3B cap exists.
- A frontier-class model to do the reasoning (the editor). Say which one you used: results are editor-dependent until proven otherwise.

## What success looks like

Not a bigger number on one benchmark. Success is: a specific failure, diagnosed in weights, fixed with a tiny edit, proven on your own tests, with the edit script in hand so anyone can reproduce it. If you get that, share your numbers — that is how this field stops being black magic.
