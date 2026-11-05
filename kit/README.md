# wsp-kit

Weight-space programming, runnable. Point it at a small model's weights
and a failure you can reproduce, and it diagnoses the failure in the
weights, applies one tiny edit, and lets an eval gate decide whether
the edit ships. No retraining, no gradients, no training loop.

The idea comes from MIT CSAIL's "Sparks of Weight-Space Programming"
(Nov 2026), where a frontier model edited 0.016% of a small model's
weights and fixed a reasoning loop. This kit is an independent
implementation of that idea, built to the rules in the repo's
[SKILL.md](../SKILL.md). It is not affiliated with MIT.

## Install

Requires Python 3.9+ and numpy. From a clone of this repo:

```
pip install ./kit
```

Or straight from GitHub:

```
pip install git+https://github.com/AuremHQ/weight-space-programming#subdirectory=kit
```

Optional extras:

```
pip install 'wsp-kit[mcp]'          # MCP server support
pip install 'wsp-kit[safetensors]'  # .safetensors weights support
```

## Quickstart — real models first

```
pip install 'wsp-kit[real]'   # torch + transformers + accelerate
wsp run --model Qwen/Qwen3-0.6B --eval evals/loop-fix-qwen3-0.6b.json
```

That runs the full gated pipeline on a real model (see "Real models" below). For a fast pipeline check with no model download:

```
wsp demo
```

This runs the toy pipeline (smoke test) end to end and writes `report.json` and
`toy_edited.npz` into `wsp-demo-out/`. Real output:

```
baseline (serving temperature 0.7, defective model)
  before
  good          75.98
  failing       64.14
  neighbours    74.80

temperature sweep, failing set only (Step 0: the cheap knob first)
  temperature 0.0  failing  81.33  good  98.33
  temperature 0.7  failing  64.14  good  75.98
  temperature 1.0  failing  56.63  good  66.72
  temperature 1.5  failing  44.84  good  52.44

causal proof: zeroing row 0 moves the failing set from 81.33 to 99.33 (+18.00 points) — causal: True

attempt 1: alpha 1.5 — 128 weights (0.77%) — passed: True

  after (serving temperature 0.7)
  good          76.81
  failing       75.43
  neighbours    76.29

gate passed: failing +11.29 points, good +0.84 points, neighbours +1.49 points
wrote wsp-demo-out/report.json
wrote wsp-demo-out/toy_edited.npz
```

Read that output the way it is meant: temperature alone did not fix
anything (the failing set got worse as temperature rose, and the good
set fell with it), the ablation proved the suspect row drives the
failure, and one edit of 128 weights (0.77%) moved the failing set
from 64.14 to 75.43 at the serving temperature, with no guard or
neighbour regression. The gate passed at the serving temperature and
at greedy, which is why the edit shipped.

## Real models (v0.2.0)

The toy proves the pipeline. The real runner runs the same pipeline
on a real open-weight model, with your eval file, from one command:

```
pip install 'wsp-kit[real]'   # torch + transformers + accelerate
wsp run --model Qwen/Qwen3-0.6B --eval eval-loop.json
```

That single command does the whole flow with no manual steps in
between: temperature sweep (Step 0, the cheap knob) → baselines →
diagnosis → causal proof → up to 3 edit attempts → gate → report.
It writes `report.json` into `wsp-real-out/` and, only on a gate
pass, the edited model plus `edit.json` (the reproducible edit
recipe). The original model files are never touched: edited weights
always go to a new directory.

**Eval files (format v1).** A JSON file with three prompt sets —
`failing`, `good`, `neighbours` — each with a scorer (`loop`,
`contains`, `finish`, or `sequence`) and prompts. The pipeline
detects whatever failure your eval shows (loops, wrong answers,
format breaks) and runs the same Step 0 → Step 1 → Step 2 → Step 3
flow. See `evals/` for the demo eval file and its format notes.

**Best passing, not first passing.** Within the skill budget (max 3
attempts, at most 1% of weights), the pipeline evaluates every
attempt variant — base alpha on the top 2 diagnosed layers, double
alpha on the same layers, base alpha on the single strongest layer —
and ships the BEST PASSING one (largest serving-temperature primary
gain, tie-break: fewer weights changed). Temperatures are evaluated
greedy-first: if the gate already fails at greedy, the remaining
temperatures are skipped for that attempt, because a gate fail at
any temperature is a gate fail overall.

**Fix on failure only.** If the baseline shows no real failure (the
failing set within 5 points of the good set at serving temperature),
the pipeline stops before any edit. A healthy model is not edited.

**Gate fail = nothing ships.** If no attempt passes, no weights are
written. The report still ranks all attempts, shows how far the best
one missed the gate (primary shortfall and guard excess, in points),
and suggests what to try next — so the best is visible without
shipping a failing edit. The gate rule is absolute: it is not
overridden by any flag.

**Watch mode (opt-in).** `wsp watch` monitors a directory for new
eval files and runs the same gated pipeline on each one:

```
wsp watch --model Qwen/Qwen3-0.6B --watch-dir ./evals-incoming --out-dir ./wsp-watch-out
```

Each run writes its own timestamped output directory with its own
report. Stop it with Ctrl+C. Watch mode NEVER applies an edit to a
live or production model — producing the edited file plus report is
the kit's job; applying it is always your deliberate step, with
`wsp apply-edit --model <id> --edit <edit.json> --out-dir <dir>`.

**Scope boundary.** The runner supports open-weight models only,
up to 3 billion parameters, loaded through HuggingFace Auto classes
with MLP layers discovered generically (Qwen, Llama, Gemma, Mistral
and other families — no hardcoded module path). Closed models (GPT,
Claude, and the like) are OUT of scope by design: their weights are
unavailable, so no weight edit is possible there.

**Worked example: Qwen3-0.6B list-loop demo (verified run, 2026-11-05).**
Eval file `evals/loop-fix-qwen3-0.6b.json`: 3 failing prompts (list
60 distinct animals / countries / fruits, `distinct` scorer, 60
expected), 4 good prompts (factual QA, `contains` scorer), 4
neighbour prompts (short 5–10 item lists, `distinct` scorer). 512
tokens, chat template on, thinking off. Command:

```
wsp run --model Qwen/Qwen3-0.6B --eval evals/loop-fix-qwen3-0.6b.json
```

Measured, not made up (596,049,920 parameters, CPU, bfloat16, ~32
minutes). Step 0 sweep, failing subset: 22.78 at greedy, 31.11 at
0.7, 32.22 at 1.0, 31.67 at 1.5 — no temperature restored it to
within 5 points of the good subset (100.00), so no cheap fix.
Baselines: failing 31.11 at serving temperature 0.7 and 22.78 at
greedy; good 100.00 at both; neighbours 100.00 at 0.7 and 95.00 at
greedy. Failure gap at serving temperature: 68.89 points — a real
failure (verified separately: a 60-animal list came back as 60 lines
but only 16 distinct animals, score 26.7). Diagnosis picked layers
27 and 26 (direction norms 82.459 and 48.027). Causal proof: zeroing
the suspect columns moved the failing subset from 22.78 to 19.44
(−3.33 points) — below the 5-point bar, so **not causal: the
pipeline stopped, no edit was made, nothing shipped**. The output
directory holds only `report.json`.

Read that outcome the way it is meant: the pipeline ran end to end
on a real model and refused to edit without a causal diagnosis. That
refusal is the feature, not a bug — it is the same discipline as
MIT's Gemma control, which could not be improved either. What this
run does NOT show is a passing edit; a passing real-model edit is
still unproven, and no number here claims otherwise. Task pack:
`tasks/loop-fix-real/task.json`.

**Qwen run 2 — same model, diagnosis contrast fixed (neighbours set
instead of the good set).** Causal proof flipped to **+32.22 (22.78 →
55.00) — PASSED**. Three edit attempts ran; **gate failed again, 0
weights**. (Note: run 2's attempt numbers were contaminated by a
restore bug — the ablation was never restored between attempts — so
they are not quoted here. A **clean re-run** with the fix (~43 min)
reproduced the sweep, baselines, diagnosis and causal proof exactly,
but all three clean attempts made the failing set *worse* at greedy:
−11.11 / −11.11 / −12.22, neighbours regressed 22.5 / 30.0 / 17.5,
best miss 32.72. The run-2 "near-miss" was entirely a bug artefact;
the edit does not reproduce the ablation benefit.)

**GPT-2 run — GPT2-124M, plain completion, no chat template (~35.5
min, CPU).** 60-item lists loop (animals 1.7, countries 5.0, fruits
1.7). Baselines @ 0.7: failing **28.33**, good 80, neighbours 100 —
a **51.67-point** gap. Diagnosis: layers **[10, 11]**, norms 21.429 /
20.613. Causal proof **−1.11 — FAILED**. **0 attempts, 0 weights,
gate false, nothing shipped.** Full numbers for all runs: the
repo's [REAL-RUNS.md](../REAL-RUNS.md).

## What the pipeline test proves, and what it does not

The pipeline test runs on a toy model: a two-layer network (64 inputs, 128
hidden, 64 outputs) with a deliberately planted defect. Its numbers
are real — they are measured, not made up — but they are toy numbers.
They prove the pipeline works end to end: baseline, cheap-fix check,
causal proof, edit, gate, report. They do not prove anything about a
language model.

For a real model, v0.2.0 ships the runner (see "Real models" above)
but not a frontier-class editor: the direction here is numerical,
estimated from measured activations, with no LLM involved. MIT's
result used GPT-6 Astra; whether other editors do this equally well
is unproven, and every report this kit writes names its editor for
exactly that reason.

## How it works

1. **Baseline.** Measure the failure first: how often it happens, how
   bad it is, at the serving temperature. No baseline, no edit.
2. **Cheap fix first.** Try prompt and decoding changes, including
   temperature, before any weight is touched. If a cheap fix works, it
   goes in the baseline, not in an edit.
3. **Causal proof.** Temporarily disable the suspect and show the
   failure moves. A suspect that does not move the failure is not a
   diagnosis and is not edited.
4. **Edit.** Subtract the estimated direction from the identified row
   only. Over 1% of weights is rejected as too broad. Original files
   are never overwritten; the edited weights are a separate artifact
   plus the script that reproduces them.
5. **Gate.** Same prompts, same settings, before and after, at the
   serving temperature and at greedy. Ships only on +5 points primary
   with no guard or neighbour regression over 2 points. At most 3
   attempts; anything else is discarded, and the report is written
   anyway.

Refusals, enforced in code: no weights, model over 3B parameters, no
test prompts, add-capability requests with no diagnosed obstacle,
production-branch edits, safety-removal requests. Any of these stops
the run with a plain message and nothing edited.

## MCP server

The same pipeline, callable from any MCP-compatible harness or agent:

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

Tools (all stable in v0.2.0):

| Tool | What it does |
|---|---|
| `wsp_version` | Kit and report-format versions |
| `toy_baseline` | Build the toy model, plant the defect, measure the baseline |
| `toy_temperature_sweep` | Step 0: try temperatures before touching weights |
| `toy_prove_cause` | Ablation test on the suspect row |
| `toy_estimate_direction` | Estimate the failure direction from hidden activations |
| `toy_apply_edit` | Subtract alpha times the direction (1% cap enforced) |
| `toy_run_gate` | Gate the candidate at serving temperature and greedy |
| `toy_demo` | The whole pipeline, returns the report |

## Layout

```
kit/
  pyproject.toml
  README.md
  wsp/
    weights.py     load/save .npz and .safetensors, size cap
    toy.py         the toy model and the planted defect
    diagnose.py    baseline, temperature sweep, causal proof, direction
    edit.py        direction subtraction, 1% cap
    gate.py        eval gate
    report.py      versioned JSON report
    pipeline.py    the full pipeline, used by CLI and MCP alike
    real_runner.py load/run real models (3B cap), greedy-first gate
    real_pipeline.py  the real-model pipeline: sweep → proof → attempts
    real_edit.py   layout-aware column edits (Linear + Conv1D)
    layers.py      generic MLP down-proj discovery across families
    evalfile.py    eval format v1 (failing / good / neighbours)
    scoring.py     scorers: distinct, contains, loop, finish, sequence
    watch.py       opt-in watch mode (wsp watch)
    cli.py         `wsp demo`, `wsp run`, `wsp watch`, `wsp apply-edit`, `wsp version`
    mcp_server.py  MCP tools
  evals/
    loop-fix-qwen3-0.6b.json
    loop-fix-gpt2-124m.json
  tasks/
    loop-fix-toy/task.json
    loop-fix-real/task.json
  tests/
    test_kit.py    unittest, no extra dependencies
```

Run the tests from the kit directory:

```
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

## Compatibility

Report format version 1 is stable. Later versions add fields; they do
not rename, remove, or re-type existing ones. The same rule covers the
MCP tools and the task format: new tools and fields may appear, old
ones keep working. A harness or agent built against v0.1.0 keeps
working after an upgrade, with no changes on its side. A breaking
change, if one ever comes, ships as a new major version alongside the
old one, never as a silent replacement.

## Limits, stated plainly

- This is a batch job, not real time. Load weights, run evaluations,
  reason, edit, re-evaluate. It never runs live inside a chat reply.
- Editing removes obstacles; it does not add abilities. A model that
  never had a capability will not gain it from an edit.
- Weight edits can bypass safety behaviour. That is why safety-removal
  requests are refused and every edit goes through evaluation and
  human review.
- Real runs are live (Qwen + GPT-2, see the repo's REAL-RUNS.md) —
  but no real-model edit has passed the gate yet. A passing
  real-model edit is still the next milestone.

## License

MIT — see the repo root [LICENSE](../LICENSE).
