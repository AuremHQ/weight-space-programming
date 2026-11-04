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

## Quickstart

```
wsp demo
```

This runs the toy pipeline end to end and writes `report.json` and
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

## What the demo proves, and what it does not

The demo runs on a toy model: a two-layer network (64 inputs, 128
hidden, 64 outputs) with a deliberately planted defect. Its numbers
are real — they are measured, not made up — but they are toy numbers.
They prove the pipeline works end to end: baseline, cheap-fix check,
causal proof, edit, gate, report. They do not prove anything about a
language model.

For a real model you need two things this kit does not ship in v0.1.0:
a runner for that model's weights (the kit works on any object with a
`forward` method returning logits and hidden states), and a
frontier-class editor model to do the reasoning. MIT's result used
GPT-6 Astra; whether other editors do this equally well is unproven,
and every report this kit writes names its editor for exactly that
reason.

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

Tools (all stable in v0.1.0):

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
    cli.py         `wsp demo`, `wsp version`
    mcp_server.py  MCP tools
  tasks/
    loop-fix-toy/task.json
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
- The toy demo is a toy. The real-model demo from the repo's DEMO-PLAN
  is still the next milestone.

## License

MIT — see the repo root [LICENSE](../LICENSE).
