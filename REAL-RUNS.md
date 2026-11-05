# Real Runs — Our Numbers

This file is the real data behind the repo. No MIT numbers here — these are **our runs**, on real models, local CPU, with a numerical editor (no LLM). Every number below comes from a `report.json` in the run's out-dir.

> **Headline: the gate has refused every real run so far. 0 weights shipped.**
> That is published here, not hidden — a gate that only says yes is not a gate.

## Results at a glance

| Run | Model | Failure | Diagnosis | Causal proof | Weights changed | Gate |
|---|---|---|---|---|---|---|
| Pipeline test | Kit weights (smoke test) | Failing 64.14 vs good 75.98 @ 0.7 | Row 0 | +18.00 — PASSED | **128 (0.77%)** | **PASSED — failing 64.14 → 75.43 (+11.29)** |
| Qwen run 1 | Qwen3-0.6B (596M, 28 layers) | 60-item lists loop (60 animals → 16 distinct) | Layers [27, 26] | −3.33 — FAILED | **0** | **FAILED — nothing shipped** |
| Qwen run 2 | Qwen3-0.6B | Same failure | Layers [27, 26] | **+32.22 — PASSED** | **0** | **FAILED — nothing shipped** |
| Qwen clean re-run | Qwen3-0.6B | Same failure, restore bug fixed | Layers [27, 26] | **+32.22 — PASSED** | **0** | **FAILED — nothing shipped** |
| GPT-2 run | GPT2-124M (124M, 12 layers) | 60-item lists loop (animals 1.7, countries 5.0, fruits 1.7) | Layers [10, 11] | −1.11 — FAILED | **0** | **FAILED — nothing shipped** |

## Pipeline test (smoke test — proves the pipeline, not a real fix)

- Failing set: 64.14 @ temp 0.7 → **75.43 after edit (+11.29)**
- Weights changed: 128 (0.77%), attempt 1, gate passed at temp 0.7 AND greedy 0.0.
- This is a synthetic test model built into the kit. It proves the pipeline runs end-to-end — it is not a language-model result.

## Qwen run 1 — Qwen3-0.6B (~32 min, CPU bf16)

- **Failure:** 60-item lists loop. 60 animals came back as 60 lines but only **16 distinct animals (score 26.7)**. Same on countries and fruits.
- **Baselines @ 0.7:** failing **31.11**, good 100, neighbours 100. Failure gap: **68.89 points**.
- **Temperature sweep:** 22.78 / 31.11 / 32.22 / 31.67 — no cheap fix.
- **Diagnosis:** layers **[27, 26]**, norms [82.459, 48.027]. Contrast was failing vs the **good set** (factual Q&A) — later found to be confounded (see run 2).
- **Causal proof:** 22.78 → 19.44 (**−3.33**, bar 5) — **FAILED**. Stopped. 0 attempts, 0 weights.
- **Gate: false.** Out-dir holds only `report.json`.

## Qwen run 2 — contrast re-run (~58 min, CPU bf16)

- **One design change:** diagnosis contrast = **neighbours set** (short healthy lists) instead of the good set.
- **Causal proof flipped: 22.78 → 55.00 (+32.22) — PASSED.** The confound hypothesis was confirmed.
- **Attempts — CONTAMINATED, do not use as results:** 3 ran, but a restore bug meant the ablation was never restored between attempts, so the numbers quoted here (+3.89 / −12.78 / −6.11, misses 4.11 / 53.78 / 94.61) are **not clean single-attempt measurements**. The gate verdict (false) still stands — only the attempt numbers were wrong.
- **Clean re-run (below) replaces these numbers.**

## Qwen clean re-run — same design, restore fixed (~43 min, CPU bf16)

- **What changed:** only the restore fix (indexed assignment in `real_runner.py`). Same eval, same neighbours contrast, same plan. Report: `runs/qwen-clean/report.json`.
- **Reproduced run 2 exactly:** sweep 22.78 / 31.11 / 32.22 / 31.67, baselines failing **31.11 @ 0.7** (gap 68.89), diagnosis layers **[27, 26]** (norms [78.048, 37.183]), causal proof **+32.22 — PASSED**. The fix changed only the attempt measurements, as it should.
- **Clean attempts (all failed at greedy, so serving temp was never evaluated):**
  - Attempt 1 (1.0×, top-2, 130,945 w, 0.022%): greedy failing **−11.11**, neighbours −22.5, miss 36.61.
  - Attempt 2 (2.0×, top-2, 130,996 w, 0.022%): greedy failing **−11.11**, neighbours −30.0, miss 44.11.
  - Attempt 3 (1.0×, top-1 layer 27, 65,487 w, 0.011%): greedy failing **−12.22**, neighbours −17.5, miss 32.72 — **best attempt**.
- **Gate: false. 0 weights.** Nothing shipped.
- **What this means:** run 2's "near-miss" story (best attempt +3.89, only 4.11 short) was entirely a bug artefact. The clean data says every edit variant made the failing set *worse* at greedy while also regressing neighbours 17.5–30 points. The direction-subtraction edit does not reproduce the ablation benefit at all — the editor is the gap, not the diagnosis.

## GPT-2 run — GPT2-124M (~35.5 min, CPU bf16, plain completion, no chat template)

- **Failure:** 60-item lists loop. 60 animals **1.7**, 60 countries **5.0**, 60 fruits **1.7** (pure "Peaches, Peaches, …"). Also "Spinach" ×60 and "Hockey" ×60 on short lists.
- **Baselines @ 0.7:** failing **28.33**, good 80, neighbours 100. Failure gap: **51.67 points**.
- **Temperature sweep:** 4.44 / 28.33 / 60.56 / 7.78 — 1.0 helps a little (60.56) but not enough (rule needs 75). No cheap fix.
- **Diagnosis:** layers **[10, 11]** (last two of 12), `c_proj`, norms [21.429, 20.613]. Contrast = neighbours set from the start.
- **Causal proof:** 4.44 → 3.33 (**−1.11**, bar 5) — **FAILED**. Stopped. 0 attempts, 0 weights.
- **Gate: false.** Out-dir holds only `report.json`.

## What this proves — and what it does not

**Proves:**
- The pipeline runs end-to-end on real models, on two families (Qwen + GPT-2), CPU-only, no LLM editor.
- It measures real failures honestly (68.89- and 51.67-point gaps).
- Diagnosis can find causal layers (Qwen run 2: +32.22).
- The gate works: a failed proof stops everything before any weight is touched.

**Does not prove:**
- A passing weight edit on a real model. None yet.
- That direction-subtraction can fix these failures. Qwen run 2 showed causal layers whose edit still could not reproduce the ablation.
- Anything about other models, other failures, or large models.

## How a run works (the pipeline)

1. **Step 0 — temperature sweep.** Temperature is the first knob; if it fixes the failure, no edit is needed.
2. **Step 1 — baselines.** Failing / good / neighbours sets at serving temp (0.7) and greedy (0.0).
3. **Step 1 — diagnosis.** MLP signals, failing vs neighbours, top-2 layers.
4. **Step 1 — causal proof.** Zero the suspect columns; need +5 points or stop.
5. **Step 2 — up to 3 edit attempts** on a fixed plan (1.0× top-2, 2.0× top-2, 1.0× top-1).
6. **Step 3 — eval gate.** Ship only if failing improves ≥ +5 with ≤ 2 regression on guards AND neighbours, at both temperatures. Otherwise nothing ships.

Editor: numerical activation-difference estimator — **no LLM**. Reports name the editor, tools, eval passes, and sampling settings.

## Limits

- CPU bf16 only; a 0.6B model takes ~30–60 minutes. Bigger models take far longer.
- Models over 3B parameters are refused by design.
- One failure type so far (list-looping), two models, one eval each.
- Not peer-reviewed, not independently replicated — same as the MIT blog this builds on.

## Reproduce

```bash
pip install './kit[real]'
wsp run --model Qwen/Qwen3-0.6B --eval kit/evals/loop-fix-qwen3-0.6b.json --out-dir out
wsp run --model gpt2 --eval kit/evals/loop-fix-gpt2-124m.json --out-dir out-gpt2
```

Run reports live under `runs/` (report.json only — no edited weights ship when the gate refuses).
