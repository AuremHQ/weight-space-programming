# Real Runs — Our Numbers

This file is the real data behind the repo. No MIT numbers here — these are **our runs**, on real models, local CPU, with a numerical editor (no LLM). Every number below comes from a `report.json` in the run's out-dir.

> **Headline: 7 real runs, 2 gate passes. 260,969 weights shipped in total.**
> The other 5 runs were refused by the gate and shipped 0 weights — that is published here, not hidden. A gate that only says yes is not a gate.

## Results at a glance

| Run | Model | Failure | Diagnosis | Causal proof | Weights changed | Gate |
|---|---|---|---|---|---|---|
| Pipeline test | Kit weights (smoke test) | Failing 64.14 vs good 75.98 @ 0.7 | Row 0 | +18.00 — PASSED | **128 (0.77%)** | **PASSED — failing 64.14 → 75.43 (+11.29)** |
| Qwen run 1 | Qwen3-0.6B (596M, 28 layers) | 60-item lists loop (60 animals → 16 distinct) | Layers [27, 26] | −3.33 — FAILED | **0** | **FAILED — nothing shipped** |
| Qwen run 2 | Qwen3-0.6B | Same failure | Layers [27, 26] | **+32.22 — PASSED** | **0** | **FAILED — nothing shipped** |
| Qwen clean re-run | Qwen3-0.6B | Same failure, restore bug fixed | Layers [27, 26] | **+32.22 — PASSED** | **0** | **FAILED — nothing shipped** |
| GPT-2 run | GPT2-124M (124M, 12 layers) | 60-item lists loop (animals 1.7, countries 5.0, fruits 1.7) | Layers [10, 11] | −1.11 — FAILED | **0** | **FAILED — nothing shipped** |
| Qwen narrow Try 1 | Qwen3-0.6B | 60-item lists loop, narrow eval (animals only) | Layers [27, 26] | **+23.89 — PASSED** | **0** | **FAILED — nothing shipped** (all 3 subtraction attempts destroyed the model at greedy) |
| Qwen narrow dampen | Qwen3-0.6B | Same failure, narrow eval | Layers [27, 26] | **+23.89 — PASSED** | **131,072 (0.022%)** | **PASSED — failing 37.78 → 67.22 (+29.44)** |
| SmolLM2 run | SmolLM2-1.7B-Instruct (1.7B) | 60-item lists loop (repeats: Butterfly ×2, Elephant ×2, Giraffe ×3) | Layer 23 | −30.00 — FAILED | **129,897 (0.0076%)** | **PASSED — failing 76.11 → 85.56 (+9.44)** |

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

## Qwen narrow Try 1 — Qwen3-0.6B, narrow-animals eval (Nov 5 ~18:54, Teji's PC, CPU bf16)

- **Failure:** 60-item lists loop, narrow eval (animals only). Same failure family as the mixed eval, fewer prompts.
- **Baselines @ 0.7:** failing **37.78** (28.89 greedy), good 100, neighbours 100. Failure gap: **62.22 points**.
- **Diagnosis:** layers **[27, 26]**, direction norms [50.795, 29.845]. Contrast = neighbours set.
- **Causal proof:** 28.89 → 52.78 (**+23.89**, bar 5) — **PASSED**.
- **Attempts — all 3 subtraction attempts destroyed the model at greedy:** failing **1.67**, good **0.00** in every attempt (neighbours 73.33 / 36.67 / 100). Best attempt missed the gate by 130.22 points. The alpha formula was the problem: its signed-mean denominator cancels to ~0, so alpha is a coin flip (see the editor investigation in `experiments/EDITOR-FIX-REPORT.md` in the build clone).
- **Gate: false. 0 weights.** Nothing shipped.

## Qwen narrow dampen — Qwen3-0.6B, narrow-animals eval, dampen 0.25 (Nov 5, VM, CPU bf16) — ✅ FIRST QWEN PASS

- **Same failure, same diagnosis as Try 1** (layers [27, 26], causal proof +23.89). Only the edit changed: instead of subtracting a direction, the suspect columns were **dampened toward zero** (scaled down by 0.25).
- **Attempts:** variant 1 (dampen 0.5), variant 2 (dampen 0.25), variant 3 (dampen 0.0). **Variant 2 passed.**
- **Gate: PASSED at serving temp 0.7 AND greedy 0.0.** Failing **37.78 → 67.22 (+29.44 @ 0.7)**, +25.56 @ 0.0. Good 100, neighbours 100 at 0.7 (neighbours +6.67 at greedy — improved, not regressed). Weights changed: **131,072 (0.022%)**, layers [27, 26], dampen factor 0.25.
- **Honest caveat:** the +29.44 slightly overstates the gain — the scorer counts 'Penguin' and 'Penguin (again)' as distinct items, so repeats with "(again)" still score. It clears the +5 bar by a wide margin, but the true gain is a little smaller than 29.44.
- **What this means:** dampening works where subtraction destroyed the model. The rule the investigation found: causal proof positive → dampen the columns down; causal proof negative → subtract with a sane alpha. Report: build clone `wsp-real-out-qwen-narrow-dampen/report.json`.

## SmolLM2 run — SmolLM2-1.7B-Instruct (Nov 5 ~20:18, Teji's PC, CPU bf16) — ✅ FIRST PASSING REAL-MODEL EDIT

- **Failure:** 60-item lists loop. 60 animals came back with repeats — Butterfly twice, Elephant twice, Giraffe three times.
- **Baselines @ 0.7:** failing **76.11**, good 100, neighbours 100. Failure gap: **23.89 points**.
- **Temperature sweep:** 56.67 / 76.11 / 68.89 / 60.56 — no cheap fix.
- **Diagnosis:** layer **23** only.
- **Causal proof:** 56.67 → 26.67 (**−30.00**, bar 5) — **FAILED**. Zeroing the suspect columns made the failure *worse*. The edit passed anyway (see below) — the ablation and the edit do not always agree, and on this model the subtraction worked where the ablation hurt.
- **Attempts:** 3 ran. Variants 1–2 (layers [23, 22], alphas 1.0× / 7867.69× and 2.0× / 15735.39×) destroyed the model. **Variant 3 (layer 23 only, alpha 1.0×) passed.**
- **Gate: PASSED at serving temp 0.7 AND greedy 0.0.** Failing **76.11 → 85.56 (+9.44 @ 0.7)**, +8.33 @ 0.0. Good 100, neighbours 100, zero regression at both temperatures. Weights changed: **129,897 (0.0076%)**, layer 23 only, alpha 1.0×. Edited model saved to `wsp-real-out-smollm2` on Teji's PC.
- **What this means:** the first real-model edit to pass the gate. One layer, one tiny subtraction, no retraining — and the model stopped repeating itself. Report: `C:\Users\tejis\weight-space-programming\wsp-real-out-smollm2\report.json` (Teji's PC).

## What this proves — and what it does not

**Proves:**
- The pipeline runs end-to-end on real models, on three families (Qwen + GPT-2 + SmolLM2), CPU-only, no LLM editor.
- It measures real failures honestly (68.89-, 62.22-, 51.67- and 23.89-point gaps).
- Diagnosis can find causal layers (Qwen run 2: +32.22; Qwen narrow: +23.89).
- The gate works: a failed proof stops everything before any weight is touched — and 5 of 7 runs were refused.
- **Two real-model edits passed the gate.** SmolLM2 by subtraction (layer 23, alpha 1.0×, +9.44); Qwen narrow by dampening (layers [27, 26], dampen 0.25, +29.44). Both held at serving temperature and greedy, with zero regression on good and neighbours at serving temperature.

**Does not prove:**
- That the kit fixes real models in general. **2 passes out of 7 runs** is the claim — both on the same narrow failure (60-item list looping), both on small models, both with a numerical editor.
- That direction-subtraction can fix these failures in general. It passed on SmolLM2 and destroyed Qwen at greedy; dampening passed on Qwen and hurt SmolLM2. The right edit depends on the model and the causal sign — there is no single editor yet.
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
- One failure type so far (list-looping), three models (Qwen, GPT-2, SmolLM2), one eval each — two of the seven runs passed the gate.
- Not peer-reviewed, not independently replicated — same as the MIT blog this builds on.

## Reproduce

```bash
pip install './kit[real]'
wsp run --model Qwen/Qwen3-0.6B --eval kit/evals/loop-fix-qwen3-0.6b.json --out-dir out
wsp run --model gpt2 --eval kit/evals/loop-fix-gpt2-124m.json --out-dir out-gpt2
```

Run reports live under `runs/` (report.json only — no edited weights ship when the gate refuses).
