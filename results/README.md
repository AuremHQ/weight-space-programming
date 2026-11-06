# Results — Our Real Runs

These are **our runs**, measured on our own models with our own kit — not MIT's numbers. Every number below comes from a `report.json` in the run's out-dir. The full detail, including what each run proves and does not prove, is in [REAL-RUNS.md](../REAL-RUNS.md).

> **Headline: 7 real runs, 2 gate passes. 260,969 weights shipped in total.**
>
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

## Notes

- The pipeline test is a synthetic 2-layer test model with a known, planted failure. It proves the pipeline runs end-to-end — it is not a language-model result.
- One honest caveat on the Qwen pass: the +29.44 slightly overstates the gain — the scorer counts 'Penguin' and 'Penguin (again)' as distinct items, so repeats with "(again)" still score. It clears the +5 bar by a wide margin, but the true gain is a little smaller than 29.44.
- The two passes used different edits: SmolLM2 by subtraction (layer 23, alpha 1.0×), Qwen narrow by dampening (layers [27, 26], dampen 0.25). The right edit depends on the model.
- **What this does not prove:** that the kit fixes real models in general. 2 passes out of 7 runs, both on the same narrow failure (60-item list looping), both on small models, both with a numerical editor.
- For the MIT research numbers that inspired this kit, see [RESEARCH.md](../RESEARCH.md) — those are MIT's results, not ours.
