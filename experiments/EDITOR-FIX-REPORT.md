# wsp-kit editor fix: direction subtraction → column dampening
(+ alpha mis-scaling found via the SmolLM2 pass)

Date: 2026-11-06 (VM work; build clone only, nothing pushed)
Models: Qwen/Qwen3-0.6B (eval `kit/evals/loop-fix-qwen3-0.6b-narrow-animals.json`),
SmolLM2-1.7B-Instruct (eval `kit/evals/loop-fix-smollm2-1.7b.json`)

## 1. Verdict (short version)

Two findings, both measured:

1. **The alpha computation is broken and is the primary defect.**
   `alpha = direction_norm / signed-mean-activation` divides by a
   quantity that cancels to ~0, so it returns either the 1.0 fallback
   or a blowup (371 on Qwen, **7867 on SmolLM2 layer 22**). The
   SmolLM2 run proves it: attempts 1–2 (layers [23,22]) were
   destroyed by layer 22's alpha of 7867.7, while attempt 3
   (layer 23 only, alpha 1.0) **PASSED the gate (+9.44 serving gain)
   — the first passing real-model edit**. The "two-layer destroys,
   single-layer works" pattern is explained by the alpha asymmetry,
   not by the layer count.
2. **Even with a sane alpha, Qwen's subtraction fails.**
   On Qwen, subtraction at every alpha from 0.5 to 371 made the
   failing set worse (probe dose-response). The fix that works on
   Qwen is a different edit on the same diagnosed columns:
   **dampening** — scale the suspect down-proj columns by a factor
   (0.5 / 0.25 / 0.0). Dampening 0.25 moved the failing subset
   26.67 → 68.33 (+41.67) with good and neighbours unchanged at 100.
   The diagnosis (layers, columns, causal proof) is unchanged; only
   the edit operation changed. The patch is in this clone
   (`kit/wsp/real_edit.py`, `real_runner.py`, `real_pipeline.py`,
   `cli.py`, `tests/test_real.py`); full-pipeline confirmation run
   below (§6). The robust editor needs BOTH: a fixed alpha (so
   subtraction stops blowing up) AND dampening (so Qwen-class
   failures, where subtraction fails at any alpha, still get fixed).
   The gate decides which ships.
3. **Dampening is not universal — the causal proof's SIGN picks
   the edit family.** Qwen's causal proof is POSITIVE (+15.56:
   zeroing helps) and dampening PASSES the gate there (+29.44).
   SmolLM2's causal proof is NEGATIVE (−30.00: zeroing hurts) and
   dampening 0.5 HURTS there (failing subset 23.33 → 18.33), while
   subtraction PASSED on SmolLM2 (+9.44). Rule: causal + → turn the
   columns down (dampening); causal − → reshape them (subtraction
   with a sane alpha). The robust editor should condition its
   attempt plan on the causal sign instead of always running the
   same 3 attempts.

## 2. The problem, as measured

5 real runs, all gate-false, 0 weights shipped. Freshest (Teji's PC,
Nov 5 ~18:54, narrow-animals): baselines failing 37.78 @0.7
(28.89 greedy), good 100, neighbours 100, gap 62.22. Diagnosis
layers [27, 26], direction norms [50.795, 29.845]. Causal proof
(zero the suspect columns): failing 28.89 → 52.78 (+23.89), causal
True. Alpha bases {27: 371.401, 26: 0.0}. All 3 subtraction attempts
at greedy: failing 1.67, good 0.00 in every attempt (neighbours
73.33 / 36.67 / 100). Best attempt missed the gate by 130.22 points.

VM clean re-run (mixed eval) showed the same shape with a small
alpha: causal +32.22 passed, attempts at alpha 1.0/2.0 made failing
worse (22.78 → 11.67) while good stayed 100.

## 3. Root cause (measured, not guessed)

Probe: `experiments/probe_alpha.py` (results
`experiments/probe_alpha_results.json`). Reproduced the PC diagnosis
exactly on the VM: layers [27, 26], norms [48.84, 31.18].

Three measured facts, each fatal to subtraction:

**(a) The alpha formula is unstable — its output is a coin flip.**
`alpha_base = direction_norm / mean_failing_activation`, where the
denominator is the SIGNED mean activation of the selected columns.
Layer 27: signed mean −0.2674, but mean |activation| 20.50.
Layer 26: signed mean −0.4764, mean |activation| 5.46. The signed
mean cancels to ~0 (activations are ±20), so the formula returns
either the 1.0 fallback (denominator ≤ 0, as on the VM) or a blowup
(371 on the PC, denominator +0.137). Same code, same model, same
eval — alpha 1.0 vs 371 depending on the sign of a near-zero mean.
It is not data-driven; it is an accident of cancellation. It also
ignores the column count: the formula aims for
alpha × mean_act ≈ direction_norm PER column, but 64 columns fire,
so the total shift is ~64× the intended one.

**(b) The selected columns are not failing-specific.**
Layer 27, sum of mean activations over the 64 columns: failing
−17.11, neighbours −5.6, **good −75.0**. Layer 26: failing −30.49,
neighbours −30.9, good −25.1 — nearly identical on all three sets.
The columns fire on good prompts as much as (or more than) on
failing ones. Any edit to them hits good behaviour too.

**(c) The per-token injected shift is huge and noisy at ANY alpha.**
Subtracting alpha·d from a column injects −alpha·a_j·d into the
residual whenever that column fires. With |a_j| ~ 5–20 and 64
columns, the per-token shift is tens to hundreds of units along d,
against an MLP-output norm of 93–353. Even alpha = 1.0 changes each
edited column by 1.2–1.8× its own norm (column norms 0.57–0.82).
The edit is a large random shove along d on every token of every
prompt — not a targeted removal of a small excess. Zeroing, by
contrast, is bounded: it only removes the columns' own (small)
contribution, which is why the causal proof passes while the edit
destroys the model.

## 4. Dose-response (subset: failing[0] + good[0,1] + neighbours[0], greedy)

Baseline (no edit): failing 26.67, good 100/100, neighbours 100.

| edit | failing[0] | good[0] | good[1] | neighbours[0] |
|---|---|---|---|---|
| subtract α=0.5 | 8.33 | 100 | 100 | 100 |
| subtract α=1.0 | 16.67 | 100 | 100 | 80 |
| subtract α=2.0 | 18.33 | 0 | 100 | 40 |
| subtract α=5.0 | 0.00 | 0 | 0 | 40 |
| subtract α=10 | 3.33 | 0 | 0 | 20 |
| subtract α=25 | 8.33 | 0 | 0 | 20 |
| subtract α=50 | 3.33 | 0 | 0 | 20 |
| subtract α=100 | 1.67 | 0 | 0 | 100 |
| subtract α=371 (PC alpha) | 5.00 | 0 | 0 | 40 |
| **dampen ×0.5** | **33.33** | **100** | **100** | **100** |
| **dampen ×0.25** | **68.33** | **100** | **100** | **100** |
| dampen ×0.0 (= zeroing) | (probe interrupted before this row; causal proof already covers it: +23.89 PC / +32.22 VM on the full failing subset) | | | |

Subtraction NEVER improves failing at any alpha; damage to good
starts at α=2 and is total by α=5. Raw outputs confirm it: α=2 gives
"The capital of France is Leclude.", α=5 gives Chinese text,
α≥25 gives "(${laughsANDARDensiveensive..." garbage, α=371 gives
repeated "其他国家...". Dampening keeps "The capital of France is
Paris." intact at every factor, and its dose-response is clean and
monotone: the more the suspect columns are scaled down, the better
the failing set gets, with zero damage.

Serving-temperature check (`experiments/probe_07.py`, subset @ 0.7):
no-edit failing[0] 35.00; dampen 0.25 → 36.67 (+1.67); dampen 0.0 →
46.67 (+11.67); good/neighbours 100 throughout. The 0.7 gain is
smaller than the greedy gain (sampling noise on one prompt); the
full-set 0.7 result is decided by the confirmation run below.

## 4a. The SmolLM2 pass — and the alpha asymmetry that explains it

New evidence (Teji's PC, Nov 5 ~20:18, run repeated twice with
identical results — first passing real-model edit):

- Baselines: failing 76.11 @0.7 (56.67 greedy), good 100,
  neighbours 100, gap 23.89 (a milder failure than Qwen's 62.22 gap).
- Diagnosis layers [23, 22], direction norms [292.896, 129.346].
- Causal proof: failing 56.67 → 26.67 (**−30.00**) yet causal: True
  (the check is |change| ≥ 5). Note the sign: zeroing made failing
  WORSE — the suspect columns were HELPING the failing set. On Qwen
  the sign was the opposite (zeroing helped +23.89/+32.22). The
  sign turns out to matter a great deal: it predicts which edit
  family works (§4c) — causal + → dampening, causal − →
  subtraction. The original causal check (|change| ≥ 5) ignores the
  sign; the robust editor should not.
- Alpha bases **{23: 1.0, 22: 7867.694}**.
- Attempts 1–2 (layers [23,22], alpha 1.0x/2.0x): failing 1.67,
  good 0.00, neighbours 17.50 — destroyed.
- Attempt 3 (layer [23] only, alpha 1.0x): **PASSED** — failing
  65.00 @0.0 / 85.56 @0.7, good 100, neighbours 100, 129897 weights
  (0.01%), gate True, serving gain **+9.44**.

**Why layer 22's alpha is 7867x layer 23's — the arithmetic:**
`alpha = direction_norm / mean_act` (signed mean activation of the
selected columns on failing prompts), or 1.0 if mean_act ≤ 1e-6.

- Layer 23: direction_norm 292.896, alpha 1.0 → its signed mean
  activation was ≤ 1e-6 (fallback). The mean cancelled to ~0.
- Layer 22: direction_norm 129.346, alpha 7867.694 → its signed
  mean activation was **+0.01644** (129.346 / 7867.694).
- The two directions are only **2.26x** apart in norm
  (292.896 / 129.346), but the alphas are **7867.7x** apart. The
  difference is entirely the denominator: one layer's signed mean
  happened to be ≤ 0 (fallback), the other's happened to be a tiny
  positive (+0.016). Same code, same model, same eval — a coin flip
  in the sign of a near-zero mean decides between alpha 1.0 and
  alpha 7867.

**What destroyed attempts 1–2:** attempt 1 applied alpha 7867.694
to layer 22 (and 1.0 to layer 23); attempt 2 applied 15735.4 to
layer 22. A column of norm ~0.5–1 receiving a 7867-unit direction
is a ~10000x rewrite — the model is destroyed before layer 23's
sane edit can matter. Attempt 3 skipped layer 22 entirely, so only
the sane alpha-1.0 edit ran — and it passed. "Two-layer destroys,
single-layer works" is the OBSERVATION; the alpha asymmetry is the
MECHANISM. Had layer 22's alpha been sane, the two-layer attempt
might have passed too — untested.

**VM reproduction (probe_smol.py):** the asymmetry reproduces
exactly on the VM — layers [23, 22], norms [282.522, 127.524]
(PC: [292.896, 129.346]); layer 23 mean_act −0.49004 → alpha 1.00,
layer 22 mean_act +0.01375 → alpha **9274.12** (PC: 7867.694).
The blowup is not a one-off: the signed mean is a different tiny
number on each machine, but it is always tiny, so the alpha is
always either 1.0 or thousands.

**Refinement — mean|act| is NOT a safe robust alpha either.**
Mean|act| gives Qwen L27 2.38 / L26 5.71 (plausible) but SmolLM2
L23 **36.15** / L22 **85.81** — far above the 1.0 that actually
passed on SmolLM2, and far above the column norms (~0.5–1).
The mean|act| alpha varies wildly across models because direction
norms and activation scales differ. The safer normalizer is the
**column-norm cap**: alpha ≤ c × mean column norm (c ≈ 1) bounds
the edit relative to the column's own size, which is
model-independent. That cap would have given ~0.5–1.0 on both
models — matching the alphas that actually worked. This is
designed, not yet tested.

**Dampening is NOT universal — the causal sign predicts the edit.**
SmolLM2's causal proof was NEGATIVE (−30.00: zeroing hurt), and
dampening 0.5 on SmolLM2 HURT the failing subset (23.33 → 18.33,
good/neighbours unchanged). Qwen's causal proof was POSITIVE
(+15.56/+23.89: zeroing helped), and dampening PASSED the gate
there. The rule that fits both models: **if zeroing helps
(causal +), the columns are hurting the failing set, so turn them
down (dampening); if zeroing hurts (causal −), the columns are
helping, so do not turn them down — reshape them instead
(subtraction, which passed on SmolLM2).** The robust editor should
condition its attempt plan on the causal sign instead of always
running the same 3 attempts. Dampen 0.25/0.0 on SmolLM2 confirmed
the hurt (0.5: −5.00, 0.25: −11.67, 0.0: −8.33) — see §4c.

## 4b. Probe 2 — negative alphas, single-layer, contribution columns
(subset: failing[0] + good[0,1] + neighbours[0], greedy;
baseline failing[0] 26.67, good 100/100, neighbours 100)

| variant | failing[0] | good[0,1] | neighbours[0] |
|---|---|---|---|
| add α=0.5 (subtract α=−0.5) | 25.00 | 100/100 | 100 |
| add α=1.0 | 18.33 | 100/100 | 100 |
| add α=2.0 | 28.33 | 100/100 | 100 |
| add α=5.0 | 3.33 | 0/0 | 20 |
| subtract α=0.5, **layer 27 only** | 25.00 | 100/100 | 100 |
| subtract α=0.5, **layer 26 only** | **40.00 (+13.33)** | 100/100 | 100 |
| subtract α=0.5, contribution-cols | 25.00 | 100/100 | 100 |
| subtract α=1.0, contribution-cols | 10.00 | 100/100 | 100 |

Adding the direction is useless at best (0.5: −1.67, 1.0: −8.33,
2.0: +1.67 = noise) and destructive at 5.0 — symmetric to
subtracting. The direction, in either sign, is not the lever for
Qwen. Contribution-weighted column selection is a dead end: it
overlaps 45/64 (layer 27) and 46/64 (layer 26) with the existing
|df−dc| selection and its mean |contribution| is only 4–5% higher —
the columns were already near-optimal; the operation was the
problem.

**One real lead:** single-layer subtraction at α=0.5 on layer 26
alone improves failing[0] by +13.33 with no damage, while layer 27
alone slightly hurts (−1.67) and both together at 0.5 destroy
(8.33). The layers interact: layer 27's edit is harmful, layer
26's is helpful, and together the harmful one dominates. This is
UNCONFIRMED — one prompt, greedy only, no 0.7 check, no full-set
run. It is a follow-up, not a result. (SmolLM2's pass was also
single-layer — layer 23 alone — which makes the layer-specificity
plausible, but Qwen's layer-26 lead is far weaker than SmolLM2's
confirmed +9.44.)

## 4c. SmolLM2 dampening (generalisation check — subset)

HuggingFaceTB/SmolLM2-1.7B-Instruct, eval
`kit/evals/loop-fix-smollm2-1.7b.json`, subset failing[0] +
good[0,1] + neighbours[0], greedy. No-edit: failing[0] 23.33,
good 100/100, neighbours 100. (Note: failing[0] is the animals
prompt — the hardest one; the full-set greedy baseline on Teji's
PC was 56.67.)

| variant | failing[0] | good[0,1] | neighbours[0] |
|---|---|---|---|
| no-edit | 23.33 | 100/100 | 100 |
| dampen 0.5 | 18.33 (−5.00) | 100/100 | 100 |
| dampen 0.25 | 11.67 (−11.67) | 100/100 | 100 |
| dampen 0.0 | 15.00 (−8.33) | 100/100 | 100 |

Dampening HURTS SmolLM2 at every factor — the opposite of Qwen,
where it helped. This is the causal-sign rule (§4a): SmolLM2's
causal proof is negative, so its suspect columns help the failing
set and turning them down hurts. Dampening is a fix for the
causal-positive case only. (First probe run crashed after the 0.5
variant on a probe-script bug — `dampen_columns` returns
`(changed, saved)` and the script passed the whole pair to
`restore`; the kit itself is correct. Re-run completed cleanly
and reproduced every number exactly, including the 0.5 result.)

## 5. The patch (clone only — NOT applied to the live repo)

- `kit/wsp/real_edit.py`: new `apply_dampen(weight, columns, factor,
  layout)` — scales selected columns by a factor in [0, 1];
  factor 0 = zero them, 1 = no-op. Same cap discipline and
  clone-don't-mutate contract as `apply_column_edit`. The subtraction
  function and `alpha_base_from_signals` are left in place (used by
  `wsp apply-edit` and the tests).
- `kit/wsp/real_runner.py`: new `dampen_columns(layer, columns,
  factor)` mirroring `apply_edit` (1% cap check, returns saved
  columns for exact restore).
- `kit/wsp/real_pipeline.py`: attempt plan changed from
  `ATTEMPT_PLAN = ((1.0,"top2"), (2.0,"top2"), (1.0,"top1"))`
  (alpha multipliers) to
  `DAMPEN_PLAN = ((0.5,"top2"), (0.25,"top2"), (0.0,"top2"))`;
  attempts now call `dampen_columns`; the pass path re-applies
  dampening; `edit.json` records `edit_type: "dampen"` +
  `dampen_factor` (no direction/alpha); `_real_block` records the
  factor. Diagnosis, causal proof, gate, and the 1% cap are
  unchanged. The dead `alpha_bases` computation was removed.
- `kit/wsp/cli.py`: `wsp apply-edit` now handles both recipe types
  (dampen recipes and legacy subtraction recipes).
- `kit/tests/test_real.py`: 8 new tests (dampen scaling/zeroing/
  Conv1D layout/bad factor/bad layout; runner dampen + restore +
  cap + bad factor). Suite: **82 tests, all passing** (was 74).

## 6. Confirmation run (full pipeline, patched code) — GATE PASSED ✓

Command (from the build clone):
```
PYTHONPATH=kit <venv>/bin/python -m wsp.cli run \
  --model Qwen/Qwen3-0.6B \
  --eval kit/evals/loop-fix-qwen3-0.6b-narrow-animals.json \
  --out-dir wsp-real-out-qwen-narrow-dampen
```
Ran ~84 min, exit 0. Report + edited model saved in
`wsp-real-out-qwen-narrow-dampen/` (report.json, edit.json,
model.safetensors).

**Gate: PASSED (both temperatures).** Before → after @ serving 0.7:
failing **37.78 → 67.22 (+29.44)**, good 100 → 100, neighbours
100 → 100. Greedy: failing 24.44 → 50.00 (+25.56), good 100,
neighbours 93.33 → 100 (improved). **131,072 weights (0.022%)**,
layers [27, 26], dampen factor 0.25. First passing real-model edit
on Qwen; second passing real-model edit overall (after SmolLM2's
subtraction pass, §4a).

Attempt table (full sets):

| variant | factor | greedy failing | 0.7 failing | good | neighbours | passed |
|---|---|---|---|---|---|---|
| 1 | 0.5 | 36.11 (+11.67) | 34.44 (−3.33) | 100 | 100 | **No** (fails 0.7) |
| 2 | 0.25 | 50.00 (+25.56) | 67.22 (+29.44) | 100 | 100 | **Yes** |
| 3 | 0.0 | 40.00 (+15.56) | 22.78 (−15.00) | 100 | 100 | **No** (fails 0.7) |

Baselines in this run: failing 37.78 @0.7 (24.44 greedy), good 100,
neighbours 100 @0.7 (93.33 greedy), gap 62.22. Diagnosis layers
[27, 26], norms [48.839, 31.178] (identical to the probe). Causal
proof: failing subset 24.44 → 40.00 (+15.56) — smaller than the PC
run's +23.89 (VM greedy subset baseline is 24.44 vs PC's 28.89;
cross-platform bf16 noise — same conclusion, smaller number).

**The 0.7 dose-response is NOT monotone — there is a sweet spot.**
Dampen 0.5 helps at greedy (+11.67) but REGRESSES at 0.7 (−3.33).
Dampen 0.0 helps at greedy (+15.56) but REGRESSES hard at 0.7
(−15.00). Only 0.25 passes both. The gate correctly picks the best
passing attempt (variant 2); the other two are honestly reported as
failures. This is why the 3-attempt plan matters — a single factor
would have been a coin flip.

**Scorer honesty note:** the after-sample lists animals in
**bold** ("1. **Lion**", "3. **Bull**", "4. **Rhinoceros**"...)
and includes "11. **Penguin (again)**" — the distinct scorer counts
distinct STRINGS, so "Penguin" vs "Penguin (again)" score as two
items. Part of the +29 gain is formatting variation, but the
after-list also contains genuinely new animals (Bull, Rhinoceros,
Ox, Cheetah, Goat, Squirrel) absent from the before-list, so the
improvement is real; the exact +29 overstates it slightly. A
stricter scorer (normalising case/formatting/parentheticals) would
be a good follow-up, but it does not change the pass/fail: even
discounting the "again" items, the gain clears +5 by a wide margin.

## 7. What was NOT tried (and why) / what is next

- **Probe 2 — DONE** (§4b): negative alphas, single-layer
  subtraction, contribution columns. Only lead: layer-26-only
  subtraction at α=0.5 (+13.33 on one prompt) — needs a full-run
  confirmation before it means anything.
- **SmolLM2 dampening probe — DONE** (§4c,
  `experiments/probe_smol.py`, results
  `experiments/probe_smol_results.json`): dampening HURTS SmolLM2
  at every factor (0.5: −5.00, 0.25: −11.67, 0.0: −8.33). This is
  the generalisation check — and the answer is that dampening does
  NOT generalise to SmolLM2, which is exactly what the causal-sign
  rule predicts (SmolLM2 causal −). The robust editor is therefore
  NOT "dampening everywhere" but "dampening when causal +, 
  subtraction (sane alpha, single layer) when causal −".
- **Robust-alpha probe — NOT yet written:** test subtraction with
  alpha from mean|act| (Qwen L27 2.38 / L26 5.71), from the
  column-sum (0.037 / 0.089), and column-norm-capped, on the Qwen
  subset. Expectation: they behave like the 0.5–2.0 alphas already
  measured (§4) — i.e. they fail on Qwen — which would confirm that
  the alpha fix prevents blowups but does not by itself save Qwen.
  The definitive test of a robust alpha is on SmolLM2 (where
  subtraction passes at alpha 1.0): does the fixed alpha reproduce
  the pass AND avoid the layer-22 blowup?
- Per-position / late-position direction, orthogonalising the
  direction against good activations: orthogonalisation cannot help
  — the damage comes from the columns FIRING on good prompts
  (injecting −d), not from d's alignment; and the direction is
  already dominated by loop states. Not run.
- Runtime steering (add −d to the residual during generation):
  not a weight edit, out of scope for the kit.

## 8. Honest limits

- **Dampening is proven on ONE model (Qwen3-0.6B) and ONE eval
  (narrow-animals) at full-gate level (§6).** Subtraction is proven
  on SmolLM2 (§4a, Teji's PC, twice). Dampening was TESTED on
  SmolLM2 and HURTS there (§4c) — so neither edit generalises to
  both models on its own. The combined rule (causal sign picks the
  edit family) fits both models at subset level, but the COMBINED
  pipeline (causal-sign-conditioned attempts + fixed alpha) is
  designed, not yet built or tested as a single pipeline. Do not
  claim "works on any model" until that combined pipeline passes
  the gate on both models.
- The Qwen gate pass is real but the eval is small (3 failing + 4
  good + 3 neighbours). Single-prompt noise is large — the 0.7
  dose-response is non-monotone (§6) and the causal proof number
  differs across platforms (+15.56 VM vs +23.89 PC). A bigger eval
  (more prompts) would firm up the numbers.
- Dampening is a quieter, less "surgical" story than MIT's
  direction subtraction: it says "these columns hurt, turn them
  down" rather than "remove exactly the loop direction". The
  before/after video still works (failing 37.78 → 67.22 is a real,
  gate-passing improvement).
- **Scorer caveat (§6):** the distinct scorer counts distinct
  STRINGS, so "Penguin" vs "Penguin (again)" score as two items.
  The +29.44 gain overstates the improvement slightly; a stricter
  scorer (normalising formatting/parentheticals) is a good
  follow-up, but the gain clears the +5 bar by a wide margin even
  discounted.
- The alpha fix is PROPOSED, not yet tested — and mean|act| is
  already ruled out as the denominator (§4a: it gives 36/86 on
  SmolLM2). The current best candidate is the **column-norm cap**
  (alpha ≤ ~1× mean column norm), which would give ~0.5–1.0 on
  both models. Even a fixed alpha would not by itself have saved
  Qwen's subtraction (§4 shows 0.5–2.0 alphas fail). The robust
  editor = fixed alpha + causal-sign-conditioned attempts (dampening
  when causal +, subtraction when causal −), gate decides. That
  combination is designed, not yet built or tested as a single
  pipeline.
