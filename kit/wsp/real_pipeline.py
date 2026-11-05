"""The universal real-model pipeline, in one place.

Same discipline as the kit pipeline in ``wsp.pipeline``, but nothing
is baked in: the model comes from a HuggingFace id, the prompts and
scorers from an eval file, the layers and columns from measured
activations. Steps, in order:

0. Temperature sweep on a failing subset (Step 0: the cheap knob).
   If some temperature alone restores the failing subset to within 5
   points of the good subset, stop — the cheap fix wins, no edit.
1. Baseline at every gate temperature (full sets, computed once and
   reused for every attempt — same prompts, same settings).
2. Diagnosis: capture MLP signals on failing vs neighbours subsets
   (same list shape, healthy — the good set is a different task
   shape and would confound the direction), pick the 2 layers with
   the largest activation-difference norm, and the top 64
   intermediate columns per layer by activation difference.
3. Causal proof: zero those columns, re-score the failing subset at
   greedy. If the failure does not move by at least 5 points, stop —
   the suspect is not a diagnosis.
4. Up to 3 edit attempts on a FIXED, documented search plan within
   the SKILL budget (variant 1: base alpha on the top 2 layers;
   variant 2: double alpha on the top 2 layers; variant 3: base
   alpha on the single strongest layer). Temperatures are evaluated
   greedy-first: if the gate already fails at greedy, the remaining
   temperatures are skipped for that attempt (a gate fail at any
   temperature is a gate fail overall, so no information is lost —
   only compute is saved). The BEST PASSING attempt ships, not the
   first passing one: among attempts that pass at every gate
   temperature, the one with the largest serving-temperature primary
   gain wins (tie-break: fewer weights changed). If no attempt
   passes, NOTHING ships — the report ranks all attempts, shows how
   far the best one missed the gate, and suggests what to try next.
   The gate rule is absolute: a failing edit is never shipped.
5. Report in schema v1 (same fields as the kit report) plus an
   additive ``real_demo`` block with the layers, columns, alphas and
   eval settings. Edited weights are saved only on a gate pass, to a
   new directory, never over the original files. The pipeline also
   refuses to edit a healthy model: if the baseline shows no real
   failure (failing set within 5 points of the good set at serving
   temperature), it stops before any edit — fix on failure only.

No LLM API is involved anywhere: the direction is a numerical
activation difference, and the editor field says so honestly.
"""

import json
import os

import numpy as np

from . import __version__, evalfile, gate, layers, report
from .real_edit import alpha_base_from_signals
from .scoring import score_text

CAUSAL_MIN_CHANGE = 5.0
CHEAP_FIX_MARGIN = 5.0
ALPHA_MULTIPLIERS = (0.5, 1.0, 2.0)
SWEEP_TEMPERATURES = (0.0, 0.7, 1.0, 1.5)
DIAGNOSIS_PER_CLASS = 6
SWEEP_SUBSET = 6
TOP_LAYERS = 2
TOP_COLUMNS = 64
# Diagnosis contrast: the set the failing activations are compared
# against. "neighbours" (same list shape, healthy) isolates the
# failure direction; "good" (short factual QA) confounds it with
# task shape. Approved design choice for this run; recorded in the
# report's real_demo block so runs stay comparable.
DIAGNOSIS_CONTRAST_SET = "neighbours"
# Fix-on-failure guard: a failing set this close to the good set is
# not a failure worth a weight edit (the gate needs a 5-point gain).
FAILURE_GAP_MARGIN = 5.0
# SKILL budget: at most 3 attempts, whatever the caller asks for.
MAX_ATTEMPTS_BUDGET = 3
# Fixed search plan (variant: alpha multiplier, layer selector).
# Variant 1 is the principled base (remove ~the observed excess on
# both diagnosed layers); variant 2 tests a stronger removal;
# variant 3 tests whether the second layer adds noise. The plan is
# fixed so runs are reproducible; adaptive search is future work.
ATTEMPT_PLAN = ((1.0, "top2"), (2.0, "top2"), (1.0, "top1"))


def failure_measured(baseline_serving):
    """True if the baseline shows a real failure at serving temp.

    Returns (measured, gap): gap is good-minus-failing. A failing set
    within FAILURE_GAP_MARGIN of the good set is not a failure worth
    editing — the model already handles it about as well as the good
    set, so there is nothing to fix.
    """
    gap = baseline_serving["good"] - baseline_serving["failing"]
    return gap > FAILURE_GAP_MARGIN, gap


def gate_distance(gate_by_temperature):
    """How far an attempt missed the gate, in points (0 = passed).

    For each evaluated temperature: the primary shortfall (how many
    points short of the minimum gain) plus, for each guard/neighbour,
    the excess regression (how many points past the allowed maximum).
    Attempts that skipped temperatures (greedy failed first) are
    measured on evaluated temperatures only; that is a lower bound
    on their true distance, and the report says which temperatures
    were skipped.
    """
    total = 0.0
    for g in gate_by_temperature.values():
        total += max(0.0, g["min_gain_points"] - g["primary_gain_points"])
        for r in g["regressions_points"].values():
            total += max(0.0, r - g["max_regress_points"])
    return total


def rank_attempts(attempts, serving):
    """Rank attempts for the report. Returns a list of entries.

    Passing attempts first, best first (largest serving-temperature
    primary gain, then fewest weights changed). Failing attempts
    next, closest to the gate first (smallest gate distance, then
    largest serving gain, then fewest weights). Each entry carries
    the attempt index so the report can point at the full record.
    """
    ranked = []
    for i, a in enumerate(attempts):
        gain_serv = None
        g = a["gate_by_temperature"].get(str(serving))
        if g is not None:
            gain_serv = g["primary_gain_points"]
        ranked.append({
            "index": i,
            # Human label: the variant number when the attempt record
            # carries one, else its position. Never use this as an
            # index — use "index" for that.
            "attempt": a.get("variant", i + 1),
            "passed": a["passed"],
            "distance_points": gate_distance(a["gate_by_temperature"])
            if not a["passed"] else 0.0,
            "primary_gain_serving": gain_serv,
            "weights_changed": a["weights_changed"],
            "evaluated_temperatures": a["evaluated_temperatures"],
        })
    passing = [r for r in ranked if r["passed"]]
    failing = [r for r in ranked if not r["passed"]]
    passing.sort(key=lambda r: (-(r["primary_gain_serving"] or 0.0),
                                r["weights_changed"]))
    failing.sort(key=lambda r: (r["distance_points"],
                                -(r["primary_gain_serving"] or -1e9),
                                r["weights_changed"]))
    return passing + failing


def select_best_passing(attempts, serving):
    """Index of the best passing attempt, or None if none passed."""
    ranked = rank_attempts(attempts, serving)
    if ranked and ranked[0]["passed"]:
        return ranked[0]["index"]
    return None


def suggest_next_steps(best_ranking, gate_by_temperature):
    """Honest next steps when no attempt passed the gate.

    Conditional on HOW the best attempt missed: primary shortfall
    suggests a different failure or a stronger edit; guard excess
    suggests a gentler edit. Always ends with the same rule: the
    gate decides, nothing ships.
    """
    suggestions = []
    missed_primary = False
    missed_guards = False
    for g in gate_by_temperature.values():
        if g["primary_gain_points"] < g["min_gain_points"]:
            missed_primary = True
        if any(r > g["max_regress_points"]
               for r in g["regressions_points"].values()):
            missed_guards = True
    if missed_primary:
        suggestions.append(
            "The diagnosed direction did not move the failure enough. "
            "Try a narrower failing set (one failure type, not mixed), "
            "more columns, or another layer — or accept that this "
            "failure is not a single-direction obstacle (MIT's Gemma "
            "control failed the same way).")
    if missed_guards:
        suggestions.append(
            "The edit helped the failing set but disturbed guards or "
            "neighbours. Try fewer columns, a smaller alpha, or a "
            "single layer.")
    suggestions.append(
        "No weights were shipped. Re-run with a revised eval file; "
        "the gate decides.")
    return suggestions


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def evaluate(model, data, temperature, samples_per_prompt, seed, log):
    """Score every set once. Greedy is deterministic, so extra samples
    at temperature 0 are skipped (they would be identical)."""
    max_tokens = data.get("max_new_tokens", 192)
    scores = {}
    for name in evalfile.SET_NAMES:
        scorer = evalfile.set_scorer(data, name)
        vals = []
        for pi, p in enumerate(evalfile.set_prompts(data, name)):
            reps = 1 if temperature == 0 else samples_per_prompt
            for s in range(reps):
                text, finished, _n = model.generate(
                    p["prompt"], max_tokens, temperature, seed + pi * 1000 + s
                )
                vals.append(score_text(scorer, text, p.get("expect"), finished,
                                       p.get("expect_items"),
                                       p.get("expect_count")))
        scores[name] = _mean(vals)
        log(f"    {name:<12} {scores[name]:6.2f}  (temperature {temperature})")
    return scores


def _subset_score(model, data, set_name, temperature, count, seed, log):
    scorer = evalfile.set_scorer(data, set_name)
    vals = []
    for pi, p in enumerate(evalfile.set_prompts(data, set_name)[:count]):
        text, finished, _n = model.generate(
            p["prompt"], data.get("max_new_tokens", 192), temperature,
            seed + pi * 1000,
        )
        vals.append(score_text(scorer, text, p.get("expect"), finished,
                               p.get("expect_items"), p.get("expect_count")))
    return _mean(vals)


def run_real_pipeline(model_id, eval_path, out_dir, dtype="bfloat16",
                      samples_per_prompt=1, seed=0, top_columns=TOP_COLUMNS,
                      max_attempts=3, thinking=False, log=print):
    """Run the full pipeline. Returns (report, out_dir, exit_code)."""
    # Eval file first: bad user input must fail fast, before the
    # (slow) model import and load.
    data = evalfile.load_eval_file(eval_path)
    # Status chip: mark this run as active so `wsp status` / `wsp
    # serve` can show RUNNING. Removed when the report is written
    # (see _write_report); a leftover marker is ignored once its
    # process is gone.
    from .status import write_running_marker
    write_running_marker(out_dir, model_id, eval_path)
    serving = data.get("serving_temperature", 0.7)
    gate_temps = data.get("gate_temperatures", [serving, 0.0])
    # SKILL requires the gate at serving temperature AND greedy, so
    # both are always evaluated even if the file lists only one of
    # them. The report records the temperatures actually evaluated.
    gate_temps = list(dict.fromkeys([serving, 0.0] + list(gate_temps)))
    max_tokens = data.get("max_new_tokens", 192)
    use_chat = data.get("use_chat_template", None)
    thinking = data.get("thinking", thinking)

    from .real_runner import RealModel

    log(f"loading {model_id} ({dtype}, CPU) ...")
    model = RealModel(model_id, dtype=dtype, use_chat_template=use_chat,
                      thinking=thinking)
    log(f"  {model.total_params:,} parameters, "
        f"{len(model.layer_infos)} decoder layers with MLP outputs")

    # Step 0: temperature sweep on subsets. A cheap fix only wins if some
    # temperature restores the failing subset WITHOUT degrading the good
    # subset at that same temperature — a knob that fixes loops by
    # wrecking answers is not a fix.
    log("Step 0: temperature sweep (failing + good subsets, cheap knob first)")
    # The sweep covers the standard knob range plus the serving
    # temperature itself, so the cheap-fix comparison never KeyErrors
    # on an unusual serving temperature.
    sweep_temps = tuple(dict.fromkeys(list(SWEEP_TEMPERATURES) + [serving]))
    sweep = {}
    sweep_good = {}
    for t in sweep_temps:
        sweep[t] = _subset_score(model, data, "failing", t, SWEEP_SUBSET,
                                 seed, log)
        sweep_good[t] = _subset_score(model, data, "good", t, SWEEP_SUBSET,
                                      seed, log)
        log(f"    temperature {t:<4} failing-subset {sweep[t]:6.2f}  "
            f"good-subset {sweep_good[t]:6.2f}")
    good_sub = sweep_good[serving]
    cheap_temps = [t for t in sweep_temps
                   if sweep[t] >= good_sub - CHEAP_FIX_MARGIN
                   and sweep_good[t] >= good_sub - CHEAP_FIX_MARGIN]
    if cheap_temps:
        best_temp = max(cheap_temps, key=lambda t: sweep[t])
        reason = (f"cheap fix wins: temperature {best_temp} alone restores "
                  f"the failing subset to {sweep[best_temp]:.2f} (good "
                  f"subset {good_sub:.2f} at serving, "
                  f"{sweep_good[best_temp]:.2f} at {best_temp}) — "
                  "no weight edit made")
        log(reason)
        rep = _fail_report(model_id, model, data, {}, {}, reason, sweep, log,
                           extra={
                               "temperature_sweep_subset": {str(k): v for k, v in sweep.items()},
                               "good_subset_serving": good_sub,
                           })
        _write_report(out_dir, rep)
        return rep, out_dir, 1

    # Step 1: baselines, once per gate temperature, reused by attempts.
    log("Step 1: baselines (full sets)")
    baseline = {}
    for t in gate_temps:
        log(f"  temperature {t}:")
        baseline[t] = evaluate(model, data, t, samples_per_prompt, seed, log)

    # Fix on failure only: if the baseline shows no real failure at
    # serving temperature, stop before any edit — a healthy model
    # must not be edited.
    measured, gap = failure_measured(baseline[serving])
    log(f"  failure gap at serving temperature: {gap:.2f} points")
    if not measured:
        reason = (f"no failure measured: failing set "
                  f"{baseline[serving]['failing']:.2f} is within "
                  f"{FAILURE_GAP_MARGIN} points of the good set "
                  f"{baseline[serving]['good']:.2f} at serving "
                  f"temperature — nothing to fix, no edit made")
        log(reason)
        rep = _fail_report(model_id, model, data, baseline[serving],
                           baseline[serving], reason, sweep, log,
                           extra={
                               "temperature_sweep_subset": {str(k): v for k, v in sweep.items()},
                               "failure_gap_points": gap,
                           })
        _write_report(out_dir, rep)
        return rep, out_dir, 1

    # Diagnosis: per-layer activation differences, failing vs the
    # contrast set (neighbours: same list shape, healthy — the good
    # set is a different task shape and would confound it).
    log(f"Step 1: diagnosis (MLP signals, failing vs {DIAGNOSIS_CONTRAST_SET})")
    layer_indices = [info["index"] for info in model.layer_infos]
    fail_caps = [model.capture(p["prompt"], max_tokens, layer_indices)
                 for p in evalfile.set_prompts(data, "failing")[:DIAGNOSIS_PER_CLASS]]
    contrast_caps = [model.capture(p["prompt"], max_tokens, layer_indices)
                     for p in evalfile.set_prompts(data, DIAGNOSIS_CONTRAST_SET)[:DIAGNOSIS_PER_CLASS]]
    per_layer = {}
    for i in layer_indices:
        mf = np.mean([c[i]["mlp_out"] for c in fail_caps], axis=0)
        mc = np.mean([c[i]["mlp_out"] for c in contrast_caps], axis=0)
        diff = mf - mc
        norm = float(np.linalg.norm(diff))
        df = np.mean([c[i]["down_in"] for c in fail_caps], axis=0)
        dc = np.mean([c[i]["down_in"] for c in contrast_caps], axis=0)
        per_layer[i] = {
            "direction": diff / norm if norm > 0 else diff,
            "direction_norm": norm,
            "col_scores": np.abs(df - dc),
            "mean_fail_down": df,
        }
    top_layers = sorted(per_layer, key=lambda i: per_layer[i]["direction_norm"],
                        reverse=True)[:TOP_LAYERS]
    columns = {}
    for i in top_layers:
        order = np.argsort(-per_layer[i]["col_scores"])[:top_columns]
        columns[i] = [int(c) for c in order]
    log(f"  picked layers {top_layers} "
        f"(direction norms {[round(per_layer[i]['direction_norm'], 3) for i in top_layers]})")

    # Causal proof: zero the suspect columns, failing subset at greedy.
    log("Step 1: causal proof (zero the suspect columns)")
    before_sub = _subset_score(model, data, "failing", 0, SWEEP_SUBSET,
                               seed, log)
    saved = [model.zero_columns(i, columns[i]) for i in top_layers]
    try:
        after_sub = _subset_score(model, data, "failing", 0, SWEEP_SUBSET,
                                  seed, log)
    finally:
        for s in saved:
            model.restore(s)
    change = after_sub - before_sub
    causal = abs(change) >= CAUSAL_MIN_CHANGE
    log(f"  failing subset {before_sub:.2f} -> {after_sub:.2f} "
        f"({change:+.2f} points) — causal: {causal}")
    if not causal:
        reason = (f"not causal: zeroing the suspect columns moved the "
                  f"failing subset by only {change:+.2f} points — no edit")
        rep = _fail_report(model_id, model, data, baseline[serving],
                           baseline[serving], reason, sweep, log,
                           extra={
                               "temperature_sweep_subset": {str(k): v for k, v in sweep.items()},
                               "causal_proof": {
                                   "failing_subset_before": before_sub,
                                   "failing_subset_after_ablation": after_sub,
                                   "change_points": change,
                               },
                               "diagnosis_contrast": DIAGNOSIS_CONTRAST_SET,
                               "diagnosis": _diagnosis_block(model, top_layers, columns, per_layer),
                           })
        _write_report(out_dir, rep)
        return rep, out_dir, 1

    # Step 2: edit attempts on the fixed search plan, within budget.
    # The BEST PASSING attempt ships, not the first passing one — so
    # every variant in budget is evaluated, even after a pass.
    alpha_bases = {}
    for i in top_layers:
        cols = columns[i]
        mean_act = float(np.mean(per_layer[i]["mean_fail_down"][cols]))
        alpha_bases[i] = alpha_base_from_signals(
            per_layer[i]["direction_norm"], mean_act)
    log(f"  alpha bases: { {i: round(alpha_bases[i], 3) for i in top_layers} }")
    n_attempts = max(0, min(max_attempts, MAX_ATTEMPTS_BUDGET))
    # Greedy-first order: 0.0 first when it is a gate temperature.
    eval_order = sorted(gate_temps, key=lambda t: (t != 0.0,
                                                  gate_temps.index(t)))
    attempts = []
    for vi, (mult, layer_sel) in enumerate(ATTEMPT_PLAN[:n_attempts]):
        attempt_layers = top_layers if layer_sel == "top2" else top_layers[:1]
        attempt_columns = {i: columns[i] for i in attempt_layers}
        attempt_alphas = {i: mult * alpha_bases[i] for i in attempt_layers}
        saved = []
        changed_total = 0
        for i in attempt_layers:
            c, s = model.apply_edit(i, attempt_columns[i],
                                    per_layer[i]["direction"],
                                    attempt_alphas[i])
            saved.append(s)
            changed_total += c
        # True percent, same as the kit pipeline since the 0.1.1
        # percent-label fix (0.77, not the fraction 0.0077).
        pct = changed_total / model.total_params * 100
        per_temp = {}
        scores_by_temp = {}
        evaluated = []
        passed = True
        for t in eval_order:
            log(f"  attempt {vi + 1} (alpha x{mult}, layers "
                f"{attempt_layers}, temperature {t}):")
            a = evaluate(model, data, t, samples_per_prompt, seed, log)
            scores_by_temp[str(t)] = a
            evaluated.append(t)
            g = gate.check_gate(baseline[t], a)
            per_temp[str(t)] = g
            if not g["passed"]:
                passed = False
                log(f"    gate failed at temperature {t} — skipping "
                    f"remaining temperatures for this attempt")
                break
        # Every attempt is restored after evaluation; the winner is
        # re-applied fresh below, so the saved model is exactly the
        # chosen attempt and nothing else.
        for s in saved:
            model.restore(s)
        attempts.append({
            "variant": vi + 1,
            "alpha_multiplier": mult,
            "alphas": {str(i): attempt_alphas[i] for i in attempt_layers},
            "layers": attempt_layers,
            "columns": {str(i): attempt_columns[i] for i in attempt_layers},
            "weights_changed": changed_total,
            "weights_changed_pct": pct,
            "gate_by_temperature": per_temp,
            "scores_by_temperature": scores_by_temp,
            "evaluated_temperatures": evaluated,
            "passed": passed and len(evaluated) == len(gate_temps),
        })
        log(f"  attempt {vi + 1}: {changed_total} weights ({pct:.2f}%) — "
            f"passed: {attempts[-1]['passed']}")

    before = baseline[serving]
    chosen_idx = select_best_passing(attempts, serving)
    if chosen_idx is None:
        # Gate rule is absolute: ship NOTHING, but show the ranking,
        # how far the best attempt missed, and what to try next.
        ranking = rank_attempts(attempts, serving)
        best = ranking[0] if ranking else None
        best_gate = (attempts[best["index"]]["gate_by_temperature"]
                     if best else {})
        suggestions = (suggest_next_steps(best, best_gate) if best
                       else ["No attempts were run."])
        reason = "no attempt passed the gate — nothing shipped"
        log(f"  ranking: {ranking}")
        if best is not None:
            log(f"  best attempt missed the gate by "
                f"{best['distance_points']:.2f} points")
        for s in suggestions:
            log(f"  suggestion: {s}")
        rep = _fail_report(model_id, model, data, before, before, reason,
                           sweep, log,
                           extra={
                               "temperature_sweep_subset": {str(k): v for k, v in sweep.items()},
                               "causal_proof": {
                                   "failing_subset_before": before_sub,
                                   "failing_subset_after_ablation": after_sub,
                                   "change_points": change,
                               },
                               "attempt_ranking": ranking,
                               "best_attempt": best,
                               "suggestions": suggestions,
                               "diagnosis_contrast": DIAGNOSIS_CONTRAST_SET,
                               "diagnosis": _diagnosis_block(model, top_layers, columns, per_layer),
                           })
        rep["attempts"] = attempts
        rep["attempts_used"] = len(attempts)
        _write_report(out_dir, rep)
        return rep, out_dir, 1

    chosen = attempts[chosen_idx]
    log(f"Step 3: best passing attempt is variant {chosen['variant']} "
        f"(serving primary gain "
        f"{chosen['gate_by_temperature'][str(serving)]['primary_gain_points']:+.2f})")
    # Reported "after" is the chosen attempt's own serving-temperature
    # scores (same prompts, same seeds) — no extra pass, no drift.
    after = chosen["scores_by_temperature"][str(serving)]
    # Before/after sample outputs (first failing prompt, serving temp):
    # model is currently original (every attempt was restored), so
    # sample, apply the chosen edit fresh, sample again.
    sample_prompt = evalfile.set_prompts(data, "failing")[0]["prompt"]
    before_text, _, _ = model.generate(sample_prompt, max_tokens, serving, seed)
    for i in chosen["layers"]:
        model.apply_edit(i, chosen["columns"][str(i)],
                         per_layer[i]["direction"],
                         chosen["alphas"][str(i)])
    after_text, _, _ = model.generate(sample_prompt, max_tokens, serving, seed)
    samples = {"prompt": sample_prompt,
               "before": before_text[:500], "after": after_text[:500]}
    log("  sample before/after captured (first failing prompt)")
    rep = report.make_report(
        model=f"{model_id} (real open-weight model, {dtype}, CPU)",
        edited_row=chosen["layers"][0],
        weights_changed=chosen["weights_changed"],
        weights_changed_pct=chosen["weights_changed_pct"],
        before=before,
        after=after,
        gate={"passed": True, "by_temperature": chosen["gate_by_temperature"]},
        temperatures_checked=list(sweep.keys()),
        editor=("numerical activation-difference estimator — no LLM; "
                "direction estimated from MLP activations"),
        tools=["torch", "transformers", "numpy"],
        attempts_used=len(attempts),
        attempts=attempts,
    )
    rep["real_demo"] = _real_block(
        model_id, model, data, dtype, chosen["layers"],
        {i: chosen["columns"][str(i)] for i in chosen["layers"]},
        per_layer, {i: chosen["alphas"][str(i)] for i in chosen["layers"]},
        sweep, good_sub, before_sub, after_sub, change,
        max_tokens, samples_per_prompt, seed, thinking, gate_temps, serving,
        samples)
    _write_report(out_dir, rep)
    edit_recipe = {
        "format_version": 1,
        "model_id": model_id,
        "dtype": dtype,
        "layers": [
            {
                "index": i,
                "columns": chosen["columns"][str(i)],
                "direction": per_layer[i]["direction"].tolist(),
                "alpha": chosen["alphas"][str(i)],
            }
            for i in chosen["layers"]
        ],
    }
    with open(os.path.join(out_dir, "edit.json"), "w") as f:
        json.dump(edit_recipe, f)
    model.save(out_dir)
    log(f"saved edited model to {out_dir}")
    return rep, out_dir, 0


def _diagnosis_block(model, top_layers, columns, per_layer):
    """Diagnosis layers/columns for fail-path real_demo blocks.

    The pass path records these in _real_block; fail paths previously
    did not, so a failed run could not say which layers were
    suspected. Cheap-fix / no-failure exits run before diagnosis and
    correctly record nothing here.
    """
    infos = {info["index"]: info for info in model.layer_infos}
    return {
        "layers": [
            {
                "index": i,
                "down_proj_name": infos[i]["down_proj_name"],
                "direction_norm": per_layer[i]["direction_norm"],
                "columns": columns[i],
            }
            for i in top_layers
        ],
    }


def _fail_report(model_id, model, data, before, after, reason, sweep, log,
                 extra=None):
    rep = report.make_report(
        model=f"{model_id} (real open-weight model, CPU)",
        edited_row=None,
        weights_changed=0,
        weights_changed_pct=0.0,
        before=before,
        after=after,
        gate={"passed": False, "reason": reason},
        temperatures_checked=list(sweep.keys()),
        editor=("numerical activation-difference estimator — no LLM; "
                "direction estimated from MLP activations"),
        tools=["torch", "transformers", "numpy"],
        attempts_used=0,
        attempts=[],
    )
    block = {
        "model_id": model_id,
        "device": "cpu",
        "total_params": model.total_params,
        "eval_name": data.get("name"),
        "eval_file": os.path.basename(data.get("_path", "")),
    }
    if extra:
        block.update(extra)
    rep["real_demo"] = block
    return rep


def _real_block(model_id, model, data, dtype, top_layers, columns, per_layer,
                alphas, sweep, good_sub, before_sub, after_sub, change,
                max_tokens, samples_per_prompt, seed, thinking, gate_temps,
                serving, samples=None):
    infos = {info["index"]: info for info in model.layer_infos}
    block = {
        "model_id": model_id,
        "dtype": dtype,
        "device": "cpu",
        "total_params": model.total_params,
        "layers": [
            {
                "index": i,
                "down_proj_name": infos[i]["down_proj_name"],
                "hidden_size": infos[i]["hidden_size"],
                "intermediate_size": infos[i]["intermediate_size"],
                "columns": columns[i],
                "direction_norm": per_layer[i]["direction_norm"],
                "alpha": alphas[i],
            }
            for i in top_layers
        ],
        "eval_name": data.get("name"),
        "eval_file": os.path.basename(data.get("_path", "")),
        "max_new_tokens": max_tokens,
        "samples_per_prompt": samples_per_prompt,
        "seed": seed,
        "thinking": thinking,
        "use_chat_template": model.use_chat_template,
        "gate_temperatures": gate_temps,
        "serving_temperature": serving,
        "diagnosis_contrast": DIAGNOSIS_CONTRAST_SET,
        "temperature_sweep_subset": {str(k): v for k, v in sweep.items()},
        "good_subset_serving": good_sub,
        "causal_proof": {
            "failing_subset_before": before_sub,
            "failing_subset_after_ablation": after_sub,
            "change_points": change,
        },
    }
    if samples is not None:
        block["samples"] = samples
    return block


def _write_report(out_dir, rep):
    # Every exit path writes the report here, so the running marker
    # is cleared here too — the status chip stops showing RUNNING
    # exactly when the run is done.
    from .status import remove_running_marker
    remove_running_marker(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
