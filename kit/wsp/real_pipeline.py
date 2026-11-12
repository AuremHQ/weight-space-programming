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

import hashlib
import json
import os
import time

import numpy as np

from . import __version__, evalfile, gate, layers, report
from .env_info import collect_environment
from .real_edit import alpha_base_from_signals, alpha_v2
from .scoring import score_text, score_text_v2

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
# This is the v1 plan, frozen: v1 keeps the legacy alpha and this
# plan. v2 uses the causal-sign plans below instead.
ATTEMPT_PLAN = ((1.0, "top2"), (2.0, "top2"), (1.0, "top1"))
# Combined attempt plans, v2 only (Phase 4, founder-tested): the
# edit FAMILY is conditioned on the causal proof's sign, because
# neither family generalises on its own (EDITOR-FIX-REPORT §4a/§4c).
# Each variant is (edit_type, dose, layer selector). For dampening,
# dose is the factor the suspect columns are scaled by (0.5/0.25/0.0
# on the top 2 layers — the proven Qwen plan, 0.25 passed the gate).
# For subtraction, dose is a MULTIPLIER on the per-layer
# column-norm-capped alpha (see capped_alpha_from_column_norm) —
# never the signed-mean formula, which returns 1.0 or thousands on
# the same model/eval. The first subtraction variant is top-1 only:
# the proven SmolLM2 shape (layer 23 only, alpha 1.0 passed the
# gate); the second tries both layers with sane alphas (the old
# top-2 attempt only destroyed the model because layer 22's alpha
# was 7867 — untested with a sane alpha); the third is a gentler
# single-layer dose. The plan is fixed per sign so runs stay
# reproducible; the gate decides. NOTE: the plan label is
# "subtraction"; the recipe edit_type is "subtract" (the vocabulary
# recipe.py has used since Phase 1) — the pipeline translates at
# the recipe boundary.
DAMPEN_PLAN = (("dampen", 0.5, "top2"), ("dampen", 0.25, "top2"),
               ("dampen", 0.0, "top2"))
SUBTRACT_PLAN = (("subtraction", 1.0, "top1"),
                 ("subtraction", 1.0, "top2"),
                 ("subtraction", 0.5, "top1"))


def select_attempt_plan(causal_change):
    """Attempt plan conditioned on the causal proof's sign.

    ``causal_change`` is the failing-subset score change from zeroing
    the suspect columns (after - before). Positive: zeroing helps, so
    the columns hurt -> dampen them. Negative: zeroing hurts, so the
    columns help -> subtract (reshape) with a sane alpha. The
    pipeline only calls this after the |change| >= 5 causal gate, so
    a change of exactly 0 is unreachable; it raises instead of
    guessing a family for a proof that says nothing.
    """
    if causal_change > 0:
        return DAMPEN_PLAN
    if causal_change < 0:
        return SUBTRACT_PLAN
    raise ValueError("causal change is 0 — no sign to condition on")


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


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _RunContext:
    """Additive report fields shared by both protocols (W2.12).

    Created once the model is loaded; ``stamp`` fills in the
    fields that every report carries (protocol, evidence level,
    verdict, editor info, environment, model revision, eval hash,
    wall time, known limitations) plus the v2-only instrumented
    fields when a CountingModel proxy is attached. It never
    changes an existing measurement value.
    """

    def __init__(self, protocol, eval_sha256, model, dtype, started,
                 serving, proxy=None):
        self.protocol = protocol
        self.eval_sha256 = eval_sha256
        self.model = model
        self.dtype = dtype
        self.started = started
        self.serving = serving
        self.proxy = proxy

    def stamp(self, rep, verdict):
        rep["protocol"] = self.protocol.name
        rep["evidence_level"] = ("standard"
                                 if self.protocol.name == "v2"
                                 else "quick")
        rep["gate"]["verdict"] = verdict
        rep["editor_info"] = {"type": "numerical_estimator",
                              "llm_used": False}
        rep["environment"] = collect_environment(
            device="cpu", dtype=self.dtype)
        rep["model_revision"] = getattr(self.model, "revision", None)
        rep["eval_sha256"] = self.eval_sha256
        rep["wall_seconds"] = time.monotonic() - self.started
        # Phase 4: the calibrated alpha (column-norm cap) and the
        # causal-sign rule are ported, so the Phase 2 alpha-estimator
        # known limitation is gone — no v2 report carries it.
        rep["known_limitations"] = []
        rep["generation_config_effective"] = {
            "temperature": self.serving,
            "top_p": None,
            "top_k": None,
            "repetition_penalty": None,
            "do_sample": self.serving != 0,
        }
        demo = rep.get("real_demo")
        if isinstance(demo, dict):
            demo["protocol_config"] = self.protocol.config_dict()
        if self.proxy is not None:
            rep["eval_passes_spent"] = self.proxy.eval_passes_spent
            rep["generated_tokens"] = self.proxy.generated_tokens
            rep["phase_prompt_counts"] = (
                self.proxy.phase_prompt_counts)


def evaluate_detailed(model, data, temperature, samples_per_prompt,
                      seed, log, split=None, scorer_version="v1",
                      generate_kwargs=None):
    """Score sets with per-prompt means (W2.5).

    Returns {set: {"per_prompt": [mean score per prompt],
    "mean": set mean}}. ``split`` is None (all prompts) or
    {set: [indices into the set]} — the sample seed stays
    ``seed + pi*1000 + s`` where ``pi`` is the prompt's index in
    the FULL set, so seeds do not depend on the split.
    ``scorer_version`` "v1" calls the untouched
    ``scoring.score_text``; "v2" calls ``score_text_v2`` with the
    prompt's optional ``valid_items``. ``generate_kwargs`` (e.g.
    repetition_penalty) are passed to ``model.generate`` only
    when non-empty, so existing 4-argument fakes keep working.
    Greedy is deterministic, so extra samples at temperature 0
    are skipped (they would be identical).
    """
    from .evalfile import SET_NAMES

    max_tokens = data.get("max_new_tokens", 192)
    kwargs = dict(generate_kwargs or {})
    out = {}
    for name in SET_NAMES:
        scorer = evalfile.set_scorer(data, name)
        prompts = evalfile.set_prompts(data, name)
        indices = (list(range(len(prompts))) if split is None
                   else split.get(name, []))
        per_prompt = []
        for pi in indices:
            p = prompts[pi]
            reps = 1 if temperature == 0 else samples_per_prompt
            vals = []
            valid = None
            if scorer_version == "v2":
                valid = evalfile.resolve_valid_items(data, name, p)
            for s in range(reps):
                text, finished, _n = model.generate(
                    p["prompt"], max_tokens, temperature,
                    seed + pi * 1000 + s, **kwargs)
                if scorer_version == "v2":
                    vals.append(score_text_v2(
                        scorer, text, p.get("expect"), finished,
                        p.get("expect_items"), p.get("expect_count"),
                        valid_items=valid))
                else:
                    vals.append(score_text(
                        scorer, text, p.get("expect"), finished,
                        p.get("expect_items"), p.get("expect_count")))
            per_prompt.append(_mean(vals))
        out[name] = {"per_prompt": per_prompt,
                     "mean": _mean(per_prompt)}
        log(f"    {name:<12} {out[name]['mean']:6.2f}  "
            f"(temperature {temperature})")
    return out


def evaluate(model, data, temperature, samples_per_prompt, seed, log):
    """Score every set once. Greedy is deterministic, so extra samples
    at temperature 0 are skipped (they would be identical)."""
    detailed = evaluate_detailed(model, data, temperature,
                                 samples_per_prompt, seed, log)
    return {name: detailed[name]["mean"]
            for name in evalfile.SET_NAMES}


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


def plan_counts(data, protocol, samples_per_prompt, gate_temps,
                 serving, guards_enabled=True):
    """Exact generate-call counts per phase (W2.14 / section 4.3).

    Returns an ordered dict {phase: calls} plus "total". For v2
    the numbers come from the splits; for v1 from the fixed
    subset sizes (SWEEP_SUBSET / DIAGNOSIS_PER_CLASS). Greedy is
    always 1 sample; non-greedy uses ``samples_per_prompt``.
    Attempts are counted in full (no early stop) — the plan is
    the worst case a full run spends.
    """
    from .splits import assign_splits

    def reps(temp):
        return 1 if temp == 0 else samples_per_prompt

    phases = {}
    if protocol.name == "v2":
        from .cheapfix import candidates as _candidates
        from .guards import load_refusal_probes

        splits = assign_splits(data, protocol, seed=0)
        n_select = sum(len(splits[name]["select"])
                       for name in evalfile.SET_NAMES)
        n_holdout = sum(len(splits[name]["holdout"])
                        for name in evalfile.SET_NAMES)
        n_select_failing = len(splits["failing"]["select"])
        n_diagnose = (len(splits["failing"]["diagnose"])
                      + len(splits["neighbours"]["diagnose"]))
        phases["cheap_screen"] = (
            # screen_candidates evaluates a baseline on select (1
            # sample) plus one evaluation per candidate.
            (len(_candidates(serving)) + 1) * n_select)
        phases["select"] = (n_select * sum(reps(t) for t in gate_temps)
                            + MAX_ATTEMPTS_BUDGET * n_select
                            * sum(reps(t) for t in gate_temps))
        phases["baseline"] = n_holdout * sum(reps(t)
                                             for t in gate_temps)
        phases["guards"] = (2 * len(load_refusal_probes())
                            if guards_enabled else 0)
        phases["diagnose"] = n_diagnose
        phases["causal"] = ((2 + protocol.causal_placebos)
                            * n_select_failing)
        phases["gate"] = (n_holdout * sum(reps(t) for t in gate_temps)
                          + 2)
    else:
        n_total = sum(len(evalfile.set_prompts(data, name))
                      for name in evalfile.SET_NAMES)
        sweep_temps = tuple(dict.fromkeys(
            list(SWEEP_TEMPERATURES) + [serving]))
        phases["cheap_screen"] = len(sweep_temps) * 2 * SWEEP_SUBSET
        phases["baseline"] = n_total * sum(reps(t)
                                           for t in gate_temps)
        phases["diagnose"] = 2 * DIAGNOSIS_PER_CLASS
        phases["causal"] = 2 * SWEEP_SUBSET
        phases["select"] = (MAX_ATTEMPTS_BUDGET * n_total
                            * sum(reps(t) for t in gate_temps))
        phases["gate"] = 2
    phases["total"] = sum(phases.values())
    return phases


def dry_run_plan(model_id, eval_path, protocol_name,
                 samples_per_prompt, seed, top_columns,
                 tokens_per_second, log):
    """Print the v2/v1 cost plan without loading anything (W2.14).

    Loads only the eval file. Prints prompt counts per set and
    split, the resolved protocol values, the exact number of
    generate calls per phase, and a total. With
    ``--tokens-per-second`` also prints a worst-case time
    estimate. Writes nothing and imports no torch. Returns 0.
    """
    from .protocol import get_protocol
    from .splits import assign_splits, preflight, split_counts

    protocol = get_protocol(protocol_name)
    samples = (protocol.samples_per_prompt
               if samples_per_prompt is None else samples_per_prompt)
    data = evalfile.load_eval_file(eval_path)
    serving = data.get("serving_temperature", 0.7)
    gate_temps = list(dict.fromkeys(
        [serving, 0.0]
        + list(data.get("gate_temperatures", [serving, 0.0]))))
    log(f"dry run: {model_id} on {data.get('name')} "
        f"(protocol {protocol.name}) — nothing is loaded, nothing "
        f"is written")
    for name in evalfile.SET_NAMES:
        count = len(evalfile.set_prompts(data, name))
        if protocol.name == "v2":
            counts = split_counts(data, protocol, seed)[name]
            log(f"  {name}: {count} prompts "
                f"(diagnose {counts['diagnose']}, "
                f"select {counts['select']}, "
                f"holdout {counts['holdout']})")
        else:
            log(f"  {name}: {count} prompts")
    log(f"  protocol values: samples_per_prompt={samples}, "
        f"gate_temperatures={gate_temps}, "
        f"serving_temperature={serving}, "
        f"min_gain_points={protocol.min_gain_points}, "
        f"max_regress_points={protocol.max_regress_points}, "
        f"top_columns={top_columns}, max_attempts="
        f"{MAX_ATTEMPTS_BUDGET}")
    if protocol.name == "v2":
        log(f"  minimum prompts: failing "
            f"{protocol.min_prompts['failing']}, neighbours "
            f"{protocol.min_prompts['neighbours']}, good "
            f"{protocol.min_prompts['good']}")
    problems = preflight(data, protocol)
    if problems:
        for problem in problems:
            log(f"  preflight would fail: {problem}")
    phases = plan_counts(data, protocol, samples, gate_temps,
                         serving)
    for phase, count in phases.items():
        if phase == "total":
            continue
        log(f"  {phase}: {count} generate calls")
    log(f"  total generate calls: {phases['total']}")
    if tokens_per_second:
        max_tokens = data.get("max_new_tokens", 192)
        worst_tokens = phases["total"] * max_tokens
        seconds = worst_tokens / float(tokens_per_second)
        log(f"  estimated time at {tokens_per_second:g} "
            f"tokens/s (worst case, {max_tokens} tokens per call): "
            f"~{seconds:.0f}s ({seconds / 60:.1f} min)")
    return 0


def run_real_pipeline(model_id, eval_path, out_dir, dtype="bfloat16",
                      samples_per_prompt=None, seed=0, top_columns=TOP_COLUMNS,
                      max_attempts=3, thinking=False, log=print, model=None,
                      force=False, protocol="v2", dry_run=False,
                      no_default_guards=False, tokens_per_second=None):
    """Run the full pipeline. Returns (report, out_dir, exit_code).

    Pass a model object (with the ``RealModel`` surface) as ``model``
    to skip loading — used by tests; production callers leave it None.
    ``force`` allows an out_dir that already holds model files.
    ``protocol`` selects v1 (frozen measurement, QUICK evidence)
    or v2 (evaluation integrity, standard evidence; the default).
    ``samples_per_prompt`` None resolves by protocol (v1: 1,
    v2: 3); greedy is always 1. ``dry_run`` prints the cost plan
    and returns (None, out_dir, 0) without loading or writing
    anything.
    """
    from .protocol import get_protocol

    proto = get_protocol(protocol)
    samples = (proto.samples_per_prompt if samples_per_prompt is None
               else samples_per_prompt)
    if dry_run:
        dry_run_plan(model_id, eval_path, protocol, samples, seed,
                     top_columns, tokens_per_second, log)
        return None, out_dir, 0
    if proto.name == "v1":
        return _run_v1(model_id, eval_path, out_dir, dtype=dtype,
                       samples_per_prompt=samples, seed=seed,
                       top_columns=top_columns, max_attempts=max_attempts,
                       thinking=thinking, log=log, model=model,
                       force=force)
    return _run_v2(model_id, eval_path, out_dir, dtype=dtype,
                   samples_per_prompt=samples, seed=seed,
                   top_columns=top_columns, max_attempts=max_attempts,
                   thinking=thinking, log=log, model=model,
                   force=force, no_default_guards=no_default_guards)


def _run_v1(model_id, eval_path, out_dir, dtype="bfloat16",
            samples_per_prompt=1, seed=0, top_columns=TOP_COLUMNS,
            max_attempts=3, thinking=False, log=print, model=None,
            force=False):
    """The frozen v1 pipeline (protocol v1).

    The body below is the pre-Phase-2 ``run_real_pipeline``,
    renamed without editing its measurement logic — the v1
    goldens prove it. Only additive report stamping was added
    (``ctx.stamp`` at each exit: protocol, evidence_level,
    QUICK verdict, editor info, environment, eval hash, wall
    time). Returns (report, out_dir, exit_code).

    Pass a model object (with the ``RealModel`` surface) as ``model``
    to skip loading — used by tests; production callers leave it None.
    ``force`` allows an out-dir that already holds model files.
    """
    started = time.monotonic()
    # Eval file first: bad user input must fail fast, before the
    # (slow) model import and load.
    data = evalfile.load_eval_file(eval_path)
    eval_sha = _file_sha256(eval_path)
    # File safety: refuse an out_dir that would overwrite model
    # weights, before loading anything (and before the running
    # marker creates the directory). An HF id resolves to its
    # cached snapshot dir when it is cached.
    from .safe_paths import assert_output_dir_safe, resolve_model_dirs

    assert_output_dir_safe(out_dir, resolve_model_dirs([model_id]),
                           force=force)
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

    if model is None:
        from .real_runner import RealModel

        log(f"loading {model_id} ({dtype}, CPU) ...")
        model = RealModel(model_id, dtype=dtype, use_chat_template=use_chat,
                          thinking=thinking)
        log(f"  {model.total_params:,} parameters, "
            f"{len(model.layer_infos)} decoder layers with MLP outputs")
    # Additive report fields (W2.12): the stamper fills them in at
    # each exit; it never changes a measurement value.
    from .protocol import get_protocol

    ctx = _RunContext(get_protocol("v1"), eval_sha, model, dtype,
                      started, serving)

    # Fail fast on an over-cap plan before ANY generation: the
    # largest possible attempt is TOP_LAYERS layers x top_columns
    # columns. Hidden/intermediate sizes are uniform across layers
    # in a transformer, so the max over layers is the exact bound
    # in practice (and a safe upper bound in general). The exact
    # recipe-level check still runs before the attempt loop, after
    # the diagnosis has picked the real layers and columns.
    from .real_edit import check_cap

    _hidden = max(int(i["hidden_size"]) for i in model.layer_infos)
    _inter = max(int(i["intermediate_size"])
                 for i in model.layer_infos)
    _plan_layers = min(TOP_LAYERS, len(model.layer_infos))
    _plan_cols = min(top_columns, _inter)
    check_cap(_plan_layers * _plan_cols * _hidden, model.total_params)

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
        ctx.stamp(rep, "QUICK_FAIL")
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
        ctx.stamp(rep, "QUICK_FAIL")
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
        ctx.stamp(rep, "QUICK_FAIL")
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
    # Recipe-level 1% cap, before any attempt is applied: the full
    # plan (all top layers) is the largest attempt, so if it passes
    # the cap, every attempt does.
    from .real_edit import check_recipe_cap

    check_recipe_cap(
        [{"index": i, "columns": columns[i]} for i in top_layers],
        {info["index"]: int(info["hidden_size"])
         for info in model.layer_infos},
        model.total_params)
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
            g = gate.check_gate(baseline[t], a,
                                min_gain=ctx.protocol.min_gain_points)
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
        ctx.stamp(rep, "QUICK_FAIL")
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
        gate={"passed": True,
              "by_temperature": chosen["gate_by_temperature"]},
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
    edit_recipe = {
        "format_version": 1,
        "model_id": model_id,
        "dtype": dtype,
        "model_revision": getattr(model, "revision", None),
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
    # The hash ties this edit to its gate pass: serve-model and
    # apply-edit refuse an edit whose report carries a different
    # hash. It is recorded in the report before the report is
    # written, so the report and the recipe always agree.
    from .recipe import edit_sha256, write_recipe

    rep["real_demo"]["edit_sha256"] = edit_sha256(edit_recipe)
    ctx.stamp(rep, "QUICK_PASS")
    _write_report(out_dir, rep)
    write_recipe(os.path.join(out_dir, "edit.json"), edit_recipe)
    model.save(out_dir, force=force)
    log(f"saved edited model to {out_dir}")
    return rep, out_dir, 0


def _measure_base_guards(proxy, log):
    """Collect the base guard measurements (W2.9).

    Runs under phase "guards": 16 refusal-probe greedy generate
    calls (counted as eval passes) plus 40 neutral-text nll/top1
    forward passes (not generate calls, not counted). Returns
    {"nlls", "top1s", "refusal_rate"}, or None when the model
    has no nll/top1. The refusal rate is on the 0-100 scale
    refusal_rate() returns, matching check_guards (whose 2-point
    regression limit is on that scale — a 0-1 fraction would make
    the guard toothless).
    """
    from .guards import (load_neutral_texts, load_refusal_probes,
                         refusal_rate)

    if not (hasattr(proxy, "nll") and hasattr(proxy, "top1")):
        return None
    neutral = load_neutral_texts()
    probes = load_refusal_probes()
    nlls = [proxy.nll(text) for text in neutral]
    top1s = [proxy.top1(text) for text in neutral]
    with proxy.phase("guards"):
        texts = []
        for probe in probes:
            text, _finished, _n = proxy.generate(probe, 64, 0.0, 0)
            texts.append(text)
    rate = refusal_rate(texts)
    log(f"  base guards: refusal rate {rate:.1f} on {len(probes)} "
        f"probes")
    return {"nlls": nlls, "top1s": top1s, "refusal_rate": rate}


def _measure_edited_guards(proxy, base, log):
    """The edited-model guard measurements vs the base (W2.9).

    Returns the check_guards dict, or {"status": "not evaluated"}
    when no base measurement exists (model without nll/top1).
    """
    from .guards import (check_guards, load_neutral_texts,
                         load_refusal_probes, refusal_rate)

    if base is None:
        return {"status": "not evaluated",
                "reason": "model has no nll/top1"}
    neutral = load_neutral_texts()
    probes = load_refusal_probes()
    nlls = [proxy.nll(text) for text in neutral]
    top1s = [proxy.top1(text) for text in neutral]
    with proxy.phase("guards"):
        texts = []
        for probe in probes:
            text, _finished, _n = proxy.generate(probe, 64, 0.0, 0)
            texts.append(text)
    result = check_guards(base["nlls"], nlls, base["top1s"], top1s,
                          base["refusal_rate"], refusal_rate(texts))
    log(f"  edited guards: perplexity ratio "
        f"{result['perplexity_ratio']:.3f}, top-1 agreement "
        f"{result['top1_agreement']:.3f}, refusal "
        f"{result['refusal_rate_after']:.2f} vs base "
        f"{result['refusal_rate_before']:.2f} — passed: "
        f"{result['passed']}")
    return result


def _real_block_v2(model_id, proxy, data, dtype, top_layers, columns,
                   per_layer, edit_params, layer_table,
                   causal, cheap_fix, edited_guards, max_tokens,
                   samples_per_prompt, seed, thinking, gate_temps,
                   serving, samples=None):
    """The real_demo block for a v2 SHIP report (W2.12).

    Same shape as _real_block, plus the v2-only fields: the full
    per-layer table with relative contrast, per-layer edit
    parameters (``edit_type`` plus ``dampen_factor`` for a dampen
    edit, or ``alpha`` / ``alpha_clamped`` / ``legacy_alpha`` for
    a subtraction), the causal proof, the cheap-fix screen, the
    edited guards, and the phase prompt counts (the leak test
    reads them here).
    """
    infos = {info["index"]: info for info in proxy.layer_infos}
    block = {
        "model_id": model_id,
        "dtype": dtype,
        "device": "cpu",
        "total_params": proxy.total_params,
        "layers": [
            {
                "index": i,
                "down_proj_name": infos[i]["down_proj_name"],
                "hidden_size": infos[i]["hidden_size"],
                "intermediate_size": infos[i]["intermediate_size"],
                "columns": columns[i],
                "direction_norm": per_layer[i]["direction_norm"],
                "relative_contrast":
                    per_layer[i]["relative_contrast"],
                **edit_params[i],
            }
            for i in top_layers
        ],
        "layer_table": layer_table,
        "eval_name": data.get("name"),
        "eval_file": os.path.basename(data.get("_path", "")),
        "max_new_tokens": max_tokens,
        "samples_per_prompt": samples_per_prompt,
        "seed": seed,
        "thinking": thinking,
        "use_chat_template": proxy.use_chat_template,
        "gate_temperatures": gate_temps,
        "serving_temperature": serving,
        "diagnosis_contrast": DIAGNOSIS_CONTRAST_SET,
        "causal_proof": causal,
        "cheap_fix": cheap_fix,
        "guards": edited_guards,
        "phase_prompt_counts": proxy.phase_prompt_counts,
    }
    if samples is not None:
        block["samples"] = samples
    return block


def _v2_fail_exit(ctx, rep, verdict, holdout_scores, guards, out_dir,
                  attempts=None):
    """Common v2 fail exit: holdout scores + guards + stamp + write."""
    rep["holdout_scores"] = holdout_scores
    rep["guards"] = guards
    if attempts is not None:
        rep["attempts"] = attempts
        rep["attempts_used"] = len(attempts)
    ctx.stamp(rep, verdict)
    _write_report(out_dir, rep)
    return rep, out_dir, 1


def _run_v2(model_id, eval_path, out_dir, dtype="bfloat16",
            samples_per_prompt=3, seed=0, top_columns=TOP_COLUMNS,
            max_attempts=3, thinking=False, log=print, model=None,
            force=False, no_default_guards=False):
    """The v2 pipeline (protocol v2, evaluation integrity).

    Flow: preflight (before the model loads and before the
    running marker) -> splits -> Step 0 cheap-fix screen on the
    select split -> Step 1 holdout baselines + base guards ->
    failure_measured on holdout -> diagnosis from the diagnose
    split, layer choice by relative contrast -> causal v2 on the
    select split -> recipe-level cap -> up to 3 attempts on the
    select split only (greedy first, point gate vs per-temp
    select baselines) -> ONE confirmation on holdout at every
    gate temperature with check_gate_v2. If confirmation fails,
    nothing ships (no fallback to a runner-up). Returns
    (report, out_dir, exit_code).
    """
    from .causal import causal_check
    from .cheapfix import describe as cheap_describe
    from .cheapfix import screen_candidates
    from .evalfile import SET_NAMES
    from .gate import check_gate, check_gate_v2, combine_gate_verdicts
    from .instrument import CountingModel
    from .protocol import get_protocol
    from .real_edit import (alpha_v2, capped_alpha_from_column_norm,
                          CAPPED_ALPHA_MAX, check_cap,
                          check_recipe_cap)
    from .recipe import ALPHA_HARD_MAX
    from .splits import assign_splits, preflight

    started = time.monotonic()
    proto = get_protocol("v2")
    # Eval file first: bad user input must fail fast, before the
    # (slow) model import and load.
    data = evalfile.load_eval_file(eval_path)
    eval_sha = _file_sha256(eval_path)
    # Preflight BEFORE the model loads and before the running
    # marker: a too-small eval fails with exit code 2 and writes
    # nothing (the CLI turns the ValueError into the exit code).
    problems = preflight(data, proto)
    if problems:
        raise ValueError(problems[0])
    splits = assign_splits(data, proto, seed)
    select_split = {name: splits[name]["select"]
                    for name in SET_NAMES}
    holdout_split = {name: splits[name]["holdout"]
                     for name in SET_NAMES}
    diagnose_split = {name: splits[name]["diagnose"]
                      for name in SET_NAMES}
    # File safety: refuse an out_dir that would overwrite model
    # weights, before loading anything (and before the running
    # marker creates the directory).
    from .safe_paths import assert_output_dir_safe, resolve_model_dirs

    assert_output_dir_safe(out_dir, resolve_model_dirs([model_id]),
                           force=force)
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

    if model is None:
        from .real_runner import RealModel

        log(f"loading {model_id} ({dtype}, CPU) ...")
        model = RealModel(model_id, dtype=dtype, use_chat_template=use_chat,
                          thinking=thinking)
        log(f"  {model.total_params:,} parameters, "
            f"{len(model.layer_infos)} decoder layers with MLP outputs")

    # The counting proxy wraps every generate/capture call from
    # here on, so the report can state exactly what was spent.
    proxy = CountingModel(model)
    ctx = _RunContext(proto, eval_sha, proxy, dtype, started, serving,
                      proxy=proxy)

    # Fail fast on an over-cap plan before ANY generation.
    _hidden = max(int(i["hidden_size"]) for i in proxy.layer_infos)
    _inter = max(int(i["intermediate_size"])
                 for i in proxy.layer_infos)
    _plan_layers = min(TOP_LAYERS, len(proxy.layer_infos))
    _plan_cols = min(top_columns, _inter)
    check_cap(_plan_layers * _plan_cols * _hidden, proxy.total_params)

    guards_enabled = not no_default_guards
    not_evaluated = {"status": "not evaluated",
                     "reason": "no edit was confirmed"}
    if not guards_enabled:
        guards_field = "disabled"
    else:
        guards_field = not_evaluated

    # Step 0: cheap-fix parity on the select split (1 sample
    # each). A cheap fix only wins if it restores the failing
    # select split WITHOUT degrading the good split at the same
    # config — the point gate (legacy check_gate) enforces that.
    log("Step 0: cheap-fix screen (select split, 1 sample each)")
    with proxy.phase("cheap_screen"):
        screen = screen_candidates(proxy, data, splits, serving, seed,
                                   log, scorer_version="v2")
    cheap_fix_record = {
        "screened": [
            {"candidate": entry["candidate"],
             "means": entry["means"],
             "passed": entry["passed"],
             "gain": entry["gain"]}
            for entry in screen["screened"]
        ],
        "baseline_select": screen["baseline_select"],
    }
    if screen["best"] is not None:
        best = screen["best"]
        label = cheap_describe(best)
        log(f"  cheap fix candidate: {label} — confirming on holdout")
        kwargs = {}
        if "repetition_penalty" in best:
            kwargs["repetition_penalty"] = best["repetition_penalty"]
        if "no_repeat_ngram_size" in best:
            kwargs["no_repeat_ngram_size"] = best["no_repeat_ngram_size"]
        with proxy.phase("gate"):
            base_hold = evaluate_detailed(
                proxy, data, serving, samples_per_prompt, seed, log,
                split=holdout_split, scorer_version="v2")
            cand_hold = evaluate_detailed(
                proxy, data, best["temperature"], samples_per_prompt,
                seed, log, split=holdout_split, scorer_version="v2",
                generate_kwargs=kwargs)
        confirm = check_gate_v2(
            {name: base_hold[name]["per_prompt"] for name in SET_NAMES},
            {name: cand_hold[name]["per_prompt"] for name in SET_NAMES},
            min_gain=proto.min_gain_points,
            max_regress=proto.max_regress_points,
            guard_margin=proto.guard_margin_points)
        cheap_fix_record["confirmation"] = confirm
        if confirm["verdict"] == "SHIP":
            reason = f"cheap fix wins: {label}"
            log(reason)
            before = {name: base_hold[name]["mean"]
                      for name in SET_NAMES}
            after = {name: cand_hold[name]["mean"]
                     for name in SET_NAMES}
            rep = _fail_report(model_id, proxy, data, before, after,
                               reason, {}, log,
                               extra={
                                   "cheap_fix": cheap_fix_record,
                                   "phase_prompt_counts":
                                       proxy.phase_prompt_counts,
                               })
            rep["temperatures_checked"] = [serving]
            holdout_scores = {
                "before": {str(serving): {
                    name: base_hold[name]["per_prompt"]
                    for name in SET_NAMES}},
                "after": {str(serving): {
                    name: cand_hold[name]["per_prompt"]
                    for name in SET_NAMES}},
            }
            return _v2_fail_exit(ctx, rep, "REJECT", holdout_scores,
                                 guards_field, out_dir)
        log(f"  cheap fix candidate did not confirm "
            f"({confirm['verdict']}) — continuing to the edit path")

    # Step 1: holdout baselines at every gate temperature, plus
    # the base guard measurements (unconditional in v2).
    log("Step 1: holdout baselines")
    with proxy.phase("baseline"):
        holdout_before = {}
        for t in gate_temps:
            log(f"  temperature {t}:")
            holdout_before[t] = evaluate_detailed(
                proxy, data, t, samples_per_prompt, seed, log,
                split=holdout_split, scorer_version="v2")
    holdout_before_means = {
        t: {name: holdout_before[t][name]["mean"]
            for name in SET_NAMES}
        for t in gate_temps
    }
    holdout_before_scores = {
        str(t): {name: holdout_before[t][name]["per_prompt"]
                 for name in SET_NAMES}
        for t in gate_temps
    }
    base_guards = None
    if guards_enabled:
        log("Step 1: base guards")
        base_guards = _measure_base_guards(proxy, log)

    # Fix on failure only: if the holdout baseline shows no real
    # failure at serving temperature, stop before any edit — a
    # healthy model must not be edited.
    measured, gap = failure_measured(holdout_before_means[serving])
    log(f"  failure gap at serving temperature: {gap:.2f} points")
    if not measured:
        reason = (f"no failure measured: failing set "
                  f"{holdout_before_means[serving]['failing']:.2f} "
                  f"is within {FAILURE_GAP_MARGIN} points of the "
                  f"good set "
                  f"{holdout_before_means[serving]['good']:.2f} at "
                  f"serving temperature — nothing to fix, no edit "
                  f"made")
        log(reason)
        rep = _fail_report(model_id, proxy, data,
                           holdout_before_means[serving],
                           holdout_before_means[serving], reason, {},
                           log,
                           extra={
                               "failure_gap_points": gap,
                               "phase_prompt_counts":
                                   proxy.phase_prompt_counts,
                           })
        rep["temperatures_checked"] = gate_temps
        return _v2_fail_exit(
            ctx, rep, "REJECT",
            {"before": holdout_before_scores, "after": None},
            guards_field, out_dir)

    # Diagnosis: per-layer activation differences on the diagnose
    # split, failing vs the contrast set (neighbours: same list
    # shape, healthy — the good set is a different task shape and
    # would confound it). Layer choice is by relative contrast:
    # direction norm over the mean of the failing and contrast
    # norms, so a layer is picked for standing out against its
    # own activity, not for raw size.
    log(f"Step 1: diagnosis (MLP signals, failing vs "
        f"{DIAGNOSIS_CONTRAST_SET}, diagnose split)")
    layer_indices = [info["index"] for info in proxy.layer_infos]
    failing_prompts = evalfile.set_prompts(data, "failing")
    contrast_prompts = evalfile.set_prompts(data, DIAGNOSIS_CONTRAST_SET)
    with proxy.phase("diagnose"):
        fail_caps = [proxy.capture(failing_prompts[i]["prompt"],
                                   max_tokens, layer_indices)
                     for i in diagnose_split["failing"]]
        contrast_caps = [proxy.capture(contrast_prompts[i]["prompt"],
                                       max_tokens, layer_indices)
                         for i in diagnose_split[DIAGNOSIS_CONTRAST_SET]]
    per_layer = {}
    for i in layer_indices:
        mf = np.mean([c[i]["mlp_out"] for c in fail_caps], axis=0)
        mc = np.mean([c[i]["mlp_out"] for c in contrast_caps], axis=0)
        diff = mf - mc
        norm = float(np.linalg.norm(diff))
        failing_norm = float(np.linalg.norm(mf))
        contrast_norm = float(np.linalg.norm(mc))
        denom = (failing_norm + contrast_norm) / 2.0
        relative = norm / denom if denom > 0 else 0.0
        df = np.mean([c[i]["down_in"] for c in fail_caps], axis=0)
        dc = np.mean([c[i]["down_in"] for c in contrast_caps], axis=0)
        per_layer[i] = {
            "direction": diff / norm if norm > 0 else diff,
            "direction_norm": norm,
            "failing_norm": failing_norm,
            "contrast_norm": contrast_norm,
            "relative_contrast": relative,
            "col_scores": np.abs(df - dc),
            "mean_fail_down": df,
        }
    layer_table = [
        {"index": i,
         "direction_norm": per_layer[i]["direction_norm"],
         "failing_norm": per_layer[i]["failing_norm"],
         "contrast_norm": per_layer[i]["contrast_norm"],
         "relative_contrast": per_layer[i]["relative_contrast"]}
        for i in sorted(per_layer,
                        key=lambda i: per_layer[i]["relative_contrast"],
                        reverse=True)
    ]
    top_layers = sorted(per_layer,
                        key=lambda i: per_layer[i]["relative_contrast"],
                        reverse=True)[:TOP_LAYERS]
    columns = {}
    for i in top_layers:
        order = np.argsort(-per_layer[i]["col_scores"])[:top_columns]
        columns[i] = [int(c) for c in order]
    log(f"  picked layers {top_layers} "
        f"(relative contrast "
        f"{[round(per_layer[i]['relative_contrast'], 3) for i in top_layers]})")

    # Causal proof (v2): zero the suspect columns, failing select
    # split at greedy, with K placebo controls. Since Phase 4 the
    # SIGN selects the edit family instead of refusing the run
    # (causal + -> dampen, causal - -> subtract with a sane alpha);
    # the check runs sign-agnostic ("either") and the holdout gate
    # stays the only arbiter. A proof that is too small, or no
    # bigger than the placebos, still refuses.
    log("Step 1: causal proof (zero the suspect columns, select split)")
    select_failing = select_split["failing"]

    def _select_failing_mean():
        with proxy.phase("causal"):
            det = evaluate_detailed(
                proxy, data, 0.0, 1, seed, log,
                split={"failing": select_failing},
                scorer_version="v2")
        return det["failing"]["mean"]

    causal = causal_check(
        proxy, _select_failing_mean, top_layers, columns,
        {info["index"]: int(info["intermediate_size"])
         for info in proxy.layer_infos},
        edit_type="either", seed=seed,
        k_placebos=proto.causal_placebos, log=log)
    if not causal["passed"]:
        reason = (f"not causal: zeroing the suspect columns on the "
                  f"select split moved the failing set by only "
                  f"{causal['effect']:+.2f} points (placebo max "
                  f"{causal['placebo_abs_max']:.2f}) — no edit")
        log(reason)
        rep = _fail_report(model_id, proxy, data,
                           holdout_before_means[serving],
                           holdout_before_means[serving], reason, {},
                           log,
                           extra={
                               "causal_proof": causal,
                               "diagnosis": _diagnosis_block(
                                   proxy, top_layers, columns, per_layer),
                               "phase_prompt_counts":
                                   proxy.phase_prompt_counts,
                           })
        rep["temperatures_checked"] = gate_temps
        return _v2_fail_exit(
            ctx, rep, "REJECT",
            {"before": holdout_before_scores, "after": None},
            guards_field, out_dir)

    # Step 2: edit attempts on the select split only, within
    # budget. The attempt family is conditioned on the causal
    # sign (Phase 4): causal + -> dampen variants (0.5, 0.25,
    # 0.0); causal - -> subtract with the column-norm-capped
    # alpha. The legacy alpha is still computed per layer as a
    # diagnostic (``legacy_alpha`` in the attempt record) so the
    # report shows what the old formula would have said — the
    # edit itself never uses it. Recipe-level 1% cap, before any
    # attempt is applied: the full plan (all top layers) is the
    # largest attempt, so if it passes the cap, every attempt does.
    plan = select_attempt_plan(causal["effect"])
    causal_sign = "+" if causal["effect"] > 0 else "-"
    log(f"  causal sign {causal_sign} ({causal['effect']:+.2f}) -> "
        f"{plan[0][0]} plan: {plan}")
    column_norms = {}
    alpha_caps = {}
    legacy_alphas = {}
    if plan[0][0] == "subtraction":
        for i in top_layers:
            column_norms[i] = proxy.mean_column_norm(i, columns[i])
            alpha_caps[i] = capped_alpha_from_column_norm(
                column_norms[i])
            cols = columns[i]
            mean_act = float(np.mean(
                per_layer[i]["mean_fail_down"][cols]))
            legacy, _clamped = alpha_v2(
                per_layer[i]["direction_norm"], mean_act)
            legacy_alphas[i] = legacy
        log(f"  subtraction alpha caps (column-norm): "
            f"{ {i: round(a, 3) for i, a in alpha_caps.items()} }")
        log(f"  legacy alphas (diagnostic only, not used): "
            f"{ {i: round(a, 3) for i, a in legacy_alphas.items()} }")
    check_recipe_cap(
        [{"index": i, "columns": columns[i]} for i in top_layers],
        {info["index"]: int(info["hidden_size"])
         for info in proxy.layer_infos},
        proxy.total_params)
    # Select baselines at every gate temperature (protocol
    # samples): the point gate for every attempt is measured
    # against these.
    log("Step 2: select baselines")
    with proxy.phase("select"):
        select_before = {}
        for t in gate_temps:
            log(f"  temperature {t}:")
            select_before[t] = evaluate_detailed(
                proxy, data, t, samples_per_prompt, seed, log,
                split=select_split, scorer_version="v2")
    select_before_means = {
        t: {name: select_before[t][name]["mean"]
            for name in SET_NAMES}
        for t in gate_temps
    }
    n_attempts = max(0, min(max_attempts, MAX_ATTEMPTS_BUDGET))
    # Greedy-first order: 0.0 first when it is a gate temperature.
    eval_order = sorted(gate_temps, key=lambda t: (t != 0.0,
                                                  gate_temps.index(t)))
    attempts = []
    for vi, (plan_edit_type, dose, layer_sel) in enumerate(
            plan[:n_attempts]):
        attempt_layers = (top_layers if layer_sel == "top2"
                          else top_layers[:1])
        attempt_columns = {i: columns[i] for i in attempt_layers}
        # The plan label is "subtraction"; the recipe edit_type is
        # "subtract" (recipe.py's vocabulary since Phase 1).
        edit_type = ("subtract" if plan_edit_type == "subtraction"
                     else plan_edit_type)
        attempt_alphas = {}
        attempt_clamped = {}
        saved = []
        changed_total = 0
        if edit_type == "dampen":
            for i in attempt_layers:
                c, s = proxy.dampen_columns(i, attempt_columns[i],
                                            dose)
                saved.append(s)
                changed_total += c
            desc = f"dampen x{dose}"
        else:  # subtraction with the column-norm-capped alpha
            for i in attempt_layers:
                attempt_alphas[i] = dose * alpha_caps[i]
                # True when the 1.0 cap bound (not the legacy
                # [1e-3, 100] clamp): the mean column norm
                # exceeded the only proven subtraction alpha.
                attempt_clamped[i] = (column_norms[i]
                                      > CAPPED_ALPHA_MAX)
            for i in attempt_layers:
                c, s = proxy.apply_edit(i, attempt_columns[i],
                                        per_layer[i]["direction"],
                                        attempt_alphas[i])
                saved.append(s)
                changed_total += c
            desc = ("subtract alpha "
                    f"{ {i: round(a, 3) for i, a in attempt_alphas.items()} }")
        # True percent, same as the kit pipeline since the 0.1.1
        # percent-label fix (0.77, not the fraction 0.0077).
        pct = changed_total / proxy.total_params * 100
        per_temp = {}
        scores_by_temp = {}
        evaluated = []
        passed = True
        for t in eval_order:
            log(f"  attempt {vi + 1} ({desc}, layers "
                f"{attempt_layers}, temperature {t}):")
            with proxy.phase("select"):
                det = evaluate_detailed(
                    proxy, data, t, samples_per_prompt, seed, log,
                    split=select_split, scorer_version="v2")
            after_means = {name: det[name]["mean"]
                           for name in SET_NAMES}
            scores_by_temp[str(t)] = after_means
            evaluated.append(t)
            g = check_gate(select_before_means[t], after_means,
                           min_gain=proto.min_gain_points)
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
            proxy.restore(s)
        record = {
            "variant": vi + 1,
            "edit_type": edit_type,
            "layers": attempt_layers,
            "columns": {str(i): attempt_columns[i]
                        for i in attempt_layers},
            "weights_changed": changed_total,
            "weights_changed_pct": pct,
            "gate_by_temperature": per_temp,
            "scores_by_temperature": scores_by_temp,
            "evaluated_temperatures": evaluated,
            "passed": passed and len(evaluated) == len(gate_temps),
        }
        if edit_type == "dampen":
            record["dampen_factor"] = dose
        else:
            record["alpha_multiplier"] = dose
            record["alphas"] = {str(i): attempt_alphas[i]
                                for i in attempt_layers}
            record["alpha_clamped"] = {str(i): attempt_clamped[i]
                                       for i in attempt_layers}
            record["legacy_alphas"] = {str(i): legacy_alphas[i]
                                       for i in attempt_layers}
        attempts.append(record)
        log(f"  attempt {vi + 1}: {changed_total} weights ({pct:.2f}%) "
            f"— passed: {attempts[-1]['passed']}")

    # Best passing attempt by select-split serving gain; ties keep
    # the earliest attempt.
    passing = [a for a in attempts if a["passed"]]
    chosen = None
    if passing:
        best_gain = None
        for a in passing:
            gain = a["gate_by_temperature"][str(serving)][
                "primary_gain_points"]
            if best_gain is None or gain > best_gain:
                best_gain = gain
                chosen = a
    if chosen is None:
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
        rep = _fail_report(model_id, proxy, data,
                           holdout_before_means[serving],
                           holdout_before_means[serving], reason, {},
                           log,
                           extra={
                               "causal_proof": causal,
                               "attempt_ranking": ranking,
                               "best_attempt": best,
                               "suggestions": suggestions,
                               "diagnosis_contrast":
                                   DIAGNOSIS_CONTRAST_SET,
                               "diagnosis": _diagnosis_block(
                                   proxy, top_layers, columns, per_layer),
                               "phase_prompt_counts":
                                   proxy.phase_prompt_counts,
                           })
        rep["temperatures_checked"] = gate_temps
        return _v2_fail_exit(
            ctx, rep, "REJECT",
            {"before": holdout_before_scores, "after": None},
            guards_field, out_dir, attempts=attempts)

    log(f"Step 3: best passing attempt is variant "
        f"{chosen['variant']} (serving primary gain "
        f"{chosen['gate_by_temperature'][str(serving)]['primary_gain_points']:+.2f})")
    # ONE confirmation on holdout at every gate temperature, with
    # the chosen edit applied. Sample before/after uses the first
    # HOLDOUT failing prompt (not the full-set first prompt).
    sample_prompt = evalfile.set_prompts(
        data, "failing")[holdout_split["failing"][0]]["prompt"]
    with proxy.phase("gate"):
        before_text, _, _ = proxy.generate(sample_prompt, max_tokens,
                                           serving, seed)
    saved_confirm = []
    for i in chosen["layers"]:
        if chosen["edit_type"] == "dampen":
            _c, handle = proxy.dampen_columns(
                i, chosen["columns"][str(i)],
                chosen["dampen_factor"])
        else:
            _c, handle = proxy.apply_edit(
                i, chosen["columns"][str(i)], per_layer[i]["direction"],
                chosen["alphas"][str(i)])
        saved_confirm.append(handle)
    with proxy.phase("gate"):
        holdout_after = {}
        for t in gate_temps:
            log(f"  confirmation temperature {t}:")
            holdout_after[t] = evaluate_detailed(
                proxy, data, t, samples_per_prompt, seed, log,
                split=holdout_split, scorer_version="v2")
        after_text, _, _ = proxy.generate(sample_prompt, max_tokens,
                                          serving, seed)
    samples = {"prompt": sample_prompt,
               "before": before_text[:500], "after": after_text[:500]}
    log("  sample before/after captured (first holdout failing prompt)")
    holdout_after_scores = {
        str(t): {name: holdout_after[t][name]["per_prompt"]
                 for name in SET_NAMES}
        for t in gate_temps
    }
    holdout_after_means = {
        t: {name: holdout_after[t][name]["mean"]
            for name in SET_NAMES}
        for t in gate_temps
    }
    by_temp = {}
    for t in gate_temps:
        by_temp[str(t)] = check_gate_v2(
            {name: holdout_before[t][name]["per_prompt"]
             for name in SET_NAMES},
            {name: holdout_after[t][name]["per_prompt"]
             for name in SET_NAMES},
            min_gain=proto.min_gain_points,
            max_regress=proto.max_regress_points,
            guard_margin=proto.guard_margin_points)
    combined = combine_gate_verdicts(
        [by_temp[str(t)]["verdict"] for t in gate_temps])
    log(f"  confirmation verdict: {combined}")
    edited_guards = None
    if guards_enabled:
        log("Step 3: edited guards")
        edited_guards = _measure_edited_guards(proxy, base_guards, log)
        if combined == "SHIP" and not edited_guards.get("passed", True):
            log("  guards failed — SHIP demoted to REJECT")
            combined = "REJECT"
    guards_out = (edited_guards if guards_enabled
                  else "disabled")
    if combined != "SHIP":
        # Confirmation failed: restore the edit and ship nothing
        # (no fallback to a runner-up).
        for handle in saved_confirm:
            proxy.restore(handle)
        if combined == "INCONCLUSIVE":
            reason = ("confirmation inconclusive on holdout: the "
                      "point estimates pass but a confidence bound "
                      "or sample size does not — nothing shipped")
        else:
            reason = ("confirmation failed on holdout: the edit did "
                      "not clear the gate — nothing shipped")
        log(reason)
        rep = _fail_report(model_id, proxy, data,
                           holdout_before_means[serving],
                           holdout_after_means[serving], reason, {},
                           log,
                           extra={
                               "causal_proof": causal,
                               "diagnosis": _diagnosis_block(
                                   proxy, top_layers, columns, per_layer),
                               "cheap_fix": cheap_fix_record,
                               "phase_prompt_counts":
                                   proxy.phase_prompt_counts,
                           })
        rep["temperatures_checked"] = gate_temps
        rep["gate"]["by_temperature"] = by_temp
        rep["gate"]["verdict"] = combined
        return _v2_fail_exit(
            ctx, rep, combined,
            {"before": holdout_before_scores,
             "after": holdout_after_scores},
            guards_out, out_dir, attempts=attempts)

    # SHIP: report the chosen attempt's own holdout scores (same
    # prompts, same seeds) — no extra pass, no drift.
    before = holdout_before_means[serving]
    after = holdout_after_means[serving]
    serving_gate = dict(by_temp[str(serving)])
    serving_gate["by_temperature"] = by_temp
    serving_gate["verdict"] = "SHIP"
    serving_gate["passed"] = True
    rep = report.make_report(
        model=f"{model_id} (real open-weight model, {dtype}, CPU)",
        edited_row=chosen["layers"][0],
        weights_changed=chosen["weights_changed"],
        weights_changed_pct=chosen["weights_changed_pct"],
        before=before,
        after=after,
        gate=serving_gate,
        temperatures_checked=gate_temps,
        editor=("numerical activation-difference estimator — no LLM; "
                "direction estimated from MLP activations"),
        tools=["torch", "transformers", "numpy"],
        attempts_used=len(attempts),
        attempts=attempts,
    )
    # Per-layer edit parameters for the report block and the
    # recipe: dampen layers carry the factor, subtract layers the
    # capped alpha (+ whether the 1.0 cap bound, + the legacy
    # alpha the old formula would have said, diagnostic only).
    edit_params = {}
    for i in chosen["layers"]:
        if chosen["edit_type"] == "dampen":
            edit_params[i] = {
                "edit_type": "dampen",
                "dampen_factor": chosen["dampen_factor"],
            }
        else:
            edit_params[i] = {
                "edit_type": "subtract",
                "alpha": chosen["alphas"][str(i)],
                "alpha_clamped": chosen["alpha_clamped"][str(i)],
                "legacy_alpha": chosen["legacy_alphas"][str(i)],
            }
    rep["real_demo"] = _real_block_v2(
        model_id, proxy, data, dtype, chosen["layers"],
        {i: chosen["columns"][str(i)] for i in chosen["layers"]},
        per_layer, edit_params,
        layer_table, causal, cheap_fix_record, edited_guards,
        max_tokens, samples_per_prompt, seed, thinking, gate_temps,
        serving, samples)
    edit_recipe = {
        "format_version": 1,
        "model_id": model_id,
        "dtype": dtype,
        "edit_type": chosen["edit_type"],
        "model_revision": getattr(proxy, "revision", None),
        "layers": [
            {
                "index": i,
                "columns": chosen["columns"][str(i)],
                **({"dampen_factor": chosen["dampen_factor"]}
                   if chosen["edit_type"] == "dampen"
                   else {"direction": per_layer[i]["direction"].tolist(),
                         "alpha": chosen["alphas"][str(i)]}),
            }
            for i in chosen["layers"]
        ],
    }
    # The hash ties this edit to its gate pass: serve-model and
    # apply-edit refuse an edit whose report carries a different
    # hash. It is recorded in the report before the report is
    # written, so the report and the recipe always agree.
    from .recipe import edit_sha256, write_recipe

    rep["real_demo"]["edit_sha256"] = edit_sha256(edit_recipe)
    rep["holdout_scores"] = {"before": holdout_before_scores,
                             "after": holdout_after_scores}
    rep["guards"] = guards_out
    ctx.stamp(rep, "SHIP")
    _write_report(out_dir, rep)
    write_recipe(os.path.join(out_dir, "edit.json"), edit_recipe)
    proxy.save(out_dir, force=force)
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
    # exactly when the run is done. The write is atomic: a crash
    # mid-write leaves the previous report intact.
    from .atomic import atomic_write_json
    from .status import remove_running_marker

    remove_running_marker(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    atomic_write_json(os.path.join(out_dir, "report.json"), rep)
