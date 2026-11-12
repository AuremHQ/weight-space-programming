"""Cheap-fix parity, v2 (F06): the same gate as an edit.

Candidates: temperatures {0.0, serving, 1.0, 1.5} plus
repetition_penalty {1.1, 1.2} and no_repeat_ngram_size {3, 4} at
the serving temperature. All candidates are screened on the
select split (1 sample each); only the best is confirmed on
holdout with the same ``check_gate_v2``. If it ships-equivalent
the run exits 1 with "cheap fix wins: <setting>" and no edit.
v1 keeps the old temperature-only rule.
"""

from . import evalfile, gate


def candidates(serving):
    """The cheap-fix candidate configs, in screen order.

    Each is a dict: {"temperature": t} plus, for the knob
    candidates, "repetition_penalty" or "no_repeat_ngram_size".
    The serving temperature itself is included (it is the
    baseline config, so it can never win — but it is screened
    like the rest, and its scores ARE the select baseline).
    """
    temps = []
    for t in (0.0, serving, 1.0, 1.5):
        if t not in temps:
            temps.append(t)
    out = [{"temperature": t} for t in temps]
    for penalty in (1.1, 1.2):
        out.append({"temperature": serving,
                    "repetition_penalty": penalty})
    for ngram in (3, 4):
        out.append({"temperature": serving,
                    "no_repeat_ngram_size": ngram})
    return out


def describe(candidate):
    """A short human label for a candidate config."""
    label = f"temperature {candidate['temperature']}"
    if "repetition_penalty" in candidate:
        label += (f" + repetition_penalty "
                  f"{candidate['repetition_penalty']}")
    if "no_repeat_ngram_size" in candidate:
        label += (f" + no_repeat_ngram_size "
                  f"{candidate['no_repeat_ngram_size']}")
    return label


def _generate_kwargs(candidate):
    """The generate kwargs for a candidate (only when set)."""
    kwargs = {}
    if "repetition_penalty" in candidate:
        kwargs["repetition_penalty"] = candidate["repetition_penalty"]
    if "no_repeat_ngram_size" in candidate:
        kwargs["no_repeat_ngram_size"] = candidate["no_repeat_ngram_size"]
    return kwargs


def screen_candidates(model, data, splits, serving, seed, log,
                      scorer_version="v2"):
    """Screen every candidate on the select split (1 sample each).

    Returns {"best", "baseline_select", "screened"} where "best"
    is the winning candidate dict (or None), "baseline_select"
    is the select-split mean scores at the serving config, and
    "screened" is a list of {candidate, means, passed, gain} in
    screen order. The best is the largest serving-temperature
    primary gain among candidates passing the point gate on
    select; ties keep the earliest candidate.
    """
    from .real_pipeline import evaluate_detailed

    select = {name: splits[name]["select"]
              for name in evalfile.SET_NAMES}
    baseline = evaluate_detailed(model, data, serving, 1, seed, log,
                                 split=select,
                                 scorer_version=scorer_version)
    baseline_select = {name: baseline[name]["mean"]
                       for name in evalfile.SET_NAMES}
    screened = []
    best = None
    best_gain = None
    for candidate in candidates(serving):
        det = evaluate_detailed(
            model, data, candidate["temperature"], 1, seed, log,
            split=select, scorer_version=scorer_version,
            generate_kwargs=_generate_kwargs(candidate))
        means = {name: det[name]["mean"]
                 for name in evalfile.SET_NAMES}
        g = gate.check_gate(baseline_select, means)
        gain = g["primary_gain_points"]
        screened.append({"candidate": candidate, "means": means,
                         "passed": g["passed"], "gain": gain})
        log(f"    cheap screen {describe(candidate)}: "
            f"failing {means['failing']:.2f} "
            f"(gain {gain:+.2f}) — point gate: {g['passed']}")
        if g["passed"] and (best_gain is None or gain > best_gain):
            best = candidate
            best_gain = gain
    return {"best": best, "baseline_select": baseline_select,
            "screened": screened}
