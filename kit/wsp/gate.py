"""The eval gate. No gate, no ship.

Same prompts, same settings, before and after. The edit ships only if
the primary metric gains at least 2 points and no guard or neighbour
metric regresses by more than 2 points. Thresholds are starting
values: a task file may set its own, but it must say what it used.
"""

MIN_GAIN_POINTS = 2.0
MAX_REGRESS_POINTS = 2.0
# Non-inferiority margin for the v2 guard lower bounds: a guard
# set may regress by up to this many points at the lower bound.
GUARD_MARGIN_POINTS = 5.0


def check_gate(before, after, primary="failing", guards=("good", "neighbours"),
               min_gain=MIN_GAIN_POINTS, max_regress=MAX_REGRESS_POINTS):
    gain = after[primary] - before[primary]
    regressions = {g: before[g] - after[g] for g in guards}
    passed = gain >= min_gain and all(r <= max_regress for r in regressions.values())
    return {
        "passed": passed,
        "primary": primary,
        "primary_gain_points": gain,
        "regressions_points": regressions,
        "min_gain_points": min_gain,
        "max_regress_points": max_regress,
    }


def check_gate_v2(before, after, primary="failing",
                  guards=("good", "neighbours"),
                  min_gain=MIN_GAIN_POINTS,
                  max_regress=MAX_REGRESS_POINTS,
                  guard_margin=GUARD_MARGIN_POINTS):
    """The v2 gate: point estimates AND one-sided 95% bounds.

    ``before`` and ``after`` map set name to a list of per-prompt
    mean scores, aligned by prompt index. Keeps every legacy key
    (so rank_attempts, gate_distance, suggest_next_steps, status
    and loop keep working) and adds ``verdict``, ``n_prompts``,
    ``primary_ci`` and ``guard_ci``.

    SHIP only when the point rule holds (primary gain >= 2, no
    guard regressing more than 2) AND the primary paired lower
    bound is > 0 AND every guard paired lower bound is >=
    -``guard_margin``. INCONCLUSIVE when the point estimates pass
    but a bound or sample size fails (n < 2 gives no bound).
    Otherwise REJECT. ``check_gate`` (legacy) is untouched.
    """
    from .stats import paired_lower_bound

    if primary not in before or primary not in after:
        raise ValueError(
            f"gate needs set {primary!r} in before and after")
    for g in guards:
        if g not in before or g not in after:
            raise ValueError(
                f"gate needs guard set {g!r} in before and after")
    primary_deltas = [a - b for b, a in zip(before[primary],
                                            after[primary])]
    primary_ci = paired_lower_bound(primary_deltas)
    guard_ci = {}
    for g in guards:
        deltas = [a - b for b, a in zip(before[g], after[g])]
        guard_ci[g] = paired_lower_bound(deltas)
    gain = primary_ci["mean"]
    regressions = {g: -guard_ci[g]["mean"] for g in guards}
    point_pass = (gain >= min_gain
                  and all(r <= max_regress
                          for r in regressions.values()))
    bounds_ok = (primary_ci["lower"] is not None
                 and primary_ci["lower"] > 0
                 and all(guard_ci[g]["lower"] is not None
                         and guard_ci[g]["lower"] >= -guard_margin
                         for g in guards))
    if point_pass and bounds_ok:
        verdict = "SHIP"
    elif point_pass:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "REJECT"
    return {
        "passed": verdict == "SHIP",
        "primary": primary,
        "primary_gain_points": gain,
        "regressions_points": regressions,
        "min_gain_points": min_gain,
        "max_regress_points": max_regress,
        "verdict": verdict,
        "n_prompts": {name: len(before[name])
                      for name in [primary] + list(guards)},
        "primary_ci": primary_ci,
        "guard_ci": guard_ci,
    }


def combine_gate_verdicts(verdicts):
    """One verdict across gate temperatures.

    SHIP only when every temperature SHIPs; INCONCLUSIVE when
    any is INCONCLUSIVE and none REJECTs; otherwise REJECT.
    """
    if not verdicts:
        return "REJECT"
    if all(v == "SHIP" for v in verdicts):
        return "SHIP"
    if any(v == "REJECT" for v in verdicts):
        return "REJECT"
    return "INCONCLUSIVE"
