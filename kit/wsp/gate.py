"""The eval gate. No gate, no ship.

Same prompts, same settings, before and after. The edit ships only if
the primary metric gains at least 5 points and no guard or neighbour
metric regresses by more than 2 points. Thresholds are starting
values: a task file may set its own, but it must say what it used.
"""

MIN_GAIN_POINTS = 5.0
MAX_REGRESS_POINTS = 2.0


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
