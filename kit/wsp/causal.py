"""Causal proof, v2 (F03): expected sign plus placebo controls.

Zeroing the suspect columns must move the failure by at least 5
points, and by more than the largest absolute effect of K seeded
random same-size column sets in the same layers. Since Phase 4 the
SIGN selects the edit family instead of refusing the run
(EDITOR-FIX-REPORT §4a/§4c): causal + (zeroing helps, the columns
hurt) -> dampen; causal - (zeroing hurts, the columns help) ->
subtract with a sane alpha. So the pipeline calls this check with
``edit_type="either"`` (no sign required, magnitude + placebo
only) and picks the family from the returned sign. Direct callers
can still require a sign: ``"dampen"`` expects
``removal_improves``, ``"subtract"`` expects ``removal_worsens``.
v1 keeps abs() (the F03 hole stays visible in v1 only). Restore
runs in ``finally``, always — even when scoring raises.
"""

import random

EXPECTED_SIGN = {
    "subtract": "removal_worsens",
    "dampen": "removal_improves",
    "either": None,
}


def causal_check(model, score_fn, layers, columns, intermediate_sizes,
                 edit_type="subtract", seed=0, k_placebos=10, log=print):
    """Run the v2 causal check.

    ``score_fn()`` returns the mean score on the select split at
    greedy (a closure over the model). ``layers`` lists the
    suspect layers; ``columns`` maps layer -> [column indices];
    ``intermediate_sizes`` maps layer -> intermediate size.
    ``edit_type`` selects the required sign: ``"subtract"`` and
    ``"dampen"`` require their family's sign, ``"either"`` (the
    pipeline's choice) requires no sign — the sign then selects
    the edit type. Returns {"before", "after", "effect", "sign",
    "placebo_effects", "placebo_abs_max", "passed"}.
    """
    expected = EXPECTED_SIGN.get(edit_type, "removal_improves")
    before = score_fn()
    saved = [model.zero_columns(i, columns[i]) for i in layers]
    try:
        after = score_fn()
    finally:
        for handle in saved:
            model.restore(handle)
    effect = after - before
    sign = "removal_improves" if effect > 0 else "removal_worsens"
    rng = random.Random(seed)
    placebo_effects = []
    for _ in range(k_placebos):
        placebo_columns = {}
        for i in layers:
            size = len(columns[i])
            placebo_columns[i] = rng.sample(
                range(intermediate_sizes[i]), size)
        saved = [model.zero_columns(i, placebo_columns[i])
                 for i in layers]
        try:
            placebo_after = score_fn()
        finally:
            for handle in saved:
                model.restore(handle)
        placebo_effects.append(placebo_after - before)
    placebo_abs_max = max((abs(e) for e in placebo_effects),
                          default=0.0)
    passed = ((expected is None or sign == expected)
              and abs(effect) >= 5.0
              and abs(effect) > placebo_abs_max)
    log(f"  causal v2: {before:.2f} -> {after:.2f} "
        f"({effect:+.2f} points, {sign}); placebo max "
        f"{placebo_abs_max:.2f} over {k_placebos} draws — "
        f"passed: {passed}")
    return {
        "before": before,
        "after": after,
        "effect": effect,
        "sign": sign,
        "placebo_effects": placebo_effects,
        "placebo_abs_max": placebo_abs_max,
        "passed": passed,
    }
