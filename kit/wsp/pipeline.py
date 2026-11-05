"""The full pipeline, in one place, so the CLI and the MCP server run
the exact same steps instead of two copies that drift apart.

Steps: baseline, temperature sweep (Step 0), causal proof, direction
estimate, then up to 3 edit attempts. The gate must pass at the serving
temperature AND at greedy (temperature 0). Anything else is a discard.
"""

from . import __version__, diagnose, edit, gate, report, toy

SERVING_TEMPERATURE = 0.7
GATE_TEMPERATURES = (0.7, 0.0)
ALPHA_CANDIDATES = (1.5, 3.0, 4.5)
MAX_ATTEMPTS = 3


def run_pipeline():
    clean = toy.build_clean_weights()
    sets = toy.build_sets()
    targets = toy.clean_targets(clean, sets)
    broken, _ = toy.plant_defect(clean, sets["failing"])

    sweep = diagnose.temperature_sweep(broken, sets, targets)
    cause = diagnose.prove_cause(broken, sets, targets)
    if not cause["causal"]:
        raise RuntimeError(
            "no causal suspect found — stopping, nothing edited. "
            f"Ablation moved the failing set by only {cause['change_points']:.1f} points."
        )
    direction = diagnose.estimate_direction(broken, sets)

    attempts = []
    chosen = None
    for alpha in ALPHA_CANDIDATES[:MAX_ATTEMPTS]:
        candidate, changed, frac = edit.apply_edit(
            broken, toy.DEFECT_CLASS, direction, alpha
        )
        per_temp = {}
        passed = True
        for t in GATE_TEMPERATURES:
            b = diagnose.baseline(broken, sets, targets, t)
            a = diagnose.baseline(candidate, sets, targets, t)
            g = gate.check_gate(b, a)
            per_temp[str(t)] = g
            passed = passed and g["passed"]
        attempts.append({
            "alpha": alpha,
            "weights_changed": changed,
            # True percent (0.77), not the fraction (0.0077): the
            # 0.1.1 percent-label fix, preserved in v0.2.0.
            "weights_changed_pct": frac * 100,
            "gate_by_temperature": per_temp,
            "passed": passed,
        })
        if passed:
            chosen = (candidate, changed, frac)
            break

    before = diagnose.baseline(broken, sets, targets, SERVING_TEMPERATURE)
    if chosen is None:
        rep = report.make_report(
            model="toy (2-layer MLP, 64-128-64, seeded — not a language model)",
            edited_row=None,
            weights_changed=0,
            weights_changed_pct=0.0,
            before=before,
            after=before,
            gate={"passed": False, "reason": "no attempt passed the gate"},
            temperatures_checked=list(sweep.keys()),
            editor="toy estimator — no LLM; direction estimated from hidden activations",
            tools=["numpy"],
            attempts_used=len(attempts),
            attempts=attempts,
        )
        return rep, None, before, before, attempts, sweep, cause

    candidate, changed, frac = chosen
    after = diagnose.baseline(candidate, sets, targets, SERVING_TEMPERATURE)
    rep = report.make_report(
        model="toy (2-layer MLP, 64-128-64, seeded — not a language model)",
        edited_row=toy.DEFECT_CLASS,
        weights_changed=changed,
        weights_changed_pct=frac * 100,
        before=before,
        after=after,
        gate={"passed": True, "by_temperature": attempts[-1]["gate_by_temperature"]},
        temperatures_checked=list(sweep.keys()),
        editor="toy estimator — no LLM; direction estimated from hidden activations",
        tools=["numpy"],
        attempts_used=len(attempts),
        attempts=attempts,
    )
    return rep, candidate, before, after, attempts, sweep, cause
