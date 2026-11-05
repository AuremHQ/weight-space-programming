"""Diagnosis: baseline, cheap-fix sweep, causal proof, direction estimate.

These functions work on any model object with the same two methods as
the kit: ``forward(X)`` returning ``(logits, hidden)`` and nothing else
fancy. A real-model runner just needs those two methods; the rest of
the kit does not care what is behind them.
"""

import numpy as np

from . import kit_test

CAUSAL_MIN_CHANGE = 5.0


def baseline(weights, sets, targets, temperature):
    return {
        name: kit_test.expected_accuracy(weights, X, targets[name], temperature)
        for name, X in sets.items()
    }


def temperature_sweep(weights, sets, targets, temperatures=(0.0, 0.7, 1.0, 1.5)):
    """Step 0, done properly: try the cheap knob first, at several
    settings, before any weight is touched. If some temperature already
    restores the failing set, there is no weight edit to do."""
    return {t: baseline(weights, sets, targets, t) for t in temperatures}


def mean_hidden(weights, X):
    _, h = kit_test.forward(weights, X)
    return h.mean(axis=0)


def estimate_direction(weights, sets):
    """Direction of the failure, estimated from data: mean hidden
    activation on failing inputs minus mean hidden on good inputs.
    This is a heuristic, not a proof — which is why the causal test
    below exists."""
    d = mean_hidden(weights, sets["failing"]) - mean_hidden(weights, sets["good"])
    norm = np.linalg.norm(d)
    if norm == 0:
        raise ValueError("no direction found: failing and good activations match")
    return d / norm


def prove_cause(weights, sets, targets, row=kit_test.DEFECT_CLASS):
    """Ablation test. Temporarily zero the suspect row and see whether
    the failure moves. A suspect that does not move the failure when
    removed is not a diagnosis, and must not be edited."""
    before = kit_test.expected_accuracy(weights, sets["failing"], targets["failing"], 0.0)
    ablated = {k: v.copy() for k, v in weights.items()}
    ablated["W2"][row] = 0.0
    after = kit_test.expected_accuracy(ablated, sets["failing"], targets["failing"], 0.0)
    change = after - before
    return {
        "suspect_row": row,
        "failing_before": before,
        "failing_after_ablation": after,
        "change_points": change,
        "causal": abs(change) >= CAUSAL_MIN_CHANGE,
    }
