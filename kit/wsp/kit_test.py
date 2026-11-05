"""The kit model for the demo.

This is a small two-layer network (64 inputs, 128 hidden, 64 outputs)
with a planted defect: on one cluster of inputs it collapses to class 0.
It is NOT a language model, it does not generate text, and its numbers
are NOT language-model results. It exists so the whole pipeline —
baseline, cheap-fix check, causal proof, edit, gate, report — can be
shown end to end in seconds on any laptop, with no GPU, no download,
and no editor model.

Ground truth for the kit is the clean model's own output. The defect
is added after the fact, so "accuracy" here means agreement with the
clean model. That is the honest way to run a planted-defect demo: the
task is to restore behaviour that was deliberately broken, not to
claim a model learned something new.
"""

import numpy as np

INPUT_DIM = 64
HIDDEN_DIM = 128
N_CLASSES = 64
DEFECT_CLASS = 0
DEFECT_STRENGTH = 3.0


def build_clean_weights(seed=42):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 0.5, size=(HIDDEN_DIM, INPUT_DIM)),
        "b1": np.zeros(HIDDEN_DIM),
        "W2": rng.normal(0, 0.5, size=(N_CLASSES, HIDDEN_DIM)),
        "b2": np.zeros(N_CLASSES),
    }


def build_sets(seed=123, n=300):
    """Three input sets. Failing is a shifted cluster; neighbours sit
    close to the good cluster and should stay unaffected by the fix."""
    rng = np.random.default_rng(seed)
    good = rng.normal(0, 1, size=(n, INPUT_DIM))
    failing = rng.normal(0, 1, size=(n, INPUT_DIM))
    failing[:, :2] += 2.0
    neighbours = rng.normal(0, 1, size=(n, INPUT_DIM))
    neighbours[:, :2] += 0.5
    return {"good": good, "failing": failing, "neighbours": neighbours}


def forward(weights, X):
    h = np.tanh(X @ weights["W1"].T + weights["b1"])
    logits = h @ weights["W2"].T + weights["b2"]
    return logits, h


def probs(logits, temperature):
    if temperature == 0:
        out = np.zeros_like(logits)
        out[np.arange(len(logits)), np.argmax(logits, axis=1)] = 1.0
        return out
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_accuracy(weights, X, targets, temperature):
    """Mean probability assigned to the target class.

    At temperature 0 this is plain greedy agreement. At higher
    temperatures it is the expected agreement of one sample, which is
    why it falls toward 1/64 as temperature rises: sampling noise is
    not a fix for a broken model.
    """
    logits, _ = forward(weights, X)
    p = probs(logits, temperature)
    return float(100.0 * p[np.arange(len(X)), targets].mean())


def clean_targets(weights, sets):
    targets = {}
    for name, X in sets.items():
        logits, _ = forward(weights, X)
        targets[name] = np.argmax(logits, axis=1)
    return targets


def plant_defect(weights, X_failing):
    """Add the defect: boost class 0 along the failing cluster's mean
    hidden direction. Returns the broken weights and the direction used
    (the demo keeps it only to describe the defect, never to fix it —
    the fix estimates its own direction from data)."""
    _, h = forward(weights, X_failing)
    u = h.mean(axis=0)
    u = u / np.linalg.norm(u)
    bad = {k: v.copy() for k, v in weights.items()}
    bad["W2"][DEFECT_CLASS] = bad["W2"][DEFECT_CLASS] + DEFECT_STRENGTH * u
    return bad, u
