"""Protocols: the frozen v1 measurement behaviour and the v2
evaluation-integrity protocol (see WSP_KIT_FIX_PROMPT.md section 4).

A ``Protocol`` is the resolved set of numbers a run uses. v1
freezes measurement behaviour (scorers, gate, causal rule, layer
choice, alpha, sampling) so old published runs stay reproducible;
v2 is the default for run / watch / loop / merge, auto stays v1
(D1) and labels its result QUICK.
"""

MIN_GAIN_POINTS = 2.0
MAX_REGRESS_POINTS = 2.0
# Non-inferiority margin for the v2 guard lower bounds (section 4.2).
GUARD_MARGIN_POINTS = 5.0
CAUSAL_MIN_CHANGE = 5.0
CAUSAL_PLACEBOS = 10
# Alpha bounds, v2 (section 4.2): the legacy formula plus bounds.
ALPHA_MIN = 1e-3
ALPHA_MAX = 100.0
# Phase 4: the v2 subtraction alpha is the column-norm-capped
# alpha (see wsp.real_edit.capped_alpha_from_column_norm), whose
# cap is 1.0 — the only subtraction alpha proven to pass a gate.
# v1 keeps the legacy estimator's bounds.
CAPPED_ALPHA_MAX = 1.0


class Protocol:
    """One protocol's resolved numbers (plain class, 3.9-safe)."""

    def __init__(self, name, samples_per_prompt, min_prompts,
                 split_quotas, scorer_version, min_gain_points,
                 max_regress_points, guard_margin_points,
                 causal_min_change, causal_placebos):
        self.name = name
        self.samples_per_prompt = samples_per_prompt
        self.min_prompts = min_prompts
        self.split_quotas = split_quotas
        self.scorer_version = scorer_version
        self.min_gain_points = min_gain_points
        self.max_regress_points = max_regress_points
        self.guard_margin_points = guard_margin_points
        self.causal_min_change = causal_min_change
        self.causal_placebos = causal_placebos

    def config_dict(self):
        """The resolved numbers, as written to report.real_demo."""
        return {
            "protocol": self.name,
            "samples_per_prompt": self.samples_per_prompt,
            "min_prompts": dict(self.min_prompts),
            "split_quotas": {k: dict(v)
                             for k, v in self.split_quotas.items()},
            "scorer_version": self.scorer_version,
            "min_gain_points": self.min_gain_points,
            "max_regress_points": self.max_regress_points,
            "guard_margin_points": self.guard_margin_points,
            "causal_min_change": self.causal_min_change,
            "causal_placebos": self.causal_placebos,
            # v2 subtraction uses the column-norm-capped alpha
            # (max CAPPED_ALPHA_MAX); v1 keeps the legacy bounds.
            "alpha_bounds": [ALPHA_MIN,
                             CAPPED_ALPHA_MAX if self.name == "v2"
                             else ALPHA_MAX],
        }


PROTOCOLS = {
    "v1": Protocol(
        name="v1",
        samples_per_prompt=1,
        min_prompts={"failing": 0, "neighbours": 0, "good": 0},
        split_quotas={},
        scorer_version="v1",
        # v1 is frozen at the published +5 bar so old published
        # runs stay reproducible (golden v1 fixtures expect 5.0).
        min_gain_points=5.0,
        max_regress_points=MAX_REGRESS_POINTS,
        guard_margin_points=GUARD_MARGIN_POINTS,
        causal_min_change=CAUSAL_MIN_CHANGE,
        causal_placebos=CAUSAL_PLACEBOS,
    ),
    "v2": Protocol(
        name="v2",
        samples_per_prompt=3,
        min_prompts={"failing": 20, "neighbours": 16, "good": 12},
        split_quotas={
            "failing": {"diagnose": 0.3, "select": 0.2,
                        "holdout": 0.5},
            "neighbours": {"diagnose": 0.3, "select": 0.2,
                           "holdout": 0.5},
            "good": {"select": 0.3, "holdout": 0.7},
        },
        scorer_version="v2",
        min_gain_points=MIN_GAIN_POINTS,
        max_regress_points=MAX_REGRESS_POINTS,
        guard_margin_points=GUARD_MARGIN_POINTS,
        causal_min_change=CAUSAL_MIN_CHANGE,
        causal_placebos=CAUSAL_PLACEBOS,
    ),
}


def get_protocol(name):
    """The Protocol for ``name``; ValueError for unknown names."""
    try:
        return PROTOCOLS[name]
    except KeyError:
        raise ValueError(
            f"unknown protocol {name!r} — use one of "
            f"{sorted(PROTOCOLS)}") from None
