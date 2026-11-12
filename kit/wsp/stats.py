"""Small-sample statistics for the v2 gate (no scipy).

One-sided 95% Student-t critical values, and the paired lower
bound on per-prompt deltas that ``wsp.gate.check_gate_v2`` uses.
"""

import math

# One-sided 95% t critical values (prompt section 4.2). The last
# entry is df = infinity (the normal 1.645).
_T_TABLE = (
    (1, 6.314),
    (2, 2.920),
    (3, 2.353),
    (4, 2.132),
    (5, 2.015),
    (7, 1.895),
    (9, 1.833),
    (19, 1.729),
    (29, 1.699),
    (float("inf"), 1.645),
)


def t_crit_one_sided(df):
    """One-sided 95% t critical value for ``df`` degrees of freedom.

    Exact on the table; linear interpolation in 1/df between table
    points (including between df=29 and infinity) otherwise.
    """
    if not isinstance(df, (int, float)) or isinstance(df, bool):
        raise ValueError(f"df must be a number, got {df!r}")
    if math.isnan(float(df)) or float(df) < 1:
        raise ValueError(f"df must be >= 1, got {df!r}")
    df = float(df)
    for table_df, value in _T_TABLE:
        if df == table_df:
            return value
    # Interpolate in x = 1/df between the bracketing table points.
    # x decreases as df grows; the table is stored in increasing df
    # (decreasing x), with infinity as x = 0.
    x = 1.0 / df
    points = [(1.0 / d if d != float("inf") else 0.0, v)
              for d, v in _T_TABLE]
    # points is in decreasing x order (1.0 ... 0.0). Find the pair
    # that brackets x.
    for (x_hi, v_hi), (x_lo, v_lo) in zip(points, points[1:]):
        if x_lo <= x <= x_hi:
            if x_hi == x_lo:  # pragma: no cover - degenerate table
                return v_hi
            frac = (x - x_hi) / (x_lo - x_hi)
            return v_hi + frac * (v_lo - v_hi)
    # df < 1 is rejected above; df between 1 and the table start
    # cannot happen. Anything else is a table bug.
    raise ValueError(f"no t critical value for df={df!r}")


def paired_lower_bound(deltas):
    """One-sided 95% lower bound on the mean of paired deltas.

    Returns {"n", "mean", "sd", "lower"} where sd is the sample
    standard deviation (ddof=1). n < 2 gives lower = None: no bound
    can be computed, which the gate reads as INCONCLUSIVE. sd = 0
    is allowed (lower = mean). NaN or Inf raises ValueError.
    """
    values = [float(d) for d in deltas]
    for v in values:
        if not math.isfinite(v):
            raise ValueError(
                "paired deltas must be finite numbers, "
                f"got {v!r}")
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "lower": None}
    mean = sum(values) / n
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "lower": None}
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    lower = mean - t_crit_one_sided(n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "lower": lower}
