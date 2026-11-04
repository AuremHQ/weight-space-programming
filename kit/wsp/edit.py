"""The edit: one direction subtracted from one row, nothing else.

The 1% cap is enforced here, in code, not just in the docs. An edit
that would touch more is rejected before anything is written, and the
caller's weights are never mutated: the function works on a copy.
"""

import numpy as np

from .weights import total_params

MAX_EDIT_FRACTION = 0.01


def apply_edit(weights, row, direction, alpha):
    """Subtract ``alpha * direction`` from one row of W2.

    Returns the new weights, the number of weights changed, and the
    fraction of the model changed. Raises ValueError if the fraction
    is over the 1% cap.
    """
    new = {k: v.copy() for k, v in weights.items()}
    before = new["W2"][row].copy()
    new["W2"][row] = before - alpha * direction
    changed = int(np.count_nonzero(new["W2"][row] - before))
    frac = changed / total_params(weights)
    if frac > MAX_EDIT_FRACTION:
        raise ValueError(
            f"edit touches {frac:.2%} of weights, over the 1% cap — "
            "rejected as too broad"
        )
    return new, changed, frac
