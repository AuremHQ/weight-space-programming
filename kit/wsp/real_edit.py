"""The real-model edit: subtract a direction from MLP output columns.

This is the transformer version of the kit edit in ``wsp.edit``. In the
kit, one row of W2 (one output class over hidden units) loses
``alpha * direction``. In a transformer MLP, the output projection
(``down_proj`` / ``c_proj`` / ``dense_4h_to_h`` — see ``wsp.layers``)
maps intermediate activations back into the residual stream, so one
COLUMN of that matrix is a vector in residual space. Subtracting a
residual-space direction from a column means: whenever that
intermediate neuron fires, the direction is removed from the
residual stream. That is the MIT Qwen shape — one direction, a small
set of layers, a small set of columns — with no training and no
gradients.

The 1% cap is enforced here, in code, before anything is written.
The caller's model is never mutated by the check: ``apply_column_edit``
works on a clone and returns it, and the runner only swaps weights in
after the cap passes. Original model files are never overwritten;
edited weights are always saved to a new directory. GPT-2's
``c_proj`` is a Conv1D whose weight is stored transposed, so
``apply_column_edit`` takes a ``layout``: ``"out_in"`` (nn.Linear,
default) edits columns, ``"in_out"`` (Conv1D) edits rows. The
direction is a residual-space vector either way.
"""

import numpy as np

MAX_EDIT_FRACTION = 0.01


def check_cap(changed, total_params):
    """Raise if an edit would touch more than 1% of weights."""
    frac = changed / total_params
    if frac > MAX_EDIT_FRACTION:
        raise ValueError(
            f"edit touches {frac:.2%} of weights, over the 1% cap — "
            "rejected as too broad"
        )
    return frac


def apply_column_edit(weight, columns, direction, alpha, layout="out_in"):
    """Subtract ``alpha * direction`` from selected columns of a matrix.

    ``weight`` is a torch tensor of shape (hidden, intermediate) for
    layout ``"out_in"`` (nn.Linear) or (intermediate, hidden) for
    layout ``"in_out"`` (Conv1D, GPT-2's ``c_proj``); ``direction``
    is a 1-D array/tensor of length hidden (unit norm); ``columns``
    lists intermediate indices. Returns (new_weight, changed_count)
    where changed_count counts entries that actually differ
    afterwards (bf16 rounding can leave tiny updates at zero, and
    those are honestly not counted as changed).

    The input tensor is never mutated.
    """
    if layout not in ("out_in", "in_out"):
        raise ValueError(f"unknown weight layout {layout!r}")
    import torch

    if isinstance(direction, np.ndarray):
        direction = torch.from_numpy(direction.astype(np.float32))
    direction = direction.to(torch.float32)
    # clone() is mandatory: .to() is a no-op copy when the dtype already
    # matches, and the edit below writes in place.
    work = weight.detach().clone().to(torch.float32)
    cols = list(columns)
    if layout == "out_in":
        work[:, cols] = work[:, cols] - float(alpha) * direction.unsqueeze(1)
    else:
        work[cols, :] = work[cols, :] - float(alpha) * direction.unsqueeze(0)
    new = work.to(weight.dtype)
    changed = int((new != weight.detach()).sum().item())
    return new, changed


def alpha_base_from_signals(direction_norm, mean_failing_activation):
    """Data-driven edit scale: how big alpha must be for the edit to
    remove roughly the observed excess direction.

    ``direction_norm`` is ||mean MLP output on failing - mean on good||
    at a layer; ``mean_failing_activation`` is the mean intermediate
    activation of the selected columns on failing prompts. The edit
    shifts the residual by alpha * activation per token, so
    alpha = direction_norm / activation removes about the excess.
    The pipeline tries 0.5x, 1x and 2x this base (max 3 attempts).
    """
    denom = float(mean_failing_activation)
    if denom <= 1e-6:
        return 1.0
    return float(direction_norm) / denom
