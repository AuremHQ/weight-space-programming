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

Dampening (Phase 4, founder-tested) is the second edit family:
``apply_dampen`` scales the suspect columns toward zero instead of
subtracting a direction. Which family runs is conditioned on the
causal proof's sign (see ``select_attempt_plan`` in
``wsp.real_pipeline``): causal + (zeroing helps) -> dampen; causal
- (zeroing hurts) -> subtract with the column-norm-capped alpha
(``capped_alpha_from_column_norm``), never the signed-mean formula.
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


def check_recipe_cap(layers, hidden_sizes, total_params):
    """Recipe-level 1% cap, checked before any attempt is applied.

    ``layers`` is a list of ``{"index": ..., "columns": [...]}``;
    ``hidden_sizes`` maps layer index to that layer's hidden size.
    Counts the UNION of distinct (layer, column) pairs, each one
    costing one hidden-size column of weights — so a column listed
    twice in one layer, or the same layer listed twice, is counted
    once. Raises ValueError above the cap; exactly 1% is allowed.
    Returns the changed count.
    """
    pairs = set()
    for layer in layers:
        for column in layer["columns"]:
            pairs.add((layer["index"], column))
    for (index, _column) in pairs:
        if index not in hidden_sizes:
            raise ValueError(
                f"recipe layer {index} is not in this model — refused")
    changed = sum(hidden_sizes[index] for (index, _column) in pairs)
    frac = changed / total_params
    if frac > MAX_EDIT_FRACTION:
        raise ValueError(
            f"recipe would touch {changed:,} weights ({frac:.2%}), "
            "over the 1% cap — rejected as too broad"
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


def apply_dampen(weight, columns, factor, layout="out_in"):
    """Scale selected columns by ``factor`` (0 = zero them, 1 = no-op).

    ``weight`` is a torch tensor of shape (hidden, intermediate) for
    layout ``"out_in"`` (nn.Linear) or (intermediate, hidden) for
    layout ``"in_out"`` (Conv1D, GPT-2's ``c_proj``); ``columns``
    lists intermediate indices; ``factor`` must be in [0, 1].
    Returns (new_weight, changed_count) where changed_count counts
    entries that actually differ afterwards. The input tensor is
    never mutated.

    Why dampening instead of subtracting a direction: subtracting
    ``alpha * direction`` injects a NEW vector into the residual
    stream on every token where the columns fire — the columns fire
    on good prompts as much as on failing ones, and their activations
    are large (|a| ~ 5-20), so the injected shift is large and noisy
    at any alpha. Measured on Qwen3-0.6B (narrow-animals eval,
    layers [27, 26], top-64 columns): subtraction at every alpha
    from 0.5 to 371 made the failing set WORSE (26.67 -> 8.33 at
    0.5x, 1.67 at 371x) and destroyed good (100 -> 0 from 2x up).
    Dampening only shrinks what the columns already output — it can
    never inject a direction, so it cannot blow up generation:
    factor 0.5 moved failing 26.67 -> 33.33, factor 0.25 -> 68.33,
    with good and neighbours unchanged at 100. See
    experiments/EDITOR-FIX-REPORT.md for the full dose-response.
    """
    if layout not in ("out_in", "in_out"):
        raise ValueError(f"unknown weight layout {layout!r}")
    factor = float(factor)
    if not 0.0 <= factor <= 1.0:
        raise ValueError(
            f"dampen factor must be in [0, 1], got {factor!r}")
    import torch

    # clone() is mandatory: the scaling below writes in place.
    work = weight.detach().clone().to(torch.float32)
    cols = list(columns)
    if layout == "out_in":
        work[:, cols] = work[:, cols] * factor
    else:
        work[cols, :] = work[cols, :] * factor
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


# v2 alpha bounds (protocol section 4.2): the legacy formula plus
# bounds. ALPHA_HARD_MAX (100.0, recipe.py) is the upper bound.
ALPHA_MIN = 1e-3


def alpha_v2(direction_norm, mean_failing_activation):
    """Legacy alpha plus v2 bounds. Returns (alpha, clamped).

    The formula is unchanged from the Phase 2 design; only the
    bounds are new. ``clamped`` is recorded per layer in the v2
    report. Since Phase 4 the v2 pipeline no longer EDITS with
    this alpha (subtraction uses the column-norm-capped alpha
    below, dampening uses no alpha); it is kept so a report can
    still show what the legacy formula would have said
    (``legacy_alpha``), and for the tests that pin its bounds.
    """
    from .recipe import ALPHA_HARD_MAX

    base = alpha_base_from_signals(direction_norm,
                                   mean_failing_activation)
    clamped = min(max(base, ALPHA_MIN), ALPHA_HARD_MAX)
    return clamped, clamped != base


# The column-norm cap for a sane subtraction alpha (Phase 4,
# founder-tested): 1.0 is the only subtraction alpha proven to
# pass a gate (SmolLM2, layer 23 only), so no alpha above it is
# trusted.
CAPPED_ALPHA_MAX = 1.0


def capped_alpha_from_column_norm(mean_column_norm, cap=CAPPED_ALPHA_MAX):
    """Sane subtraction scale: alpha = mean column norm, capped at ``cap``.

    The old scale (``alpha_base_from_signals``) divides the direction
    norm by the SIGNED mean activation of the selected columns, which
    cancels to ~0 — so it returns either the 1.0 fallback or a blowup
    (1.0 vs 7867 on the same model, same eval). This scale bounds the
    edit relative to the columns' own size instead: alpha <= 1x the
    mean column norm keeps the injected shift at most about one
    column-norm per firing column, which is model-independent.
    Measured expectation (EDITOR-FIX-REPORT §4a): ~0.5-1.0 on both
    Qwen3-0.6B and SmolLM2-1.7B-Instruct — matching the alphas that
    actually passed. The ``cap`` (default 1.0) is a safety rail: 1.0
    is the only subtraction alpha proven to pass a gate (SmolLM2,
    layer 23 only), so no alpha above it is trusted. A non-positive
    norm (degenerate all-zero columns) falls back to 1.0, the same
    value as the old fallback. mean|act| as the denominator is ruled
    out — it gives 36/86 on SmolLM2, far above the column norms.
    """
    norm = float(mean_column_norm)
    if not np.isfinite(norm) or norm <= 0:
        return 1.0
    return min(norm, float(cap))
