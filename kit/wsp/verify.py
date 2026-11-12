"""Tensor-hash verify (Phase 4, W4.4): did an applied recipe
reproduce the recorded edited weights?

``changed_tensors_sha256(model, layers)`` hashes the full
down-projection weight tensor of every edited layer, in ascending
layer order, as raw dtype bytes. The hash covers the WHOLE tensor,
not just the edited columns: the unedited columns are identical to
the original, so a MATCH still proves the edited columns
reproduced, and any difference anywhere in those tensors shows up
as DIVERGED.

The recorded value for the Qwen narrow-dampen pass lives in
``runs/qwen-narrow-dampen/PROVENANCE.md`` (sha256 of the raw BF16
bytes of ``model.layers.26.mlp.down_proj.weight`` then
``model.layers.27.mlp.down_proj.weight``, read from the edited
``model.safetensors``). ``wsp apply-edit --verify`` recomputes the
same hash from the freshly edited model and prints MATCH or
DIVERGED. A MATCH assumes the same base model revision was loaded;
a different base revision honestly reports DIVERGED.

The model only needs ``down_proj_weight(layer_index)``:
``RealModel`` returns the detached CPU torch tensor,
``FakeRealModel`` a numpy array. No weights are copied anywhere.
"""

import hashlib

import numpy as np


def _tensor_bytes(tensor):
    """Raw dtype bytes of a torch tensor or a numpy array.

    Torch bf16 has no ``.numpy()`` (it raises), so torch tensors
    are read through a uint8 view — raw bytes, any dtype. Numpy
    arrays are read contiguously.
    """
    if hasattr(tensor, "detach"):
        import torch

        view = tensor.detach().cpu().contiguous().view(torch.uint8)
        return view.numpy().tobytes()
    return np.ascontiguousarray(tensor).tobytes()


def changed_tensors_sha256(model, layers):
    """SHA-256 (hex) of the edited layers' down-proj tensors.

    ``layers`` is a list of ``{"index": ...}`` (a recipe's layers,
    or any list of layer dicts). Layers are hashed in ascending
    index order, so the result does not depend on the recipe's
    layer order.
    """
    h = hashlib.sha256()
    for layer in sorted(layers, key=lambda l: l["index"]):
        weight = model.down_proj_weight(layer["index"])
        h.update(_tensor_bytes(weight))
    return h.hexdigest()
