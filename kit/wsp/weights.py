"""Loading, saving, and counting weights.

Two formats, on purpose. ``.npz`` always works (numpy only) and is what
the kit model and the tests use. ``.safetensors`` works when the
``safetensors`` package is installed, which is the format real
open-weight models ship in. Nothing here trains, and nothing here ever
overwrites the file it loaded from: edits are saved to a new path.
"""

import os

import numpy as np


def total_params(weights):
    return int(sum(int(v.size) for v in weights.values()))


def check_size(weights):
    """Return the model's parameter count.

    No model-size cap: use small models for better performance.
    """
    return total_params(weights)


def load_weights(path):
    """Load a weights file into a dict of numpy arrays.

    ``.npz`` is always supported. ``.safetensors`` needs the safetensors
    package (``pip install 'wsp-kit[safetensors]'``).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no weights file at {path}")
    if path.endswith(".npz"):
        with np.load(path) as data:
            weights = {k: np.array(data[k]) for k in data.files}
    elif path.endswith(".safetensors"):
        try:
            from safetensors.numpy import load_file
        except ImportError:
            raise ImportError(
                "safetensors support needs the safetensors package: "
                "pip install 'wsp-kit[safetensors]'"
            ) from None
        weights = load_file(path)
    else:
        raise ValueError(
            f"unsupported weights format for {path} — use .npz or .safetensors"
        )
    if not weights:
        raise ValueError(f"no weights found in {path}")
    check_size(weights)
    return weights


def save_weights(path, weights):
    """Save weights to a new file. Never pass the path you loaded from."""
    if path.endswith(".npz"):
        np.savez(path, **weights)
    elif path.endswith(".safetensors"):
        try:
            from safetensors.numpy import save_file
        except ImportError:
            raise ImportError(
                "safetensors support needs the safetensors package: "
                "pip install 'wsp-kit[safetensors]'"
            ) from None
        save_file(weights, path, metadata={"format": "pt"})
    else:
        raise ValueError(
            f"unsupported weights format for {path} — use .npz or .safetensors"
        )
    return path
