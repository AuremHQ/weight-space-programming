"""Environment record for reports (W2.12).

``collect_environment()`` never raises: a missing package is
recorded as None, never an exception. Versions come from
importlib.metadata (no imports — importing torch just to read its
version would be slow and, in a dry run, forbidden). The kit
commit comes from ``git rev-parse`` only when a .git directory
exists above the package; it never touches the network.
"""

import os
import platform
import subprocess
import sys

try:
    from importlib import metadata as _metadata
except ImportError:  # pragma: no cover - Python 3.7 fallback
    _metadata = None

import numpy

from . import __version__


def _package_version(name):
    if _metadata is None:
        return None
    try:
        return _metadata.version(name)
    except Exception:
        return None


def _kit_commit():
    """HEAD commit, or None (no .git, no git, or any error)."""
    here = os.path.dirname(os.path.abspath(__file__))
    node = here
    while True:
        if os.path.isdir(os.path.join(node, ".git")):
            break
        parent = os.path.dirname(node)
        if parent == node:
            return None
        node = parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=node, capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    commit = out.stdout.strip()
    return commit or None


def collect_environment(device="cpu", dtype=None):
    """Environment facts for a report. Never raises."""
    try:
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "safetensors": _package_version("safetensors"),
            "device": device,
            "dtype": dtype,
            "kit_version": __version__,
            "kit_commit": _kit_commit(),
        }
    except Exception:  # pragma: no cover - defensive; never raise
        return {
            "python": None,
            "platform": None,
            "numpy": None,
            "torch": None,
            "transformers": None,
            "accelerate": None,
            "safetensors": None,
            "device": device,
            "dtype": dtype,
            "kit_version": __version__,
            "kit_commit": None,
        }
