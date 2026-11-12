"""Output-directory safety for run / apply-edit / merge.

An output directory that IS a model directory, sits inside one, or
contains one would let the pipeline overwrite the model's own
weights — the exact mistake the 1% cap exists to bound. The same
goes for an output directory that already holds model files:
writing there would silently mix edited weights with the originals.

Checks are by real (symlink-resolved) path, so a symlink cannot
smuggle a model directory in under a different name. HF ids are
not local paths and are ignored here. The caller passes ``force``
only when the user explicitly asked for it (``--force``).
"""

import os

WEIGHT_EXTENSIONS = (".safetensors", ".bin", ".pt", ".pth", ".onnx")
CONFIG_NAMES = ("config.json",)


def _norm(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def resolve_model_dirs(refs):
    """Local directories for model refs (HF ids resolved from cache).

    A ref that is already a local directory is itself. An HF id is
    resolved to its cached snapshot directory when it is cached
    (local_files_only — never a download); an uncached id resolves
    to nothing, because there are no local weights to clobber yet.
    Call sites pass the result to assert_output_dir_safe, so an
    out_dir that would overwrite a cached model's weights is
    refused the same way as one that would overwrite a local dir.
    """
    dirs = []
    for ref in refs:
        if not ref:
            continue
        if os.path.isdir(ref):
            dirs.append(ref)
            continue
        try:
            from huggingface_hub import snapshot_download

            dirs.append(snapshot_download(ref, local_files_only=True))
        except Exception:
            continue
    return dirs


def _is_model_file(name):
    lower = name.lower()
    return lower in CONFIG_NAMES or lower.endswith(WEIGHT_EXTENSIONS)


def assert_output_dir_safe(out_dir, model_dirs, force=False):
    """Refuse an output directory that risks clobbering model weights.

    ``model_dirs`` is a list of local model directories (None, HF
    ids and non-directories are ignored). Raises ValueError with a
    plain reason when the out_dir equals, sits inside, or contains
    any of them, or when it already holds config.json or weight
    files and ``force`` is False.
    """
    out = _norm(out_dir)
    for model_dir in model_dirs:
        if not model_dir or not os.path.isdir(model_dir):
            continue
        model = _norm(model_dir)
        if out == model:
            raise ValueError(
                f"output directory {out_dir!r} IS the model "
                "directory — refused")
        if out.startswith(model + os.sep):
            raise ValueError(
                f"output directory {out_dir!r} is INSIDE the model "
                f"directory {model_dir!r} — refused")
        if model.startswith(out + os.sep):
            raise ValueError(
                f"output directory {out_dir!r} CONTAINS the model "
                f"directory {model_dir!r} — refused")
    if os.path.isdir(out_dir) and not force:
        for name in os.listdir(out_dir):
            if _is_model_file(name):
                raise ValueError(
                    f"output directory {out_dir!r} already holds "
                    f"{name!r} — refused; use --force to overwrite")
