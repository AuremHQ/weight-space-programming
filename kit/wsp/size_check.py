"""Estimate a model's parameter count BEFORE any weight is downloaded
or loaded (F16).

The estimate reports a model's size before the machine spends time
and bandwidth on it. Callers use it as a fail-closed resolvability
check: when the size cannot be determined at all, it raises
ValueError. It is not a cap — there is no model-size cap. Use small
models for better performance. It never reads weights and never
trusts remote code: ``trust_remote_code`` is never passed anywhere
in this module.

Local directory: read its config.json and instantiate the
architecture on the meta device (``accelerate.init_empty_weights``
when accelerate is present, else a ``torch.device("meta")``
context), then sum ``numel`` over the parameters. HF id: try a
cached snapshot's config first, else fall back to the hub's
``safetensors.total`` metadata. When the size cannot be determined
at all, raise ValueError — callers must refuse (fail closed).
"""

import os


def _count_from_config(config):
    import torch
    from transformers import AutoModelForCausalLM

    try:
        from accelerate import init_empty_weights
    except ImportError:
        init_empty_weights = None
    if init_empty_weights is not None:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
    else:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
    # from_config does NOT tie weights on a meta device, so a tied
    # model (lm_head sharing the embedding) would count its
    # vocabulary twice. Tie first, then the shared parameter is
    # counted once — the same number count_params() reports.
    tie = getattr(model, "tie_weights", None)
    if callable(tie):
        tie()
    return sum(p.numel() for p in model.parameters())


def _count_from_local_dir(path):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        path, trust_remote_code=False, local_files_only=True)
    return _count_from_config(config)


def estimate_param_count(model_ref):
    """Return the model's parameter count, or raise ValueError.

    ``model_ref`` is a local directory or an HF id. Raises
    ValueError when the size cannot be determined — the caller must
    refuse to load the model.
    """
    if os.path.isdir(model_ref):
        try:
            return _count_from_local_dir(model_ref)
        except Exception as exc:
            raise ValueError(
                f"cannot determine the size of the model at "
                f"{model_ref!r}: {exc}") from exc
    try:
        from huggingface_hub import snapshot_download

        local = snapshot_download(model_ref, local_files_only=True)
        return _count_from_local_dir(local)
    except Exception:
        pass
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_ref)
        total = getattr(info.safetensors, "total", None)
        if total is None:
            raise ValueError(
                f"the hub has no safetensors.total metadata for "
                f"{model_ref!r} — size unknown, refused")
        return int(total)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"cannot determine the size of the model "
            f"{model_ref!r}: {exc}") from exc
