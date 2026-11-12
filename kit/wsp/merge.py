"""Merge mode (R6): combine two same-base open-weight models.

SEPARATE from surgical editing (SKILL Steps 0-3). A merge blends two
whole models — it is NOT covered by the 1% weight cap, it never edits
a production branch, and it ships only if its own full eval gate
passes (+2 primary, <=2 regression, serving temperature AND greedy).

Methods shipped: ``linear`` (weighted average) and ``slerp``
(spherical linear interpolation, MergeKit-style). Task Arithmetic,
TIES, DARE and Model Soups are future work, not silent defaults.

The same-base check and the merge math are pure numpy and need no
model download. Loading, saving and gating real models need the
``real`` extra (torch, transformers, safetensors); those imports are
lazy, so importing this module never fails without them.
"""

import hashlib
import json
import os
import shutil

import numpy as np

METHODS = ("linear", "slerp")
# No model-size cap: use small models for better performance.
# Config keys that define a base. A key present in one config and
# missing in the other is a mismatch, not a default.
CONFIG_KEYS = (
    "architectures",
    "model_type",
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "n_embd",
    "n_layer",
    "n_head",
    "n_positions",
    "n_ctx",
    "max_position_embeddings",
    "tie_word_embeddings",
)
# Tokenizer files. The check does not demand identical file SETS:
# ``tokenizer.json`` (fast tokenizer) and ``tokenizer_config.json``
# are authoritative — they must match when present in both, and a
# model that has one while the other does not is a mismatch (sameness
# cannot be proven cheaply). The slow files (vocab.json, merges.txt
# and friends) are redundant when a fast tokenizer exists, so they
# are only compared when NEITHER model has tokenizer.json.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)
# Files compared only for slow-tokenizer-only models.
SLOW_TOKENIZER_FILES = (
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)


def _require_real():
    """Import the real-model stack, or raise the plain install hint."""
    try:
        import safetensors.torch  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "real-model support needs torch + transformers: "
            "pip install 'wsp-kit[real]'"
        ) from exc


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenizer_hashes(model_dir):
    """SHA-256 of each tokenizer file present in the directory."""
    out = {}
    for name in TOKENIZER_FILES:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            out[name] = _file_sha256(path)
    return out


def check_same_base(config_a, config_b, tok_a, tok_b):
    """Check two models share a base. Returns a list of mismatches.

    Empty list = same base. A non-empty list names every difference
    found, so the refusal message says exactly what does not match.
    """
    mismatches = []
    for key in CONFIG_KEYS:
        in_a = key in config_a
        in_b = key in config_b
        if in_a and in_b:
            if config_a[key] != config_b[key]:
                mismatches.append(
                    f"config {key!r}: {config_a[key]!r} != {config_b[key]!r}"
                )
        elif in_a or in_b:
            mismatches.append(
                f"config {key!r}: present in "
                f"{'A' if in_a else 'B'} only"
            )
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if name in tok_a and name in tok_b:
            if tok_a[name] != tok_b[name]:
                mismatches.append(f"tokenizer file {name!r}: differs")
        elif name in tok_a or name in tok_b:
            mismatches.append(
                f"tokenizer file {name!r}: present in "
                f"{'A' if name in tok_a else 'B'} only"
            )
    if "tokenizer.json" not in tok_a and "tokenizer.json" not in tok_b:
        for name in SLOW_TOKENIZER_FILES:
            if name in tok_a and name in tok_b:
                if tok_a[name] != tok_b[name]:
                    mismatches.append(f"tokenizer file {name!r}: differs")
            elif name in tok_a or name in tok_b:
                mismatches.append(
                    f"tokenizer file {name!r}: present in "
                    f"{'A' if name in tok_a else 'B'} only"
                )
    return mismatches


def merge_linear(a, b, t):
    """Weighted average: (1-t)*a + t*b. t=0 returns A, t=1 returns B."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    out = (1.0 - float(t)) * a.astype(np.float32) + float(t) * b.astype(np.float32)
    return out.astype(a.dtype)


def merge_slerp(a, b, t):
    """Spherical linear interpolation (direction SLERP + magnitude lerp).

    Falls back to the linear average when the two vectors are
    (near-)parallel, (near-)opposite, or either is (near-)zero —
    SLERP is undefined there, and the fallback is exact in the
    parallel case. t=0 returns A, t=1 returns B.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    t = float(t)
    a32 = a.astype(np.float32).ravel()
    b32 = b.astype(np.float32).ravel()
    na = float(np.linalg.norm(a32))
    nb = float(np.linalg.norm(b32))
    if na < 1e-12 or nb < 1e-12:
        return merge_linear(a, b, t)
    ua = a32 / na
    ub = b32 / nb
    dot = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    omega = float(np.arccos(dot))
    if omega < 1e-6 or omega > np.pi - 1e-6:
        return merge_linear(a, b, t)
    so = float(np.sin(omega))
    direction = (
        (np.sin((1.0 - t) * omega) * ua + np.sin(t * omega) * ub) / so
    )
    magnitude = (1.0 - t) * na + t * nb
    return (direction * magnitude).reshape(a.shape).astype(a.dtype)


_MERGE_FNS = {"linear": merge_linear, "slerp": merge_slerp}


def merge_state_dicts(state_a, state_b, method, t):
    """Merge two state dicts tensor by tensor.

    Floating tensors are merged by ``method``; non-floating tensors
    (ids, masks, positions) must be exactly equal and are copied
    from A. Returns (merged_dict, stats). Raises ValueError on key,
    shape or non-float mismatch.
    """
    if method not in _MERGE_FNS:
        raise ValueError(
            f"unknown merge method {method!r} — use one of {list(METHODS)}"
        )
    if not 0.0 <= float(t) <= 1.0:
        raise ValueError(f"blend weight t must be in [0, 1], got {t!r}")
    keys_a = set(state_a)
    keys_b = set(state_b)
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)[:5]
        only_b = sorted(keys_b - keys_a)[:5]
        raise ValueError(
            "state dict keys differ — not the same base "
            f"(only in A: {only_a}, only in B: {only_b})"
        )
    fn = _MERGE_FNS[method]
    merged = {}
    n_merged = 0
    n_copied = 0
    params_merged = 0
    for name in sorted(keys_a):
        a = np.asarray(state_a[name])
        b = np.asarray(state_b[name])
        if a.shape != b.shape:
            raise ValueError(
                f"tensor {name!r} shape mismatch: {a.shape} vs {b.shape}"
            )
        if np.issubdtype(a.dtype, np.floating):
            merged[name] = fn(a, b, t)
            n_merged += 1
            params_merged += int(a.size)
        else:
            if not np.array_equal(a, b):
                raise ValueError(
                    f"non-float tensor {name!r} differs between models — "
                    "not the same base"
                )
            merged[name] = a
            n_copied += 1
    stats = {
        "tensors_merged": n_merged,
        "tensors_copied": n_copied,
        "params_merged": params_merged,
    }
    return merged, stats


def count_eval_passes(data, temperatures, samples_per_prompt):
    """Total generate calls the gate run will spend.

    Greedy is deterministic, so extra samples at temperature 0 are
    skipped (same rule as wsp.real_pipeline.evaluate). Counts both
    models: the baseline and the merged model are each evaluated.
    """
    from .evalfile import SET_NAMES, set_prompts

    per_model = 0
    for temp in temperatures:
        for name in SET_NAMES:
            reps = 1 if temp == 0 else samples_per_prompt
            per_model += len(set_prompts(data, name)) * reps
    return per_model * 2


def _resolve_model_dir(model):
    """A local directory, or a HuggingFace id resolved from the cache."""
    if os.path.isdir(model):
        return model
    _require_real()
    from huggingface_hub import snapshot_download

    return snapshot_download(model)


def load_model_dir(model):
    """Load one model: (config, state dict, tokenizer hashes, params,
    dtypes).

    ``model`` is a local directory or a HuggingFace id. The state
    dict is plain numpy arrays; bf16/fp16 weights are widened to
    float32 for the merge math (numpy has no bf16), and ``dtypes``
    records each tensor's stored dtype so the merged model is saved
    back in exactly the original dtypes. Nothing is loaded onto a
    device here.
    """
    _require_real()
    import torch
    from safetensors.torch import load_file

    model_dir = _resolve_model_dir(model)
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"no config.json in {model_dir}")
    with open(config_path) as f:
        config = json.load(f)
    state = {}
    dtypes = {}
    shard_files = sorted(
        fn for fn in os.listdir(model_dir)
        if fn.endswith(".safetensors")
    )
    if shard_files:
        for fn in shard_files:
            for name, tensor in load_file(
                    os.path.join(model_dir, fn)).items():
                dtypes[name] = str(tensor.dtype)
                if tensor.dtype in (torch.bfloat16, torch.float16):
                    state[name] = tensor.to(torch.float32).numpy()
                else:
                    state[name] = tensor.numpy()
    else:
        bin_files = sorted(
            fn for fn in os.listdir(model_dir) if fn.endswith(".bin")
        )
        if not bin_files:
            raise FileNotFoundError(
                f"no .safetensors or .bin weights in {model_dir}"
            )
        for fn in bin_files:
            loaded = torch.load(os.path.join(model_dir, fn),
                                map_location="cpu", weights_only=True)
            for name, tensor in loaded.items():
                dtypes[name] = str(tensor.dtype)
                if tensor.dtype in (torch.bfloat16, torch.float16):
                    state[name] = tensor.to(torch.float32).numpy()
                else:
                    state[name] = tensor.numpy()
    total = sum(int(v.size) for v in state.values())
    return config, state, tokenizer_hashes(model_dir), total, dtypes


def save_merged_model(out_dir, config, model_dir_a, merged_state, dtypes):
    """Save the merged model to a NEW directory. Never overwrites.

    Each tensor is cast back to its stored dtype (see
    ``load_model_dir``), so a bf16 base stays a bf16 merged model.
    """
    _require_real()
    import torch
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    for name in os.listdir(model_dir_a):
        if name in TOKENIZER_FILES or name == "generation_config.json":
            src = os.path.join(model_dir_a, name)
            if os.path.isfile(src):
                with open(src, "rb") as fin, open(
                        os.path.join(out_dir, name), "wb") as fout:
                    fout.write(fin.read())
    torch_state = {}
    for name, arr in merged_state.items():
        tensor = torch.from_numpy(np.ascontiguousarray(arr))
        stored = dtypes.get(name)
        if stored is not None:
            tensor = tensor.to(getattr(torch, stored.split(".")[-1]))
        torch_state[name] = tensor
    save_file(torch_state, os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt"})
    return out_dir


def run_merge(model_a, model_b, method, t, eval_path, out_dir,
              dtype="bfloat16", samples_per_prompt=None, seed=0, log=print,
              force=False, protocol="v2"):
    """Merge two models and run the eval gate on the merged model.

    Baseline is model A, evaluated at the serving temperature AND
    greedy; the merged model must gain >=5 on the primary set and
    regress <=2 on guards/neighbours at BOTH temperatures. The
    merged model is written to ``<out_dir>.partial-<pid>`` and
    evaluated from there; it moves into ``out_dir`` only on a gate
    pass — a refused merge ships no model, only merge.json +
    report.json. Returns (report, out_dir, exit_code): 0 on a gate
    pass, 1 on a gate fail or a refusal recorded in the report.
    Raises ValueError on a same-base mismatch, an over-cap model, or
    a bad method/t.

    ``protocol`` selects v1 (legacy point gate, QUICK evidence) or
    v2 (check_gate_v2 on all prompts — no splits, no
    diagnosis/selection stage — with the same minimums; a
    below-minimum eval is INCONCLUSIVE and ships nothing).
    ``samples_per_prompt`` None resolves by protocol (v1: 1, v2: 3).
    """
    from . import evalfile, gate
    from .atomic import atomic_write_json
    from .protocol import get_protocol

    proto = get_protocol(protocol)
    samples = (proto.samples_per_prompt if samples_per_prompt is None
               else samples_per_prompt)
    if method not in METHODS:
        raise ValueError(
            f"unknown merge method {method!r} — use one of {list(METHODS)}"
        )
    if not 0.0 <= float(t) <= 1.0:
        raise ValueError(f"blend weight t must be in [0, 1], got {t!r}")
    data = evalfile.load_eval_file(eval_path)
    # File safety, before any weight is loaded: the out_dir must not
    # be (in/around) either input model. An HF id resolves to its
    # cached snapshot dir when it is cached. No model-size cap: use
    # small models for better performance.
    from .safe_paths import assert_output_dir_safe, resolve_model_dirs

    assert_output_dir_safe(out_dir,
                           resolve_model_dirs([model_a, model_b]),
                           force=force)
    serving = data.get("serving_temperature", 0.7)
    gate_temps = list(dict.fromkeys([serving, 0.0] +
                                    list(data.get("gate_temperatures",
                                                [serving, 0.0]))))
    # v2 preflight: a below-minimum eval cannot SHIP, so refuse it
    # before loading any model. The report is INCONCLUSIVE (not a
    # gate failure on the numbers) and no model ships.
    from .splits import preflight

    problems = preflight(data, proto)
    if problems:
        os.makedirs(out_dir, exist_ok=True)
        from . import __version__

        report = {
            "report_version": 1,
            "kit_version": __version__,
            "mode": "merge",
            "method": method,
            "t": float(t),
            "model_a": model_a,
            "model_b": model_b,
            "protocol": proto.name,
            "evidence_level": ("standard" if proto.name == "v2"
                               else "quick"),
            "before": None,
            "after": None,
            "gate": {"passed": False, "verdict": "INCONCLUSIVE",
                     "reason": problems[0], "by_temperature": {}},
            "eval_name": data.get("name"),
            "eval_file": os.path.basename(data.get("_path", "")),
            "eval_passes_spent": 0,
            "serving_temperature": serving,
            "note": (
                "Merge mode (R6): the eval file is below the v2 "
                "minimums, so this merge cannot be judged — nothing "
                "ships. Add more of YOUR OWN prompts: the kit never "
                "invents them."),
        }
        atomic_write_json(os.path.join(out_dir, "report.json"), report)
        log(f"merge refused: {problems[0]}")
        return report, out_dir, 1
    _require_real()
    log(f"loading {model_a} ...")
    config_a, state_a, tok_a, params_a, dtypes_a = load_model_dir(model_a)
    log(f"loading {model_b} ...")
    config_b, state_b, tok_b, params_b, _dtypes_b = load_model_dir(model_b)
    mismatches = check_same_base(config_a, config_b, tok_a, tok_b)
    if mismatches:
        raise ValueError(
            "not the same base — merge refused:\n  " +
            "\n  ".join(mismatches)
        )
    log(f"  same base confirmed ({params_a:,} parameters, "
        f"{len(state_a)} tensors)")
    log(f"merging ({method}, t={t}) ...")
    merged_state, stats = merge_state_dicts(state_a, state_b, method, t)
    log(f"  merged {stats['tensors_merged']} tensors "
        f"({stats['params_merged']:,} parameters), "
        f"copied {stats['tensors_copied']} non-float tensors")
    # The merged model lives in a partial directory until the gate
    # decides: it moves into out_dir only on a pass, and the partial
    # directory is deleted in the finally otherwise.
    partial_dir = f"{out_dir}.partial-{os.getpid()}"
    try:
        save_merged_model(partial_dir, config_a,
                          _resolve_model_dir(model_a),
                          merged_state, dtypes_a)
        merge_recipe = {
            "format_version": 1,
            "mode": "merge",
            "method": method,
            "t": float(t),
            "model_a": model_a,
            "model_b": model_b,
            "total_params": params_a,
            "same_base_mismatches": [],
            "tensors_merged": stats["tensors_merged"],
            "tensors_copied": stats["tensors_copied"],
            "params_merged": stats["params_merged"],
        }
        # Free the raw states before loading models for evaluation.
        del state_a, state_b, merged_state
        import gc
        gc.collect()

        from .evalfile import SET_NAMES
        from .real_runner import RealModel

        passes = count_eval_passes(data, gate_temps, samples)
        log(f"evaluating baseline ({model_a}) ...")
        base_model = RealModel(model_a, dtype=dtype)
        if proto.name == "v2":
            from .real_pipeline import evaluate_detailed

            base_det = {}
            for temp in gate_temps:
                log(f"  temperature {temp}:")
                base_det[temp] = evaluate_detailed(
                    base_model, data, temp, samples, seed, log,
                    scorer_version=proto.scorer_version)
            baseline = {t: {name: base_det[t][name]["mean"]
                            for name in SET_NAMES}
                        for t in gate_temps}
        else:
            from .real_pipeline import evaluate

            baseline = {}
            for temp in gate_temps:
                log(f"  temperature {temp}:")
                baseline[temp] = evaluate(base_model, data, temp,
                                          samples, seed, log)
        del base_model
        gc.collect()
        log(f"evaluating merged model ({partial_dir}) ...")
        merged_model = RealModel(partial_dir, dtype=dtype)
        if proto.name == "v2":
            after_det = {}
            for temp in gate_temps:
                log(f"  temperature {temp}:")
                after_det[temp] = evaluate_detailed(
                    merged_model, data, temp, samples, seed, log,
                    scorer_version=proto.scorer_version)
            after = {t: {name: after_det[t][name]["mean"]
                         for name in SET_NAMES}
                     for t in gate_temps}
        else:
            after = {}
            for temp in gate_temps:
                log(f"  temperature {temp}:")
                after[temp] = evaluate(merged_model, data, temp,
                                       samples, seed, log)
        del merged_model
        gc.collect()


        gate_by_temp = {}
        if proto.name == "v2":
            # No splits in merge mode: the gate runs on all prompts.
            for temp in gate_temps:
                gate_by_temp[str(temp)] = gate.check_gate_v2(
                    {name: base_det[temp][name]["per_prompt"]
                     for name in SET_NAMES},
                    {name: after_det[temp][name]["per_prompt"]
                     for name in SET_NAMES},
                    min_gain=proto.min_gain_points,
                    max_regress=proto.max_regress_points,
                    guard_margin=proto.guard_margin_points)
            verdict = gate.combine_gate_verdicts(
                [gate_by_temp[str(t)]["verdict"]
                 for t in gate_temps])
            passed = verdict == "SHIP"
        else:
            for temp in gate_temps:
                g = gate.check_gate(baseline[temp], after[temp],
                                    min_gain=proto.min_gain_points)
                gate_by_temp[str(temp)] = g
            passed = all(g["passed"]
                         for g in gate_by_temp.values())
            verdict = "QUICK_PASS" if passed else "QUICK_FAIL"
        from . import __version__

        report = {
            "report_version": 1,
            "kit_version": __version__,
            "mode": "merge",
            "method": method,
            "t": float(t),
            "model_a": model_a,
            "model_b": model_b,
            "total_params": params_a,
            "protocol": proto.name,
            "evidence_level": ("standard" if proto.name == "v2"
                               else "quick"),
            "same_base": {
                "checked": list(CONFIG_KEYS) + ["tokenizer_files"],
                "mismatches": [],
            },
            "tensors_merged": stats["tensors_merged"],
            "tensors_copied": stats["tensors_copied"],
            "params_merged": stats["params_merged"],
            "before": baseline[serving],
            "after": after[serving],
            "scores_by_temperature": {
                "before": {str(k): v for k, v in baseline.items()},
                "after": {str(k): v for k, v in after.items()},
            },
            "gate": {"passed": passed, "verdict": verdict,
                     "by_temperature": gate_by_temp},
            "eval_name": data.get("name"),
            "eval_file": os.path.basename(data.get("_path", "")),
            "eval_passes_spent": passes,
            "serving_temperature": serving,
            "note": (
                "Merge mode (R6): this model is a blend of two same-base "
                "models, not a surgical edit. Merges are NOT covered by "
                "the 1% weight cap. Background: the LLM-editor method "
                "was demonstrated by MIT with GPT-6 Astra; blog post, "
                "not peer reviewed. Editor of THIS merge: arithmetic "
                "on weights (no LLM, no editor model). This "
                "merge passed the eval gate below; verify independently "
                "before use."
                if passed else
                "Merge mode (R6): the gate refused this merge — nothing "
                "ships. A refused merge is a correct result, not a "
                "failure to hide."
            ),
        }
        if passed:
            # Move the merged model into place: a whole-directory
            # os.replace when out_dir does not exist yet, else move
            # the files (out_dir may already hold an old report.json
            # from a refused merge).
            if not os.path.exists(out_dir):
                os.replace(partial_dir, out_dir)
            else:
                for name in os.listdir(partial_dir):
                    os.replace(os.path.join(partial_dir, name),
                               os.path.join(out_dir, name))
                os.rmdir(partial_dir)
        os.makedirs(out_dir, exist_ok=True)
        atomic_write_json(os.path.join(out_dir, "merge.json"),
                          merge_recipe)
        atomic_write_json(os.path.join(out_dir, "report.json"), report)
        if passed:
            log(f"gate passed: failing "
                f"{after[serving]['failing'] - baseline[serving]['failing']:+.2f} "
                f"points at serving temperature")
        else:
            log("gate failed — merged model not shipped")
        return report, out_dir, 0 if passed else 1
    finally:
        # A refused merge leaves no partial directory behind.
        if os.path.isdir(partial_dir):
            shutil.rmtree(partial_dir, ignore_errors=True)
