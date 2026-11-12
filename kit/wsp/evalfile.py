"""Eval file format v1: user-supplied prompts + scoring for `wsp run`.

An eval file is plain JSON. It is the task definition for a real-model
run: which model, which prompts, which scorer per set, which sampling
settings. The pipeline never invents prompts — if the file is missing
or invalid, the run stops before any weight is touched.

Example (shortened)::

    {
      "format_version": 1,
      "name": "loop-fix-qwen3-0.6b",
      "model": "Qwen/Qwen3-0.6B",
      "serving_temperature": 0.7,
      "gate_temperatures": [0.7, 0.0],
      "max_new_tokens": 192,
      "use_chat_template": true,
      "sets": {
        "failing": {"scorer": "loop", "prompts": [{"prompt": "..."}]},
        "good": {"scorer": "contains",
                 "prompts": [{"prompt": "...", "expect": "..."}]},
        "neighbours": {"scorer": "loop", "prompts": [{"prompt": "..."}]}
      }
    }

Rules (all enforced, all failures plain):

- ``format_version`` must be 1. Later versions may ADD fields; they do
  not rename, remove, or re-type existing ones.
- The three set names are fixed — ``failing``, ``good``, ``neighbours``
  — because the gate compares exactly those three.
- Every prompt is an object with a non-empty ``prompt`` string.
- A set scored with ``contains`` needs an ``expect`` string on every
  prompt in that set.
- Temperatures live in [0, 2]; ``max_new_tokens`` is a positive int.
- ``use_chat_template`` defaults to true when the tokenizer defines a
  chat template, false otherwise; the file may force either.
"""

import json
import os

FORMAT_VERSION = 1
SET_NAMES = ("failing", "good", "neighbours")


def load_eval_file(path):
    """Load and validate an eval file. Returns the parsed dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no eval file at {path}")
    with open(path) as f:
        data = json.load(f)
    data = validate_eval(data, path)
    data["_path"] = path
    return data


def validate_eval(data, path="<eval>"):
    """Validate a parsed eval dict in place. Returns it unchanged."""
    if not isinstance(data, dict):
        raise ValueError(f"{path}: eval file must be a JSON object")
    if data.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{path}: format_version must be {FORMAT_VERSION}, "
            f"got {data.get('format_version')!r}"
        )
    serving = data.get("serving_temperature", 0.7)
    if not isinstance(serving, (int, float)) or not 0 <= serving <= 2:
        raise ValueError(f"{path}: serving_temperature must be in [0, 2]")
    gate_temps = data.get("gate_temperatures", [serving, 0.0])
    if (not isinstance(gate_temps, list) or not gate_temps
            or any(not isinstance(t, (int, float)) or not 0 <= t <= 2
                   for t in gate_temps)):
        raise ValueError(f"{path}: gate_temperatures must be a non-empty "
                         "list of temperatures in [0, 2]")
    max_tokens = data.get("max_new_tokens", 192)
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(f"{path}: max_new_tokens must be a positive int")
    use_chat = data.get("use_chat_template", None)
    if use_chat is not None and not isinstance(use_chat, bool):
        raise ValueError(f"{path}: use_chat_template must be true or false")
    sets = data.get("sets")
    if not isinstance(sets, dict):
        raise ValueError(f"{path}: 'sets' must be an object")
    for name in SET_NAMES:
        if name not in sets:
            raise ValueError(f"{path}: missing set {name!r} "
                             f"(need {list(SET_NAMES)})")
        entry = sets[name]
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: set {name!r} must be an object")
        scorer = entry.get("scorer")
        if scorer not in ("loop", "contains", "finish", "sequence",
                          "distinct"):
            raise ValueError(
                f"{path}: set {name!r} has unknown scorer {scorer!r} "
                "(use 'loop', 'contains', 'finish', 'sequence' or "
                "'distinct')"
            )
        set_valid = entry.get("valid_items")
        if set_valid is not None:
            _validate_valid_items(set_valid, scorer, path, name, None)
        prompts = entry.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(f"{path}: set {name!r} needs a non-empty "
                             "'prompts' list")
        for i, p in enumerate(prompts):
            if not isinstance(p, dict) or not p.get("prompt"):
                raise ValueError(
                    f"{path}: set {name!r} prompt {i} needs a non-empty "
                    "'prompt' string"
                )
            family = p.get("family")
            if family is not None and not isinstance(family, str):
                raise ValueError(
                    f"{path}: set {name!r} prompt {i} 'family' must "
                    "be a string"
                )
            valid = p.get("valid_items")
            if valid is not None:
                _validate_valid_items(valid, scorer, path, name, i)
            if scorer == "contains" and not p.get("expect"):
                raise ValueError(
                    f"{path}: set {name!r} prompt {i} uses the 'contains' "
                    "scorer and needs an 'expect' string"
                )
            if scorer == "sequence" and not p.get("expect_items"):
                raise ValueError(
                    f"{path}: set {name!r} prompt {i} uses the 'sequence' "
                    "scorer and needs an 'expect_items' list"
                )
            if scorer == "distinct":
                ec = p.get("expect_count")
                if not isinstance(ec, int) or ec <= 0:
                    raise ValueError(
                        f"{path}: set {name!r} prompt {i} uses the "
                        "'distinct' scorer and needs a positive "
                        "'expect_count' int"
                    )
    return data


def _validate_valid_items(valid, scorer, path, name, index):
    """Validate a ``valid_items`` field (prompt or set level).

    NEW in v2, only with the 'distinct' scorer (never reuse
    ``expect_items``, which belongs to 'sequence'). Either a list
    of non-empty strings, or a non-empty string naming a file with
    one item per line. Raises ValueError otherwise.
    """
    where = f"{path}: set {name!r}" + (
        f" prompt {index}" if index is not None else "")
    if scorer != "distinct":
        raise ValueError(
            f"{where} has 'valid_items', which is only allowed "
            "with the 'distinct' scorer")
    if isinstance(valid, str):
        if not valid:
            raise ValueError(f"{where} 'valid_items' file path "
                             "must be a non-empty string")
        return
    if (not isinstance(valid, list) or not valid
            or any(not isinstance(v, str) or not v for v in valid)):
        raise ValueError(
            f"{where} 'valid_items' must be a list of non-empty "
            "strings (or a file path string)")


def resolve_valid_items(data, name, prompt):
    """The resolved ``valid_items`` list for one prompt, or None.

    Prompt-level wins over set-level. A string value names a file
    with one item per line, resolved relative to the eval file's
    directory when relative. Returns None when the prompt/set
    carries no ``valid_items``.
    """
    valid = prompt.get("valid_items")
    if valid is None:
        valid = data["sets"][name].get("valid_items")
    if valid is None:
        return None
    if isinstance(valid, list):
        return list(valid)
    path = valid
    if not os.path.isabs(path):
        base = os.path.dirname(os.path.abspath(
            data.get("_path", "") or ""))
        path = os.path.join(base, path)
    try:
        with open(path) as f:
            items = [ln.strip() for ln in f if ln.strip()]
    except OSError as exc:
        raise ValueError(
            f"valid_items file not readable: {valid} ({exc})") from exc
    if not items:
        raise ValueError(
            f"valid_items file is empty: {valid}")
    return items


def set_prompts(data, name):
    """Prompt dicts for one set, in file order."""
    return data["sets"][name]["prompts"]


def set_scorer(data, name):
    """Scorer name for one set."""
    return data["sets"][name]["scorer"]
