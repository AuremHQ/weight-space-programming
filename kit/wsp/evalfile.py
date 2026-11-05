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


def set_prompts(data, name):
    """Prompt dicts for one set, in file order."""
    return data["sets"][name]["prompts"]


def set_scorer(data, name):
    """Scorer name for one set."""
    return data["sets"][name]["scorer"]
