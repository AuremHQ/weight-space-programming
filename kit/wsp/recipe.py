"""Edit recipes (edit.json): validation, hashing, provenance.

An edit.json is the recipe for reproducing one gate-passing edit:
which model, which layers, which columns, which direction, which
alpha. A recipe is DATA, and a hand-written or tampered one must
never reach a model unchecked, so there are two gates:

1. ``load_recipe`` — structural validation of the file itself:
   size, JSON (NaN/Infinity are rejected, not silently parsed),
   format version, model id, layer count, column/alpha shape.
   No model is needed for this step.
2. ``validate_against_model`` — apply-time validation against the
   loaded model: layer indices exist, columns are unique and in
   range, the direction length equals the hidden size, every number
   is finite, alpha is in range, and the recipe-level 1% cap holds.

``verify_provenance`` ties a recipe to its gate pass: a report.json
must sit next to the edit.json, show a gate pass, carry the same
edit hash and the same model id. It returns the evidence level:
``"ship"`` for a SHIP verdict, ``"quick"`` for QUICK_PASS or for
reports with no verdict at all (every v1 report). This is an
integrity check against accidents and casual tampering, NOT
authentication: someone who can write both files can forge both.
Do not point serve-model at a directory untrusted users can write.

Two edit types exist (Phase 4): ``"subtract"`` (direction + alpha
per layer) and ``"dampen"`` (``dampen_factor`` in [0, 1] per layer,
no direction, no alpha). A dampen layer carrying direction or alpha
is refused: a mixed recipe is malformed.
"""

import hashlib
import json
import math
import os
from datetime import datetime, timezone

from . import __version__ as KIT_VERSION
from .atomic import atomic_write_json
from .real_edit import check_recipe_cap

MAX_RECIPE_BYTES = 5_000_000
MAX_RECIPE_LAYERS = 8
ALPHA_HARD_MAX = 100.0


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _reject_constant(name):
    raise ValueError(f"invalid JSON constant {name!r} in edit.json")


def _parse_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite number {text!r} in edit.json")
    return value


def canonical_layers_bytes(recipe):
    """Canonical bytes of the edit itself (what the hash covers).

    Only the defining fields: format version, model id, dtype, edit
    type, layers. Additive metadata (kit_version, created_at, ...)
    is deliberately excluded, so re-stamping a recipe does not
    change its identity.
    """
    defining = {
        "format_version": 1,
        "model_id": recipe["model_id"],
        "dtype": recipe.get("dtype", "bfloat16"),
        "edit_type": recipe.get("edit_type", "subtract"),
        "layers": recipe["layers"],
    }
    return json.dumps(defining, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def edit_sha256(recipe):
    """SHA-256 of the canonical edit bytes (hex)."""
    return hashlib.sha256(canonical_layers_bytes(recipe)).hexdigest()


def write_recipe(path, recipe):
    """Write a recipe atomically, filling in additive fields.

    Additive fields (never part of the hash): kit_version, protocol,
    model_revision (None if unknown), created_at, edit_type,
    base_state. The caller's dict is not mutated.
    """
    out = dict(recipe)
    out.setdefault("kit_version", KIT_VERSION)
    out.setdefault("protocol", "v1")
    out.setdefault("model_revision", None)
    out.setdefault("created_at", datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"))
    out.setdefault("edit_type", "subtract")
    out.setdefault("base_state", "original")
    atomic_write_json(path, out)


def load_recipe(path):
    """Load and structurally validate an edit.json recipe.

    Returns the recipe dict (edit_type normalised to "subtract"
    when absent). Raises ValueError on any structural problem and
    FileNotFoundError when the file is missing.
    """
    size = os.path.getsize(path)
    if size > MAX_RECIPE_BYTES:
        raise ValueError(
            f"edit.json is {size:,} bytes, over the "
            f"{MAX_RECIPE_BYTES:,} byte limit — refused")
    with open(path) as f:
        recipe = json.load(f, parse_constant=_reject_constant,
                           parse_float=_parse_float)
    if not isinstance(recipe, dict):
        raise ValueError("edit.json must be a JSON object")
    if recipe.get("format_version") != 1:
        raise ValueError(
            f"unsupported edit.json format_version "
            f"{recipe.get('format_version')!r}")
    model_id = recipe.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("edit.json is missing model_id")
    if "dtype" in recipe and not isinstance(recipe["dtype"], str):
        raise ValueError("edit.json dtype must be a string")
    layers = recipe.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("edit.json is missing layers")
    if len(layers) > MAX_RECIPE_LAYERS:
        raise ValueError(
            f"edit.json has {len(layers)} layers, over the "
            f"{MAX_RECIPE_LAYERS} layer limit — refused")
    edit_type = recipe.get("edit_type", "subtract")
    if edit_type not in ("subtract", "dampen"):
        raise ValueError(
            f"unsupported edit_type {edit_type!r} — only 'subtract' "
            "and 'dampen' exist")
    recipe["edit_type"] = edit_type
    for layer in layers:
        _validate_layer_shape(layer, edit_type)
    return recipe


def _validate_layer_shape(layer, edit_type):
    if not isinstance(layer, dict):
        raise ValueError("edit.json layer must be an object")
    index = layer.get("index")
    if not _is_int(index) or index < 0:
        raise ValueError(
            f"edit.json layer index {index!r} must be a "
            "non-negative integer")
    columns = layer.get("columns")
    if (not isinstance(columns, list) or not columns
            or any(not _is_int(c) for c in columns)):
        raise ValueError(
            f"edit.json layer {index}: columns must be a non-empty "
            "list of integers")
    if edit_type == "dampen":
        _validate_dampen_layer(layer, index)
        return
    direction = layer.get("direction")
    if (not isinstance(direction, list) or not direction
            or any(not _is_finite_number(v) for v in direction)):
        raise ValueError(
            f"edit.json layer {index}: direction must be a non-empty "
            "list of finite numbers")
    alpha = layer.get("alpha")
    if (not _is_finite_number(alpha)
            or not 0 < float(alpha) <= ALPHA_HARD_MAX):
        raise ValueError(
            f"edit.json layer {index}: alpha {alpha!r} must be finite "
            f"and in (0, {ALPHA_HARD_MAX}]")


def _validate_dampen_layer(layer, index):
    """A dampen layer carries only a factor in [0, 1]."""
    if "direction" in layer or "alpha" in layer:
        raise ValueError(
            f"edit.json layer {index}: a dampen layer must not carry "
            "direction or alpha — refused")
    factor = layer.get("dampen_factor")
    if (not _is_finite_number(factor)
            or not 0.0 <= float(factor) <= 1.0):
        raise ValueError(
            f"edit.json layer {index}: dampen_factor {factor!r} must "
            "be finite and in [0, 1]")


def validate_against_model(recipe, layer_infos, total_params):
    """Apply-time validation against the loaded model.

    Layer indices must exist, columns must be unique ints in
    [0, intermediate_size), and the recipe-level 1% cap must hold.
    Subtract layers additionally need a direction whose length
    equals the hidden size and an alpha in (0, ALPHA_HARD_MAX];
    dampen layers need a dampen_factor in [0, 1]. Every number
    must be finite. Raises ValueError with a plain reason on any
    failure.
    """
    edit_type = recipe.get("edit_type", "subtract")
    if edit_type not in ("subtract", "dampen"):
        raise ValueError(
            f"unsupported edit_type {edit_type!r} — only 'subtract' "
            "and 'dampen' exist")
    infos = {info["index"]: info for info in layer_infos}
    hidden_sizes = {}
    for layer in recipe["layers"]:
        index = layer["index"]
        info = infos.get(index)
        if info is None:
            raise ValueError(
                f"edit.json layer {index} is not in this model "
                f"({len(infos)} layers) — refused")
        columns = layer["columns"]
        if len(set(columns)) != len(columns):
            raise ValueError(
                f"edit.json layer {index}: duplicate columns — refused")
        intermediate = int(info["intermediate_size"])
        for c in columns:
            if c < 0 or c >= intermediate:
                raise ValueError(
                    f"edit.json layer {index}: column {c} is outside "
                    f"[0, {intermediate}) — refused")
        if edit_type == "dampen":
            factor = layer.get("dampen_factor")
            if (not _is_finite_number(factor)
                    or not 0.0 <= float(factor) <= 1.0):
                raise ValueError(
                    f"edit.json layer {index}: dampen_factor "
                    f"{factor!r} must be finite and in [0, 1] — "
                    "refused")
            hidden_sizes[index] = int(info["hidden_size"])
            continue
        direction = layer["direction"]
        hidden = int(info["hidden_size"])
        if len(direction) != hidden:
            raise ValueError(
                f"edit.json layer {index}: direction has "
                f"{len(direction)} entries, this model needs {hidden} "
                "— refused")
        if any(not _is_finite_number(v) for v in direction):
            raise ValueError(
                f"edit.json layer {index}: direction is not finite — "
                "refused")
        alpha = layer["alpha"]
        if (not _is_finite_number(alpha)
                or not 0 < float(alpha) <= ALPHA_HARD_MAX):
            raise ValueError(
                f"edit.json layer {index}: alpha {alpha!r} must be "
                f"finite and in (0, {ALPHA_HARD_MAX}] — refused")
        hidden_sizes[index] = hidden
    check_recipe_cap(recipe["layers"], hidden_sizes, total_params)


def verify_provenance(recipe, report):
    """Tie a recipe to its gate pass. Returns the evidence level.

    ``report`` is the report.json next to the edit.json (or None).
    Requires: the report shows a gate pass, its
    real_demo.edit_sha256 equals this recipe's hash, and its
    real_demo.model_id equals the recipe's model_id. Returns
    ``"ship"`` for a SHIP verdict, ``"quick"`` for QUICK_PASS or
    for reports with no verdict at all. Raises ValueError with a
    plain reason when the provenance does not hold.
    """
    if report is None:
        raise ValueError(
            "no report.json next to this edit.json — the edit cannot "
            "be tied to a gate pass")
    if not isinstance(report, dict):
        raise ValueError("report.json next to this edit.json is not "
                         "a JSON object")
    gate = report.get("gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ValueError(
            "the report.json next to this edit.json does not show a "
            "gate pass — refused")
    demo = report.get("real_demo")
    if not isinstance(demo, dict):
        raise ValueError(
            "the report.json next to this edit.json has no real_demo "
            "block — refused")
    if demo.get("edit_sha256") != edit_sha256(recipe):
        raise ValueError(
            "the report.json next to this edit.json carries a "
            "different edit hash — refused")
    if demo.get("model_id") != recipe.get("model_id"):
        raise ValueError(
            "the report.json next to this edit.json is for a "
            "different model — refused")
    if gate.get("verdict") == "SHIP":
        return "ship"
    return "quick"
