"""Weight deltas: one verified edit, saved small, applied safely.

A delta is ONLY the weights a gate-passing edit actually changed —
a sparse diff of a weights dict — plus a JSON proof (``proof.json``)
that says which model it came from, what it was measured on, and
what the gate said. Nothing here trains, nothing here phones home,
and nothing here ever overwrites a model: applying a delta writes a
new file or a new directory, and the old model stays exactly as it
was.

Format (version 1). One delta lives in ``<store>/<delta-id>/``:

- ``delta.npz`` — for each changed tensor ``name``: ``{name}__idx``
  (flat C-order indices of the changed elements) and ``{name}__val``
  (their new values, in the target tensor's dtype). Indices are
  int32 when they fit, else int64.
- ``proof.json`` — the manifest: format version, delta id, kit
  version, timestamp, model id + revision (the base version),
  per-tensor shapes/dtypes, the sha256 of the patched tensors as
  they were BEFORE the edit (the base check), the edit itself
  (type, layers, weights changed, the 1% cap), the eval (name,
  file, before/after scores) and the gate (passed, verdict, gain,
  regressions, thresholds), plus the sha256 of ``delta.npz``.

The 1% cap is enforced twice: when a delta is built and when it is
applied. A delta that would touch more is refused before anything
is written, and an apply that fails any check — proof shows no gate
pass, base does not match, cap broken, round-trip not exact —
writes nothing and leaves the old model alone.
"""

import hashlib
import json
import os
import secrets
from datetime import datetime, timezone

import numpy as np

from . import __version__ as KIT_VERSION
from .atomic import atomic_write_json

DELTA_FORMAT_VERSION = 1
# The same 1% cap as the edit itself (wsp.edit / wsp.real_edit):
# a delta is a record of an edit, so it may never cover more.
MAX_DELTA_FRACTION = 0.01
DELTA_DIR_PREFIX = "dlt-"


def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_delta_id():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{DELTA_DIR_PREFIX}{stamp}-{secrets.token_hex(4)}"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensors_sha256(weights, names):
    """SHA-256 (hex) of named tensors, in sorted name order.

    Covers the name, shape, dtype and raw bytes of each tensor, so
    the hash identifies the exact base the delta was made from.
    """
    h = hashlib.sha256()
    for name in sorted(names):
        arr = np.ascontiguousarray(weights[name])
        h.update(name.encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _check_fraction(changed, total):
    if total <= 0:
        raise ValueError("cannot check the 1% cap: empty weights")
    frac = changed / total
    if frac > MAX_DELTA_FRACTION:
        raise ValueError(
            f"delta touches {frac:.2%} of weights, over the 1% cap — "
            "refused as too broad"
        )
    return frac


def compute_delta(before, after):
    """Sparse diff of two weights dicts.

    Returns (delta, stats) where delta maps each changed tensor
    name to ``{"indices", "values", "shape", "dtype"}`` and stats
    carries ``weights_changed``, ``weights_changed_fraction``,
    ``total_params`` and ``tensors_changed``. Only elements that
    actually differ are stored. Raises ValueError on key/shape
    mismatch, on a changed non-float tensor (deltas cover float
    weights only), or on a breach of the 1% cap. Neither input is
    mutated.
    """
    if set(before) != set(after):
        only_before = sorted(set(before) - set(after))[:5]
        only_after = sorted(set(after) - set(before))[:5]
        raise ValueError(
            "weights keys differ — not the same model "
            f"(only in before: {only_before}, only in after: {only_after})"
        )
    total = 0
    changed = 0
    delta = {}
    for name in sorted(before):
        b = np.asarray(before[name])
        a = np.asarray(after[name])
        if b.shape != a.shape:
            raise ValueError(
                f"tensor {name!r} shape mismatch: {b.shape} vs {a.shape}"
            )
        total += int(b.size)
        if np.array_equal(b, a):
            continue
        if not np.issubdtype(b.dtype, np.floating):
            raise ValueError(
                f"non-float tensor {name!r} changed — deltas cover "
                "float weights only"
            )
        diff = b != a
        idx = np.flatnonzero(diff)
        delta[name] = {
            "indices": idx.astype(np.int32)
            if idx.size == 0 or int(idx[-1]) < 2 ** 31 else idx.astype(np.int64),
            "values": np.ascontiguousarray(a.ravel()[idx]),
            "shape": tuple(int(d) for d in b.shape),
            "dtype": str(b.dtype),
        }
        changed += int(idx.size)
    frac = _check_fraction(changed, total)
    stats = {
        "weights_changed": changed,
        "weights_changed_fraction": frac,
        "total_params": total,
        "tensors_changed": len(delta),
    }
    return delta, stats


def build_delta_from_layer_edits(layer_edits, total_params):
    """Build a delta from per-layer edit records.

    Each record is ``{"name", "layout", "columns", "orig_slice",
    "new_tensor"}``: ``name`` is the state-dict name of the edited
    tensor, ``layout`` is ``"out_in"`` (a column is ``weight[:, c]``)
    or ``"in_out"`` (a column is ``weight[c, :]``), ``columns`` are
    the edited column indices, ``orig_slice`` holds their ORIGINAL
    values and ``new_tensor`` the full CURRENT tensor. Only elements
    whose value actually changed are stored. Returns (delta, stats,
    base_hashes) where base_hashes maps each edited tensor to the
    sha256 of its original (pre-edit) values — the base check an
    apply verifies before touching a model.
    """
    delta = {}
    base_hashes = {}
    changed = 0
    for edit in layer_edits:
        name = edit["name"]
        layout = edit["layout"]
        if layout not in ("out_in", "in_out"):
            raise ValueError(
                f"unknown layout {layout!r} for tensor {name!r} — "
                "use 'out_in' or 'in_out'"
            )
        columns = [int(c) for c in edit["columns"]]
        new_full = np.asarray(edit["new_tensor"])
        orig_slice = np.asarray(edit["orig_slice"])
        if layout == "out_in":
            if orig_slice.shape != (new_full.shape[0], len(columns)):
                raise ValueError(
                    f"orig_slice shape {orig_slice.shape} does not match "
                    f"tensor {name!r} shape {new_full.shape} with "
                    f"{len(columns)} out_in columns"
                )
        else:
            if orig_slice.shape != (len(columns), new_full.shape[1]):
                raise ValueError(
                    f"orig_slice shape {orig_slice.shape} does not match "
                    f"tensor {name!r} shape {new_full.shape} with "
                    f"{len(columns)} in_out columns"
                )
        orig_full = new_full.copy()
        if layout == "out_in":
            orig_full[:, columns] = orig_slice
        else:
            orig_full[columns, :] = orig_slice
        base_hashes[name] = tensors_sha256({name: orig_full}, [name])
        if layout == "out_in":
            region_orig = orig_full[:, columns]
            region_new = new_full[:, columns]
            rows = np.arange(new_full.shape[0])[:, None]
            cols = np.array(columns)[None, :]
            flat = (rows * new_full.shape[1] + cols).ravel()
        else:
            region_orig = orig_full[columns, :]
            region_new = new_full[columns, :]
            rows = np.array(columns)[:, None]
            cols = np.arange(new_full.shape[1])[None, :]
            flat = (rows * new_full.shape[1] + cols).ravel()
        diff = region_orig != region_new
        keep = np.flatnonzero(diff.ravel())
        if keep.size == 0:
            continue
        idx = flat[keep]
        vals = np.ascontiguousarray(region_new.ravel()[keep])
        delta[name] = {
            "indices": idx.astype(np.int32)
            if int(idx[-1]) < 2 ** 31 else idx.astype(np.int64),
            "values": vals,
            "shape": tuple(int(d) for d in new_full.shape),
            "dtype": str(new_full.dtype),
        }
        changed += int(keep.size)
    frac = _check_fraction(changed, total_params)
    stats = {
        "weights_changed": changed,
        "weights_changed_fraction": frac,
        "total_params": int(total_params),
        "tensors_changed": len(delta),
    }
    return delta, stats, base_hashes


def _delta_npz_arrays(delta):
    arrays = {}
    for name in sorted(delta):
        arrays[f"{name}__idx"] = delta[name]["indices"]
        arrays[f"{name}__val"] = delta[name]["values"]
    return arrays


def _delta_from_npz(path, proof):
    """Load the sparse arrays, checked against the proof."""
    tensors = proof.get("tensors", {})
    with np.load(path) as data:
        files = set(data.files)
        delta = {}
        for name, info in tensors.items():
            idx_key = f"{name}__idx"
            val_key = f"{name}__val"
            if idx_key not in files or val_key not in files:
                raise ValueError(
                    f"delta file is missing arrays for tensor {name!r}"
                )
            idx = np.array(data[idx_key])
            vals = np.array(data[val_key])
            if idx.shape != vals.shape:
                raise ValueError(
                    f"delta arrays for tensor {name!r} disagree: "
                    f"{idx.shape} indices vs {vals.shape} values"
                )
            if int(idx.size) != int(info["changed"]):
                raise ValueError(
                    f"delta for tensor {name!r} holds {idx.size} entries, "
                    f"proof says {info['changed']}"
                )
            delta[name] = {
                "indices": idx,
                "values": vals,
                "shape": tuple(int(d) for d in info["shape"]),
                "dtype": info["dtype"],
            }
    expected = {f"{name}__idx" for name in tensors}
    expected |= {f"{name}__val" for name in tensors}
    if files != expected:
        extra = sorted(files - expected)[:5]
        raise ValueError(
            f"delta file holds unexpected arrays: {extra}"
        )
    return delta


def _validate_proof(proof):
    """Structural check of a proof.json. Raises ValueError."""
    if not isinstance(proof, dict):
        raise ValueError("proof.json is not an object")
    for key in ("format_version", "delta_id", "kit_version",
                "created_at", "model", "base", "edit", "eval",
                "gate", "files", "tensors"):
        if key not in proof:
            raise ValueError(f"proof.json is missing {key!r}")
    if proof["format_version"] != DELTA_FORMAT_VERSION:
        raise ValueError(
            f"unsupported delta format version "
            f"{proof['format_version']!r} — this kit writes "
            f"version {DELTA_FORMAT_VERSION}"
        )
    for key in ("model_id", "total_params"):
        if key not in proof["model"]:
            raise ValueError(f"proof.json model is missing {key!r}")
    if "tensors_sha256" not in proof["base"]:
        raise ValueError("proof.json base is missing 'tensors_sha256'")
    for key in ("edit_type", "weights_changed",
                "weights_changed_pct", "cap_fraction"):
        if key not in proof["edit"]:
            raise ValueError(f"proof.json edit is missing {key!r}")
    for key in ("before", "after"):
        if key not in proof["eval"]:
            raise ValueError(f"proof.json eval is missing {key!r}")
    for key in ("passed", "verdict"):
        if key not in proof["gate"]:
            raise ValueError(f"proof.json gate is missing {key!r}")
    if "delta" not in proof["files"] or "delta_sha256" not in proof["files"]:
        raise ValueError("proof.json files is missing delta/delta_sha256")
    if not isinstance(proof["tensors"], dict) or not proof["tensors"]:
        raise ValueError("proof.json tensors is empty — a delta "
                         "with no changed tensor is not a delta")
    for name, info in proof["tensors"].items():
        for key in ("shape", "dtype", "changed"):
            if key not in info:
                raise ValueError(
                    f"proof.json tensors[{name!r}] is missing {key!r}"
                )


def check_delta_gate(proof):
    """The local gate check for an apply: (ok, reason).

    A delta may only be applied when its proof records a gate pass,
    its fraction is within the 1% cap, and it carries the base hash
    an apply verifies. This checks the record, not the model — the
    base-hash and round-trip checks run inside ``apply_delta``.
    """
    try:
        _validate_proof(proof)
    except ValueError as exc:
        return False, f"invalid proof: {exc}"
    gate = proof["gate"]
    if not gate["passed"] or gate["verdict"] not in ("SHIP", "QUICK_PASS"):
        return False, (
            f"proof shows no gate pass (verdict "
            f"{gate['verdict']!r}) — a delta with no verified edit "
            f"is never applied"
        )
    if float(proof["edit"]["cap_fraction"]) > MAX_DELTA_FRACTION + 1e-12:
        return False, (
            f"proof cap {proof['edit']['cap_fraction']} is over the "
            f"1% cap — refused"
        )
    if float(proof["edit"]["weights_changed_pct"]) > MAX_DELTA_FRACTION + 1e-12:
        return False, (
            f"proof says the delta touches "
            f"{proof['edit']['weights_changed_pct']:.2%} of weights, "
            f"over the 1% cap — refused"
        )
    if not proof["base"]["tensors_sha256"]:
        return False, "proof carries no base hash — cannot verify the base"
    return True, "gate pass recorded, cap holds, base hash present"


def save_delta(store_dir, delta, proof_fields):
    """Save a delta + proof to a new directory in the store.

    ``proof_fields`` holds the proof minus the fields this function
    fills in: ``delta_id``, ``kit_version``, ``created_at`` and
    ``files`` (delta filename + sha256). It must already carry the
    rest — ``model``, ``base``, ``edit``, ``eval``, ``gate`` and
    ``tensors``. The directory is new every time — an existing id is
    never reused and nothing already saved is overwritten. A proof
    that fails validation writes nothing at all. Returns
    (delta_id, proof).
    """
    if not delta:
        raise ValueError("cannot save an empty delta — nothing changed")
    os.makedirs(store_dir, exist_ok=True)
    for _ in range(10):
        delta_id = _new_delta_id()
        delta_dir = os.path.join(store_dir, delta_id)
        if not os.path.exists(delta_dir):
            break
    else:
        raise ValueError("could not find a free delta id — store is wedged")
    # Build the full proof first: a proof that fails validation
    # must not leave a half-written directory behind.
    proof = {
        "format_version": DELTA_FORMAT_VERSION,
        "delta_id": delta_id,
        "kit_version": KIT_VERSION,
        "created_at": _utc_now(),
        "files": {"delta": "delta.npz", "delta_sha256": ""},
    }
    proof.update(proof_fields)
    proof["format_version"] = DELTA_FORMAT_VERSION
    proof["delta_id"] = delta_id
    proof["kit_version"] = KIT_VERSION
    proof["created_at"] = _utc_now()
    proof["files"] = {"delta": "delta.npz", "delta_sha256": ""}
    _validate_proof(proof)
    os.makedirs(delta_dir)
    try:
        delta_path = os.path.join(delta_dir, "delta.npz")
        np.savez(delta_path, **_delta_npz_arrays(delta))
        proof["files"] = {
            "delta": "delta.npz",
            "delta_sha256": _sha256_file(delta_path),
        }
        atomic_write_json(os.path.join(delta_dir, "proof.json"), proof)
    except BaseException:
        import shutil

        shutil.rmtree(delta_dir, ignore_errors=True)
        raise
    return delta_id, proof


def save_verified_delta(store_dir, before, after, proof_fields):
    """Save a VERIFIED edit as a delta. The hook helper.

    Computes the sparse diff, refuses when the gate in
    ``proof_fields`` did not pass or the diff breaches the 1% cap,
    and saves the delta + proof. Returns (delta_id, proof, stats).
    Raises ValueError on a failed gate, a cap breach, or an
    incomplete proof — a delta is only ever written for a verified
    edit, so a caller that cannot prove the pass must not call this.
    """
    gate = proof_fields.get("gate", {})
    if not gate.get("passed"):
        raise ValueError(
            "refusing to save a delta with no gate pass — deltas "
            "are only written for verified edits"
        )
    delta, stats = compute_delta(before, after)
    if not delta:
        raise ValueError("refusing to save an empty delta — nothing changed")
    full_proof = dict(proof_fields)
    full_proof["base"] = {
        "tensors_sha256": {name: tensors_sha256(before, [name])
                           for name in sorted(delta)},
    }
    full_proof["edit"] = dict(proof_fields.get("edit", {}))
    full_proof["edit"].update({
        "weights_changed": stats["weights_changed"],
        "weights_changed_pct": stats["weights_changed_fraction"],
        "cap_fraction": MAX_DELTA_FRACTION,
    })
    full_proof["tensors"] = {
        name: {
            "shape": list(delta[name]["shape"]),
            "dtype": delta[name]["dtype"],
            "changed": int(delta[name]["indices"].size),
        }
        for name in sorted(delta)
    }
    full_proof.setdefault("model", {})
    full_proof["model"]["total_params"] = stats["total_params"]
    delta_id, proof = save_delta(store_dir, delta, full_proof)
    return delta_id, proof, stats


def load_delta(store_dir, delta_id):
    """Load a saved delta: (delta, proof).

    Verifies the proof structure and the sha256 of ``delta.npz``
    before returning anything. Raises FileNotFoundError when the
    delta is missing and ValueError on any tampering or mismatch.
    """
    delta_dir = os.path.join(store_dir, delta_id)
    proof_path = os.path.join(delta_dir, "proof.json")
    delta_path = os.path.join(delta_dir, "delta.npz")
    if not os.path.isfile(proof_path):
        raise FileNotFoundError(f"no delta {delta_id!r} in {store_dir}")
    if not os.path.isfile(delta_path):
        raise FileNotFoundError(
            f"delta {delta_id!r} has no delta.npz in {store_dir}"
        )
    with open(proof_path) as f:
        proof = json.load(f)
    _validate_proof(proof)
    if proof["delta_id"] != delta_id:
        raise ValueError(
            f"proof.json says {proof['delta_id']!r}, directory says "
            f"{delta_id!r} — mismatched delta"
        )
    actual = _sha256_file(delta_path)
    if actual != proof["files"]["delta_sha256"]:
        raise ValueError(
            f"delta.npz sha256 mismatch for {delta_id!r} — the file "
            "was changed after it was saved"
        )
    delta = _delta_from_npz(delta_path, proof)
    return delta, proof


def list_deltas(store_dir):
    """Proofs of every saved delta, oldest first.

    An empty or missing store is an empty list, not an error. A
    corrupt entry raises ValueError naming the id — a broken proof
    in the store is surfaced, not silently skipped.
    """
    if not os.path.isdir(store_dir):
        return []
    proofs = []
    for name in sorted(os.listdir(store_dir)):
        delta_dir = os.path.join(store_dir, name)
        if not os.path.isdir(delta_dir):
            continue
        proof_path = os.path.join(delta_dir, "proof.json")
        if not os.path.isfile(proof_path):
            raise ValueError(
                f"delta directory {name!r} has no proof.json"
            )
        with open(proof_path) as f:
            proof = json.load(f)
        _validate_proof(proof)
        proofs.append(proof)
    proofs.sort(key=lambda p: (p["created_at"], p["delta_id"]))
    return proofs


def apply_delta(weights, delta, proof):
    """Apply a delta to weights, returning a NEW dict.

    Checks, in order: the proof's local gate (``check_delta_gate``),
    every patched tensor exists with the recorded shape, the base
    hash of each patched tensor matches the proof (the model this
    delta was made from), every index is in range, every value is
    finite, and the fraction is within the 1% cap. Then it patches
    a copy and verifies the round-trip: the diff of the patched
    weights against the input must equal the delta exactly. Any
    failure raises ValueError and the input is untouched — the
    caller writes nothing.
    """
    ok, reason = check_delta_gate(proof)
    if not ok:
        raise ValueError(f"delta refused: {reason}")
    total = sum(int(v.size) for v in weights.values())
    changed = 0
    for name, entry in delta.items():
        if name not in weights:
            raise ValueError(
                f"delta patches tensor {name!r}, which the model "
                "does not have — wrong base"
            )
        base = np.asarray(weights[name])
        if tuple(int(d) for d in base.shape) != tuple(entry["shape"]):
            raise ValueError(
                f"tensor {name!r} shape mismatch: model has "
                f"{base.shape}, delta was made from {entry['shape']} "
                "— wrong base"
            )
        if int(entry["indices"].size) == 0:
            continue
        if int(entry["indices"].min()) < 0 or \
                int(entry["indices"].max()) >= int(base.size):
            raise ValueError(
                f"delta index out of range for tensor {name!r} — "
                "corrupt or wrong base"
            )
        if not np.all(np.isfinite(entry["values"])):
            raise ValueError(
                f"delta values for tensor {name!r} are not finite — "
                "corrupt delta"
            )
        changed += int(entry["indices"].size)
    _check_fraction(changed, total)
    base_hashes = proof["base"]["tensors_sha256"]
    for name in delta:
        expected = base_hashes.get(name)
        if expected is None:
            raise ValueError(
                f"proof carries no base hash for tensor {name!r} — "
                "cannot verify the base"
            )
        actual = tensors_sha256(weights, [name])
        if actual != expected:
            raise ValueError(
                f"base mismatch for tensor {name!r}: this delta was "
                "made from a different base — refusing to apply"
            )
    new = {k: np.array(v, copy=True) for k, v in weights.items()}
    for name, entry in delta.items():
        if int(entry["indices"].size) == 0:
            continue
        target = new[name]
        vals = np.asarray(entry["values"]).astype(target.dtype,
                                                  copy=False)
        flat = target.ravel()
        flat[entry["indices"]] = vals
        new[name] = flat.reshape(target.shape)
    # Round-trip: the patched weights must differ from the input by
    # exactly this delta — nothing more, nothing less.
    check, _stats = compute_delta(weights, new)
    for name, entry in delta.items():
        got = check.get(name)
        if got is None:
            raise ValueError(
                f"round-trip failed for tensor {name!r}: apply "
                "changed nothing — corrupt delta"
            )
        if not np.array_equal(got["indices"], entry["indices"]) or \
                not np.array_equal(got["values"].astype(entry["values"].dtype),
                                   entry["values"]):
            raise ValueError(
                f"round-trip failed for tensor {name!r}: applied "
                "weights do not match the delta — refusing"
            )
    if set(check) != set(delta):
        raise ValueError(
            "round-trip failed: apply changed tensors outside the "
            "delta — refusing"
        )
    return new
