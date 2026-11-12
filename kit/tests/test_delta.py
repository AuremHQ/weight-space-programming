"""Tests for weight deltas (Phase 1): sparse diff, proof, store,
gate-checked apply, and the `wsp delta` commands.

A delta is only the weights a verified edit actually changed, plus
a JSON proof. Nothing here needs torch: deltas work on plain numpy
weights dicts (the same representation as wsp.weights), and the
real-model hook is exercised through save_verified_delta with
synthetic weights. Run from the kit directory:
  PYTHONPATH=. python3 -m unittest discover -s tests -v
"""

import contextlib
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from wsp import cli, delta, weights


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rng_weights(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 1, (64, 32)).astype(np.float32),
        "W2": rng.normal(0, 1, (32, 64)).astype(np.float32),
        "b": rng.normal(0, 1, (32,)).astype(np.float32),
    }


def _edited(weights_dict, seed=1):
    """A copy with a few elements changed (well under the 1% cap)."""
    rng = np.random.default_rng(seed)
    new = {k: v.copy() for k, v in weights_dict.items()}
    new["W2"][3, 5] += 0.5
    new["W2"][10, 0] -= 1.25
    new["W1"][0, 0] = rng.normal()
    return new


def _proof_fields(**over):
    fields = {
        "model": {"model_id": "toy-model", "model_revision": "abc123"},
        "base": {"tensors_sha256": {}},
        "edit": {"edit_type": "subtract",
                 "layers": [{"index": 0, "columns": [5]}]},
        "eval": {"eval_name": "toy-eval", "eval_file": "toy.json",
                 "before": {"failing": 40.0, "good": 80.0,
                            "neighbours": 90.0},
                 "after": {"failing": 70.0, "good": 79.0,
                           "neighbours": 89.0}},
        "gate": {"passed": True, "verdict": "SHIP",
                 "min_gain_points": 2.0, "max_regress_points": 2.0,
                 "primary_gain_points": 30.0,
                 "regressions_points": {"good": 1.0, "neighbours": 1.0}},
    }
    fields.update(over)
    return fields


def _run_cli(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(argv)
    return code, out.getvalue()


class TestComputeDelta(unittest.TestCase):
    def test_round_trip_is_exact(self):
        before = _rng_weights()
        after = _edited(before)
        before_copy = {k: v.copy() for k, v in before.items()}
        d, stats = delta.compute_delta(before, after)
        self.assertEqual(stats["weights_changed"], 3)
        self.assertEqual(stats["tensors_changed"], 2)
        self.assertLessEqual(stats["weights_changed_fraction"], 0.01)
        self.assertEqual(stats["total_params"], 64 * 32 + 32 * 64 + 32)
        proof = _proof_fields()
        # A round-trip through apply needs a complete proof (the
        # format fields are normally filled in by save_delta).
        proof["format_version"] = delta.DELTA_FORMAT_VERSION
        proof["delta_id"] = "dlt-test"
        proof["kit_version"] = delta.KIT_VERSION
        proof["created_at"] = "2026-11-11T00:00:00Z"
        proof["files"] = {"delta": "delta.npz", "delta_sha256": "x"}
        proof["model"]["total_params"] = stats["total_params"]
        proof["base"] = {"tensors_sha256":
                         {name: delta.tensors_sha256(before, [name])
                          for name in d}}
        proof["tensors"] = {
            name: {"shape": list(d[name]["shape"]),
                   "dtype": d[name]["dtype"],
                   "changed": int(d[name]["indices"].size)}
            for name in d
        }
        proof["edit"]["weights_changed"] = stats["weights_changed"]
        proof["edit"]["weights_changed_pct"] = (
            stats["weights_changed_fraction"])
        proof["edit"]["cap_fraction"] = delta.MAX_DELTA_FRACTION
        result = delta.apply_delta(before, d, proof)
        for name in before:
            self.assertTrue(np.array_equal(result[name], after[name]),
                            f"tensor {name} differs after round-trip")
        # The input is untouched by apply.
        for name in before:
            self.assertTrue(np.array_equal(before[name], before_copy[name]),
                            f"apply mutated the input tensor {name}")
        self.assertEqual(result["W2"][3, 5], after["W2"][3, 5])

    def test_no_change_is_empty(self):
        before = _rng_weights()
        d, stats = delta.compute_delta(before, before)
        self.assertEqual(d, {})
        self.assertEqual(stats["weights_changed"], 0)

    def test_cap_is_enforced(self):
        before = _rng_weights()
        after = {k: v.copy() for k, v in before.items()}
        # Change 60 of 4128 total params (1.45%) — over the 1% cap.
        after["W2"].ravel()[:60] += 1.0
        with self.assertRaises(ValueError):
            delta.compute_delta(before, after)

    def test_key_mismatch_refused(self):
        before = _rng_weights()
        after = dict(before)
        after["W3"] = np.zeros((4,), dtype=np.float32)
        with self.assertRaises(ValueError):
            delta.compute_delta(before, after)

    def test_non_float_change_refused(self):
        before = {"ids": np.array([1, 2, 3]),
                  "W": np.zeros((10, 10), dtype=np.float32)}
        after = {"ids": np.array([1, 2, 4]),
                 "W": np.zeros((10, 10), dtype=np.float32)}
        with self.assertRaises(ValueError):
            delta.compute_delta(before, after)


class TestLayerEdits(unittest.TestCase):
    def test_out_in_layout(self):
        # weight (4, 6); columns 1 and 4 edited; only 3 elements differ.
        new_full = np.zeros((4, 6), dtype=np.float32)
        new_full[:, 1] = [1, 2, 3, 4]
        new_full[:, 4] = [5, 5, 5, 5]
        orig_slice = np.zeros((4, 2), dtype=np.float32)
        orig_slice[:, 0] = [1, 2, 3, 9]  # last element differs
        orig_slice[:, 1] = [5, 5, 0, 5]  # third element differs
        new_full[3, 1] = 4  # differs from 9
        new_full[2, 4] = 5  # differs from 0
        edits = [{"name": "W", "layout": "out_in", "columns": [1, 4],
                  "orig_slice": orig_slice, "new_tensor": new_full}]
        d, stats, hashes = delta.build_delta_from_layer_edits(edits, 10000)
        self.assertEqual(stats["weights_changed"], 2)
        self.assertEqual(sorted(d["W"]["indices"].tolist()),
                         sorted([3 * 6 + 1, 2 * 6 + 4]))
        self.assertIn("W", hashes)
        # The base hash is the ORIGINAL tensor: rebuild it and compare.
        orig_full = new_full.copy()
        orig_full[:, [1, 4]] = orig_slice
        self.assertEqual(hashes["W"],
                         delta.tensors_sha256({"W": orig_full}, ["W"]))

    def test_in_out_layout(self):
        new_full = np.zeros((6, 4), dtype=np.float32)
        new_full[2, :] = [1, 1, 1, 1]
        orig_slice = np.zeros((1, 4), dtype=np.float32)
        orig_slice[0] = [1, 1, 9, 1]  # third element differs
        new_full[2, 2] = 1  # differs from 9
        edits = [{"name": "W", "layout": "in_out", "columns": [2],
                  "orig_slice": orig_slice, "new_tensor": new_full}]
        d, stats, _hashes = delta.build_delta_from_layer_edits(edits, 10000)
        self.assertEqual(stats["weights_changed"], 1)
        self.assertEqual(d["W"]["indices"].tolist(), [2 * 4 + 2])

    def test_unchanged_columns_store_nothing(self):
        # An edit that changes no values is not a delta.
        new_full = np.zeros((4, 6), dtype=np.float32)
        edits = [{"name": "W", "layout": "out_in", "columns": [1],
                  "orig_slice": np.zeros((4, 1), dtype=np.float32),
                  "new_tensor": new_full}]
        d, stats, _hashes = delta.build_delta_from_layer_edits(edits, 10000)
        self.assertEqual(d, {})
        self.assertEqual(stats["weights_changed"], 0)

    def test_layer_edit_cap(self):
        new_full = np.ones((100, 100), dtype=np.float32)
        orig_slice = np.zeros((100, 2), dtype=np.float32)
        edits = [{"name": "W", "layout": "out_in", "columns": [0, 1],
                  "orig_slice": orig_slice, "new_tensor": new_full}]
        # 200 changed of 10000 total = 2% — over the cap.
        with self.assertRaises(ValueError):
            delta.build_delta_from_layer_edits(edits, 10000)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = os.path.join(self.tmp, "deltas")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _saved(self, seed=0):
        before = _rng_weights(seed)
        after = _edited(before, seed + 1)
        fields = _proof_fields()
        return delta.save_verified_delta(self.store, before, after,
                                         fields), before, after

    def test_save_load_round_trip(self):
        (delta_id, proof, stats), before, after = self._saved()
        self.assertTrue(delta_id.startswith("dlt-"))
        d, loaded = delta.load_delta(self.store, delta_id)
        self.assertEqual(loaded["delta_id"], delta_id)
        self.assertEqual(loaded["format_version"], 1)
        self.assertEqual(loaded["kit_version"], delta.KIT_VERSION)
        self.assertIn("created_at", loaded)
        self.assertEqual(loaded["model"]["model_id"], "toy-model")
        self.assertEqual(loaded["model"]["model_revision"], "abc123")
        self.assertEqual(loaded["model"]["total_params"],
                         stats["total_params"])
        self.assertEqual(loaded["edit"]["edit_type"], "subtract")
        self.assertEqual(loaded["edit"]["weights_changed"], 3)
        self.assertEqual(loaded["eval"]["before"]["failing"], 40.0)
        self.assertEqual(loaded["eval"]["after"]["failing"], 70.0)
        self.assertTrue(loaded["gate"]["passed"])
        self.assertEqual(loaded["gate"]["verdict"], "SHIP")
        self.assertEqual(loaded["gate"]["primary_gain_points"], 30.0)
        self.assertIn("delta_sha256", loaded["files"])
        result = delta.apply_delta(before, d, loaded)
        for name in before:
            self.assertTrue(np.array_equal(result[name], after[name]))

    def test_ids_are_unique(self):
        (id1, _p1, _s1), _, _ = self._saved(seed=0)
        (id2, _p2, _s2), _, _ = self._saved(seed=10)
        self.assertNotEqual(id1, id2)
        proofs = delta.list_deltas(self.store)
        self.assertEqual(len(proofs), 2)
        self.assertEqual({p["delta_id"] for p in proofs}, {id1, id2})

    def test_list_empty_store(self):
        self.assertEqual(delta.list_deltas(self.store), [])
        self.assertEqual(delta.list_deltas(
            os.path.join(self.tmp, "nope")), [])

    def test_save_refuses_unverified_edit(self):
        before = _rng_weights()
        after = _edited(before)
        fields = _proof_fields()
        fields["gate"] = {"passed": False, "verdict": "REJECT"}
        with self.assertRaises(ValueError):
            delta.save_verified_delta(self.store, before, after, fields)
        self.assertEqual(delta.list_deltas(self.store), [])

    def test_save_refuses_empty_delta(self):
        before = _rng_weights()
        with self.assertRaises(ValueError):
            delta.save_verified_delta(self.store, before, before,
                                      _proof_fields())

    def test_tampered_delta_file_refused(self):
        (delta_id, _proof, _stats), _, _ = self._saved()
        path = os.path.join(self.store, delta_id, "delta.npz")
        with open(path, "ab") as f:
            f.write(b"tamper")
        with self.assertRaises(ValueError):
            delta.load_delta(self.store, delta_id)

    def test_missing_delta_refused(self):
        with self.assertRaises(FileNotFoundError):
            delta.load_delta(self.store, "dlt-does-not-exist")


class TestApplyGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = os.path.join(self.tmp, "deltas")
        before = _rng_weights()
        after = _edited(before)
        self.delta_id, self.proof, _stats = delta.save_verified_delta(
            self.store, before, after, _proof_fields())
        self.before = before
        self.after = after
        self.delta, self.proof = delta.load_delta(self.store,
                                                  self.delta_id)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_gate_check_ok(self):
        ok, _reason = delta.check_delta_gate(self.proof)
        self.assertTrue(ok)

    def test_gate_check_rejects_failed_proof(self):
        bad = dict(self.proof)
        bad["gate"] = dict(self.proof["gate"])
        bad["gate"]["passed"] = False
        bad["gate"]["verdict"] = "REJECT"
        ok, reason = delta.check_delta_gate(bad)
        self.assertFalse(ok)
        self.assertIn("no gate pass", reason)

    def test_apply_refuses_failed_proof(self):
        bad = dict(self.proof)
        bad["gate"] = dict(self.proof["gate"])
        bad["gate"]["passed"] = False
        bad["gate"]["verdict"] = "REJECT"
        with self.assertRaises(ValueError):
            delta.apply_delta(self.before, self.delta, bad)

    def test_apply_refuses_wrong_base(self):
        other = _rng_weights(seed=99)
        with self.assertRaises(ValueError) as ctx:
            delta.apply_delta(other, self.delta, self.proof)
        self.assertIn("base mismatch", str(ctx.exception))

    def test_apply_refuses_missing_tensor(self):
        smaller = {"W1": self.before["W1"], "b": self.before["b"]}
        with self.assertRaises(ValueError):
            delta.apply_delta(smaller, self.delta, self.proof)


class TestDeltaCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = os.path.join(self.tmp, "deltas")
        before = _rng_weights()
        after = _edited(before)
        self.delta_id, _proof, _stats = delta.save_verified_delta(
            self.store, before, after, _proof_fields())
        self.before = before
        self.after = after
        self.base_path = os.path.join(self.tmp, "base.npz")
        weights.save_weights(self.base_path, before)
        self.base_sha = _sha256_file(self.base_path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_list(self):
        code, out = _run_cli(["delta", "list", "--store", self.store])
        self.assertEqual(code, 0)
        self.assertIn(self.delta_id, out)
        self.assertIn("toy-model", out)
        self.assertIn("SHIP", out)

    def test_list_empty(self):
        code, out = _run_cli(["delta", "list", "--store",
                              os.path.join(self.tmp, "empty")])
        self.assertEqual(code, 0)
        self.assertIn("no deltas", out)

    def test_show(self):
        code, out = _run_cli(["delta", "show", self.delta_id,
                              "--store", self.store])
        self.assertEqual(code, 0)
        shown = json.loads(out)
        self.assertEqual(shown["delta_id"], self.delta_id)
        self.assertEqual(shown["gate"]["verdict"], "SHIP")

    def test_show_missing(self):
        code, out = _run_cli(["delta", "show", "dlt-nope",
                              "--store", self.store])
        self.assertEqual(code, 2)
        self.assertIn("error", out)

    def test_apply_npz_end_to_end(self):
        out_path = os.path.join(self.tmp, "edited.npz")
        code, out = _run_cli(["delta", "apply", self.delta_id,
                              "--model", self.base_path,
                              "--out", out_path,
                              "--store", self.store])
        self.assertEqual(code, 0, out)
        self.assertIn("applied delta", out)
        self.assertIn("old model untouched", out)
        edited = weights.load_weights(out_path)
        for name in self.before:
            self.assertTrue(np.array_equal(edited[name], self.after[name]),
                            f"tensor {name} differs after CLI apply")
        # The input file is byte-identical: never overwritten.
        self.assertEqual(_sha256_file(self.base_path), self.base_sha)

    def test_apply_refuses_existing_out(self):
        out_path = os.path.join(self.tmp, "edited.npz")
        code, _out = _run_cli(["delta", "apply", self.delta_id,
                               "--model", self.base_path,
                               "--out", out_path,
                               "--store", self.store])
        self.assertEqual(code, 0)
        code, out = _run_cli(["delta", "apply", self.delta_id,
                              "--model", self.base_path,
                              "--out", out_path,
                              "--store", self.store])
        self.assertEqual(code, 2)
        self.assertIn("already exists", out)

    def test_apply_refuses_same_path(self):
        code, out = _run_cli(["delta", "apply", self.delta_id,
                              "--model", self.base_path,
                              "--out", self.base_path,
                              "--store", self.store])
        self.assertEqual(code, 2)
        self.assertIn("never overwrite", out)
        self.assertEqual(_sha256_file(self.base_path), self.base_sha)

    def test_apply_refuses_failed_gate_proof(self):
        # Rewrite the proof with a failed gate, keeping the delta.
        proof_path = os.path.join(self.store, self.delta_id, "proof.json")
        with open(proof_path) as f:
            proof = json.load(f)
        proof["gate"]["passed"] = False
        proof["gate"]["verdict"] = "REJECT"
        with open(proof_path, "w") as f:
            json.dump(proof, f, indent=2)
        out_path = os.path.join(self.tmp, "edited.npz")
        code, out = _run_cli(["delta", "apply", self.delta_id,
                              "--model", self.base_path,
                              "--out", out_path,
                              "--store", self.store])
        self.assertEqual(code, 1)
        self.assertIn("refused", out)
        self.assertFalse(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()
