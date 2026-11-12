"""Helpers for the v1 golden (characterization) tests.

Reports are additive by contract (see ``wsp.report``): later versions
may ADD fields but never rename, remove or re-type existing ones. The
golden check is therefore a recursive SUBSET check — every key and
value in the golden file must be present and equal in the new output,
and extra keys are allowed.

Volatile keys are stripped before comparing, both when generating a
golden and when testing against one: ``kit_version`` (changes every
release), the free-text ``note`` (its wording changes on purpose in
Phase 3), and any timestamp / wall-time field. Measurements are never
stripped.
"""

import copy

# Keys stripped wherever they appear (recursively).
VOLATILE_KEYS = frozenset({
    "kit_version",
    "note",
    "started",
    "finished_at",
    "created_at",
    "timestamp",
    "wall_seconds",
    "wall_time",
    "duration_seconds",
    "elapsed",
})


def strip_volatile(obj):
    """Deep-copy ``obj`` with every volatile key removed."""
    if isinstance(obj, dict):
        return {k: strip_volatile(v) for k, v in obj.items()
                if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [strip_volatile(v) for v in obj]
    return obj


def assert_golden_subset(testcase, golden, actual, path="$"):
    """Assert every golden key/value is present and equal in actual.

    Dicts compare by subset (extra actual keys are allowed); lists
    compare element-wise and must have the same length; scalars
    compare with ==. Failure messages name the JSON path.
    """
    if isinstance(golden, dict):
        testcase.assertIsInstance(actual, dict, f"{path}: not an object")
        for key, want in golden.items():
            testcase.assertIn(key, actual, f"{path}.{key}: missing")
            assert_golden_subset(testcase, want, actual[key],
                                 f"{path}.{key}")
    elif isinstance(golden, list):
        testcase.assertIsInstance(actual, list, f"{path}: not a list")
        testcase.assertEqual(len(actual), len(golden),
                             f"{path}: length {len(actual)} != {len(golden)}")
        for i, (want, got) in enumerate(zip(golden, actual)):
            assert_golden_subset(testcase, want, got, f"{path}[{i}]")
    else:
        testcase.assertEqual(actual, golden, f"{path}: {actual!r} != {golden!r}")
