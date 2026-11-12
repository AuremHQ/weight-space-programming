"""Deterministic numpy-only fake of ``RealModel`` for golden tests.

The fake models one small transformer well enough for
``run_real_pipeline`` to run end to end without torch and without any
model download. It is NOT a model: it returns canned text whose
``distinct`` score is set by a per-scenario table, and canned MLP
vectors whose diagnosis is fixed (top layers [0, 1], all 16 columns,
direction [1, 0, ...] on each).

Architecture (fixed): 4 layers, hidden size 8, intermediate size 16,
``total_params`` 1,000,000. ``layer_infos`` carries exactly the keys
``layers.describe_layers`` returns.

Prompt classification (tied to ``tests/golden/v1/eval.json``):
- "List 60 different <category>" prompts are the failing set; the
  category word selects the row of the score table.
- "List 5/10 ..." prompts are the neighbours set (always perfect).
- The twelve factual questions are the good set (always answered
  correctly, so the guards never regress). ``generate`` also
  accepts the v2 decoding-knob kwargs (ignored) and the fake
  implements ``nll`` / ``top1`` as edit-independent constants, so
  the v2 perplexity / top-1 guards pass on it; ablation is
  proportional to the overlap with the suspect columns
  {0, 1, 2, 3}, so a random placebo set scores below the real
  ablation (v1 zeroes all 16 columns: full overlap, unchanged).

State: ``zero_columns`` records an ablation, ``apply_edit`` records an
edit (per layer), ``restore`` undoes either by the handle it returned.
``dampen_columns`` (Phase 4) records a dampen edit the same way, and
all three mutate a tiny deterministic weight store (all ones) so the
tensor-hash verify can run against the fake; ``mean_column_norm``
and ``down_proj_weight`` read that store.
``generate`` scores failing prompts from the scenario table for the
current state (ablated > edited > base). Temperature matters only in
the ``cheap_fix`` scenario, where 1.0 and 1.5 restore the failing
subset — that is what Step 0 of the pipeline is for.

Scenarios (``FakeRealModel(scenario)``):
- ``cheap_fix``: a temperature alone fixes the failure (Step 0 wins).
- ``no_failure``: baseline failing set within 5 points of good.
- ``not_causal``: ablation moves the failure by 0 points.
- ``pass``: causal +40, the edit passes the gate at both temps.
- ``no_attempt_passes``: causal +40, but the edit changes nothing.
- ``negative_causal``: ablation makes the failure WORSE by 30 points
  (v1 counts ``abs()`` as proof) and the edit still passes the gate.
- ``partial_pass``: causal +40, but the edit fixes only animals
  and countries — the v2 holdout lower bound does not clear zero
  on some splits (INCONCLUSIVE), while v1 would ship it.
"""

import numpy as np

N_LAYERS = 4
HIDDEN_SIZE = 8
INTERMEDIATE_SIZE = 16
TOTAL_PARAMS = 1_000_000

SCENARIOS = ("cheap_fix", "no_failure", "not_causal", "pass",
             "no_attempt_passes", "negative_causal", "partial_pass")

# Failing prompts, in eval-file order. The first six are the Step 0 /
# diagnosis / causal subset; the last two only appear in full-set
# baselines and gate scores.
_FAILING_CATEGORIES = ("animals", "countries", "fruits", "vegetables",
                       "colors", "instruments", "sports", "cities")
_SINGULAR = {"animals": "Animal", "countries": "Country",
             "fruits": "Fruit", "vegetables": "Vegetable",
             "colors": "Color", "instruments": "Instrument",
             "sports": "Sport", "cities": "City"}

# Base distinct-item counts (of 60) per category, per scenario.
_BASE_COUNTS = {
    "pass": {"animals": 18, "countries": 18, "fruits": 18,
             "vegetables": 18, "colors": 18, "instruments": 18,
             "sports": 60, "cities": 60},
    "not_causal": {"animals": 18, "countries": 18, "fruits": 18,
                   "vegetables": 18, "colors": 18, "instruments": 18,
                   "sports": 60, "cities": 60},
    "negative_causal": {"animals": 18, "countries": 18, "fruits": 18,
                        "vegetables": 18, "colors": 18, "instruments": 18,
                        "sports": 60, "cities": 60},
    "no_attempt_passes": {"animals": 18, "countries": 18, "fruits": 18,
                          "vegetables": 18, "colors": 18,
                          "instruments": 18, "sports": 60, "cities": 60},
    "cheap_fix": {"animals": 18, "countries": 18, "fruits": 18,
                  "vegetables": 18, "colors": 18, "instruments": 18,
                  "sports": 60, "cities": 60},
    # Tuned so Step 0 does NOT trigger (subset 94.17 < 95) but Step 1
    # finds no failure (full-set mean 95.625, gap 4.375 <= 5).
    "no_failure": {"animals": 57, "countries": 57, "fruits": 57,
                   "vegetables": 57, "colors": 57, "instruments": 54,
                   "sports": 60, "cities": 60},
    # Partial effect (v2 INCONCLUSIVE case): the edit fixes only
    # two categories, so the holdout mean gain is real but the
    # lower bound does not clear zero on some splits.
    "partial_pass": {"animals": 18, "countries": 18, "fruits": 18,
                     "vegetables": 18, "colors": 18, "instruments": 18,
                     "sports": 60, "cities": 60},
}

# Distinct-item counts while columns are zeroed (causal ablation).
_ABLATED_COUNTS = {
    "pass": {"animals": 42, "countries": 42, "fruits": 42,
             "vegetables": 42, "colors": 42, "instruments": 42,
             "sports": 60, "cities": 60},
    "partial_pass": {"animals": 42, "countries": 42, "fruits": 42,
                     "vegetables": 42, "colors": 42, "instruments": 42,
                     "sports": 60, "cities": 60},
    "no_attempt_passes": {"animals": 42, "countries": 42, "fruits": 42,
                          "vegetables": 42, "colors": 42,
                          "instruments": 42, "sports": 60, "cities": 60},
    "negative_causal": {"animals": 0, "countries": 0, "fruits": 0,
                        "vegetables": 0, "colors": 0, "instruments": 0,
                        "sports": 60, "cities": 60},
    "not_causal": {"animals": 18, "countries": 18, "fruits": 18,
                   "vegetables": 18, "colors": 18, "instruments": 18,
                   "sports": 60, "cities": 60},
}

# Distinct-item counts while an edit is applied. "full" = two layers
# edited (variants 1 and 2); "partial" = one layer (variant 3).
_EDITED_COUNTS = {
    "pass": {"full": {"animals": 60, "countries": 60, "fruits": 60,
                      "vegetables": 60, "colors": 60, "instruments": 60,
                      "sports": 60, "cities": 60},
             "partial": {"animals": 48, "countries": 48, "fruits": 48,
                         "vegetables": 48, "colors": 48,
                         "instruments": 48, "sports": 60, "cities": 60}},
    "negative_causal": {"full": {"animals": 60, "countries": 60,
                                 "fruits": 60, "vegetables": 60,
                                 "colors": 60, "instruments": 60,
                                 "sports": 60, "cities": 60},
                        "partial": {"animals": 48, "countries": 48,
                                    "fruits": 48, "vegetables": 48,
                                    "colors": 48, "instruments": 48,
                                    "sports": 60, "cities": 60}},
    # Only animals + countries improve (60); everything else stays.
    "partial_pass": {"full": {"animals": 60, "countries": 60,
                              "fruits": 18, "vegetables": 18,
                              "colors": 18, "instruments": 18,
                              "sports": 60, "cities": 60},
                     "partial": {"animals": 48, "countries": 48,
                                 "fruits": 18, "vegetables": 18,
                                 "colors": 18, "instruments": 18,
                                 "sports": 60, "cities": 60}},
    # The edit changes nothing: every attempt fails the gate.
    "no_attempt_passes": {"full": {"animals": 18, "countries": 18,
                                   "fruits": 18, "vegetables": 18,
                                   "colors": 18, "instruments": 18,
                                   "sports": 60, "cities": 60},
                          "partial": {"animals": 18, "countries": 18,
                                      "fruits": 18, "vegetables": 18,
                                      "colors": 18, "instruments": 18,
                                      "sports": 60, "cities": 60}},
}

# The golden eval's four good prompts and their expected answers,
# plus eight more (v2 tests need 12 good prompts: the v2 minimum).
_GOOD_ANSWERS = (
    ("capital of France", "Paris"),
    ("2 + 3", "5"),
    ("color is the sky", "blue"),
    ("Romeo and Juliet", "Shakespeare"),
    ("capital of Japan", "Tokyo"),
    ("capital of Italy", "Rome"),
    ("7 + 8", "15"),
    ("color is grass", "green"),
    ("largest planet", "Jupiter"),
    ("boiling point of water", "100"),
    ("painted the Mona Lisa", "da Vinci"),
    ("first president of the USA", "Washington"),
)

# Columns the diagnosis suspects (the top columns by col_scores,
# see _DOWN_FAIL): ablation of THESE columns is the real effect.
# A placebo (random same-size column set) only overlaps them
# partly, so its effect is proportionally smaller. This is what
# lets the v2 causal test (real effect vs placebo maximum) pass
# on the fake while v1 behaviour (zeroing all 16 columns, full
# overlap) stays byte-identical to the goldens.
SUSPECT_COLUMNS = frozenset((0, 1, 2, 3))

# MLP vectors for capture(). Contrast (neighbours) mlp_out is [1, 0,
# ...] on every layer; failing mlp_out differs by layer, so the
# direction norms are 3.0, 1.0, 0.5, 0.2 and the top layers are
# [0, 1]. down_in is [16..1] on failing prompts and zeros on
# neighbours, so col_scores are [16..1] and every column is picked.
_MLP_FAIL = {
    0: [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    1: [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    2: [1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    3: [1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}
_MLP_CONTRAST = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_DOWN_FAIL = [float(v) for v in range(INTERMEDIATE_SIZE, 0, -1)]
_DOWN_CONTRAST = [0.0] * INTERMEDIATE_SIZE


def _failing_category(prompt):
    """Category word of a 'List 60 different <category>' prompt."""
    marker = "60 different "
    at = prompt.find(marker)
    if at < 0:
        return None
    word = prompt[at + len(marker):].split()[0].strip(",.")
    return word if word in _SINGULAR else None


def _list_text(singular, count):
    """A numbered list with exactly ``count`` distinct items."""
    return "".join(f"{i + 1}. {singular} {i + 1}\n"
                   for i in range(count))


class FakeRealModel:
    """A deterministic stand-in for ``RealModel`` (see module docstring)."""

    def __init__(self, scenario):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown fake scenario {scenario!r} "
                             f"(use one of {list(SCENARIOS)})")
        self.scenario = scenario
        self.total_params = TOTAL_PARAMS
        self.use_chat_template = True
        self.layer_infos = [{
            "index": i,
            "block_path": f"model.layers[{i}]",
            "mlp_name": "mlp",
            "down_proj_name": "down_proj",
            "weight_layout": "out_in",
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "down_proj_numel": HIDDEN_SIZE * INTERMEDIATE_SIZE,
        } for i in range(N_LAYERS)]
        self._ablated = set()
        self._ablated_cols = {}
        self._edited = {}
        self.revision = None
        self.save_calls = []
        # A tiny deterministic weight store, so the dampen / verify
        # paths can be exercised without torch: one float32 matrix
        # per layer (hidden x intermediate), all ones. Edits mutate
        # it exactly like RealModel mutates real weights; scoring
        # still reads _edited / _ablated only, so the goldens and
        # the v2 fixture scores are unchanged.
        self._weights = {
            i: np.ones((HIDDEN_SIZE, INTERMEDIATE_SIZE),
                       dtype=np.float32)
            for i in range(N_LAYERS)
        }
        self._saved_slices = {}  # (kind, layer) -> (columns, copy)

    # -- generation --------------------------------------------------

    def _ablated_fraction(self):
        """Mean overlap of the zeroed columns with SUSPECT_COLUMNS.

        1.0 when every suspect column is zeroed (the v1 ablation,
        which zeroes all 16 columns, and the v2 ablation with
        top_columns=4); proportionally less for a random placebo
        set. 0.0 when nothing is ablated.
        """
        if not self._ablated_cols:
            return 0.0
        fracs = []
        for cols in self._ablated_cols.values():
            fracs.append(len(set(cols) & SUSPECT_COLUMNS)
                         / len(SUSPECT_COLUMNS))
        return sum(fracs) / len(fracs)

    def _failing_count(self, category, temperature):
        if self.scenario == "cheap_fix" and temperature in (1.0, 1.5):
            return 60
        if self._ablated:
            table = _ABLATED_COUNTS.get(self.scenario)
            if table is not None:
                base = _BASE_COUNTS[self.scenario][category]
                return int(round(
                    base + (table[category] - base)
                    * self._ablated_fraction()))
        if self._edited:
            table = _EDITED_COUNTS.get(self.scenario)
            if table is not None:
                kind = "full" if len(self._edited) >= 2 else "partial"
                return table[kind][category]
        return _BASE_COUNTS[self.scenario][category]

    def generate(self, prompt, max_new_tokens, temperature, seed,
                 repetition_penalty=None, no_repeat_ngram_size=None):
        """Return (text, finished, new_token_count). Deterministic.

        The two decoding-knob kwargs are accepted and ignored: the
        fake's outputs do not depend on them (only the cheap_fix
        scenario's temperature does). RealModel passes them through
        to HF generate; the fake keeps the 4-argument call working.
        """
        category = _failing_category(prompt)
        if category is not None:
            count = self._failing_count(category, temperature)
            text = _list_text(_SINGULAR[category], count)
            return text, True, len(text.split())
        if prompt.startswith("List "):
            # Neighbours set: short lists, always complete.
            parts = prompt.split()
            try:
                count = int(parts[1])
            except (IndexError, ValueError):
                count = 5
            word = "Animal"
            for key in _SINGULAR:
                if key in prompt:
                    word = _SINGULAR[key]
                    break
            text = _list_text(word, count)
            return text, True, len(text.split())
        for needle, answer in _GOOD_ANSWERS:
            if needle in prompt:
                text = f"The answer is {answer}."
                return text, True, len(text.split())
        return "I don't know.", True, 3

    # -- diagnosis capture -------------------------------------------

    def capture(self, prompt, max_new_tokens, layer_indices):
        """Mean MLP vectors per layer (see module docstring)."""
        failing = _failing_category(prompt) is not None
        result = {}
        for i in layer_indices:
            if failing:
                mlp = np.array(_MLP_FAIL[i], dtype=np.float64)
                down = np.array(_DOWN_FAIL, dtype=np.float64)
            else:
                mlp = np.array(_MLP_CONTRAST, dtype=np.float64)
                down = np.array(_DOWN_CONTRAST, dtype=np.float64)
            result[i] = {"mlp_out": mlp, "down_in": down}
        return result

    # -- edits ---------------------------------------------------------

    def apply_edit(self, layer_index, columns, direction, alpha):
        """Record the edit. Returns (changed_count, saved_handle)."""
        changed = len(columns) * HIDDEN_SIZE
        self._saved_slices[("edit", layer_index)] = (
            list(columns), self._weights[layer_index][:, list(columns)].copy())
        direction = np.asarray(direction, dtype=np.float32)
        self._weights[layer_index][:, list(columns)] -= (
            float(alpha) * direction[:, None])
        self._edited[layer_index] = float(alpha)
        return changed, ("edit", layer_index)

    def dampen_columns(self, layer_index, columns, factor):
        """Scale columns by ``factor`` in [0, 1] (Phase 4).

        Records the edit like apply_edit (so scoring treats a
        dampened layer as edited), and mutates the weight store.
        Returns (changed_count, saved_handle). Raises ValueError
        for a factor outside [0, 1].
        """
        factor = float(factor)
        if not 0.0 <= factor <= 1.0:
            raise ValueError(
                f"dampen factor must be in [0, 1], got {factor!r}")
        changed = len(columns) * HIDDEN_SIZE
        self._saved_slices[("dampen", layer_index)] = (
            list(columns), self._weights[layer_index][:, list(columns)].copy())
        self._weights[layer_index][:, list(columns)] *= factor
        self._edited[layer_index] = factor
        return changed, ("dampen", layer_index)

    def mean_column_norm(self, layer_index, columns):
        """Mean L2 norm of the selected down-proj columns (Phase 4)."""
        cols = self._weights[layer_index][:, list(columns)]
        return float(np.linalg.norm(cols, axis=0).mean())

    def down_proj_weight(self, layer_index):
        """The down-proj weight matrix (for the tensor-hash verify)."""
        return self._weights[layer_index]

    def zero_columns(self, layer_index, columns):
        """Record the ablation. Returns a saved handle for restore."""
        self._saved_slices[("zero", layer_index)] = (
            list(columns), self._weights[layer_index][:, list(columns)].copy())
        self._weights[layer_index][:, list(columns)] = 0
        self._ablated.add(layer_index)
        self._ablated_cols[layer_index] = set(columns)
        return ("zero", layer_index)

    def restore(self, saved):
        """Undo one apply_edit / dampen_columns / zero_columns handle."""
        kind, layer_index = saved
        if kind == "edit":
            self._edited.pop(layer_index, None)
        elif kind == "dampen":
            self._edited.pop(layer_index, None)
        elif kind == "zero":
            self._ablated.discard(layer_index)
            self._ablated_cols.pop(layer_index, None)
        else:  # pragma: no cover - defensive; handles are ours
            raise ValueError(f"unknown saved handle {saved!r}")
        entry = self._saved_slices.pop((kind, layer_index), None)
        if entry is not None:
            columns, orig = entry
            self._weights[layer_index][:, list(columns)] = orig

    # -- guards (v2): canned, edit-independent values -----------------

    def nll(self, text):
        """Mean negative log-likelihood per token (fake: constant).

        Constant across edits, so the v2 perplexity ratio is 1.0.
        """
        return 2.0

    def top1(self, text):
        """Top-1 next-token id after ``text`` (fake: constant).

        Constant across edits, so v2 top-1 agreement is 1.0.
        """
        return 5

    def save(self, out_dir, force=False):
        """Record the save (no weights exist for a fake)."""
        import json
        import os

        record = {"scenario": self.scenario,
                  "edited_layers": sorted(self._edited)}
        with open(os.path.join(out_dir, "fake-model.json"), "w") as f:
            json.dump(record, f)
        self.save_calls.append(out_dir)
        return out_dir
