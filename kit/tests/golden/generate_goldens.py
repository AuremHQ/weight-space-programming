"""Regenerate the v1 golden files from the CURRENT pipeline code.

Run from the kit directory (``PYTHONPATH=.`` is not needed — the
script puts the kit directory on ``sys.path`` itself)::

    python3 tests/golden/generate_goldens.py

For every scenario in ``tests.fakes.SCENARIOS`` it runs
``run_real_pipeline`` with a ``FakeRealModel`` into a temp dir and
writes the stripped ``report.json`` to
``tests/golden/v1/<scenario>.json`` (and the stripped ``edit.json``
of the ``pass`` scenario to ``tests/golden/v1/pass.edit.json``).

Only run this when the pipeline's v1 MEASUREMENT behaviour has
deliberately changed and the change is understood: a regenerated
golden hides a behaviour change instead of pinning it. The golden
tests fail first on any such change — that failure is the point.
"""

import json
import os
import sys
import tempfile

KIT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(KIT_DIR))

from tests.fakes import FakeRealModel, SCENARIOS  # noqa: E402
from tests.golden_utils import strip_volatile  # noqa: E402
from wsp.real_pipeline import run_real_pipeline  # noqa: E402

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "v1")
MODEL_ID = "fake-model"


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for scenario in SCENARIOS:
        with tempfile.TemporaryDirectory() as d:
            _rep, out_dir, code = run_real_pipeline(
                model_id=MODEL_ID,
                eval_path=os.path.join(GOLDEN_DIR, "eval.json"),
                out_dir=d,
                model=FakeRealModel(scenario),
                log=lambda *a: None,
            )
            with open(os.path.join(out_dir, "report.json")) as f:
                rep = json.load(f)
            path = os.path.join(GOLDEN_DIR, f"{scenario}.json")
            with open(path, "w") as f:
                json.dump(strip_volatile(rep), f, indent=2)
                f.write("\n")
            print(f"{scenario}: exit {code} -> {path}")
            if scenario == "pass" and code == 0:
                with open(os.path.join(out_dir, "edit.json")) as f:
                    edit = json.load(f)
                edit_path = os.path.join(GOLDEN_DIR, "pass.edit.json")
                with open(edit_path, "w") as f:
                    json.dump(strip_volatile(edit), f, indent=2)
                    f.write("\n")
                print(f"{scenario}: edit.json -> {edit_path}")


if __name__ == "__main__":
    main()
