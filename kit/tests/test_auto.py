"""Tests for `wsp auto` (easy mode). Plain unittest, no extra deps.

Run from the kit directory: PYTHONPATH=. python3 -m unittest discover -s tests -v
"""

import json
import os
import tempfile
import unittest

from wsp import auto
from wsp.evalfile import load_eval_file


class TestBuildEval(unittest.TestCase):
    def test_valid_eval_passes_validation(self):
        data = auto.build_eval(
            "Qwen/Qwen3-0.6B", "auto-test",
            [("List 60 animals.", "dog")],
            [("What is 2 + 3?", "5")],
            [("List 5 animals.", "dog")])
        self.assertEqual(data["format_version"], 1)
        self.assertEqual(data["model"], "Qwen/Qwen3-0.6B")
        for name in ("failing", "good", "neighbours"):
            self.assertEqual(data["sets"][name]["scorer"], "contains")
        self.assertEqual(
            data["sets"]["failing"]["prompts"][0]["prompt"],
            "List 60 animals.")
        self.assertEqual(
            data["sets"]["failing"]["prompts"][0]["expect"], "dog")

    def test_empty_prompt_rejected(self):
        with self.assertRaises(ValueError):
            auto.build_eval("m", "n", [("", "x")],
                            [("p", "e")], [("p", "e")])

    def test_empty_expect_rejected(self):
        with self.assertRaises(ValueError):
            auto.build_eval("m", "n", [("p", "")],
                            [("p", "e")], [("p", "e")])

    def test_neighbours_default_to_good(self):
        data = auto.build_eval(
            "m", "n", [("f", "e")], [("g", "e")], None)
        self.assertEqual(data["sets"]["neighbours"]["prompts"],
                         data["sets"]["good"]["prompts"])

    def test_eval_roundtrip(self):
        data = auto.build_eval(
            "Qwen/Qwen3-0.6B", "auto-test",
            [("List 60 animals.", "dog")],
            [("What is 2 + 3?", "5")],
            [("List 5 animals.", "dog")])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "e.json")
            with open(path, "w") as f:
                json.dump(data, f)
            back = load_eval_file(path)
            self.assertEqual(back["model"], "Qwen/Qwen3-0.6B")
            self.assertEqual(
                back["sets"]["failing"]["prompts"][0]["expect"], "dog")

    def test_slugify_model(self):
        self.assertEqual(auto.slugify_model("Qwen/Qwen3-0.6B"),
                         "qwen-qwen3-0-6b")
        self.assertEqual(auto.slugify_model("gpt2"), "gpt2")


class TestAskQuestions(unittest.TestCase):
    def test_answers_build_eval(self):
        answers = iter([
            "Qwen/Qwen3-0.6B",        # model
            "List 60 animals.", "dog",  # failing 1
            "",                        # done failing
            "What is 2 + 3?", "5",    # good 1
            "",                        # done good
            "",                        # neighbours: reuse good
        ])
        model, failing, good, neighbours = auto.ask_questions(
            input_fn=lambda prompt: next(answers),
            print_fn=lambda *a: None)
        self.assertEqual(model, "Qwen/Qwen3-0.6B")
        self.assertEqual(failing, [("List 60 animals.", "dog")])
        self.assertEqual(good, [("What is 2 + 3?", "5")])
        self.assertEqual(neighbours, good)

    def test_eof_cancels(self):
        def _raise(prompt):
            raise EOFError

        with self.assertRaises(EOFError):
            auto.ask_questions(input_fn=_raise,
                               print_fn=lambda *a: None)


if __name__ == "__main__":
    unittest.main()
