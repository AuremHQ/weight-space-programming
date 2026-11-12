"""Phase 2 (W2.11): scorers v2 — strict list parsing, normalisation,
valid_items. Every v1 function stays untouched (asserted below).
"""

import time
import unittest

from wsp.scoring import distinct_score, parse_list_items, score_text


class TestNormalizeItem(unittest.TestCase):
    def test_penguin_again_is_penguin(self):
        from wsp.scoring import normalize_item

        self.assertEqual(normalize_item("Penguin (again)"), "penguin")
        self.assertEqual(normalize_item("Penguin."), "penguin")
        self.assertEqual(normalize_item("penguin"), "penguin")

    def test_trailing_notes_dropped(self):
        from wsp.scoring import normalize_item

        self.assertEqual(normalize_item("Lion (repeat)"), "lion")
        self.assertEqual(normalize_item("Tiger (2)"), "tiger")
        self.assertEqual(normalize_item("Bear (note: see above)"), "bear")

    def test_unicode_nfkc(self):
        from wsp.scoring import normalize_item

        self.assertEqual(normalize_item("\uff30\uff45\uff4e\uff47\uff55\uff49\uff4e"),
                         "penguin")
        self.assertEqual(normalize_item("  Penguin   King  "),
                         "penguin king")

    def test_empty_after_normalise(self):
        from wsp.scoring import normalize_item

        self.assertEqual(normalize_item("..."), "")
        self.assertEqual(normalize_item("(again)"), "")


class TestParseListItemsV2(unittest.TestCase):
    def test_markers_only_when_markers_exist(self):
        from wsp.scoring import parse_list_items_v2

        text = ("Sure! Here is your list:\n1. Penguin\n2. Lion\n"
                "Sorry, I hope this helps!")
        self.assertEqual(parse_list_items_v2(text),
                         ["Penguin", "Lion"])

    def test_markdown_bullets(self):
        from wsp.scoring import parse_list_items_v2

        self.assertEqual(
            parse_list_items_v2("- Penguin\n* Lion\n\u2022 Tiger"),
            ["Penguin", "Lion", "Tiger"])

    def test_paren_numbering(self):
        from wsp.scoring import parse_list_items_v2

        self.assertEqual(parse_list_items_v2("1) Penguin\n2) Lion"),
                         ["Penguin", "Lion"])

    def test_no_markers_short_lines_count(self):
        from wsp.scoring import parse_list_items_v2

        self.assertEqual(parse_list_items_v2("Penguin\nLion"),
                         ["Penguin", "Lion"])

    def test_prose_lines_over_12_words_never_count(self):
        from wsp.scoring import parse_list_items_v2

        prose = ("This is a long prose line with more than twelve "
                 "words in it for sure yes indeed")
        self.assertEqual(parse_list_items_v2(prose), [])
        self.assertEqual(
            parse_list_items_v2("1. " + prose), [])

    def test_empty(self):
        from wsp.scoring import parse_list_items_v2

        self.assertEqual(parse_list_items_v2(""), [])
        self.assertEqual(parse_list_items_v2("   \n\n  "), [])

    def test_unicode_items(self):
        from wsp.scoring import parse_list_items_v2

        self.assertEqual(
            parse_list_items_v2("1. Caf\u00e9\n2. Na\u00efve"),
            ["Caf\u00e9", "Na\u00efve"])


class TestDistinctScoreV2(unittest.TestCase):
    def test_penguin_again_is_one(self):
        from wsp.scoring import distinct_score_v2

        text = "1. Penguin\n2. Penguin (again)\n3. Penguin."
        self.assertAlmostEqual(distinct_score_v2(text, 60),
                               100.0 / 60)

    def test_case_and_punctuation_are_one(self):
        from wsp.scoring import distinct_score_v2

        text = "1. Penguin.\n2. penguin\n3. PENGUIN"
        self.assertAlmostEqual(distinct_score_v2(text, 60),
                               100.0 / 60)

    def test_preamble_not_counted(self):
        from wsp.scoring import distinct_score_v2

        text = ("Sure! Here is a list of animals:\n"
                "1. Penguin\n2. Lion\nSorry about that!")
        self.assertAlmostEqual(distinct_score_v2(text, 60),
                               200.0 / 60)

    def test_valid_items_filtering(self):
        from wsp.scoring import distinct_score_v2

        text = "1. Penguin\n2. Tiger\n3. Lion"
        self.assertAlmostEqual(
            distinct_score_v2(text, 60, valid_items=["penguin", "lion"]),
            200.0 / 60)
        self.assertAlmostEqual(
            distinct_score_v2(text, 60, valid_items=["Tiger"]),
            100.0 / 60)

    def test_valid_items_normalised_too(self):
        from wsp.scoring import distinct_score_v2

        text = "1. Penguin (again)\n2. Lion."
        self.assertAlmostEqual(
            distinct_score_v2(text, 60, valid_items=["PENGUIN.", "lion"]),
            200.0 / 60)

    def test_empty_scores_zero(self):
        from wsp.scoring import distinct_score_v2

        self.assertEqual(distinct_score_v2("", 60), 0.0)
        self.assertEqual(distinct_score_v2("   \n", 60), 0.0)

    def test_prose_only_scores_zero(self):
        from wsp.scoring import distinct_score_v2

        prose = ("This is a long prose line with more than twelve "
                 "words in it for sure yes indeed\n" * 3)
        self.assertEqual(distinct_score_v2(prose, 60), 0.0)

    def test_100k_lines_under_one_second(self):
        from wsp.scoring import distinct_score_v2

        lines = "".join(f"{i + 1}. Item {i}\n" for i in range(100000))
        start = time.monotonic()
        score = distinct_score_v2(lines, 60)
        elapsed = time.monotonic() - start
        self.assertEqual(score, 100.0)
        self.assertLess(elapsed, 1.0, f"took {elapsed:.2f}s")

    def test_bad_expect_count_raises(self):
        from wsp.scoring import distinct_score_v2

        with self.assertRaises(ValueError):
            distinct_score_v2("1. Penguin", 0)


class TestScoreTextV2(unittest.TestCase):
    def test_distinct_uses_v2(self):
        from wsp.scoring import score_text_v2

        self.assertAlmostEqual(
            score_text_v2("distinct", "1. Penguin\n2. Penguin (again)",
                          expect_count=60),
            100.0 / 60)

    def test_other_scorers_delegate_to_v1(self):
        from wsp.scoring import score_text_v2

        self.assertEqual(
            score_text_v2("contains", "The answer is Paris.",
                          expect="Paris"), 100.0)
        self.assertEqual(
            score_text_v2("loop", "hello world", finished=True), 100.0)

    def test_valid_items_passed_through(self):
        from wsp.scoring import score_text_v2

        self.assertAlmostEqual(
            score_text_v2("distinct", "1. Penguin\n2. Tiger",
                          expect_count=60, valid_items=["penguin"]),
            100.0 / 60)

    def test_unknown_scorer_raises(self):
        from wsp.scoring import score_text_v2

        with self.assertRaises(ValueError):
            score_text_v2("nope", "text", expect_count=60)


class TestV1Untouched(unittest.TestCase):
    def test_v1_still_counts_penguin_again_as_two(self):
        # F07 as v1 saw it: kept frozen, on purpose.
        self.assertEqual(
            len(parse_list_items("1. Penguin\n2. Penguin (again)")), 2)
        self.assertAlmostEqual(
            distinct_score("1. Penguin\n2. Penguin (again)", 60),
            200.0 / 60)

    def test_v1_score_text_unchanged(self):
        self.assertAlmostEqual(
            score_text("distinct", "1. Penguin\n2. Penguin (again)",
                       expect_count=60), 200.0 / 60)


if __name__ == "__main__":
    unittest.main()
