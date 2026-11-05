"""Generic scorers for real-model eval sets.

Every scorer returns a 0-100 number where HIGHER is better, so the
gate in ``wsp.gate`` works unchanged: primary gain >= 5, guards and
neighbours regression <= 2. A scorer never sees the model, only the
generated text (and, for ``contains``, the expected answer).

Scorers shipped in v0.2.0:

- ``loop``: non-loop score. 100 means no repeated 8-word block was
  found; 0 means the whole output is one repeated block. This is the
  score for the loop failure: a model stuck repeating text scores
  near 0, a model that finishes cleanly scores near 100.
- ``contains``: 100 if the expected substring appears in the output
  (case-insensitive), else 0. For factual guard prompts.
- ``finish``: 100 if generation stopped with EOS before the token
  limit, else 0. For format-break failures where stopping matters.
- ``sequence``: prefix-match score for ordered-list tasks (counting,
  alphabets, days). The prompt carries ``expect_items`` — the exact
  lines expected, in order. Score = 100 * (length of the longest
  correct prefix of the output's non-empty lines) / len(expect_items).
  A model that loops after 47 correct items scores 47 on a 100-item
  list; a clean completion scores 100. This measures the damage a
  loop does, not just whether repetition exists.
- ``distinct``: distinct-delivery score for list tasks. The prompt
  carries ``expect_count`` (how many distinct items were asked for).
  Score = 100 * min(distinct items delivered, expect_count) /
  expect_count. A model that cycles through 15 animals when asked
  for 60 scores 25; a model that delivers 60 distinct items scores
  100. This is the score for the list-loop failure: a small model
  that cannot sustain a long list repeats items instead of finding
  new ones, and a bare repetition check cannot see it when the items
  are numbered (the numbers differ, so no word block repeats).

Custom scorers can be added to SCORERS without touching the gate:
the gate only ever sees 0-100 numbers.
"""

import re

NGRAM = 8


def _words(text):
    return re.findall(r"\S+", text)


def repeated_word_fraction(text, n=NGRAM):
    """Fraction of words covered by some repeated n-word block.

    A block counts as repeated when the same n-word sequence appears
    at least twice. Words inside any repeated block count as covered.
    Short outputs (fewer than n words) cannot repeat a block, so they
    score 0 by definition.
    """
    words = _words(text)
    if len(words) < n:
        return 0.0
    seen = {}
    covered = [False] * len(words)
    for i in range(len(words) - n + 1):
        block = tuple(words[i:i + n])
        if block in seen:
            for j in range(i, i + n):
                covered[j] = True
            for j in range(seen[block], seen[block] + n):
                covered[j] = True
        else:
            seen[block] = i
    return sum(covered) / len(words)


def loop_score(text):
    """Non-loop score: 100 minus the repeated-word percentage."""
    return 100.0 * (1.0 - repeated_word_fraction(text))


def contains_score(text, expect):
    """100 if the expected answer appears, else 0."""
    if expect is None:
        raise ValueError("contains scorer needs an 'expect' string")
    return 100.0 if expect.lower() in text.lower() else 0.0


def finish_score(_text, finished):
    """100 if generation stopped with EOS before the token limit."""
    return 100.0 if finished else 0.0


def sequence_score(text, expect_items):
    """Prefix-match score for ordered-list outputs.

    Compares the output's non-empty lines, in order, against
    ``expect_items``. The score is the longest correct prefix as a
    percentage of the expected list. A loop after 47 correct items
    scores 47 on a 100-item list; a clean completion scores 100.
    """
    if not expect_items:
        raise ValueError("sequence scorer needs a non-empty 'expect_items' list")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    prefix = 0
    for got, want in zip(lines, expect_items):
        if got == want:
            prefix += 1
        else:
            break
    return 100.0 * prefix / len(expect_items)


_ITEM_RE = re.compile(r"^\s*\d+[.\)]\s*(.+?)\s*$")


def parse_list_items(text):
    """Item lines from a list output, numbering stripped.

    A line counts as an item once list numbering has started: either
    the line itself carries a number prefix ("1. Japan"), or it is a
    non-empty line after the first numbered line (covers lists where
    numbering stops mid-way). Preamble lines before the first numbered
    line are ignored. If no line is numbered at all, every non-empty
    line counts (an unnumbered list is still a list; preamble then
    inflates the count by at most a line or two, capped by min() in
    the score). Returns the item strings in order.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    numbered = []
    started = False
    for ln in lines:
        if not ln:
            continue
        m = _ITEM_RE.match(ln)
        if m:
            started = True
            numbered.append(m.group(1).strip())
        elif started:
            numbered.append(ln)
    if numbered:
        return numbered
    return [ln for ln in lines if ln]


def distinct_score(text, expect_count):
    """Distinct-delivery score for list tasks.

    Score = 100 * min(distinct items, expect_count) / expect_count.
    Matching is case-insensitive. An output with no lines at all
    scores 0 (no list was produced).
    """
    if not isinstance(expect_count, int) or expect_count <= 0:
        raise ValueError("distinct scorer needs a positive 'expect_count'")
    items = parse_list_items(text)
    if not items:
        return 0.0
    distinct = len({it.lower() for it in items})
    return 100.0 * min(distinct, expect_count) / expect_count


SCORERS = {
    "loop": lambda text, expect, finished: loop_score(text),
    "contains": lambda text, expect, finished: contains_score(text, expect),
    "finish": lambda text, expect, finished: finish_score(text, finished),
}


def score_text(scorer, text, expect=None, finished=True, expect_items=None,
               expect_count=None):
    """Score one generation. Raises ValueError on an unknown scorer."""
    if scorer == "sequence":
        return float(sequence_score(text, expect_items))
    if scorer == "distinct":
        return float(distinct_score(text, expect_count))
    try:
        fn = SCORERS[scorer]
    except KeyError:
        raise ValueError(
            f"unknown scorer {scorer!r} — use one of "
            f"{sorted(SCORERS) + ['sequence', 'distinct']}"
        ) from None
    return float(fn(text, expect, finished))
