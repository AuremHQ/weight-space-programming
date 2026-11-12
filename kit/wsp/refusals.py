"""The five hard refusals (SKILL.md): which are enforced in code,
which live in the skill text only.

Truth, as the code has it today (Phase 3, W3.1):

- no_eval_file: yes — the run stops with a plain message, exit 2.
- no_weights: PARTIAL — there is no dedicated pre-check; the model
  loader fails (an API-only id has no weights to load) and the CLI
  prints the loader error plainly and exits 2, nothing edited.
- add_capability, production_branch: no — SKILL.md only.
- safety_removal: no — not prevented in code; the v2 default guard
  bundle DETECTS refusal regression after an edit, it does not
  prevent safety removal.

``render_refusal_block`` renders the README text from this registry,
so the docs cannot drift from the code again: the test asserts the
rendered block appears verbatim in README.md and kit/README.md.
"""

REFUSALS = [
    {
        "id": "no_eval_file",
        "text": "No eval file: no test prompts from you, and no eval "
                "set in the repo.",
        "enforced_in_code": True,
        "where": "evalfile.load_eval_file raises FileNotFoundError; "
                 "cli.cmd_run prints it plainly and exits 2.",
        "test": "test_phase3.TestRefusals."
                "test_no_eval_file_plain_message",
    },
    {
        "id": "no_weights",
        "text": "No weights found: the model is served by an API.",
        "enforced_in_code": False,
        "where": "PARTIAL: no dedicated check — the model loader "
                 "fails (an API-only id has no weights to load) and "
                 "cli.cmd_run prints the loader error plainly and "
                 "exits 2, nothing edited.",
        "test": "test_phase3.TestRefusals."
                "test_no_weights_api_only_id_plain_message",
    },
    {
        "id": "add_capability",
        "text": "'Add capability X' with no diagnosed obstacle.",
        "enforced_in_code": False,
        "where": "SKILL.md only — no code check.",
        "test": None,
    },
    {
        "id": "production_branch",
        "text": "Edit a production branch in place.",
        "enforced_in_code": False,
        "where": "SKILL.md only — no code check.",
        "test": None,
    },
    {
        "id": "safety_removal",
        "text": "Remove safety/refusal behaviour.",
        "enforced_in_code": False,
        "where": "SKILL.md only — not prevented in code; detected "
                 "by the v2 refusal guard (the default guard bundle "
                 "reports refusal regression after an edit).",
        "test": None,
    },
]


def render_refusal_block():
    """The refusal list as it appears in README.md and kit/README.md.

    One line per refusal, with its enforcement state. Generated from
    the registry so the docs stay true to the code.
    """
    labels = {
        "no_eval_file": "No eval file (no test prompts from you)",
        "no_weights": "No weights (API-only model)",
        "add_capability": "Add capability with no diagnosed obstacle",
        "production_branch": "Edit a production branch in place",
        "safety_removal": "Remove safety/refusal behaviour",
    }
    details = {
        "no_eval_file": "the run stops with a plain message (exit 2), "
                        "nothing edited",
        "no_weights": "the model loader fails and the run stops with a "
                      "plain message (exit 2), nothing edited; there is "
                      "no dedicated pre-check",
        "add_capability": "not enforced in code",
        "production_branch": "not enforced in code",
        "safety_removal": "not prevented in code; the v2 default guard "
                          "bundle detects refusal regression after an "
                          "edit, it does not prevent safety removal",
    }
    states = {
        "no_eval_file": "enforced in code",
        "no_weights": "partially enforced",
        "add_capability": "SKILL.md only",
        "production_branch": "SKILL.md only",
        "safety_removal": "SKILL.md only",
    }
    lines = ["Refusals \u2014 what the kit refuses, and how each is "
             "enforced:"]
    for row in REFUSALS:
        rid = row["id"]
        lines.append(f"- {labels[rid]}: {states[rid]} \u2014 "
                     f"{details[rid]}.")
    return "\n".join(lines)

