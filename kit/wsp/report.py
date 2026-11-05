"""The report: one JSON object, versioned, no surprises.

Report format version 1 is stable. Later versions may ADD fields; they
do not rename, remove, or re-type existing ones. Anything built
against v1 — a harness, a script, a dashboard — keeps working.

The report records the sampling settings and the editor, because a
result without its temperature and without its editor is not
reproducible and must not be trusted as if it were.
"""

from . import REPORT_VERSION, __version__


def _report_note(gate):
    """Honest note: the canned 'passed' wording must never appear on
    a failed gate. Pass path keeps the original wording; fail path
    states nothing shipped and why (the gate reason)."""
    base = ("Weight-space editing was demonstrated with GPT-6 Astra "
            "and is unproven for other models. ")
    if gate.get("passed"):
        return (base + "This edit passed the eval gate above; verify "
                "independently before use.")
    reason = gate.get("reason", "the eval gate did not pass")
    return (base + "This edit did NOT pass the eval gate "
            f"({reason}) \u2014 nothing shipped; verify independently "
            "before use.")


def make_report(model, edited_row, weights_changed, weights_changed_pct,
                before, after, gate, temperatures_checked, editor, tools,
                attempts_used, attempts):
    return {
        "report_version": REPORT_VERSION,
        "kit_version": __version__,
        "model": model,
        "edited_row": edited_row,
        "weights_changed": weights_changed,
        "weights_changed_pct": weights_changed_pct,
        "before": before,
        "after": after,
        "gate": gate,
        "temperatures_checked": temperatures_checked,
        "editor": editor,
        "tools": tools,
        "attempts_used": attempts_used,
        "attempts": attempts,
        "note": _report_note(gate),
    }
