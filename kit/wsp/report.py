"""The report: one JSON object, versioned, no surprises.

Report format version 1 is stable. Later versions may ADD fields; they
do not rename, remove, or re-type existing ones. Anything built
against v1 — a harness, a script, a dashboard — keeps working.

The report records the sampling settings and the editor, because a
result without its temperature and without its editor is not
reproducible and must not be trusted as if it were.
"""

from . import REPORT_VERSION, __version__


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
        "note": (
            "Weight-space editing was demonstrated with GPT-6 Astra and is "
            "unproven for other models. This edit passed the eval gate "
            "above; verify independently before use."
        ),
    }
