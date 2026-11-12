"""wsp status: the wins feed. Green only, by design.

`wsp status` prints one line; `wsp serve` shows the page. Both read
real report.json files — never hardcoded numbers — and show ONLY
gate-passing runs. A run the gate refused never appears here: the full
honest record lives in the repo's REAL-RUNS.md, linked from the page
footer. The gate is framed the way it works: every edit earns its
place — only gate-passing edits ship.

Chip states: RUNNING while a run is active (a `.wsp-running` marker
the real pipeline writes at run start and removes when the report is
written), PASSED when the latest completed run passed the gate, IDLE
otherwise — always showing the last win. A leftover marker from a
crashed run is ignored once its process is gone, so the chip can
never get stuck on RUNNING.
"""

import html
import json
import os
import time
from datetime import datetime, timezone

MARKER_NAME = ".wsp-running"
REPO_URL = "https://github.com/AuremHQ/weight-space-programming"
RUNS_LOG_URL = REPO_URL + "/blob/main/REAL-RUNS.md"
REFRESH_SECONDS = 15


# ---------------------------------------------------------------- reports

def find_reports(runs_dir):
    """All report.json files under runs_dir, sorted by path.

    Hidden directories and __pycache__ are skipped. Files that do not
    parse as wsp reports are dropped later by load_report.
    """
    found = []
    for root, dirs, files in os.walk(runs_dir):
        dirs[:] = sorted(d for d in dirs
                         if not d.startswith(".") and d != "__pycache__")
        if "report.json" in files:
            found.append(os.path.join(root, "report.json"))
    return sorted(found)


def load_report(path):
    """Parse one report.json; None unless it looks like a wsp report
    (a gate dict plus before/after dicts with a failing score)."""
    try:
        with open(path) as f:
            rep = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(rep, dict):
        return None
    gate = rep.get("gate")
    before = rep.get("before")
    after = rep.get("after")
    if not isinstance(gate, dict) or "passed" not in gate:
        return None
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    try:
        float(before["failing"])
        float(after["failing"])
    except (KeyError, TypeError, ValueError):
        return None
    return rep


def _short_model(model):
    """'Qwen/Qwen3-0.6B (real open-weight model, CPU)' -> 'Qwen/Qwen3-0.6B'."""
    if not isinstance(model, str):
        return "unknown model"
    return model.split(" (")[0].strip() or "unknown model"


def verdict_label(verdict):
    """The plain-English label for a gate verdict (W3.8).

    SHIP is standard evidence; QUICK_PASS — and legacy reports with
    no verdict at all — are a quick check, not evidence. Anything
    else (REJECT, QUICK_FAIL, INCONCLUSIVE) is not a pass, so it has
    no label here: it never reaches a win card.
    """
    if verdict == "SHIP":
        return "Approved (standard evidence)"
    if verdict in ("QUICK_PASS", None):
        return "Quick check passed (not evidence)"
    return None


def parse_win(rep, path):
    """A gate-passing report as a win card, else None.

    The kit test is labelled 'Pipeline test (smoke test)' so nobody
    mistakes it for a real-model fix; real runs are named by their
    eval file.
    """
    if not rep.get("gate", {}).get("passed"):
        return None
    before = float(rep["before"]["failing"])
    after = float(rep["after"]["failing"])
    is_real = isinstance(rep.get("real_demo"), dict)
    if is_real:
        name = rep["real_demo"].get("eval_name") or _short_model(rep.get("model"))
        short = name
    else:
        name = "Pipeline test (smoke test)"
        short = "pipeline test"
    gate_temps = rep.get("gate", {}).get("by_temperature", {})
    if isinstance(gate_temps, dict):
        try:
            # JSON object keys are strings ("0.7"); normalise to floats.
            temps = sorted(float(t) for t in gate_temps)
        except (TypeError, ValueError):
            temps = []
    else:
        temps = []
    if not temps and isinstance(rep.get("temperatures_checked"), list):
        temps = [float(t) for t in rep["temperatures_checked"]]
    mtime = os.path.getmtime(path)
    model = _short_model(rep.get("model"))
    if model == "toy":
        model = "kit"
    return {
        "name": name,
        "short": short,
        "model": model,
        "is_real": is_real,
        "verdict": (rep.get("gate") or {}).get("verdict"),
        "verdict_label": verdict_label((rep.get("gate") or {}).get("verdict")),
        "failing_before": before,
        "failing_after": after,
        "gain": after - before,
        "weights_changed": int(rep.get("weights_changed") or 0),
        "weights_changed_pct": float(rep.get("weights_changed_pct") or 0.0),
        "gate_temps": temps,
        "date": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d"),
        "mtime": mtime,
    }


def _causal_proof(rep):
    """Best causal-proof change in a real report, else None."""
    demo = rep.get("real_demo")
    if not isinstance(demo, dict):
        return None
    proof = demo.get("causal_proof")
    if not isinstance(proof, dict):
        return None
    try:
        return float(proof["change_points"])
    except (KeyError, TypeError, ValueError):
        return None


def scan_reports(runs_dir="."):
    """Everything the page and the one-liner need, from real files.

    Returns a dict: wins (newest first), reports (all parsed reports),
    running (marker info or None), best_causal_proof, real_runs,
    real_passes.
    """
    reports = []
    for path in find_reports(runs_dir):
        rep = load_report(path)
        if rep is None:
            continue
        reports.append({
            "rep": rep,
            "path": path,
            "mtime": os.path.getmtime(path),
            "passed": bool(rep["gate"]["passed"]),
            "is_real": isinstance(rep.get("real_demo"), dict),
            "causal": _causal_proof(rep),
        })
    reports.sort(key=lambda r: r["mtime"])
    wins = []
    for r in reports:
        win = parse_win(r["rep"], r["path"])
        if win is not None:
            wins.append(win)
    wins.sort(key=lambda w: w["mtime"], reverse=True)
    causal = [r["causal"] for r in reports if r["causal"] is not None]
    # Verdict counts (W2.13): approved = SHIP (standard evidence);
    # quick passes = gate passed with a QUICK_PASS verdict or no
    # verdict at all (legacy v1 reports); refused = everything else
    # (REJECT, QUICK_FAIL, INCONCLUSIVE — nothing shipped).
    approved = sum(
        1 for r in reports
        if (r["rep"].get("gate") or {}).get("verdict") == "SHIP")
    quick_passes = sum(
        1 for r in reports
        if r["passed"] and (r["rep"].get("gate") or {}).get("verdict")
        in ("QUICK_PASS", None))
    total = len(reports)
    return {
        "wins": wins,
        "reports": reports,
        "running": _find_running_marker(runs_dir),
        "best_causal_proof": max(causal) if causal else None,
        "real_runs": sum(1 for r in reports if r["is_real"]),
        "real_passes": sum(1 for w in wins if w["is_real"]),
        "approved": approved,
        "quick_passes": quick_passes,
        "refused": total - approved - quick_passes,
        "total": total,
    }


# ---------------------------------------------------------------- marker

def write_running_marker(out_dir, model_id, eval_path):
    """Mark a real run as active. Written at run start; removed when
    the report is written (see _write_report in real_pipeline)."""
    os.makedirs(out_dir, exist_ok=True)
    data = {
        "pid": os.getpid(),
        "model_id": model_id,
        "eval": os.path.basename(eval_path),
        "started": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(out_dir, MARKER_NAME), "w") as f:
        json.dump(data, f)


def remove_running_marker(out_dir):
    """Remove the marker; missing is fine (already removed, or the run
    never wrote one)."""
    try:
        os.remove(os.path.join(out_dir, MARKER_NAME))
    except OSError:
        pass


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False
    return True


def _find_running_marker(runs_dir):
    """The live run's marker, or None. A marker whose process is gone
    (a crashed run) is stale and ignored."""
    for root, dirs, _files in os.walk(runs_dir):
        dirs[:] = sorted(d for d in dirs
                         if not d.startswith(".") and d != "__pycache__")
        path = os.path.join(root, MARKER_NAME)
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            pid = int(data["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if _pid_alive(pid):
            return data
    return None


# ---------------------------------------------------------------- chip + line

def chip_state(scan):
    """RUNNING while a run is active, PASSED when the latest completed
    run passed the gate, IDLE otherwise."""
    if scan["running"] is not None:
        return "RUNNING"
    if scan["reports"] and scan["reports"][-1]["passed"]:
        return "PASSED"
    return "IDLE"


def _denominator_text(scan):
    """'{A} Approved (standard evidence), {M} Quick check passed (not
    evidence), {K} refused of {T} runs found here' — the honest
    denominator the line and the page share, in the verdict wording
    (W3.8)."""
    quick = scan["quick_passes"]
    return (f"{scan['approved']} Approved (standard evidence), "
            f"{quick} Quick check passed (not evidence), "
            f"{scan['refused']} refused of "
            f"{scan['total']} runs found here")


def status_line(runs_dir="."):
    """The one-liner `wsp status` prints. Green-only and factual.

    The denominator names every run: approved (SHIP, standard
    evidence), quick passes (gate passed, QUICK evidence or legacy
    v1), refused (nothing shipped). The best gain is shown when a
    win exists; with runs but no wins, the line says so plainly.
    """
    scan = scan_reports(runs_dir)
    chip = chip_state(scan)
    if scan["total"] == 0:
        return f"wsp-kit by Aurem \u25cf {chip} \u2014 no runs found yet"
    denom = _denominator_text(scan)
    wins = scan["wins"]
    if wins:
        best = max(wins, key=lambda w: w["gain"])
        return (f"wsp-kit by Aurem \u25cf {chip} \u2014 {denom} "
                f"\u00b7 best +{best['gain']:.2f} ({best['short']})")
    return (f"wsp-kit by Aurem \u25cf {chip} \u2014 {denom} "
            f"\u00b7 no approved improvements yet")


# ---------------------------------------------------------------- page

def _chip_html(chip, running):
    if chip == "RUNNING":
        label = "RUNNING"
        if running:
            label += " \u2014 " + html.escape(str(running.get("model_id", "")))
        return (f'<span class="chip run"><span class="dot"></span>'
                f'{label}</span>')
    if chip == "PASSED":
        return '<span class="chip">&#10003; PASSED</span>'
    return '<span class="chip idle">&#9675; IDLE</span>'


def _win_card(win):
    temps = ""
    if win["gate_temps"]:
        temps = "Gate passed at " + ", ".join(f"{t:g}" for t in win["gate_temps"])
    label = win.get("verdict_label") or "Quick check passed (not evidence)"
    return f"""<div class="card">
      <div class="card-top"><h3>{html.escape(win["name"])}</h3>
      <span class="pill">&#10003; Gate passed</span></div>
      <div class="meta">{html.escape(win["model"])} &middot; {html.escape(win["date"])} &middot; {html.escape(label)}</div>
      <div class="scores">Failing {win["failing_before"]:.2f} &rarr; <b>{win["failing_after"]:.2f}</b></div>
      <div class="gain">+{win["gain"]:.2f} points</div>
      <div class="meta">{win["weights_changed"]:,} weights ({win["weights_changed_pct"]:.2f}%) &middot; {html.escape(temps)}</div>
    </div>"""


def _momentum_html(scan):
    items = []
    if scan["real_runs"] > 0:
        items.append(('<span class="tick">&#10003;</span> Pipeline proven '
                      'end-to-end',
                      f'{scan["real_runs"]} real-model run'
                      f'{"s" if scan["real_runs"] != 1 else ""} completed '
                      f'on CPU, no LLM editor'))
    best = scan["best_causal_proof"]
    if best is not None and best > 0:
        items.append(('<span class="tick">&#10003;</span> Diagnosis finds '
                      'causal layers',
                      f'Strongest causal proof so far: +{best:.2f} points'))
    if scan["real_passes"] == 0:
        items.append(('<span class="next-tag">NEXT</span> First passing '
                      'real-model edit',
                      'The next milestone — the gate decides when it ships'))
    else:
        items.append(('<span class="tick">&#10003;</span> Real-model edits '
                      'approved',
                      f'{scan["real_passes"]} approved so far'))
    out = ['<div class="section"><h2>Momentum</h2>']
    for title, sub in items:
        cls = "mile next" if "NEXT" in title else "mile"
        out.append(f'<div class="{cls}"><div class="mile-title">{title}</div>'
                   f'<div class="mile-sub">{html.escape(sub)}</div></div>')
    out.append('</div>')
    return "\n".join(out)


def render_html(scan, updated=None):
    """The full status page, generated fresh from real reports.

    Green, gold, navy and purple only — no red anywhere, and no
    failure language: refused runs simply never appear.
    """
    chip = chip_state(scan)
    wins = scan["wins"]
    best = max(wins, key=lambda w: w["gain"]) if wins else None
    if updated is None:
        updated = datetime.now().strftime("%Y-%m-%d %H:%M")
    hero_best = (f'<div class="best">Biggest gain: '
                 f'+{best["gain"]:.2f} ({html.escape(best["short"])})</div>'
                 if best else "")
    if wins:
        cards = "\n".join(_win_card(w) for w in wins)
    else:
        cards = ('<div class="card"><h3>No approved improvements yet</h3>'
                 '<div class="meta">The first gate-passing edit will '
                 'appear here.</div></div>')
    plural = "s" if len(wins) != 1 else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{REFRESH_SECONDS}">
<title>wsp-kit by Aurem &mdash; Approved improvements</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0a1128; color: #f1f5f9; }}
.wrap {{ max-width: 680px; margin: 0 auto; padding: 20px 16px 40px; }}
header {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
header .name {{ font-size: 20px; font-weight: 800; }}
header .ver {{ color: #94a3b8; font-size: 13px; }}
.chip {{ margin-left: auto; display: inline-flex; align-items: center; gap: 6px; font-weight: 700; font-size: 13px; padding: 6px 12px; border-radius: 999px; background: #16a34a; color: #fff; }}
.chip.idle {{ background: transparent; border: 1px solid #4ade80; color: #4ade80; }}
.dot {{ width: 8px; height: 8px; border-radius: 50%; background: #fff; animation: pulse 1.6s infinite; }}
@keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: .3; }} 100% {{ opacity: 1; }} }}
.tagline {{ color: #cbd5e1; font-size: 14px; margin-top: 10px; }}
.hero {{ text-align: center; padding: 26px 0 6px; }}
.hero .count {{ font-size: 64px; font-weight: 800; color: #f5c542; line-height: 1; }}
.hero .label {{ font-size: 15px; color: #e2e8f0; margin-top: 6px; }}
.hero .best {{ color: #4ade80; font-weight: 700; font-size: 17px; margin-top: 10px; }}
.hero .denom {{ color: #cbd5e1; font-size: 14px; margin-top: 10px; }}
.section {{ margin-top: 22px; }}
.section h2 {{ font-size: 16px; margin: 0 0 4px; color: #f5c542; }}
.card {{ background: #fff; color: #0f172a; border-radius: 14px; padding: 16px; margin-top: 12px; border-left: 6px solid #7c3aed; }}
.card-top {{ display: flex; align-items: center; gap: 8px; justify-content: space-between; }}
.card h3 {{ margin: 0; font-size: 16px; }}
.pill {{ background: #16a34a; color: #fff; font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 999px; white-space: nowrap; }}
.meta {{ color: #475569; font-size: 13px; margin-top: 4px; }}
.scores {{ font-size: 15px; margin-top: 10px; }}
.gain {{ color: #15803d; font-weight: 800; font-size: 22px; margin-top: 4px; }}
.mile {{ background: #7c3aed; color: #fff; border-radius: 12px; padding: 14px; margin-top: 10px; }}
.mile.next {{ background: transparent; border: 1px dashed #f5c542; color: #fde68a; }}
.mile-title {{ font-weight: 700; font-size: 15px; }}
.mile-sub {{ font-size: 13px; opacity: .9; margin-top: 3px; }}
.tick {{ color: #4ade80; }}
.next-tag {{ background: #f5c542; color: #0a1128; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 999px; }}
.cta {{ display: block; text-align: center; margin-top: 24px; background: #f5c542; color: #0a1128; font-weight: 800; font-size: 16px; padding: 14px; border-radius: 12px; text-decoration: none; }}
footer {{ text-align: center; color: #94a3b8; font-size: 13px; margin-top: 18px; line-height: 1.7; }}
footer a {{ color: #4ade80; }}
</style>
</head>
<body>
<div class="wrap">
<header><span class="name">wsp-kit by Aurem</span>{_chip_html(chip, scan["running"])}</header>
<div class="tagline">Every edit earns its place &mdash; only gate-passing edits ship.</div>
<div class="hero">
<div class="count">{len(wins)}</div>
<div class="label">Approved improvement{plural}</div>
{hero_best}
<div class="denom">{html.escape(_denominator_text(scan))}</div>
</div>
<div class="section"><h2>Approved improvements</h2>
{cards}
</div>
{_momentum_html(scan)}
<a class="cta" href="{REPO_URL}">&#9733; Star on GitHub</a>
<footer>Full run log: <a href="{RUNS_LOG_URL}">REAL-RUNS.md</a><br>
Updated {html.escape(updated)} &middot; refreshes every {REFRESH_SECONDS}s &middot; numbers read from report.json</footer>
</div>
</body>
</html>
"""
