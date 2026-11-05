"""MCP server: the same pipeline, callable from any MCP harness.

Tools are versioned by the kit version (see ``wsp_version``). Tool
names and parameters are stable: later versions add tools, they do
not rename or remove these ones, so harnesses built against v0.1.0
keep working.

State is per-process and in memory: run ``toy_baseline`` first, then
the other tools work on that session's model. The full pipeline is
``toy_demo``; the granular tools exist so a harness can drive the
steps itself and inspect each one.

Needs the ``mcp`` package: pip install 'wsp-kit[mcp]'.
"""

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    raise SystemExit(
        "the mcp package is not installed — run: pip install 'wsp-kit[mcp]'"
    ) from None

from . import __version__, diagnose, edit, gate, pipeline, toy

mcp = MCPServer("wsp-kit")

STATE = {}


def _need_session():
    if "broken" not in STATE:
        raise RuntimeError("no model loaded — run toy_baseline first")
    return STATE


def _as_floats(metrics):
    return {k: float(v) for k, v in metrics.items()}


@mcp.tool()
def wsp_version():
    """Kit and report-format versions."""
    from . import REPORT_VERSION
    return {"kit_version": __version__, "report_version": REPORT_VERSION}


@mcp.tool()
def toy_baseline():
    """Build the toy model, plant the defect, and measure the baseline."""
    clean = toy.build_clean_weights()
    sets = toy.build_sets()
    targets = toy.clean_targets(clean, sets)
    broken, _ = toy.plant_defect(clean, sets["failing"])
    STATE.update(clean=clean, sets=sets, targets=targets, broken=broken)
    return {
        "serving_temperature": pipeline.SERVING_TEMPERATURE,
        "baseline": _as_floats(
            diagnose.baseline(broken, sets, targets, pipeline.SERVING_TEMPERATURE)
        ),
    }


@mcp.tool()
def toy_temperature_sweep():
    """Step 0: try temperatures before touching any weight."""
    s = _need_session()
    sweep = diagnose.temperature_sweep(s["broken"], s["sets"], s["targets"])
    return {str(t): _as_floats(m) for t, m in sweep.items()}


@mcp.tool()
def toy_prove_cause():
    """Ablation test: does zeroing the suspect row move the failure?"""
    s = _need_session()
    return diagnose.prove_cause(s["broken"], s["sets"], s["targets"])


@mcp.tool()
def toy_estimate_direction():
    """Estimate the failure direction from hidden activations."""
    s = _need_session()
    direction = diagnose.estimate_direction(s["broken"], s["sets"])
    STATE["direction"] = direction
    return {"suspect_row": toy.DEFECT_CLASS, "direction": [float(x) for x in direction]}


@mcp.tool()
def toy_apply_edit(alpha: float):
    """Subtract alpha * direction from the suspect row (1% cap enforced)."""
    s = _need_session()
    if "direction" not in STATE:
        raise RuntimeError("no direction estimated — run toy_estimate_direction first")
    candidate, changed, frac = edit.apply_edit(
        s["broken"], toy.DEFECT_CLASS, STATE["direction"], alpha
    )
    STATE["candidate"] = candidate
    return {"weights_changed": changed, "weights_changed_pct": frac * 100}


@mcp.tool()
def toy_run_gate():
    """Gate the current candidate at the serving temperature and greedy."""
    s = _need_session()
    if "candidate" not in STATE:
        raise RuntimeError("no candidate edit — run toy_apply_edit first")
    per_temp = {}
    passed = True
    for t in pipeline.GATE_TEMPERATURES:
        b = diagnose.baseline(s["broken"], s["sets"], s["targets"], t)
        a = diagnose.baseline(s["candidate"], s["sets"], s["targets"], t)
        g = gate.check_gate(b, a)
        per_temp[str(t)] = g
        passed = passed and g["passed"]
    return {"passed": passed, "by_temperature": per_temp}


@mcp.tool()
def toy_demo():
    """Run the whole toy pipeline and return the report."""
    rep, _edited, _before, _after, _attempts, _sweep, _cause = pipeline.run_pipeline()
    return rep


if __name__ == "__main__":
    mcp.run()
