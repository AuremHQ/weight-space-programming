"""Deterministic prompt splits and preflight for protocol v2.

Splits are EXACT: prompts are ranked by sha256 of (seed, set
name, normalised prompt text) and cut by the protocol quotas with
largest-remainder rounding, so 20 failing prompts always give
6/4/10 whatever the file order. When prompts carry an optional
``family``, the quotas apply inside each family and the counts
per split are reported. Preflight runs BEFORE the model is loaded
and before the running marker is written.
"""

import hashlib

from .evalfile import SET_NAMES, set_prompts

PREFLIGHT_FAILING_TEXT = (
    "need at least {n} failing prompts for a decision (found {k}). "
    "Add more of YOUR OWN failing prompts: the kit never invents "
    "them. For a quick personal check use --protocol v1 (result is "
    "labelled QUICK, not evidence)."
)


def normalize_prompt(text):
    """Normalised prompt text for ranking and duplicate checks."""
    return " ".join(str(text).split()).casefold()


def _rank_key(seed, set_name, prompt):
    digest = hashlib.sha256(
        f"{seed}:{set_name}:{normalize_prompt(prompt)}".encode("utf-8")
    ).hexdigest()
    return digest


def _cut(ranked, quotas):
    """Cut an ordered index list by quotas (largest remainder).

    ``ranked`` is the list of indices in rank order; ``quotas``
    maps split name to a fraction (summing to 1). Returns
    {split: [indices]} with exact counts: floors first, then the
    leftover slots go to the largest fractional parts (ties broken
    by rank order, so the result is deterministic).
    """
    names = list(quotas)
    total = len(ranked)
    exact = {name: total * quotas[name] for name in names}
    counts = {name: int(exact[name]) for name in names}
    leftover = total - sum(counts.values())
    # Largest fractional part first; ties by quota order.
    order = sorted(names, key=lambda n: (-(exact[n] - counts[n]),
                                         names.index(n)))
    for name in order[:leftover]:
        counts[name] += 1
    out = {}
    at = 0
    for name in names:
        out[name] = ranked[at:at + counts[name]]
        at += counts[name]
    return out


def assign_splits(data, protocol, seed):
    """Split every set's prompts into diagnose / select / holdout.

    Returns {set_name: {split: [indices into the set's prompt
    list]}}. For protocol v1 (no quotas) every prompt lands in
    "holdout" and the other splits are empty. Family-stratified
    when prompts carry ``family``: quotas apply inside each family.
    """
    result = {}
    for name in SET_NAMES:
        prompts = set_prompts(data, name)
        quotas = protocol.split_quotas.get(name)
        if not quotas:
            result[name] = {"diagnose": [], "select": [],
                            "holdout": list(range(len(prompts)))}
            continue
        # Group indices by family (None = no family given).
        families = {}
        for i, p in enumerate(prompts):
            families.setdefault(p.get("family"), []).append(i)
        merged = {split: [] for split in
                  ("diagnose", "select", "holdout")}
        for family in sorted(families, key=lambda f: "" if f is None
                             else str(f)):
            members = families[family]
            ranked = sorted(
                members,
                key=lambda i: _rank_key(seed, name,
                                        prompts[i]["prompt"]))
            cut = _cut(ranked, quotas)
            for split in quotas:
                merged[split].extend(cut[split])
        # Keep each split in rank order for stable reporting.
        for split in merged:
            merged[split].sort(
                key=lambda i: _rank_key(seed, name,
                                        prompts[i]["prompt"]))
        result[name] = merged
    return result


def split_counts(data, protocol, seed):
    """{set: {split: count}} for the assignment (reporting)."""
    splits = assign_splits(data, protocol, seed)
    return {name: {split: len(idxs)
                   for split, idxs in by_split.items()}
            for name, by_split in splits.items()}


def preflight(data, protocol):
    """Plain problems with this eval under ``protocol``.

    v1 has no minimums and no splits: always []. v2 checks the
    minimum prompts per set, duplicate normalised prompts across
    or within sets, and empty required splits. The failing-count
    problem uses the exact text the pipeline raises (and the CLI
    prints); the other minimums use the same shape.
    """
    if protocol.name == "v1":
        return []
    problems = []
    minimums = protocol.min_prompts
    counts = {}
    for name in SET_NAMES:
        counts[name] = len(set_prompts(data, name))
    if counts["failing"] < minimums["failing"]:
        problems.append(PREFLIGHT_FAILING_TEXT.format(
            n=minimums["failing"], k=counts["failing"]))
    for name, label in (("neighbours", "neighbour"),
                        ("good", "good")):
        if counts[name] < minimums[name]:
            problems.append(
                f"need at least {minimums[name]} {label} prompts "
                f"for a decision (found {counts[name]}). Add more "
                f"of YOUR OWN {label} prompts: the kit never "
                f"invents them. For a quick personal check use "
                f"--protocol v1 (result is labelled QUICK, not "
                f"evidence).")
    # Duplicates: same normalised prompt twice, in one set or
    # across sets (a prompt cannot be both a failure and a guard).
    seen = {}
    for name in SET_NAMES:
        for i, p in enumerate(set_prompts(data, name)):
            norm = normalize_prompt(p["prompt"])
            if norm in seen:
                problems.append(
                    f"duplicate prompt {p['prompt']!r} in sets "
                    f"{seen[norm]} and {name} (prompt {i}) — a "
                    f"prompt cannot appear twice")
            else:
                seen[norm] = name
    # Empty required splits (checked on the assignment itself, so
    # a pathological family split is caught even when the totals
    # look fine).
    splits = assign_splits(data, protocol, seed=0)
    for name in SET_NAMES:
        quotas = protocol.split_quotas.get(name, {})
        for split in quotas:
            if not splits[name][split]:
                problems.append(
                    f"empty split: set {name!r} has no {split!r} "
                    f"prompts under protocol {protocol.name}")
    return problems
