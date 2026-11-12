# Provenance — Qwen narrow dampen run (first Qwen gate pass)

Run: Qwen/Qwen3-0.6B, eval `loop-fix-qwen3-0.6b-narrow-animals`,
Nov 5 (VM, CPU bf16). Failing 37.78 → 67.22 (+29.44 @ 0.7),
131,072 weights (0.022%), layers [27, 26], dampen factor 0.25.
Evidence level: quick (3 failing prompts). This report predates
the kit's provenance hash (no `edit_sha256` in `real_demo`), so
`wsp apply-edit` needs `--allow-unverified` for this run — the
hash check cannot tie a pre-Phase-1 report to its recipe. The
tensor hash below is the check that does tie them: apply the
recipe to the original model and compare.

## Files (sha256 of the file bytes)

| File | sha256 |
| --- | --- |
| `report.json` | `0903367db18e7343223ee15a05bf357b2dbe6041b3000a1121d93eef3129fed8` |
| `edit.json` | `0fde5a2600ead6ffa02c6b469e99d46d85bb12bb8f76810496d59558da8e9e21` |
| `../../kit/evals/loop-fix-qwen3-0.6b-narrow-animals.json` | `66e74922b6848e649df0da95b5c3debda916a9bbc2aa37e23d4273a0a1fed241` |

Copied byte-identical from the build clone
(`wsp-real-out-qwen-narrow-dampen/`); nothing was re-created,
re-run or edited. The original sha256 values are recorded here
beside the copies so anyone can verify them.

## Edited weights (hashes only — no weights ship in this repo)

The edited model (`model.safetensors`, 1,192,135,096 bytes) still
exists in the build clone only. Its file hash:

- `model.safetensors` sha256:
  `b98602661bb2d0914f3c3e66a4867838861209433676336a8734ff03ab30b125`

Changed-tensors sha256 — the value `wsp apply-edit --verify`
compares against. Definition: sha256 over the raw BF16 bytes of
the full down-projection tensors of the edited layers, in
ascending layer order (`model.layers.26.mlp.down_proj.weight`
then `model.layers.27.mlp.down_proj.weight`):

- changed tensors sha256:
  `842972c87e1e99bbb6ee8cd40f8a1d5d96f554939a43673b9f986f8a553a35fc`

Re-derive it from the edited file (read-only, no weights copied):

```python
import hashlib, json, struct
path = "wsp-real-out-qwen-narrow-dampen/model.safetensors"
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    start = 8 + n
    h = hashlib.sha256()
    for name in ("model.layers.26.mlp.down_proj.weight",
                 "model.layers.27.mlp.down_proj.weight"):
        s, e = header[name]["data_offsets"]
        f.seek(start + s)
        h.update(f.read(e - s))
print(h.hexdigest())
```

A MATCH assumes the same base model revision was loaded; a
different base revision honestly reports DIVERGED.

## What is NOT here

The SmolLM2 run (`wsp-real-out-smollm2`) has no report.json or
edit.json in either founder-approved search location (the build
clone or the combined-pipeline staging dir), so nothing for that
run is shipped in this phase — only its eval file
(`kit/evals/loop-fix-smollm2-1.7b.json`,
`c25d05a010c0e6838f40e45a3f4055e52847705dedf05e97a7a3b6e8f7b14c01`).
See the Phase 4 report for the exact missing list.
