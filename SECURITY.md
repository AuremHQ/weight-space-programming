# Security policy

## Threat model
- The status page (`wsp serve`) and serve-model bind localhost only (127.0.0.1) — they are local tools, not public services.
- Both servers check Host, Origin, and Content-Type, cap request bodies, and clamp generation parameters.
- serve-model applies only gate-passing edits for the exact model, verified against the recipe hash and its report.
- A bearer token (`--token` / `WSP_TOKEN`) is optional; without it, anyone on the machine can POST /generate and /rollback.
- The kit never downloads or runs code from eval files, failures.jsonl, or model outputs — data is data, not instructions.
- What is NOT checked: the recipe + report pair is an integrity check, not authentication — anyone who can write both files can forge both.
- What is NOT prevented: safety-removal is detected by the v2 guard bundle after an edit, not prevented before it.
- Report a vulnerability by email to hello@auremcto.com — do not open a public issue with exploit details.

## What is checked
- Recipe validation (layers, columns, direction length, alpha, 1% cap), provenance (report + edit hash), output-dir safety, and the local-server rules above. See the kit README "Safety flags" table.

## What is not checked
- The kit does not authenticate the model source: a model id you point it at is trusted. Only run models you trust, from sources you trust.
