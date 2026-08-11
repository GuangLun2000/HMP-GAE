# Documentation map

HMP-GAE uses one owner for each kind of information. Link to that owner instead
of copying its content into another document.

| Source of truth | Owns | Does not own |
|---|---|---|
| [`../main.py`](../main.py) → `main()` | Active experiment values | Explanations or historical results |
| [`../README.md`](../README.md) | Setup, execution, outputs, short method overview | Full equations or version history |
| [`../MATH_LOGIC.md`](../MATH_LOGIC.md) | Current mathematical mechanism and code-symbol mapping | Active config values or experiment outcomes |
| [`DECISION.md`](DECISION.md) | Dated design decisions, rejected alternatives, preregistration, falsification rules | Quick-start instructions |
| [`../AGENTS.md`](../AGENTS.md) | Non-obvious workflow constraints and operational pitfalls | Repeated algorithm derivations |
| Code docstrings and tests | Local contracts and executable invariants | Paper narrative |

`CLAUDE.md` is intentionally only an import shim for `AGENTS.md`; never maintain
a second copy of agent guidance there.

## Maintenance rules

When a change affects documentation:

1. Update the owner document first.
2. Replace duplicate explanations elsewhere with a short summary and link.
3. Keep active defaults only in `main()`; persist exact values in each result's
   archived config rather than in prose.
4. Do not attach current performance claims to mechanism descriptions. Put the
   acceptance criteria in `DECISION.md` and judge each run from its archived
   outputs.
5. Refer to code by file and symbol, not by volatile line number.
6. Run `python check_docs.py` after every Markdown edit.

## Change routing

| Change | Required documentation update |
|---|---|
| New user-facing command or output | README |
| Current algorithm or symbol changes | MATH_LOGIC plus local code docstring |
| New design choice, threshold, or rejected alternative | DECISION |
| New workflow hazard or verification requirement | AGENTS |
| Config-only experiment arm | `main()` and the archived result config; no prose snapshot |

Historical decision entries are append-oriented evidence. Correct factual
errors explicitly, but do not rewrite old reasoning to make later outcomes look
preordained.
