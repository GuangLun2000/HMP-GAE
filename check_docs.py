#!/usr/bin/env python3
"""Doc/code consistency guard for HMP-GAE.

Cheap (~0.1s, stdlib-only, CPU) static check that the context docs
(AGENTS.md / CLAUDE.md at the root, everything under docs/, README.md) stay
honest against the code. Encodes the anti-drift rule established 2026-07-08:

  * NO `main.py:<line>` / `main.py#L<line>` refs in the agent docs — main.py
    churns constantly, so line numbers rot. Refer to it by symbol instead
    (`main()` / config-key name).
  * Every relative markdown link target must exist on disk. Links are
    resolved relative to the linking file's own directory, so docs/*.md
    `../` links are checked correctly (2026-08-11, when MATH_LOGIC.md moved
    into docs/ and only code + agent-tool configs stayed at the root).
  * Every config key the docs treat as authoritative must exist in main.py.
  * Every code symbol the docs name (in STABLE files) must exist there.

Line refs INTO stable files (hmp_gae/*, client.py, trust_scorer.py, server.py)
are allowed and intentionally NOT checked here — those files rarely reorder.

Run:  python check_docs.py      (exit 0 = clean, 1 = drift found)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Docs whose main.py line-refs we forbid (the agent-facing convention docs).
AGENT_DOCS = ["AGENTS.md", "CLAUDE.md", "docs/MATH_LOGIC.md"]
# Docs whose markdown links we resolve (every prose doc in the repo).
LINK_DOCS = ["README.md", "AGENTS.md", "CLAUDE.md",
             "docs/README.md", "docs/MATH_LOGIC.md", "docs/DECISION.md"]

# Config keys the docs rely on as authoritative knobs — must exist in main.py.
CONFIG_KEYS = [
    "experiment_name", "num_clients", "num_attackers", "num_rounds",
    "model_name", "dataset", "data_distribution", "dirichlet_alpha",
    "attack_method", "defense_method", "defense_config",
    "hallu_flip_ratio_range", "hallu_flip_mode", "hallu_flip_map",
    "semantic_weight", "trust_mode", "num_byzantine",
    "v4_tau_ratio", "v4_reject_mult", "v5_m_floor", "v5_r_hard",
    "knn_k", "semantic_probe_stratified",
]

# Symbols the docs name, keyed by the file that must define them (def/class).
SYMBOLS = {
    "main.py": ["main"],
    "client.py": ["BenignClient", "local_train"],
    "server.py": ["Server", "run_round"],
    "defense/__init__.py": ["HMPGAEDefense", "FedAvgDefense", "build_defense"],
    "attack/hallucination.py": ["HallucinationAttackerClient", "FlippedLabelDataset"],
    "hmp_gae/runtime.py": ["HMPGAERuntime", "aggregate", "_update_history"],
    "hmp_gae/trust_scorer.py": [
        "weighted_aggregate",
        "v4_cse_reject_weights", "v5_cse_reject_weights",
        "v8_hmp_cse_propagation_weights",
    ],
    "hmp_gae/hypergraph.py": [
        "knn_hypergraph", "semantic_js_similarity",
        "consensus_propagation_hypergraph", "hypergraph_propagation_matrix",
    ],
    "hmp_gae/node_features.py": ["compute_node_features"],
    "hmp_gae/encoder.py": ["HMPEncoder", "HMPLayer"],
    "hmp_gae/decoder.py": ["HyperedgeDecoder", "normalized_cosine_decoder"],
    "hmp_gae/losses.py": ["total_loss_v8", "adjacency_recon_loss", "hist_loss"],
}

MAIN_PY_LINEREF = re.compile(r"main\.py(?::\d+|#L\d+)")
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def check_no_mainpy_linerefs() -> list[str]:
    fails = []
    for doc in AGENT_DOCS:
        for i, line in enumerate(_read(doc).splitlines(), 1):
            for m in MAIN_PY_LINEREF.finditer(line):
                fails.append(f"{doc}:{i}: forbidden main.py line-ref `{m.group(0)}` (use a symbol instead)")
    return fails


def check_links_resolve() -> tuple[list[str], int]:
    fails, n = [], 0
    for doc in LINK_DOCS:
        # Markdown resolves relative links against the LINKING file's own
        # directory, not the repo root — docs/*.md must be able to say ../.
        base = (ROOT / doc).parent
        for target in MD_LINK.findall(_read(doc)):
            t = target.strip()
            if t.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_part = t.split("#", 1)[0].rstrip("/")
            if not path_part:
                continue
            n += 1
            if not (base / path_part).exists():
                fails.append(f"{doc}: dead link -> {t}")
    return fails, n


def check_config_keys() -> tuple[list[str], int]:
    main_src = _read("main.py")
    fails = []
    for key in CONFIG_KEYS:
        if not re.search(r"""['"]%s['"]\s*:""" % re.escape(key), main_src):
            fails.append(f"main.py: config key '{key}' referenced by docs is missing")
    return fails, len(CONFIG_KEYS)


def check_symbols() -> tuple[list[str], int]:
    fails, n = [], 0
    for rel, names in SYMBOLS.items():
        src = _read(rel)
        for name in names:
            n += 1
            if not re.search(r"(?:def|class)\s+%s\b" % re.escape(name), src):
                fails.append(f"{rel}: symbol `{name}` named in docs not found (def/class)")
    return fails, n


def main() -> int:
    print("HMP-GAE doc/code consistency check")
    print("=" * 42)
    checks = [
        ("No stale main.py line-refs in agent docs", lambda: (check_no_mainpy_linerefs(), None)),
        ("Markdown links resolve", lambda: check_links_resolve()),
        ("Config keys exist in main.py", lambda: check_config_keys()),
        ("Referenced symbols exist in code", lambda: check_symbols()),
    ]
    all_fails = []
    for label, fn in checks:
        result = fn()
        fails = result[0]
        count = result[1] if len(result) > 1 else None
        status = "PASS" if not fails else "FAIL"
        suffix = f" ({count} checked)" if count is not None else ""
        print(f"[{status}] {label}{suffix}")
        for f in fails:
            print(f"        - {f}")
        all_fails.extend(fails)
    print("=" * 42)
    if all_fails:
        print(f"FAILED: {len(all_fails)} issue(s). Fix the docs (or update this guard's curated lists).")
        return 1
    print("OK: docs and code are consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
