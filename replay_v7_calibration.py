#!/usr/bin/env python3
"""
V7 offline replay calibration over archived run logs (docs/DECISION.md "V7").

Stdlib-only (no torch / numpy) so it runs on the pure-editing Mac or in Colab
against the archived ``*_results.json`` files. NOTHING here trains or tunes:
this script decides whether the pre-committed V7 candidate grids contain a
zero-false-positive plateau, and if so, which constants the plateau's
pre-registered selection rules pick. If the plateau is empty, V7 is reported
dead — that is a negative result for docs/DECISION.md, not a reason to widen
the grids.

The V7 Tier-2 rule being replayed (trust_scorer.v7_cse_reject_corrob_weights):
inside the cold window (1-indexed logged rounds round_min..round_max), a
client NOT CSE-flagged as-run is corroboration-flagged iff

    v4_ratio > tau_lo   AND   residual >= iso_min

subject to the shared rank-cap budget
(min(k_cap, N - max(1, keep_min)) - #as-run CSE flags), candidates ranked by
(residual desc, ratio desc, index asc). 'residual' is the RAW graph_residual
channel the server persists every round. Rounds where the graph channel was
resolution-gated (graph_min_distinct) are EXCLUDED from every statistic —
the live rule abstains there (runtime passes iso=None), so counting them
would calibrate a rule the live run cannot execute. For archives predating
the 'graph_channel_gated' key, gating is reconstructed from the persisted
residual list: distinct-level count < graph_min_distinct (the archived
defense_config value when present, else the canonical default 4).

ARCHIVE COMPATIBILITY (pre-V4 runs): rounds missing the as-run 'v4_ratio' /
'v4_flagged' channels (archives written before 2026-07-28) are admitted by
exact reconstruction from the per-round 'local_cse' dict:

    ratio_i = cse_i / max(median_lower(cse), 1e-6)

where median_lower is the LOWER of the two middle values (torch.median's
convention — NOT statistics.median, which averages), and the as-run V4 flag
set is re-derived by the V4 rule: rank by (ratio desc, index asc), cap at
min(k_cap, N - max(1, keep_min)), flag only ratio > 1.85. This is a float64
reconstruction of a float32 rule; the ~1e-6 relative error is 4+ orders of
magnitude below the 0.05 tau_lo grid step. Rounds without a complete
local_cse dict are excluded and counted (Phase 0).

PRE-COMMITTED GRIDS AND SELECTION RULES (written before any replay result):
  tau_lo    in {1.30, 1.35, ..., 1.80}
  iso_min   in {5/12, 7/12}          (inter-level midpoints at N=7, knn_k=2)
  round_max in {5, 8, 10}            (round_min fixed at 3)
  * PASS   = zero benign flags (cap-free AND budgeted) over ALL admitted
             benign client-round decisions, on a CONTIGUOUS tau_lo region
             (>= 2 grid points, a plateau not a point) at some
             (iso_min, round_max).
  * SELECT (fully deterministic; windows NEST — the flag set at a smaller
    round_max is a subset of the flag set at a larger one, so zero-FP at the
    largest passing round_max implies zero-FP at every smaller candidate):
      1. iso_min  = the most conservative (largest) iso_min with a passing
                    plateau at any round_max.
      2. The CERTIFIED plateau is the plateau at the LARGEST passing
                    round_max for that iso_min.
      3. tau_lo   = the certified plateau's midpoint, rounded DOWN to the
                    grid (lower-middle element).
      4. round_max = the smallest candidate whose recall at the selected
                    (tau_lo, iso_min) covers >= 80% of the recall at the
                    largest passing round_max.
  * Kill criteria: no >= 2-point plateau anywhere, or zero recall at every
    certified point, or the benign iso distribution shifting >= one
    quantization level across cells at matched round buckets (Phase 5).
    A kill is recorded in docs/DECISION.md; the grids are NOT widened.
  * The always-on arm (window = 3..end) is replayed for documentation: its
    expected steady-state FP failures are the recorded justification for the
    cold window's existence.

Exit codes (so notebook / automation cells cannot misread a kill as a pass):
  0 = a plateau exists and a selection was printed
  1 = no *_results.json files found
  2 = no usable runs, or no usable CLEAN baseline (zero-FP premise
      uncertifiable — V7 must not run)
  3 = negative result (grids contain no passing plateau with recall)

Usage:
    python replay_v7_calibration.py [--dir results] [FILE.json ...]
                                    [--out results/v7_replay_summary.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

TAU_LO_GRID = [round(1.30 + 0.05 * i, 2) for i in range(11)]   # 1.30..1.80
ISO_MIN_GRID = [5.0 / 12.0, 7.0 / 12.0]
ROUND_MAX_GRID = [5, 8, 10]
ROUND_MIN = 3
V4_TAU = 1.85            # the pre-registered V4 threshold
V4_EPS = 1e-6            # v4_cse_reject_weights' median clamp
TAU_HI_PLATEAU_LO = 1.785  # bottom of the archived zero-FP plateau: flags
#                            with r below this are CSE-impossible detections
RECALL_COVERAGE = 0.80   # round_max selection coverage rule
DEFAULT_GMD = 4          # canonical graph_min_distinct when not archived
DEFAULT_KEEP_MIN = 1     # trust_scorer keep_min default
LEVEL_EPS = 1e-9         # one-quantization-level comparison guard (Phase 5)


def _iso_name(v: float) -> str:
    return {5.0 / 12.0: "5/12", 7.0 / 12.0: "7/12"}.get(v, f"{v:.4f}")


def _median_lower(vals):
    """torch.median semantics: the LOWER of the two middle values at even n
    (statistics.median averages them — deliberately NOT used)."""
    s = sorted(vals)
    return s[(len(s) - 1) // 2]


def _finite_list(x, n):
    return (isinstance(x, list) and len(x) == n
            and all(isinstance(v, (int, float)) and math.isfinite(v)
                    for v in x))


def _recon_v4(cse, k_cap, keep_min):
    """Reconstruct (ratio, flagged) exactly as v4_cse_reject_weights would.
    Rank ties broken index-ascending (real CSE floats never tie exactly;
    the convention is stated for determinism)."""
    med = max(_median_lower(cse), V4_EPS)
    ratio = [c / med for c in cse]
    n = len(ratio)
    max_flags = min(max(0, int(k_cap)), max(0, n - max(1, int(keep_min))))
    flagged = [False] * n
    for j in sorted(range(n), key=lambda i: (-ratio[i], i))[:max_flags]:
        if ratio[j] > V4_TAU:
            flagged[j] = True
    return ratio, flagged


def load_run(path: Path):
    """Extract (and where needed reconstruct) the per-round channels replay
    needs; None if the file is unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [skip] {path.name}: unreadable ({e})")
        return None
    cfg = data.get("config") or {}
    dcfg = cfg.get("defense_config") or {}
    k_cap = int(cfg.get("num_byzantine", dcfg.get("num_byzantine", 2)))
    keep_min = int(dcfg.get("keep_min", DEFAULT_KEEP_MIN))
    gmd = int(dcfg.get("graph_min_distinct", DEFAULT_GMD))
    rounds = []
    tally = {"asrun": 0, "recon": 0, "no_cse": 0, "no_residual": 0,
             "gated_derived": 0}
    for r in data.get("results") or []:
        agg = r.get("aggregation") or {}
        clients = [int(c) for c in (agg.get("accepted_clients") or [])]
        n = len(clients)
        row = {"round": int(r.get("round", 0)),  # 1-indexed in the logs
               "clients": clients, "usable": False, "gated": None,
               "ratio": None, "residual": None, "flagged": None}
        if n == 0:
            rounds.append(row)
            continue
        residual = agg.get("residual")
        if not _finite_list(residual, n):
            tally["no_residual"] += 1
            rounds.append(row)
            continue
        row["residual"] = residual
        # Gating: as-run bool when archived; else reconstructed from the
        # persisted residual levels with the archived graph_min_distinct.
        gated = agg.get("graph_channel_gated")
        if isinstance(gated, bool):
            row["gated"] = gated
        else:
            row["gated"] = len({round(v, 6) for v in residual}) < gmd
            tally["gated_derived"] += 1
        ratio, flagged = agg.get("v4_ratio"), agg.get("v4_flagged")
        if _finite_list(ratio, n) and isinstance(flagged, list) \
                and len(flagged) == n:
            row["ratio"] = ratio
            row["flagged"] = [bool(f) for f in flagged]
            tally["asrun"] += 1
        else:
            # Pre-V4 archive: reconstruct from the per-round local_cse dict.
            cse_map = r.get("local_cse") or {}
            cse = []
            for cid in clients:
                v = cse_map.get(str(cid), cse_map.get(cid))
                if not (isinstance(v, (int, float)) and math.isfinite(v)):
                    cse = None
                    break
                cse.append(float(v))
            if cse is None:
                tally["no_cse"] += 1
                rounds.append(row)
                continue
            row["ratio"], row["flagged"] = _recon_v4(cse, k_cap, keep_min)
            tally["recon"] += 1
        row["usable"] = True
        rounds.append(row)
    attackers = set(int(a) for a in (data.get("attacker_ids") or []))
    return {
        "name": path.name,
        "model": str(cfg.get("model_name", "?")),
        "dataset": str(cfg.get("dataset", "?")),
        "seed": cfg.get("seed"),
        "k_cap": k_cap,
        "keep_min": keep_min,
        "gmd": gmd,
        "trust_mode": str(dcfg.get("trust_mode", "?")),
        "attackers": attackers,
        "rounds": rounds,
        "tally": tally,
    }


def replay_flags(run, tau_lo, iso_min, round_min, round_max):
    """Return (benign_flags, attacker_flags, benign_joint_capfree) lists of
    (round, client, ratio, iso) under the V7 Tier-2 rule. round_max=None
    means always-on (the documentation arm). Gated rounds are skipped for
    BOTH statistics — the live rule abstains there (iso=None)."""
    benign, attacker, benign_capfree = [], [], []
    for row in run["rounds"]:
        rnd = row["round"]
        if rnd < round_min:
            continue
        if round_max is not None and rnd > round_max:
            continue
        if not row["usable"] or row["gated"]:
            continue
        ratio, iso, flagged = row["ratio"], row["residual"], row["flagged"]
        clients = row["clients"]
        n = len(clients)
        # cap-free joint exceedance (the plateau statistic — critique: the
        # budget shield is NOT V4's structural one in this regime, so the
        # zero-FP claim must hold before the cap is even applied)
        for i in range(n):
            if (not flagged[i] and ratio[i] > tau_lo and iso[i] >= iso_min
                    and clients[i] not in run["attackers"]):
                benign_capfree.append((rnd, clients[i], ratio[i], iso[i]))
        # budgeted rule (what the live run would do), including the keep_min
        # clamp trust_scorer applies to max_flags
        max_flags = min(max(0, run["k_cap"]),
                        max(0, n - max(1, run["keep_min"])))
        budget = max_flags - sum(1 for f in flagged if f)
        if budget <= 0:
            continue
        cand = [i for i in range(n)
                if not flagged[i] and ratio[i] > tau_lo and iso[i] >= iso_min]
        cand.sort(key=lambda i: (-iso[i], -ratio[i], i))
        for i in cand[:budget]:
            rec = (rnd, clients[i], ratio[i], iso[i])
            if clients[i] in run["attackers"]:
                attacker.append(rec)
            else:
                benign.append(rec)
    return benign, attacker, benign_capfree


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("files", nargs="*", help="archived *_results.json files")
    ap.add_argument("--dir", default="results",
                    help="directory to glob *_results.json from (default: results)")
    ap.add_argument("--out", default=None,
                    help="optional JSON summary path (e.g. results/v7_replay_summary.json)")
    args = ap.parse_args(argv)

    paths = [Path(f) for f in args.files]
    if not paths:
        paths = sorted(Path(args.dir).glob("*_results.json"))
    if not paths:
        print(f"No *_results.json found (dir={args.dir!r}). Point this at the "
              "archived run logs (Drive/Colab archive).")
        return 1

    runs = [r for r in (load_run(p) for p in paths) if r is not None]

    # ---- Phase 0: schema audit (blocking — denominators come from here) ---
    print("=" * 72)
    print("Phase 0 — schema audit (usable = as-run OR reconstructed channels)")
    print("=" * 72)
    usable = []
    win_hi = max(ROUND_MAX_GRID)
    for r in runs:
        n_rounds = len(r["rounds"])
        n_usable = sum(1 for row in r["rounds"] if row["usable"])
        gated_win = sum(1 for row in r["rounds"]
                        if row["usable"] and row["gated"]
                        and ROUND_MIN <= row["round"] <= win_hi)
        t = r["tally"]
        tag = "CLEAN" if not r["attackers"] else f"ATK{sorted(r['attackers'])}"
        print(f"  {r['name']}: {r['model']} / {r['dataset']} seed={r['seed']} "
              f"mode={r['trust_mode']} {tag} rounds={n_rounds} "
              f"[as-run={t['asrun']} recon={t['recon']} "
              f"no_cse={t['no_cse']} no_residual={t['no_residual']}] "
              f"gated-in-window={gated_win}"
              + (f" (gating derived from residual levels, "
                 f"graph_min_distinct={r['gmd']})" if t["gated_derived"]
                 else ""))
        if n_usable:
            usable.append(r)
    clean_runs = [r for r in usable if not r["attackers"]]
    atk_runs = [r for r in usable if r["attackers"]]
    print(f"usable: {len(usable)} ({len(clean_runs)} clean, {len(atk_runs)} attack)")
    no_clean = not clean_runs
    if no_clean:
        print("⚠ NO usable clean baselines — the zero-FP premise cannot be "
              "certified. V7 must not run until clean baselines with "
              "per-round 'residual' plus either as-run 'v4_ratio' or a "
              "complete 'local_cse' dict exist (rerunning them is offline "
              "calibration, not tuning). Exit code will be 2.")
    if not usable:
        print("No usable runs.")
        return 2

    # Admitted-decision denominators (in-window, usable, NOT gated).
    print()
    for rmax in ROUND_MAX_GRID:
        adm_b = adm_a = 0
        for r in usable:
            for row in r["rounds"]:
                if (row["usable"] and not row["gated"]
                        and ROUND_MIN <= row["round"] <= rmax):
                    a = sum(1 for c in row["clients"] if c in r["attackers"])
                    adm_a += a
                    adm_b += len(row["clients"]) - a
        print(f"  admitted decisions (round_max={rmax}): "
              f"benign={adm_b} attacker={adm_a}")

    # ---- Phase 1: anchor extremes -----------------------------------------
    print()
    print("=" * 72)
    print("Phase 1 — benign ratio extremes and their same-round isolation")
    print("=" * 72)
    for r in usable:
        best = None
        for row in r["rounds"]:
            if not row["usable"]:
                continue
            for i, c in enumerate(row["clients"]):
                if c in r["attackers"]:
                    continue
                if best is None or row["ratio"][i] > best[2]:
                    best = (row["round"], c, row["ratio"][i],
                            row["residual"][i], row["gated"])
        if best:
            rnd, c, rat, iso_v, gated = best
            danger = (ROUND_MIN <= rnd <= win_hi and not gated
                      and iso_v >= min(ISO_MIN_GRID))
            print(f"  {r['name']}: max benign r={rat:.4f} (c{c}, R{rnd}, "
                  f"iso={iso_v:.4f}" + (", gated" if gated else "") + ")"
                  + ("  << IN-WINDOW + ISOLATED — plateau hazard"
                     if danger else ""))

    # ---- Phase 2: zero-FP frontier over the pre-committed grid ------------
    print()
    print("=" * 72)
    print("Phase 2 — zero-FP frontier (cap-free AND budgeted must both be 0; "
          "gated rounds excluded)")
    print("=" * 72)
    frontier = {}

    def plateau_at(iso_min, rmax):
        """Maximal contiguous zero-FP tau_lo region at (iso_min, rmax)."""
        plateau, cur = [], []
        for t in TAU_LO_GRID:
            fb, fc, _ = frontier[(t, iso_min, rmax)]
            if fb == 0 and fc == 0:
                cur.append(t)
                if len(cur) > len(plateau):
                    plateau = list(cur)
            else:
                cur = []
        return plateau

    for iso_min in ISO_MIN_GRID:
        for round_max in ROUND_MAX_GRID:
            for tau_lo in TAU_LO_GRID:
                fp_budget = fp_capfree = rec = 0
                for r in usable:
                    b, a, bc = replay_flags(r, tau_lo, iso_min,
                                            ROUND_MIN, round_max)
                    fp_budget += len(b)
                    fp_capfree += len(bc)
                    rec += len(a)
                frontier[(tau_lo, iso_min, round_max)] = (
                    fp_budget, fp_capfree, rec)
            plateau = plateau_at(iso_min, round_max)
            desc = (f"plateau tau_lo=[{plateau[0]}, {plateau[-1]}]"
                    if len(plateau) >= 2 else
                    ("single point — NOT a plateau" if plateau else "EMPTY"))
            print(f"  iso_min={_iso_name(iso_min)} round_max={round_max}: {desc}")
            for t in TAU_LO_GRID:
                fb, fc, rc = frontier[(t, iso_min, round_max)]
                print(f"    tau_lo={t:.2f}: benign budgeted={fb} "
                      f"capfree={fc} attacker_recovered={rc}")

    # ---- Phase 3: incremental recall at every passing point ---------------
    print()
    print("=" * 72)
    print("Phase 3 — attacker recall decomposition (the load-bearing statistic)")
    print("=" * 72)
    print("  'CSE-impossible' = ratio < 1.785 (below the archived V4 plateau "
          "bottom: no legal CSE threshold could have flagged it)")
    for (tau_lo, iso_min, round_max), (fb, fc, rc) in sorted(frontier.items()):
        if fb or fc or rc == 0:
            continue
        impossible = borderline = 0
        by_bucket = {}
        by_level = {}
        for r in atk_runs:
            _, a, _ = replay_flags(r, tau_lo, iso_min, ROUND_MIN, round_max)
            for rnd, c, rat, iso_v in a:
                if rat < TAU_HI_PLATEAU_LO:
                    impossible += 1
                elif rat <= V4_TAU:
                    borderline += 1
                b = ("R3-5" if rnd <= 5 else "R6-8" if rnd <= 8 else "R9-10")
                by_bucket[b] = by_bucket.get(b, 0) + 1
                lv = round(iso_v, 6)
                by_level[lv] = by_level.get(lv, 0) + 1
        print(f"  tau_lo={tau_lo:.2f} iso_min={_iso_name(iso_min)} "
              f"round_max={round_max}: recovered={rc} "
              f"(CSE-impossible={impossible}, borderline={borderline}) "
              f"buckets={by_bucket} iso_levels={by_level}")

    # ---- Phase 4: always-on documentation arm -----------------------------
    print()
    print("=" * 72)
    print("Phase 4 — always-on arm (window 3..end): documents WHY the cold "
          "window exists")
    print("=" * 72)
    for iso_min in ISO_MIN_GRID:
        for tau_lo in (TAU_LO_GRID[0], TAU_LO_GRID[-1]):
            fp = sum(len(replay_flags(r, tau_lo, iso_min, ROUND_MIN,
                                      None)[0]) for r in usable)
            print(f"  iso_min={_iso_name(iso_min)} tau_lo={tau_lo:.2f}: "
                  f"benign flags (budgeted, all rounds) = {fp}")

    # ---- Phase 5: channel resolution + cross-cell iso stability -----------
    print()
    print("=" * 72)
    print("Phase 5 — in-window channel resolution and benign iso stability "
          "(the DECISION kill criterion: benign iso shifting >= one "
          "quantization level across cells at matched round buckets)")
    print("=" * 72)
    stability = {}
    for r in usable:
        counts = []
        medians = {}
        for row in r["rounds"]:
            if not (row["usable"] and ROUND_MIN <= row["round"] <= win_hi):
                continue
            counts.append(len({round(v, 6) for v in row["residual"]}))
            if row["gated"]:
                continue
            b = ("R3-5" if row["round"] <= 5
                 else "R6-8" if row["round"] <= 8 else "R9-10")
            for i, c in enumerate(row["clients"]):
                if c not in r["attackers"]:
                    medians.setdefault(b, []).append(row["residual"][i])
        med_by_bucket = {b: round(_median_lower(v), 6)
                         for b, v in medians.items() if v}
        stability[r["name"]] = med_by_bucket
        if counts:
            print(f"  {r['name']}: in-window distinct levels "
                  f"min={min(counts)} median={sorted(counts)[len(counts)//2]} "
                  f"(graph_min_distinct={r['gmd']} gates rounds below) "
                  f"benign iso median/bucket={med_by_bucket}")
    for b in ("R3-5", "R6-8", "R9-10"):
        vals = [m[b] for m in stability.values() if b in m]
        if len(vals) >= 2:
            spread = max(vals) - min(vals)
            n_clients = max((len(row["clients"]) for r in usable
                             for row in r["rounds"] if row["usable"]),
                            default=7)
            level = 1.0 / max(1, n_clients - 1)
            if spread >= level - LEVEL_EPS:
                print(f"  ⚠ iso stability kill criterion at {b}: benign "
                      f"median spread {spread:.4f} >= one level ({level:.4f})")

    # ---- Selection (pre-committed deterministic rules) --------------------
    print()
    print("=" * 72)
    print("Selection under the pre-committed rules (verify by eye before "
          "writing the DECISION entry):")
    selection = None
    for iso_min in sorted(ISO_MIN_GRID, reverse=True):
        passing = [m for m in ROUND_MAX_GRID
                   if len(plateau_at(iso_min, m)) >= 2]
        if not passing:
            continue
        m_cert = max(passing)   # windows nest: certifies every smaller m too
        plat = plateau_at(iso_min, m_cert)
        tau_sel = plat[(len(plat) - 1) // 2]   # midpoint, rounded DOWN
        recall_ref = frontier[(tau_sel, iso_min, m_cert)][2]
        if recall_ref == 0:
            print(f"  iso_min={_iso_name(iso_min)}: plateau exists "
                  f"(certified at round_max={m_cert}) but recall=0 at "
                  f"tau_lo={tau_sel:.2f} — kill criterion, trying next "
                  "iso_min")
            continue
        m_sel = next((m for m in sorted(ROUND_MAX_GRID)
                      if frontier[(tau_sel, iso_min, m)][2]
                      >= RECALL_COVERAGE * recall_ref), m_cert)
        selection = {"tau_lo": tau_sel, "iso_min": iso_min,
                     "round_max": m_sel, "certified_round_max": m_cert,
                     "plateau": plat, "recall_at_selected":
                         frontier[(tau_sel, iso_min, m_sel)][2],
                     "recall_at_certified": recall_ref}
        print(f"  SELECTED: tau_lo={tau_sel:.2f} "
              f"iso_min={_iso_name(iso_min)} round_max={m_sel} "
              f"(plateau [{plat[0]}, {plat[-1]}] certified at "
              f"round_max={m_cert}; recall {selection['recall_at_selected']}"
              f"/{recall_ref})")
        break
    if selection is None:
        print("  NO passing (plateau AND recall) point in the grids -> V7 is "
              "a NEGATIVE result: record it in docs/DECISION.md and do NOT "
              "widen the grids.")

    if args.out:
        out = {
            "grids": {"tau_lo": TAU_LO_GRID,
                      "iso_min": ISO_MIN_GRID,
                      "round_max": ROUND_MAX_GRID,
                      "round_min": ROUND_MIN},
            "frontier": {
                f"tau_lo={t}|iso={_iso_name(i)}|rmax={m}": {
                    "benign_budgeted": fb, "benign_capfree": fc,
                    "attacker_recovered": rc}
                for (t, i, m), (fb, fc, rc) in frontier.items()},
            "selection": selection,
            "benign_iso_median_by_bucket": stability,
            "runs": [{"name": r["name"], "model": r["model"],
                      "dataset": r["dataset"], "seed": r["seed"],
                      "clean": not r["attackers"],
                      "tally": r["tally"]} for r in usable],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nSummary written to {args.out}")

    if no_clean:
        return 2
    return 0 if selection is not None else 3


if __name__ == "__main__":
    sys.exit(main())
