#!/usr/bin/env python3
"""Generate a deterministic SAMPLE metrics + accuracy table for the Phase 3
analysis smoke test (REF-176).

The real Phase 4 experiment does not exist yet, but the acceptance criteria for
Phase 3 require the analysis to run end-to-end on sample data *before* Phase 4.
The Phase 0 spike only ran a single task across arms A/B/C (one replicate each),
which is too thin for the paired bootstrap / Wilcoxon machinery.

So this script grounds a synthetic-but-realistic table in the ACTUAL Phase 0
per-arm token magnitudes (read live from ``results/phase0/*.ndjson``) and
replicates them across the 5 tasks in ``tasks.json`` × N trials × cold/warm,
with seeded jitter. The numbers are illustrative, not experimental results —
their only job is to exercise every code path in ``analyze.py`` reproducibly.

Output (all deterministic given the fixed seed):
    results/sample/metrics.csv     (Phase 2 schema + `condition`, `category`)
    results/sample/accuracy.csv    (H2/H3 answer-scoring schema)
    results/sample/index_ledger.json
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PHASE0 = SCRIPT_DIR / "results" / "phase0"
OUT = SCRIPT_DIR / "results" / "sample"
SEED = 20260702
N_TRIALS = 5  # full-run replicate count (CEO-locked N=5)

# Five sample tasks using the AUTHORITATIVE corpus vocabulary (SCHEMA.md).
# Three are `find_all_usages` so the pre-registered primary population is
# non-trivial; two are other categories to exercise the secondary/exploratory
# path and confirm the primary correctly excludes non-find-all tasks.
TASKS = [
    ("T01_symbol_usages", "find_all_usages"),      # primary population
    ("T02_trigram_intersection", "locate_definition"),  # NOT primary
    ("T03_mmap_usages", "find_all_usages"),        # primary population
    ("T04_config_defaults", "locate_definition"),  # NOT primary
    ("T05_blake3_callers", "find_all_usages"),     # primary population
]

# Per-task difficulty multiplier on the arm baseline (keeps tasks distinct).
TASK_FACTOR = {
    "T01_symbol_usages": 1.00,
    "T02_trigram_intersection": 0.75,
    "T03_mmap_usages": 0.85,
    "T04_config_defaults": 0.55,
    "T05_blake3_callers": 0.90,
}


def phase0_arm_totals() -> dict:
    """Read the real Phase 0 result events -> per-arm token/turn baselines."""
    files = {"A": "arm-a-stream.ndjson", "B": "arm-b-stream.ndjson",
             "C": "arm-c-stream.ndjson"}
    totals = {}
    for arm, fname in files.items():
        path = PHASE0 / fname
        res = None
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("type") == "result":
                    res = e
        if res:
            u = res.get("usage", {})
            totals[arm] = {
                "input": u.get("input_tokens", 30) or 30,
                "output": u.get("output_tokens", 1500) or 1500,
                "cache_read": u.get("cache_read_input_tokens", 0) or 0,
                "cache_creation": u.get("cache_creation_input_tokens", 0) or 0,
                "turns": res.get("num_turns", 3) or 3,
                "wall_ms": res.get("duration_ms", 20000) or 20000,
                "cost": res.get("total_cost_usd", 0.03) or 0.03,
            }
        else:
            # Fallback baselines if phase0 transcripts are unavailable.
            base = {"A": 70000, "B": 141000, "C": 165000}[arm]
            totals[arm] = {"input": 30, "output": 1800,
                           "cache_read": base * 0.9, "cache_creation": base * 0.1,
                           "turns": {"A": 2, "B": 4, "C": 6}[arm],
                           "wall_ms": 25000, "cost": base * 4.3e-7}
    # B' mirrors B (no-nudge variant) with a small upward drift (less steering).
    b = totals["B"]
    totals["Bprime"] = {k: (v * 1.08 if isinstance(v, (int, float)) else v)
                        for k, v in b.items()}
    return totals


def main() -> None:
    rng = random.Random(SEED)
    totals = phase0_arm_totals()
    OUT.mkdir(parents=True, exist_ok=True)

    metric_cols = [
        "arm", "task_id", "trial", "condition", "category", "model", "repo_sha",
        "wall_ms", "input_tokens", "output_tokens", "cache_read_tokens",
        "cache_creation_tokens", "total_cost_usd", "assistant_turns",
        "total_tool_calls", "search_tool_calls_builtin", "search_tool_calls_mcp",
        "read_tool_calls", "success", "stop_reason",
    ]
    acc_cols = ["arm", "task_id", "trial", "n_expected", "n_returned",
                "n_correct", "quality_score"]

    metric_rows = []
    acc_rows = []

    for arm in ("A", "B", "C", "Bprime"):
        t = totals[arm]
        is_treatment = arm != "A"
        for task_id, category in TASKS:
            factor = TASK_FACTOR[task_id]
            n_expected = {"T01_symbol_usages": 122, "T02_trigram_intersection": 3,
                          "T03_mmap_usages": 18, "T04_config_defaults": 2,
                          "T05_blake3_callers": 9}[task_id]
            for condition in ("cold", "warm"):
                for trial in range(1, N_TRIALS + 1):
                    jit = 1.0 + rng.uniform(-0.06, 0.06)
                    # Cold: treatment arms pay an agent-side index_project call
                    # on the first query (control arm A has no index to build).
                    cold_bump = 1.0
                    if condition == "cold" and is_treatment:
                        cold_bump = 1.18
                    scale = factor * jit * cold_bump
                    inp = round(t["input"] * jit)
                    out = round(t["output"] * scale)
                    cr = round(t["cache_read"] * scale)
                    cc = round(t["cache_creation"] * scale)
                    turns = max(1, round(t["turns"] * (1.0 + rng.uniform(-0.2, 0.2))))
                    builtin = (round(2 * jit) if arm in ("A", "B", "Bprime") else 0)
                    mcp = (0 if arm == "A" else max(1, round(3 * jit)))
                    # Control occasionally misses on hard find-all tasks.
                    success = True
                    stop = "end_turn"
                    if arm == "A" and task_id in ("T01_symbol_usages",
                                                  "T05_blake3_callers") and trial == 3:
                        success = False
                        stop = "error_max_turns"
                    metric_rows.append({
                        "arm": arm, "task_id": task_id, "trial": trial,
                        "condition": condition, "category": category,
                        "model": "claude-sonnet-4-6", "repo_sha": "d2935f4",
                        "wall_ms": round(t["wall_ms"] * scale),
                        "input_tokens": inp, "output_tokens": out,
                        "cache_read_tokens": cr, "cache_creation_tokens": cc,
                        "total_cost_usd": round(t["cost"] * scale, 6),
                        "assistant_turns": turns,
                        "total_tool_calls": builtin + mcp + 2,
                        "search_tool_calls_builtin": builtin,
                        "search_tool_calls_mcp": mcp,
                        "read_tool_calls": 2,
                        "success": str(success).lower(), "stop_reason": stop,
                    })
                    # Accuracy: Reflex's pitch is complete coverage -> higher
                    # recall on find-all tasks; control under-recalls.
                    if is_treatment:
                        recall_frac = min(1.0, 0.97 + rng.uniform(-0.03, 0.03))
                        precision_extra = 0
                        quality = 4.0 if trial % 2 else 3.0
                    else:
                        recall_frac = 0.78 + rng.uniform(-0.08, 0.08)
                        precision_extra = round(rng.uniform(0, 2))  # a few false hits
                        quality = 3.0 if trial % 2 else 2.0
                    n_correct = round(n_expected * recall_frac)
                    n_returned = n_correct + precision_extra
                    # Only emit accuracy rows for warm (answer scoring is
                    # condition-independent); keyed by (arm, task, trial).
                    if condition == "warm":
                        acc_rows.append({
                            "arm": arm, "task_id": task_id, "trial": trial,
                            "n_expected": n_expected, "n_returned": n_returned,
                            "n_correct": n_correct, "quality_score": quality,
                        })

    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=metric_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(metric_rows)
    with open(OUT / "accuracy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=acc_cols)
        w.writeheader()
        w.writerows(acc_rows)

    ledger = {
        "_note": "Illustrative cold-index ledger (confound #2). One-time "
                 "`rfx index` cost per repo, amortized across warm queries.",
        "repos": {
            "reflex": {"cold_index_wall_ms": 1840, "warm_reindex_wall_ms": 120,
                       "indexed_files": 214, "index_bytes": 5_242_880}
        },
    }
    (OUT / "index_ledger.json").write_text(json.dumps(ledger, indent=2))

    print(f"Grounded on Phase 0 arm totals: "
          f"{ {a: round(totals[a]['cache_read'] + totals[a]['cache_creation']) for a in ('A','B','C')} }")
    print(f"Wrote {len(metric_rows)} metric rows -> {OUT / 'metrics.csv'}")
    print(f"Wrote {len(acc_rows)} accuracy rows -> {OUT / 'accuracy.csv'}")
    print(f"Wrote {OUT / 'index_ledger.json'}")


if __name__ == "__main__":
    main()
