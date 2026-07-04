#!/usr/bin/env python3
"""REF-225 Phase 2 authoritative analysis — FROZEN valid-task set.

The Phase 1 validity gate (computed on arm-A data ONLY, before any arm-B result
was unblinded) passed 13 of 16 tasks. Three tasks are EXCLUDED because arm-A
solved them in <= 3 median turns, i.e. they are not iteration-forcing and would
only reproduce REF-222 parity:

    EXCLUDED (validity gate FAIL):
      ripgrep-comp-parallel   (arm-A median num_turns 3.5)
      tokio-comp-task-abort   (arm-A median num_turns 2.0)
      tokio-comp-io-driver    (arm-A median num_turns 2.0)

Per the pre-registration, the Phase 2 primary analysis runs on the 13 VALID
tasks below. This file is committed BEFORE arm-B data is analysed so the
valid-task set is frozen pre-unblinding (REF-176 anti-p-fishing discipline);
the git timestamp is the audit trail.

Primary endpoints (Holm-Bonferroni over K=3): assistant_turns, total_tool_calls,
total_tokens. Paired Wilcoxon signed-rank per (task, trial) + bootstrap ratio CI.
Recall guardrail: arm-B median file-coverage recall must not drop >5pp below A.

Usage:
    python3 benches/efficacy/ref225-phase2-analysis.py \
        [--metrics benches/efficacy/results/metrics_ref225.csv] \
        [--recall  benches/efficacy/results/recall_ref225.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path("benches/efficacy")))
from stats import wilcoxon_signed_rank, paired_bootstrap_ratio_ci, median  # noqa: E402

# The 13 tasks that PASSED the arm-A validity gate (median num_turns >= 4).
VALID_TASK_IDS = [
    "reflex-comp-query-path",        # 12.0
    "reflex-comp-trigram-extract",   # 4.0 (marginal)
    "reflex-comp-deleted-file",      # 8.5
    "reflex-comp-mcp-dispatch",      # 23.5
    "ripgrep-comp-binary",           # 11.0
    "ripgrep-comp-type-flag",        # 17.0
    "tokio-comp-work-steal",         # 6.5
    "reflex-trans-content-store",    # 6.5
    "reflex-trans-regex-trigrams",   # 4.0 (marginal)
    "reflex-cm-language-dispatch",   # 5.5
    "ripgrep-cm-printer-chain",      # 7.5
    "ripgrep-cm-stats-tracking",     # 8.0
    "tokio-cm-spawn-chain",          # 21.5
]
EXCLUDED_TASK_IDS = [
    "ripgrep-comp-parallel",         # 3.5  FAIL
    "tokio-comp-task-abort",         # 2.0  FAIL
    "tokio-comp-io-driver",          # 2.0  FAIL
]

ENDPOINTS = ["assistant_turns", "total_tool_calls", "total_tokens"]
ALPHA = 0.05


def get_trial_pairs(rows, endpoint):
    """Paired (a, b) values per (task_id, trial) over the VALID task set."""
    a_map, b_map = {}, {}
    for r in rows:
        if r.get("task_id") not in VALID_TASK_IDS:
            continue
        v = (r.get(endpoint) or "").strip()
        if not v:
            continue
        try:
            fv = float(v)
        except ValueError:
            continue
        key = (r["task_id"], r.get("trial", ""))
        if r.get("arm") == "A":
            a_map[key] = fv
        elif r.get("arm") == "B":
            b_map[key] = fv
    return [(a_map[k], b_map[k]) for k in a_map if k in b_map]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="benches/efficacy/results/metrics_ref225.csv")
    ap.add_argument("--recall", default="benches/efficacy/results/recall_ref225.csv")
    args = ap.parse_args()

    print("REF-225 Phase 2 authoritative analysis (13 valid tasks; 3 excluded by validity gate)")
    print(f"  Excluded (arm-A median num_turns < 4): {', '.join(EXCLUDED_TASK_IDS)}")
    print()

    mpath = Path(args.metrics)
    if not mpath.exists():
        print(f"  metrics CSV not found: {mpath} — run Phase 2 first.")
        sys.exit(1)
    rows = list(csv.DictReader(open(mpath)))

    print("PRIMARY ENDPOINTS — paired Wilcoxon signed-rank + 95% bootstrap ratio CI")
    print(f"  K=3 tests, Holm-Bonferroni FWER at alpha={ALPHA}")
    print(f"  {'Endpoint':<22} {'A med':>8} {'B med':>8} {'ratio':>7} {'n':>4} {'p':>7}  {'95% CI':>16} adj")
    print("  " + "-" * 84)

    results = []
    for ep in ENDPOINTS:
        pairs = get_trial_pairs(rows, ep)
        if not pairs:
            print(f"  {ep:<22}  NO DATA")
            continue
        a_vals = [p[0] for p in pairs]
        b_vals = [p[1] for p in pairs]
        wres = wilcoxon_signed_rank(a_vals, b_vals)
        ci = paired_bootstrap_ratio_ci(pairs, n_boot=2000, level=0.95, seed=42)
        med_a, med_b = median(a_vals), median(b_vals)
        ratio = med_b / med_a if med_a > 0 else float("inf")
        results.append(dict(ep=ep, med_a=med_a, med_b=med_b, ratio=ratio,
                            p=wres.p_value, ci=ci, n=wres.n))

    # Holm-Bonferroni
    for rank, res in enumerate(sorted(results, key=lambda r: r["p"])):
        res["holm_sig"] = res["p"] < ALPHA / (len(results) - rank)

    order = {ep: i for i, ep in enumerate(ENDPOINTS)}
    for res in sorted(results, key=lambda r: order[r["ep"]]):
        ci = res["ci"]
        sig = "*" if res["holm_sig"] else "ns"
        print(f"  {res['ep']:<22} {res['med_a']:>8.1f} {res['med_b']:>8.1f} "
              f"{res['ratio']:>7.3f} {res['n']:>4} {res['p']:>7.4f}  "
              f"[{ci.low:.3f},{ci.high:.3f}]  {sig}")

    print()
    if results:
        dirs = ["B<A" if r["ratio"] < 1 else ("B>A" if r["ratio"] > 1 else "tie") for r in results]
        consistent = len(set(dirs)) == 1
        all_sig = all(r["holm_sig"] for r in results)
        all_ns = all(not r["holm_sig"] for r in results)
        print(f"  Co-primary direction consistency: {'YES ('+dirs[0]+')' if consistent else 'NO — mixed'}")
        if all_sig and consistent and dirs[0] == "B<A":
            print("  ENDPOINT VERDICT: B WINS — Reflex reduces iterations (all 3 endpoints sig)")
        elif all_sig and consistent and dirs[0] == "B>A":
            print("  ENDPOINT VERDICT: B LOSES — Reflex increases iterations (all 3 sig)")
        elif all_ns:
            print("  ENDPOINT VERDICT: PARITY — no significant difference (Holm-adjusted)")
        elif not consistent:
            print("  ENDPOINT VERDICT: MIXED — endpoints disagree; report individually")
        else:
            print("  ENDPOINT VERDICT: INDETERMINATE — some sig, some not; see per-endpoint p")

    # Recall guardrail
    rpath = Path(args.recall)
    if rpath.exists():
        rrows = [r for r in csv.DictReader(open(rpath)) if r.get("task_id") in VALID_TASK_IDS]
        a = [float(r["recall"]) for r in rrows if r["arm"] == "A" and r.get("recall")]
        b = [float(r["recall"]) for r in rrows if r["arm"] == "B" and r.get("recall")]
        if a and b:
            ma, mb = median(a), median(b)
            delta = mb - ma
            status = "PASS" if delta >= -0.05 else "FAIL — arm B sacrifices coverage"
            print()
            print(f"  RECALL GUARDRAIL: A median={ma:.3f}, B median={mb:.3f}, "
                  f"delta={delta:+.3f} -> {status}")
    else:
        print()
        print(f"  RECALL GUARDRAIL: recall CSV not found ({rpath}) — run score_recall_ref225.py")


if __name__ == "__main__":
    main()
