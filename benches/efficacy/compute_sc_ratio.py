#!/usr/bin/env python3
"""REF-204 (REF-196 Phase 4): compute the B_sc / B token ratio for structuredContent.

The efficacy harness ships a general A/B analysis (`analyze.py`) whose headline is
the pre-registered B/A endpoint. This script is a focused companion for the
structuredContent question only: **does the additive `structuredContent` field
(REF-202) change token efficiency relative to plain Reflex MCP?**

The comparison is the paired arms produced by `runner.py`:
  - `B_sc`   : Reflex MCP with structuredContent ON  (REF-202 shipped default)
  - `B_nosc` : Reflex MCP with structuredContent OFF (REFLEX_MCP_STRUCTURED_CONTENT=0)
Both are byte-for-byte identical arm-B configs except that one field, so any
token delta is attributable to `structuredContent` alone. `B_nosc` IS "plain
Reflex MCP" — i.e. the REF-204 "B" in the "B_sc / B" ratio.

Token metric mirrors the pre-registered total in `analyze.py`:
    total_tokens = input + output + cache_read + cache_creation
(cache is included on purpose — the MCP payload tax must be visible).

Two ratios are reported, both deterministic (seeded bootstrap, no clock/global RNG):
  1. Per-task ratio-of-medians with a two-sample bootstrap 95% CI over trials.
  2. Across-task headline: median of per-task ratios with a paired bootstrap 95% CI
     over tasks (the study's exchangeable unit), plus the pre-registered
     direction verdict.

Usage:
    python3 benches/efficacy/compute_sc_ratio.py --metrics results/metrics.csv
    python3 benches/efficacy/compute_sc_ratio.py --metrics results/metrics.csv --md out.md
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
import stats  # noqa: E402  (harness stdlib-only bootstrap/CI helpers)

N_BOOT = 10000
LEVEL = 0.95
SEED = 20260703  # fixed => reproducible CIs (REF-204)

TREATMENT = "B_sc"
BASELINE = "B_nosc"

TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens")


def _num(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def total_tokens(row: dict) -> float:
    return sum(_num(row.get(f, 0)) for f in TOKEN_FIELDS)


def load(metrics_csv: Path, treatment: str, baseline: str) -> dict[str, dict[str, list[float]]]:
    """Return {arm: {task_id: [total_tokens per trial]}} for the two arms."""
    by_arm: dict[str, dict[str, list[float]]] = {treatment: defaultdict(list), baseline: defaultdict(list)}
    with open(metrics_csv, newline="") as f:
        for row in csv.DictReader(f):
            arm = row.get("arm")
            if arm in by_arm and row.get("task_id") and row.get("task_id") != "_baseline":
                by_arm[arm][row["task_id"]].append(total_tokens(row))
    return by_arm


def two_sample_ratio_ci(nosc: list[float], sc: list[float]) -> stats.CI:
    """Bootstrap 95% CI for median(sc)/median(nosc), resampling each arm's trials."""
    point = stats.median(sc) / stats.median(nosc)
    if len(nosc) < 2 or len(sc) < 2:
        return stats.CI(point=point, low=point, high=point, level=LEVEL, n=min(len(nosc), len(sc)), n_boot=0)
    import random
    rng = random.Random(SEED)
    boot = []
    na, nb = len(nosc), len(sc)
    for _ in range(N_BOOT):
        a = stats.median([nosc[rng.randrange(na)] for _ in range(na)])
        b = stats.median([sc[rng.randrange(nb)] for _ in range(nb)])
        if a > 0:
            boot.append(b / a)
    boot.sort()
    lo = stats.quantile(boot, (1 - LEVEL) / 2)
    hi = stats.quantile(boot, 1 - (1 - LEVEL) / 2)
    return stats.CI(point=point, low=lo, high=hi, level=LEVEL, n=min(na, nb), n_boot=N_BOOT)


def verdict(r: float, lo: float, hi: float) -> str:
    """Pre-registered direction rule (REF-176), applied to B_sc vs baseline.

    Here "better" means B_sc spends FEWER tokens than plain Reflex MCP.
    """
    if r < 0.90 and hi < 1.0:
        return "better (structuredContent reduces tokens)"
    if r > 1.10 and lo > 1.0:
        return "worse (structuredContent increases tokens)"
    if lo <= 1.0 <= hi:
        return "no difference (95% CI straddles 1.0)"
    return "indeterminate (excludes parity but misses the ±10% thresholds)"


def main() -> None:
    ap = argparse.ArgumentParser(description="REF-204 B_sc/B structuredContent token ratio")
    ap.add_argument("--metrics", type=Path, default=SCRIPT_DIR / "results" / "metrics.csv")
    ap.add_argument("--treatment", default=TREATMENT)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--md", type=Path, default=None, help="Also write the markdown report to this path")
    args = ap.parse_args()

    by_arm = load(args.metrics, args.treatment, args.baseline)
    t_tasks, b_tasks = by_arm[args.treatment], by_arm[args.baseline]
    tasks = sorted(set(t_tasks) & set(b_tasks))
    if not tasks:
        sys.exit(f"ERROR: no tasks present in BOTH {args.treatment} and {args.baseline} in {args.metrics}")

    lines: list[str] = []
    lines.append(f"Token metric: total = {' + '.join(TOKEN_FIELDS)}")
    lines.append(f"Treatment = {args.treatment} (structuredContent ON) | Baseline = {args.baseline} (OFF)")
    lines.append("")
    lines.append(f"| task | n({args.baseline}) | n({args.treatment}) | median {args.baseline} | median {args.treatment} | ratio B_sc/B | 95% CI |")
    lines.append("|------|------|------|------|------|------|------|")

    pairs: list[tuple[float, float]] = []  # (baseline_median, treatment_median) per task
    for task in tasks:
        nosc, sc = b_tasks[task], t_tasks[task]
        mb, mt = stats.median(nosc), stats.median(sc)
        ci = two_sample_ratio_ci(nosc, sc)
        pairs.append((mb, mt))
        lines.append(
            f"| {task} | {len(nosc)} | {len(sc)} | {mb:,.0f} | {mt:,.0f} | "
            f"{ci.point:.3f} | [{ci.low:.3f}, {ci.high:.3f}] |"
        )

    # Across-task headline: median of per-task ratios + paired bootstrap over tasks.
    agg = stats.paired_bootstrap_ratio_ci(pairs, n_boot=N_BOOT, level=LEVEL, seed=SEED)
    v = verdict(agg.point, agg.low, agg.high)
    lines.append("")
    lines.append(f"**Across-task headline (median of per-task B_sc/B ratios):**")
    lines.append(f"- ratio = **{agg.point:.3f}**  95% CI [{agg.low:.3f}, {agg.high:.3f}]  (n={agg.n} tasks, {agg.n_boot} bootstrap)")
    lines.append(f"- direction verdict: **{v}**")

    report = "\n".join(lines)
    print(report)
    if args.md:
        args.md.write_text(report + "\n")
        print(f"\n[wrote {args.md}]")


if __name__ == "__main__":
    main()
