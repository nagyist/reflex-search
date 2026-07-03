#!/usr/bin/env python3
"""REF-225 Phase 1 validity gate — standalone, robust, re-runnable.

Reads arm-A trial transcripts directly from the results tree and computes, per
task, the median ``num_turns`` (the pre-registered ``assistant_turns`` metric —
NOT a raw count of ``type:assistant`` lines, which inflates 2-4×; see
extract_metrics.py:244). Applies the validity gate:

    median arm-A num_turns >= 4  ->  task PASSES

The gate passes overall when >= 12 of the 16 tasks pass. This script is
deliberately decoupled from ``run-ref225-arm-a.sh``'s inline hook, which chains
extract_metrics (hard-fails on ANY incomplete transcript) + the gate check: if
even one of the 128 trials fails to produce a terminal ``result`` event (agent
error, timeout, rate-limit truncation), that inline chain aborts and the gate is
never computed. This script instead **skips incomplete trials with a warning**
and computes the gate from whatever completed — so a single bad trial can't sink
the whole Phase 1 decision.

Usage:
    python3 benches/efficacy/ref225-validity-gate.py \
        [--results-dir benches/efficacy/results/] [--min-trials 5]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

IF_TASK_IDS = [
    "reflex-comp-query-path", "reflex-comp-trigram-extract", "reflex-comp-deleted-file",
    "reflex-comp-mcp-dispatch", "ripgrep-comp-binary", "ripgrep-comp-type-flag",
    "ripgrep-comp-parallel", "tokio-comp-work-steal", "tokio-comp-task-abort",
    "tokio-comp-io-driver", "reflex-trans-content-store", "reflex-trans-regex-trigrams",
    "reflex-cm-language-dispatch", "ripgrep-cm-printer-chain", "ripgrep-cm-stats-tracking",
    "tokio-cm-spawn-chain",
]

GATE_THRESHOLD = 4          # median num_turns must be >= this
MIN_TASKS_PASS = 12         # >= this many tasks must pass to proceed to Phase 2


def num_turns(transcript_path: Path):
    """Return the result event's num_turns, or None if the trial is incomplete."""
    result_ev = None
    try:
        for raw in transcript_path.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "result":
                result_ev = ev
    except Exception:
        return None
    if result_ev is None:
        return None
    nt = result_ev.get("num_turns")
    return int(nt) if isinstance(nt, (int, float)) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="benches/efficacy/results/")
    ap.add_argument("--min-trials", type=int, default=5,
                    help="minimum completed trials required to judge a task")
    args = ap.parse_args()

    arm_a = Path(args.results_dir) / "A"
    passed, failed, marginal, incomplete_tasks = [], [], [], []
    skipped_incomplete = 0

    print("REF-225 Phase 1 validity gate (metric: median num_turns, threshold >= 4)")
    print(f"  {'task':<30} {'n':>3} {'turns (completed trials)':<28} {'median':>7}  gate")
    print("  " + "-" * 78)

    for tid in IF_TASK_IDS:
        task_dir = arm_a / tid
        turns = []
        if task_dir.is_dir():
            for tp in sorted(task_dir.glob("trial_*.ndjson")):
                nt = num_turns(tp)
                if nt is None:
                    skipped_incomplete += 1
                else:
                    turns.append(nt)
        if not turns:
            print(f"  {tid:<30} {0:>3} {'(no completed trials)':<28} {'—':>7}  NO DATA")
            failed.append(tid)
            incomplete_tasks.append(tid)
            continue
        med = statistics.median(turns)
        gate = "PASS" if med >= GATE_THRESHOLD else "FAIL"
        flag = ""
        if med == GATE_THRESHOLD:
            flag = " (marginal)"
            marginal.append(tid)
        n = len(turns)
        note = "" if n >= args.min_trials else f"  <{args.min_trials} trials!"
        print(f"  {tid:<30} {n:>3} {str(turns):<28} {med:>7.1f}  {gate}{flag}{note}")
        (passed if med >= GATE_THRESHOLD else failed).append(tid)

    print()
    if skipped_incomplete:
        print(f"  NOTE: skipped {skipped_incomplete} incomplete trial(s) (no result event) "
              f"— robust to mid-run failures.")
    n_pass = len(passed)
    print(f"  VALIDITY GATE RESULT: {n_pass}/{len(IF_TASK_IDS)} tasks pass "
          f"(median num_turns >= {GATE_THRESHOLD})")
    if marginal:
        print(f"  MARGINAL tasks (median exactly {GATE_THRESHOLD}): {', '.join(marginal)}")
    if n_pass >= MIN_TASKS_PASS:
        print(f"  -> PHASE 1 PASSES ({n_pass} >= {MIN_TASKS_PASS}): proceed to Phase 2 "
              f"(launch run-ref225-phase2.sh)")
    else:
        print(f"  -> PHASE 1 FAILS ({n_pass} < {MIN_TASKS_PASS}): publish the null result.")
        print("     Null: 'Could not construct >= 12 iteration-forcing tasks that arm-A")
        print("     cannot answer in <= 3 turns; REF-222 parity is robust.'")


if __name__ == "__main__":
    main()
