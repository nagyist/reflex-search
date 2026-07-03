#!/usr/bin/env python3
"""REF-225 recall guardrail: deterministic file-coverage recall for rubric tasks.

The REF-225 iteration-forcing tasks are all ``ground_truth.type == "rubric"``
(comprehension / reverse_dependency / cross_module). ``score_accuracy.py`` only
grades ``match_set`` tasks (it extracts file:line pairs and compares to a ripgrep
oracle), so it skips every REF-225 task. Full rubric quality grading (0–4) needs
an LLM judge; but the **recall guardrail** the Phase 2 acceptance criteria call
for only needs a coverage measure: did the agent's answer reference the files the
task's ``expected_files`` says the answer must cover?

This scorer computes, per trial, deterministic **file-coverage recall**:

    recall = |expected_files mentioned in the final answer| / |expected_files|

A file counts as "mentioned" if its path (or a path suffix — agents often drop
the leading ``src/`` or ``crates/core/``) appears anywhere in the agent's final
answer text. This is intentionally lenient on precision (the guardrail only cares
that arm B does not *lose* coverage vs arm A), and is fully deterministic — no
network, no LLM, same transcript → same recall.

Output: a CSV with (arm, task_id, trial, n_expected, n_covered, recall) that
feeds the same guardrail comparison used for the efficiency endpoints.

Usage:
    python3 benches/efficacy/score_recall_ref225.py \
        --tasks benches/efficacy/tasks/iteration-forcing.yaml \
        --results-dir benches/efficacy/results/ \
        --out benches/efficacy/results/recall_ref225.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _extract_final_text(transcript_path: Path) -> str:
    """Extract the last assistant text block from an NDJSON transcript."""
    last_text = ""
    try:
        for raw in transcript_path.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if ev.get("type") != "assistant":
                continue
            for block in ev.get("message", {}).get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    last_text = block.get("text", "")
    except Exception:
        return ""
    return last_text


def _file_mentioned(expected: str, text: str) -> bool:
    """True if ``expected`` file path (or a meaningful suffix) appears in text.

    Agents frequently drop leading path components (``query/mod.rs`` for
    ``src/query/mod.rs``). We match if the full path appears, OR if the last
    two path components (``dir/file.ext``) appear as a token in the text.
    """
    if expected in text:
        return True
    parts = expected.split("/")
    if len(parts) >= 2:
        suffix = "/".join(parts[-2:])
        if suffix in text:
            return True
    # Fall back to the bare filename only if it is distinctive (has an extension
    # and is not a super-common name like mod.rs / lib.rs that would false-match).
    fname = parts[-1]
    common = {"mod.rs", "lib.rs", "main.rs"}
    if fname not in common and re.search(r"(?:^|[\s`*/(,\[])" + re.escape(fname) + r"(?:$|[\s`*.:,)\]])", text):
        return True
    return False


def score_trial(transcript_path: Path, expected_files: list[str]) -> dict:
    text = _extract_final_text(transcript_path)
    n_expected = len(expected_files)
    if not text:
        return {"n_expected": n_expected, "n_covered": 0, "recall": 0.0, "notes": "no_final_text"}
    covered = [f for f in expected_files if _file_mentioned(f, text)]
    recall = len(covered) / n_expected if n_expected else 0.0
    return {"n_expected": n_expected, "n_covered": len(covered),
            "recall": round(recall, 4), "notes": ""}


def load_tasks(tasks_path: Path) -> dict:
    """Minimal YAML reader for the iteration-forcing task file.

    Avoids a PyYAML dependency (harness is stdlib-only). Parses just the fields
    this scorer needs: id and ground_truth.expected_files.
    """
    tasks = {}
    cur_id = None
    in_expected = False
    expected: list[str] = []
    for raw in tasks_path.read_text().splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        # New task entry
        m = re.match(r"-\s+id:\s*(\S+)", stripped)
        if m:
            if cur_id:
                tasks[cur_id] = expected
            cur_id = m.group(1)
            expected = []
            in_expected = False
            continue
        if stripped.startswith("expected_files:"):
            in_expected = True
            continue
        if in_expected:
            im = re.match(r"-\s+(\S+)", stripped)
            # a list item at the expected_files indentation
            if im and (line.lstrip().startswith("-")) and "/" in im.group(1):
                expected.append(im.group(1))
                continue
            # any other key at same/shallower indent ends the list
            if stripped and not stripped.startswith("-"):
                in_expected = False
    if cur_id:
        tasks[cur_id] = expected
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=["A", "B"])
    args = ap.parse_args()

    tasks = load_tasks(Path(args.tasks))
    print(f"Loaded {len(tasks)} tasks with expected_files")

    results_dir = Path(args.results_dir)
    rows = []
    for arm in args.arms:
        for task_id, expected_files in tasks.items():
            if not expected_files:
                continue
            task_dir = results_dir / arm / task_id
            if not task_dir.is_dir():
                continue
            for tp in sorted(task_dir.glob("trial_*.ndjson")):
                trial = tp.stem.replace("trial_", "")
                sc = score_trial(tp, expected_files)
                rows.append({"arm": arm, "task_id": task_id, "trial": trial, **sc})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "task_id", "trial",
                                          "n_expected", "n_covered", "recall", "notes"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} recall rows to {args.out}")

    # Per-arm median recall summary (the guardrail comparison)
    for arm in args.arms:
        recs = [r["recall"] for r in rows if r["arm"] == arm]
        if recs:
            recs_sorted = sorted(recs)
            med = recs_sorted[len(recs_sorted) // 2]
            print(f"  arm {arm}: median file-coverage recall = {med:.3f} (n={len(recs)})")


if __name__ == "__main__":
    main()
