#!/usr/bin/env python3
"""
Diagnose WHY Reflex-enabled arms cost more tokens than the built-in control (REF-176).

Decomposes the token disparity and pins the culprit. Reads NDJSON transcripts on
disk and emits aggregates only — never loads raw transcripts into an agent context
(the failure mode that corrupted the Phase 4 run; see run-detached.sh).

Findings feed the "how to make Reflex win" recommendations. Run:

    python3 benches/efficacy/diagnose_disparity.py [--results-dir results/]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()


def parse(path):
    """Resilient NDJSON parse (skips malformed lines). Returns list of events."""
    events = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def content_bytes(content):
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    if isinstance(content, list):
        return sum(content_bytes(c) for c in content)
    if isinstance(content, dict):
        return len(str(content.get("text", content.get("content", ""))).encode("utf-8"))
    return len(str(content).encode("utf-8"))


def tool_group(name: str) -> str:
    if name.startswith("mcp__reflex__"):
        return "reflex_mcp"
    if name in ("Grep", "Glob"):
        return "builtin_search"
    if name == "Read":
        return "Read"
    if name == "Bash":
        return "Bash"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=SCRIPT_DIR / "results")
    args = ap.parse_args()

    # Per-arm accumulators
    tool_calls = defaultdict(lambda: defaultdict(float))   # arm -> tool_name -> count
    group_calls = defaultdict(lambda: defaultdict(float))  # arm -> group -> count
    result_bytes = defaultdict(lambda: defaultdict(float)) # arm -> group -> total result bytes
    result_n = defaultdict(lambda: defaultdict(float))     # arm -> group -> #results
    trials = defaultdict(int)

    for nd in sorted(args.results_dir.rglob("*.ndjson")):
        parts = nd.relative_to(args.results_dir).parts
        if len(parts) < 3 or parts[1] == "_baseline":
            continue
        arm = parts[0]
        if arm not in ("A", "B", "C"):
            continue
        events = parse(nd)
        trials[arm] += 1
        tool_use_map = {}
        for ev in events:
            if ev.get("type") == "assistant":
                for b in ev.get("message", {}).get("content", []):
                    if b.get("type") == "tool_use":
                        name = b.get("name", "")
                        tool_use_map[b.get("id")] = name
                        tool_calls[arm][name] += 1
                        group_calls[arm][tool_group(name)] += 1
        for ev in events:
            if ev.get("type") == "user":
                for b in ev.get("message", {}).get("content", []):
                    if b.get("type") == "tool_result":
                        g = tool_group(tool_use_map.get(b.get("tool_use_id"), ""))
                        result_bytes[arm][g] += content_bytes(b.get("content", ""))
                        result_n[arm][g] += 1

    print("=== Avg tool calls per trial, by group ===")
    groups = ["builtin_search", "reflex_mcp", "Read", "Bash", "other"]
    hdr = f"{'arm':>4} " + " ".join(f"{g:>15}" for g in groups) + f"{'TOTAL':>9}"
    print(hdr)
    for a in ["A", "B", "C"]:
        n = trials[a] or 1
        cells = " ".join(f"{group_calls[a][g]/n:>15.2f}" for g in groups)
        tot = sum(group_calls[a].values()) / n
        print(f"{a:>4} {cells}{tot:>9.2f}")

    print("\n=== Avg result payload bytes PER CALL, by group (context each call injects) ===")
    print(f"{'arm':>4} " + " ".join(f"{g:>15}" for g in groups))
    for a in ["A", "B", "C"]:
        cells = []
        for g in groups:
            nn = result_n[a][g]
            cells.append(f"{(result_bytes[a][g]/nn if nn else 0):>15.0f}")
        print(f"{a:>4} " + " ".join(cells))

    print("\n=== Total result bytes injected per trial, by group ===")
    print(f"{'arm':>4} " + " ".join(f"{g:>15}" for g in groups) + f"{'TOTAL':>10}")
    for a in ["A", "B", "C"]:
        n = trials[a] or 1
        cells = " ".join(f"{result_bytes[a][g]/n:>15.0f}" for g in groups)
        tot = sum(result_bytes[a].values()) / n
        print(f"{a:>4} {cells}{tot:>10.0f}")

    print("\n=== Top reflex MCP tools called (avg/trial across B+C) ===")
    combined = defaultdict(float)
    ntrials = trials["B"] + trials["C"] or 1
    for a in ("B", "C"):
        for name, c in tool_calls[a].items():
            if name.startswith("mcp__reflex__"):
                combined[name] += c
    for name, c in sorted(combined.items(), key=lambda x: -x[1]):
        print(f"   {name:<34} {c/ntrials:.2f}")


if __name__ == "__main__":
    main()
