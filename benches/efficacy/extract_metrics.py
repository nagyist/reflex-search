#!/usr/bin/env python3
"""
Phase 2 metrics extractor for the Reflex efficacy study.

Parses NDJSON transcripts → tidy CSV (one row per trial) with all H1
efficiency fields. Fails loudly if a transcript is missing required usage fields.

Usage:
    python3 benches/efficacy/extract_metrics.py [--results-dir results/]
                                                [--out metrics.csv]
                                                [--arms A B C Bprime]
                                                [--include-baselines]
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / "results"

# Tool categories for metric columns
SEARCH_TOOLS_BUILTIN = frozenset(["Grep", "Glob"])
SEARCH_TOOLS_MCP = frozenset([
    "mcp__reflex__search_code",
    "mcp__reflex__search_regex",
    "mcp__reflex__search_ast",
    "mcp__reflex__find_references",
    "mcp__reflex__gather_context",
    "mcp__reflex__list_locations",
    "mcp__reflex__count_occurrences",
    "mcp__reflex__find_circular",
    "mcp__reflex__find_hotspots",
    "mcp__reflex__find_islands",
    "mcp__reflex__find_unused",
    "mcp__reflex__get_dependencies",
    "mcp__reflex__get_dependents",
    "mcp__reflex__get_transitive_deps",
    "mcp__reflex__check_index_status",
    "mcp__reflex__index_project",
    "mcp__reflex__analyze_summary",
    "mcp__reflex__list_locations",
])
READ_TOOL = "Read"


CSV_COLUMNS = [
    "arm",
    "task_id",
    "trial",
    "model",
    "repo_sha",
    # Efficiency metrics (H1 hypothesis fields)
    "wall_ms",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "total_cost_usd",
    "assistant_turns",
    "total_tool_calls",
    "toolsearch_calls",
    "search_tool_calls_builtin",
    "search_tool_calls_mcp",
    "read_tool_calls",
    "bytes_read",
    "lines_read",
    # Arm verification fields
    "mcp_servers_active",
    "grep_glob_blocked",
    "reflex_tools_used",
    # Result quality
    "success",
    "stop_reason",
    "permission_denials",
    "parse_warnings",
    # Transcript path (for manual audit)
    "transcript_path",
]


def _content_bytes(content: Any) -> int:
    """Measure the byte size of a tool result's content field."""
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    if isinstance(content, list):
        return sum(_content_bytes(item) for item in content)
    if isinstance(content, dict):
        text = content.get("text", content.get("content", ""))
        return len(str(text).encode("utf-8"))
    return len(str(content).encode("utf-8"))


def _content_lines(content: Any) -> int:
    """Count newlines in a tool result's content field."""
    if content is None:
        return 0
    if isinstance(content, str):
        return content.count("\n")
    if isinstance(content, list):
        return sum(_content_lines(item) for item in content)
    if isinstance(content, dict):
        text = content.get("text", content.get("content", ""))
        return str(text).count("\n")
    return str(content).count("\n")


def parse_transcript(path: Path) -> dict:
    """
    Parse a single NDJSON transcript and return a metrics dict.

    Resilient to individual malformed lines: a stray non-UTF8 byte (e.g. a
    smart-quote in tool output) or a single unparseable stream-json line should
    NOT discard an otherwise-complete, result-bearing trial. Such lines are
    skipped and counted in `parse_warnings` so downstream analysis can flag or
    exclude degraded trials. We still fail loudly if the terminal `result`
    event (which carries the token/usage metrics) is missing — that is a
    genuinely unusable trial.

    Non-UTF8 bytes are tolerated via errors="replace" so a single bad byte does
    not abort the whole file read.
    """
    events = []
    skipped_lines = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip this malformed line but keep parsing the rest of the
                # transcript. Losing one intermediate line degrades secondary
                # metrics (bytes_read / tool counts) but preserves the result
                # event's token usage. Count it for transparency.
                skipped_lines += 1
                continue

    # --- Extract harness metadata (first line we wrote) ---
    metadata = {}
    for ev in events:
        if ev.get("type") == "harness_metadata":
            metadata = ev
            break

    # --- Extract init event (model, mcp_servers) ---
    init_ev = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None)
    model = metadata.get("model") or (init_ev.get("model") if init_ev else "unknown")
    repo_sha = metadata.get("repo_sha", "unknown")
    arm = metadata.get("arm", path.parts[-3] if len(path.parts) >= 3 else "unknown")
    task_id = metadata.get("task_id", path.parts[-2] if len(path.parts) >= 2 else "unknown")
    trial = metadata.get("trial", int(path.stem.replace("trial_", "")) if "trial_" in path.stem else 0)

    mcp_servers_active = []
    grep_glob_blocked = False
    if init_ev:
        mcp_servers_active = [s["name"] for s in init_ev.get("mcp_servers", []) if s.get("status") == "connected"]
        # Grep/Glob blocked if they are absent from the tools list
        tools_list = init_ev.get("tools", [])
        if tools_list:
            grep_glob_blocked = "Grep" not in tools_list and "Glob" not in tools_list

    # --- Extract result event ---
    result_ev = next(
        (e for e in reversed(events) if e.get("type") == "result"),
        None,
    )
    if result_ev is None:
        raise ValueError(f"No terminal 'result' event found in {path}")

    usage = result_ev.get("usage")
    if not usage:
        raise ValueError(
            f"Missing 'usage' in result event of {path}. "
            "Cannot compute token metrics — fail loudly per Phase 0 recipe."
        )

    # --- Build tool_use_id → tool_name map from assistant events ---
    tool_use_map: dict[str, str] = {}
    for ev in events:
        if ev.get("type") == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    tool_use_map[block["id"]] = block["name"]

    # --- Count tool calls ---
    total_tool_calls = 0
    toolsearch_calls = 0
    search_builtin = 0
    search_mcp = 0
    read_calls = 0
    bytes_read = 0
    lines_read = 0
    reflex_tools_used: set[str] = set()

    # Tool calls come from assistant events (tool_use blocks)
    for ev in events:
        if ev.get("type") == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    name = block.get("name", "")
                    total_tool_calls += 1
                    if name == "ToolSearch":
                        toolsearch_calls += 1
                    if name in SEARCH_TOOLS_BUILTIN:
                        search_builtin += 1
                    if name in SEARCH_TOOLS_MCP:
                        search_mcp += 1
                        reflex_tools_used.add(name)
                    if name == READ_TOOL:
                        read_calls += 1

    # Read bytes/lines from user events (tool results)
    for ev in events:
        if ev.get("type") == "user":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id", "")
                    tool_name = tool_use_map.get(tool_use_id, "")
                    if tool_name == READ_TOOL:
                        content = block.get("content", "")
                        bytes_read += _content_bytes(content)
                        lines_read += _content_lines(content)

    return {
        "arm": arm,
        "task_id": task_id,
        "trial": trial,
        "model": model,
        "repo_sha": repo_sha,
        # Core efficiency metrics
        "wall_ms": result_ev.get("duration_ms", ""),
        "input_tokens": usage.get("input_tokens", ""),
        "output_tokens": usage.get("output_tokens", ""),
        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
        "total_cost_usd": result_ev.get("total_cost_usd", ""),
        "assistant_turns": result_ev.get("num_turns", ""),
        # Tool call counts
        "total_tool_calls": total_tool_calls,
        "toolsearch_calls": toolsearch_calls,
        "search_tool_calls_builtin": search_builtin,
        "search_tool_calls_mcp": search_mcp,
        "read_tool_calls": read_calls,
        "bytes_read": bytes_read,
        "lines_read": lines_read,
        # Arm verification
        "mcp_servers_active": "|".join(mcp_servers_active),
        "grep_glob_blocked": grep_glob_blocked,
        "reflex_tools_used": "|".join(sorted(reflex_tools_used)),
        # Quality
        "success": not result_ev.get("is_error", False),
        "stop_reason": result_ev.get("stop_reason", ""),
        "permission_denials": len(result_ev.get("permission_denials", [])),
        # Number of NDJSON lines that failed to parse and were skipped. >0 means
        # secondary metrics (bytes_read/tool counts) may be undercounted for this
        # trial; token/usage metrics from the result event are still authoritative.
        "parse_warnings": skipped_lines,
        "transcript_path": str(path),
    }


def discover_transcripts(results_dir: Path, arms: list[str] | None, include_baselines: bool) -> list[Path]:
    """Walk results_dir and return all .ndjson transcript paths."""
    paths = []
    for ndjson in sorted(results_dir.rglob("*.ndjson")):
        # Structure: results/{arm}/{task_id}/trial_NN.ndjson
        parts = ndjson.relative_to(results_dir).parts
        if len(parts) < 3:
            continue
        arm = parts[0]
        task_id = parts[1]

        if arms and arm not in arms:
            continue
        if not include_baselines and task_id == "_baseline":
            continue
        paths.append(ndjson)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reflex efficacy metrics extractor (Phase 2)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Root directory for transcript files (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SCRIPT_DIR / "results" / "metrics.csv",
        help="Output CSV path (default: results/metrics.csv)",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        metavar="ARM",
        help="Filter to specific arms (default: all)",
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Include _baseline rows in output CSV",
    )
    parser.add_argument(
        "--no-fail-on-missing-usage",
        action="store_true",
        default=False,
        dest="skip_incomplete",
        help="Tolerate transcripts missing a terminal result event (e.g. in-flight trials)",
    )
    args = parser.parse_args()

    transcripts = discover_transcripts(args.results_dir, args.arms, args.include_baselines)
    if not transcripts:
        sys.exit(
            f"ERROR: No transcript .ndjson files found under {args.results_dir}.\n"
            "Run runner.py first."
        )

    print(f"Found {len(transcripts)} transcript(s) to process.")

    rows = []
    errors = []
    for path in transcripts:
        try:
            row = parse_transcript(path)
            rows.append(row)
        except ValueError as e:
            errors.append((path, str(e)))
            print(f"  ERROR: {e}", file=sys.stderr)

    if errors and not args.skip_incomplete:
        print(
            f"\n{len(errors)} transcript(s) failed parsing. "
            "Fix the above errors before producing metrics.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Write CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")

    # Quick sanity print: per-arm summary
    from collections import defaultdict
    by_arm: dict[str, list] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)

    print("\nPer-arm summary (excluding baselines):")
    for arm, arm_rows in sorted(by_arm.items()):
        non_base = [r for r in arm_rows if r["task_id"] != "_baseline"]
        if not non_base:
            continue
        def safe_avg(field):
            vals = [r[field] for r in non_base if isinstance(r.get(field), (int, float))]
            return round(sum(vals) / len(vals), 1) if vals else "n/a"

        print(
            f"  {arm}: n={len(non_base)}"
            f"  wall_ms={safe_avg('wall_ms')}"
            f"  total_tokens={safe_avg('input_tokens')}"
            f"  tool_calls={safe_avg('total_tool_calls')}"
            f"  mcp_calls={safe_avg('search_tool_calls_mcp')}"
            f"  builtin_search_calls={safe_avg('search_tool_calls_builtin')}"
        )


if __name__ == "__main__":
    main()
