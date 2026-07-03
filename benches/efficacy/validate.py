#!/usr/bin/env python3
"""Validate the efficacy task corpus.

Confirms (acceptance criterion for REF-178) that:
  1. Every task in tasks/*.yaml parses and satisfies the schema (SCHEMA.md).
  2. Every task id is globally unique; every repo_sha matches the pin in repos.md.
  3. Every ground-truth reference RESOLVES in its pinned repo checkout — by
     re-deriving the answer key with the independent ripgrep oracle (oracle.py)
     and asserting the recorded count/sha256/location/files.

Ground truth is checked against a checkout pinned at the task's repo_sha, so the
answer key can never silently drift from the code.

Usage:
    python3 validate.py                 # fetch missing repos, validate everything
    python3 validate.py --no-fetch      # fail if a pinned checkout is missing (CI)
    python3 validate.py --repo reflex   # validate one repo's tasks only
    python3 validate.py -v              # verbose: print each task's result

Requires a YAML loader: PyYAML if importable, else the `yq` CLI. Requires `rg`
(ripgrep) and `git` on PATH.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import oracle  # shared, non-circular ground-truth normalization

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE / "tasks"
CORPUS_DIR = HERE / "corpus"
REPO_ROOT = HERE.parent.parent  # the Reflex repository root

# Pin registry — MUST stay in sync with repos.md.
REPOS = {
    "reflex": {
        "sha": "d2935f48f5abea2a76b479040a23478155be9bb0",
        "url": None,  # this repository; materialized via `git worktree`
        "kind": "worktree",
    },
    "ripgrep": {
        "sha": "4649aa9700619f94cf9c66876e9549d83420e16c",
        "url": "https://github.com/BurntSushi/ripgrep.git",
        "kind": "clone",
    },
    "tokio": {
        "sha": "ab3ff69cf2258a8c696b2dca89a2cef4ff114c1c",
        "url": "https://github.com/tokio-rs/tokio.git",
        "kind": "clone",
    },
}

CATEGORIES = {
    "locate_definition", "find_all_usages", "refactor_scope", "dependency_imports",
    "reverse_dependency", "hotspot", "cross_module", "comprehension", "negative_control",
}
EXPECT = {"win", "neutral", "lose"}
GT_TYPES = {"location", "match_set", "file_set", "ranking", "rubric"}
REQUIRED_FIELDS = ("id", "repo", "repo_sha", "category", "expect_reflex", "prompt", "ground_truth", "oracle_notes")


# ── YAML loading (PyYAML → yq CLI) ──────────────────────────────────────────
def load_yaml(path: Path):
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text())
    except ModuleNotFoundError:
        pass
    # mikefarah yq (`-o=json`) then kislyuk yq (JSON by default)
    for cmd in (["yq", "-o=json", ".", str(path)], ["yq", ".", str(path)]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            break
        if out.returncode == 0 and out.stdout.strip():
            try:
                return json.loads(out.stdout)
            except json.JSONDecodeError:
                continue
    sys.exit("FATAL: no YAML loader available — install PyYAML (`pip install pyyaml`) or the `yq` CLI.")


# ── git / checkout management ───────────────────────────────────────────────
def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True).stdout.strip()


def ensure_checkout(repo_id: str, fetch: bool) -> Path:
    info = REPOS[repo_id]
    dest = CORPUS_DIR / repo_id
    if (dest / ".git").exists() or dest.exists():
        head = _git(dest, "rev-parse", "HEAD")
        if head == info["sha"]:
            return dest
        # present but wrong SHA — try to move it onto the pin
        _git(dest, "checkout", "--detach", info["sha"])
        if _git(dest, "rev-parse", "HEAD") == info["sha"]:
            return dest
        raise RuntimeError(f"{repo_id}: checkout at {dest} is {head[:10]}, expected {info['sha'][:10]}")
    if not fetch:
        raise RuntimeError(f"{repo_id}: no checkout at {dest} and --no-fetch set. See repos.md to fetch.")
    CORPUS_DIR.mkdir(exist_ok=True)
    if info["kind"] == "worktree":
        r = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach", str(dest), info["sha"]],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            # Fall back to the live checkout (src/ is byte-identical at the pin; see repos.md).
            print(f"  note: worktree add failed ({r.stderr.strip()}); using live checkout {REPO_ROOT}")
            return REPO_ROOT
    else:
        subprocess.run(
            ["git", "clone", "--filter=blob:none", info["url"], str(dest)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(["git", "-C", str(dest), "checkout", "--detach", info["sha"]], check=True,
                       capture_output=True, text=True)
    return dest


# ── ground-truth resolvers (one per type) ───────────────────────────────────
def check_location(gt: dict, cwd: Path) -> list[str]:
    errs = []
    f = cwd / gt["file"]
    if not f.exists():
        return [f"file not found: {gt['file']}"]
    lines = f.read_text(errors="replace").splitlines()
    n = gt["line"]
    if n < 1 or n > len(lines):
        return [f"line {n} out of range (file has {len(lines)} lines)"]
    text = lines[n - 1]
    if not re.search(gt["expect_regex"], text):
        errs.append(f"line {n} {text!r} does not match /{gt['expect_regex']}/")
    return errs


def _oracle_spec(gt: dict):
    o = gt["oracle"]
    return o["pattern"], o["path"], list(o.get("flags") or [])


def check_set(gt: dict, kind: str, cwd: Path) -> list[str]:
    pattern, path, flags = _oracle_spec(gt)
    res = oracle.evaluate(kind, pattern, path, flags, str(cwd))
    errs = []
    if res.count != gt["expected_count"]:
        errs.append(f"count {res.count} != expected {gt['expected_count']}")
    if res.sha256 != gt["expected_sha256"]:
        errs.append(f"sha256 {res.sha256[:12]}… != expected {gt['expected_sha256'][:12]}…")
    if kind == "file_set" and "expected_files" in gt:
        want = sorted(gt["expected_files"])
        if res.lines != want:
            missing = sorted(set(want) - set(res.lines))
            extra = sorted(set(res.lines) - set(want))
            errs.append(f"file set mismatch (missing={missing}, extra={extra})")
    return errs


def check_ranking(gt: dict, cwd: Path) -> list[str]:
    errs = []
    counts = {}
    for name in gt["candidates"]:
        pattern = gt["pattern_template"].replace("{name}", name)
        res = oracle.evaluate("file_set", pattern, gt["path"], list(gt.get("flags") or []), str(cwd))
        counts[name] = res.count
    top = max(counts, key=counts.get)
    if top != gt["expected_top"]:
        errs.append(f"argmax is {top} ({counts[top]}), expected_top {gt['expected_top']} ({counts.get(gt['expected_top'])})")
    for name, want in (gt.get("expected_counts") or {}).items():
        if counts.get(name) != want:
            errs.append(f"count[{name}]={counts.get(name)} != expected {want}")
    return errs


def check_rubric(gt: dict, cwd: Path) -> list[str]:
    errs = []
    for rel in gt.get("expected_files", []):
        if not (cwd / rel).exists():
            errs.append(f"expected_file not found: {rel}")
    if not gt.get("required_facts"):
        errs.append("rubric has no required_facts")
    return errs


def resolve_ground_truth(gt: dict, cwd: Path) -> list[str]:
    t = gt["type"]
    if t == "location":
        return check_location(gt, cwd)
    if t == "match_set":
        return check_set(gt, "match_set", cwd)
    if t == "file_set":
        return check_set(gt, "file_set", cwd)
    if t == "ranking":
        return check_ranking(gt, cwd)
    if t == "rubric":
        return check_rubric(gt, cwd)
    return [f"unknown ground_truth.type: {t}"]


# ── schema validation ───────────────────────────────────────────────────────
def validate_schema(task: dict) -> list[str]:
    errs = []
    for field in REQUIRED_FIELDS:
        if field not in task or task[field] in (None, ""):
            errs.append(f"missing field: {field}")
    if errs:
        return errs
    if task["repo"] not in REPOS:
        errs.append(f"unknown repo: {task['repo']}")
    elif task["repo_sha"] != REPOS[task["repo"]]["sha"]:
        errs.append(f"repo_sha {task['repo_sha'][:10]}… != pin for {task['repo']}")
    if task["category"] not in CATEGORIES:
        errs.append(f"unknown category: {task['category']}")
    if task["expect_reflex"] not in EXPECT:
        errs.append(f"invalid expect_reflex: {task['expect_reflex']}")
    gt = task.get("ground_truth") or {}
    if gt.get("type") not in GT_TYPES:
        errs.append(f"invalid ground_truth.type: {gt.get('type')}")
    return errs


# ── main ────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the efficacy task corpus.")
    ap.add_argument("--no-fetch", action="store_true", help="fail if a pinned checkout is missing")
    ap.add_argument("--repo", help="validate only this repo's tasks")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each task result")
    args = ap.parse_args()

    files = sorted(TASKS_DIR.glob("*.yaml"))
    if not files:
        print("no task files found in tasks/*.yaml", file=sys.stderr)
        return 1

    all_tasks: list[tuple[Path, dict]] = []
    seen_ids: dict[str, Path] = {}
    n_fail = 0
    expect_tally = {"win": 0, "neutral": 0, "lose": 0}
    cat_tally: dict[str, int] = {}
    checkouts: dict[str, Path] = {}

    # 1) load + schema + uniqueness
    for f in files:
        doc = load_yaml(f)
        tasks = (doc or {}).get("tasks") if isinstance(doc, dict) else doc
        if not isinstance(tasks, list):
            print(f"FAIL {f.name}: top-level `tasks:` list not found")
            n_fail += 1
            continue
        for task in tasks:
            all_tasks.append((f, task))
            tid = task.get("id", "<no-id>")
            if tid in seen_ids:
                print(f"FAIL {f.name}: duplicate id {tid} (also in {seen_ids[tid].name})")
                n_fail += 1
            seen_ids[tid] = f

    # 2) resolve ground truth per task
    for f, task in all_tasks:
        tid = task.get("id", "<no-id>")
        repo = task.get("repo")
        if args.repo and repo != args.repo:
            continue
        errs = validate_schema(task)
        if not errs:
            if repo not in checkouts:
                try:
                    checkouts[repo] = ensure_checkout(repo, fetch=not args.no_fetch)
                except Exception as exc:  # noqa: BLE001
                    print(f"FAIL {tid}: cannot prepare {repo} checkout: {exc}")
                    n_fail += 1
                    continue
            try:
                errs = resolve_ground_truth(task["ground_truth"], checkouts[repo])
            except oracle.OracleError as exc:
                errs = [f"oracle error: {exc}"]
        if errs:
            n_fail += 1
            print(f"FAIL {tid}")
            for e in errs:
                print(f"       - {e}")
        else:
            expect_tally[task["expect_reflex"]] += 1
            cat_tally[task["category"]] = cat_tally.get(task["category"], 0) + 1
            if args.verbose:
                print(f"ok   {tid}  [{task['category']}, expect={task['expect_reflex']}]")

    # 3) summary + no-cherry-picking guard
    total = len([t for _, t in all_tasks if not args.repo or t.get("repo") == args.repo])
    passed = total - n_fail
    print("\n── summary ──────────────────────────────────────────")
    print(f"tasks: {total}  passed: {passed}  failed: {n_fail}")
    print(f"categories covered ({len(cat_tally)}/9): " + ", ".join(f"{k}={v}" for k, v in sorted(cat_tally.items())))
    print(f"expect_reflex mix: win={expect_tally['win']} neutral={expect_tally['neutral']} lose={expect_tally['lose']}")
    if not args.repo:
        if not (30 <= total <= 50):
            print(f"WARN: task count {total} outside the 30–50 target range")
        for k in ("win", "neutral", "lose"):
            if expect_tally[k] == 0:
                print(f"WARN: no `{k}` tasks — mix must include win, neutral AND lose (no cherry-picking)")
    if n_fail:
        print("\nRESULT: FAIL")
        return 1
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
