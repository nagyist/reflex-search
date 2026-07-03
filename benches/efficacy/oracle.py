#!/usr/bin/env python3
"""Canonical ground-truth oracle for the Reflex efficacy corpus.

This module is the SINGLE SOURCE OF TRUTH for how a task's ground truth is
derived from a pinned repository checkout. Both the task-authoring step and the
validation script (``validate.py``) import these functions so that the numbers
baked into ``tasks/*.yaml`` are computed the exact same way they are checked.

Design principle: the oracle is **independent of Reflex** (it shells out to
ripgrep, an orthogonal tool) so ground truth cannot be circular — we never grade
Reflex against Reflex. See ``tasks/SCHEMA.md`` for the task schema.
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass


class OracleError(RuntimeError):
    """Raised when ripgrep is missing or an oracle cannot be evaluated."""


def _run_rg(args: list[str], cwd: str) -> str:
    """Run ripgrep in ``cwd`` and return stdout. Exit code 1 (no matches) is OK."""
    try:
        proc = subprocess.run(
            ["rg", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment guard
        raise OracleError("ripgrep (`rg`) not found on PATH") from exc
    # rg exit codes: 0 = matches, 1 = no matches, 2 = actual error.
    if proc.returncode == 2:
        raise OracleError(f"ripgrep error: {proc.stderr.strip()}")
    return proc.stdout


def match_set(pattern: str, path: str, flags: list[str] | None, cwd: str) -> list[str]:
    """Return the sorted-unique ``file:line`` set for a pattern in a checkout.

    This is the deterministic 'complete coverage' answer key for
    find-all / usages / occurrence tasks. ``flags`` are extra rg flags such as
    ``-w`` (word boundary) or ``-F`` (fixed/literal string).
    """
    flags = flags or []
    # --with-filename forces the `path:` prefix even when `path` is a single file
    # (rg omits it otherwise, which would break the `path:line` parser).
    out = _run_rg(
        ["--no-heading", "--line-number", "--with-filename", "--color", "never",
         *flags, "--", pattern, path],
        cwd,
    )
    pairs = set()
    for line in out.splitlines():
        # rg output is `path:line:content`; path never contains ':' in our repos.
        parts = line.split(":", 2)
        if len(parts) >= 2 and parts[1].isdigit():
            pairs.add(f"{parts[0]}:{parts[1]}")
    return sorted(pairs)


def file_set(pattern: str, path: str, flags: list[str] | None, cwd: str) -> list[str]:
    """Return the sorted-unique set of files containing a pattern.

    This is the answer key for refactor-scoping ('every file you'd touch to
    rename X') and reverse-dependency ('what depends on F') tasks.
    """
    flags = flags or []
    out = _run_rg(
        ["--no-heading", "--files-with-matches", "--color", "never", *flags, "--", pattern, path],
        cwd,
    )
    return sorted({ln for ln in out.splitlines() if ln})


def canonical_sha256(lines: list[str]) -> str:
    """Stable checksum of a ground-truth set: sorted lines joined with '\\n'."""
    blob = "\n".join(lines) + ("\n" if lines else "")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class OracleResult:
    lines: list[str]
    count: int
    sha256: str


def evaluate(kind: str, pattern: str, path: str, flags: list[str] | None, cwd: str) -> OracleResult:
    """Evaluate a ``match_set`` or ``file_set`` oracle against a checkout."""
    if kind == "match_set":
        lines = match_set(pattern, path, flags, cwd)
    elif kind == "file_set":
        lines = file_set(pattern, path, flags, cwd)
    else:
        raise OracleError(f"unknown oracle kind: {kind}")
    return OracleResult(lines=lines, count=len(lines), sha256=canonical_sha256(lines))


if __name__ == "__main__":
    # Authoring helper: `python3 oracle.py <match_set|file_set> <pattern> <path> [flags...] [--cwd DIR]`
    import sys

    argv = sys.argv[1:]
    cwd = "."
    if "--cwd" in argv:
        i = argv.index("--cwd")
        cwd = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    kind, pattern, path, *flags = argv
    res = evaluate(kind, pattern, path, flags, cwd)
    print(f"count={res.count}")
    print(f"sha256={res.sha256}")
    for ln in res.lines:
        print(ln)
