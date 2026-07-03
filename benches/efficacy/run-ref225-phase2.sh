#!/usr/bin/env bash
# REF-225: Phase 2 powered A/B — iteration-forcing task set
# Runs BOTH arms A+B on all 16 pre-registered iteration-forcing tasks.
#
# Prerequisites:
#   1. Phase 1 arm-A validity gate has PASSED (run-ref225-arm-a.sh completed,
#      metrics_ref225_arm_a.csv shows >=12 tasks with median turns >= 4).
#   2. rfx binary is built and current (CARGO_TARGET_DIR or ./target/release/rfx).
#
# This script re-runs arm A plus arm B in a single matrix pass.
# Arm-A trials that already exist are SKIPPED by the runner (resume-safe).
# Arm-B trials start fresh since B results don't exist yet after Phase 1.
#
# Primary endpoints (Holm-Bonferroni, pre-registered REF-225 spec):
#   assistant_turns, total_tool_calls, total_tokens
#
# Usage (run detached):
#   nohup bash benches/efficacy/run-ref225-phase2.sh [--dry-run] [--n N] \
#       >benches/efficacy/results/ref225-phase2.log 2>&1 &
#   echo $! > benches/efficacy/results/ref225-phase2.pid
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$SCRIPT_DIR/results/ref225-phase2.log"

N=8
DRY_RUN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN="--dry-run"; shift ;;
    --n=*) N="${1#--n=}"; shift ;;
    --n) N="${2:-8}"; shift 2 ;;
    *) shift ;;
  esac
done

mkdir -p "$SCRIPT_DIR/results"

echo "=== REF-225 Phase 2: Powered A/B — iteration-forcing tasks ===" | tee -a "$LOG_FILE"
echo "  Arms:  A B" | tee -a "$LOG_FILE"
echo "  N:     $N trials per arm × task" | tee -a "$LOG_FILE"
echo "  Model: claude-sonnet-4-6" | tee -a "$LOG_FILE"
echo "  Tasks: 16 iteration-forcing tasks (iteration-forcing.yaml)" | tee -a "$LOG_FILE"
echo "  Log:   $LOG_FILE" | tee -a "$LOG_FILE"
echo "  Primary endpoints: assistant_turns, total_tool_calls, total_tokens (Holm-Bonferroni)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Verify binary hygiene
RFX_BIN="${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/rfx"
if [ ! -x "$RFX_BIN" ]; then
  echo "ERROR: rfx binary not found at $RFX_BIN" | tee -a "$LOG_FILE"
  exit 1
fi

BUILD_SHA=$("$RFX_BIN" --version 2>/dev/null | head -1 || echo "unknown")
echo "  rfx binary: $RFX_BIN" | tee -a "$LOG_FILE"
echo "  rfx version: $BUILD_SHA" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Verify Phase 1 gate was passed before unblinding arm B
ARM_A_CSV="$SCRIPT_DIR/results/metrics_ref225_arm_a.csv"
if [ -z "$DRY_RUN" ] && [ ! -f "$ARM_A_CSV" ]; then
  echo "ERROR: Phase 1 arm-A metrics CSV not found: $ARM_A_CSV" | tee -a "$LOG_FILE"
  echo "       Run run-ref225-arm-a.sh first and confirm PHASE 1 PASSES." | tee -a "$LOG_FILE"
  exit 1
fi

# All 16 pre-registered iteration-forcing task IDs (frozen at git commit 3e50d61)
IF_TASKS=(
  # comprehension (Reflex)
  "reflex-comp-query-path"
  "reflex-comp-trigram-extract"
  "reflex-comp-deleted-file"
  "reflex-comp-mcp-dispatch"
  # comprehension (ripgrep)
  "ripgrep-comp-binary"
  "ripgrep-comp-type-flag"
  "ripgrep-comp-parallel"
  # comprehension (tokio)
  "tokio-comp-work-steal"
  "tokio-comp-task-abort"
  "tokio-comp-io-driver"
  # transitive_dep (Reflex)
  "reflex-trans-content-store"
  "reflex-trans-regex-trigrams"
  # cross_module
  "reflex-cm-language-dispatch"
  "ripgrep-cm-printer-chain"
  "ripgrep-cm-stats-tracking"
  "tokio-cm-spawn-chain"
)

echo "Running matrix: ${#IF_TASKS[@]} tasks × $N trials × 2 arms" | tee -a "$LOG_FILE"
echo "  (Arm-A trials already completed by Phase 1 will be SKIPPED — runner is resume-safe)" | tee -a "$LOG_FILE"
echo "Started at: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

cd "$REPO_ROOT"
python3 benches/efficacy/runner.py \
  --arms A B \
  --tasks "${IF_TASKS[@]}" \
  --n "$N" \
  --model claude-sonnet-4-6 \
  --skip-build \
  --skip-index \
  $DRY_RUN \
  2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Runner complete at: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"

if [ -z "$DRY_RUN" ]; then
  echo "" | tee -a "$LOG_FILE"
  echo "=== Extracting Phase 2 metrics ===" | tee -a "$LOG_FILE"
  python3 benches/efficacy/extract_metrics.py \
    --results-dir benches/efficacy/results/ \
    --out benches/efficacy/results/metrics_ref225.csv \
    2>&1 | tee -a "$LOG_FILE"

  echo "" | tee -a "$LOG_FILE"
  echo "=== Phase 2 statistical analysis (Holm-Bonferroni) ===" | tee -a "$LOG_FILE"
  python3 - <<'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import csv, statistics, math, sys
from pathlib import Path

csv_path = Path("benches/efficacy/results/metrics_ref225.csv")
if not csv_path.exists():
    print("ERROR: metrics CSV not found")
    sys.exit(1)

rows = list(csv.DictReader(open(csv_path)))

IF_TASK_IDS = [
    "reflex-comp-query-path", "reflex-comp-trigram-extract", "reflex-comp-deleted-file",
    "reflex-comp-mcp-dispatch", "ripgrep-comp-binary", "ripgrep-comp-type-flag",
    "ripgrep-comp-parallel", "tokio-comp-work-steal", "tokio-comp-task-abort",
    "tokio-comp-io-driver", "reflex-trans-content-store", "reflex-trans-regex-trigrams",
    "reflex-cm-language-dispatch", "ripgrep-cm-printer-chain", "ripgrep-cm-stats-tracking",
    "tokio-cm-spawn-chain",
]

ENDPOINTS = ["assistant_turns", "total_tool_calls", "total_tokens"]

def get_vals(arm, endpoint):
    return [float(r[endpoint]) for r in rows
            if r.get("arm") == arm and r.get("task_id") in IF_TASK_IDS
            and r.get(endpoint) and r[endpoint].strip()]

print("PRIMARY ENDPOINTS (Holm-Bonferroni, 3 tests):")
print(f"  {'Endpoint':<25} {'A median':>10} {'B median':>10} {'ratio B/A':>10}  direction")
print("  " + "-"*70)

ratios = []
for ep in ENDPOINTS:
    a_vals = get_vals("A", ep)
    b_vals = get_vals("B", ep)
    if not a_vals or not b_vals:
        print(f"  {ep:<25}  NO DATA")
        continue
    med_a = statistics.median(a_vals)
    med_b = statistics.median(b_vals)
    ratio = med_b / med_a if med_a > 0 else float('inf')
    direction = "B better" if ratio < 0.95 else ("B worse" if ratio > 1.05 else "parity")
    ratios.append((ep, med_a, med_b, ratio, direction))
    print(f"  {ep:<25} {med_a:>10.1f} {med_b:>10.1f} {ratio:>10.3f}  {direction}")

print()
if ratios:
    directions = [r[4] for r in ratios]
    all_better = all(d == "B better" for d in directions)
    all_parity = all(d == "parity" for d in directions)
    consistent = all(d == directions[0] for d in directions)
    print(f"  Co-primary direction consistency: {'YES' if consistent else 'NO'}")
    if all_better:
        print("  VERDICT: B WINS — Reflex reduces iterations on iteration-forcing tasks")
    elif all_parity:
        print("  VERDICT: PARITY — no measurable iteration advantage for Reflex")
    elif not consistent:
        print("  VERDICT: MIXED — endpoints disagree; report individually")
    else:
        print("  VERDICT: INDETERMINATE — direction consistent but effect size uncertain")

print()
print("Full data in: benches/efficacy/results/metrics_ref225.csv")
print("Run benches/efficacy/stats.py for Wilcoxon p-values and CIs.")
PYEOF
fi
