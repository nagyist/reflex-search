#!/usr/bin/env bash
# REF-217: Extract metrics and run analysis after the A/B run completes.
set -e
cd "$(dirname "$0")/../.."
RESULTS="benches/efficacy/results"
EFFICACY="benches/efficacy"

echo "=== REF-217 Post-run analysis ==="
echo "Extracting metrics..."
python3 "$EFFICACY/extract_metrics.py" \
  --arms A B \
  --out "$RESULTS/metrics.csv" \
  --no-fail-on-missing-usage

echo "Running analyze.py..."
python3 "$EFFICACY/analyze.py" \
  --metrics "$RESULTS/metrics.csv" \
  --outdir "$RESULTS/analysis" \
  --plotdir "$RESULTS/plots"

echo "=== Done ==="
cat "$RESULTS/analysis/summary.md" 2>/dev/null || cat "$RESULTS/analysis/summary.json" 2>/dev/null
