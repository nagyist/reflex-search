# Performance Research & Baselines

## Criterion Benchmark Baseline (2026-05-13)

Measured on the `feature/code-quality-refactor` branch.
Run with: `cargo bench --bench trigram_bench`

### trigram_extraction

Measures raw trigram extraction throughput from a 10 KB synthetic Rust source.

| Variant | Min | Median | Max |
|---------|-----|--------|-----|
| `no_locations_10kb` | 7.619 µs | **7.698 µs** | 7.783 µs |
| `with_locations_10kb` | 15.99 µs | **16.20 µs** | 16.47 µs |

**Takeaway**: Adding location tracking roughly doubles extraction cost (~2×), which is acceptable since it is only triggered during indexing, not queries.

### posting_list_intersection

3-trigram intersection across a synthetic 10 K-file index. Pattern `"abcde"` generates
`abc`→5 000 entries, `bcd`→5 000 entries, `cde`→5 000 entries; expected result ~2 500 files.

| Variant | Min | Median | Max |
|---------|-----|--------|-----|
| `3gram_10k_files` | 2.758 ms | **2.840 ms** | 2.934 ms |

**Takeaway**: Intersection for a 3-trigram pattern against 10 K files costs ~2.8 ms. The HashSet-based
implementation allocates once per list pair; see `Algorithmic complexity` lens. Large posting lists
from high-frequency trigrams (e.g., `" th"`) dominate this cost — use `max_posting_list_entries` to cap.

### index_and_query_roundtrip

Full index build + query on 1 000 synthetic Rust files (10 samples due to I/O expense).

| Variant | Min | Median | Max |
|---------|-----|--------|-----|
| `1k_files` | 776.7 ms | **804.2 ms** | 840.5 ms |

**Takeaway**: Indexing 1 K files costs ~800 ms (dominated by disk I/O and content hashing). This benchmark intentionally deletes the `.reflex/` cache each iteration to simulate cold-index builds.

### symbol_query_tree_sitter

Parses 100 candidate Rust files through tree-sitter to extract symbol definitions.
This simulates the symbol-query hot path when the trigram filter returns 100 candidates.

| Variant | Min | Median | Max |
|---------|-----|--------|-----|
| `100_candidates_rust` | 1.651 s | **1.671 s** | 1.692 s |

**Takeaway**: ~16.7 ms per candidate file for tree-sitter Rust parsing. This confirms the
`Tree-sitter query performance` lens: AST queries are ~1 000× slower than trigram search
(2 µs per trigram extraction vs 16.7 ms per tree-sitter parse). Always require `--glob` with `--ast`.

## Analysis

- **Query hot path** (trigram extraction only): <10 µs for 10 KB, scales linearly with source size.
- **Symbol hot path** (trigram + tree-sitter): trigram gets you to ~10–100 candidates in <1 ms, then tree-sitter adds ~16.7 ms/file overhead.
- **Cold indexing**: ~800 ms for 1 K files → ~50 K files/minute throughput. Parallel rayon indexing makes this viable.
- **Posting list** budget: 2.8 ms for a 10 K-file corpus with dense trigrams. For 100 K-file corpora this would be ~28 ms; use `max_posting_list_entries` to keep query latency under 10 ms.

## Benchmark Design Decisions

- **Bench 1 (extraction)**: Uses `make_rust_source(10_240)` which generates deterministic Rust with realistic identifier density. The "no locations" variant mirrors the query path; "with locations" mirrors the index path.
- **Bench 2 (intersection)**: Synthetic posting lists chosen to produce ~50% overlap, stress-testing the intersection algorithm without needing real files.
- **Bench 3 (roundtrip)**: `sample_size(10)` due to disk I/O; uses `TempDir` with per-iteration cache deletion to guarantee cold starts.
- **Bench 4 (tree-sitter)**: 100 files × ~200 lines each. Realistic: trigram filter would return ~10–100 candidates on a large codebase. Rust grammar chosen as the most mature and commonly benchmarked.
