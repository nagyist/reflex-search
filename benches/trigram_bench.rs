//! Criterion benchmarks for the trigram indexing and query pipeline.
//!
//! Run with: cargo bench
//! HTML reports: target/criterion/

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use reflex::models::{IndexConfig, Language};
use reflex::parsers::ParserFactory;
use reflex::trigram::{
    FileLocation, TrigramIndex, extract_trigrams, extract_trigrams_with_locations,
};
use reflex::{CacheManager, Indexer, QueryEngine, QueryFilter};
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

// ─── helpers ──────────────────────────────────────────────────────────────────

fn make_rust_source(approx_bytes: usize) -> String {
    let mut s = String::with_capacity(approx_bytes + 256);
    let mut i = 0usize;
    while s.len() < approx_bytes {
        s.push_str(&format!(
            "fn func_{i}(x: i32, y: i32) -> i32 {{\n\
             \tlet a = x + y * {i};\n\
             \tlet b = a.wrapping_add({i});\n\
             \tprintln!(\"{{}} {{}}\", a, b);\n\
             \ta + b\n\
             }}\n\n"
        ));
        i += 1;
    }
    s.truncate(approx_bytes);
    s
}

// ─── 1. Trigram extraction ─────────────────────────────────────────────────────
//
// Measures the raw throughput of extracting trigrams from a 10 KB Rust source.
// Two variants: without location info (used for query pattern matching) and
// with location info (used during indexing).

fn bench_trigram_extraction(c: &mut Criterion) {
    let source = make_rust_source(10 * 1024);

    let mut g = c.benchmark_group("trigram_extraction");

    g.bench_function("no_locations_10kb", |b| {
        b.iter(|| extract_trigrams(black_box(&source)))
    });

    g.bench_function("with_locations_10kb", |b| {
        b.iter(|| extract_trigrams_with_locations(black_box(&source), 0))
    });

    g.finish();
}

// ─── 2. Posting-list intersection ─────────────────────────────────────────────
//
// Builds a synthetic in-memory index spanning 10 K synthetic files, then
// measures the cost of a 3-trigram intersection (the hot path of every query).
//
// Layout (pattern "abcde" → trigrams "abc", "bcd", "cde"):
//   "abc" → files 0–4 999      (5 000 entries)
//   "bcd" → files 2 500–7 499  (5 000 entries)
//   "cde" → files 0–4 999      (5 000 entries)
// Expected intersection: files 2 500–4 999 (~2 500 results)
//
// Lens: Algorithmic complexity — intersection is O(n) per list pair;
// the HashSet-based implementation incurs one allocation per list.

fn bench_posting_list_intersection(c: &mut Criterion) {
    let abc: u32 = (b'a' as u32) << 16 | (b'b' as u32) << 8 | b'c' as u32;
    let bcd: u32 = (b'b' as u32) << 16 | (b'c' as u32) << 8 | b'd' as u32;
    let cde: u32 = (b'c' as u32) << 16 | (b'd' as u32) << 8 | b'e' as u32;

    let mut raw: Vec<(u32, FileLocation)> = Vec::with_capacity(15_000);
    // "abc": files 0–4 999
    for fid in 0u32..5_000 {
        raw.push((abc, FileLocation::new(fid, 1, 0)));
    }
    // "bcd": files 2 500–7 499  (overlaps abc in files 2 500–4 999)
    for fid in 2_500u32..7_500 {
        raw.push((bcd, FileLocation::new(fid, 1, 1)));
    }
    // "cde": files 0–4 999
    for fid in 0u32..5_000 {
        raw.push((cde, FileLocation::new(fid, 1, 2)));
    }

    let mut idx = TrigramIndex::new();
    idx.build_from_trigrams(raw);

    let mut g = c.benchmark_group("posting_list_intersection");
    g.bench_function("3gram_10k_files", |b| {
        // "abcde" → 3 trigrams; search returns the pre-computed intersection
        b.iter(|| idx.search(black_box("abcde")))
    });
    g.finish();
}

// ─── 3. Full index + query roundtrip ──────────────────────────────────────────
//
// Creates 1 000 synthetic Rust files on disk, then on each benchmark iteration:
//   (a) deletes the .reflex cache so the index is always built from scratch
//   (b) indexes all 1 000 files
//   (c) runs a full-text query that matches every file
//
// Uses BatchSize::PerIteration so setup (cache deletion) runs every iteration.
// sample_size is capped at 10 because full indexing is expensive I/O-bound work.

fn bench_index_and_query_roundtrip(c: &mut Criterion) {
    let temp = TempDir::new().unwrap();
    let project = temp.path().to_path_buf();

    for i in 0u32..1_000 {
        fs::write(
            project.join(format!("file_{i}.rs")),
            format!(
                "fn func_{i}(x: i32) -> i32 {{\n\
                 \tlet y = x * {i};\n\
                 \tprintln!(\"{{y}}\");\n\
                 \ty\n\
                 }}\n"
            ),
        )
        .unwrap();
    }

    let mut g = c.benchmark_group("index_and_query_roundtrip");
    g.sample_size(10);

    g.bench_function("1k_files", |b| {
        b.iter_batched(
            || {
                let cache_dir = project.join(".reflex");
                if cache_dir.exists() {
                    fs::remove_dir_all(&cache_dir).unwrap();
                }
            },
            |_| {
                let cache = CacheManager::new(&project);
                let indexer = Indexer::new(cache, IndexConfig::default());
                indexer.index(&project, false).unwrap();

                let cache = CacheManager::new(&project);
                let engine = QueryEngine::new(cache);
                let filter = QueryFilter {
                    limit: None,
                    ..Default::default()
                };
                let results = engine.search(black_box("func_"), filter).unwrap();
                black_box(results)
            },
            BatchSize::PerIteration,
        )
    });

    g.finish();
}

// ─── 4. Symbol query through tree-sitter ──────────────────────────────────────
//
// Simulates the symbol-query path where the trigram filter returns 100 candidate
// files, each of which must be parsed by tree-sitter to extract symbol definitions.
//
// Lens: Tree-sitter query performance — parsing is 100–1000× slower than trigram
// search; this benchmark quantifies the per-file cost to inform caching decisions.

fn bench_symbol_query_tree_sitter(c: &mut Criterion) {
    let sources: Vec<String> = (0u32..100)
        .map(|i| {
            format!(
                "fn func_{i}(x: i32) -> i32 {{\n\
                 \tx * {i}\n\
                 }}\n\n\
                 struct S_{i} {{\n\
                 \tv: i32,\n\
                 }}\n\n\
                 impl S_{i} {{\n\
                 \tfn new(v: i32) -> Self {{\n\
                 \t\tS_{i} {{ v }}\n\
                 \t}}\n\
                 }}\n"
            )
        })
        .collect();
    let paths: Vec<String> = (0u32..100).map(|i| format!("file_{i}.rs")).collect();

    let mut g = c.benchmark_group("symbol_query_tree_sitter");
    g.bench_function("100_candidates_rust", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for (path, src) in paths.iter().zip(sources.iter()) {
                let syms = ParserFactory::parse(path, black_box(src), Language::Rust).unwrap();
                total += syms.len();
            }
            black_box(total)
        })
    });
    g.finish();
}

// ─── registry ─────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_trigram_extraction,
    bench_posting_list_intersection,
    bench_index_and_query_roundtrip,
    bench_symbol_query_tree_sitter,
);
criterion_main!(benches);
