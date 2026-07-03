//! Result assembly utilities: trigram index reconstruction and file-id resolution

use anyhow::{Context, Result};

use crate::content_store::ContentReader;
use crate::trigram::TrigramIndex;

/// Find a file_id by its path string in the content store.
pub fn find_file_id(content_reader: &ContentReader, target_path: &str) -> Option<u32> {
    for file_id in 0..content_reader.file_count() {
        if let Some(path) = content_reader.get_file_path(file_id as u32)
            && path.to_string_lossy() == target_path
        {
            return Some(file_id as u32);
        }
    }
    None
}

/// Rebuild a trigram index from content store (fallback when trigrams.bin is missing).
pub fn rebuild_trigram_index(content_reader: &ContentReader) -> Result<TrigramIndex> {
    log::debug!(
        "Rebuilding trigram index from {} files",
        content_reader.file_count()
    );
    let mut trigram_index = TrigramIndex::new();

    for file_id in 0..content_reader.file_count() {
        let file_path = content_reader
            .get_file_path(file_id as u32)
            .context("Invalid file_id")?
            .to_path_buf();
        let content = content_reader.get_file_content(file_id as u32)?;

        let idx = trigram_index.add_file(file_path);
        trigram_index.index_file(idx, content);
    }

    trigram_index.finalize();
    log::debug!(
        "Trigram index rebuilt with {} trigrams",
        trigram_index.trigram_count()
    );

    Ok(trigram_index)
}

/// Normalize a glob pattern so it matches the way indexed paths are stored.
///
/// Indexed paths are stored **relative and without a `./` prefix** (e.g.
/// `src/parsers/rust.rs`) — see the `strip_prefix("./")` normalization applied
/// throughout `query::mod`. A relative glob such as `src/**` must therefore be
/// anchored so it can match those bare paths. We prepend `**/` (not `./`):
/// `./src/**` fails to match `src/parsers/rust.rs` because of the leading `./`,
/// whereas `**/src/**` matches it. This mirrors the convention the integration
/// tests already rely on and is forgiving of LLM-authored patterns that omit a
/// leading `**/` (REF-191).
///
/// Examples:
/// - "src/**/*.rs" → "**/src/**/*.rs"
/// - "main.rs"     → "**/main.rs"
/// - "./services/**/*.php" → unchanged (already anchored)
/// - "/abs/path"   → unchanged (absolute)
/// - "**/foo"      → unchanged (already prefixed)
pub fn normalize_glob_pattern(pattern: &str) -> String {
    if pattern.starts_with('.') || pattern.starts_with('/') || pattern.starts_with('*') {
        pattern.to_string()
    } else {
        format!("**/{}", pattern)
    }
}

#[cfg(test)]
mod normalize_glob_tests {
    use super::normalize_glob_pattern;
    use globset::Glob;

    fn matches(pattern: &str, path: &str) -> bool {
        let normalized = normalize_glob_pattern(pattern);
        Glob::new(&normalized)
            .unwrap()
            .compile_matcher()
            .is_match(path)
    }

    #[test]
    fn relative_patterns_get_recursive_prefix() {
        assert_eq!(normalize_glob_pattern("src/**/*.rs"), "**/src/**/*.rs");
        assert_eq!(normalize_glob_pattern("main.rs"), "**/main.rs");
        assert_eq!(normalize_glob_pattern("src/**"), "**/src/**");
    }

    #[test]
    fn anchored_and_prefixed_patterns_are_unchanged() {
        assert_eq!(
            normalize_glob_pattern("./services/**/*.php"),
            "./services/**/*.php"
        );
        assert_eq!(normalize_glob_pattern("/abs/path/*.rs"), "/abs/path/*.rs");
        assert_eq!(normalize_glob_pattern("**/foo"), "**/foo");
        assert_eq!(normalize_glob_pattern("*.rs"), "*.rs");
    }

    /// REF-191 regression: the natural `src/**` an LLM writes must match
    /// bare stored paths like `src/parsers/rust.rs`. The old `./`-prefix
    /// normalization returned zero matches for this, causing agents to
    /// distrust Reflex and fall back to Grep on find-all tasks.
    #[test]
    fn src_glob_matches_bare_stored_paths() {
        assert!(matches("src/**", "src/parsers/rust.rs"));
        assert!(matches("src/**", "src/mcp.rs"));
        assert!(matches("src/**/*.rs", "src/parsers/rust.rs"));
        assert!(matches("src/**/*.rs", "src/mcp.rs"));
    }

    #[test]
    fn src_glob_does_not_match_unrelated_paths() {
        // Component boundary: `src` must be a whole path component.
        assert!(!matches("src/**", "src_helpers/foo.rs"));
        assert!(!matches("src/**/*.rs", "benches/foo.rs"));
    }

    #[test]
    fn bare_filename_matches_at_any_depth() {
        assert!(matches("main.rs", "src/main.rs"));
        assert!(matches("main.rs", "main.rs"));
    }
}
