//! LLM narration helpers for Pulse
//!
//! Provides centralized LLM calling for digest and wiki surfaces.
//! Handles provider setup, caching, content gating, and async bridging.

use anyhow::Result;
use std::path::Path;

use crate::semantic::config;
use crate::semantic::providers::{self, LlmProvider};

use super::llm_cache::LlmCache;

/// System prompt for digest section narration
const DIGEST_SYSTEM_PROMPT: &str = "\
You are a technical writer narrating a codebase change report.
You may ONLY describe facts present in the STRUCTURAL CONTEXT below.
Do NOT speculate, infer intent, or add information not in the context.
Write 2-4 concise sentences summarizing the key points.

STRUCTURAL CONTEXT:
";

/// System prompt for wiki module summary
const WIKI_SYSTEM_PROMPT: &str = "\
You are a technical writer creating a module overview for a codebase wiki.
You may ONLY describe facts present in the STRUCTURAL CONTEXT below.
Do NOT speculate about design intent or add information not in the context.
Write a 3-5 sentence summary of this module's purpose and structure.

STRUCTURAL CONTEXT:
";

/// Minimum word count to attempt narration.
/// Sections below this threshold are too brief to produce useful summaries.
const MIN_CONTENT_WORDS: usize = 15;

/// Create an LLM provider using the user's ~/.reflex/config.toml (same config as `rfx ask`)
pub fn create_pulse_provider() -> Result<Box<dyn LlmProvider>> {
    let semantic_config = config::load_config(Path::new("."))?;
    let api_key = config::get_api_key(&semantic_config.provider)?;

    let model = if semantic_config.model.is_some() {
        semantic_config.model.clone()
    } else {
        config::get_user_model(&semantic_config.provider)
    };

    let options = config::get_provider_options(&semantic_config.provider);

    providers::create_provider(&semantic_config.provider, api_key, model, options)
}

/// Narrate a structural context block using LLM.
///
/// Returns `None` if:
/// - Content is too brief (fewer than MIN_CONTENT_WORDS words)
/// - LLM call fails (degrades gracefully, logs warning)
/// - Cache hit returns previously generated narration
///
/// Checks `LlmCache` first; stores response on success.
pub fn narrate_section(
    provider: &dyn LlmProvider,
    system_prompt: &str,
    structural_context: &str,
    cache: &LlmCache,
    snapshot_id: &str,
    cache_key_suffix: &str,
) -> Option<String> {
    // Check minimum content length
    let word_count = structural_context.split_whitespace().count();
    if word_count < MIN_CONTENT_WORDS {
        eprintln!("  Skipping: {} (too brief, {} words)", cache_key_suffix, word_count);
        return None;
    }

    // Check cache
    let cache_key = LlmCache::compute_key(snapshot_id, cache_key_suffix, structural_context);
    match cache.get(&cache_key) {
        Ok(Some(cached)) => {
            log::debug!("LLM cache hit for '{}'", cache_key_suffix);
            eprintln!("  Narrating: {} (cached)", cache_key_suffix);
            return Some(cached.response);
        }
        Ok(None) => {}
        Err(e) => {
            log::warn!("Failed to read LLM cache: {}", e);
        }
    }

    // Build prompt
    let prompt = format!("{}{}", system_prompt, structural_context);

    eprintln!("  Narrating: {}...", cache_key_suffix);

    // Call LLM with retry (sync bridge over async)
    let result = call_llm_sync(provider, &prompt);

    match result {
        Ok(response) => {
            let response = response.trim().to_string();

            // Cache the response
            let context_hash = blake3::hash(structural_context.as_bytes()).to_hex().to_string();
            if let Err(e) = cache.put(&cache_key, &context_hash, &response) {
                log::warn!("Failed to write LLM cache: {}", e);
            }

            Some(response)
        }
        Err(e) => {
            log::warn!("LLM narration failed for '{}': {}", cache_key_suffix, e);
            None
        }
    }
}

/// Get the system prompt for digest narration
pub fn digest_system_prompt() -> &'static str {
    DIGEST_SYSTEM_PROMPT
}

/// Get the system prompt for wiki narration
pub fn wiki_system_prompt() -> &'static str {
    WIKI_SYSTEM_PROMPT
}

/// Synchronous LLM call with retry logic.
/// Uses tokio runtime to bridge async provider calls.
fn call_llm_sync(provider: &dyn LlmProvider, prompt: &str) -> Result<String> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let mut last_error = None;
        let max_retries = 2;

        for attempt in 0..=max_retries {
            if attempt > 0 {
                log::debug!("Retrying LLM narration (attempt {}/{})", attempt + 1, max_retries + 1);
                tokio::time::sleep(tokio::time::Duration::from_millis(500 * attempt as u64)).await;
            }

            match provider.complete(prompt, false).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    log::debug!("LLM call attempt {} failed: {}", attempt + 1, e);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("LLM call failed")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_count_sufficient() {
        // 15+ words should pass the gate
        let text = "src/parsers/rust.rs has 250 lines and contains extract_symbols fn_name and other important functions used for parsing code";
        let count = text.split_whitespace().count();
        assert!(count >= MIN_CONTENT_WORDS, "Word count {} should be >= {}", count, MIN_CONTENT_WORDS);
    }

    #[test]
    fn test_word_count_too_brief() {
        // < 15 words should be rejected
        let text = "No data available yet.";
        let count = text.split_whitespace().count();
        assert!(count < MIN_CONTENT_WORDS, "Word count {} should be < {}", count, MIN_CONTENT_WORDS);
    }

    #[test]
    fn test_word_count_empty() {
        let count = "".split_whitespace().count();
        assert!(count < MIN_CONTENT_WORDS);
    }

    #[test]
    fn test_word_count_wiki_structural() {
        // Typical wiki page with markdown table + file list should pass
        let text = "| Language | Files | Lines |\n| --- | --- | --- |\n| Rust | 45 | 12,500 |\n\n**Files:** src/main.rs src/lib.rs src/query/mod.rs src/parsers/rust.rs";
        let count = text.split_whitespace().count();
        assert!(count >= MIN_CONTENT_WORDS, "Wiki structural word count {} should be >= {}", count, MIN_CONTENT_WORDS);
    }

    #[test]
    fn test_word_count_digest_bootstrap() {
        // Typical digest with structural data should pass
        let text = "Branch: feature/pulse Commit: abc1234 Files: 120 Edges: 340 Modules: src tests build.rs config.toml main.rs lib.rs";
        let count = text.split_whitespace().count();
        assert!(count >= MIN_CONTENT_WORDS, "Digest bootstrap word count {} should be >= {}", count, MIN_CONTENT_WORDS);
    }

    #[test]
    fn test_digest_system_prompt() {
        assert!(digest_system_prompt().contains("STRUCTURAL CONTEXT"));
    }

    #[test]
    fn test_wiki_system_prompt() {
        assert!(wiki_system_prompt().contains("STRUCTURAL CONTEXT"));
    }
}
