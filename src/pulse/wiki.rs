//! Wiki generation: per-module documentation pages
//!
//! Generates a living wiki page for each detected module (directory) in the codebase.
//! Pages include structural sections (dependencies, dependents, key symbols, metrics)
//! and optional LLM-generated summaries.

use anyhow::{Context, Result};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::cache::CacheManager;
use crate::dependency::DependencyIndex;
use crate::models::{Language, SymbolKind};
use crate::parsers::ParserFactory;
use crate::semantic::context::CodebaseContext;
use crate::semantic::providers::LlmProvider;

use super::llm_cache::LlmCache;
use super::narrate;

/// A detected module in the codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDefinition {
    /// Module path (e.g., "src", "tests", "src/parsers")
    pub path: String,
    /// Module tier: 1 = top-level, 2 = depth-2/3
    pub tier: u8,
    /// Number of files in this module
    pub file_count: usize,
    /// Total line count
    pub total_lines: usize,
    /// Languages present in this module
    pub languages: Vec<String>,
}

/// A generated wiki page for a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WikiPage {
    pub module_path: String,
    pub title: String,
    pub sections: WikiSections,
}

/// Structural sections of a wiki page (all built without LLM)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WikiSections {
    pub summary: Option<String>,
    pub structure: String,
    pub dependencies: String,
    pub dependents: String,
    pub key_symbols: String,
    pub metrics: String,
    pub recent_changes: Option<String>,
}

/// Detect modules in the codebase using CodebaseContext
pub fn detect_modules(cache: &CacheManager) -> Result<Vec<ModuleDefinition>> {
    let context = CodebaseContext::extract(cache)
        .context("Failed to extract codebase context")?;

    let db_path = cache.path().join("meta.db");
    let conn = Connection::open(&db_path)?;

    let mut modules = Vec::new();

    // Tier 1: top-level directories
    for dir in &context.top_level_dirs {
        let dir_path = dir.trim_end_matches('/');
        if let Some(module) = build_module_def(&conn, dir_path, 1)? {
            modules.push(module);
        }
    }

    // Tier 2: common paths (depth 2-3)
    for path in &context.common_paths {
        let path_str = path.trim_end_matches('/');
        // Skip if already covered by a Tier 1 module
        let already_covered = modules.iter().any(|m| path_str.starts_with(&m.path));
        if !already_covered {
            if let Some(module) = build_module_def(&conn, path_str, 2)? {
                modules.push(module);
            }
        }
    }

    // Sort by path for deterministic output
    modules.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(modules)
}

/// Generate a wiki page for a single module
pub fn generate_wiki_page(
    cache: &CacheManager,
    module: &ModuleDefinition,
    diff: Option<&super::diff::SnapshotDiff>,
    no_llm: bool,
    provider: Option<&dyn LlmProvider>,
    llm_cache: Option<&LlmCache>,
    snapshot_id: &str,
) -> Result<WikiPage> {
    let db_path = cache.path().join("meta.db");
    let conn = Connection::open(&db_path)?;
    let deps_index = DependencyIndex::new(cache.clone());

    // Build structural sections
    let structure = build_structure_section(&conn, &module.path)?;
    let dependencies = build_dependencies_section(&conn, &module.path)?;
    let dependents = build_dependents_section(&conn, &deps_index, &module.path)?;
    let key_symbols = build_key_symbols_section(&conn, &module.path);
    let metrics = build_metrics_section(module);
    let recent_changes = diff.map(|d| build_recent_changes(d, &module.path));

    // Generate LLM summary when provider is available
    let summary = if !no_llm {
        if let (Some(provider), Some(llm_cache)) = (provider, llm_cache) {
            // Build combined structural context for the summary
            let mut context = String::new();
            context.push_str(&format!("Module: {}\n\n", module.path));
            context.push_str(&format!("## Structure\n{}\n\n", structure));
            context.push_str(&format!("## Dependencies\n{}\n\n", dependencies));
            context.push_str(&format!("## Dependents\n{}\n\n", dependents));
            context.push_str(&format!("## Key Symbols\n{}\n\n", key_symbols));
            context.push_str(&format!("## Metrics\n{}\n", metrics));

            narrate::narrate_section(
                provider,
                narrate::wiki_system_prompt(),
                &context,
                llm_cache,
                snapshot_id,
                &module.path,
            )
        } else {
            None
        }
    } else {
        None
    };

    Ok(WikiPage {
        module_path: module.path.clone(),
        title: format!("{}/", module.path),
        sections: WikiSections {
            summary,
            structure,
            dependencies,
            dependents,
            key_symbols,
            metrics,
            recent_changes,
        },
    })
}

/// Generate wiki pages for all detected modules
///
/// `provider` and `llm_cache` are created by the caller (site.rs or CLI handler).
pub fn generate_all_pages(
    cache: &CacheManager,
    diff: Option<&super::diff::SnapshotDiff>,
    no_llm: bool,
    snapshot_id: &str,
    provider: Option<&dyn LlmProvider>,
    llm_cache: Option<&LlmCache>,
) -> Result<Vec<WikiPage>> {
    let modules = detect_modules(cache)?;
    let mut pages = Vec::new();

    if provider.is_some() {
        eprintln!("Generating wiki summaries...");
    }

    for module in &modules {
        match generate_wiki_page(
            cache,
            module,
            diff,
            no_llm,
            provider,
            llm_cache,
            snapshot_id,
        ) {
            Ok(page) => pages.push(page),
            Err(e) => {
                log::warn!("Failed to generate wiki page for {}: {}", module.path, e);
            }
        }
    }

    Ok(pages)
}

/// Render wiki pages as (filename, markdown) pairs
pub fn render_wiki_markdown(pages: &[WikiPage]) -> Vec<(String, String)> {
    pages.iter().map(|page| {
        let filename = page.module_path.replace('/', "_") + ".md";
        let mut md = String::new();

        md.push_str(&format!("# {}\n\n", page.title));

        if let Some(summary) = &page.sections.summary {
            md.push_str(summary);
            md.push_str("\n\n");
        }

        md.push_str("## Structure\n\n");
        md.push_str(&page.sections.structure);
        md.push_str("\n\n");

        md.push_str("## Dependencies\n\n");
        md.push_str(&page.sections.dependencies);
        md.push_str("\n\n");

        md.push_str("## Dependents\n\n");
        md.push_str(&page.sections.dependents);
        md.push_str("\n\n");

        md.push_str("## Key Symbols\n\n");
        md.push_str(&page.sections.key_symbols);
        md.push_str("\n\n");

        md.push_str("## Metrics\n\n");
        md.push_str(&page.sections.metrics);
        md.push_str("\n\n");

        if let Some(changes) = &page.sections.recent_changes {
            md.push_str("## Recent Changes\n\n");
            md.push_str(changes);
            md.push_str("\n\n");
        }

        (filename, md)
    }).collect()
}

// --- Private helpers ---

fn build_module_def(conn: &Connection, path: &str, tier: u8) -> Result<Option<ModuleDefinition>> {
    let pattern = format!("{}/%", path);

    let file_count: usize = conn.query_row(
        "SELECT COUNT(*) FROM files WHERE path LIKE ?1 OR path LIKE ?2",
        rusqlite::params![format!("{}/%", path), format!("{}", path)],
        |row| row.get(0),
    )?;

    if file_count == 0 {
        return Ok(None);
    }

    let total_lines: usize = conn.query_row(
        "SELECT COALESCE(SUM(line_count), 0) FROM files WHERE path LIKE ?1",
        [&pattern],
        |row| row.get(0),
    )?;

    let mut stmt = conn.prepare(
        "SELECT DISTINCT language FROM files WHERE path LIKE ?1 AND language IS NOT NULL"
    )?;
    let languages: Vec<String> = stmt.query_map([&pattern], |row| row.get(0))?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Some(ModuleDefinition {
        path: path.to_string(),
        tier,
        file_count,
        total_lines,
        languages,
    }))
}

fn build_structure_section(conn: &Connection, module_path: &str) -> Result<String> {
    let pattern = format!("{}/%", module_path);
    let mut stmt = conn.prepare(
        "SELECT path, language, COALESCE(line_count, 0) FROM files
         WHERE path LIKE ?1
         ORDER BY path
         LIMIT 50"
    )?;

    let files: Vec<(String, Option<String>, i64)> = stmt.query_map([&pattern], |row| {
        Ok((row.get(0)?, row.get(1)?, row.get(2)?))
    })?.collect::<Result<Vec<_>, _>>()?;

    let mut content = String::new();

    // Group by language
    let mut by_lang: HashMap<String, usize> = HashMap::new();
    for (_, lang, _) in &files {
        let lang = lang.as_deref().unwrap_or("other");
        *by_lang.entry(lang.to_string()).or_insert(0) += 1;
    }

    content.push_str("| Language | Files |\n|---|---|\n");
    let mut lang_counts: Vec<_> = by_lang.into_iter().collect();
    lang_counts.sort_by(|a, b| b.1.cmp(&a.1));
    for (lang, count) in &lang_counts {
        content.push_str(&format!("| {} | {} |\n", lang, count));
    }

    content.push_str("\n**Files:**\n");
    for (path, _, lines) in files.iter().take(30) {
        content.push_str(&format!("- `{}` ({} lines)\n", path, lines));
    }
    let total: usize = conn.query_row(
        "SELECT COUNT(*) FROM files WHERE path LIKE ?1",
        [&pattern], |row| row.get(0)
    )?;
    if total > 30 {
        content.push_str(&format!("- ... and {} more files\n", total - 30));
    }

    Ok(content)
}

fn build_dependencies_section(conn: &Connection, module_path: &str) -> Result<String> {
    let pattern = format!("{}/%", module_path);
    let mut stmt = conn.prepare(
        "SELECT DISTINCT f2.path
         FROM file_dependencies fd
         JOIN files f1 ON fd.file_id = f1.id
         JOIN files f2 ON fd.resolved_file_id = f2.id
         WHERE f1.path LIKE ?1 AND f2.path NOT LIKE ?1
         ORDER BY f2.path
         LIMIT 30"
    )?;

    let deps: Vec<String> = stmt.query_map([&pattern], |row| row.get(0))?
        .collect::<Result<Vec<_>, _>>()?;

    if deps.is_empty() {
        return Ok("No outgoing dependencies detected.".to_string());
    }

    let mut content = String::new();
    for dep in &deps {
        content.push_str(&format!("- `{}`\n", dep));
    }
    Ok(content)
}

fn build_dependents_section(conn: &Connection, _deps_index: &DependencyIndex, module_path: &str) -> Result<String> {
    let pattern = format!("{}/%", module_path);
    let mut stmt = conn.prepare(
        "SELECT DISTINCT f1.path
         FROM file_dependencies fd
         JOIN files f1 ON fd.file_id = f1.id
         JOIN files f2 ON fd.resolved_file_id = f2.id
         WHERE f2.path LIKE ?1 AND f1.path NOT LIKE ?1
         ORDER BY f1.path
         LIMIT 30"
    )?;

    let dependents: Vec<String> = stmt.query_map([&pattern], |row| row.get(0))?
        .collect::<Result<Vec<_>, _>>()?;

    if dependents.is_empty() {
        return Ok("No incoming dependencies detected.".to_string());
    }

    let mut content = String::new();
    for dep in &dependents {
        content.push_str(&format!("- `{}`\n", dep));
    }
    Ok(content)
}

fn build_key_symbols_section(conn: &Connection, module_path: &str) -> String {
    let pattern = format!("{}/%", module_path);
    let mut stmt = match conn.prepare(
        "SELECT path, language FROM files
         WHERE path LIKE ?1 AND language IS NOT NULL
         ORDER BY COALESCE(line_count, 0) DESC
         LIMIT 20"
    ) {
        Ok(s) => s,
        Err(_) => return "No symbols extracted.".to_string(),
    };

    let files: Vec<(String, String)> = match stmt.query_map([&pattern], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }) {
        Ok(rows) => rows.filter_map(|r| r.ok()).collect(),
        Err(_) => return "No symbols extracted.".to_string(),
    };

    if files.is_empty() {
        return "No files in this module.".to_string();
    }

    // Parse each file and collect symbols
    let mut by_kind: HashMap<String, Vec<(String, String, usize)>> = HashMap::new(); // kind -> [(name, path, size)]
    let mut total_symbols = 0usize;

    for (path, lang_str) in &files {
        let language = match Language::from_name(lang_str) {
            Some(l) => l,
            None => continue,
        };

        // Read source from disk
        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let symbols = match ParserFactory::parse(path, &source, language) {
            Ok(s) => s,
            Err(_) => continue,
        };

        for sym in symbols {
            if let Some(name) = &sym.symbol {
                // Skip imports, exports, and unknown kinds
                match &sym.kind {
                    SymbolKind::Import | SymbolKind::Export | SymbolKind::Unknown(_) => continue,
                    _ => {}
                }

                let kind_name = format!("{}", sym.kind);
                let size = sym.span.end_line.saturating_sub(sym.span.start_line) + 1;
                by_kind
                    .entry(kind_name)
                    .or_default()
                    .push((name.clone(), path.clone(), size));
                total_symbols += 1;
            }
        }
    }

    if total_symbols == 0 {
        return "No symbols extracted.".to_string();
    }

    // Sort each kind by size (descending) and take top 10
    let display_order = [
        "Function", "Struct", "Class", "Trait", "Interface",
        "Enum", "Method", "Constant", "Type", "Macro",
        "Variable", "Module", "Namespace", "Property", "Attribute",
    ];

    let mut content = String::new();
    let mut symbols_shown = 0;
    let max_total = 50;

    for kind in &display_order {
        if symbols_shown >= max_total {
            break;
        }
        let kind_str = kind.to_string();
        if let Some(entries) = by_kind.get_mut(&kind_str) {
            entries.sort_by(|a, b| b.2.cmp(&a.2));
            let count = entries.len();
            let take = 10.min(max_total - symbols_shown);
            content.push_str(&format!("**{}** ({}):\n", kind, count));
            for (name, path, _) in entries.iter().take(take) {
                content.push_str(&format!("- `{}` ({})\n", name, path));
                symbols_shown += 1;
            }
            content.push('\n');
        }
    }

    // Handle any kinds not in display_order
    for (kind, entries) in &mut by_kind {
        if symbols_shown >= max_total {
            break;
        }
        if display_order.contains(&kind.as_str()) {
            continue;
        }
        entries.sort_by(|a, b| b.2.cmp(&a.2));
        let count = entries.len();
        let take = 10.min(max_total - symbols_shown);
        content.push_str(&format!("**{}** ({}):\n", kind, count));
        for (name, path, _) in entries.iter().take(take) {
            content.push_str(&format!("- `{}` ({})\n", name, path));
            symbols_shown += 1;
        }
        content.push('\n');
    }

    if content.is_empty() {
        "No symbols extracted.".to_string()
    } else {
        content
    }
}

fn build_metrics_section(module: &ModuleDefinition) -> String {
    format!(
        "| Metric | Value |\n|---|---|\n\
         | Files | {} |\n\
         | Total lines | {} |\n\
         | Languages | {} |\n\
         | Tier | {} |",
        module.file_count,
        module.total_lines,
        module.languages.join(", "),
        module.tier,
    )
}

fn build_recent_changes(diff: &super::diff::SnapshotDiff, module_path: &str) -> String {
    let prefix = format!("{}/", module_path);
    let mut content = String::new();

    let added: Vec<_> = diff.files_added.iter()
        .filter(|f| f.path.starts_with(&prefix))
        .collect();
    let removed: Vec<_> = diff.files_removed.iter()
        .filter(|f| f.path.starts_with(&prefix))
        .collect();
    let modified: Vec<_> = diff.files_modified.iter()
        .filter(|f| f.path.starts_with(&prefix))
        .collect();

    if added.is_empty() && removed.is_empty() && modified.is_empty() {
        return "No changes in this module since last snapshot.".to_string();
    }

    if !added.is_empty() {
        content.push_str(&format!("**Added** ({}):\n", added.len()));
        for f in added.iter().take(10) {
            content.push_str(&format!("- `{}`\n", f.path));
        }
    }
    if !removed.is_empty() {
        content.push_str(&format!("**Removed** ({}):\n", removed.len()));
        for f in removed.iter().take(10) {
            content.push_str(&format!("- `{}`\n", f.path));
        }
    }
    if !modified.is_empty() {
        content.push_str(&format!("**Modified** ({}):\n", modified.len()));
        for f in modified.iter().take(10) {
            let delta = f.new_line_count as i64 - f.old_line_count as i64;
            content.push_str(&format!("- `{}` ({:+} lines)\n", f.path, delta));
        }
    }

    content
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_definition_serialization() {
        let module = ModuleDefinition {
            path: "src".to_string(),
            tier: 1,
            file_count: 50,
            total_lines: 5000,
            languages: vec!["Rust".to_string()],
        };
        let json = serde_json::to_string(&module).unwrap();
        assert!(json.contains("src"));
    }

    #[test]
    fn test_render_wiki_page() {
        let page = WikiPage {
            module_path: "src".to_string(),
            title: "src/".to_string(),
            sections: WikiSections {
                summary: None,
                structure: "test structure".to_string(),
                dependencies: "test deps".to_string(),
                dependents: "test dependents".to_string(),
                key_symbols: "test symbols".to_string(),
                metrics: "test metrics".to_string(),
                recent_changes: None,
            },
        };
        let rendered = render_wiki_markdown(&[page]);
        assert_eq!(rendered.len(), 1);
        assert_eq!(rendered[0].0, "src.md");
        assert!(rendered[0].1.contains("# src/"));
    }
}
