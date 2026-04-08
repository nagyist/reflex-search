//! Static site generator
//!
//! Orchestrates wiki, digest, and map into a Zola project.
//! Generates markdown content with TOML front matter, Tera templates,
//! and a Zola config. Optionally runs `zola build` to produce HTML.

use anyhow::{Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};

use crate::cache::CacheManager;
use crate::semantic::providers::LlmProvider;
use super::digest;
use super::diff;
use super::map::{self, MapFormat, MapZoom};
use super::narrate;
use super::snapshot;
use super::wiki;
use super::zola;

/// Truncate a string to at most `max_chars` Unicode characters, appending "..." if truncated.
fn truncate_str(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{}...", truncated)
    } else {
        truncated
    }
}

/// Site generation configuration
#[derive(Debug, Clone)]
pub struct SiteConfig {
    pub output_dir: PathBuf,
    pub base_url: String,
    pub title: String,
    pub surfaces: Vec<Surface>,
    pub no_llm: bool,
    pub clean: bool,
    pub force_renarrate: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Surface {
    Wiki,
    Digest,
    Map,
}

impl Default for SiteConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("pulse-site"),
            base_url: "/".to_string(),
            title: "Pulse".to_string(),
            surfaces: vec![Surface::Wiki, Surface::Digest, Surface::Map],
            no_llm: true,
            clean: false,
            force_renarrate: false,
        }
    }
}

/// Report from site generation
#[derive(Debug, Clone, Serialize)]
pub struct SiteReport {
    pub output_dir: String,
    pub pages_generated: usize,
    pub digest_generated: bool,
    pub map_generated: bool,
    pub narration_mode: String,
    pub build_success: bool,
}

/// Generate the complete Zola project and optionally build it
pub fn generate_site(cache: &CacheManager, config: &SiteConfig) -> Result<SiteReport> {
    // Clean output dir if requested
    if config.clean && config.output_dir.exists() {
        std::fs::remove_dir_all(&config.output_dir)
            .context("Failed to clean output directory")?;
    }

    // Create Zola project structure
    create_directory_structure(&config.output_dir)?;

    // Write Zola config
    write_zola_config(&config.output_dir, &config.base_url, &config.title)?;

    // Write templates
    write_templates(&config.output_dir)?;

    // Write static assets
    write_static_assets(&config.output_dir)?;

    // Get snapshots for diff
    let snapshots = snapshot::list_snapshots(cache)?;
    let current_snapshot = snapshots.first();
    let baseline_snapshot = snapshots.get(1);

    let snapshot_diff = match (current_snapshot, baseline_snapshot) {
        (Some(current), Some(baseline)) => {
            let pulse_config = super::config::load_pulse_config(cache.path())?;
            diff::compute_diff(&baseline.path, &current.path, &pulse_config.thresholds).ok()
        }
        _ => None,
    };

    // Clear LLM cache if force-renarrate is set
    if config.force_renarrate && !config.no_llm {
        let llm_cache = super::llm_cache::LlmCache::new(cache.path());
        if let Err(e) = llm_cache.clear() {
            log::warn!("Failed to clear LLM cache: {}", e);
        }
    }

    // Create LLM provider once for all surfaces
    let provider: Option<Box<dyn LlmProvider>> = if !config.no_llm {
        match narrate::create_pulse_provider() {
            Ok(p) => {
                eprintln!("LLM provider ready, narration enabled.");
                Some(p)
            }
            Err(e) => {
                eprintln!("LLM narration unavailable: {}", e);
                None
            }
        }
    } else {
        None
    };

    let llm_cache = provider.as_ref().map(|_| super::llm_cache::LlmCache::new(cache.path()));

    let mut pages_generated = 0;
    let mut digest_generated = false;
    let mut map_generated = false;
    let mut has_narration = false;

    // Collect wiki pages for the home page index
    let mut wiki_page_index: Vec<WikiPageMeta> = Vec::new();

    // Generate wiki pages
    if config.surfaces.contains(&Surface::Wiki) {
        let snapshot_id = snapshots.first().map(|s| s.id.as_str()).unwrap_or("unknown");
        let wiki_pages = wiki::generate_all_pages(
            cache,
            snapshot_diff.as_ref(),
            config.no_llm,
            snapshot_id,
            provider.as_ref().map(|p| p.as_ref()),
            llm_cache.as_ref(),
        )?;

        if wiki_pages.iter().any(|p| p.sections.summary.is_some()) {
            has_narration = true;
        }

        // Write wiki section index
        write_wiki_section_index(&config.output_dir)?;

        // Detect modules for metadata
        let modules = wiki::detect_modules(cache)?;
        let module_map: std::collections::HashMap<&str, &wiki::ModuleDefinition> = modules.iter()
            .map(|m| (m.path.as_str(), m))
            .collect();

        for (i, page) in wiki_pages.iter().enumerate() {
            let module = module_map.get(page.module_path.as_str());
            let slug = page.module_path.replace('/', "-");

            let summary_preview = page.sections.summary.as_deref()
                .map(|s| s.chars().take(200).collect::<String>())
                .unwrap_or_else(|| format!("{} files", module.map(|m| m.file_count).unwrap_or(0)));

            wiki_page_index.push(WikiPageMeta {
                title: page.title.clone(),
                slug: slug.clone(),
                file_count: module.map(|m| m.file_count).unwrap_or(0),
                total_lines: module.map(|m| m.total_lines).unwrap_or(0),
                description: summary_preview,
            });

            write_wiki_page(
                &config.output_dir,
                page,
                module,
                i + 1,
            )?;
            pages_generated += 1;
        }
    }

    // Generate digest
    if config.surfaces.contains(&Surface::Digest) {
        if let Some(current) = current_snapshot {
            let digest_data = digest::generate_digest(
                snapshot_diff.as_ref(),
                current,
                Some(cache),
                config.no_llm,
                provider.as_ref().map(|p| p.as_ref()),
                llm_cache.as_ref(),
            )?;

            if digest_data.sections.iter().any(|s| s.narrative.is_some()) {
                has_narration = true;
            }

            let digest_md = digest::render_markdown(&digest_data);
            write_digest_page(&config.output_dir, &digest_md, &digest_data)?;
            digest_generated = true;
        }
    }

    // Generate map
    if config.surfaces.contains(&Surface::Map) {
        let map_content = map::generate_map(cache, &MapZoom::Repo, MapFormat::Mermaid)?;
        write_map_page(&config.output_dir, &map_content)?;
        map_generated = true;
    }

    // Write home page
    write_home_page(
        &config.output_dir,
        &config.title,
        &wiki_page_index,
        digest_generated,
        map_generated,
    )?;

    // Compute narration mode
    let narration_mode = if config.no_llm {
        "disabled".to_string()
    } else if has_narration {
        "narrated".to_string()
    } else {
        "structural".to_string()
    };

    // Try to build with Zola
    let build_success = try_zola_build(&config.output_dir);

    Ok(SiteReport {
        output_dir: config.output_dir.display().to_string(),
        pages_generated,
        digest_generated,
        map_generated,
        narration_mode,
        build_success,
    })
}

// ── Directory structure ──────────────────────────────────────

fn create_directory_structure(output_dir: &Path) -> Result<()> {
    let dirs = [
        "",
        "content",
        "content/wiki",
        "content/digest",
        "content/map",
        "templates",
        "templates/shortcodes",
        "static",
        "sass",
    ];

    for dir in &dirs {
        std::fs::create_dir_all(output_dir.join(dir))
            .with_context(|| format!("Failed to create directory: {}", dir))?;
    }

    Ok(())
}

// ── Zola config ──────────────────────────────────────────────

fn write_zola_config(output_dir: &Path, base_url: &str, title: &str) -> Result<()> {
    let config = format!(
r#"# Zola configuration — generated by rfx pulse generate
base_url = "{base_url}"
title = "{title}"
description = "Auto-generated codebase documentation"
compile_sass = false
build_search_index = false
generate_feeds = false
minify_html = false

[markdown]
highlight_code = true
highlight_theme = "base16-ocean-dark"
render_emoji = false
external_links_target_blank = true
smart_punctuation = true

[extra]
generated_by = "Reflex Pulse"
"#);

    std::fs::write(output_dir.join("config.toml"), config)
        .context("Failed to write Zola config.toml")
}

// ── Templates ────────────────────────────────────────────────

fn write_templates(output_dir: &Path) -> Result<()> {
    // Base template
    let base_html = r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{% block title %}{{ config.title }}{% endblock title %}</title>
    <link rel="stylesheet" href="{{ get_url(path='style.css') }}">
</head>
<body>
    <div class="layout">
        <nav class="sidebar">
            <div class="sidebar-header">
                <a href="{{ get_url(path='/') }}"><h2>{{ config.title }}</h2></a>
            </div>
            <ul class="nav-list">
                <li><a href="{{ get_url(path='/') }}">Home</a></li>
                {% set wiki_section = get_section(path="wiki/_index.md", metadata_only=true) %}
                <li class="nav-section">
                    <a href="{{ get_url(path='wiki') }}">Wiki</a>
                    <ul>
                        {% for page in wiki_section.pages %}
                        <li><a href="{{ page.permalink }}">{{ page.title }}</a></li>
                        {% endfor %}
                    </ul>
                </li>
                <li><a href="{{ get_url(path='digest') }}">Digest</a></li>
                <li><a href="{{ get_url(path='map') }}">Map</a></li>
            </ul>
        </nav>
        <main class="content">
            {% block content %}{% endblock content %}
        </main>
    </div>
    {% block scripts %}{% endblock scripts %}
</body>
</html>"##;

    // Index (home) template
    let index_html = r#"{% extends "base.html" %}
{% block title %}{{ config.title }}{% endblock title %}
{% block content %}
{{ section.content | safe }}
{% endblock content %}"#;

    // Section template (wiki/, digest/, map/)
    let section_html = r#"{% extends "base.html" %}
{% block title %}{{ section.title }} — {{ config.title }}{% endblock title %}
{% block content %}
<h1>{{ section.title }}</h1>
{{ section.content | safe }}
{% if section.pages %}
<div class="page-list">
    {% for page in section.pages %}
    <div class="page-card">
        <h3><a href="{{ page.permalink }}">{{ page.title }}</a></h3>
        {% if page.description %}
        <p>{{ page.description }}</p>
        {% endif %}
    </div>
    {% endfor %}
</div>
{% endif %}
{% endblock content %}"#;

    // Page template (individual wiki modules)
    let page_html = r#"{% extends "base.html" %}
{% block title %}{{ page.title }} — {{ config.title }}{% endblock title %}
{% block content %}
<h1>{{ page.title }}</h1>
{% if page.extra.tier %}
<div class="page-meta">
    <span class="badge tier-{{ page.extra.tier }}">Tier {{ page.extra.tier }}</span>
    {% if page.extra.file_count %}
    <span class="badge">{{ page.extra.file_count }} files</span>
    {% endif %}
    {% if page.extra.languages %}
    <span class="badge">{{ page.extra.languages }}</span>
    {% endif %}
</div>
{% endif %}
{{ page.content | safe }}
{% endblock content %}
{% block scripts %}
{% if page.extra.has_mermaid is defined %}
<script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({ startOnLoad: true });
</script>
{% endif %}
{% endblock scripts %}"#;

    // Mermaid shortcode
    let mermaid_shortcode = r#"<pre class="mermaid">
{{ body }}
</pre>"#;

    std::fs::write(output_dir.join("templates/base.html"), base_html)?;
    std::fs::write(output_dir.join("templates/index.html"), index_html)?;
    std::fs::write(output_dir.join("templates/section.html"), section_html)?;
    std::fs::write(output_dir.join("templates/page.html"), page_html)?;
    std::fs::write(output_dir.join("templates/shortcodes/mermaid.html"), mermaid_shortcode)?;

    Ok(())
}

// ── Static assets ────────────────────────────────────────────

fn write_static_assets(output_dir: &Path) -> Result<()> {
    let css = r#":root {
    --bg: #1a1b26;
    --bg-surface: #24283b;
    --bg-hover: #292e42;
    --fg: #c0caf5;
    --fg-muted: #565f89;
    --fg-accent: #7aa2f7;
    --fg-green: #9ece6a;
    --fg-yellow: #e0af68;
    --fg-red: #f7768e;
    --border: #3b4261;
    --sidebar-width: 260px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--fg);
    line-height: 1.6;
}

.layout {
    display: flex;
    min-height: 100vh;
}

.sidebar {
    width: var(--sidebar-width);
    background: var(--bg-surface);
    border-right: 1px solid var(--border);
    padding: 1.5rem 0;
    position: fixed;
    height: 100vh;
    overflow-y: auto;
}

.sidebar-header {
    padding: 0 1.25rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.5rem;
}

.sidebar-header h2 {
    font-size: 1.1rem;
    color: var(--fg-accent);
}

.sidebar a {
    color: var(--fg);
    text-decoration: none;
}

.sidebar a:hover {
    color: var(--fg-accent);
}

.nav-list {
    list-style: none;
    padding: 0;
}

.nav-list > li {
    padding: 0.35rem 1.25rem;
}

.nav-list > li > a {
    font-weight: 500;
    font-size: 0.95rem;
}

.nav-section ul {
    list-style: none;
    padding-left: 0.75rem;
    margin-top: 0.25rem;
}

.nav-section ul li {
    padding: 0.15rem 0;
}

.nav-section ul li a {
    font-size: 0.85rem;
    color: var(--fg-muted);
}

.nav-section ul li a:hover {
    color: var(--fg-accent);
}

.content {
    margin-left: var(--sidebar-width);
    padding: 2rem 3rem;
    max-width: 900px;
    flex: 1;
}

h1 { font-size: 1.8rem; margin-bottom: 1rem; color: var(--fg); }
h2 { font-size: 1.4rem; margin: 1.5rem 0 0.75rem; color: var(--fg-accent); border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }
h3 { font-size: 1.15rem; margin: 1.2rem 0 0.5rem; color: var(--fg); }

p { margin-bottom: 0.75rem; }

a { color: var(--fg-accent); text-decoration: none; }
a:hover { text-decoration: underline; }

code {
    background: var(--bg-surface);
    padding: 0.15em 0.4em;
    border-radius: 3px;
    font-size: 0.9em;
    font-family: "JetBrains Mono", "Fira Code", monospace;
}

pre {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem;
    overflow-x: auto;
    margin: 0.75rem 0;
}

pre code {
    background: none;
    padding: 0;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.75rem 0;
}

th, td {
    text-align: left;
    padding: 0.5rem 0.75rem;
    border: 1px solid var(--border);
}

th {
    background: var(--bg-surface);
    font-weight: 600;
    color: var(--fg-accent);
}

tr:hover { background: var(--bg-hover); }

ul, ol { padding-left: 1.5rem; margin-bottom: 0.75rem; }
li { margin-bottom: 0.25rem; }

.page-meta {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
}

.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 500;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    color: var(--fg-muted);
}

.tier-1 { color: var(--fg-accent); border-color: var(--fg-accent); }
.tier-2 { color: var(--fg-green); border-color: var(--fg-green); }

.page-card {
    padding: 1rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 0.75rem;
    background: var(--bg-surface);
}

.page-card:hover { background: var(--bg-hover); }
.page-card h3 { margin: 0 0 0.25rem; font-size: 1rem; }
.page-card p { margin: 0; font-size: 0.9rem; color: var(--fg-muted); }

.mermaid {
    background: var(--bg-surface);
    padding: 1rem;
    border-radius: 6px;
    text-align: center;
}

.module-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 0.75rem;
    margin: 1rem 0;
}

@media (max-width: 768px) {
    .sidebar { display: none; }
    .content { margin-left: 0; padding: 1rem; }
}
"#;

    std::fs::write(output_dir.join("static/style.css"), css)
        .context("Failed to write style.css")
}

// ── Content generation ───────────────────────────────────────

struct WikiPageMeta {
    title: String,
    slug: String,
    file_count: usize,
    total_lines: usize,
    description: String,
}

fn write_home_page(
    output_dir: &Path,
    title: &str,
    wiki_pages: &[WikiPageMeta],
    has_digest: bool,
    has_map: bool,
) -> Result<()> {
    let mut content = String::new();

    content.push_str("+++\n");
    content.push_str(&format!("title = \"{}\"\n", title));
    content.push_str("sort_by = \"weight\"\n");
    content.push_str("+++\n\n");

    content.push_str(&format!("# {}\n\n", title));
    content.push_str("Auto-generated codebase documentation powered by [Reflex](https://github.com/reflex-search/reflex).\n\n");

    // Module index
    if !wiki_pages.is_empty() {
        content.push_str("## Modules\n\n");
        content.push_str("| Module | Files | Lines | Description |\n|---|---|---|---|\n");
        for page in wiki_pages {
            let desc = truncate_str(&page.description, 77);
            content.push_str(&format!(
                "| [{}](@/wiki/{}.md) | {} | {} | {} |\n",
                page.title, page.slug, page.file_count, page.total_lines, desc
            ));
        }
        content.push('\n');
    }

    // Navigation
    content.push_str("## Sections\n\n");
    content.push_str("- [Wiki](/wiki/) — Per-module documentation\n");
    if has_digest {
        content.push_str("- [Digest](/digest/) — Structural change report\n");
    }
    if has_map {
        content.push_str("- [Map](/map/) — Architecture dependency graph\n");
    }

    std::fs::write(output_dir.join("content/_index.md"), content)
        .context("Failed to write home page")
}

fn write_wiki_section_index(output_dir: &Path) -> Result<()> {
    let content = r#"+++
title = "Wiki"
sort_by = "weight"
template = "section.html"
+++

Per-module documentation pages. Each page covers a detected module's structure,
dependencies, key symbols, and metrics.
"#;

    std::fs::write(output_dir.join("content/wiki/_index.md"), content)
        .context("Failed to write wiki section index")
}

fn write_wiki_page(
    output_dir: &Path,
    page: &wiki::WikiPage,
    module: Option<&&wiki::ModuleDefinition>,
    weight: usize,
) -> Result<()> {
    let slug = page.module_path.replace('/', "-");
    let mut content = String::new();

    // TOML front matter
    content.push_str("+++\n");
    content.push_str(&format!("title = \"{}\"\n", page.title));
    content.push_str(&format!("weight = {}\n", weight));
    if let Some(summary) = &page.sections.summary {
        let desc = truncate_str(summary, 200)
            .replace('\\', "\\\\")
            .replace('"', "'")
            .replace('\n', " ");
        content.push_str(&format!("description = \"{}\"\n", desc));
    }

    content.push_str("\n[extra]\n");
    if let Some(m) = module {
        content.push_str(&format!("tier = {}\n", m.tier));
        content.push_str(&format!("file_count = {}\n", m.file_count));
        content.push_str(&format!("total_lines = {}\n", m.total_lines));
        content.push_str(&format!("languages = \"{}\"\n", m.languages.join(", ")));
    }
    content.push_str("+++\n\n");

    // Page content
    if let Some(summary) = &page.sections.summary {
        content.push_str(summary);
        content.push_str("\n\n");
    }

    content.push_str("## Structure\n\n");
    content.push_str(&page.sections.structure);
    content.push_str("\n\n");

    content.push_str("## Dependencies\n\n");
    content.push_str(&page.sections.dependencies);
    content.push_str("\n\n");

    content.push_str("## Dependents\n\n");
    content.push_str(&page.sections.dependents);
    content.push_str("\n\n");

    content.push_str("## Key Symbols\n\n");
    content.push_str(&page.sections.key_symbols);
    content.push_str("\n\n");

    content.push_str("## Metrics\n\n");
    content.push_str(&page.sections.metrics);
    content.push_str("\n\n");

    if let Some(changes) = &page.sections.recent_changes {
        content.push_str("## Recent Changes\n\n");
        content.push_str(changes);
        content.push_str("\n\n");
    }

    let filename = format!("{}.md", slug);
    std::fs::write(output_dir.join("content/wiki").join(&filename), content)
        .with_context(|| format!("Failed to write wiki page: {}", filename))
}

fn write_digest_page(
    output_dir: &Path,
    digest_md: &str,
    digest_data: &digest::Digest,
) -> Result<()> {
    // Section index
    let mut index_content = String::new();
    index_content.push_str("+++\n");
    index_content.push_str(&format!("title = \"{}\"\n", digest_data.title));
    index_content.push_str("template = \"section.html\"\n");
    index_content.push_str("+++\n\n");
    index_content.push_str(digest_md);

    std::fs::write(output_dir.join("content/digest/_index.md"), index_content)
        .context("Failed to write digest page")
}

fn write_map_page(output_dir: &Path, mermaid_content: &str) -> Result<()> {
    let mut content = String::new();
    content.push_str("+++\n");
    content.push_str("title = \"Architecture Map\"\n");
    content.push_str("template = \"section.html\"\n");
    content.push_str("\n[extra]\n");
    content.push_str("has_mermaid = true\n");
    content.push_str("+++\n\n");

    content.push_str("Module-level dependency graph showing how code modules relate to each other.\n\n");
    content.push_str("{% mermaid() %}\n");
    content.push_str(mermaid_content);
    content.push_str("{% end %}\n");

    // Also include mermaid JS loading in the map section template
    // We need a dedicated map section template for the mermaid script
    std::fs::write(output_dir.join("content/map/_index.md"), content)
        .context("Failed to write map page")
}

// ── Zola build ───────────────────────────────────────────────

fn try_zola_build(output_dir: &Path) -> bool {
    match zola::ensure_zola() {
        Ok(zola_path) => {
            eprintln!("Building site with Zola...");
            let public_dir = output_dir.join("public");

            let result = std::process::Command::new(&zola_path)
                .current_dir(output_dir)
                .arg("build")
                .arg("--output-dir")
                .arg(&public_dir)
                .output();

            match result {
                Ok(output) if output.status.success() => {
                    // Count HTML files in public/
                    let html_count = count_html_files(&public_dir);
                    eprintln!("Site built at {}/ ({} pages)", public_dir.display(), html_count);
                    true
                }
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    eprintln!("Zola build failed:\n{}", stderr);
                    eprintln!("The Zola project was generated at {}/ — you can build manually with:", output_dir.display());
                    eprintln!("  cd {} && zola build", output_dir.display());
                    false
                }
                Err(e) => {
                    eprintln!("Failed to run Zola: {}", e);
                    false
                }
            }
        }
        Err(e) => {
            eprintln!("Could not download Zola: {}", e);
            eprintln!("The Zola project was generated at {}/ — install Zola and run:", output_dir.display());
            eprintln!("  cd {} && zola build", output_dir.display());
            eprintln!("Install Zola: https://www.getzola.org/documentation/getting-started/installation/");
            false
        }
    }
}

fn count_html_files(dir: &Path) -> usize {
    if !dir.exists() {
        return 0;
    }
    walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|ext| ext == "html").unwrap_or(false))
        .count()
}
