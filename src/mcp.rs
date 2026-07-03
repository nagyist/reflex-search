//! MCP (Model Context Protocol) server implementation
//!
//! This module implements the MCP protocol directly over stdio using JSON-RPC 2.0.
//! It exposes Reflex's code search capabilities as MCP tools for AI coding assistants.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use crate::cache::CacheManager;
use crate::dependency::DependencyIndex;
use crate::indexer::Indexer;
use crate::line_filter;
use crate::models::{IndexConfig, IndexStatus, Language, SymbolKind};
use crate::query::{QueryEngine, QueryFilter};
use crate::semantic::config::load_mcp_config;

/// Default preview truncation length for MCP responses (characters).
/// Raised from 100 so typical Rust/TS/Python function signatures fit without truncation.
const DEFAULT_MCP_PREVIEW_LENGTH: usize = 180;

/// Default page size for MCP list results when the caller does not specify a
/// `limit`. Raised from 50 → 200 (REF-191).
///
/// Rationale: the efficacy benchmark's residual gap was **turn count** — Reflex
/// reached parity only when it answered in the same number of turns as `grep`.
/// `grep -rn` returns every occurrence in one shot; a 50-result default forces
/// an agent doing a find-all task (e.g. 122 occurrences of `extract_symbols`)
/// to paginate, spending an extra MCP call for the same answer. 200 covers the
/// overwhelming majority of find-all result sets in a single "decisive" call
/// while still bounding token cost (hard cap remains 500 via the `min(500)`
/// clamp on explicit limits). Callers who want a cheap cardinality probe should
/// use `mode="count"`.
const DEFAULT_MCP_RESULT_LIMIT: usize = 200;

/// Returns true if every occurrence of `pattern` in `preview` falls inside a
/// string literal or comment for the given `lang`. Conservative: returns false
/// (keep the match) when the language has no filter or the pattern is not found.
fn is_in_string_or_comment(lang: Language, preview: &str, pattern: &str) -> bool {
    let Some(filter) = line_filter::get_filter(lang) else {
        return false;
    };

    let mut pos = 0;
    let mut found = false;

    while pos < preview.len() {
        let Some(rel) = preview[pos..].find(pattern) else {
            break;
        };
        let abs = pos + rel;
        found = true;
        if !filter.is_in_comment(preview, abs) && !filter.is_in_string(preview, abs) {
            return false; // at least one occurrence is in real code — keep the match
        }
        pos = abs + 1;
    }

    found
}

/// JSON-RPC 2.0 request
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

/// JSON-RPC 2.0 response
#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    // Per JSON-RPC 2.0, Notifications must omit `id` entirely (not set it to null).
    // We never construct a JsonRpcResponse for a Notification, but skip_serializing_if
    // is a defensive guard against ever emitting `"id": null`, which strict clients
    // (e.g. Claude Code's Zod validators) reject.
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error
#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

/// Parse language string to Language enum
fn parse_language(lang: Option<String>) -> Option<Language> {
    lang.as_deref().and_then(Language::from_name)
}

/// Parse symbol kind string to SymbolKind enum
fn parse_symbol_kind(kind: Option<String>) -> Option<SymbolKind> {
    kind.as_deref().and_then(|s| {
        let capitalized = {
            let mut chars = s.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first
                    .to_uppercase()
                    .chain(chars.flat_map(|c| c.to_lowercase()))
                    .collect(),
            }
        };

        capitalized
            .parse::<SymbolKind>()
            .ok()
            .or_else(|| Some(SymbolKind::Unknown(s.to_string())))
    })
}

/// Handle initialize request
fn handle_initialize(_params: Option<Value>) -> Result<Value> {
    Ok(json!({
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "reflex",
            "version": env!("CARGO_PKG_VERSION")
        },
        "instructions": "Reflex MCP tools are pre-loaded — do NOT call ToolSearch. Available tools (all prefixed mcp__reflex__): check_index_status · search_code · search_regex · list_locations · count_occurrences · find_references · gather_context · index_project · get_dependencies · get_dependents · find_hotspots · search_ast. Structural tools (if enabled): find_circular · find_islands · find_unused · analyze_summary · get_transitive_deps. Prefer these over Grep and Glob for code search. If you see 'Index not found', call mcp__reflex__index_project first, then retry."
    }))
}

// ---------------------------------------------------------------------------
// outputSchema builders (REF-203 / REF-196 Phase 3)
//
// Each tool's `outputSchema` declares the JSON Schema of the `structuredContent`
// object produced by `make_tool_result` in the corresponding `handle_call_tool`
// arm. Declaring it lets MCP clients validate responses and gives the LLM
// type awareness of the shape it will receive.
//
// These are composed from small reusable pieces so the declared schema tracks
// the real serialization of `QueryResponse` (see `src/models.rs`). A drift test
// (`tests/mcp_output_schema.rs`) validates a live `search_code` response against
// `schema_search_output()` to catch divergence.
//
// The schemas are intentionally permissive on additional properties
// (forward-compatible: new optional fields must not break existing clients) but
// strict on `required` — the fields that are ALWAYS present in that response.

/// `{ start_line, end_line }` — the serialized shape of `models::Span`.
fn schema_span() -> Value {
    json!({
        "type": "object",
        "properties": {
            "start_line": { "type": "integer" },
            "end_line": { "type": "integer" }
        },
        "required": ["start_line", "end_line"]
    })
}

/// One match within a file — serialized `models::MatchResult`.
/// `kind` and `symbol` are omitted for plain text matches, so only
/// `span` + `preview` are guaranteed.
fn schema_match() -> Value {
    json!({
        "type": "object",
        "properties": {
            "kind": { "type": "string", "description": "Symbol kind; present only for symbol matches" },
            "symbol": { "type": "string" },
            "span": schema_span(),
            "preview": { "type": "string" },
            "context_before": { "type": "array", "items": { "type": "string" } },
            "context_after": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["span", "preview"]
    })
}

/// A file-grouped result — serialized `models::FileGroupedResult`.
fn schema_file_result() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": { "type": "string" },
            "language": { "type": "string", "description": "Detected language, lowercased (e.g. \"rust\"); \"unknown\" for unrecognized files" },
            "dependencies": { "type": "array", "items": { "type": "object" } },
            "matches": { "type": "array", "items": schema_match() }
        },
        "required": ["path", "language", "matches"]
    })
}

/// Pagination metadata — serialized `models::PaginationInfo`.
/// `limit` defaults to `DEFAULT_MCP_RESULT_LIMIT` (200) for list-mode MCP calls.
fn schema_pagination() -> Value {
    json!({
        "type": "object",
        "properties": {
            "total": { "type": "integer", "description": "Total matches before offset/limit" },
            "count": { "type": "integer", "description": "Matches returned in this page" },
            "offset": { "type": "integer" },
            "limit": { "type": "integer", "default": 200, "description": "Max results per page (MCP default 200)" },
            "has_more": { "type": "boolean" }
        },
        "required": ["total", "count", "offset", "has_more"]
    })
}

/// Reduced `count`-mode payload — `{ count, pattern }`. Emitted by
/// `search_code`, `search_regex`, and `find_references` when `mode="count"`.
fn schema_count_result() -> Value {
    json!({
        "type": "object",
        "properties": {
            "count": { "type": "integer" },
            "pattern": { "type": "string" }
        },
        "required": ["count", "pattern"]
    })
}

/// List-mode result for `search_code` / `search_regex`: serialized
/// `QueryResponse` plus the flattened `has_more`/`total_count`/`returned_count`
/// scalars that `handle_call_tool` inserts.
fn schema_search_list() -> Value {
    json!({
        "type": "object",
        "properties": {
            "status": { "type": "string", "enum": ["fresh", "stale"] },
            "can_trust_results": { "type": "boolean" },
            "warning": { "type": "object" },
            "ai_instruction": { "type": "string" },
            "pagination": schema_pagination(),
            "results": { "type": "array", "items": schema_file_result() },
            "total_count": { "type": "integer" },
            "returned_count": { "type": "integer" },
            "has_more": { "type": "boolean" }
        },
        "required": ["status", "pagination", "results", "total_count", "returned_count", "has_more"]
    })
}

/// Columnar list-mode result for `search_code` / `search_regex` (REF-209).
///
/// The file-grouped `results` array is replaced by a `{ columns, rows }` pair:
/// each match becomes a positional row aligned to `columns`, so match-field keys
/// appear once instead of once per match (~41% payload reduction on large
/// results — see REF-196 analysis). Top-level metadata (`status`, `pagination`,
/// the flattened `has_more`/`total_count`/`returned_count` scalars) is unchanged.
fn schema_search_columnar() -> Value {
    json!({
        "type": "object",
        "properties": {
            "status": { "type": "string", "enum": ["fresh", "stale"] },
            "can_trust_results": { "type": "boolean" },
            "warning": { "type": "object" },
            "ai_instruction": { "type": "string" },
            "pagination": schema_pagination(),
            "columns": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Column names for each row. Always begins with path, language, start_line, end_line, preview; kind/symbol/context_before/context_after/dependencies are appended only when at least one match carries them."
            },
            "rows": {
                "type": "array",
                "items": { "type": "array" },
                "description": "One array per match; element i corresponds to columns[i]. path/language repeat per row so each row is self-contained."
            },
            "total_count": { "type": "integer" },
            "returned_count": { "type": "integer" },
            "has_more": { "type": "boolean" }
        },
        "required": ["status", "pagination", "columns", "rows", "total_count", "returned_count", "has_more"]
    })
}

/// `outputSchema` for `search_code` and `search_regex`. `list` mode returns the
/// full result set; `count` mode returns `{ count, pattern }`.
///
/// The list-mode branch tracks the runtime `REFLEX_MCP_COLUMNAR` toggle (REF-209)
/// so the declared schema always matches what the server actually emits: the
/// columnar `{ columns, rows }` shape by default, or the legacy file-grouped
/// `results` array when columnar output is disabled.
fn schema_search_output() -> Value {
    let list = if columnar_enabled() {
        schema_search_columnar()
    } else {
        schema_search_list()
    };
    json!({
        // MCP requires `outputSchema` to be an object-type JSON Schema (the
        // `structuredContent` payload is always a JSON object). Both `oneOf`
        // branches are themselves `type: object`, so declaring the root as
        // `object` is consistent and is REQUIRED: Claude Code's client validates
        // `outputSchema.type === "object"` and rejects the *entire* tools/list
        // batch (dropping all tools, server still "connected") if any tool's
        // root schema is a bare `oneOf` without it (REF-210).
        "type": "object",
        "description": "list mode (default) returns full results in columnar {columns, rows} form (REF-209); count mode returns {count, pattern}. Set REFLEX_MCP_COLUMNAR=0 to restore the legacy results[] shape.",
        "oneOf": [list, schema_count_result()]
    })
}

/// `outputSchema` for `find_references`. `list` mode returns the definition plus
/// flattened references; `count` mode returns `{ count, pattern }`.
fn schema_find_references_output() -> Value {
    let list = json!({
        "type": "object",
        "properties": {
            "status": { "type": "string", "enum": ["fresh", "stale"] },
            "definition": {
                "type": ["object", "null"],
                "description": "First matching symbol definition, or null if none exists",
                "properties": {
                    "path": { "type": "string" },
                    "kind": { "type": "string" },
                    "symbol": { "type": "string" },
                    "span": schema_span(),
                    "preview": { "type": "string" }
                }
            },
            "references": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "line": { "type": "integer" },
                        "preview": { "type": "string" }
                    },
                    "required": ["path", "line", "preview"]
                }
            },
            "total_references": { "type": "integer" },
            "total_count": { "type": "integer" },
            "returned_count": { "type": "integer" },
            "has_more": { "type": "boolean" },
            "pagination": schema_pagination()
        },
        "required": ["status", "definition", "references", "total_references", "total_count", "returned_count", "has_more", "pagination"]
    });
    json!({
        // Root must be `type: object` — see the note in `schema_search_output`.
        // A bare top-level `oneOf` here fails Claude Code's outputSchema
        // validation and silently drops every Reflex tool (REF-210).
        "type": "object",
        "description": "list mode (default) returns definition + references; count mode returns {count, pattern}",
        "oneOf": [list, schema_count_result()]
    })
}

/// `outputSchema` for `gather_context`. Returns human-readable prose wrapped as
/// `{ context }` (this arm does not go through `make_tool_result`).
fn schema_gather_context_output() -> Value {
    json!({
        "type": "object",
        "properties": {
            "context": { "type": "string", "description": "Human-readable project orientation summary" }
        },
        "required": ["context"]
    })
}

/// `outputSchema` for `list_locations`: `{ status, total_locations, locations[] }`.
fn schema_list_locations_output() -> Value {
    json!({
        "type": "object",
        "properties": {
            "status": { "type": "string", "enum": ["fresh", "stale"] },
            "total_locations": { "type": "integer" },
            "locations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "line": { "type": "integer" }
                    },
                    "required": ["path", "line"]
                }
            }
        },
        "required": ["status", "total_locations", "locations"]
    })
}

/// `outputSchema` for `count_occurrences`: `{ status, pattern, total, files }`.
fn schema_count_occurrences_output() -> Value {
    json!({
        "type": "object",
        "properties": {
            "status": { "type": "string", "enum": ["fresh", "stale"] },
            "pattern": { "type": "string" },
            "total": { "type": "integer", "description": "Total occurrences across all files" },
            "files": { "type": "integer", "description": "Number of distinct files containing the pattern" }
        },
        "required": ["status", "pattern", "total", "files"]
    })
}

/// Handle tools/list request
///
/// When `enable_structural` is false, the five structural-analysis-only tools
/// (`find_circular`, `find_islands`, `find_unused`, `analyze_summary`,
/// `get_transitive_deps`) are omitted from the response. Controlled by the
/// `[mcp] enable_structural_tools` flag in `~/.reflex/config.toml`.
fn handle_list_tools(_params: Option<Value>, enable_structural: bool) -> Result<Value> {
    // Names of tools gated behind enable_structural_tools config flag.
    // These are structural-analysis tools rarely needed for day-to-day code search.
    const STRUCTURAL_TOOLS: &[&str] = &[
        "find_circular",
        "find_islands",
        "find_unused",
        "analyze_summary",
        "get_transitive_deps",
    ];

    let all_tools = json!({
        "tools": [
            {
                "name": "list_locations",
                "description": "Fast location discovery with minimal token usage.\n\n**Purpose:** Find where a pattern occurs (file + line) without loading previews or detailed context.\n\n**Returns:** Array of {path, line} objects - one per match location.\n\n**Use this when:**\n- Starting exploration (\"where is X used?\")\n- Counting affected locations\n- Building a list for targeted Read operations\n- You need locations only, not code content\n\n**Workflow:**\n1. Use list_locations to discover (cheap, returns locations only)\n2. Use Read tool or search_code on specific files if you need content (targeted)\n\n**Supports:** lang, file, glob, exclude filters\n**No limit:** Returns ALL matching locations\n\n**Error Handling:** If you receive an error message containing \"Index not found\" or \"stale\", immediately call the index_project tool, wait for it to complete, then retry this operation.\n\n**Example:** Pattern \"CourtCase\" → [{\"path\": \"app/Models/CourtCase.php\", \"line\": 15}, {\"path\": \"app/Http/Controllers/CourtController.php\", \"line\": 42}]",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (text to find)"
                        },
                        "lang": {
                            "type": "string",
                            "description": "Filter by language (php, rust, typescript, python, etc.)"
                        },
                        "file": {
                            "type": "string",
                            "description": "Filter by file path substring (e.g., 'Controllers')"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching patterns (e.g., ['app/**/*.php'])"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching patterns (e.g., ['vendor/**', 'tests/**'])"
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        },
                        "dependencies": {
                            "type": "boolean",
                            "description": "Include dependency information (imports) in results. Only extracts static imports."
                        }
                    },
                    "required": ["pattern"]
                },
                "outputSchema": schema_list_locations_output()
            },
            {
                "name": "count_occurrences",
                "description": "Quick statistics - count how many times a pattern occurs.\n\n**Purpose:** Get total occurrence count and file count without loading any content.\n\n**Use this when:**\n- You need quick stats (\"how many times is X used?\")\n- Checking impact before refactoring\n- Validating search scope\n\n**Returns:** {total: count, files: count, pattern: string}\n\n**Supports:** All filters (lang, file, glob, exclude, symbols)\n\n**Error Handling:** If you receive an error message containing \"Index not found\" or \"stale\", immediately call the index_project tool, wait for it to complete, then retry this operation.\n\n**Example output:** {\"total\": 87, \"files\": 12, \"pattern\": \"CourtCase\"}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (text to find)"
                        },
                        "lang": {
                            "type": "string",
                            "description": "Filter by language"
                        },
                        "symbols": {
                            "type": "boolean",
                            "description": "Count symbol definitions only (not usages)"
                        },
                        "kind": {
                            "type": "string",
                            "description": "Filter by symbol kind (function, class, etc.)"
                        },
                        "file": {
                            "type": "string",
                            "description": "Filter by file path substring"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching patterns"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching patterns"
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        },
                        "dependencies": {
                            "type": "boolean",
                            "description": "Include dependency information (imports) in results. Only extracts static imports."
                        }
                    },
                    "required": ["pattern"]
                },
                "outputSchema": schema_count_occurrences_output()
            },
            {
                "name": "search_code",
                "description": "Comprehensive full-codebase search — equivalent to `grep -rn`. Returns ALL occurrences across every indexed file in a single call. Do NOT chain multiple search_code calls for the same pattern — one call is already exhaustive. Use mode=\"count\" to get just the match count before deciding whether to paginate.\n\n**Search modes:**\n- Full-text (default): Finds ALL occurrences — definitions + usages\n- Symbol-only (symbols=true): Finds ONLY definitions where symbols are declared\n\n**When to use search_regex instead:**\n- Patterns with special characters: -> :: () [] {} . * + ? \\\\ | ^ $\n- Complex pattern matching: wildcards, alternation, anchors\n- Examples: '->with(', '::new', 'function*', '[derive]', 'fn (get|set)_.*'\n\n**Use this for:**\n- Simple text patterns (alphanumeric, underscores, hyphens)\n- Detailed analysis with line numbers and code previews\n- Symbol definition searches\n\n**Count mode:** Pass mode=\"count\" to get {\"count\": N, \"pattern\": \"...\"} with no match bodies. Faster than list mode — use this to check cardinality before paginating.\n\n**Result shape (list mode):** Results are columnar to save tokens: {\"columns\": [...], \"rows\": [[...]]}. Each row is one match with values positionally aligned to `columns` (always path, language, start_line, end_line, preview; then kind/symbol/context when present). Read `columns` to map positions. Set env REFLEX_MCP_COLUMNAR=0 to restore the legacy results[] object shape.\n\n**Pagination:** Check response.pagination.has_more. If true, use offset parameter to fetch next page.\n\n**Error Handling:** If you receive an error message containing \"Index not found\" or \"stale\", immediately call the index_project tool, wait for it to complete, then retry this operation.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (text to find)"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["list", "count"],
                            "description": "Response mode: \"list\" (default) returns full match results; \"count\" returns only {count, pattern} — faster, skips match body serialization."
                        },
                        "lang": {
                            "type": "string",
                            "description": "Filter by language (rust, typescript, python, etc.)"
                        },
                        "kind": {
                            "type": "string",
                            "description": "Filter by symbol kind (function, class, struct, etc.)"
                        },
                        "symbols": {
                            "type": "boolean",
                            "description": "Symbol-only search (definitions, not usage)"
                        },
                        "exact": {
                            "type": "boolean",
                            "description": "Exact match (no substring matching)"
                        },
                        "file": {
                            "type": "string",
                            "description": "Filter by file path (substring)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results per page (default: 200, max: 500). The 200-result default covers most find-all tasks in a single call. IMPORTANT: If response.has_more is true, you MUST fetch more pages using offset parameter."
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N results). ALWAYS paginate when has_more=true. Example: First call offset=0, second call offset=100, third offset=200, etc."
                        },
                        "expand": {
                            "type": "boolean",
                            "description": "Show full symbol body (not just signature)"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching glob patterns (e.g., 'src/**/*.rs')"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching glob patterns (e.g., 'target/**')"
                        },
                        "paths": {
                            "type": "boolean",
                            "description": "Return only unique file paths (not full results)"
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        },
                        "dependencies": {
                            "type": "boolean",
                            "description": "Include dependency information (imports) in results. **IMPORTANT:** Currently only supported for Rust files — passing this with any other language (typescript, python, go, etc.) will produce no dependency data. Only extracts static imports (string literals); dynamic imports are filtered. See CLAUDE.md for details."
                        },
                        "preview_length": {
                            "type": "integer",
                            "description": "Maximum characters per preview line (default: 180). Use a smaller value (e.g. 60) for wide-result scans where short previews are sufficient."
                        }
                    },
                    "required": ["pattern"]
                },
                "outputSchema": schema_search_output()
            },
            {
                "name": "search_regex",
                "description": "Regex-based code search for complex pattern matching (e.g., 'fn (get|set)_\\\\w+').\n\n**Use for:**\n- Patterns with special characters: -> :: () [] {} . * + ? \\\\ | ^ $\n- Pattern matching: wildcards (.*), alternation (a|b), anchors (^$)\n- Complex searches: case-insensitive variants, word boundaries\n\n**Common examples:**\n- Method calls: '->with\\\\(', '->map\\\\(', '::new\\\\('\n- Operators: '->', '::', '||', '&&'\n- Functions: 'fn (get|set)_\\\\\\\\w+' (getter/setter functions)\n- Attributes: '\\\\\\\\[(derive|test)\\\\\\\\]' (Rust attributes)\n\n**Escaping rules:**\n- Must escape: ( ) [ ] { } . * + ? \\\\ | ^ $\n- No escaping needed: -> :: - _ / = < >\n- Use double backslash in JSON: \\\\\\\\( \\\\\\\\) \\\\\\\\[ \\\\\\\\]\n\n**Count mode:** Pass mode=\"count\" to get {\"count\": N, \"pattern\": \"...\"} with no match bodies — faster than list mode.\n\n**Result shape (list mode):** Results are columnar to save tokens: {\"columns\": [...], \"rows\": [[...]]}. Each row is one match with values positionally aligned to `columns` (always path, language, start_line, end_line, preview; then kind/symbol/context when present). Read `columns` to map positions. Set env REFLEX_MCP_COLUMNAR=0 to restore the legacy results[] object shape.\n\n**Error Handling:** If you receive an error message containing \"Index not found\" or \"stale\", immediately call the index_project tool, wait for it to complete, then retry this operation.\n\n**Don't use for:**\n- Simple text searches (use search_code instead - faster)\n- Symbol definitions (use search_code with symbols=true instead)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["list", "count"],
                            "description": "Response mode: \"list\" (default) returns full match results; \"count\" returns only {count, pattern} — faster, skips match body serialization."
                        },
                        "lang": {
                            "type": "string",
                            "description": "Filter by language"
                        },
                        "file": {
                            "type": "string",
                            "description": "Filter by file path"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 200, max: 500). Use with offset for pagination."
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N results after sorting)"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching glob patterns"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching glob patterns"
                        },
                        "paths": {
                            "type": "boolean",
                            "description": "Return only unique file paths"
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        },
                        "dependencies": {
                            "type": "boolean",
                            "description": "Include dependency information (imports) in results. Only extracts static imports."
                        }
                    },
                    "required": ["pattern"]
                },
                "outputSchema": schema_search_output()
            },
            {
                "name": "search_ast",
                "description": "⚠️ ADVANCED USERS ONLY - DO NOT USE UNLESS ABSOLUTELY NECESSARY ⚠️\n\nStructure-aware code search using Tree-sitter AST patterns (S-expressions).\n\n**PERFORMANCE WARNING:** AST queries bypass trigram optimization and scan the ENTIRE codebase (500ms-10s+).\n\n**WHEN TO USE (RARE):**\n- You need to match code structure, not just text (e.g., \"all async functions with try/catch blocks\")\n- --symbols search is insufficient (e.g., need to match specific AST node types)\n- You have a very specific structural pattern that cannot be expressed as text\n\n**IN 95% OF CASES, USE search_code with symbols=true INSTEAD** (10-100x faster).\n\n**REQUIRED:** You MUST use glob patterns to limit scope (e.g., glob=['src/**/*.rs']) to avoid scanning thousands of files.\n\n**Token efficiency:** Previews are auto-truncated to ~100 chars. Use limit parameter to control result count.\n\n**Error Handling:** If you receive an error message containing \"Index not found\" or \"stale\", immediately call the index_project tool, wait for it to complete, then retry this operation.\n\n**Example AST patterns:**\n- Rust: '(function_item) @fn' (all functions)\n- Python: '(function_definition) @fn' (all functions)\n- TypeScript: '(class_declaration) @class' (all classes)\n\nRefer to Tree-sitter documentation for each language's grammar.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "AST pattern (Tree-sitter S-expression, e.g., '(function_item) @fn')"
                        },
                        "lang": {
                            "type": "string",
                            "description": "Language (REQUIRED: rust, typescript, javascript, python, go, java, c, cpp, csharp, php, ruby, kotlin, zig)"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching glob patterns (STRONGLY RECOMMENDED to limit scope, e.g., ['src/**/*.rs'])"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching glob patterns (e.g., ['target/**', 'node_modules/**'])"
                        },
                        "file": {
                            "type": "string",
                            "description": "Filter by file path (substring)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (use with offset for pagination)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N results after sorting)"
                        },
                        "paths": {
                            "type": "boolean",
                            "description": "Return only unique file paths"
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        },
                        "dependencies": {
                            "type": "boolean",
                            "description": "Include dependency information (imports) in results. Only extracts static imports."
                        }
                    },
                    "required": ["pattern", "lang"]
                }
            },
            {
                "name": "index_project",
                "description": "Rebuild or update the code search index. Run this when:\n\n- After code changes (user edits, git operations, file creation/deletion)\n- Search results seem stale or missing new files\n- Empty/error results (may indicate missing/corrupt index)\n\n**Modes:**\n- Incremental (default): Only re-indexes changed files (fast)\n- Full rebuild (force=true): Re-indexes everything (use if index seems corrupted)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "force": {
                            "type": "boolean",
                            "description": "Force full rebuild (ignore incremental)"
                        },
                        "languages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Languages to include (empty = all)"
                        }
                    }
                }
            },
            {
                "name": "get_dependencies",
                "description": "Get all dependencies (imports) of a specific file.\n\n**Purpose:** Analyze what modules/files a given file imports.\n\n**Returns:** Array of dependency objects with import path, line number, type (internal/external/stdlib), and optional symbols.\n\n**Use this when:**\n- Understanding file dependencies\n- Analyzing import structure\n- Finding what a file depends on\n\n**IMPORTANT:** Only extracts **static imports** (string literals). Dynamic imports (variables, template literals, expressions) are automatically filtered by tree-sitter query design. See CLAUDE.md section \"Dependency/Import Extraction\" for details.\n\n**Note:** Path matching is fuzzy - supports exact paths, fragments, or just filenames.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path (supports fuzzy matching: 'Controllers/FooController.php' or just 'FooController.php')"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "get_dependents",
                "description": "Get all files that depend on (import) a specific file.\n\n**Purpose:** Find what other files import this file (reverse dependency lookup).\n\n**Returns:** Array of file paths that import the specified file.\n\n**Use this when:**\n- Understanding impact of changes\n- Finding usages of a module\n- Analyzing file importance\n\n**IMPORTANT:** Only considers **static imports** (string literals). Dynamic imports are filtered. See CLAUDE.md section \"Dependency/Import Extraction\" for details.\n\n**Note:** Path matching is fuzzy - supports exact paths, fragments, or just filenames.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path (supports fuzzy matching: 'models/User.php' or just 'User.php')"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "get_transitive_deps",
                "description": "Get transitive dependencies of a file up to a specified depth.\n\n**Purpose:** Find not just direct dependencies, but dependencies of dependencies (the full dependency tree).\n\n**Returns:** Object mapping file IDs to their depth in the dependency tree.\n\n**Use this when:**\n- Understanding full dependency chain\n- Analyzing deep coupling\n- Planning refactoring impact\n\n**IMPORTANT:** Only follows **static imports** (string literals). Dynamic imports are filtered. See CLAUDE.md section \"Dependency/Import Extraction\" for details.\n\n**Example:** depth=2 finds: file → deps → deps of deps",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path (supports fuzzy matching)"
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Maximum depth to traverse (default: 3, max recommended: 5)"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "find_hotspots",
                "description": "Find the most-imported files in the codebase (dependency hotspots). This is the definitive tool for 'which file is imported by the most other files' — Reflex answers this in one call from its pre-built dependency index, which grep cannot do without scanning every file.\n\n**Purpose:** Identify files that many other files depend on — critical path analysis, refactoring impact estimation, and architecture review.\n\n**Pagination:** Default limit of 200 results per page. Check response.pagination.has_more to fetch more pages.\n\n**Sorting:** Default order is descending (most imports first). Use sort parameter to change.\n\n**Returns:** Object with pagination metadata and array of {path, import_count} objects sorted by import count.\n\n**Use this when:**\n- Finding critical files (\"what does every module depend on?\")\n- Identifying potential bottlenecks before refactoring\n- Understanding module boundaries and coupling\n- Planning refactoring priorities by blast radius\n\n**IMPORTANT:** Only counts **static imports** (string literals). Dynamic imports are filtered. See CLAUDE.md section \"Dependency/Import Extraction\" for details.\n\n**Example output:** {\"pagination\": {...}, \"results\": [{\"path\": \"src/models.rs\", \"import_count\": 27}]}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of hotspots per page (default: 200)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N results). Use with limit for pagination."
                        },
                        "min_dependents": {
                            "type": "integer",
                            "description": "Minimum number of dependents to include (default: 2)"
                        },
                        "sort": {
                            "type": "string",
                            "description": "Sort order: 'asc' (least imports first) or 'desc' (most imports first, default)"
                        }
                    }
                }
            },
            {
                "name": "find_circular",
                "description": "Detect circular dependencies in the codebase.\n\n**Purpose:** Find dependency cycles (A → B → C → A).\n\n**Pagination:** Default limit of 200 results per page. Check response.pagination.has_more to fetch more pages.\n\n**Sorting:** Default order is descending (longest cycles first). Use sort parameter to change.\n\n**Returns:** Object with pagination metadata and array of cycles, where each cycle is an array of file paths forming the circular path.\n\n**Use this when:**\n- Debugging circular dependency issues\n- Improving code architecture\n- Validating refactoring\n\n**IMPORTANT:** Only detects cycles in **static imports** (string literals). Dynamic imports are filtered. See CLAUDE.md section \"Dependency/Import Extraction\" for details.\n\n**Note:** Circular dependencies can cause compilation issues and indicate architectural problems.\n\n**Example output:** {\"pagination\": {...}, \"results\": [{\"paths\": [\"a.rs\", \"b.rs\", \"a.rs\"]}]}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of cycles per page (default: 200)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N cycles). Use with limit for pagination."
                        },
                        "sort": {
                            "type": "string",
                            "description": "Sort order: 'asc' (shortest cycles first) or 'desc' (longest cycles first, default)"
                        }
                    }
                }
            },
            {
                "name": "find_unused",
                "description": "Find unused files that no other files import.\n\n**Purpose:** Identify orphaned files that could be safely removed.\n\n**Pagination:** Default limit of 200 results per page. Check response.pagination.has_more to fetch more pages.\n\n**Returns:** Object with pagination metadata and flat array of file path strings (no wrapping objects).\n\n**Use this when:**\n- Cleaning up dead code\n- Reducing codebase size\n- Identifying test-only or entry-point files\n\n**Note:** Entry points (main.rs, index.ts) will appear as unused but should not be deleted.\n\n**Example output:** {\"pagination\": {...}, \"results\": [\"src/unused.rs\", \"tests/old.rs\"]}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of unused files per page (default: 200)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N files). Use with limit for pagination."
                        }
                    }
                }
            },
            {
                "name": "find_islands",
                "description": "Find disconnected components (islands) in the dependency graph.\n\n**Purpose:** Identify groups of files that are isolated from the rest of the codebase (no dependencies between groups).\n\n**Pagination:** Default limit of 200 results per page. Check response.pagination.has_more to fetch more pages.\n\n**Sorting:** Default order is descending (largest islands first). Use sort parameter to change.\n\n**Returns:** Object with pagination metadata and array of islands, where each island contains multiple file paths that depend on each other.\n\n**Use this when:**\n- Identifying isolated subsystems\n- Understanding codebase modularity\n- Finding potential code splitting opportunities\n- Detecting disconnected feature modules\n\n**IMPORTANT:** Only considers **static imports** (string literals). Dynamic imports are filtered. See CLAUDE.md section \"Dependency/Import Extraction\" for details.\n\n**Size filtering:** Use min_island_size and max_island_size to filter by component size. Default: 2-500 files (or 50% of total files).\n\n**Example output:** {\"pagination\": {...}, \"results\": [{\"island_id\": 1, \"size\": 5, \"paths\": [\"a.rs\", \"b.rs\", \"c.rs\", \"d.rs\", \"e.rs\"]}]}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of islands per page (default: 200)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset (skip first N islands). Use with limit for pagination."
                        },
                        "min_island_size": {
                            "type": "integer",
                            "description": "Minimum files in an island to include (default: 2)"
                        },
                        "max_island_size": {
                            "type": "integer",
                            "description": "Maximum files in an island to include (default: 500 or 50% of total files)"
                        },
                        "sort": {
                            "type": "string",
                            "description": "Sort order: 'asc' (smallest islands first) or 'desc' (largest islands first, default)"
                        }
                    }
                }
            },
            {
                "name": "analyze_summary",
                "description": "Get a summary of all dependency analyses.\n\n**Purpose:** Quick overview of codebase dependency health.\n\n**Returns:** Object with counts: {circular_dependencies, hotspots, unused_files, islands, min_dependents}\n\n**Use this when:**\n- Getting a quick health check of the codebase\n- Understanding overall dependency structure\n- Deciding which specific analysis to run next\n\n**IMPORTANT:** Only considers **static imports** (string literals). Dynamic imports are filtered. See CLAUDE.md section \"Dependency/Import Extraction\" for details.\n\n**Example output:** {\"circular_dependencies\": 17, \"hotspots\": 10, \"unused_files\": 82, \"islands\": 81, \"min_dependents\": 2}",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "min_dependents": {
                            "type": "integer",
                            "description": "Minimum number of dependents for hotspots (default: 2)"
                        }
                    }
                }
            },
            {
                "name": "find_references",
                "description": "Find ALL code locations that reference a symbol or pattern in a single call — do NOT chain with search_code for the same pattern; this already covers the full codebase.\n\n**Purpose:** Eliminates the two-step pattern of search_code(symbols=true) + search_code(). Returns both the definition and all usages atomically in one call. Results are complete — no need to follow up with additional searches.\n\n**Filtering:** By default, matches inside string literals and comments are excluded (e.g., test fixture strings, doc comments). Pass `include_strings: true` to restore all occurrences.\n\n**Returns:** {definition, references, total_references, pagination, status}\n\n**Use this when:**\n- \"Find all callers of X\" — the most common agent refactoring pattern\n- Code review: understand impact before changing a function or class\n- Rename planning: see every site that needs updating\n- Dead code detection: confirm nothing calls a function before removing it\n\n**definition:** First matching symbol definition {path, line, kind, symbol, span, preview}, or null if no symbol definition exists for the pattern.\n\n**references:** Flat array of {path, line, preview} — all textual occurrences including the definition site itself.\n\n**Pagination applies to references only.** Use limit and offset. Check pagination.has_more for more pages.\n\n**Error Handling:** If you receive an error message containing \"Index not found\" or \"stale\", immediately call the index_project tool, wait for it to complete, then retry this operation.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Symbol name or text pattern to find references for (e.g., 'CacheManager', 'extract_symbols')"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["list", "count"],
                            "description": "Response mode: \"list\" (default) returns full results with definition + references; \"count\" returns only {count, pattern} — faster, skips match body serialization."
                        },
                        "kind": {
                            "type": "string",
                            "description": "Filter definition lookup by symbol kind (function, class, struct, trait, etc.)"
                        },
                        "lang": {
                            "type": "string",
                            "description": "Filter by language (rust, typescript, python, go, etc.)"
                        },
                        "glob": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Include files matching glob patterns (e.g., ['src/**/*.rs'])"
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude files matching glob patterns (e.g., ['target/**', 'tests/**'])"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max references per page (default: 200, max: 500). The 200-result default covers most find-all tasks in a single call. Pagination applies to references only."
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset for references (skip first N). Use with limit."
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force execution of potentially expensive queries (bypasses broad query detection)"
                        },
                        "include_strings": {
                            "type": "boolean",
                            "description": "Include matches inside string literals and comments (default: false). By default these are excluded to focus on real call sites."
                        }
                    },
                    "required": ["pattern"]
                },
                "outputSchema": schema_find_references_output()
            },
            {
                "name": "gather_context",
                "description": "Collects comprehensive codebase information.\n\n**Parameters:**\n- `structure` (bool): Show directory tree\n- `file_types` (bool): Show file type distribution\n- `project_type` (bool): Detect project type (CLI/library/webapp)\n- `framework` (bool): Detect frameworks (React, Django, etc.)\n- `entry_points` (bool): Find main/index files\n- `test_layout` (bool): Show test organization\n- `config_files` (bool): List configuration files\n- `depth` (int): Tree depth for structure (default: 2)\n- `path` (string, optional): Focus on specific directory\n\n**When to use:**\n- Understanding project structure and organization\n- Finding which frameworks/languages are used\n- Locating entry points and test layouts\n- Getting file statistics and distribution\n\n**When NOT to use:**\n- Finding conceptual/architectural information (use search_documentation)\n- Understanding high-level how things work (use search_documentation)\n\n**Note:** By default (no parameters), all context types are gathered.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "structure": {
                            "type": "boolean",
                            "description": "Show directory structure"
                        },
                        "file_types": {
                            "type": "boolean",
                            "description": "Show file type distribution"
                        },
                        "project_type": {
                            "type": "boolean",
                            "description": "Detect project type (CLI/library/webapp/monorepo)"
                        },
                        "framework": {
                            "type": "boolean",
                            "description": "Detect frameworks and conventions"
                        },
                        "entry_points": {
                            "type": "boolean",
                            "description": "Show entry point files"
                        },
                        "test_layout": {
                            "type": "boolean",
                            "description": "Show test organization pattern"
                        },
                        "config_files": {
                            "type": "boolean",
                            "description": "List important configuration files"
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Tree depth for structure (default: 2)"
                        },
                        "path": {
                            "type": "string",
                            "description": "Focus on specific directory path"
                        }
                    }
                },
                "outputSchema": schema_gather_context_output()
            },
            {
                "name": "check_index_status",
                "description": "Check whether the Reflex search index is fresh, stale, or missing — without running any search.\n\n**CALL THIS FIRST** at the start of every session and before any significant search task. If the index is stale or missing, call index_project before searching.\n\n**Returns:**\n- `status`: `\"fresh\"` | `\"stale\"` | `\"missing\"`\n- `reason`: why the index is stale (branch not indexed, commit changed, files modified)\n- `action_required`: command to fix the issue (always `rfx index` when stale)\n- `files_modified`: number of recently modified files detected (only present for mtime-based staleness)\n\n**When to call:**\n- At the start of every agent session\n- Before any bulk search or refactoring task\n- After a git operation (checkout, merge, rebase, pull)\n\n**Example fresh response:** `{\"status\": \"fresh\"}`\n**Example stale response:** `{\"status\": \"stale\", \"reason\": \"Commit changed from abc1234 to def5678\", \"action_required\": \"rfx index\"}`",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    });

    if enable_structural {
        return Ok(all_tools);
    }

    // Filter out structural-analysis tools when disabled in config
    let tools: Vec<Value> = all_tools["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| !STRUCTURAL_TOOLS.contains(&t["name"].as_str().unwrap_or("")))
        .cloned()
        .collect();
    Ok(json!({ "tools": tools }))
}

/// Handle tools/call request
/// Build a successful MCP `tools/call` result carrying both the legacy
/// `content[text]` JSON string (unchanged, fully backwards-compatible) and the
/// 2025-11-25 `structuredContent` object holding the same data natively.
///
/// Stage 1 (REF-202): additive only. `content[text]` remains the full JSON
/// string so existing clients keep working; clients that don't recognise
/// `structuredContent` simply ignore it. Only success paths use this — error
/// results are surfaced through the JSON-RPC error channel in `process_request`,
/// never as a tool result, so there are no `isError` responses to preserve here.
fn make_tool_result(data: Value) -> Value {
    make_tool_result_with(data, structured_content_enabled(), sc_stage2_enabled())
}

/// REF-204: A/B efficacy switch for the additive `structuredContent` field.
///
/// Returns `false` only when `REFLEX_MCP_STRUCTURED_CONTENT` is explicitly set to
/// a falsey value (`0`/`false`/`off`/`no`, case-insensitive); otherwise the field
/// is emitted (the shipped REF-202 default). Disabling it reproduces the
/// pre-REF-202 `content[text]`-only tool-result shape from the *same* binary, so
/// the B_sc-vs-B efficacy comparison isolates exactly one variable — the presence
/// of `structuredContent` — while `content[text]` stays byte-identical.
fn structured_content_enabled() -> bool {
    match std::env::var("REFLEX_MCP_STRUCTURED_CONTENT") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    }
}

/// REF-196 Stage 2: replace `content[text]` with a brief human summary.
///
/// When enabled, `content[text]` carries a short description (e.g. "Found 42
/// matches in 3 files — see structuredContent") while the full data lives in
/// `structuredContent`. This eliminates the Stage 1 double-payload cost: Stage 1
/// emitted both the full JSON string AND the native object; Stage 2 emits only
/// the summary string + the native object, so total token cost ≈ data once.
///
/// Only meaningful when `REFLEX_MCP_STRUCTURED_CONTENT` is also enabled.
fn sc_stage2_enabled() -> bool {
    match std::env::var("REFLEX_MCP_SC_STAGE2") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        ),
        Err(_) => false,
    }
}

/// REF-209: columnar result-format toggle for `search_code` / `search_regex`.
///
/// Default ON. Returns `false` only when `REFLEX_MCP_COLUMNAR` is explicitly set
/// to a falsey value (`0`/`false`/`off`/`no`, case-insensitive), which restores
/// the legacy file-grouped `results` array for backwards compatibility. Both the
/// emitted payload (`to_columnar`) and the declared `outputSchema`
/// (`schema_search_output`) consult this, so they never drift.
fn columnar_enabled() -> bool {
    match std::env::var("REFLEX_MCP_COLUMNAR") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    }
}

/// REF-209: reshape a search response's file-grouped `results` array into a
/// columnar `{ columns, rows }` pair to cut key-repetition token cost.
///
/// One row is emitted per match; `path`/`language` (and any file-level
/// `dependencies`) repeat per row so each row is self-contained and needs no
/// back-reference. Columns are emitted dynamically: the five always-present
/// fields (`path`, `language`, `start_line`, `end_line`, `preview`) plus any
/// optional field (`kind`, `symbol`, `context_before`, `context_after`,
/// `dependencies`) that at least one match/file actually carries — so the common
/// full-text case stays at five columns with no all-`null` padding.
///
/// Objects without a `results` array (count-mode `{count, pattern}`, error
/// shapes) are returned unchanged, so this is safe to call on any success value.
fn to_columnar(mut value: Value) -> Value {
    // Only transform success objects that carry a `results` array.
    if !value.get("results").map(Value::is_array).unwrap_or(false) {
        return value;
    }
    let obj = value
        .as_object_mut()
        .expect("value has a `results` member, so it is an object");
    let results = match obj.remove("results") {
        Some(Value::Array(results)) => results,
        // Unreachable given the guard above, but stay total rather than panic.
        other => {
            if let Some(other) = other {
                obj.insert("results".to_string(), other);
            }
            return value;
        }
    };

    // Pass 1: decide which optional columns any row needs.
    let mut has_kind = false;
    let mut has_symbol = false;
    let mut has_ctx_before = false;
    let mut has_ctx_after = false;
    let mut has_deps = false;
    for file in &results {
        if file.get("dependencies").is_some() {
            has_deps = true;
        }
        if let Some(matches) = file.get("matches").and_then(Value::as_array) {
            for m in matches {
                has_kind |= m.get("kind").is_some();
                has_symbol |= m.get("symbol").is_some();
                has_ctx_before |= m.get("context_before").is_some();
                has_ctx_after |= m.get("context_after").is_some();
            }
        }
    }

    // Fixed base columns; optional ones appended only when present, in a stable
    // order so `columns[i]` is deterministic for a given result set.
    let mut columns: Vec<&'static str> =
        vec!["path", "language", "start_line", "end_line", "preview"];
    if has_kind {
        columns.push("kind");
    }
    if has_symbol {
        columns.push("symbol");
    }
    if has_ctx_before {
        columns.push("context_before");
    }
    if has_ctx_after {
        columns.push("context_after");
    }
    if has_deps {
        columns.push("dependencies");
    }

    // Pass 2: project each match into a positional row aligned to `columns`.
    let mut rows: Vec<Value> = Vec::new();
    for file in &results {
        let path = file.get("path").cloned().unwrap_or(Value::Null);
        let language = file.get("language").cloned().unwrap_or(Value::Null);
        let deps = file.get("dependencies").cloned().unwrap_or(Value::Null);
        let Some(matches) = file.get("matches").and_then(Value::as_array) else {
            continue;
        };
        for m in matches {
            let span = m.get("span");
            let start_line = span
                .and_then(|s| s.get("start_line"))
                .cloned()
                .unwrap_or(Value::Null);
            let end_line = span
                .and_then(|s| s.get("end_line"))
                .cloned()
                .unwrap_or(Value::Null);
            let preview = m.get("preview").cloned().unwrap_or(Value::Null);

            let mut row: Vec<Value> = vec![
                path.clone(),
                language.clone(),
                start_line,
                end_line,
                preview,
            ];
            if has_kind {
                row.push(m.get("kind").cloned().unwrap_or(Value::Null));
            }
            if has_symbol {
                row.push(m.get("symbol").cloned().unwrap_or(Value::Null));
            }
            if has_ctx_before {
                row.push(m.get("context_before").cloned().unwrap_or(Value::Null));
            }
            if has_ctx_after {
                row.push(m.get("context_after").cloned().unwrap_or(Value::Null));
            }
            if has_deps {
                row.push(deps.clone());
            }
            rows.push(Value::Array(row));
        }
    }

    obj.insert("columns".to_string(), json!(columns));
    obj.insert("rows".to_string(), Value::Array(rows));
    value
}

/// Generate a brief human-readable summary of a tool result for Stage 2
/// `content[text]`. Inspects common result shape fields to produce a concise
/// description. Falls back to a generic message for unrecognised shapes.
fn brief_summary(data: &Value) -> String {
    // find_references: definition + usages
    if let Some(usages) = data.get("usages").and_then(|v| v.as_array()) {
        let n = data
            .get("total_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(usages.len() as u64);
        let has_def = !data.get("definition").map(|v| v.is_null()).unwrap_or(true);
        return if has_def {
            format!("Definition + {} usage(s) — see structuredContent", n)
        } else {
            format!("{} usage(s) — see structuredContent", n)
        };
    }
    // search_code / search_regex / find_references pagination shape
    if let Some(pagination) = data.get("pagination") {
        let total = pagination
            .get("total")
            .and_then(|v| v.as_u64())
            .or_else(|| data.get("total_count").and_then(|v| v.as_u64()))
            .unwrap_or(0);
        let file_count = data
            .get("results")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        return if file_count > 0 {
            format!(
                "Found {} match(es) across {} file(s) — see structuredContent",
                total, file_count
            )
        } else {
            format!("Found {} match(es) — see structuredContent", total)
        };
    }
    // count_occurrences
    if let Some(count) = data.get("count").and_then(|v| v.as_u64()) {
        return format!("{} occurrence(s) — see structuredContent", count);
    }
    // list_locations
    if let Some(n) = data
        .get("total_locations")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            data.get("locations")
                .and_then(|v| v.as_array())
                .map(|a| a.len() as u64)
        })
    {
        return format!("{} location(s) — see structuredContent", n);
    }
    // gather_context / project summary
    if data.get("entry_points").is_some() || data.get("directory_structure").is_some() {
        let files = data
            .get("index_stats")
            .and_then(|s| s.get("total_files"))
            .and_then(|v| v.as_u64());
        return match files {
            Some(n) => format!("Project context: {} file(s) — see structuredContent", n),
            None => "Project context — see structuredContent".to_string(),
        };
    }
    // generic dependency / hotspot results
    if let Some(arr) = data
        .get("dependencies")
        .or_else(|| data.get("dependents"))
        .or_else(|| data.get("hotspots"))
        .and_then(|v| v.as_array())
    {
        return format!("{} item(s) — see structuredContent", arr.len());
    }
    "Result — see structuredContent".to_string()
}

/// Inner builder split out from [`make_tool_result`] so both output shapes are
/// unit-testable without mutating process-global env (which would race cargo's
/// parallel test threads).
///
/// - `structured = false`: legacy `content[text]`-only JSON string (B_nosc arm)
/// - `structured = true, stage2 = false`: Stage 1 — full JSON in both fields (B_sc arm)
/// - `structured = true, stage2 = true`: Stage 2 — brief summary in `content[text]`,
///   full data in `structuredContent` (B_sc2 arm; the token-saving production target)
fn make_tool_result_with(data: Value, structured: bool, stage2: bool) -> Value {
    let text = if structured && stage2 {
        brief_summary(&data)
    } else {
        serde_json::to_string(&data).unwrap_or_default()
    };
    let content = json!([{"type": "text", "text": text}]);
    if structured {
        json!({ "content": content, "structuredContent": data })
    } else {
        json!({ "content": content })
    }
}

fn handle_call_tool(params: Option<Value>) -> Result<Value> {
    let params = params.ok_or_else(|| anyhow::anyhow!("Missing params for tools/call"))?;

    let name = params["name"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing tool name"))?;

    let arguments = params["arguments"].clone();

    match name {
        "list_locations" => {
            // Location discovery tool (minimal token usage)
            let pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern"))?
                .to_string();

            let lang = arguments["lang"].as_str().map(|s| s.to_string());
            let file = arguments["file"].as_str().map(|s| s.to_string());
            let glob_patterns = arguments["glob"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let exclude_patterns = arguments["exclude"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let force = arguments["force"].as_bool().unwrap_or(false);
            let dependencies = arguments["dependencies"].as_bool().unwrap_or(false);

            let language = parse_language(lang);

            let filter = QueryFilter {
                language,
                kind: None,
                use_ast: false,
                use_regex: false,
                limit: None, // No limit for paths-only mode
                symbols_mode: false,
                expand: false,
                file_pattern: file,
                exact: false,
                use_contains: false,
                timeout_secs: 30,
                glob_patterns,
                exclude_patterns,
                paths_only: true, // KEY: Enable paths-only mode
                offset: None,
                force,
                suppress_output: true, // MCP always returns JSON
                include_dependencies: dependencies,
                ..Default::default()
            };

            let cache = CacheManager::new(".");
            let engine = QueryEngine::new(cache);
            let response = engine.search_with_metadata(&pattern, filter)?;

            // Extract locations (path + line) for each match
            let locations: Vec<serde_json::Value> = response
                .results
                .iter()
                .flat_map(|file_group| {
                    file_group.matches.iter().map(move |m| {
                        json!({
                            "path": file_group.path.clone(),
                            "line": m.span.start_line
                        })
                    })
                })
                .collect();

            // Return compact response (just locations + count)
            let compact_response = json!({
                "status": response.status,
                "total_locations": locations.len(),
                "locations": locations
            });

            Ok(make_tool_result(compact_response))
        }
        "count_occurrences" => {
            // Quick stats tool (minimal token usage)
            let pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern"))?
                .to_string();

            let lang = arguments["lang"].as_str().map(|s| s.to_string());
            let kind = arguments["kind"].as_str().map(|s| s.to_string());
            let symbols = arguments["symbols"].as_bool();
            let file = arguments["file"].as_str().map(|s| s.to_string());
            let glob_patterns = arguments["glob"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let exclude_patterns = arguments["exclude"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let force = arguments["force"].as_bool().unwrap_or(false);
            let dependencies = arguments["dependencies"].as_bool().unwrap_or(false);

            let language = parse_language(lang);
            let parsed_kind = parse_symbol_kind(kind);
            let symbols_mode = symbols.unwrap_or(false) || parsed_kind.is_some();

            let filter = QueryFilter {
                language,
                kind: parsed_kind,
                use_ast: false,
                use_regex: false,
                limit: None, // No limit for counting
                symbols_mode,
                expand: false,
                file_pattern: file,
                exact: false,
                use_contains: false,
                timeout_secs: 30,
                glob_patterns,
                exclude_patterns,
                paths_only: false, // Need to count all occurrences
                offset: None,
                force,
                suppress_output: true, // MCP always returns JSON
                include_dependencies: dependencies,
                ..Default::default()
            };

            let cache = CacheManager::new(".");
            let engine = QueryEngine::new(cache);
            let response = engine.search_with_metadata(&pattern, filter)?;

            // Count unique files
            use std::collections::HashSet;
            let unique_files: HashSet<String> =
                response.results.iter().map(|fg| fg.path.clone()).collect();

            // Return minimal stats
            let stats = json!({
                "status": response.status,
                "pattern": pattern,
                "total": response.pagination.total,
                "files": unique_files.len()
            });

            Ok(make_tool_result(stats))
        }
        "search_code" => {
            let pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern"))?
                .to_string();

            let lang = arguments["lang"].as_str().map(|s| s.to_string());
            let kind = arguments["kind"].as_str().map(|s| s.to_string());
            let symbols = arguments["symbols"].as_bool();
            let exact = arguments["exact"].as_bool();
            let file = arguments["file"].as_str().map(|s| s.to_string());
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let expand = arguments["expand"].as_bool();
            let glob_patterns: Vec<String> = arguments["glob"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let exclude_patterns = arguments["exclude"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let paths_only = arguments["paths"].as_bool().unwrap_or(false);
            let force = arguments["force"].as_bool().unwrap_or(false);
            let dependencies = arguments["dependencies"].as_bool().unwrap_or(false);
            let preview_length = arguments["preview_length"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(DEFAULT_MCP_PREVIEW_LENGTH);

            let language = parse_language(lang.clone());

            // Build warning for unsupported language + dependencies combination (REF-171)
            let deps_lang_warning: Option<String> =
                if dependencies && matches!(language, Some(l) if l != Language::Rust) {
                    Some(format!(
                        "Warning: dependencies is currently only supported for Rust files. \
                         No dependency data will be included for {} files.",
                        lang.as_deref().unwrap_or("non-Rust")
                    ))
                } else {
                    None
                };

            let parsed_kind = parse_symbol_kind(kind);
            let symbols_mode = symbols.unwrap_or(false) || parsed_kind.is_some();

            let offset = arguments["offset"].as_u64().map(|n| n as usize);

            // Smart limit handling:
            // 1. If --paths is set and user didn't specify limit: no limit (None)
            // 2. If user specified limit: use that value, capped at 500
            // 3. Otherwise: use the agent-oriented default (REF-191) so find-all
            //    tasks come back in one call instead of paginating.
            let final_limit = if paths_only && limit.is_none() {
                None // --paths without explicit limit means no limit
            } else if let Some(user_limit) = limit {
                Some(user_limit.min(500)) // Use user-specified limit, capped at 500
            } else {
                Some(DEFAULT_MCP_RESULT_LIMIT)
            };

            let mode = arguments["mode"].as_str().unwrap_or("list");

            // Count mode: run query but return only the total match count.
            // Skips preview truncation and full result serialization for speed.
            if mode == "count" {
                let count_filter = QueryFilter {
                    language,
                    kind: parsed_kind,
                    use_ast: false,
                    use_regex: false,
                    limit: None, // count everything
                    symbols_mode,
                    expand: false,
                    file_pattern: file,
                    exact: exact.unwrap_or(false),
                    use_contains: false,
                    timeout_secs: 30,
                    glob_patterns,
                    exclude_patterns,
                    paths_only: false,
                    offset: None,
                    force,
                    suppress_output: true,
                    include_dependencies: false,
                    ..Default::default()
                };
                let cache = CacheManager::new(".");
                let engine = QueryEngine::new(cache);
                let response = engine.search_with_metadata(&pattern, count_filter)?;
                let result = json!({"count": response.pagination.total, "pattern": pattern});
                return Ok(make_tool_result(result));
            }

            let filter = QueryFilter {
                language,
                kind: parsed_kind,
                use_ast: false,
                use_regex: false,
                limit: final_limit,
                symbols_mode,
                expand: expand.unwrap_or(false),
                file_pattern: file,
                exact: exact.unwrap_or(false),
                use_contains: false, // Default to word-boundary matching for MCP
                timeout_secs: 30,    // Default 30 second timeout for MCP queries
                glob_patterns: glob_patterns.clone(),
                exclude_patterns,
                paths_only,
                offset,
                force,
                suppress_output: true, // MCP always returns JSON
                include_dependencies: dependencies,
                ..Default::default()
            };

            let cache = CacheManager::new(".");
            let engine = QueryEngine::new(cache);
            let mut response = engine.search_with_metadata(&pattern, filter)?;

            // Apply preview truncation for token efficiency
            for file_group in response.results.iter_mut() {
                for m in file_group.matches.iter_mut() {
                    m.preview = crate::cli::truncate_preview(&m.preview, preview_length);
                }
            }

            // Calculate result count for AI instruction
            let result_count: usize = response.results.iter().map(|fg| fg.matches.len()).sum();

            // Generate AI instruction (MCP always uses AI mode)
            response.ai_instruction = crate::query::generate_ai_instruction(
                result_count,
                response.pagination.total,
                response.pagination.has_more,
                symbols_mode,
                paths_only,
                false, // use_ast
                false, // use_regex
                language.is_some(),
                !glob_patterns.is_empty(),
                exact.unwrap_or(false),
            );

            // Prepend language limitation warning to AI instruction (REF-171)
            if let Some(warn) = deps_lang_warning {
                response.ai_instruction = Some(match response.ai_instruction.take() {
                    Some(existing) => format!("{warn}\n\n{existing}"),
                    None => warn,
                });
            }

            // Extract pagination scalars before consuming response (REF-185)
            let has_more = response.pagination.has_more;
            let total_count = response.pagination.total;

            let mut response_val = serde_json::to_value(response)?;
            if let serde_json::Value::Object(ref mut map) = response_val {
                map.insert("has_more".to_string(), json!(has_more));
                map.insert("total_count".to_string(), json!(total_count));
                map.insert("returned_count".to_string(), json!(result_count));
            }

            // REF-209: emit the token-efficient columnar shape by default; the
            // env toggle restores the legacy results[] array for compatibility.
            if columnar_enabled() {
                response_val = to_columnar(response_val);
            }

            Ok(make_tool_result(response_val))
        }
        "search_regex" => {
            let pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern"))?
                .to_string();

            let lang = arguments["lang"].as_str().map(|s| s.to_string());
            let file = arguments["file"].as_str().map(|s| s.to_string());
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let glob_patterns: Vec<String> = arguments["glob"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let exclude_patterns = arguments["exclude"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let paths_only = arguments["paths"].as_bool().unwrap_or(false);
            let force = arguments["force"].as_bool().unwrap_or(false);
            let dependencies = arguments["dependencies"].as_bool().unwrap_or(false);

            let language = parse_language(lang);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);

            // Smart limit handling (same as search_code)
            let final_limit = if paths_only && limit.is_none() {
                None // --paths without explicit limit means no limit
            } else if let Some(user_limit) = limit {
                Some(user_limit.min(500)) // Use user-specified limit, capped at 500
            } else {
                Some(DEFAULT_MCP_RESULT_LIMIT) // REF-191: one-call default
            };

            let mode = arguments["mode"].as_str().unwrap_or("list");

            // Count mode: return only the total match count, no match bodies.
            if mode == "count" {
                let count_filter = QueryFilter {
                    language,
                    kind: None,
                    use_ast: false,
                    use_regex: true,
                    limit: None, // count everything
                    symbols_mode: false,
                    expand: false,
                    file_pattern: file,
                    exact: false,
                    use_contains: false,
                    timeout_secs: 30,
                    glob_patterns,
                    exclude_patterns,
                    paths_only: false,
                    offset: None,
                    force,
                    suppress_output: true,
                    include_dependencies: false,
                    ..Default::default()
                };
                let cache = CacheManager::new(".");
                let engine = QueryEngine::new(cache);
                let response = engine.search_with_metadata(&pattern, count_filter)?;
                let result = json!({"count": response.pagination.total, "pattern": pattern});
                return Ok(make_tool_result(result));
            }

            let filter = QueryFilter {
                language,
                kind: None,
                use_ast: false,
                use_regex: true,
                limit: final_limit,
                symbols_mode: false,
                expand: false,
                file_pattern: file,
                exact: false,
                use_contains: false, // Regex mode uses substring matching via use_regex flag
                timeout_secs: 30,    // Default 30 second timeout for MCP queries
                glob_patterns: glob_patterns.clone(),
                exclude_patterns,
                paths_only,
                offset,
                force,
                suppress_output: true, // MCP always returns JSON
                include_dependencies: dependencies,
                ..Default::default()
            };

            let cache = CacheManager::new(".");
            let engine = QueryEngine::new(cache);
            let mut response = engine.search_with_metadata(&pattern, filter)?;

            // Apply preview truncation for token efficiency
            for file_group in response.results.iter_mut() {
                for m in file_group.matches.iter_mut() {
                    m.preview =
                        crate::cli::truncate_preview(&m.preview, DEFAULT_MCP_PREVIEW_LENGTH);
                }
            }

            // Calculate result count for AI instruction
            let result_count: usize = response.results.iter().map(|fg| fg.matches.len()).sum();

            // Generate AI instruction (MCP always uses AI mode)
            response.ai_instruction = crate::query::generate_ai_instruction(
                result_count,
                response.pagination.total,
                response.pagination.has_more,
                false, // symbols_mode
                paths_only,
                false, // use_ast
                true,  // use_regex
                language.is_some(),
                !glob_patterns.is_empty(),
                false, // exact
            );

            // Extract pagination scalars before consuming response (REF-185)
            let has_more = response.pagination.has_more;
            let total_count = response.pagination.total;

            let mut response_val = serde_json::to_value(response)?;
            if let serde_json::Value::Object(ref mut map) = response_val {
                map.insert("has_more".to_string(), json!(has_more));
                map.insert("total_count".to_string(), json!(total_count));
                map.insert("returned_count".to_string(), json!(result_count));
            }

            // REF-209: emit the token-efficient columnar shape by default; the
            // env toggle restores the legacy results[] array for compatibility.
            if columnar_enabled() {
                response_val = to_columnar(response_val);
            }

            Ok(make_tool_result(response_val))
        }
        "search_ast" => {
            // AST pattern (Tree-sitter S-expression)
            let ast_pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern (AST S-expression)"))?
                .to_string();

            let lang_str = arguments["lang"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing lang (required for AST queries)"))?
                .to_string();

            let file = arguments["file"].as_str().map(|s| s.to_string());
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let glob_patterns: Vec<String> = arguments["glob"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let exclude_patterns: Vec<String> = arguments["exclude"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let paths_only = arguments["paths"].as_bool().unwrap_or(false);
            let force = arguments["force"].as_bool().unwrap_or(false);
            let dependencies = arguments["dependencies"].as_bool().unwrap_or(false);

            let language = parse_language(Some(lang_str)).ok_or_else(|| {
                anyhow::anyhow!("Invalid or unsupported language for AST queries")
            })?;

            // Warn if glob patterns are not provided (performance issue)
            if glob_patterns.is_empty() && exclude_patterns.is_empty() {
                log::warn!(
                    "⚠️  AST query without glob patterns will scan the ENTIRE codebase. This may take 2-10+ seconds."
                );
                log::warn!(
                    "    Strongly recommend using glob patterns, e.g., glob=['src/**/*.rs']"
                );
            }

            let offset = arguments["offset"].as_u64().map(|n| n as usize);

            // Smart limit handling (same as search_code)
            let final_limit = if paths_only && limit.is_none() {
                None // --paths without explicit limit means no limit
            } else if let Some(user_limit) = limit {
                Some(user_limit) // Use user-specified limit
            } else {
                Some(100) // Default: limit to 100 results for token efficiency
            };

            let filter = QueryFilter {
                language: Some(language),
                kind: None,
                use_ast: true,
                use_regex: false,
                limit: final_limit,
                symbols_mode: false,
                expand: false,
                file_pattern: file,
                exact: false,
                use_contains: false,
                timeout_secs: 60, // Longer timeout for AST queries (they're slow)
                glob_patterns,
                exclude_patterns,
                paths_only,
                offset,
                force,
                suppress_output: true, // MCP always returns JSON
                include_dependencies: dependencies,
                ..Default::default()
            };

            let cache = CacheManager::new(".");
            let engine = QueryEngine::new(cache);

            // Use the new search_ast_all_files method (no trigram filtering)
            let mut results = engine.search_ast_all_files(&ast_pattern, filter)?;

            // Apply preview truncation for token efficiency
            for result in &mut results {
                result.preview =
                    crate::cli::truncate_preview(&result.preview, DEFAULT_MCP_PREVIEW_LENGTH);
            }

            Ok(make_tool_result(serde_json::to_value(&results)?))
        }
        "index_project" => {
            let force = arguments["force"].as_bool();
            let languages = arguments["languages"].as_array().map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
            });

            let cache = CacheManager::new(".");

            if force.unwrap_or(false) {
                log::info!("Force rebuild requested, clearing existing cache");
                cache.clear()?;
            }

            let lang_filters: Vec<Language> = languages
                .unwrap_or_default()
                .iter()
                .filter_map(|s| parse_language(Some(s.clone())))
                .collect();

            let config = IndexConfig {
                languages: lang_filters,
                ..Default::default()
            };

            let indexer = Indexer::new(cache, config);
            let path = PathBuf::from(".");
            let stats = indexer.index(&path, false)?;

            Ok(make_tool_result(serde_json::to_value(&stats)?))
        }
        "get_dependencies" => {
            let path = arguments["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing path"))?
                .to_string();

            let cache = CacheManager::new(".");
            let deps_index = DependencyIndex::new(cache);

            // Fuzzy path matching
            let file_id = deps_index
                .get_file_id_by_path(&path)?
                .ok_or_else(|| anyhow::anyhow!("File '{}' not found in index", path))?;

            let dependencies = deps_index.get_dependencies_info(file_id)?;

            Ok(make_tool_result(serde_json::to_value(&dependencies)?))
        }
        "get_dependents" => {
            let path = arguments["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing path"))?
                .to_string();

            let cache = CacheManager::new(".");
            let deps_index = DependencyIndex::new(cache);

            // Fuzzy path matching
            let file_id = deps_index
                .get_file_id_by_path(&path)?
                .ok_or_else(|| anyhow::anyhow!("File '{}' not found in index", path))?;

            let dependents = deps_index.get_dependents(file_id)?;
            let paths = deps_index.get_file_paths(&dependents)?;

            // Convert to array of paths
            let path_list: Vec<String> = dependents
                .iter()
                .filter_map(|id| paths.get(id).cloned())
                .collect();

            Ok(make_tool_result(serde_json::to_value(&path_list)?))
        }
        "get_transitive_deps" => {
            let path = arguments["path"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing path"))?
                .to_string();

            let depth = arguments["depth"].as_u64().map(|n| n as usize).unwrap_or(3); // Default depth of 3

            let cache = CacheManager::new(".");
            let deps_index = DependencyIndex::new(cache);

            // Fuzzy path matching
            let file_id = deps_index
                .get_file_id_by_path(&path)?
                .ok_or_else(|| anyhow::anyhow!("File '{}' not found in index", path))?;

            let transitive = deps_index.get_transitive_deps(file_id, depth)?;

            // Get paths for all file IDs
            let file_ids: Vec<i64> = transitive.keys().copied().collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            // Build result with path → depth mapping
            let result: Vec<serde_json::Value> = transitive
                .iter()
                .filter_map(|(id, depth)| {
                    paths.get(id).map(|path| {
                        json!({
                            "path": path,
                            "depth": depth
                        })
                    })
                })
                .collect();

            Ok(make_tool_result(serde_json::to_value(&result)?))
        }
        "find_hotspots" => {
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);
            let min_dependents = arguments["min_dependents"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(2);
            let sort = arguments["sort"].as_str().map(|s| s.to_string());

            let cache = CacheManager::new(".");
            let deps_index = DependencyIndex::new(cache);

            // Get all hotspots first (without limit) to track total count
            let mut all_hotspots = deps_index.find_hotspots(None, min_dependents)?;

            // Apply sorting (default: descending - most imports first)
            let sort_order = sort.as_deref().unwrap_or("desc");
            match sort_order {
                "asc" => {
                    // Ascending: least imports first
                    all_hotspots.sort_by_key(|a| a.1);
                }
                "desc" => {
                    // Descending: most imports first (default)
                    all_hotspots.sort_by_key(|a| std::cmp::Reverse(a.1));
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Invalid sort order '{}'. Supported: asc, desc",
                        sort_order
                    ));
                }
            }

            let total_count = all_hotspots.len();

            // Apply offset pagination
            let offset_val = offset.unwrap_or(0);
            let mut hotspots: Vec<_> = all_hotspots.into_iter().skip(offset_val).collect();

            // Apply limit (default 200)
            let limit_val = limit.unwrap_or(200);
            hotspots.truncate(limit_val);

            let count = hotspots.len();
            let has_more = offset_val + count < total_count;

            // Get paths for all file IDs
            let file_ids: Vec<i64> = hotspots.iter().map(|(id, _)| *id).collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            // Build result with path + import_count (no file_id)
            let results: Vec<serde_json::Value> = hotspots
                .iter()
                .filter_map(|(id, import_count)| {
                    paths.get(id).map(|path| {
                        json!({
                            "path": path,
                            "import_count": import_count,
                        })
                    })
                })
                .collect();

            let response = json!({
                "pagination": {
                    "total": total_count,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit_val,
                    "has_more": has_more,
                },
                "results": results,
            });

            Ok(make_tool_result(response))
        }
        "find_circular" => {
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);
            let sort = arguments["sort"].as_str().map(|s| s.to_string());

            let cache = CacheManager::new(".");
            let deps_index = DependencyIndex::new(cache);

            let mut all_cycles = deps_index.detect_circular_dependencies()?;

            // Apply sorting (default: descending - longest cycles first)
            let sort_order = sort.as_deref().unwrap_or("desc");
            match sort_order {
                "asc" => {
                    // Ascending: shortest cycles first
                    all_cycles.sort_by_key(|cycle| cycle.len());
                }
                "desc" => {
                    // Descending: longest cycles first (default)
                    all_cycles.sort_by_key(|cycle| std::cmp::Reverse(cycle.len()));
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Invalid sort order '{}'. Supported: asc, desc",
                        sort_order
                    ));
                }
            }

            let total_count = all_cycles.len();

            // Apply offset pagination
            let offset_val = offset.unwrap_or(0);
            let mut cycles: Vec<_> = all_cycles.into_iter().skip(offset_val).collect();

            // Apply limit (default 200)
            let limit_val = limit.unwrap_or(200);
            cycles.truncate(limit_val);

            let count = cycles.len();
            let has_more = offset_val + count < total_count;

            // Convert cycles to paths (without file_ids)
            let file_ids: Vec<i64> = cycles.iter().flat_map(|c| c.iter()).copied().collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            let results: Vec<serde_json::Value> = cycles
                .iter()
                .map(|cycle| {
                    let cycle_paths: Vec<_> = cycle
                        .iter()
                        .filter_map(|id| paths.get(id).cloned())
                        .collect();
                    json!({
                        "paths": cycle_paths,
                    })
                })
                .collect();

            let response = json!({
                "pagination": {
                    "total": total_count,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit_val,
                    "has_more": has_more,
                },
                "results": results,
            });

            Ok(make_tool_result(response))
        }
        "find_unused" => {
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);

            let cache = CacheManager::new(".");
            let deps_index = DependencyIndex::new(cache);

            let all_unused = deps_index.find_unused_files()?;
            let total_count = all_unused.len();

            // Apply offset pagination
            let offset_val = offset.unwrap_or(0);
            let mut unused: Vec<_> = all_unused.into_iter().skip(offset_val).collect();

            // Apply limit (default 200)
            let limit_val = limit.unwrap_or(200);
            unused.truncate(limit_val);

            let count = unused.len();
            let has_more = offset_val + count < total_count;

            // Get paths for all unused file IDs
            let paths = deps_index.get_file_paths(&unused)?;

            // Build result (flat array of path strings)
            let results: Vec<String> = unused
                .iter()
                .filter_map(|id| paths.get(id).cloned())
                .collect();

            let response = json!({
                "pagination": {
                    "total": total_count,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit_val,
                    "has_more": has_more,
                },
                "results": results,
            });

            Ok(make_tool_result(response))
        }
        "find_islands" => {
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);
            let min_island_size = arguments["min_island_size"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(2);
            let max_island_size = arguments["max_island_size"].as_u64().map(|n| n as usize);
            let sort = arguments["sort"].as_str().map(|s| s.to_string());

            let cache = CacheManager::new(".");
            let deps_index = DependencyIndex::new(cache);

            let all_islands = deps_index.find_islands()?;
            let total_components = all_islands.len();

            // Get total file count for percentage calculation
            let total_files = deps_index.get_cache().stats()?.total_files;

            // Calculate max_island_size default: min of 500 or 50% of total files
            let max_size = max_island_size.unwrap_or_else(|| {
                let fifty_percent = (total_files as f64 * 0.5) as usize;
                fifty_percent.min(500)
            });

            // Filter islands by size
            let mut islands: Vec<_> = all_islands
                .into_iter()
                .filter(|island| {
                    let size = island.len();
                    size >= min_island_size && size <= max_size
                })
                .collect();

            // Apply sorting (default: descending - largest islands first)
            let sort_order = sort.as_deref().unwrap_or("desc");
            match sort_order {
                "asc" => {
                    // Ascending: smallest islands first
                    islands.sort_by_key(|island| island.len());
                }
                "desc" => {
                    // Descending: largest islands first (default)
                    islands.sort_by_key(|island| std::cmp::Reverse(island.len()));
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Invalid sort order '{}'. Supported: asc, desc",
                        sort_order
                    ));
                }
            }

            let _filtered_count = total_components - islands.len();
            let total_after_filter = islands.len();

            // Apply offset pagination
            let offset_val = offset.unwrap_or(0);
            if offset_val > 0 && offset_val < islands.len() {
                islands = islands.into_iter().skip(offset_val).collect();
            } else if offset_val >= islands.len() {
                islands.clear();
            }

            // Apply limit (default 200)
            let limit_val = limit.unwrap_or(200);
            islands.truncate(limit_val);

            let count = islands.len();
            let has_more = offset_val + count < total_after_filter;

            // Get all file IDs from all islands
            let file_ids: Vec<i64> = islands
                .iter()
                .flat_map(|island| island.iter())
                .copied()
                .collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            // Build result (array of islands with paths, no file_ids)
            let results: Vec<serde_json::Value> = islands
                .iter()
                .enumerate()
                .map(|(idx, island)| {
                    let island_paths: Vec<_> = island
                        .iter()
                        .filter_map(|id| paths.get(id).cloned())
                        .collect();
                    json!({
                        "island_id": idx + 1,
                        "size": island.len(),
                        "paths": island_paths,
                    })
                })
                .collect();

            let response = json!({
                "pagination": {
                    "total": total_after_filter,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit_val,
                    "has_more": has_more,
                },
                "results": results,
            });

            Ok(make_tool_result(response))
        }
        "analyze_summary" => {
            let min_dependents = arguments["min_dependents"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(2);

            let cache = CacheManager::new(".");
            let deps_index = DependencyIndex::new(cache);

            let cycles = deps_index.detect_circular_dependencies()?;
            let hotspots = deps_index.find_hotspots(None, min_dependents)?;
            let unused = deps_index.find_unused_files()?;
            let all_islands = deps_index.find_islands()?;

            let summary = json!({
                "circular_dependencies": cycles.len(),
                "hotspots": hotspots.len(),
                "unused_files": unused.len(),
                "islands": all_islands.len(),
                "min_dependents": min_dependents,
            });

            Ok(make_tool_result(summary))
        }
        "gather_context" => {
            // Parse optional parameters
            let structure = arguments["structure"].as_bool().unwrap_or(false);
            let file_types = arguments["file_types"].as_bool().unwrap_or(false);
            let project_type = arguments["project_type"].as_bool().unwrap_or(false);
            let framework = arguments["framework"].as_bool().unwrap_or(false);
            let entry_points = arguments["entry_points"].as_bool().unwrap_or(false);
            let test_layout = arguments["test_layout"].as_bool().unwrap_or(false);
            let config_files = arguments["config_files"].as_bool().unwrap_or(false);
            let depth = arguments["depth"].as_u64().map(|n| n as usize).unwrap_or(2);
            let path = arguments["path"].as_str().map(|s| s.to_string());

            // Build context options
            let mut opts = crate::context::ContextOptions {
                structure,
                path,
                file_types,
                project_type,
                framework,
                entry_points,
                test_layout,
                config_files,
                depth,
                json: false, // MCP always returns text format
            };

            // If no context flags specified, return minimal orientation context only.
            // Requesting all types by default floods agent context windows (2000-5000 tokens).
            let no_flags_set = opts.is_empty();
            if no_flags_set {
                opts.project_type = true;
                opts.entry_points = true;
            }

            let cache = CacheManager::new(".");
            let context = crate::context::generate_context(&cache, &opts)?;

            let hint = if no_flags_set {
                "\n\n---\nHint: this is the minimal orientation view. Pass any combination of these flags for more detail: structure, file_types, framework, test_layout, config_files."
            } else {
                ""
            };

            // gather_context returns human-readable prose, not JSON. Running it
            // through make_tool_result would JSON-encode (quote + escape) the text
            // and change content[text]; instead keep content[text] as the raw
            // string and expose the same prose under structuredContent (REF-202).
            let context_text = format!("{}{}", context, hint);
            let mut result = json!({
                "content": [{
                    "type": "text",
                    "text": context_text.clone()
                }]
            });
            // REF-204: gate the additive structuredContent for A/B measurement,
            // keeping content[text] byte-identical (see structured_content_enabled).
            if structured_content_enabled() {
                result["structuredContent"] = json!({ "context": context_text });
            }
            Ok(result)
        }
        "check_index_status" => {
            let cache = CacheManager::new(".");

            if !cache.exists() {
                let result = json!({
                    "status": "missing",
                    "action_required": "rfx index"
                });
                return Ok(make_tool_result(result));
            }

            let engine = QueryEngine::new(cache);
            let (status, _can_trust, warning) = engine.get_index_status()?;

            let status_str = match status {
                IndexStatus::Fresh => "fresh",
                IndexStatus::Stale => "stale",
            };

            let result = if let Some(w) = warning {
                let mut obj = json!({
                    "status": status_str,
                    "reason": w.reason,
                    "action_required": w.action_required
                });
                if let Some(fm) = w.files_modified {
                    obj["files_modified"] = json!(fm);
                }
                obj
            } else {
                json!({ "status": status_str })
            };

            Ok(make_tool_result(result))
        }
        "find_references" => {
            let pattern = arguments["pattern"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing pattern"))?
                .to_string();

            let lang = arguments["lang"].as_str().map(|s| s.to_string());
            let kind = arguments["kind"].as_str().map(|s| s.to_string());
            let limit = arguments["limit"].as_u64().map(|n| n as usize);
            let offset = arguments["offset"].as_u64().map(|n| n as usize);
            let glob_patterns: Vec<String> = arguments["glob"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let exclude_patterns: Vec<String> = arguments["exclude"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let force = arguments["force"].as_bool().unwrap_or(false);
            let include_strings = arguments["include_strings"].as_bool().unwrap_or(false);

            let language = parse_language(lang);
            let parsed_kind = parse_symbol_kind(kind);

            let mode = arguments["mode"].as_str().unwrap_or("list");

            // Count mode: skip the definition lookup and just count textual references.
            if mode == "count" {
                let count_filter = QueryFilter {
                    language,
                    kind: None,
                    use_ast: false,
                    use_regex: false,
                    limit: None, // count everything
                    symbols_mode: false,
                    expand: false,
                    file_pattern: None,
                    exact: false,
                    use_contains: false,
                    timeout_secs: 30,
                    glob_patterns,
                    exclude_patterns,
                    paths_only: false,
                    offset: None,
                    force,
                    suppress_output: true,
                    include_dependencies: false,
                    ..Default::default()
                };
                let cache = CacheManager::new(".");
                let engine = QueryEngine::new(cache);
                let response = engine.search_with_metadata(&pattern, count_filter)?;
                let result = json!({"count": response.pagination.total, "pattern": pattern});
                return Ok(make_tool_result(result));
            }

            // Search 1: Find symbol definition (symbols_mode=true, small cap)
            let def_filter = QueryFilter {
                language,
                kind: parsed_kind,
                use_ast: false,
                use_regex: false,
                limit: Some(5),
                symbols_mode: true,
                expand: false,
                file_pattern: None,
                exact: false,
                use_contains: false,
                timeout_secs: 30,
                glob_patterns: glob_patterns.clone(),
                exclude_patterns: exclude_patterns.clone(),
                paths_only: false,
                offset: None,
                force,
                suppress_output: true,
                include_dependencies: false,
                ..Default::default()
            };

            let cache = CacheManager::new(".");
            let engine = QueryEngine::new(cache);
            let def_response = engine.search_with_metadata(&pattern, def_filter)?;

            // Extract first definition as a compact object (reuse MatchResult's Serialize impl)
            let definition: Option<serde_json::Value> =
                def_response.results.first().and_then(|fg| {
                    fg.matches.first().map(|m| {
                        let mut def_obj = serde_json::to_value(m).unwrap_or(json!({}));
                        if let serde_json::Value::Object(ref mut map) = def_obj {
                            map.insert("path".to_string(), json!(fg.path.clone()));
                            // Truncate preview if present
                            if let Some(preview) = map.get("preview").and_then(|v| v.as_str()) {
                                let truncated = crate::cli::truncate_preview(
                                    preview,
                                    DEFAULT_MCP_PREVIEW_LENGTH,
                                );
                                map.insert("preview".to_string(), json!(truncated));
                            }
                        }
                        def_obj
                    })
                });

            // Search 2: Find all textual references (symbols_mode=false)
            let ref_filter = QueryFilter {
                language,
                kind: None,
                use_ast: false,
                use_regex: false,
                // REF-191: default to the one-call page size so "find all callers"
                // returns the full set instead of paginating at 50.
                limit: limit.map(|l| l.min(500)).or(Some(DEFAULT_MCP_RESULT_LIMIT)),
                symbols_mode: false,
                expand: false,
                file_pattern: None,
                exact: false,
                use_contains: false,
                timeout_secs: 30,
                glob_patterns,
                exclude_patterns,
                paths_only: false,
                offset,
                force,
                suppress_output: true,
                include_dependencies: false,
                ..Default::default()
            };

            let ref_response = engine.search_with_metadata(&pattern, ref_filter)?;

            // Flatten references to compact {path, line, preview} array,
            // excluding matches inside string literals or comments (unless include_strings).
            let references: Vec<serde_json::Value> = ref_response.results.iter()
                .flat_map(|fg| {
                    fg.matches.iter()
                        .filter(|m| {
                            include_strings
                                || !is_in_string_or_comment(fg.language, &m.preview, &pattern)
                        })
                        .map(move |m| {
                            json!({
                                "path": fg.path,
                                "line": m.span.start_line,
                                "preview": crate::cli::truncate_preview(&m.preview, DEFAULT_MCP_PREVIEW_LENGTH)
                            })
                        })
                })
                .collect();

            let total_references = references.len();
            let has_more = ref_response.pagination.has_more;
            let returned_count = references.len();

            let response = json!({
                "status": ref_response.status,
                "definition": definition,
                "references": references,
                "total_references": total_references,
                "total_count": total_references,
                "returned_count": returned_count,
                "has_more": has_more,
                "pagination": ref_response.pagination,
            });

            Ok(make_tool_result(response))
        }
        _ => Err(anyhow::anyhow!("Unknown tool: {}", name)),
    }
}

/// Process a single JSON-RPC request
fn process_request(request: JsonRpcRequest, enable_structural: bool) -> JsonRpcResponse {
    log::debug!("MCP request: method={}", request.method);

    let result = match request.method.as_str() {
        "initialize" => handle_initialize(request.params),
        "tools/list" => handle_list_tools(request.params, enable_structural),
        "tools/call" => handle_call_tool(request.params),
        _ => Err(anyhow::anyhow!("Unknown method: {}", request.method)),
    };

    match result {
        Ok(value) => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: Some(value),
            error: None,
        },
        Err(e) => {
            log::error!("MCP error: {}", e);
            let msg = e.to_string();
            // REF-67: map to the correct JSON-RPC error code instead of always using -32603.
            let (code, kind, message) =
                if let Some(re) = e.downcast_ref::<crate::errors::ReflexError>() {
                    let code = match re {
                        crate::errors::ReflexError::QuerySyntaxError(_) => -32602, // Invalid params
                        _ => -32603,                                               // Internal error
                    };
                    (code, re.kind().to_string(), re.to_string())
                } else if msg.starts_with("Unknown method:") {
                    (-32601, "MethodNotFound".to_string(), msg)
                } else if msg.starts_with("Missing")
                    || msg.starts_with("Unknown tool:")
                    || msg.starts_with("Invalid or unsupported")
                {
                    (-32602, "InvalidParams".to_string(), msg)
                } else {
                    (-32603, "InternalError".to_string(), msg)
                };
            JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(JsonRpcError {
                    code,
                    message: message.clone(),
                    data: Some(json!({ "kind": kind })),
                }),
            }
        }
    }
}

/// Handle a JSON-RPC Notification. Notifications never receive a response.
fn handle_notification(method: &str, _params: Option<Value>) {
    match method {
        "notifications/initialized" | "notifications/cancelled" => {
            log::debug!("MCP notification: {}", method);
        }
        other => {
            log::debug!("MCP unknown notification: {}", other);
        }
    }
}

/// Run the MCP server on stdio.
pub fn run_mcp_server() -> Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    run_mcp_server_io(stdin.lock(), stdout.lock())
}

/// Run the MCP server reading JSON-RPC messages from `reader` and writing
/// responses to `writer`. Exposed at crate-level for integration tests.
pub fn run_mcp_server_io<R: BufRead, W: Write>(reader: R, writer: W) -> Result<()> {
    let mcp_config = load_mcp_config();
    run_mcp_server_io_impl(reader, writer, mcp_config.enable_structural_tools)
}

/// Inner server loop. Accepts `enable_structural` so tests can drive the flag
/// without touching the filesystem.
fn run_mcp_server_io_impl<R: BufRead, W: Write>(
    reader: R,
    mut writer: W,
    enable_structural: bool,
) -> Result<()> {
    log::info!("Starting Reflex MCP server on stdio");

    for line in reader.lines() {
        let line = line?;

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        log::debug!("MCP input: {}", line);

        // Parse JSON-RPC message
        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(req) => req,
            Err(e) => {
                log::error!("Failed to parse JSON-RPC request: {}", e);
                // REF-61: send -32700 Parse error instead of silently dropping.
                // Per JSON-RPC 2.0, use null id when the id cannot be determined.
                let error_response = JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: Some(Value::Null),
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("Parse error: {}", e),
                        data: None,
                    }),
                };
                let response_json = serde_json::to_string(&error_response)?;
                writeln!(writer, "{}", response_json)?;
                writer.flush()?;
                continue;
            }
        };

        // Notifications (no `id`) must not receive a response per JSON-RPC 2.0.
        if request.id.is_none() {
            handle_notification(&request.method, request.params);
            continue;
        }

        // Process request and write response
        let response = process_request(request, enable_structural);
        let response_json = serde_json::to_string(&response)?;
        writeln!(writer, "{}", response_json)?;
        writer.flush()?;

        log::debug!("MCP output: {}", response_json);
    }

    log::info!("Reflex MCP server stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn call_server(input: &str) -> String {
        call_server_with_structural(input, true)
    }

    fn call_server_with_structural(input: &str, enable_structural: bool) -> String {
        let reader = Cursor::new(input.as_bytes());
        let mut output = Vec::new();
        run_mcp_server_io_impl(reader, &mut output, enable_structural).unwrap();
        String::from_utf8(output).unwrap()
    }

    fn parse_first_response(raw: &str) -> serde_json::Value {
        let line = raw.lines().next().expect("no response line");
        serde_json::from_str(line).expect("invalid JSON response")
    }

    // REF-61: malformed JSON must return -32700 Parse error (not silence)
    #[test]
    fn test_parse_error_returns_32700() {
        let raw = call_server("not-valid-json\n");
        let resp = parse_first_response(&raw);
        assert_eq!(resp["jsonrpc"], "2.0");
        assert_eq!(resp["error"]["code"], -32700);
        // Per JSON-RPC 2.0, id must be null when id cannot be determined
        assert!(resp["id"].is_null(), "id must be null for parse errors");
    }

    // REF-67: unknown method returns -32601 (Method not found)
    #[test]
    fn test_unknown_method_returns_32601() {
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"no_such_method","params":null}"#;
        let raw = call_server(&format!("{}\n", req));
        let resp = parse_first_response(&raw);
        assert_eq!(resp["error"]["code"], -32601);
    }

    // REF-67: missing required param returns -32602 (Invalid params), not -32603
    #[test]
    fn test_missing_param_returns_32602() {
        // search_code requires "pattern" — omit it
        let req = r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"search_code","arguments":{}}}"#;
        let raw = call_server(&format!("{}\n", req));
        let resp = parse_first_response(&raw);
        assert_eq!(resp["error"]["code"], -32602);
    }

    // Notification (no id) must produce no response
    #[test]
    fn test_notification_produces_no_response() {
        let notif = r#"{"jsonrpc":"2.0","method":"notifications/initialized","params":null}"#;
        let raw = call_server(&format!("{}\n", notif));
        assert!(
            raw.trim().is_empty(),
            "notification must not get a response"
        );
    }

    // REF-197: initialize handshake must carry the MCP `instructions` field that
    // nudges clients to prefer reflex tools (moved out of the per-repo CLAUDE.md).
    #[test]
    fn test_initialize_includes_instructions() {
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}"#;
        let raw = call_server(&format!("{}\n", req));
        let resp = parse_first_response(&raw);

        let instructions = resp["result"]["instructions"]
            .as_str()
            .expect("instructions must be a string");
        assert!(!instructions.is_empty(), "instructions must be non-empty");
        assert!(
            instructions.contains("mcp__reflex__"),
            "instructions must reference the mcp__reflex__ tool prefix"
        );
        assert!(
            instructions.contains("index_project"),
            "instructions must mention the index_project recovery step"
        );

        // Pre-existing handshake fields must be unchanged.
        assert_eq!(resp["result"]["protocolVersion"], "2025-11-25");
        assert_eq!(resp["result"]["serverInfo"]["name"], "reflex");
    }

    // REF-202: make_tool_result must emit BOTH the legacy content[text] JSON
    // string (unchanged) and the 2025-11-25 structuredContent object (same data).
    #[test]
    fn test_make_tool_result_dual_shape() {
        let data = json!({"status": "fresh", "count": 3});
        let result = super::make_tool_result(data.clone());

        // structuredContent carries the native object verbatim.
        assert_eq!(result["structuredContent"], data);

        // content[text] is the same data serialized as a JSON string, so existing
        // clients that only read content[text] keep getting identical output.
        assert_eq!(result["content"][0]["type"], "text");
        let text = result["content"][0]["text"]
            .as_str()
            .expect("content[0].text must be a string");
        let roundtrip: serde_json::Value =
            serde_json::from_str(text).expect("content[text] must be valid JSON");
        assert_eq!(roundtrip, data);
    }

    // REF-204: the structuredContent field is a runtime A/B toggle. With it
    // disabled, content[text] must be byte-identical to the enabled shape and
    // structuredContent must be absent (reproducing pre-REF-202 output). Tested
    // via the inner builder so no process-global env is mutated (race-free under
    // cargo's parallel test threads).
    #[test]
    fn test_make_tool_result_toggle_off_omits_structured_content() {
        let data = json!({"status": "fresh", "count": 3});
        let on = super::make_tool_result_with(data.clone(), true, false);
        let off = super::make_tool_result_with(data.clone(), false, false);

        // Enabled shape matches the shipped REF-202 default.
        assert_eq!(on["structuredContent"], data);

        // Disabled shape drops structuredContent entirely...
        assert!(off.get("structuredContent").is_none());
        // ...but leaves content[text] byte-identical, so the only measured
        // difference between the B and B_sc arms is the extra field.
        assert_eq!(off["content"], on["content"]);
    }

    // REF-196 Stage 2: brief summary in content[text], full data in structuredContent.
    #[test]
    fn test_make_tool_result_stage2_summary() {
        let data = json!({
            "status": "ok",
            "pagination": {"total": 42, "count": 42, "offset": 0, "limit": 200, "has_more": false},
            "results": [{"path": "a.rs"}, {"path": "b.rs"}],
            "total_count": 42,
            "returned_count": 42
        });
        let stage2 = super::make_tool_result_with(data.clone(), true, true);
        // structuredContent still holds the full data
        assert_eq!(stage2["structuredContent"], data);
        // content[text] is a brief summary, not the full JSON blob
        let text = stage2["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("42"),
            "summary should mention match count: {}",
            text
        );
        assert!(
            text.contains("structuredContent"),
            "summary should reference structuredContent: {}",
            text
        );
        assert!(
            text.len() < 120,
            "summary should be brief (< 120 chars): {}",
            text
        );
    }

    // REF-204: default (env unset) must keep structuredContent enabled so the
    // shipped contract is unchanged unless a caller explicitly opts out.
    #[test]
    fn test_structured_content_enabled_default_on() {
        // NOTE: relies on REFLEX_MCP_STRUCTURED_CONTENT being unset in the test
        // environment (the harness never sets it); asserts the default only.
        assert!(super::structured_content_enabled());
    }

    // REF-209: columnar output is the shipped default unless a caller opts out.
    #[test]
    fn test_columnar_enabled_default_on() {
        // Relies on REFLEX_MCP_COLUMNAR being unset in the harness environment.
        assert!(super::columnar_enabled());
    }

    /// A representative two-file list-mode search response (post-flattening),
    /// mixing a symbol match and a plain-text match.
    fn sample_search_response() -> serde_json::Value {
        json!({
            "status": "fresh",
            "pagination": {"total": 2, "count": 2, "offset": 0, "limit": 200, "has_more": false},
            "results": [
                {
                    "path": "src/mcp.rs",
                    "language": "rust",
                    "matches": [
                        {"kind": "Function", "symbol": "make_tool_result",
                         "span": {"start_line": 955, "end_line": 957}, "preview": "fn make_tool_result"}
                    ]
                },
                {
                    "path": "src/query.rs",
                    "language": "rust",
                    "matches": [
                        {"span": {"start_line": 10, "end_line": 10}, "preview": "let x = 1;"},
                        {"span": {"start_line": 20, "end_line": 20}, "preview": "let y = 2;"}
                    ]
                }
            ],
            "total_count": 2,
            "returned_count": 2,
            "has_more": false
        })
    }

    // REF-209: `results` array is replaced by columns/rows; one row per match.
    #[test]
    fn test_to_columnar_reshapes_results() {
        let out = super::to_columnar(sample_search_response());

        // `results` is gone, replaced by `columns` + `rows`.
        assert!(out.get("results").is_none(), "results must be removed");
        let columns = out["columns"].as_array().expect("columns array");
        let rows = out["rows"].as_array().expect("rows array");

        // The five base columns always lead; kind/symbol appended because one
        // match carries them. No context columns (none present).
        let col_names: Vec<&str> = columns.iter().filter_map(|c| c.as_str()).collect();
        assert_eq!(
            col_names,
            vec![
                "path",
                "language",
                "start_line",
                "end_line",
                "preview",
                "kind",
                "symbol"
            ]
        );

        // One row per match across all files (1 + 2 = 3).
        assert_eq!(rows.len(), 3);

        // Symbol row is fully populated and positionally aligned to columns.
        assert_eq!(
            rows[0],
            json!([
                "src/mcp.rs",
                "rust",
                955,
                957,
                "fn make_tool_result",
                "Function",
                "make_tool_result"
            ])
        );
        // Plain-text rows carry null in the trailing optional columns.
        assert_eq!(
            rows[1],
            json!(["src/query.rs", "rust", 10, 10, "let x = 1;", null, null])
        );
        assert_eq!(
            rows[2],
            json!(["src/query.rs", "rust", 20, 20, "let y = 2;", null, null])
        );
    }

    // REF-209: top-level metadata (status/pagination/scalars) survives the reshape.
    #[test]
    fn test_to_columnar_preserves_metadata() {
        let out = super::to_columnar(sample_search_response());
        assert_eq!(out["status"], "fresh");
        assert_eq!(out["total_count"], 2);
        assert_eq!(out["returned_count"], 2);
        assert_eq!(out["has_more"], false);
        assert_eq!(out["pagination"]["total"], 2);
        assert_eq!(out["pagination"]["limit"], 200);
    }

    // REF-209: a pure full-text result (no kind/symbol/context) stays at the five
    // base columns — no all-null padding claws back the token saving.
    #[test]
    fn test_to_columnar_omits_absent_optional_columns() {
        let data = json!({
            "status": "fresh",
            "pagination": {"total": 1, "count": 1, "offset": 0, "limit": 200, "has_more": false},
            "results": [{
                "path": "a.rs", "language": "rust",
                "matches": [{"span": {"start_line": 1, "end_line": 1}, "preview": "struct X"}]
            }],
            "total_count": 1, "returned_count": 1, "has_more": false
        });
        let out = super::to_columnar(data);
        let col_names: Vec<&str> = out["columns"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|c| c.as_str())
            .collect();
        assert_eq!(
            col_names,
            vec!["path", "language", "start_line", "end_line", "preview"]
        );
        assert_eq!(out["rows"][0], json!(["a.rs", "rust", 1, 1, "struct X"]));
    }

    // REF-209: context columns appear only when a match carries context lines.
    #[test]
    fn test_to_columnar_includes_context_columns_when_present() {
        let data = json!({
            "status": "fresh",
            "pagination": {"total": 1, "count": 1, "offset": 0, "limit": 200, "has_more": false},
            "results": [{
                "path": "a.rs", "language": "rust",
                "matches": [{
                    "span": {"start_line": 5, "end_line": 5}, "preview": "hit",
                    "context_before": ["above"], "context_after": ["below"]
                }]
            }],
            "total_count": 1, "returned_count": 1, "has_more": false
        });
        let out = super::to_columnar(data);
        let col_names: Vec<&str> = out["columns"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|c| c.as_str())
            .collect();
        assert_eq!(
            col_names,
            vec![
                "path",
                "language",
                "start_line",
                "end_line",
                "preview",
                "context_before",
                "context_after"
            ]
        );
        assert_eq!(
            out["rows"][0],
            json!(["a.rs", "rust", 5, 5, "hit", ["above"], ["below"]])
        );
    }

    // REF-209: count-mode / non-results shapes pass through untouched, so
    // `to_columnar` is safe to call on any success value.
    #[test]
    fn test_to_columnar_passthrough_non_results() {
        let count = json!({"count": 7, "pattern": "foo"});
        assert_eq!(super::to_columnar(count.clone()), count);

        let scalar = json!("not an object");
        assert_eq!(super::to_columnar(scalar.clone()), scalar);
    }

    // REF-200: tool schemas must advertise the correct default limit (200, raised from 50 in REF-191) and max cap (500)
    #[test]
    fn test_tool_schema_limit_defaults() {
        let req = r#"{"jsonrpc":"2.0","id":5,"method":"tools/list","params":null}"#;
        let raw = call_server(&format!("{}\n", req));
        let resp = parse_first_response(&raw);
        let tools = resp["result"]["tools"].as_array().expect("tools array");

        let find_tool = |name: &str| {
            tools
                .iter()
                .find(|t| t["name"] == name)
                .unwrap_or_else(|| panic!("tool '{}' not found", name))
        };

        for tool_name in &["search_code", "search_regex", "find_references"] {
            let tool = find_tool(tool_name);
            let limit_desc = tool["inputSchema"]["properties"]["limit"]["description"]
                .as_str()
                .unwrap_or_else(|| panic!("{}: missing limit description", tool_name));
            assert!(
                limit_desc.contains("200"),
                "{}: limit description should mention default 200, got: {}",
                tool_name,
                limit_desc
            );
            assert!(
                limit_desc.contains("500"),
                "{}: limit description should mention max 500, got: {}",
                tool_name,
                limit_desc
            );
        }
    }

    // REF-189: structural tools absent when enable_structural_tools = false
    #[test]
    fn test_structural_tools_gated_by_flag() {
        const STRUCTURAL: &[&str] = &[
            "find_circular",
            "find_islands",
            "find_unused",
            "analyze_summary",
            "get_transitive_deps",
        ];
        const ALWAYS_ON: &[&str] = &["search_code", "find_hotspots", "get_dependencies"];

        let req = r#"{"jsonrpc":"2.0","id":10,"method":"tools/list","params":null}"#;

        // Default (structural enabled): all 5 structural tools present
        let raw_on = call_server_with_structural(&format!("{}\n", req), true);
        let resp_on = parse_first_response(&raw_on);
        let tools_on = resp_on["result"]["tools"].as_array().expect("tools array");
        let names_on: Vec<&str> = tools_on.iter().filter_map(|t| t["name"].as_str()).collect();
        for name in STRUCTURAL {
            assert!(
                names_on.contains(name),
                "structural tool '{}' should appear when flag=true",
                name
            );
        }

        // Disabled: structural tools absent, day-to-day tools still present
        let raw_off = call_server_with_structural(&format!("{}\n", req), false);
        let resp_off = parse_first_response(&raw_off);
        let tools_off = resp_off["result"]["tools"].as_array().expect("tools array");
        let names_off: Vec<&str> = tools_off
            .iter()
            .filter_map(|t| t["name"].as_str())
            .collect();
        for name in STRUCTURAL {
            assert!(
                !names_off.contains(name),
                "structural tool '{}' must be absent when flag=false",
                name
            );
        }
        for name in ALWAYS_ON {
            assert!(
                names_off.contains(name),
                "always-on tool '{}' must remain when flag=false",
                name
            );
        }
    }

    // REF-186: find_references should filter string/comment matches by default
    #[test]
    fn test_is_in_string_or_comment_filters_comment() {
        // Pattern inside a Rust single-line comment should be filtered
        let line = "let x = 5; // extract_symbols here";
        assert!(
            super::is_in_string_or_comment(crate::models::Language::Rust, line, "extract_symbols"),
            "pattern in comment should be classified as non-code"
        );
    }

    #[test]
    fn test_is_in_string_or_comment_filters_string_literal() {
        // Pattern inside a string literal should be filtered
        let line = r#"let s = "extract_symbols";"#;
        assert!(
            super::is_in_string_or_comment(crate::models::Language::Rust, line, "extract_symbols"),
            "pattern in string literal should be classified as non-code"
        );
    }

    #[test]
    fn test_is_in_string_or_comment_keeps_real_code() {
        // Pattern in real code should NOT be filtered
        let line = "fn extract_symbols(source: &str) -> Vec<SearchResult> {";
        assert!(
            !super::is_in_string_or_comment(crate::models::Language::Rust, line, "extract_symbols"),
            "real function name should not be classified as non-code"
        );
    }

    #[test]
    fn test_is_in_string_or_comment_mixed_line_keeps_match() {
        // When a line has the pattern both in a string AND in real code, the match
        // should be kept (conservative: real code occurrence wins)
        let line = r#"let _s = "extract_symbols"; extract_symbols(data);"#;
        assert!(
            !super::is_in_string_or_comment(crate::models::Language::Rust, line, "extract_symbols"),
            "when pattern appears in code on the same line, match should be kept"
        );
    }

    #[test]
    fn test_is_in_string_or_comment_unknown_language_keeps_match() {
        // Unknown language has no filter — always keep the match (conservative)
        let line = "extract_symbols in some unknown syntax";
        assert!(
            !super::is_in_string_or_comment(
                crate::models::Language::Unknown,
                line,
                "extract_symbols"
            ),
            "unknown language should never filter matches"
        );
    }

    #[test]
    fn test_find_references_schema_has_include_strings() {
        let tools_json =
            call_server(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":null}"#);
        let resp = parse_first_response(&tools_json);
        let tools = resp["result"]["tools"].as_array().expect("tools array");
        let find_refs = tools
            .iter()
            .find(|t| t["name"] == "find_references")
            .expect("find_references tool");
        let props = &find_refs["inputSchema"]["properties"];
        assert!(
            !props["include_strings"].is_null(),
            "find_references inputSchema must expose include_strings parameter"
        );
    }

    // REF-187: search_code, search_regex, and find_references must expose mode parameter
    #[test]
    fn test_count_mode_schema_exposed_on_search_tools() {
        let raw = call_server(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":null}"#);
        let resp = parse_first_response(&raw);
        let tools = resp["result"]["tools"].as_array().expect("tools array");

        for tool_name in &["search_code", "search_regex", "find_references"] {
            let tool = tools
                .iter()
                .find(|t| t["name"] == *tool_name)
                .unwrap_or_else(|| panic!("tool '{}' not found", tool_name));
            let props = &tool["inputSchema"]["properties"];
            assert!(
                !props["mode"].is_null(),
                "'{}' inputSchema must expose 'mode' parameter (REF-187)",
                tool_name
            );
            let enum_vals = props["mode"]["enum"]
                .as_array()
                .unwrap_or_else(|| panic!("'{}' mode must have enum values", tool_name));
            let vals: Vec<&str> = enum_vals.iter().filter_map(|v| v.as_str()).collect();
            assert!(
                vals.contains(&"count") && vals.contains(&"list"),
                "'{}' mode enum must contain 'count' and 'list', got {:?}",
                tool_name,
                vals
            );
        }
    }

    // REF-187: count mode must return {count, pattern} without match bodies
    // This test verifies the handler shape via a missing-index path (schema-level only,
    // since integration tests with a real index live in tests/).
    #[test]
    fn test_count_mode_missing_required_param_still_returns_32602() {
        // Ensure count mode parsing doesn't interfere with required param validation.
        let req = r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"search_code","arguments":{"mode":"count"}}}"#;
        let raw = call_server(&format!("{}\n", req));
        let resp = parse_first_response(&raw);
        // Missing pattern → must still return InvalidParams (-32602), not a crash
        assert_eq!(
            resp["error"]["code"], -32602,
            "count mode must not bypass required-param validation"
        );
    }
}
