//! Schema-drift prevention for MCP `outputSchema` (REF-203 / REF-196 Phase 3).
//!
//! Each priority tool declares an `outputSchema` in `handle_list_tools`. This
//! test makes REAL `tools/call` requests against a throwaway indexed fixture and
//! validates the returned `structuredContent` against the tool's declared
//! `outputSchema`. If the response shape ever drifts from the declared schema
//! (a renamed field, a dropped key, a changed type), this test fails.
//!
//! Only one test here (`structured_content_matches_declared_output_schemas`)
//! calls `set_current_dir` (process-global) so the MCP handlers'
//! `CacheManager::new(".")` resolve to the fixture. The second test
//! (`validator_rejects_drift`) only issues a `tools/list` call, which does not
//! touch the index and is therefore cwd-independent, so the two can run
//! concurrently without racing. Integration-test files run as separate
//! processes, so this chdir cannot affect any other test file.

use reflex::mcp::run_mcp_server_io;
use reflex::{CacheManager, IndexConfig, Indexer};
use serde_json::{Value, json};
use std::io::Cursor;

/// Send one JSON-RPC request through the server and return the parsed result.
fn call(request: &str) -> Value {
    let reader = Cursor::new(format!("{request}\n").into_bytes());
    let mut output: Vec<u8> = Vec::new();
    run_mcp_server_io(reader, &mut output).expect("server should not error");
    let raw = String::from_utf8(output).expect("utf8");
    let line = raw.lines().next().expect("one response line");
    serde_json::from_str(line).expect("valid JSON response")
}

/// Make a `tools/call` and return the `structuredContent` object.
fn call_tool(name: &str, args: Value) -> Value {
    let req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": { "name": name, "arguments": args }
    });
    let resp = call(&req.to_string());
    assert!(
        resp.get("error").is_none(),
        "tool '{name}' returned an error: {resp}"
    );
    resp["result"]["structuredContent"].clone()
}

/// Minimal recursive JSON-Schema validator supporting the exact subset the MCP
/// server emits: `type` (string or `[string,...]`), `properties`, `required`,
/// `items`, `enum`, and `oneOf`. Additional properties are always permitted
/// (forward-compatible). Returns `Err(path-qualified message)` on the first
/// violation.
///
/// This is deliberately hand-rolled rather than pulling in a `jsonschema`
/// dependency: it keeps the drift test fully offline (Reflex is local-first) and
/// covers only the vocabulary we control. Its correctness is pinned by the
/// negative self-checks in `validator_rejects_drift`.
fn validate(schema: &Value, data: &Value, path: &str) -> Result<(), String> {
    if let Some(t) = schema.get("type") {
        let ok = match t {
            Value::String(s) => type_matches(s, data),
            Value::Array(arr) => arr
                .iter()
                .any(|x| x.as_str().is_some_and(|s| type_matches(s, data))),
            _ => true,
        };
        if !ok {
            return Err(format!("{path}: expected type {t}, got {data}"));
        }
    }

    if let Some(Value::Array(allowed)) = schema.get("enum")
        && !allowed.iter().any(|v| v == data)
    {
        return Err(format!("{path}: value {data} not in enum {allowed:?}"));
    }

    if let Some(Value::Array(branches)) = schema.get("oneOf") {
        let matched = branches
            .iter()
            .filter(|b| validate(b, data, path).is_ok())
            .count();
        if matched != 1 {
            return Err(format!(
                "{path}: oneOf matched {matched} branches (expected exactly 1)"
            ));
        }
    }

    if data.is_object() {
        if let Some(Value::Array(required)) = schema.get("required") {
            for r in required {
                if let Some(key) = r.as_str()
                    && data.get(key).is_none()
                {
                    return Err(format!("{path}: missing required property '{key}'"));
                }
            }
        }
        if let Some(Value::Object(props)) = schema.get("properties") {
            for (key, subschema) in props {
                if let Some(child) = data.get(key) {
                    validate(subschema, child, &format!("{path}.{key}"))?;
                }
            }
        }
    }

    if let (Some(items), Some(arr)) = (schema.get("items"), data.as_array()) {
        for (i, elem) in arr.iter().enumerate() {
            validate(items, elem, &format!("{path}[{i}]"))?;
        }
    }

    Ok(())
}

fn type_matches(ty: &str, v: &Value) -> bool {
    match ty {
        "object" => v.is_object(),
        "array" => v.is_array(),
        "string" => v.is_string(),
        "integer" => v.is_i64() || v.is_u64(),
        "number" => v.is_number(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        _ => true, // unknown type keyword: don't fail the whole validation
    }
}

/// Fetch the map of tool name -> outputSchema from `tools/list`.
fn output_schemas() -> std::collections::HashMap<String, Value> {
    let resp = call(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":null}"#);
    let tools = resp["result"]["tools"].as_array().expect("tools array");
    tools
        .iter()
        .filter_map(|t| {
            let name = t["name"].as_str()?.to_string();
            let schema = t.get("outputSchema")?.clone();
            Some((name, schema))
        })
        .collect()
}

/// Build a throwaway indexed fixture and chdir into it so the MCP handlers'
/// `CacheManager::new(".")` finds it. Uses a tempdir so no git-tracked file is
/// touched and the fixture content is fully deterministic.
fn setup_fixture() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().to_path_buf();
    std::fs::write(
        root.join("sample.rs"),
        "pub struct Widget {\n    pub id: u32,\n}\n\n\
         pub struct Gadget {\n    pub name: String,\n}\n\n\
         pub fn make_widget() -> Widget {\n    Widget { id: 1 }\n}\n",
    )
    .expect("write fixture source");

    let cache = CacheManager::new(&root);
    let indexer = Indexer::new(cache, IndexConfig::default());
    indexer.index(&root, true).expect("index fixture");

    // Keep the tempdir alive for the process lifetime; chdir into it.
    std::env::set_current_dir(&root).expect("chdir into fixture");
    std::mem::forget(tmp);
}

#[test]
fn structured_content_matches_declared_output_schemas() {
    setup_fixture();
    let schemas = output_schemas();

    // Acceptance criterion: all six priority tools declare an outputSchema.
    for tool in [
        "search_code",
        "search_regex",
        "find_references",
        "gather_context",
        "list_locations",
        "count_occurrences",
    ] {
        assert!(
            schemas.contains_key(tool),
            "tool '{tool}' must declare an outputSchema"
        );
    }

    // Acceptance criterion: the search_code pagination limit default is 200
    // (DEFAULT_MCP_RESULT_LIMIT, per REF-191/REF-200 — never 50).
    let limit_default = &schemas["search_code"]["oneOf"][0]["properties"]["pagination"]["properties"]
        ["limit"]["default"];
    assert_eq!(
        limit_default,
        &json!(200),
        "search_code outputSchema pagination.limit default must be 200, got {limit_default}"
    );

    // --- search_code, list mode: primary drift check ------------------------
    // REF-209: list mode returns the columnar {columns, rows} shape by default.
    let sc = call_tool("search_code", json!({ "pattern": "struct" }));
    validate(&schemas["search_code"], &sc, "search_code")
        .expect("search_code structuredContent must match its outputSchema");
    // The fixture defines two structs, so rows must be non-empty — this exercises
    // the columnar columns/rows schema, not just the envelope.
    assert!(
        sc["total_count"].as_u64().unwrap_or(0) >= 1,
        "expected matches for 'struct' in fixture, got {sc}"
    );
    let columns: Vec<&str> = sc["columns"]
        .as_array()
        .expect("columnar response must carry a columns array")
        .iter()
        .filter_map(|c| c.as_str())
        .collect();
    assert_eq!(
        &columns[..5],
        &["path", "language", "start_line", "end_line", "preview"],
        "columnar header must begin with the five base columns, got {columns:?}"
    );
    assert!(
        sc["rows"]
            .as_array()
            .map(|a| !a.is_empty())
            .unwrap_or(false),
        "expected non-empty rows array to exercise columnar schema, got {sc}"
    );

    // --- search_code, count mode: proves the oneOf count branch -------------
    let sc_count = call_tool(
        "search_code",
        json!({ "pattern": "struct", "mode": "count" }),
    );
    validate(&schemas["search_code"], &sc_count, "search_code(count)")
        .expect("search_code count-mode structuredContent must match its outputSchema");
    assert!(
        sc_count.get("count").is_some(),
        "count mode must return count"
    );

    // --- search_regex, list mode -------------------------------------------
    let sr = call_tool("search_regex", json!({ "pattern": "struct" }));
    validate(&schemas["search_regex"], &sr, "search_regex")
        .expect("search_regex structuredContent must match its outputSchema");

    // --- find_references (definition is null for the keyword 'struct') ------
    let fr = call_tool("find_references", json!({ "pattern": "Widget" }));
    validate(&schemas["find_references"], &fr, "find_references")
        .expect("find_references structuredContent must match its outputSchema");

    // --- list_locations -----------------------------------------------------
    let ll = call_tool("list_locations", json!({ "pattern": "struct" }));
    validate(&schemas["list_locations"], &ll, "list_locations")
        .expect("list_locations structuredContent must match its outputSchema");

    // --- count_occurrences --------------------------------------------------
    let co = call_tool("count_occurrences", json!({ "pattern": "struct" }));
    validate(&schemas["count_occurrences"], &co, "count_occurrences")
        .expect("count_occurrences structuredContent must match its outputSchema");

    // --- gather_context -----------------------------------------------------
    let gc = call_tool("gather_context", json!({}));
    validate(&schemas["gather_context"], &gc, "gather_context")
        .expect("gather_context structuredContent must match its outputSchema");
}

/// Prove the hand-rolled validator actually rejects drift — otherwise a passing
/// positive check above could be meaningless (a validator that accepts anything).
#[test]
fn validator_rejects_drift() {
    // A well-formed columnar list-mode envelope (REF-209).
    let good = json!({
        "status": "fresh",
        "pagination": { "total": 1, "count": 1, "offset": 0, "limit": 200, "has_more": false },
        "columns": ["path", "language", "start_line", "end_line", "preview"],
        "rows": [["a.rs", "rust", 1, 1, "struct X"]],
        "total_count": 1,
        "returned_count": 1,
        "has_more": false
    });

    // Reconstruct the same schema the server would emit for search_code by
    // asking the running server for it (guarantees the test tracks the source).
    // We build a fixture-free tools/list call — it does not need an index.
    let resp = call(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":null}"#);
    let tools = resp["result"]["tools"].as_array().expect("tools array");
    let schema = tools
        .iter()
        .find(|t| t["name"] == "search_code")
        .and_then(|t| t.get("outputSchema"))
        .expect("search_code outputSchema")
        .clone();

    // Sanity: the good envelope validates.
    validate(&schema, &good, "good").expect("well-formed envelope must validate");

    // Drift 1: drop a required top-level field -> both oneOf branches fail.
    let mut missing_rows = good.clone();
    missing_rows.as_object_mut().unwrap().remove("rows");
    assert!(
        validate(&schema, &missing_rows, "missing_rows").is_err(),
        "removing required 'rows' must fail validation"
    );

    // Drift 2: wrong type for a nested field (pagination.total as a string).
    let mut wrong_type = good.clone();
    wrong_type["pagination"]["total"] = json!("oops");
    assert!(
        validate(&schema, &wrong_type, "wrong_type").is_err(),
        "wrong nested type must fail validation"
    );

    // Drift 3: invalid enum value for status.
    let mut bad_enum = good.clone();
    bad_enum["status"] = json!("nonsense");
    assert!(
        validate(&schema, &bad_enum, "bad_enum").is_err(),
        "invalid status enum must fail validation"
    );
}

/// REF-210 regression: every declared `outputSchema` MUST have a root
/// `"type": "object"`.
///
/// The MCP spec requires `outputSchema` to describe the `structuredContent`
/// object, and Claude Code's client validates `outputSchema.type === "object"`
/// on the *entire* `tools/list` batch. A single tool whose root schema is a bare
/// `oneOf` (no `type`) makes the client reject the whole response, dropping every
/// Reflex tool while the server still reports `status: connected` — the 0-tools
/// failure that invalidated the REF-196 B_sc2 efficacy trials. This guards the
/// `oneOf`-rooted schemas (`search_code`, `search_regex`, `find_references`) that
/// caused it, plus any future tool.
#[test]
fn every_output_schema_root_is_object_typed() {
    let schemas = output_schemas();
    assert!(
        !schemas.is_empty(),
        "expected at least one tool to declare an outputSchema"
    );
    for (name, schema) in &schemas {
        assert_eq!(
            schema.get("type").and_then(|t| t.as_str()),
            Some("object"),
            "tool '{name}' outputSchema root must be `type: \"object\"` \
             (Claude Code rejects the entire tools/list batch otherwise — REF-210); \
             got root schema: {schema}"
        );
    }
}
