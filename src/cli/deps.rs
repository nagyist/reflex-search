use anyhow::Result;
use std::path::PathBuf;
use crate::cache::CacheManager;
use crate::output;


/// Handle the `analyze` subcommand
#[allow(clippy::too_many_arguments)]
pub(super) fn handle_analyze(
    circular: bool,
    hotspots: bool,
    min_dependents: usize,
    unused: bool,
    islands: bool,
    min_island_size: usize,
    max_island_size: Option<usize>,
    format: String,
    as_json: bool,
    pretty_json: bool,
    count_only: bool,
    all: bool,
    plain: bool,
    _glob_patterns: Vec<String>,
    _exclude_patterns: Vec<String>,
    _force: bool,
    limit: Option<usize>,
    offset: Option<usize>,
    sort: Option<String>,
) -> Result<()> {
    use crate::dependency::DependencyIndex;

    log::info!("Starting analyze command");

    let cache = CacheManager::new(".");

    if !cache.exists() {
        anyhow::bail!(
            "No index found in current directory.\n\
             \n\
             Run 'rfx index' to build the code search index first.\n\
             \n\
             Example:\n\
             $ rfx index             # Index current directory\n\
             $ rfx analyze           # Run dependency analysis"
        );
    }

    let deps_index = DependencyIndex::new(cache);

    // JSON mode overrides format
    let format = if as_json { "json" } else { &format };

    // Smart limit handling for analyze commands (default: 200 per page)
    let final_limit = if all {
        None  // --all means no limit
    } else if let Some(user_limit) = limit {
        Some(user_limit)  // Use user-specified limit
    } else {
        Some(200)  // Default: limit to 200 results per page for token efficiency
    };

    // If no specific flags, show summary
    if !circular && !hotspots && !unused && !islands {
        return handle_analyze_summary(&deps_index, min_dependents, count_only, as_json, pretty_json);
    }

    // Run specific analyses based on flags
    if circular {
        handle_deps_circular(&deps_index, format, pretty_json, final_limit, offset, count_only, plain, sort.clone())?;
    }

    if hotspots {
        handle_deps_hotspots(&deps_index, format, pretty_json, final_limit, offset, min_dependents, count_only, plain, sort.clone())?;
    }

    if unused {
        handle_deps_unused(&deps_index, format, pretty_json, final_limit, offset, count_only, plain)?;
    }

    if islands {
        handle_deps_islands(&deps_index, format, pretty_json, final_limit, offset, min_island_size, max_island_size, count_only, plain, sort.clone())?;
    }

    Ok(())
}


/// Handle analyze summary (default --analyze behavior)
fn handle_analyze_summary(
    deps_index: &crate::dependency::DependencyIndex,
    min_dependents: usize,
    count_only: bool,
    as_json: bool,
    pretty_json: bool,
) -> Result<()> {
    // Gather counts
    let cycles = deps_index.detect_circular_dependencies()?;
    let hotspots = deps_index.find_hotspots(None, min_dependents)?;
    let unused = deps_index.find_unused_files()?;
    let all_islands = deps_index.find_islands()?;

    if as_json {
        // JSON output
        let summary = serde_json::json!({
            "circular_dependencies": cycles.len(),
            "hotspots": hotspots.len(),
            "unused_files": unused.len(),
            "islands": all_islands.len(),
            "min_dependents": min_dependents,
        });

        let json_str = if pretty_json {
            serde_json::to_string_pretty(&summary)?
        } else {
            serde_json::to_string(&summary)?
        };
        println!("{}", json_str);
    } else if count_only {
        // Just show counts without any extra formatting
        println!("{} circular dependencies", cycles.len());
        println!("{} hotspots ({}+ dependents)", hotspots.len(), min_dependents);
        println!("{} unused files", unused.len());
        println!("{} islands", all_islands.len());
    } else {
        // Full summary with headers and suggestions
        println!("Dependency Analysis Summary\n");

        // Circular dependencies
        println!("Circular Dependencies: {} cycle(s)", cycles.len());

        // Hotspots
        println!("Hotspots: {} file(s) with {}+ dependents", hotspots.len(), min_dependents);

        // Unused
        println!("Unused Files: {} file(s)", unused.len());

        // Islands
        println!("Islands: {} disconnected component(s)", all_islands.len());

        println!("\nUse specific flags for detailed results:");
        println!("  rfx analyze --circular");
        println!("  rfx analyze --hotspots");
        println!("  rfx analyze --unused");
        println!("  rfx analyze --islands");
    }

    Ok(())
}


/// Handle the `deps` subcommand
pub(super) fn handle_deps(
    file: PathBuf,
    reverse: bool,
    depth: usize,
    format: String,
    as_json: bool,
    pretty_json: bool,
) -> Result<()> {
    use crate::dependency::DependencyIndex;

    log::info!("Starting deps command");

    let cache = CacheManager::new(".");

    if !cache.exists() {
        anyhow::bail!(
            "No index found in current directory.\n\
             \n\
             Run 'rfx index' to build the code search index first.\n\
             \n\
             Example:\n\
             $ rfx index          # Index current directory\n\
             $ rfx deps <file>    # Analyze dependencies"
        );
    }

    let deps_index = DependencyIndex::new(cache);

    // JSON mode overrides format
    let format = if as_json { "json" } else { &format };

    // Convert file path to string
    let file_str = file.to_string_lossy().to_string();

    // Get file ID
    let file_id = deps_index.get_file_id_by_path(&file_str)?
        .ok_or_else(|| anyhow::anyhow!("File '{}' not found in index", file_str))?;

    if reverse {
        // Show dependents (who imports this file)
        let dependents = deps_index.get_dependents(file_id)?;
        let paths = deps_index.get_file_paths(&dependents)?;

        match format.as_ref() {
            "json" => {
                let output: Vec<_> = dependents.iter()
                    .filter_map(|id| paths.get(id).map(|path| serde_json::json!({
                        "file_id": id,
                        "path": path,
                    })))
                    .collect();

                let json_str = if pretty_json {
                    serde_json::to_string_pretty(&output)?
                } else {
                    serde_json::to_string(&output)?
                };
                println!("{}", json_str);
                eprintln!("Found {} files that import {}", dependents.len(), file_str);
            }
            "tree" => {
                println!("Files that import {}:", file_str);
                for (id, path) in &paths {
                    if dependents.contains(id) {
                        println!("  └─ {}", path);
                    }
                }
                eprintln!("\nFound {} dependents", dependents.len());
            }
            "table" => {
                println!("ID     Path");
                println!("-----  ----");
                for id in &dependents {
                    if let Some(path) = paths.get(id) {
                        println!("{:<5}  {}", id, path);
                    }
                }
                eprintln!("\nFound {} dependents", dependents.len());
            }
            _ => {
                anyhow::bail!("Unknown format '{}'. Supported: json, tree, table, dot", format);
            }
        }
    } else {
        // Show dependencies (what this file imports)
        if depth == 1 {
            // Direct dependencies only
            let deps = deps_index.get_dependencies(file_id)?;

            match format.as_ref() {
                "json" => {
                    let output: Vec<_> = deps.iter()
                        .map(|dep| serde_json::json!({
                            "imported_path": dep.imported_path,
                            "resolved_file_id": dep.resolved_file_id,
                            "import_type": match dep.import_type {
                                crate::models::ImportType::Internal => "internal",
                                crate::models::ImportType::External => "external",
                                crate::models::ImportType::Stdlib => "stdlib",
                            },
                            "line": dep.line_number,
                            "symbols": dep.imported_symbols,
                        }))
                        .collect();

                    let json_str = if pretty_json {
                        serde_json::to_string_pretty(&output)?
                    } else {
                        serde_json::to_string(&output)?
                    };
                    println!("{}", json_str);
                    eprintln!("Found {} dependencies for {}", deps.len(), file_str);
                }
                "tree" => {
                    println!("Dependencies of {}:", file_str);
                    for dep in &deps {
                        let type_label = match dep.import_type {
                            crate::models::ImportType::Internal => "[internal]",
                            crate::models::ImportType::External => "[external]",
                            crate::models::ImportType::Stdlib => "[stdlib]",
                        };
                        println!("  └─ {} {} (line {})", dep.imported_path, type_label, dep.line_number);
                    }
                    eprintln!("\nFound {} dependencies", deps.len());
                }
                "table" => {
                    println!("Path                          Type       Line");
                    println!("----------------------------  ---------  ----");
                    for dep in &deps {
                        let type_str = match dep.import_type {
                            crate::models::ImportType::Internal => "internal",
                            crate::models::ImportType::External => "external",
                            crate::models::ImportType::Stdlib => "stdlib",
                        };
                        println!("{:<28}  {:<9}  {}", dep.imported_path, type_str, dep.line_number);
                    }
                    eprintln!("\nFound {} dependencies", deps.len());
                }
                _ => {
                    anyhow::bail!("Unknown format '{}'. Supported: json, tree, table, dot", format);
                }
            }
        } else {
            // Transitive dependencies (depth > 1)
            let transitive = deps_index.get_transitive_deps(file_id, depth)?;
            let file_ids: Vec<_> = transitive.keys().copied().collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            match format.as_ref() {
                "json" => {
                    let output: Vec<_> = transitive.iter()
                        .filter_map(|(id, d)| {
                            paths.get(id).map(|path| serde_json::json!({
                                "file_id": id,
                                "path": path,
                                "depth": d,
                            }))
                        })
                        .collect();

                    let json_str = if pretty_json {
                        serde_json::to_string_pretty(&output)?
                    } else {
                        serde_json::to_string(&output)?
                    };
                    println!("{}", json_str);
                    eprintln!("Found {} transitive dependencies (depth {})", transitive.len(), depth);
                }
                "tree" => {
                    println!("Transitive dependencies of {} (depth {}):", file_str, depth);
                    // Group by depth for tree display
                    let mut by_depth: std::collections::HashMap<usize, Vec<i64>> = std::collections::HashMap::new();
                    for (id, d) in &transitive {
                        by_depth.entry(*d).or_insert_with(Vec::new).push(*id);
                    }

                    for depth_level in 0..=depth {
                        if let Some(ids) = by_depth.get(&depth_level) {
                            let indent = "  ".repeat(depth_level);
                            for id in ids {
                                if let Some(path) = paths.get(id) {
                                    if depth_level == 0 {
                                        println!("{}{} (self)", indent, path);
                                    } else {
                                        println!("{}└─ {}", indent, path);
                                    }
                                }
                            }
                        }
                    }
                    eprintln!("\nFound {} transitive dependencies", transitive.len());
                }
                "table" => {
                    println!("Depth  File ID  Path");
                    println!("-----  -------  ----");
                    let mut sorted: Vec<_> = transitive.iter().collect();
                    sorted.sort_by_key(|(_, d)| *d);
                    for (id, d) in sorted {
                        if let Some(path) = paths.get(id) {
                            println!("{:<5}  {:<7}  {}", d, id, path);
                        }
                    }
                    eprintln!("\nFound {} transitive dependencies", transitive.len());
                }
                _ => {
                    anyhow::bail!("Unknown format '{}'. Supported: json, tree, table, dot", format);
                }
            }
        }
    }

    Ok(())
}


/// Handle --circular flag (detect cycles)
fn handle_deps_circular(
    deps_index: &crate::dependency::DependencyIndex,
    format: &str,
    pretty_json: bool,
    limit: Option<usize>,
    offset: Option<usize>,
    count_only: bool,
    _plain: bool,
    sort: Option<String>,
) -> Result<()> {
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
            anyhow::bail!("Invalid sort order '{}'. Supported: asc, desc", sort_order);
        }
    }

    let total_count = all_cycles.len();

    if count_only {
        println!("Found {} circular dependencies", total_count);
        return Ok(());
    }

    if all_cycles.is_empty() {
        println!("No circular dependencies found.");
        return Ok(());
    }

    // Apply offset pagination
    let offset_val = offset.unwrap_or(0);
    let mut cycles: Vec<_> = all_cycles.into_iter().skip(offset_val).collect();

    // Apply limit
    if let Some(lim) = limit {
        cycles.truncate(lim);
    }

    if cycles.is_empty() {
        println!("No circular dependencies found at offset {}.", offset_val);
        return Ok(());
    }

    let count = cycles.len();
    let has_more = offset_val + count < total_count;

    match format {
        "json" => {
            let file_ids: Vec<i64> = cycles.iter().flat_map(|c| c.iter()).copied().collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            let results: Vec<_> = cycles.iter()
                .map(|cycle| {
                    let cycle_paths: Vec<_> = cycle.iter()
                        .filter_map(|id| paths.get(id).cloned())
                        .collect();
                    serde_json::json!({
                        "paths": cycle_paths,
                    })
                })
                .collect();

            let output = serde_json::json!({
                "pagination": {
                    "total": total_count,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit,
                    "has_more": has_more,
                },
                "results": results,
            });

            let json_str = if pretty_json {
                serde_json::to_string_pretty(&output)?
            } else {
                serde_json::to_string(&output)?
            };
            println!("{}", json_str);
            if total_count > count {
                eprintln!("Found {} circular dependencies ({} total)", count, total_count);
            } else {
                eprintln!("Found {} circular dependencies", count);
            }
        }
        "tree" => {
            println!("Circular Dependencies Found:");
            let file_ids: Vec<i64> = cycles.iter().flat_map(|c| c.iter()).copied().collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            for (idx, cycle) in cycles.iter().enumerate() {
                println!("\nCycle {}:", idx + 1);
                for id in cycle {
                    if let Some(path) = paths.get(id) {
                        println!("  → {}", path);
                    }
                }
                // Show cycle completion
                if let Some(first_id) = cycle.first() {
                    if let Some(path) = paths.get(first_id) {
                        println!("  → {} (cycle completes)", path);
                    }
                }
            }
            if total_count > count {
                eprintln!("\nFound {} cycles ({} total)", count, total_count);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else {
                eprintln!("\nFound {} cycles", count);
            }
        }
        "table" => {
            println!("Cycle  Files in Cycle");
            println!("-----  --------------");
            let file_ids: Vec<i64> = cycles.iter().flat_map(|c| c.iter()).copied().collect();
            let paths = deps_index.get_file_paths(&file_ids)?;

            for (idx, cycle) in cycles.iter().enumerate() {
                let cycle_str = cycle.iter()
                    .filter_map(|id| paths.get(id).map(|p| p.as_str()))
                    .collect::<Vec<_>>()
                    .join(" → ");
                println!("{:<5}  {}", idx + 1, cycle_str);
            }
            if total_count > count {
                eprintln!("\nFound {} cycles ({} total)", count, total_count);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else {
                eprintln!("\nFound {} cycles", count);
            }
        }
        _ => {
            anyhow::bail!("Unknown format '{}'. Supported: json, tree, table", format);
        }
    }

    Ok(())
}


/// Handle --hotspots flag (most-imported files)
fn handle_deps_hotspots(
    deps_index: &crate::dependency::DependencyIndex,
    format: &str,
    pretty_json: bool,
    limit: Option<usize>,
    offset: Option<usize>,
    min_dependents: usize,
    count_only: bool,
    _plain: bool,
    sort: Option<String>,
) -> Result<()> {
    // Get all hotspots without limit first to track total count
    let mut all_hotspots = deps_index.find_hotspots(None, min_dependents)?;

    // Apply sorting (default: descending - most imports first)
    let sort_order = sort.as_deref().unwrap_or("desc");
    match sort_order {
        "asc" => {
            // Ascending: least imports first
            all_hotspots.sort_by(|a, b| a.1.cmp(&b.1));
        }
        "desc" => {
            // Descending: most imports first (default)
            all_hotspots.sort_by(|a, b| b.1.cmp(&a.1));
        }
        _ => {
            anyhow::bail!("Invalid sort order '{}'. Supported: asc, desc", sort_order);
        }
    }

    let total_count = all_hotspots.len();

    if count_only {
        println!("Found {} hotspots with {}+ dependents", total_count, min_dependents);
        return Ok(());
    }

    if all_hotspots.is_empty() {
        println!("No hotspots found.");
        return Ok(());
    }

    // Apply offset pagination
    let offset_val = offset.unwrap_or(0);
    let mut hotspots: Vec<_> = all_hotspots.into_iter().skip(offset_val).collect();

    // Apply limit
    if let Some(lim) = limit {
        hotspots.truncate(lim);
    }

    if hotspots.is_empty() {
        println!("No hotspots found at offset {}.", offset_val);
        return Ok(());
    }

    let count = hotspots.len();
    let has_more = offset_val + count < total_count;

    let file_ids: Vec<i64> = hotspots.iter().map(|(id, _)| *id).collect();
    let paths = deps_index.get_file_paths(&file_ids)?;

    match format {
        "json" => {
            let results: Vec<_> = hotspots.iter()
                .filter_map(|(id, import_count)| {
                    paths.get(id).map(|path| serde_json::json!({
                        "path": path,
                        "import_count": import_count,
                    }))
                })
                .collect();

            let output = serde_json::json!({
                "pagination": {
                    "total": total_count,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit,
                    "has_more": has_more,
                },
                "results": results,
            });

            let json_str = if pretty_json {
                serde_json::to_string_pretty(&output)?
            } else {
                serde_json::to_string(&output)?
            };
            println!("{}", json_str);
            if total_count > count {
                eprintln!("Found {} hotspots ({} total)", count, total_count);
            } else {
                eprintln!("Found {} hotspots", count);
            }
        }
        "tree" => {
            println!("Hotspots (Most-Imported Files):");
            for (idx, (id, import_count)) in hotspots.iter().enumerate() {
                if let Some(path) = paths.get(id) {
                    println!("  {}. {} ({} imports)", idx + 1, path, import_count);
                }
            }
            if total_count > count {
                eprintln!("\nFound {} hotspots ({} total)", count, total_count);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else {
                eprintln!("\nFound {} hotspots", count);
            }
        }
        "table" => {
            println!("Rank  Imports  File");
            println!("----  -------  ----");
            for (idx, (id, import_count)) in hotspots.iter().enumerate() {
                if let Some(path) = paths.get(id) {
                    println!("{:<4}  {:<7}  {}", idx + 1, import_count, path);
                }
            }
            if total_count > count {
                eprintln!("\nFound {} hotspots ({} total)", count, total_count);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else {
                eprintln!("\nFound {} hotspots", count);
            }
        }
        _ => {
            anyhow::bail!("Unknown format '{}'. Supported: json, tree, table", format);
        }
    }

    Ok(())
}


/// Handle --unused flag (orphaned files)
fn handle_deps_unused(
    deps_index: &crate::dependency::DependencyIndex,
    format: &str,
    pretty_json: bool,
    limit: Option<usize>,
    offset: Option<usize>,
    count_only: bool,
    _plain: bool,
) -> Result<()> {
    let all_unused = deps_index.find_unused_files()?;
    let total_count = all_unused.len();

    if count_only {
        println!("Found {} unused files", total_count);
        return Ok(());
    }

    if all_unused.is_empty() {
        println!("No unused files found (all files have incoming dependencies).");
        return Ok(());
    }

    // Apply offset pagination
    let offset_val = offset.unwrap_or(0);
    let mut unused: Vec<_> = all_unused.into_iter().skip(offset_val).collect();

    if unused.is_empty() {
        println!("No unused files found at offset {}.", offset_val);
        return Ok(());
    }

    // Apply limit
    if let Some(lim) = limit {
        unused.truncate(lim);
    }

    let count = unused.len();
    let has_more = offset_val + count < total_count;

    let paths = deps_index.get_file_paths(&unused)?;

    match format {
        "json" => {
            // Return flat array of path strings (no "path" key wrapper)
            let results: Vec<String> = unused.iter()
                .filter_map(|id| paths.get(id).cloned())
                .collect();

            let output = serde_json::json!({
                "pagination": {
                    "total": total_count,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit,
                    "has_more": has_more,
                },
                "results": results,
            });

            let json_str = if pretty_json {
                serde_json::to_string_pretty(&output)?
            } else {
                serde_json::to_string(&output)?
            };
            println!("{}", json_str);
            if total_count > count {
                eprintln!("Found {} unused files ({} total)", count, total_count);
            } else {
                eprintln!("Found {} unused files", count);
            }
        }
        "tree" => {
            println!("Unused Files (No Incoming Dependencies):");
            for (idx, id) in unused.iter().enumerate() {
                if let Some(path) = paths.get(id) {
                    println!("  {}. {}", idx + 1, path);
                }
            }
            if total_count > count {
                eprintln!("\nFound {} unused files ({} total)", count, total_count);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else {
                eprintln!("\nFound {} unused files", count);
            }
        }
        "table" => {
            println!("Path");
            println!("----");
            for id in &unused {
                if let Some(path) = paths.get(id) {
                    println!("{}", path);
                }
            }
            if total_count > count {
                eprintln!("\nFound {} unused files ({} total)", count, total_count);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else {
                eprintln!("\nFound {} unused files", count);
            }
        }
        _ => {
            anyhow::bail!("Unknown format '{}'. Supported: json, tree, table", format);
        }
    }

    Ok(())
}


/// Handle --islands flag (disconnected components)
fn handle_deps_islands(
    deps_index: &crate::dependency::DependencyIndex,
    format: &str,
    pretty_json: bool,
    limit: Option<usize>,
    offset: Option<usize>,
    min_island_size: usize,
    max_island_size: Option<usize>,
    count_only: bool,
    _plain: bool,
    sort: Option<String>,
) -> Result<()> {
    let all_islands = deps_index.find_islands()?;
    let total_components = all_islands.len();

    // Get total file count from the cache for percentage calculation
    let cache = deps_index.get_cache();
    let total_files = cache.stats()?.total_files as usize;

    // Calculate max_island_size default: min of 500 or 50% of total files
    let max_size = max_island_size.unwrap_or_else(|| {
        let fifty_percent = (total_files as f64 * 0.5) as usize;
        fifty_percent.min(500)
    });

    // Filter islands by size
    let mut islands: Vec<_> = all_islands.into_iter()
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
            anyhow::bail!("Invalid sort order '{}'. Supported: asc, desc", sort_order);
        }
    }

    let filtered_count = total_components - islands.len();

    if count_only {
        if filtered_count > 0 {
            println!("Found {} islands (filtered {} of {} total components by size: {}-{})",
                islands.len(), filtered_count, total_components, min_island_size, max_size);
        } else {
            println!("Found {} islands", islands.len());
        }
        return Ok(());
    }

    // Apply offset pagination first
    let offset_val = offset.unwrap_or(0);
    if offset_val > 0 && offset_val < islands.len() {
        islands = islands.into_iter().skip(offset_val).collect();
    } else if offset_val >= islands.len() {
        if filtered_count > 0 {
            println!("No islands found at offset {} (filtered {} of {} total components by size: {}-{}).",
                offset_val, filtered_count, total_components, min_island_size, max_size);
        } else {
            println!("No islands found at offset {}.", offset_val);
        }
        return Ok(());
    }

    // Apply limit to number of islands
    if let Some(lim) = limit {
        islands.truncate(lim);
    }

    if islands.is_empty() {
        if filtered_count > 0 {
            println!("No islands found matching criteria (filtered {} of {} total components by size: {}-{}).",
                filtered_count, total_components, min_island_size, max_size);
        } else {
            println!("No islands found.");
        }
        return Ok(());
    }

    // Get all file IDs from all islands and track pagination
    let count = islands.len();
    let has_more = offset_val + count < total_components - filtered_count;

    let file_ids: Vec<i64> = islands.iter().flat_map(|island| island.iter()).copied().collect();
    let paths = deps_index.get_file_paths(&file_ids)?;

    match format {
        "json" => {
            let results: Vec<_> = islands.iter()
                .enumerate()
                .map(|(idx, island)| {
                    let island_paths: Vec<_> = island.iter()
                        .filter_map(|id| paths.get(id).cloned())
                        .collect();
                    serde_json::json!({
                        "island_id": idx + 1,
                        "size": island.len(),
                        "paths": island_paths,
                    })
                })
                .collect();

            let output = serde_json::json!({
                "pagination": {
                    "total": total_components - filtered_count,
                    "count": count,
                    "offset": offset_val,
                    "limit": limit,
                    "has_more": has_more,
                },
                "results": results,
            });

            let json_str = if pretty_json {
                serde_json::to_string_pretty(&output)?
            } else {
                serde_json::to_string(&output)?
            };
            println!("{}", json_str);
            if filtered_count > 0 {
                eprintln!("Found {} islands (filtered {} of {} total components by size: {}-{})",
                    count, filtered_count, total_components, min_island_size, max_size);
            } else if total_components - filtered_count > count {
                eprintln!("Found {} islands ({} total)", count, total_components - filtered_count);
            } else {
                eprintln!("Found {} islands (disconnected components)", count);
            }
        }
        "tree" => {
            println!("Islands (Disconnected Components):");
            for (idx, island) in islands.iter().enumerate() {
                println!("\nIsland {} ({} files):", idx + 1, island.len());
                for id in island {
                    if let Some(path) = paths.get(id) {
                        println!("  ├─ {}", path);
                    }
                }
            }
            if filtered_count > 0 {
                eprintln!("\nFound {} islands (filtered {} of {} total components by size: {}-{})",
                    count, filtered_count, total_components, min_island_size, max_size);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else if total_components - filtered_count > count {
                eprintln!("\nFound {} islands ({} total)", count, total_components - filtered_count);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else {
                eprintln!("\nFound {} islands", count);
            }
        }
        "table" => {
            println!("Island  Size  Files");
            println!("------  ----  -----");
            for (idx, island) in islands.iter().enumerate() {
                let island_files = island.iter()
                    .filter_map(|id| paths.get(id).map(|p| p.as_str()))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("{:<6}  {:<4}  {}", idx + 1, island.len(), island_files);
            }
            if filtered_count > 0 {
                eprintln!("\nFound {} islands (filtered {} of {} total components by size: {}-{})",
                    count, filtered_count, total_components, min_island_size, max_size);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else if total_components - filtered_count > count {
                eprintln!("\nFound {} islands ({} total)", count, total_components - filtered_count);
                if has_more {
                    eprintln!("Use --limit and --offset to paginate");
                }
            } else {
                eprintln!("\nFound {} islands", count);
            }
        }
        _ => {
            anyhow::bail!("Unknown format '{}'. Supported: json, tree, table", format);
        }
    }

    Ok(())
}
