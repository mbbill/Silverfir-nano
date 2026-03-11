//! Static fusion pattern discovery.
//!
//! Analyzes Wasm binary files without executing them, extracts instruction
//! N-grams weighted by loop nesting depth (with interprocedural call graph
//! propagation), and produces TOML output compatible with handlers_fused.toml.

mod discovery;
mod trie;

use discovery::DiscoveryConfig;
use trie::PatternTrie;
use wasmparser::{Operator, Parser, Payload, TypeRef};

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::{env, fs, process};

// =============================================================================
// Operator → handler name mapping
// =============================================================================

fn operator_to_handler(op: &Operator) -> Option<&'static str> {
    use Operator::*;
    match op {
        LocalGet { .. } => Some("local_get"),
        LocalSet { .. } => Some("local_set"),
        LocalTee { .. } => Some("local_tee"),
        I32Const { .. } => Some("i32_const"),
        I64Const { .. } => Some("i64_const"),
        If { .. } => Some("if_"),
        BrIf { .. } => Some("br_if"),
        // i32 arithmetic
        I32Add => Some("i32_add"),
        I32Sub => Some("i32_sub"),
        I32Mul => Some("i32_mul"),
        I32DivS => Some("i32_div_s"),
        I32DivU => Some("i32_div_u"),
        I32RemS => Some("i32_rem_s"),
        I32RemU => Some("i32_rem_u"),
        I32And => Some("i32_and"),
        I32Or => Some("i32_or"),
        I32Xor => Some("i32_xor"),
        I32Shl => Some("i32_shl"),
        I32ShrS => Some("i32_shr_s"),
        I32ShrU => Some("i32_shr_u"),
        I32Rotl => Some("i32_rotl"),
        I32Rotr => Some("i32_rotr"),
        // i32 comparison
        I32Eqz => Some("i32_eqz"),
        I32Eq => Some("i32_eq"),
        I32Ne => Some("i32_ne"),
        I32LtS => Some("i32_lt_s"),
        I32LtU => Some("i32_lt_u"),
        I32GtS => Some("i32_gt_s"),
        I32GtU => Some("i32_gt_u"),
        I32LeS => Some("i32_le_s"),
        I32LeU => Some("i32_le_u"),
        I32GeS => Some("i32_ge_s"),
        I32GeU => Some("i32_ge_u"),
        // i32 unary
        I32Clz => Some("i32_clz"),
        I32Ctz => Some("i32_ctz"),
        I32Popcnt => Some("i32_popcnt"),
        // i64 arithmetic
        I64Add => Some("i64_add"),
        I64Sub => Some("i64_sub"),
        I64Mul => Some("i64_mul"),
        I64DivS => Some("i64_div_s"),
        I64DivU => Some("i64_div_u"),
        I64RemS => Some("i64_rem_s"),
        I64RemU => Some("i64_rem_u"),
        I64And => Some("i64_and"),
        I64Or => Some("i64_or"),
        I64Xor => Some("i64_xor"),
        I64Shl => Some("i64_shl"),
        I64ShrS => Some("i64_shr_s"),
        I64ShrU => Some("i64_shr_u"),
        I64Rotl => Some("i64_rotl"),
        I64Rotr => Some("i64_rotr"),
        // i64 comparison
        I64Eqz => Some("i64_eqz"),
        I64Eq => Some("i64_eq"),
        I64Ne => Some("i64_ne"),
        I64LtS => Some("i64_lt_s"),
        I64LtU => Some("i64_lt_u"),
        I64GtS => Some("i64_gt_s"),
        I64GtU => Some("i64_gt_u"),
        I64LeS => Some("i64_le_s"),
        I64LeU => Some("i64_le_u"),
        I64GeS => Some("i64_ge_s"),
        I64GeU => Some("i64_ge_u"),
        // i64 unary
        I64Clz => Some("i64_clz"),
        I64Ctz => Some("i64_ctz"),
        I64Popcnt => Some("i64_popcnt"),
        // Conversions
        I32WrapI64 => Some("i32_wrap_i64"),
        I64ExtendI32S => Some("i64_extend_i32_s"),
        I64ExtendI32U => Some("i64_extend_i32_u"),
        I32Extend8S => Some("i32_extend8_s"),
        I32Extend16S => Some("i32_extend16_s"),
        I64Extend8S => Some("i64_extend8_s"),
        I64Extend16S => Some("i64_extend16_s"),
        I64Extend32S => Some("i64_extend32_s"),
        // Reinterpret
        I32ReinterpretF32 => Some("i32_reinterpret_f32"),
        I64ReinterpretF64 => Some("i64_reinterpret_f64"),
        F32ReinterpretI32 => Some("f32_reinterpret_i32"),
        F64ReinterpretI64 => Some("f64_reinterpret_i64"),
        // f32 arithmetic
        F32Add => Some("f32_add"),
        F32Sub => Some("f32_sub"),
        F32Mul => Some("f32_mul"),
        F32Div => Some("f32_div"),
        F32Min => Some("f32_min"),
        F32Max => Some("f32_max"),
        F32Copysign => Some("f32_copysign"),
        // f32 comparison
        F32Eq => Some("f32_eq"),
        F32Ne => Some("f32_ne"),
        F32Lt => Some("f32_lt"),
        F32Gt => Some("f32_gt"),
        F32Le => Some("f32_le"),
        F32Ge => Some("f32_ge"),
        // f32 unary
        F32Abs => Some("f32_abs"),
        F32Neg => Some("f32_neg"),
        F32Ceil => Some("f32_ceil"),
        F32Floor => Some("f32_floor"),
        F32Trunc => Some("f32_trunc"),
        F32Nearest => Some("f32_nearest"),
        F32Sqrt => Some("f32_sqrt"),
        // f64 arithmetic
        F64Add => Some("f64_add"),
        F64Sub => Some("f64_sub"),
        F64Mul => Some("f64_mul"),
        F64Div => Some("f64_div"),
        F64Min => Some("f64_min"),
        F64Max => Some("f64_max"),
        F64Copysign => Some("f64_copysign"),
        // f64 comparison
        F64Eq => Some("f64_eq"),
        F64Ne => Some("f64_ne"),
        F64Lt => Some("f64_lt"),
        F64Gt => Some("f64_gt"),
        F64Le => Some("f64_le"),
        F64Ge => Some("f64_ge"),
        // f64 unary
        F64Abs => Some("f64_abs"),
        F64Neg => Some("f64_neg"),
        F64Ceil => Some("f64_ceil"),
        F64Floor => Some("f64_floor"),
        F64Trunc => Some("f64_trunc"),
        F64Nearest => Some("f64_nearest"),
        F64Sqrt => Some("f64_sqrt"),
        // Float conversions
        F32ConvertI32S => Some("f32_convert_i32_s"),
        F32ConvertI32U => Some("f32_convert_i32_u"),
        F32ConvertI64S => Some("f32_convert_i64_s"),
        F32ConvertI64U => Some("f32_convert_i64_u"),
        F64ConvertI32S => Some("f64_convert_i32_s"),
        F64ConvertI32U => Some("f64_convert_i32_u"),
        F64ConvertI64S => Some("f64_convert_i64_s"),
        F64ConvertI64U => Some("f64_convert_i64_u"),
        F32DemoteF64 => Some("f32_demote_f64"),
        F64PromoteF32 => Some("f64_promote_f32"),
        // Memory load
        I32Load { .. } => Some("i32_load"),
        I64Load { .. } => Some("i64_load"),
        F32Load { .. } => Some("f32_load"),
        F64Load { .. } => Some("f64_load"),
        I32Load8S { .. } => Some("i32_load8_s"),
        I32Load8U { .. } => Some("i32_load8_u"),
        I32Load16S { .. } => Some("i32_load16_s"),
        I32Load16U { .. } => Some("i32_load16_u"),
        I64Load8S { .. } => Some("i64_load8_s"),
        I64Load8U { .. } => Some("i64_load8_u"),
        I64Load16S { .. } => Some("i64_load16_s"),
        I64Load16U { .. } => Some("i64_load16_u"),
        I64Load32S { .. } => Some("i64_load32_s"),
        I64Load32U { .. } => Some("i64_load32_u"),
        // Memory store
        I32Store { .. } => Some("i32_store"),
        I64Store { .. } => Some("i64_store"),
        F32Store { .. } => Some("f32_store"),
        F64Store { .. } => Some("f64_store"),
        I32Store8 { .. } => Some("i32_store8"),
        I32Store16 { .. } => Some("i32_store16"),
        I64Store8 { .. } => Some("i64_store8"),
        I64Store16 { .. } => Some("i64_store16"),
        I64Store32 { .. } => Some("i64_store32"),
        _ => None,
    }
}

// =============================================================================
// Loop weight
// =============================================================================

const WEIGHTS: [u64; 7] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000];

#[inline]
fn loop_weight(depth: u32) -> u64 {
    WEIGHTS[depth.min(6) as usize]
}

// =============================================================================
// Block tracking for proper loop depth
// =============================================================================

#[derive(Clone, Copy, PartialEq)]
enum BlockKind {
    Block,
    Loop,
    If,
}

// =============================================================================
// Phase 1: Call graph extraction
// =============================================================================

struct CallSite {
    callee_idx: u32,
    loop_depth: u32,
}

fn extract_call_sites(body: &wasmparser::FunctionBody) -> Vec<CallSite> {
    let mut sites = Vec::new();
    let mut loop_depth: u32 = 0;
    let mut block_stack: Vec<BlockKind> = Vec::new();

    let reader = body.get_operators_reader().unwrap();
    for op in reader {
        let op = op.unwrap();
        match &op {
            Operator::Block { .. } => block_stack.push(BlockKind::Block),
            Operator::Loop { .. } => {
                block_stack.push(BlockKind::Loop);
                loop_depth += 1;
            }
            Operator::If { .. } => block_stack.push(BlockKind::If),
            Operator::End => {
                if let Some(BlockKind::Loop) = block_stack.pop() {
                    loop_depth = loop_depth.saturating_sub(1);
                }
            }
            Operator::Call { function_index } => {
                sites.push(CallSite {
                    callee_idx: *function_index,
                    loop_depth,
                });
            }
            _ => {}
        }
    }
    sites
}

// =============================================================================
// Phase 2: Call graph propagation
// =============================================================================

fn propagate_call_graph(
    num_functions: usize,
    num_imports: usize,
    call_graph: &[Vec<CallSite>],
    export_indices: &[usize],
    start_index: Option<usize>,
) -> Vec<u32> {
    let mut base_depth = vec![0u32; num_functions];
    let mut visited = vec![false; num_functions];

    let mut queue: VecDeque<(usize, u32)> = VecDeque::new();

    for &idx in export_indices {
        if idx >= num_imports && idx < num_functions {
            queue.push_back((idx, 0));
        }
    }
    if let Some(idx) = start_index {
        if idx >= num_imports && idx < num_functions {
            queue.push_back((idx, 0));
        }
    }

    if queue.is_empty() {
        for idx in num_imports..num_functions {
            queue.push_back((idx, 0));
        }
    }

    const MAX_DEPTH: u32 = 6;

    while let Some((func_idx, depth)) = queue.pop_front() {
        if func_idx < num_imports || func_idx >= num_functions {
            continue;
        }
        let local_idx = func_idx - num_imports;
        if local_idx >= call_graph.len() {
            continue;
        }

        if visited[func_idx] && depth <= base_depth[func_idx] {
            continue;
        }
        base_depth[func_idx] = base_depth[func_idx].max(depth);
        visited[func_idx] = true;

        for site in &call_graph[local_idx] {
            let callee = site.callee_idx as usize;
            if callee < num_imports || callee >= num_functions {
                continue;
            }
            let new_depth = (depth + site.loop_depth).min(MAX_DEPTH);
            if new_depth > base_depth[callee] {
                queue.push_back((callee, new_depth));
            }
        }
    }

    base_depth
}

// =============================================================================
// Phase 3: N-gram extraction
// =============================================================================

fn analyze_operators(
    body: &wasmparser::FunctionBody,
    base_depth: u32,
    window_size: usize,
    counts: &mut HashMap<Vec<String>, u64>,
    total: &mut u64,
) {
    let mut window: VecDeque<String> = VecDeque::new();
    let mut loop_depth: u32 = 0;
    let mut block_stack: Vec<BlockKind> = Vec::new();

    let reader = body.get_operators_reader().unwrap();
    for op in reader {
        let op = op.unwrap();

        // Track block nesting
        match &op {
            Operator::Block { .. } => block_stack.push(BlockKind::Block),
            Operator::Loop { .. } => {
                block_stack.push(BlockKind::Loop);
                loop_depth += 1;
            }
            Operator::If { .. } => block_stack.push(BlockKind::If),
            Operator::End => {
                if let Some(BlockKind::Loop) = block_stack.pop() {
                    loop_depth = loop_depth.saturating_sub(1);
                }
            }
            _ => {}
        }

        let handler = operator_to_handler(&op);
        let effective_depth = base_depth + loop_depth;
        let weight = loop_weight(effective_depth);
        *total += weight;

        match handler {
            Some(name) => {
                window.push_back(name.to_string());
                if window.len() > window_size {
                    window.pop_front();
                }

                let wlen = window.len();
                for ngram_len in 1..=wlen {
                    let start = wlen - ngram_len;
                    let ngram: Vec<String> = window.iter().skip(start).cloned().collect();
                    *counts.entry(ngram).or_insert(0) += weight;
                }
            }
            None => {
                window.clear();
            }
        }
    }
}

// =============================================================================
// Per-file analysis
// =============================================================================

struct FileStats {
    counts: HashMap<Vec<String>, u64>,
    total: u64,
}

fn analyze_wasm_file(path: &Path, window_size: usize) -> FileStats {
    let wasm = fs::read(path).unwrap_or_else(|err| {
        eprintln!("Error reading '{}': {}", path.display(), err);
        process::exit(1);
    });

    // Pass 1: collect module structure + call graph
    let mut num_func_imports: usize = 0;
    let mut num_type_funcs: usize = 0;
    let mut export_indices: Vec<usize> = Vec::new();
    let mut start_index: Option<usize> = None;
    let mut call_graph: Vec<Vec<CallSite>> = Vec::new();

    for payload in Parser::new(0).parse_all(&wasm) {
        let payload = payload.unwrap_or_else(|err| {
            eprintln!("Error parsing '{}': {}", path.display(), err);
            process::exit(1);
        });
        match payload {
            Payload::ImportSection(reader) => {
                for import in reader {
                    if matches!(import.unwrap().ty, TypeRef::Func(_)) {
                        num_func_imports += 1;
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                num_type_funcs = reader.count() as usize;
            }
            Payload::ExportSection(reader) => {
                for export in reader {
                    let export = export.unwrap();
                    if let wasmparser::ExternalKind::Func = export.kind {
                        export_indices.push(export.index as usize);
                    }
                }
            }
            Payload::StartSection { func, .. } => {
                start_index = Some(func as usize);
            }
            Payload::CodeSectionEntry(body) => {
                call_graph.push(extract_call_sites(&body));
            }
            _ => {}
        }
    }

    let num_functions = num_func_imports + num_type_funcs;

    // Propagate call graph
    let base_depths = propagate_call_graph(
        num_functions,
        num_func_imports,
        &call_graph,
        &export_indices,
        start_index,
    );

    // Pass 2: N-gram extraction
    let mut counts: HashMap<Vec<String>, u64> = HashMap::new();
    let mut total: u64 = 0;
    let mut local_idx: usize = 0;

    for payload in Parser::new(0).parse_all(&wasm) {
        if let Ok(Payload::CodeSectionEntry(body)) = payload {
            let abs_idx = num_func_imports + local_idx;
            let base_depth = base_depths[abs_idx];
            analyze_operators(&body, base_depth, window_size, &mut counts, &mut total);
            local_idx += 1;
        }
    }

    eprintln!(
        "  {} — {} functions ({} local), {} weighted instructions",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
        num_functions,
        call_graph.len(),
        total
    );

    FileStats { counts, total }
}

// =============================================================================
// Multi-file merge (max-frequency)
// =============================================================================

fn merge_file_stats(all: Vec<FileStats>) -> FileStats {
    if all.len() == 1 {
        return all.into_iter().next().unwrap();
    }

    let mut max_freq: HashMap<Vec<String>, f64> = HashMap::new();
    let mut sum_total: u64 = 0;

    for stats in &all {
        if stats.total == 0 {
            continue;
        }
        sum_total += stats.total;
        for (key, &count) in &stats.counts {
            let freq = count as f64 / stats.total as f64;
            let entry = max_freq.entry(key.clone()).or_insert(0.0);
            if freq > *entry {
                *entry = freq;
            }
        }
    }

    let avg_total = sum_total / all.len() as u64;

    let mut merged_counts: HashMap<Vec<String>, u64> = HashMap::new();
    for (key, freq) in max_freq {
        let count = (freq * avg_total as f64) as u64;
        if count > 0 {
            merged_counts.insert(key, count);
        }
    }

    FileStats {
        counts: merged_counts,
        total: avg_total,
    }
}

// =============================================================================
// Handler name loading
// =============================================================================

fn load_handler_names() -> HashSet<String> {
    let candidates = [
        PathBuf::from("sf-nano-core/src/vm/interp/fast/handlers.toml"),
        PathBuf::from("../sf-nano-core/src/vm/interp/fast/handlers.toml"),
        PathBuf::from("../../sf-nano-core/src/vm/interp/fast/handlers.toml"),
    ];

    for toml_path in &candidates {
        if let Ok(content) = fs::read_to_string(toml_path) {
            return parse_handler_names(&content);
        }
    }

    eprintln!("Warning: Could not find handlers.toml — name collisions possible");
    HashSet::new()
}

fn parse_handler_names(content: &str) -> HashSet<String> {
    let mut names = HashSet::new();
    let mut in_handler = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "[[handler]]" {
            in_handler = true;
            continue;
        }
        if trimmed.starts_with("[[") {
            in_handler = false;
            continue;
        }
        if in_handler && trimmed.starts_with("op = ") {
            if let Some(rest) = trimmed.strip_prefix("op = ") {
                let rest = rest.trim().trim_matches('"');
                names.insert(rest.to_string());
            }
        }
    }
    names
}

// =============================================================================
// Hot-local expansion
// =============================================================================

const HOT_LOCAL_VARIANTS: [&str; 3] = ["_l0", "_l1", "_l2"];

fn is_local_op(op: &str) -> bool {
    matches!(op, "local_get" | "local_set" | "local_tee")
}

/// Expand each pattern: for every local op position, produce variants with
/// the original base name plus _l0, _l1, _l2 suffixes (4 per position).
/// Positions are independent, so a pattern with N local ops produces 4^N variants.
fn expand_hot_locals(
    candidates: Vec<discovery::FusionCandidate>,
    max_total: usize,
) -> Vec<discovery::FusionCandidate> {
    let mut used_names: HashSet<String> = candidates.iter().map(|c| c.name.clone()).collect();
    let mut result = Vec::new();

    for c in candidates {
        // Find positions of local ops
        let local_positions: Vec<usize> = c
            .pattern
            .iter()
            .enumerate()
            .filter(|(_, op)| is_local_op(op))
            .map(|(i, _)| i)
            .collect();

        if local_positions.is_empty() {
            // No locals — keep as-is
            result.push(c);
            if result.len() >= max_total {
                return result;
            }
            continue;
        }

        // Generate all combinations: each position gets 4 choices (base + 3 variants)
        // Total combos = 4^local_positions.len()
        let num_combos = 4usize.pow(local_positions.len() as u32);

        for combo in 0..num_combos {
            let mut pattern = c.pattern.clone();
            let mut rem = combo;
            for &pos in &local_positions {
                let choice = rem % 4;
                rem /= 4;
                if choice > 0 {
                    // choice 1=_l0, 2=_l1, 3=_l2
                    pattern[pos] = format!("{}{}", pattern[pos], HOT_LOCAL_VARIANTS[choice - 1]);
                }
                // choice 0 = keep base name
            }

            let pattern_refs: Vec<&str> = pattern.iter().map(|s| s.as_str()).collect();
            let tos = discovery::compute_tos_pattern(&pattern_refs);
            let fields = discovery::auto_encoding_fields(&pattern_refs);
            let name = discovery::auto_name(&pattern_refs, &used_names);
            used_names.insert(name.clone());

            result.push(discovery::FusionCandidate {
                pattern,
                name,
                raw_count: c.raw_count,
                effective_count: c.effective_count,
                savings: c.savings,
                tos_pattern: tos,
                encoding_fields: fields,
            });

            if result.len() >= max_total {
                return result;
            }
        }
    }

    result
}

// =============================================================================
// CLI
// =============================================================================

struct Args {
    files: Vec<PathBuf>,
    window_size: usize,
    top: usize,
    min_savings_pct: f64,
    output: PathBuf,
    show_trie: bool,
}

fn print_usage() {
    eprintln!("Static fusion pattern discovery for Silverfir-nano.");
    eprintln!();
    eprintln!("Analyzes Wasm binaries without executing them, extracting instruction");
    eprintln!("N-grams weighted by loop depth with interprocedural call graph propagation.");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  static-discover [OPTIONS] <wasm-file> [<wasm-file> ...]");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  -w, --window <N>          N-gram window size [default: 8]");
    eprintln!("  -n, --top <N>             Max fusion candidates [default: 500]");
    eprintln!("      --min-savings <PCT>   Minimum savings %% [default: 0.005]");
    eprintln!(
        "  -o, --output <PATH>       Output TOML path [default: handlers_fused_discovered.toml]"
    );
    eprintln!("      --show-trie           Print pattern trie");
    eprintln!("  -h, --help                Print this help message");
}

fn parse_args() -> Args {
    let argv: Vec<String> = env::args().collect();
    let mut args = Args {
        files: Vec::new(),
        window_size: 8,
        top: 500,
        min_savings_pct: 0.005,
        output: PathBuf::from("handlers_fused_discovered.toml"),
        show_trie: false,
    };

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            "-w" | "--window" => {
                i += 1;
                args.window_size = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(8);
            }
            "-n" | "--top" => {
                i += 1;
                args.top = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(500);
            }
            "--min-savings" => {
                i += 1;
                args.min_savings_pct = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.005);
            }
            "-o" | "--output" => {
                i += 1;
                if let Some(s) = argv.get(i) {
                    args.output = PathBuf::from(s);
                }
            }
            "--show-trie" => {
                args.show_trie = true;
            }
            _ => {
                args.files.push(PathBuf::from(&argv[i]));
            }
        }
        i += 1;
    }

    if args.files.is_empty() {
        print_usage();
        process::exit(1);
    }

    args
}

fn main() {
    let args = parse_args();

    eprintln!(
        "Static fusion discovery (window={}, top={}, min_savings={:.3}%)",
        args.window_size, args.top, args.min_savings_pct
    );
    eprintln!();

    // Analyze each file
    let t0 = std::time::Instant::now();
    let all_stats: Vec<FileStats> = args
        .files
        .iter()
        .map(|f| analyze_wasm_file(f, args.window_size))
        .collect();
    let elapsed = t0.elapsed();
    eprintln!();
    eprintln!("Analysis completed in {:.2}s", elapsed.as_secs_f64());

    // Merge
    let merged = merge_file_stats(all_stats);
    if merged.total == 0 {
        eprintln!("No instructions found.");
        return;
    }
    eprintln!("Merged total (weighted): {}", merged.total);

    // Print 1-gram breakdown
    {
        let mut unigrams: Vec<(&str, u64)> = Vec::new();
        for (key, &count) in &merged.counts {
            if key.len() == 1 {
                unigrams.push((&key[0], count));
            }
        }
        unigrams.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!();
        eprintln!("Per-handler instruction counts (1-grams, weighted):");
        eprintln!("{:<30} {:>12} {:>8}", "Handler", "Count", "Pct");
        eprintln!("{}", "-".repeat(52));
        for (name, count) in &unigrams {
            let pct = *count as f64 / merged.total as f64 * 100.0;
            eprintln!("{:<30} {:>12} {:>7.2}%", name, count, pct);
        }
        eprintln!("{}", "-".repeat(52));
        eprintln!("{:<30} {:>12}", "TOTAL (weighted)", merged.total);
        eprintln!();
    }

    // Build pattern trie
    eprintln!("Building pattern trie...");
    let mut trie = PatternTrie::new(merged.total);
    for (key, count) in &merged.counts {
        if key.len() >= 2 {
            let refs: Vec<&str> = key.iter().map(|s| s.as_str()).collect();
            trie.insert(&refs, *count);
        }
    }

    // Trie stats
    let depth_stats = trie.depth_stats();
    eprintln!();
    eprintln!("Trie Statistics");
    eprintln!("{}", "-".repeat(50));
    eprintln!("Total weighted instructions: {}", merged.total);
    let mut depths: Vec<_> = depth_stats.iter().collect();
    depths.sort_by_key(|(d, _)| *d);
    for (depth, count) in &depths {
        eprintln!("  Unique {}-grams: {}", depth, count);
    }
    eprintln!();

    if args.show_trie {
        let min_savings_abs =
            (args.min_savings_pct / 100.0 * trie.total_instructions as f64) as u64;
        trie.print_tree(
            args.window_size,
            min_savings_abs / (args.window_size as u64).max(1),
        );
        eprintln!();
    }

    // Load reserved handler names
    let reserved_names = load_handler_names();
    eprintln!("Reserved handler names: {}", reserved_names.len());
    eprintln!();

    // Run discovery
    let config = DiscoveryConfig {
        max_candidates: args.top,
        min_savings_pct: args.min_savings_pct,
        reserved_names,
    };

    let candidates = discovery::discover(&trie, &config);

    if candidates.is_empty() {
        eprintln!("No fusion candidates found above threshold.");
        return;
    }

    // Print summary
    eprintln!("Discovered Candidates");
    eprintln!("{}", "-".repeat(70));
    let total_savings: u64 = candidates.iter().map(|c| c.savings).sum();
    let total_pct = (total_savings as f64 / merged.total as f64) * 100.0;

    for (i, c) in candidates.iter().enumerate() {
        let pct = (c.savings as f64 / merged.total as f64) * 100.0;
        eprintln!(
            "  {:>3}. {} [{}] count={}, savings={} ({:.2}%)",
            i + 1,
            c.name,
            c.pattern.join(" -> "),
            c.effective_count,
            c.savings,
            pct
        );
    }
    eprintln!();
    eprintln!(
        "Total estimated dispatch reduction: {} ({:.2}%)",
        total_savings, total_pct
    );
    eprintln!();

    // Expand hot-local variants, capped at the requested --top limit
    let candidates = expand_hot_locals(candidates, args.top);
    eprintln!("After hot-local expansion: {} patterns", candidates.len());
    eprintln!();

    // Generate TOML
    let toml = discovery::format_all_toml(&candidates);

    // Write output
    fs::write(&args.output, &toml).unwrap_or_else(|err| {
        eprintln!("Error writing output: {}", err);
        process::exit(1);
    });
    eprintln!(
        "Written {} fused patterns to: {}",
        candidates.len(),
        args.output.display()
    );
}
