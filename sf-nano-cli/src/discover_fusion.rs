//! CLI command for automatic fusion candidate discovery.
//!
//! Profiles one or more WASM workloads, builds a pattern trie capturing all
//! N-gram instruction sequences, and selects optimal fusion candidates using
//! a greedy algorithm with prefix overlap adjustment.
//!
//! Supports multi-workload merging: when multiple `--workload` flags are given,
//! instruction statistics are frequency-averaged across workloads to produce
//! fusion patterns that generalize across diverse programs.

use sf_nano_core::vm::interp::fast::fusion::discovery::{self, DiscoveryConfig};
use sf_nano_core::vm::interp::fast::fusion::pattern_trie::PatternTrie;
use sf_nano_core::vm::interp::fast::fusion::profiler;
use sf_nano_core::wasi::{set_wasi_ctx, wasi_imports, WasiContextBuilder};
use sf_nano_core::{set_backend_mode, BackendMode, Instance};

use std::collections::HashSet;
use std::path::PathBuf;
use std::{fs, process};

/// Default output path for discovered fusion patterns.
const DEFAULT_FUSED_TOML: &str = "handlers_fused_discovered.toml";

pub struct Workload {
    pub path: PathBuf,
    pub prog_args: Vec<String>,
    pub dir: Option<PathBuf>,
}

pub struct DiscoverFusionArgs {
    pub workloads: Vec<Workload>,
    pub max_window: usize,
    pub top: usize,
    pub min_savings_pct: f64,
    pub output: Option<PathBuf>,
    pub show_trie: bool,
}

impl Default for DiscoverFusionArgs {
    fn default() -> Self {
        Self {
            workloads: Vec::new(),
            max_window: 4,
            top: 32,
            min_savings_pct: 0.005,
            output: None,
            show_trie: false,
        }
    }
}

/// Global options that can appear anywhere before/between workloads.
const GLOBAL_OPTIONS: &[&str] = &[
    "--window", "-w", "--top", "-n", "--min-savings", "--output", "-o", "--show-trie",
];

fn is_global_option(s: &str) -> bool {
    GLOBAL_OPTIONS.contains(&s)
}

fn print_usage() {
    eprintln!("Discover optimal instruction fusion patterns by profiling WASM workloads.");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  sf-nano-cli discover-fusion [OPTIONS] <wasm-file> [-- args...]");
    eprintln!("  sf-nano-cli discover-fusion [OPTIONS] --workload <wasm> [args...] ...");
    eprintln!();
    eprintln!("The first form profiles a single WASM module. The second form profiles");
    eprintln!("multiple workloads and merges their instruction statistics using frequency");
    eprintln!("averaging, producing fusion patterns that work well across diverse programs.");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  -w, --window <N>                N-gram window size for profiling [default: 4]");
    eprintln!("  -n, --top <N>                   Maximum number of fusion candidates [default: 32]");
    eprintln!("      --min-savings <PCT>           Minimum savings as %% of total instructions [default: 0.005]");
    eprintln!("  -o, --output <PATH>              Output TOML file path [default: handlers_fused_discovered.toml]");
    eprintln!("      --show-trie                  Print the pattern trie before discovery");
    eprintln!("      --workload <WASM> [ARGS...]  Add a workload to profile (repeatable)");
    eprintln!("        --dir <PATH>               Preopen directory for this workload (per-workload)");
    eprintln!("  -h, --help                       Print this help message");
    eprintln!();
    eprintln!("EXAMPLES:");
    eprintln!("  # Single workload (coremark):");
    eprintln!("  sf-nano-cli discover-fusion --top 500 --window 5 coremark.wasm");
    eprintln!();
    eprintln!("  # Multiple workloads (merged):");
    eprintln!("  sf-nano-cli discover-fusion --top 500 --window 5 \\");
    eprintln!("    --workload coremark.wasm \\");
    eprintln!("    --workload lua.wasm fib.lua");
    eprintln!();
    eprintln!("  # Custom output path:");
    eprintln!("  sf-nano-cli discover-fusion -o sf-nano-core/src/vm/interp/fast/handlers_fused.toml \\");
    eprintln!("    --workload coremark.wasm");
    eprintln!();
    eprintln!("MULTI-WORKLOAD MERGING:");
    eprintln!("  When --workload is specified multiple times, each workload is profiled");
    eprintln!("  independently. The resulting instruction statistics are then merged by:");
    eprintln!("    1. Normalizing each workload's counts to frequencies (count / total)");
    eprintln!("    2. Averaging frequencies across all workloads");
    eprintln!("    3. Scaling back to absolute counts using the average total");
    eprintln!("  This ensures patterns common across workloads are weighted appropriately.");
    eprintln!();
    eprintln!("After generating the TOML, copy it to the handlers_fused.toml path and rebuild:");
    eprintln!("  cp handlers_fused_discovered.toml sf-nano-core/src/vm/interp/fast/handlers_fused.toml");
    eprintln!("  cargo build --release");
}

/// Parse discover-fusion args from command line.
///
/// Supports two syntaxes:
///   Legacy:  discover-fusion [options] <wasm-file> [-- args...]
///   Multi:   discover-fusion [options] --workload <wasm> [args...] [--workload <wasm> [args...] ...]
pub fn parse_args(args: &[String]) -> DiscoverFusionArgs {
    let mut result = DiscoverFusionArgs::default();
    let mut i = 0;

    // First pass: collect global options and workloads
    let mut bare_path: Option<PathBuf> = None;
    let mut bare_args: Vec<String> = Vec::new();

    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            "--window" | "-w" => {
                i += 1;
                if i < args.len() {
                    result.max_window = args[i].parse().unwrap_or(4);
                }
            }
            "--top" | "-n" => {
                i += 1;
                if i < args.len() {
                    result.top = args[i].parse().unwrap_or(32);
                }
            }
            "--min-savings" => {
                i += 1;
                if i < args.len() {
                    result.min_savings_pct = args[i].parse().unwrap_or(0.01);
                }
            }
            "--output" | "-o" => {
                i += 1;
                if i < args.len() {
                    result.output = Some(PathBuf::from(&args[i]));
                }
            }
            "--show-trie" => {
                result.show_trie = true;
            }
            "--workload" => {
                // Collect workload: next arg is module path, rest until next
                // --workload or global option are program args.
                // An optional --dir <path> can appear among the workload args.
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: --workload requires a wasm file path");
                    process::exit(1);
                }
                let path = PathBuf::from(&args[i]);
                i += 1;
                let mut prog_args = Vec::new();
                let mut dir: Option<PathBuf> = None;
                while i < args.len() && args[i] != "--workload" && !is_global_option(&args[i]) {
                    if args[i] == "--dir" {
                        i += 1;
                        if i < args.len() {
                            dir = Some(PathBuf::from(&args[i]));
                        }
                    } else {
                        prog_args.push(args[i].clone());
                    }
                    i += 1;
                }
                result.workloads.push(Workload { path, prog_args, dir });
                continue; // don't increment i again
            }
            "--" => {
                // Legacy: rest are program args for the bare path
                bare_args = args[i + 1..].to_vec();
                break;
            }
            _ => {
                if bare_path.is_none() && result.workloads.is_empty() {
                    bare_path = Some(PathBuf::from(&args[i]));
                } else if bare_path.is_some() && result.workloads.is_empty() {
                    // Remaining args are program args (legacy mode)
                    bare_args = args[i..].to_vec();
                    break;
                }
            }
        }
        i += 1;
    }

    // If we have a bare path and no --workload entries, use legacy single-workload mode
    if let Some(path) = bare_path {
        if result.workloads.is_empty() {
            result.workloads.push(Workload {
                path,
                prog_args: bare_args,
                dir: None,
            });
        }
    }

    // Handle --help / -h anywhere in args
    if args.iter().any(|a| a == "--help" || a == "-h") || result.workloads.is_empty() {
        print_usage();
        process::exit(if result.workloads.is_empty() { 1 } else { 0 });
    }

    result
}

pub fn run_from_args(args: &[String]) {
    let cmd = parse_args(args);
    run(cmd);
}

/// Run a single workload and return its profiling stats.
fn run_workload(workload: &Workload, window_size: usize) -> profiler::FastProfileStats {
    // Force the base backend so profiling sees the raw instruction stream.
    set_backend_mode(BackendMode::Base);

    profiler::enable(window_size);
    eprintln!(
        "Profiling: {} (window size: {})",
        workload.path.display(),
        window_size
    );

    // Read WASM binary
    let data = fs::read(&workload.path).unwrap_or_else(|err| {
        eprintln!("Error reading '{}': {}", workload.path.display(), err);
        process::exit(1);
    });

    let module_name = workload
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("module");

    // Build WASI context
    let mut wasi_args = vec![module_name.to_string()];
    wasi_args.extend(workload.prog_args.clone());

    let preopen = workload.dir.as_deref().unwrap_or_else(|| std::path::Path::new("."));
    let ctx = WasiContextBuilder::new()
        .args(&wasi_args)
        .preopen_dir(".", preopen)
        .inherit_env()
        .build();
    set_wasi_ctx(ctx);

    // Create instance and run
    let imports = wasi_imports();
    let mut instance = Instance::new(&data, &imports).unwrap_or_else(|err| {
        eprintln!("Error instantiating module: {}", err);
        process::exit(1);
    });

    let result = instance.invoke("_start", &[]);
    match result {
        Ok(_) => {}
        Err(ref err) if err.to_string().contains("not found") => {
            let _ = instance.invoke("main", &[]);
        }
        Err(_) => {}
    }

    // Collect stats
    profiler::take_stats()
}

pub fn run(cmd: DiscoverFusionArgs) {
    let multi = cmd.workloads.len() > 1;

    // Run each workload and collect stats
    let mut all_stats = Vec::new();
    for (i, workload) in cmd.workloads.iter().enumerate() {
        if multi {
            eprintln!();
            eprintln!("=== Workload {}/{} ===", i + 1, cmd.workloads.len());
        }
        let t0 = std::time::Instant::now();
        let stats = run_workload(workload, cmd.max_window);
        let elapsed = t0.elapsed();
        eprintln!(
            "  {} instructions profiled in {:.1}s",
            stats.total_instructions,
            elapsed.as_secs_f64()
        );
        all_stats.push(stats);
    }

    // Merge stats (frequency-averaged if multiple workloads)
    let stats = profiler::merge_stats(all_stats);

    if stats.total_instructions == 0 {
        eprintln!("No instructions were profiled.");
        return;
    }

    if multi {
        eprintln!();
        eprintln!("Merged (frequency-averaged): {} virtual instructions", stats.total_instructions);
    }

    // Print 1-gram (per-handler) breakdown
    {
        let mut unigrams: Vec<_> = stats.sequences.iter()
            .filter(|(k, _)| k.len() == 1)
            .map(|(k, &v)| (k.ops()[0], v))
            .collect();
        unigrams.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!();
        eprintln!("Per-handler instruction counts (1-grams):");
        eprintln!("{:<30} {:>12} {:>8}", "Handler", "Count", "Pct");
        eprintln!("{}", "-".repeat(52));
        for (name, count) in &unigrams {
            let pct = *count as f64 / stats.total_instructions as f64 * 100.0;
            eprintln!("{:<30} {:>12} {:>7.2}%", name, count, pct);
        }
        eprintln!("{}", "-".repeat(52));
        eprintln!("{:<30} {:>12}", "TOTAL", stats.total_instructions);
        eprintln!();
    }

    // Build pattern trie
    eprintln!("Building pattern trie...");
    let trie = PatternTrie::from_stats(&stats);

    // Print trie stats
    let depth_stats = trie.depth_stats();
    eprintln!();
    eprintln!("Trie Statistics");
    eprintln!("{}", "-".repeat(50));
    eprintln!("Total instructions profiled: {}", stats.total_instructions);
    let mut depths: Vec<_> = depth_stats.iter().collect();
    depths.sort_by_key(|(d, _)| *d);
    for (depth, count) in &depths {
        eprintln!("  Unique {}-grams: {}", depth, count);
    }
    eprintln!();

    if cmd.show_trie {
        let min_savings_abs = (cmd.min_savings_pct / 100.0 * trie.total_instructions as f64) as u64;
        trie.print_tree(cmd.max_window, min_savings_abs / (cmd.max_window as u64));
        eprintln!();
    }

    // Load handler op names to avoid name collisions
    let reserved_names = load_handler_names();
    eprintln!("Reserved handler names: {}", reserved_names.len());
    eprintln!();

    // Run discovery
    let config = DiscoveryConfig {
        max_candidates: cmd.top,
        min_savings_pct: cmd.min_savings_pct,
        reserved_names,
    };

    let candidates = discovery::discover(&trie, &config);

    if candidates.is_empty() {
        eprintln!("No new fusion candidates found above threshold.");
        return;
    }

    // Print summary
    eprintln!("Discovered Candidates");
    eprintln!("{}", "-".repeat(70));
    let total_savings: u64 = candidates.iter().map(|c| c.savings).sum();
    let total_pct = if stats.total_instructions > 0 {
        (total_savings as f64 / stats.total_instructions as f64) * 100.0
    } else {
        0.0
    };

    for (i, c) in candidates.iter().enumerate() {
        let pct = if stats.total_instructions > 0 {
            (c.savings as f64 / stats.total_instructions as f64) * 100.0
        } else {
            0.0
        };
        eprintln!(
            "  {:>2}. {} [{}] count={}, savings={} ({:.2}%)",
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

    // Generate TOML
    let toml = discovery::format_all_toml(&candidates);

    // Output
    let output_path = cmd.output.unwrap_or_else(|| PathBuf::from(DEFAULT_FUSED_TOML));
    fs::write(&output_path, &toml).unwrap_or_else(|err| {
        eprintln!("Error writing output file: {}", err);
        process::exit(1);
    });
    eprintln!("Written {} fused patterns to: {}", candidates.len(), output_path.display());
    eprintln!("Rebuild with: cargo build --release");
}

/// Load [[handler]] op names from handlers.toml to avoid name collisions.
fn load_handler_names() -> HashSet<String> {
    // Try to find handlers.toml relative to the binary or CWD
    let candidates = [
        PathBuf::from("sf-nano-core/src/vm/interp/fast/handlers.toml"),
        PathBuf::from("../sf-nano-core/src/vm/interp/fast/handlers.toml"),
    ];

    for toml_path in &candidates {
        if let Ok(content) = fs::read_to_string(toml_path) {
            return parse_handler_names(&content);
        }
    }

    eprintln!("Warning: Could not find handlers.toml");
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
