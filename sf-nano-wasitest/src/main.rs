//! WASI test runner for sf-nano
//!
//! Discovers WASI test `.wasm` files and runs each one via the `sf-nano-cli`
//! binary as a subprocess, validating exit codes and stdout/stderr output.

use log::{error, info, warn};
use serde::Deserialize;
use structopt::StructOpt;

use std::{
    collections::HashMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

const WASI_TESTSUITE_DIR: &str = env!("WASI_TESTSUITE_DIR");
const TEST_TIMEOUT_SECS: u64 = 10;

// Tests to skip (add entries as needed)
const SKIP_LIST: &[&str] = &[];

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(StructOpt, Debug)]
#[structopt(
    name = "sf-nano-wasitest",
    about = "WASI test runner for sf-nano (subprocess-based)"
)]
struct Cli {
    /// Substring filters for test names. Multiple allowed.
    #[structopt(name = "FILTERS")]
    patterns: Vec<String>,

    /// Path to the sf-nano-cli binary
    #[structopt(long = "cli-path")]
    cli_path: Option<PathBuf>,

    /// Log level (trace, debug, info, warn, error)
    #[structopt(long = "log-level")]
    log_level: Option<String>,

    /// List matching tests and exit
    #[structopt(long = "list")]
    list_only: bool,

    /// Stop after first failure
    #[structopt(long = "fail-fast")]
    fail_fast: bool,

    /// Quiet output
    #[structopt(short = "q", long = "quiet")]
    quiet: bool,
}

// ---------------------------------------------------------------------------
// Test config (JSON)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
struct TestConfig {
    args: Option<Vec<String>>,
    env: Option<HashMap<String, String>>,
    exit_code: Option<i32>,
    dirs: Option<Vec<String>>,
    stdout: Option<String>,
    stderr: Option<String>,
}

// ---------------------------------------------------------------------------
// Test result types
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct TestOutput {
    exit_code: i32,
    stdout: String,
    stderr: String,
}

#[derive(Debug)]
struct TestFailure {
    kind: String,
    message: String,
}

#[derive(Debug)]
struct TestResult {
    name: String,
    skipped: bool,
    failures: Vec<TestFailure>,
    duration: Duration,
}

impl TestResult {
    fn passed(&self) -> bool {
        !self.skipped && self.failures.is_empty()
    }
    fn failed(&self) -> bool {
        !self.failures.is_empty()
    }
    fn status(&self) -> &'static str {
        if self.skipped {
            "SKIPPED"
        } else if self.failed() {
            "FAILED"
        } else {
            "PASSED"
        }
    }
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

struct Summary {
    passed: usize,
    failed: usize,
    skipped: usize,
}

impl Summary {
    fn total(&self) -> usize {
        self.passed + self.failed + self.skipped
    }
}

fn print_summary(s: &Summary, duration: Duration) {
    println!();
    println!("=== sf-nano WASI Test Summary ===");
    println!("Total:   {}", s.total());
    println!(
        "Passed:  {} ({:.1}%)",
        s.passed,
        if s.total() > 0 {
            s.passed as f64 / s.total() as f64 * 100.0
        } else {
            0.0
        }
    );
    println!(
        "Failed:  {} ({:.1}%)",
        s.failed,
        if s.total() > 0 {
            s.failed as f64 / s.total() as f64 * 100.0
        } else {
            0.0
        }
    );
    println!(
        "Skipped: {} ({:.1}%)",
        s.skipped,
        if s.total() > 0 {
            s.skipped as f64 / s.total() as f64 * 100.0
        } else {
            0.0
        }
    );
    println!("Duration: {:.2?}", duration);
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

fn discover_wasm_files(dir: &Path) -> Vec<PathBuf> {
    let mut result = Vec::new();
    collect_wasm_files(dir, &mut result);
    result.sort();
    result
}

fn collect_wasm_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_wasm_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "wasm") {
            out.push(path);
        }
    }
}

fn match_filters(patterns: &[String], name: &str) -> bool {
    if patterns.is_empty() {
        return true;
    }
    let lower = name.to_lowercase();
    patterns.iter().any(|p| lower.contains(&p.to_lowercase()))
}

fn read_test_config(wasm_file: &Path) -> TestConfig {
    let json_path = wasm_file.with_extension("json");
    if !json_path.exists() {
        return TestConfig::default();
    }
    match fs::read_to_string(&json_path) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_else(|e| {
            warn!("Failed to parse {}: {}", json_path.display(), e);
            TestConfig::default()
        }),
        Err(e) => {
            warn!("Failed to read {}: {}", json_path.display(), e);
            TestConfig::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Locate sf-nano-cli
// ---------------------------------------------------------------------------

fn find_cli_binary() -> Option<PathBuf> {
    let exe_name = if cfg!(windows) {
        "sf-nano-cli.exe"
    } else {
        "sf-nano-cli"
    };

    // 1. Next to the current binary
    if let Ok(self_exe) = std::env::current_exe() {
        if let Some(dir) = self_exe.parent() {
            let candidate = dir.join(exe_name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    // 2. Common cargo target directories
    for profile in &["debug", "release"] {
        let candidate = PathBuf::from(format!("target/{}/{}", profile, exe_name));
        if candidate.exists() {
            return Some(candidate);
        }
    }

    // 3. Fall back to PATH
    if Command::new(exe_name)
        .arg("--help")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
    {
        return Some(PathBuf::from(exe_name));
    }

    None
}

// ---------------------------------------------------------------------------
// Run a single test
// ---------------------------------------------------------------------------

fn run_test(wasm_file: &Path, cli_path: &Path) -> TestResult {
    let test_name = wasm_file
        .file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    let start = Instant::now();

    // Check skip list
    if SKIP_LIST.contains(&test_name.as_str()) {
        return TestResult {
            name: test_name,
            skipped: true,
            failures: vec![],
            duration: start.elapsed(),
        };
    }

    let config = read_test_config(wasm_file);

    // Build command
    let mut cmd = Command::new(cli_path);
    cmd.arg(wasm_file);

    // Add test arguments
    if let Some(ref args) = config.args {
        cmd.args(args);
    }

    // Environment: clear, preserve PATH, then set test-specified vars
    cmd.env_clear();
    if let Ok(path) = std::env::var("PATH") {
        cmd.env("PATH", path);
    }
    if let Some(ref env_vars) = config.env {
        for (key, value) in env_vars {
            cmd.env(key, value);
        }
    }

    // Set working directory to parent of wasm file for dir preopens
    if let Some(parent) = wasm_file.parent() {
        cmd.current_dir(parent);
    }

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    // Spawn and wait with timeout
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return TestResult {
                name: test_name,
                skipped: false,
                failures: vec![TestFailure {
                    kind: "spawn".into(),
                    message: format!("Failed to spawn sf-nano-cli: {}", e),
                }],
                duration: start.elapsed(),
            };
        }
    };

    let timeout = Duration::from_secs(TEST_TIMEOUT_SECS);
    let poll_interval = Duration::from_millis(100);

    let output = loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout_buf = Vec::new();
                let mut stderr_buf = Vec::new();
                if let Some(mut out) = child.stdout.take() {
                    let _ = out.read_to_end(&mut stdout_buf);
                }
                if let Some(mut err) = child.stderr.take() {
                    let _ = err.read_to_end(&mut stderr_buf);
                }
                break Ok(TestOutput {
                    exit_code: status.code().unwrap_or(-1),
                    stdout: String::from_utf8_lossy(&stdout_buf).to_string(),
                    stderr: String::from_utf8_lossy(&stderr_buf).to_string(),
                });
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    break Err(format!(
                        "Test timed out after {} seconds",
                        timeout.as_secs()
                    ));
                }
                std::thread::sleep(poll_interval);
            }
            Err(e) => {
                let _ = child.kill();
                break Err(format!("Failed to check process status: {}", e));
            }
        }
    };

    match output {
        Ok(output) => {
            let failures = validate_output(&config, &output);
            TestResult {
                name: test_name,
                skipped: false,
                failures,
                duration: start.elapsed(),
            }
        }
        Err(msg) => TestResult {
            name: test_name,
            skipped: false,
            failures: vec![TestFailure {
                kind: "execution".into(),
                message: msg,
            }],
            duration: start.elapsed(),
        },
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_output(config: &TestConfig, output: &TestOutput) -> Vec<TestFailure> {
    let mut failures = Vec::new();

    // Exit code
    let expected_exit = config.exit_code.unwrap_or(0);
    if expected_exit != output.exit_code {
        let mut msg = format!("expected {} == actual {}", expected_exit, output.exit_code);
        let stderr_trim = output.stderr.trim();
        if !stderr_trim.is_empty() {
            let snippet = if stderr_trim.len() > 500 {
                format!("{}…", &stderr_trim[..500])
            } else {
                stderr_trim.to_string()
            };
            msg.push_str(" | stderr: ");
            msg.push_str(&snippet);
        }
        failures.push(TestFailure {
            kind: "exit_code".into(),
            message: msg,
        });
    }

    // Stdout
    if let Some(ref expected) = config.stdout {
        if expected != &output.stdout {
            failures.push(TestFailure {
                kind: "stdout".into(),
                message: format!("expected '{}' == actual '{}'", expected, output.stdout),
            });
        }
    }

    // Stderr
    if let Some(ref expected) = config.stderr {
        if expected != &output.stderr {
            failures.push(TestFailure {
                kind: "stderr".into(),
                message: format!("expected '{}' == actual '{}'", expected, output.stderr),
            });
        }
    }

    failures
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::from_args();

    // Initialize logging
    if let Some(ref level_str) = cli.log_level {
        let log_level = match level_str.to_lowercase().as_str() {
            "trace" => log::LevelFilter::Trace,
            "debug" => log::LevelFilter::Debug,
            "info" => log::LevelFilter::Info,
            "warn" => log::LevelFilter::Warn,
            "error" => log::LevelFilter::Error,
            _ => {
                eprintln!(
                    "Invalid log level '{}'. Valid: trace, debug, info, warn, error",
                    level_str
                );
                std::process::exit(1);
            }
        };
        env_logger::Builder::new().filter_level(log_level).init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    }

    // Resolve sf-nano-cli path
    let cli_path = cli.cli_path.clone().or_else(find_cli_binary);
    let cli_path = match cli_path {
        Some(p) => p,
        None => {
            error!("Could not find sf-nano-cli binary. Use --cli-path or build it first.");
            std::process::exit(1);
        }
    };
    info!("Using sf-nano-cli: {}", cli_path.display());

    let testsuite_path = Path::new(WASI_TESTSUITE_DIR);
    if !testsuite_path.exists() {
        error!(
            "WASI testsuite not found at: {}. Run 'cargo build' first.",
            WASI_TESTSUITE_DIR
        );
        std::process::exit(1);
    }
    info!("Using WASI testsuite from: {}", WASI_TESTSUITE_DIR);

    // Discover tests
    let all_wasm = discover_wasm_files(testsuite_path);
    let filtered: Vec<_> = all_wasm
        .iter()
        .filter(|p| {
            let name = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            match_filters(&cli.patterns, name)
        })
        .collect();

    if cli.list_only {
        for wasm in &filtered {
            println!(
                "{}",
                wasm.file_stem().and_then(|s| s.to_str()).unwrap_or("?")
            );
        }
        println!("Total matching tests: {}", filtered.len());
        return;
    }

    if !cli.quiet {
        println!("Running {} WASI tests...", filtered.len());
        println!();
    }

    let start = Instant::now();
    let mut results = Vec::new();

    for wasm in &filtered {
        let result = run_test(wasm, &cli_path);

        if !cli.quiet {
            let symbol = match result.status() {
                "PASSED" => "✅",
                "FAILED" => "❌",
                "SKIPPED" => "⏭",
                _ => "❓",
            };
            if result.failed() {
                let msg = result
                    .failures
                    .iter()
                    .map(|f| format!("{}: {}", f.kind, f.message))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!(
                    "  {} {} ({:.2?}) - {}",
                    symbol, result.name, result.duration, msg
                );
            } else {
                println!("  {} {} ({:.2?})", symbol, result.name, result.duration);
            }
        }

        let should_stop = cli.fail_fast && result.failed();
        results.push(result);
        if should_stop {
            break;
        }
    }

    let summary = Summary {
        passed: results.iter().filter(|r| r.passed()).count(),
        failed: results.iter().filter(|r| r.failed()).count(),
        skipped: results.iter().filter(|r| r.skipped).count(),
    };

    print_summary(&summary, start.elapsed());

    if summary.failed > 0 {
        std::process::exit(1);
    }
}
