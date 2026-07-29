//! WASI test runner for silverfir-nano.
//!
//! This runner is intentionally aligned with the `silverfir-rs` WASI harness:
//! same testsuite layout, same suite discovery rules, and the same subprocess
//! execution model with timeout protection. The subprocess target is
//! `sf-nano-cli`, so failures reflect nano behavior instead of the old
//! interpreter.

mod discovery;
mod summary;
mod types;
mod utils;

use discovery::{discover_suite_tests, find_test_suite_directories, match_filters, read_manifest};
use log::{error, info, warn};
use summary::print_test_summary;
use types::{TestConfig, TestFailure, TestOutput, TestResult, TestSuite};
use utils::copy_dir_recursive;

use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};
use structopt::StructOpt;

const WASI_TESTSUITE_DIR: &str = env!("WASI_TESTSUITE_DIR");
const TEST_TIMEOUT_SECS: u64 = 10;
// qemu-riscv32-static in the Colima/Linux runner reports RV32 Linux timestamp
// update syscalls used by Rust's filetime path as unimplemented (`statx`
// succeeds, then syscall 412 and the legacy syscall 88 both return ENOSYS).
// Keep this as an explicit runner option so native hosts and other targets
// continue to execute the same WASI tests.
const RV32_QEMU_TIMESTAMP_TESTS: &[&str] =
    &["fd_filestat_set", "path_filestat", "symlink_filestat"];

#[derive(StructOpt, Debug, Clone)]
#[structopt(
    name = "sf-nano-wasitest",
    about = "Run wasi-testsuite against silverfir-nano"
)]
struct Cli {
    /// Substring filters for test names (file stems). Multiple allowed.
    #[structopt(name = "FILTERS", parse(from_str))]
    patterns: Vec<String>,

    /// Path to the sf-nano-cli binary.
    #[structopt(long = "cli-path")]
    cli_path: Option<PathBuf>,

    /// List matching tests and exit.
    #[structopt(long = "list")]
    list_only: bool,

    /// Stop after first failing suite.
    #[structopt(long = "fail-fast")]
    fail_fast: bool,

    /// Log level (trace, debug, info, warn, error).
    #[structopt(long = "log-level")]
    log_level: Option<String>,

    /// Quiet output (suppress per-test prints).
    #[structopt(short = "q", long = "quiet")]
    quiet: bool,

    /// Backend to use for execution (auto, native, jit).
    #[structopt(long = "backend")]
    backend: Option<String>,

    /// Run every test through the interpreter instead of the JIT. Forwarded
    /// to sf-nano-cli, which needs to have been built with `interp`.
    #[structopt(long = "interp")]
    interp: bool,

    /// Override per-function compiler RAM budget passed to sf-nano-cli, e.g. 0, 400K, 2M.
    #[structopt(long = "compiler-ram-budget")]
    compiler_ram_budget: Option<String>,

    /// Skip RV32 qemu-user timestamp-setting tests whose syscalls are not implemented.
    #[structopt(long = "skip-rv32-qemu-timestamp-tests")]
    skip_rv32_qemu_timestamp_tests: bool,
}

fn main() {
    let cli = Cli::from_args();
    init_logging(cli.log_level.as_deref());

    let cli_path = resolve_cli_path(cli.cli_path.clone());
    info!("Using sf-nano-cli: {}", cli_path.display());
    info!("Using WASI testsuite from: {}", WASI_TESTSUITE_DIR);

    let testsuite_path = Path::new(WASI_TESTSUITE_DIR);
    if !testsuite_path.exists() {
        error!(
            "WASI testsuite directory not found at: {}",
            WASI_TESTSUITE_DIR
        );
        error!("This should have been downloaded during build. Try running `cargo build` first.");
        std::process::exit(1);
    }

    let test_suite_dirs = find_test_suite_directories(testsuite_path);
    info!("Found {} test suite directories", test_suite_dirs.len());
    if test_suite_dirs.is_empty() {
        // A suite that discovers nothing has not passed, it has not run. This
        // exited 0 once, and a testsuite pin carrying no `.wasm` files sat in
        // CI reporting green 0.0s runs on every architecture because of it.
        error!(
            "No test suite directories found under {}. The testsuite checkout \
             has no .wasm files -- check the pin in build.rs.",
            testsuite_path.display()
        );
        std::process::exit(1);
    }

    if cli.list_only {
        let mut total = 0usize;
        for suite_dir in &test_suite_dirs {
            let (suite_name, tests) = discover_suite_tests(suite_dir);
            let mut matched: Vec<_> = tests
                .into_iter()
                .filter(|(_, name)| match_filters(&cli.patterns, name))
                .collect();
            if !matched.is_empty() {
                println!("Suite: {}", suite_name);
                matched.sort_by(|a, b| a.1.cmp(&b.1));
                for (_, name) in matched {
                    println!("  - {}", name);
                    total += 1;
                }
            }
        }
        println!("Total matching tests: {}", total);
        return;
    }

    let start_time = Instant::now();
    let mut suites = Vec::new();

    for suite_dir in test_suite_dirs {
        let suite = run_test_suite(&suite_dir, &cli_path, &cli);
        info!(
            "Suite '{}': {} total, {} passed, {} failed, {} skipped",
            suite.name,
            suite.total_count(),
            suite.passed_count(),
            suite.failed_count(),
            suite.skipped_count()
        );
        let suite_failed = suite.failed_count() > 0;
        suites.push(suite);
        if cli.fail_fast && suite_failed {
            break;
        }
    }

    let total_duration = start_time.elapsed();
    print_test_summary(&suites, total_duration);

    let total_failures: usize = suites.iter().map(|suite| suite.failed_count()).sum();
    if total_failures > 0 {
        std::process::exit(1);
    }
}

fn init_logging(level: Option<&str>) {
    if let Some(level_str) = level {
        let log_level = match level_str.to_lowercase().as_str() {
            "trace" => log::LevelFilter::Trace,
            "debug" => log::LevelFilter::Debug,
            "info" => log::LevelFilter::Info,
            "warn" => log::LevelFilter::Warn,
            "error" => log::LevelFilter::Error,
            _ => {
                eprintln!(
                    "Invalid log level '{}'. Valid options: trace, debug, info, warn, error",
                    level_str
                );
                std::process::exit(1);
            }
        };
        env_logger::Builder::new().filter_level(log_level).init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    }
}

fn resolve_cli_path(cli_path: Option<PathBuf>) -> PathBuf {
    if let Some(path) = cli_path {
        return path;
    }
    find_cli_binary().unwrap_or_else(|| {
        error!("Could not find sf-nano-cli. Use --cli-path or build it first.");
        std::process::exit(1);
    })
}

fn find_cli_binary() -> Option<PathBuf> {
    let exe_name = if cfg!(windows) {
        "sf-nano-cli.exe"
    } else {
        "sf-nano-cli"
    };

    if let Ok(self_exe) = std::env::current_exe() {
        if let Some(dir) = self_exe.parent() {
            let candidate = dir.join(exe_name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    for profile in &["debug", "release"] {
        let candidate = PathBuf::from(format!("target/{}/{}", profile, exe_name));
        if candidate.exists() {
            return Some(candidate);
        }
    }

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

fn run_test_suite(suite_dir: &Path, cli_path: &Path, cli: &Cli) -> TestSuite {
    let suite_name = read_manifest(suite_dir).unwrap_or_else(|| {
        suite_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string()
    });
    let (_, tests) = discover_suite_tests(suite_dir);

    let tests: Vec<_> = tests
        .into_iter()
        .filter(|(_, name)| match_filters(&cli.patterns, name))
        .collect();

    if !cli.quiet && !tests.is_empty() {
        println!("Running suite: {}", suite_name);
    }

    let mut test_results = Vec::new();
    for (wasm_file, test_name) in tests {
        let result = if should_skip_test(&test_name, cli) {
            TestResult {
                name: test_name,
                is_executed: false,
                skip_reason: Some(rv32_qemu_timestamp_skip_reason()),
                failures: Vec::new(),
                duration: Duration::default(),
            }
        } else {
            execute_single_test(&wasm_file, &test_name, cli_path, cli)
        };
        if !cli.quiet {
            print_test_result(&result);
        }
        test_results.push(result);
    }

    if !cli.quiet && !test_results.is_empty() {
        println!();
    }

    TestSuite {
        name: suite_name,
        test_results,
    }
}

fn should_skip_test(test_name: &str, cli: &Cli) -> bool {
    cli.skip_rv32_qemu_timestamp_tests && RV32_QEMU_TIMESTAMP_TESTS.contains(&test_name)
}

fn rv32_qemu_timestamp_skip_reason() -> String {
    "qemu-riscv32-static reports RV32 Linux timestamp-setting syscalls as ENOSYS".to_string()
}

fn print_test_result(result: &TestResult) {
    let status_symbol = match result.status() {
        "PASSED" => "OK",
        "FAILED" => "FAIL",
        "SKIPPED" => "SKIP",
        _ => "?",
    };
    if result.failed() {
        let failure_msg = result
            .failures
            .iter()
            .map(|failure| format!("{}: {}", failure.failure_type, failure.message))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "  {} {} ({:.2?}) - {}",
            status_symbol, result.name, result.duration, failure_msg
        );
    } else if !result.is_executed {
        if let Some(reason) = result.skip_reason.as_deref() {
            println!(
                "  {} {} ({:.2?}) - {}",
                status_symbol, result.name, result.duration, reason
            );
        } else {
            println!(
                "  {} {} ({:.2?})",
                status_symbol, result.name, result.duration
            );
        }
    } else {
        println!(
            "  {} {} ({:.2?})",
            status_symbol, result.name, result.duration
        );
    }
}

fn execute_single_test(
    wasm_file: &Path,
    test_name: &str,
    cli_path: &Path,
    cli: &Cli,
) -> TestResult {
    let start_time = Instant::now();
    let config = read_test_config(wasm_file);
    let wasm_data = match fs::read(wasm_file) {
        Ok(data) => data,
        Err(err) => {
            return TestResult {
                name: test_name.to_string(),
                is_executed: true,
                skip_reason: None,
                failures: vec![TestFailure {
                    failure_type: "file_read".to_string(),
                    message: format!("Failed to read test file: {}", err),
                }],
                duration: start_time.elapsed(),
            };
        }
    };

    match run_wasm_test_with_subprocess(
        test_name,
        &wasm_data,
        &config,
        wasm_file.parent(),
        cli_path,
        cli,
    ) {
        Ok(output) => {
            let failures = validate_test_output(&config, &output);
            TestResult {
                name: test_name.to_string(),
                is_executed: true,
                skip_reason: None,
                failures,
                duration: start_time.elapsed(),
            }
        }
        Err(err) => TestResult {
            name: test_name.to_string(),
            is_executed: true,
            skip_reason: None,
            failures: vec![TestFailure {
                failure_type: "execution".to_string(),
                message: err,
            }],
            duration: start_time.elapsed(),
        },
    }
}

fn read_test_config(wasm_file: &Path) -> TestConfig {
    let config_file = wasm_file.with_extension("json");
    if !config_file.exists() {
        return TestConfig::default();
    }

    match fs::read_to_string(&config_file) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_else(|err| {
            warn!(
                "Failed to parse config file {}: {}",
                config_file.display(),
                err
            );
            TestConfig::default()
        }),
        Err(err) => {
            warn!(
                "Failed to read config file {}: {}",
                config_file.display(),
                err
            );
            TestConfig::default()
        }
    }
}

/// Remove `*.cleanup` entries anywhere under `dir`, the suite's marker for
/// files a test creates and does not guarantee to remove.
fn remove_cleanup_entries(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let is_cleanup = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".cleanup"));
        if is_cleanup {
            let _ = if path.is_dir() {
                fs::remove_dir_all(&path)
            } else {
                fs::remove_file(&path)
            };
        } else if path.is_dir() {
            remove_cleanup_entries(&path);
        }
    }
}

fn run_wasm_test_with_subprocess(
    test_name: &str,
    wasm_data: &[u8],
    config: &TestConfig,
    fixture_dir: Option<&Path>,
    cli_path: &Path,
    cli: &Cli,
) -> Result<TestOutput, String> {
    let temp_root = std::env::temp_dir().join("sf-nano-wasitest").join(format!(
        "{}-{}",
        std::process::id(),
        test_name
    ));
    fs::create_dir_all(&temp_root)
        .map_err(|err| format!("Failed to create temp directory: {}", err))?;
    let temp_wasm_path = temp_root.join(format!("{}.wasm", test_name));
    fs::write(&temp_wasm_path, wasm_data)
        .map_err(|err| format!("Failed to write temp WASM file: {}", err))?;

    let mut cmd = Command::new(cli_path);
    if cli.interp {
        cmd.arg("--interp");
    }
    if let Some(ref backend) = cli.backend {
        cmd.arg("--backend").arg(backend);
    }
    if let Some(budget) = compiler_ram_budget_setting(cli) {
        cmd.arg("--compiler-ram-budget").arg(budget);
    }

    // `root` preopens a directory as the guest's `/`, per the suite's
    // specification.md. Every filesystem test in the current suite declares
    // its scratch area this way, so a runner that honours only `dirs` passes
    // no preopen at all and each one dies looking for its working directory.
    if let Some(ref root) = config.root {
        let src = fixture_dir
            .map(|dir| dir.join(root))
            .filter(|path| path.is_dir())
            .ok_or_else(|| {
                format!(
                    "test declares root `{}` but no such directory sits beside it",
                    root
                )
            })?;
        let dst = temp_root.join("root");
        fs::create_dir_all(&dst)
            .map_err(|err| format!("Failed to create root preopen {}: {}", dst.display(), err))?;
        copy_dir_recursive(&src, &dst).map_err(|err| {
            format!(
                "Failed to copy root fixture into {}: {}",
                dst.display(),
                err
            )
        })?;
        // The suite's convention: `*.cleanup` entries are scratch a previous
        // run may have left behind, and a test is entitled to a root without
        // them. The fixture is copied, never written through, so clearing the
        // copy is enough.
        remove_cleanup_entries(&dst);
        cmd.arg("--dir").arg(format!("/::{}", dst.display()));
    }

    cmd.arg(&temp_wasm_path);

    if let Some(ref args) = config.args {
        cmd.args(args);
    }

    cmd.env_clear();
    if let Ok(path) = std::env::var("PATH") {
        cmd.env("PATH", path);
    }
    cmd.env("WASI_TEST_ONLY_SPECIFIED_VARS", "1");
    if let Some(ref env_vars) = config.env {
        for (key, value) in env_vars {
            cmd.env(key, value);
        }
    }

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|err| format!("Failed to spawn sf-nano-cli: {}", err))?;

    let timeout = Duration::from_secs(TEST_TIMEOUT_SECS);
    let poll_interval = Duration::from_millis(100);
    let start = Instant::now();

    let output = loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout_buf = Vec::new();
                let mut stderr_buf = Vec::new();
                if let Some(mut stdout) = child.stdout.take() {
                    let _ = stdout.read_to_end(&mut stdout_buf);
                }
                if let Some(mut stderr) = child.stderr.take() {
                    let _ = stderr.read_to_end(&mut stderr_buf);
                }

                break Ok(TestOutput {
                    exit_code: if status.success() {
                        0
                    } else {
                        status.code().unwrap_or(-1)
                    },
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
            Err(err) => {
                let _ = child.kill();
                break Err(format!("Failed to check process status: {}", err));
            }
        }
    };

    let _ = fs::remove_file(&temp_wasm_path);
    let _ = fs::remove_dir_all(&temp_root);

    output
}

fn compiler_ram_budget_setting(cli: &Cli) -> Option<String> {
    cli.compiler_ram_budget
        .clone()
        .or_else(|| std::env::var("SF_NANO_COMPILER_RAM_BUDGET").ok())
}

fn validate_test_output(config: &TestConfig, output: &TestOutput) -> Vec<TestFailure> {
    let mut failures = Vec::new();

    let expected_exit = config.exit_code.unwrap_or(0);
    if expected_exit != output.exit_code {
        let mut message = format!("expected {} == actual {}", expected_exit, output.exit_code);
        let stderr_trim = output.stderr.trim();
        if !stderr_trim.is_empty() {
            let snippet = if stderr_trim.len() > 500 {
                format!("{}...", &stderr_trim[..500])
            } else {
                stderr_trim.to_string()
            };
            message.push_str(" | stderr: ");
            message.push_str(&snippet);
        }
        failures.push(TestFailure {
            failure_type: "exit_code".to_string(),
            message,
        });
    }

    if let Some(ref expected_stdout) = config.stdout {
        if expected_stdout != &output.stdout {
            failures.push(TestFailure {
                failure_type: "stdout".to_string(),
                message: format!(
                    "expected {:?} == actual {:?}",
                    expected_stdout, output.stdout
                ),
            });
        }
    }

    if let Some(ref expected_stderr) = config.stderr {
        if expected_stderr != &output.stderr {
            failures.push(TestFailure {
                failure_type: "stderr".to_string(),
                message: format!(
                    "expected {:?} == actual {:?}",
                    expected_stderr, output.stderr
                ),
            });
        }
    }

    failures
}
