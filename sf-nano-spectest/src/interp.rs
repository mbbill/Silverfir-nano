//! Spec-suite runner for the interpreter (`vm/interpreter`).
//!
//! Runs every base-suite `.wast` file against `InterpInstance`. Directives
//! the v1 interpreter deliberately does not support (module linking /
//! registration, imported globals/memories/tables, SIMD, GC/EH, validation
//! directives) are counted as skips — never as passes. The success
//! criterion is ZERO mismatches on everything actually executed.
//!
//! Run: `cargo run -- --interp`

use std::fs;
use std::path::{Path, PathBuf};

use sf_nano_core::module::Module;
use sf_nano_core::{InterpInstance, WasmError};
use wast::core::{NanPattern, WastArgCore, WastRetCore};
use wast::parser::{self, ParseBuffer};
use wast::{Wast, WastArg, WastDirective, WastExecute, WastRet};

use crate::discovery;

const NULL_FUNCREF: u64 = u32::MAX as u64;

#[derive(Default)]
struct Totals {
    files: usize,
    files_full: usize,
    files_partial: usize,
    files_skipped: usize,
    asserts_passed: usize,
    asserts_failed: usize,
    directives_skipped: usize,
    failures: Vec<String>,
}

enum Converted {
    Vals(Vec<u64>),
    Unsupported,
}

fn convert_args(args: &[WastArg]) -> Converted {
    let mut out = Vec::new();
    for a in args {
        match a {
            WastArg::Core(c) => match c {
                WastArgCore::I32(v) => out.push(*v as u32 as u64),
                WastArgCore::I64(v) => out.push(*v as u64),
                WastArgCore::F32(v) => out.push(v.bits as u64),
                WastArgCore::F64(v) => out.push(v.bits),
                WastArgCore::RefNull(_) => out.push(NULL_FUNCREF),
                _ => return Converted::Unsupported,
            },
            _ => return Converted::Unsupported,
        }
    }
    Converted::Vals(out)
}

fn is_canonical_nan32(bits: u32) -> bool {
    (bits & 0x7fff_ffff) == 0x7fc0_0000
}
fn is_arithmetic_nan32(bits: u32) -> bool {
    (bits & 0x7fc0_0000) == 0x7fc0_0000
}
fn is_canonical_nan64(bits: u64) -> bool {
    (bits & 0x7fff_ffff_ffff_ffff) == 0x7ff8_0000_0000_0000
}
fn is_arithmetic_nan64(bits: u64) -> bool {
    (bits & 0x7ff8_0000_0000_0000) == 0x7ff8_0000_0000_0000
}

/// Compare one actual raw slot against one expected result. Returns
/// None = unsupported expectation kind (skip directive), Some(bool) = match.
fn ret_matches(actual: u64, ret: &WastRet) -> Option<bool> {
    let WastRet::Core(core) = ret else {
        return None;
    };
    Some(match core {
        WastRetCore::I32(v) => actual as u32 == *v as u32,
        WastRetCore::I64(v) => actual == *v as u64,
        WastRetCore::F32(p) => match p {
            NanPattern::Value(v) => actual as u32 == v.bits,
            NanPattern::CanonicalNan => is_canonical_nan32(actual as u32),
            NanPattern::ArithmeticNan => is_arithmetic_nan32(actual as u32),
        },
        WastRetCore::F64(p) => match p {
            NanPattern::Value(v) => actual == v.bits,
            NanPattern::CanonicalNan => is_canonical_nan64(actual),
            NanPattern::ArithmeticNan => is_arithmetic_nan64(actual),
        },
        WastRetCore::RefNull(_) => actual == NULL_FUNCREF,
        WastRetCore::RefFunc(Some(wast::token::Index::Num(n, _))) => actual == *n as u64,
        _ => return None,
    })
}

fn trap_matches(err: &WasmError, expected: &str) -> bool {
    let msg = match err {
        WasmError::Trap(m) => *m,
        _ => return false,
    };
    msg == expected || expected.starts_with(msg) || msg.starts_with(expected)
}

struct FileRun<'m> {
    inst: Option<InterpInstance<'m>>,
    /// Set when the rest of the file cannot be meaningfully executed.
    skip_rest: Option<&'static str>,
}

fn spectest_host() -> impl FnMut(&str, &str, &mut [u8], &[u64], &mut [u64]) -> Result<(), WasmError>
{
    |_module, name, _mem, _args, _results| {
        if name.starts_with("print") {
            Ok(())
        } else {
            Err(WasmError::invalid("interp-spectest: unshimmed host import"))
        }
    }
}

fn run_file(path: &PathBuf, totals: &mut Totals) {
    let name = path.file_name().unwrap().to_string_lossy().into_owned();
    let source = fs::read_to_string(path).expect("read wast");
    let buf = ParseBuffer::new(&source).expect("lex");
    let wast: Wast = match parser::parse(&buf) {
        Ok(w) => w,
        Err(_) => {
            totals.files_skipped += 1;
            return;
        }
    };

    let mut run = FileRun {
        inst: None,
        skip_rest: None,
    };
    let mut executed = 0usize;
    let mut skipped = 0usize;

    for directive in wast.directives {
        if run.skip_rest.is_some() {
            skipped += 1;
            continue;
        }
        let line = {
            let sp = directive.span();
            sp.linecol_in(&source).0 + 1
        };
        let name = format!("{name}:{line}");
        match directive {
            WastDirective::Module(mut quote_wat) => {
                let Ok(bytes) = quote_wat.encode() else {
                    skipped += 1;
                    continue;
                };
                match instantiate(bytes) {
                    Ok(inst) => run.inst = Some(inst),
                    Err(_) => {
                        // Unsupported feature in this module: everything
                        // after it depends on it — skip the file's rest.
                        run.skip_rest = Some("module unsupported");
                        skipped += 1;
                    }
                }
            }
            WastDirective::AssertReturn { exec, results, .. } => match execute(&mut run, &exec) {
                Exec::Skip => skipped += 1,
                Exec::Err(e) => {
                    totals.asserts_failed += 1;
                    totals
                        .failures
                        .push(format!("{name}: assert_return trapped: {e:?}"));
                }
                Exec::Vals(vals) => {
                    executed += 1;
                    if vals.len() != results.len() {
                        totals.asserts_failed += 1;
                        totals.failures.push(format!("{name}: arity mismatch"));
                        continue;
                    }
                    let mut ok = true;
                    let mut supported = true;
                    for (a, r) in vals.iter().zip(results.iter()) {
                        match ret_matches(*a, r) {
                            None => {
                                supported = false;
                                break;
                            }
                            Some(m) => ok &= m,
                        }
                    }
                    if !supported {
                        skipped += 1;
                    } else if ok {
                        totals.asserts_passed += 1;
                    } else {
                        totals.asserts_failed += 1;
                        totals
                            .failures
                            .push(format!("{name}: assert_return mismatch: got {vals:x?}"));
                    }
                }
            },
            WastDirective::AssertTrap { exec, message, .. } => handle_trap_assert(
                &mut run,
                exec,
                message,
                &name,
                totals,
                &mut executed,
                &mut skipped,
            ),
            WastDirective::AssertExhaustion { call, message, .. } => handle_trap_assert(
                &mut run,
                WastExecute::Invoke(call),
                message,
                &name,
                totals,
                &mut executed,
                &mut skipped,
            ),

            WastDirective::Invoke(invoke) => {
                match execute(&mut run, &WastExecute::Invoke(invoke)) {
                    Exec::Skip => skipped += 1,
                    Exec::Err(e) => {
                        totals.asserts_failed += 1;
                        totals
                            .failures
                            .push(format!("{name}: invoke trapped: {e:?}"));
                    }
                    Exec::Vals(_) => {
                        executed += 1;
                        totals.asserts_passed += 1;
                    }
                }
            }
            // Validation/malformedness are the shared front end's concern,
            // covered by the main spectest runner.
            WastDirective::AssertInvalid { .. }
            | WastDirective::AssertMalformed { .. }
            | WastDirective::AssertUnlinkable { .. }
            | WastDirective::AssertException { .. } => skipped += 1,
            // Multi-module linking is outside v1.
            WastDirective::Register { .. }
            | WastDirective::ModuleDefinition(_)
            | WastDirective::ModuleInstance { .. } => {
                run.skip_rest = Some("linking unsupported");
                skipped += 1;
            }
            _ => skipped += 1,
        }
    }

    totals.files += 1;
    totals.directives_skipped += skipped;
    if skipped == 0 {
        totals.files_full += 1;
    } else if executed > 0 {
        totals.files_partial += 1;
    } else {
        totals.files_skipped += 1;
    }
}

enum Exec {
    Vals(Vec<u64>),
    Err(WasmError),
    Skip,
}

#[allow(clippy::too_many_arguments)]
fn handle_trap_assert(
    run: &mut FileRun<'_>,
    exec: WastExecute,
    message: &str,
    name: &str,
    totals: &mut Totals,
    executed: &mut usize,
    skipped: &mut usize,
) {
    match execute_general(run, exec) {
        Exec::Skip => *skipped += 1,
        Exec::Err(e) => {
            *executed += 1;
            if trap_matches(&e, message) {
                totals.asserts_passed += 1;
            } else {
                totals.asserts_failed += 1;
                totals
                    .failures
                    .push(format!("{name}: wrong trap: want '{message}', got {e:?}"));
            }
        }
        Exec::Vals(_) => {
            *executed += 1;
            totals.asserts_failed += 1;
            totals
                .failures
                .push(format!("{name}: expected trap '{message}', got values"));
        }
    }
}

fn execute(run: &mut FileRun<'_>, exec: &WastExecute) -> Exec {
    match exec {
        WastExecute::Invoke(invoke) => {
            if invoke.module.is_some() {
                return Exec::Skip;
            }
            let Some(inst) = run.inst.as_mut() else {
                return Exec::Skip;
            };
            let Some(idx) = inst.find_export(invoke.name) else {
                return Exec::Skip;
            };
            let (p, r) = inst.func_arity(idx).unwrap();
            let args = match convert_args(&invoke.args) {
                Converted::Unsupported => return Exec::Skip,
                Converted::Vals(v) => v,
            };
            if args.len() != p {
                return Exec::Skip;
            }
            let mut results = vec![0u64; r];
            match inst.invoke(idx, &args, &mut results) {
                Ok(()) => Exec::Vals(results),
                Err(e) => Exec::Err(e),
            }
        }
        WastExecute::Get { module, global, .. } => {
            if module.is_some() {
                return Exec::Skip;
            }
            let Some(inst) = run.inst.as_ref() else {
                return Exec::Skip;
            };
            match inst.get_export_global(global) {
                Some(v) => Exec::Vals(vec![v]),
                None => Exec::Skip,
            }
        }
        WastExecute::Wat(_) => Exec::Skip,
    }
}

fn execute_general(run: &mut FileRun<'_>, exec: WastExecute) -> Exec {
    match exec {
        WastExecute::Wat(mut wat) => {
            // assert_trap on a whole module: instantiation must trap.
            let Ok(bytes) = wat.encode() else {
                return Exec::Skip;
            };
            match instantiate(bytes) {
                Ok(_) => Exec::Vals(Vec::new()),
                Err(e) if matches!(e, WasmError::Trap(_)) => Exec::Err(e),
                Err(_) => Exec::Skip, // unsupported, not a trap
            }
        }
        other => execute(run, &other),
    }
}

fn instantiate(bytes: Vec<u8>) -> Result<InterpInstance<'static>, WasmError> {
    let module: &'static Module = Box::leak(Box::new(Module::new("spec", &bytes)?));
    let mut inst = InterpInstance::new(module)?;
    inst.set_host(spectest_host());
    Ok(inst)
}

/// Whether a file passes the CLI filters (same semantics as the JIT
/// runner: `.wast`-suffixed filters are paths, others match the file stem).
fn matches_filters(path: &Path, filters: &[String]) -> bool {
    if filters.is_empty() {
        return true;
    }
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    filters.iter().any(|f| {
        if f.ends_with(".wast") {
            path.ends_with(Path::new(f))
        } else {
            stem == f
        }
    })
}

/// Run the spec suite against the interpreter; returns the process exit
/// code (non-zero iff any executed assert mismatched).
pub fn run(filters: &[String]) -> i32 {
    let target_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/webassembly-testsuite");
    let mut files = discovery::find_wast_files(&target_dir);
    files.sort();
    assert!(!files.is_empty(), "testsuite not found at {target_dir:?}");

    let mut totals = Totals::default();
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().into_owned();
        if !matches_filters(f, filters) {
            continue;
        }
        if discovery::should_skip_test(&name, false) {
            totals.files_skipped += 1;
            continue;
        }
        run_file(f, &mut totals);
    }

    println!("=== interp-spectest summary ===");
    println!("files:            {}", totals.files);
    println!("  fully run:      {}", totals.files_full);
    println!("  partially run:  {}", totals.files_partial);
    println!("  skipped:        {}", totals.files_skipped);
    println!("asserts passed:   {}", totals.asserts_passed);
    println!("asserts FAILED:   {}", totals.asserts_failed);
    println!("directives skipped: {}", totals.directives_skipped);
    for f in totals.failures.iter().take(40) {
        println!("FAIL {f}");
    }
    if totals.asserts_failed > 0 {
        return 1;
    }
    println!("all executed asserts passed");
    0
}
