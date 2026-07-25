//! End-to-end: the interpreter runs real WASI benchmark binaries from
//! `benchmarks/wasi/` through a minimal host shim. This is the completion
//! check for the import/host boundary: argument marshalling, memory access
//! from host functions, and program output all flow through the
//! interpreter's own execution.
//!
//! Gated on the target having a generated dispatch engine — the same
//! condition `enable_native_dispatch` uses, so one grep finds both. A
//! target without one fails `InterpInstance::new` cleanly, and `interp` is
//! a default feature of the CLI and spectest crates, so an ungated test
//! would fail `cargo test --workspace` there.
#![cfg(sf_interp_engine)]

use std::cell::{Cell, RefCell};

use sf_nano_core::module::Module;
use sf_nano_core::{InterpInstance, WasmError};

fn r32(mem: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(mem[p..p + 4].try_into().unwrap())
}

fn w32(mem: &mut [u8], p: usize, v: u32) {
    mem[p..p + 4].copy_from_slice(&v.to_le_bytes());
}

fn w64(mem: &mut [u8], p: usize, v: u64) {
    mem[p..p + 8].copy_from_slice(&v.to_le_bytes());
}

struct HostState {
    fake_nanos: Cell<u64>,
    output: RefCell<Vec<u8>>,
    exit_code: Cell<Option<u32>>,
}

fn run_wasi(path: &str) -> (Result<(), WasmError>, HostState) {
    let wasm = std::fs::read(path).expect("read wasm");
    let module = Module::new("wasi", &wasm).expect("parse");
    let state = HostState {
        fake_nanos: Cell::new(0),
        output: RefCell::new(Vec::new()),
        exit_code: Cell::new(None),
    };
    let result = {
        let mut inst = InterpInstance::new(&module).expect("instantiate");
        inst.set_host(
            |_mod: &str, name: &str, mem: &mut [u8], args: &[u64], results: &mut [u64]| {
                match name {
                    "clock_time_get" => {
                        // Advance 11 fake seconds per read so self-timing
                        // benchmark loops converge immediately.
                        let t = state.fake_nanos.get() + 11_000_000_000;
                        state.fake_nanos.set(t);
                        w64(mem, args[2] as usize, t);
                        results[0] = 0;
                    }
                    "fd_write" => {
                        let iovs = args[1] as usize;
                        let n = args[2] as usize;
                        let mut written = 0u32;
                        for k in 0..n {
                            let ptr = r32(mem, iovs + k * 8) as usize;
                            let len = r32(mem, iovs + k * 8 + 4) as usize;
                            state
                                .output
                                .borrow_mut()
                                .extend_from_slice(&mem[ptr..ptr + len]);
                            written += len as u32;
                        }
                        w32(mem, args[3] as usize, written);
                        results[0] = 0;
                    }
                    "args_sizes_get" => {
                        w32(mem, args[0] as usize, 0);
                        w32(mem, args[1] as usize, 0);
                        results[0] = 0;
                    }
                    "args_get" => results[0] = 0,
                    "fd_close" | "fd_seek" | "fd_fdstat_get" => {
                        // benign stubs: zero any out-parameters conservatively
                        if name == "fd_seek" {
                            w64(mem, args[3] as usize, 0);
                        }
                        if name == "fd_fdstat_get" {
                            let p = args[1] as usize;
                            mem[p..p + 24].fill(0);
                        }
                        results[0] = 0;
                    }
                    "proc_exit" => {
                        state.exit_code.set(Some(args[0] as u32));
                        return Err(WasmError::trap("proc_exit"));
                    }
                    other => panic!("unshimmed WASI import: {other}"),
                }
                Ok(())
            },
        );
        let start = inst.find_export("_start").expect("_start export");
        inst.invoke(start, &[], &mut [])
    };
    (result, state)
}

fn completed_ok(result: &Result<(), WasmError>, state: &HostState) -> bool {
    match result {
        Ok(()) => true,
        Err(WasmError::Trap("proc_exit")) => state.exit_code.get() == Some(0),
        _ => false,
    }
}

#[test]
fn interpreter_runs_coremark() {
    let (result, state) = run_wasi(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../benchmarks/wasi/coremark/coremark.wasm"
    ));
    assert!(
        completed_ok(&result, &state),
        "coremark did not complete: {result:?}"
    );
    let out = String::from_utf8_lossy(&state.output.borrow()).into_owned();
    assert!(
        out.contains("CoreMark") || out.contains("Iterations"),
        "unexpected coremark output: {out}"
    );
    assert!(
        out.contains("Correct operation validated"),
        "coremark did not validate: {out}"
    );
}

#[test]
#[ignore = "~5 s under native dispatch but minutes on stage-A-only builds; run explicitly with --ignored"]
fn interpreter_runs_sha256_benchmark() {
    let (result, state) = run_wasi(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../benchmarks/wasi/sha256/sha256.wasm"
    ));
    assert!(
        completed_ok(&result, &state),
        "sha256 did not complete: {result:?}"
    );
    let out = String::from_utf8_lossy(&state.output.borrow()).into_owned();
    assert!(!out.is_empty(), "sha256 produced no output");
}
