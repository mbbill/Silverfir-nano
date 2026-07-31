//! End-to-end: the interpreter runs real WASI benchmark binaries from
//! `benchmarks/wasi/` through a minimal host shim. This is the completion
//! check for the import/host boundary: argument marshalling, memory access
//! from host functions, and program output all flow through the
//! interpreter's own execution.
//!
//! Gated on the target having a generated dispatch engine — the same
//! condition `enable_native_dispatch` uses, so one grep finds both. A
//! target without one fails interpreter instantiation cleanly, and `interp` is
//! a default feature of the CLI and spectest crates, so an ungated test
//! would fail `cargo test --workspace` there.
#![cfg(sf_interp)]

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use sf_nano_core::{Caller, Config, Engine, Import, Instance, Tier, Value, WasmError};

const WASI_MODULE: &str = "wasi_snapshot_preview1";

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

fn arg_u32(args: &[Value], index: usize) -> u32 {
    match args[index] {
        Value::I32(value) => value as u32,
        _ => panic!("WASI argument {index} is not i32"),
    }
}

fn set_success(results: &mut [Value]) {
    results[0] = Value::I32(0);
}

fn clock_time_get(
    state: &HostState,
    caller: &mut Caller<'_>,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    // Advance 11 fake seconds per read so self-timing benchmark loops
    // converge immediately.
    let t = state.fake_nanos.get() + 11_000_000_000;
    state.fake_nanos.set(t);
    let mem = caller.memory_mut().expect("clock_time_get needs memory");
    w64(mem, arg_u32(args, 2) as usize, t);
    set_success(results);
    Ok(())
}

fn clock_res_get(
    caller: &mut Caller<'_>,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let mem = caller.memory_mut().expect("clock_res_get needs memory");
    w64(mem, arg_u32(args, 1) as usize, 1);
    set_success(results);
    Ok(())
}

fn fd_write(
    state: &HostState,
    caller: &mut Caller<'_>,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let mem = caller.memory_mut().expect("fd_write needs memory");
    let iovs = arg_u32(args, 1) as usize;
    let n = arg_u32(args, 2) as usize;
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
    w32(mem, arg_u32(args, 3) as usize, written);
    set_success(results);
    Ok(())
}

fn args_sizes_get(
    caller: &mut Caller<'_>,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let mem = caller.memory_mut().expect("args_sizes_get needs memory");
    w32(mem, arg_u32(args, 0) as usize, 0);
    w32(mem, arg_u32(args, 1) as usize, 0);
    set_success(results);
    Ok(())
}

fn args_get(
    _caller: &mut Caller<'_>,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    set_success(results);
    Ok(())
}

fn fd_close(
    _caller: &mut Caller<'_>,
    _args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    set_success(results);
    Ok(())
}

fn fd_seek(
    caller: &mut Caller<'_>,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let mem = caller.memory_mut().expect("fd_seek needs memory");
    w64(mem, arg_u32(args, 3) as usize, 0);
    set_success(results);
    Ok(())
}

fn fd_fdstat_get(
    caller: &mut Caller<'_>,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let mem = caller.memory_mut().expect("fd_fdstat_get needs memory");
    let p = arg_u32(args, 1) as usize;
    mem[p..p + 24].fill(0);
    set_success(results);
    Ok(())
}

fn proc_exit(
    state: &HostState,
    _caller: &mut Caller<'_>,
    args: &[Value],
    _results: &mut [Value],
) -> Result<(), WasmError> {
    state.exit_code.set(Some(arg_u32(args, 0)));
    Err(WasmError::trap("proc_exit"))
}

fn wasi_imports(state: &Rc<HostState>) -> [Import; 9] {
    let clock_state = Rc::clone(state);
    let write_state = Rc::clone(state);
    let exit_state = Rc::clone(state);
    [
        Import::func(
            WASI_MODULE,
            "clock_time_get",
            move |caller, args, results| clock_time_get(&clock_state, caller, args, results),
        ),
        Import::func(WASI_MODULE, "clock_res_get", clock_res_get),
        Import::func(WASI_MODULE, "fd_write", move |caller, args, results| {
            fd_write(&write_state, caller, args, results)
        }),
        Import::func(WASI_MODULE, "args_sizes_get", args_sizes_get),
        Import::func(WASI_MODULE, "args_get", args_get),
        Import::func(WASI_MODULE, "fd_close", fd_close),
        Import::func(WASI_MODULE, "fd_seek", fd_seek),
        Import::func(WASI_MODULE, "fd_fdstat_get", fd_fdstat_get),
        Import::func(WASI_MODULE, "proc_exit", move |caller, args, results| {
            proc_exit(&exit_state, caller, args, results)
        }),
    ]
}

fn run_wasi(path: &str) -> (Result<(), WasmError>, HostState) {
    let wasm = std::fs::read(path).expect("read wasm");
    // Host callbacks are `'static`, so shared state reaches them by refcount
    // rather than by borrow.
    let state = Rc::new(HostState {
        fake_nanos: Cell::new(0),
        output: RefCell::new(Vec::new()),
        exit_code: Cell::new(None),
    });
    let result = {
        let imports = wasi_imports(&state);
        let mut inst = Instance::new(&engine(), &wasm, &imports).expect("instantiate");
        inst.invoke("_start", &[]).map(|_| ())
    };
    (
        result,
        Rc::try_unwrap(state).ok().expect("host state still shared"),
    )
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

/// The interpreter engine under test.
fn engine() -> Engine {
    Engine::new(Config::new().tier(Tier::Interp)).expect("interpreter engine configuration failed")
}
