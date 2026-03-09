use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::ffi::c_char;
use core::fmt::Write as _;
use core::hash::Hasher;
use core::sync::atomic::{AtomicU64, Ordering};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Mutex, OnceLock};

use crate::error::WasmError;
use crate::module::entities::FunctionSpec;
use crate::vm::entities::ModuleInst;
use crate::vm::interp::fast::context as fast_ctx;
use crate::vm::interp::fast::instruction::Instruction;
use crate::vm::native::context as native_ctx;
use crate::vm::native::instruction::NativeEntry;
use crate::vm::store::Store;

const TRACE_ENV: &str = "SF_FUNCTION_TRACE";
const TRACE_MEMORY_ENV: &str = "SF_FUNCTION_TRACE_MEMORY";
const TRACE_VERSION: &str = "sf-nano-function-trace-v1";

#[unsafe(no_mangle)]
pub static FUNCTION_TRACE_ACTIVE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendTag {
    Fast,
    Native,
}

impl BackendTag {
    fn as_str(self) -> &'static str {
        match self {
            BackendTag::Fast => "fast",
            BackendTag::Native => "native",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventKind {
    Entry,
    Exit,
    Trap,
}

impl EventKind {
    fn as_str(self) -> &'static str {
        match self {
            EventKind::Entry => "entry",
            EventKind::Exit => "exit",
            EventKind::Trap => "trap",
        }
    }
}

struct TraceRecorder {
    writer: BufWriter<File>,
    seq: u64,
    include_memory: bool,
    backend: Option<BackendTag>,
}

impl TraceRecorder {
    fn new(path: &std::ffi::OsStr) -> Option<Self> {
        let file = File::create(path).ok()?;
        Some(Self {
            writer: BufWriter::new(file),
            seq: 0,
            include_memory: std::env::var_os(TRACE_MEMORY_ENV).is_some(),
            backend: None,
        })
    }

    fn record(
        &mut self,
        backend: BackendTag,
        kind: EventKind,
        func_idx: u32,
        depth: u32,
        results: &[u64],
        globals_hash: u64,
        memory_hash: Option<u64>,
        error: Option<&str>,
    ) {
        if self.backend.is_none() {
            self.backend = Some(backend);
            let _ = writeln!(
                self.writer,
                "# {} backend={} memory={}",
                TRACE_VERSION,
                backend.as_str(),
                if self.include_memory { 1 } else { 0 }
            );
        }

        let mut result_buf = String::new();
        if results.is_empty() {
            result_buf.push('-');
        } else {
            for (idx, value) in results.iter().enumerate() {
                if idx != 0 {
                    result_buf.push(',');
                }
                let _ = write!(&mut result_buf, "{:016x}", value);
            }
        }

        let memory_text = memory_hash
            .map(|hash| format!("{:016x}", hash))
            .unwrap_or_else(|| "-".into());
        let error_text = error
            .map(sanitize_error)
            .unwrap_or_else(|| "-".into());

        let _ = writeln!(
            self.writer,
            "T1\t{}\t{}\t{}\t{}\t{}\t{}\t{:016x}\t{}\t{}",
            self.seq,
            backend.as_str(),
            kind.as_str(),
            depth,
            func_idx,
            result_buf,
            globals_hash,
            memory_text,
            error_text,
        );
        let _ = self.writer.flush();
        self.seq += 1;
    }
}

fn sanitize_error(text: &str) -> String {
    text.replace('\t', " ").replace('\n', " ")
}

#[derive(Default)]
struct TraceState {
    recorder: Option<TraceRecorder>,
    path: Option<std::ffi::OsString>,
    include_memory: bool,
}

impl TraceState {
    fn sync_from_env(&mut self) {
        let path = std::env::var_os(TRACE_ENV);
        let include_memory = std::env::var_os(TRACE_MEMORY_ENV).is_some();
        let changed = self.path != path || self.include_memory != include_memory;
        if !changed {
            FUNCTION_TRACE_ACTIVE.store(self.recorder.is_some() as u64, Ordering::Release);
            return;
        }

        self.path = path.clone();
        self.include_memory = include_memory;
        self.recorder = path.as_deref().and_then(TraceRecorder::new);
        FUNCTION_TRACE_ACTIVE.store(self.recorder.is_some() as u64, Ordering::Release);
    }
}

fn state() -> &'static Mutex<TraceState> {
    static STATE: OnceLock<Mutex<TraceState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(TraceState::default()))
}

pub fn init_from_env() {
    if let Ok(mut guard) = state().lock() {
        guard.sync_from_env();
    } else {
        FUNCTION_TRACE_ACTIVE.store(0, Ordering::Release);
    }
}

#[inline]
pub fn enabled() -> bool {
    FUNCTION_TRACE_ACTIVE.load(Ordering::Acquire) != 0
}

fn record_event(
    backend: BackendTag,
    kind: EventKind,
    func_idx: u32,
    depth: u32,
    results: &[u64],
    store: &Store,
    error: Option<&str>,
) {
    if !enabled() {
        return;
    }

    let Ok(mut guard) = state().lock() else {
        return;
    };
    let Some(recorder) = guard.recorder.as_mut() else {
        return;
    };
    let globals_hash = hash_globals(store.module());
    let memory_hash = if recorder.include_memory {
        Some(hash_memories(store.module()))
    } else {
        None
    };
    recorder.record(
        backend,
        kind,
        func_idx,
        depth,
        results,
        globals_hash,
        memory_hash,
        error,
    );
}

fn hash_globals(module: &ModuleInst) -> u64 {
    let mut hasher = Fnv64::default();
    for global in &module.globals {
        hasher.write_u64(global.value.to_raw());
    }
    hasher.finish()
}

fn hash_memories(module: &ModuleInst) -> u64 {
    let mut hasher = Fnv64::default();
    for memory in &module.memories {
        hasher.write(memory.data.as_slice());
    }
    hasher.finish()
}

#[derive(Default)]
struct Fnv64(u64);

impl Hasher for Fnv64 {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut state = if self.0 == 0 {
            0xcbf29ce484222325u64
        } else {
            self.0
        };
        for &byte in bytes {
            state ^= byte as u64;
            state = state.wrapping_mul(0x100000001b3);
        }
        self.0 = state;
    }
}

fn find_func_idx_by_fast_entry(module: &ModuleInst, entry: *mut Instruction) -> Option<u32> {
    module
        .functions
        .iter()
        .enumerate()
        .find_map(|(idx, func)| match func {
            crate::vm::entities::FunctionInst::Local { spec, .. }
                if spec.has_fast_code() && spec.fast_cache().entry() == entry =>
            {
                Some(idx as u32)
            }
            _ => None,
        })
}

fn find_func_idx_by_native_entry(module: &ModuleInst, entry: NativeEntry) -> Option<u32> {
    module
        .functions
        .iter()
        .enumerate()
        .find_map(|(idx, func)| match func {
            crate::vm::entities::FunctionInst::Local { spec, .. }
                if spec.has_native_code()
                    && std::ptr::fn_addr_eq(spec.native_cache().entry(), entry) =>
            {
                Some(idx as u32)
            }
            _ => None,
        })
}

fn find_func_idx_by_spec(module: &ModuleInst, spec: &FunctionSpec) -> Option<u32> {
    module
        .functions
        .iter()
        .enumerate()
        .find_map(|(idx, func)| match func {
            crate::vm::entities::FunctionInst::Local { spec: candidate, .. }
                if core::ptr::eq(candidate, spec) =>
            {
                Some(idx as u32)
            }
            _ => None,
        })
}

pub fn fast_root_entry(ctx: &mut fast_ctx::Context, spec: &FunctionSpec) {
    init_from_env();
    if !enabled() {
        return;
    }
    let Some(module) = ctx.current_module() else {
        return;
    };
    let Some(func_idx) = find_func_idx_by_spec(module, spec) else {
        return;
    };
    ctx.trace_stack.clear();
    ctx.trace_stack.push(func_idx);
    let store = ctx.store_mut();
    record_event(BackendTag::Fast, EventKind::Entry, func_idx, 0, &[], store, None);
}

pub fn record_root_exit(store: &Store, backend: BackendTag, spec: &FunctionSpec, results: &[u64]) {
    if !enabled() {
        return;
    }
    let Some(func_idx) = find_func_idx_by_spec(store.module(), spec) else {
        return;
    };
    record_event(backend, EventKind::Exit, func_idx, 0, results, store, None);
}

#[no_mangle]
pub unsafe extern "C" fn fast_function_trace_enter_entry(
    ctx: *mut fast_ctx::Context,
    entry: *mut Instruction,
) {
    if !enabled() {
        return;
    }
    if ctx.is_null() {
        return;
    }
    let ctx = &mut *ctx;
    let Some(module) = ctx.current_module() else {
        return;
    };
    let Some(func_idx) = find_func_idx_by_fast_entry(module, entry) else {
        return;
    };
    ctx.trace_stack.push(func_idx);
    let depth = ctx.hot.call_depth as u32;
    let store = ctx.store_mut();
    record_event(BackendTag::Fast, EventKind::Entry, func_idx, depth, &[], store, None);
}

#[no_mangle]
pub unsafe extern "C" fn fast_function_trace_exit(
    ctx: *mut fast_ctx::Context,
    fp: *mut u64,
    arity: u16,
) {
    if !enabled() {
        return;
    }
    if ctx.is_null() {
        return;
    }
    let ctx = &mut *ctx;
    let Some(func_idx) = ctx.trace_stack.pop() else {
        return;
    };
    let depth = ctx.hot.call_depth as u32;
    let results = core::slice::from_raw_parts(fp, arity as usize);
    let store = ctx.store_mut();
    record_event(BackendTag::Fast, EventKind::Exit, func_idx, depth, results, store, None);
}

pub fn fast_trap_current(ctx: &mut fast_ctx::Context, error: &WasmError) {
    if !enabled() {
        return;
    }
    let Some(&func_idx) = ctx.trace_stack.last() else {
        return;
    };
    let depth = ctx.hot.call_depth as u32;
    let store = ctx.store_mut();
    record_event(
        BackendTag::Fast,
        EventKind::Trap,
        func_idx,
        depth,
        &[],
        store,
        Some(&error.message()),
    );
    ctx.trace_stack.clear();
}

pub fn native_root_entry(ctx: &mut native_ctx::Context, spec: &FunctionSpec) {
    init_from_env();
    if !enabled() {
        return;
    }
    if ctx.current_module.is_null() {
        return;
    }
    let module = unsafe { &*ctx.current_module };
    let Some(func_idx) = find_func_idx_by_spec(module, spec) else {
        return;
    };
    ctx.trace_stack.clear();
    ctx.trace_stack.push(func_idx);
    let store = unsafe { &mut *ctx.store };
    record_event(BackendTag::Native, EventKind::Entry, func_idx, 0, &[], store, None);
}

#[no_mangle]
pub unsafe extern "C" fn native_function_trace_enter_entry(
    ctx: *mut native_ctx::Context,
    entry: NativeEntry,
) {
    if !enabled() {
        return;
    }
    if ctx.is_null() {
        return;
    }
    let ctx = &mut *ctx;
    if ctx.current_module.is_null() {
        return;
    }
    let module = &*ctx.current_module;
    let Some(func_idx) = find_func_idx_by_native_entry(module, entry) else {
        return;
    };
    ctx.trace_stack.push(func_idx);
    let depth = ctx.hot.call_depth as u32;
    let store = &mut *ctx.store;
    record_event(BackendTag::Native, EventKind::Entry, func_idx, depth, &[], store, None);
}

#[no_mangle]
pub unsafe extern "C" fn native_function_trace_exit(
    ctx: *mut native_ctx::Context,
    fp: *mut u64,
    arity: u16,
) {
    if !enabled() {
        return;
    }
    if ctx.is_null() {
        return;
    }
    let ctx = &mut *ctx;
    let Some(func_idx) = ctx.trace_stack.pop() else {
        return;
    };
    let depth = ctx.hot.call_depth as u32;
    let results = core::slice::from_raw_parts(fp, arity as usize);
    let store = &mut *ctx.store;
    record_event(BackendTag::Native, EventKind::Exit, func_idx, depth, results, store, None);
}

pub fn native_trap_current(ctx: &mut native_ctx::Context, error: &WasmError) {
    if !enabled() {
        return;
    }
    let Some(&func_idx) = ctx.trace_stack.last() else {
        return;
    };
    let depth = ctx.hot.call_depth as u32;
    let store = unsafe { &mut *ctx.store };
    record_event(
        BackendTag::Native,
        EventKind::Trap,
        func_idx,
        depth,
        &[],
        store,
        Some(&error.message()),
    );
    ctx.trace_stack.clear();
}

pub fn trap_message_from_cstr(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return "trap".into();
    }
    unsafe {
        core::ffi::CStr::from_ptr(ptr)
            .to_str()
            .unwrap_or("trap")
            .to_string()
    }
}
