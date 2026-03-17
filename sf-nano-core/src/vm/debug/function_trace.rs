//! Sparse function-boundary trace support.
//!
//! The trace format is intentionally small and backend-neutral:
//! - one file per run
//! - semantic Wasm function boundaries only
//! - stable/canonicalized values to avoid false diffs between correct runs

use alloc::{string::String, vec::Vec};

use crate::vm::value::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceEventKind {
    Entry,
    Exit,
    Trap,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionTraceId {
    pub ordinal: u64,
    pub func_idx: u32,
    pub call_depth: u32,
    pub kind: TraceEventKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionTraceEvent {
    pub id: FunctionTraceId,
    pub result_values: Vec<Value>,
    pub globals_hash: u64,
    pub memory_hash: Option<u64>,
    pub error_class: Option<String>,
}

pub trait FunctionTraceSink {
    fn record(&mut self, event: FunctionTraceEvent);
}

#[cfg(feature = "function-trace")]
mod imp {
    use alloc::format;
    use alloc::string::{String, ToString};
    use core::ffi::c_char;
    use core::fmt::Write as _;
    use core::hash::Hasher;
    use core::sync::atomic::{AtomicU64, Ordering};
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use std::sync::{Mutex, OnceLock};

    use crate::error::WasmError;
    use crate::module::entities::FunctionSpec;
    use crate::value_type::ValueType;
    use crate::vm::entities::{FunctionInst, ModuleInst};
    #[cfg(feature = "interp")]
    use crate::vm::interp::{context as interp_ctx, instruction::Instruction};
    use crate::vm::native::runtime::context::NativeContext;
    use crate::vm::store::Store;

    const TRACE_ENV: &str = "SF_FUNCTION_TRACE";
    const TRACE_MEMORY_ENV: &str = "SF_FUNCTION_TRACE_MEMORY";
    const TRACE_VERSION: &str = "sf-nano-function-trace-v2";
    const CANONICAL_F32_NAN: u32 = 0x7fc0_0000;
    const CANONICAL_F64_NAN: u64 = 0x7ff8_0000_0000_0000;

    #[unsafe(no_mangle)]
    pub static FUNCTION_TRACE_ACTIVE: AtomicU64 = AtomicU64::new(0);

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum EventKind {
        Entry,
        Exit,
        Trap,
    }

    impl EventKind {
        #[inline]
        const fn as_str(self) -> &'static str {
            match self {
                Self::Entry => "entry",
                Self::Exit => "exit",
                Self::Trap => "trap",
            }
        }
    }

    struct TraceRecorder {
        writer: BufWriter<File>,
        seq: u64,
        include_memory: bool,
        backend: Option<&'static str>,
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

        fn ensure_header(&mut self, backend: Option<&'static str>) {
            if self.backend.is_some() {
                return;
            }
            let backend = backend.unwrap_or("unknown");
            self.backend = Some(backend);
            let _ = writeln!(
                self.writer,
                "# {} backend={} memory={}",
                TRACE_VERSION,
                backend,
                if self.include_memory { 1 } else { 0 }
            );
        }

        fn record(
            &mut self,
            backend: Option<&'static str>,
            kind: EventKind,
            func_idx: u32,
            depth: u32,
            results: &str,
            globals_hash: u64,
            memory_hash: Option<u64>,
            error_class: Option<&str>,
        ) {
            self.ensure_header(backend);
            let memory_text = memory_hash
                .map(|hash| format!("{:016x}", hash))
                .unwrap_or_else(|| "-".into());
            let error_text = error_class.unwrap_or("-");
            let _ = writeln!(
                self.writer,
                "T2\t{}\t{}\t{}\t{}\t{}\t{:016x}\t{}\t{}",
                self.seq,
                kind.as_str(),
                depth,
                func_idx,
                results,
                globals_hash,
                memory_text,
                error_text,
            );
            let _ = self.writer.flush();
            self.seq += 1;
        }
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

    #[inline]
    fn canonicalize_raw(raw: u64, ty: ValueType) -> u64 {
        match ty {
            ValueType::I32 => raw as u32 as u64,
            ValueType::I64 => raw,
            ValueType::F32 => {
                let bits = raw as u32;
                if f32::from_bits(bits).is_nan() {
                    CANONICAL_F32_NAN as u64
                } else {
                    bits as u64
                }
            }
            ValueType::F64 => {
                if f64::from_bits(raw).is_nan() {
                    CANONICAL_F64_NAN
                } else {
                    raw
                }
            }
            ValueType::Ref(_) => raw,
            _ => raw,
        }
    }

    fn hash_globals(module: &ModuleInst) -> u64 {
        let mut hasher = Fnv64::default();
        for global in &module.globals {
            hasher.write_u64(canonicalize_raw(global.raw(), global.value_type));
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

    fn format_results(module: &ModuleInst, func_idx: u32, results: &[u64]) -> String {
        let Some(func) = module.functions.get(func_idx as usize) else {
            return "-".into();
        };
        let result_types = func.func_type().results();
        if results.is_empty() {
            return "-".into();
        }
        let mut out = String::new();
        for (index, raw) in results.iter().copied().enumerate() {
            if index != 0 {
                out.push(',');
            }
            let ty = result_types.get(index).copied().unwrap_or(ValueType::I64);
            let _ = write!(&mut out, "{:016x}", canonicalize_raw(raw, ty));
        }
        out
    }

    fn record_event(
        backend: Option<&'static str>,
        kind: EventKind,
        func_idx: u32,
        depth: u32,
        results: &[u64],
        store: &Store,
        error: Option<&WasmError>,
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

        let module = store.module();
        let globals_hash = hash_globals(module);
        let memory_hash = if recorder.include_memory {
            Some(hash_memories(module))
        } else {
            None
        };
        let results = format_results(module, func_idx, results);
        recorder.record(
            backend,
            kind,
            func_idx,
            depth,
            &results,
            globals_hash,
            memory_hash,
            error.map(WasmError::class),
        );
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

    #[cfg(feature = "interp")]
    fn find_func_idx_by_fast_entry(module: &ModuleInst, entry: *mut Instruction) -> Option<u32> {
        module
            .functions
            .iter()
            .enumerate()
            .find_map(|(idx, func)| match func {
                FunctionInst::Local { spec, .. }
                    if spec.has_fast_code() && spec.fast_cache().entry() == entry =>
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
                FunctionInst::Local {
                    spec: candidate, ..
                } if core::ptr::eq(candidate, spec) => Some(idx as u32),
                _ => None,
            })
    }

    #[cfg(feature = "interp")]
    pub fn fast_root_entry(ctx: &mut interp_ctx::Context, spec: &FunctionSpec) {
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
        record_event(
            Some("interpreter"),
            EventKind::Entry,
            func_idx,
            0,
            &[],
            store,
            None,
        );
    }

    #[cfg(feature = "interp")]
    pub fn fast_root_exit(store: &Store, spec: &FunctionSpec, results: &[u64]) {
        if !enabled() {
            return;
        }
        let Some(func_idx) = find_func_idx_by_spec(store.module(), spec) else {
            return;
        };
        record_event(
            Some("interpreter"),
            EventKind::Exit,
            func_idx,
            0,
            results,
            store,
            None,
        );
    }

    #[cfg(feature = "interp")]
    pub fn fast_trap_current(ctx: &mut interp_ctx::Context, error: &WasmError) {
        if !enabled() {
            return;
        }
        let Some(&func_idx) = ctx.trace_stack.last() else {
            return;
        };
        let depth = ctx.trace_stack.len() as u32;
        let store = ctx.store_mut();
        record_event(
            None,
            EventKind::Trap,
            func_idx,
            depth,
            &[],
            store,
            Some(error),
        );
        ctx.trace_stack.clear();
    }

    #[cfg(feature = "interp")]
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn fast_function_trace_enter_entry(
        ctx: *mut interp_ctx::Context,
        entry: *mut Instruction,
    ) {
        if !enabled() || ctx.is_null() {
            return;
        }
        let ctx = unsafe { &mut *ctx };
        let Some(module) = ctx.current_module() else {
            return;
        };
        let Some(func_idx) = find_func_idx_by_fast_entry(module, entry) else {
            return;
        };
        ctx.trace_stack.push(func_idx);
        let depth = ctx.trace_stack.len() as u32;
        let store = ctx.store_mut();
        record_event(None, EventKind::Entry, func_idx, depth, &[], store, None);
    }

    #[cfg(feature = "interp")]
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn fast_function_trace_exit(
        ctx: *mut interp_ctx::Context,
        fp: *mut u64,
        arity: u16,
    ) {
        if !enabled() || ctx.is_null() {
            return;
        }
        let ctx = unsafe { &mut *ctx };
        let Some(func_idx) = ctx.trace_stack.pop() else {
            return;
        };
        let depth = ctx.trace_stack.len() as u32;
        let results = unsafe { core::slice::from_raw_parts(fp, arity as usize) };
        let store = ctx.store_mut();
        record_event(None, EventKind::Exit, func_idx, depth, results, store, None);
    }

    pub fn native_root_entry(ctx: &mut NativeContext, spec: &FunctionSpec, backend: &'static str) {
        if !enabled() {
            return;
        }
        let Some(store) = (unsafe { ctx.store.as_ref() }) else {
            return;
        };
        let Some(func_idx) = find_func_idx_by_spec(store.module(), spec) else {
            return;
        };
        ctx.trace_stack.clear();
        ctx.trace_stack.push(func_idx);
        record_event(
            Some(backend),
            EventKind::Entry,
            func_idx,
            0,
            &[],
            store,
            None,
        );
    }

    pub fn native_root_exit(ctx: &mut NativeContext, spec: &FunctionSpec, results: &[u64]) {
        if !enabled() {
            return;
        }
        let Some(store) = (unsafe { ctx.store.as_ref() }) else {
            return;
        };
        let Some(func_idx) = find_func_idx_by_spec(store.module(), spec) else {
            return;
        };
        record_event(None, EventKind::Exit, func_idx, 0, results, store, None);
        ctx.trace_stack.clear();
    }

    pub fn native_trap_current(ctx: &mut NativeContext, error: &WasmError) {
        if !enabled() {
            return;
        }
        let Some(&func_idx) = ctx.trace_stack.last() else {
            return;
        };
        let Some(store) = (unsafe { ctx.store.as_ref() }) else {
            return;
        };
        record_event(
            None,
            EventKind::Trap,
            func_idx,
            ctx.trace_stack.len() as u32,
            &[],
            store,
            Some(error),
        );
        ctx.trace_stack.clear();
    }

    pub fn native_function_trace_enter_func_idx(ctx: &mut NativeContext, func_idx: u32) {
        if !enabled() {
            return;
        }
        let Some(store) = (unsafe { ctx.store.as_ref() }) else {
            return;
        };
        ctx.trace_stack.push(func_idx);
        record_event(
            None,
            EventKind::Entry,
            func_idx,
            ctx.trace_stack.len() as u32,
            &[],
            store,
            None,
        );
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn native_function_trace_enter_func_idx_entry(
        ctx: *mut NativeContext,
        func_idx: u64,
    ) {
        if !enabled() || ctx.is_null() {
            return;
        }
        native_function_trace_enter_func_idx(unsafe { &mut *ctx }, func_idx as u32);
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn native_function_trace_exit(
        ctx: *mut NativeContext,
        fp: *mut u64,
        arity: u64,
    ) {
        if !enabled() || ctx.is_null() {
            return;
        }
        let ctx = unsafe { &mut *ctx };
        let Some(func_idx) = ctx.trace_stack.pop() else {
            return;
        };
        let Some(store) = (unsafe { ctx.store.as_ref() }) else {
            return;
        };
        let results = unsafe { core::slice::from_raw_parts(fp, arity as usize) };
        record_event(
            None,
            EventKind::Exit,
            func_idx,
            ctx.trace_stack.len() as u32,
            results,
            store,
            None,
        );
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
}

#[cfg(feature = "function-trace")]
pub use imp::*;

#[cfg(not(feature = "function-trace"))]
#[inline]
pub fn init_from_env() {}

#[cfg(not(feature = "function-trace"))]
#[inline]
pub const fn enabled() -> bool {
    false
}
