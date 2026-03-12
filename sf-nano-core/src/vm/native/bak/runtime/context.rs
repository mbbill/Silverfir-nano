//! Native runtime context.
//!
//! This is the backend-owned runtime state shared by native entries and the
//! native reference machine. It stays at the native/runtime boundary and must not carry
//! interpreter instruction-stream state.

use crate::{
    error::WasmError,
    vm::{entities::GlobalInst, store::Store},
};
use crate::vm::native::code::NativeCode;

use super::entry::NativeEntry;

/// Native runtime context shared by native entries and recursive native calls.
#[repr(C)]
#[derive(Debug)]
pub struct NativeContext {
    pub store: *mut Store,
    pub memory0_base: *mut u8,
    pub memory0_len: usize,
    pub globals_base: *mut GlobalInst,
    pub globals_len: usize,
    pub stack_end: *mut u64,
    pub call_depth: u64,
    pub error: Option<WasmError>,
    pub current_code: *const NativeCode,
    pub term_entry: Option<NativeEntry>,
    #[cfg(feature = "function-trace")]
    pub trace_stack: std::vec::Vec<u32>,
}

impl NativeContext {
    #[inline]
    pub fn new(
        store: *mut Store,
        stack_end: *mut u64,
        current_code: *const NativeCode,
    ) -> Self {
        let mut ctx = Self {
            store,
            memory0_base: core::ptr::null_mut(),
            memory0_len: 0,
            globals_base: core::ptr::null_mut(),
            globals_len: 0,
            stack_end,
            call_depth: 0,
            error: None,
            current_code,
            term_entry: None,
            #[cfg(feature = "function-trace")]
            trace_stack: std::vec::Vec::new(),
        };
        ctx.refresh_cached_views();
        ctx
    }

    #[inline]
    pub fn refresh_cached_views(&mut self) {
        self.refresh_memory0_view();
        self.refresh_globals_view();
    }

    fn refresh_memory0_view(&mut self) {
        let Some(store) = (unsafe { self.store.as_ref() }) else {
            self.memory0_base = core::ptr::null_mut();
            self.memory0_len = 0;
            return;
        };
        if let Some(memory) = store.module().memories.first() {
            self.memory0_base = memory.data.as_ptr() as *mut u8;
            self.memory0_len = memory.data.len();
        } else {
            self.memory0_base = core::ptr::null_mut();
            self.memory0_len = 0;
        }
    }

    fn refresh_globals_view(&mut self) {
        let Some(store) = (unsafe { self.store.as_ref() }) else {
            self.globals_base = core::ptr::null_mut();
            self.globals_len = 0;
            return;
        };
        self.globals_base = store.module().globals.as_ptr() as *mut GlobalInst;
        self.globals_len = store.module().globals.len();
    }
}
