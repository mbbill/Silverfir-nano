//! Native runtime context.
//!
//! This is the backend-owned runtime state shared by native entries and the
//! native reference machine. It stays at the native/runtime boundary and must not carry
//! interpreter instruction-stream state.

use crate::{error::WasmError, vm::store::Store};

use super::code::NativeCode;

/// Native runtime context shared by native entries and recursive native calls.
#[repr(C)]
#[derive(Debug)]
pub struct NativeContext {
    pub store: *mut Store,
    pub stack_end: *mut u64,
    pub call_depth: u64,
    pub error: Option<WasmError>,
    pub current_code: *const NativeCode,
}

impl NativeContext {
    #[inline]
    pub const fn new(
        store: *mut Store,
        stack_end: *mut u64,
        current_code: *const NativeCode,
    ) -> Self {
        Self {
            store,
            stack_end,
            call_depth: 0,
            error: None,
            current_code,
        }
    }
}
