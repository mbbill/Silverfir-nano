//! Interpreter context & hot state layout.

use core::ffi::c_char;

use crate::error::WasmError;
use crate::vm::store::Store;
use crate::vm::{entities::ModuleInst, interp::instruction::Instruction};

/// C-visible hot prefix of the fast interpreter context.
///
/// SAFETY: Field order, offsets, and size must match `vm_trampoline.h` `CtxHot`.
#[repr(C)]
pub struct ContextHot {
    pub stack_end: *mut u64,
    pub call_depth: u64,
    pub mem0_base: *mut u8,
    pub mem0_size: u64,
    pub trap_message: *const c_char,
    pub term_inst: *mut Instruction,
}

impl ContextHot {
    #[inline]
    pub fn new(stack_end: *mut u64, mem0_base: *mut u8, mem0_size: u64) -> Self {
        Self {
            stack_end,
            call_depth: 0,
            mem0_base,
            mem0_size,
            trap_message: core::ptr::null(),
            term_inst: core::ptr::null_mut(),
        }
    }
}

/// Byte offsets into the hot context prefix, used by JIT-emitted code.
pub mod ctx_offset {
    use super::ContextHot;

    pub const STACK_END: u32 = core::mem::offset_of!(ContextHot, stack_end) as u32;
    pub const CALL_DEPTH: u32 = core::mem::offset_of!(ContextHot, call_depth) as u32;
    pub const MEM0_BASE: u32 = core::mem::offset_of!(ContextHot, mem0_base) as u32;
    pub const MEM0_SIZE: u32 = core::mem::offset_of!(ContextHot, mem0_size) as u32;
    pub const TRAP_MESSAGE: u32 = core::mem::offset_of!(ContextHot, trap_message) as u32;
    pub const TERM_INST: u32 = core::mem::offset_of!(ContextHot, term_inst) as u32;
}

const _: [(); 0] = [(); core::mem::offset_of!(Context, hot)];
const _: [(); 0] = [(); ctx_offset::STACK_END as usize];
const _: [(); 8] = [(); ctx_offset::CALL_DEPTH as usize];
const _: [(); 16] = [(); ctx_offset::MEM0_BASE as usize];
const _: [(); 24] = [(); ctx_offset::MEM0_SIZE as usize];
const _: [(); 32] = [(); ctx_offset::TRAP_MESSAGE as usize];
const _: [(); 40] = [(); ctx_offset::TERM_INST as usize];
const _: [(); 48] = [(); core::mem::size_of::<ContextHot>()];

/// Opaque context passed across the C trampoline boundary.
///
/// SAFETY: The first field must remain `ContextHot` so the C trampoline and JIT
/// can treat a `*mut Context` as a pointer to the hot prefix.
#[repr(C)]
pub struct Context {
    pub hot: ContextHot,
    pub store: *mut Store,
    pub current_module: *const ModuleInst,
    pub error: Option<WasmError>,
    #[cfg(feature = "function-trace")]
    pub trace_stack: std::vec::Vec<u32>,
}

impl Context {
    #[inline]
    pub fn new(
        store: *mut Store,
        current_module: *const ModuleInst,
        stack_end: *mut u64,
        mem0_base: *mut u8,
        mem0_size: u64,
    ) -> Self {
        Self {
            hot: ContextHot::new(stack_end, mem0_base, mem0_size),
            store,
            current_module,
            error: None,
            #[cfg(feature = "function-trace")]
            trace_stack: std::vec::Vec::new(),
        }
    }

    #[inline]
    pub fn store(&self) -> &Store {
        unsafe { &*self.store }
    }

    #[inline]
    pub fn store_mut(&self) -> &mut Store {
        unsafe { &mut *self.store }
    }

    #[inline]
    pub fn current_module(&self) -> Option<&ModuleInst> {
        if self.current_module.is_null() {
            None
        } else {
            Some(unsafe { &*self.current_module })
        }
    }
}
