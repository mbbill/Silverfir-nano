//! Interpreter context & hot state layout.

use core::ffi::c_char;

use crate::vm::entities::ModuleInst;
use crate::vm::interp::fast::instruction::Instruction;
use crate::vm::store::Store;
use crate::error::WasmError;

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
    #[cfg(feature = "lockstep-debug")]
    pub checkpoint_pc: *mut Instruction,
    #[cfg(feature = "lockstep-debug")]
    pub checkpoint_fp: *mut u64,
    #[cfg(feature = "lockstep-debug")]
    pub checkpoint_ordinal: u64,
    #[cfg(feature = "lockstep-debug")]
    pub checkpoint_l0: u64,
    #[cfg(feature = "lockstep-debug")]
    pub checkpoint_l1: u64,
    #[cfg(feature = "lockstep-debug")]
    pub checkpoint_l2: u64,
    #[cfg(feature = "lockstep-debug")]
    pub checkpoint_t0: u64,
    #[cfg(feature = "lockstep-debug")]
    pub checkpoint_t1: u64,
    #[cfg(feature = "lockstep-debug")]
    pub checkpoint_t2: u64,
    #[cfg(feature = "lockstep-debug")]
    pub checkpoint_t3: u64,
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
            #[cfg(feature = "lockstep-debug")]
            checkpoint_pc: core::ptr::null_mut(),
            #[cfg(feature = "lockstep-debug")]
            checkpoint_fp: core::ptr::null_mut(),
            #[cfg(feature = "lockstep-debug")]
            checkpoint_ordinal: 0,
            #[cfg(feature = "lockstep-debug")]
            checkpoint_l0: 0,
            #[cfg(feature = "lockstep-debug")]
            checkpoint_l1: 0,
            #[cfg(feature = "lockstep-debug")]
            checkpoint_l2: 0,
            #[cfg(feature = "lockstep-debug")]
            checkpoint_t0: 0,
            #[cfg(feature = "lockstep-debug")]
            checkpoint_t1: 0,
            #[cfg(feature = "lockstep-debug")]
            checkpoint_t2: 0,
            #[cfg(feature = "lockstep-debug")]
            checkpoint_t3: 0,
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
