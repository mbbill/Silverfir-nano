//! Native runtime context and hot state layout.

use core::ffi::c_char;

use crate::error::WasmError;
use crate::vm::entities::ModuleInst;
use crate::vm::native::instruction::NativeEntry;
use crate::vm::store::Store;

#[repr(C)]
pub struct ContextHot {
    pub stack_end: *mut u64,
    pub call_depth: u64,
    pub mem0_base: *mut u8,
    pub mem0_size: u64,
    pub trap_message: *const c_char,
    pub term_entry: NativeEntry,
    pub resume_fp: *mut u64,
    #[cfg(feature = "lockstep-debug")]
    pub checkpoint_next_inst: *mut crate::vm::native::instruction::NativeInst,
    #[cfg(feature = "lockstep-debug")]
    pub checkpoint_ordinal: u64,
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
            term_entry: crate::vm::native::runtime::term_entry(),
            resume_fp: core::ptr::null_mut(),
            #[cfg(feature = "lockstep-debug")]
            checkpoint_next_inst: core::ptr::null_mut(),
            #[cfg(feature = "lockstep-debug")]
            checkpoint_ordinal: 0,
        }
    }
}

pub mod ctx_offset {
    use super::ContextHot;

    pub const STACK_END: u32 = core::mem::offset_of!(ContextHot, stack_end) as u32;
    pub const CALL_DEPTH: u32 = core::mem::offset_of!(ContextHot, call_depth) as u32;
    pub const MEM0_BASE: u32 = core::mem::offset_of!(ContextHot, mem0_base) as u32;
    pub const MEM0_SIZE: u32 = core::mem::offset_of!(ContextHot, mem0_size) as u32;
    pub const TRAP_MESSAGE: u32 = core::mem::offset_of!(ContextHot, trap_message) as u32;
    pub const TERM_ENTRY: u32 = core::mem::offset_of!(ContextHot, term_entry) as u32;
    pub const RESUME_FP: u32 = core::mem::offset_of!(ContextHot, resume_fp) as u32;
    #[cfg(feature = "lockstep-debug")]
    pub const CHECKPOINT_NEXT_INST: u32 =
        core::mem::offset_of!(ContextHot, checkpoint_next_inst) as u32;
    #[cfg(feature = "lockstep-debug")]
    pub const CHECKPOINT_ORDINAL: u32 =
        core::mem::offset_of!(ContextHot, checkpoint_ordinal) as u32;
    pub const TERM_PC: u32 = TERM_ENTRY;
}

const _: [(); 0] = [(); core::mem::offset_of!(Context, hot)];
const _: [(); 0] = [(); ctx_offset::STACK_END as usize];
const _: [(); 8] = [(); ctx_offset::CALL_DEPTH as usize];
const _: [(); 16] = [(); ctx_offset::MEM0_BASE as usize];
const _: [(); 24] = [(); ctx_offset::MEM0_SIZE as usize];
const _: [(); 32] = [(); ctx_offset::TRAP_MESSAGE as usize];
const _: [(); 40] = [(); ctx_offset::TERM_ENTRY as usize];
const _: [(); 48] = [(); ctx_offset::RESUME_FP as usize];
#[cfg(not(feature = "lockstep-debug"))]
const _: [(); 56] = [(); core::mem::size_of::<ContextHot>()];
#[cfg(feature = "lockstep-debug")]
const _: [(); 56] = [(); ctx_offset::CHECKPOINT_NEXT_INST as usize];
#[cfg(feature = "lockstep-debug")]
const _: [(); 64] = [(); ctx_offset::CHECKPOINT_ORDINAL as usize];
#[cfg(feature = "lockstep-debug")]
const _: [(); 72] = [(); core::mem::size_of::<ContextHot>()];

#[repr(C)]
pub struct Context {
    pub hot: ContextHot,
    pub store: *mut Store,
    pub current_module: *const ModuleInst,
    pub error: Option<WasmError>,
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

#[cfg(feature = "lockstep-debug")]
pub mod ctx_debug_offset {
    use super::Context;

    pub const CHECKPOINT_L0: u32 = core::mem::offset_of!(Context, checkpoint_l0) as u32;
    pub const CHECKPOINT_L1: u32 = core::mem::offset_of!(Context, checkpoint_l1) as u32;
    pub const CHECKPOINT_L2: u32 = core::mem::offset_of!(Context, checkpoint_l2) as u32;
    pub const CHECKPOINT_T0: u32 = core::mem::offset_of!(Context, checkpoint_t0) as u32;
    pub const CHECKPOINT_T1: u32 = core::mem::offset_of!(Context, checkpoint_t1) as u32;
    pub const CHECKPOINT_T2: u32 = core::mem::offset_of!(Context, checkpoint_t2) as u32;
    pub const CHECKPOINT_T3: u32 = core::mem::offset_of!(Context, checkpoint_t3) as u32;
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
}
