use alloc::vec::Vec;

use crate::vm::machine::machine_ir::{
    MachineBlockId, MachineBlockParam, MachineFloatWidth, MachineFuncId, MachineReg, MachineValue,
};

use super::text_emitter::TextEmitter;

// ── Edge stubs ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub(crate) struct EdgeStub {
    pub label: usize,
    pub target: MachineBlockId,
    pub params: Vec<MachineBlockParam>,
    pub args: Vec<MachineValue>,
    pub arg_float_widths: Vec<Option<MachineFloatWidth>>,
}

// ── Patch types ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LocalPtrPatch {
    pub literal_offset: usize,
    pub target_offset: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PendingLocalPtrPatch {
    pub literal_offset: usize,
    pub target_label: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DirectCallPatch {
    pub literal_offset: usize,
    pub callee: MachineFuncId,
}

// ── Function artifact ────────────────────────────────────────────────────────

/// Output of compiling a single function, before linking.
#[derive(Debug)]
pub(crate) struct FunctionArtifact {
    pub text: TextEmitter,
    pub local_ptr_patches: Vec<LocalPtrPatch>,
    pub direct_call_patches: Vec<DirectCallPatch>,
    pub function_table_patches: Vec<usize>,
    pub root_return_offset: usize,
    #[cfg(has_guard_pages)]
    pub return_error_offset: usize,
    pub internal_entry_offset: usize,
    pub debug_regions: Vec<DebugRegion>,
}

// ── Function info table entry ────────────────────────────────────────────────

/// Per-function metadata written to the executable code buffer for indirect
/// call dispatch. Layout is identical across arm64 and x86_64.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NativeFunctionInfo {
    pub(crate) entry: u64,
    pub(crate) total_frame_bytes: u64,
    pub(crate) frame_prefix_slots: u64,
    pub(crate) call_scratch_base_slot: u64,
}

pub(crate) const NATIVE_FUNCTION_INFO_SIZE: usize = core::mem::size_of::<NativeFunctionInfo>();

/// Re-export the canonical DebugRegion from ir_dump.
pub(crate) use crate::vm::debug::ir_dump::DebugRegion;

// ── Parallel move source ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub(crate) enum ParallelSource {
    Reg {
        reg: MachineReg,
        float_width: Option<MachineFloatWidth>,
    },
    Imm(u64),
    /// GP cycle-break temp. The `u8` is the scratch pool index
    /// allocated by `alloc_gp_scratch` and freed by `free_gp_scratch`.
    GpTemp(u8),
    /// FP cycle-break temp. The `u8` is the scratch pool index.
    FpTemp(u8, MachineFloatWidth),
}

impl From<u64> for ParallelSource {
    fn from(value: u64) -> Self {
        Self::Imm(value)
    }
}
