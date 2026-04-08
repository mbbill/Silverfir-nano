use alloc::string::String;
use alloc::vec::Vec;

use crate::vm::machine::machine_ir::{
    MachineBlockId, MachineBlockParam, MachineFloatWidth, MachineFuncId, MachineReg, MachineValue,
};
use crate::vm::runtime::dispatch_view::NativeLocalCallInfo64;

use super::text_emitter::TextEmitter;

// ── Debug region ─────────────────────────────────────────────────────────────

/// One debug region within a compiled function.
///
/// Populated during code emission and consumed by the optional `ir_dump`
/// debug dumper (`sf_ir_dump`) and the profiler (`sf_has_std`) when enabled.
/// Always present so the data can flow through the arch layer regardless of
/// which consumer is compiled in.
#[derive(Clone, Debug)]
pub(crate) struct DebugRegion {
    /// Byte offset within the function text.
    pub offset: usize,
    /// Byte length of this region.
    pub len: usize,
    /// Human-readable label (e.g. "b0", "edge_3", "prologue", "return_ok").
    pub label: String,
}

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
    /// Offset within `text` of the function's `body_local_error_label`.
    /// Used by the guard-page signal handler to redirect a faulting PC to
    /// the trap propagation tail.
    #[cfg(sf_has_guard_pages)]
    pub body_local_error_offset: usize,
    pub internal_entry_offset: usize,
    pub debug_regions: Vec<DebugRegion>,
}

// ── Function info table entry ────────────────────────────────────────────────

/// Per-function metadata written to the executable code buffer for indirect
/// call dispatch. Layout is identical across arm64 and x86_64.
pub(crate) type NativeFunctionInfo = NativeLocalCallInfo64;

pub(crate) const NATIVE_FUNCTION_INFO_SIZE: usize = core::mem::size_of::<NativeFunctionInfo>();

// ── Parallel move source ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub(crate) enum ParallelSource {
    Reg {
        reg: MachineReg,
        float_width: Option<MachineFloatWidth>,
    },
    /// Edge-level cached-local reservation with no real incoming value move.
    ReservedReg(MachineReg),
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
