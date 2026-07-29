use crate::collections;

#[cfg(sf_has_debug_regions)]
use tracked_alloc::string::String;

use crate::vm::jit::machine::machine_ir::{
    MachineBlockId, MachineBlockParam, MachineFloatWidth, MachineFuncId, MachineReg, MachineValue,
};

use super::text_emitter::TextEmitter;

// ── Debug region ─────────────────────────────────────────────────────────────

/// One debug region within a compiled function.
///
/// Populated during code emission and consumed by the optional `ir_dump`
/// (`sf_ir_dump`) and `jitdump` (`sf_jitdump`) debug tools. Only compiled
/// when at least one consumer is enabled (`sf_has_debug_regions`).
#[cfg(sf_has_debug_regions)]
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
    pub params: collections::Vec<MachineBlockParam>,
    pub args: collections::Vec<MachineValue>,
    pub arg_float_widths: collections::Vec<Option<MachineFloatWidth>>,
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
    pub callee: MachineFuncId,
    pub site: DirectCallPatchSite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DirectCallPatchSite {
    /// A raw callee address written into the instruction stream. arm64
    /// patches its `bl`/`b` encodings in place instead.
    #[cfg(not(sf_backend_arm64))]
    AddressLiteral { offset: usize },
    #[cfg(sf_backend_arm64)]
    Arm64Bl {
        inst_offset: usize,
        fallback_veneer_offset: usize,
        fallback_literal_offset: usize,
    },
    #[cfg(sf_backend_arm64)]
    Arm64B {
        inst_offset: usize,
        fallback_veneer_offset: usize,
        fallback_literal_offset: usize,
    },
}

impl DirectCallPatch {
    #[cfg(not(sf_backend_arm64))]
    pub(crate) const fn address_literal(offset: usize, callee: MachineFuncId) -> Self {
        Self {
            callee,
            site: DirectCallPatchSite::AddressLiteral { offset },
        }
    }

    #[cfg(sf_backend_arm64)]
    pub(crate) const fn arm64_bl(
        inst_offset: usize,
        fallback_veneer_offset: usize,
        fallback_literal_offset: usize,
        callee: MachineFuncId,
    ) -> Self {
        Self {
            callee,
            site: DirectCallPatchSite::Arm64Bl {
                inst_offset,
                fallback_veneer_offset,
                fallback_literal_offset,
            },
        }
    }

    #[cfg(sf_backend_arm64)]
    pub(crate) const fn arm64_b(
        inst_offset: usize,
        fallback_veneer_offset: usize,
        fallback_literal_offset: usize,
        callee: MachineFuncId,
    ) -> Self {
        Self {
            callee,
            site: DirectCallPatchSite::Arm64B {
                inst_offset,
                fallback_veneer_offset,
                fallback_literal_offset,
            },
        }
    }
}

// ── Function artifact ────────────────────────────────────────────────────────

/// Output of compiling a single function, before linking.
#[derive(Debug)]
pub(crate) struct FunctionArtifact {
    pub text: TextEmitter,
    pub local_ptr_patches: collections::Vec<LocalPtrPatch>,
    pub direct_call_patches: collections::Vec<DirectCallPatch>,
    /// Offset within `text` of the function's `body_local_error_label`.
    /// Used by the guard-page signal handler to redirect a faulting PC to
    /// the trap propagation tail.
    #[cfg(sf_has_guard_pages)]
    pub body_local_error_offset: usize,
    pub internal_entry_offset: usize,
    #[cfg(sf_has_debug_regions)]
    pub debug_regions: collections::Vec<DebugRegion>,
}

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
