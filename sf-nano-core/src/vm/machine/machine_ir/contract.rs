//! Shared runtime contract types carried alongside machine IR.
//!
//! These are produced by lowering and consumed by backends and the runtime.
//! They describe frame layout agreements and helper symbol bindings, not
//! the MachineIR instruction vocabulary itself.

use super::types::{MachineExternId, MachineFuncId};

/// Shared local-call call-link layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct MachineCallLinkLayout {
    /// Byte offset of the saved continuation pointer within call scratch.
    pub continuation_offset: i32,
    /// Byte offset of the saved caller frame pointer within call scratch.
    pub caller_frame_offset: i32,
    /// Byte offset of the saved caller result-base slot within call scratch.
    pub caller_result_base_offset: i32,
    /// Number of call-scratch slots reserved by the call-link prefix.
    pub slot_count: u16,
}

/// One frame-relative region in the machine runtime contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct MachineFrameRegion {
    pub base_slot: u16,
    pub slots: u16,
}

/// One per-function runtime record derived once from the shared frame plan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct MachineFunctionRuntime {
    pub id: MachineFuncId,
    pub frame_prefix_slots: u16,
    pub total_frame_slots: u16,
    pub call_scratch: Option<MachineFrameRegion>,
    pub helper_scratch: Option<MachineFrameRegion>,
    pub return_results: Option<MachineFrameRegion>,
}

/// Shared runtime contract for machine IR execution.
///
/// This contains only execution-boundary details that remain after VM/runtime
/// semantics have been lowered above machine IR.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MachineRuntimeContract {
    pub call_link: MachineCallLinkLayout,
    pub functions: alloc::vec::Vec<MachineFunctionRuntime>,
}

/// One closed helper symbol that may remain as a true runtime boundary.
///
/// The backend resolves this symbol to the real helper wrapper address for the
/// active ISA/runtime implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum MachineHelperSymbol {
    CallExternal,
    CallIndirectExternal,
}

/// One opaque external binding referenced from machine IR.
///
/// The machine layer uses only the id. Sidecar runtime data owns the meaning
/// of each external target and how it resolves during finalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MachineExternBinding {
    pub id: MachineExternId,
    pub symbol: MachineHelperSymbol,
}
