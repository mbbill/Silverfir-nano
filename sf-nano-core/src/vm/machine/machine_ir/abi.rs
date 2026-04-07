//! Static ABI/layout metadata carried alongside machine IR.
//!
//! These records are produced by machine lowering and consumed by backend
//! codegen, debug dumps, and compiled-module packaging. They are part of the
//! backend-facing MachineIR artifact, but not part of the executable
//! instruction vocabulary itself.

use super::types::MachineFuncId;

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

/// One frame-relative region in the machine module ABI.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct MachineFrameRegion {
    pub base_slot: u16,
    pub slots: u16,
}

/// One per-function ABI record derived from the shared frame plan.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct MachineFunctionAbi {
    pub id: MachineFuncId,
    pub frame_prefix_slots: u16,
    pub total_frame_slots: u16,
    pub call_scratch: Option<MachineFrameRegion>,
    pub helper_scratch: Option<MachineFrameRegion>,
    pub return_results: Option<MachineFrameRegion>,
    /// Non-param local slot indices that may be read before being written.
    /// These slots must be zero-initialized by the callee at function entry.
    /// Locals not listed here are guaranteed to be written before any read,
    /// so the wasm zero-init contract is satisfied without an explicit store.
    pub init_locals: alloc::vec::Vec<u16>,
}

/// Module-wide ABI metadata carried alongside MachineIR.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MachineModuleAbi {
    pub call_link: MachineCallLinkLayout,
    pub functions: alloc::vec::Vec<MachineFunctionAbi>,
}
