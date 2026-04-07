//! Static ABI/layout metadata carried alongside machine IR.
//!
//! These records are produced by machine lowering and consumed by backend
//! codegen, debug dumps, and compiled-module packaging. They are part of the
//! backend-facing MachineIR artifact, but not part of the executable
//! instruction vocabulary itself.

use super::types::MachineFuncId;

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
    /// Helper-scratch frame region: scratch slots reserved for runtime
    /// helpers that take a frame-relative scratch base. Distinct from the
    /// (now-deleted) call-link record, which the new local-call ABI no
    /// longer needs — caller/callee state travels via a backend-private
    /// host-stack call record instead. See `docs/ABI_PLAN.md` §6.
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
    pub functions: alloc::vec::Vec<MachineFunctionAbi>,
}
