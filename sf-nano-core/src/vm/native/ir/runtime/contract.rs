use crate::vm::native::ir::machine::MachineFuncId;

use super::MachineCallLinkLayout;

/// One frame-relative region in the machine runtime contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MachineFrameRegion {
    pub base_slot: u16,
    pub slots: u16,
}

/// One per-function runtime record derived once from the shared frame plan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MachineFunctionRuntime {
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
pub struct MachineRuntimeContract {
    pub call_link: MachineCallLinkLayout,
    pub functions: alloc::vec::Vec<MachineFunctionRuntime>,
}
