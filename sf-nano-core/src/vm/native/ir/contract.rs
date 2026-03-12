//! Shared runtime contract for the machine-facing IR.
//!
//! The final machine IR is only the code contract. This module defines the
//! runtime-side metadata that both real ISA backends and the debug emulator
//! must share.

use alloc::vec::Vec;

use super::machine::{MachineHelperId, MachineReg};

/// One distinguished runtime-provided input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MachinePinnedInputKind {
    RuntimeBase,
    FrameBase,
    StackLimit,
}

/// One pinned input register in the machine ABI contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MachinePinnedInput {
    pub reg: MachineReg,
    pub kind: MachinePinnedInputKind,
}

/// One local-call call-link layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MachineCallLinkContract {
    pub continuation_slot_offset: i32,
    pub caller_frame_slot_offset: i32,
}

/// One fallthrough helper signature.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineHelperSignature {
    pub id: MachineHelperId,
    pub arg_regs: Vec<MachineReg>,
    pub result_regs: Vec<MachineReg>,
    pub clobbers: Vec<MachineReg>,
    pub may_trap: bool,
    pub may_change_runtime_views: bool,
}

/// Runtime layout offsets used for address derivation above the final machine
/// IR. ISA backends should consume only the resulting address math.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MachineRuntimeLayout {
    pub memory0_base_offset: i32,
    pub memory0_len_offset: i32,
    pub globals_base_offset: i32,
    pub globals_len_offset: i32,
}

/// Shared machine runtime contract for one compiled module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineRuntimeContract {
    pub pinned_inputs: Vec<MachinePinnedInput>,
    pub call_link: MachineCallLinkContract,
    pub helper_signatures: Vec<MachineHelperSignature>,
    pub runtime_layout: MachineRuntimeLayout,
}
