//! Static ABI/layout metadata carried alongside machine IR.
//!
//! These records are produced by machine lowering and consumed by backend
//! codegen, debug dumps, and compiled-module packaging. They are part of the
//! backend-facing MachineIR artifact, but not part of the executable
//! instruction vocabulary itself.

use crate::collections;

use super::types::{MachineFuncId, MachineReg, MachineStorageType};
use crate::vm::jit::middle::frame::FrameSlot;

/// One frame-relative region in the machine module ABI.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct MachineFrameRegion {
    pub base_slot: u16,
    pub slots: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineParamLoc {
    Frame {
        param_index: u16,
        slot: FrameSlot,
    },
    GpArg {
        param_index: u16,
        lane: u8,
        ty: MachineStorageType,
    },
    GpArgPair {
        param_index: u16,
        lo_lane: u8,
        hi_lane: u8,
    },
    FpArg {
        param_index: u16,
        lane: u8,
        ty: MachineStorageType,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct InternalCallArgBudget {
    pub gp_units: u8,
    pub fp_units: u8,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MachineCallArgs {
    pub frame_params: MachineFrameRegion,
    pub lane_args: collections::Vec<MachineCallLaneArg>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MachineCallLaneArg {
    Gp {
        param_index: u16,
        lane: u8,
        src: MachineArgSrc,
        ty: MachineStorageType,
    },
    GpPair {
        param_index: u16,
        lo_lane: u8,
        hi_lane: u8,
        src: MachineArgSrcPair,
    },
    Fp {
        param_index: u16,
        lane: u8,
        src: MachineArgSrc,
        ty: MachineStorageType,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineArgSrc {
    Reg(MachineReg),
    FrameSlot(FrameSlot),
    /// `byte_offset` must be nonnegative; MachineIR never reaches backward
    /// into a caller frame.
    FrameSlotOffset {
        slot: FrameSlot,
        byte_offset: i8,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MachineArgSrcPair {
    pub lo: MachineArgSrc,
    pub hi: MachineArgSrc,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MachineCallResults {
    None,
    ScalarGp {
        dst: MachineResultDst,
        ty: MachineStorageType,
    },
    ScalarGpPair {
        lo: MachineResultDst,
        hi: MachineResultDst,
    },
    ScalarFp {
        dst: MachineResultDst,
        ty: MachineStorageType,
    },
    FrameFallback {
        callee_results: MachineFrameRegion,
        caller_results: MachineFrameRegion,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineResultDst {
    FrameSlot(FrameSlot),
    Reg(MachineReg),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineResultSrc {
    FrameSlot(FrameSlot),
    /// `byte_offset` must be nonnegative; MachineIR never reaches backward
    /// into a caller frame.
    FrameSlotOffset {
        slot: FrameSlot,
        byte_offset: i8,
    },
    Reg(MachineReg),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MachineReturnValue {
    ScalarGp {
        src: MachineResultSrc,
        ty: MachineStorageType,
    },
    ScalarGpPair {
        lo: MachineResultSrc,
        hi: MachineResultSrc,
    },
    ScalarFp {
        src: MachineResultSrc,
        ty: MachineStorageType,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MachineReturnAbi {
    None,
    ScalarGp { ty: MachineStorageType },
    ScalarGpPair,
    ScalarFp { ty: MachineStorageType },
    FrameFallback { results: MachineFrameRegion },
}

impl Default for MachineReturnAbi {
    fn default() -> Self {
        Self::None
    }
}

/// One per-function ABI record derived from the shared frame plan.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct MachineFunctionAbi {
    pub id: MachineFuncId,
    pub param_locs: collections::Vec<MachineParamLoc>,
    pub frame_prefix_slots: u16,
    pub total_frame_slots: u16,
    /// Helper-scratch frame region: scratch slots reserved for runtime
    /// helpers that take a frame-relative scratch base. Local wasm-to-wasm
    /// calls do not reserve a software call-link record; caller/callee state is
    /// carried by the MachineIR call ABI and the native call/return pair.
    pub helper_scratch: Option<MachineFrameRegion>,
    pub return_results: Option<MachineFrameRegion>,
    pub return_abi: MachineReturnAbi,
    /// Non-param local slot indices that may be read before being written.
    /// These slots must be zero-initialized by the callee at function entry.
    /// Locals not listed here are guaranteed to be written before any read,
    /// so the wasm zero-init contract is satisfied without an explicit store.
    pub init_locals: collections::Vec<u16>,
}

/// Module-wide ABI metadata carried alongside MachineIR.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MachineModuleAbi {
    pub functions: collections::Vec<MachineFunctionAbi>,
}
