//! ABI-visible metadata for the runtime-call entry.

use crate::{error::WasmError, vm::runtime::common::internal_error};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub(crate) enum RuntimeCallTargetKind {
    Immediate = 0,
    FrameSlot = 1,
}

impl TryFrom<u32> for RuntimeCallTargetKind {
    type Error = WasmError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Immediate),
            1 => Ok(Self::FrameSlot),
            _ => Err(internal_error(
                "runtime-call entry received unknown func_idx source kind",
            )),
        }
    }
}

/// One frame-relative region passed to the runtime-call entrypoint.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct RuntimeCallFrameRegion {
    pub base_slot: u16,
    pub slots: u16,
}

/// Shared metadata for runtime calls.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct RuntimeCallMeta {
    /// Either a direct function index or a frame slot containing the resolved
    /// function index, selected by `func_idx_source_kind`.
    ///
    /// The stored field stays a raw `u32` because this record is serialized
    /// into the machine const-pool and decoded through a runtime ABI boundary.
    pub func_idx_source: u32,
    pub func_idx_source_kind: u32,
    /// Expected caller-side function type index for `call_ref`.
    /// `u32::MAX` disables dynamic type checking in the runtime entry.
    pub expected_type_idx: u32,
    pub args: RuntimeCallFrameRegion,
    pub results: RuntimeCallFrameRegion,
}

impl RuntimeCallMeta {
    #[inline]
    pub(crate) fn target_kind(self) -> Result<RuntimeCallTargetKind, WasmError> {
        self.func_idx_source_kind.try_into()
    }
}
