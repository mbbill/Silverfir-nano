//! ABI-visible metadata for the external-call runtime entry.

use crate::{error::WasmError, vm::runtime::common::internal_error};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub(crate) enum ExternalCallTargetKind {
    Immediate = 0,
    FrameSlot = 1,
}

impl TryFrom<u32> for ExternalCallTargetKind {
    type Error = WasmError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Immediate),
            1 => Ok(Self::FrameSlot),
            _ => Err(internal_error(
                "external-call entry received unknown func_idx source kind",
            )),
        }
    }
}

/// One frame-relative region passed to the external-call entrypoint.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct ExternalCallFrameRegion {
    pub base_slot: u16,
    pub slots: u16,
}

/// Shared metadata for external calls.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct ExternalCallMeta {
    /// Either a direct function index or a frame slot containing the resolved
    /// function index, selected by `func_idx_source_kind`.
    ///
    /// The stored field stays a raw `u32` because this record is serialized
    /// into the machine const-pool and decoded through a runtime ABI boundary.
    pub func_idx_source: u32,
    pub func_idx_source_kind: u32,
    pub args: ExternalCallFrameRegion,
    pub results: ExternalCallFrameRegion,
}

impl ExternalCallMeta {
    #[inline]
    pub(crate) fn target_kind(self) -> Result<ExternalCallTargetKind, WasmError> {
        self.func_idx_source_kind.try_into()
    }
}
