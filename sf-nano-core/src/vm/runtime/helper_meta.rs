use alloc::vec::Vec;
use core::mem::{align_of, size_of};

use crate::vm::middle::frame::FrameSpan;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct HelperFrameRegion {
    pub base_slot: u16,
    pub slots: u16,
}

impl From<FrameSpan> for HelperFrameRegion {
    #[inline]
    fn from(value: FrameSpan) -> Self {
        Self {
            base_slot: value.start.0,
            slots: value.count,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct CallExternalMeta {
    pub func_idx: u32,
    pub args: HelperFrameRegion,
    pub results: HelperFrameRegion,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct CallIndirectExternalMeta {
    pub func_idx_slot: u32,
    pub args: HelperFrameRegion,
    pub results: HelperFrameRegion,
}

#[inline]
pub(crate) fn encode_record<T: Copy>(record: &T) -> (u32, Vec<u8>) {
    let bytes =
        unsafe { core::slice::from_raw_parts((record as *const T).cast::<u8>(), size_of::<T>()) };
    (align_of::<T>() as u32, bytes.to_vec())
}
