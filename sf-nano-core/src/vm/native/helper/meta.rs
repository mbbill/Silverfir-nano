use alloc::vec::Vec;
use core::mem::{align_of, size_of};

use crate::vm::plan::frame::FrameSpan;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HelperFrameRegion {
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
pub struct CallExternalMeta {
    pub func_idx: u32,
    pub args: HelperFrameRegion,
    pub results: HelperFrameRegion,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MemoryGrowMeta {
    pub mem_idx: u32,
    pub io: HelperFrameRegion,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TableGrowMeta {
    pub table_idx: u32,
    pub args: HelperFrameRegion,
    pub results: HelperFrameRegion,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MemoryInitMeta {
    pub data_idx: u32,
    pub mem_idx: u32,
    pub args: HelperFrameRegion,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DataDropMeta {
    pub data_idx: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TableInitMeta {
    pub elem_idx: u32,
    pub table_idx: u32,
    pub args: HelperFrameRegion,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ElemDropMeta {
    pub elem_idx: u32,
}

#[inline]
pub fn encode_record<T: Copy>(record: &T) -> (u32, Vec<u8>) {
    let bytes = unsafe {
        core::slice::from_raw_parts((record as *const T).cast::<u8>(), size_of::<T>())
    };
    (align_of::<T>() as u32, bytes.to_vec())
}
