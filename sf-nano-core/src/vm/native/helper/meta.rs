//! Typed helper metadata records.
//!
//! This replaces generic `imm0/imm1/imm2` and generic TOS-slot descriptors.

use alloc::vec::Vec;

use crate::vm::lir::slot::{FrameSlot, FrameSpan};
use crate::vm::native::runtime::entry::NativeEntry;

/// Common helper-call result.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HelperResult {
    pub next_entry: NativeEntry,
}

/// Uniform helper-call metadata header.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HelperMetadataHeader {
    pub kind: ColdHelperKind,
    pub next_entry: NativeEntry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColdHelperKind {
    CallExternal,
    CallIndirect,
    MemoryGrow,
    TableGrow,
    TableFill,
    TableCopy,
    TableInit,
    GlobalGet,
    GlobalSet,
    DataDrop,
    ElemDrop,
}

/// Example typed metadata for `call_indirect`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CallIndirectMeta {
    pub header: HelperMetadataHeader,
    pub index_slot: FrameSlot,
    pub args: FrameSpan,
    pub results: FrameSpan,
    pub type_idx: u32,
    pub table_idx: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CallExternalMeta {
    pub header: HelperMetadataHeader,
    pub func_idx: u32,
    pub args: FrameSpan,
    pub results: FrameSpan,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GlobalGetMeta {
    pub header: HelperMetadataHeader,
    pub global_idx: u32,
    pub result_slot: FrameSlot,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GlobalSetMeta {
    pub header: HelperMetadataHeader,
    pub global_idx: u32,
    pub value_slot: FrameSlot,
}

#[derive(Clone, Debug)]
pub enum HelperMetadataRecord {
    CallIndirect(CallIndirectMeta),
    CallExternal(CallExternalMeta),
    GlobalGet(GlobalGetMeta),
    GlobalSet(GlobalSetMeta),
}

impl HelperMetadataRecord {
    #[inline]
    pub fn kind(&self) -> ColdHelperKind {
        match self {
            HelperMetadataRecord::CallIndirect(_) => ColdHelperKind::CallIndirect,
            HelperMetadataRecord::CallExternal(_) => ColdHelperKind::CallExternal,
            HelperMetadataRecord::GlobalGet(_) => ColdHelperKind::GlobalGet,
            HelperMetadataRecord::GlobalSet(_) => ColdHelperKind::GlobalSet,
        }
    }

    #[inline]
    pub fn header(&self) -> &HelperMetadataHeader {
        match self {
            HelperMetadataRecord::CallIndirect(meta) => &meta.header,
            HelperMetadataRecord::CallExternal(meta) => &meta.header,
            HelperMetadataRecord::GlobalGet(meta) => &meta.header,
            HelperMetadataRecord::GlobalSet(meta) => &meta.header,
        }
    }
}

pub type HelperMetadataArena = Vec<HelperMetadataRecord>;
