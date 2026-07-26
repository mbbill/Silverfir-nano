//! Runtime-published dispatch metadata shared by native lowering, backends,
//! and lowering.

use crate::collections;

use core::cell::Cell;
use tracked_alloc::boxed::Box;

use crate::vm::{jit::backend::BackendConfig, jit::machine::machine_ir::MachineModuleAbi};

pub(crate) mod function_kind {
    pub(crate) const LOCAL: u32 = 0;
    pub(crate) const EXTERNAL: u32 = 1;
    pub(crate) const INVALID: u32 = 2;
}

/// Per-function dispatch facts used by indirect dispatch.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct CallDispatchView {
    pub(crate) kind: u32,
    pub(crate) type_canon: u32,
    pub(crate) local_target: u32,
}

/// Per-function metadata table entry used by 64-bit native backends for local
/// call setup after indirect dispatch resolves the callee.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct NativeLocalCallInfo64 {
    pub(crate) entry: u64,
    pub(crate) total_frame_bytes: u64,
    pub(crate) frame_prefix_slots: u64,
}

/// Per-function metadata table entry used by 32-bit native backends for local
/// call setup after indirect dispatch resolves the callee.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct NativeLocalCallInfo32 {
    pub(crate) entry: u32,
    pub(crate) total_frame_bytes: u32,
    pub(crate) frame_prefix_slots: u32,
}

/// One fixed-table dispatch entry.
///
/// The entry is intentionally 16 bytes on both 32-bit and 64-bit targets so
/// lowering can scale the table index with a shift instead of multiplying by a
/// target-dependent record size. On 32-bit targets the low 32 bits of `entry`
/// hold the callee entry word.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NativeFixedCallTableEntry {
    pub(crate) type_canon: u32,
    pub(crate) local_target: u32,
    pub(crate) entry: u64,
}

/// Per-table fast dispatch view for immutable local-only tables.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NativeFixedCallTableView {
    pub(crate) entry_base: *const NativeFixedCallTableEntry,
    pub(crate) len: usize,
}

/// Late-published metadata table used by local `call_indirect` lowering.
///
/// The module ABI already knows every row in this table, but native backends do
/// not know the final address of the executable-embedded copy until code
/// emission finishes. We therefore keep one owned fallback copy here and expose
/// a single published base pointer that can later be redirected to the final
/// executable buffer.
#[derive(Debug)]
pub(crate) struct LocalCallInfoTable {
    owned_records: Box<[u8]>,
    published_base: Cell<*const u8>,
    len: usize,
}

impl LocalCallInfoTable {
    pub(crate) fn new(backend: BackendConfig, abi: &MachineModuleAbi) -> Self {
        let owned_records = build_local_call_info_records(backend, abi);
        let published_base = if owned_records.is_empty() {
            core::ptr::null()
        } else {
            owned_records.as_ptr()
        };
        Self {
            owned_records,
            published_base: Cell::new(published_base),
            len: abi.functions.len(),
        }
    }

    #[inline]
    pub(crate) fn publish(&self, base: *const u8) {
        let published = if self.len == 0 {
            core::ptr::null()
        } else {
            base
        };
        self.published_base.set(published);
    }

    #[inline]
    pub(crate) fn base(&self) -> *const u8 {
        let published = self.published_base.get();
        if published.is_null() && !self.owned_records.is_empty() {
            self.owned_records.as_ptr()
        } else {
            published
        }
    }

    #[inline]
    pub(crate) const fn len(&self) -> usize {
        self.len
    }
}

/// Runtime-published dispatch metadata associated with one compiled native
/// module.
///
/// Today only the local indirect-call table is compile-time metadata; the
/// function views and canonical type table still come from the live store and
/// are refreshed directly into the runtime context.
#[derive(Debug)]
pub(crate) struct NativeDispatchMetadata {
    local_call_infos: LocalCallInfoTable,
}

impl NativeDispatchMetadata {
    pub(crate) fn new(backend: BackendConfig, abi: &MachineModuleAbi) -> Self {
        Self {
            local_call_infos: LocalCallInfoTable::new(backend, abi),
        }
    }

    #[inline]
    pub(crate) const fn local_call_infos(&self) -> &LocalCallInfoTable {
        &self.local_call_infos
    }
}

fn build_local_call_info_records(backend: BackendConfig, abi: &MachineModuleAbi) -> Box<[u8]> {
    match backend.gp_unit_bytes {
        4 => {
            let mut bytes = collections::Vec::with_capacity(
                abi.functions.len() * core::mem::size_of::<NativeLocalCallInfo32>(),
            );
            for function in &abi.functions {
                let info = NativeLocalCallInfo32 {
                    entry: function.id.0,
                    total_frame_bytes: u32::from(function.total_frame_slots) * 8,
                    frame_prefix_slots: u32::from(function.frame_prefix_slots),
                };
                bytes.extend_from_slice(&info.entry.to_le_bytes());
                bytes.extend_from_slice(&info.total_frame_bytes.to_le_bytes());
                bytes.extend_from_slice(&info.frame_prefix_slots.to_le_bytes());
            }
            bytes.into_boxed_slice()
        }
        8 => {
            let mut bytes = collections::Vec::with_capacity(
                abi.functions.len() * core::mem::size_of::<NativeLocalCallInfo64>(),
            );
            for function in &abi.functions {
                let info = NativeLocalCallInfo64 {
                    entry: u64::from(function.id.0),
                    total_frame_bytes: u64::from(function.total_frame_slots) * 8,
                    frame_prefix_slots: u64::from(function.frame_prefix_slots),
                };
                bytes.extend_from_slice(&info.entry.to_le_bytes());
                bytes.extend_from_slice(&info.total_frame_bytes.to_le_bytes());
                bytes.extend_from_slice(&info.frame_prefix_slots.to_le_bytes());
            }
            bytes.into_boxed_slice()
        }
        other => {
            debug_assert!(false, "unsupported GP unit width {other}");
            Box::from([])
        }
    }
}
