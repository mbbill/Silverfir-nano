//! Runtime-published dispatch metadata shared by native lowering, backends,
//! and the emulator.

pub(crate) mod function_kind {
    pub(crate) const LOCAL: u32 = 0;
    pub(crate) const EXTERNAL: u32 = 1;
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
    pub(crate) call_scratch_base_slot: u64,
}

/// Per-function metadata table entry used by 32-bit native backends for local
/// call setup after indirect dispatch resolves the callee.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct NativeLocalCallInfo32 {
    pub(crate) entry: u32,
    pub(crate) total_frame_bytes: u32,
    pub(crate) frame_prefix_slots: u32,
    pub(crate) call_scratch_base_slot: u32,
}
