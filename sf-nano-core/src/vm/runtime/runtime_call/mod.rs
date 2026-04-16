//! Runtime support for Wasm calls that must round-trip through runtime dispatch.

pub(crate) mod abi;
mod entry;

pub(crate) use abi::{RuntimeCallFrameRegion, RuntimeCallMeta, RuntimeCallTargetKind};
pub(crate) use entry::call_runtime_entry_ptr;
