//! Runtime support for Wasm calls that target external host functions.

pub(crate) mod abi;
mod entry;

pub(crate) use abi::{ExternalCallFrameRegion, ExternalCallMeta, ExternalCallTargetKind};
pub(crate) use entry::call_external_entry_ptr;
