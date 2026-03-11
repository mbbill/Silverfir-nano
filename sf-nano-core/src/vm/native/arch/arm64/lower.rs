//! Lower target-independent native IR into ARM64 code.
//!
//! This is the ISA boundary. It should translate native IR mechanically into
//! encoded ARM64 bytes and direct-entry patch sites.

use alloc::vec::Vec;

use crate::{error::WasmError, vm::native::ir::NativeProgram};

use super::entry::Arm64EntryPatch;

/// Result of lowering one function to ARM64.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Arm64LoweredFunction {
    pub text: Vec<u8>,
    pub entry_patches: Vec<Arm64EntryPatch>,
}

pub fn lower_arm64(_program: &NativeProgram) -> Result<Arm64LoweredFunction, WasmError> {
    // TODO: Translate native IR into ARM64 bytes and patch sites.
    Ok(Arm64LoweredFunction::default())
}
