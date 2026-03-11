//! ARM64 native entry metadata.
//!
//! This file should stay limited to ARM64 entry/patch representation, not
//! frontend semantics.

use crate::vm::native::ir::NativeBlockId;

/// One unresolved ARM64 block-entry patch site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arm64EntryPatch {
    pub code_offset: u32,
    pub target: NativeBlockId,
}
