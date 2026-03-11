//! Target-independent native peepholes.
//!
//! This pass should stay small and mechanical:
//! - remove redundant moves/copies
//! - normalize parallel-copy shapes
//! - fuse obvious compare+branch tails
//! - clean up after placement
//!
//! It must not become another semantic optimizer layer.

use crate::vm::native::ir::NativeProgram;

#[inline]
pub(super) fn run_native_peepholes(_program: &mut NativeProgram) {}
