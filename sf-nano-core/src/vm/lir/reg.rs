//! Backend-facing register classes.
//!
//! Note:
//! - this does *not* name concrete `T0..T3`
//! - it only distinguishes register classes

use crate::vm::plan::tos::TosRotation;

/// Register class visible to backend-facing IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegisterClass {
    HotLocal,
    TosWindow,
}

/// Compute the physical rotating-window register index for a value at
/// `depth_from_top`.
///
/// This is the only stack-like register mapping backends should need after
/// LIR: rotation + stack effect determines concrete window register use.
#[inline]
pub const fn window_register(rotation: TosRotation, depth_from_top: u8) -> u8 {
    rotation.nth_from_top(depth_from_top)
}
