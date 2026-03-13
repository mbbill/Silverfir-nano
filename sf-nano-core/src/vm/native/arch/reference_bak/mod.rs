//! Reference consumer for target-independent native IR.
//!
//! This module is the validation backend for `NativeProgram`.
//! It should consume the same input IR shape as a real ISA backend such as
//! `arch/arm64`, so bugs can be split cleanly into:
//! - native lowering / placement bugs
//! - ISA emission bugs

#[cfg(debug_assertions)]
mod compile;
#[cfg(debug_assertions)]
mod entry;
pub(crate) use crate::vm::native::machine;

#[cfg(debug_assertions)]
pub use compile::compile_program;
