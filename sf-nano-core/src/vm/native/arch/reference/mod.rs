//! Reference consumer for target-independent native IR.
//!
//! This module is the validation backend for `NativeProgram`.
//! It should consume the same input IR shape as a real ISA backend such as
//! `arch/arm64`, so bugs can be split cleanly into:
//! - native lowering / placement bugs
//! - ISA emission bugs

mod compile;
mod entry;
mod machine;

pub use compile::compile_program;
pub use machine::{execute_program, ReferenceMachine};
