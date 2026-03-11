//! Reference consumer for target-independent native IR.
//!
//! This module is the validation backend for `NativeProgram`.
//! It should consume the same input IR shape as a real ISA backend such as
//! `arch/arm64`, so bugs can be split cleanly into:
//! - native lowering / placement bugs
//! - ISA emission bugs

mod entry;
mod layout;
mod machine;

pub use entry::shared_native_entry;
pub use layout::frame_slots_used;
pub use machine::{execute_program, ReferenceMachine};
