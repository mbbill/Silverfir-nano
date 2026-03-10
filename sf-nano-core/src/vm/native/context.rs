//! Native runtime context ABI.

use super::entry::NativeEntry;

/// Native runtime context.
///
/// This should hold only backend-independent runtime state plus native-owned
/// entry/termination state. It must not embed interpreter instruction streams.
#[repr(C)]
#[derive(Debug)]
pub struct NativeContext {
    pub stack_end: *mut u64,
    pub call_depth: u64,
    pub trap_message: *const u8,
    pub term_entry: Option<NativeEntry>,
}
