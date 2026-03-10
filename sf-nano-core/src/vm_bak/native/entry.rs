//! Native entry ABI helpers.

pub type NativeEntry = unsafe extern "C" fn();

pub const INVALID_RESUME_SLOT: u16 = u16::MAX;
pub const EMPTY_RESUME_SLOTS: u64 = u64::MAX;
