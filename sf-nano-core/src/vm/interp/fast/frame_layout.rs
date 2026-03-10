//! Fast-interpreter frame layout.
//!
//! This is specific to the handler/trampoline family. Native must not depend on
//! these constants.

/// Metadata slots between frame locals and the operand area.
pub const METADATA_SLOTS: usize = 3;
/// Home slots reserved for the fast backend's local register cache.
pub const LOCAL_CACHE_HOME_SLOTS: usize = 3;

#[inline]
pub const fn frame_prefix_size(logical_frame_size: usize) -> usize {
    if logical_frame_size < LOCAL_CACHE_HOME_SLOTS {
        LOCAL_CACHE_HOME_SLOTS
    } else {
        logical_frame_size
    }
}

#[inline]
pub const fn operand_stack_base(frame_size: usize) -> usize {
    frame_size + METADATA_SLOTS
}

#[inline]
pub const fn operand_base_offset(frame_size: usize) -> usize {
    operand_stack_base(frame_size) * 8
}

#[inline]
pub const fn return_pc_slot(frame_size: usize) -> usize {
    frame_size
}

#[inline]
pub const fn saved_fp_slot(frame_size: usize) -> usize {
    frame_size + 1
}

#[inline]
pub const fn saved_module_slot(frame_size: usize) -> usize {
    frame_size + 2
}
