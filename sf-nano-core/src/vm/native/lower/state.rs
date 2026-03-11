//! Shared lowering and placement state.
//!
//! This file is the intended home for native-only state that should not leak
//! into either LIR or ISA-specific lowering.

use alloc::vec::Vec;

use crate::vm::native::abi::NativeLocation;

#[derive(Clone, Debug, Default)]
pub(super) struct PlacementState {
    pub incoming_tos: u8,
    pub hot_locals_live: Vec<bool>,
    pub aliases: Vec<NativeLocation>,
}
