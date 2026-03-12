//! Shared lowering and placement state.
//!
//! This file is the intended home for native-only state that should not leak
//! into either LIR or ISA-specific lowering.

use alloc::vec::Vec;

use crate::vm::{
    lir::ir::LirValue,
    native::abi::{NativePlace, NativeSpillSlot, NativeValue},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PlacedValue {
    Place(NativePlace),
    Imm64(u64),
}

impl PlacedValue {
    #[inline]
    pub(super) const fn as_native_value(self) -> NativeValue {
        match self {
            Self::Place(place) => NativeValue::Place(place),
            Self::Imm64(value) => NativeValue::Imm64(value),
        }
    }

    #[inline]
    pub(super) const fn as_place(self) -> Option<NativePlace> {
        match self {
            Self::Place(place) => Some(place),
            Self::Imm64(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct PlacementState {
    homes: Vec<Option<PlacedValue>>,
    next_spill: u16,
}

impl PlacementState {
    pub(super) fn new(value_count: usize) -> Self {
        Self {
            homes: alloc::vec![None; value_count],
            next_spill: 0,
        }
    }

    #[inline]
    pub(super) fn assign(&mut self, value: LirValue, place: NativePlace) {
        self.homes[value.0 as usize] = Some(PlacedValue::Place(place));
    }

    #[inline]
    pub(super) fn assign_value(&mut self, value: LirValue, placed: PlacedValue) {
        self.homes[value.0 as usize] = Some(placed);
    }

    #[inline]
    pub(super) fn home(&self, value: LirValue) -> Option<NativePlace> {
        self.homes
            .get(value.0 as usize)
            .copied()
            .flatten()
            .and_then(PlacedValue::as_place)
    }

    #[inline]
    pub(super) fn placed(&self, value: LirValue) -> Option<PlacedValue> {
        self.homes.get(value.0 as usize).copied().flatten()
    }

    #[inline]
    pub(super) fn snapshot_spill(&mut self) -> NativePlace {
        let place = NativePlace::Spill(NativeSpillSlot(self.next_spill));
        self.next_spill = self.next_spill.saturating_add(1);
        place
    }

    #[inline]
    pub(super) const fn spill_slots(&self) -> u16 {
        self.next_spill
    }
}
