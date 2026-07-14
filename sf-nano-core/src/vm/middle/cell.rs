//! Cell identity.
//!
//! A cell is the middle-end's multi-use value slot: named, written and read
//! any number of times, with its lifetime managed by the residency planner —
//! in contrast to the anonymous single-use linear SSA values. Wasm locals are
//! one origin of cells (today the only one); cell `i` corresponds 1:1 to wasm
//! local `i`. A cell's frame home is a separate property published in
//! `SsaProgram::cell_homes`, keyed by this identity — `CellId` itself is not
//! a frame address.

/// Identity of a cell. Distinct from `FrameSlot` (frame geometry): plan rows,
/// resident sets, and cache ops are keyed by `CellId`; frame addressing goes
/// through the cell's home slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub(crate) struct CellId(pub u16);
