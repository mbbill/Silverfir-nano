//! Target-independent native VM ABI vocabulary.
//!
//! This file defines the fixed machine roles that the final placed native IR
//! will target before any ISA-specific lowering begins.
//!
//! Important:
//! - these roles are backend ABI concepts, not Wasm semantics
//! - the reference machine and real ISA backends should consume the same
//!   placed contracts
//! - TOS lanes remain valid here only as concrete VM locations, not as a
//!   stack-machine abstraction

use crate::vm::lir::slot::FrameSlot;

/// One fixed VM register role in the target-independent native ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NativeLocation {
    Ctx,
    Fp,
    Hot(u8),
    Tos(u8),
    Tmp(u8),
}

/// One placed value source/destination used by the final machine-shaped IR.
///
/// The live native IR has not been migrated to this representation yet, but
/// all new lowering work should move in this direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NativeStorage {
    Location(NativeLocation),
    Frame(FrameSlot),
    Imm64(u64),
}

/// Block-entry ABI contract after placement.
///
/// This is intentionally small: entry TOS width plus the assumption that hot
/// locals live in their function-static ABI locations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockAbi {
    pub tos_width: u8,
}
