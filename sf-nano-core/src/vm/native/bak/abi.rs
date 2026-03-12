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

use crate::vm::{backend::BackendConfig, lir::slot::FrameSlot};

/// Native-visible backend budget.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeAbi {
    pub ctx_register_count: u8,
    pub fp_register_count: u8,
    pub tmp_register_count: u8,
    pub hot_local_count: u8,
    pub tos_register_count: u8,
}

impl From<BackendConfig> for NativeAbi {
    #[inline]
    fn from(value: BackendConfig) -> Self {
        Self {
            ctx_register_count: value.ctx_register_count,
            fp_register_count: value.fp_register_count,
            tmp_register_count: value.tmp_register_count,
            hot_local_count: value.hot_local_count,
            tos_register_count: value.tos_register_count,
        }
    }
}

/// One fixed VM register role in the target-independent native ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NativeLocation {
    Ctx,
    Fp,
    Hot(u8),
    Tos(u8),
    Tmp(u8),
}

/// One backend-owned spill slot beyond the planned Wasm frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NativeSpillSlot(pub u16);

/// One writable placed storage location in the machine IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NativePlace {
    Location(NativeLocation),
    Frame(FrameSlot),
    Spill(NativeSpillSlot),
}

/// One readable machine operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NativeValue {
    Place(NativePlace),
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
