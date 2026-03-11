//! Target-independent native IR.
//!
//! This is the single IR layer after shared CFG + SSA LIR.
//! It owns:
//! - placed VM ABI locations
//! - backend-owned spill slots
//! - explicit edge copies
//! - native-owned call/helper lowering shape
//!
//! It must not reintroduce stack height, TOS rotation, or legacy LIR metadata.

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::{
    lir::{
        leaf::LirLeafOp,
        target::LirTarget,
    },
    plan::{frame::FrameLayoutPlan, group::GroupPlan, hot_local::HotLocalPlan},
};

use super::abi::{BlockAbi, NativeAbi, NativePlace, NativeValue};

/// One native CFG block id.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NativeBlockId(pub u32);

impl NativeBlockId {
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl From<LirTarget> for NativeBlockId {
    #[inline]
    fn from(value: LirTarget) -> Self {
        Self(value.0)
    }
}

/// Full native IR program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeProgram {
    pub entry: NativeBlockId,
    pub blocks: Vec<NativeBlock>,
    pub abi: NativeAbi,
    pub frame: FrameLayoutPlan,
    pub spill_slots: u16,
    pub groups: GroupPlan,
    pub hot_locals: Option<HotLocalPlan>,
}

impl NativeProgram {
    #[cfg(any(debug_assertions, test))]
    pub fn validate(&self) -> Result<(), WasmError> {
        if self.blocks.is_empty() {
            if self.entry.as_usize() != 0 {
                return Err(WasmError::internal(
                    "empty native program must use entry block 0".into(),
                ));
            }
            return Ok(());
        }

        if self.entry.as_usize() >= self.blocks.len() {
            return Err(WasmError::internal(alloc::format!(
                "native entry block {} is out of range for {} blocks",
                self.entry.as_usize(),
                self.blocks.len(),
            )));
        }

        for (index, block) in self.blocks.iter().enumerate() {
            if block.id.as_usize() != index {
                return Err(WasmError::internal(alloc::format!(
                    "native block {} has mismatched id {}",
                    index,
                    block.id.as_usize(),
                )));
            }
            if block.entry_abi.tos_width > self.abi.tos_register_count {
                return Err(WasmError::internal(alloc::format!(
                    "native block {} entry ABI needs {} TOS lanes, exceeding ABI width {}",
                    index,
                    block.entry_abi.tos_width,
                    self.abi.tos_register_count,
                )));
            }

            for inst in &block.ops {
                self.validate_inst(inst)?;
            }
            self.validate_terminator(&block.terminator, index)?;
        }

        Ok(())
    }

    #[cfg(not(any(debug_assertions, test)))]
    #[inline]
    pub fn validate(&self) -> Result<(), WasmError> {
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_inst(&self, inst: &NativeInst) -> Result<(), WasmError> {
        match &inst.kind {
            NativeInstKind::Move(mov) => {
                self.validate_place(mov.dst)?;
                self.validate_value(mov.src)?;
            }
            NativeInstKind::Leaf { args, results, .. }
            | NativeInstKind::CallExternal { args, results, .. }
            | NativeInstKind::CallLocal { args, results, .. } => {
                for arg in args {
                    self.validate_value(*arg)?;
                }
                for result in results {
                    self.validate_place(*result)?;
                }
            }
            NativeInstKind::CallIndirect {
                index,
                args,
                results,
                ..
            } => {
                self.validate_value(*index)?;
                for arg in args {
                    self.validate_value(*arg)?;
                }
                for result in results {
                    self.validate_place(*result)?;
                }
            }
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_terminator(
        &self,
        term: &NativeTerminator,
        source_block: usize,
    ) -> Result<(), WasmError> {
        match term {
            NativeTerminator::Goto(edge) => self.validate_edge(edge, source_block),
            NativeTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                self.validate_value(*cond)?;
                self.validate_edge(then_edge, source_block)?;
                self.validate_edge(else_edge, source_block)
            }
            NativeTerminator::BrTable { index, entries } => {
                self.validate_value(*index)?;
                for edge in entries {
                    self.validate_edge(edge, source_block)?;
                }
                Ok(())
            }
            NativeTerminator::Return { values } => {
                for value in values {
                    self.validate_value(*value)?;
                }
                Ok(())
            }
            NativeTerminator::TrapUnreachable => Ok(()),
        }
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_edge(&self, edge: &NativeEdge, source_block: usize) -> Result<(), WasmError> {
        let Some(target) = self.blocks.get(edge.target.as_usize()) else {
            return Err(WasmError::internal(alloc::format!(
                "native block {} has edge to out-of-range target {}",
                source_block,
                edge.target.as_usize(),
            )));
        };

        for mov in &edge.copies {
            self.validate_value(mov.src)?;
            let NativePlace::Location(super::abi::NativeLocation::Tos(lane)) = mov.dst else {
                return Err(WasmError::internal(alloc::format!(
                    "native edge b{} -> b{} writes non-TOS destination {:?}",
                    source_block,
                    edge.target.as_usize(),
                    mov.dst,
                )));
            };
            if lane >= target.entry_abi.tos_width {
                return Err(WasmError::internal(alloc::format!(
                    "native edge b{} -> b{} writes TOS lane {} but target expects {} lanes",
                    source_block,
                    edge.target.as_usize(),
                    lane,
                    target.entry_abi.tos_width,
                )));
            }
        }
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_value(&self, value: NativeValue) -> Result<(), WasmError> {
        match value {
            NativeValue::Place(place) => self.validate_place(place),
            NativeValue::Imm64(_) => Ok(()),
        }
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_place(&self, place: NativePlace) -> Result<(), WasmError> {
        match place {
            NativePlace::Location(super::abi::NativeLocation::Ctx) => {
                if self.abi.ctx_register_count == 0 {
                    return Err(WasmError::internal(
                        "native IR references Ctx with zero ctx-register budget".into(),
                    ));
                }
            }
            NativePlace::Location(super::abi::NativeLocation::Fp) => {
                if self.abi.fp_register_count == 0 {
                    return Err(WasmError::internal(
                        "native IR references Fp with zero fp-register budget".into(),
                    ));
                }
            }
            NativePlace::Location(super::abi::NativeLocation::Hot(reg)) => {
                if reg >= self.abi.hot_local_count {
                    return Err(WasmError::internal(alloc::format!(
                        "native IR references hot-local lane {} beyond ABI width {}",
                        reg,
                        self.abi.hot_local_count,
                    )));
                }
            }
            NativePlace::Location(super::abi::NativeLocation::Tos(lane)) => {
                if lane >= self.abi.tos_register_count {
                    return Err(WasmError::internal(alloc::format!(
                        "native IR references TOS lane {} beyond ABI width {}",
                        lane,
                        self.abi.tos_register_count,
                    )));
                }
            }
            NativePlace::Location(super::abi::NativeLocation::Tmp(reg)) => {
                if reg >= self.abi.tmp_register_count {
                    return Err(WasmError::internal(alloc::format!(
                        "native IR references tmp lane {} beyond ABI width {}",
                        reg,
                        self.abi.tmp_register_count,
                    )));
                }
            }
            NativePlace::Frame(slot) => {
                if slot.0 >= self.frame.operands.end().0 {
                    return Err(WasmError::internal(alloc::format!(
                        "native IR references frame slot {} beyond planned frame end {}",
                        slot.0,
                        self.frame.operands.end().0,
                    )));
                }
            }
            NativePlace::Spill(slot) => {
                if slot.0 >= self.spill_slots {
                    return Err(WasmError::internal(alloc::format!(
                        "native IR references spill slot {} beyond spill budget {}",
                        slot.0,
                        self.spill_slots,
                    )));
                }
            }
        }
        Ok(())
    }
}

/// One native IR basic block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeBlock {
    pub id: NativeBlockId,
    pub entry_abi: BlockAbi,
    pub ops: Vec<NativeInst>,
    pub terminator: NativeTerminator,
}

/// One native IR instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeInst {
    pub kind: NativeInstKind,
}

/// One edge-local copy needed to satisfy successor live-ins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeMove {
    pub dst: NativePlace,
    pub src: NativeValue,
}

/// One explicit native CFG edge.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeEdge {
    pub target: NativeBlockId,
    pub copies: Vec<NativeMove>,
}

/// Native instruction vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeInstKind {
    Move(NativeMove),
    Leaf {
        op: LirLeafOp,
        args: Vec<NativeValue>,
        results: Vec<NativePlace>,
    },
    CallExternal {
        func_idx: u32,
        args: Vec<NativeValue>,
        results: Vec<NativePlace>,
    },
    CallLocal {
        callee: u32,
        args: Vec<NativeValue>,
        results: Vec<NativePlace>,
    },
    CallIndirect {
        type_idx: u32,
        table_idx: u32,
        index: NativeValue,
        args: Vec<NativeValue>,
        results: Vec<NativePlace>,
    },
}

/// Native IR terminator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NativeTerminator {
    Goto(NativeEdge),
    Branch {
        cond: NativeValue,
        then_edge: NativeEdge,
        else_edge: NativeEdge,
    },
    BrTable {
        index: NativeValue,
        entries: Vec<NativeEdge>,
    },
    Return {
        values: Vec<NativeValue>,
    },
    TrapUnreachable,
}
