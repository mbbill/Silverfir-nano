//! CFG + SSA backend-facing LIR.
//!
//! This is the shared semantic handoff between planning/grouping and the
//! backend family. It must not carry rotating-window metadata, stack height,
//! or backend-side stack reconstruction hints.

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::plan::hot_local::HotLocalPlan;

use super::{
    leaf::LirLeafOp,
    slot::{FrameSlot, FrameSpan},
    target::LirTarget,
};

/// One SSA value inside LIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LirValue(pub u32);

/// LIR-visible subset of the backend VM register budget.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LirAbi {
    pub tos_register_count: u8,
    pub hot_local_count: u8,
}

/// Full LIR program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LirProgram {
    pub entry: LirTarget,
    pub blocks: Vec<LirBlock>,
    pub abi: LirAbi,
    /// Function-static mapping for named hot-local slots.
    pub hot_locals: Option<HotLocalPlan>,
}

impl LirProgram {
    #[cfg(any(debug_assertions, test))]
    pub fn validate(&self) -> Result<(), WasmError> {
        if self.blocks.is_empty() {
            if self.entry.as_usize() != 0 {
                return Err(WasmError::internal(
                    "empty LIR program must use entry block 0".into(),
                ));
            }
            return Ok(());
        }

        if self.entry.as_usize() >= self.blocks.len() {
            return Err(WasmError::internal(alloc::format!(
                "LIR entry block {} is out of range for {} blocks",
                self.entry.as_usize(),
                self.blocks.len(),
            )));
        }

        if let Some(hot_locals) = &self.hot_locals {
            if hot_locals.slot_count() != self.abi.hot_local_count as usize {
                return Err(WasmError::internal(alloc::format!(
                    "LIR hot-local mapping width {} does not match ABI hot-local count {}",
                    hot_locals.slot_count(),
                    self.abi.hot_local_count,
                )));
            }
        }

        for (index, block) in self.blocks.iter().enumerate() {
            if block.id.as_usize() != index {
                return Err(WasmError::internal(alloc::format!(
                    "LIR block {} has mismatched id {}",
                    index,
                    block.id.as_usize(),
                )));
            }
            if block.params.tos.len() > self.abi.tos_register_count as usize {
                return Err(WasmError::internal(alloc::format!(
                    "LIR block {} expects {} TOS params, exceeding ABI width {}",
                    index,
                    block.params.tos.len(),
                    self.abi.tos_register_count,
                )));
            }

            for op in &block.ops {
                match &op.kind {
                    LirInstKind::ReadHotLocal { reg, .. }
                    | LirInstKind::WriteHotLocal { reg, .. } => {
                        if *reg >= self.abi.hot_local_count {
                            return Err(WasmError::internal(alloc::format!(
                                "LIR hot-local reg {} exceeds ABI hot-local count {}",
                                reg,
                                self.abi.hot_local_count,
                            )));
                        }
                    }
                    _ => {}
                }
            }

            match &block.terminator {
                LirTerminator::Goto(edge) => self.validate_edge(edge, index)?,
                LirTerminator::Branch {
                    then_edge,
                    else_edge,
                    ..
                } => {
                    self.validate_edge(then_edge, index)?;
                    self.validate_edge(else_edge, index)?;
                }
                LirTerminator::BrTable { entries, .. } => {
                    for edge in entries {
                        self.validate_edge(edge, index)?;
                    }
                }
                LirTerminator::Return { .. } | LirTerminator::TrapUnreachable => {}
            }
        }

        Ok(())
    }

    #[cfg(not(any(debug_assertions, test)))]
    #[inline]
    pub fn validate(&self) -> Result<(), WasmError> {
        Ok(())
    }

    #[cfg(any(debug_assertions, test))]
    fn validate_edge(&self, edge: &LirEdge, source_block: usize) -> Result<(), WasmError> {
        let Some(target) = self.blocks.get(edge.target.as_usize()) else {
            return Err(WasmError::internal(alloc::format!(
                "LIR block {} has edge to out-of-range target {}",
                source_block,
                edge.target.as_usize(),
            )));
        };
        if edge.tos.len() != target.params.tos.len() {
            return Err(WasmError::internal(alloc::format!(
                "LIR edge b{} -> b{} has {} TOS args, but target expects {}",
                source_block,
                edge.target.as_usize(),
                edge.tos.len(),
                target.params.tos.len(),
            )));
        }
        if edge.tos.len() > self.abi.tos_register_count as usize {
            return Err(WasmError::internal(alloc::format!(
                "LIR edge b{} -> b{} exceeds ABI TOS width {}",
                source_block,
                edge.target.as_usize(),
                self.abi.tos_register_count,
            )));
        }
        Ok(())
    }
}

/// One LIR basic block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LirBlock {
    pub id: LirTarget,
    pub params: LirBlockParams,
    pub ops: Vec<LirInst>,
    pub terminator: LirTerminator,
}

/// Explicit incoming block state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LirBlockParams {
    /// Incoming TOS-lane values. Position is the lane index.
    ///
    /// Hot locals have function-static identity and therefore do not appear
    /// here as edge-threaded state.
    pub tos: Vec<LirValue>,
}

/// One SSA operation inside a block body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LirInst {
    pub kind: LirInstKind,
}

/// One control-flow edge with explicit outgoing state arguments.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LirEdge {
    pub target: LirTarget,
    /// Outgoing TOS-lane values for the successor. Position is the lane index.
    ///
    /// Hot locals have function-static identity and therefore do not appear
    /// here as edge-threaded state.
    pub tos: Vec<LirValue>,
}

/// Target-facing operation vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LirInstKind {
    Leaf {
        op: LirLeafOp,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
    WriteOperandSlot {
        slot: FrameSlot,
        src: LirValue,
    },
    ReadOperandSlot {
        slot: FrameSlot,
        dst: LirValue,
    },
    ReadHotLocal {
        /// Hot-local identity comes from `LirProgram::hot_locals`.
        reg: u8,
        dst: LirValue,
    },
    WriteHotLocal {
        /// Hot-local identity comes from `LirProgram::hot_locals`.
        reg: u8,
        src: LirValue,
    },
    ReadFrameLocal {
        frame_slot: FrameSlot,
        dst: LirValue,
    },
    WriteFrameLocal {
        frame_slot: FrameSlot,
        src: LirValue,
    },
    CallExternal {
        func_idx: u32,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
    CallInternal {
        callee: u32,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
    CallIndirect {
        type_idx: u32,
        table_idx: u32,
        index: LirValue,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
}

/// Explicit CFG terminator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LirTerminator {
    Goto(LirEdge),
    Branch {
        cond: LirValue,
        then_edge: LirEdge,
        else_edge: LirEdge,
    },
    BrTable {
        index: LirValue,
        entries: Vec<LirEdge>,
    },
    Return {
        values: Vec<LirValue>,
    },
    TrapUnreachable,
}

impl LirInstKind {
    pub fn reads_frame(&self) -> Vec<FrameSpan> {
        match self {
            LirInstKind::Leaf { .. } => Vec::new(),
            LirInstKind::WriteOperandSlot { .. } => Vec::new(),
            LirInstKind::ReadOperandSlot { slot, .. } => alloc::vec![FrameSpan::single(*slot)],
            LirInstKind::ReadHotLocal { .. } | LirInstKind::WriteHotLocal { .. } => Vec::new(),
            LirInstKind::ReadFrameLocal { frame_slot, .. } => {
                alloc::vec![FrameSpan::single(*frame_slot)]
            }
            LirInstKind::WriteFrameLocal { .. } => Vec::new(),
            LirInstKind::CallExternal { .. }
            | LirInstKind::CallInternal { .. }
            | LirInstKind::CallIndirect { .. } => Vec::new(),
        }
    }

    pub fn writes_frame(&self) -> Vec<FrameSpan> {
        match self {
            LirInstKind::Leaf { .. } => Vec::new(),
            LirInstKind::WriteOperandSlot { slot, .. } => alloc::vec![FrameSpan::single(*slot)],
            LirInstKind::ReadOperandSlot { .. } => Vec::new(),
            LirInstKind::ReadHotLocal { .. } | LirInstKind::WriteHotLocal { .. } => Vec::new(),
            LirInstKind::ReadFrameLocal { .. } => Vec::new(),
            LirInstKind::WriteFrameLocal { frame_slot, .. } => {
                alloc::vec![FrameSpan::single(*frame_slot)]
            }
            LirInstKind::CallExternal { .. }
            | LirInstKind::CallInternal { .. }
            | LirInstKind::CallIndirect { .. } => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_edge_param_arity_mismatch() {
        let program = LirProgram {
            entry: LirTarget(0),
            abi: LirAbi {
                tos_register_count: 1,
                hot_local_count: 0,
            },
            hot_locals: None,
            blocks: alloc::vec![
                LirBlock {
                    id: LirTarget(0),
                    params: LirBlockParams::default(),
                    ops: Vec::new(),
                    terminator: LirTerminator::Goto(LirEdge {
                        target: LirTarget(1),
                        tos: alloc::vec![LirValue(0)],
                    }),
                },
                LirBlock {
                    id: LirTarget(1),
                    params: LirBlockParams::default(),
                    ops: Vec::new(),
                    terminator: LirTerminator::Return { values: Vec::new() },
                },
            ],
        };

        let error = program.validate().expect_err("LIR validation should fail");
        assert!(error
            .message()
            .contains("has 1 TOS args, but target expects 0"));
    }

    #[test]
    fn rejects_hot_local_reg_outside_abi() {
        let program = LirProgram {
            entry: LirTarget(0),
            abi: LirAbi {
                tos_register_count: 0,
                hot_local_count: 0,
            },
            hot_locals: None,
            blocks: alloc::vec![LirBlock {
                id: LirTarget(0),
                params: LirBlockParams::default(),
                ops: alloc::vec![LirInst {
                    kind: LirInstKind::ReadHotLocal {
                        reg: 0,
                        dst: LirValue(0),
                    },
                }],
                terminator: LirTerminator::Return { values: Vec::new() },
            }],
        };

        let error = program.validate().expect_err("LIR validation should fail");
        assert!(error.message().contains("exceeds ABI hot-local count"));
    }
}
