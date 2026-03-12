//! CFG + SSA semantic LIR.
//!
//! This is the shared semantic handoff between planning/grouping and the
//! machine-prep stage. It must not carry cache-budget policy, rotating-window
//! metadata, or backend-side stack reconstruction hints.

use alloc::vec::Vec;

use crate::error::WasmError;

use super::{
    leaf::LirLeafOp,
    runtime::LirRuntimeOp,
    slot::{FrameSlot, FrameSpan},
    target::LirTarget,
};

/// One SSA value inside LIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LirValue(pub u32);

/// Full LIR program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LirProgram {
    pub entry: LirTarget,
    pub blocks: Vec<LirBlock>,
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

        for (index, block) in self.blocks.iter().enumerate() {
            if block.id.as_usize() != index {
                return Err(WasmError::internal(alloc::format!(
                    "LIR block {} has mismatched id {}",
                    index,
                    block.id.as_usize(),
                )));
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
        if edge.args.len() != target.params.values.len() {
            return Err(WasmError::internal(alloc::format!(
                "LIR edge b{} -> b{} has {} args, but target expects {}",
                source_block,
                edge.target.as_usize(),
                edge.args.len(),
                target.params.values.len(),
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
    /// Full incoming operand-stack state for the block, from bottom to top.
    pub values: Vec<LirValue>,
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
    /// Full outgoing operand-stack state for the successor, from bottom to top.
    pub args: Vec<LirValue>,
}

/// Target-facing operation vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LirInstKind {
    Leaf {
        op: LirLeafOp,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
    Runtime {
        op: LirRuntimeOp,
        args: Vec<LirValue>,
        results: Vec<LirValue>,
    },
    ReadSlot {
        slot: FrameSlot,
        dst: LirValue,
    },
    WriteSlot {
        slot: FrameSlot,
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
            LirInstKind::Leaf { .. }
            | LirInstKind::Runtime { .. }
            | LirInstKind::WriteSlot { .. }
            | LirInstKind::CallExternal { .. }
            | LirInstKind::CallInternal { .. }
            | LirInstKind::CallIndirect { .. } => Vec::new(),
            LirInstKind::ReadSlot { slot, .. } => alloc::vec![FrameSpan::single(*slot)],
        }
    }

    pub fn writes_frame(&self) -> Vec<FrameSpan> {
        match self {
            LirInstKind::Leaf { .. }
            | LirInstKind::Runtime { .. }
            | LirInstKind::ReadSlot { .. }
            | LirInstKind::CallExternal { .. }
            | LirInstKind::CallInternal { .. }
            | LirInstKind::CallIndirect { .. } => Vec::new(),
            LirInstKind::WriteSlot { slot, .. } => alloc::vec![FrameSpan::single(*slot)],
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
            blocks: alloc::vec![
                LirBlock {
                    id: LirTarget(0),
                    params: LirBlockParams::default(),
                    ops: Vec::new(),
                    terminator: LirTerminator::Goto(LirEdge {
                        target: LirTarget(1),
                        args: alloc::vec![LirValue(0)],
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
        assert!(error.message().contains("has 1 args, but target expects 0"));
    }
}
