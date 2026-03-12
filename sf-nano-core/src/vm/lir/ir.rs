//! Prepared backend-facing LIR.
//!
//! This is the frontend/native handoff for the engine's prepared single-pass
//! pipeline:
//! - canonical locals and deep stack values live in frame slots
//! - only a bounded top-of-stack window stays live as SSA values
//! - explicit spill/fill ops publish and reload transient values from operand
//!   slots so the backend never needs general register allocation

use alloc::vec::Vec;

use crate::error::WasmError;

use super::{
    leaf::LirLeafOp,
    runtime::LirRuntimeOp,
    slot::{FrameSlot, FrameSpan},
    target::LirTarget,
};

/// One SSA value inside the live TOS window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LirValue(pub u32);

/// Preferred canonical local-slot ranking selected by planning.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LirLocalCachePlan {
    pub preferred_slots: Vec<FrameSlot>,
}

/// Full prepared LIR program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LirProgram {
    pub entry: LirTarget,
    pub local_cache: LirLocalCachePlan,
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

            self.validate_boundary(&block.params, alloc::format!("block b{index} params"))?;

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

        for (index, slot) in self.local_cache.preferred_slots.iter().enumerate() {
            if self.local_cache.preferred_slots[..index].contains(slot) {
                return Err(WasmError::internal(alloc::format!(
                    "LIR local-cache preferences contain duplicate slot {:?}",
                    slot,
                )));
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
    fn validate_boundary(
        &self,
        boundary: &LirBlockParams,
        label: alloc::string::String,
    ) -> Result<(), WasmError> {
        if boundary.tos.len() > boundary.stack_height as usize {
            return Err(WasmError::internal(alloc::format!(
                "{label} has {} live TOS values for stack height {}",
                boundary.tos.len(),
                boundary.stack_height,
            )));
        }
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

        self.validate_boundary(
            &LirBlockParams {
                stack_height: edge.stack_height,
                tos: edge.tos.clone(),
            },
            alloc::format!("edge b{source_block} -> b{}", edge.target.as_usize()),
        )?;

        if edge.stack_height != target.params.stack_height {
            return Err(WasmError::internal(alloc::format!(
                "LIR edge b{} -> b{} has stack height {}, but target expects {}",
                source_block,
                edge.target.as_usize(),
                edge.stack_height,
                target.params.stack_height,
            )));
        }
        if edge.tos.len() != target.params.tos.len() {
            return Err(WasmError::internal(alloc::format!(
                "LIR edge b{} -> b{} has {} TOS args, but target expects {}",
                source_block,
                edge.target.as_usize(),
                edge.tos.len(),
                target.params.tos.len(),
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
    /// Full logical stack height on entry. Values below the live TOS window are
    /// already published to canonical operand slots.
    pub stack_height: u16,
    /// Live TOS-window values on entry, from bottom to top.
    pub tos: Vec<LirValue>,
}

/// One SSA operation inside a block body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LirInst {
    pub kind: LirInstKind,
}

/// One control-flow edge with explicit outgoing live TOS-window state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LirEdge {
    pub target: LirTarget,
    pub stack_height: u16,
    /// Outgoing live TOS-window values for the successor, from bottom to top.
    pub tos: Vec<LirValue>,
}

/// Prepared frontend operation vocabulary.
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
    /// Read a canonical frame slot, usually a local slot.
    ReadSlot {
        slot: FrameSlot,
        dst: LirValue,
    },
    /// Write a canonical frame slot, usually a local slot.
    WriteSlot {
        slot: FrameSlot,
        src: LirValue,
    },
    /// Publish one transient live value into its canonical operand slot.
    Spill {
        slot: FrameSlot,
        src: LirValue,
    },
    /// Reload one transient live value from its canonical operand slot.
    Fill {
        slot: FrameSlot,
        dst: LirValue,
    },
    /// Call using canonical frame spans for arguments and results.
    CallExternal {
        func_idx: u32,
        args: FrameSpan,
        results: FrameSpan,
    },
    /// Call using canonical frame spans for arguments and results.
    CallInternal {
        callee: u32,
        args: FrameSpan,
        results: FrameSpan,
    },
    /// Indirect call using canonical frame spans plus an explicit spilled index.
    CallIndirect {
        type_idx: u32,
        table_idx: u32,
        index_slot: FrameSlot,
        args: FrameSpan,
        results: FrameSpan,
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
    /// Return using canonical frame result slots prepared before the terminator.
    Return {
        results: Option<FrameSpan>,
    },
    TrapUnreachable,
}

impl LirInstKind {
    pub fn reads_frame(&self) -> Vec<FrameSpan> {
        match self {
            LirInstKind::Leaf { .. }
            | LirInstKind::Runtime { .. }
            | LirInstKind::WriteSlot { .. }
            | LirInstKind::Spill { .. } => Vec::new(),
            LirInstKind::ReadSlot { slot, .. } | LirInstKind::Fill { slot, .. } => {
                alloc::vec![FrameSpan::single(*slot)]
            }
            LirInstKind::CallExternal { args, .. } | LirInstKind::CallInternal { args, .. } => {
                alloc::vec![*args]
            }
            LirInstKind::CallIndirect {
                index_slot, args, ..
            } => alloc::vec![FrameSpan::single(*index_slot), *args],
        }
    }

    pub fn writes_frame(&self) -> Vec<FrameSpan> {
        match self {
            LirInstKind::Leaf { .. }
            | LirInstKind::Runtime { .. }
            | LirInstKind::ReadSlot { .. }
            | LirInstKind::Fill { .. } => Vec::new(),
            LirInstKind::WriteSlot { slot, .. } | LirInstKind::Spill { slot, .. } => {
                alloc::vec![FrameSpan::single(*slot)]
            }
            LirInstKind::CallExternal { results, .. }
            | LirInstKind::CallInternal { results, .. }
            | LirInstKind::CallIndirect { results, .. } => alloc::vec![*results],
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
            local_cache: LirLocalCachePlan::default(),
            blocks: alloc::vec![
                LirBlock {
                    id: LirTarget(0),
                    params: LirBlockParams::default(),
                    ops: Vec::new(),
                    terminator: LirTerminator::Goto(LirEdge {
                        target: LirTarget(1),
                        stack_height: 1,
                        tos: alloc::vec![LirValue(0)],
                    }),
                },
                LirBlock {
                    id: LirTarget(1),
                    params: LirBlockParams::default(),
                    ops: Vec::new(),
                    terminator: LirTerminator::Return { results: None },
                },
            ],
        };

        let error = program.validate().expect_err("LIR validation should fail");
        assert!(error.message().contains("has stack height 1"));
    }

    #[test]
    fn rejects_boundary_with_more_live_tos_than_stack_height() {
        let program = LirProgram {
            entry: LirTarget(0),
            local_cache: LirLocalCachePlan::default(),
            blocks: alloc::vec![LirBlock {
                id: LirTarget(0),
                params: LirBlockParams {
                    stack_height: 2,
                    tos: alloc::vec![LirValue(0), LirValue(1), LirValue(2)],
                },
                ops: Vec::new(),
                terminator: LirTerminator::Return { results: None },
            }],
        };

        let error = program.validate().expect_err("LIR validation should fail");
        assert!(error.message().contains("stack height 2"));
    }
}
