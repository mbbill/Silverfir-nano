//! Slot-only SSA skeleton lowering.
//!
//! This stage turns explicit semantic CFG blocks into a slot-only block IR.
//! Locals are normalized to slot-form operations here, but cache/spill policy
//! is still deferred to the later planner and rewrite.

use crate::collections;

use crate::{
    error::WasmError,
    vm::{
        middle::{
            cfg::{CfgBlockId, SemanticCfg},
            frame::FrameLayoutPlan,
        },
        wasm::{
            common::SemanticTarget,
            primitive_op::PrimitiveOpKind,
            semantic_ir::{SemanticOpKind, SemanticProgram},
        },
    },
};

/// Slot-only function skeleton.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SlotSsaProgram {
    pub entry: CfgBlockId,
    pub blocks: collections::Vec<SlotBlock>,
}

/// Slot-only block skeleton.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SlotBlock {
    pub id: CfgBlockId,
    pub range: core::ops::Range<usize>,
    pub body: collections::Vec<SlotInst>,
    pub terminator: SlotTerminator,
}

/// Slot-only body instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SlotInst {
    pub semantic_index: usize,
    pub kind: SlotInstKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SlotInstKind {
    Primitive(PrimitiveOpKind),
    LocalGetSlot {
        local: u16,
    },
    LocalSetSlot {
        local: u16,
    },
    LocalTeeSlot {
        local: u16,
    },
    CallDirect {
        callee: u32,
        params: u16,
        results: u16,
    },
    CallIndirect {
        type_idx: u32,
        table_idx: u32,
        params: u16,
        results: u16,
    },
    CallRef {
        type_idx: u32,
        params: u16,
        results: u16,
    },
}

/// Slot-only block terminator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SlotTerminator {
    pub semantic_index: usize,
    pub kind: SlotTerminatorKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SlotTerminatorKind {
    Goto(CfgBlockId),
    Branch {
        then_target: CfgBlockId,
        else_target: CfgBlockId,
    },
    BrTable {
        targets: collections::Vec<CfgBlockId>,
    },
    Return,
    TrapUnreachable,
}

pub(crate) fn lower_slot_only_ssa(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    _frame: FrameLayoutPlan,
) -> Result<SlotSsaProgram, WasmError> {
    let mut blocks = collections::Vec::with_capacity(cfg.blocks.len());
    for block in &cfg.blocks {
        let Some(last_index) = block.range.end.checked_sub(1) else {
            return Err(WasmError::internal("cfg block cannot be empty"));
        };
        let mut body = collections::Vec::new();
        for semantic_index in block.range.start..last_index {
            if let Some(kind) = lower_body_inst_kind(&semantic.ops[semantic_index].kind) {
                body.push(SlotInst {
                    semantic_index,
                    kind,
                });
            }
        }
        blocks.push(SlotBlock {
            id: block.id,
            range: block.range.clone(),
            body,
            terminator: SlotTerminator {
                semantic_index: last_index,
                kind: lower_terminator_kind(
                    last_index,
                    semantic,
                    cfg,
                    &semantic.ops[last_index].kind,
                )?,
            },
        });
    }

    Ok(SlotSsaProgram {
        entry: cfg.entry,
        blocks,
    })
}

fn lower_body_inst_kind(kind: &SemanticOpKind) -> Option<SlotInstKind> {
    match kind {
        SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable) => None,
        SemanticOpKind::Primitive(kind) => Some(SlotInstKind::Primitive(kind.clone())),
        SemanticOpKind::LocalGet { idx } => Some(SlotInstKind::LocalGetSlot { local: *idx }),
        SemanticOpKind::LocalSet { idx } => Some(SlotInstKind::LocalSetSlot { local: *idx }),
        SemanticOpKind::LocalTee { idx } => Some(SlotInstKind::LocalTeeSlot { local: *idx }),
        SemanticOpKind::CallDirect {
            callee,
            params,
            results,
        } => Some(SlotInstKind::CallDirect {
            callee: *callee,
            params: *params,
            results: *results,
        }),
        SemanticOpKind::CallIndirect {
            type_idx,
            table_idx,
            params,
            results,
        } => Some(SlotInstKind::CallIndirect {
            type_idx: *type_idx,
            table_idx: *table_idx,
            params: *params,
            results: *results,
        }),
        SemanticOpKind::CallRef {
            type_idx,
            params,
            results,
        } => Some(SlotInstKind::CallRef {
            type_idx: *type_idx,
            params: *params,
            results: *results,
        }),
        SemanticOpKind::Block { .. }
        | SemanticOpKind::Loop { .. }
        | SemanticOpKind::If { .. }
        | SemanticOpKind::Else { .. }
        | SemanticOpKind::End
        | SemanticOpKind::Br { .. }
        | SemanticOpKind::BrIf { .. }
        | SemanticOpKind::BrTable { .. }
        | SemanticOpKind::ReturnVoid
        | SemanticOpKind::ReturnOne
        | SemanticOpKind::Return { .. } => None,
    }
}

fn lower_terminator_kind(
    semantic_index: usize,
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    kind: &SemanticOpKind,
) -> Result<SlotTerminatorKind, WasmError> {
    let fallthrough = || {
        let target = semantic_index
            .checked_add(1)
            .filter(|next| *next < semantic.ops.len())
            .ok_or_else(|| WasmError::invalid("missing fallthrough target"))?;
        Ok::<_, WasmError>(cfg.semantic_to_block[target])
    };

    Ok(match kind {
        SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable) => {
            SlotTerminatorKind::TrapUnreachable
        }
        SemanticOpKind::ReturnVoid | SemanticOpKind::ReturnOne | SemanticOpKind::Return { .. } => {
            SlotTerminatorKind::Return
        }
        SemanticOpKind::Br { target, .. } => SlotTerminatorKind::Goto(target_block(cfg, *target)?),
        SemanticOpKind::BrIf { target, .. } => SlotTerminatorKind::Branch {
            then_target: target_block(cfg, *target)?,
            else_target: fallthrough()?,
        },
        SemanticOpKind::BrTable { entries } => SlotTerminatorKind::BrTable {
            targets: entries
                .iter()
                .map(|entry| target_block(cfg, entry.target))
                .collect::<Result<collections::Vec<_>, _>>()?,
        },
        SemanticOpKind::If { else_target, .. } => SlotTerminatorKind::Branch {
            then_target: fallthrough()?,
            else_target: target_block(cfg, *else_target)?,
        },
        SemanticOpKind::Else { end_target } => {
            SlotTerminatorKind::Goto(target_block(cfg, *end_target)?)
        }
        _ => SlotTerminatorKind::Goto(fallthrough()?),
    })
}

fn target_block(cfg: &SemanticCfg, target: SemanticTarget) -> Result<CfgBlockId, WasmError> {
    cfg.semantic_to_block
        .get(target.index().as_usize())
        .copied()
        .ok_or_else(|| WasmError::invalid("semantic target out of range"))
}
