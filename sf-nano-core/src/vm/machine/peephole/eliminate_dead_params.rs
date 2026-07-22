//! Remove block parameters whose incoming values are only carried around CFG
//! cycles and never consumed.

use crate::collections;
use crate::vm::machine::machine_ir::{
    MachineBlock, MachineBlockId, MachineEdge, MachineReg, MachineTerminator, MachineValue,
};

use super::helpers::{inst_defines, inst_uses_value, terminator_uses_reg};

pub(super) fn eliminate_dead_params(blocks: &mut [MachineBlock]) {
    let mut useful: collections::Vec<collections::Vec<bool>> = blocks
        .iter()
        .map(|block| collections::vec![false; block.params.len()])
        .collect();

    for (block_index, block) in blocks.iter().enumerate() {
        for (param_index, param) in block.params.iter().enumerate() {
            useful[block_index][param_index] = incoming_value_is_consumed(block, param.reg);
        }
    }

    loop {
        let mut changed = false;
        for (block_index, block) in blocks.iter().enumerate() {
            for (param_index, param) in block.params.iter().enumerate() {
                if useful[block_index][param_index]
                    || block
                        .ops
                        .iter()
                        .any(|inst| inst_defines(&inst.kind, param.reg))
                {
                    continue;
                }
                let mut needed_by_edge = false;
                visit_edges(&block.terminator, |edge| {
                    let Some(target_index) = block_index_for_id(blocks, edge.target) else {
                        return;
                    };
                    for (arg_index, arg) in edge.args.iter().enumerate() {
                        if matches!(arg, MachineValue::Reg(reg) if *reg == param.reg)
                            && useful
                                .get(target_index)
                                .and_then(|params| params.get(arg_index))
                                .copied()
                                .unwrap_or(true)
                        {
                            needed_by_edge = true;
                            break;
                        }
                    }
                });
                if needed_by_edge {
                    useful[block_index][param_index] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    let remove: collections::Vec<collections::Vec<bool>> = useful
        .iter()
        .map(|params| params.iter().map(|is_useful| !is_useful).collect())
        .collect();
    let block_ids: collections::Vec<MachineBlockId> = blocks.iter().map(|block| block.id).collect();
    for block in blocks.iter_mut() {
        visit_edges_mut(&mut block.terminator, |edge| {
            let Some(target_index) = block_index_for_ids(&block_ids, edge.target) else {
                return;
            };
            let mut index = 0usize;
            edge.args.retain(|_| {
                let keep = !remove[target_index].get(index).copied().unwrap_or(false);
                index += 1;
                keep
            });
        });
    }
    for (block_index, block) in blocks.iter_mut().enumerate() {
        let mut index = 0usize;
        block.params.retain(|_| {
            let keep = !remove[block_index][index];
            index += 1;
            keep
        });
    }
}

fn incoming_value_is_consumed(block: &MachineBlock, reg: MachineReg) -> bool {
    for inst in &block.ops {
        if inst_uses_value(&inst.kind, reg) {
            return true;
        }
        if inst_defines(&inst.kind, reg) {
            return false;
        }
    }
    let mut terminator = block.terminator.clone();
    visit_edges_mut(&mut terminator, |edge| edge.args.clear());
    terminator_uses_reg(&terminator, reg)
}

fn block_index_for_id(blocks: &[MachineBlock], id: MachineBlockId) -> Option<usize> {
    let index = id.as_usize();
    if blocks.get(index).is_some_and(|block| block.id == id) {
        Some(index)
    } else {
        blocks.iter().position(|block| block.id == id)
    }
}

fn block_index_for_ids(ids: &[MachineBlockId], id: MachineBlockId) -> Option<usize> {
    let index = id.as_usize();
    if ids.get(index).copied() == Some(id) {
        Some(index)
    } else {
        ids.iter().position(|candidate| *candidate == id)
    }
}

fn visit_edges(terminator: &MachineTerminator, mut visit: impl FnMut(&MachineEdge)) {
    match terminator {
        MachineTerminator::Jump(edge) => visit(edge),
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            visit(then_edge);
            visit(else_edge);
        }
        MachineTerminator::JumpTable { entries, .. } => {
            for edge in entries {
                visit(edge);
            }
        }
        MachineTerminator::Call { success, .. } => visit(success),
        MachineTerminator::TailCall { .. }
        | MachineTerminator::Return
        | MachineTerminator::ReturnScalar { .. }
        | MachineTerminator::Trap { .. } => {}
    }
}

fn visit_edges_mut(terminator: &mut MachineTerminator, mut visit: impl FnMut(&mut MachineEdge)) {
    match terminator {
        MachineTerminator::Jump(edge) => visit(edge),
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            visit(then_edge);
            visit(else_edge);
        }
        MachineTerminator::JumpTable { entries, .. } => {
            for edge in entries {
                visit(edge);
            }
        }
        MachineTerminator::Call { success, .. } => visit(success),
        MachineTerminator::TailCall { .. }
        | MachineTerminator::Return
        | MachineTerminator::ReturnScalar { .. }
        | MachineTerminator::Trap { .. } => {}
    }
}
