//! Remove block parameters whose incoming values are only carried around CFG
//! cycles and never consumed.

use crate::collections;
use crate::vm::machine::machine_ir::{
    MachineBlock, MachineBlockId, MachineEdge, MachineReg, MachineTerminator, MachineValue,
};

use super::helpers::{terminator_non_edge_uses_reg, visit_source_values};

pub(super) fn eliminate_dead_params(blocks: &mut [MachineBlock]) {
    let mut param_offsets = collections::Vec::with_capacity(blocks.len() + 1);
    param_offsets.push(0);
    for block in blocks.iter() {
        param_offsets.push(param_offsets.last().copied().unwrap_or(0) + block.params.len());
    }
    let param_count = param_offsets.last().copied().unwrap_or(0);

    // Keep one sorted register-to-node table for all blocks. The block offsets
    // split it into independently searchable slices without one allocation per
    // block.
    let mut param_lookup = collections::Vec::with_capacity(param_count);
    for (block_index, block) in blocks.iter().enumerate() {
        let base = param_offsets[block_index];
        param_lookup.extend(
            block
                .params
                .iter()
                .enumerate()
                .map(|(param_index, param)| (param.reg, base + param_index)),
        );
        param_lookup[base..param_offsets[block_index + 1]].sort_unstable_by_key(|(reg, _)| *reg);
    }

    let mut useful = collections::vec![false; param_count];
    let mut defined = collections::vec![false; param_count];

    // Discover direct reads in one pass over each block. Uses are visited
    // before definitions so an instruction that reads and overwrites the same
    // register still consumes the incoming block parameter.
    for (block_index, block) in blocks.iter().enumerate() {
        let lookup = &param_lookup[param_offsets[block_index]..param_offsets[block_index + 1]];
        for inst in &block.ops {
            visit_source_values(&inst.kind, |value| {
                let MachineValue::Reg(reg) = value else {
                    return;
                };
                if let Some(node) = param_node_for_reg(lookup, *reg) {
                    if !defined[node] {
                        useful[node] = true;
                    }
                }
            });
            inst.kind.for_each_defined_reg(|reg| {
                if let Some(node) = param_node_for_reg(lookup, reg) {
                    defined[node] = true;
                }
            });
        }

        // Edge arguments are dependencies, not direct uses. Inspect only the
        // terminator's control/call/return operands rather than cloning and
        // stripping every edge payload.
        for &(reg, node) in lookup {
            if !defined[node] && terminator_non_edge_uses_reg(&block.terminator, reg) {
                useful[node] = true;
            }
        }
    }

    let mut block_lookup: collections::Vec<(MachineBlockId, usize)> = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.id, index))
        .collect();
    block_lookup.sort_unstable_by_key(|(id, _)| *id);

    // A source parameter carried to a target parameter is useful exactly when
    // the target is useful. Store the reverse graph in three flat arrays so
    // propagation is a worklist walk rather than repeated full-CFG scans.
    let mut dependency_head = collections::vec![usize::MAX; param_count];
    let mut dependency_source = collections::Vec::new();
    let mut dependency_next = collections::Vec::new();
    for (block_index, block) in blocks.iter().enumerate() {
        let source_lookup =
            &param_lookup[param_offsets[block_index]..param_offsets[block_index + 1]];
        visit_edges(&block.terminator, |edge| {
            let Some(target_index) = block_index_for_id(&block_lookup, edge.target) else {
                return;
            };
            let target_base = param_offsets[target_index];
            let target_end = param_offsets[target_index + 1];
            for (arg_index, arg) in edge.args.iter().enumerate() {
                let MachineValue::Reg(reg) = arg else {
                    continue;
                };
                let Some(source_node) = param_node_for_reg(source_lookup, *reg) else {
                    continue;
                };
                if useful[source_node] || defined[source_node] {
                    continue;
                }
                let target_node = target_base + arg_index;
                if target_node >= target_end {
                    useful[source_node] = true;
                    continue;
                }
                let dependency = dependency_source.len();
                dependency_source.push(source_node);
                dependency_next.push(dependency_head[target_node]);
                dependency_head[target_node] = dependency;
            }
        });
    }

    let mut worklist: collections::Vec<usize> = useful
        .iter()
        .enumerate()
        .filter_map(|(node, is_useful)| is_useful.then_some(node))
        .collect();
    while let Some(target_node) = worklist.pop() {
        let mut dependency = dependency_head[target_node];
        while dependency != usize::MAX {
            let source_node = dependency_source[dependency];
            if !useful[source_node] {
                useful[source_node] = true;
                worklist.push(source_node);
            }
            dependency = dependency_next[dependency];
        }
    }

    for block in blocks.iter_mut() {
        visit_edges_mut(&mut block.terminator, |edge| {
            let Some(target_index) = block_index_for_id(&block_lookup, edge.target) else {
                return;
            };
            let target_base = param_offsets[target_index];
            let target_end = param_offsets[target_index + 1];
            let mut index = 0usize;
            edge.args.retain(|_| {
                let target_node = target_base + index;
                let keep = target_node >= target_end || useful[target_node];
                index += 1;
                keep
            });
        });
    }
    for (block_index, block) in blocks.iter_mut().enumerate() {
        let base = param_offsets[block_index];
        let mut index = 0usize;
        block.params.retain(|_| {
            let keep = useful[base + index];
            index += 1;
            keep
        });
    }
}

fn param_node_for_reg(lookup: &[(MachineReg, usize)], reg: MachineReg) -> Option<usize> {
    lookup
        .binary_search_by_key(&reg, |(candidate, _)| *candidate)
        .ok()
        .map(|index| lookup[index].1)
}

fn block_index_for_id(lookup: &[(MachineBlockId, usize)], id: MachineBlockId) -> Option<usize> {
    lookup
        .binary_search_by_key(&id, |(candidate, _)| *candidate)
        .ok()
        .map(|index| lookup[index].1)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::machine::machine_ir::{
        MachineBlockParam, MachineBranchCond, MachineInst, MachineInstKind, MachineRegOwner,
        MachineStorageType,
    };

    fn edge(target: u32, args: &[MachineReg]) -> MachineEdge {
        MachineEdge {
            target: MachineBlockId(target),
            args: args.iter().copied().map(MachineValue::Reg).collect(),
        }
    }

    fn block(
        id: u32,
        params: &[MachineReg],
        ops: collections::Vec<MachineInst>,
        terminator: MachineTerminator,
    ) -> MachineBlock {
        MachineBlock {
            id: MachineBlockId(id),
            params: params
                .iter()
                .copied()
                .map(MachineBlockParam::gp_word)
                .collect(),
            ops,
            terminator,
        }
    }

    #[test]
    fn removes_params_only_carried_around_a_cycle() {
        let carried = MachineReg(4);
        let mut blocks = collections::vec![
            block(
                10,
                &[carried],
                collections::Vec::new(),
                MachineTerminator::Jump(edge(30, &[carried])),
            ),
            block(
                30,
                &[carried],
                collections::Vec::new(),
                MachineTerminator::Jump(edge(10, &[carried])),
            ),
        ];

        eliminate_dead_params(&mut blocks);

        assert!(blocks.iter().all(|block| block.params.is_empty()));
        for block in &blocks {
            let MachineTerminator::Jump(edge) = &block.terminator else {
                panic!("expected jump");
            };
            assert!(edge.args.is_empty());
        }
    }

    #[test]
    fn propagates_usefulness_back_through_multiple_edges() {
        let carried = MachineReg(4);
        let mut blocks = collections::vec![
            block(
                10,
                &[carried],
                collections::Vec::new(),
                MachineTerminator::Jump(edge(30, &[carried])),
            ),
            block(
                30,
                &[carried],
                collections::Vec::new(),
                MachineTerminator::Jump(edge(50, &[carried])),
            ),
            block(
                50,
                &[carried],
                collections::vec![MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(5),
                        src: MachineValue::Reg(carried),
                    },
                }],
                MachineTerminator::Return,
            ),
        ];

        eliminate_dead_params(&mut blocks);

        assert!(blocks.iter().all(|block| block.params.len() == 1));
        for block in &blocks[..2] {
            let MachineTerminator::Jump(edge) = &block.terminator else {
                panic!("expected jump");
            };
            assert_eq!(edge.args.as_slice(), &[MachineValue::Reg(carried)]);
        }
    }

    #[test]
    fn ignores_uses_after_the_incoming_value_is_redefined() {
        let carried = MachineReg(4);
        let mut blocks = collections::vec![
            block(
                10,
                &[carried],
                collections::vec![MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: carried,
                        src: MachineValue::Imm64(1),
                    },
                }],
                MachineTerminator::Jump(edge(30, &[carried])),
            ),
            block(
                30,
                &[carried],
                collections::vec![MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(5),
                        src: MachineValue::Reg(carried),
                    },
                }],
                MachineTerminator::Return,
            ),
        ];

        eliminate_dead_params(&mut blocks);

        assert!(blocks[0].params.is_empty());
        assert_eq!(blocks[1].params.len(), 1);
        let MachineTerminator::Jump(edge) = &blocks[0].terminator else {
            panic!("expected jump");
        };
        assert_eq!(edge.args.as_slice(), &[MachineValue::Reg(carried)]);
    }

    #[test]
    fn keeps_param_used_by_branch_condition() {
        let carried = MachineReg(4);
        let mut blocks = collections::vec![block(
            10,
            &[carried],
            collections::Vec::new(),
            MachineTerminator::Branch {
                cond: MachineBranchCond::Value(MachineValue::Reg(carried)),
                then_edge: edge(20, &[]),
                else_edge: edge(30, &[]),
            },
        )];

        eliminate_dead_params(&mut blocks);

        assert_eq!(blocks[0].params.len(), 1);
    }
}
