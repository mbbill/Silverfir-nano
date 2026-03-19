use alloc::vec;
use alloc::vec::Vec;

use crate::vm::{
    lir::ir::{LirBlock, LirEdge, LirInst, LirInstKind, LirProgram, LirTerminator, LirValue},
    plan::frame::FrameLayoutPlan,
};

pub(super) fn optimize_lir(program: &mut LirProgram, frame: FrameLayoutPlan) {
    for block in &mut program.blocks {
        forward_slot_values(block, frame, &program.value_types);
    }
}

fn forward_slot_values(
    block: &mut LirBlock,
    frame: FrameLayoutPlan,
    value_types: &[crate::value_type::ValueType],
) {
    let alias_len = max_value_index(block)
        .map(|value| value.0 as usize + 1)
        .unwrap_or(0);
    if alias_len == 0 {
        return;
    }

    let last_uses = compute_last_uses(block, alias_len);
    let mut aliases = vec![None; alias_len];
    let mut slot_values = vec![None; frame.total_slots() as usize];
    let original_ops = core::mem::take(&mut block.ops);
    let mut rewritten = Vec::with_capacity(original_ops.len());

    for mut inst in original_ops {
        rewrite_inst_uses(&mut inst.kind, &aliases);

        match &inst.kind {
            LirInstKind::LoadSlot { slot, dst } => {
                if let Some(src) = slot_values.get(slot.0 as usize).copied().flatten() {
                    let resolved_src = resolve_alias(src, &aliases);
                    if can_forward_load(resolved_src, *dst, &last_uses)
                        && can_forward_load_type(resolved_src, *dst, value_types)
                    {
                        aliases[dst.0 as usize] = Some(resolved_src);
                        continue;
                    }
                }
            }
            LirInstKind::StoreSlot { slot, src } => {
                slot_values[slot.0 as usize] = Some(*src);
            }
            LirInstKind::Boundary(_) => {
                slot_values.fill(None);
            }
            LirInstKind::Value { .. } => {}
        }

        rewritten.push(inst);
    }

    rewrite_terminator_uses(&mut block.terminator, &aliases);
    block.ops = rewritten;
}

fn max_value_index(block: &LirBlock) -> Option<LirValue> {
    let mut max_value = block.params.iter().copied().max();

    for inst in &block.ops {
        match &inst.kind {
            LirInstKind::Value { args, results, .. } => {
                max_value = max_value.max(args.iter().copied().max());
                max_value = max_value.max(results.iter().copied().max());
            }
            LirInstKind::LoadSlot { dst, .. } => {
                max_value = max_value.max(Some(*dst));
            }
            LirInstKind::StoreSlot { src, .. } => {
                max_value = max_value.max(Some(*src));
            }
            LirInstKind::Boundary(_) => {}
        }
    }

    match &block.terminator {
        LirTerminator::Goto(edge) => {
            max_value = max_value.max(edge.bindings.iter().map(|binding| binding.value).max());
        }
        LirTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            max_value = max_value.max(Some(*cond));
            max_value = max_value.max(then_edge.bindings.iter().map(|binding| binding.value).max());
            max_value = max_value.max(else_edge.bindings.iter().map(|binding| binding.value).max());
        }
        LirTerminator::BrTable { index, entries } => {
            max_value = max_value.max(Some(*index));
            for edge in entries {
                max_value = max_value.max(edge.bindings.iter().map(|binding| binding.value).max());
            }
        }
        LirTerminator::Return { .. } | LirTerminator::TrapUnreachable => {}
    }

    max_value
}

fn compute_last_uses(block: &LirBlock, len: usize) -> Vec<Option<u32>> {
    let mut last_uses = vec![None; len];

    for (index, inst) in block.ops.iter().enumerate() {
        let pos = index as u32;
        match &inst.kind {
            LirInstKind::Value { args, .. } => {
                for value in args {
                    last_uses[value.0 as usize] = Some(pos);
                }
            }
            LirInstKind::StoreSlot { src, .. } => {
                last_uses[src.0 as usize] = Some(pos);
            }
            LirInstKind::LoadSlot { .. } | LirInstKind::Boundary(_) => {}
        }
    }

    let term_pos = block.ops.len() as u32;
    match &block.terminator {
        LirTerminator::Goto(edge) => mark_edge_uses(edge, term_pos, &mut last_uses),
        LirTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            last_uses[cond.0 as usize] = Some(term_pos);
            mark_edge_uses(then_edge, term_pos, &mut last_uses);
            mark_edge_uses(else_edge, term_pos, &mut last_uses);
        }
        LirTerminator::BrTable { index, entries } => {
            last_uses[index.0 as usize] = Some(term_pos);
            for edge in entries {
                mark_edge_uses(edge, term_pos, &mut last_uses);
            }
        }
        LirTerminator::Return { .. } | LirTerminator::TrapUnreachable => {}
    }

    last_uses
}

fn mark_edge_uses(edge: &LirEdge, pos: u32, last_uses: &mut [Option<u32>]) {
    for binding in &edge.bindings {
        last_uses[binding.value.0 as usize] = Some(pos);
    }
}

fn can_forward_load(src: LirValue, dst: LirValue, last_uses: &[Option<u32>]) -> bool {
    let Some(dst_last_use) = last_uses.get(dst.0 as usize).copied().flatten() else {
        return false;
    };
    let Some(src_last_use) = last_uses.get(src.0 as usize).copied().flatten() else {
        return false;
    };
    dst_last_use <= src_last_use
}

fn can_forward_load_type(
    src: LirValue,
    dst: LirValue,
    value_types: &[crate::value_type::ValueType],
) -> bool {
    match (
        value_types.get(src.0 as usize).copied(),
        value_types.get(dst.0 as usize).copied(),
    ) {
        (Some(src_ty), Some(dst_ty)) => src_ty == dst_ty,
        _ => true,
    }
}

fn rewrite_inst_uses(kind: &mut LirInstKind, aliases: &[Option<LirValue>]) {
    match kind {
        LirInstKind::Value { args, .. } => {
            for value in args {
                *value = resolve_alias(*value, aliases);
            }
        }
        LirInstKind::StoreSlot { src, .. } => {
            *src = resolve_alias(*src, aliases);
        }
        LirInstKind::LoadSlot { .. } | LirInstKind::Boundary(_) => {}
    }
}

fn rewrite_terminator_uses(term: &mut LirTerminator, aliases: &[Option<LirValue>]) {
    match term {
        LirTerminator::Goto(edge) => rewrite_edge_uses(edge, aliases),
        LirTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            *cond = resolve_alias(*cond, aliases);
            rewrite_edge_uses(then_edge, aliases);
            rewrite_edge_uses(else_edge, aliases);
        }
        LirTerminator::BrTable { index, entries } => {
            *index = resolve_alias(*index, aliases);
            for edge in entries {
                rewrite_edge_uses(edge, aliases);
            }
        }
        LirTerminator::Return { .. } | LirTerminator::TrapUnreachable => {}
    }
}

fn rewrite_edge_uses(edge: &mut LirEdge, aliases: &[Option<LirValue>]) {
    for binding in &mut edge.bindings {
        binding.value = resolve_alias(binding.value, aliases);
    }
}

fn resolve_alias(value: LirValue, aliases: &[Option<LirValue>]) -> LirValue {
    let mut resolved = value;
    while let Some(Some(next)) = aliases.get(resolved.0 as usize) {
        if *next == resolved {
            break;
        }
        resolved = *next;
    }
    resolved
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use crate::value_type::ValueType;
    use crate::vm::{
        lir::{
            ir::{LirBlock, LirInstKind, LirProgram, LirTerminator, LirValue},
            leaf::LirLeafOp,
            slot::FrameSlot,
            target::LirTarget,
        },
        plan::{frame::plan_frame_layout, prepare::optimize::optimize_lir},
        wasm::primitive_op::PrimitiveOpKind,
    };

    #[test]
    fn forwards_store_reload_when_source_already_lives_long_enough() {
        let mut program = LirProgram {
            entry: LirTarget(0),
            local_cache: Default::default(),
            blocks: alloc::vec![LirBlock {
                id: LirTarget(0),
                params: Vec::new(),
                ops: alloc::vec![
                    crate::vm::lir::ir::LirInst {
                        kind: LirInstKind::Value {
                            op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 7 })
                                .unwrap(),
                            args: Vec::new(),
                            results: alloc::vec![LirValue(0)],
                        },
                    },
                    crate::vm::lir::ir::LirInst {
                        kind: LirInstKind::StoreSlot {
                            slot: FrameSlot(0),
                            src: LirValue(0),
                        },
                    },
                    crate::vm::lir::ir::LirInst {
                        kind: LirInstKind::LoadSlot {
                            slot: FrameSlot(0),
                            dst: LirValue(1),
                        },
                    },
                    crate::vm::lir::ir::LirInst {
                        kind: LirInstKind::Value {
                            op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Add).unwrap(),
                            args: alloc::vec![LirValue(0), LirValue(1)],
                            results: alloc::vec![LirValue(2)],
                        },
                    },
                ],
                terminator: LirTerminator::Return { results: None },
            }],
            value_types: alloc::vec![],
        };

        optimize_lir(&mut program, plan_frame_layout(1, 2, 0));

        let block = &program.blocks[0];
        assert_eq!(block.ops.len(), 3);
        assert!(matches!(
            &block.ops[2].kind,
            LirInstKind::Value {
                args,
                ..
            } if args == &alloc::vec![LirValue(0), LirValue(0)]
        ));
    }

    #[test]
    fn does_not_forward_duplicate_slot_loads_without_same_block_store() {
        let mut program = LirProgram {
            entry: LirTarget(0),
            local_cache: Default::default(),
            blocks: alloc::vec![LirBlock {
                id: LirTarget(0),
                params: Vec::new(),
                ops: alloc::vec![
                    crate::vm::lir::ir::LirInst {
                        kind: LirInstKind::LoadSlot {
                            slot: FrameSlot(0),
                            dst: LirValue(0),
                        },
                    },
                    crate::vm::lir::ir::LirInst {
                        kind: LirInstKind::LoadSlot {
                            slot: FrameSlot(0),
                            dst: LirValue(1),
                        },
                    },
                    crate::vm::lir::ir::LirInst {
                        kind: LirInstKind::Value {
                            op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Add).unwrap(),
                            args: alloc::vec![LirValue(0), LirValue(1)],
                            results: alloc::vec![LirValue(2)],
                        },
                    },
                ],
                terminator: LirTerminator::Return { results: None },
            }],
            value_types: alloc::vec![],
        };

        optimize_lir(&mut program, plan_frame_layout(1, 2, 0));

        let block = &program.blocks[0];
        assert!(matches!(block.ops[1].kind, LirInstKind::LoadSlot { .. }));
        assert!(matches!(
            &block.ops[2].kind,
            LirInstKind::Value {
                args,
                ..
            } if args == &alloc::vec![LirValue(0), LirValue(1)]
        ));
    }

    #[test]
    fn keeps_reload_when_forwarding_would_extend_the_source_lifetime() {
        let mut program = LirProgram {
            entry: LirTarget(0),
            local_cache: Default::default(),
            blocks: alloc::vec![LirBlock {
                id: LirTarget(0),
                params: Vec::new(),
                ops: alloc::vec![
                    crate::vm::lir::ir::LirInst {
                        kind: LirInstKind::LoadSlot {
                            slot: FrameSlot(0),
                            dst: LirValue(0),
                        },
                    },
                    crate::vm::lir::ir::LirInst {
                        kind: LirInstKind::Value {
                            op: LirLeafOp::from_primitive(PrimitiveOpKind::Drop).unwrap(),
                            args: alloc::vec![LirValue(0)],
                            results: Vec::new(),
                        },
                    },
                    crate::vm::lir::ir::LirInst {
                        kind: LirInstKind::LoadSlot {
                            slot: FrameSlot(0),
                            dst: LirValue(1),
                        },
                    },
                    crate::vm::lir::ir::LirInst {
                        kind: LirInstKind::Value {
                            op: LirLeafOp::from_primitive(PrimitiveOpKind::Drop).unwrap(),
                            args: alloc::vec![LirValue(1)],
                            results: Vec::new(),
                        },
                    },
                ],
                terminator: LirTerminator::Return { results: None },
            }],
            value_types: alloc::vec![],
        };

        optimize_lir(&mut program, plan_frame_layout(1, 2, 0));

        let block = &program.blocks[0];
        assert!(matches!(block.ops[2].kind, LirInstKind::LoadSlot { .. }));
        assert!(matches!(
            &block.ops[3].kind,
            LirInstKind::Value {
                args,
                ..
            } if args == &alloc::vec![LirValue(1)]
        ));
    }

    #[test]
    fn does_not_forward_slot_reload_across_mismatched_value_types() {
        let mut program = LirProgram {
            entry: LirTarget(0),
            local_cache: Default::default(),
            blocks: alloc::vec![LirBlock {
                id: LirTarget(0),
                params: Vec::new(),
                ops: alloc::vec![
                    crate::vm::lir::ir::LirInst {
                        kind: LirInstKind::Value {
                            op: LirLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 1 })
                                .unwrap(),
                            args: Vec::new(),
                            results: alloc::vec![LirValue(0)],
                        },
                    },
                    crate::vm::lir::ir::LirInst {
                        kind: LirInstKind::StoreSlot {
                            slot: FrameSlot(0),
                            src: LirValue(0),
                        },
                    },
                    crate::vm::lir::ir::LirInst {
                        kind: LirInstKind::LoadSlot {
                            slot: FrameSlot(0),
                            dst: LirValue(1),
                        },
                    },
                    crate::vm::lir::ir::LirInst {
                        kind: LirInstKind::Value {
                            op: LirLeafOp::from_primitive(PrimitiveOpKind::Drop).unwrap(),
                            args: alloc::vec![LirValue(1)],
                            results: Vec::new(),
                        },
                    },
                ],
                terminator: LirTerminator::Return { results: None },
            }],
            value_types: alloc::vec![ValueType::I32, ValueType::I64],
        };

        optimize_lir(&mut program, plan_frame_layout(1, 2, 0));

        let block = &program.blocks[0];
        assert!(matches!(block.ops[2].kind, LirInstKind::LoadSlot { .. }));
        assert!(matches!(
            &block.ops[3].kind,
            LirInstKind::Value {
                args,
                ..
            } if args == &alloc::vec![LirValue(1)]
        ));
    }
}
