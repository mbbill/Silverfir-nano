//! Post-rewrite cleanup passes.
//!
//! Rewrite intentionally emits explicit boundary-repair blocks because they
//! make the planner contract easy to reason about. Once the full SSA program
//! exists, we can simplify the CFG mechanically:
//!
//! - canonicalize cache-only repair runs
//! - thread empty goto blocks
//! - merge unconditional single-predecessor successor blocks
//! - drop blocks that became unreachable
//!
//! These are structural cleanups only. They must preserve the prepared SSA
//! semantics exactly; they are not heuristic optimizations.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use super::ssa_ir::{
    ir::{
        SsaBinding, SsaEdge, SsaInst, SsaInstKind, SsaOperand, SsaProgram, SsaTerminator, SsaValue,
    },
    target::SsaTarget,
};

pub(crate) fn cleanup_program(program: &mut SsaProgram) {
    loop {
        let mut changed = false;

        if simplify_cache_only_runs(program) {
            changed = true;
        }
        if thread_one_empty_goto_block(program) {
            continue;
        }
        if merge_one_goto_successor(program) {
            continue;
        }
        if remove_unreachable_blocks(program) {
            continue;
        }

        if !changed {
            break;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CachePresence {
    Reserved,
    Ensured,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CacheRunState {
    needs_drop: bool,
    present: Option<CachePresence>,
}

impl CacheRunState {
    fn apply(&mut self, kind: &SsaInstKind) {
        match *kind {
            SsaInstKind::LocalDropCache { .. } => {
                self.needs_drop = true;
                self.present = None;
            }
            SsaInstKind::LocalReserveCache { .. } => {
                if self.present != Some(CachePresence::Ensured) {
                    self.present = Some(CachePresence::Reserved);
                }
            }
            SsaInstKind::LocalEnsureCache { .. } => {
                self.present = Some(CachePresence::Ensured);
            }
            _ => {}
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EdgeLocation {
    Goto { block: usize },
    BranchThen { block: usize },
    BranchElse { block: usize },
    BrTable { block: usize, entry: usize },
}

fn simplify_cache_only_runs(program: &mut SsaProgram) -> bool {
    let mut changed = false;

    for block in &mut program.blocks {
        let mut new_ops = Vec::with_capacity(block.ops.len());
        let mut index = 0usize;
        while index < block.ops.len() {
            if !is_cache_only_op(&block.ops[index].kind) {
                new_ops.push(block.ops[index].clone());
                index += 1;
                continue;
            }

            let start = index;
            while index < block.ops.len() && is_cache_only_op(&block.ops[index].kind) {
                index += 1;
            }

            let replacement = simplify_cache_run(&block.ops[start..index]);
            if replacement.len() != index - start
                || replacement
                    .iter()
                    .zip(block.ops[start..index].iter())
                    .any(|(new_inst, old_inst)| new_inst.kind != old_inst.kind)
            {
                changed = true;
            }
            new_ops.extend(replacement);
        }

        block.ops = new_ops;
    }

    changed
}

fn simplify_cache_run(run: &[SsaInst]) -> Vec<SsaInst> {
    let mut by_slot = BTreeMap::<u16, CacheRunState>::new();
    for inst in run {
        let slot =
            cache_run_slot(&inst.kind).expect("cache-only run should only contain cache ops");
        by_slot.entry(slot.0).or_default().apply(&inst.kind);
    }

    let mut out = Vec::new();
    for (&slot_index, state) in &by_slot {
        let slot = super::frame::FrameSlot(slot_index);
        if state.needs_drop {
            out.push(SsaInst {
                kind: SsaInstKind::LocalDropCache { slot },
            });
        }
    }
    for (&slot_index, state) in &by_slot {
        let slot = super::frame::FrameSlot(slot_index);
        match state.present {
            Some(CachePresence::Ensured) => out.push(SsaInst {
                kind: SsaInstKind::LocalEnsureCache { slot },
            }),
            Some(CachePresence::Reserved) => out.push(SsaInst {
                kind: SsaInstKind::LocalReserveCache { slot },
            }),
            None => {}
        }
    }
    out
}

#[inline]
fn is_cache_only_op(kind: &SsaInstKind) -> bool {
    matches!(
        kind,
        SsaInstKind::LocalEnsureCache { .. }
            | SsaInstKind::LocalReserveCache { .. }
            | SsaInstKind::LocalDropCache { .. }
    )
}

#[inline]
fn cache_run_slot(kind: &SsaInstKind) -> Option<super::frame::FrameSlot> {
    match *kind {
        SsaInstKind::LocalEnsureCache { slot }
        | SsaInstKind::LocalReserveCache { slot }
        | SsaInstKind::LocalDropCache { slot } => Some(slot),
        _ => None,
    }
}

fn thread_one_empty_goto_block(program: &mut SsaProgram) -> bool {
    for block_index in 0..program.blocks.len() {
        let Some(out_edge) = (match &program.blocks[block_index].terminator {
            SsaTerminator::Goto(edge) if program.blocks[block_index].ops.is_empty() => {
                Some(edge.clone())
            }
            _ => None,
        }) else {
            continue;
        };
        if out_edge.target.as_usize() == block_index {
            continue;
        }

        let params = program.blocks[block_index].params.clone();
        if block_index == program.entry.as_usize() {
            if !params.is_empty() || !out_edge.bindings.is_empty() {
                continue;
            }
            if program
                .block_entry_cached_slots
                .get(out_edge.target.as_usize())
                .map(|slots| !slots.is_empty())
                .unwrap_or(false)
            {
                continue;
            }
            merge_block_origins_into_target(program, block_index, out_edge.target.as_usize());
            program.entry = out_edge.target;
            remove_blocks(program, &[block_index]);
            return true;
        }

        let incoming = incoming_edge_locations(program, block_index);
        if incoming.is_empty() {
            continue;
        }

        let mut composed = Vec::with_capacity(incoming.len());
        for &loc in &incoming {
            let edge = edge_at(program, loc);
            let Some(new_edge) = compose_edge(edge, &params, &out_edge) else {
                composed.clear();
                break;
            };
            composed.push(new_edge);
        }
        if composed.is_empty() {
            continue;
        }

        for (loc, new_edge) in incoming.into_iter().zip(composed.into_iter()) {
            *edge_at_mut(program, loc) = new_edge;
        }
        merge_block_origins_into_target(program, block_index, out_edge.target.as_usize());
        remove_blocks(program, &[block_index]);
        return true;
    }

    false
}

fn merge_one_goto_successor(program: &mut SsaProgram) -> bool {
    let predecessor_counts = predecessor_counts(program);
    for pred_index in 0..program.blocks.len() {
        let Some(edge) = (match &program.blocks[pred_index].terminator {
            SsaTerminator::Goto(edge) => Some(edge.clone()),
            _ => None,
        }) else {
            continue;
        };
        let succ_index = edge.target.as_usize();
        if succ_index == pred_index || succ_index >= program.blocks.len() {
            continue;
        }
        if predecessor_counts[succ_index] != 1 {
            continue;
        }

        let succ = program.blocks[succ_index].clone();
        let Some(subst) = binding_substitution(&succ.params, &edge.bindings) else {
            continue;
        };

        let merged_ops = succ
            .ops
            .iter()
            .cloned()
            .map(|inst| substitute_inst(inst, &subst))
            .collect::<Vec<_>>();
        let merged_terminator = substitute_terminator(succ.terminator, &subst);

        program.blocks[pred_index].ops.extend(merged_ops);
        program.blocks[pred_index].terminator = merged_terminator;
        merge_block_origins(pred_index, succ_index, &mut program.block_cfg_origins);
        remove_blocks(program, &[succ_index]);
        return true;
    }

    false
}

fn remove_unreachable_blocks(program: &mut SsaProgram) -> bool {
    if program.blocks.is_empty() {
        return false;
    }

    let mut reachable = vec![false; program.blocks.len()];
    let mut stack = vec![program.entry.as_usize()];
    while let Some(block_index) = stack.pop() {
        if block_index >= program.blocks.len() || reachable[block_index] {
            continue;
        }
        reachable[block_index] = true;
        for edge in outgoing_edges(&program.blocks[block_index].terminator) {
            stack.push(edge.target.as_usize());
        }
    }

    let removed = reachable
        .iter()
        .enumerate()
        .filter_map(|(index, keep)| (!*keep).then_some(index))
        .collect::<Vec<_>>();
    if removed.is_empty() {
        return false;
    }
    remove_blocks(program, &removed);
    true
}

fn predecessor_counts(program: &SsaProgram) -> Vec<usize> {
    let mut counts = vec![0usize; program.blocks.len()];
    for block in &program.blocks {
        for edge in outgoing_edges(&block.terminator) {
            if let Some(count) = counts.get_mut(edge.target.as_usize()) {
                *count += 1;
            }
        }
    }
    counts
}

fn incoming_edge_locations(program: &SsaProgram, target_index: usize) -> Vec<EdgeLocation> {
    let mut incoming = Vec::new();
    for (block_index, block) in program.blocks.iter().enumerate() {
        match &block.terminator {
            SsaTerminator::Goto(edge) => {
                if edge.target.as_usize() == target_index {
                    incoming.push(EdgeLocation::Goto { block: block_index });
                }
            }
            SsaTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                if then_edge.target.as_usize() == target_index {
                    incoming.push(EdgeLocation::BranchThen { block: block_index });
                }
                if else_edge.target.as_usize() == target_index {
                    incoming.push(EdgeLocation::BranchElse { block: block_index });
                }
            }
            SsaTerminator::BrTable { entries, .. } => {
                for (entry_index, edge) in entries.iter().enumerate() {
                    if edge.target.as_usize() == target_index {
                        incoming.push(EdgeLocation::BrTable {
                            block: block_index,
                            entry: entry_index,
                        });
                    }
                }
            }
            SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
        }
    }
    incoming
}

fn compose_edge(in_edge: &SsaEdge, params: &[SsaValue], out_edge: &SsaEdge) -> Option<SsaEdge> {
    let param_set = params.iter().copied().collect::<BTreeSet<_>>();
    let param_map = in_edge
        .bindings
        .iter()
        .map(|binding| (binding.param, binding.value))
        .collect::<BTreeMap<_, _>>();

    let mut bindings = Vec::with_capacity(out_edge.bindings.len());
    for binding in &out_edge.bindings {
        let value = if param_set.contains(&binding.value) {
            *param_map.get(&binding.value)?
        } else {
            binding.value
        };
        bindings.push(SsaBinding {
            param: binding.param,
            value,
        });
    }

    Some(SsaEdge {
        target: out_edge.target,
        bindings,
    })
}

fn binding_substitution(
    params: &[SsaValue],
    bindings: &[SsaBinding],
) -> Option<BTreeMap<SsaValue, SsaValue>> {
    let map = bindings
        .iter()
        .map(|binding| (binding.param, binding.value))
        .collect::<BTreeMap<_, _>>();
    for &param in params {
        map.get(&param)?;
    }
    Some(map)
}

fn substitute_inst(inst: SsaInst, subst: &BTreeMap<SsaValue, SsaValue>) -> SsaInst {
    SsaInst {
        kind: match inst.kind {
            SsaInstKind::Value { op, args, results } => SsaInstKind::Value {
                op,
                args: args
                    .into_iter()
                    .map(|arg| match arg {
                        SsaOperand::Value(value) => {
                            SsaOperand::Value(substitute_value(value, subst))
                        }
                        SsaOperand::Const(bits) => SsaOperand::Const(bits),
                    })
                    .collect(),
                results,
            },
            SsaInstKind::Fill { slot, dst } => SsaInstKind::Fill { slot, dst },
            SsaInstKind::Spill { slot, src } => SsaInstKind::Spill {
                slot,
                src: substitute_value(src, subst),
            },
            SsaInstKind::LocalGetSlot { slot, dst } => SsaInstKind::LocalGetSlot { slot, dst },
            SsaInstKind::LocalGetCache { slot, dst } => SsaInstKind::LocalGetCache { slot, dst },
            SsaInstKind::LocalSetSlot { slot, src } => SsaInstKind::LocalSetSlot {
                slot,
                src: substitute_value(src, subst),
            },
            SsaInstKind::LocalSetCache { slot, src } => SsaInstKind::LocalSetCache {
                slot,
                src: substitute_value(src, subst),
            },
            SsaInstKind::LocalEnsureCache { slot } => SsaInstKind::LocalEnsureCache { slot },
            SsaInstKind::LocalReserveCache { slot } => SsaInstKind::LocalReserveCache { slot },
            SsaInstKind::LocalDropCache { slot } => SsaInstKind::LocalDropCache { slot },
            SsaInstKind::Call(op) => SsaInstKind::Call(op),
        },
    }
}

fn substitute_terminator(
    terminator: SsaTerminator,
    subst: &BTreeMap<SsaValue, SsaValue>,
) -> SsaTerminator {
    match terminator {
        SsaTerminator::Goto(edge) => SsaTerminator::Goto(substitute_edge(edge, subst)),
        SsaTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => SsaTerminator::Branch {
            cond: substitute_value(cond, subst),
            then_edge: substitute_edge(then_edge, subst),
            else_edge: substitute_edge(else_edge, subst),
        },
        SsaTerminator::BrTable { index, entries } => SsaTerminator::BrTable {
            index: substitute_value(index, subst),
            entries: entries
                .into_iter()
                .map(|edge| substitute_edge(edge, subst))
                .collect(),
        },
        SsaTerminator::Return { results } => SsaTerminator::Return { results },
        SsaTerminator::TrapUnreachable => SsaTerminator::TrapUnreachable,
    }
}

fn substitute_edge(edge: SsaEdge, subst: &BTreeMap<SsaValue, SsaValue>) -> SsaEdge {
    SsaEdge {
        target: edge.target,
        bindings: edge
            .bindings
            .into_iter()
            .map(|binding| SsaBinding {
                param: binding.param,
                value: substitute_value(binding.value, subst),
            })
            .collect(),
    }
}

#[inline]
fn substitute_value(value: SsaValue, subst: &BTreeMap<SsaValue, SsaValue>) -> SsaValue {
    subst.get(&value).copied().unwrap_or(value)
}

fn remove_blocks(program: &mut SsaProgram, removed: &[usize]) {
    if removed.is_empty() {
        return;
    }

    let removed = removed.iter().copied().collect::<BTreeSet<_>>();
    let mut mapping = vec![SsaTarget::default(); program.blocks.len()];
    let mut next = 0u32;
    for (old_index, _) in program.blocks.iter().enumerate() {
        if removed.contains(&old_index) {
            continue;
        }
        mapping[old_index] = SsaTarget(next);
        next += 1;
    }

    let mut new_blocks = Vec::with_capacity(program.blocks.len() - removed.len());
    let mut new_entries = Vec::with_capacity(
        program
            .block_entry_cached_slots
            .len()
            .saturating_sub(removed.len()),
    );
    let keep_origins = !program.block_cfg_origins.is_empty();
    let mut new_origins = if keep_origins {
        Vec::with_capacity(
            program
                .block_cfg_origins
                .len()
                .saturating_sub(removed.len()),
        )
    } else {
        Vec::new()
    };
    for (old_index, mut block) in program.blocks.drain(..).enumerate() {
        if removed.contains(&old_index) {
            continue;
        }
        remap_terminator_targets(&mut block.terminator, &mapping);
        block.id = mapping[old_index];
        new_blocks.push(block);
        new_entries.push(program.block_entry_cached_slots[old_index].clone());
        if keep_origins {
            new_origins.push(program.block_cfg_origins[old_index].clone());
        }
    }

    program.entry = mapping[program.entry.as_usize()];
    program.blocks = new_blocks;
    program.block_entry_cached_slots = new_entries;
    if keep_origins {
        program.block_cfg_origins = new_origins;
    }
}

fn merge_block_origins_into_target(program: &mut SsaProgram, from: usize, to: usize) {
    if program.block_cfg_origins.is_empty()
        || from == to
        || from >= program.block_cfg_origins.len()
        || to >= program.block_cfg_origins.len()
    {
        return;
    }
    let from_origins = program.block_cfg_origins[from].clone();
    for origin in from_origins {
        if !program.block_cfg_origins[to].contains(&origin) {
            program.block_cfg_origins[to].push(origin);
        }
    }
    program.block_cfg_origins[to].sort_unstable();
}

fn merge_block_origins(dst: usize, src: usize, origins: &mut [Vec<u32>]) {
    if origins.is_empty() || dst == src || dst >= origins.len() || src >= origins.len() {
        return;
    }
    let src_origins = origins[src].clone();
    for origin in src_origins {
        if !origins[dst].contains(&origin) {
            origins[dst].push(origin);
        }
    }
    origins[dst].sort_unstable();
}

fn remap_terminator_targets(term: &mut SsaTerminator, mapping: &[SsaTarget]) {
    match term {
        SsaTerminator::Goto(edge) => edge.target = mapping[edge.target.as_usize()],
        SsaTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            then_edge.target = mapping[then_edge.target.as_usize()];
            else_edge.target = mapping[else_edge.target.as_usize()];
        }
        SsaTerminator::BrTable { entries, .. } => {
            for edge in entries {
                edge.target = mapping[edge.target.as_usize()];
            }
        }
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
    }
}

fn outgoing_edges(term: &SsaTerminator) -> Vec<&SsaEdge> {
    match term {
        SsaTerminator::Goto(edge) => alloc::vec![edge],
        SsaTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => alloc::vec![then_edge, else_edge],
        SsaTerminator::BrTable { entries, .. } => entries.iter().collect(),
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => Vec::new(),
    }
}

fn edge_at(program: &SsaProgram, location: EdgeLocation) -> &SsaEdge {
    match location {
        EdgeLocation::Goto { block } => match &program.blocks[block].terminator {
            SsaTerminator::Goto(edge) => edge,
            _ => unreachable!("edge location should point at a goto edge"),
        },
        EdgeLocation::BranchThen { block } => match &program.blocks[block].terminator {
            SsaTerminator::Branch { then_edge, .. } => then_edge,
            _ => unreachable!("edge location should point at a branch-then edge"),
        },
        EdgeLocation::BranchElse { block } => match &program.blocks[block].terminator {
            SsaTerminator::Branch { else_edge, .. } => else_edge,
            _ => unreachable!("edge location should point at a branch-else edge"),
        },
        EdgeLocation::BrTable { block, entry } => match &program.blocks[block].terminator {
            SsaTerminator::BrTable { entries, .. } => &entries[entry],
            _ => unreachable!("edge location should point at a br_table edge"),
        },
    }
}

fn edge_at_mut(program: &mut SsaProgram, location: EdgeLocation) -> &mut SsaEdge {
    match location {
        EdgeLocation::Goto { block } => match &mut program.blocks[block].terminator {
            SsaTerminator::Goto(edge) => edge,
            _ => unreachable!("edge location should point at a goto edge"),
        },
        EdgeLocation::BranchThen { block } => match &mut program.blocks[block].terminator {
            SsaTerminator::Branch { then_edge, .. } => then_edge,
            _ => unreachable!("edge location should point at a branch-then edge"),
        },
        EdgeLocation::BranchElse { block } => match &mut program.blocks[block].terminator {
            SsaTerminator::Branch { else_edge, .. } => else_edge,
            _ => unreachable!("edge location should point at a branch-else edge"),
        },
        EdgeLocation::BrTable { block, entry } => match &mut program.blocks[block].terminator {
            SsaTerminator::BrTable { entries, .. } => &mut entries[entry],
            _ => unreachable!("edge location should point at a br_table edge"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::middle::{
        frame::FrameSlot,
        ssa_ir::{ir::SsaBlock, leaf::SsaLeafOp, validate::validate_program},
    };
    use crate::vm::wasm::primitive_op::PrimitiveOpKind;

    fn value_inst(result: u32) -> SsaInst {
        SsaInst {
            kind: SsaInstKind::Value {
                op: SsaLeafOp::from_primitive(PrimitiveOpKind::I32Const { value: 1 }).unwrap(),
                args: Vec::new(),
                results: alloc::vec![SsaValue(result)],
            },
        }
    }

    #[test]
    fn simplify_cache_run_keeps_required_drop_then_materialization() {
        let run = alloc::vec![
            SsaInst {
                kind: SsaInstKind::LocalReserveCache { slot: FrameSlot(0) },
            },
            SsaInst {
                kind: SsaInstKind::LocalDropCache { slot: FrameSlot(0) },
            },
            SsaInst {
                kind: SsaInstKind::LocalEnsureCache { slot: FrameSlot(0) },
            },
            SsaInst {
                kind: SsaInstKind::LocalEnsureCache { slot: FrameSlot(0) },
            },
        ];
        let simplified = simplify_cache_run(&run);
        assert_eq!(
            simplified.iter().map(|inst| &inst.kind).collect::<Vec<_>>(),
            alloc::vec![
                &SsaInstKind::LocalDropCache { slot: FrameSlot(0) },
                &SsaInstKind::LocalEnsureCache { slot: FrameSlot(0) },
            ]
        );
    }

    #[test]
    fn threads_empty_goto_block_and_composes_bindings() {
        let mut program = SsaProgram {
            entry: SsaTarget(0),
            blocks: alloc::vec![
                SsaBlock {
                    id: SsaTarget(0),
                    params: Vec::new(),
                    ops: Vec::new(),
                    terminator: SsaTerminator::Goto(SsaEdge {
                        target: SsaTarget(1),
                        bindings: alloc::vec![SsaBinding {
                            param: SsaValue(10),
                            value: SsaValue(0),
                        }],
                    }),
                },
                SsaBlock {
                    id: SsaTarget(1),
                    params: alloc::vec![SsaValue(10)],
                    ops: Vec::new(),
                    terminator: SsaTerminator::Goto(SsaEdge {
                        target: SsaTarget(2),
                        bindings: alloc::vec![SsaBinding {
                            param: SsaValue(20),
                            value: SsaValue(10),
                        }],
                    }),
                },
                SsaBlock {
                    id: SsaTarget(2),
                    params: alloc::vec![SsaValue(20)],
                    ops: Vec::new(),
                    terminator: SsaTerminator::Return { results: None },
                },
            ],
            local_slot_types: Vec::new(),
            local_slot_info: Vec::new(),
            block_entry_cached_slots: alloc::vec![Vec::new(), Vec::new(), Vec::new()],
            block_cfg_origins: alloc::vec![],
            value_types: alloc::vec![crate::value_type::ValueType::I32; 32],
            value_sink_local: alloc::vec![None; 32],
        };

        assert!(thread_one_empty_goto_block(&mut program));
        assert_eq!(program.blocks.len(), 2);
        let edge = match &program.blocks[0].terminator {
            SsaTerminator::Goto(edge) => edge,
            other => panic!("expected goto after threading, got {other:?}"),
        };
        assert_eq!(edge.target, SsaTarget(1));
        assert_eq!(
            edge.bindings,
            alloc::vec![SsaBinding {
                param: SsaValue(20),
                value: SsaValue(0),
            }]
        );
        validate_program(&program).unwrap();
    }

    #[test]
    fn merges_goto_successor_with_param_substitution() {
        let mut program = SsaProgram {
            entry: SsaTarget(0),
            blocks: alloc::vec![
                SsaBlock {
                    id: SsaTarget(0),
                    params: Vec::new(),
                    ops: alloc::vec![value_inst(0)],
                    terminator: SsaTerminator::Goto(SsaEdge {
                        target: SsaTarget(1),
                        bindings: alloc::vec![SsaBinding {
                            param: SsaValue(10),
                            value: SsaValue(0),
                        }],
                    }),
                },
                SsaBlock {
                    id: SsaTarget(1),
                    params: alloc::vec![SsaValue(10)],
                    ops: alloc::vec![SsaInst {
                        kind: SsaInstKind::LocalSetCache {
                            slot: FrameSlot(0),
                            src: SsaValue(10),
                        },
                    }],
                    terminator: SsaTerminator::Return { results: None },
                },
            ],
            local_slot_types: alloc::vec![crate::value_type::ValueType::I32],
            local_slot_info: alloc::vec![Default::default()],
            block_entry_cached_slots: alloc::vec![Vec::new(), Vec::new()],
            block_cfg_origins: alloc::vec![],
            value_types: alloc::vec![crate::value_type::ValueType::I32; 16],
            value_sink_local: alloc::vec![None; 16],
        };

        assert!(merge_one_goto_successor(&mut program));
        assert_eq!(program.blocks.len(), 1);
        assert_eq!(program.blocks[0].ops.len(), 2);
        assert!(matches!(
            program.blocks[0].ops[1].kind,
            SsaInstKind::LocalSetCache {
                src: SsaValue(0),
                ..
            }
        ));
        assert!(matches!(
            program.blocks[0].terminator,
            SsaTerminator::Return { .. }
        ));
        validate_program(&program).unwrap();
    }

    #[test]
    fn remove_unreachable_blocks_prunes_dead_chain() {
        let mut program = SsaProgram {
            entry: SsaTarget(0),
            blocks: alloc::vec![
                SsaBlock {
                    id: SsaTarget(0),
                    params: Vec::new(),
                    ops: Vec::new(),
                    terminator: SsaTerminator::Return { results: None },
                },
                SsaBlock {
                    id: SsaTarget(1),
                    params: Vec::new(),
                    ops: Vec::new(),
                    terminator: SsaTerminator::Goto(SsaEdge {
                        target: SsaTarget(2),
                        bindings: Vec::new(),
                    }),
                },
                SsaBlock {
                    id: SsaTarget(2),
                    params: Vec::new(),
                    ops: Vec::new(),
                    terminator: SsaTerminator::Return { results: None },
                },
            ],
            local_slot_types: Vec::new(),
            local_slot_info: Vec::new(),
            block_entry_cached_slots: alloc::vec![Vec::new(), Vec::new(), Vec::new()],
            block_cfg_origins: alloc::vec![],
            value_types: Vec::new(),
            value_sink_local: Vec::new(),
        };

        assert!(remove_unreachable_blocks(&mut program));
        assert_eq!(program.blocks.len(), 1);
        assert_eq!(program.entry, SsaTarget(0));
    }
}
