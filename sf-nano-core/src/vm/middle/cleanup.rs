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

use alloc::vec;
use alloc::vec::Vec;

use super::ssa_ir::{
    ir::{
        SsaBinding, SsaBlock, SsaEdge, SsaInst, SsaInstKind, SsaOperand, SsaProgram, SsaTerminator,
        SsaValue,
    },
    target::SsaTarget,
};

pub(crate) fn cleanup_program(program: &mut SsaProgram) {
    loop {
        simplify_cache_only_runs(program);
        while thread_one_empty_goto_block(program) {}
        if merge_one_goto_successor(program) {
            continue;
        }
        if remove_unreachable_blocks(program) {
            continue;
        }
        break;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ValueSubstitution {
    from: SsaValue,
    to: SsaValue,
}

fn simplify_cache_only_runs(program: &mut SsaProgram) {
    for block in &mut program.blocks {
        let old_ops = core::mem::take(&mut block.ops);
        let mut new_ops = Vec::with_capacity(old_ops.len());
        let mut cache_run = Vec::new();
        for inst in old_ops {
            if !is_cache_only_op(&inst.kind) {
                flush_cache_run(&mut new_ops, &mut cache_run);
                new_ops.push(inst);
                continue;
            }
            accumulate_cache_run_state(&mut cache_run, &inst.kind);
        }
        flush_cache_run(&mut new_ops, &mut cache_run);

        block.ops = new_ops;
    }
}

#[cfg(test)]
fn simplify_cache_run(run: &[SsaInst]) -> Vec<SsaInst> {
    let mut by_slot = Vec::<(u16, CacheRunState)>::new();
    for inst in run {
        accumulate_cache_run_state(&mut by_slot, &inst.kind);
    }

    let mut out = Vec::with_capacity(by_slot.len().saturating_mul(2));
    flush_cache_run(&mut out, &mut by_slot);
    out
}

fn accumulate_cache_run_state(by_slot: &mut Vec<(u16, CacheRunState)>, kind: &SsaInstKind) {
    let slot = cache_run_slot(kind).expect("cache-only run should only contain cache ops");
    match by_slot
        .iter_mut()
        .find(|(slot_index, _)| *slot_index == slot.0)
    {
        Some((_, state)) => state.apply(kind),
        None => {
            let mut state = CacheRunState::default();
            state.apply(kind);
            by_slot.push((slot.0, state));
        }
    }
}

fn flush_cache_run(out: &mut Vec<SsaInst>, by_slot: &mut Vec<(u16, CacheRunState)>) {
    if by_slot.is_empty() {
        return;
    }
    by_slot.sort_unstable_by_key(|(slot_index, _)| *slot_index);
    for &(slot_index, state) in by_slot.iter() {
        let slot = super::frame::FrameSlot(slot_index);
        if state.needs_drop {
            out.push(SsaInst {
                kind: SsaInstKind::LocalDropCache { slot },
            });
        }
    }
    for &(slot_index, state) in by_slot.iter() {
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
    by_slot.clear();
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
        let (target, bindings_empty) = match &program.blocks[block_index].terminator {
            SsaTerminator::Goto(edge) if program.blocks[block_index].ops.is_empty() => {
                (edge.target, edge.bindings.is_empty())
            }
            _ => continue,
        };
        if target.as_usize() == block_index {
            continue;
        }

        if block_index == program.entry.as_usize() {
            if !program.blocks[block_index].params.is_empty() || !bindings_empty {
                continue;
            }
            if program
                .block_entry_cached_slots
                .get(target.as_usize())
                .map(|slots| !slots.is_empty())
                .unwrap_or(false)
            {
                continue;
            }
            merge_block_origins_into_target(program, block_index, target.as_usize());
            program.entry = target;
            remove_blocks(program, &[block_index]);
            return true;
        }

        let incoming = incoming_edge_locations(program, block_index);
        if incoming.is_empty() {
            continue;
        }

        let mut composed = Vec::with_capacity(incoming.len());
        {
            let block = &program.blocks[block_index];
            let params = &block.params;
            let SsaTerminator::Goto(out_edge) = &block.terminator else {
                unreachable!("empty goto block should still end in goto");
            };
            for &loc in &incoming {
                let edge = edge_at(program, loc);
                let Some(new_edge) = compose_edge(edge, params, out_edge) else {
                    composed.clear();
                    break;
                };
                composed.push(new_edge);
            }
        }
        if composed.is_empty() {
            continue;
        }

        for (loc, new_edge) in incoming.into_iter().zip(composed.into_iter()) {
            *edge_at_mut(program, loc) = new_edge;
        }
        merge_block_origins_into_target(program, block_index, target.as_usize());
        remove_blocks(program, &[block_index]);
        return true;
    }

    false
}

fn merge_one_goto_successor(program: &mut SsaProgram) -> bool {
    let predecessor_counts = predecessor_counts(program);
    for pred_index in 0..program.blocks.len() {
        let SsaTerminator::Goto(edge) = &program.blocks[pred_index].terminator else {
            continue;
        };
        let succ_index = edge.target.as_usize();
        if succ_index == pred_index || succ_index >= program.blocks.len() {
            continue;
        }
        if succ_index == program.entry.as_usize() {
            continue;
        }
        if predecessor_counts[succ_index] != 1 {
            continue;
        }

        let Some(subst) = binding_substitution(&program.blocks[succ_index].params, &edge.bindings)
        else {
            continue;
        };
        let succ = core::mem::replace(
            &mut program.blocks[succ_index],
            SsaBlock {
                id: SsaTarget(u32::MAX),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: SsaTerminator::TrapUnreachable,
            },
        );

        let merged_ops = succ
            .ops
            .into_iter()
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
        visit_outgoing_edges(&program.blocks[block_index].terminator, |edge| {
            stack.push(edge.target.as_usize());
        });
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
        visit_outgoing_edges(&block.terminator, |edge| {
            if let Some(count) = counts.get_mut(edge.target.as_usize()) {
                *count += 1;
            }
        });
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
    let mut bindings = Vec::with_capacity(out_edge.bindings.len());
    for binding in &out_edge.bindings {
        let value = if params.contains(&binding.value) {
            find_binding_value(&in_edge.bindings, binding.value)?
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
) -> Option<Vec<ValueSubstitution>> {
    let mut subst = Vec::with_capacity(params.len());
    for &param in params {
        subst.push(ValueSubstitution {
            from: param,
            to: find_binding_value(bindings, param)?,
        });
    }
    Some(subst)
}

fn substitute_inst(inst: SsaInst, subst: &[ValueSubstitution]) -> SsaInst {
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

fn substitute_terminator(terminator: SsaTerminator, subst: &[ValueSubstitution]) -> SsaTerminator {
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

fn substitute_edge(edge: SsaEdge, subst: &[ValueSubstitution]) -> SsaEdge {
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
fn substitute_value(value: SsaValue, subst: &[ValueSubstitution]) -> SsaValue {
    subst
        .iter()
        .find(|entry| entry.from == value)
        .map_or(value, |entry| entry.to)
}

fn remove_blocks(program: &mut SsaProgram, removed: &[usize]) {
    if removed.is_empty() {
        return;
    }
    if removed.len() == 1 {
        remove_one_block(program, removed[0]);
        return;
    }

    let block_len = program.blocks.len();
    let mut removed_mask = vec![false; block_len];
    for &index in removed {
        debug_assert!(index < block_len, "removed block index out of range");
        if let Some(slot) = removed_mask.get_mut(index) {
            *slot = true;
        }
    }

    let mut mapping = vec![SsaTarget::default(); block_len];
    let mut next = 0u32;
    for (old_index, is_removed) in removed_mask.iter().copied().enumerate() {
        if is_removed {
            continue;
        }
        mapping[old_index] = SsaTarget(next);
        next += 1;
    }

    let old_blocks = core::mem::take(&mut program.blocks);
    let old_entries = core::mem::take(&mut program.block_entry_cached_slots);
    let keep_origins = !program.block_cfg_origins.is_empty();
    let old_origins = if keep_origins {
        core::mem::take(&mut program.block_cfg_origins)
    } else {
        Vec::new()
    };
    let kept_blocks = block_len.saturating_sub(removed.len());
    let mut new_blocks = Vec::with_capacity(kept_blocks);
    let mut new_entries = Vec::with_capacity(old_entries.len().saturating_sub(removed.len()));
    let mut new_origins = if keep_origins {
        Vec::with_capacity(old_origins.len().saturating_sub(removed.len()))
    } else {
        Vec::new()
    };

    if keep_origins {
        for (old_index, ((mut block, entry_slots), origins)) in old_blocks
            .into_iter()
            .zip(old_entries.into_iter())
            .zip(old_origins.into_iter())
            .enumerate()
        {
            if removed_mask[old_index] {
                continue;
            }
            remap_terminator_targets(&mut block.terminator, &mapping);
            block.id = mapping[old_index];
            new_blocks.push(block);
            new_entries.push(entry_slots);
            new_origins.push(origins);
        }
    } else {
        for (old_index, (mut block, entry_slots)) in old_blocks
            .into_iter()
            .zip(old_entries.into_iter())
            .enumerate()
        {
            if removed_mask[old_index] {
                continue;
            }
            remap_terminator_targets(&mut block.terminator, &mapping);
            block.id = mapping[old_index];
            new_blocks.push(block);
            new_entries.push(entry_slots);
        }
    }

    debug_assert!(removed_mask.get(program.entry.as_usize()).copied() != Some(true));
    program.entry = mapping[program.entry.as_usize()];
    program.blocks = new_blocks;
    program.block_entry_cached_slots = new_entries;
    if keep_origins {
        program.block_cfg_origins = new_origins;
    }
}

fn remove_one_block(program: &mut SsaProgram, removed_index: usize) {
    debug_assert!(
        removed_index < program.blocks.len(),
        "removed block index out of range"
    );
    debug_assert_ne!(program.entry.as_usize(), removed_index);

    program.blocks.remove(removed_index);
    program.block_entry_cached_slots.remove(removed_index);
    if !program.block_cfg_origins.is_empty() {
        program.block_cfg_origins.remove(removed_index);
    }

    if program.entry.as_usize() > removed_index {
        program.entry = SsaTarget(program.entry.0 - 1);
    }

    for (index, block) in program.blocks.iter_mut().enumerate().skip(removed_index) {
        block.id = SsaTarget(index as u32);
    }
    for block in &mut program.blocks {
        remap_terminator_target_after_single_removal(&mut block.terminator, removed_index);
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

    let (from_origins, to_origins) = if from < to {
        let (head, tail) = program.block_cfg_origins.split_at_mut(to);
        (&head[from], &mut tail[0])
    } else {
        let (head, tail) = program.block_cfg_origins.split_at_mut(from);
        (&tail[0], &mut head[to])
    };
    merge_origin_lists(from_origins, to_origins);
}

fn merge_block_origins(dst: usize, src: usize, origins: &mut [Vec<u32>]) {
    if origins.is_empty() || dst == src || dst >= origins.len() || src >= origins.len() {
        return;
    }

    let (src_origins, dst_origins) = if src < dst {
        let (head, tail) = origins.split_at_mut(dst);
        (&head[src], &mut tail[0])
    } else {
        let (head, tail) = origins.split_at_mut(src);
        (&tail[0], &mut head[dst])
    };
    merge_origin_lists(src_origins, dst_origins);
}

fn merge_origin_lists(src: &[u32], dst: &mut Vec<u32>) {
    for &origin in src {
        if !dst.contains(&origin) {
            dst.push(origin);
        }
    }
    dst.sort_unstable();
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

fn remap_terminator_target_after_single_removal(term: &mut SsaTerminator, removed_index: usize) {
    match term {
        SsaTerminator::Goto(edge) => {
            remap_target_after_single_removal(&mut edge.target, removed_index);
        }
        SsaTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            remap_target_after_single_removal(&mut then_edge.target, removed_index);
            remap_target_after_single_removal(&mut else_edge.target, removed_index);
        }
        SsaTerminator::BrTable { entries, .. } => {
            for edge in entries {
                remap_target_after_single_removal(&mut edge.target, removed_index);
            }
        }
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
    }
}

#[inline]
fn remap_target_after_single_removal(target: &mut SsaTarget, removed_index: usize) {
    let target_index = target.as_usize();
    debug_assert_ne!(target_index, removed_index);
    if target_index > removed_index {
        target.0 -= 1;
    }
}

fn visit_outgoing_edges(term: &SsaTerminator, mut visit: impl FnMut(&SsaEdge)) {
    match term {
        SsaTerminator::Goto(edge) => visit(edge),
        SsaTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            visit(then_edge);
            visit(else_edge);
        }
        SsaTerminator::BrTable { entries, .. } => {
            for edge in entries {
                visit(edge);
            }
        }
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
    }
}

#[inline]
fn find_binding_value(bindings: &[SsaBinding], param: SsaValue) -> Option<SsaValue> {
    bindings
        .iter()
        .find(|binding| binding.param == param)
        .map(|binding| binding.value)
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

    #[test]
    fn does_not_merge_unreachable_predecessor_into_entry() {
        let mut program = SsaProgram {
            entry: SsaTarget(1),
            blocks: alloc::vec![
                SsaBlock {
                    id: SsaTarget(0),
                    params: Vec::new(),
                    ops: alloc::vec![value_inst(0)],
                    terminator: SsaTerminator::Goto(SsaEdge {
                        target: SsaTarget(1),
                        bindings: Vec::new(),
                    }),
                },
                SsaBlock {
                    id: SsaTarget(1),
                    params: Vec::new(),
                    ops: alloc::vec![value_inst(1)],
                    terminator: SsaTerminator::Return { results: None },
                },
            ],
            local_slot_types: Vec::new(),
            local_slot_info: Vec::new(),
            block_entry_cached_slots: alloc::vec![Vec::new(), Vec::new()],
            block_cfg_origins: alloc::vec![],
            value_types: alloc::vec![crate::value_type::ValueType::I32; 2],
            value_sink_local: alloc::vec![None; 2],
        };

        assert!(!merge_one_goto_successor(&mut program));
        assert!(remove_unreachable_blocks(&mut program));
        assert_eq!(program.entry, SsaTarget(0));
        assert_eq!(program.blocks.len(), 1);
        assert_eq!(program.blocks[0].id, SsaTarget(0));
        assert!(matches!(
            program.blocks[0].terminator,
            SsaTerminator::Return { .. }
        ));
        validate_program(&program).unwrap();
    }

    #[test]
    fn remove_single_block_reindexes_targets_and_entry() {
        let mut program = SsaProgram {
            entry: SsaTarget(2),
            blocks: alloc::vec![
                SsaBlock {
                    id: SsaTarget(0),
                    params: Vec::new(),
                    ops: Vec::new(),
                    terminator: SsaTerminator::Goto(SsaEdge {
                        target: SsaTarget(2),
                        bindings: Vec::new(),
                    }),
                },
                SsaBlock {
                    id: SsaTarget(1),
                    params: Vec::new(),
                    ops: Vec::new(),
                    terminator: SsaTerminator::Return { results: None },
                },
                SsaBlock {
                    id: SsaTarget(2),
                    params: Vec::new(),
                    ops: Vec::new(),
                    terminator: SsaTerminator::Goto(SsaEdge {
                        target: SsaTarget(3),
                        bindings: Vec::new(),
                    }),
                },
                SsaBlock {
                    id: SsaTarget(3),
                    params: Vec::new(),
                    ops: Vec::new(),
                    terminator: SsaTerminator::Return { results: None },
                },
            ],
            local_slot_types: Vec::new(),
            local_slot_info: Vec::new(),
            block_entry_cached_slots: alloc::vec![
                alloc::vec![FrameSlot(0)],
                alloc::vec![FrameSlot(1)],
                alloc::vec![FrameSlot(2)],
                alloc::vec![FrameSlot(3)],
            ],
            block_cfg_origins: alloc::vec![
                alloc::vec![10],
                alloc::vec![11],
                alloc::vec![12],
                alloc::vec![13],
            ],
            value_types: Vec::new(),
            value_sink_local: Vec::new(),
        };

        remove_blocks(&mut program, &[1]);

        assert_eq!(program.entry, SsaTarget(1));
        assert_eq!(program.blocks.len(), 3);
        assert_eq!(
            program
                .blocks
                .iter()
                .map(|block| block.id)
                .collect::<Vec<_>>(),
            alloc::vec![SsaTarget(0), SsaTarget(1), SsaTarget(2)]
        );
        assert_eq!(
            match &program.blocks[0].terminator {
                SsaTerminator::Goto(edge) => edge.target,
                other => panic!("expected goto, got {other:?}"),
            },
            SsaTarget(1)
        );
        assert_eq!(
            match &program.blocks[1].terminator {
                SsaTerminator::Goto(edge) => edge.target,
                other => panic!("expected goto, got {other:?}"),
            },
            SsaTarget(2)
        );
        assert_eq!(
            program.block_entry_cached_slots,
            alloc::vec![
                alloc::vec![FrameSlot(0)],
                alloc::vec![FrameSlot(2)],
                alloc::vec![FrameSlot(3)],
            ]
        );
        assert_eq!(
            program.block_cfg_origins,
            alloc::vec![alloc::vec![10], alloc::vec![12], alloc::vec![13],]
        );
        validate_program(&program).unwrap();
    }
}
