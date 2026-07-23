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

use crate::collections;

use super::ssa_ir::{
    ir::{
        SsaBinding, SsaBlock, SsaEdge, SsaInst, SsaOp, SsaOperand, SsaProgram, SsaScalarResultLoc,
        SsaTerminator, SsaValue,
    },
    target::SsaTarget,
};

pub(crate) fn cleanup_program(program: &mut SsaProgram) {
    loop {
        thread_empty_goto_blocks(program);
        merge_goto_successors(program);
        if remove_unreachable_blocks(program) {
            continue;
        }
        break;
    }
    fold_block_params_sourced_from_cached_cells(program);
    // Cache-run canonicalization neither creates empty goto blocks nor changes
    // CFG edges, so it cannot enable any of the structural transformations
    // above.  Running it inside the fixed-point loop rebuilt every block's op
    // vector after each single-block merge/removal, making cleanup quadratic on
    // large functions.  Structural convergence can concatenate cache runs, so
    // canonicalize once on the final block layout instead.
    simplify_cache_only_runs(program);
}

/// Replace a block parameter with a cached-local read when every incoming edge
/// obtains that parameter from the same cached local.
///
/// Typed loops often carry a value that is also assigned to a local with
/// `local.tee`. Keeping both the transient block parameter and the cached local
/// forces MachineIR to dedicate two registers to one value and shuffle between
/// them on every backedge. When all predecessors prove the two identities are
/// equal, define the former block-param value with `cell.get_cache` at the
/// target instead. The final cache-requirement pass will upgrade the target
/// cell from write-first Reserve to Ensure and the normal cache edge will carry
/// the value.
fn fold_block_params_sourced_from_cached_cells(program: &mut SsaProgram) -> bool {
    let mut cache_get_cell = collections::vec![None; program.value_types.len()];
    for block in &program.blocks {
        for inst in &block.ops {
            if inst.op != SsaOp::CELL_GET_CACHE || !inst.result.is_some() {
                continue;
            }
            if let Some(slot) = cache_get_cell.get_mut(inst.result.0 as usize) {
                *slot = Some(super::cell::CellId(inst.meta));
            }
        }
    }

    let incoming_by_block = all_incoming_edge_locations(program);
    let mut param_location = collections::vec![None; program.value_types.len()];
    for (block_index, block) in program.blocks.iter().enumerate() {
        for (param_index, param) in block.params.iter().copied().enumerate() {
            if let Some(slot) = param_location.get_mut(param.0 as usize) {
                *slot = Some((block_index, param_index));
            }
        }
    }
    let mut folds = collections::Vec::<(usize, SsaValue, super::cell::CellId)>::new();
    let mut common_cells = collections::Vec::new();
    let mut seen_count = collections::Vec::new();
    let mut valid = collections::Vec::new();
    for block_index in 0..program.blocks.len() {
        if block_index == program.entry.as_usize() {
            continue;
        }
        let incoming = &incoming_by_block[block_index];
        if incoming.is_empty() {
            continue;
        }
        let params = &program.blocks[block_index].params;
        common_cells.clear();
        common_cells.resize(params.len(), None);
        seen_count.clear();
        seen_count.resize(params.len(), 0usize);
        valid.clear();
        valid.resize(params.len(), true);
        for &location in incoming {
            for binding in &edge_at(program, location).bindings {
                let Some((target_block, param_index)) = param_location
                    .get(binding.param.0 as usize)
                    .copied()
                    .flatten()
                else {
                    continue;
                };
                if target_block != block_index {
                    continue;
                }
                seen_count[param_index] += 1;
                let Some(source_cell) = cache_get_cell
                    .get(binding.value.0 as usize)
                    .copied()
                    .flatten()
                else {
                    valid[param_index] = false;
                    continue;
                };
                if common_cells[param_index].is_some_and(|candidate| candidate != source_cell) {
                    valid[param_index] = false;
                } else {
                    common_cells[param_index] = Some(source_cell);
                }
            }
        }
        for (param_index, &param) in params.iter().enumerate() {
            let Some(param_ty) = program.value_types.get(param.0 as usize).copied() else {
                continue;
            };
            if param_ty.is_ref() {
                continue;
            }
            if !valid[param_index] || seen_count[param_index] != incoming.len() {
                continue;
            }
            let Some(cell) = common_cells[param_index] else {
                continue;
            };
            if program.cell_types.get(cell.0 as usize).copied() != Some(param_ty)
                || !program
                    .block_entry_cached_cells
                    .get(block_index)
                    .is_some_and(|cells| cells.contains(&cell))
            {
                continue;
            }
            folds.push((block_index, param, cell));
        }
    }

    if folds.is_empty() {
        return false;
    }

    let mut folded_param = collections::vec![false; program.value_types.len()];
    for &(_, param, _) in &folds {
        if let Some(slot) = folded_param.get_mut(param.0 as usize) {
            *slot = true;
        }
    }

    let mut fold_index = 0;
    while fold_index < folds.len() {
        let block_index = folds[fold_index].0;
        let group_start = fold_index;
        while fold_index < folds.len() && folds[fold_index].0 == block_index {
            fold_index += 1;
        }
        program.blocks[block_index]
            .params
            .retain(|param| !folded_param.get(param.0 as usize).copied().unwrap_or(false));
        for &location in &incoming_by_block[block_index] {
            edge_at_mut(program, location).bindings.retain(|binding| {
                !folded_param
                    .get(binding.param.0 as usize)
                    .copied()
                    .unwrap_or(false)
            });
        }
        let old_ops = core::mem::take(&mut program.blocks[block_index].ops);
        let mut new_ops = collections::Vec::with_capacity(
            old_ops
                .len()
                .saturating_add(fold_index.saturating_sub(group_start)),
        );
        for &(_, param, cell) in &folds[group_start..fold_index] {
            new_ops.push(SsaInst::cell_get_cache(cell, param));
        }
        new_ops.extend(old_ops);
        program.blocks[block_index].ops = new_ops;
    }
    true
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
    fn apply(&mut self, op: SsaOp) {
        match op {
            SsaOp::CELL_DROP_CACHE => {
                self.needs_drop = true;
                self.present = None;
            }
            SsaOp::CELL_RESERVE_CACHE => {
                if self.present != Some(CachePresence::Ensured) {
                    self.present = Some(CachePresence::Reserved);
                }
            }
            SsaOp::CELL_ENSURE_CACHE => {
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
        let mut new_ops = collections::Vec::with_capacity(old_ops.len());
        let mut cache_run = collections::Vec::new();
        for inst in old_ops {
            if !is_cache_only_op(inst.op) {
                flush_cache_run(&mut new_ops, &mut cache_run);
                new_ops.push(inst);
                continue;
            }
            accumulate_cache_run_state(&mut cache_run, &inst);
        }
        flush_cache_run(&mut new_ops, &mut cache_run);

        block.ops = new_ops;
    }
}

#[cfg(test)]
fn simplify_cache_run(run: &[SsaInst]) -> collections::Vec<SsaInst> {
    let mut by_slot = collections::Vec::<(u16, CacheRunState)>::new();
    for inst in run {
        accumulate_cache_run_state(&mut by_slot, inst);
    }

    let mut out = collections::Vec::with_capacity(by_slot.len().saturating_mul(2));
    flush_cache_run(&mut out, &mut by_slot);
    out
}

fn accumulate_cache_run_state(
    by_slot: &mut collections::Vec<(u16, CacheRunState)>,
    inst: &SsaInst,
) {
    let slot = cache_run_slot(inst).expect("cache-only run should only contain cache ops");
    match by_slot
        .iter_mut()
        .find(|(slot_index, _)| *slot_index == slot.0)
    {
        Some((_, state)) => state.apply(inst.op),
        None => {
            let mut state = CacheRunState::default();
            state.apply(inst.op);
            by_slot.push((slot.0, state));
        }
    }
}

fn flush_cache_run(
    out: &mut collections::Vec<SsaInst>,
    by_slot: &mut collections::Vec<(u16, CacheRunState)>,
) {
    if by_slot.is_empty() {
        return;
    }
    by_slot.sort_unstable_by_key(|(slot_index, _)| *slot_index);
    for &(slot_index, state) in by_slot.iter() {
        let slot = super::cell::CellId(slot_index);
        if state.needs_drop {
            out.push(SsaInst::cell_drop_cache(slot));
        }
    }
    for &(slot_index, state) in by_slot.iter() {
        let slot = super::cell::CellId(slot_index);
        match state.present {
            Some(CachePresence::Ensured) => out.push(SsaInst::cell_ensure_cache(slot)),
            Some(CachePresence::Reserved) => out.push(SsaInst::cell_reserve_cache(slot)),
            None => {}
        }
    }
    by_slot.clear();
}

#[inline]
fn is_cache_only_op(op: SsaOp) -> bool {
    matches!(
        op,
        SsaOp::CELL_ENSURE_CACHE | SsaOp::CELL_RESERVE_CACHE | SsaOp::CELL_DROP_CACHE
    )
}

#[inline]
fn cache_run_slot(inst: &SsaInst) -> Option<super::cell::CellId> {
    match inst.op {
        SsaOp::CELL_ENSURE_CACHE | SsaOp::CELL_RESERVE_CACHE | SsaOp::CELL_DROP_CACHE => {
            Some(super::cell::CellId(inst.meta))
        }
        _ => None,
    }
}

fn thread_empty_goto_blocks(program: &mut SsaProgram) -> bool {
    let block_count = program.blocks.len();
    let mut incoming_by_block = all_incoming_edge_locations(program);
    let mut removed = collections::vec![false; block_count];
    let mut removed_indices = collections::Vec::new();
    let mut queued = collections::vec![true; block_count];
    let mut worklist = (0..block_count).collect::<collections::Vec<_>>();

    while let Some(block_index) = worklist.pop() {
        queued[block_index] = false;
        if removed[block_index] {
            continue;
        }
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
                .block_entry_cached_cells
                .get(target.as_usize())
                .map(|slots| !slots.is_empty())
                .unwrap_or(false)
            {
                continue;
            }
            if incoming_by_block[block_index]
                .iter()
                .any(|location| !removed[edge_location_block(*location)])
            {
                continue;
            }
            program.entry = target;
            removed[block_index] = true;
            removed_indices.push(block_index);
            queue_cleanup_block(target.as_usize(), &removed, &mut queued, &mut worklist);
            continue;
        }

        let incoming = core::mem::take(&mut incoming_by_block[block_index])
            .into_iter()
            .filter(|location| !removed[edge_location_block(*location)])
            .collect::<collections::Vec<_>>();
        if incoming.is_empty() {
            continue;
        }

        let mut composed = collections::Vec::with_capacity(incoming.len());
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
            incoming_by_block[block_index] = incoming;
            continue;
        }

        for (loc, new_edge) in incoming.into_iter().zip(composed.into_iter()) {
            *edge_at_mut(program, loc) = new_edge;
            incoming_by_block[target.as_usize()].push(loc);
            queue_cleanup_block(
                edge_location_block(loc),
                &removed,
                &mut queued,
                &mut worklist,
            );
        }
        removed[block_index] = true;
        removed_indices.push(block_index);
        queue_cleanup_block(target.as_usize(), &removed, &mut queued, &mut worklist);
    }

    if removed_indices.is_empty() {
        false
    } else {
        remove_blocks(program, &removed_indices);
        true
    }
}

fn queue_cleanup_block(
    block_index: usize,
    removed: &[bool],
    queued: &mut [bool],
    worklist: &mut collections::Vec<usize>,
) {
    if block_index < queued.len() && !removed[block_index] && !queued[block_index] {
        queued[block_index] = true;
        worklist.push(block_index);
    }
}

fn edge_location_block(location: EdgeLocation) -> usize {
    match location {
        EdgeLocation::Goto { block }
        | EdgeLocation::BranchThen { block }
        | EdgeLocation::BranchElse { block }
        | EdgeLocation::BrTable { block, .. } => block,
    }
}

fn merge_goto_successors(program: &mut SsaProgram) -> bool {
    let mut removed = collections::Vec::new();
    let predecessor_counts = predecessor_counts(program);
    for pred_index in 0..program.blocks.len() {
        while let Some(succ_index) = merge_goto_successor(program, pred_index, &predecessor_counts)
        {
            removed.push(succ_index);
        }
    }

    if removed.is_empty() {
        false
    } else {
        remove_blocks(program, &removed);
        true
    }
}

fn merge_goto_successor(
    program: &mut SsaProgram,
    pred_index: usize,
    predecessor_counts: &[usize],
) -> Option<usize> {
    let SsaTerminator::Goto(edge) = &program.blocks[pred_index].terminator else {
        return None;
    };
    let succ_index = edge.target.as_usize();
    if succ_index == pred_index || succ_index >= program.blocks.len() {
        return None;
    }
    if succ_index == program.entry.as_usize() {
        return None;
    }
    if predecessor_counts[succ_index] != 1 {
        return None;
    }

    // Boundary-repair blocks emitted by `rewrite/edge.rs` contain only
    // cache-only ops (CellEnsureCache / CellReserveCache / CellDropCache).
    // They exist precisely to give the machine layer a fresh `cache_bindings`
    // slate when crossing a region boundary: each new block starts with all
    // cache bindings = None. Merging such a successor into its predecessor
    // makes the merged block need simultaneous bindings for
    // `pred_exit_cache ∪ succ_ensure_slots`, which can exceed the configured
    // dynamic-lane budget once entry register params and transient values are
    // accounted for. Keeping the repair block separate is the middle-layer
    // mechanism that preserves the budget proof handed to machine lowering.
    let succ_ops = &program.blocks[succ_index].ops;
    if !succ_ops.is_empty() && succ_ops.iter().all(|inst| is_cache_only_op(inst.op)) {
        return None;
    }

    let subst = binding_substitution(&program.blocks[succ_index].params, &edge.bindings)?;

    // The merged block's extra_args is concatenated
    // `[pred.extra_args ++ succ.extra_args]`. `SsaInst.meta` indexes
    // extra_args with a u16, so the combined length must fit in u16. If it
    // wouldn't, skip this merge rather than wrap indices and miscompile
    // primitive overflow-operand ranges.
    let combined_extra = program.blocks[pred_index]
        .extra_args
        .len()
        .checked_add(program.blocks[succ_index].extra_args.len())?;
    if combined_extra > u16::MAX as usize {
        return None;
    }

    let succ = core::mem::replace(
        &mut program.blocks[succ_index],
        SsaBlock {
            id: SsaTarget(u32::MAX),
            params: collections::Vec::new(),
            ops: collections::Vec::new(),
            extra_args: collections::Vec::new(),
            terminator: SsaTerminator::TrapUnreachable,
        },
    );

    // Precompute whether each succ op uses overflow operands while the
    // primitive pool is borrowable. We need this when rebasing `extra_args`
    // indices: only primitives with more than two operands read `meta` as an
    // extra_args start index.
    let has_extra_args: collections::Vec<bool> = succ
        .ops
        .iter()
        .map(|inst| {
            inst.op.as_primitive_idx().is_some_and(|idx| {
                crate::vm::wasm::primitive_op::stack_effect(&program.primitive_pool[idx as usize]).0
                    > 2
            })
        })
        .collect();

    // Substitute extra_args in-place, and remember how succ's indices map into
    // the appended region of pred.extra_args.
    let mut succ_extra = succ.extra_args;
    for operand in succ_extra.iter_mut() {
        *operand = substitute_operand(*operand, &subst);
    }

    let pred = &mut program.blocks[pred_index];
    let extra_args_base = pred.extra_args.len() as u16;
    pred.extra_args.extend(succ_extra);

    let merged_ops = succ
        .ops
        .into_iter()
        .zip(has_extra_args)
        .map(|(inst, has_extra_args)| {
            substitute_inst(inst, &subst, extra_args_base, has_extra_args)
        })
        .collect::<collections::Vec<_>>();
    let merged_terminator = substitute_terminator(succ.terminator, &subst);

    pred.ops.extend(merged_ops);
    pred.terminator = merged_terminator;
    Some(succ_index)
}

fn remove_unreachable_blocks(program: &mut SsaProgram) -> bool {
    if program.blocks.is_empty() {
        return false;
    }

    let mut reachable = collections::vec![false; program.blocks.len()];
    let mut stack = collections::vec![program.entry.as_usize()];
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
        .collect::<collections::Vec<_>>();
    if removed.is_empty() {
        return false;
    }
    remove_blocks(program, &removed);
    true
}

fn predecessor_counts(program: &SsaProgram) -> collections::Vec<usize> {
    let mut counts = collections::vec![0usize; program.blocks.len()];
    for block in &program.blocks {
        visit_outgoing_edges(&block.terminator, |edge| {
            if let Some(count) = counts.get_mut(edge.target.as_usize()) {
                *count += 1;
            }
        });
    }
    counts
}

fn all_incoming_edge_locations(
    program: &SsaProgram,
) -> collections::Vec<collections::Vec<EdgeLocation>> {
    let mut incoming = collections::vec![collections::Vec::new(); program.blocks.len()];
    for (block_index, block) in program.blocks.iter().enumerate() {
        let mut push = |target: SsaTarget, location| {
            if let Some(edges) = incoming.get_mut(target.as_usize()) {
                edges.push(location);
            }
        };
        match &block.terminator {
            SsaTerminator::Goto(edge) => {
                push(edge.target, EdgeLocation::Goto { block: block_index });
            }
            SsaTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                push(
                    then_edge.target,
                    EdgeLocation::BranchThen { block: block_index },
                );
                push(
                    else_edge.target,
                    EdgeLocation::BranchElse { block: block_index },
                );
            }
            SsaTerminator::BrTable { entries, .. } => {
                for (entry_index, edge) in entries.iter().enumerate() {
                    push(
                        edge.target,
                        EdgeLocation::BrTable {
                            block: block_index,
                            entry: entry_index,
                        },
                    );
                }
            }
            SsaTerminator::Return { .. }
            | SsaTerminator::ReturnScalar { .. }
            | SsaTerminator::TailCallDirect { .. }
            | SsaTerminator::TailCallIndirect { .. }
            | SsaTerminator::TailCallRef { .. }
            | SsaTerminator::TrapUnreachable
            | SsaTerminator::EhThrow { .. }
            | SsaTerminator::EhThrowRef { .. } => {}
        }
    }
    incoming
}

fn compose_edge(in_edge: &SsaEdge, params: &[SsaValue], out_edge: &SsaEdge) -> Option<SsaEdge> {
    let mut bindings = collections::Vec::with_capacity(out_edge.bindings.len());
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
) -> Option<collections::Vec<ValueSubstitution>> {
    let mut subst = collections::Vec::with_capacity(params.len());
    for &param in params {
        subst.push(ValueSubstitution {
            from: param,
            to: find_binding_value(bindings, param)?,
        });
    }
    Some(subst)
}

/// Rewrite the value references in an instruction according to `subst`, and
/// for primitives with overflow operands shift the extra-args index by
/// `extra_args_base`.
///
/// `has_extra_args` must be `true` exactly when `inst.op` is a primitive op
/// whose `stack_effect(...).0 > 2`; for everything else (non-primitives and
/// 0/1/2-arg primitives) `meta` is left untouched.
///
/// Callers must ensure the merged block's total extra_args length stays
/// within u16; `merge_one_goto_successor` pre-checks that.
fn substitute_inst(
    mut inst: SsaInst,
    subst: &[ValueSubstitution],
    extra_args_base: u16,
    has_extra_args: bool,
) -> SsaInst {
    if inst.op.is_primitive() {
        if has_extra_args {
            // Pre-checked by the caller: combined extra_args len <= u16::MAX.
            // `inst.meta` is a valid index into the source block's extra_args
            // (< succ.extra_args.len()); shifting by `extra_args_base` (which
            // is `pred.extra_args.len()` before the extend) keeps the sum
            // below u16::MAX.
            inst.meta = inst
                .meta
                .checked_add(extra_args_base)
                .expect("extra_args index overflow: merge_one_goto_successor should have rejected this merge");
        }
        inst.args = [
            substitute_operand(inst.args[0], subst),
            substitute_operand(inst.args[1], subst),
        ];
        return inst;
    }

    match inst.op {
        SsaOp::SPILL | SsaOp::CELL_SET_SLOT | SsaOp::CELL_SET_CACHE => {
            inst.args[0] = substitute_operand(inst.args[0], subst);
        }
        SsaOp::FILL
        | SsaOp::CELL_GET_SLOT
        | SsaOp::CELL_GET_CACHE
        | SsaOp::CELL_ENSURE_CACHE
        | SsaOp::CELL_RESERVE_CACHE
        | SsaOp::CELL_DROP_CACHE
        | SsaOp::CALL => {}
        _ => {}
    }
    inst
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
        SsaTerminator::ReturnScalar { result, results } => SsaTerminator::ReturnScalar {
            result: substitute_scalar_result(result, subst),
            results,
        },
        SsaTerminator::TailCallDirect {
            callee,
            args,
            return_results,
        } => SsaTerminator::TailCallDirect {
            callee,
            args,
            return_results,
        },
        SsaTerminator::TailCallIndirect {
            type_idx,
            table_idx,
            index,
            args,
            return_results,
        } => SsaTerminator::TailCallIndirect {
            type_idx,
            table_idx,
            index,
            args,
            return_results,
        },
        SsaTerminator::TailCallRef {
            type_idx,
            callee_ref,
            args,
            return_results,
        } => SsaTerminator::TailCallRef {
            type_idx,
            callee_ref,
            args,
            return_results,
        },
        SsaTerminator::TrapUnreachable => SsaTerminator::TrapUnreachable,
        SsaTerminator::EhThrow { tag_idx, args } => SsaTerminator::EhThrow { tag_idx, args },
        SsaTerminator::EhThrowRef { exnref_slot } => SsaTerminator::EhThrowRef { exnref_slot },
    }
}

fn substitute_scalar_result(
    result: SsaScalarResultLoc,
    subst: &[ValueSubstitution],
) -> SsaScalarResultLoc {
    match result {
        SsaScalarResultLoc::Stack { slot, ty } => SsaScalarResultLoc::Stack { slot, ty },
        SsaScalarResultLoc::Live { value, ty, slot } => SsaScalarResultLoc::Live {
            value: substitute_value(value, subst),
            ty,
            slot,
        },
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

#[inline]
fn substitute_operand(operand: SsaOperand, subst: &[ValueSubstitution]) -> SsaOperand {
    match operand.as_value() {
        Some(value) => {
            let substituted = substitute_value(value, subst);
            if substituted == value {
                operand
            } else {
                SsaOperand::value(substituted)
            }
        }
        None => operand,
    }
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
    let mut removed_mask = collections::vec![false; block_len];
    for &index in removed {
        debug_assert!(index < block_len, "removed block index out of range");
        if let Some(slot) = removed_mask.get_mut(index) {
            *slot = true;
        }
    }

    let mut mapping = collections::vec![SsaTarget::default(); block_len];
    let mut next = 0u32;
    for (old_index, is_removed) in removed_mask.iter().copied().enumerate() {
        if is_removed {
            continue;
        }
        mapping[old_index] = SsaTarget(next);
        next += 1;
    }

    let old_blocks = core::mem::take(&mut program.blocks);
    let old_entries = core::mem::take(&mut program.block_entry_cached_cells);
    let old_entry_reqs = core::mem::take(&mut program.block_entry_cache_requirements);
    debug_assert_eq!(old_blocks.len(), old_entries.len());
    debug_assert!(old_entry_reqs.is_empty() || old_entry_reqs.len() == old_blocks.len());
    let kept_blocks = block_len.saturating_sub(removed.len());
    let mut new_blocks = collections::Vec::with_capacity(kept_blocks);
    let mut new_entries =
        collections::Vec::with_capacity(old_entries.len().saturating_sub(removed.len()));
    let mut new_entry_reqs = if old_entry_reqs.is_empty() {
        collections::Vec::new()
    } else {
        collections::Vec::with_capacity(old_entry_reqs.len().saturating_sub(removed.len()))
    };

    if old_entry_reqs.is_empty() {
        for (old_index, (mut block, entry_slots)) in
            old_blocks.into_iter().zip(old_entries).enumerate()
        {
            if removed_mask[old_index] {
                continue;
            }
            remap_terminator_targets(&mut block.terminator, &mapping);
            block.id = mapping[old_index];
            new_blocks.push(block);
            new_entries.push(entry_slots);
        }
    } else {
        for (old_index, ((mut block, entry_slots), entry_reqs)) in old_blocks
            .into_iter()
            .zip(old_entries)
            .zip(old_entry_reqs)
            .enumerate()
        {
            if removed_mask[old_index] {
                continue;
            }
            remap_terminator_targets(&mut block.terminator, &mapping);
            block.id = mapping[old_index];
            new_blocks.push(block);
            new_entries.push(entry_slots);
            new_entry_reqs.push(entry_reqs);
        }
    }

    debug_assert!(removed_mask.get(program.entry.as_usize()).copied() != Some(true));
    program.entry = mapping[program.entry.as_usize()];
    program.blocks = new_blocks;
    program.block_entry_cached_cells = new_entries;
    program.block_entry_cache_requirements = new_entry_reqs;
}

fn remove_one_block(program: &mut SsaProgram, removed_index: usize) {
    debug_assert!(
        removed_index < program.blocks.len(),
        "removed block index out of range"
    );
    debug_assert_ne!(program.entry.as_usize(), removed_index);

    program.blocks.remove(removed_index);
    program.block_entry_cached_cells.remove(removed_index);
    if !program.block_entry_cache_requirements.is_empty() {
        program.block_entry_cache_requirements.remove(removed_index);
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
        SsaTerminator::Return { .. }
        | SsaTerminator::ReturnScalar { .. }
        | SsaTerminator::TailCallDirect { .. }
        | SsaTerminator::TailCallIndirect { .. }
        | SsaTerminator::TailCallRef { .. }
        | SsaTerminator::TrapUnreachable
        | SsaTerminator::EhThrow { .. }
        | SsaTerminator::EhThrowRef { .. } => {}
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
        SsaTerminator::Return { .. }
        | SsaTerminator::ReturnScalar { .. }
        | SsaTerminator::TailCallDirect { .. }
        | SsaTerminator::TailCallIndirect { .. }
        | SsaTerminator::TailCallRef { .. }
        | SsaTerminator::TrapUnreachable
        | SsaTerminator::EhThrow { .. }
        | SsaTerminator::EhThrowRef { .. } => {}
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
        SsaTerminator::Return { .. }
        | SsaTerminator::ReturnScalar { .. }
        | SsaTerminator::TailCallDirect { .. }
        | SsaTerminator::TailCallIndirect { .. }
        | SsaTerminator::TailCallRef { .. }
        | SsaTerminator::TrapUnreachable
        | SsaTerminator::EhThrow { .. }
        | SsaTerminator::EhThrowRef { .. } => {}
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
    use crate::value_type::ValueType;
    use crate::vm::middle::{
        cell::CellId,
        ssa_ir::{
            ir::{EntryCacheRequirement, SsaBlock},
            validate::validate_program,
        },
    };
    use crate::vm::wasm::primitive_op::PrimitiveOpKind;

    fn value_inst(program: &mut SsaProgram, result: u32) -> SsaInst {
        let pool_idx = program
            .intern_primitive(PrimitiveOpKind::I32Const { value: 1 })
            .unwrap();
        SsaInst::primitive(
            pool_idx,
            SsaValue(result),
            [SsaOperand::NONE, SsaOperand::NONE],
            0,
        )
    }

    #[test]
    fn simplify_cache_run_keeps_required_drop_then_materialization() {
        let run = collections::vec![
            SsaInst::cell_reserve_cache(CellId(0)),
            SsaInst::cell_drop_cache(CellId(0)),
            SsaInst::cell_ensure_cache(CellId(0)),
            SsaInst::cell_ensure_cache(CellId(0)),
        ];
        let simplified = simplify_cache_run(&run);
        assert_eq!(
            simplified
                .iter()
                .map(|inst| inst.op)
                .collect::<collections::Vec<_>>(),
            collections::vec![SsaOp::CELL_DROP_CACHE, SsaOp::CELL_ENSURE_CACHE]
        );
        assert!(simplified.iter().all(|inst| inst.meta == 0));
    }

    #[test]
    fn folds_loop_param_sourced_from_same_cached_cell_on_every_edge() {
        let param = SsaValue(10);
        let mut program = SsaProgram {
            cell_homes: collections::Vec::new(),
            entry: SsaTarget(0),
            blocks: collections::vec![
                SsaBlock {
                    id: SsaTarget(0),
                    params: collections::Vec::new(),
                    ops: collections::vec![SsaInst::cell_get_cache(CellId(0), SsaValue(0))],
                    extra_args: collections::Vec::new(),
                    terminator: SsaTerminator::Goto(SsaEdge {
                        target: SsaTarget(1),
                        bindings: collections::vec![SsaBinding {
                            param,
                            value: SsaValue(0),
                        }],
                    }),
                },
                SsaBlock {
                    id: SsaTarget(1),
                    params: collections::vec![param],
                    ops: collections::vec![SsaInst::cell_get_cache(CellId(0), SsaValue(1))],
                    extra_args: collections::Vec::new(),
                    terminator: SsaTerminator::Branch {
                        cond: param,
                        then_edge: SsaEdge {
                            target: SsaTarget(1),
                            bindings: collections::vec![SsaBinding {
                                param,
                                value: SsaValue(1),
                            }],
                        },
                        else_edge: SsaEdge {
                            target: SsaTarget(2),
                            bindings: collections::Vec::new(),
                        },
                    },
                },
                SsaBlock {
                    id: SsaTarget(2),
                    params: collections::Vec::new(),
                    ops: collections::Vec::new(),
                    extra_args: collections::Vec::new(),
                    terminator: SsaTerminator::Return { results: None },
                },
            ],
            cell_types: collections::vec![ValueType::I32],
            result_types: collections::Vec::new(),
            cell_info: collections::vec![Default::default()],
            block_entry_cached_cells: collections::vec![
                collections::vec![CellId(0)],
                collections::vec![CellId(0)],
                collections::Vec::new(),
            ],
            block_entry_cache_requirements: collections::vec![
                collections::vec![EntryCacheRequirement::Ensure],
                collections::vec![EntryCacheRequirement::Ensure],
                collections::Vec::new(),
            ],
            preferred_preserved: collections::Vec::new(),
            value_types: collections::vec![ValueType::I32; 11],
            value_sink_cell: collections::vec![None; 11],
            const_pool: collections::Vec::new(),
            primitive_pool: collections::Vec::new(),
            call_ops: collections::Vec::new(),
        };

        assert!(fold_block_params_sourced_from_cached_cells(&mut program));
        let loop_block = &program.blocks[1];
        assert!(loop_block.params.is_empty());
        assert_eq!(loop_block.ops[0].op, SsaOp::CELL_GET_CACHE);
        assert_eq!(loop_block.ops[0].meta, 0);
        assert_eq!(loop_block.ops[0].result, param);
        assert!(matches!(
            &program.blocks[0].terminator,
            SsaTerminator::Goto(edge) if edge.bindings.is_empty()
        ));
        assert!(matches!(
            &loop_block.terminator,
            SsaTerminator::Branch { then_edge, .. } if then_edge.bindings.is_empty()
        ));
        validate_program(&program).unwrap();
    }

    #[test]
    fn threads_empty_goto_chain_and_composes_bindings() {
        let mut program = SsaProgram {
            cell_homes: collections::Vec::new(),
            entry: SsaTarget(0),
            blocks: collections::vec![
                SsaBlock {
                    id: SsaTarget(0),
                    params: collections::Vec::new(),
                    ops: collections::Vec::new(),
                    extra_args: collections::Vec::new(),
                    terminator: SsaTerminator::Goto(SsaEdge {
                        target: SsaTarget(1),
                        bindings: collections::vec![SsaBinding {
                            param: SsaValue(10),
                            value: SsaValue(0),
                        }],
                    }),
                },
                SsaBlock {
                    id: SsaTarget(1),
                    params: collections::vec![SsaValue(10)],
                    ops: collections::Vec::new(),
                    extra_args: collections::Vec::new(),
                    terminator: SsaTerminator::Goto(SsaEdge {
                        target: SsaTarget(2),
                        bindings: collections::vec![SsaBinding {
                            param: SsaValue(20),
                            value: SsaValue(10),
                        }],
                    }),
                },
                SsaBlock {
                    id: SsaTarget(2),
                    params: collections::vec![SsaValue(20)],
                    ops: collections::Vec::new(),
                    extra_args: collections::Vec::new(),
                    terminator: SsaTerminator::Goto(SsaEdge {
                        target: SsaTarget(3),
                        bindings: collections::vec![SsaBinding {
                            param: SsaValue(30),
                            value: SsaValue(20),
                        }],
                    }),
                },
                SsaBlock {
                    id: SsaTarget(3),
                    params: collections::vec![SsaValue(30)],
                    ops: collections::Vec::new(),
                    extra_args: collections::Vec::new(),
                    terminator: SsaTerminator::Return { results: None },
                },
            ],
            cell_types: collections::Vec::new(),
            result_types: collections::Vec::new(),
            cell_info: collections::Vec::new(),
            block_entry_cached_cells: collections::vec![
                collections::Vec::new(),
                collections::Vec::new(),
                collections::Vec::new(),
                collections::Vec::new()
            ],
            block_entry_cache_requirements: collections::vec![collections::Vec::new(); 4],
            preferred_preserved: collections::Vec::new(),
            value_types: collections::vec![ValueType::I32; 32],
            value_sink_cell: collections::vec![None; 32],
            const_pool: collections::Vec::new(),
            primitive_pool: collections::Vec::new(),
            call_ops: collections::Vec::new(),
        };

        assert!(thread_empty_goto_blocks(&mut program));
        assert_eq!(program.blocks.len(), 2);
        let edge = match &program.blocks[0].terminator {
            SsaTerminator::Goto(edge) => edge,
            other => panic!("expected goto after threading, got {other:?}"),
        };
        assert_eq!(edge.target, SsaTarget(1));
        assert_eq!(
            edge.bindings,
            collections::vec![SsaBinding {
                param: SsaValue(30),
                value: SsaValue(0),
            }]
        );
        validate_program(&program).unwrap();
    }

    #[test]
    fn merges_goto_successor_with_param_substitution() {
        let mut program = SsaProgram {
            cell_homes: collections::Vec::new(),
            entry: SsaTarget(0),
            blocks: collections::Vec::new(),
            cell_types: collections::vec![ValueType::I32],
            result_types: collections::Vec::new(),
            cell_info: collections::vec![Default::default()],
            block_entry_cached_cells: collections::vec![
                collections::Vec::new(),
                collections::Vec::new()
            ],
            block_entry_cache_requirements: collections::vec![collections::Vec::new(); 2],
            preferred_preserved: collections::Vec::new(),
            value_types: collections::vec![ValueType::I32; 16],
            value_sink_cell: collections::vec![None; 16],
            const_pool: collections::Vec::new(),
            primitive_pool: collections::Vec::new(),
            call_ops: collections::Vec::new(),
        };
        let vi = value_inst(&mut program, 0);
        program.blocks = collections::vec![
            SsaBlock {
                id: SsaTarget(0),
                params: collections::Vec::new(),
                ops: collections::vec![vi],
                extra_args: collections::Vec::new(),
                terminator: SsaTerminator::Goto(SsaEdge {
                    target: SsaTarget(1),
                    bindings: collections::vec![SsaBinding {
                        param: SsaValue(10),
                        value: SsaValue(0),
                    }],
                }),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: collections::vec![SsaValue(10)],
                ops: collections::vec![SsaInst::cell_set_cache(CellId(0), SsaValue(10))],
                extra_args: collections::Vec::new(),
                terminator: SsaTerminator::Return { results: None },
            },
        ];

        assert!(merge_goto_successors(&mut program));
        assert_eq!(program.blocks.len(), 1);
        assert_eq!(program.blocks[0].ops.len(), 2);
        let set_cache = &program.blocks[0].ops[1];
        assert_eq!(set_cache.op, SsaOp::CELL_SET_CACHE);
        assert_eq!(set_cache.args[0].as_value(), Some(SsaValue(0)));
        assert!(matches!(
            program.blocks[0].terminator,
            SsaTerminator::Return { .. }
        ));
        validate_program(&program).unwrap();
    }

    #[test]
    fn merges_entire_goto_chain_before_reindexing() {
        let mut program = SsaProgram {
            cell_homes: collections::Vec::new(),
            entry: SsaTarget(0),
            blocks: collections::Vec::new(),
            cell_types: collections::Vec::new(),
            result_types: collections::Vec::new(),
            cell_info: collections::Vec::new(),
            block_entry_cached_cells: collections::vec![collections::Vec::new(); 3],
            // Rewrite intentionally leaves this final-SSA side table absent.
            block_entry_cache_requirements: collections::Vec::new(),
            preferred_preserved: collections::Vec::new(),
            value_types: collections::vec![ValueType::I32; 3],
            value_sink_cell: collections::vec![None; 3],
            const_pool: collections::Vec::new(),
            primitive_pool: collections::Vec::new(),
            call_ops: collections::Vec::new(),
        };
        let vi0 = value_inst(&mut program, 0);
        let vi1 = value_inst(&mut program, 1);
        let vi2 = value_inst(&mut program, 2);
        program.blocks = collections::vec![
            SsaBlock {
                id: SsaTarget(0),
                params: collections::Vec::new(),
                ops: collections::vec![vi0],
                extra_args: collections::Vec::new(),
                terminator: SsaTerminator::Goto(SsaEdge {
                    target: SsaTarget(1),
                    bindings: collections::Vec::new(),
                }),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: collections::Vec::new(),
                ops: collections::vec![vi1],
                extra_args: collections::Vec::new(),
                terminator: SsaTerminator::Goto(SsaEdge {
                    target: SsaTarget(2),
                    bindings: collections::Vec::new(),
                }),
            },
            SsaBlock {
                id: SsaTarget(2),
                params: collections::Vec::new(),
                ops: collections::vec![vi2],
                extra_args: collections::Vec::new(),
                terminator: SsaTerminator::Return { results: None },
            },
        ];

        assert!(merge_goto_successors(&mut program));
        assert_eq!(program.blocks.len(), 1);
        assert_eq!(program.blocks[0].ops.len(), 3);
        assert!(matches!(
            program.blocks[0].terminator,
            SsaTerminator::Return { .. }
        ));
        validate_program(&program).unwrap();
    }

    #[test]
    fn remove_unreachable_blocks_prunes_dead_chain() {
        let mut program = SsaProgram {
            cell_homes: collections::Vec::new(),
            entry: SsaTarget(0),
            blocks: collections::vec![
                SsaBlock {
                    id: SsaTarget(0),
                    params: collections::Vec::new(),
                    ops: collections::Vec::new(),
                    extra_args: collections::Vec::new(),
                    terminator: SsaTerminator::Return { results: None },
                },
                SsaBlock {
                    id: SsaTarget(1),
                    params: collections::Vec::new(),
                    ops: collections::Vec::new(),
                    extra_args: collections::Vec::new(),
                    terminator: SsaTerminator::Goto(SsaEdge {
                        target: SsaTarget(2),
                        bindings: collections::Vec::new(),
                    }),
                },
                SsaBlock {
                    id: SsaTarget(2),
                    params: collections::Vec::new(),
                    ops: collections::Vec::new(),
                    extra_args: collections::Vec::new(),
                    terminator: SsaTerminator::Return { results: None },
                },
            ],
            cell_types: collections::Vec::new(),
            result_types: collections::Vec::new(),
            cell_info: collections::Vec::new(),
            block_entry_cached_cells: collections::vec![
                collections::Vec::new(),
                collections::Vec::new(),
                collections::Vec::new()
            ],
            // Rewrite intentionally leaves this final-SSA side table absent.
            block_entry_cache_requirements: collections::Vec::new(),
            preferred_preserved: collections::Vec::new(),
            value_types: collections::Vec::new(),
            value_sink_cell: collections::Vec::new(),
            const_pool: collections::Vec::new(),
            primitive_pool: collections::Vec::new(),
            call_ops: collections::Vec::new(),
        };

        assert!(remove_unreachable_blocks(&mut program));
        assert_eq!(program.blocks.len(), 1);
        assert_eq!(program.entry, SsaTarget(0));
        assert!(program.block_entry_cache_requirements.is_empty());
    }

    #[test]
    fn does_not_merge_unreachable_predecessor_into_entry() {
        let mut program = SsaProgram {
            cell_homes: collections::Vec::new(),
            entry: SsaTarget(1),
            blocks: collections::Vec::new(),
            cell_types: collections::Vec::new(),
            result_types: collections::Vec::new(),
            cell_info: collections::Vec::new(),
            block_entry_cached_cells: collections::vec![
                collections::Vec::new(),
                collections::Vec::new()
            ],
            block_entry_cache_requirements: collections::vec![collections::Vec::new(); 2],
            preferred_preserved: collections::Vec::new(),
            value_types: collections::vec![ValueType::I32; 2],
            value_sink_cell: collections::vec![None; 2],
            const_pool: collections::Vec::new(),
            primitive_pool: collections::Vec::new(),
            call_ops: collections::Vec::new(),
        };
        let vi0 = value_inst(&mut program, 0);
        let vi1 = value_inst(&mut program, 1);
        program.blocks = collections::vec![
            SsaBlock {
                id: SsaTarget(0),
                params: collections::Vec::new(),
                ops: collections::vec![vi0],
                extra_args: collections::Vec::new(),
                terminator: SsaTerminator::Goto(SsaEdge {
                    target: SsaTarget(1),
                    bindings: collections::Vec::new(),
                }),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: collections::Vec::new(),
                ops: collections::vec![vi1],
                extra_args: collections::Vec::new(),
                terminator: SsaTerminator::Return { results: None },
            },
        ];

        assert!(!merge_goto_successors(&mut program));
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
            cell_homes: collections::Vec::new(),
            entry: SsaTarget(2),
            blocks: collections::vec![
                SsaBlock {
                    id: SsaTarget(0),
                    params: collections::Vec::new(),
                    ops: collections::Vec::new(),
                    extra_args: collections::Vec::new(),
                    terminator: SsaTerminator::Goto(SsaEdge {
                        target: SsaTarget(2),
                        bindings: collections::Vec::new(),
                    }),
                },
                SsaBlock {
                    id: SsaTarget(1),
                    params: collections::Vec::new(),
                    ops: collections::Vec::new(),
                    extra_args: collections::Vec::new(),
                    terminator: SsaTerminator::Return { results: None },
                },
                SsaBlock {
                    id: SsaTarget(2),
                    params: collections::Vec::new(),
                    ops: collections::Vec::new(),
                    extra_args: collections::Vec::new(),
                    terminator: SsaTerminator::Goto(SsaEdge {
                        target: SsaTarget(3),
                        bindings: collections::Vec::new(),
                    }),
                },
                SsaBlock {
                    id: SsaTarget(3),
                    params: collections::Vec::new(),
                    ops: collections::Vec::new(),
                    extra_args: collections::Vec::new(),
                    terminator: SsaTerminator::Return { results: None },
                },
            ],
            cell_types: collections::Vec::new(),
            result_types: collections::Vec::new(),
            cell_info: collections::Vec::new(),
            block_entry_cached_cells: collections::vec![
                collections::vec![CellId(0)],
                collections::vec![CellId(1)],
                collections::vec![CellId(2)],
                collections::vec![CellId(3)],
            ],
            block_entry_cache_requirements: collections::vec![
                collections::vec![EntryCacheRequirement::Ensure],
                collections::vec![EntryCacheRequirement::Ensure],
                collections::vec![EntryCacheRequirement::Ensure],
                collections::vec![EntryCacheRequirement::Ensure],
            ],
            preferred_preserved: collections::Vec::new(),
            value_types: collections::Vec::new(),
            value_sink_cell: collections::Vec::new(),
            const_pool: collections::Vec::new(),
            primitive_pool: collections::Vec::new(),
            call_ops: collections::Vec::new(),
        };

        remove_blocks(&mut program, &[1]);

        assert_eq!(program.entry, SsaTarget(1));
        assert_eq!(program.blocks.len(), 3);
        assert_eq!(
            program
                .blocks
                .iter()
                .map(|block| block.id)
                .collect::<collections::Vec<_>>(),
            collections::vec![SsaTarget(0), SsaTarget(1), SsaTarget(2)]
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
            program.block_entry_cached_cells,
            collections::vec![
                collections::vec![CellId(0)],
                collections::vec![CellId(2)],
                collections::vec![CellId(3)],
            ]
        );
        validate_program(&program).unwrap();
    }
}
