//! CFG simplification: jump threading and block merging.
//!
//! Two passes run in sequence:
//! 1. **Jump threading** — redirects edges that target trivial (empty goto)
//!    blocks straight to the final destination, composing bindings along the
//!    chain.
//! 2. **Single-predecessor merging** — when a block unconditionally gotos a
//!    successor that has exactly one predecessor, the two blocks are fused.
//!
//! After each pass, dead blocks are removed and the remaining blocks are
//! renumbered so the invariant `blocks[i].id == SsaTarget(i)` holds.
//!
//! ## Entry block invariant
//!
//! `program.entry` is never changed.  Machine lowering (`lower_context.rs`)
//! treats the original entry block as the semantic function entry. With the
//! explicit-cache pipeline, cache state starts empty at block entry and is
//! rebuilt only by explicit SSA ops. Redirecting `program.entry` into a loop
//! body would still violate entry semantics by skipping the original frontend
//! prologue shape and reinterpreting loop entry as function entry.

use alloc::vec;
use alloc::vec::Vec;

use crate::vm::middle::ssa_ir::{
    ir::{
        SsaBinding, SsaBlock, SsaEdge, SsaInstKind, SsaOperand, SsaProgram, SsaTerminator, SsaValue,
    },
    target::SsaTarget,
};

// ---------------------------------------------------------------------------
// Public entry
// ---------------------------------------------------------------------------

pub(super) fn simplify_cfg(program: &mut SsaProgram) {
    if program.blocks.len() <= 1 {
        return;
    }
    thread_jumps(program);
    merge_single_predecessor_blocks(program);
}

// ---------------------------------------------------------------------------
// Pass 1: Jump Threading
// ---------------------------------------------------------------------------

fn is_trivial_block(block: &SsaBlock) -> bool {
    block.ops.is_empty() && matches!(block.terminator, SsaTerminator::Goto(_))
}

/// Follows a chain of trivial blocks starting at `start`, returning the final
/// non-trivial target and the composed outgoing bindings (mapping from the
/// final target's params to values expressed in terms of `start`'s params).
///
/// Stops on: non-trivial blocks, self-loops (an intentional infinite loop in
/// the source program), and revisited blocks (a cycle of trivial blocks, also
/// an infinite loop).  A visited set plus a step bound guarantee termination.
fn resolve_trivial_chain(blocks: &[SsaBlock], start: SsaTarget) -> (SsaTarget, Vec<SsaBinding>) {
    let max_steps = blocks.len();
    let mut visited = vec![false; blocks.len()];

    let mut current = start;
    // Accumulated bindings: maps the *next* target's params to values in terms
    // of `start`'s params.  `None` means "identity so far" (we haven't
    // composed anything yet).
    let mut composed: Option<Vec<SsaBinding>> = None;

    for _ in 0..max_steps {
        let block = &blocks[current.as_usize()];
        if !is_trivial_block(block) {
            break;
        }

        let SsaTerminator::Goto(ref edge) = block.terminator else {
            unreachable!();
        };

        // Self-loop: this block is an intentional infinite loop.
        if edge.target == current {
            break;
        }

        // Cycle among trivial blocks (also an infinite loop).
        if visited[current.as_usize()] {
            break;
        }
        visited[current.as_usize()] = true;

        // Compose bindings through this step.
        composed = Some(match composed {
            None => {
                // First step: the outgoing bindings are already in terms of
                // `start`'s params (since current == start on the first step
                // and the block's goto bindings reference current's params).
                edge.bindings.clone()
            }
            Some(prev) => compose_bindings(&prev, &block.params, &edge.bindings),
        });

        current = edge.target;
    }

    (current, composed.unwrap_or_default())
}

/// Compose two binding steps.
///
/// `pred_bindings` maps an intermediate block's params to predecessor values.
/// `intermediate_params` are the intermediate block's params.
/// `outgoing_bindings` maps the final target's params to values that may
/// reference `intermediate_params`.
///
/// Returns bindings mapping the final target's params to predecessor values.
fn compose_bindings(
    pred_bindings: &[SsaBinding],
    intermediate_params: &[SsaValue],
    outgoing_bindings: &[SsaBinding],
) -> Vec<SsaBinding> {
    outgoing_bindings
        .iter()
        .map(|out_b| {
            let resolved_value = intermediate_params
                .iter()
                .position(|p| *p == out_b.value)
                .and_then(|idx| {
                    let param = intermediate_params[idx];
                    pred_bindings.iter().find(|pb| pb.param == param)
                })
                .map(|pb| pb.value)
                .unwrap_or(out_b.value);
            SsaBinding {
                param: out_b.param,
                value: resolved_value,
            }
        })
        .collect()
}

/// Redirect entry: final target, params of the trivial block (needed for
/// binding composition), and composed outgoing bindings.
struct Redirect {
    final_target: SsaTarget,
    intermediate_params: Vec<SsaValue>,
    composed_bindings: Vec<SsaBinding>,
}

/// Rewrite edges that target trivial blocks to skip them.
///
/// A trivial block has no ops and a `Goto` terminator.  For each such block
/// we follow the chain to the final non-trivial target, composing edge
/// bindings along the way.  Every edge in the program that pointed at a
/// trivial block is then rewritten to the final target.
///
/// The entry block itself may be trivial, but we never redirect
/// `program.entry` — see the module-level doc comment for why.
fn thread_jumps(program: &mut SsaProgram) {
    let block_count = program.blocks.len();

    // Build redirect table: for each trivial block, where it ultimately leads.
    let mut redirects: Vec<Option<Redirect>> = Vec::with_capacity(block_count);
    for i in 0..block_count {
        if is_trivial_block(&program.blocks[i]) {
            let (target, bindings) = resolve_trivial_chain(&program.blocks, SsaTarget(i as u32));
            if target != SsaTarget(i as u32) {
                redirects.push(Some(Redirect {
                    final_target: target,
                    intermediate_params: program.blocks[i].params.clone(),
                    composed_bindings: bindings,
                }));
            } else {
                redirects.push(None);
            }
        } else {
            redirects.push(None);
        }
    }

    // Early exit if nothing to do.
    if redirects.iter().all(|r| r.is_none()) {
        return;
    }

    // Rewrite every edge in the program.
    for i in 0..block_count {
        rewrite_terminator_edges(&mut program.blocks[i].terminator, &redirects);
    }

    // Do NOT update program.entry here — see module doc comment.

    remove_dead_and_renumber(program);
}

fn rewrite_terminator_edges(term: &mut SsaTerminator, redirects: &[Option<Redirect>]) {
    match term {
        SsaTerminator::Goto(edge) => rewrite_edge(edge, redirects),
        SsaTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            rewrite_edge(then_edge, redirects);
            rewrite_edge(else_edge, redirects);
        }
        SsaTerminator::BrTable { entries, .. } => {
            for edge in entries {
                rewrite_edge(edge, redirects);
            }
        }
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
    }
}

fn rewrite_edge(edge: &mut SsaEdge, redirects: &[Option<Redirect>]) {
    if let Some(Some(r)) = redirects.get(edge.target.as_usize()) {
        let new_bindings =
            compose_bindings(&edge.bindings, &r.intermediate_params, &r.composed_bindings);
        edge.target = r.final_target;
        edge.bindings = new_bindings;
    }
}

// ---------------------------------------------------------------------------
// Pass 2: Single-Predecessor Merging
// ---------------------------------------------------------------------------

fn build_predecessor_counts(program: &SsaProgram) -> Vec<u32> {
    let mut counts = vec![0u32; program.blocks.len()];
    for block in &program.blocks {
        let mut count_edge = |target: SsaTarget| {
            counts[target.as_usize()] += 1;
        };
        match &block.terminator {
            SsaTerminator::Goto(edge) => count_edge(edge.target),
            SsaTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                count_edge(then_edge.target);
                count_edge(else_edge.target);
            }
            SsaTerminator::BrTable { entries, .. } => {
                for edge in entries {
                    count_edge(edge.target);
                }
            }
            SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
        }
    }
    counts
}

/// Merge a block into its sole predecessor when the predecessor ends with
/// `Goto` to that block.  The successor's params are substituted with the
/// goto's binding values throughout its ops and terminator.
///
/// The entry block is never absorbed as a successor (it has no SSA
/// predecessors), but it *can* absorb its own successor.
fn merge_single_predecessor_blocks(program: &mut SsaProgram) {
    let mut ever_merged = false;
    let mut changed = true;
    while changed {
        changed = false;
        let pred_counts = build_predecessor_counts(program);

        for i in 0..program.blocks.len() {
            let (target_idx, bindings) = match &program.blocks[i].terminator {
                SsaTerminator::Goto(edge) => (edge.target.as_usize(), edge.bindings.clone()),
                _ => continue,
            };

            if target_idx == i {
                continue; // self-loop
            }
            if program.entry.as_usize() == target_idx {
                continue; // entry block cannot become a successor (see module doc)
            }
            if pred_counts[target_idx] != 1 {
                continue;
            }

            // Build value substitution: successor param -> predecessor value.
            let succ_params = program.blocks[target_idx].params.clone();
            let subst: Vec<(SsaValue, SsaValue)> = succ_params
                .iter()
                .filter_map(|param| {
                    bindings
                        .iter()
                        .find(|b| b.param == *param)
                        .map(|b| (*param, b.value))
                })
                .collect();

            // Take the successor's ops and terminator.
            let mut succ_ops = core::mem::take(&mut program.blocks[target_idx].ops);
            let mut succ_term = core::mem::replace(
                &mut program.blocks[target_idx].terminator,
                SsaTerminator::TrapUnreachable,
            );

            // Apply substitution.
            for inst in &mut succ_ops {
                substitute_inst_uses(&mut inst.kind, &subst);
            }
            substitute_terminator_uses(&mut succ_term, &subst);

            // Merge into predecessor.
            program.blocks[i].ops.append(&mut succ_ops);
            program.blocks[i].terminator = succ_term;

            changed = true;
            ever_merged = true;
            break; // restart — indices and pred counts are stale
        }
    }

    if ever_merged {
        remove_dead_and_renumber(program);
    }
}

fn substitute_value(v: SsaValue, subst: &[(SsaValue, SsaValue)]) -> SsaValue {
    for &(from, to) in subst {
        if v == from {
            return to;
        }
    }
    v
}

fn substitute_inst_uses(kind: &mut SsaInstKind, subst: &[(SsaValue, SsaValue)]) {
    match kind {
        SsaInstKind::Value { args, .. } => {
            for operand in args {
                if let SsaOperand::Value(v) = operand {
                    *v = substitute_value(*v, subst);
                }
            }
        }
        SsaInstKind::LocalSetSlot { src, .. }
        | SsaInstKind::LocalSetCache { src, .. }
        | SsaInstKind::Spill { src, .. } => {
            *src = substitute_value(*src, subst);
        }
        SsaInstKind::LocalGetSlot { .. }
        | SsaInstKind::LocalGetCache { .. }
        | SsaInstKind::Fill { .. }
        | SsaInstKind::LocalDropCache { .. }
        | SsaInstKind::Call(_) => {}
    }
}

fn substitute_terminator_uses(term: &mut SsaTerminator, subst: &[(SsaValue, SsaValue)]) {
    match term {
        SsaTerminator::Goto(edge) => substitute_edge_uses(edge, subst),
        SsaTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            *cond = substitute_value(*cond, subst);
            substitute_edge_uses(then_edge, subst);
            substitute_edge_uses(else_edge, subst);
        }
        SsaTerminator::BrTable { index, entries } => {
            *index = substitute_value(*index, subst);
            for edge in entries {
                substitute_edge_uses(edge, subst);
            }
        }
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
    }
}

fn substitute_edge_uses(edge: &mut SsaEdge, subst: &[(SsaValue, SsaValue)]) {
    for binding in &mut edge.bindings {
        binding.value = substitute_value(binding.value, subst);
    }
}

// ---------------------------------------------------------------------------
// Shared: dead block removal + renumbering
// ---------------------------------------------------------------------------

/// Remove unreachable blocks and renumber the survivors so that
/// `blocks[i].id == SsaTarget(i)`.  All downstream consumers (machine
/// lowering, interpreter finalizer, arch backend) use block IDs as direct
/// array indices, so this invariant must hold after any block removal.
fn remove_dead_and_renumber(program: &mut SsaProgram) {
    let block_count = program.blocks.len();
    if block_count == 0 {
        return;
    }

    // Reachability BFS from entry.
    let mut reachable = vec![false; block_count];
    let mut worklist = vec![program.entry.as_usize()];
    while let Some(idx) = worklist.pop() {
        if idx >= block_count || reachable[idx] {
            continue;
        }
        reachable[idx] = true;
        for_each_successor_target(&program.blocks[idx].terminator, |target| {
            let t = target.as_usize();
            if t < block_count && !reachable[t] {
                worklist.push(t);
            }
        });
    }

    let live_count = reachable.iter().filter(|&&r| r).count();
    if live_count == block_count {
        return;
    }

    // Build old→new mapping.
    let mut old_to_new = vec![SsaTarget(u32::MAX); block_count];
    let mut new_idx = 0u32;
    for (old_idx, &is_reachable) in reachable.iter().enumerate() {
        if is_reachable {
            old_to_new[old_idx] = SsaTarget(new_idx);
            new_idx += 1;
        }
    }

    // Rewrite entry.
    program.entry = old_to_new[program.entry.as_usize()];

    // Compact and rewrite.
    let mut new_blocks = Vec::with_capacity(live_count);
    for (old_idx, block) in program.blocks.drain(..).enumerate() {
        if !reachable[old_idx] {
            continue;
        }
        let mut block = block;
        block.id = old_to_new[old_idx];
        rewrite_terminator_targets(&mut block.terminator, &old_to_new);
        new_blocks.push(block);
    }
    program.blocks = new_blocks;
}

fn for_each_successor_target(term: &SsaTerminator, mut f: impl FnMut(SsaTarget)) {
    match term {
        SsaTerminator::Goto(edge) => f(edge.target),
        SsaTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            f(then_edge.target);
            f(else_edge.target);
        }
        SsaTerminator::BrTable { entries, .. } => {
            for edge in entries {
                f(edge.target);
            }
        }
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
    }
}

fn rewrite_terminator_targets(term: &mut SsaTerminator, map: &[SsaTarget]) {
    let remap = |edge: &mut SsaEdge| {
        edge.target = map[edge.target.as_usize()];
    };
    match term {
        SsaTerminator::Goto(edge) => remap(edge),
        SsaTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            remap(then_edge);
            remap(else_edge);
        }
        SsaTerminator::BrTable { entries, .. } => {
            for edge in entries {
                remap(edge);
            }
        }
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::middle::ssa_ir::ir::{SsaInst, SsaLocalCachePrefs};
    use alloc::vec;

    fn empty_program() -> SsaProgram {
        SsaProgram {
            entry: SsaTarget(0),
            local_cache: SsaLocalCachePrefs::default(),
            blocks: Vec::new(),
            block_entry_cached_slots: Vec::new(),
            value_types: Vec::new(),
            value_sink_local: Vec::new(),
        }
    }

    fn make_goto(target: u32, bindings: Vec<SsaBinding>) -> SsaTerminator {
        SsaTerminator::Goto(SsaEdge {
            target: SsaTarget(target),
            bindings,
        })
    }

    fn make_branch(cond: SsaValue, then_target: u32, else_target: u32) -> SsaTerminator {
        SsaTerminator::Branch {
            cond,
            then_edge: SsaEdge {
                target: SsaTarget(then_target),
                bindings: Vec::new(),
            },
            else_edge: SsaEdge {
                target: SsaTarget(else_target),
                bindings: Vec::new(),
            },
        }
    }

    #[test]
    fn empty_program_is_noop() {
        let mut prog = empty_program();
        simplify_cfg(&mut prog);
        assert!(prog.blocks.is_empty());
    }

    #[test]
    fn simple_empty_block_threaded() {
        // b0(entry, empty) -> b1(empty) -> b2(return)
        // Entry is not redirected. b0's goto b1 is rewritten to b2.
        // b1 becomes dead. b0 and b2 survive. Then merge: b0→b2 single-pred → merged.
        let mut prog = empty_program();
        prog.blocks = vec![
            SsaBlock {
                id: SsaTarget(0),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: make_goto(1, Vec::new()),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: make_goto(2, Vec::new()),
            },
            SsaBlock {
                id: SsaTarget(2),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: SsaTerminator::Return { results: None },
            },
        ];
        simplify_cfg(&mut prog);
        // b0 stays as entry. b1 dead. b0→b2 merged into one block.
        assert_eq!(prog.blocks.len(), 1);
        assert_eq!(prog.entry, SsaTarget(0));
        assert!(matches!(
            prog.blocks[0].terminator,
            SsaTerminator::Return { .. }
        ));
    }

    #[test]
    fn chain_of_empty_blocks() {
        // b0(entry) -> b1(empty) -> b2(empty) -> b3(return)
        // b0's goto is rewritten to b3. b1, b2 dead. b0→b3 merged.
        let mut prog = empty_program();
        prog.blocks = vec![
            SsaBlock {
                id: SsaTarget(0),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: make_goto(1, Vec::new()),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: make_goto(2, Vec::new()),
            },
            SsaBlock {
                id: SsaTarget(2),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: make_goto(3, Vec::new()),
            },
            SsaBlock {
                id: SsaTarget(3),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: SsaTerminator::Return { results: None },
            },
        ];
        simplify_cfg(&mut prog);
        assert_eq!(prog.blocks.len(), 1);
        assert!(matches!(
            prog.blocks[0].terminator,
            SsaTerminator::Return { .. }
        ));
    }

    #[test]
    fn self_loop_preserved() {
        // b0(entry, empty goto b1) -> b1(empty, self-loop)
        // b0 is entry → not redirected. b0's goto b1 stays (b1 is self-loop,
        // no redirect for b1). Both blocks survive.
        let mut prog = empty_program();
        prog.blocks = vec![
            SsaBlock {
                id: SsaTarget(0),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: make_goto(1, Vec::new()),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: make_goto(1, Vec::new()),
            },
        ];
        simplify_cfg(&mut prog);
        assert_eq!(prog.blocks.len(), 2);
        assert_eq!(prog.entry, SsaTarget(0));
        match &prog.blocks[1].terminator {
            SsaTerminator::Goto(edge) => assert_eq!(edge.target, SsaTarget(1)),
            _ => panic!("expected self-loop Goto"),
        }
    }

    #[test]
    fn branch_targets_threaded() {
        // b0: branch cond then b1(empty->b3) else b2(empty->b3)
        // b3: return
        let mut prog = empty_program();
        prog.blocks = vec![
            SsaBlock {
                id: SsaTarget(0),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: make_branch(SsaValue(0), 1, 2),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: make_goto(3, Vec::new()),
            },
            SsaBlock {
                id: SsaTarget(2),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: make_goto(3, Vec::new()),
            },
            SsaBlock {
                id: SsaTarget(3),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: SsaTerminator::Return { results: None },
            },
        ];
        simplify_cfg(&mut prog);
        // b1 and b2 are gone; b0 branches directly to the return block.
        assert_eq!(prog.blocks.len(), 2);
        match &prog.blocks[0].terminator {
            SsaTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                assert_eq!(then_edge.target, SsaTarget(1));
                assert_eq!(else_edge.target, SsaTarget(1));
            }
            _ => panic!("expected Branch"),
        }
    }

    #[test]
    fn single_predecessor_merge() {
        // b0(has ops) -> goto b1(single pred, has ops) -> return
        use crate::vm::middle::frame::FrameSlot;
        let mut prog = empty_program();
        prog.blocks = vec![
            SsaBlock {
                id: SsaTarget(0),
                params: Vec::new(),
                ops: vec![SsaInst {
                    kind: SsaInstKind::LocalGetSlot {
                        slot: FrameSlot(0),
                        dst: SsaValue(0),
                    },
                }],
                terminator: make_goto(1, Vec::new()),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: Vec::new(),
                ops: vec![SsaInst {
                    kind: SsaInstKind::LocalGetSlot {
                        slot: FrameSlot(1),
                        dst: SsaValue(1),
                    },
                }],
                terminator: SsaTerminator::Return { results: None },
            },
        ];
        simplify_cfg(&mut prog);
        // Merged into a single block.
        assert_eq!(prog.blocks.len(), 1);
        assert_eq!(prog.blocks[0].ops.len(), 2);
    }

    #[test]
    fn entry_block_not_redirected() {
        // entry=b0(empty goto b1) -> b1(return)
        // Entry is not redirected (machine lowering adds cached-local init
        // to entry block).  b1 is single-predecessor of b0 → merged.
        let mut prog = empty_program();
        prog.blocks = vec![
            SsaBlock {
                id: SsaTarget(0),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: make_goto(1, Vec::new()),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: SsaTerminator::Return { results: None },
            },
        ];
        prog.entry = SsaTarget(0);
        simplify_cfg(&mut prog);
        assert_eq!(prog.entry, SsaTarget(0));
        // b0 + b1 merged into one block.
        assert_eq!(prog.blocks.len(), 1);
        assert!(matches!(
            prog.blocks[0].terminator,
            SsaTerminator::Return { .. }
        ));
    }

    #[test]
    fn empty_block_with_params_composed() {
        // b0: branch v99 then b1 [v1=v0] else b2 [v3=v0]
        // b1(params=[v1]): goto b2 [v3=v1]   (trivial, forwards param)
        // b2(params=[v3]): return
        // After threading: b0 branches then→b2 [v3=v0] else→b2 [v3=v0].
        // b1 is dead, removed. Two blocks remain.
        let mut prog = empty_program();
        prog.blocks = vec![
            SsaBlock {
                id: SsaTarget(0),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: SsaTerminator::Branch {
                    cond: SsaValue(99),
                    then_edge: SsaEdge {
                        target: SsaTarget(1),
                        bindings: vec![SsaBinding {
                            param: SsaValue(1),
                            value: SsaValue(0),
                        }],
                    },
                    else_edge: SsaEdge {
                        target: SsaTarget(2),
                        bindings: vec![SsaBinding {
                            param: SsaValue(3),
                            value: SsaValue(0),
                        }],
                    },
                },
            },
            SsaBlock {
                id: SsaTarget(1),
                params: vec![SsaValue(1)],
                ops: Vec::new(),
                terminator: SsaTerminator::Goto(SsaEdge {
                    target: SsaTarget(2),
                    bindings: vec![SsaBinding {
                        param: SsaValue(3),
                        value: SsaValue(1),
                    }],
                }),
            },
            SsaBlock {
                id: SsaTarget(2),
                params: vec![SsaValue(3)],
                ops: Vec::new(),
                terminator: SsaTerminator::Return { results: None },
            },
        ];
        simplify_cfg(&mut prog);
        // b1 threaded away; b0 branches directly to b2 (renumbered b1).
        assert_eq!(prog.blocks.len(), 2);
        match &prog.blocks[0].terminator {
            SsaTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                assert_eq!(then_edge.target, SsaTarget(1));
                assert_eq!(then_edge.bindings.len(), 1);
                assert_eq!(then_edge.bindings[0].param, SsaValue(3));
                assert_eq!(then_edge.bindings[0].value, SsaValue(0));
                assert_eq!(else_edge.target, SsaTarget(1));
            }
            _ => panic!("expected Branch"),
        }
    }

    #[test]
    fn cycle_of_trivial_blocks_preserved() {
        // b0(entry, empty→b1), b1(empty→b2), b2(empty→b1) — b1↔b2 cycle
        // Entry not redirected. b0's edge to b1: redirect[b1]=b2, rewrite → b2.
        // b1's edge to b2: redirect[b2]=b1, rewrite → b1 (self-loop).
        // b2's edge to b1: redirect[b1]=b2, rewrite → b2 (self-loop).
        // Reachable from b0: {b0, b2}. b1 dead. 2 blocks remain.
        let mut prog = empty_program();
        prog.blocks = vec![
            SsaBlock {
                id: SsaTarget(0),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: make_goto(1, Vec::new()),
            },
            SsaBlock {
                id: SsaTarget(1),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: make_goto(2, Vec::new()),
            },
            SsaBlock {
                id: SsaTarget(2),
                params: Vec::new(),
                ops: Vec::new(),
                terminator: make_goto(1, Vec::new()),
            },
        ];
        simplify_cfg(&mut prog);
        // b0(entry) + b2(self-loop) survive. No panic, no hang.
        assert_eq!(prog.blocks.len(), 2);
        assert_eq!(prog.entry, SsaTarget(0));
    }
}
