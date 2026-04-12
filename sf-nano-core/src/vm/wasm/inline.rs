//! Function inlining at the semantic IR level.
//!
//! Replaces `CallDirect` ops with the callee's body when the callee is a
//! small leaf function. The wrapper `Block/End` pair provides the target for
//! any branch that would have exited the callee.
//!
//! Runs on a decoded caller using a retained set of tiny semantic callee
//! bodies, before `prepare_function` / IR lowering.

use crate::collections;

use crate::value_type::ValueType;

use super::common::{BrTableEntry, SemanticTarget};
use super::semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram};

/// Maximum number of semantic ops in a callee for it to be inlined.
/// Keep this low to match LLVM-level inlining behavior — only trivial
/// wrappers and tiny arithmetic helpers, not large leaf computations
/// like CRC routines.
const MAX_INLINE_OPS: usize = 12;

/// Multiplier applied to `MAX_INLINE_OPS` when the call site is inside a loop.
const LOOP_INLINE_MULTIPLIER: usize = 10;

/// Maximum number of parameters for an inline candidate.
const MAX_INLINE_PARAMS: usize = 16;

/// Check whether a callee is structurally eligible for inlining.
///
/// The current inliner is only proven on straight-line leaf helpers. Raising
/// the size thresholds is fine for larger arithmetic/local-only bodies, but it
/// must not silently start inlining structured control-flow callees, because
/// the `Return -> Br`/wrapper lowering is not robust enough there yet.
fn is_leaf_inline_candidate(callee: &SemanticProgram) -> bool {
    if callee.params as usize > MAX_INLINE_PARAMS {
        return false;
    }
    let len = callee.ops.len();
    for (index, op) in callee.ops.iter().enumerate() {
        let is_trailing_end = index + 2 == len && matches!(op.kind, SemanticOpKind::End);
        let is_trailing_return = index + 1 == len
            && matches!(
                op.kind,
                SemanticOpKind::ReturnVoid
                    | SemanticOpKind::ReturnOne
                    | SemanticOpKind::Return { .. }
            );
        if is_trailing_end || is_trailing_return {
            continue;
        }
        match &op.kind {
            SemanticOpKind::Primitive(_)
            | SemanticOpKind::LocalGet { .. }
            | SemanticOpKind::LocalSet { .. }
            | SemanticOpKind::LocalTee { .. } => {}
            _ => return false,
        }
    }
    true
}

/// Return whether this semantic program is worth retaining as an inline seed.
///
/// Call-site-specific budgets are still checked during inlining. This only
/// keeps leaf callees that are small enough to inline somewhere, including the
/// larger hot-loop budget.
pub(crate) fn retain_inline_candidate(callee: &SemanticProgram) -> bool {
    is_leaf_inline_candidate(callee) && callee.ops.len() <= inline_ops_limit(true)
}

/// Return the inline ops budget for a call site.  Sites inside loops get a
/// much larger budget because eliminating call overhead in a hot loop is
/// disproportionately valuable.
#[inline]
fn inline_ops_limit(in_loop: bool) -> usize {
    if in_loop {
        MAX_INLINE_OPS * LOOP_INLINE_MULTIPLIER
    } else {
        MAX_INLINE_OPS
    }
}

// ── Stack-depth tracking for Return → Br conversion ─────────────────────────

/// A recorded Return site: op index + the stack_drop needed for the equivalent
/// `Br` to the wrapper block.
struct ReturnSite {
    op_index: usize,
    stack_drop: u32,
    arity: u16,
}

/// Walk the callee's semantic ops and compute the `stack_drop` value for every
/// `Return`/`ReturnOne`/`ReturnVoid` op. Returns an empty vec when there are
/// no explicit returns.
fn find_return_sites(callee: &SemanticProgram) -> collections::Vec<ReturnSite> {
    let mut sites = collections::Vec::new();
    let mut depth: i32 = 0;
    let mut control: collections::Vec<(i32, u16, u16)> = collections::Vec::new(); // (height_below_params, params, results)
    let mut unreachable = false;

    for (i, op) in callee.ops.iter().enumerate() {
        if unreachable {
            match &op.kind {
                SemanticOpKind::Block { params, results }
                | SemanticOpKind::Loop { params, results } => {
                    control.push((0, *params, *results)); // dummy
                }
                SemanticOpKind::If {
                    params, results, ..
                } => {
                    control.push((0, *params, *results)); // dummy
                }
                SemanticOpKind::Else { .. } => {
                    if let Some(&(h, p, _r)) = control.last() {
                        depth = h + p as i32;
                        unreachable = false;
                    }
                }
                SemanticOpKind::End => {
                    if let Some((h, _p, r)) = control.pop() {
                        depth = h + r as i32;
                        unreachable = false;
                    }
                }
                _ => {}
            }
            continue;
        }

        match &op.kind {
            SemanticOpKind::Block { params, results }
            | SemanticOpKind::Loop { params, results } => {
                control.push((depth - *params as i32, *params, *results));
            }
            SemanticOpKind::If {
                params, results, ..
            } => {
                depth -= 1;
                control.push((depth - *params as i32, *params, *results));
            }
            SemanticOpKind::Else { .. } => {
                if let Some(&(h, p, _r)) = control.last() {
                    depth = h + p as i32;
                }
            }
            SemanticOpKind::End => {
                if let Some((h, _p, r)) = control.pop() {
                    depth = h + r as i32;
                }
            }
            SemanticOpKind::Br { .. } | SemanticOpKind::BrTable { .. } => {
                unreachable = true;
            }
            SemanticOpKind::BrIf { .. } => {
                depth -= 1;
            }
            SemanticOpKind::ReturnVoid => {
                sites.push(ReturnSite {
                    op_index: i,
                    stack_drop: depth as u32,
                    arity: 0,
                });
                unreachable = true;
            }
            SemanticOpKind::ReturnOne => {
                sites.push(ReturnSite {
                    op_index: i,
                    stack_drop: (depth - 1) as u32,
                    arity: 1,
                });
                unreachable = true;
            }
            SemanticOpKind::Return { arity } => {
                sites.push(ReturnSite {
                    op_index: i,
                    stack_drop: (depth - *arity as i32) as u32,
                    arity: *arity,
                });
                unreachable = true;
            }
            SemanticOpKind::CallDirect {
                params, results, ..
            } => {
                depth -= *params as i32;
                depth += *results as i32;
            }
            SemanticOpKind::CallIndirect {
                params, results, ..
            } => {
                depth -= 1;
                depth -= *params as i32;
                depth += *results as i32;
            }
            SemanticOpKind::Primitive(p) => {
                let (pops, pushes) = super::primitive_op::stack_effect(p);
                depth -= pops as i32;
                depth += pushes as i32;
            }
            SemanticOpKind::LocalGet { .. } => depth += 1,
            SemanticOpKind::LocalSet { .. } => depth -= 1,
            SemanticOpKind::LocalTee { .. } => {}
        }
    }

    sites
}

/// Apply inlining to a single caller function. `semantics` is the full array
/// of decoded programs (indexed by module function index, `None` for imports).
/// `caller_func_idx` is the caller's index into that array.
///
/// Returns `true` if any inlining was performed (the caller's program was
/// modified).
pub(crate) fn inline_calls_in_function(
    caller: &mut SemanticProgram,
    caller_func_idx: u32,
    semantics: &[Option<SemanticProgram>],
) -> bool {
    // Collect inline sites (process back-to-front so earlier indices stay valid).
    // Track loop depth so that call sites inside loops get a larger inline budget.
    let mut sites: collections::Vec<(usize, u32)> = collections::Vec::new(); // (op_index, callee_func_idx)
    let mut loop_depth: u32 = 0;
    let mut control_is_loop: collections::Vec<bool> = collections::Vec::new();
    for (i, op) in caller.ops.iter().enumerate() {
        match &op.kind {
            SemanticOpKind::Loop { .. } => {
                control_is_loop.push(true);
                loop_depth += 1;
            }
            SemanticOpKind::Block { .. } | SemanticOpKind::If { .. } => {
                control_is_loop.push(false);
            }
            SemanticOpKind::End => {
                if let Some(true) = control_is_loop.pop() {
                    loop_depth -= 1;
                }
            }
            SemanticOpKind::CallDirect { callee, .. } => {
                if *callee == caller_func_idx {
                    continue; // skip direct recursion
                }
                if let Some(Some(callee_prog)) = semantics.get(*callee as usize) {
                    let in_loop = loop_depth > 0;
                    if is_leaf_inline_candidate(callee_prog)
                        && callee_prog.ops.len() <= inline_ops_limit(in_loop)
                    {
                        sites.push((i, *callee));
                    }
                }
            }
            _ => {}
        }
    }
    if sites.is_empty() {
        return false;
    }

    // Process sites back-to-front so insertions don't shift earlier indices.
    sites.reverse();
    for &(site_idx, callee_func_idx) in &sites {
        let callee = semantics[callee_func_idx as usize].as_ref().unwrap();
        inline_single_call(caller, site_idx, callee);
    }

    // Recompute max_stack_height from the final ops.
    caller.max_stack_height = recompute_max_stack_height(caller);
    true
}

/// Replace `caller.ops[site]` (a `CallDirect`) with the inlined callee body.
fn inline_single_call(caller: &mut SemanticProgram, site: usize, callee: &SemanticProgram) {
    let call_op = &caller.ops[site];
    let (call_params, call_results) = match &call_op.kind {
        SemanticOpKind::CallDirect {
            params, results, ..
        } => (*params, *results),
        _ => unreachable!("inline_single_call called on non-CallDirect"),
    };

    // --- Allocate new locals for the callee's params + locals ---
    let local_offset = caller.local_count;
    let callee_total_locals = callee.local_count;
    let caller_local_count_before = caller.local_count;
    caller.local_count += callee_total_locals;
    merge_inlined_local_types(
        caller,
        caller_local_count_before,
        callee,
        callee_total_locals,
    );

    // --- Build the replacement op sequence ---
    //
    // Layout:
    //   ops[site]:                Block { params: 0, results: call_results }
    //   ops[site+1 .. +params]:  LocalSet for each param (reverse order)
    //   ops[site+1+params .. ]:  callee body (ops[0..N-1], excluding final End)
    //   ops[last]:               End   (wrapper block End)
    //
    // The callee's final End is NOT copied; the wrapper End takes its place.
    // Because targets are absolute indices, the callee's internal targets that
    // pointed at the final End (= callee.ops.len()-1) will, after the offset
    // adjustment, point exactly at the wrapper End.

    // The decoder always emits [body..., End, Return] where End is the
    // function-level block terminator and Return is the implicit return.
    // We must exclude BOTH: the wrapper Block's End replaces the function End,
    // and the wrapper's Br (for explicit returns) or fallthrough handles the return.
    let has_trailing_return = matches!(
        callee.ops.last().map(|op| &op.kind),
        Some(
            SemanticOpKind::ReturnVoid | SemanticOpKind::ReturnOne | SemanticOpKind::Return { .. }
        )
    );
    let callee_body_len = if has_trailing_return {
        callee.ops.len() - 2 // exclude End + Return
    } else {
        callee.ops.len() - 1 // just exclude End (shouldn't happen in practice)
    };
    let prefix_len = 1 + call_params as usize; // Block + LocalSets
    let total_inserted = prefix_len + callee_body_len + 1; // +1 for wrapper End

    // The offset applied to all callee targets: each callee op index I maps to
    // caller op index (site + prefix_len + I).
    let target_offset = site + prefix_len;

    let mut inserted = collections::Vec::with_capacity(total_inserted);

    // 1. Wrapper Block — params must match the call's params so that the
    //    values sitting on the caller's stack are consumed by the block, giving
    //    the correct post-block stack height (caller_depth - params + results).
    inserted.push(SemanticOp {
        kind: SemanticOpKind::Block {
            params: call_params,
            results: call_results,
        },
    });

    // 2. Store params into new locals (reverse order — wasm stack is LIFO)
    for i in (0..call_params).rev() {
        inserted.push(SemanticOp {
            kind: SemanticOpKind::LocalSet {
                idx: local_offset + i,
            },
        });
    }

    // 3. Callee body (remapped), converting Return → Br to wrapper End.
    let return_sites = find_return_sites(callee);
    // The wrapper End will be at position: site + prefix_len + callee_body_len
    // In callee-local coordinates that's callee_body_len (= callee.ops.len()-1).
    // After target_offset, it maps to: target_offset + callee_body_len
    //   = site + prefix_len + callee_body_len.
    // But the wrapper End replaces the callee's function-level End (which was at
    // callee.ops.len()-1). After target_offset: target_offset + (callee.ops.len()-1)
    //   = site + prefix_len + callee_body_len.  Same position. ✓
    let wrapper_end_abs = target_offset + callee_body_len;

    for (callee_idx, callee_op) in callee.ops[..callee_body_len].iter().enumerate() {
        if let Some(rs) = return_sites.iter().find(|rs| rs.op_index == callee_idx) {
            // Convert Return → Br to wrapper End
            inserted.push(SemanticOp {
                kind: SemanticOpKind::Br {
                    stack_drop: rs.stack_drop,
                    arity: rs.arity,
                    target: SemanticTarget::new(wrapper_end_abs),
                },
            });
        } else {
            inserted.push(SemanticOp {
                kind: remap_op(&callee_op.kind, local_offset, target_offset),
            });
        }
    }

    // 4. Wrapper End
    inserted.push(SemanticOp {
        kind: SemanticOpKind::End,
    });

    // --- Patch caller targets that point past the call site ---
    // Replacing 1 op with `total_inserted` ops shifts everything after `site`
    // by `total_inserted - 1`.
    let shift = total_inserted as i64 - 1;
    if shift != 0 {
        shift_targets_after(&mut caller.ops, site, shift);
    }

    // --- Splice into caller ---
    caller.ops.splice(site..=site, inserted);

    // --- Patch op_result_types keys ---
    // 1. Remove the entry for the old CallDirect (now replaced by Block).
    // 2. Shift entries with key > site by the insertion delta.
    // 3. Reattach the old call's result types to the new wrapper Block.
    // 4. Copy callee's op_result_types with target_offset applied.
    {
        let wrapper_result_types = caller.op_result_types.get(&site).cloned();
        let shifted: collections::Vec<_> = caller
            .op_result_types
            .iter()
            .filter(|(&k, _)| k != site) // remove old call entry
            .map(|(&k, v)| {
                let new_k = if k > site {
                    (k as i64 + shift) as usize
                } else {
                    k
                };
                (new_k, v.clone())
            })
            .collect();
        caller.op_result_types.clear();
        for (k, v) in shifted {
            caller.op_result_types.insert(k, v);
        }
        if let Some(wrapper_result_types) = wrapper_result_types {
            caller.op_result_types.insert(site, wrapper_result_types);
        }
        // Transfer callee's op_result_types (for multi-value blocks etc.)
        for (&k, v) in &callee.op_result_types {
            if k < callee_body_len {
                caller.op_result_types.insert(target_offset + k, v.clone());
            }
        }
    }

    // max_stack_height is recomputed by the caller after all sites are inlined.
}

fn merge_inlined_local_types(
    caller: &mut SemanticProgram,
    caller_local_count_before: u16,
    callee: &SemanticProgram,
    callee_total_locals: u16,
) {
    if callee_total_locals == 0 {
        return;
    }

    if !caller.local_types.is_empty() {
        if callee.local_types.is_empty() {
            // Callee has no type info — preserve the caller's typed state with
            // a conservative fallback for the appended locals.
            caller
                .local_types
                .extend((0..callee_total_locals).map(|_| ValueType::I64));
        } else {
            caller.local_types.extend_from_slice(&callee.local_types);
        }
        return;
    }

    if caller_local_count_before == 0 && !callee.local_types.is_empty() {
        caller.local_types = callee.local_types.clone();
    }
}

/// Walk the semantic ops and compute the true max operand stack height.
fn recompute_max_stack_height(program: &SemanticProgram) -> u16 {
    let mut depth: i32 = 0;
    let mut max_depth: i32 = 0;
    let mut control: collections::Vec<(i32, u16, u16)> = collections::Vec::new();
    let mut unreachable = false;

    for op in &program.ops {
        if unreachable {
            match &op.kind {
                SemanticOpKind::Block { params, results }
                | SemanticOpKind::Loop { params, results } => {
                    control.push((0, *params, *results));
                }
                SemanticOpKind::If {
                    params, results, ..
                } => {
                    control.push((0, *params, *results));
                }
                SemanticOpKind::Else { .. } => {
                    if let Some(&(h, p, _)) = control.last() {
                        depth = h + p as i32;
                        unreachable = false;
                    }
                }
                SemanticOpKind::End => {
                    if let Some((h, _, r)) = control.pop() {
                        depth = h + r as i32;
                        unreachable = false;
                    }
                }
                _ => {}
            }
            continue;
        }

        match &op.kind {
            SemanticOpKind::Block { params, results }
            | SemanticOpKind::Loop { params, results } => {
                control.push((depth - *params as i32, *params, *results));
            }
            SemanticOpKind::If {
                params, results, ..
            } => {
                depth -= 1;
                control.push((depth - *params as i32, *params, *results));
            }
            SemanticOpKind::Else { .. } => {
                if let Some(&(h, p, _)) = control.last() {
                    depth = h + p as i32;
                }
            }
            SemanticOpKind::End => {
                if let Some((h, _, r)) = control.pop() {
                    depth = h + r as i32;
                }
            }
            SemanticOpKind::Br { .. }
            | SemanticOpKind::BrTable { .. }
            | SemanticOpKind::ReturnVoid
            | SemanticOpKind::ReturnOne
            | SemanticOpKind::Return { .. } => {
                unreachable = true;
            }
            SemanticOpKind::BrIf { .. } => {
                depth -= 1;
            }
            SemanticOpKind::CallDirect {
                params, results, ..
            } => {
                depth -= *params as i32;
                depth += *results as i32;
            }
            SemanticOpKind::CallIndirect {
                params, results, ..
            } => {
                depth -= 1;
                depth -= *params as i32;
                depth += *results as i32;
            }
            SemanticOpKind::Primitive(p) => {
                let (pops, pushes) = super::primitive_op::stack_effect(p);
                depth -= pops as i32;
                depth += pushes as i32;
            }
            SemanticOpKind::LocalGet { .. } => depth += 1,
            SemanticOpKind::LocalSet { .. } => depth -= 1,
            SemanticOpKind::LocalTee { .. } => {}
        }
        if depth > max_depth {
            max_depth = depth;
        }
    }

    max_depth.max(0) as u16
}

/// Remap a single callee op: shift local indices by `local_offset` and branch
/// targets by `target_offset`.
fn remap_op(kind: &SemanticOpKind, local_offset: u16, target_offset: usize) -> SemanticOpKind {
    match kind {
        // --- Local access: shift index ---
        SemanticOpKind::LocalGet { idx } => SemanticOpKind::LocalGet {
            idx: idx + local_offset,
        },
        SemanticOpKind::LocalSet { idx } => SemanticOpKind::LocalSet {
            idx: idx + local_offset,
        },
        SemanticOpKind::LocalTee { idx } => SemanticOpKind::LocalTee {
            idx: idx + local_offset,
        },

        // --- Branch targets: shift by target_offset ---
        SemanticOpKind::Br {
            stack_drop,
            arity,
            target,
        } => SemanticOpKind::Br {
            stack_drop: *stack_drop,
            arity: *arity,
            target: offset_target(*target, target_offset),
        },
        SemanticOpKind::BrIf {
            stack_drop,
            arity,
            target,
        } => SemanticOpKind::BrIf {
            stack_drop: *stack_drop,
            arity: *arity,
            target: offset_target(*target, target_offset),
        },
        SemanticOpKind::BrTable { entries } => SemanticOpKind::BrTable {
            entries: entries
                .iter()
                .map(|e| BrTableEntry {
                    target: offset_target(e.target, target_offset),
                    stack_drop: e.stack_drop,
                    arity: e.arity,
                })
                .collect(),
        },

        // --- Control flow with targets ---
        SemanticOpKind::If {
            params,
            results,
            else_target,
        } => SemanticOpKind::If {
            params: *params,
            results: *results,
            else_target: offset_target(*else_target, target_offset),
        },
        SemanticOpKind::Else { end_target } => SemanticOpKind::Else {
            end_target: offset_target(*end_target, target_offset),
        },

        // --- Everything else: clone as-is ---
        SemanticOpKind::Block { params, results } => SemanticOpKind::Block {
            params: *params,
            results: *results,
        },
        SemanticOpKind::Loop { params, results } => SemanticOpKind::Loop {
            params: *params,
            results: *results,
        },
        SemanticOpKind::End => SemanticOpKind::End,
        SemanticOpKind::Primitive(p) => SemanticOpKind::Primitive(p.clone()),

        // These should not appear (is_inline_candidate rejects them) but
        // handle gracefully by cloning.
        other => other.clone(),
    }
}

fn offset_target(target: SemanticTarget, offset: usize) -> SemanticTarget {
    if target.is_pending() {
        target
    } else {
        SemanticTarget::new(target.index().as_usize() + offset)
    }
}

/// Shift all `SemanticTarget` values in `ops` that point past `after` by
/// `shift` positions. This accounts for ops being inserted/removed at `after`.
fn shift_targets_after(ops: &mut [SemanticOp], after: usize, shift: i64) {
    for op in ops.iter_mut() {
        match &mut op.kind {
            SemanticOpKind::Br { target, .. } | SemanticOpKind::BrIf { target, .. } => {
                *target = shift_target(*target, after, shift);
            }
            SemanticOpKind::BrTable { entries } => {
                for entry in entries.iter_mut() {
                    entry.target = shift_target(entry.target, after, shift);
                }
            }
            SemanticOpKind::If { else_target, .. } => {
                *else_target = shift_target(*else_target, after, shift);
            }
            SemanticOpKind::Else { end_target } => {
                *end_target = shift_target(*end_target, after, shift);
            }
            _ => {}
        }
    }
}

fn shift_target(target: SemanticTarget, after: usize, shift: i64) -> SemanticTarget {
    if target.is_pending() {
        return target;
    }
    let idx = target.index().as_usize();
    if idx > after {
        SemanticTarget::new((idx as i64 + shift) as usize)
    } else {
        target
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::wasm::primitive_op::PrimitiveOpKind;

    #[test]
    fn inline_preserves_wrapper_block_result_types() {
        let mut caller = SemanticProgram {
            params: 0,
            results: 1,
            local_count: 0,
            max_stack_height: 1,
            ops: collections::vec![
                SemanticOp {
                    kind: SemanticOpKind::CallDirect {
                        callee: 1,
                        params: 0,
                        results: 1,
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
            local_types: collections::vec![],
            result_types: collections::vec![ValueType::I32],
            op_result_types: tracked_alloc::collections::BTreeMap::from([(
                0usize,
                collections::vec![ValueType::I32],
            )]),
        };
        let callee = SemanticProgram {
            params: 0,
            results: 1,
            local_count: 0,
            max_stack_height: 1,
            ops: collections::vec![
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 7 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
            local_types: collections::vec![],
            result_types: collections::vec![ValueType::I32],
            op_result_types: tracked_alloc::collections::BTreeMap::new(),
        };

        inline_single_call(&mut caller, 0, &callee);

        assert!(matches!(
            caller.ops.first().map(|op| &op.kind),
            Some(SemanticOpKind::Block {
                params: 0,
                results: 1
            })
        ));
        assert_eq!(
            caller.op_result_types.get(&0),
            Some(&collections::vec![ValueType::I32]),
            "wrapper Block must keep the inlined call's result types",
        );
    }

    #[test]
    fn inline_adopts_typed_callee_locals_for_zero_local_caller() {
        let mut caller = SemanticProgram {
            params: 0,
            results: 1,
            local_count: 0,
            max_stack_height: 1,
            ops: collections::vec![
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::CallDirect {
                        callee: 1,
                        params: 1,
                        results: 1,
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
            local_types: collections::vec![],
            result_types: collections::vec![ValueType::I32],
            op_result_types: tracked_alloc::collections::BTreeMap::from([(
                1usize,
                collections::vec![ValueType::I32],
            )]),
        };
        let callee = SemanticProgram {
            params: 1,
            results: 1,
            local_count: 1,
            max_stack_height: 1,
            ops: collections::vec![
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
            local_types: collections::vec![ValueType::I32],
            result_types: collections::vec![ValueType::I32],
            op_result_types: tracked_alloc::collections::BTreeMap::new(),
        };

        inline_single_call(&mut caller, 1, &callee);

        assert_eq!(caller.local_count, 1);
        assert_eq!(caller.local_types, collections::vec![ValueType::I32]);
    }

    #[test]
    fn structured_control_callee_is_not_inline_candidate() {
        let callee = SemanticProgram {
            params: 1,
            results: 1,
            local_count: 1,
            max_stack_height: 2,
            ops: collections::vec![
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::If {
                        params: 0,
                        results: 1,
                        else_target: SemanticTarget::new(4),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 1 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::Else {
                        end_target: SemanticTarget::new(5),
                    },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Const { value: 0 }),
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                },
                SemanticOp {
                    kind: SemanticOpKind::ReturnOne,
                },
            ],
            local_types: collections::vec![ValueType::I32],
            result_types: collections::vec![ValueType::I32],
            op_result_types: tracked_alloc::collections::BTreeMap::from([(
                1usize,
                collections::vec![ValueType::I32],
            )]),
        };

        assert!(
            !is_leaf_inline_candidate(&callee),
            "raising the inline size budget must not silently admit structured-control callees into the current straight-line inliner"
        );
    }
}
