//! Liveness analysis: which non-param locals can be read before being written?
//!
//! Wasm requires non-param locals to start as zero. The JIT can elide that
//! initialization for any local that is provably written along every reachable
//! path before its first read. This module computes that set.
//!
//! The result is consumed by:
//! - `SsaProgram.local_slot_info[i].reads_before_write` (one bit per slot)
//! - machine lowering, which emits zero stores at the callee entry only for
//!   slots that need them
//!
//! ## Algorithm
//!
//! Forward dataflow over the structured semantic program. State is `def_set`,
//! a per-local boolean: "this local has definitely been written on the current
//! reachable path". A LocalGet of a local with `def_set[i] == false` marks
//! that local as needing initialization (`reads_init[i] = true`).
//!
//! Structured control flow is tracked with a frame stack:
//! - **Block** scopes: branches out (`Br`/`BrIf`/`BrTable`) intersect the
//!   current `def_set` into the frame's `branch_def_set`. At `End`, the
//!   post-block `def_set` is the meet of the fall-through state (if reachable)
//!   and the branch state.
//! - **If/Else** scopes: after `Else`, restore entry `def_set`; at `End`, meet
//!   both arms (each with its own reachability) plus any branches out.
//! - **Loop** scopes: branches out of loops target the loop header (which is
//!   the entry, not the merge), so `Br` to a loop frame is intentionally NOT
//!   recorded as an exit-merge contribution. The loop body sees the entry
//!   `def_set` for the first iteration; subsequent iterations would see at
//!   least the same writes, so a single forward pass is conservatively
//!   correct.
//!
//! Reachability: `Br`/`BrTable`/`Return*`/`Unreachable` set `reachable=false`.
//! Reads on unreachable code are not counted.

use crate::collections;

use crate::vm::wasm::{
    primitive_op::PrimitiveOpKind,
    semantic_ir::{SemanticOpKind, SemanticProgram},
};

/// For each local index `i`, returns `true` if local `i` may be read before
/// being written along some reachable execution path. Such locals require
/// zero initialization at function entry.
///
/// Params (`i < semantic.params`) are always considered initialized by the
/// caller (their values are passed in), so they never need init even if
/// `reads_init[i]` is true. Callers that want the "needs zero init" set should
/// also filter out params.
pub(crate) fn locals_reads_before_write(semantic: &SemanticProgram) -> collections::Vec<bool> {
    let n = semantic.local_count as usize;
    if n == 0 {
        return collections::Vec::new();
    }

    let mut def_set = collections::vec![false; n];
    // Params are guaranteed-initialized by the caller.
    for i in 0..(semantic.params as usize).min(n) {
        def_set[i] = true;
    }
    let mut reads_init = collections::vec![false; n];
    let mut reachable = true;

    let mut stack: collections::Vec<Frame> = collections::Vec::new();
    stack.push(Frame {
        kind: FrameKind::Block,
        entry_def_set: def_set.clone(),
        branch_def_set: None,
        if_arm_def_set: None,
        if_arm_reachable: false,
    });

    for op in &semantic.ops {
        match op.kind {
            SemanticOpKind::Block { .. } => {
                stack.push(Frame {
                    kind: FrameKind::Block,
                    entry_def_set: def_set.clone(),
                    branch_def_set: None,
                    if_arm_def_set: None,
                    if_arm_reachable: false,
                });
            }
            SemanticOpKind::Loop { .. } => {
                stack.push(Frame {
                    kind: FrameKind::Loop,
                    entry_def_set: def_set.clone(),
                    branch_def_set: None,
                    if_arm_def_set: None,
                    if_arm_reachable: false,
                });
            }
            SemanticOpKind::If { .. } => {
                stack.push(Frame {
                    kind: FrameKind::If,
                    entry_def_set: def_set.clone(),
                    branch_def_set: None,
                    if_arm_def_set: None,
                    if_arm_reachable: false,
                });
            }
            SemanticOpKind::Else { .. } => {
                if let Some(frame) = stack.last_mut() {
                    frame.if_arm_def_set = Some(def_set.clone());
                    frame.if_arm_reachable = reachable;
                    def_set.copy_from_slice(&frame.entry_def_set);
                    reachable = true;
                }
            }
            SemanticOpKind::End => {
                if let Some(frame) = stack.pop() {
                    match frame.kind {
                        FrameKind::Block => {
                            def_set = merge_paths(
                                if reachable { Some(&def_set) } else { None },
                                frame.branch_def_set.as_deref(),
                                n,
                            );
                            reachable = reachable || frame.branch_def_set.is_some();
                        }
                        FrameKind::If => {
                            if let Some(if_arm) = &frame.if_arm_def_set {
                                def_set = intersect_selected_paths(
                                    [
                                        frame.if_arm_reachable.then_some(if_arm.as_slice()),
                                        reachable.then_some(def_set.as_slice()),
                                        frame.branch_def_set.as_deref(),
                                    ],
                                    n,
                                );
                                reachable = reachable
                                    || frame.if_arm_reachable
                                    || frame.branch_def_set.is_some();
                            } else {
                                def_set = intersect_selected_paths(
                                    [
                                        reachable.then_some(def_set.as_slice()),
                                        Some(frame.entry_def_set.as_slice()),
                                        frame.branch_def_set.as_deref(),
                                    ],
                                    n,
                                );
                                reachable = true;
                            }
                        }
                        FrameKind::Loop => {}
                    }
                }
            }
            SemanticOpKind::Br { .. } => {
                if reachable {
                    for frame in stack.iter_mut() {
                        if frame.kind == FrameKind::Loop {
                            continue;
                        }
                        frame.branch_def_set = Some(match &frame.branch_def_set {
                            Some(existing) => intersect_vecs(existing, &def_set),
                            None => def_set.clone(),
                        });
                    }
                }
                reachable = false;
            }
            SemanticOpKind::BrIf { .. } => {
                if reachable {
                    for frame in stack.iter_mut() {
                        if frame.kind == FrameKind::Loop {
                            continue;
                        }
                        frame.branch_def_set = Some(match &frame.branch_def_set {
                            Some(existing) => intersect_vecs(existing, &def_set),
                            None => def_set.clone(),
                        });
                    }
                }
            }
            SemanticOpKind::BrTable { .. } => {
                if reachable {
                    for frame in stack.iter_mut() {
                        if frame.kind == FrameKind::Loop {
                            continue;
                        }
                        frame.branch_def_set = Some(match &frame.branch_def_set {
                            Some(existing) => intersect_vecs(existing, &def_set),
                            None => def_set.clone(),
                        });
                    }
                }
                reachable = false;
            }
            SemanticOpKind::ReturnVoid
            | SemanticOpKind::ReturnOne
            | SemanticOpKind::Return { .. }
            | SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable) => {
                reachable = false;
            }
            SemanticOpKind::LocalGet { idx } => {
                if reachable {
                    let i = idx as usize;
                    if i < n && !def_set[i] {
                        reads_init[i] = true;
                    }
                }
            }
            SemanticOpKind::LocalSet { idx } | SemanticOpKind::LocalTee { idx } => {
                if reachable {
                    let i = idx as usize;
                    if i < n {
                        def_set[i] = true;
                    }
                }
            }
            _ => {}
        }
    }

    reads_init
}

#[derive(Clone, Copy, PartialEq)]
enum FrameKind {
    Block,
    Loop,
    If,
}

struct Frame {
    kind: FrameKind,
    entry_def_set: collections::Vec<bool>,
    branch_def_set: Option<collections::Vec<bool>>,
    if_arm_def_set: Option<collections::Vec<bool>>,
    if_arm_reachable: bool,
}

fn intersect_vecs(a: &[bool], b: &[bool]) -> collections::Vec<bool> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x && y).collect()
}

fn intersect_selected_paths(paths: [Option<&[bool]>; 3], n: usize) -> collections::Vec<bool> {
    let mut out = collections::vec![false; n];
    let mut initialized = false;
    for path in paths.into_iter().flatten() {
        if !initialized {
            out = path.to_vec().into();
            initialized = true;
            continue;
        }
        for (slot, bit) in out.iter_mut().zip(path.iter()) {
            *slot &= *bit;
        }
    }
    out
}

fn merge_paths(a: Option<&[bool]>, b: Option<&[bool]>, n: usize) -> collections::Vec<bool> {
    match (a, b) {
        (None, None) => collections::vec![false; n],
        (Some(a), None) | (None, Some(a)) => a.to_vec().into(),
        (Some(a), Some(b)) => intersect_vecs(a, b),
    }
}
