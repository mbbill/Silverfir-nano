//! Per-bank local-cache preference analysis.
//!
//! Selects ranked lists of canonical local slots for GP and FP cached-local
//! registers. GP cache holds i32/i64/ref locals; FP cache holds f32/f64 locals.

use alloc::vec::Vec;

use crate::vm::{
    lir::ir::{CachedLocalInfo, LirLocalCachePrefs},
    wasm::{
        primitive_op::PrimitiveOpKind,
        semantic_ir::{SemanticOpKind, SemanticProgram},
    },
};

use super::frame::FrameLayoutPlan;

pub fn analyze_local_cache_prefs(
    semantic: &SemanticProgram,
    gp_slots: u8,
    fp_slots: u8,
    frame: FrameLayoutPlan,
) -> LirLocalCachePrefs {
    if semantic.local_count == 0 {
        return LirLocalCachePrefs::default();
    }

    let weights = local_weights(semantic);
    let entry_reads_before_write = entry_scope_reads_before_write(semantic);

    let gp = select_top_n(&weights, gp_slots as usize, |idx| {
        semantic
            .local_types
            .get(idx)
            .map_or(true, |ty| !ty.is_float())
    });
    let fp = select_top_n(&weights, fp_slots as usize, |idx| {
        semantic
            .local_types
            .get(idx)
            .map_or(false, |ty| ty.is_float())
    });

    let local_info = |idx: u32| CachedLocalInfo {
        is_param: (idx as u16) < semantic.params,
        reads_before_write: entry_reads_before_write
            .get(idx as usize)
            .copied()
            .unwrap_or(true),
    };

    LirLocalCachePrefs {
        gp_local_info: gp.iter().map(|idx| local_info(*idx)).collect(),
        gp_preferred_slots: gp
            .into_iter()
            .map(|idx| frame.local_slot(idx as u16))
            .collect(),
        fp_local_info: fp.iter().map(|idx| local_info(*idx)).collect(),
        fp_preferred_slots: fp
            .iter()
            .map(|idx| frame.local_slot(*idx as u16))
            .collect(),
        fp_preferred_types: fp
            .into_iter()
            .map(|idx| semantic.local_types[idx as usize])
            .collect(),
    }
}

fn select_top_n(
    weights: &[u64],
    count: usize,
    eligible: impl Fn(usize) -> bool,
) -> Vec<u32> {
    if count == 0 {
        return Vec::new();
    }
    let mut best: Vec<Option<(u32, u64)>> = alloc::vec![None; count];
    let mut found = 0usize;

    for (idx, &weight) in weights.iter().enumerate() {
        if weight == 0 || !eligible(idx) {
            continue;
        }

        let mut insert_at = found.min(count);
        for pos in 0..found.min(count) {
            let (best_idx, best_weight) = best[pos].expect("filled prefix");
            if weight > best_weight || (weight == best_weight && idx < best_idx as usize) {
                insert_at = pos;
                break;
            }
        }

        if insert_at < count {
            let mut pos = found.min(count.saturating_sub(1));
            while pos > insert_at {
                best[pos] = best[pos - 1];
                pos -= 1;
            }
            best[insert_at] = Some((idx as u32, weight));
            found += 1;
        }
    }

    best.into_iter()
        .filter_map(|entry| entry.map(|(idx, _)| idx))
        .collect()
}

/// For each local, determine whether its initial zero value may be observed
/// on any execution path from function entry. Uses a structured must-set
/// analysis that walks the wasm control flow.
///
/// The analysis tracks which locals are *definitely written* on all paths
/// reaching each program point. A `LocalGet` at a point where the local is
/// not definitely written means the initial zero is observable.
///
/// Returns `true` for locals whose initial zero may be observed.
fn entry_scope_reads_before_write(semantic: &SemanticProgram) -> Vec<bool> {
    let n = semantic.local_count as usize;
    if n == 0 {
        return Vec::new();
    }

    let mut def_set = alloc::vec![false; n];
    let mut reads_init = alloc::vec![false; n];
    let mut reachable = true;

    #[derive(Clone, Copy, PartialEq)]
    enum FrameKind {
        Block,
        Loop,
        If,
    }

    struct Frame {
        kind: FrameKind,
        entry_def_set: Vec<bool>,
        /// Intersection of def_sets from branches targeting this scope's End.
        branch_def_set: Option<Vec<bool>>,
        /// Saved if-arm state for If scopes with Else.
        if_arm_def_set: Option<Vec<bool>>,
        if_arm_reachable: bool,
    }

    let mut stack: Vec<Frame> = Vec::new();

    // Implicit function-level block scope.
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
                            // Merge fallthrough + all branch paths.
                            def_set = merge_paths(
                                if reachable { Some(&def_set) } else { None },
                                frame.branch_def_set.as_deref(),
                                n,
                            );
                            reachable = reachable || frame.branch_def_set.is_some();
                        }
                        FrameKind::If => {
                            if let Some(if_arm) = &frame.if_arm_def_set {
                                // Has Else: merge if-arm + else-arm + branches.
                                let mut paths: Vec<&[bool]> = Vec::new();
                                if frame.if_arm_reachable {
                                    paths.push(if_arm);
                                }
                                if reachable {
                                    paths.push(&def_set);
                                }
                                if let Some(br) = &frame.branch_def_set {
                                    paths.push(br);
                                }
                                def_set = intersect_all(&paths, n);
                                reachable = reachable
                                    || frame.if_arm_reachable
                                    || frame.branch_def_set.is_some();
                            } else {
                                // No Else: merge fallthrough + entry (skip path)
                                // + branches.
                                let mut paths: Vec<&[bool]> = Vec::new();
                                if reachable {
                                    paths.push(&def_set);
                                }
                                paths.push(&frame.entry_def_set); // always reachable
                                if let Some(br) = &frame.branch_def_set {
                                    paths.push(br);
                                }
                                def_set = intersect_all(&paths, n);
                                reachable = true;
                            }
                        }
                        FrameKind::Loop => {
                            // Back-edge branches target loop header, not End.
                            // Fallthrough continues; if dead, stays dead.
                        }
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
                // Fallthrough continues, still reachable.
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


/// Intersect two def_set vecs element-wise (AND).
fn intersect_vecs(a: &[bool], b: &[bool]) -> Vec<bool> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x && y).collect()
}

/// Intersect all paths. If no paths are reachable, return all-false.
fn intersect_all(paths: &[&[bool]], n: usize) -> Vec<bool> {
    if paths.is_empty() {
        return alloc::vec![false; n];
    }
    let mut result = paths[0].to_vec();
    for path in &paths[1..] {
        for (r, &p) in result.iter_mut().zip(path.iter()) {
            *r = *r && p;
        }
    }
    result
}

/// Merge fallthrough (if reachable) with branch paths at a Block's End.
fn merge_paths(
    fallthrough: Option<&[bool]>,
    branches: Option<&[bool]>,
    n: usize,
) -> Vec<bool> {
    match (fallthrough, branches) {
        (Some(ft), Some(br)) => intersect_vecs(ft, br),
        (Some(ft), None) => ft.to_vec(),
        (None, Some(br)) => br.to_vec(),
        (None, None) => alloc::vec![false; n],
    }
}


fn local_weights(semantic: &SemanticProgram) -> Vec<u64> {
    let mut weights = alloc::vec![0u64; semantic.local_count as usize];
    let mut loop_depth = 0u32;
    let mut control_stack = Vec::new();

    for op in &semantic.ops {
        match op.kind {
            SemanticOpKind::Block { .. } | SemanticOpKind::If { .. } => {
                control_stack.push(false);
            }
            SemanticOpKind::Loop { .. } => {
                loop_depth = loop_depth.saturating_add(1);
                control_stack.push(true);
            }
            SemanticOpKind::Else { .. } => {}
            SemanticOpKind::End => {
                if control_stack.pop().unwrap_or(false) {
                    loop_depth = loop_depth.saturating_sub(1);
                }
            }
            SemanticOpKind::LocalGet { idx }
            | SemanticOpKind::LocalSet { idx }
            | SemanticOpKind::LocalTee { idx } => {
                if let Some(weight) = weights.get_mut(idx as usize) {
                    *weight = weight.saturating_add(loop_weight(loop_depth));
                }
            }
            _ => {}
        }
    }

    weights
}

#[inline]
fn loop_weight(depth: u32) -> u64 {
    const WEIGHTS: [u64; 7] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000];
    WEIGHTS[depth.min(6) as usize]
}

#[cfg(test)]
mod tests {
    use super::analyze_local_cache_prefs;
    use crate::vm::{
        plan::frame::plan_frame_layout,
        wasm::semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
    };

    #[test]
    fn prefers_more_frequently_used_locals() {
        let semantic = SemanticProgram {
            params: 0,
            results: 0,
            local_count: 3,
            max_stack_height: 0,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 1 },
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalSet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 1 },
                },
            ],
            local_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let frame = plan_frame_layout(semantic.local_count, semantic.max_stack_height, 0);
        let prefs = analyze_local_cache_prefs(&semantic, 2, 0, frame);
        assert_eq!(
            prefs.gp_preferred_slots,
            alloc::vec![frame.local_slot(1), frame.local_slot(0)],
        );
    }
}
