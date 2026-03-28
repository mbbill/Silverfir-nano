//! Per-bank local-cache preference analysis.
//!
//! Selects ranked lists of canonical local slots for GP and FP cached-local
//! registers. GP cache holds i32/i64/ref locals; FP cache holds f32/f64 locals.

use alloc::vec::Vec;
use core::ops::Range;

use crate::value_type::ValueType;
use crate::vm::{
    middle::ssa_ir::ir::{CachedLocalInfo, SsaLocalCachePrefs},
    wasm::{
        primitive_op::PrimitiveOpKind,
        semantic_ir::{SemanticOpKind, SemanticProgram},
    },
};

use super::frame::FrameLayoutPlan;
use super::state::gp_value_budget_units;

/// Returns `(cache_prefs, continuation_skip_reload)`.
///
/// `continuation_skip_reload` is a temporary consumed during prepare: one
/// `Vec<bool>` per call site in SemanticProgram op order, parallel to the
/// cached-local ordering (GP then FP).  Each entry is moved onto the
/// corresponding `SsaBoundaryOp::skip_reload` field and never stored
/// persistently.
pub(super) fn analyze_local_cache_prefs(
    semantic: &SemanticProgram,
    gp_unit_bytes: u8,
    gp_budget_units: u8,
    fp_budget_units: u8,
    frame: FrameLayoutPlan,
) -> (SsaLocalCachePrefs, Vec<Vec<bool>>) {
    if semantic.local_count == 0 {
        return (SsaLocalCachePrefs::default(), Vec::new());
    }

    let weights = local_weights(semantic);
    let entry_reads_before_write = entry_scope_reads_before_write(semantic);

    let gp = select_with_budget(
        &weights,
        gp_budget_units as usize,
        |idx| {
            semantic
                .local_types
                .get(idx)
                .map_or(true, |ty| !ty.is_float())
        },
        |idx| {
            semantic
                .local_types
                .get(idx)
                .copied()
                .map_or(1, |ty| gp_value_budget_units(ty, gp_unit_bytes))
        },
    );
    let fp = select_top_n(&weights, fp_budget_units as usize, |idx| {
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

    // Compute per-call-site reload-skip sets.  The cached-local indices are
    // GP then FP, matching the order used everywhere else.
    let cached_local_indices: Vec<u32> = gp.iter().chain(fp.iter()).copied().collect();
    let skip_reload = continuation_skip_reload(semantic, &cached_local_indices);

    let prefs = SsaLocalCachePrefs {
        gp_local_info: gp.iter().map(|idx| local_info(*idx)).collect(),
        gp_preferred_slots: gp.iter().map(|idx| frame.local_slot(*idx as u16)).collect(),
        gp_preferred_types: gp
            .iter()
            .map(|idx| {
                semantic
                    .local_types
                    .get(*idx as usize)
                    .copied()
                    .unwrap_or(ValueType::I32)
            })
            .collect(),
        fp_local_info: fp.iter().map(|idx| local_info(*idx)).collect(),
        fp_preferred_slots: fp.iter().map(|idx| frame.local_slot(*idx as u16)).collect(),
        fp_preferred_types: fp
            .into_iter()
            .map(|idx| semantic.local_types[idx as usize])
            .collect(),
    };
    (prefs, skip_reload)
}

fn select_top_n(weights: &[u64], count: usize, eligible: impl Fn(usize) -> bool) -> Vec<u32> {
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

fn select_with_budget(
    weights: &[u64],
    budget: usize,
    eligible: impl Fn(usize) -> bool,
    cost: impl Fn(usize) -> usize,
) -> Vec<u32> {
    if budget == 0 {
        return Vec::new();
    }

    let mut best: Vec<Option<(u64, Vec<u32>)>> = alloc::vec![None; budget + 1];
    best[0] = Some((0, Vec::new()));

    for (idx, &weight) in weights.iter().enumerate() {
        if weight == 0 || !eligible(idx) {
            continue;
        }
        let item_cost = cost(idx);
        if item_cost == 0 || item_cost > budget {
            continue;
        }

        for used in (item_cost..=budget).rev() {
            let Some((base_weight, base_items)) = best[used - item_cost].clone() else {
                continue;
            };
            let mut candidate_items = base_items;
            candidate_items.push(idx as u32);
            let candidate = (base_weight + weight, candidate_items);
            if select_state_better(&candidate, best[used].as_ref()) {
                best[used] = Some(candidate);
            }
        }
    }

    let selected = best
        .iter()
        .filter_map(|entry| entry.as_ref())
        .max_by(|lhs, rhs| compare_selection_state(lhs, rhs))
        .map(|(_, items)| items.clone())
        .unwrap_or_default();

    let mut ranked = selected;
    ranked.sort_by(|lhs, rhs| {
        let lhs_weight = weights[*lhs as usize];
        let rhs_weight = weights[*rhs as usize];
        rhs_weight.cmp(&lhs_weight).then_with(|| lhs.cmp(rhs))
    });
    ranked
}

fn select_state_better(candidate: &(u64, Vec<u32>), current: Option<&(u64, Vec<u32>)>) -> bool {
    current.is_none_or(|existing| compare_selection_state(candidate, existing).is_gt())
}

fn compare_selection_state(lhs: &(u64, Vec<u32>), rhs: &(u64, Vec<u32>)) -> core::cmp::Ordering {
    lhs.0
        .cmp(&rhs.0)
        .then_with(|| rhs.1.len().cmp(&lhs.1.len()))
        .then_with(|| rhs.1.cmp(&lhs.1))
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

/// For each call/call_indirect site in the function, compute which cached
/// locals are *definitely written before read* on the straight-line path
/// immediately following the call.  Those locals do not need reloading at the
/// continuation because their cached-register value will be overwritten before
/// anyone reads it.
///
/// The scan is intentionally conservative: it walks forward from the call site
/// until it hits a branch, another call, a loop, or end-of-function, recording
/// `LocalSet`/`LocalTee` as writes and `LocalGet` as reads.  Any cached local
/// that is written before being read in that window gets `skip_reload = true`.
///
/// Returns one `Vec<bool>` per call site (in SemanticProgram op order), with
/// one entry per cached local (GP then FP, same order as the preferred-slots
/// vectors).  `true` = skip reload.
fn continuation_skip_reload(
    semantic: &SemanticProgram,
    cached_local_indices: &[u32],
) -> Vec<Vec<bool>> {
    let n = cached_local_indices.len();
    if n == 0 {
        // No cached locals at all — return an empty skip set per call site.
        let call_count = semantic
            .ops
            .iter()
            .filter(|op| {
                matches!(
                    op.kind,
                    SemanticOpKind::CallInternal { .. }
                        | SemanticOpKind::CallExternal { .. }
                        | SemanticOpKind::CallIndirect { .. }
                )
            })
            .count();
        return alloc::vec![Vec::new(); call_count];
    }

    // Build a reverse map: wasm local index → position in cached_local_indices.
    let max_local = cached_local_indices.iter().copied().max().unwrap_or(0) as usize;
    let mut local_to_cache = alloc::vec![usize::MAX; max_local + 1];
    for (pos, &idx) in cached_local_indices.iter().enumerate() {
        local_to_cache[idx as usize] = pos;
    }

    let mut results = Vec::new();

    for (op_index, op) in semantic.ops.iter().enumerate() {
        let is_call = matches!(
            op.kind,
            SemanticOpKind::CallInternal { .. }
                | SemanticOpKind::CallExternal { .. }
                | SemanticOpKind::CallIndirect { .. }
        );
        if !is_call {
            continue;
        }

        // Walk forward from the op after this call.
        let mut written = alloc::vec![false; n];
        let mut read = alloc::vec![false; n];

        for subsequent in &semantic.ops[op_index + 1..] {
            match subsequent.kind {
                // Stop at control flow or another call — we can't see past these.
                SemanticOpKind::Block { .. }
                | SemanticOpKind::Loop { .. }
                | SemanticOpKind::If { .. }
                | SemanticOpKind::Else { .. }
                | SemanticOpKind::End
                | SemanticOpKind::Br { .. }
                | SemanticOpKind::BrIf { .. }
                | SemanticOpKind::BrTable { .. }
                | SemanticOpKind::CallInternal { .. }
                | SemanticOpKind::CallExternal { .. }
                | SemanticOpKind::CallIndirect { .. }
                | SemanticOpKind::ReturnVoid
                | SemanticOpKind::ReturnOne
                | SemanticOpKind::Return { .. }
                | SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable) => break,

                SemanticOpKind::LocalGet { idx } => {
                    let i = idx as usize;
                    if i <= max_local {
                        let pos = local_to_cache[i];
                        if pos != usize::MAX && !written[pos] {
                            read[pos] = true;
                        }
                    }
                }
                SemanticOpKind::LocalSet { idx } | SemanticOpKind::LocalTee { idx } => {
                    let i = idx as usize;
                    if i <= max_local {
                        let pos = local_to_cache[i];
                        if pos != usize::MAX && !read[pos] {
                            written[pos] = true;
                        }
                    }
                }
                _ => {}
            }
        }

        results.push(written);
    }

    results
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
fn merge_paths(fallthrough: Option<&[bool]>, branches: Option<&[bool]>, n: usize) -> Vec<bool> {
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

// ---------------------------------------------------------------------------
// Per-block local demand analysis (register-agnostic)
// ---------------------------------------------------------------------------

/// Per-block local access demand. Register-agnostic — just weight vectors.
/// The machine lowering layer applies budget-aware selection per block.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockLocalDemand {
    /// One weight vector per SSA block (indexed by block position in
    /// `block_ranges`). Each inner Vec has length == `local_count`.
    pub block_weights: Vec<Vec<u64>>,
    /// Loop depth at each block's entry point.
    pub block_loop_depth: Vec<u32>,
    /// Per-local: can the initial zero be observed? Whole-function analysis,
    /// used only by the entry block for cached-local initialization.
    pub reads_before_write: Vec<bool>,
    /// Per-call-site: local indices that are written-before-read after the
    /// call. One `Vec<u32>` per call site (in semantic op order). Machine
    /// lowering maps these to the continuation block's cache indices.
    pub continuation_skip_locals: Vec<Vec<u32>>,
    /// Local types from the semantic program, needed for GP/FP bank selection.
    pub local_types: Vec<ValueType>,
    /// Number of function parameters (locals with index < param_count).
    pub param_count: u16,
}

/// Compute per-block local demand from a semantic program and its block ranges.
///
/// `block_ranges` are the contiguous semantic-op ranges produced by
/// `build_block_ranges()` + `retain_reachable_blocks()`.
pub(super) fn analyze_per_block_demand(
    semantic: &SemanticProgram,
    block_ranges: &[Range<usize>],
) -> BlockLocalDemand {
    let n = semantic.local_count as usize;
    if n == 0 || block_ranges.is_empty() {
        return BlockLocalDemand {
            block_weights: alloc::vec![Vec::new(); block_ranges.len()],
            block_loop_depth: alloc::vec![0; block_ranges.len()],
            reads_before_write: Vec::new(),
            continuation_skip_locals: Vec::new(),
            local_types: semantic.local_types.clone(),
            param_count: semantic.params,
        };
    }

    let (block_weights, block_loop_depth) =
        per_block_local_weights(semantic, block_ranges);
    let reads_before_write = entry_scope_reads_before_write(semantic);
    let continuation_skip_locals = compute_continuation_skip_locals(semantic);

    BlockLocalDemand {
        block_weights,
        block_loop_depth,
        reads_before_write,
        continuation_skip_locals,
        local_types: semantic.local_types.clone(),
        param_count: semantic.params,
    }
}

/// Compute per-block local access weight vectors and loop depths.
///
/// Walks all semantic ops sequentially (tracking loop depth via a control
/// stack), accumulating weights into the current block's weight vector.
fn per_block_local_weights(
    semantic: &SemanticProgram,
    block_ranges: &[Range<usize>],
) -> (Vec<Vec<u64>>, Vec<u32>) {
    let n = semantic.local_count as usize;
    let block_count = block_ranges.len();
    let mut weights = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        weights.push(alloc::vec![0u64; n]);
    }
    let mut depths = alloc::vec![0u32; block_count];

    // Build a reverse map: semantic_op_index → block index.
    // For ops outside any block range, block_of[i] = usize::MAX.
    let op_count = semantic.ops.len();
    let mut block_of = alloc::vec![usize::MAX; op_count];
    for (bi, range) in block_ranges.iter().enumerate() {
        for idx in range.clone() {
            if idx < op_count {
                block_of[idx] = bi;
            }
        }
    }

    let mut loop_depth = 0u32;
    let mut control_stack: Vec<bool> = Vec::new();

    for (op_index, op) in semantic.ops.iter().enumerate() {
        // Record loop depth at block start.
        let bi = block_of[op_index];
        if bi != usize::MAX {
            let range = &block_ranges[bi];
            if op_index == range.start {
                depths[bi] = loop_depth;
            }
        }

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
                if bi != usize::MAX {
                    if let Some(w) = weights[bi].get_mut(idx as usize) {
                        *w = w.saturating_add(loop_weight(loop_depth));
                    }
                }
            }
            _ => {}
        }
    }

    (weights, depths)
}

/// For each call site, compute the set of local indices that are
/// written-before-read on the straight-line path immediately after the call.
///
/// Returns one `Vec<u32>` per call site (in semantic op order), containing
/// the local indices that can skip reload. This is register-agnostic — the
/// machine lowering maps these indices to the continuation block's cache.
fn compute_continuation_skip_locals(semantic: &SemanticProgram) -> Vec<Vec<u32>> {
    let mut results = Vec::new();

    for (op_index, op) in semantic.ops.iter().enumerate() {
        let is_call = matches!(
            op.kind,
            SemanticOpKind::CallInternal { .. }
                | SemanticOpKind::CallExternal { .. }
                | SemanticOpKind::CallIndirect { .. }
        );
        if !is_call {
            continue;
        }

        let mut written = Vec::new();
        let mut read = Vec::new();

        for subsequent in &semantic.ops[op_index + 1..] {
            match subsequent.kind {
                // Stop at control flow or another call.
                SemanticOpKind::Block { .. }
                | SemanticOpKind::Loop { .. }
                | SemanticOpKind::If { .. }
                | SemanticOpKind::Else { .. }
                | SemanticOpKind::End
                | SemanticOpKind::Br { .. }
                | SemanticOpKind::BrIf { .. }
                | SemanticOpKind::BrTable { .. }
                | SemanticOpKind::CallInternal { .. }
                | SemanticOpKind::CallExternal { .. }
                | SemanticOpKind::CallIndirect { .. }
                | SemanticOpKind::ReturnVoid
                | SemanticOpKind::ReturnOne
                | SemanticOpKind::Return { .. }
                | SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable) => break,

                SemanticOpKind::LocalGet { idx } => {
                    let i = idx as u32;
                    if !written.contains(&i) {
                        if !read.contains(&i) {
                            read.push(i);
                        }
                    }
                }
                SemanticOpKind::LocalSet { idx } | SemanticOpKind::LocalTee { idx } => {
                    let i = idx as u32;
                    if !read.contains(&i) {
                        if !written.contains(&i) {
                            written.push(i);
                        }
                    }
                }
                _ => {}
            }
        }

        results.push(written);
    }

    results
}

/// Make `select_with_budget` available to the machine lowering layer.
pub(crate) fn select_locals_with_budget(
    weights: &[u64],
    budget: usize,
    eligible: impl Fn(usize) -> bool,
    cost: impl Fn(usize) -> usize,
) -> Vec<u32> {
    select_with_budget(weights, budget, eligible, cost)
}

/// Make `select_top_n` available to the machine lowering layer.
pub(crate) fn select_locals_top_n(
    weights: &[u64],
    count: usize,
    eligible: impl Fn(usize) -> bool,
) -> Vec<u32> {
    select_top_n(weights, count, eligible)
}

#[cfg(test)]
mod tests {
    use super::analyze_local_cache_prefs;
    use crate::value_type::ValueType;
    use crate::vm::{
        middle::frame::plan_frame_layout,
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
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let frame = plan_frame_layout(semantic.local_count, semantic.max_stack_height, 0);
        let (prefs, _skip_reload) = analyze_local_cache_prefs(&semantic, 8, 2, 0, frame);
        assert_eq!(
            prefs.gp_preferred_slots,
            alloc::vec![frame.local_slot(1), frame.local_slot(0)],
        );
    }

    #[test]
    fn i64_locals_consume_two_gp_cache_units_on_32_bit() {
        let semantic = SemanticProgram {
            params: 0,
            results: 0,
            local_count: 2,
            max_stack_height: 0,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 1 },
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 1 },
                },
            ],
            local_types: alloc::vec![ValueType::I64, ValueType::I64],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };

        let frame = plan_frame_layout(semantic.local_count, semantic.max_stack_height, 0);
        let (prefs_64, _skip_reload) = analyze_local_cache_prefs(&semantic, 8, 2, 0, frame);
        let (prefs_32, _skip_reload) = analyze_local_cache_prefs(&semantic, 4, 2, 0, frame);

        assert_eq!(
            prefs_64.gp_preferred_slots,
            alloc::vec![frame.local_slot(1), frame.local_slot(0)],
        );
        assert_eq!(
            prefs_32.gp_preferred_slots,
            alloc::vec![frame.local_slot(1)]
        );
    }

    #[test]
    fn per_block_weights_splits_by_range() {
        // Two blocks: ops 0..2 and 2..4.
        // Block 0 accesses local 0 twice, block 1 accesses local 1 twice.
        let semantic = SemanticProgram {
            params: 0,
            results: 0,
            local_count: 2,
            max_stack_height: 0,
            ops: alloc::vec![
                SemanticOp { kind: SemanticOpKind::LocalGet { idx: 0 } },
                SemanticOp { kind: SemanticOpKind::LocalGet { idx: 0 } },
                SemanticOp { kind: SemanticOpKind::LocalGet { idx: 1 } },
                SemanticOp { kind: SemanticOpKind::LocalGet { idx: 1 } },
            ],
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };
        let ranges = alloc::vec![0..2, 2..4];
        let demand = super::analyze_per_block_demand(&semantic, &ranges);
        // Block 0: local 0 has weight 2, local 1 has weight 0
        assert_eq!(demand.block_weights[0], alloc::vec![2, 0]);
        // Block 1: local 0 has weight 0, local 1 has weight 2
        assert_eq!(demand.block_weights[1], alloc::vec![0, 2]);
        assert_eq!(demand.block_loop_depth, alloc::vec![0, 0]);
    }

    #[test]
    fn per_block_weights_respects_loop_depth() {
        // Block 0 (ops 0..1): outside loop
        // Block 1 (ops 1..3): inside loop (loop header + local access)
        let semantic = SemanticProgram {
            params: 0,
            results: 0,
            local_count: 1,
            max_stack_height: 0,
            ops: alloc::vec![
                SemanticOp { kind: SemanticOpKind::LocalGet { idx: 0 } },
                SemanticOp { kind: SemanticOpKind::Loop { params: 0, results: 0 } },
                SemanticOp { kind: SemanticOpKind::LocalGet { idx: 0 } },
                SemanticOp { kind: SemanticOpKind::End },
            ],
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };
        let ranges = alloc::vec![0..1, 1..4];
        let demand = super::analyze_per_block_demand(&semantic, &ranges);
        // Block 0: depth 0, weight = 1
        assert_eq!(demand.block_weights[0], alloc::vec![1]);
        // Block 1: loop starts at op 1, local.get at op 2 is at depth 1 (weight 10)
        assert_eq!(demand.block_weights[1], alloc::vec![10]);
        assert_eq!(demand.block_loop_depth[0], 0);
        assert_eq!(demand.block_loop_depth[1], 0); // loop op itself is still at depth 0 entry
    }

    #[test]
    fn continuation_skip_locals_basic() {
        // call; local.set 2; local.get 0; local.set 1
        // local 2 is written before read → skip
        // local 0 is read first → no skip
        // local 1 is written but after read of 0 → skip (it's not read before written)
        let semantic = SemanticProgram {
            params: 0,
            results: 0,
            local_count: 3,
            max_stack_height: 0,
            ops: alloc::vec![
                SemanticOp { kind: SemanticOpKind::CallExternal { func_idx: 0, params: 0, results: 0 } },
                SemanticOp { kind: SemanticOpKind::LocalSet { idx: 2 } },
                SemanticOp { kind: SemanticOpKind::LocalGet { idx: 0 } },
                SemanticOp { kind: SemanticOpKind::LocalSet { idx: 1 } },
            ],
            local_types: alloc::vec![],
            result_types: alloc::vec![],
            op_result_types: alloc::collections::BTreeMap::new(),
        };
        let skip = super::compute_continuation_skip_locals(&semantic);
        assert_eq!(skip.len(), 1);
        // Locals 2 and 1 are written before read
        let mut skip_sorted = skip[0].clone();
        skip_sorted.sort();
        assert_eq!(skip_sorted, alloc::vec![1, 2]);
    }
}
