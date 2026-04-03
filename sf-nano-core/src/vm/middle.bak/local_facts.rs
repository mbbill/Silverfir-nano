//! Stable per-local facts for joint planning.
//!
//! This stage intentionally does not perform whole-function cache ranking.
//! It only records facts that remain valid regardless of the later planning
//! algorithm, such as slot/type information and whether the entry zero value
//! may be observed before a write.

use alloc::vec::Vec;

use crate::value_type::ValueType;
use crate::vm::{
    middle::ssa_ir::ir::LocalSlotInfo,
    wasm::{
        primitive_op::PrimitiveOpKind,
        semantic_ir::{SemanticOpKind, SemanticProgram},
    },
};

use super::frame::FrameLayoutPlan;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct LocalSlotFacts {
    pub(super) slot_types: Vec<ValueType>,
    pub(super) slot_info: Vec<LocalSlotInfo>,
}

pub(super) fn collect_local_slot_facts(
    semantic: &SemanticProgram,
    _gp_unit_bytes: u8,
    frame: FrameLayoutPlan,
) -> LocalSlotFacts {
    if semantic.local_count == 0 {
        return LocalSlotFacts::default();
    }

    let entry_reads_before_write = entry_scope_reads_before_write(semantic);

    let local_info = |idx: u32| LocalSlotInfo {
        is_param: (idx as u16) < semantic.params,
        reads_before_write: entry_reads_before_write
            .get(idx as usize)
            .copied()
            .unwrap_or(true),
    };

    let mut slot_types = Vec::with_capacity(semantic.local_count as usize);
    let mut slot_info = Vec::with_capacity(semantic.local_count as usize);

    for local_idx in 0..semantic.local_count as usize {
        let ty = semantic
            .local_types
            .get(local_idx)
            .copied()
            .unwrap_or(ValueType::I64);
        debug_assert_eq!(frame.local_slot(local_idx as u16).0 as usize, local_idx);
        slot_types.push(ty);
        slot_info.push(local_info(local_idx as u32));
    }

    LocalSlotFacts {
        slot_types,
        slot_info,
    }
}

/// For each local, determine whether its initial zero value may be observed
/// on any execution path from function entry.
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
        branch_def_set: Option<Vec<bool>>,
        if_arm_def_set: Option<Vec<bool>>,
        if_arm_reachable: bool,
    }

    let mut stack: Vec<Frame> = Vec::new();
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
                                let mut paths: Vec<&[bool]> = Vec::new();
                                if reachable {
                                    paths.push(&def_set);
                                }
                                paths.push(&frame.entry_def_set);
                                if let Some(br) = &frame.branch_def_set {
                                    paths.push(br);
                                }
                                def_set = intersect_all(&paths, n);
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

fn intersect_vecs(a: &[bool], b: &[bool]) -> Vec<bool> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x && y).collect()
}

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

fn merge_paths(fallthrough: Option<&[bool]>, branches: Option<&[bool]>, n: usize) -> Vec<bool> {
    match (fallthrough, branches) {
        (Some(ft), Some(br)) => intersect_vecs(ft, br),
        (Some(ft), None) => ft.to_vec(),
        (None, Some(br)) => br.to_vec(),
        (None, None) => alloc::vec![false; n],
    }
}
