//! Hot-local planning.
//!
//! This module analyzes semantic local accesses and selects which locals live
//! in the dedicated hot-local register class. It must remain entirely on the
//! planning side.

use alloc::vec::Vec;

use crate::vm::wasm::semantic_ir::{SemanticOpKind, SemanticProgram};

use super::{config::PlanConfig, frame::FrameLayoutPlan, types::PlannedHotLocalInit};

/// Function-level hot-local mapping selected during planning.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HotLocalPlan {
    mapping: Vec<Option<u32>>,
}

impl HotLocalPlan {
    #[inline]
    pub fn new(mapping: Vec<Option<u32>>) -> Self {
        Self { mapping }
    }

    #[inline]
    pub fn slot_count(&self) -> usize {
        self.mapping.len()
    }

    #[inline]
    pub fn mapping(&self) -> &[Option<u32>] {
        &self.mapping
    }

    #[inline]
    pub fn resolve(&self, local_idx: u32) -> Option<u8> {
        self.mapping
            .iter()
            .position(|mapped| *mapped == Some(local_idx))
            .map(|slot| slot as u8)
    }
}

pub fn analyze_hot_locals(semantic: &SemanticProgram, config: PlanConfig) -> Option<HotLocalPlan> {
    let slot_count = config.hot_local_count as usize;
    if slot_count == 0 || semantic.local_count == 0 {
        return None;
    }

    let weights = local_weights(semantic);
    let mut best = alloc::vec![None; slot_count];
    let mut found = 0usize;

    for (idx, &weight) in weights.iter().enumerate() {
        if weight == 0 {
            continue;
        }

        let mut insert_at = found.min(slot_count);
        for pos in 0..found.min(slot_count) {
            let (best_idx, best_weight) = best[pos].expect("filled prefix");
            if weight > best_weight || (weight == best_weight && idx < best_idx as usize) {
                insert_at = pos;
                break;
            }
        }

        if insert_at < slot_count {
            let mut pos = found.min(slot_count.saturating_sub(1));
            while pos > insert_at {
                best[pos] = best[pos - 1];
                pos -= 1;
            }
            best[insert_at] = Some((idx as u32, weight));
            found += 1;
        }
    }

    let mapping = best
        .into_iter()
        .map(|entry| entry.map(|(idx, _)| idx))
        .collect::<Vec<_>>();

    if mapping.iter().all(Option::is_none) {
        None
    } else {
        Some(HotLocalPlan::new(mapping))
    }
}

pub fn plan_hot_local_inits(
    hot_locals: Option<&HotLocalPlan>,
    frame: FrameLayoutPlan,
) -> Vec<PlannedHotLocalInit> {
    hot_locals
        .into_iter()
        .flat_map(|plan| plan.mapping().iter().enumerate())
        .filter_map(|(reg, local_idx)| {
            local_idx.map(|idx| PlannedHotLocalInit {
                reg: reg as u8,
                frame_slot: frame.local_slot(idx as u16),
            })
        })
        .collect()
}

fn local_weights(semantic: &SemanticProgram) -> Vec<u64> {
    let mut weights = alloc::vec![0; semantic.local_count as usize];
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
            SemanticOpKind::Else => {}
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
    use super::analyze_hot_locals;
    use crate::vm::{
        backend::BackendKind,
        plan::config::PlanConfig,
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
                    next: None,
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalSet { idx: 0 },
                    next: None,
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 1 },
                    next: None,
                    alt: None,
                },
            ],
        };

        let plan = analyze_hot_locals(
            &semantic,
            PlanConfig::for_backend(
                BackendKind::Base,
                crate::vm::backend::BackendConfig {
                    ctx_register_count: 1,
                    fp_register_count: 1,
                    tmp_register_count: 1,
                    hot_local_count: 2,
                    tos_register_count: 0,
                },
            ),
        )
        .expect("hot locals analyzed");

        assert_eq!(plan.mapping(), &[Some(1), Some(0)]);
    }

    #[test]
    fn loop_weighting_prefers_inner_loop_accesses() {
        let semantic = SemanticProgram {
            params: 0,
            results: 0,
            local_count: 2,
            max_stack_height: 0,
            ops: alloc::vec![
                SemanticOp {
                    kind: SemanticOpKind::Loop {
                        params: 0,
                        results: 0,
                    },
                    next: None,
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                    next: None,
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::End,
                    next: None,
                    alt: None,
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 1 },
                    next: None,
                    alt: None,
                },
            ],
        };

        let plan = analyze_hot_locals(
            &semantic,
            PlanConfig::for_backend(
                BackendKind::Base,
                crate::vm::backend::BackendConfig {
                    ctx_register_count: 1,
                    fp_register_count: 1,
                    tmp_register_count: 1,
                    hot_local_count: 1,
                    tos_register_count: 0,
                },
            ),
        )
        .expect("hot locals analyzed");

        assert_eq!(plan.mapping(), &[Some(0)]);
    }
}
