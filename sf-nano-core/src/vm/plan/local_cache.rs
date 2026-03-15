//! Per-bank local-cache preference analysis.
//!
//! Selects ranked lists of canonical local slots for GP and FP cached-local
//! registers. GP cache holds i32/i64/ref locals; FP cache holds f32/f64 locals.

use alloc::vec::Vec;

use crate::vm::{
    lir::ir::LirLocalCachePrefs,
    wasm::semantic_ir::{SemanticOpKind, SemanticProgram},
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

    LirLocalCachePrefs {
        gp_preferred_slots: gp
            .into_iter()
            .map(|idx| frame.local_slot(idx as u16))
            .collect(),
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
