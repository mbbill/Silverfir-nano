//! Block-entry boundary shaping for semantic LIR lowering.

use alloc::vec::Vec;

use crate::vm::{
    lir::ir::LirBlockParams,
    wasm::semantic_ir::{SemanticOpKind, SemanticProgram},
};

use super::{input::SemanticPlannedOp, state::ValueAlloc};

#[derive(Clone, Copy, Debug)]
struct ControlFrame {
    start_height: u16,
    results: u16,
}

pub(super) fn make_block_params(height: u16, values: &mut ValueAlloc) -> LirBlockParams {
    LirBlockParams {
        values: values.many(height as usize),
    }
}

pub(super) fn compute_entry_heights(
    semantic: &SemanticProgram,
    mapped: &[SemanticPlannedOp<'_>],
) -> Vec<u16> {
    let mut heights = Vec::with_capacity(mapped.len());
    let mut control = alloc::vec![ControlFrame {
        start_height: 0,
        results: semantic.results,
    }];

    for op in mapped {
        let entry_height = match &op.semantic.kind {
            SemanticOpKind::Else | SemanticOpKind::End => control
                .last()
                .map_or(op.planned.height, |frame| frame.start_height + frame.results),
            _ => op.planned.height,
        };
        heights.push(entry_height);

        match &op.semantic.kind {
            SemanticOpKind::Block { params, results }
            | SemanticOpKind::Loop { params, results } => {
                control.push(ControlFrame {
                    start_height: entry_height.saturating_sub(*params),
                    results: *results,
                });
            }
            SemanticOpKind::If { params, results } => {
                control.push(ControlFrame {
                    start_height: entry_height.saturating_sub(1).saturating_sub(*params),
                    results: *results,
                });
            }
            SemanticOpKind::End => {
                let _ = control.pop();
            }
            _ => {}
        }
    }

    heights
}
