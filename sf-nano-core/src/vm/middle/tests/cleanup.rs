use crate::collections;
use crate::vm::middle::ssa_ir::ir::SsaTerminator;

use crate::vm::wasm::{primitive_op::PrimitiveOpKind, semantic_ir::SemanticOpKind};

use super::helpers::{i32_program, op, prepare_i32_program, prim, target};

#[test]
fn synthetic_entry_repair_block_is_merged_into_entry_block() {
    let semantic = i32_program(
        1,
        2,
        1,
        collections::vec![
            op(SemanticOpKind::LocalGet { idx: 0 }),
            op(SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::I32Add),
            op(SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    assert_eq!(
        prepared.ssa.blocks.len(),
        1,
        "synthetic entry repair should be merged into the real entry block instead of surviving as a separate goto block"
    );
    assert_eq!(prepared.ssa.entry.as_usize(), 0);
}

#[test]
fn prepared_ssa_contains_no_empty_identity_goto_blocks_after_cleanup() {
    let semantic = i32_program(
        1,
        1,
        0,
        collections::vec![
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            op(SemanticOpKind::If {
                params: 0,
                results: 0,
                else_target: target(4),
            }),
            prim(PrimitiveOpKind::Nop),
            op(SemanticOpKind::Else {
                end_target: target(5),
            }),
            prim(PrimitiveOpKind::Nop),
            op(SemanticOpKind::End),
            op(SemanticOpKind::ReturnVoid),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    for block in &prepared.ssa.blocks {
        if !block.ops.is_empty() {
            continue;
        }
        let SsaTerminator::Goto(edge) = &block.terminator else {
            continue;
        };
        let Some(target_block) = prepared.ssa.blocks.get(edge.target.as_usize()) else {
            continue;
        };
        let is_identity = target_block.params.len() == edge.bindings.len()
            && target_block
                .params
                .iter()
                .zip(edge.bindings.iter())
                .all(|(param, binding)| binding.param == *param && binding.value == *param);
        assert!(
            !is_identity,
            "cleanup should remove empty goto blocks whose edge is only an identity pass-through; blocks={:?}",
            prepared.ssa.blocks
        );
    }
}
