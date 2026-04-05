use crate::vm::middle::ssa_ir::ir::{SsaInstKind, SsaOperand};
use crate::vm::wasm::primitive_op::PrimitiveOpKind;

use super::helpers::{i32_program, op, prepare_i32_program, prim};

#[test]
fn prepared_ssa_absorbs_single_use_const_into_arithmetic_operand() {
    let semantic = i32_program(
        1,
        2,
        1,
        alloc::vec![
            op(crate::vm::wasm::semantic_ir::SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            prim(PrimitiveOpKind::I32Add),
            op(crate::vm::wasm::semantic_ir::SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let add = prepared
        .ssa
        .blocks
        .iter()
        .flat_map(|block| block.ops.iter())
        .find_map(|inst| match &inst.kind {
            SsaInstKind::Value { op, args, .. }
                if matches!(op.primitive(), PrimitiveOpKind::I32Add) =>
            {
                Some(args)
            }
            _ => None,
        })
        .expect("prepared SSA should still contain the mixed i32.add");

    assert!(
        add.iter().any(|arg| matches!(arg, SsaOperand::Const(7))),
        "middle const absorption should fold the single-use i32.const directly into the i32.add operand; blocks={:?}",
        prepared.ssa.blocks
    );
}

#[test]
fn prepared_ssa_folds_fully_constant_expression_to_single_const() {
    let semantic = i32_program(
        0,
        2,
        1,
        alloc::vec![
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            prim(PrimitiveOpKind::I32Const { value: 2 }),
            prim(PrimitiveOpKind::I32Add),
            op(crate::vm::wasm::semantic_ir::SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let mut saw_const_three = false;

    for inst in prepared.ssa.blocks.iter().flat_map(|block| block.ops.iter()) {
        if let SsaInstKind::Value { op, args, .. } = &inst.kind {
            assert!(
                !matches!(op.primitive(), PrimitiveOpKind::I32Add),
                "fully constant i32.add should fold away in middle SSA; blocks={:?}",
                prepared.ssa.blocks
            );
            if matches!(op.primitive(), PrimitiveOpKind::I32Const { value: 3 }) {
                assert!(
                    args.is_empty(),
                    "folded const producer should not keep stale operands"
                );
                saw_const_three = true;
            }
        }
    }

    assert!(
        saw_const_three,
        "fully constant expression should leave behind a single i32.const 3 producer; blocks={:?}",
        prepared.ssa.blocks
    );
}
