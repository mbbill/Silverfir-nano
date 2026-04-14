use crate::collections;
use crate::vm::middle::ssa_ir::ir::{DecodedOperand, SsaInstView};

use crate::vm::wasm::primitive_op::PrimitiveOpKind;

use super::helpers::{i32_program, op, prepare_i32_program, prim};

#[test]
fn prepared_ssa_absorbs_single_use_const_into_arithmetic_operand() {
    let semantic = i32_program(
        1,
        2,
        1,
        collections::vec![
            op(crate::vm::wasm::semantic_ir::SemanticOpKind::LocalGet { idx: 0 }),
            prim(PrimitiveOpKind::I32Const { value: 7 }),
            prim(PrimitiveOpKind::I32Add),
            op(crate::vm::wasm::semantic_ir::SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let program = &prepared.ssa;
    let mut add_args: Option<collections::Vec<DecodedOperand>> = None;
    'outer: for block in program.blocks.iter() {
        for (idx, _) in block.ops.iter().enumerate() {
            if let SsaInstView::Value { op, args, .. } = block.view(idx, program) {
                if matches!(op, PrimitiveOpKind::I32Add) {
                    add_args = Some(args.iter().map(|a| a.decode()).collect());
                    break 'outer;
                }
            }
        }
    }
    let add_args = add_args.expect("prepared SSA should still contain the mixed i32.add");

    assert!(
        add_args.iter().any(|arg| match arg {
            DecodedOperand::Const(idx) => program.const_pool[*idx as usize] == 7,
            _ => false,
        }),
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
        collections::vec![
            prim(PrimitiveOpKind::I32Const { value: 1 }),
            prim(PrimitiveOpKind::I32Const { value: 2 }),
            prim(PrimitiveOpKind::I32Add),
            op(crate::vm::wasm::semantic_ir::SemanticOpKind::ReturnOne),
        ],
    );

    let prepared = prepare_i32_program(&semantic, 2, 0);
    let program = &prepared.ssa;
    let mut saw_const_three = false;

    for block in program.blocks.iter() {
        for (idx, _) in block.ops.iter().enumerate() {
            if let SsaInstView::Value { op, args, .. } = block.view(idx, program) {
                assert!(
                    !matches!(op, PrimitiveOpKind::I32Add),
                    "fully constant i32.add should fold away in middle SSA; blocks={:?}",
                    prepared.ssa.blocks
                );
                if matches!(op, PrimitiveOpKind::I32Const { value: 3 }) {
                    assert_eq!(
                        args.len(),
                        0,
                        "folded const producer should not keep stale operands"
                    );
                    saw_const_three = true;
                }
            }
        }
    }

    assert!(
        saw_const_three,
        "fully constant expression should leave behind a single i32.const 3 producer; blocks={:?}",
        prepared.ssa.blocks
    );
}
