//! Delay the native frame until after an empty scalar-return entry guard.
//!
//! The guard reads registers only. Its returning arm never changes a
//! preserved lane or touches the Wasm/native frame, so it needs neither
//! the body's alignment shim nor its frame probe. The ordinary arm still
//! executes the complete prelude, probe, and normal return/error paths.

use crate::{
    error::WasmError,
    vm::jit::machine::machine_ir::{
        MachineBlockId, MachineBranchCond, MachineCompareKind, MachineEdge, MachineInst,
        MachineInstKind, MachineIntWidth, MachineProgram, MachineReg, MachineResultSrc,
        MachineReturnValue, MachineSign, MachineTerminator, MachineValue,
    },
};

use super::{abi, backend::X86_64Backend, enc, fusion::map_int_cond};

#[derive(Debug, PartialEq, Eq)]
struct EntryGuard {
    width: MachineIntWidth,
    kind: MachineCompareKind,
    sign: MachineSign,
    lhs: MachineValue,
    rhs: MachineValue,
    result: EntryResult,
    return_on_true: bool,
}

#[derive(Debug, PartialEq, Eq)]
enum EntryResult {
    Reg(MachineReg),
    Constant(u64),
}

fn identity_edge(program: &MachineProgram, edge: &MachineEdge) -> bool {
    program
        .blocks
        .get(edge.target.as_usize())
        .is_some_and(|block| {
            block.params.len() == edge.args.len()
                && block
                    .params
                    .iter()
                    .zip(&edge.args)
                    .all(|(param, arg)| *arg == MachineValue::Reg(param.reg))
        })
}

fn targets_entry(term: &MachineTerminator, entry: MachineBlockId) -> bool {
    match term {
        MachineTerminator::Jump(edge) => edge.target == entry,
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => then_edge.target == entry || else_edge.target == entry,
        MachineTerminator::JumpTable { entries, .. } => {
            entries.iter().any(|edge| edge.target == entry)
        }
        MachineTerminator::Call { success, .. } => success.target == entry,
        MachineTerminator::TailCall { .. }
        | MachineTerminator::Return
        | MachineTerminator::ReturnScalar { .. }
        | MachineTerminator::Trap { .. } => false,
    }
}

fn entry_guard(program: &MachineProgram, next: Option<MachineBlockId>) -> Option<EntryGuard> {
    let entry = program.blocks.get(program.entry.as_usize())?;
    if !entry.ops.is_empty() {
        return None;
    }
    let MachineTerminator::Branch {
        cond:
            MachineBranchCond::IntCompare {
                width,
                kind,
                sign,
                lhs,
                rhs,
            },
        then_edge,
        else_edge,
    } = &entry.terminator
    else {
        return None;
    };
    if matches!(lhs, MachineValue::ReservedReg(_))
        || matches!(rhs, MachineValue::ReservedReg(_))
        || !identity_edge(program, then_edge)
        || !identity_edge(program, else_edge)
    {
        return None;
    }
    let (fast, return_on_true) = if Some(else_edge.target) == next {
        (then_edge, true)
    } else if Some(then_edge.target) == next {
        (else_edge, false)
    } else {
        return None;
    };
    let fast_block = program.blocks.get(fast.target.as_usize())?;
    let MachineTerminator::ReturnScalar {
        value:
            MachineReturnValue::ScalarGp {
                src: MachineResultSrc::Reg(result),
                ty: result_ty,
            },
    } = fast_block.terminator
    else {
        return None;
    };
    // A literal can go directly into the result lane without clobbering the
    // block's destination lane (which may be callee-preserved). Every other
    // instruction keeps the ordinary framed lowering.
    let result = match fast_block.ops.as_slice() {
        [] => EntryResult::Reg(result),
        [MachineInst {
            kind:
                MachineInstKind::Move {
                    dst,
                    src: MachineValue::Imm64(value),
                    ty,
                    ..
                },
        }] if *dst == result && *ty == result_ty => EntryResult::Constant(*value),
        _ => return None,
    };
    // Backedges enter with an existing body frame; they must never execute
    // the frameless return. Keep the normal entry lowering in that case.
    if program
        .blocks
        .iter()
        .any(|block| targets_entry(&block.terminator, program.entry))
    {
        return None;
    }
    Some(EntryGuard {
        width: *width,
        kind: *kind,
        sign: *sign,
        lhs: *lhs,
        rhs: *rhs,
        result,
        return_on_true,
    })
}

impl X86_64Backend<'_> {
    pub(super) fn emit_body_entry_guard(
        &mut self,
        next: Option<MachineBlockId>,
    ) -> Result<bool, WasmError> {
        let function = self.core.mir_function()?;
        let Some(guard) = entry_guard(&function.program, next) else {
            return Ok(false);
        };
        let entry_label = self.core.block_label(function.program.entry)?;
        self.core.bind_label(entry_label);
        self.lower_cmp_values(guard.width, guard.lhs, guard.rhs)?;
        let body = self.core.new_label();
        let cc = map_int_cond(guard.kind, guard.sign);
        self.emit_jcc(
            if guard.return_on_true {
                cc.invert()
            } else {
                cc
            },
            body,
        );
        match guard.result {
            EntryResult::Reg(reg) => {
                let src = self.map_gp_reg(reg)?;
                if src != abi::W2W_GP_RET0 {
                    enc::mov_rr_64(&mut self.core.text, abi::W2W_GP_RET0, src);
                }
            }
            EntryResult::Constant(value) => {
                enc::mov_ri_64(&mut self.core.text, abi::W2W_GP_RET0, value);
            }
        }
        enc::xor_rr_32(&mut self.core.text, abi::C_RET0, abi::C_RET0);
        enc::ret(&mut self.core.text);
        self.core.bind_label(body);
        self.flags32 = None;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        collections,
        vm::jit::machine::machine_ir::{
            MachineBlock, MachineBlockParam, MachineRegOwner, MachineStorageType,
        },
    };

    fn program() -> MachineProgram {
        let edge = |target| MachineEdge {
            target: MachineBlockId(target),
            args: collections::vec![MachineValue::Reg(MachineReg(4))],
        };
        MachineProgram {
            entry: MachineBlockId(0),
            fp_reg_init_widths: collections::vec![],
            blocks: collections::vec![
                MachineBlock {
                    id: MachineBlockId(0),
                    params: collections::vec![MachineBlockParam::gp_i64(MachineReg(4))],
                    ops: collections::vec![],
                    terminator: MachineTerminator::Branch {
                        cond: MachineBranchCond::IntCompare {
                            width: MachineIntWidth::I64,
                            kind: MachineCompareKind::Lt,
                            sign: MachineSign::Signed,
                            lhs: MachineValue::Reg(MachineReg(4)),
                            rhs: MachineValue::Imm64(37),
                        },
                        then_edge: edge(1),
                        else_edge: edge(2),
                    },
                },
                MachineBlock {
                    id: MachineBlockId(1),
                    params: collections::vec![MachineBlockParam::gp_i64(MachineReg(4))],
                    ops: collections::vec![],
                    terminator: MachineTerminator::ReturnScalar {
                        value: MachineReturnValue::ScalarGp {
                            ty: MachineStorageType::GpI64,
                            src: MachineResultSrc::Reg(MachineReg(4)),
                        },
                    },
                },
                MachineBlock {
                    id: MachineBlockId(2),
                    params: collections::vec![MachineBlockParam::gp_i64(MachineReg(4))],
                    ops: collections::vec![],
                    terminator: MachineTerminator::Return,
                },
            ],
        }
    }

    #[test]
    fn accepts_both_guard_directions_but_requires_the_other_arm_to_fall_through() {
        let mut p = program();
        let guard = entry_guard(&p, Some(MachineBlockId(2))).unwrap();
        assert!(guard.return_on_true);
        assert_eq!(guard.rhs, MachineValue::Imm64(37));
        assert!(entry_guard(&p, None).is_none());
        assert!(entry_guard(&p, Some(MachineBlockId(1))).is_none());
        let MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } = &mut p.blocks[0].terminator
        else {
            unreachable!()
        };
        core::mem::swap(then_edge, else_edge);
        assert!(
            !entry_guard(&p, Some(MachineBlockId(2)))
                .unwrap()
                .return_on_true
        );
    }

    #[test]
    fn rejects_instructions_frame_results_and_edge_moves_before_the_frame() {
        let mut p = program();
        let op = MachineInst {
            kind: MachineInstKind::Move {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpI64,
                dst: MachineReg(5),
                src: MachineValue::Imm64(5),
            },
        };
        for block in [0, 1] {
            p.blocks[block].ops.push(op.clone());
            assert!(entry_guard(&p, Some(MachineBlockId(2))).is_none());
            p.blocks[block].ops.clear();
        }
        let MachineTerminator::ReturnScalar {
            value: MachineReturnValue::ScalarGp { src, .. },
        } = &mut p.blocks[1].terminator
        else {
            unreachable!()
        };
        *src = MachineResultSrc::FrameSlot(crate::vm::jit::middle::frame::FrameSlot(0));
        assert!(entry_guard(&p, Some(MachineBlockId(2))).is_none());
        for fast in [true, false] {
            let mut p = program();
            let MachineTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } = &mut p.blocks[0].terminator
            else {
                unreachable!()
            };
            let edge = if fast { then_edge } else { else_edge };
            edge.args[0] = MachineValue::Reg(MachineReg(5));
            assert!(entry_guard(&p, Some(MachineBlockId(2))).is_none());
        }
    }

    #[test]
    fn constant_return_materializes_only_the_return_lane() {
        let mut p = program();
        for value in [0, u64::MAX, 1 << 63, 0xabc0_1234_5678] {
            p.blocks[1].ops = collections::vec![MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::LinearValue,
                    ty: MachineStorageType::GpI64,
                    dst: MachineReg(4),
                    src: MachineValue::Imm64(value),
                }
            }];
            assert_eq!(
                entry_guard(&p, Some(MachineBlockId(2))).unwrap().result,
                EntryResult::Constant(value)
            );
        }
        let extra = p.blocks[1].ops[0].clone();
        p.blocks[1].ops.push(extra);
        assert!(entry_guard(&p, Some(MachineBlockId(2))).is_none());
    }

    #[test]
    fn rejects_cfg_reentry_even_when_hidden_in_a_jump_table() {
        let mut p = program();
        p.blocks[2].terminator = MachineTerminator::JumpTable {
            index: MachineValue::Reg(MachineReg(4)),
            entries: collections::vec![MachineEdge {
                target: MachineBlockId(0),
                args: collections::vec![MachineValue::Reg(MachineReg(4))],
            }],
        };
        assert!(entry_guard(&p, Some(MachineBlockId(2))).is_none());
        p.blocks[2].terminator = MachineTerminator::Jump(MachineEdge {
            target: MachineBlockId(2),
            args: collections::vec![MachineValue::Reg(MachineReg(4))],
        });
        assert!(entry_guard(&p, Some(MachineBlockId(2))).is_some());
    }
}
