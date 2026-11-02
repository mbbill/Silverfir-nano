//! Float comparisons read straight from UCOMISS/UCOMISD flags.
//!
//! UCOMIS sets CF=1 for `lhs < rhs` *and* for unordered, ZF=1 for equal and
//! for unordered, and PF=1 only for unordered. Wasm ordered relations are
//! false on NaN, so comparing `lt`/`le` with swapped operands turns every
//! ordered relation into one condition (`A` or `AE`) that already excludes
//! unordered. Only `eq` (ZF=1 and PF=0) and `ne` (ZF=0 or PF=1) need two
//! flag tests.
//!
//! A boolean whose only consumer is the block's branch, or an adjacent FP
//! select, is never materialized: the consumer jumps on the flags.

use crate::{
    error::WasmError,
    vm::jit::machine::machine_ir::{
        MachineBlock, MachineBranchCond, MachineCompareKind, MachineFloatWidth, MachineInst,
        MachineInstKind, MachineIntWidth, MachineReg, MachineStorageType, MachineTerminator,
        MachineValue,
    },
};

use super::{
    backend::X86_64Backend,
    enc::{self, Cc},
    narrow_equality::{dead_on_edge, dynamic_gp},
};

/// How a Wasm float comparison reads from the flags of one UCOMIS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FloatCond {
    /// True exactly when this condition code holds.
    Single(Cc),
    /// Ordered and equal: ZF=1 and PF=0.
    Equal,
    /// Unordered or not equal: ZF=0 or PF=1.
    NotEqual,
}

/// Whether UCOMIS takes `(rhs, lhs)` rather than `(lhs, rhs)`, and the
/// condition that is then true exactly when the Wasm comparison is.
pub(super) fn float_cond(kind: MachineCompareKind) -> (bool, FloatCond) {
    match kind {
        MachineCompareKind::Eq => (false, FloatCond::Equal),
        MachineCompareKind::Ne => (false, FloatCond::NotEqual),
        MachineCompareKind::Gt => (false, FloatCond::Single(Cc::A)),
        MachineCompareKind::Ge => (false, FloatCond::Single(Cc::AE)),
        MachineCompareKind::Lt => (true, FloatCond::Single(Cc::A)),
        MachineCompareKind::Le => (true, FloatCond::Single(Cc::AE)),
    }
}

/// A block ending in `FloatCompare` whose boolean is read only by the
/// branch terminator, either directly or through `== 0` / `!= 0`, and is
/// dead on both edges. The compare op is not emitted; the branch emits the
/// UCOMIS and jumps on its flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FloatBranch {
    pub index: usize,
    pub width: MachineFloatWidth,
    pub kind: MachineCompareKind,
    pub lhs: MachineValue,
    pub rhs: MachineValue,
    /// The then-edge is taken when the comparison is false (`bool == 0`).
    pub negate: bool,
}

pub(super) fn float_compare_branch(
    block: &MachineBlock,
    blocks: &[MachineBlock],
) -> Option<FloatBranch> {
    let index = block.ops.len().checked_sub(1)?;
    let MachineInstKind::FloatCompare {
        width,
        kind,
        dst,
        lhs,
        rhs,
    } = block.ops[index].kind
    else {
        return None;
    };
    let MachineTerminator::Branch {
        cond,
        then_edge,
        else_edge,
    } = &block.terminator
    else {
        return None;
    };
    let negate = match *cond {
        MachineBranchCond::Value(MachineValue::Reg(reg)) if reg == dst => false,
        MachineBranchCond::IntCompare {
            width: MachineIntWidth::I32,
            kind: kind @ (MachineCompareKind::Eq | MachineCompareKind::Ne),
            lhs: a,
            rhs: b,
            ..
        } if (a == MachineValue::Reg(dst) && b == MachineValue::Imm64(0))
            || (a == MachineValue::Imm64(0) && b == MachineValue::Reg(dst)) =>
        {
            kind == MachineCompareKind::Eq
        }
        _ => return None,
    };
    if !dynamic_gp(dst)
        || !dead_on_edge(blocks, then_edge, dst)
        || !dead_on_edge(blocks, else_edge, dst)
    {
        return None;
    }
    Some(FloatBranch {
        index,
        width,
        kind,
        lhs,
        rhs,
        negate,
    })
}

/// A `FloatCompare + Select` pair where the boolean is the FP select's
/// condition. The caller must additionally prove `bool_reg` dead after the
/// select.
pub(super) struct FloatCompareSelect {
    pub width: MachineFloatWidth,
    pub kind: MachineCompareKind,
    pub bool_reg: MachineReg,
    pub lhs: MachineValue,
    pub rhs: MachineValue,
}

pub(super) fn float_compare_select_fusion(
    compare: &MachineInst,
    select: &MachineInst,
) -> Option<FloatCompareSelect> {
    let MachineInstKind::FloatCompare {
        width,
        kind,
        dst: bool_reg,
        lhs,
        rhs,
    } = compare.kind
    else {
        return None;
    };
    let MachineInstKind::Select {
        ty,
        cond: MachineValue::Reg(cond),
        on_true,
        on_false,
        ..
    } = select.kind
    else {
        return None;
    };
    if ty.float_width().is_none()
        || cond != bool_reg
        || on_true == MachineValue::Reg(bool_reg)
        || on_false == MachineValue::Reg(bool_reg)
    {
        return None;
    }
    Some(FloatCompareSelect {
        width,
        kind,
        bool_reg,
        lhs,
        rhs,
    })
}

impl<'a> X86_64Backend<'a> {
    /// Emit the UCOMIS for `lhs kind rhs`; the returned condition is true
    /// exactly when the Wasm comparison is.
    pub(super) fn emit_float_compare_flags(
        &mut self,
        width: MachineFloatWidth,
        kind: MachineCompareKind,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<FloatCond, WasmError> {
        let (swap, cond) = float_cond(kind);
        let gp0 = self.gp_scratch.scoped_alloc().detach();
        let gp1 = self.gp_scratch.scoped_alloc().detach();
        let fp0 = self.fp_scratch.scoped_alloc().detach();
        let fp1 = self.fp_scratch.scoped_alloc().detach();
        let lhs_fp = self.prepare_float_operand(width, lhs, *gp0, *fp0)?;
        // When lhs already lives in a mapped FP register (not `fp0`), `fp0`
        // is free for rhs; otherwise use `fp1` so lhs is not clobbered.
        let rhs_scratch = if lhs_fp != *fp0 { *fp0 } else { *fp1 };
        let rhs_fp = self.prepare_float_operand(width, rhs, *gp1, rhs_scratch)?;
        let (first, second) = if swap {
            (rhs_fp, lhs_fp)
        } else {
            (lhs_fp, rhs_fp)
        };
        self.ucomis(width, first as u8, second as u8);
        Ok(cond)
    }

    /// Jump to `target` when the comparison's truth equals `when`; fall
    /// through otherwise. Leaves EFLAGS unchanged.
    pub(super) fn emit_float_jump(&mut self, cond: FloatCond, when: bool, target: usize) {
        match (cond, when) {
            (FloatCond::Single(cc), true) => self.emit_jcc(cc, target),
            (FloatCond::Single(cc), false) => self.emit_jcc(cc.invert(), target),
            // ZF=1 and PF=0.
            (FloatCond::Equal, true) | (FloatCond::NotEqual, false) => {
                let unordered = self.core.new_label();
                self.emit_jcc(Cc::P, unordered);
                self.emit_jcc(Cc::E, target);
                self.core.bind_label(unordered);
            }
            // ZF=0 or PF=1.
            (FloatCond::Equal, false) | (FloatCond::NotEqual, true) => {
                self.emit_jcc(Cc::NE, target);
                self.emit_jcc(Cc::P, target);
            }
        }
    }

    pub(super) fn lower_float_compare(
        &mut self,
        width: MachineFloatWidth,
        kind: MachineCompareKind,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_gp = self.map_gp_reg(dst)?;
        let cond = self.emit_float_compare_flags(width, kind, lhs, rhs)?;
        let other = self.gp_scratch.scoped_alloc().detach();
        let text = &mut self.core.text;
        match cond {
            FloatCond::Single(cc) => enc::setcc(text, cc, dst_gp),
            FloatCond::Equal => {
                enc::setcc(text, Cc::E, dst_gp);
                enc::setcc(text, Cc::NP, *other);
                enc::and_rr_32(text, dst_gp, *other);
            }
            FloatCond::NotEqual => {
                enc::setcc(text, Cc::NE, dst_gp);
                enc::setcc(text, Cc::P, *other);
                enc::or_rr_32(text, dst_gp, *other);
            }
        }
        enc::movzx_r32_r8(text, dst_gp, dst_gp);
        Ok(())
    }

    /// Branch on a planned trailing float compare (see [`FloatBranch`]).
    pub(super) fn lower_float_branch(
        &mut self,
        plan: FloatBranch,
        then_label: Option<usize>,
        else_label: Option<usize>,
    ) -> Result<(), WasmError> {
        if then_label.is_none() && else_label.is_none() {
            return Ok(());
        }
        let cond = self.emit_float_compare_flags(plan.width, plan.kind, plan.lhs, plan.rhs)?;
        let then_when = !plan.negate;
        match (then_label, else_label) {
            (Some(then_label), None) => self.emit_float_jump(cond, then_when, then_label),
            (None, Some(else_label)) => self.emit_float_jump(cond, !then_when, else_label),
            (Some(then_label), Some(else_label)) => {
                self.emit_float_jump(cond, then_when, then_label);
                self.emit_jmp(else_label);
            }
            (None, None) => unreachable!("handled above"),
        }
        Ok(())
    }

    pub(super) fn lower_fused_float_compare_select(
        &mut self,
        fusion: &FloatCompareSelect,
        ty: MachineStorageType,
        dst: MachineReg,
        on_true: MachineValue,
        on_false: MachineValue,
    ) -> Result<(), WasmError> {
        let width = ty
            .float_width()
            .ok_or_else(|| WasmError::internal("float compare-select fusion on a GP select"))?;
        let cond =
            self.emit_float_compare_flags(fusion.width, fusion.kind, fusion.lhs, fusion.rhs)?;
        self.lower_float_select_on_flags(width, cond, dst, on_true, on_false)
    }

    /// `dst = cond ? on_true : on_false` for an FP value, with `cond`
    /// already in EFLAGS. Operand preparation (register reads, literal-pool
    /// loads, XORPS zeroing, GP-to-XMM moves) leaves EFLAGS intact.
    pub(super) fn lower_float_select_on_flags(
        &mut self,
        width: MachineFloatWidth,
        cond: FloatCond,
        dst: MachineReg,
        on_true: MachineValue,
        on_false: MachineValue,
    ) -> Result<(), WasmError> {
        let dst_fp = self.map_fp_reg(dst)? as u8;
        let gp0 = self.gp_scratch.scoped_alloc().detach();
        let gp1 = self.gp_scratch.scoped_alloc().detach();
        let fp0 = self.fp_scratch.scoped_alloc().detach();
        let fp1 = self.fp_scratch.scoped_alloc().detach();
        let true_fp = self.prepare_float_operand(width, on_true, *gp0, *fp0)? as u8;
        let false_fp = self.prepare_float_operand(width, on_false, *gp1, *fp1)? as u8;
        let done = self.core.new_label();
        if false_fp == dst_fp {
            self.emit_float_jump(cond, false, done);
            if true_fp != dst_fp {
                enc::movaps_rr(&mut self.core.text, dst_fp, true_fp);
            }
        } else if true_fp == dst_fp {
            self.emit_float_jump(cond, true, done);
            enc::movaps_rr(&mut self.core.text, dst_fp, false_fp);
        } else {
            enc::movaps_rr(&mut self.core.text, dst_fp, false_fp);
            self.emit_float_jump(cond, false, done);
            enc::movaps_rr(&mut self.core.text, dst_fp, true_fp);
        }
        self.core.bind_label(done);
        self.core.set_fp_reg_width(dst, width)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collections;
    use crate::vm::jit::machine::machine_ir::{MachineBlockId, MachineEdge, MachineSign};

    /// Flags UCOMIS produces for `a` vs `b`: (CF, ZF, PF).
    fn ucomis_flags(a: f64, b: f64) -> (bool, bool, bool) {
        if a.is_nan() || b.is_nan() {
            (true, true, true)
        } else if a < b {
            (true, false, false)
        } else if a == b {
            (false, true, false)
        } else {
            (false, false, false)
        }
    }

    fn holds(cc: Cc, (cf, zf, pf): (bool, bool, bool)) -> bool {
        match cc {
            Cc::A => !cf && !zf,
            Cc::AE => !cf,
            Cc::B => cf,
            Cc::BE => cf || zf,
            Cc::E => zf,
            Cc::NE => !zf,
            Cc::P => pf,
            Cc::NP => !pf,
            _ => unreachable!("not a UCOMIS condition"),
        }
    }

    fn truth(cond: FloatCond, flags: (bool, bool, bool)) -> bool {
        match cond {
            FloatCond::Single(cc) => holds(cc, flags),
            FloatCond::Equal => holds(Cc::E, flags) && holds(Cc::NP, flags),
            FloatCond::NotEqual => holds(Cc::NE, flags) || holds(Cc::P, flags),
        }
    }

    #[test]
    fn every_kind_matches_wasm_semantics_including_nan() {
        let values = [
            f64::NAN,
            f64::NEG_INFINITY,
            -1.5,
            -0.0,
            0.0,
            2.0,
            f64::INFINITY,
        ];
        let kinds = [
            MachineCompareKind::Eq,
            MachineCompareKind::Ne,
            MachineCompareKind::Lt,
            MachineCompareKind::Le,
            MachineCompareKind::Gt,
            MachineCompareKind::Ge,
        ];
        for kind in kinds {
            let (swap, cond) = float_cond(kind);
            for a in values {
                for b in values {
                    let expected = match kind {
                        MachineCompareKind::Eq => a == b,
                        MachineCompareKind::Ne => a != b,
                        MachineCompareKind::Lt => a < b,
                        MachineCompareKind::Le => a <= b,
                        MachineCompareKind::Gt => a > b,
                        MachineCompareKind::Ge => a >= b,
                    };
                    let flags = if swap {
                        ucomis_flags(b, a)
                    } else {
                        ucomis_flags(a, b)
                    };
                    assert_eq!(truth(cond, flags), expected, "{kind:?} {a} {b}");
                    if let FloatCond::Single(cc) = cond {
                        assert_eq!(holds(cc.invert(), flags), !expected, "{kind:?} {a} {b}");
                    }
                }
            }
        }
    }

    fn edge(id: u32) -> MachineEdge {
        MachineEdge {
            target: MachineBlockId(id),
            args: collections::vec![],
        }
    }

    fn program(
        cond: MachineBranchCond,
        successor_reads_bool: bool,
    ) -> collections::Vec<MachineBlock> {
        let compare = MachineInst {
            kind: MachineInstKind::FloatCompare {
                width: MachineFloatWidth::F64,
                kind: MachineCompareKind::Lt,
                dst: MachineReg(7),
                lhs: MachineValue::Reg(MachineReg(40)),
                rhs: MachineValue::Imm64(0),
            },
        };
        let mut successor_ops = collections::vec![];
        if successor_reads_bool {
            successor_ops.push(MachineInst {
                kind: MachineInstKind::IntCompare {
                    width: MachineIntWidth::I32,
                    kind: MachineCompareKind::Eq,
                    sign: MachineSign::Unsigned,
                    dst: MachineReg(8),
                    lhs: MachineValue::Reg(MachineReg(7)),
                    rhs: MachineValue::Imm64(1),
                },
            });
        }
        collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::vec![],
                ops: collections::vec![compare],
                terminator: MachineTerminator::Branch {
                    cond,
                    then_edge: edge(1),
                    else_edge: edge(2),
                },
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::vec![],
                ops: successor_ops,
                terminator: MachineTerminator::Return,
            },
            MachineBlock {
                id: MachineBlockId(2),
                params: collections::vec![],
                ops: collections::vec![],
                terminator: MachineTerminator::Return,
            },
        ]
    }

    fn eq_zero(kind: MachineCompareKind, swapped: bool) -> MachineBranchCond {
        let (lhs, rhs) = if swapped {
            (MachineValue::Imm64(0), MachineValue::Reg(MachineReg(7)))
        } else {
            (MachineValue::Reg(MachineReg(7)), MachineValue::Imm64(0))
        };
        MachineBranchCond::IntCompare {
            width: MachineIntWidth::I32,
            kind,
            sign: MachineSign::Unsigned,
            lhs,
            rhs,
        }
    }

    #[test]
    fn plans_direct_and_zero_tested_branches() {
        let direct = program(
            MachineBranchCond::Value(MachineValue::Reg(MachineReg(7))),
            false,
        );
        let plan = float_compare_branch(&direct[0], &direct).expect("direct branch");
        assert!(!plan.negate);
        assert_eq!(plan.index, 0);
        for swapped in [false, true] {
            let eqz = program(eq_zero(MachineCompareKind::Eq, swapped), false);
            assert!(float_compare_branch(&eqz[0], &eqz).expect("eqz").negate);
            let nez = program(eq_zero(MachineCompareKind::Ne, swapped), false);
            assert!(!float_compare_branch(&nez[0], &nez).expect("nez").negate);
        }
    }

    #[test]
    fn keeps_bool_that_a_successor_reads_or_that_is_compared_otherwise() {
        let live = program(
            MachineBranchCond::Value(MachineValue::Reg(MachineReg(7))),
            true,
        );
        assert_eq!(float_compare_branch(&live[0], &live), None);
        let ordered = program(eq_zero(MachineCompareKind::Lt, false), false);
        assert_eq!(float_compare_branch(&ordered[0], &ordered), None);
        let other_reg = program(
            MachineBranchCond::Value(MachineValue::Reg(MachineReg(9))),
            false,
        );
        assert_eq!(float_compare_branch(&other_reg[0], &other_reg), None);
    }
}
