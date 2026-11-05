//! Read-modify-write fusion of `load; op; store` to one memory word.
//!
//! ```text
//! indexed_load  t  <- [base + index + off]
//! iN.op         t' <- t, x        (add/sub/and/or/xor; x = imm or reg)
//! indexed_store [base + index + off] <- t'
//! ```
//!
//! becomes `OP [base + index + off], x` when neither `t` nor `t'` is read
//! afterwards. The fused instruction reads the same address first, so an
//! out-of-bounds access faults before anything is written, exactly as the
//! separate load would: guarded-memory traps resolve by function range and
//! fault address, not by instruction.

use crate::{
    error::WasmError,
    vm::jit::machine::machine_ir::{
        MachineIndexExtend, MachineInst, MachineInstKind, MachineIntBinaryOp, MachineIntWidth,
        MachineLoadExtension, MachineMemWidth, MachineReg, MachineValue,
    },
};

use super::{
    backend::X86_64Backend,
    enc::{self, MemAluOp},
    narrow_equality::dynamic_gp,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RmwOperand {
    Imm(i32),
    Reg(MachineReg),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RmwFusion {
    pub wide: bool,
    pub op: MemAluOp,
    pub base: MachineReg,
    pub index: MachineReg,
    pub index_extend: MachineIndexExtend,
    pub offset: i32,
    pub operand: RmwOperand,
    /// The loaded value and the stored result; both must be proven dead
    /// after the store by the caller.
    pub loaded: MachineReg,
    pub result: MachineReg,
}

fn mem_alu_op(op: MachineIntBinaryOp) -> Option<MemAluOp> {
    Some(match op {
        MachineIntBinaryOp::Add => MemAluOp::Add,
        MachineIntBinaryOp::Sub => MemAluOp::Sub,
        MachineIntBinaryOp::And => MemAluOp::And,
        MachineIntBinaryOp::Or => MemAluOp::Or,
        MachineIntBinaryOp::Xor => MemAluOp::Xor,
        _ => return None,
    })
}

fn rmw_operand(value: MachineValue, wide: bool) -> Option<RmwOperand> {
    match value {
        // IntBinary operands are GP registers by construction.
        MachineValue::Reg(reg) => Some(RmwOperand::Reg(reg)),
        // An i32 op uses the low 32 bits of the immediate as-is; an i64
        // op sign-extends imm32, so only those values qualify.
        MachineValue::Imm64(imm) if !wide => Some(RmwOperand::Imm(imm as u32 as i32)),
        MachineValue::Imm64(imm) => i32::try_from(imm as i64).ok().map(RmwOperand::Imm),
        _ => None,
    }
}

pub(super) fn rmw_fusion(
    load: &MachineInst,
    alu: &MachineInst,
    store: &MachineInst,
) -> Option<RmwFusion> {
    let MachineInstKind::IndexedLoad {
        dst: loaded,
        base,
        index,
        index_extend,
        offset,
        width: load_width,
        extension: MachineLoadExtension::None | MachineLoadExtension::ZeroExtend,
    } = load.kind
    else {
        return None;
    };
    let MachineInstKind::IntBinary {
        width,
        op,
        dst: result,
        lhs,
        rhs,
    } = alu.kind
    else {
        return None;
    };
    let MachineInstKind::IndexedStore {
        base: store_base,
        index: store_index,
        index_extend: store_extend,
        offset: store_offset,
        width: store_width,
        src: MachineValue::Reg(stored),
    } = store.kind
    else {
        return None;
    };
    let wide = match (load_width, width) {
        (MachineMemWidth::U32, MachineIntWidth::I32) => false,
        (MachineMemWidth::U64, MachineIntWidth::I64) => true,
        _ => return None,
    };
    let op = mem_alu_op(op)?;
    if store_width != load_width
        || (store_base, store_index, store_extend, store_offset)
            != (base, index, index_extend, offset)
        || stored != result
        || !dynamic_gp(loaded)
        || !dynamic_gp(result)
        || [base, index].contains(&loaded)
        || [base, index].contains(&result)
    {
        return None;
    }
    let commutative = op != MemAluOp::Sub;
    let other = if lhs == MachineValue::Reg(loaded) {
        rhs
    } else if commutative && rhs == MachineValue::Reg(loaded) {
        lhs
    } else {
        return None;
    };
    if other == MachineValue::Reg(loaded) {
        return None;
    }
    Some(RmwFusion {
        wide,
        op,
        base,
        index,
        index_extend,
        offset,
        operand: rmw_operand(other, wide)?,
        loaded,
        result,
    })
}

impl<'a> X86_64Backend<'a> {
    pub(super) fn lower_rmw(&mut self, fusion: &RmwFusion) -> Result<(), WasmError> {
        let base = self.map_gp_reg(fusion.base)?;
        let index = self.map_gp_reg(fusion.index)?;
        let operand = match fusion.operand {
            RmwOperand::Reg(reg) => Some(self.map_gp_reg(reg)?),
            RmwOperand::Imm(_) => None,
        };
        let scratch0 = self.gp_scratch.scoped_alloc().detach();
        let scratch1 = self.gp_scratch.scoped_alloc().detach();
        let index = match fusion.index_extend {
            MachineIndexExtend::None => index,
            MachineIndexExtend::ZeroExtend32 => {
                // The MachineIR contract leaves the upper half undefined;
                // extend into a scratch that is neither base nor operand.
                let idx = if base == *scratch0 || operand == Some(*scratch0) {
                    *scratch1
                } else {
                    *scratch0
                };
                enc::mov_rr_32(&mut self.core.text, idx, index);
                idx
            }
        };
        let text = &mut self.core.text;
        match (fusion.operand, operand) {
            (RmwOperand::Imm(imm), _) => enc::alu_mi_idx(
                text,
                fusion.wide,
                fusion.op,
                base,
                index,
                fusion.offset,
                imm,
            ),
            (RmwOperand::Reg(_), Some(src)) => enc::alu_mr_idx(
                text,
                fusion.wide,
                fusion.op,
                base,
                index,
                fusion.offset,
                src,
            ),
            (RmwOperand::Reg(_), None) => unreachable!("mapped above"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(dst: u16, width: MachineMemWidth) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IndexedLoad {
                dst: MachineReg(dst),
                base: MachineReg(2),
                index: MachineReg(9),
                index_extend: MachineIndexExtend::None,
                offset: 16,
                width,
                extension: MachineLoadExtension::ZeroExtend,
            },
        }
    }

    fn alu(
        op: MachineIntBinaryOp,
        width: MachineIntWidth,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width,
                op,
                dst: MachineReg(7),
                lhs,
                rhs,
            },
        }
    }

    fn store(width: MachineMemWidth, offset: i32) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IndexedStore {
                base: MachineReg(2),
                index: MachineReg(9),
                index_extend: MachineIndexExtend::None,
                offset,
                width,
                src: MachineValue::Reg(MachineReg(7)),
            },
        }
    }

    const T: MachineValue = MachineValue::Reg(MachineReg(7));

    #[test]
    fn fuses_word_and_doubleword_updates() {
        let fused = rmw_fusion(
            &load(7, MachineMemWidth::U32),
            &alu(
                MachineIntBinaryOp::Or,
                MachineIntWidth::I32,
                T,
                MachineValue::Imm64(0xffdf_ffff),
            ),
            &store(MachineMemWidth::U32, 16),
        )
        .expect("i32 or");
        assert_eq!(fused.operand, RmwOperand::Imm(0xffdf_ffffu32 as i32));
        assert!(!fused.wide);
        // Commuted register operand.
        let fused = rmw_fusion(
            &load(7, MachineMemWidth::U64),
            &alu(
                MachineIntBinaryOp::Add,
                MachineIntWidth::I64,
                MachineValue::Reg(MachineReg(8)),
                T,
            ),
            &store(MachineMemWidth::U64, 16),
        )
        .expect("i64 add");
        assert_eq!(fused.operand, RmwOperand::Reg(MachineReg(8)));
        assert!(fused.wide);
    }

    #[test]
    fn rejects_mismatches() {
        let i32_or = alu(
            MachineIntBinaryOp::Or,
            MachineIntWidth::I32,
            T,
            MachineValue::Imm64(1),
        );
        // Different address.
        assert_eq!(
            rmw_fusion(
                &load(7, MachineMemWidth::U32),
                &i32_or,
                &store(MachineMemWidth::U32, 20)
            ),
            None
        );
        // Narrow access.
        assert_eq!(
            rmw_fusion(
                &load(7, MachineMemWidth::U8),
                &i32_or,
                &store(MachineMemWidth::U8, 16)
            ),
            None
        );
        // Width mismatch between op and memory.
        assert_eq!(
            rmw_fusion(
                &load(7, MachineMemWidth::U64),
                &i32_or,
                &store(MachineMemWidth::U64, 16)
            ),
            None
        );
        // Sub is not commutative.
        let rsub = alu(
            MachineIntBinaryOp::Sub,
            MachineIntWidth::I32,
            MachineValue::Imm64(1),
            T,
        );
        assert_eq!(
            rmw_fusion(
                &load(7, MachineMemWidth::U32),
                &rsub,
                &store(MachineMemWidth::U32, 16)
            ),
            None
        );
        // Not an ALU op the memory form supports.
        let mul = alu(
            MachineIntBinaryOp::Mul,
            MachineIntWidth::I32,
            T,
            MachineValue::Imm64(3),
        );
        assert_eq!(
            rmw_fusion(
                &load(7, MachineMemWidth::U32),
                &mul,
                &store(MachineMemWidth::U32, 16)
            ),
            None
        );
        // An i64 immediate outside sign-extended imm32.
        let big = alu(
            MachineIntBinaryOp::Add,
            MachineIntWidth::I64,
            T,
            MachineValue::Imm64(0x8000_0000),
        );
        assert_eq!(
            rmw_fusion(
                &load(7, MachineMemWidth::U64),
                &big,
                &store(MachineMemWidth::U64, 16)
            ),
            None
        );
        // The loaded value is also the other operand (x + x).
        let double = alu(MachineIntBinaryOp::Add, MachineIntWidth::I32, T, T);
        assert_eq!(
            rmw_fusion(
                &load(7, MachineMemWidth::U32),
                &double,
                &store(MachineMemWidth::U32, 16)
            ),
            None
        );
        // The loaded register is the address index.
        assert_eq!(
            rmw_fusion(
                &load(9, MachineMemWidth::U32),
                &alu(
                    MachineIntBinaryOp::Or,
                    MachineIntWidth::I32,
                    MachineValue::Reg(MachineReg(9)),
                    MachineValue::Imm64(1)
                ),
                &store(MachineMemWidth::U32, 16)
            ),
            None
        );
    }
}
