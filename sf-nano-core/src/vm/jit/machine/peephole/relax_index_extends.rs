//! Elide provably redundant index zero-extensions on indexed memory ops.
//!
//! `IndexedLoad`/`IndexedStore` with `MachineIndexExtend::ZeroExtend32`
//! obligate the backend to zero-extend the 32-bit index before use. On
//! backends where every 32-bit integer instruction already writes a
//! zero-extended destination (x86_64: all r32-form ops clear bits 63:32),
//! that obligation is vacuous whenever the index register's most recent
//! in-block definition is such an instruction — the extend can be relaxed
//! to `None` and the backend indexes the register directly, saving one
//! `mov` per memory access (three to four per iteration in the stream
//! kernels).
//!
//! Gated on `BackendConfig::gp32_defs_zero_extend`; backends whose 32-bit
//! ops sign-extend (riscv64) or that get the extension for free in the
//! addressing mode (arm64 UXTW) leave it off.
//!
//! Soundness: a register is tracked as clean only from a whitelisted
//! 32-bit-form definition to its next redefinition, within one block.
//! Block entry states (params, values live across edges) are never
//! trusted.

use crate::vm::jit::machine::machine_ir::{
    MachineBlock, MachineIndexExtend, MachineInstKind, MachineIntWidth, MachineLoadExtension,
    MachineMemWidth, MachineReg,
};

/// True when this instruction's destination is written by a 32-bit-form
/// instruction on a `gp32_defs_zero_extend` backend, leaving the upper
/// half zero.
fn def_zero_extends(kind: &MachineInstKind) -> Option<MachineReg> {
    match kind {
        MachineInstKind::IntBinary {
            width: MachineIntWidth::I32,
            dst,
            ..
        }
        | MachineInstKind::IntUnary {
            width: MachineIntWidth::I32,
            dst,
            ..
        } => Some(*dst),
        MachineInstKind::IntCompare { dst, .. } => Some(*dst),
        MachineInstKind::Load {
            dst,
            width: MachineMemWidth::U8 | MachineMemWidth::U16 | MachineMemWidth::U32,
            extension: MachineLoadExtension::None | MachineLoadExtension::ZeroExtend,
            ..
        }
        | MachineInstKind::IndexedLoad {
            dst,
            width: MachineMemWidth::U8 | MachineMemWidth::U16 | MachineMemWidth::U32,
            extension: MachineLoadExtension::None | MachineLoadExtension::ZeroExtend,
            ..
        } => Some(*dst),
        _ => None,
    }
}

pub(super) fn relax_index_extends(block: &mut MachineBlock) {
    // Bitset over MachineReg numbers; register files are small.
    let mut clean = 0u128;
    let bit = |reg: MachineReg| 1u128 << (reg.0 as u32 & 127);

    for inst in &mut block.ops {
        // Uses first: relax this op's own index against the state built
        // by earlier ops.
        match &mut inst.kind {
            MachineInstKind::IndexedLoad {
                index,
                index_extend: index_extend @ MachineIndexExtend::ZeroExtend32,
                ..
            }
            | MachineInstKind::IndexedStore {
                index,
                index_extend: index_extend @ MachineIndexExtend::ZeroExtend32,
                ..
            } => {
                if clean & bit(*index) != 0 {
                    *index_extend = MachineIndexExtend::None;
                }
            }
            _ => {}
        }

        // Defs second: every redefinition invalidates; whitelisted
        // 32-bit-form defs re-establish cleanliness.
        inst.kind.for_each_defined_reg(|reg| clean &= !bit(reg));
        if let Some(dst) = def_zero_extends(&inst.kind) {
            clean |= bit(dst);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collections;
    use crate::vm::jit::machine::machine_ir::{
        MachineAddr, MachineBlockId, MachineInst, MachineIntBinaryOp, MachineStorageType,
        MachineTerminator, MachineValue,
    };

    fn block(ops: collections::Vec<MachineInst>) -> MachineBlock {
        MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops,
            terminator: MachineTerminator::Return,
        }
    }

    fn add32(dst: u16, lhs: u16) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: MachineIntWidth::I32,
                op: MachineIntBinaryOp::Add,
                dst: MachineReg(dst),
                lhs: MachineValue::Reg(MachineReg(lhs)),
                rhs: MachineValue::Imm64(8),
            },
        }
    }

    fn add64(dst: u16, lhs: u16) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: MachineIntWidth::I64,
                op: MachineIntBinaryOp::Add,
                dst: MachineReg(dst),
                lhs: MachineValue::Reg(MachineReg(lhs)),
                rhs: MachineValue::Imm64(8),
            },
        }
    }

    fn indexed_load(index: u16) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IndexedLoad {
                dst: MachineReg(9),
                base: MachineReg(2),
                index: MachineReg(index),
                index_extend: MachineIndexExtend::ZeroExtend32,
                offset: 0,
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::None,
            },
        }
    }

    fn extend_of(block: &MachineBlock, index: usize) -> MachineIndexExtend {
        match &block.ops[index].kind {
            MachineInstKind::IndexedLoad { index_extend, .. }
            | MachineInstKind::IndexedStore { index_extend, .. } => *index_extend,
            _ => panic!("not an indexed memory op"),
        }
    }

    fn store_addr_dummy() -> MachineInst {
        MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: MachineAddr {
                    base: MachineReg(2),
                    offset: 0,
                },
                width: MachineMemWidth::U64,
                src: MachineValue::Reg(MachineReg(9)),
            },
        }
    }

    #[test]
    fn relaxes_after_32bit_def() {
        let mut b = block(collections::vec![add32(5, 5), indexed_load(5)]);
        relax_index_extends(&mut b);
        assert_eq!(extend_of(&b, 1), MachineIndexExtend::None);
    }

    #[test]
    fn keeps_extend_without_in_block_def() {
        let mut b = block(collections::vec![indexed_load(5)]);
        relax_index_extends(&mut b);
        assert_eq!(extend_of(&b, 0), MachineIndexExtend::ZeroExtend32);
    }

    #[test]
    fn keeps_extend_after_64bit_def() {
        let mut b = block(collections::vec![add64(5, 5), indexed_load(5)]);
        relax_index_extends(&mut b);
        assert_eq!(extend_of(&b, 1), MachineIndexExtend::ZeroExtend32);
    }

    #[test]
    fn redefinition_invalidates_cleanliness() {
        let mut b = block(collections::vec![add32(5, 5), add64(5, 5), indexed_load(5),]);
        relax_index_extends(&mut b);
        assert_eq!(extend_of(&b, 2), MachineIndexExtend::ZeroExtend32);
    }

    #[test]
    fn cleanliness_survives_unrelated_ops() {
        let mut b = block(collections::vec![
            add32(5, 5),
            store_addr_dummy(),
            indexed_load(5),
        ]);
        relax_index_extends(&mut b);
        assert_eq!(extend_of(&b, 2), MachineIndexExtend::None);
    }

    #[test]
    fn clean_load_result_relaxes_following_index_use() {
        let mut b = block(collections::vec![indexed_load(5), {
            MachineInst {
                kind: MachineInstKind::IndexedLoad {
                    dst: MachineReg(10),
                    base: MachineReg(2),
                    index: MachineReg(9),
                    index_extend: MachineIndexExtend::ZeroExtend32,
                    offset: 0,
                    width: MachineMemWidth::U32,
                    extension: MachineLoadExtension::None,
                },
            }
        }]);
        relax_index_extends(&mut b);
        // First load's index has no in-block def; second indexes the
        // first's zero-extending U32 result.
        assert_eq!(extend_of(&b, 0), MachineIndexExtend::ZeroExtend32);
        assert_eq!(extend_of(&b, 1), MachineIndexExtend::None);
    }
}
