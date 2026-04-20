//! Constant-indexed memory fusion.
//!
//! Before: the SSA-level `i32.const N; f64.load` (the standard C/Rust
//! compiler-emitted global-variable access) lowers through
//! `fuse_indexed_memory` to:
//!
//! ```text
//! move.linear.gp rIDX <- imm32
//! indexed_load  rDST <- [BASE + rIDX(ZX32) + OFFSET]
//! ```
//!
//! After this pass:
//!
//! ```text
//! load rDST <- [BASE + (imm32 + OFFSET)]
//! ```
//!
//! On x86_64 this becomes one `mov/movsd disp32(base), reg` instruction
//! (~9 bytes) instead of the original 4-instruction sequence
//! `mov imm, %r10d; mov %r10d, %reg; add %r12, %reg; load (%reg)` (~20 bytes).
//! ARM64 and ARMv7 similarly prefer the non-indexed form when the index
//! collapses to a constant.
//!
//! Safety conditions:
//! - the `Move` source must be an `Imm64` (so the value is known at lowering
//!   time, not an arbitrary register copy);
//! - the index register must be dead after the memory op (otherwise we would
//!   have to keep the `Move` alive, which negates the saving);
//! - the combined displacement `imm_zx32 + offset` must fit in `i32` so a
//!   single `MachineAddr { base, offset }` can represent it.

use crate::collections;

use crate::vm::backend::BackendConfig;
use crate::vm::machine::machine_ir::{
    is_fp_reg, MachineAddr, MachineBlock, MachineIndexExtend, MachineInst, MachineInstKind,
    MachineMemWidth, MachineReg, MachineRegOwner, MachineStorageType, MachineTerminator,
    MachineValue,
};

use super::helpers::reg_live_after;

pub(super) fn fuse_const_indexed(block: &mut MachineBlock, config: BackendConfig) {
    let ops = core::mem::take(&mut block.ops);
    let term = &block.terminator;
    let mut out: collections::Vec<MachineInst> = collections::Vec::with_capacity(ops.len());
    let mut i = 0;

    while i < ops.len() {
        if let Some((fused, consumed)) = try_fuse(&ops[i..], term, config) {
            out.push(fused);
            i += consumed;
            continue;
        }
        out.push(ops[i].clone());
        i += 1;
    }

    block.ops = out;
}

fn try_fuse(
    ops: &[MachineInst],
    term: &MachineTerminator,
    config: BackendConfig,
) -> Option<(MachineInst, usize)> {
    if ops.len() < 2 {
        return None;
    }

    // [0]: move.linear.gp rIDX <- Imm64(imm)
    let (idx_reg, imm) = match ops[0].kind {
        MachineInstKind::Move {
            owner: MachineRegOwner::LinearValue,
            ty,
            dst,
            src: MachineValue::Imm64(bits),
        } if matches!(ty, MachineStorageType::GpWord | MachineStorageType::GpI64) => (dst, bits),
        _ => return None,
    };

    // Wasm addresses materialized from `i32.const` are zero-extended to 64.
    // Preserve that by taking the low 32 bits, then widening as u64 → i64.
    // If the move was for an i64 const (rare for memory addresses but possible
    // for `i32.const N` followed by extension), the high bits must be zero for
    // the fold to be safe.
    if (imm >> 32) != 0 {
        return None;
    }
    let imm32 = imm as u32;

    // [1]: indexed_load/store that uses rIDX as index with ZeroExtend32.
    match ops[1].kind {
        MachineInstKind::IndexedLoad {
            dst,
            base,
            index,
            index_extend: MachineIndexExtend::ZeroExtend32,
            offset,
            width,
            extension,
        } if index == idx_reg && base != idx_reg => {
            let total = combined_offset(imm32, offset)?;
            // rIDX must not be used past the indexed load (which is the last
            // inst we consume). `reg_live_after` scans `ops[2..]` + terminator.
            if reg_live_after(&ops[2..], term, idx_reg) {
                return None;
            }
            let ty = load_storage_type(dst, width, config);
            let fused = MachineInst {
                kind: MachineInstKind::Load {
                    owner: MachineRegOwner::LinearValue,
                    ty,
                    dst,
                    addr: MachineAddr { base, offset: total },
                    width,
                    extension,
                },
            };
            Some((fused, 2))
        }
        MachineInstKind::IndexedStore {
            base,
            index,
            index_extend: MachineIndexExtend::ZeroExtend32,
            offset,
            width,
            src,
        } if index == idx_reg
            && base != idx_reg
            && !matches!(src, MachineValue::Reg(r) if r == idx_reg) =>
        {
            let total = combined_offset(imm32, offset)?;
            if reg_live_after(&ops[2..], term, idx_reg) {
                return None;
            }
            let ty = store_storage_type(src, width, config);
            let fused = MachineInst {
                kind: MachineInstKind::Store {
                    ty,
                    addr: MachineAddr { base, offset: total },
                    width,
                    src,
                },
            };
            Some((fused, 2))
        }
        _ => None,
    }
}

/// Combine a zero-extended i32 wasm address with the memory op's compile-time
/// offset. The result must fit in `i32` to be representable in `MachineAddr`.
fn combined_offset(imm32: u32, offset: i32) -> Option<i32> {
    let combined = (imm32 as i64).wrapping_add(offset as i64);
    if (i32::MIN as i64..=i32::MAX as i64).contains(&combined) {
        Some(combined as i32)
    } else {
        None
    }
}

fn load_storage_type(
    dst: MachineReg,
    width: MachineMemWidth,
    config: BackendConfig,
) -> MachineStorageType {
    if is_fp_reg(dst, config) {
        match width {
            MachineMemWidth::U32 => MachineStorageType::Fp32,
            _ => MachineStorageType::Fp64,
        }
    } else {
        match width {
            MachineMemWidth::U64 => MachineStorageType::GpI64,
            _ => MachineStorageType::GpWord,
        }
    }
}

fn store_storage_type(
    src: MachineValue,
    width: MachineMemWidth,
    config: BackendConfig,
) -> MachineStorageType {
    if let MachineValue::Reg(reg) = src {
        if is_fp_reg(reg, config) {
            return match width {
                MachineMemWidth::U32 => MachineStorageType::Fp32,
                _ => MachineStorageType::Fp64,
            };
        }
    }
    match width {
        MachineMemWidth::U64 => MachineStorageType::GpI64,
        _ => MachineStorageType::GpWord,
    }
}
