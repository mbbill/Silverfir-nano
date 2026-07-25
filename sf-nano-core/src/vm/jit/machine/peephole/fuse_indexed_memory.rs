//! Indexed memory address fusion pass.
//!
//! Fuses address computation + load/store into `IndexedLoad`/`IndexedStore`.
//!
//! **Pattern A** (with zero-extend, 3->1):
//! ```text
//! cvt.I64ExtendI32U  r <- addr
//! [i64.Add           r <- r IMM]      // optional: Wasm load/store offset
//! i64.Add            r <- base r
//! load/store         .. [r + 0]
//! ```
//! -> `IndexedLoad/Store { base, index=addr, extend=ZeroExtend32, offset }`
//!
//! **Pattern B** (no extend, 2->1):
//! ```text
//! i64.Add            r <- base index
//! load/store         .. [r + 0]
//! ```
//! -> `IndexedLoad/Store { base, index, extend=None, offset=0 }`
//!
//! **Pattern C** (constant address, 4/5->1):
//! ```text
//! move               r <- IMM           // constant Wasm address
//! cvt.I64ExtendI32U  r <- r
//! [i64.Add           r <- r IMM]        // optional: Wasm load/store offset
//! i64.Add            r <- base r
//! load/store         .. [r + 0]
//! ```
//! -> the original `Load`/`Store` with `addr = [base + zext32(IMM) + offset]`,
//! dropping the materialization and address arithmetic entirely.

use crate::vm::jit::machine::machine_ir::{
    MachineAddr, MachineBlock, MachineConvertOp, MachineIndexExtend, MachineInst, MachineInstKind,
    MachineIntBinaryOp, MachineIntWidth, MachineStorageType, MachineTerminator, MachineValue,
};

use super::helpers::reg_live_after;

pub(super) fn fuse_indexed_memory(block: &mut MachineBlock) {
    if block.ops.is_empty() {
        return;
    }

    let term = &block.terminator;
    let len = block.ops.len();
    let mut read = 0;
    let mut write = 0;

    while read < len {
        // --- Pattern C: const_move + cvt + [offset_add] + base_add + load/store ---
        if let Some(consumed) = try_fuse_const_addr(&block.ops[read..len], term) {
            block.ops[write] = consumed.fused;
            read += consumed.skip;
            write += 1;
            continue;
        }

        // --- Pattern A: cvt + [offset_add] + base_add + load/store ---
        if let Some(consumed) = try_fuse_uxtw_indexed(&block.ops[read..len], term) {
            block.ops[write] = consumed.fused;
            read += consumed.skip;
            write += 1;
            continue;
        }

        // --- Pattern B: base_add + load/store ---
        if let Some(consumed) = try_fuse_indexed(&block.ops[read..len], term) {
            block.ops[write] = consumed.fused;
            read += consumed.skip;
            write += 1;
            continue;
        }

        if write != read {
            block.ops[write] = block.ops[read].clone();
        }
        read += 1;
        write += 1;
    }

    block.ops.truncate(write);
}

struct FusedResult {
    fused: MachineInst,
    skip: usize,
}

/// Try to fuse `const_move + cvt.I64ExtendI32U + [offset_add] + base_add +
/// load/store` into the original load/store with a pure `base + imm` address.
///
/// The Wasm address is a constant, so the whole address computation folds into
/// the memory op's immediate offset. Semantics are exact: the effective address
/// is `base + zext32(IMM) + wasm_offset`, identical to the register sequence.
fn try_fuse_const_addr(ops: &[MachineInst], term: &MachineTerminator) -> Option<FusedResult> {
    // [0] move const_dst <- IMM  (GP constant materialization)
    let (const_dst, const_imm) = match ops.get(0)?.kind {
        MachineInstKind::Move {
            ty: MachineStorageType::GpWord | MachineStorageType::GpI64,
            dst,
            src: MachineValue::Imm64(imm),
            ..
        } => (dst, imm),
        _ => return None,
    };

    // [1] cvt.I64ExtendI32U ext_dst <- const_dst
    let ext_dst = match ops.get(1)?.kind {
        MachineInstKind::Convert {
            op: MachineConvertOp::I64ExtendI32U,
            dst,
            src: MachineValue::Reg(src),
        } if src == const_dst => dst,
        _ => return None,
    };

    // [2] optional: i64.Add ext_dst <- ext_dst IMM  (Wasm offset)
    let (wasm_offset, offset_count) = match ops.get(2)?.kind {
        MachineInstKind::IntBinary {
            width: MachineIntWidth::I64,
            op: MachineIntBinaryOp::Add,
            dst,
            lhs: MachineValue::Reg(lhs),
            rhs: MachineValue::Imm64(imm),
        } if dst == ext_dst && lhs == ext_dst && imm <= i32::MAX as u64 => (imm, 1),
        _ => (0u64, 0),
    };

    let base_idx = 2 + offset_count;

    // [base_idx] i64.Add ext_dst <- base ext_dst
    let base_reg = match ops.get(base_idx)?.kind {
        MachineInstKind::IntBinary {
            width: MachineIntWidth::I64,
            op: MachineIntBinaryOp::Add,
            dst,
            lhs: MachineValue::Reg(base),
            rhs: MachineValue::Reg(rhs),
        } if dst == ext_dst && rhs == ext_dst => base,
        _ => return None,
    };
    // The constant clobbered `const_dst`; folding reads `base` directly, so the
    // base must be a different register still holding its original value.
    if base_reg == const_dst {
        return None;
    }

    // Combined immediate must fit the memory op's i32 offset field. Backends
    // fall back to scratch materialization for offsets beyond their addressing
    // modes, which is never worse than the original register sequence.
    let combined = u64::from(const_imm as u32) + wasm_offset;
    if combined > i32::MAX as u64 {
        return None;
    }
    let combined = combined as i32;

    let mem_idx = base_idx + 1;
    let later = if ops.len() > mem_idx + 1 {
        &ops[mem_idx + 1..]
    } else {
        &[]
    };

    // [mem_idx] load or store using ext_dst with addr.offset == 0. The original
    // op is kept verbatim (owner/ty/width/extension) with only `addr` rewritten.
    match ops.get(mem_idx)?.kind {
        MachineInstKind::Load {
            owner,
            ty,
            dst,
            addr,
            width,
            extension,
        } if addr.base == ext_dst && addr.offset == 0 => {
            // const_dst and ext_dst must both be dead after the load.
            if dst != ext_dst && reg_live_after(later, term, ext_dst) {
                return None;
            }
            if dst != const_dst && const_dst != ext_dst && reg_live_after(later, term, const_dst) {
                return None;
            }
            Some(FusedResult {
                fused: MachineInst {
                    kind: MachineInstKind::Load {
                        owner,
                        ty,
                        dst,
                        addr: MachineAddr {
                            base: base_reg,
                            offset: combined,
                        },
                        width,
                        extension,
                    },
                },
                skip: mem_idx + 1,
            })
        }
        MachineInstKind::Store {
            ty,
            addr,
            width,
            src,
        } if addr.base == ext_dst
            && addr.offset == 0
            && !matches!(src, MachineValue::Reg(r) if r == ext_dst || r == const_dst) =>
        {
            if reg_live_after(later, term, ext_dst)
                || (const_dst != ext_dst && reg_live_after(later, term, const_dst))
            {
                return None;
            }
            Some(FusedResult {
                fused: MachineInst {
                    kind: MachineInstKind::Store {
                        ty,
                        addr: MachineAddr {
                            base: base_reg,
                            offset: combined,
                        },
                        width,
                        src,
                    },
                },
                skip: mem_idx + 1,
            })
        }
        _ => None,
    }
}

/// Try to fuse `cvt.I64ExtendI32U + [offset_add] + base_add + load/store`.
fn try_fuse_uxtw_indexed(ops: &[MachineInst], term: &MachineTerminator) -> Option<FusedResult> {
    // [0] cvt.I64ExtendI32U ext_dst <- wasm_addr
    let (ext_dst, wasm_addr) = match ops.get(0)?.kind {
        MachineInstKind::Convert {
            op: MachineConvertOp::I64ExtendI32U,
            dst,
            src: MachineValue::Reg(src),
        } => (dst, src),
        _ => return None,
    };

    // [1] optional: i64.Add ext_dst <- ext_dst IMM  (Wasm offset)
    let (offset, offset_count) = match ops.get(1)?.kind {
        MachineInstKind::IntBinary {
            width: MachineIntWidth::I64,
            op: MachineIntBinaryOp::Add,
            dst,
            lhs: MachineValue::Reg(lhs),
            rhs: MachineValue::Imm64(imm),
        } if dst == ext_dst && lhs == ext_dst && imm <= i32::MAX as u64 => (imm as i32, 1),
        _ => (0i32, 0),
    };

    let base_idx = 1 + offset_count;

    // [base_idx] i64.Add ext_dst <- base ext_dst
    let base_reg = match ops.get(base_idx)?.kind {
        MachineInstKind::IntBinary {
            width: MachineIntWidth::I64,
            op: MachineIntBinaryOp::Add,
            dst,
            lhs: MachineValue::Reg(base),
            rhs: MachineValue::Reg(rhs),
        } if dst == ext_dst && rhs == ext_dst => base,
        _ => return None,
    };

    let mem_idx = base_idx + 1;
    let later = if ops.len() > mem_idx + 1 {
        &ops[mem_idx + 1..]
    } else {
        &[]
    };

    // [mem_idx] load or store using ext_dst with addr.offset == 0
    match ops.get(mem_idx)?.kind {
        MachineInstKind::Load {
            dst,
            addr,
            width,
            extension,
            ..
        } if addr.base == ext_dst && addr.offset == 0 => {
            // ext_dst must be dead after the load (overwritten by dst, or unused).
            if dst != ext_dst && reg_live_after(later, term, ext_dst) {
                return None;
            }
            Some(FusedResult {
                fused: MachineInst {
                    kind: MachineInstKind::IndexedLoad {
                        dst,
                        base: base_reg,
                        index: wasm_addr,
                        index_extend: MachineIndexExtend::ZeroExtend32,
                        offset,
                        width,
                        extension,
                    },
                },
                skip: mem_idx + 1,
            })
        }
        MachineInstKind::Store {
            addr, width, src, ..
        } if addr.base == ext_dst
            && addr.offset == 0
            && !matches!(src, MachineValue::Reg(r) if r == ext_dst) =>
        {
            if reg_live_after(later, term, ext_dst) {
                return None;
            }
            Some(FusedResult {
                fused: MachineInst {
                    kind: MachineInstKind::IndexedStore {
                        base: base_reg,
                        index: wasm_addr,
                        index_extend: MachineIndexExtend::ZeroExtend32,
                        offset,
                        width,
                        src,
                    },
                },
                skip: mem_idx + 1,
            })
        }
        _ => None,
    }
}

/// Try to fuse `i64.Add(base, index) + load/store`.
fn try_fuse_indexed(ops: &[MachineInst], term: &MachineTerminator) -> Option<FusedResult> {
    // [0] i64.Add add_dst <- base index
    let (add_dst, base_reg, index_reg) = match ops.get(0)?.kind {
        MachineInstKind::IntBinary {
            width: MachineIntWidth::I64,
            op: MachineIntBinaryOp::Add,
            dst,
            lhs: MachineValue::Reg(base),
            rhs: MachineValue::Reg(index),
        } => (dst, base, index),
        _ => return None,
    };

    let later = if ops.len() > 2 { &ops[2..] } else { &[] };

    // [1] load or store using add_dst with addr.offset == 0
    match ops.get(1)?.kind {
        MachineInstKind::Load {
            dst,
            addr,
            width,
            extension,
            ..
        } if addr.base == add_dst && addr.offset == 0 => {
            if dst != add_dst && reg_live_after(later, term, add_dst) {
                return None;
            }
            Some(FusedResult {
                fused: MachineInst {
                    kind: MachineInstKind::IndexedLoad {
                        dst,
                        base: base_reg,
                        index: index_reg,
                        index_extend: MachineIndexExtend::None,
                        offset: 0,
                        width,
                        extension,
                    },
                },
                skip: 2,
            })
        }
        MachineInstKind::Store {
            addr, width, src, ..
        } if addr.base == add_dst
            && addr.offset == 0
            && !matches!(src, MachineValue::Reg(r) if r == add_dst) =>
        {
            if reg_live_after(later, term, add_dst) {
                return None;
            }
            Some(FusedResult {
                fused: MachineInst {
                    kind: MachineInstKind::IndexedStore {
                        base: base_reg,
                        index: index_reg,
                        index_extend: MachineIndexExtend::None,
                        offset: 0,
                        width,
                        src,
                    },
                },
                skip: 2,
            })
        }
        _ => None,
    }
}
