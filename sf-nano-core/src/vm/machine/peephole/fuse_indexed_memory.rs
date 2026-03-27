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

use alloc::vec::Vec;

use crate::vm::machine::machine_ir::{
    MachineBlock, MachineConvertOp, MachineIndexExtend, MachineInst, MachineInstKind,
    MachineIntBinaryOp, MachineIntWidth, MachineTerminator, MachineValue,
};

use super::helpers::reg_live_after;

pub(super) fn fuse_indexed_memory(block: &mut MachineBlock) {
    let ops = &block.ops;
    let term = &block.terminator;
    let mut out: Vec<MachineInst> = Vec::with_capacity(ops.len());
    let mut i = 0;

    while i < ops.len() {
        // --- Pattern A: cvt + [offset_add] + base_add + load/store ---
        if let Some(consumed) = try_fuse_uxtw_indexed(&ops[i..], term) {
            out.push(consumed.fused);
            i += consumed.skip;
            continue;
        }

        // --- Pattern B: base_add + load/store ---
        if let Some(consumed) = try_fuse_indexed(&ops[i..], term) {
            out.push(consumed.fused);
            i += consumed.skip;
            continue;
        }

        out.push(ops[i].clone());
        i += 1;
    }

    block.ops = out;
}

struct FusedResult {
    fused: MachineInst,
    skip: usize,
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
    let later = if ops.len() > mem_idx + 1 { &ops[mem_idx + 1..] } else { &[] };

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
