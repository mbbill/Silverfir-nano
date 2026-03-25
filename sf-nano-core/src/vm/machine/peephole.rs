//! MachineIR peephole optimization pass.
//!
//! Runs within a single block. Current optimizations:
//!
//! 1. **Constant deduplication**: When the same constant value is materialized
//!    multiple times (`Move { src: Imm64 }` or `FloatConst`), subsequent
//!    occurrences are replaced with register copies from the first.
//!
//! 2. **Constant folding into operands**: `move rX <- C; op rD <- ... rX ...`
//!    → `op rD <- ... C ...` when rX has no other uses before redefinition.
//!
//! 3. **Copy propagation**: `move rTmp <- rSrc; op ... rTmp ...`
//!    → rewrite later uses to `rSrc` and remove the transient move.
//!
//! 4. **Store-to-load forwarding**: `store.u64 [addr] <- X; ...; load.u64 rY <- [addr]`
//!    → `move rY <- X` when intervening ops cannot invalidate the exact address.
//!
//! 5. **Compare-and-branch fusion**: `IntCompare { dst } + Branch { Value(Reg(dst)) }`
//!    → `Branch { IntCompare { ... } }` when the compare result register is dead in
//!    both successor blocks. Eliminates the boolean materialization (CSET on ARM64,
//!    SETCC+MOVZX on x86_64) and enables hardware compare-and-branch fusion.
//!    FloatCompare is NOT fused because x86_64 requires multi-instruction NaN
//!    handling that cannot be expressed as a single conditional branch.

use alloc::vec;
use alloc::vec::Vec;

use crate::vm::backend::BackendConfig;
use super::machine_ir::{
    MachineAddr, MachineBlock, MachineBlockId, MachineBranchCond, MachineConvertOp, MachineEdge,
    MachineFloatWidth, MachineIndexExtend, MachineInst, MachineInstKind, MachineIntBinaryOp,
    MachineIntWidth, MachineProgram, MachineReg, MachineStorageType, MachineTerminator,
    MachineValue,
};

#[derive(Clone, Copy)]
struct TrackedStore {
    addr: MachineAddr,
    src: MachineValue,
}

#[derive(Clone, Copy)]
struct TrackedLoad {
    addr: MachineAddr,
    ty: MachineStorageType,
    width: super::machine_ir::MachineMemWidth,
    extension: super::machine_ir::MachineLoadExtension,
    reg: MachineReg,
}

/// Run peephole optimizations on all blocks in a program.
///
/// Register classification is derived from `config` — the single source of
/// truth for the register layout.
pub(crate) fn optimize(program: &mut MachineProgram, config: BackendConfig) {
    let first_fp_reg = config.first_fp_reg();
    let gp_reg_width = config.gp_unit_bytes;
    let mut cp_scratch = CopyPropagateScratch::new(config.total_reg_count() as usize);
    for block in &mut program.blocks {
        deduplicate_constants(block, first_fp_reg);
        copy_propagate(block, config, &mut cp_scratch);
        forward_stored_values(block, config);
        reuse_loaded_values(block, config);
        fuse_indexed_memory(block);
        copy_propagate(block, config, &mut cp_scratch);
    }
    fuse_compare_branch(&mut program.blocks, gp_reg_width);
}

/// Replace duplicate constant materializations with register copies.
///
/// Within a block, if the same constant value is materialized into multiple
/// registers (via `Move { src: Imm64 }` or `FloatConst`), the second and
/// subsequent materializations are replaced with register-to-register copies.
///
/// Runs before `fold_constants` so that shared constants keep their defining
/// instruction alive, preventing fold from inlining the same expensive constant
/// into multiple consumers independently.
fn deduplicate_constants(block: &mut MachineBlock, first_fp_reg: u16) {
    let mut gp_consts: Vec<(u64, MachineReg)> = Vec::new();
    let mut fp_consts: Vec<(u64, MachineFloatWidth, MachineReg)> = Vec::new();

    for inst in &mut block.ops {
        if matches!(inst.kind, MachineInstKind::CallHelper(_)) {
            gp_consts.clear();
            fp_consts.clear();
            continue;
        }

        let mut new_gp = None;
        let mut new_fp = None;

        match &mut inst.kind {
            MachineInstKind::Move {
                dst,
                src: src @ MachineValue::Imm64(..),
                ..
            } if dst.0 < first_fp_reg => {
                let bits = match *src {
                    MachineValue::Imm64(b) => b,
                    _ => unreachable!(),
                };
                // Skip zero: fold_constants can inline Imm64(0) into consumers
                // for free (str xzr, cmp #0, etc.), so dedup would be a regression.
                if bits != 0 {
                    if let Some(&(_, prev)) =
                        gp_consts.iter().find(|(b, r)| *b == bits && *r != *dst)
                    {
                        *src = MachineValue::Reg(prev);
                    }
                    new_gp = Some((bits, *dst));
                }
            }
            MachineInstKind::FloatConst { dst, bits, width } => {
                let (d, b, w) = (*dst, *bits, *width);
                // Skip zero: fcmp d, #0.0 is free when folded as Imm64(0).
                if b != 0 {
                    if let Some(&(_, _, prev)) = fp_consts
                        .iter()
                        .find(|(bb, ww, r)| *bb == b && *ww == w && *r != d)
                    {
                        inst.kind = MachineInstKind::Move {
                            ty: match w {
                                MachineFloatWidth::F32 => MachineStorageType::Fp32,
                                MachineFloatWidth::F64 => MachineStorageType::Fp64,
                            },
                            dst: d,
                            src: MachineValue::Reg(prev),
                        };
                    }
                    new_fp = Some((b, w, d));
                }
            }
            _ => {}
        }

        // Invalidate tracking for any register redefined by this instruction.
        if let Some(def) = defined_reg(&inst.kind) {
            gp_consts.retain(|(_, r)| *r != def);
            fp_consts.retain(|(_, _, r)| *r != def);
        }

        if let Some(e) = new_gp {
            gp_consts.push(e);
        }
        if let Some(e) = new_fp {
            fp_consts.push(e);
        }
    }
}

/// Forward exact `store.u64` values into later exact `load.u64` instructions
/// within a block when no intervening instruction can change the address or
/// the stored source value.
fn forward_stored_values(block: &mut MachineBlock, config: BackendConfig) {
    let mut tracked = Vec::<TrackedStore>::new();
    let mut rewritten = Vec::with_capacity(block.ops.len());

    for mut inst in block.ops.drain(..) {
        let mut keep_inst = true;

        match &mut inst.kind {
            MachineInstKind::Load {
                ty,
                dst,
                addr,
                width: super::machine_ir::MachineMemWidth::U64,
                extension: super::machine_ir::MachineLoadExtension::None,
            } => {
                if let Some(src) = tracked
                    .iter()
                    .rev()
                    .find(|entry| entry.addr == *addr)
                    .map(|entry| entry.src)
                {
                    if matches!(src, MachineValue::Reg(src_reg) if src_reg == *dst) {
                        keep_inst = false;
                    } else if let Some(move_ty) =
                        rewrite_move_storage_type(*dst, src, *ty, config)
                    {
                        inst.kind = MachineInstKind::Move {
                            ty: move_ty,
                            dst: *dst,
                            src,
                        };
                    }
                }
            }
            MachineInstKind::Store {
                addr, width, src, ..
            } => {
                tracked.retain(|entry| {
                    !addrs_overlap(entry.addr, super::machine_ir::MachineMemWidth::U64, *addr, *width)
                });
                if *width == super::machine_ir::MachineMemWidth::U64 {
                    tracked.push(TrackedStore {
                        addr: *addr,
                        src: *src,
                    });
                }
            }
            MachineInstKind::CallHelper(_) => {
                tracked.clear();
            }
            _ => {}
        }

        if keep_inst {
            if let Some(dst) = defined_reg(&inst.kind) {
                kill_tracked_stores_by_reg(&mut tracked, dst);
            }
            rewritten.push(inst);
        }
    }

    block.ops = rewritten;
}

/// Reuse an earlier exact load result for a later identical load when no
/// intervening store, helper call, or register redefinition can invalidate it.
fn reuse_loaded_values(block: &mut MachineBlock, config: BackendConfig) {
    let mut tracked = Vec::<TrackedLoad>::new();
    let mut rewritten = Vec::with_capacity(block.ops.len());

    for mut inst in block.ops.drain(..) {
        let mut keep_inst = true;
        let mut produced_load = None;
        let mut rewrite_load = None;

        match &inst.kind {
            MachineInstKind::Load {
                ty,
                dst,
                addr,
                width,
                extension,
            } => {
                if let Some(src_reg) = tracked
                    .iter()
                    .rev()
                    .find(|entry| {
                        entry.addr == *addr
                            && entry.ty == *ty
                            && entry.width == *width
                            && entry.extension == *extension
                    })
                    .map(|entry| entry.reg)
                {
                    if src_reg == *dst {
                        keep_inst = false;
                    } else if let Some(move_ty) = rewrite_move_storage_type(
                        *dst,
                        MachineValue::Reg(src_reg),
                        *ty,
                        config,
                    ) {
                        rewrite_load = Some((*dst, src_reg, move_ty));
                        produced_load = Some(TrackedLoad {
                            addr: *addr,
                            ty: *ty,
                            width: *width,
                            extension: *extension,
                            reg: *dst,
                        });
                    }
                } else {
                    produced_load = Some(TrackedLoad {
                        addr: *addr,
                        ty: *ty,
                        width: *width,
                        extension: *extension,
                        reg: *dst,
                    });
                }
            }
            MachineInstKind::Store { addr, width, .. } => {
                tracked.retain(|entry| !addrs_overlap(entry.addr, entry.width, *addr, *width));
            }
            MachineInstKind::CallHelper(_) => {
                tracked.clear();
            }
            _ => {}
        }

        if keep_inst {
            if let Some((dst, src_reg, ty)) = rewrite_load {
                inst.kind = MachineInstKind::Move {
                    ty,
                    dst,
                    src: MachineValue::Reg(src_reg),
                };
            }
            if let Some(dst) = defined_reg(&inst.kind) {
                kill_tracked_loads_by_reg(&mut tracked, dst);
            }
            if let Some(load) = produced_load {
                tracked.push(load);
            }
            rewritten.push(inst);
        }
    }

    block.ops = rewritten;
}

/// Track transient register aliases within a block and rewrite later uses to
/// the original source register. Cached-local and fixed-register writes are
/// preserved, but their sources are still canonicalized.
/// Reusable scratch buffers for copy_propagate to avoid per-block allocation.
struct CopyPropagateScratch {
    aliases: Vec<Option<MachineReg>>,
    float_aliases: Vec<Option<MachineReg>>,
    rewritten: Vec<MachineInst>,
}

impl CopyPropagateScratch {
    fn new(reg_count: usize) -> Self {
        Self {
            aliases: vec![None; reg_count],
            float_aliases: vec![None; reg_count],
            rewritten: Vec::new(),
        }
    }

    fn clear(&mut self) {
        for a in &mut self.aliases {
            *a = None;
        }
        for a in &mut self.float_aliases {
            *a = None;
        }
        self.rewritten.clear();
    }
}

fn copy_propagate(
    block: &mut MachineBlock,
    config: BackendConfig,
    scratch: &mut CopyPropagateScratch,
) {
    scratch.clear();
    let original_ops = core::mem::take(&mut block.ops);
    scratch.rewritten.reserve(
        original_ops
            .len()
            .saturating_sub(scratch.rewritten.capacity()),
    );
    let aliases = &mut scratch.aliases;
    let float_aliases = &mut scratch.float_aliases;
    let rewritten = &mut scratch.rewritten;

    for (index, mut inst) in original_ops.iter().cloned().enumerate() {
        rewrite_sources(&mut inst.kind, aliases);
        rewrite_float_alias_sources(&mut inst.kind, float_aliases);

        if matches!(inst.kind, MachineInstKind::CallHelper(_)) {
            clear_aliases(aliases);
            clear_aliases(float_aliases);
            rewritten.push(inst);
            continue;
        }

        if let Some(dst) = defined_reg(&inst.kind) {
            kill_alias(aliases, dst);
            kill_alias(float_aliases, dst);
        }

        match &inst.kind {
            MachineInstKind::Move {
                ty: _,
                dst,
                src: MachineValue::Reg(src),
            } => {
                if *dst == *src {
                    continue;
                }
                // Only transient-to-transient copies are safe to elide here.
                // Moves from fixed or cached-local registers into a transient
                // often act as snapshots, not just aliases.
                if super::machine_ir::is_transient_reg(*dst, config)
                    && super::machine_ir::is_transient_reg(*src, config)
                    && super::machine_ir::same_reg_bank(*dst, *src, config)
                    && can_elide_reg_move(&original_ops, &block.terminator, index, *dst, *src)
                {
                    aliases[dst.0 as usize] = Some(*src);
                    continue;
                }
                if !super::machine_ir::is_fp_reg(*dst, config)
                    && super::machine_ir::is_fp_reg(*src, config)
                {
                    float_aliases[dst.0 as usize] = Some(*src);
                }
            }
            _ => {}
        }

        rewritten.push(inst);
    }

    rewrite_terminator_sources(&mut block.terminator, aliases);
    rewrite_float_alias_terminator_sources(&mut block.terminator, float_aliases);
    block.ops = core::mem::take(rewritten);
}

/// Fuse address computation + load/store into `IndexedLoad`/`IndexedStore`.
///
/// Recognizes two patterns and replaces them with first-class fused
/// instructions that each backend maps to its best addressing mode:
///
/// **Pattern A** (with zero-extend, 3→1):
/// ```text
/// cvt.I64ExtendI32U  r <- addr        // zero-extend Wasm address
/// [i64.Add           r <- r IMM]      // optional: Wasm load/store offset
/// i64.Add            r <- base r      // add linear-memory base
/// load/store         .. [r + 0]       // memory access (offset may be non-zero
///                                     //   if already folded by an earlier run)
/// ```
/// → `IndexedLoad/Store { base, index=addr, extend=ZeroExtend32, offset }`
///
/// **Pattern B** (no extend, 2→1):
/// ```text
/// i64.Add            r <- base index  // base + index
/// load/store         .. [r + 0]       // memory access
/// ```
/// → `IndexedLoad/Store { base, index, extend=None, offset=0 }`
fn fuse_indexed_memory(block: &mut MachineBlock) {
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

/// Check if `reg` is used by any instruction in `ops` or the terminator before
/// being redefined.
fn reg_live_after(ops: &[MachineInst], term: &MachineTerminator, reg: MachineReg) -> bool {
    for inst in ops {
        if inst_uses_value(&inst.kind, reg) {
            return true;
        }
        if inst_defines(&inst.kind, reg) {
            return false;
        }
    }
    terminator_uses_reg(term, reg)
}

/// Check if an instruction uses `reg` as a source operand.
fn inst_uses_value(kind: &MachineInstKind, reg: MachineReg) -> bool {
    let mut found = false;
    visit_source_values(kind, |v| {
        if matches!(v, MachineValue::Reg(r) if *r == reg) {
            found = true;
        }
    });
    found
}

// --- helpers ---

/// Count how many times `reg` appears as a source operand in `kind`.
fn count_value_uses(kind: &MachineInstKind, reg: MachineReg) -> usize {
    let mut count = 0;
    visit_source_values(kind, |v| {
        if matches!(v, MachineValue::Reg(r) if *r == reg) {
            count += 1;
        }
    });
    count
}

/// Check if `kind` defines (writes to) `reg`.
fn inst_defines(kind: &MachineInstKind, reg: MachineReg) -> bool {
    match kind {
        MachineInstKind::Move { dst, .. }
        | MachineInstKind::FloatConst { dst, .. }
        | MachineInstKind::Load { dst, .. }
        | MachineInstKind::IntUnary { dst, .. }
        | MachineInstKind::IntBinary { dst, .. }
        | MachineInstKind::IntCompare { dst, .. }
        | MachineInstKind::FloatUnary { dst, .. }
        | MachineInstKind::FloatBinary { dst, .. }
        | MachineInstKind::FloatCompare { dst, .. }
        | MachineInstKind::Convert { dst, .. }
        | MachineInstKind::Select { dst, .. }
        | MachineInstKind::IndexedLoad { dst, .. } => *dst == reg,
        MachineInstKind::Int64PairBinary { dst_lo, dst_hi, .. } => *dst_lo == reg || *dst_hi == reg,
        MachineInstKind::Int64PairUnary { dst_lo, dst_hi, .. } => *dst_lo == reg || *dst_hi == reg,
        MachineInstKind::Int64PairDivRem { dst_lo, dst_hi, .. } => *dst_lo == reg || *dst_hi == reg,
        MachineInstKind::Int64PairShift { dst_lo, dst_hi, .. } => *dst_lo == reg || *dst_hi == reg,
        MachineInstKind::Int64PairCompare { dst, .. } => *dst == reg,
        MachineInstKind::ConvertFloatToI64Pair { dst_lo, dst_hi, .. } => {
            *dst_lo == reg || *dst_hi == reg
        }
        MachineInstKind::ReinterpretF64ToI64Pair { dst_lo, dst_hi, .. } => {
            *dst_lo == reg || *dst_hi == reg
        }
        MachineInstKind::ConvertI64PairToFloat { dst, .. }
        | MachineInstKind::ReinterpretI64PairToF64 { dst, .. } => *dst == reg,
        MachineInstKind::Store { .. }
        | MachineInstKind::IndexedStore { .. }
        | MachineInstKind::TrapIf { .. }
        | MachineInstKind::CallHelper(_) => false,
    }
}

fn defined_reg(kind: &MachineInstKind) -> Option<MachineReg> {
    match kind {
        MachineInstKind::Move { dst, .. }
        | MachineInstKind::FloatConst { dst, .. }
        | MachineInstKind::Load { dst, .. }
        | MachineInstKind::IntUnary { dst, .. }
        | MachineInstKind::IntBinary { dst, .. }
        | MachineInstKind::IntCompare { dst, .. }
        | MachineInstKind::FloatUnary { dst, .. }
        | MachineInstKind::FloatBinary { dst, .. }
        | MachineInstKind::FloatCompare { dst, .. }
        | MachineInstKind::Convert { dst, .. }
        | MachineInstKind::Select { dst, .. }
        | MachineInstKind::IndexedLoad { dst, .. } => Some(*dst),
        MachineInstKind::Int64PairBinary { .. } => None,
        MachineInstKind::Int64PairUnary { .. } => None,
        MachineInstKind::Int64PairDivRem { .. } => None,
        MachineInstKind::Int64PairShift { .. } => None,
        MachineInstKind::Int64PairCompare { dst, .. } => Some(*dst),
        MachineInstKind::ConvertFloatToI64Pair { .. } => None,
        MachineInstKind::ConvertI64PairToFloat { dst, .. }
        | MachineInstKind::ReinterpretI64PairToF64 { dst, .. } => Some(*dst),
        MachineInstKind::ReinterpretF64ToI64Pair { .. } => None,
        MachineInstKind::Store { .. }
        | MachineInstKind::IndexedStore { .. }
        | MachineInstKind::TrapIf { .. }
        | MachineInstKind::CallHelper(_) => None,
    }
}

/// Check if a terminator reads from `reg`.
fn terminator_uses_reg(term: &super::machine_ir::MachineTerminator, reg: MachineReg) -> bool {
    match term {
        super::machine_ir::MachineTerminator::Jump(edge) => edge_uses_reg(edge, reg),
        super::machine_ir::MachineTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            branch_cond_uses_reg(cond, reg)
                || edge_uses_reg(then_edge, reg)
                || edge_uses_reg(else_edge, reg)
        }
        super::machine_ir::MachineTerminator::JumpTable { index, entries } => {
            value_is_reg(index, reg) || entries.iter().any(|e| edge_uses_reg(e, reg))
        }
        super::machine_ir::MachineTerminator::CallDirect {
            callee_frame_base, ..
        } => *callee_frame_base == reg,
        super::machine_ir::MachineTerminator::CallIndirect {
            callee_target,
            callee_frame_base,
            ..
        } => value_is_reg(callee_target, reg) || *callee_frame_base == reg,
        super::machine_ir::MachineTerminator::Return | super::machine_ir::MachineTerminator::Trap { .. } => false,
    }
}

fn edge_uses_reg(edge: &super::machine_ir::MachineEdge, reg: MachineReg) -> bool {
    edge.args.iter().any(|v| value_is_reg(v, reg))
}

fn branch_cond_uses_reg(cond: &super::machine_ir::MachineBranchCond, reg: MachineReg) -> bool {
    match cond {
        super::machine_ir::MachineBranchCond::Value(v) => value_is_reg(v, reg),
        super::machine_ir::MachineBranchCond::IntCompare { lhs, rhs, .. } => {
            value_is_reg(lhs, reg) || value_is_reg(rhs, reg)
        }
    }
}

fn value_is_reg(v: &MachineValue, reg: MachineReg) -> bool {
    matches!(v, MachineValue::Reg(r) if *r == reg)
}

/// Visit all source (read) values in an instruction.
fn visit_source_values(kind: &MachineInstKind, mut f: impl FnMut(&MachineValue)) {
    match kind {
        MachineInstKind::Move { src, .. } => f(src),
        MachineInstKind::FloatConst { .. } => {}
        MachineInstKind::Load { addr, .. } => {
            f(&MachineValue::Reg(addr.base));
        }
        MachineInstKind::Store { addr, src, .. } => {
            f(&MachineValue::Reg(addr.base));
            f(src);
        }
        MachineInstKind::IndexedLoad { base, index, .. } => {
            f(&MachineValue::Reg(*base));
            f(&MachineValue::Reg(*index));
        }
        MachineInstKind::IndexedStore { base, index, src, .. } => {
            f(&MachineValue::Reg(*base));
            f(&MachineValue::Reg(*index));
            f(src);
        }
        MachineInstKind::IntUnary { src, .. } => f(src),
        MachineInstKind::IntBinary { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        MachineInstKind::Int64PairBinary {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            f(lhs_lo);
            f(lhs_hi);
            f(rhs_lo);
            f(rhs_hi);
        }
        MachineInstKind::Int64PairUnary { src_lo, src_hi, .. } => {
            f(src_lo);
            f(src_hi);
        }
        MachineInstKind::Int64PairDivRem {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            f(lhs_lo);
            f(lhs_hi);
            f(rhs_lo);
            f(rhs_hi);
        }
        MachineInstKind::Int64PairShift {
            lhs_lo,
            lhs_hi,
            rhs,
            ..
        } => {
            f(lhs_lo);
            f(lhs_hi);
            f(rhs);
        }
        MachineInstKind::Int64PairCompare {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            f(lhs_lo);
            f(lhs_hi);
            f(rhs_lo);
            f(rhs_hi);
        }
        MachineInstKind::IntCompare { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        MachineInstKind::FloatUnary { src, .. } => f(src),
        MachineInstKind::FloatBinary { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        MachineInstKind::FloatCompare { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
        MachineInstKind::Convert { src, .. } => f(src),
        MachineInstKind::ConvertI64PairToFloat { src_lo, src_hi, .. } => {
            f(src_lo);
            f(src_hi);
        }
        MachineInstKind::ConvertFloatToI64Pair { src, .. }
        | MachineInstKind::ReinterpretF64ToI64Pair { src, .. } => f(src),
        MachineInstKind::ReinterpretI64PairToF64 { src_lo, src_hi, .. } => {
            f(src_lo);
            f(src_hi);
        }
        MachineInstKind::Select {
            on_true,
            on_false,
            cond,
            ..
        } => {
            f(on_true);
            f(on_false);
            f(cond);
        }
        MachineInstKind::TrapIf { cond, .. } => visit_branch_cond_values(cond, &mut f),
        MachineInstKind::CallHelper(_) => {}
    }
}

fn rewrite_sources(kind: &mut MachineInstKind, aliases: &[Option<MachineReg>]) {
    match kind {
        MachineInstKind::Move { src, .. }
        | MachineInstKind::IntUnary { src, .. }
        | MachineInstKind::FloatUnary { src, .. }
        | MachineInstKind::Convert { src, .. }
        | MachineInstKind::ConvertFloatToI64Pair { src, .. }
        | MachineInstKind::ReinterpretF64ToI64Pair { src, .. } => rewrite_value(src, aliases),
        MachineInstKind::ConvertI64PairToFloat { src_lo, src_hi, .. }
        | MachineInstKind::ReinterpretI64PairToF64 { src_lo, src_hi, .. } => {
            rewrite_value(src_lo, aliases);
            rewrite_value(src_hi, aliases);
        }
        MachineInstKind::FloatConst { .. } => {}
        MachineInstKind::Load { addr, .. } => rewrite_addr(addr, aliases),
        MachineInstKind::Store { addr, src, .. } => {
            rewrite_addr(addr, aliases);
            rewrite_value(src, aliases);
        }
        MachineInstKind::IndexedLoad { base, index, .. } => {
            *base = resolve_alias(*base, aliases);
            *index = resolve_alias(*index, aliases);
        }
        MachineInstKind::IndexedStore { base, index, src, .. } => {
            *base = resolve_alias(*base, aliases);
            *index = resolve_alias(*index, aliases);
            rewrite_value(src, aliases);
        }
        MachineInstKind::IntBinary { lhs, rhs, .. }
        | MachineInstKind::IntCompare { lhs, rhs, .. }
        | MachineInstKind::FloatBinary { lhs, rhs, .. }
        | MachineInstKind::FloatCompare { lhs, rhs, .. } => {
            rewrite_value(lhs, aliases);
            rewrite_value(rhs, aliases);
        }
        MachineInstKind::Int64PairBinary {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            rewrite_value(lhs_lo, aliases);
            rewrite_value(lhs_hi, aliases);
            rewrite_value(rhs_lo, aliases);
            rewrite_value(rhs_hi, aliases);
        }
        MachineInstKind::Int64PairDivRem {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            rewrite_value(lhs_lo, aliases);
            rewrite_value(lhs_hi, aliases);
            rewrite_value(rhs_lo, aliases);
            rewrite_value(rhs_hi, aliases);
        }
        MachineInstKind::Int64PairUnary { src_lo, src_hi, .. } => {
            rewrite_value(src_lo, aliases);
            rewrite_value(src_hi, aliases);
        }
        MachineInstKind::Int64PairShift {
            lhs_lo,
            lhs_hi,
            rhs,
            ..
        } => {
            rewrite_value(lhs_lo, aliases);
            rewrite_value(lhs_hi, aliases);
            rewrite_value(rhs, aliases);
        }
        MachineInstKind::Int64PairCompare {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            rewrite_value(lhs_lo, aliases);
            rewrite_value(lhs_hi, aliases);
            rewrite_value(rhs_lo, aliases);
            rewrite_value(rhs_hi, aliases);
        }
        MachineInstKind::Select {
            on_true,
            on_false,
            cond,
            ..
        } => {
            rewrite_value(on_true, aliases);
            rewrite_value(on_false, aliases);
            rewrite_value(cond, aliases);
        }
        MachineInstKind::TrapIf { cond, .. } => rewrite_branch_cond(cond, aliases),
        MachineInstKind::CallHelper(_) => {}
    }
}

fn rewrite_terminator_sources(term: &mut MachineTerminator, aliases: &[Option<MachineReg>]) {
    match term {
        MachineTerminator::Jump(edge) => rewrite_edge(edge, aliases),
        MachineTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            rewrite_branch_cond(cond, aliases);
            rewrite_edge(then_edge, aliases);
            rewrite_edge(else_edge, aliases);
        }
        MachineTerminator::JumpTable { index, entries } => {
            rewrite_value(index, aliases);
            for edge in entries {
                rewrite_edge(edge, aliases);
            }
        }
        MachineTerminator::CallDirect {
            callee_frame_base, ..
        } => {
            *callee_frame_base = resolve_alias(*callee_frame_base, aliases);
        }
        MachineTerminator::CallIndirect {
            callee_target,
            callee_frame_base,
            ..
        } => {
            rewrite_value(callee_target, aliases);
            *callee_frame_base = resolve_alias(*callee_frame_base, aliases);
        }
        MachineTerminator::Return | MachineTerminator::Trap { .. } => {}
    }
}

fn rewrite_float_alias_terminator_sources(
    term: &mut MachineTerminator,
    aliases: &[Option<MachineReg>],
) {
    match term {
        MachineTerminator::Branch { cond, .. } => rewrite_float_alias_branch_cond(cond, aliases),
        MachineTerminator::Jump(_)
        | MachineTerminator::JumpTable { .. }
        | MachineTerminator::CallDirect { .. }
        | MachineTerminator::CallIndirect { .. }
        | MachineTerminator::Return
        | MachineTerminator::Trap { .. } => {}
    }
}

fn rewrite_branch_cond(cond: &mut MachineBranchCond, aliases: &[Option<MachineReg>]) {
    match cond {
        MachineBranchCond::Value(value) => rewrite_value(value, aliases),
        MachineBranchCond::IntCompare { lhs, rhs, .. } => {
            rewrite_value(lhs, aliases);
            rewrite_value(rhs, aliases);
        }
    }
}

fn rewrite_float_alias_branch_cond(cond: &mut MachineBranchCond, aliases: &[Option<MachineReg>]) {
    let _ = (cond, aliases);
}

fn rewrite_edge(edge: &mut MachineEdge, aliases: &[Option<MachineReg>]) {
    for arg in &mut edge.args {
        rewrite_value(arg, aliases);
    }
}

fn rewrite_addr(addr: &mut MachineAddr, aliases: &[Option<MachineReg>]) {
    addr.base = resolve_alias(addr.base, aliases);
}

fn rewrite_value(value: &mut MachineValue, aliases: &[Option<MachineReg>]) {
    if let MachineValue::Reg(reg) = value {
        *reg = resolve_alias(*reg, aliases);
    }
}

fn rewrite_float_alias_sources(kind: &mut MachineInstKind, aliases: &[Option<MachineReg>]) {
    match kind {
        MachineInstKind::FloatUnary { src, .. } => rewrite_float_alias_value(src, aliases),
        MachineInstKind::FloatBinary { lhs, rhs, .. }
        | MachineInstKind::FloatCompare { lhs, rhs, .. } => {
            rewrite_float_alias_value(lhs, aliases);
            rewrite_float_alias_value(rhs, aliases);
        }
        MachineInstKind::Store { width, src, .. }
            if matches!(
                width,
                super::machine_ir::MachineMemWidth::U32 | super::machine_ir::MachineMemWidth::U64
            ) =>
        {
            rewrite_float_alias_value(src, aliases);
        }
        MachineInstKind::TrapIf { cond, .. } => rewrite_float_alias_branch_cond(cond, aliases),
        MachineInstKind::Convert { op, src, .. } if convert_src_accepts_fp(*op) => {
            rewrite_float_alias_value(src, aliases);
        }
        _ => {}
    }
}

fn visit_branch_cond_values(cond: &MachineBranchCond, mut f: impl FnMut(&MachineValue)) {
    match cond {
        MachineBranchCond::Value(value) => f(value),
        MachineBranchCond::IntCompare { lhs, rhs, .. } => {
            f(lhs);
            f(rhs);
        }
    }
}

fn rewrite_float_alias_value(value: &mut MachineValue, aliases: &[Option<MachineReg>]) {
    let MachineValue::Reg(reg) = value else {
        return;
    };
    if let Some(Some(src)) = aliases.get(reg.0 as usize) {
        *value = MachineValue::Reg(*src);
    }
}

fn resolve_alias(reg: MachineReg, aliases: &[Option<MachineReg>]) -> MachineReg {
    let mut resolved = reg;
    while let Some(Some(next)) = aliases.get(resolved.0 as usize) {
        if *next == resolved {
            break;
        }
        resolved = *next;
    }
    resolved
}

fn convert_src_accepts_fp(op: super::machine_ir::MachineConvertOp) -> bool {
    matches!(
        op,
        super::machine_ir::MachineConvertOp::I32TruncF32S
            | super::machine_ir::MachineConvertOp::I32TruncF32U
            | super::machine_ir::MachineConvertOp::I32TruncF64S
            | super::machine_ir::MachineConvertOp::I32TruncF64U
            | super::machine_ir::MachineConvertOp::I64TruncF32S
            | super::machine_ir::MachineConvertOp::I64TruncF32U
            | super::machine_ir::MachineConvertOp::I64TruncF64S
            | super::machine_ir::MachineConvertOp::I64TruncF64U
            | super::machine_ir::MachineConvertOp::I32TruncSatF32S
            | super::machine_ir::MachineConvertOp::I32TruncSatF32U
            | super::machine_ir::MachineConvertOp::I32TruncSatF64S
            | super::machine_ir::MachineConvertOp::I32TruncSatF64U
            | super::machine_ir::MachineConvertOp::I64TruncSatF32S
            | super::machine_ir::MachineConvertOp::I64TruncSatF32U
            | super::machine_ir::MachineConvertOp::I64TruncSatF64S
            | super::machine_ir::MachineConvertOp::I64TruncSatF64U
            | super::machine_ir::MachineConvertOp::F32DemoteF64
            | super::machine_ir::MachineConvertOp::F64PromoteF32
            | super::machine_ir::MachineConvertOp::I32ReinterpretF32
            | super::machine_ir::MachineConvertOp::I64ReinterpretF64
    )
}

fn kill_alias(aliases: &mut [Option<MachineReg>], reg: MachineReg) {
    if let Some(slot) = aliases.get_mut(reg.0 as usize) {
        *slot = None;
    }
    for alias in aliases.iter_mut() {
        if *alias == Some(reg) {
            *alias = None;
        }
    }
}

fn clear_aliases(aliases: &mut [Option<MachineReg>]) {
    for alias in aliases.iter_mut() {
        *alias = None;
    }
}

fn reg_move_rewrite_supported(dst: MachineReg, src: MachineReg, config: BackendConfig) -> bool {
    let dst_is_fp = super::machine_ir::is_fp_reg(dst, config);
    let src_is_fp = super::machine_ir::is_fp_reg(src, config);
    !dst_is_fp || src_is_fp
}

fn move_rewrite_supported(dst: MachineReg, src: MachineValue, config: BackendConfig) -> bool {
    match src {
        MachineValue::Reg(src_reg) => reg_move_rewrite_supported(dst, src_reg, config),
        MachineValue::Imm64(_) => super::machine_ir::is_gp_reg(dst, config),
    }
}

fn rewrite_move_storage_type(
    dst: MachineReg,
    src: MachineValue,
    ty: MachineStorageType,
    config: BackendConfig,
) -> Option<MachineStorageType> {
    if super::machine_ir::is_fp_reg(dst, config) != ty.is_fp() {
        return None;
    }
    move_rewrite_supported(dst, src, config).then_some(ty)
}

fn can_elide_reg_move(
    ops: &[MachineInst],
    terminator: &MachineTerminator,
    start_idx: usize,
    dst: MachineReg,
    src: MachineReg,
) -> bool {
    let mut source_stable = true;

    for (later_index, inst) in ops[start_idx + 1..].iter().enumerate() {
        if count_value_uses(&inst.kind, dst) > 0 && !source_stable {
            return false;
        }
        if inst_defines(&inst.kind, dst) {
            return true;
        }
        if matches!(inst.kind, MachineInstKind::CallHelper(_)) {
            // copy_propagate clears aliases at helper calls, so a move can only
            // disappear here if its destination is dead after the barrier.
            let remaining = &ops[start_idx + 1 + later_index + 1..];
            return !reg_live_after(remaining, terminator, dst);
        }
        if inst_defines(&inst.kind, src) {
            source_stable = false;
        }
    }

    source_stable || !terminator_uses_reg(terminator, dst)
}

fn kill_tracked_stores_by_reg(tracked: &mut Vec<TrackedStore>, reg: MachineReg) {
    tracked.retain(|entry| entry.addr.base != reg && !value_is_reg(&entry.src, reg));
}

fn kill_tracked_loads_by_reg(tracked: &mut Vec<TrackedLoad>, reg: MachineReg) {
    tracked.retain(|entry| entry.addr.base != reg && entry.reg != reg);
}

fn addrs_overlap(
    lhs_addr: MachineAddr,
    lhs_width: super::machine_ir::MachineMemWidth,
    rhs_addr: MachineAddr,
    rhs_width: super::machine_ir::MachineMemWidth,
) -> bool {
    if lhs_addr.base != rhs_addr.base {
        return false;
    }

    let lhs_start = i64::from(lhs_addr.offset);
    let lhs_end = lhs_start + mem_width_bytes(lhs_width);
    let rhs_start = i64::from(rhs_addr.offset);
    let rhs_end = rhs_start + mem_width_bytes(rhs_width);

    lhs_start < rhs_end && rhs_start < lhs_end
}

fn mem_width_bytes(width: super::machine_ir::MachineMemWidth) -> i64 {
    match width {
        super::machine_ir::MachineMemWidth::U8 => 1,
        super::machine_ir::MachineMemWidth::U16 => 2,
        super::machine_ir::MachineMemWidth::U32 => 4,
        super::machine_ir::MachineMemWidth::U64 => 8,
    }
}

// ---------------------------------------------------------------------------
// Compare-and-branch fusion
// ---------------------------------------------------------------------------

/// Rewrite `IntCompare/FloatCompare { dst } + Branch { Value(Reg(dst)) }`
/// into `Branch { IntCompare/FloatCompare { ... } }` when the compare result
/// register is provably dead in both successor blocks.
///
/// This is a cross-block pass: it reads successor blocks to check liveness,
/// so it must run after the per-block optimizations are done.
fn fuse_compare_branch(blocks: &mut [MachineBlock], gp_reg_width: u8) {
    for idx in 0..blocks.len() {
        // Check the last op and the terminator of this block.
        let last_op = match blocks[idx].ops.last() {
            Some(op) => op,
            None => continue,
        };

        // Terminator must be Branch { cond: Value(Reg(cond_reg)) }.
        let (cond_reg, then_target, else_target) = match &blocks[idx].terminator {
            MachineTerminator::Branch {
                cond: MachineBranchCond::Value(MachineValue::Reg(r)),
                then_edge,
                else_edge,
            } => (*r, then_edge.target, else_edge.target),
            _ => continue,
        };

        // Build the fused branch condition, or skip.
        //
        // Only IntCompare is fused here. FloatCompare is NOT fused because
        // on x86_64 Wasm float comparisons require multi-instruction NaN
        // handling (SETCC+SETNP+AND) that cannot be expressed as a single
        // conditional branch. ARM64's FCMP condition codes handle NaN
        // correctly with a single B.cond, but since this is a shared pass
        // it must be safe for all backends.
        let fused_cond = match &last_op.kind {
            MachineInstKind::IntCompare {
                width,
                ..
            } if *width == super::machine_ir::MachineIntWidth::I64 && gp_reg_width == 4 => continue,
            MachineInstKind::IntCompare {
                width,
                kind,
                sign,
                dst,
                lhs,
                rhs,
            } if *dst == cond_reg => MachineBranchCond::IntCompare {
                width: *width,
                kind: *kind,
                sign: *sign,
                lhs: *lhs,
                rhs: *rhs,
            },
            _ => continue,
        };

        // Reject if any edge passes dst as an arg.
        if term_edge_uses_value(&blocks[idx].terminator, cond_reg) {
            continue;
        }

        // Reject if dst is live-in to either successor.
        if !reg_dead_at_block_entry(blocks, then_target, cond_reg) {
            continue;
        }
        if !reg_dead_at_block_entry(blocks, else_target, cond_reg) {
            continue;
        }

        // Safe to fuse: remove the compare op and rewrite the terminator.
        blocks[idx].ops.pop();
        if let MachineTerminator::Branch { cond, .. } = &mut blocks[idx].terminator {
            *cond = fused_cond;
        }
    }
}

/// Check whether any edge arg in the terminator references `reg`.
fn term_edge_uses_value(term: &MachineTerminator, reg: MachineReg) -> bool {
    let check = |e: &MachineEdge| e.args.iter().any(|a| value_is_reg(a, reg));
    match term {
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => check(then_edge) || check(else_edge),
        MachineTerminator::Jump(edge) => check(edge),
        MachineTerminator::JumpTable { entries, .. } => entries.iter().any(|e| check(e)),
        _ => false,
    }
}

/// Returns true if `reg` is provably dead at the beginning of `target`:
/// either the block defines it before any use, the block has it as a
/// parameter, or the block never touches it.
pub(crate) fn reg_dead_at_block_entry(
    blocks: &[MachineBlock],
    target: MachineBlockId,
    reg: MachineReg,
) -> bool {
    let Some(block) = blocks.get(target.as_usize()) else {
        return false;
    };
    // If the target has reg as a param, it will be defined by the edge.
    if block.params.iter().any(|p| p.reg == reg) {
        return true;
    }
    // Scan ops: defined before used → dead at entry.
    for op in &block.ops {
        if inst_defines(&op.kind, reg) {
            return true;
        }
        if count_value_uses(&op.kind, reg) > 0 {
            return false;
        }
    }
    // Reached terminator without touching reg.
    !terminator_uses_reg(&block.terminator, reg)
}
