//! Recover bulk memory operations from overlap-safe scalar byte-copy loops.
//!
//! Producers can lower `memory.copy` to a loop with a direction-dependent
//! step. Prove both directions, the byte accesses, and the carried state,
//! whether the step remains in a register or is reloaded from a frame slot.
//! A widened bounds guard retains the original loop for ranges where bulk
//! validation would change zero-length, address-wrap or partial-trap behaviour.

use crate::collections;
use crate::vm::jit::backend::BackendConfig;
use crate::vm::jit::machine::machine_ir::{
    MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineBranchCond,
    MachineCompareKind, MachineConvertOp, MachineEdge, MachineIndexExtend, MachineInst,
    MachineInstKind, MachineIntBinaryOp, MachineIntWidth, MachineLoadExtension, MachineMemWidth,
    MachineProgram, MachineReg, MachineSign, MachineStorageType, MachineTerminator, MachineValue,
    MACHINE_FP_REG, MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
};

pub(super) fn recognize_memmove(program: &mut MachineProgram, config: BackendConfig) {
    // Widened endpoint checks below require a 64-bit GP carrier. Leave the
    // original byte loop intact when the target cannot express those checks.
    if config.gp_unit_bytes != 8 {
        return;
    }
    let Some((dispatch_index, dest, src, len)) = find_memmove_dispatch(&program.blocks) else {
        return;
    };
    let operands = [dest, src, len];
    let mut free = (BackendConfig::FIXED
        ..BackendConfig::FIXED + u16::from(config.allocatable_gp_dynamic_budget()))
        .map(MachineReg)
        .filter(|r| !operands.contains(r));
    let (Some(end), Some(count)) = (free.next(), free.next()) else {
        return;
    };
    let original = program.blocks[dispatch_index].clone();
    let fallback_id = MachineBlockId(program.blocks.len() as u32);
    let source_check_id = MachineBlockId(fallback_id.0 + 1);
    let copy_id = MachineBlockId(fallback_id.0 + 2);
    let dest_bound_id = MachineBlockId(fallback_id.0 + 3);
    let source_bound_id = MachineBlockId(fallback_id.0 + 4);
    let edge = |target| MachineEdge {
        target,
        args: operands.iter().copied().map(reg).collect(),
    };
    let check = |id, start, success| MachineBlock {
        id,
        params: original.params.clone(),
        ops: collections::vec![
            MachineInst {
                kind: MachineInstKind::Convert {
                    op: MachineConvertOp::I64ExtendI32U,
                    dst: end,
                    src: reg(start),
                }
            },
            MachineInst {
                kind: MachineInstKind::Convert {
                    op: MachineConvertOp::I64ExtendI32U,
                    dst: count,
                    src: reg(len),
                }
            },
            MachineInst {
                kind: MachineInstKind::IntBinary {
                    width: MachineIntWidth::I64,
                    op: MachineIntBinaryOp::Add,
                    dst: end,
                    lhs: reg(end),
                    rhs: reg(count),
                }
            },
        ],
        terminator: MachineTerminator::Branch {
            cond: MachineBranchCond::IntCompare {
                width: MachineIntWidth::I64,
                kind: MachineCompareKind::Le,
                sign: MachineSign::Unsigned,
                lhs: reg(end),
                // The original addresses wrap at 32 bits even if memory 0
                // itself is memory64. An endpoint of 2^32 is still safe:
                // the last accessed byte is at 2^32 - 1.
                rhs: MachineValue::Imm64(1u64 << 32),
            },
            then_edge: MachineEdge {
                target: success,
                args: operands.iter().copied().chain([end]).map(reg).collect(),
            },
            else_edge: edge(fallback_id),
        },
    };
    let bound = |id, success| {
        let mut params = original.params.clone();
        params.push(MachineBlockParam::gp_word(end));
        MachineBlock {
            id,
            params,
            ops: collections::vec![],
            terminator: MachineTerminator::Branch {
                cond: MachineBranchCond::IntCompare {
                    width: MachineIntWidth::I64,
                    kind: MachineCompareKind::Le,
                    sign: MachineSign::Unsigned,
                    lhs: reg(end),
                    rhs: reg(MACHINE_MEM0_SIZE_REG),
                },
                then_edge: edge(success),
                else_edge: edge(fallback_id),
            },
        }
    };
    // A byte loop may modify a prefix before trapping, or do no accesses for
    // len=0 even when its pointers are out of bounds. Bulk memory instead
    // validates both complete ranges first. Only take the bulk path when
    // every original access is in bounds; otherwise execute the original
    // control flow, including its exact trap and partial-write behaviour.
    program.blocks[dispatch_index] = check(original.id, dest, dest_bound_id);
    program.blocks.push(MachineBlock {
        id: fallback_id,
        ..original.clone()
    });
    program
        .blocks
        .push(check(source_check_id, src, source_bound_id));
    let dest_bound = bound(dest_bound_id, source_check_id);
    let source_bound = bound(source_bound_id, copy_id);
    program.blocks.push(MachineBlock {
        id: copy_id,
        params: original.params,
        ops: collections::vec![MachineInst {
            kind: MachineInstKind::MemoryCopy {
                dst_mem: 0,
                src_mem: 0,
                dest: reg(dest),
                src: reg(src),
                len: reg(len),
            },
        }],
        terminator: MachineTerminator::Return,
    });
    program.blocks.push(dest_bound);
    program.blocks.push(source_bound);
}

fn find_memmove_dispatch(
    blocks: &[MachineBlock],
) -> Option<(usize, MachineReg, MachineReg, MachineReg)> {
    blocks.iter().enumerate().find_map(|(index, block)| {
        match_memmove_dispatch(blocks, block).map(|regs| (index, regs.0, regs.1, regs.2))
    })
}

fn match_memmove_dispatch(
    blocks: &[MachineBlock],
    dispatch: &MachineBlock,
) -> Option<(MachineReg, MachineReg, MachineReg)> {
    if !dispatch.ops.is_empty() {
        return None;
    }
    let [dest_param, src_param, len_param] = dispatch.params.as_slice() else {
        return None;
    };
    if [dest_param.ty, src_param.ty, len_param.ty]
        .iter()
        .any(|ty| *ty != MachineStorageType::GpWord)
    {
        return None;
    }
    let (dest, src, len) = (dest_param.reg, src_param.reg, len_param.reg);
    let MachineTerminator::Branch {
        cond:
            MachineBranchCond::IntCompare {
                width: MachineIntWidth::I32,
                kind: MachineCompareKind::Gt,
                sign: MachineSign::Unsigned,
                lhs: MachineValue::Reg(cond_dest),
                rhs: MachineValue::Reg(cond_src),
            },
        then_edge: backward_edge,
        else_edge: forward_edge,
    } = &dispatch.terminator
    else {
        return None;
    };
    if (*cond_dest, *cond_src) != (dest, src)
        || !edge_args_are(backward_edge, &[dest, src, len])
        || !edge_args_are(forward_edge, &[dest, src, len])
    {
        return None;
    }

    let backward = block(blocks, backward_edge.target)?;
    let forward = block(blocks, forward_edge.target)?;
    if !params_are(backward, &[dest, src, len]) || !params_are(forward, &[dest, src, len]) {
        return None;
    }

    let [back_index_inst, back_end_inst] = backward.ops.as_slice() else {
        return None;
    };
    let back_index = match_i32_sub_one(back_index_inst, len)?;
    let back_end = match_move_imm(back_end_inst, u32::MAX as u64)?;
    if [dest, src, back_end].contains(&back_index) || [dest, src].contains(&back_end) {
        return None;
    }
    let MachineTerminator::Jump(back_join) = &backward.terminator else {
        return None;
    };
    if !edge_values_are(
        back_join,
        &[
            reg(back_end),
            reg(src),
            reg(back_end),
            reg(back_index),
            reg(dest),
        ],
    ) {
        return None;
    }

    let [forward_index_inst, forward_step_inst] = forward.ops.as_slice() else {
        return None;
    };
    let forward_index = match_move_imm(forward_index_inst, 0)?;
    let forward_step = match_move_imm(forward_step_inst, 1)?;
    if [dest, src, len].contains(&forward_index)
        || [dest, src, len, forward_index].contains(&forward_step)
    {
        return None;
    }
    let MachineTerminator::Jump(forward_join) = &forward.terminator else {
        return None;
    };
    if back_join.target != forward_join.target
        || !edge_values_are(
            forward_join,
            &[
                reg(forward_step),
                reg(src),
                reg(len),
                reg(forward_index),
                reg(dest),
            ],
        )
    {
        return None;
    }

    let join = block(blocks, back_join.target)?;
    let [join_step_param, join_src, join_end, join_index, join_dest] = join.params.as_slice()
    else {
        return None;
    };
    let [join_move] = join.ops.as_slice() else {
        return None;
    };
    let join_step = match_step_binding(join_move, join_step_param.reg)?;
    if let CopyStep::Register(step) = join_step {
        if [join_src.reg, join_end.reg, join_index.reg, join_dest.reg].contains(&step) {
            return None;
        }
    }
    let MachineTerminator::Jump(header_edge) = &join.terminator else {
        return None;
    };
    let joined = loop_regs(
        join_src.reg,
        join_end.reg,
        join_index.reg,
        join_step,
        join_dest.reg,
    );
    if !edge_args_are(header_edge, &joined) {
        return None;
    }
    let header = block(blocks, header_edge.target)?;
    let (header_regs, header_step) = copy_loop_params(header, join_step)?;
    let [header_src, header_end, header_index, header_dest] = header_regs;
    let (body_edge, return_edge) = copy_loop_edges(header, header_end, header_index)?;
    let carried = loop_regs(
        header_src,
        header_end,
        header_index,
        header_step,
        header_dest,
    );
    if !edge_args_are(body_edge, &carried) || !return_edge.args.is_empty() {
        return None;
    }
    let return_block = block(blocks, return_edge.target)?;
    if !return_block.ops.is_empty() || !matches!(return_block.terminator, MachineTerminator::Return)
    {
        return None;
    }
    let body = block(blocks, body_edge.target)?;
    let (body_regs, body_step) = copy_loop_params(body, header_step)?;
    if !match_copy_body(body, body_regs, body_step, header.id) {
        return None;
    }
    Some((dest, src, len))
}

#[derive(Clone, Copy)]
enum CopyStep {
    Register(MachineReg),
    Frame(MachineAddr),
}

fn match_step_binding(inst: &MachineInst, source: MachineReg) -> Option<CopyStep> {
    if let Some(dst) = match_move_reg(inst, source) {
        return Some(CopyStep::Register(dst));
    }
    match inst.kind {
        MachineInstKind::Store {
            ty: MachineStorageType::GpWord,
            addr,
            width: MachineMemWidth::U64,
            src: MachineValue::Reg(value),
        } if addr.base == MACHINE_FP_REG && value == source => Some(CopyStep::Frame(addr)),
        _ => None,
    }
}

fn loop_regs(
    src: MachineReg,
    end: MachineReg,
    index: MachineReg,
    step: CopyStep,
    dest: MachineReg,
) -> collections::Vec<MachineReg> {
    let mut regs = collections::vec![src, end, index];
    if let CopyStep::Register(step) = step {
        regs.push(step);
    }
    regs.push(dest);
    regs
}

fn copy_loop_params(block: &MachineBlock, step: CopyStep) -> Option<([MachineReg; 4], CopyStep)> {
    if block
        .params
        .iter()
        .any(|p| p.ty != MachineStorageType::GpWord)
    {
        return None;
    }
    match (block.params.as_slice(), step) {
        ([src, end, index, step, dest], CopyStep::Register(_)) => Some((
            [src.reg, end.reg, index.reg, dest.reg],
            CopyStep::Register(step.reg),
        )),
        ([src, end, index, dest], CopyStep::Frame(addr)) => Some((
            [src.reg, end.reg, index.reg, dest.reg],
            CopyStep::Frame(addr),
        )),
        _ => None,
    }
}

fn copy_loop_edges(
    block: &MachineBlock,
    end: MachineReg,
    index: MachineReg,
) -> Option<(&MachineEdge, &MachineEdge)> {
    let MachineTerminator::Branch {
        cond,
        then_edge,
        else_edge,
    } = &block.terminator
    else {
        return None;
    };
    let MachineBranchCond::IntCompare {
        width: MachineIntWidth::I32,
        kind,
        sign: MachineSign::Unsigned,
        lhs,
        rhs,
    } = cond
    else {
        return None;
    };
    let equal_on_true = match (block.ops.as_slice(), lhs, rhs) {
        ([], MachineValue::Reg(a), MachineValue::Reg(b))
            if (*a, *b) == (end, index) || (*a, *b) == (index, end) =>
        {
            match kind {
                MachineCompareKind::Eq => true,
                MachineCompareKind::Ne => false,
                _ => return None,
            }
        }
        ([compare], MachineValue::Reg(result), MachineValue::Imm64(0))
            if match_i32_eq(compare, end, index) == Some(*result)
                && !block.params.iter().any(|param| param.reg == *result) =>
        {
            match kind {
                MachineCompareKind::Eq => false,
                MachineCompareKind::Ne => true,
                _ => return None,
            }
        }
        _ => return None,
    };
    if equal_on_true {
        Some((else_edge, then_edge))
    } else {
        Some((then_edge, else_edge))
    }
}

fn match_copy_body(
    body: &MachineBlock,
    regs: [MachineReg; 4],
    step: CopyStep,
    header: MachineBlockId,
) -> bool {
    let [src, end, index, dest] = regs;
    let (copy_ops, increment, step_reg) = match (body.ops.as_slice(), step) {
        ([dest_add, src_add, load, store, increment], CopyStep::Register(step)) => {
            ([dest_add, src_add, load, store], increment, step)
        }
        ([dest_add, src_add, load, store, reload, increment], CopyStep::Frame(addr)) => {
            let MachineInstKind::Load {
                ty: MachineStorageType::GpWord,
                dst,
                addr: loaded_addr,
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
                ..
            } = reload.kind
            else {
                return false;
            };
            if loaded_addr != addr || regs.contains(&dst) {
                return false;
            }
            ([dest_add, src_add, load, store], increment, dst)
        }
        _ => return false,
    };
    let [dest_add, src_add, load, store] = copy_ops;
    let Some(dest_addr) = match_i32_add(dest_add, dest, index) else {
        return false;
    };
    let Some(src_addr) = match_i32_add(src_add, src, index) else {
        return false;
    };
    let carried = loop_regs(src, end, index, step, dest);
    if carried.contains(&dest_addr) || carried.contains(&src_addr) || dest_addr == src_addr {
        return false;
    }
    if !matches!(load.kind, MachineInstKind::IndexedLoad {
        dst, base: MACHINE_MEM0_BASE_REG, index: load_index,
        index_extend: MachineIndexExtend::ZeroExtend32, offset: 0,
        width: MachineMemWidth::U8, extension: MachineLoadExtension::ZeroExtend,
    } if dst == src_addr && load_index == src_addr)
        || !matches!(store.kind, MachineInstKind::IndexedStore {
            base: MACHINE_MEM0_BASE_REG, index: store_index,
            index_extend: MachineIndexExtend::ZeroExtend32, offset: 0,
            width: MachineMemWidth::U8, src: MachineValue::Reg(value),
        } if store_index == dest_addr && value == src_addr)
        || match_i32_add(increment, index, step_reg) != Some(index)
    {
        return false;
    }
    matches!(&body.terminator, MachineTerminator::Jump(edge)
        if edge.target == header && edge_args_are(edge, &carried))
}

fn block(blocks: &[MachineBlock], id: MachineBlockId) -> Option<&MachineBlock> {
    blocks
        .get(id.as_usize())
        .filter(|block| block.id == id)
        .or_else(|| blocks.iter().find(|block| block.id == id))
}

fn params_are(block: &MachineBlock, regs: &[MachineReg]) -> bool {
    block.params.len() == regs.len()
        && block
            .params
            .iter()
            .zip(regs)
            .all(|(param, reg)| param.reg == *reg)
}

fn edge_args_are(
    edge: &crate::vm::jit::machine::machine_ir::MachineEdge,
    regs: &[MachineReg],
) -> bool {
    edge_values_are(
        edge,
        &regs
            .iter()
            .copied()
            .map(reg)
            .collect::<collections::Vec<_>>(),
    )
}

fn edge_values_are(
    edge: &crate::vm::jit::machine::machine_ir::MachineEdge,
    values: &[MachineValue],
) -> bool {
    edge.args.as_slice() == values
}

const fn reg(value: MachineReg) -> MachineValue {
    MachineValue::Reg(value)
}

fn match_move_imm(inst: &MachineInst, imm: u64) -> Option<MachineReg> {
    match inst.kind {
        MachineInstKind::Move {
            dst,
            src: MachineValue::Imm64(value),
            ..
        } if value == imm => Some(dst),
        _ => None,
    }
}

fn match_move_reg(inst: &MachineInst, src: MachineReg) -> Option<MachineReg> {
    match inst.kind {
        MachineInstKind::Move {
            dst,
            src: MachineValue::Reg(value),
            ..
        } if value == src => Some(dst),
        _ => None,
    }
}

fn match_i32_sub_one(inst: &MachineInst, lhs: MachineReg) -> Option<MachineReg> {
    match inst.kind {
        MachineInstKind::IntBinary {
            width: MachineIntWidth::I32,
            op: MachineIntBinaryOp::Sub,
            dst,
            lhs: MachineValue::Reg(value),
            rhs: MachineValue::Imm64(1),
        } if value == lhs => Some(dst),
        _ => None,
    }
}

fn match_i32_add(inst: &MachineInst, lhs: MachineReg, rhs: MachineReg) -> Option<MachineReg> {
    match inst.kind {
        MachineInstKind::IntBinary {
            width: MachineIntWidth::I32,
            op: MachineIntBinaryOp::Add,
            dst,
            lhs: MachineValue::Reg(a),
            rhs: MachineValue::Reg(b),
        } if (a, b) == (lhs, rhs) || (a, b) == (rhs, lhs) => Some(dst),
        _ => None,
    }
}

fn match_i32_eq(inst: &MachineInst, lhs: MachineReg, rhs: MachineReg) -> Option<MachineReg> {
    match inst.kind {
        MachineInstKind::IntCompare {
            width: MachineIntWidth::I32,
            kind: MachineCompareKind::Eq,
            sign: MachineSign::Unsigned,
            dst,
            lhs: MachineValue::Reg(a),
            rhs: MachineValue::Reg(b),
        } if (a, b) == (lhs, rhs) || (a, b) == (rhs, lhs) => Some(dst),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::jit::machine::machine_ir::MachineRegOwner;

    fn inst(kind: MachineInstKind) -> MachineInst {
        MachineInst { kind }
    }
    fn imm(dst: u16, value: u64) -> MachineInst {
        inst(MachineInstKind::Move {
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            dst: MachineReg(dst),
            src: MachineValue::Imm64(value),
        })
    }
    fn add(dst: u16, lhs: u16, rhs: u16) -> MachineInst {
        inst(MachineInstKind::IntBinary {
            width: MachineIntWidth::I32,
            op: MachineIntBinaryOp::Add,
            dst: MachineReg(dst),
            lhs: reg(MachineReg(lhs)),
            rhs: reg(MachineReg(rhs)),
        })
    }
    fn edge(target: u32, regs: &[u16]) -> MachineEdge {
        MachineEdge {
            target: MachineBlockId(target),
            args: regs.iter().map(|r| reg(MachineReg(*r))).collect(),
        }
    }
    fn block(
        id: u32,
        params: &[u16],
        ops: collections::Vec<MachineInst>,
        terminator: MachineTerminator,
    ) -> MachineBlock {
        MachineBlock {
            id: MachineBlockId(id),
            params: params
                .iter()
                .map(|r| MachineBlockParam::gp_word(MachineReg(*r)))
                .collect(),
            ops,
            terminator,
        }
    }
    fn fixture(frame_step: bool, double_compare: bool) -> MachineProgram {
        let addr = MachineAddr {
            base: MACHINE_FP_REG,
            offset: 32,
        };
        let carried = if frame_step {
            collections::vec![5, 6, 7, 8]
        } else {
            collections::vec![5, 6, 7, 9, 8]
        };
        let temp = if frame_step { 9 } else { 10 };
        let mut body = collections::vec![
            add(4, 8, 7),
            add(temp, 5, 7),
            inst(MachineInstKind::IndexedLoad {
                dst: MachineReg(temp),
                base: MACHINE_MEM0_BASE_REG,
                index: MachineReg(temp),
                index_extend: MachineIndexExtend::ZeroExtend32,
                offset: 0,
                width: MachineMemWidth::U8,
                extension: MachineLoadExtension::ZeroExtend
            }),
            inst(MachineInstKind::IndexedStore {
                base: MACHINE_MEM0_BASE_REG,
                index: MachineReg(4),
                index_extend: MachineIndexExtend::ZeroExtend32,
                offset: 0,
                width: MachineMemWidth::U8,
                src: reg(MachineReg(temp))
            }),
        ];
        if frame_step {
            body.push(inst(MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: MachineReg(4),
                addr,
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
            }));
        }
        body.push(add(7, 7, if frame_step { 4 } else { 9 }));
        let compare = MachineInstKind::IntCompare {
            width: MachineIntWidth::I32,
            kind: MachineCompareKind::Eq,
            sign: MachineSign::Unsigned,
            dst: MachineReg(4),
            lhs: reg(MachineReg(6)),
            rhs: reg(MachineReg(7)),
        };
        let condition = if double_compare {
            MachineBranchCond::IntCompare {
                width: MachineIntWidth::I32,
                kind: MachineCompareKind::Eq,
                sign: MachineSign::Unsigned,
                lhs: reg(MachineReg(4)),
                rhs: MachineValue::Imm64(0),
            }
        } else {
            MachineBranchCond::IntCompare {
                width: MachineIntWidth::I32,
                kind: MachineCompareKind::Ne,
                sign: MachineSign::Unsigned,
                lhs: reg(MachineReg(6)),
                rhs: reg(MachineReg(7)),
            }
        };
        MachineProgram {
            entry: MachineBlockId(0),
            fp_reg_init_widths: collections::vec![],
            blocks: collections::vec![
                block(
                    0,
                    &[4, 5, 6],
                    collections::vec![],
                    MachineTerminator::Branch {
                        cond: MachineBranchCond::IntCompare {
                            width: MachineIntWidth::I32,
                            kind: MachineCompareKind::Gt,
                            sign: MachineSign::Unsigned,
                            lhs: reg(MachineReg(4)),
                            rhs: reg(MachineReg(5))
                        },
                        then_edge: edge(1, &[4, 5, 6]),
                        else_edge: edge(2, &[4, 5, 6])
                    }
                ),
                block(
                    1,
                    &[4, 5, 6],
                    collections::vec![
                        inst(MachineInstKind::IntBinary {
                            width: MachineIntWidth::I32,
                            op: MachineIntBinaryOp::Sub,
                            dst: MachineReg(7),
                            lhs: reg(MachineReg(6)),
                            rhs: MachineValue::Imm64(1)
                        }),
                        imm(6, u32::MAX as u64)
                    ],
                    MachineTerminator::Jump(edge(3, &[6, 5, 6, 7, 4]))
                ),
                block(
                    2,
                    &[4, 5, 6],
                    collections::vec![imm(7, 0), imm(8, 1)],
                    MachineTerminator::Jump(edge(3, &[8, 5, 6, 7, 4]))
                ),
                block(
                    3,
                    &[4, 5, 6, 7, 8],
                    collections::vec![if frame_step {
                        inst(MachineInstKind::Store {
                            ty: MachineStorageType::GpWord,
                            addr,
                            width: MachineMemWidth::U64,
                            src: reg(MachineReg(4)),
                        })
                    } else {
                        inst(MachineInstKind::Move {
                            owner: MachineRegOwner::LinearValue,
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(9),
                            src: reg(MachineReg(4)),
                        })
                    }],
                    MachineTerminator::Jump(edge(4, &carried))
                ),
                block(
                    4,
                    &carried,
                    if double_compare {
                        collections::vec![inst(compare)]
                    } else {
                        collections::vec![]
                    },
                    MachineTerminator::Branch {
                        cond: condition,
                        then_edge: edge(5, &carried),
                        else_edge: edge(6, &[])
                    }
                ),
                block(
                    5,
                    &carried,
                    body,
                    MachineTerminator::Jump(edge(4, &carried))
                ),
                block(6, &[], collections::vec![], MachineTerminator::Return),
            ],
        }
    }
    fn config() -> BackendConfig {
        BackendConfig::new(8, 8, 0, 3)
    }

    #[test]
    fn recognizes_both_step_locations_and_predicates_with_guarded_fallback() {
        for frame in [false, true] {
            for double in [false, true] {
                let mut program = fixture(frame, double);
                let original = program.clone();
                recognize_memmove(&mut program, config());
                assert_eq!(program.blocks.len(), original.blocks.len() + 5);
                let mut fallback = program.blocks[7].clone();
                fallback.id = MachineBlockId(0);
                assert_eq!(fallback, original.blocks[0]);
                assert_eq!(&program.blocks[1..7], &original.blocks[1..]);
                for index in [10, 11] {
                    assert!(matches!(
                        program.blocks[index].terminator,
                        MachineTerminator::Branch {
                            cond: MachineBranchCond::IntCompare {
                                width: MachineIntWidth::I64,
                                kind: MachineCompareKind::Le,
                                rhs: MachineValue::Reg(MACHINE_MEM0_SIZE_REG),
                                ..
                            },
                            else_edge: MachineEdge {
                                target: MachineBlockId(7),
                                ..
                            },
                            ..
                        }
                    ));
                }
                for index in [0, 8] {
                    assert!(matches!(
                        program.blocks[index].terminator,
                        MachineTerminator::Branch {
                            cond: MachineBranchCond::IntCompare {
                                width: MachineIntWidth::I64,
                                kind: MachineCompareKind::Le,
                                rhs: MachineValue::Imm64(0x1_0000_0000),
                                ..
                            },
                            else_edge: MachineEdge {
                                target: MachineBlockId(7),
                                ..
                            },
                            ..
                        }
                    ));
                }
                assert!(matches!(
                    program.blocks[9].ops[0].kind,
                    MachineInstKind::MemoryCopy { .. }
                ));
            }
        }
    }

    #[test]
    fn rejects_changes_to_copy_semantics_or_step_storage() {
        for mutation in 0..9 {
            let mut program = fixture(mutation != 7, mutation == 8);
            match mutation {
                0 => program.blocks[0].ops.push(imm(7, 42)),
                1 => program.blocks[2].ops[1] = imm(8, 2),
                2 => {
                    if let MachineInstKind::IndexedLoad { offset, .. } =
                        &mut program.blocks[5].ops[2].kind
                    {
                        *offset = 1;
                    }
                }
                3 => {
                    if let MachineInstKind::IndexedStore { width, .. } =
                        &mut program.blocks[5].ops[3].kind
                    {
                        *width = MachineMemWidth::U16;
                    }
                }
                4 => {
                    if let MachineInstKind::Load { addr, .. } = &mut program.blocks[5].ops[4].kind {
                        addr.offset += 8;
                    }
                }
                5 => program.blocks[5].ops[5] = add(7, 7, 6),
                6 => program.blocks[5].ops[0] = add(5, 8, 7),
                7 => {
                    if let MachineInstKind::Move { dst, .. } = &mut program.blocks[3].ops[0].kind {
                        *dst = MachineReg(5);
                    }
                }
                8 => {
                    if let MachineInstKind::IntCompare { dst, .. } =
                        &mut program.blocks[4].ops[0].kind
                    {
                        *dst = MachineReg(6);
                    }
                    if let MachineTerminator::Branch {
                        cond: MachineBranchCond::IntCompare { lhs, .. },
                        ..
                    } = &mut program.blocks[4].terminator
                    {
                        *lhs = reg(MachineReg(6));
                    }
                }
                _ => unreachable!(),
            }
            let original = program.clone();
            recognize_memmove(&mut program, config());
            assert_eq!(program, original, "mutation {mutation}");
        }
    }

    #[test]
    fn leaves_32bit_targets_on_the_original_loop() {
        let mut program = fixture(false, true);
        let original = program.clone();
        recognize_memmove(&mut program, BackendConfig::new(4, 12, 0, 8));
        assert_eq!(program, original);
    }
}
