//! Recognize the canonical overlap-safe byte-copy loop emitted for Rust's
//! `memmove` intrinsic and replace it with MachineIR's bulk memory operation.
//!
//! LLVM occasionally leaves this helper as two scalar byte loops in Wasm
//! rather than emitting `memory.copy`. Replaying that loop in native code is
//! especially expensive for parsers and allocators. The matcher below is
//! deliberately strict: it proves both the forward and backward copy paths,
//! their direction selection, and the byte load/store before rewriting the
//! dispatch block.

use crate::collections;
use crate::vm::machine::machine_ir::{
    MachineBlock, MachineBlockId, MachineBranchCond, MachineCompareKind, MachineIndexExtend,
    MachineInst, MachineInstKind, MachineIntBinaryOp, MachineIntWidth, MachineLoadExtension,
    MachineMemWidth, MachineProgram, MachineReg, MachineSign, MachineStorageType,
    MachineTerminator, MachineValue, MACHINE_MEM0_BASE_REG,
};

pub(super) fn recognize_memmove(program: &mut MachineProgram) {
    let Some((dispatch_index, dest, src, len)) = find_memmove_dispatch(&program.blocks) else {
        return;
    };
    let dispatch = &mut program.blocks[dispatch_index];
    dispatch.ops.clear();
    dispatch.ops.push(MachineInst {
        kind: MachineInstKind::MemoryCopy {
            dst_mem: 0,
            src_mem: 0,
            dest: MachineValue::Reg(dest),
            src: MachineValue::Reg(src),
            len: MachineValue::Reg(len),
        },
    });
    dispatch.terminator = MachineTerminator::Return;
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
    let join_step = match_move_reg(join_move, join_step_param.reg)?;
    let MachineTerminator::Jump(header_edge) = &join.terminator else {
        return None;
    };
    if !edge_values_are(
        header_edge,
        &[
            reg(join_src.reg),
            reg(join_end.reg),
            reg(join_index.reg),
            reg(join_step),
            reg(join_dest.reg),
        ],
    ) {
        return None;
    }

    let header = block(blocks, header_edge.target)?;
    let [header_src, header_end, header_index, header_step, header_dest] = header.params.as_slice()
    else {
        return None;
    };
    let [compare] = header.ops.as_slice() else {
        return None;
    };
    let compare_result = match_i32_eq(compare, header_end.reg, header_index.reg)?;
    let MachineTerminator::Branch {
        cond:
            MachineBranchCond::IntCompare {
                width: MachineIntWidth::I32,
                kind: MachineCompareKind::Eq,
                sign: MachineSign::Unsigned,
                lhs: MachineValue::Reg(cond_result),
                rhs: MachineValue::Imm64(0),
            },
        then_edge: body_edge,
        else_edge: return_edge,
    } = &header.terminator
    else {
        return None;
    };
    if *cond_result != compare_result
        || !edge_args_are(
            body_edge,
            &[
                header_src.reg,
                header_end.reg,
                header_index.reg,
                header_step.reg,
                header_dest.reg,
            ],
        )
        || !return_edge.args.is_empty()
    {
        return None;
    }

    let return_block = block(blocks, return_edge.target)?;
    if !return_block.ops.is_empty() || !matches!(return_block.terminator, MachineTerminator::Return)
    {
        return None;
    }
    let body = block(blocks, body_edge.target)?;
    let [body_src, body_end, body_index, body_step, body_dest] = body.params.as_slice() else {
        return None;
    };
    if !match_copy_body(
        body,
        body_src.reg,
        body_end.reg,
        body_index.reg,
        body_step.reg,
        body_dest.reg,
        header.id,
    ) {
        return None;
    }
    Some((dest, src, len))
}

fn match_copy_body(
    body: &MachineBlock,
    src: MachineReg,
    end: MachineReg,
    index: MachineReg,
    step: MachineReg,
    dest: MachineReg,
    header: MachineBlockId,
) -> bool {
    let [dest_add, src_add, load, store, increment] = body.ops.as_slice() else {
        return false;
    };
    let Some(dest_addr) = match_i32_add(dest_add, dest, index) else {
        return false;
    };
    let Some(src_addr) = match_i32_add(src_add, src, index) else {
        return false;
    };
    if !matches!(
        load.kind,
        MachineInstKind::IndexedLoad {
            dst,
            base: MACHINE_MEM0_BASE_REG,
            index: load_index,
            index_extend: MachineIndexExtend::ZeroExtend32,
            offset: 0,
            width: MachineMemWidth::U8,
            extension: MachineLoadExtension::ZeroExtend,
        } if dst == src_addr && load_index == src_addr
    ) || !matches!(
        store.kind,
        MachineInstKind::IndexedStore {
            base: MACHINE_MEM0_BASE_REG,
            index: store_index,
            index_extend: MachineIndexExtend::ZeroExtend32,
            offset: 0,
            width: MachineMemWidth::U8,
            src: MachineValue::Reg(value),
        } if store_index == dest_addr && value == src_addr
    ) || match_i32_add(increment, index, step) != Some(index)
    {
        return false;
    }
    matches!(
        &body.terminator,
        MachineTerminator::Jump(edge)
            if edge.target == header && edge_args_are(edge, &[src, end, index, step, dest])
    )
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

fn edge_args_are(edge: &crate::vm::machine::machine_ir::MachineEdge, regs: &[MachineReg]) -> bool {
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
    edge: &crate::vm::machine::machine_ir::MachineEdge,
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
