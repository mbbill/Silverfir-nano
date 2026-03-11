//! Finalize shared LIR into fast-interpreter instructions.

use alloc::vec::Vec;

use crate::vm::{
    entities::{FunctionInst, ModuleInst},
    interp::{
        build::FastBuildBundle, encoding, fast_code::FastCode, frame_layout, handlers::full_set,
        instruction::Instruction, resolve,
    },
    lir::legacy::ir::{LirBrTableEntry, LirFrameCopy, LirMarkerKind, LirOp, LirOpKind},
    plan::{
        frame::{FrameLayoutPlan, FrameSlot, FrameSpan},
        tos::{SpillArtifact, TosTransferDirection},
    },
    wasm::core_op::CoreOpKind,
};

#[derive(Clone, Debug, Default)]
pub struct FastFinalized {
    pub code: FastCode,
}

pub fn finalize_fast(bundle: &FastBuildBundle, module: &ModuleInst) -> FastFinalized {
    let frame = bundle.planned.frame;
    let mut instructions = Vec::with_capacity(bundle.lir.ops.len() + 2);
    let mut start_indices = Vec::with_capacity(bundle.lir.ops.len());

    for op in &bundle.lir.ops {
        start_indices.push(instructions.len());
        instructions.push(encode_op(op, module, frame));
        if let LirOpKind::BrTable { entries, .. } = &op.kind {
            for _ in 0..data_slot_count(entries.len()) {
                instructions.push(Instruction::new_handler_only(full_set::op_data));
            }
        }
    }

    let term_index = instructions.len();
    instructions.push(Instruction::new_handler_only(full_set::op_term));
    instructions.push(Instruction::new_handler_only(full_set::op_term));

    patch_br_tables(
        &mut instructions,
        &bundle.lir.ops,
        &start_indices,
        term_index,
        frame,
    );

    // Direct targets are absolute instruction pointers, so patch them only
    // after the instruction buffer is in its final stable allocation.
    let mut code = FastCode::from_instructions(instructions);
    patch_direct_targets(
        &mut code.instructions,
        &bundle.lir.ops,
        &start_indices,
        term_index,
        frame,
        module,
    );

    FastFinalized { code }
}

fn encode_op(op: &LirOp, module: &ModuleInst, frame: FrameLayoutPlan) -> Instruction {
    let handler = resolve::resolve_handler_for_op(&op.kind, resolve::handler_variant(op));
    let (imm0, imm1, imm2) = encode_immediates(op, module, frame, 0);
    Instruction::new(handler, imm0, imm1, imm2)
}

fn encode_immediates(
    op: &LirOp,
    module: &ModuleInst,
    frame: FrameLayoutPlan,
    target_ptr: u64,
) -> (u64, u64, u64) {
    match &op.kind {
        LirOpKind::Core(kind) => encode_core(kind),
        LirOpKind::InitLocals { l0, l1, l2 } => encoding::init_locals::encode(l0.0, l1.0, l2.0),
        LirOpKind::CacheTransfer(SpillArtifact::TosTransfer(transfer)) => {
            let slot = physical_top_slot(frame, transfer.frame);
            match (transfer.direction, transfer.frame.count) {
                (TosTransferDirection::Spill, 1) => encoding::spill_1::encode(slot),
                (TosTransferDirection::Spill, 2) => encoding::spill_2::encode(slot),
                (TosTransferDirection::Spill, 3) => encoding::spill_3::encode(slot),
                (TosTransferDirection::Spill, _) => encoding::spill_4::encode(slot),
                (TosTransferDirection::Fill, 1) => encoding::fill_1::encode(slot),
                (TosTransferDirection::Fill, 2) => encoding::fill_2::encode(slot),
                (TosTransferDirection::Fill, 3) => encoding::fill_3::encode(slot),
                (TosTransferDirection::Fill, _) => encoding::fill_4::encode(slot),
            }
        }
        LirOpKind::LocalGetHot { reg } => encoding::local_get::encode(*reg as u16),
        LirOpKind::LocalSetHot { reg } => encoding::local_set::encode(*reg as u16),
        LirOpKind::LocalTeeHot { reg } => encoding::local_tee::encode(*reg as u16),
        LirOpKind::LocalGetFrame { frame_slot } => {
            encoding::local_get::encode(physical_slot(frame, *frame_slot))
        }
        LirOpKind::LocalSetFrame { frame_slot } => {
            encoding::local_set::encode(physical_slot(frame, *frame_slot))
        }
        LirOpKind::LocalTeeFrame { frame_slot } => {
            encoding::local_tee::encode(physical_slot(frame, *frame_slot))
        }
        LirOpKind::Marker(LirMarkerKind::If { .. }) => encoding::if_::encode(target_ptr),
        LirOpKind::Marker(LirMarkerKind::Else) => encoding::else_::encode(target_ptr),
        LirOpKind::Marker(_) => (0, 0, 0),
        LirOpKind::Branch {
            kind: crate::vm::plan::PlannedBranchKind::Br,
            fixup,
            ..
        } => {
            let (src_slot, dst_slot, count) = copy_slots(frame, *fixup);
            encoding::br::encode(target_ptr, src_slot, dst_slot, count)
        }
        LirOpKind::Branch {
            kind: crate::vm::plan::PlannedBranchKind::BrIf,
            fixup,
            ..
        } => {
            let (src_slot, dst_slot, count) = copy_slots(frame, *fixup);
            if count == 0 {
                encoding::br_if_simple::encode(target_ptr)
            } else {
                encoding::br_if::encode(target_ptr, src_slot, dst_slot, count)
            }
        }
        LirOpKind::BrTable { entries, .. } => {
            encoding::br_table::encode(entries.len() as u64, data_slot_count(entries.len()) as u64)
        }
        LirOpKind::CallExternal { func_idx, args, .. } => {
            encoding::call_external::encode(*func_idx as u64, physical_slot(frame, args.start))
        }
        LirOpKind::CallInternal { callee, args, .. } => encoding::call_internal::encode(
            module_function_ptr(module, *callee),
            physical_slot(frame, args.start),
        ),
        LirOpKind::CallIndirect {
            type_idx,
            table_idx,
            index_slot,
            args,
            ..
        } => encoding::call_indirect::encode(
            *type_idx as u64,
            *table_idx as u64,
            physical_slot(frame, args.start),
            physical_slot(frame, *index_slot),
        ),
        LirOpKind::Return { results: None } => encoding::return_void::encode(frame.frame_size),
        LirOpKind::Return {
            results: Some(results),
        } if results.count == 1 => {
            encoding::return_one::encode(frame.frame_size, physical_slot(frame, results.start))
        }
        LirOpKind::Return {
            results: Some(results),
        } => encoding::r#return::encode(
            results.count,
            frame.frame_size,
            physical_slot(frame, results.start),
        ),
    }
}

fn patch_direct_targets(
    instructions: &mut [Instruction],
    ops: &[LirOp],
    start_indices: &[usize],
    term_index: usize,
    frame: FrameLayoutPlan,
    module: &ModuleInst,
) {
    let base = instructions.as_ptr() as usize;

    for (idx, op) in ops.iter().enumerate() {
        let target = match &op.kind {
            LirOpKind::Marker(LirMarkerKind::If { .. })
            | LirOpKind::Marker(LirMarkerKind::Else) => op.alt,
            LirOpKind::Branch { target, .. } => *target,
            _ => None,
        };

        let Some(target) = target else {
            continue;
        };
        let target_index = start_indices
            .get(target.as_usize())
            .copied()
            .unwrap_or(term_index);
        let target_ptr = (base + target_index * core::mem::size_of::<Instruction>()) as u64;
        let (imm0, imm1, imm2) = encode_immediates(op, module, frame, target_ptr);
        instructions[start_indices[idx]].imm0 = imm0;
        instructions[start_indices[idx]].imm1 = imm1;
        instructions[start_indices[idx]].imm2 = imm2;
    }
}

fn patch_br_tables(
    instructions: &mut [Instruction],
    ops: &[LirOp],
    start_indices: &[usize],
    term_index: usize,
    frame: FrameLayoutPlan,
) {
    for (op_idx, op) in ops.iter().enumerate() {
        let LirOpKind::BrTable { entries, .. } = &op.kind else {
            continue;
        };

        let inst_index = start_indices[op_idx];
        for slot_idx in 0..data_slot_count(entries.len()) {
            let data_index = inst_index + 1 + slot_idx;
            let entry = entries.get(slot_idx).copied();
            let (imm0, imm1, imm2) =
                pack_br_table_entry(entry, inst_index, start_indices, term_index, frame);
            instructions[data_index].handler = full_set::op_data;
            instructions[data_index].imm0 = imm0;
            instructions[data_index].imm1 = imm1;
            instructions[data_index].imm2 = imm2;
        }
    }
}

fn pack_br_table_entry(
    entry: Option<LirBrTableEntry>,
    br_index: usize,
    start_indices: &[usize],
    term_index: usize,
    frame: FrameLayoutPlan,
) -> (u64, u64, u64) {
    let Some(entry) = entry else {
        return (0, 0, 0);
    };

    let rel = br_table_rel(entry.target, br_index, start_indices, term_index);
    let (src_slot, dst_slot, count) = copy_slots(frame, entry.fixup);
    (
        rel as i32 as u64,
        (src_slot as u64) | ((dst_slot as u64) << 16) | ((count as u64) << 32),
        0,
    )
}

fn br_table_rel(
    target: Option<crate::vm::lir::target::LirTarget>,
    br_index: usize,
    start_indices: &[usize],
    term_index: usize,
) -> i32 {
    let target_index = target
        .and_then(|target| start_indices.get(target.as_usize()).copied())
        .unwrap_or(term_index);
    target_index as i32 - br_index as i32
}

fn encode_core(kind: &CoreOpKind) -> (u64, u64, u64) {
    match kind {
        CoreOpKind::I32Const { value } => encoding::r#const::encode(*value as u64),
        CoreOpKind::I64Const { value } => encoding::r#const::encode(*value),
        CoreOpKind::F32Const { value } => encoding::r#const::encode(*value as u64),
        CoreOpKind::F64Const { value } => encoding::r#const::encode(*value),

        CoreOpKind::I32Load { offset, memidx }
        | CoreOpKind::I64Load { offset, memidx }
        | CoreOpKind::F32Load { offset, memidx }
        | CoreOpKind::F64Load { offset, memidx }
        | CoreOpKind::I32Load8S { offset, memidx }
        | CoreOpKind::I32Load8U { offset, memidx }
        | CoreOpKind::I32Load16S { offset, memidx }
        | CoreOpKind::I32Load16U { offset, memidx }
        | CoreOpKind::I64Load8S { offset, memidx }
        | CoreOpKind::I64Load8U { offset, memidx }
        | CoreOpKind::I64Load16S { offset, memidx }
        | CoreOpKind::I64Load16U { offset, memidx }
        | CoreOpKind::I64Load32S { offset, memidx }
        | CoreOpKind::I64Load32U { offset, memidx } => encoding::load::encode(*offset, *memidx),

        CoreOpKind::I32Store { offset, memidx }
        | CoreOpKind::I64Store { offset, memidx }
        | CoreOpKind::F32Store { offset, memidx }
        | CoreOpKind::F64Store { offset, memidx }
        | CoreOpKind::I32Store8 { offset, memidx }
        | CoreOpKind::I32Store16 { offset, memidx }
        | CoreOpKind::I64Store8 { offset, memidx }
        | CoreOpKind::I64Store16 { offset, memidx }
        | CoreOpKind::I64Store32 { offset, memidx } => encoding::store::encode(*offset, *memidx),

        CoreOpKind::GlobalGet { idx } | CoreOpKind::GlobalSet { idx } => {
            encoding::global::encode(*idx as u64)
        }
        CoreOpKind::MemorySize { mem_idx } => encoding::memory_size::encode(*mem_idx as u64),
        CoreOpKind::MemoryGrow { mem_idx } => encoding::memory_grow::encode(*mem_idx as u64),
        CoreOpKind::MemoryFill { imm0, imm1 }
        | CoreOpKind::MemoryCopy { imm0, imm1 }
        | CoreOpKind::MemoryInit { imm0, imm1 }
        | CoreOpKind::TableFill { imm0, imm1 }
        | CoreOpKind::TableCopy { imm0, imm1 }
        | CoreOpKind::TableInit { imm0, imm1 } => {
            encoding::ternary::encode(*imm0 as u64, *imm1 as u64)
        }
        CoreOpKind::DataDrop { data_idx } => encoding::drop_op::encode(*data_idx as u64),
        CoreOpKind::TableGet { table_idx } => encoding::table_get::encode(*table_idx as u64),
        CoreOpKind::TableSet { table_idx } => encoding::table_set::encode(*table_idx as u64),
        CoreOpKind::TableSize { table_idx } => encoding::table_size::encode(*table_idx as u64),
        CoreOpKind::TableGrow { table_idx } => encoding::table_grow::encode(*table_idx as u64),
        CoreOpKind::ElemDrop { elem_idx } => encoding::drop_op::encode(*elem_idx as u64),
        CoreOpKind::RefFunc { func_idx } => encoding::ref_func::encode(*func_idx as u64),

        _ => (0, 0, 0),
    }
}

fn data_slot_count(entry_count: usize) -> usize {
    entry_count
}

fn module_function_ptr(module: &ModuleInst, func_idx: u32) -> u64 {
    module
        .functions
        .get(func_idx as usize)
        .map(|func| func as *const FunctionInst as u64)
        .unwrap_or(0)
}

fn physical_top_slot(frame: FrameLayoutPlan, span: FrameSpan) -> u16 {
    physical_slot(frame, FrameSlot(span.end().0.saturating_sub(1)))
}

fn copy_slots(frame: FrameLayoutPlan, fixup: Option<LirFrameCopy>) -> (u16, u16, u16) {
    let Some(fixup) = fixup else {
        return (0, 0, 0);
    };
    (
        physical_slot(frame, fixup.src.start),
        physical_slot(frame, fixup.dst.start),
        fixup.src.count,
    )
}

fn physical_slot(frame: FrameLayoutPlan, slot: FrameSlot) -> u16 {
    if slot.0 < frame.operand_base {
        slot.0
    } else {
        frame_layout::operand_stack_base(frame.frame_size as usize) as u16 + slot.0
            - frame.operand_base
    }
}
