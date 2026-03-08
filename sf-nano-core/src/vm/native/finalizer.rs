//! Finalizer: Vec<ResolvedNativeInst> -> Box<[NativeInst]>.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use crate::vm::compile::StackTracker;
use crate::vm::compaction::CompactionDisposition;
use crate::vm::lowered::{IrOpKind, OpIndex, SlotRef};
use crate::vm::operand_encoding::encode_operands;

use super::CodeBuffer;
use super::debug_map;
use super::bridge::{self, HelperMetadata};
use super::instruction::{EMPTY_TOS_SLOTS, INVALID_TOS_SLOT, NativeEntry, NativeInst};
use super::resolved::{NativeResolvedVec, ResolvedNativeInst};
use super::runtime::term_entry;
use super::arm64::{EntryPatchSites, emit};

pub fn finalize(
    mut ops: NativeResolvedVec,
    stack: &mut StackTracker,
    buf: &mut CodeBuffer,
    module_name: &str,
    func_idx: u32,
) -> (Box<[NativeInst]>, Box<[HelperMetadata]>) {
    append_terminals(&mut ops);
    route_terminals(&mut ops);
    default_alt_to_term(&mut ops);
    let ops = expand_br_tables(ops);
    let keep: Vec<bool> = ops.iter().map(|op| op.compaction.is_kept()).collect();
    validate_removed_targets(&ops, &keep);
    let index_map = build_index_map(&keep);
    let compacted = compact_and_patch(ops, &keep, &index_map);
    build_instructions(compacted, stack.operand_base(), buf, module_name, func_idx)
}

fn append_terminals(ops: &mut NativeResolvedVec) {
    let term = ResolvedNativeInst {
        entry: term_entry(),
        kind: IrOpKind::Term,
        pre_height: 0,
        alt_target: None,
        has_target: false,
        cold_helper: None,
        cold_stack_slot: None,
        entry_patches: EntryPatchSites::default(),
        compaction: CompactionDisposition::Keep,
    };
    ops.push(term.clone());
    ops.push(term);
}

fn route_terminals(ops: &mut NativeResolvedVec) {
    let term_idx = ops.len() - 1;
    for op in ops.iter_mut() {
        match &op.kind {
            IrOpKind::ReturnVoid { .. }
            | IrOpKind::ReturnOne { .. }
            | IrOpKind::Return { .. }
            | IrOpKind::Unreachable => op.alt_target = Some(OpIndex::from(term_idx)),
            _ => {}
        }
    }
}

fn default_alt_to_term(ops: &mut NativeResolvedVec) {
    let term_idx = ops.len() - 1;
    for op in ops.iter_mut() {
        if op.alt_target.is_none() {
            op.alt_target = Some(OpIndex::from(term_idx));
        }
    }
}

fn expand_br_tables(ops: NativeResolvedVec) -> NativeResolvedVec {
    let mut expansion_at: Vec<usize> = vec![0; ops.len()];
    let mut total_expansion = 0;

    for (i, op) in ops.iter().enumerate() {
        expansion_at[i] = total_expansion;
        if let IrOpKind::BrTable { entries, .. } = &op.kind {
            total_expansion += (entries.len() + 1) / 2;
        }
    }

    if total_expansion == 0 {
        return ops;
    }

    let old_to_new: Vec<usize> = expansion_at
        .iter()
        .enumerate()
        .map(|(i, &exp)| i + exp)
        .collect();

    let mut ops = ops;
    for op in ops.iter_mut() {
        if let Some(ref mut alt) = op.alt_target {
            if alt.as_usize() < old_to_new.len() {
                *alt = OpIndex::from(old_to_new[alt.as_usize()]);
            }
        }
        if let IrOpKind::BrTable { ref mut entries, .. } = op.kind {
            for e in entries.iter_mut() {
                if let Some(ref mut tgt) = e.target_idx {
                    if tgt.as_usize() < old_to_new.len() {
                        *tgt = OpIndex::from(old_to_new[tgt.as_usize()]);
                    }
                }
            }
        }
    }

    let mut result = Vec::with_capacity(ops.len() + total_expansion);
    for op in ops {
        let data_slot_count = if let IrOpKind::BrTable { ref entries, .. } = op.kind {
            (entries.len() + 1) / 2
        } else {
            0
        };
        result.push(op);
        for _ in 0..data_slot_count {
            result.push(ResolvedNativeInst {
                entry: term_entry(),
                kind: IrOpKind::Data {
                    imm0: 0,
                    imm1: 0,
                    imm2: 0,
                },
                pre_height: 0,
                alt_target: None,
                has_target: false,
                cold_helper: None,
                cold_stack_slot: None,
                entry_patches: EntryPatchSites::default(),
                compaction: CompactionDisposition::Keep,
            });
        }
    }

    result
}

fn build_index_map(keep: &[bool]) -> Vec<Option<usize>> {
    let mut map = vec![None; keep.len()];
    let mut new_idx = 0;
    for (old_idx, &k) in keep.iter().enumerate() {
        if k {
            map[old_idx] = Some(new_idx);
            new_idx += 1;
        }
    }
    map
}

fn incoming_targets(ops: &[ResolvedNativeInst]) -> Vec<bool> {
    let mut incoming = vec![false; ops.len()];
    for op in ops {
        if let Some(target) = op.alt_target {
            if target.as_usize() < incoming.len() {
                incoming[target.as_usize()] = true;
            }
        }
        if let IrOpKind::BrTable { entries, .. } = &op.kind {
            for entry in entries {
                if let Some(target) = entry.target_idx {
                    if target.as_usize() < incoming.len() {
                        incoming[target.as_usize()] = true;
                    }
                }
            }
        }
    }
    incoming
}

fn validate_removed_targets(ops: &[ResolvedNativeInst], keep: &[bool]) {
    let incoming = incoming_targets(ops);
    for (idx, op) in ops.iter().enumerate() {
        if keep[idx] || !incoming[idx] {
            continue;
        }
        assert!(
            op.redirects_branch_target(),
            "branch target {} points to removed internal-only op {:?}",
            idx,
            op.kind,
        );
    }
}

fn remap_target(
    old_target: OpIndex,
    ops: &[ResolvedNativeInst],
    index_map: &[Option<usize>],
) -> Option<OpIndex> {
    let mut target = old_target.as_usize();
    while target < index_map.len() {
        if let Some(new_target) = index_map[target] {
            return Some(OpIndex::from(new_target));
        }
        assert!(
            ops[target].redirects_branch_target(),
            "branch target {} points to removed internal-only op {:?}",
            old_target.as_usize(),
            ops[target].kind,
        );
        target += 1;
    }
    None
}

fn compact_and_patch(
    ops: NativeResolvedVec,
    keep: &[bool],
    index_map: &[Option<usize>],
) -> NativeResolvedVec {
    let original_ops = ops.clone();
    let mut compacted = Vec::with_capacity(ops.len());
    let mut ops_iter = ops.into_iter().enumerate().peekable();

    while let Some((old_idx, op)) = ops_iter.next() {
        if !keep[old_idx] {
            continue;
        }

        let mut op = op;
        if let Some(alt) = op.alt_target {
            op.alt_target = remap_target(alt, &original_ops, index_map);
        }

        if let IrOpKind::BrTable {
            ref mut entries,
            ref mut entry_count,
            ref mut data_slot_count,
            ..
        } = op.kind
        {
            let taken_entries = mem::take(entries);
            let br_table_new_idx = compacted.len();
            let ec = taken_entries.len();
            let dsc = (ec + 1) / 2;

            *entry_count = ec as u32;
            *data_slot_count = dsc as u32;
            compacted.push(op);

            let mut data_slots: Vec<(u64, u64, u64)> = vec![(0, 0, 0); dsc];

            for (entry_idx, entry) in taken_entries.iter().enumerate() {
                if let Some(tgt_old) = entry.target_idx {
                    if let Some(tgt_new) = remap_target(tgt_old, &original_ops, index_map) {
                        let rel = (tgt_new.as_usize() as i32) - (br_table_new_idx as i32);
                        let stack_drop = entry.stack_offset as u32;
                        let arity = entry.arity as u32;
                        let slot_idx = entry_idx / 2;
                        let entry_in_slot = entry_idx % 2;

                        if entry_in_slot == 0 {
                            data_slots[slot_idx].0 = rel as i32 as u64;
                            data_slots[slot_idx].1 = ((stack_drop << 16) | arity) as u64;
                        } else {
                            data_slots[slot_idx].2 = ((rel as u64) << 32)
                                | ((stack_drop as u64) << 16)
                                | (arity as u64);
                        }
                    }
                }
            }

            for (imm0, imm1, imm2) in data_slots {
                if let Some((_, mut data_op)) = ops_iter.next() {
                    data_op.kind = IrOpKind::Data { imm0, imm1, imm2 };
                    compacted.push(data_op);
                }
            }
            continue;
        }

        compacted.push(op);
    }

    compacted
}

fn build_instructions(
    ops: NativeResolvedVec,
    operand_base: usize,
    buf: &mut CodeBuffer,
    module_name: &str,
    func_idx: u32,
) -> (Box<[NativeInst]>, Box<[HelperMetadata]>) {
    if ops.is_empty() {
        return (Box::new([]), Box::new([]));
    }

    let fix_slot = |slot: SlotRef| -> u16 { slot.resolve(operand_base) };

    fn pack_tos_slots(pre_height: u16, operand_base: usize) -> u64 {
        let h = pre_height as usize;
        if h == 0 {
            return EMPTY_TOS_SLOTS;
        }

        let variant = ((h - 1) % 4) + 1;
        let mut slots = [INVALID_TOS_SLOT; 4];
        let live = h.min(4);
        for pos in 1..=live {
            let reg_idx = (variant + 4 - pos) % 4;
            let slot_idx = operand_base + h - pos;
            assert!(slot_idx <= u16::MAX as usize, "native TOS slot index overflow");
            slots[reg_idx] = slot_idx as u16;
        }

        (slots[0] as u64)
            | ((slots[1] as u64) << 16)
            | ((slots[2] as u64) << 32)
            | ((slots[3] as u64) << 48)
    }

    let instructions: Vec<NativeInst> = ops
        .iter()
        .map(|op| {
            let tos_slots = pack_tos_slots(op.pre_height, operand_base);
            if op.cold_helper.is_some() {
                NativeInst::new_with_tos_slots(op.entry, 0, 0, 0, tos_slots)
            } else {
                let (imm0, imm1, imm2) = encode_operands(&op.kind, 0, &fix_slot);
                NativeInst::new_with_tos_slots(op.entry, imm0, imm1, imm2, tos_slots)
            }
        })
        .collect();

    let mut code_box: Box<[NativeInst]> = instructions.into_boxed_slice();
    let base = code_box.as_mut_ptr();

    for (i, op) in ops.iter().enumerate() {
        if let Some(alt_idx) = op.alt_target {
            if op.has_target && op.cold_helper.is_none() {
                unsafe {
                    let target_ptr = base.add(alt_idx.as_usize()) as u64;
                    let (imm0, imm1, imm2) = encode_operands(&op.kind, target_ptr, &fix_slot);
                    (*base.add(i)).imm0 = imm0;
                    (*base.add(i)).imm1 = imm1;
                    (*base.add(i)).imm2 = imm2;
                }
            }
        }
    }

    let helper_sites: Vec<(usize, NativeEntry)> = ops
        .iter()
        .enumerate()
        .filter_map(|(i, op)| op.cold_helper.map(|_| (i, code_box[i].entry)))
        .collect();

    let placeholder = HelperMetadata::new(
        bridge::call_external_helper,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    );
    let mut metadata_box = vec![placeholder; helper_sites.len()].into_boxed_slice();

    let needs_code_patch = !helper_sites.is_empty()
        || ops.iter().any(|op| {
            op.entry_patches.fallthrough_literal.is_some() || op.entry_patches.alt_literal.is_some()
        });

    if needs_code_patch {
        let mut patch_start = buf.len();
        buf.begin_write();

        for (meta_idx, (code_idx, helper_entry)) in helper_sites.iter().copied().enumerate() {
            let meta_ptr = unsafe { metadata_box.as_ptr().add(meta_idx) };
            let (wrapper_off, wrapper_len) = emit::emit_helper_wrapper(buf, helper_entry, meta_ptr);
            let wrapper_entry: NativeEntry = unsafe { buf.fn_ptr(wrapper_off) };
            code_box[code_idx].entry = wrapper_entry;
            if let Some(cold_helper) = ops[code_idx].cold_helper {
                debug_map::record_wrapper(
                    buf.base_ptr(),
                    wrapper_off,
                    wrapper_len,
                    module_name,
                    func_idx,
                    code_idx,
                    cold_helper_name(cold_helper),
                    &ops[code_idx].kind,
                );
            }
        }

        for (meta_idx, (code_idx, _)) in helper_sites.iter().copied().enumerate() {
            let op = &ops[code_idx];
            let cold_helper = op.cold_helper.expect("helper site must have cold helper");
            let next = unsafe { base.add(code_idx + 1) };
            let branch_target = op
                .alt_target
                .map(|alt_idx| unsafe { base.add(alt_idx.as_usize()) });
            metadata_box[meta_idx] = bridge::build_metadata(
                cold_helper,
                &op.kind,
                unsafe { base.add(code_idx) },
                next,
                branch_target,
                op.cold_stack_slot,
                &fix_slot,
            );
        }

        for (i, op) in ops.iter().enumerate() {
            if let Some(literal_off) = op.entry_patches.fallthrough_literal {
                patch_start = patch_start.min(literal_off);
                let target_entry = code_box
                    .get(i + 1)
                    .map(|inst| inst.entry)
                    .unwrap_or_else(term_entry);
                buf.patch_u64(literal_off, target_entry as usize as u64);
            }
            if let Some(literal_off) = op.entry_patches.alt_literal {
                patch_start = patch_start.min(literal_off);
                let target_entry = op
                    .alt_target
                    .and_then(|alt_idx| code_box.get(alt_idx.as_usize()).map(|inst| inst.entry))
                    .unwrap_or_else(term_entry);
                buf.patch_u64(literal_off, target_entry as usize as u64);
            }
        }

        let written_len = buf.len() - patch_start;
        buf.finish_write(patch_start, written_len);
    }

    (code_box, metadata_box)
}

fn cold_helper_name(helper: bridge::ColdHelperKind) -> &'static str {
    match helper {
        bridge::ColdHelperKind::Br => "br",
        bridge::ColdHelperKind::BrIf => "br_if",
        bridge::ColdHelperKind::If => "if",
        bridge::ColdHelperKind::Else => "else",
        bridge::ColdHelperKind::CallExternal => "call_external",
        bridge::ColdHelperKind::CallInternal => "call_internal",
        bridge::ColdHelperKind::CallIndirect => "call_indirect",
        bridge::ColdHelperKind::BrTable => "br_table",
        bridge::ColdHelperKind::GlobalGet => "global_get",
        bridge::ColdHelperKind::GlobalSet => "global_set",
        bridge::ColdHelperKind::MemoryGrow => "memory_grow",
        bridge::ColdHelperKind::MemoryCopy => "memory_copy",
        bridge::ColdHelperKind::MemoryFill => "memory_fill",
        bridge::ColdHelperKind::MemoryInit => "memory_init",
        bridge::ColdHelperKind::DataDrop => "data_drop",
        bridge::ColdHelperKind::I32Popcnt => "i32_popcnt",
        bridge::ColdHelperKind::I64Popcnt => "i64_popcnt",
        bridge::ColdHelperKind::F32Copysign => "f32_copysign",
        bridge::ColdHelperKind::F64Copysign => "f64_copysign",
        bridge::ColdHelperKind::I32TruncF32S => "i32_trunc_f32_s",
        bridge::ColdHelperKind::I32TruncF32U => "i32_trunc_f32_u",
        bridge::ColdHelperKind::I32TruncF64S => "i32_trunc_f64_s",
        bridge::ColdHelperKind::I32TruncF64U => "i32_trunc_f64_u",
        bridge::ColdHelperKind::I64TruncF32S => "i64_trunc_f32_s",
        bridge::ColdHelperKind::I64TruncF32U => "i64_trunc_f32_u",
        bridge::ColdHelperKind::I64TruncF64S => "i64_trunc_f64_s",
        bridge::ColdHelperKind::I64TruncF64U => "i64_trunc_f64_u",
        bridge::ColdHelperKind::TableGet => "table_get",
        bridge::ColdHelperKind::TableSet => "table_set",
        bridge::ColdHelperKind::TableSize => "table_size",
        bridge::ColdHelperKind::TableGrow => "table_grow",
        bridge::ColdHelperKind::TableFill => "table_fill",
        bridge::ColdHelperKind::TableCopy => "table_copy",
        bridge::ColdHelperKind::TableInit => "table_init",
        bridge::ColdHelperKind::ElemDrop => "elem_drop",
        bridge::ColdHelperKind::RefNull => "ref_null",
        bridge::ColdHelperKind::RefIsNull => "ref_is_null",
        bridge::ColdHelperKind::RefFunc => "ref_func",
    }
}
