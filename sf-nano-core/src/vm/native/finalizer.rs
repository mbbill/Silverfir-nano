//! Finalizer: Vec<ResolvedNativeInst> -> Box<[NativeInst]>.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use crate::vm::compile::StackTracker;
use crate::vm::compaction::CompactionDisposition;
use crate::vm::lowered::{IrOpKind, OpIndex, SlotRef};
use crate::vm::operand_encoding::encode_operands;

use super::instruction::{NativeEntry, NativeInst};
use super::resolved::{NativeResolvedVec, ResolvedNativeInst};
use super::runtime::term_entry;

pub fn finalize(
    mut ops: NativeResolvedVec,
    stack: &mut StackTracker,
) -> Box<[NativeInst]> {
    append_terminals(&mut ops);
    route_terminals(&mut ops);
    default_alt_to_term(&mut ops);
    let ops = expand_br_tables(ops);
    let keep: Vec<bool> = ops.iter().map(|op| op.compaction.is_kept()).collect();
    validate_removed_targets(&ops, &keep);
    let index_map = build_index_map(&keep);
    let compacted = compact_and_patch(ops, &keep, &index_map);
    build_instructions(compacted, stack.operand_base())
}

fn append_terminals(ops: &mut NativeResolvedVec) {
    let term = ResolvedNativeInst {
        entry: term_entry(),
        kind: IrOpKind::Term,
        alt_target: None,
        has_target: false,
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
                alt_target: None,
                has_target: false,
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

fn build_instructions(ops: NativeResolvedVec, operand_base: usize) -> Box<[NativeInst]> {
    if ops.is_empty() {
        return Box::new([]);
    }

    let fix_slot = |slot: SlotRef| -> u16 { slot.resolve(operand_base) };

    let instructions: Vec<NativeInst> = ops
        .iter()
        .map(|op| {
            let (imm0, imm1, imm2) = encode_operands(&op.kind, 0, &fix_slot);
            NativeInst::new(op.entry, imm0, imm1, imm2)
        })
        .collect();

    let mut code_box: Box<[NativeInst]> = instructions.into_boxed_slice();
    let base = code_box.as_mut_ptr();
    for (i, op) in ops.iter().enumerate() {
        if let Some(alt_idx) = op.alt_target {
            if op.has_target {
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
    code_box
}
