//! Finalizer: Vec<ResolvedInst> → Box<[Instruction]>.
//!
//! Compacts removed markers, patches branch targets, encodes immediates.

use alloc::boxed::Box;
use alloc::vec::Vec;
use alloc::vec;
use core::mem;

use crate::vm::compaction::CompactionDisposition;
use crate::vm::{compile::StackTracker, lowered::{IrOpKind, OpIndex, SlotRef}, operand_encoding::encode_operands};
use super::ir_resolve::resolve_handler;
use super::super::handlers::OpHandler as Handler;
use super::super::resolved::ResolvedInst;
use crate::vm::interp::fast::instruction::Instruction;

/// Finalize resolved instructions into final code.
pub fn finalize(
    mut ops: Vec<ResolvedInst>,
    stack: &mut StackTracker,
) -> Box<[Instruction]> {
    // Append terminal instructions
    append_terminals(&mut ops);

    // Route RETURN and UNREACHABLE to terminal
    route_terminals(&mut ops);

    // Default alt to terminal (for trap paths)
    default_alt_to_term(&mut ops);

    // Expand br_tables by inserting inline data pseudo-instructions
    let ops = expand_br_tables(ops);

    // Compute which instructions remain in final code.
    let keep: Vec<bool> = ops.iter().map(|op| op.compaction.is_kept()).collect();

    // Validate that any removed target is an explicitly redirectable marker.
    validate_removed_targets(&ops, &keep);

    // Build old->new index mapping
    let index_map = build_index_map(&keep);

    // Compact ops and patch indices (including br_table inline data)
    let compacted = compact_and_patch(ops, &keep, &index_map);

    // Build final Instruction array
    let operand_base = stack.operand_base();
    build_instructions(compacted, operand_base)
}

/// Append terminal (Term) instruction and arena sentinel.
fn append_terminals(ops: &mut Vec<ResolvedInst>) {
    let term = ResolvedInst {
        handler: resolve_handler(&IrOpKind::Term, 0),
        kind: IrOpKind::Term,
        alt_target: None,
        has_target: false,
        compaction: CompactionDisposition::Keep,
    };

    ops.push(term.clone());

    // Arena sentinel: ensures pc_next of any instruction is a valid read.
    ops.push(term);
}

/// Route RETURN and UNREACHABLE to the single terminal instruction.
fn route_terminals(ops: &mut Vec<ResolvedInst>) {
    let term_idx = ops.len() - 1;

    for op in ops.iter_mut() {
        match &op.kind {
            IrOpKind::ReturnVoid { .. } | IrOpKind::ReturnOne { .. } |
            IrOpKind::Return { .. } | IrOpKind::Unreachable => {
                op.alt_target = Some(OpIndex::from(term_idx));
            }
            _ => {}
        }
    }
}

/// Default alt to terminal for instructions without alt.
fn default_alt_to_term(ops: &mut Vec<ResolvedInst>) {
    let term_idx = ops.len() - 1;
    for op in ops.iter_mut() {
        if op.alt_target.is_none() {
            op.alt_target = Some(OpIndex::from(term_idx));
        }
    }
}

/// Expand br_tables by inserting inline data pseudo-instructions.
fn expand_br_tables(ops: Vec<ResolvedInst>) -> Vec<ResolvedInst> {
    // First pass: calculate expansion
    let mut expansion_at: Vec<usize> = vec![0; ops.len()];
    let mut total_expansion = 0;

    for (i, op) in ops.iter().enumerate() {
        expansion_at[i] = total_expansion;
        if let IrOpKind::BrTable { entries, .. } = &op.kind {
            let data_slot_count = (entries.len() + 1) / 2;
            total_expansion += data_slot_count;
        }
    }

    if total_expansion == 0 {
        return ops;
    }

    // Build old->new index mapping
    let old_to_new: Vec<usize> = expansion_at
        .iter()
        .enumerate()
        .map(|(i, &exp)| i + exp)
        .collect();

    // Update all index references
    let mut ops = ops;
    for op in ops.iter_mut() {
        if let Some(ref mut alt) = op.alt_target {
            if alt.as_usize() < old_to_new.len() {
                *alt = OpIndex::from(old_to_new[alt.as_usize()]);
            }
        }
        // Patch br_table entry targets
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

    // Build expanded Vec with data slots
    let mut result = Vec::with_capacity(ops.len() + total_expansion);

    for op in ops {
        let data_slot_count = if let IrOpKind::BrTable { ref entries, .. } = op.kind {
            (entries.len() + 1) / 2
        } else {
            0
        };

        result.push(op);

        // Insert data pseudo-instructions after br_table
        for _ in 0..data_slot_count {
            result.push(ResolvedInst {
                handler: resolve_handler(&IrOpKind::Data { imm0: 0, imm1: 0, imm2: 0 }, 0),
                kind: IrOpKind::Data { imm0: 0, imm1: 0, imm2: 0 },
                alt_target: None,
                has_target: false,
                compaction: CompactionDisposition::Keep,
            });
        }
    }

    result
}

/// Build old->new index mapping.
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

/// Mark every resolved index that can be reached by a non-fallthrough branch.
fn incoming_targets(ops: &[ResolvedInst]) -> Vec<bool> {
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

fn validate_removed_targets(ops: &[ResolvedInst], keep: &[bool]) {
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
    ops: &[ResolvedInst],
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

/// Compact ops and patch all indices.
fn compact_and_patch(
    ops: Vec<ResolvedInst>,
    keep: &[bool],
    index_map: &[Option<usize>],
) -> Vec<ResolvedInst> {
    let original_ops = ops.clone();
    let mut compacted = Vec::with_capacity(ops.len());
    let mut ops_iter = ops.into_iter().enumerate().peekable();

    while let Some((old_idx, op)) = ops_iter.next() {
        if !keep[old_idx] {
            continue;
        }

        let mut op = op;

        // Patch alt_target
        if let Some(alt) = op.alt_target {
            op.alt_target = remap_target(alt, &original_ops, index_map);
        }

        // Handle br_table: fill inline data slots
        if let IrOpKind::BrTable { ref mut entries, ref mut entry_count, ref mut data_slot_count, .. } = op.kind {
            let taken_entries = mem::take(entries);
            let br_table_new_idx = compacted.len();
            let ec = taken_entries.len();
            let dsc = (ec + 1) / 2;

            // Set counts for encoding
            *entry_count = ec as u32;
            *data_slot_count = dsc as u32;

            compacted.push(op);

            // Build packed data for each slot
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
                            let packed = ((rel as u64) << 32)
                                | ((stack_drop as u64) << 16)
                                | (arity as u64);
                            data_slots[slot_idx].2 = packed;
                        }
                    }
                }
            }

            // Fill data pseudo-instructions
            for (imm0, imm1, imm2) in data_slots {
                if let Some((data_old_idx, mut data_op)) = ops_iter.next() {
                    debug_assert!(keep[data_old_idx]);
                    debug_assert!(matches!(data_op.kind, IrOpKind::Data { .. }));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved(kind: IrOpKind, alt_target: Option<OpIndex>, has_target: bool, compaction: CompactionDisposition) -> ResolvedInst {
        ResolvedInst {
            handler: resolve_handler(&kind, 0),
            kind,
            alt_target,
            has_target,
            compaction,
        }
    }

    #[test]
    fn test_removed_redirect_target_maps_to_next_kept() {
        let ops = vec![
            resolved(IrOpKind::If, Some(OpIndex::from(1)), true, CompactionDisposition::Keep),
            resolved(IrOpKind::Block, None, false, CompactionDisposition::RedirectBranchTarget),
            resolved(IrOpKind::I32Const { value: 7 }, None, false, CompactionDisposition::Keep),
        ];
        let keep: Vec<bool> = ops.iter().map(|op| op.compaction.is_kept()).collect();
        let index_map = build_index_map(&keep);

        validate_removed_targets(&ops, &keep);
        let compacted = compact_and_patch(ops, &keep, &index_map);

        assert_eq!(compacted[0].alt_target, Some(OpIndex::from(1)));
    }

    #[test]
    #[should_panic(expected = "removed internal-only op")]
    fn test_removed_internal_only_target_panics() {
        let ops = vec![
            resolved(IrOpKind::If, Some(OpIndex::from(1)), true, CompactionDisposition::Keep),
            resolved(IrOpKind::Nop, None, false, CompactionDisposition::InternalOnly),
            resolved(IrOpKind::I32Const { value: 7 }, None, false, CompactionDisposition::Keep),
        ];
        let keep: Vec<bool> = ops.iter().map(|op| op.compaction.is_kept()).collect();

        validate_removed_targets(&ops, &keep);
    }
}

/// Build final Instruction array.
///
/// Handler is pre-resolved on ResolvedInst. Encoding uses encode_operands.
fn build_instructions(
    ops: Vec<ResolvedInst>,
    operand_base: usize,
) -> Box<[Instruction]> {
    if ops.is_empty() {
        return Box::new([]);
    }

    let fix_slot = |slot: SlotRef| -> u16 {
        slot.resolve(operand_base)
    };

    // First pass: encode all instructions (target_ptr = 0 for now)
    let instructions: Vec<Instruction> = ops
        .iter()
        .map(|op| {
            let (imm0, imm1, imm2) = encode_operands(&op.kind, 0, &fix_slot);
            Instruction::new(op.handler, imm0, imm1, imm2)
        })
        .collect();

    // Convert to Box (stable heap allocation)
    let mut code_box: Box<[Instruction]> = instructions.into_boxed_slice();

    // Second pass: patch branch target pointers
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
