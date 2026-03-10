//! Finalizer: Vec<ResolvedNativeInst> -> native entry table + metadata.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::mem;

use crate::vm::wasm::StackTracker;
use crate::vm::abi::compaction::CompactionDisposition;
use crate::vm::entities::FunctionInst;
use crate::vm::lir::{IrOpKind, OpIndex, SlotRef};

use super::CodeBuffer;
use super::code::DirectCallEntryPatch;
use super::map;
#[cfg(feature = "native-dump")]
use super::dump;
use super::bridge::{self, HelperMetadata};
use super::entry::{EMPTY_RESUME_SLOTS, INVALID_RESUME_SLOT, NativeEntry};
use super::resolved::{NativeResolvedVec, ResolvedNativeInst};
use super::runtime::term_entry;
use super::arm64::{EntryPatchSites, current_variant_from_window, emit, tos_reg};
use super::arm64::reg::Reg;

#[derive(Clone, Copy)]
struct FinalEntryInfo {
    entry: NativeEntry,
    resume_slots: u64,
}

fn absolute_frame_slot(slot: Option<SlotRef>, context: &str) -> u16 {
    match slot {
        Some(SlotRef::Absolute(slot)) => slot,
        Some(SlotRef::OperandRelative(offset)) => {
            panic!("{context} expects an absolute frame slot, got operand-relative offset {offset}")
        }
        None => panic!("{context} expects a frame slot"),
    }
}

pub fn finalize(
    mut ops: NativeResolvedVec,
    stack: &mut StackTracker,
    buf: &mut CodeBuffer,
    module_name: &str,
    func_idx: u32,
) -> (Box<[NativeEntry]>, Box<[HelperMetadata]>, Box<[DirectCallEntryPatch]>) {
    append_terminals(&mut ops);
    route_terminals(&mut ops);
    default_alt_to_term(&mut ops);
    let ops = expand_br_tables(ops);
    let keep: Vec<bool> = ops.iter().map(|op| op.compaction.is_kept()).collect();
    validate_removed_targets(&ops, &keep);
    let index_map = build_index_map(&keep);
    let (compacted, original_indices) = compact_and_patch(ops, &keep, &index_map);
    build_instructions(
        compacted,
        original_indices,
        stack.operand_base(),
        buf,
        module_name,
        func_idx,
    )
}

fn append_terminals(ops: &mut NativeResolvedVec) {
    let term = ResolvedNativeInst {
        entry: term_entry(),
        kind: IrOpKind::Term,
        window: crate::vm::lir::window_from_height(0),
        entry_input_count: 0,
        frame_slot: None,
        #[cfg(feature = "native-dump")]
        original_ir_idx: usize::MAX,
        alt_target: None,
        has_target: false,
        cold_helper: None,
        cold_frame_slot: None,
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
        #[cfg(feature = "native-dump")]
        let original_ir_idx = op.original_ir_idx;
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
                window: crate::vm::lir::window_from_height(0),
                entry_input_count: 0,
                frame_slot: None,
                #[cfg(feature = "native-dump")]
                original_ir_idx,
                alt_target: None,
                has_target: false,
                cold_helper: None,
                cold_frame_slot: None,
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

#[inline]
fn debug_ir_idx(op: &ResolvedNativeInst, fallback: usize) -> usize {
    #[cfg(feature = "native-dump")]
    {
        op.original_ir_idx
    }
    #[cfg(not(feature = "native-dump"))]
    {
        fallback
    }
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
) -> (NativeResolvedVec, Vec<usize>) {
    let original_ops = ops.clone();
    let mut compacted = Vec::with_capacity(ops.len());
    let mut original_indices = Vec::with_capacity(ops.len());
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
            original_indices.push(old_idx);

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
                if let Some((data_old_idx, mut data_op)) = ops_iter.next() {
                    data_op.kind = IrOpKind::Data { imm0, imm1, imm2 };
                    compacted.push(data_op);
                    original_indices.push(data_old_idx);
                }
            }
            continue;
        }

        compacted.push(op);
        original_indices.push(old_idx);
    }

    (compacted, original_indices)
}

fn build_instructions(
    ops: NativeResolvedVec,
    original_indices: Vec<usize>,
    operand_base: usize,
    buf: &mut CodeBuffer,
    module_name: &str,
    func_idx: u32,
) -> (Box<[NativeEntry]>, Box<[HelperMetadata]>, Box<[DirectCallEntryPatch]>) {
    if ops.is_empty() {
        return (Box::new([]), Box::new([]), Box::new([]));
    }

    let fix_slot = |slot: SlotRef| -> u16 { slot.resolve(operand_base) };

    fn pack_resume_slots(window: u8, operand_base: usize, input_count: u8) -> u64 {
        if input_count == 0 {
            return EMPTY_RESUME_SLOTS;
        }
        let mut slots = [INVALID_RESUME_SLOT; 4];
        for pos in 1..=input_count as usize {
            let reg_idx = (window as usize + 4 - pos) % 4;
            let slot_idx = operand_base + 4 - pos;
            assert!(slot_idx <= u16::MAX as usize, "native resume slot index overflow");
            slots[reg_idx] = slot_idx as u16;
        }

        (slots[0] as u64)
            | ((slots[1] as u64) << 16)
            | ((slots[2] as u64) << 32)
            | ((slots[3] as u64) << 48)
    }

    let mut entry_info: Vec<FinalEntryInfo> = ops
        .iter()
        .map(|op| FinalEntryInfo {
            entry: op.entry,
            resume_slots: pack_resume_slots(op.window, operand_base, op.entry_input_count),
        })
        .collect();

    let mut entries: Vec<NativeEntry> = entry_info.iter().map(|entry| entry.entry).collect();

    let helper_sites: Vec<(usize, NativeEntry)> = ops
        .iter()
        .zip(entries.iter().copied())
        .enumerate()
        .filter_map(|(i, (op, entry))| op.cold_helper.map(|_| (i, entry)))
        .collect();
    let br_table_sites: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter_map(|(i, op)| matches!(op.kind, IrOpKind::BrTable { .. }).then_some(i))
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
    let mut br_table_entry_literals: Vec<(usize, usize)> = Vec::new();
    let mut direct_call_entry_patches: Vec<DirectCallEntryPatch> = Vec::new();
    let mut direct_call_next_entry_literals: Vec<(usize, usize, usize)> = Vec::new();

    let needs_code_patch = !helper_sites.is_empty()
        || !br_table_sites.is_empty()
        || ops.iter().any(|op| {
            op.entry_patches.fallthrough_literal.is_some() || op.entry_patches.alt_literal.is_some()
        });

    if needs_code_patch {
        let mut patch_start = buf.len();
        buf.begin_write();

        for &code_idx in &br_table_sites {
            let op = &ops[code_idx];
            let (
                IrOpKind::BrTable {
                    entry_count,
                    ..
                }
            ) = &op.kind else {
                unreachable!("br_table site must be br_table");
            };

            let mut records = Vec::with_capacity(*entry_count as usize);
            let mut target_indices = Vec::with_capacity(*entry_count as usize);
            for entry_idx in 0..(*entry_count as usize) {
                let data_idx = code_idx + 1 + (entry_idx / 2);
                let data_inst = &ops[data_idx];
                let (imm0, imm1, imm2) = match data_inst.kind {
                    IrOpKind::Data { imm0, imm1, imm2 } => (imm0, imm1, imm2),
                    _ => unreachable!("br_table data slot must be an IrOpKind::Data"),
                };
                let (rel, packed_branch) = if entry_idx % 2 == 0 {
                    (imm0 as i32, imm1)
                } else {
                    (((imm2 >> 32) as u32) as i32, imm2 & 0xffff_ffff)
                };
                let target_idx = ((code_idx as isize) + (rel as isize)) as usize;
                target_indices.push(target_idx);
                records.push(emit::DirectBrTableRecord { packed_branch });
            }
            let src_reg = match tos_reg(op.window, 1) {
                reg @ (Reg::T0 | Reg::T1 | Reg::T2 | Reg::T3) => reg,
                _ => unreachable!("br_table source must be a TOS register"),
            };
            let top_slot = absolute_frame_slot(op.frame_slot, "br_table")
                .checked_sub(1)
                .expect("br_table expects a carried-value slot below the popped index");
            let emit_site = emit::emit_direct_br_table_entry(
                buf,
                src_reg,
                &records,
                top_slot,
            );
            for (target_idx, literal_off) in target_indices
                .into_iter()
                .zip(emit_site.target_entry_literals.into_iter())
            {
                br_table_entry_literals.push((target_idx, literal_off));
            }
            let wrapper_entry: NativeEntry = unsafe { buf.fn_ptr(emit_site.start) };
            entries[code_idx] = wrapper_entry;
            entry_info[code_idx].entry = wrapper_entry;
            map::record_wrapper(
                buf.base_ptr(),
                emit_site.start,
                emit_site.len,
                module_name,
                func_idx,
                debug_ir_idx(op, original_indices[code_idx]),
                "br_table",
                &op.kind,
            );
            #[cfg(feature = "native-dump")]
            dump::record_wrapper(
                emit_site.start,
                emit_site.len,
                module_name,
                func_idx,
                debug_ir_idx(op, original_indices[code_idx]),
                "br_table",
                &op.kind,
            );
        }

        for (meta_idx, (code_idx, helper_entry)) in helper_sites.iter().copied().enumerate() {
            let op = &ops[code_idx];
            let cold_helper = op.cold_helper.expect("helper site must have cold helper");
            let meta_ptr = unsafe { metadata_box.as_ptr().add(meta_idx) };
            let (wrapper_off, wrapper_len) = match (&op.kind, cold_helper) {
                (
                    IrOpKind::CallInternal {
                        callee,
                        delta,
                        ..
                    },
                    bridge::ColdHelperKind::CallInternal,
                ) => {
                    let callee_ptr = *callee as *const FunctionInst;
                    let callee = unsafe { &*callee_ptr };
                    let params_count = callee.func_type().params().len() as u16;
                    let callee_spec = callee.spec();
                    let locals_count = callee_spec
                        .map(|spec| spec.locals().len() as u16)
                        .unwrap_or(0);
                    let emit_site = emit::emit_direct_call_internal_entry(
                        buf,
                        params_count,
                        locals_count,
                        fix_slot(*delta),
                    );
                    if let Some(spec) = callee_spec {
                        if spec.has_native_code() {
                            buf.patch_u64(
                                emit_site.callee_entry_literal,
                                spec.native_cache().entry() as usize as u64,
                            );
                        } else {
                            direct_call_entry_patches.push(DirectCallEntryPatch {
                                callee: callee_ptr,
                                literal_off: emit_site.callee_entry_literal,
                            });
                        }
                    }
                    direct_call_next_entry_literals.push((
                        code_idx + 1,
                        emit_site.next_entry_literal,
                        emit_site.next_resume_slots_literal,
                    ));
                    (emit_site.start, emit_site.len)
                }
                _ => emit::emit_helper_wrapper(buf, helper_entry, meta_ptr),
            };
            let wrapper_entry: NativeEntry = unsafe { buf.fn_ptr(wrapper_off) };
            entries[code_idx] = wrapper_entry;
            entry_info[code_idx].entry = wrapper_entry;
            map::record_wrapper(
                buf.base_ptr(),
                wrapper_off,
                wrapper_len,
                module_name,
                func_idx,
                debug_ir_idx(op, original_indices[code_idx]),
                cold_helper_name(cold_helper),
                &op.kind,
            );
            #[cfg(feature = "native-dump")]
            dump::record_wrapper(
                wrapper_off,
                wrapper_len,
                module_name,
                func_idx,
                debug_ir_idx(op, original_indices[code_idx]),
                cold_helper_name(cold_helper),
                &op.kind,
            );
        }

        for (meta_idx, (code_idx, _)) in helper_sites.iter().copied().enumerate() {
            let op = &ops[code_idx];
            let cold_helper = op.cold_helper.expect("helper site must have cold helper");
            let branch_target = op
                .alt_target
                .and_then(|alt_idx| entry_info.get(alt_idx.as_usize()).copied())
                .map(|info| (info.entry, info.resume_slots));
            let next_info = entry_info
                .get(code_idx + 1)
                .copied()
                .unwrap_or(FinalEntryInfo {
                    entry: term_entry(),
                    resume_slots: EMPTY_RESUME_SLOTS,
                });
            metadata_box[meta_idx] = bridge::build_metadata(
                cold_helper,
                &op.kind,
                op.window,
                next_info.entry,
                next_info.resume_slots,
                branch_target,
                op.cold_frame_slot,
                &fix_slot,
            );
            #[cfg(feature = "native-dump")]
            dump::write_helper_metadata(
                module_name,
                func_idx,
                debug_ir_idx(op, original_indices[code_idx]),
                cold_helper_name(cold_helper),
                &metadata_box[meta_idx],
            );
        }

        let mut resume_entry_cache: Vec<(usize, NativeEntry)> = Vec::new();

        for (i, op) in ops.iter().enumerate() {
            if let Some(literal_off) = op.entry_patches.fallthrough_literal {
                patch_start = patch_start.min(literal_off);
                let target_entry = entry_info
                    .get(i + 1)
                    .map(|entry| entry.entry)
                    .unwrap_or_else(term_entry);
                let patched_entry = if needs_fallthrough_resume(&op.kind) {
                    ensure_resume_entry(
                        &mut resume_entry_cache,
                        &entry_info,
                        &ops,
                        buf,
                        module_name,
                        func_idx,
                        i + 1,
                        target_entry,
                        debug_ir_idx(&ops[i + 1], original_indices[i + 1]),
                    )
                } else {
                    target_entry
                };
                buf.patch_u64(literal_off, patched_entry as usize as u64);
            }
            if let Some(literal_off) = op.entry_patches.alt_literal {
                patch_start = patch_start.min(literal_off);
                let target_entry = op
                    .alt_target
                    .and_then(|alt_idx| entry_info.get(alt_idx.as_usize()).map(|entry| entry.entry))
                    .unwrap_or_else(term_entry);
                let patched_entry = if needs_branch_resume(&op.kind) {
                    let target_idx = op
                        .alt_target
                        .map(|idx| idx.as_usize())
                        .unwrap_or(ops.len().saturating_sub(1));
                    ensure_resume_entry(
                        &mut resume_entry_cache,
                        &entry_info,
                        &ops,
                        buf,
                        module_name,
                        func_idx,
                        target_idx,
                        target_entry,
                        debug_ir_idx(&ops[target_idx], original_indices[target_idx]),
                    )
                } else {
                    target_entry
                };
                buf.patch_u64(literal_off, patched_entry as usize as u64);
            }
        }

        for (target_idx, literal_off) in br_table_entry_literals {
            patch_start = patch_start.min(literal_off);
            let target_entry = entry_info
                .get(target_idx)
                .map(|entry| entry.entry)
                .unwrap_or_else(term_entry);
            let patched_entry = ensure_resume_entry(
                &mut resume_entry_cache,
                &entry_info,
                &ops,
                buf,
                module_name,
                func_idx,
                target_idx,
                target_entry,
                debug_ir_idx(&ops[target_idx], original_indices[target_idx]),
            );
            buf.patch_u64(literal_off, patched_entry as usize as u64);
        }

        for (target_idx, entry_literal_off, resume_slots_literal_off) in direct_call_next_entry_literals {
            patch_start = patch_start.min(entry_literal_off.min(resume_slots_literal_off));
            let target_entry = entry_info
                .get(target_idx)
                .map(|entry| entry.entry)
                .unwrap_or_else(term_entry);
            let target_resume_slots = entry_info
                .get(target_idx)
                .map(|entry| entry.resume_slots)
                .unwrap_or(EMPTY_RESUME_SLOTS);
            buf.patch_u64(entry_literal_off, target_entry as usize as u64);
            buf.patch_u64(resume_slots_literal_off, target_resume_slots);
        }

        let written_len = buf.len() - patch_start;
        buf.finish_write(patch_start, written_len);
        #[cfg(feature = "native-dump")]
        dump::flush_function(buf.base_ptr(), module_name, func_idx);
    }

    (
        entries.into_boxed_slice(),
        metadata_box,
        direct_call_entry_patches.into_boxed_slice(),
    )
}

fn needs_branch_resume(kind: &IrOpKind) -> bool {
    matches!(
        kind,
        IrOpKind::BrIf {
            stack_drop,
            arity,
            ..
        } if *stack_drop > 0 && *arity > 0
    ) || matches!(
        kind,
        IrOpKind::Br {
            stack_drop,
            arity,
            ..
        } if *stack_drop > 0 && *arity > 0
    )
}

fn needs_fallthrough_resume(_kind: &IrOpKind) -> bool {
    false
}

fn ensure_resume_entry(
    cache: &mut Vec<(usize, NativeEntry)>,
    entry_info: &[FinalEntryInfo],
    ops: &[ResolvedNativeInst],
    buf: &mut CodeBuffer,
    module_name: &str,
    func_idx: u32,
    target_idx: usize,
    fallback_entry: NativeEntry,
    original_ir_idx: usize,
) -> NativeEntry {
    let target_resume = entry_info
        .get(target_idx)
        .map(|entry| entry.resume_slots)
        .unwrap_or(EMPTY_RESUME_SLOTS);
    if target_resume == EMPTY_RESUME_SLOTS {
        return fallback_entry;
    }
    if let Some((_, entry)) = cache.iter().find(|(idx, _)| *idx == target_idx) {
        return *entry;
    }

    let (start, len) = emit::emit_resume_entry(buf, fallback_entry, target_resume);
    let entry: NativeEntry = unsafe { buf.fn_ptr(start) };
    cache.push((target_idx, entry));
    map::record_wrapper(
        buf.base_ptr(),
        start,
        len,
        module_name,
        func_idx,
        original_ir_idx,
        "resume",
        &ops[target_idx].kind,
    );
    #[cfg(feature = "native-dump")]
    dump::record_wrapper(
        start,
        len,
        module_name,
        func_idx,
        original_ir_idx,
        "resume",
        &ops[target_idx].kind,
    );
    entry
}

fn cold_helper_name(helper: bridge::ColdHelperKind) -> &'static str {
    match helper {
        bridge::ColdHelperKind::CallExternal => "call_external",
        bridge::ColdHelperKind::CallInternal => "call_internal",
        bridge::ColdHelperKind::CallIndirect => "call_indirect",
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
