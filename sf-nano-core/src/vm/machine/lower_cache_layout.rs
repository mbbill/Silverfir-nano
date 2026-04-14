use crate::collections;
use tracked_alloc::collections::BTreeMap;

use core::cmp::Reverse;

use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::MachineStorageType,
        middle::{
            frame::FrameSlot,
            ssa_ir::{
                ir::{
                    block_entry_cache_requirement, EntryCacheRequirement, SsaBlock, SsaOp,
                    SsaProgram, SsaTerminator,
                },
                target::SsaTarget,
            },
        },
    },
};

use super::{
    lower_context::{CachedLocal, EntryCacheParam, ValueRegs},
    lower_module::{preferred_fp_dynamic_reg, preferred_gp_dynamic_reg},
    lower_regalloc::MachineRegFile,
};

/// Sentinel for "no lane assigned" in the slot->lane layout matrices.
/// Replaces `None` in what was previously `Option<usize>` (halves the
/// per-element size from 16 to 4 bytes).
const LANE_UNASSIGNED: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayoutBank {
    Gp,
    Fp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SlotLayoutMeta {
    bank: LayoutBank,
    width: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ParamPrefixUsage {
    gp: usize,
    fp: usize,
}

pub(super) fn compute_block_entry_cache_params(
    regfile: &MachineRegFile,
    program: &SsaProgram,
    cached_locals: &[CachedLocal],
    gp_reg_width: u8,
) -> Result<collections::Vec<collections::Vec<EntryCacheParam>>, WasmError> {
    if program.blocks.is_empty() || cached_locals.is_empty() {
        return Ok(collections::vec![collections::Vec::new(); program.blocks.len()]);
    }

    let slot_to_cached_index = cached_locals
        .iter()
        .enumerate()
        .map(|(cached_index, cached)| (cached.slot, cached_index))
        .collect::<BTreeMap<_, _>>();
    let slot_meta = cached_locals
        .iter()
        .map(|cached| SlotLayoutMeta {
            bank: if cached.ty.is_fp() {
                LayoutBank::Fp
            } else {
                LayoutBank::Gp
            },
            width: if gp_reg_width == 4 && matches!(cached.ty, MachineStorageType::GpI64) {
                2
            } else {
                1
            },
        })
        .collect::<collections::Vec<_>>();
    let param_usage = compute_param_prefix_usage(program, gp_reg_width);
    let predecessors = compute_predecessors(program);
    let rpo = compute_reverse_postorder(program);
    let idom =
        compute_immediate_dominators(program.blocks.len(), program.entry, &predecessors, &rpo);
    let idom_children = build_idom_children(program.blocks.len(), program.entry.as_usize(), &idom);
    let bank_slots = compute_block_bank_slots(program, &slot_to_cached_index, &slot_meta);

    let lanes = cached_locals.len();
    let n_blocks = program.blocks.len();
    let mut gp_layouts = collections::vec![LANE_UNASSIGNED; n_blocks * lanes];
    let mut fp_layouts = collections::vec![LANE_UNASSIGNED; n_blocks * lanes];
    let mut gp_exit_layouts = collections::vec![LANE_UNASSIGNED; n_blocks * lanes];
    let mut fp_exit_layouts = collections::vec![LANE_UNASSIGNED; n_blocks * lanes];
    let mut visited = collections::vec![false; program.blocks.len()];
    if program.entry.as_usize() < program.blocks.len() {
        assign_bank_layouts_from_root(
            program.entry.as_usize(),
            None,
            LayoutBank::Gp,
            regfile.gp_allocatable_count(),
            program,
            &slot_to_cached_index,
            &idom_children,
            &bank_slots,
            &slot_meta,
            &param_usage,
            &mut gp_layouts,
            &mut gp_exit_layouts,
            lanes,
            &mut visited,
        )?;
        assign_bank_layouts_from_root(
            program.entry.as_usize(),
            None,
            LayoutBank::Fp,
            regfile.fp_dynamic_count(),
            program,
            &slot_to_cached_index,
            &idom_children,
            &bank_slots,
            &slot_meta,
            &param_usage,
            &mut fp_layouts,
            &mut fp_exit_layouts,
            lanes,
            &mut collections::vec![false; program.blocks.len()],
        )?;
    }

    for block_index in 0..program.blocks.len() {
        if visited[block_index] {
            continue;
        }
        assign_bank_layouts_from_root(
            block_index,
            None,
            LayoutBank::Gp,
            regfile.gp_allocatable_count(),
            program,
            &slot_to_cached_index,
            &idom_children,
            &bank_slots,
            &slot_meta,
            &param_usage,
            &mut gp_layouts,
            &mut gp_exit_layouts,
            lanes,
            &mut visited,
        )?;
    }
    let mut fp_visited = collections::vec![false; program.blocks.len()];
    if program.entry.as_usize() < program.blocks.len() {
        mark_idom_reachable(program.entry.as_usize(), &idom_children, &mut fp_visited);
    }
    for block_index in 0..program.blocks.len() {
        if fp_visited[block_index] {
            continue;
        }
        assign_bank_layouts_from_root(
            block_index,
            None,
            LayoutBank::Fp,
            regfile.fp_dynamic_count(),
            program,
            &slot_to_cached_index,
            &idom_children,
            &bank_slots,
            &slot_meta,
            &param_usage,
            &mut fp_layouts,
            &mut fp_exit_layouts,
            lanes,
            &mut fp_visited,
        )?;
    }

    let mut layouts = collections::vec![collections::Vec::new(); program.blocks.len()];
    for (block_index, block) in program.blocks.iter().enumerate() {
        let entry_slots = program
            .block_entry_cached_slots
            .get(block_index)
            .map(|s| s.as_slice())
            .unwrap_or(&[]);
        // Upper bound: one entry per entry slot, minus any that miss
        // slot_to_cached_index. Sizing tight avoids ~2x push-growth overallocation
        // on the per-block Vec, which otherwise dominates mir_lower peak memory.
        let mut entries =
            collections::Vec::<(u16, u16, EntryCacheParam)>::with_capacity(entry_slots.len());
        for &slot in entry_slots {
            let Some(&cached_index) = slot_to_cached_index.get(&slot) else {
                continue;
            };
            let cached_index_u16 = u16::try_from(cached_index).map_err(|_| {
                WasmError::internal("cached local index overflowed entry-cache metadata")
            })?;
            let row_base = block_index * lanes;
            let lane_val = match slot_meta[cached_index].bank {
                LayoutBank::Gp => gp_layouts[row_base + cached_index],
                LayoutBank::Fp => fp_layouts[row_base + cached_index],
            };
            if lane_val == LANE_UNASSIGNED {
                return Err(WasmError::internal(
                    "cache lane layout missing for slot in block b",
                ));
            }
            let start = lane_val as usize;
            let regs = regs_for_segment(
                regfile,
                slot_meta[cached_index].bank,
                start,
                slot_meta[cached_index].width,
            )?;
            let requirement =
                block_entry_cache_requirement(entry_slots, block, slot).ok_or_else(|| {
                    WasmError::internal("entry cache slot in block b has no entry requirement")
                })?;
            entries.push((
                regs.lo.0,
                cached_index_u16,
                EntryCacheParam {
                    cached_index: cached_index_u16,
                    regs,
                    needs_value: matches!(requirement, EntryCacheRequirement::Ensure),
                },
            ));
        }
        entries.sort_by_key(|(reg, cached_index, _)| (*reg, *cached_index));
        layouts[block_index] = entries.into_iter().map(|(_, _, entry)| entry).collect();
    }

    Ok(layouts)
}

fn compute_param_prefix_usage(
    program: &SsaProgram,
    gp_reg_width: u8,
) -> collections::Vec<ParamPrefixUsage> {
    program
        .blocks
        .iter()
        .map(|block| {
            let mut usage = ParamPrefixUsage::default();
            for &param in &block.params {
                match super::lower_regalloc::lir_value_storage_type(program, param) {
                    MachineStorageType::Fp32 | MachineStorageType::Fp64 => usage.fp += 1,
                    MachineStorageType::GpI64 if gp_reg_width == 4 => usage.gp += 2,
                    _ => usage.gp += 1,
                }
            }
            usage
        })
        .collect()
}

fn compute_predecessors(program: &SsaProgram) -> collections::Vec<collections::Vec<usize>> {
    let mut predecessors = collections::vec![collections::Vec::new(); program.blocks.len()];
    for (block_index, block) in program.blocks.iter().enumerate() {
        for target in block_successors(&block.terminator) {
            if let Some(preds) = predecessors.get_mut(target.as_usize()) {
                preds.push(block_index);
            }
        }
    }
    predecessors
}

fn compute_reverse_postorder(program: &SsaProgram) -> collections::Vec<usize> {
    fn dfs(
        block_index: usize,
        program: &SsaProgram,
        seen: &mut [bool],
        order: &mut collections::Vec<usize>,
    ) {
        if block_index >= program.blocks.len() || seen[block_index] {
            return;
        }
        seen[block_index] = true;
        for succ in block_successors(&program.blocks[block_index].terminator) {
            dfs(succ.as_usize(), program, seen, order);
        }
        order.push(block_index);
    }

    let mut seen = collections::vec![false; program.blocks.len()];
    let mut order = collections::Vec::with_capacity(program.blocks.len());
    dfs(program.entry.as_usize(), program, &mut seen, &mut order);
    order.reverse();
    order
}

fn compute_immediate_dominators(
    block_count: usize,
    entry: SsaTarget,
    predecessors: &[collections::Vec<usize>],
    rpo: &[usize],
) -> collections::Vec<Option<usize>> {
    let mut rpo_index = collections::vec![usize::MAX; block_count];
    for (index, &block_index) in rpo.iter().enumerate() {
        if block_index < block_count {
            rpo_index[block_index] = index;
        }
    }

    let entry_index = entry.as_usize();
    let mut idom = collections::vec![None; block_count];
    if entry_index >= block_count {
        return idom;
    }
    idom[entry_index] = Some(entry_index);

    let mut changed = true;
    while changed {
        changed = false;
        for &block_index in rpo.iter().skip(1) {
            let mut preds = predecessors[block_index]
                .iter()
                .copied()
                .filter(|pred| idom[*pred].is_some());
            let Some(mut new_idom) = preds.next() else {
                continue;
            };
            for pred in preds {
                new_idom = intersect_idom(&idom, &rpo_index, pred, new_idom);
            }
            if idom[block_index] != Some(new_idom) {
                idom[block_index] = Some(new_idom);
                changed = true;
            }
        }
    }

    idom
}

fn intersect_idom(
    idom: &[Option<usize>],
    rpo_index: &[usize],
    mut lhs: usize,
    mut rhs: usize,
) -> usize {
    while lhs != rhs {
        while rpo_index[lhs] > rpo_index[rhs] {
            lhs = idom[lhs].expect("reachable block must have idom");
        }
        while rpo_index[rhs] > rpo_index[lhs] {
            rhs = idom[rhs].expect("reachable block must have idom");
        }
    }
    lhs
}

fn build_idom_children(
    block_count: usize,
    entry_index: usize,
    idom: &[Option<usize>],
) -> collections::Vec<collections::Vec<usize>> {
    let mut children = collections::vec![collections::Vec::new(); block_count];
    for block_index in 0..block_count {
        let Some(parent) = idom[block_index] else {
            continue;
        };
        if block_index == entry_index {
            continue;
        }
        children[parent].push(block_index);
    }
    children
}

fn mark_idom_reachable(block_index: usize, children: &[collections::Vec<usize>], out: &mut [bool]) {
    if block_index >= out.len() || out[block_index] {
        return;
    }
    out[block_index] = true;
    for &child in &children[block_index] {
        mark_idom_reachable(child, children, out);
    }
}

fn compute_block_bank_slots(
    program: &SsaProgram,
    slot_to_cached_index: &BTreeMap<FrameSlot, usize>,
    slot_meta: &[SlotLayoutMeta],
) -> collections::Vec<[collections::Vec<usize>; 2]> {
    (0..program.blocks.len())
        .map(|block_index| {
            let mut gp = collections::Vec::new();
            let mut fp = collections::Vec::new();
            let slots = program
                .block_entry_cached_slots
                .get(block_index)
                .map(|slots| slots.as_slice())
                .unwrap_or(&[]);
            for slot in slots {
                let Some(&cached_index) = slot_to_cached_index.get(slot) else {
                    continue;
                };
                match slot_meta[cached_index].bank {
                    LayoutBank::Gp => gp.push(cached_index),
                    LayoutBank::Fp => fp.push(cached_index),
                }
            }
            [gp, fp]
        })
        .collect()
}

fn assign_bank_layouts_from_root(
    block_index: usize,
    parent_layout: Option<&[u32]>,
    bank: LayoutBank,
    lane_count: usize,
    program: &SsaProgram,
    slot_to_cached_index: &BTreeMap<FrameSlot, usize>,
    idom_children: &[collections::Vec<usize>],
    bank_slots: &[[collections::Vec<usize>; 2]],
    slot_meta: &[SlotLayoutMeta],
    param_usage: &[ParamPrefixUsage],
    layouts: &mut [u32],
    exit_layouts: &mut [u32],
    lanes: usize,
    visited: &mut [bool],
) -> Result<(), WasmError> {
    if block_index >= visited.len() || visited[block_index] {
        return Ok(());
    }
    visited[block_index] = true;
    let slots = match bank {
        LayoutBank::Gp => &bank_slots[block_index][0],
        LayoutBank::Fp => &bank_slots[block_index][1],
    };
    let prefix = match bank {
        LayoutBank::Gp => param_usage[block_index].gp,
        LayoutBank::Fp => param_usage[block_index].fp,
    };
    let row_base = block_index * lanes;
    let entry_row =
        build_block_bank_layout(slots, parent_layout, slot_meta, lane_count, prefix, bank)?;
    layouts[row_base..row_base + lanes].copy_from_slice(&entry_row);
    let exit_row = simulate_block_exit_layout(
        &program.blocks[block_index],
        &layouts[row_base..row_base + lanes],
        slot_to_cached_index,
        slot_meta,
        bank,
        lane_count,
        prefix,
    )?;
    exit_layouts[row_base..row_base + lanes].copy_from_slice(&exit_row);
    let current = exit_row;
    for &child in &idom_children[block_index] {
        assign_bank_layouts_from_root(
            child,
            Some(&current),
            bank,
            lane_count,
            program,
            slot_to_cached_index,
            idom_children,
            bank_slots,
            slot_meta,
            param_usage,
            layouts,
            exit_layouts,
            lanes,
            visited,
        )?;
    }
    Ok(())
}

fn simulate_block_exit_layout(
    block: &SsaBlock,
    entry_layout: &[u32],
    slot_to_cached_index: &BTreeMap<FrameSlot, usize>,
    slot_meta: &[SlotLayoutMeta],
    bank: LayoutBank,
    lane_count: usize,
    prefix_occupied: usize,
) -> Result<collections::Vec<u32>, WasmError> {
    let mut layout: collections::Vec<u32> = entry_layout.to_vec().into();
    let mut occupied = collections::vec![false; lane_count];
    for lane in 0..prefix_occupied.min(lane_count) {
        occupied[lane] = true;
    }
    for (slot_index, start) in layout.iter().copied().enumerate() {
        if slot_meta.get(slot_index).map(|meta| meta.bank) != Some(bank) {
            continue;
        }
        if start != LANE_UNASSIGNED {
            occupy_segment(
                &mut occupied,
                start as usize,
                slot_meta[slot_index].width,
                true,
            );
        }
    }

    for inst in &block.ops {
        match inst.op {
            SsaOp::LOCAL_ENSURE_CACHE | SsaOp::LOCAL_RESERVE_CACHE | SsaOp::LOCAL_SET_CACHE => {
                let slot = FrameSlot(inst.meta);
                let Some(&slot_index) = slot_to_cached_index.get(&slot) else {
                    continue;
                };
                if slot_meta[slot_index].bank != bank || layout[slot_index] != LANE_UNASSIGNED {
                    continue;
                }
                let width = slot_meta[slot_index].width;
                if let Some(start) = choose_hole_start(&occupied, width) {
                    occupy_segment(&mut occupied, start, width, true);
                    layout[slot_index] = start as u32;
                } else if bank == LayoutBank::Gp {
                    let preserve = layout.clone();
                    let mut active = layout
                        .iter()
                        .enumerate()
                        .filter_map(|(cached_index, start)| {
                            (slot_meta[cached_index].bank == bank && *start != LANE_UNASSIGNED)
                                .then_some(cached_index)
                        })
                        .collect::<collections::Vec<_>>();
                    active.push(slot_index);
                    layout = exact_gp_layout(
                        &active,
                        Some(&preserve),
                        slot_meta,
                        lane_count,
                        prefix_occupied,
                    )?;
                    occupied.fill(false);
                    for lane in 0..prefix_occupied.min(lane_count) {
                        occupied[lane] = true;
                    }
                    for (cached_index, start) in layout.iter().copied().enumerate() {
                        if slot_meta.get(cached_index).map(|meta| meta.bank) != Some(bank) {
                            continue;
                        }
                        if start != LANE_UNASSIGNED {
                            occupy_segment(
                                &mut occupied,
                                start as usize,
                                slot_meta[cached_index].width,
                                true,
                            );
                        }
                    }
                } else {
                    return Err(WasmError::internal(
                        "FP cache exit layout unexpectedly failed without a feasible hole".into(),
                    ));
                }
            }
            SsaOp::LOCAL_DROP_CACHE => {
                let slot = FrameSlot(inst.meta);
                let Some(&slot_index) = slot_to_cached_index.get(&slot) else {
                    continue;
                };
                if slot_meta[slot_index].bank != bank {
                    continue;
                }
                let prev = core::mem::replace(&mut layout[slot_index], LANE_UNASSIGNED);
                if prev != LANE_UNASSIGNED {
                    occupy_segment(
                        &mut occupied,
                        prev as usize,
                        slot_meta[slot_index].width,
                        false,
                    );
                }
            }
            SsaOp::CALL => {
                for (slot_index, lane) in layout.iter_mut().enumerate() {
                    if slot_meta.get(slot_index).map(|meta| meta.bank) != Some(bank) {
                        continue;
                    }
                    if *lane != LANE_UNASSIGNED {
                        occupy_segment(
                            &mut occupied,
                            *lane as usize,
                            slot_meta[slot_index].width,
                            false,
                        );
                        *lane = LANE_UNASSIGNED;
                    }
                }
            }
            // Value (primitive) / Fill / Spill / LocalGetSlot / LocalGetCache /
            // LocalSetSlot: no cache layout effects.
            _ => {}
        }
    }

    Ok(layout)
}

fn build_block_bank_layout(
    slots: &[usize],
    parent_layout: Option<&[u32]>,
    slot_meta: &[SlotLayoutMeta],
    lane_count: usize,
    prefix_occupied: usize,
    bank: LayoutBank,
) -> Result<collections::Vec<u32>, WasmError> {
    let mut layout = collections::vec![LANE_UNASSIGNED; slot_meta.len()];
    if slots.is_empty() {
        return Ok(layout);
    }
    let mut occupied = collections::vec![false; lane_count];
    for lane in 0..prefix_occupied.min(lane_count) {
        occupied[lane] = true;
    }

    let mut additions = collections::Vec::new();
    if let Some(parent_layout) = parent_layout {
        for &slot in slots {
            let parent_start = parent_layout[slot];
            if parent_start != LANE_UNASSIGNED {
                let start = parent_start as usize;
                let width = slot_meta[slot].width;
                if segment_available(&occupied, start, width) {
                    occupy_segment(&mut occupied, start, width, true);
                    layout[slot] = parent_start;
                    continue;
                }
            }
            additions.push(slot);
        }
    } else {
        additions.extend_from_slice(slots);
    }

    additions.sort_by_key(|&slot| (Reverse(slot_meta[slot].width), slot));
    let mut needs_repack = false;
    for slot in additions {
        let width = slot_meta[slot].width;
        if let Some(start) = choose_hole_start(&occupied, width) {
            occupy_segment(&mut occupied, start, width, true);
            layout[slot] = start as u32;
        } else {
            needs_repack = true;
            break;
        }
    }

    if !needs_repack {
        return Ok(layout);
    }

    if bank != LayoutBank::Gp {
        return Err(WasmError::internal(
            "FP cache lane layout unexpectedly failed without a feasible hole".into(),
        ));
    }

    exact_gp_layout(slots, parent_layout, slot_meta, lane_count, prefix_occupied)
}

fn exact_gp_layout(
    slots: &[usize],
    parent_layout: Option<&[u32]>,
    slot_meta: &[SlotLayoutMeta],
    lane_count: usize,
    prefix_occupied: usize,
) -> Result<collections::Vec<u32>, WasmError> {
    let mut ordered = slots.to_vec();
    ordered.sort_by_key(|&slot| {
        (
            Reverse(slot_meta[slot].width),
            Reverse(
                parent_layout
                    .map(|layout| layout[slot] != LANE_UNASSIGNED)
                    .unwrap_or(false),
            ),
            slot,
        )
    });
    let mut occupied = collections::vec![false; lane_count];
    for lane in 0..prefix_occupied.min(lane_count) {
        occupied[lane] = true;
    }
    let mut current = collections::vec![LANE_UNASSIGNED; slot_meta.len()];
    let mut best: Option<(usize, collections::Vec<u32>)> = None;
    search_exact_gp_layout(
        &ordered,
        0,
        parent_layout,
        slot_meta,
        &mut occupied,
        &mut current,
        0,
        &mut best,
    );
    best.map(|(_, layout)| layout).ok_or_else(|| {
        WasmError::internal("GP cache lane exact layout search found no feasible assignment")
    })
}

fn search_exact_gp_layout(
    ordered: &[usize],
    index: usize,
    parent_layout: Option<&[u32]>,
    slot_meta: &[SlotLayoutMeta],
    occupied: &mut [bool],
    current: &mut [u32],
    current_cost: usize,
    best: &mut Option<(usize, collections::Vec<u32>)>,
) {
    if let Some((best_cost, _)) = best {
        if current_cost > *best_cost {
            return;
        }
    }
    if index == ordered.len() {
        let replace = match best {
            None => true,
            Some((best_cost, best_layout)) => {
                current_cost < *best_cost
                    || (current_cost == *best_cost
                        && lexicographically_better(current, best_layout))
            }
        };
        if replace {
            *best = Some((current_cost, current.to_vec().into()));
        }
        return;
    }

    let slot = ordered[index];
    let width = slot_meta[slot].width;
    let mut starts = feasible_starts(occupied, width);
    let parent_start_opt = parent_layout.and_then(|layout| {
        let v = layout[slot];
        if v != LANE_UNASSIGNED {
            Some(v as usize)
        } else {
            None
        }
    });
    if let Some(parent_start) = parent_start_opt {
        if let Some(pos) = starts.iter().position(|&start| start == parent_start) {
            starts.swap(0, pos);
        }
    }
    for start in starts {
        occupy_segment(occupied, start, width, true);
        current[slot] = start as u32;
        let added_cost = match parent_start_opt {
            Some(parent_start) if parent_start != start => width,
            _ => 0,
        };
        search_exact_gp_layout(
            ordered,
            index + 1,
            parent_layout,
            slot_meta,
            occupied,
            current,
            current_cost + added_cost,
            best,
        );
        current[slot] = LANE_UNASSIGNED;
        occupy_segment(occupied, start, width, false);
    }
}

fn lexicographically_better(lhs: &[u32], rhs: &[u32]) -> bool {
    for (left, right) in lhs.iter().zip(rhs.iter()) {
        let left_set = *left != LANE_UNASSIGNED;
        let right_set = *right != LANE_UNASSIGNED;
        match (left_set, right_set) {
            (true, true) if *left != *right => return *left < *right,
            (true, false) => return true,
            (false, true) => return false,
            _ => {}
        }
    }
    false
}

fn feasible_starts(occupied: &[bool], width: usize) -> collections::Vec<usize> {
    let mut starts = collections::Vec::new();
    if width == 0 || width > occupied.len() {
        return starts;
    }
    for start in 0..=occupied.len() - width {
        if segment_available(occupied, start, width) {
            starts.push(start);
        }
    }
    starts
}

fn choose_hole_start(occupied: &[bool], width: usize) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    let mut lane = 0usize;
    while lane < occupied.len() {
        if occupied[lane] {
            lane += 1;
            continue;
        }
        let start = lane;
        while lane < occupied.len() && !occupied[lane] {
            lane += 1;
        }
        let len = lane - start;
        if len < width {
            continue;
        }
        let rank = (usize::from(len != width), len);
        if best
            .map(|(best_start, best_len)| {
                rank < (usize::from(best_len != width), best_len)
                    || (rank == (usize::from(best_len != width), best_len) && start < best_start)
            })
            .unwrap_or(true)
        {
            best = Some((start, len));
        }
    }
    best.map(|(start, _)| start)
}

fn segment_available(occupied: &[bool], start: usize, width: usize) -> bool {
    start
        .checked_add(width)
        .map(|end| end <= occupied.len() && occupied[start..end].iter().all(|lane| !*lane))
        .unwrap_or(false)
}

fn occupy_segment(occupied: &mut [bool], start: usize, width: usize, value: bool) {
    for lane in &mut occupied[start..start + width] {
        *lane = value;
    }
}

fn regs_for_segment(
    regfile: &MachineRegFile,
    bank: LayoutBank,
    start: usize,
    width: usize,
) -> Result<ValueRegs, WasmError> {
    match (bank, width) {
        (LayoutBank::Fp, 1) => Ok(ValueRegs {
            lo: preferred_fp_dynamic_reg(regfile, start).ok_or_else(|| {
                WasmError::internal("FP cache lane layout exceeded dynamic register budget")
            })?,
            hi: None,
        }),
        (LayoutBank::Gp, 2) => Ok(ValueRegs {
            lo: preferred_gp_dynamic_reg(regfile, start)
                .ok_or_else(|| WasmError::internal("GP cache lane layout exceeded pair budget"))?,
            hi: Some(
                preferred_gp_dynamic_reg(regfile, start + 1).ok_or_else(|| {
                    WasmError::internal("GP cache lane layout exceeded pair budget")
                })?,
            ),
        }),
        (LayoutBank::Gp, 1) => Ok(ValueRegs {
            lo: preferred_gp_dynamic_reg(regfile, start).ok_or_else(|| {
                WasmError::internal("GP cache lane layout exceeded dynamic register budget")
            })?,
            hi: None,
        }),
        _ => Err(WasmError::internal(
            "unsupported cache lane segment width for bank".into(),
        )),
    }
}

fn block_successors(terminator: &SsaTerminator) -> collections::Vec<SsaTarget> {
    match terminator {
        SsaTerminator::Goto(edge) => collections::vec![edge.target],
        SsaTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => collections::vec![then_edge.target, else_edge.target],
        SsaTerminator::BrTable { entries, .. } => {
            entries.iter().map(|entry| entry.target).collect()
        }
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => collections::Vec::new(),
    }
}
