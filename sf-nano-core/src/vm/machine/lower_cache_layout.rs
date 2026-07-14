use crate::collections;
use tracked_alloc::collections::BTreeMap;

use core::cmp::Reverse;

use crate::{
    error::WasmError,
    vm::{
        entities::TableDispatchMode,
        machine::machine_ir::MachineStorageType,
        middle::{
            cell::CellId,
            ssa_ir::{
                ir::{
                    EntryCacheRequirement, SsaBlock, SsaCallOp, SsaOp, SsaProgram, SsaTerminator,
                },
                target::SsaTarget,
            },
        },
    },
};

use super::{
    lower_context::{CachedCell, EntryCacheParam, ValueRegs},
    lower_module::{preferred_fp_dynamic_reg, preferred_gp_dynamic_reg},
    lower_regalloc::MachineRegFile,
};

/// Sentinel for "no lane assigned" in the cell->lane layout matrices.
/// Replaces `None` in what was previously `Option<usize>` (halves the
/// per-element size from 16 to 4 bytes).
const LANE_UNASSIGNED: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayoutBank {
    Gp,
    Fp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CellLayoutMeta {
    bank: LayoutBank,
    width: usize,
    is_ref: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ParamPrefixUsage {
    gp: usize,
    fp: usize,
}

pub(super) fn compute_block_entry_cache_params(
    regfile: &MachineRegFile,
    program: &SsaProgram,
    cached_cells: &[CachedCell],
    gp_reg_width: u8,
    is_local_func: &[bool],
    table_dispatch_modes: &[TableDispatchMode],
    preferred_preserved: &[bool],
) -> Result<collections::Vec<collections::Vec<EntryCacheParam>>, WasmError> {
    if program.blocks.is_empty() || cached_cells.is_empty() {
        return Ok(collections::vec![collections::Vec::new(); program.blocks.len()]);
    }

    let cell_to_cached_index = cached_cells
        .iter()
        .enumerate()
        .map(|(cached_index, cached)| (cached.cell, cached_index))
        .collect::<BTreeMap<_, _>>();
    let cell_meta = cached_cells
        .iter()
        .map(|cached| CellLayoutMeta {
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
            is_ref: cached.value_ty.is_ref(),
        })
        .collect::<collections::Vec<_>>();
    let param_usage = compute_param_prefix_usage(program, gp_reg_width);
    // Preference is the whole-function `preferred_preserved` flag (per local
    // cell — the residency solver's preserved-class nomination, published
    // through the plan).
    // Promote the per-cell flag to the per-block-per-cached-index shape the
    // layout consumes; the flag is already function-global, so every block gets
    // the same row.
    let call_preserve_preferences =
        promote_preferred_preserved(cached_cells, preferred_preserved, program.blocks.len());
    let predecessors = compute_predecessors(program);
    let rpo = compute_reverse_postorder(program);
    let idom =
        compute_immediate_dominators(program.blocks.len(), program.entry, &predecessors, &rpo);
    let idom_children = build_idom_children(program.blocks.len(), program.entry.as_usize(), &idom);
    let bank_cells = compute_block_bank_slots(program, &cell_to_cached_index, &cell_meta);

    let lanes = cached_cells.len();
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
            &cell_to_cached_index,
            &idom_children,
            &bank_cells,
            &cell_meta,
            &param_usage,
            is_local_func,
            table_dispatch_modes,
            &call_preserve_preferences,
            &regfile.gp_allocatable_preserved_lanes(),
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
            &cell_to_cached_index,
            &idom_children,
            &bank_cells,
            &cell_meta,
            &param_usage,
            is_local_func,
            table_dispatch_modes,
            &call_preserve_preferences,
            &regfile.fp_allocatable_preserved_lanes(),
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
            &cell_to_cached_index,
            &idom_children,
            &bank_cells,
            &cell_meta,
            &param_usage,
            is_local_func,
            table_dispatch_modes,
            &call_preserve_preferences,
            &regfile.gp_allocatable_preserved_lanes(),
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
            &cell_to_cached_index,
            &idom_children,
            &bank_cells,
            &cell_meta,
            &param_usage,
            is_local_func,
            table_dispatch_modes,
            &call_preserve_preferences,
            &regfile.fp_allocatable_preserved_lanes(),
            &mut fp_layouts,
            &mut fp_exit_layouts,
            lanes,
            &mut fp_visited,
        )?;
    }

    improve_gp_layouts_for_incoming_edges(
        program,
        &predecessors,
        &cell_to_cached_index,
        &bank_cells,
        &cell_meta,
        &param_usage,
        is_local_func,
        table_dispatch_modes,
        &call_preserve_preferences,
        &regfile.gp_allocatable_preserved_lanes(),
        regfile.gp_allocatable_count(),
        &mut gp_layouts,
        &mut gp_exit_layouts,
        lanes,
    )?;

    let mut layouts = collections::vec![collections::Vec::new(); program.blocks.len()];
    for block_index in 0..program.blocks.len() {
        let entry_slots = program
            .block_entry_cached_cells
            .get(block_index)
            .map(|s| s.as_slice())
            .unwrap_or(&[]);
        let entry_requirements = program
            .block_entry_cache_requirements
            .get(block_index)
            .map(|r| r.as_slice())
            .unwrap_or(&[]);
        // Upper bound: one entry per entry cell, minus any that miss
        // cell_to_cached_index. Sizing tight avoids ~2x push-growth overallocation
        // on the per-block Vec, which otherwise dominates mir_lower peak memory.
        let mut entries =
            collections::Vec::<(u16, u16, EntryCacheParam)>::with_capacity(entry_slots.len());
        for (position, &cell) in entry_slots.iter().enumerate() {
            let Some(&cached_index) = cell_to_cached_index.get(&cell) else {
                continue;
            };
            let cached_index_u16 = u16::try_from(cached_index).map_err(|_| {
                WasmError::internal("cached local index overflowed entry-cache metadata")
            })?;
            let row_base = block_index * lanes;
            let lane_val = match cell_meta[cached_index].bank {
                LayoutBank::Gp => gp_layouts[row_base + cached_index],
                LayoutBank::Fp => fp_layouts[row_base + cached_index],
            };
            if lane_val == LANE_UNASSIGNED {
                return Err(WasmError::internal(
                    "cache lane layout missing for cell in block b",
                ));
            }
            let start = lane_val as usize;
            let regs = regs_for_segment(
                regfile,
                cell_meta[cached_index].bank,
                start,
                cell_meta[cached_index].width,
            )?;
            // needs_value comes from the plan's row requirement (Ensure vs
            // Reserve), not a machine op re-scan.
            let requirement = entry_requirements.get(position).copied().ok_or_else(|| {
                WasmError::internal("entry cache cell in block b has no entry requirement")
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
                let ty = super::lower_regalloc::lir_value_storage_type(program, param);
                if ty.is_fp() {
                    usage.fp += 1;
                } else if gp_reg_width == 4 && matches!(ty, MachineStorageType::GpI64) {
                    usage.gp += 2;
                } else {
                    usage.gp += 1;
                }
            }
            usage
        })
        .collect()
}

/// Promote the whole-function `preferred_preserved` flag (per local cell,
/// the residency solver's nomination, published through the plan) to the per-block-per-cached-index
/// shape the layout consumes. The flag is already function-global, so every
/// block gets the same row.
pub(super) fn promote_preferred_preserved(
    cached_cells: &[CachedCell],
    preferred_preserved: &[bool],
    block_count: usize,
) -> collections::Vec<collections::Vec<bool>> {
    let per_cached: collections::Vec<bool> = cached_cells
        .iter()
        .map(|cached| {
            preferred_preserved
                .get(cached.cell.0 as usize)
                .copied()
                .unwrap_or(false)
        })
        .collect();
    collections::vec![per_cached; block_count]
}

/// Lane-stability classification for the layout sim: calls across which a
/// cached local's LANE ASSIGNMENT may be kept (direct local-JIT calls, plus
/// fixed-local-only indirect dispatch whose continuation resumes locally).
/// This deliberately credits more calls than the value-carry gate
/// (`is_direct_local_ssa_call` / the plan's survivable classification):
/// keeping the lane assigned means the post-call reload lands in the same
/// lane and spares cross-edge shuffles, even when the value itself dies at
/// the boundary. Narrowing this to direct-only was measured at +1,314
/// instructions on sqlite.
pub(super) fn is_local_jit_call(
    program: &SsaProgram,
    is_local_func: &[bool],
    table_dispatch_modes: &[TableDispatchMode],
    call_index: u16,
) -> bool {
    match program.call_ops.get(call_index as usize) {
        Some(SsaCallOp::CallDirect { callee, .. }) => is_local_func
            .get(*callee as usize)
            .copied()
            .unwrap_or(false),
        Some(SsaCallOp::CallIndirect { table_idx, .. }) => matches!(
            table_dispatch_modes.get(*table_idx as usize).copied(),
            Some(TableDispatchMode::FixedLocalOnly { .. })
        ),
        Some(SsaCallOp::CallRef { .. }) | None => false,
    }
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
    let mut seen = collections::vec![false; program.blocks.len()];
    let mut order = collections::Vec::with_capacity(program.blocks.len());
    let mut stack = collections::vec![(program.entry.as_usize(), false)];
    while let Some((block_index, expanded)) = stack.pop() {
        if block_index >= program.blocks.len() {
            continue;
        }
        if expanded {
            order.push(block_index);
            continue;
        }
        if seen[block_index] {
            continue;
        }
        seen[block_index] = true;
        stack.push((block_index, true));
        let successors = block_successors(&program.blocks[block_index].terminator);
        for succ in successors.iter().rev() {
            let succ_index = succ.as_usize();
            if succ_index < program.blocks.len() && !seen[succ_index] {
                stack.push((succ_index, false));
            }
        }
    }
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
    let mut stack = collections::vec![block_index];
    while let Some(block_index) = stack.pop() {
        if block_index >= out.len() || out[block_index] {
            continue;
        }
        out[block_index] = true;
        for &child in children[block_index].iter().rev() {
            stack.push(child);
        }
    }
}

fn compute_block_bank_slots(
    program: &SsaProgram,
    cell_to_cached_index: &BTreeMap<CellId, usize>,
    cell_meta: &[CellLayoutMeta],
) -> collections::Vec<[collections::Vec<usize>; 2]> {
    (0..program.blocks.len())
        .map(|block_index| {
            let mut gp = collections::Vec::new();
            let mut fp = collections::Vec::new();
            let cells = program
                .block_entry_cached_cells
                .get(block_index)
                .map(|cells| cells.as_slice())
                .unwrap_or(&[]);
            for cell in cells {
                let Some(&cached_index) = cell_to_cached_index.get(cell) else {
                    continue;
                };
                match cell_meta[cached_index].bank {
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
    cell_to_cached_index: &BTreeMap<CellId, usize>,
    idom_children: &[collections::Vec<usize>],
    bank_cells: &[[collections::Vec<usize>; 2]],
    cell_meta: &[CellLayoutMeta],
    param_usage: &[ParamPrefixUsage],
    is_local_func: &[bool],
    table_dispatch_modes: &[TableDispatchMode],
    call_preserve_preferences: &[collections::Vec<bool>],
    preserved_lanes: &[bool],
    layouts: &mut [u32],
    exit_layouts: &mut [u32],
    lanes: usize,
    visited: &mut [bool],
) -> Result<(), WasmError> {
    let mut stack = collections::Vec::new();
    stack.push((block_index, parent_layout.map(collections::Vec::from)));
    while let Some((block_index, parent_layout)) = stack.pop() {
        if block_index >= visited.len() || visited[block_index] {
            continue;
        }
        visited[block_index] = true;
        let cells = match bank {
            LayoutBank::Gp => &bank_cells[block_index][0],
            LayoutBank::Fp => &bank_cells[block_index][1],
        };
        let prefix = match bank {
            LayoutBank::Gp => param_usage[block_index].gp,
            LayoutBank::Fp => param_usage[block_index].fp,
        };
        let row_base = block_index * lanes;
        let entry_row = build_block_bank_layout(
            cells,
            parent_layout.as_deref(),
            cell_meta,
            lane_count,
            prefix,
            bank,
            call_preserve_preferences
                .get(block_index)
                .map(|bits| bits.as_slice())
                .unwrap_or(&[]),
            preserved_lanes,
        )?;
        layouts[row_base..row_base + lanes].copy_from_slice(&entry_row);
        let exit_row = simulate_block_exit_layout(
            &program.blocks[block_index],
            &layouts[row_base..row_base + lanes],
            cell_to_cached_index,
            cell_meta,
            bank,
            lane_count,
            prefix,
            program,
            is_local_func,
            table_dispatch_modes,
            call_preserve_preferences
                .get(block_index)
                .map(|bits| bits.as_slice())
                .unwrap_or(&[]),
            preserved_lanes,
        )?;
        exit_layouts[row_base..row_base + lanes].copy_from_slice(&exit_row);
        for &child in idom_children[block_index].iter().rev() {
            stack.push((child, Some(exit_row.clone())));
        }
    }
    Ok(())
}

fn improve_gp_layouts_for_incoming_edges(
    program: &SsaProgram,
    predecessors: &[collections::Vec<usize>],
    cell_to_cached_index: &BTreeMap<CellId, usize>,
    bank_cells: &[[collections::Vec<usize>; 2]],
    cell_meta: &[CellLayoutMeta],
    param_usage: &[ParamPrefixUsage],
    is_local_func: &[bool],
    table_dispatch_modes: &[TableDispatchMode],
    call_preserve_preferences: &[collections::Vec<bool>],
    preserved_lanes: &[bool],
    lane_count: usize,
    layouts: &mut [u32],
    exit_layouts: &mut [u32],
    lanes: usize,
) -> Result<(), WasmError> {
    for _ in 0..2 {
        let mut changed = false;
        for block_index in 0..program.blocks.len() {
            let Some(preds) = predecessors.get(block_index) else {
                continue;
            };
            if preds.is_empty() {
                continue;
            }
            let cells = &bank_cells[block_index][0];
            if cells.is_empty() {
                continue;
            }
            let row_base = block_index * lanes;
            let current = collections::Vec::from(&layouts[row_base..row_base + lanes]);
            let current = current.as_slice();
            let pred_layouts = preds
                .iter()
                .filter_map(|pred| {
                    let start = pred.checked_mul(lanes)?;
                    exit_layouts
                        .get(start..start + lanes)
                        .map(collections::Vec::from)
                })
                .collect::<collections::Vec<_>>();
            if pred_layouts.is_empty() {
                continue;
            }
            let pred_layout_refs = pred_layouts
                .iter()
                .map(|layout| layout.as_slice())
                .collect::<collections::Vec<_>>();
            let needs_value = entry_cache_needs_value(
                program,
                block_index,
                cell_to_cached_index,
                cell_meta.len(),
            );
            let Some(candidate) = edge_preferred_gp_layout(
                cells,
                current,
                &pred_layout_refs,
                &needs_value,
                cell_meta,
                lane_count,
                param_usage[block_index].gp,
                call_preserve_preferences
                    .get(block_index)
                    .map(|bits| bits.as_slice())
                    .unwrap_or(&[]),
                preserved_lanes,
            ) else {
                continue;
            };
            let current_cost = incoming_edge_layout_cost(
                cells,
                current,
                &pred_layout_refs,
                &needs_value,
                cell_meta,
                call_preserve_preferences
                    .get(block_index)
                    .map(|bits| bits.as_slice())
                    .unwrap_or(&[]),
                preserved_lanes,
            );
            let candidate_cost = incoming_edge_layout_cost(
                cells,
                &candidate,
                &pred_layout_refs,
                &needs_value,
                cell_meta,
                call_preserve_preferences
                    .get(block_index)
                    .map(|bits| bits.as_slice())
                    .unwrap_or(&[]),
                preserved_lanes,
            );
            if candidate_cost >= current_cost || candidate.as_slice() == current {
                continue;
            }
            layouts[row_base..row_base + lanes].copy_from_slice(&candidate);
            let exit_row = simulate_block_exit_layout(
                &program.blocks[block_index],
                &layouts[row_base..row_base + lanes],
                cell_to_cached_index,
                cell_meta,
                LayoutBank::Gp,
                lane_count,
                param_usage[block_index].gp,
                program,
                is_local_func,
                table_dispatch_modes,
                call_preserve_preferences
                    .get(block_index)
                    .map(|bits| bits.as_slice())
                    .unwrap_or(&[]),
                preserved_lanes,
            )?;
            exit_layouts[row_base..row_base + lanes].copy_from_slice(&exit_row);
            changed = true;
        }
        if !changed {
            break;
        }
    }
    Ok(())
}

fn entry_cache_needs_value(
    program: &SsaProgram,
    block_index: usize,
    cell_to_cached_index: &BTreeMap<CellId, usize>,
    cell_count: usize,
) -> collections::Vec<bool> {
    let mut needs = collections::vec![false; cell_count];
    let Some(entry_slots) = program.block_entry_cached_cells.get(block_index) else {
        return needs;
    };
    let entry_requirements = program
        .block_entry_cache_requirements
        .get(block_index)
        .map(|r| r.as_slice())
        .unwrap_or(&[]);
    for (position, &cell) in entry_slots.iter().enumerate() {
        if entry_requirements.get(position).copied() != Some(EntryCacheRequirement::Ensure) {
            continue;
        }
        if let Some(&cached_index) = cell_to_cached_index.get(&cell) {
            if let Some(bit) = needs.get_mut(cached_index) {
                *bit = true;
            }
        }
    }
    needs
}

fn edge_preferred_gp_layout(
    cells: &[usize],
    current_layout: &[u32],
    pred_layouts: &[&[u32]],
    needs_value: &[bool],
    cell_meta: &[CellLayoutMeta],
    lane_count: usize,
    prefix_occupied: usize,
    call_preserve_preferences: &[bool],
    preserved_lanes: &[bool],
) -> Option<collections::Vec<u32>> {
    let mut layout = collections::vec![LANE_UNASSIGNED; cell_meta.len()];
    let mut occupied = collections::vec![false; lane_count];
    for lane in 0..prefix_occupied.min(lane_count) {
        occupied[lane] = true;
    }
    let mut ordered = cells.to_vec();
    ordered.sort_by_key(|&cell| {
        (
            Reverse(cell_meta[cell].width),
            Reverse(
                call_preserve_preferences
                    .get(cell)
                    .copied()
                    .unwrap_or(false),
            ),
            cell,
        )
    });

    for cell in ordered {
        let width = cell_meta[cell].width;
        let prefer_preserved = call_preserve_preferences
            .get(cell)
            .copied()
            .unwrap_or(false);
        let mut best: Option<(usize, usize)> = None;
        for start in 0..=lane_count.saturating_sub(width) {
            if !segment_available(&occupied, start, width) {
                continue;
            }
            let cost = edge_slot_layout_cost(
                cell,
                start,
                width,
                pred_layouts,
                needs_value,
                preserved_lanes,
                prefer_preserved,
            )
            .saturating_add(
                current_layout
                    .get(cell)
                    .copied()
                    .filter(|current| *current != LANE_UNASSIGNED && *current as usize != start)
                    .map(|_| 1)
                    .unwrap_or(0),
            );
            if best
                .map(|(best_cost, best_start)| {
                    cost < best_cost || (cost == best_cost && start < best_start)
                })
                .unwrap_or(true)
            {
                best = Some((cost, start));
            }
        }
        let (_, start) = best?;
        occupy_segment(&mut occupied, start, width, true);
        layout[cell] = start as u32;
    }
    Some(layout)
}

fn incoming_edge_layout_cost(
    cells: &[usize],
    layout: &[u32],
    pred_layouts: &[&[u32]],
    needs_value: &[bool],
    cell_meta: &[CellLayoutMeta],
    call_preserve_preferences: &[bool],
    preserved_lanes: &[bool],
) -> usize {
    cells
        .iter()
        .copied()
        .map(|cell| {
            let start = layout[cell];
            if start == LANE_UNASSIGNED {
                return usize::MAX / 4;
            }
            let width = cell_meta[cell].width;
            edge_slot_layout_cost(
                cell,
                start as usize,
                width,
                pred_layouts,
                needs_value,
                preserved_lanes,
                call_preserve_preferences
                    .get(cell)
                    .copied()
                    .unwrap_or(false),
            )
        })
        .sum()
}

fn edge_slot_layout_cost(
    cell: usize,
    start: usize,
    width: usize,
    pred_layouts: &[&[u32]],
    needs_value: &[bool],
    preserved_lanes: &[bool],
    prefer_preserved: bool,
) -> usize {
    let mut cost = preference_mismatch_cost(preserved_lanes, start, width, prefer_preserved);
    if needs_value.get(cell).copied().unwrap_or(true) {
        for pred in pred_layouts {
            match pred.get(cell).copied().unwrap_or(LANE_UNASSIGNED) {
                LANE_UNASSIGNED => cost = cost.saturating_add(width.saturating_mul(2)),
                pred_start if pred_start as usize != start => {
                    cost = cost.saturating_add(width);
                }
                _ => {}
            }
        }
    }
    cost
}

fn simulate_block_exit_layout(
    block: &SsaBlock,
    entry_layout: &[u32],
    cell_to_cached_index: &BTreeMap<CellId, usize>,
    cell_meta: &[CellLayoutMeta],
    bank: LayoutBank,
    lane_count: usize,
    prefix_occupied: usize,
    program: &SsaProgram,
    is_local_func: &[bool],
    table_dispatch_modes: &[TableDispatchMode],
    call_preserve_preferences: &[bool],
    preserved_lanes: &[bool],
) -> Result<collections::Vec<u32>, WasmError> {
    let mut layout: collections::Vec<u32> = entry_layout.to_vec().into();
    let mut occupied = collections::vec![false; lane_count];
    for lane in 0..prefix_occupied.min(lane_count) {
        occupied[lane] = true;
    }
    for (cell_index, start) in layout.iter().copied().enumerate() {
        if cell_meta.get(cell_index).map(|meta| meta.bank) != Some(bank) {
            continue;
        }
        if start != LANE_UNASSIGNED {
            occupy_segment(
                &mut occupied,
                start as usize,
                cell_meta[cell_index].width,
                true,
            );
        }
    }

    for inst in &block.ops {
        match inst.op {
            SsaOp::CELL_ENSURE_CACHE => {
                let cell = CellId(inst.meta);
                let Some(&cell_index) = cell_to_cached_index.get(&cell) else {
                    continue;
                };
                if cell_meta[cell_index].bank != bank {
                    continue;
                }
                if layout[cell_index] != LANE_UNASSIGNED {
                    continue;
                }
                let width = cell_meta[cell_index].width;
                let prefer_preserved = call_preserve_preferences
                    .get(cell_index)
                    .copied()
                    .unwrap_or(false);
                if let Some(start) =
                    choose_hole_start(&occupied, width, preserved_lanes, prefer_preserved)
                {
                    occupy_segment(&mut occupied, start, width, true);
                    layout[cell_index] = start as u32;
                } else if bank == LayoutBank::Gp {
                    let preserve = layout.clone();
                    let mut active = layout
                        .iter()
                        .enumerate()
                        .filter_map(|(cached_index, start)| {
                            (cell_meta[cached_index].bank == bank && *start != LANE_UNASSIGNED)
                                .then_some(cached_index)
                        })
                        .collect::<collections::Vec<_>>();
                    active.push(cell_index);
                    layout = exact_gp_layout(
                        &active,
                        Some(&preserve),
                        cell_meta,
                        lane_count,
                        prefix_occupied,
                        call_preserve_preferences,
                        preserved_lanes,
                    )?;
                    occupied.fill(false);
                    for lane in 0..prefix_occupied.min(lane_count) {
                        occupied[lane] = true;
                    }
                    for (cached_index, start) in layout.iter().copied().enumerate() {
                        if cell_meta.get(cached_index).map(|meta| meta.bank) != Some(bank) {
                            continue;
                        }
                        if start != LANE_UNASSIGNED {
                            occupy_segment(
                                &mut occupied,
                                start as usize,
                                cell_meta[cached_index].width,
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
            SsaOp::CELL_RESERVE_CACHE | SsaOp::CELL_SET_CACHE => {
                let cell = CellId(inst.meta);
                let Some(&cell_index) = cell_to_cached_index.get(&cell) else {
                    continue;
                };
                if cell_meta[cell_index].bank != bank {
                    continue;
                }
                if layout[cell_index] == LANE_UNASSIGNED {
                    let width = cell_meta[cell_index].width;
                    let prefer_preserved = call_preserve_preferences
                        .get(cell_index)
                        .copied()
                        .unwrap_or(false);
                    if let Some(start) =
                        choose_hole_start(&occupied, width, preserved_lanes, prefer_preserved)
                    {
                        occupy_segment(&mut occupied, start, width, true);
                        layout[cell_index] = start as u32;
                    } else if bank == LayoutBank::Gp {
                        let preserve = layout.clone();
                        let mut active = layout
                            .iter()
                            .enumerate()
                            .filter_map(|(cached_index, start)| {
                                (cell_meta[cached_index].bank == bank && *start != LANE_UNASSIGNED)
                                    .then_some(cached_index)
                            })
                            .collect::<collections::Vec<_>>();
                        active.push(cell_index);
                        layout = exact_gp_layout(
                            &active,
                            Some(&preserve),
                            cell_meta,
                            lane_count,
                            prefix_occupied,
                            call_preserve_preferences,
                            preserved_lanes,
                        )?;
                        occupied.fill(false);
                        for lane in 0..prefix_occupied.min(lane_count) {
                            occupied[lane] = true;
                        }
                        for (cached_index, start) in layout.iter().copied().enumerate() {
                            if cell_meta.get(cached_index).map(|meta| meta.bank) != Some(bank) {
                                continue;
                            }
                            if start != LANE_UNASSIGNED {
                                occupy_segment(
                                    &mut occupied,
                                    start as usize,
                                    cell_meta[cached_index].width,
                                    true,
                                );
                            }
                        }
                    } else {
                        return Err(WasmError::internal(
                            "FP cache exit layout unexpectedly failed without a feasible hole"
                                .into(),
                        ));
                    }
                }
            }
            SsaOp::CELL_DROP_CACHE => {
                let cell = CellId(inst.meta);
                let Some(&cell_index) = cell_to_cached_index.get(&cell) else {
                    continue;
                };
                if cell_meta[cell_index].bank != bank {
                    continue;
                }
                let prev = core::mem::replace(&mut layout[cell_index], LANE_UNASSIGNED);
                if prev != LANE_UNASSIGNED {
                    occupy_segment(
                        &mut occupied,
                        prev as usize,
                        cell_meta[cell_index].width,
                        false,
                    );
                }
            }
            SsaOp::CALL => {
                let local_jit_call =
                    is_local_jit_call(program, is_local_func, table_dispatch_modes, inst.meta);
                for (cell_index, lane) in layout.iter_mut().enumerate() {
                    if cell_meta.get(cell_index).map(|meta| meta.bank) != Some(bank) {
                        continue;
                    }
                    if *lane == LANE_UNASSIGNED {
                        continue;
                    }
                    if local_jit_call
                        && call_preserve_preferences
                            .get(cell_index)
                            .copied()
                            .unwrap_or(false)
                        && !cell_meta[cell_index].is_ref
                    {
                        let start = *lane as usize;
                        let width = cell_meta[cell_index].width;
                        if segment_is_preserved(preserved_lanes, start, width) {
                            continue;
                        }
                    }
                    occupy_segment(
                        &mut occupied,
                        *lane as usize,
                        cell_meta[cell_index].width,
                        false,
                    );
                    *lane = LANE_UNASSIGNED;
                }
            }
            // Value (primitive) / Fill / Spill / CellGetSlot / CellGetCache /
            // CellSetSlot: no cache layout effects.
            _ => {}
        }
    }

    Ok(layout)
}

fn build_block_bank_layout(
    cells: &[usize],
    parent_layout: Option<&[u32]>,
    cell_meta: &[CellLayoutMeta],
    lane_count: usize,
    prefix_occupied: usize,
    bank: LayoutBank,
    call_preserve_preferences: &[bool],
    preserved_lanes: &[bool],
) -> Result<collections::Vec<u32>, WasmError> {
    let mut layout = collections::vec![LANE_UNASSIGNED; cell_meta.len()];
    if cells.is_empty() {
        return Ok(layout);
    }
    let mut occupied = collections::vec![false; lane_count];
    for lane in 0..prefix_occupied.min(lane_count) {
        occupied[lane] = true;
    }

    let mut additions = collections::Vec::new();
    if let Some(parent_layout) = parent_layout {
        for &cell in cells {
            let parent_start = parent_layout[cell];
            if parent_start != LANE_UNASSIGNED {
                let start = parent_start as usize;
                let width = cell_meta[cell].width;
                let prefer_preserved = call_preserve_preferences
                    .get(cell)
                    .copied()
                    .unwrap_or(false);
                if segment_available(&occupied, start, width)
                    && segment_matches_preference(preserved_lanes, start, width, prefer_preserved)
                {
                    occupy_segment(&mut occupied, start, width, true);
                    layout[cell] = parent_start;
                    continue;
                }
            }
            additions.push(cell);
        }
    } else {
        additions.extend_from_slice(cells);
    }

    additions.sort_by_key(|&cell| {
        (
            Reverse(cell_meta[cell].width),
            Reverse(
                call_preserve_preferences
                    .get(cell)
                    .copied()
                    .unwrap_or(false),
            ),
            cell,
        )
    });
    let mut needs_repack = false;
    for cell in additions {
        let width = cell_meta[cell].width;
        let prefer_preserved = call_preserve_preferences
            .get(cell)
            .copied()
            .unwrap_or(false);
        if let Some(start) = choose_hole_start(&occupied, width, preserved_lanes, prefer_preserved)
        {
            occupy_segment(&mut occupied, start, width, true);
            layout[cell] = start as u32;
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

    exact_gp_layout(
        cells,
        parent_layout,
        cell_meta,
        lane_count,
        prefix_occupied,
        call_preserve_preferences,
        preserved_lanes,
    )
}

fn exact_gp_layout(
    cells: &[usize],
    parent_layout: Option<&[u32]>,
    cell_meta: &[CellLayoutMeta],
    lane_count: usize,
    prefix_occupied: usize,
    call_preserve_preferences: &[bool],
    preserved_lanes: &[bool],
) -> Result<collections::Vec<u32>, WasmError> {
    let mut ordered = cells.to_vec();
    ordered.sort_by_key(|&cell| {
        (
            Reverse(cell_meta[cell].width),
            Reverse(
                parent_layout
                    .map(|layout| layout[cell] != LANE_UNASSIGNED)
                    .unwrap_or(false),
            ),
            cell,
        )
    });
    let mut occupied = collections::vec![false; lane_count];
    for lane in 0..prefix_occupied.min(lane_count) {
        occupied[lane] = true;
    }
    let mut current = collections::vec![LANE_UNASSIGNED; cell_meta.len()];
    let mut best: Option<(usize, collections::Vec<u32>)> = None;
    search_exact_gp_layout(
        &ordered,
        0,
        parent_layout,
        cell_meta,
        call_preserve_preferences,
        preserved_lanes,
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
    cell_meta: &[CellLayoutMeta],
    call_preserve_preferences: &[bool],
    preserved_lanes: &[bool],
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

    let cell = ordered[index];
    let width = cell_meta[cell].width;
    let prefer_preserved = call_preserve_preferences
        .get(cell)
        .copied()
        .unwrap_or(false);
    let mut starts = feasible_starts(occupied, width, preserved_lanes, prefer_preserved);
    let parent_start_opt = parent_layout.and_then(|layout| {
        let v = layout[cell];
        if v != LANE_UNASSIGNED {
            Some(v as usize)
        } else {
            None
        }
    });
    if let Some(parent_start) = parent_start_opt {
        if segment_matches_preference(preserved_lanes, parent_start, width, prefer_preserved) {
            if let Some(pos) = starts.iter().position(|&start| start == parent_start) {
                starts.swap(0, pos);
            }
        }
    }
    for start in starts {
        occupy_segment(occupied, start, width, true);
        current[cell] = start as u32;
        let move_cost = match parent_start_opt {
            Some(parent_start) if parent_start != start => width,
            _ => 0,
        };
        let preference_cost =
            preference_mismatch_cost(preserved_lanes, start, width, prefer_preserved);
        let added_cost = move_cost.saturating_add(preference_cost);
        search_exact_gp_layout(
            ordered,
            index + 1,
            parent_layout,
            cell_meta,
            call_preserve_preferences,
            preserved_lanes,
            occupied,
            current,
            current_cost + added_cost,
            best,
        );
        current[cell] = LANE_UNASSIGNED;
        occupy_segment(occupied, start, width, false);
    }
}

fn preference_mismatch_cost(
    preserved_lanes: &[bool],
    start: usize,
    width: usize,
    prefer_preserved: bool,
) -> usize {
    if segment_matches_preference(preserved_lanes, start, width, prefer_preserved) {
        0
    } else {
        width.saturating_mul(preserved_lanes.len().saturating_add(1))
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

fn feasible_starts(
    occupied: &[bool],
    width: usize,
    preserved_lanes: &[bool],
    prefer_preserved: bool,
) -> collections::Vec<usize> {
    let mut starts = collections::Vec::new();
    if width == 0 || width > occupied.len() {
        return starts;
    }
    for start in 0..=occupied.len() - width {
        if segment_available(occupied, start, width)
            && segment_matches_preference(preserved_lanes, start, width, prefer_preserved)
        {
            starts.push(start);
        }
    }
    for start in 0..=occupied.len() - width {
        if segment_available(occupied, start, width)
            && !segment_matches_preference(preserved_lanes, start, width, prefer_preserved)
        {
            starts.push(start);
        }
    }
    starts
}

fn choose_hole_start(
    occupied: &[bool],
    width: usize,
    preserved_lanes: &[bool],
    prefer_preserved: bool,
) -> Option<usize> {
    choose_hole_start_with_filter(occupied, width, |candidate| {
        segment_matches_preference(preserved_lanes, candidate, width, prefer_preserved)
    })
    .or_else(|| choose_hole_start_with_filter(occupied, width, |_| true))
}

fn choose_hole_start_with_filter(
    occupied: &[bool],
    width: usize,
    allow_start: impl Fn(usize) -> bool,
) -> Option<usize> {
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
        for candidate in start..=lane - width {
            if !allow_start(candidate) {
                continue;
            }
            if best
                .map(|(best_start, best_len)| {
                    rank < (usize::from(best_len != width), best_len)
                        || (rank == (usize::from(best_len != width), best_len)
                            && candidate < best_start)
                })
                .unwrap_or(true)
            {
                best = Some((candidate, len));
            }
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

fn segment_is_preserved(preserved_lanes: &[bool], start: usize, width: usize) -> bool {
    start
        .checked_add(width)
        .map(|end| {
            end <= preserved_lanes.len() && preserved_lanes[start..end].iter().all(|lane| *lane)
        })
        .unwrap_or(false)
}

fn segment_uses_preserved(preserved_lanes: &[bool], start: usize, width: usize) -> bool {
    start
        .checked_add(width)
        .map(|end| {
            end <= preserved_lanes.len() && preserved_lanes[start..end].iter().any(|lane| *lane)
        })
        .unwrap_or(false)
}

fn segment_matches_preference(
    preserved_lanes: &[bool],
    start: usize,
    width: usize,
    prefer_preserved: bool,
) -> bool {
    if prefer_preserved {
        segment_is_preserved(preserved_lanes, start, width)
    } else {
        !segment_uses_preserved(preserved_lanes, start, width)
    }
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
        SsaTerminator::Return { .. }
        | SsaTerminator::ReturnScalar { .. }
        | SsaTerminator::TailCallDirect { .. }
        | SsaTerminator::TailCallIndirect { .. }
        | SsaTerminator::TailCallRef { .. }
        | SsaTerminator::TrapUnreachable
        | SsaTerminator::EhThrow { .. }
        | SsaTerminator::EhThrowRef { .. } => collections::Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::middle::ssa_ir::ir::SsaEdge;

    fn long_linear_program(block_count: usize) -> SsaProgram {
        let blocks = (0..block_count)
            .map(|index| {
                let terminator = if index + 1 < block_count {
                    SsaTerminator::Goto(SsaEdge {
                        target: SsaTarget((index + 1) as u32),
                        bindings: collections::Vec::new(),
                    })
                } else {
                    SsaTerminator::Return { results: None }
                };
                SsaBlock {
                    id: SsaTarget(index as u32),
                    params: collections::Vec::new(),
                    ops: collections::Vec::new(),
                    extra_args: collections::Vec::new(),
                    terminator,
                }
            })
            .collect();

        SsaProgram {
            entry: SsaTarget(0),
            blocks,
            ..SsaProgram::default()
        }
    }

    #[test]
    fn reverse_postorder_handles_long_linear_cfg_without_recursion() {
        let block_count = 12_000;
        let program = long_linear_program(block_count);

        let rpo = compute_reverse_postorder(&program);

        assert_eq!(rpo.len(), block_count);
        assert_eq!(rpo.first().copied(), Some(0));
        assert_eq!(rpo.last().copied(), Some(block_count - 1));
    }

    #[test]
    fn idom_layout_walk_handles_long_linear_tree_without_recursion() {
        let block_count = 12_000;
        let program = long_linear_program(block_count);
        let mut idom_children = collections::vec![collections::Vec::new(); block_count];
        for index in 0..block_count - 1 {
            idom_children[index].push(index + 1);
        }
        let bank_cells = (0..block_count)
            .map(|_| [collections::Vec::new(), collections::Vec::new()])
            .collect::<collections::Vec<_>>();
        let cell_to_cached_index = BTreeMap::new();
        let cell_meta = collections::Vec::new();
        let param_usage = collections::vec![ParamPrefixUsage::default(); block_count];
        let mut layouts = collections::Vec::new();
        let mut exit_layouts = collections::Vec::new();
        let mut visited = collections::vec![false; block_count];

        assign_bank_layouts_from_root(
            0,
            None,
            LayoutBank::Gp,
            1,
            &program,
            &cell_to_cached_index,
            &idom_children,
            &bank_cells,
            &cell_meta,
            &param_usage,
            &[],
            &[],
            &[],
            &[],
            &mut layouts,
            &mut exit_layouts,
            0,
            &mut visited,
        )
        .expect("long linear idom tree should not overflow the native stack");

        assert!(visited.into_iter().all(|visited| visited));
    }

    #[test]
    fn inherited_call_preserved_cache_moves_to_preserved_lane() {
        let cells = collections::vec![0usize];
        let parent_layout = collections::vec![0u32];
        let cell_meta = collections::vec![CellLayoutMeta {
            bank: LayoutBank::Gp,
            width: 1,
            is_ref: false,
        }];
        let call_preserve_preferences = collections::vec![true];
        let preserved_lanes = collections::vec![false, false, true];

        let layout = build_block_bank_layout(
            &cells,
            Some(&parent_layout),
            &cell_meta,
            3,
            0,
            LayoutBank::Gp,
            &call_preserve_preferences,
            &preserved_lanes,
        )
        .expect("one-cell preserved layout should be feasible");

        assert_eq!(
            layout[0], 2,
            "a call-crossing inherited cache must move out of volatile lanes"
        );
    }
}
