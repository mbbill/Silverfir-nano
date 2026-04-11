//! Cost-based public cached-local residency on the Wasm loop region tree.
//!
//! This implements the `ALGORITHM4.md` public-state solver:
//! - regions are `{ root } + one node per semantic Wasm loop`
//! - rewards are weighted frame-op savings minus a simple call tax
//! - mismatch cost is paid only when residency changes at a region boundary
//! - capacities are enforced per region/bank during the final extraction pass

use crate::collections;

use crate::{
    value_type::ValueType,
    vm::{
        middle::{
            budget::{count_live_bank_budget_units, gp_value_budget_units},
            cfg::SemanticCfg,
            frame::FrameSlot,
            joint_plan::facts::{BlockLocalRegion, OpPlan},
        },
        wasm::semantic_ir::{SemanticOpKind, SemanticProgram},
    },
};

const ASSUMED_TRIP_COUNT: f64 = 8.0;
const PRICE_ITERS: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bank {
    Gp,
    Fp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LocalMeta {
    bank: Bank,
    units: usize,
}

#[derive(Clone, Debug, Default)]
struct RegionNode {
    children: collections::Vec<usize>,
    depth: usize,
    start_index: usize,
    end_index: usize,
    owned_blocks: collections::Vec<usize>,
    entry_freq: f64,
    exit_freq: f64,
    gp_capacity: usize,
    fp_capacity: usize,
}

#[derive(Clone, Debug)]
struct RegionTree {
    nodes: collections::Vec<RegionNode>,
    owner_by_block: collections::Vec<usize>,
}

#[derive(Clone, Debug)]
struct SlotDp {
    force_value: collections::Vec<[[f64; 2]; 2]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StructuredFrame {
    Loop(usize),
    Other,
}

impl SlotDp {
    fn new(region_count: usize) -> Self {
        Self {
            force_value: collections::vec![[[0.0; 2]; 2]; region_count],
        }
    }

    #[inline]
    fn best(&self, region: usize, parent_state: usize) -> f64 {
        self.force_value[region][parent_state][0].max(self.force_value[region][parent_state][1])
    }
}

pub(super) fn solve_public_cache_sets(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    gp_unit_bytes: u8,
    gp_dynamic_budget: u8,
    fp_dynamic_budget: u8,
    op_plans: &[OpPlan],
    block_regions: &[BlockLocalRegion],
) -> collections::Vec<collections::Vec<FrameSlot>> {
    let local_count = semantic.local_count as usize;
    if cfg.blocks.is_empty() || local_count == 0 {
        return collections::vec![collections::Vec::new(); cfg.blocks.len()];
    }

    let mut regions = build_region_tree(semantic, cfg);
    let local_meta = build_local_meta(&semantic.local_types, local_count, gp_unit_bytes);
    let block_weights = compute_block_weights(&regions);
    let block_call_counts = compute_block_call_counts(semantic, cfg);
    let (peak_gp, peak_fp) = compute_block_peak_live_units(cfg, op_plans, gp_unit_bytes);

    for region in &mut regions.nodes {
        let mut gp_headroom = 0usize;
        let mut fp_headroom = 0usize;
        for &block_index in &region.owned_blocks {
            gp_headroom = gp_headroom.max(peak_gp[block_index]);
            fp_headroom = fp_headroom.max(peak_fp[block_index]);
        }
        region.gp_capacity = usize::from(gp_dynamic_budget).saturating_sub(gp_headroom);
        region.fp_capacity = usize::from(fp_dynamic_budget).saturating_sub(fp_headroom);
    }

    let region_count = regions.nodes.len();
    let mut benefit = collections::vec![collections::vec![0.0; local_count]; region_count];
    let mut call_tax = collections::vec![collections::vec![0.0; local_count]; region_count];

    for (block_index, region_id) in regions.owner_by_block.iter().copied().enumerate() {
        let weight = block_weights[block_index];
        for info in block_regions[block_index].locals.iter().flatten() {
            let slot_index = info.slot.0 as usize;
            let access_count = f64::from(info.read_count.saturating_add(info.write_count));
            if access_count > 0.0 {
                benefit[region_id][slot_index] += weight * access_count;
            }
        }

        let calls = block_call_counts[block_index];
        if calls == 0 {
            continue;
        }
        let call_weight = weight * f64::from(calls);
        for (slot_index, meta) in local_meta.iter().copied().enumerate() {
            call_tax[region_id][slot_index] += call_weight * meta.units as f64;
        }
    }

    let mut selected = collections::vec![collections::vec![false; local_count]; region_count];
    solve_bank(
        Bank::Gp,
        &regions,
        &local_meta,
        &benefit,
        &call_tax,
        &mut selected,
    );
    solve_bank(
        Bank::Fp,
        &regions,
        &local_meta,
        &benefit,
        &call_tax,
        &mut selected,
    );

    cfg.blocks
        .iter()
        .enumerate()
        .map(|(block_index, _)| {
            let owner = regions.owner_by_block[block_index];
            let mut slots = collections::Vec::new();
            for (slot_index, is_selected) in selected[owner].iter().copied().enumerate() {
                if is_selected {
                    slots.push(FrameSlot(slot_index as u16));
                }
            }
            slots
        })
        .collect()
}

fn solve_bank(
    bank: Bank,
    regions: &RegionTree,
    local_meta: &[LocalMeta],
    benefit: &[collections::Vec<f64>],
    call_tax: &[collections::Vec<f64>],
    selected: &mut [collections::Vec<bool>],
) {
    let slots = local_meta
        .iter()
        .enumerate()
        .filter_map(|(slot_index, meta)| (meta.bank == bank).then_some(slot_index))
        .collect::<collections::Vec<_>>();
    if slots.is_empty() {
        return;
    }

    let capacities = regions
        .nodes
        .iter()
        .map(|region| match bank {
            Bank::Gp => region.gp_capacity,
            Bank::Fp => region.fp_capacity,
        })
        .collect::<collections::Vec<_>>();
    let mut lambdas = collections::vec![0.0; regions.nodes.len()];
    let mut slot_dps = slots
        .iter()
        .map(|_| SlotDp::new(regions.nodes.len()))
        .collect::<collections::Vec<_>>();
    let mut slot_choice = slots
        .iter()
        .map(|_| collections::vec![false; regions.nodes.len()])
        .collect::<collections::Vec<_>>();

    for _ in 0..PRICE_ITERS {
        let mut demand = collections::vec![0usize; regions.nodes.len()];
        for (slot_pos, &slot_index) in slots.iter().enumerate() {
            compute_slot_dp(
                slot_index,
                local_meta[slot_index],
                regions,
                benefit,
                call_tax,
                &lambdas,
                &mut slot_dps[slot_pos],
            );
            extract_unconstrained_states(
                0,
                0,
                regions,
                &slot_dps[slot_pos],
                &mut slot_choice[slot_pos],
            );
            for (region_id, chosen) in slot_choice[slot_pos].iter().copied().enumerate() {
                if chosen {
                    demand[region_id] += local_meta[slot_index].units;
                }
            }
        }

        for region_id in 0..regions.nodes.len() {
            let cap = capacities[region_id];
            let overload = demand[region_id] as isize - cap as isize;
            let step = if cap == 0 { 1.0 } else { 1.0 / cap as f64 };
            lambdas[region_id] = (lambdas[region_id] + step * overload as f64).max(0.0);
        }
    }

    let zero_lambdas = collections::vec![0.0; regions.nodes.len()];
    for (slot_pos, &slot_index) in slots.iter().enumerate() {
        compute_slot_dp(
            slot_index,
            local_meta[slot_index],
            regions,
            benefit,
            call_tax,
            &zero_lambdas,
            &mut slot_dps[slot_pos],
        );
    }

    let parent_selection = collections::vec![false; local_meta.len()];
    extract_feasible_states(
        0,
        &parent_selection,
        &slots,
        regions,
        local_meta,
        &capacities,
        &slot_dps,
        selected,
    );
}

fn compute_slot_dp(
    slot_index: usize,
    meta: LocalMeta,
    regions: &RegionTree,
    benefit: &[collections::Vec<f64>],
    call_tax: &[collections::Vec<f64>],
    lambdas: &[f64],
    dp: &mut SlotDp,
) {
    let units = meta.units as f64;
    for region_id in (0..regions.nodes.len()).rev() {
        let region = &regions.nodes[region_id];
        let reward = benefit[region_id][slot_index]
            - call_tax[region_id][slot_index]
            - lambdas[region_id] * units;

        let mut child_if_absent = 0.0;
        let mut child_if_resident = 0.0;
        for &child in &region.children {
            child_if_absent += dp.best(child, 0);
            child_if_resident += dp.best(child, 1);
        }

        for parent_state in 0..=1 {
            dp.force_value[region_id][parent_state][0] =
                child_if_absent - edge_cost(regions, region_id, parent_state, 0, units);
            dp.force_value[region_id][parent_state][1] =
                reward + child_if_resident - edge_cost(regions, region_id, parent_state, 1, units);
        }
    }
}

fn extract_unconstrained_states(
    region_id: usize,
    parent_state: usize,
    regions: &RegionTree,
    dp: &SlotDp,
    out: &mut [bool],
) {
    let resident =
        dp.force_value[region_id][parent_state][1] > dp.force_value[region_id][parent_state][0];
    out[region_id] = resident;
    let child_parent = usize::from(resident);
    for &child in &regions.nodes[region_id].children {
        extract_unconstrained_states(child, child_parent, regions, dp, out);
    }
}

fn extract_feasible_states(
    region_id: usize,
    parent_selection: &[bool],
    bank_slots: &[usize],
    regions: &RegionTree,
    local_meta: &[LocalMeta],
    capacities: &[usize],
    slot_dps: &[SlotDp],
    selected: &mut [collections::Vec<bool>],
) {
    let cap = capacities[region_id];
    let neg_inf = f64::NEG_INFINITY;
    let mut values = collections::vec![collections::vec![neg_inf; cap + 1]; bank_slots.len() + 1];
    let mut take = collections::vec![collections::vec![false; cap + 1]; bank_slots.len() + 1];

    for used in 0..=cap {
        values[0][used] = 0.0;
    }

    for (item_index, &slot_index) in bank_slots.iter().enumerate() {
        let weight = local_meta[slot_index].units;
        let parent_state = if region_id == 0 {
            0
        } else {
            usize::from(parent_selection[slot_index])
        };
        let absent = slot_dps[item_index].force_value[region_id][parent_state][0];
        let resident = slot_dps[item_index].force_value[region_id][parent_state][1];

        for used in 0..=cap {
            let mut best = values[item_index][used] + absent;
            let mut choose_resident = false;
            if used >= weight {
                let candidate = values[item_index][used - weight] + resident;
                if candidate > best {
                    best = candidate;
                    choose_resident = true;
                }
            }
            values[item_index + 1][used] = best;
            take[item_index + 1][used] = choose_resident;
        }
    }

    let mut used = 0usize;
    for candidate in 1..=cap {
        if values[bank_slots.len()][candidate] > values[bank_slots.len()][used] {
            used = candidate;
        }
    }

    for item_index in (0..bank_slots.len()).rev() {
        let slot_index = bank_slots[item_index];
        let chose_resident = take[item_index + 1][used];
        selected[region_id][slot_index] = chose_resident;
        if chose_resident {
            used = used.saturating_sub(local_meta[slot_index].units);
        }
    }

    for &child in &regions.nodes[region_id].children {
        let parent_snapshot = selected[region_id].clone();
        extract_feasible_states(
            child,
            &parent_snapshot,
            bank_slots,
            regions,
            local_meta,
            capacities,
            slot_dps,
            selected,
        );
    }
}

fn edge_cost(
    regions: &RegionTree,
    region_id: usize,
    parent_state: usize,
    state: usize,
    units: f64,
) -> f64 {
    let region = &regions.nodes[region_id];
    if parent_state == state {
        return 0.0;
    }
    if region_id == 0 {
        if state == 1 {
            region.entry_freq * units
        } else {
            0.0
        }
    } else {
        (region.entry_freq + region.exit_freq) * units
    }
}

fn build_region_tree(semantic: &SemanticProgram, cfg: &SemanticCfg) -> RegionTree {
    let mut nodes = collections::vec![RegionNode {
        depth: 0,
        start_index: 0,
        end_index: semantic.ops.len(),
        entry_freq: 1.0,
        exit_freq: 1.0,
        ..RegionNode::default()
    }];
    let mut stack = collections::Vec::<StructuredFrame>::new();

    for (index, op) in semantic.ops.iter().enumerate() {
        match op.kind {
            SemanticOpKind::Loop { .. } => {
                let parent = current_parent_region(&stack);
                let region_id = nodes.len();
                nodes.push(RegionNode {
                    depth: nodes[parent].depth + 1,
                    start_index: index,
                    end_index: semantic.ops.len(),
                    ..RegionNode::default()
                });
                nodes[parent].children.push(region_id);
                stack.push(StructuredFrame::Loop(region_id));
            }
            SemanticOpKind::Block { .. } | SemanticOpKind::If { .. } => {
                stack.push(StructuredFrame::Other);
            }
            SemanticOpKind::End => {
                if let Some(frame) = stack.pop() {
                    if let StructuredFrame::Loop(region_id) = frame {
                        nodes[region_id].end_index = index;
                    }
                }
            }
            SemanticOpKind::Else { .. } => {}
            _ => {}
        }
    }

    let mut owner_by_block = collections::vec![0usize; cfg.blocks.len()];
    for (block_index, block) in cfg.blocks.iter().enumerate() {
        let start_index = block
            .range
            .clone()
            .find(|&semantic_index| {
                !matches!(
                    semantic.ops[semantic_index].kind,
                    SemanticOpKind::Block { .. } | SemanticOpKind::Loop { .. }
                )
            })
            .unwrap_or(block.range.start);
        let mut owner = 0usize;
        let mut best_depth = 0usize;
        for region_id in 1..nodes.len() {
            let region = &nodes[region_id];
            if start_index >= region.start_index
                && start_index < region.end_index
                && region.depth >= best_depth
            {
                owner = region_id;
                best_depth = region.depth;
            }
        }
        owner_by_block[block_index] = owner;
        nodes[owner].owned_blocks.push(block_index);
    }

    for region_id in 1..nodes.len() {
        let header_weight = block_weight(nodes[region_id].depth);
        let entry_freq = header_weight / ASSUMED_TRIP_COUNT;
        nodes[region_id].entry_freq = entry_freq;
        nodes[region_id].exit_freq = entry_freq;
    }

    RegionTree {
        nodes,
        owner_by_block,
    }
}

fn current_parent_region(stack: &[StructuredFrame]) -> usize {
    stack
        .iter()
        .rev()
        .find_map(|frame| match *frame {
            StructuredFrame::Loop(region_id) => Some(region_id),
            StructuredFrame::Other => None,
        })
        .unwrap_or(0)
}

fn build_local_meta(
    local_types: &[ValueType],
    local_count: usize,
    gp_unit_bytes: u8,
) -> collections::Vec<LocalMeta> {
    (0..local_count)
        .map(|slot_index| {
            let ty = local_types
                .get(slot_index)
                .copied()
                .unwrap_or(ValueType::I64);
            if ty.is_float() {
                LocalMeta {
                    bank: Bank::Fp,
                    units: 1,
                }
            } else {
                LocalMeta {
                    bank: Bank::Gp,
                    units: gp_value_budget_units(ty, gp_unit_bytes),
                }
            }
        })
        .collect()
}

fn compute_block_weights(regions: &RegionTree) -> collections::Vec<f64> {
    regions
        .owner_by_block
        .iter()
        .copied()
        .map(|owner| block_weight(regions.nodes[owner].depth))
        .collect()
}

fn block_weight(depth: usize) -> f64 {
    let mut weight = 1.0;
    for _ in 0..depth {
        weight *= ASSUMED_TRIP_COUNT;
    }
    weight
}

fn compute_block_call_counts(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
) -> collections::Vec<u32> {
    cfg.blocks
        .iter()
        .map(|block| {
            block
                .range
                .clone()
                .filter(|&semantic_index| {
                    matches!(
                        semantic.ops[semantic_index].kind,
                        SemanticOpKind::CallDirect { .. } | SemanticOpKind::CallIndirect { .. }
                    )
                })
                .count() as u32
        })
        .collect()
}

fn compute_block_peak_live_units(
    cfg: &SemanticCfg,
    op_plans: &[OpPlan],
    gp_unit_bytes: u8,
) -> (collections::Vec<usize>, collections::Vec<usize>) {
    let mut peak_gp = collections::vec![0usize; cfg.blocks.len()];
    let mut peak_fp = collections::vec![0usize; cfg.blocks.len()];

    for (block_index, block) in cfg.blocks.iter().enumerate() {
        for semantic_index in block.range.clone() {
            let op_plan = &op_plans[semantic_index];
            let (before_gp, before_fp) =
                count_live_bank_budget_units(&op_plan.before.live_types, gp_unit_bytes);
            peak_gp[block_index] = peak_gp[block_index].max(before_gp);
            peak_fp[block_index] = peak_fp[block_index].max(before_fp);

            let (after_gp, after_fp) =
                count_live_bank_budget_units(&op_plan.after.live_types, gp_unit_bytes);
            peak_gp[block_index] = peak_gp[block_index].max(after_gp);
            peak_fp[block_index] = peak_fp[block_index].max(after_fp);
        }
    }

    (peak_gp, peak_fp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::middle::cfg::{CfgBlock, CfgBlockFlags, CfgBlockId, CfgTerminator, SemanticCfg};
    use crate::vm::middle::joint_plan::facts::EntryState;

    #[test]
    fn block_peak_live_units_include_post_op_results() {
        let cfg = SemanticCfg {
            entry: CfgBlockId(0),
            blocks: collections::vec![CfgBlock {
                id: CfgBlockId(0),
                range: 0..1,
                preds: collections::Vec::new(),
                succs: collections::Vec::new(),
                terminator: CfgTerminator::Return { op_index: 0 },
                flags: CfgBlockFlags {
                    is_entry: true,
                    ..CfgBlockFlags::default()
                },
            }],
            semantic_to_block: collections::vec![CfgBlockId(0)],
        };
        let op_plans = collections::vec![OpPlan {
            before: EntryState::default(),
            after: EntryState {
                stack_height: 1,
                spill_depth: 0,
                stack_types: collections::vec![ValueType::I64],
                live_types: collections::vec![ValueType::I64],
            },
        }];

        let (peak_gp, peak_fp) = compute_block_peak_live_units(&cfg, &op_plans, 8);

        assert_eq!(peak_gp, collections::vec![1]);
        assert_eq!(peak_fp, collections::vec![0]);
    }
}
