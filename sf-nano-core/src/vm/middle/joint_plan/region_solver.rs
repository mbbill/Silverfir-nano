//! Cost-based public cached-local residency on the Wasm loop region tree.
//!
//! This implements the `ALGORITHM4.md` public-state solver:
//! - regions are `{ root } + one node per semantic Wasm loop`
//! - rewards are weighted frame-op savings minus a simple call tax
//! - mismatch cost is paid only when residency changes at a region boundary
//! - capacities are enforced per region/bank during the final extraction pass

use crate::collections;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        middle::{
            budget::gp_value_budget_units, cfg::SemanticCfg, frame::FrameSlot,
            joint_plan::facts::BlockLocalSummary,
        },
        wasm::semantic_ir::{SemanticOpKind, SemanticProgram},
    },
};

const DEFAULT_ASSUMED_TRIP_COUNT: f64 = 8.0;
const DEFAULT_PRICE_ITERS: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bank {
    Gp,
    Fp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Algorithm4Params {
    assumed_trip_count: f64,
    price_iters: usize,
    edge_cost_scale: f64,
    call_tax_scale: f64,
}

impl Default for Algorithm4Params {
    fn default() -> Self {
        Self {
            assumed_trip_count: DEFAULT_ASSUMED_TRIP_COUNT,
            price_iters: DEFAULT_PRICE_ITERS,
            edge_cost_scale: 1.0,
            call_tax_scale: 1.0,
        }
    }
}

impl Algorithm4Params {
    #[inline]
    const fn benefit_scale(&self) -> f64 {
        1.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum ResidencyPolicy {
    Algorithm4(Algorithm4Params),
    DiagnosticStatic,
    DiagnosticNone,
}

impl ResidencyPolicy {
    pub(super) fn from_env() -> Result<Self, WasmError> {
        #[cfg(any(sf_has_std, test))]
        {
            match std::env::var("SF_CACHE_POLICY") {
                Ok(spec) => Self::parse(&spec),
                Err(std::env::VarError::NotPresent) => Ok(Self::default()),
                Err(_) => Err(WasmError::internal("invalid SF_CACHE_POLICY")),
            }
        }

        #[cfg(not(any(sf_has_std, test)))]
        {
            Ok(Self::default())
        }
    }

    #[cfg(any(sf_has_std, test))]
    fn parse(spec: &str) -> Result<Self, WasmError> {
        let trimmed = spec.trim();
        if trimmed.is_empty() || trimmed == "algorithm4" {
            return Ok(Self::default());
        }
        if trimmed == "diag_static" || trimmed == "simplify_static_global" {
            return Ok(Self::DiagnosticStatic);
        }
        if trimmed == "diag_none" {
            return Ok(Self::DiagnosticNone);
        }
        if trimmed == "simplify_no_call_tax" {
            let mut cfg = Algorithm4Params::default();
            cfg.call_tax_scale = 0.0;
            return Ok(Self::Algorithm4(cfg));
        }
        if trimmed == "simplify_no_prices" {
            let mut cfg = Algorithm4Params::default();
            cfg.price_iters = 0;
            return Ok(Self::Algorithm4(cfg));
        }
        if trimmed == "simplify_no_edge_cost" {
            let mut cfg = Algorithm4Params::default();
            cfg.edge_cost_scale = 0.0;
            return Ok(Self::Algorithm4(cfg));
        }
        if trimmed == "simplify_no_loop_weight" {
            let mut cfg = Algorithm4Params::default();
            cfg.assumed_trip_count = 1.0;
            return Ok(Self::Algorithm4(cfg));
        }
        let Some(params) = trimmed.strip_prefix("algorithm4:") else {
            return Err(WasmError::internal("unknown SF_CACHE_POLICY"));
        };
        let mut cfg = Algorithm4Params::default();
        for part in params.split(',') {
            let Some((key, value)) = part.split_once('=') else {
                return Err(WasmError::internal("invalid SF_CACHE_POLICY option"));
            };
            match key.trim() {
                "trip" => {
                    cfg.assumed_trip_count = value
                        .trim()
                        .parse::<f64>()
                        .map_err(|_| WasmError::internal("invalid SF_CACHE_POLICY trip value"))?;
                    if cfg.assumed_trip_count <= 0.0 {
                        return Err(WasmError::internal("invalid SF_CACHE_POLICY trip value"));
                    }
                }
                "iters" => {
                    cfg.price_iters = value
                        .trim()
                        .parse::<usize>()
                        .map_err(|_| WasmError::internal("invalid SF_CACHE_POLICY iters value"))?;
                }
                "edge" => {
                    cfg.edge_cost_scale = value
                        .trim()
                        .parse::<f64>()
                        .map_err(|_| WasmError::internal("invalid SF_CACHE_POLICY edge value"))?;
                    if cfg.edge_cost_scale < 0.0 {
                        return Err(WasmError::internal("invalid SF_CACHE_POLICY edge value"));
                    }
                }
                "call" => {
                    cfg.call_tax_scale = value
                        .trim()
                        .parse::<f64>()
                        .map_err(|_| WasmError::internal("invalid SF_CACHE_POLICY call value"))?;
                    if cfg.call_tax_scale < 0.0 {
                        return Err(WasmError::internal("invalid SF_CACHE_POLICY call value"));
                    }
                }
                _ => return Err(WasmError::internal("unknown SF_CACHE_POLICY option")),
            }
        }
        Ok(Self::Algorithm4(cfg))
    }
}

impl Default for ResidencyPolicy {
    fn default() -> Self {
        Self::Algorithm4(Algorithm4Params::default())
    }
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
    block_peak_gp: &[usize],
    block_peak_fp: &[usize],
    block_local_summaries: &[BlockLocalSummary],
    policy: ResidencyPolicy,
) -> collections::Vec<collections::Vec<FrameSlot>> {
    let local_count = semantic.local_count as usize;
    if cfg.blocks.is_empty() || local_count == 0 {
        return collections::vec![collections::Vec::new(); cfg.blocks.len()];
    }

    let policy_params = match policy {
        ResidencyPolicy::Algorithm4(params) => params,
        ResidencyPolicy::DiagnosticStatic | ResidencyPolicy::DiagnosticNone => {
            Algorithm4Params::default()
        }
    };
    let mut regions = build_region_tree(semantic, cfg, &policy_params);
    let local_meta = build_local_meta(&semantic.local_types, local_count, gp_unit_bytes);
    let block_weights = compute_block_weights(&regions, &policy_params);
    let block_call_counts = compute_block_call_counts(semantic, cfg);
    let (peak_gp, peak_fp) = (block_peak_gp, block_peak_fp);

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
        for score in &block_local_summaries[block_index].slot_scores {
            let slot_index = score.slot.0 as usize;
            let access_count = f64::from(score.access_count);
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
    match policy {
        ResidencyPolicy::Algorithm4(params) => {
            solve_bank(
                Bank::Gp,
                &regions,
                &local_meta,
                &benefit,
                &call_tax,
                &params,
                &mut selected,
            );
            solve_bank(
                Bank::Fp,
                &regions,
                &local_meta,
                &benefit,
                &call_tax,
                &params,
                &mut selected,
            );
        }
        ResidencyPolicy::DiagnosticStatic => {
            selected = static_global_set(&regions, &local_meta, &benefit, &call_tax);
        }
        ResidencyPolicy::DiagnosticNone => {}
    }

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
    params: &Algorithm4Params,
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

    for iter in 0..params.price_iters {
        let mut demand = collections::vec![0usize; regions.nodes.len()];
        for (slot_pos, &slot_index) in slots.iter().enumerate() {
            compute_slot_dp(
                slot_index,
                local_meta[slot_index],
                regions,
                benefit,
                call_tax,
                params,
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
            let damping = 1.0 / (iter + 2) as f64;
            let step = if cap == 0 {
                damping
            } else {
                damping / cap as f64
            };
            lambdas[region_id] = (lambdas[region_id] + step * overload as f64).max(0.0);
        }
    }

    for (slot_pos, &slot_index) in slots.iter().enumerate() {
        compute_slot_dp(
            slot_index,
            local_meta[slot_index],
            regions,
            benefit,
            call_tax,
            params,
            &lambdas,
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
    params: &Algorithm4Params,
    lambdas: &[f64],
    dp: &mut SlotDp,
) {
    let units = meta.units as f64;
    for region_id in (0..regions.nodes.len()).rev() {
        let region = &regions.nodes[region_id];
        let reward = benefit[region_id][slot_index] * params.benefit_scale()
            - call_tax[region_id][slot_index] * params.call_tax_scale
            - lambdas[region_id] * units;

        let mut child_if_absent = 0.0;
        let mut child_if_resident = 0.0;
        for &child in &region.children {
            child_if_absent += dp.best(child, 0);
            child_if_resident += dp.best(child, 1);
        }

        for parent_state in 0..=1 {
            dp.force_value[region_id][parent_state][0] =
                child_if_absent - edge_cost(regions, region_id, parent_state, 0, units, params);
            dp.force_value[region_id][parent_state][1] = reward + child_if_resident
                - edge_cost(regions, region_id, parent_state, 1, units, params);
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

    values[0][0] = 0.0;

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
    params: &Algorithm4Params,
) -> f64 {
    let region = &regions.nodes[region_id];
    if parent_state == state {
        return 0.0;
    }
    if region_id == 0 {
        if state == 1 {
            region.entry_freq * units * params.edge_cost_scale
        } else {
            0.0
        }
    } else {
        (region.entry_freq + region.exit_freq) * units * params.edge_cost_scale
    }
}

fn build_region_tree(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    params: &Algorithm4Params,
) -> RegionTree {
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
        let header_weight = block_weight(nodes[region_id].depth, params);
        let entry_freq = header_weight / params.assumed_trip_count;
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
            if ty.is_fp() {
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

fn compute_block_weights(regions: &RegionTree, params: &Algorithm4Params) -> collections::Vec<f64> {
    regions
        .owner_by_block
        .iter()
        .copied()
        .map(|owner| block_weight(regions.nodes[owner].depth, params))
        .collect()
}

fn block_weight(depth: usize, params: &Algorithm4Params) -> f64 {
    let mut weight = 1.0;
    for _ in 0..depth {
        weight *= params.assumed_trip_count;
    }
    weight
}

fn static_global_set(
    regions: &RegionTree,
    local_meta: &[LocalMeta],
    benefit: &[collections::Vec<f64>],
    call_tax: &[collections::Vec<f64>],
) -> collections::Vec<collections::Vec<bool>> {
    let region_count = regions.nodes.len();
    let local_count = local_meta.len();

    let mut net = collections::vec![0.0f64; local_count];
    for region_id in 0..region_count {
        for slot in 0..local_count {
            net[slot] += benefit[region_id][slot] - call_tax[region_id][slot];
        }
    }

    let mut chosen = collections::vec![false; local_count];
    for bank in [Bank::Gp, Bank::Fp] {
        let cap = regions
            .nodes
            .iter()
            .map(|region| match bank {
                Bank::Gp => region.gp_capacity,
                Bank::Fp => region.fp_capacity,
            })
            .min()
            .unwrap_or(0);
        let items = (0..local_count)
            .filter(|&slot| local_meta[slot].bank == bank)
            .collect::<collections::Vec<_>>();
        let mut value =
            collections::vec![collections::vec![f64::NEG_INFINITY; cap + 1]; items.len() + 1];
        let mut take = collections::vec![collections::vec![false; cap + 1]; items.len() + 1];
        for used in 0..=cap {
            value[0][used] = 0.0;
        }
        for (item_index, &slot) in items.iter().enumerate() {
            let weight = local_meta[slot].units;
            for used in 0..=cap {
                let mut best = value[item_index][used];
                let mut chose = false;
                if used >= weight {
                    let candidate = value[item_index][used - weight] + net[slot];
                    if candidate > best {
                        best = candidate;
                        chose = true;
                    }
                }
                value[item_index + 1][used] = best;
                take[item_index + 1][used] = chose;
            }
        }
        let mut used = 0usize;
        for candidate in 1..=cap {
            if value[items.len()][candidate] > value[items.len()][used] {
                used = candidate;
            }
        }
        for item_index in (0..items.len()).rev() {
            let slot = items[item_index];
            if take[item_index + 1][used] {
                chosen[slot] = true;
                used = used.saturating_sub(local_meta[slot].units);
            }
        }
    }

    collections::vec![chosen; region_count]
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
                        SemanticOpKind::CallDirect { .. }
                            | SemanticOpKind::CallIndirect { .. }
                            | SemanticOpKind::CallRef { .. }
                    )
                })
                .count() as u32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feasible_extraction_backtracks_the_best_capacity_choice() {
        let regions = RegionTree {
            nodes: collections::vec![
                RegionNode {
                    children: collections::vec![1, 2],
                    entry_freq: 1.0,
                    exit_freq: 1.0,
                    gp_capacity: 1,
                    ..RegionNode::default()
                },
                RegionNode {
                    entry_freq: 1.0,
                    exit_freq: 1.0,
                    gp_capacity: 1,
                    ..RegionNode::default()
                },
                RegionNode {
                    entry_freq: 1.0,
                    exit_freq: 1.0,
                    gp_capacity: 1,
                    ..RegionNode::default()
                },
            ],
            owner_by_block: collections::Vec::new(),
        };
        let local_meta = collections::vec![
            LocalMeta {
                bank: Bank::Gp,
                units: 1,
            },
            LocalMeta {
                bank: Bank::Gp,
                units: 1,
            },
            LocalMeta {
                bank: Bank::Gp,
                units: 1,
            },
        ];
        let benefit = collections::vec![
            collections::vec![0.0, 0.0, 0.0],
            collections::vec![0.0, 0.0, 0.0],
            collections::vec![0.0, 2.0, 3.0],
        ];
        let call_tax = collections::vec![
            collections::vec![0.0, 0.0, 0.0],
            collections::vec![0.0, 0.0, 0.0],
            collections::vec![0.0, 0.0, 0.0],
        ];
        let mut selected = collections::vec![collections::vec![false; 3]; 3];

        solve_bank(
            Bank::Gp,
            &regions,
            &local_meta,
            &benefit,
            &call_tax,
            &Algorithm4Params::default(),
            &mut selected,
        );

        assert!(
            selected.iter().all(|region| region[2] && !region[1]),
            "the one-slot plan should carry the higher-value local2, not reconstruct the weaker local1; selected={selected:?}"
        );
    }
}
