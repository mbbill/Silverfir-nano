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
        backend::BackendConfig,
        middle::{
            budget::gp_value_budget_units, cell::CellId, cfg::SemanticCfg,
            joint_plan::facts::BlockCellSummary,
        },
        wasm::semantic_ir::{SemanticOpKind, SemanticProgram},
    },
};

const DEFAULT_ASSUMED_TRIP_COUNT: f64 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bank {
    Gp,
    Fp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Algorithm4Params {
    assumed_trip_count: f64,
    edge_cost_scale: f64,
    call_tax_scale: f64,
    /// Residual per-crossing cost of a preserved-nominated resident relative
    /// to a volatile one at a direct local-JIT call: the value survives in its
    /// callee-saved register, so only the safepoint publish of a dirty cache
    /// remains (no drop, no reload). 1.0 disables the class distinction.
    preserved_keep_factor: f64,
}

impl Default for Algorithm4Params {
    fn default() -> Self {
        Self {
            assumed_trip_count: DEFAULT_ASSUMED_TRIP_COUNT,
            edge_cost_scale: 1.0,
            call_tax_scale: 1.0,
            preserved_keep_factor: 0.5,
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
        // Only `algorithm4` and its parameterized form (`algorithm4:trip/
        // edge/call=…`) are supported; the named diagnostic shorthands were
        // retired — express them as `algorithm4:edge=0`, etc.
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
                "pkeep" => {
                    cfg.preserved_keep_factor = value
                        .trim()
                        .parse::<f64>()
                        .map_err(|_| WasmError::internal("invalid SF_CACHE_POLICY pkeep value"))?;
                    if !(0.0..=1.0).contains(&cfg.preserved_keep_factor) {
                        return Err(WasmError::internal("invalid SF_CACHE_POLICY pkeep value"));
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
struct CellMeta {
    bank: Bank,
    units: usize,
    is_ref: bool,
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

/// The residency solver's output: per-block resident sets plus the
/// function-scope preserved-class nomination the residency was priced with.
pub(super) struct ResidencySolution {
    pub block_residents: collections::Vec<collections::Vec<CellId>>,
    pub preferred_preserved: collections::Vec<bool>,
}

pub(super) fn solve_public_cache_sets(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    config: BackendConfig,
    block_peak_gp: &[usize],
    block_peak_fp: &[usize],
    block_local_summaries: &[BlockCellSummary],
    is_local_func: &[bool],
    policy: ResidencyPolicy,
) -> ResidencySolution {
    let gp_dynamic_budget = config.allocatable_gp_dynamic_budget();
    let fp_dynamic_budget = config.fp_dynamic_budget;
    let local_count = semantic.local_count as usize;
    if cfg.blocks.is_empty() || local_count == 0 {
        return ResidencySolution {
            block_residents: collections::vec![collections::Vec::new(); cfg.blocks.len()],
            preferred_preserved: collections::vec![false; local_count],
        };
    }

    let ResidencyPolicy::Algorithm4(policy_params) = policy;
    let mut regions = build_region_tree(semantic, cfg, &policy_params);
    let cell_meta = build_cell_meta(&semantic.local_types, local_count, config.gp_unit_bytes);
    let block_weights = compute_block_weights(&regions, &policy_params);
    let call_counts = compute_block_call_counts(semantic, cfg, is_local_func);
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
    let mut survivable_call_weight = collections::vec![0.0; region_count];
    let mut barrier_call_weight = collections::vec![0.0; region_count];

    for (block_index, region_id) in regions.owner_by_block.iter().copied().enumerate() {
        let weight = block_weights[block_index];
        for score in &block_local_summaries[block_index].slot_scores {
            let slot_index = score.slot.0 as usize;
            let access_count = f64::from(score.access_count);
            if access_count > 0.0 {
                benefit[region_id][slot_index] += weight * access_count;
            }
        }
        survivable_call_weight[region_id] +=
            weight * f64::from(call_counts.survivable[block_index]);
        barrier_call_weight[region_id] += weight * f64::from(call_counts.barrier[block_index]);
    }

    let preferred_preserved = nominate_preserved(
        &cell_meta,
        &benefit,
        &survivable_call_weight,
        config,
        &policy_params,
    );

    // Class-aware call tax: a barrier call (non-local direct, indirect, ref)
    // kills every resident, so everyone pays full freight there. A survivable
    // call kills only volatile-class residents; a preserved-nominated resident
    // rides it out in its callee-saved register and pays only the residual
    // safepoint-publish factor.
    let mut call_tax = collections::vec![collections::vec![0.0; local_count]; region_count];
    for region_id in 0..region_count {
        for (slot_index, meta) in cell_meta.iter().copied().enumerate() {
            let survivable_factor = if preferred_preserved[slot_index] {
                policy_params.preserved_keep_factor
            } else {
                1.0
            };
            call_tax[region_id][slot_index] = (barrier_call_weight[region_id]
                + survivable_call_weight[region_id] * survivable_factor)
                * meta.units as f64;
        }
    }

    let mut selected = collections::vec![collections::vec![false; local_count]; region_count];
    solve_bank(
        Bank::Gp,
        &regions,
        &cell_meta,
        &benefit,
        &call_tax,
        &policy_params,
        &mut selected,
    );
    solve_bank(
        Bank::Fp,
        &regions,
        &cell_meta,
        &benefit,
        &call_tax,
        &policy_params,
        &mut selected,
    );

    let block_residents = cfg
        .blocks
        .iter()
        .enumerate()
        .map(|(block_index, _)| {
            let owner = regions.owner_by_block[block_index];
            let mut slots = collections::Vec::new();
            for (slot_index, is_selected) in selected[owner].iter().copied().enumerate() {
                if is_selected {
                    slots.push(CellId(slot_index as u16));
                }
            }
            slots
        })
        .collect();
    ResidencySolution {
        block_residents,
        preferred_preserved,
    }
}

/// Function-scope preserved-class nomination.
///
/// A local is nominated when the trip-weighted tax relief it would earn at
/// survivable calls (summed over the regions where it has any access benefit,
/// i.e. everywhere it could plausibly be resident) amortizes the per-lane
/// save/restore overhead a callee-saved register costs the function. Winners
/// are chosen by descending relief and clamped to the bank's preserved-lane
/// capacity in budget units, slot index as the deterministic tiebreaker.
/// Reference-typed locals are never nominated: they must be frame-visible
/// across every call boundary for GC-root visibility.
fn nominate_preserved(
    cell_meta: &[CellMeta],
    benefit: &[collections::Vec<f64>],
    survivable_call_weight: &[f64],
    config: BackendConfig,
    params: &Algorithm4Params,
) -> collections::Vec<bool> {
    let mut preferred = collections::vec![false; cell_meta.len()];
    let relief_factor = 1.0 - params.preserved_keep_factor;
    if relief_factor <= 0.0 {
        return preferred;
    }
    for bank in [Bank::Gp, Bank::Fp] {
        let capacity = usize::from(match bank {
            Bank::Gp => config.gp_preserved_dynamic,
            Bank::Fp => config.fp_preserved_dynamic,
        });
        if capacity == 0 {
            continue;
        }
        let overhead = f64::from(config.preserved_lane_save_overhead);
        let mut candidates: collections::Vec<(f64, usize)> = collections::Vec::new();
        for (slot_index, meta) in cell_meta.iter().copied().enumerate() {
            if meta.bank != bank || meta.is_ref {
                continue;
            }
            let mut relief = 0.0;
            for (region_id, weight) in survivable_call_weight.iter().copied().enumerate() {
                if weight > 0.0 && benefit[region_id][slot_index] > 0.0 {
                    relief += weight * relief_factor * meta.units as f64;
                }
            }
            if relief > overhead * meta.units as f64 {
                candidates.push((relief, slot_index));
            }
        }
        candidates.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        let mut used = 0usize;
        for (_, slot_index) in candidates {
            let units = cell_meta[slot_index].units;
            if used + units > capacity {
                continue;
            }
            preferred[slot_index] = true;
            used += units;
            if used == capacity {
                break;
            }
        }
    }
    preferred
}

fn solve_bank(
    bank: Bank,
    regions: &RegionTree,
    cell_meta: &[CellMeta],
    benefit: &[collections::Vec<f64>],
    call_tax: &[collections::Vec<f64>],
    params: &Algorithm4Params,
    selected: &mut [collections::Vec<bool>],
) {
    let slots = cell_meta
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
    let mut slot_dps = slots
        .iter()
        .map(|_| SlotDp::new(regions.nodes.len()))
        .collect::<collections::Vec<_>>();

    for (slot_pos, &slot_index) in slots.iter().enumerate() {
        compute_slot_dp(
            slot_index,
            cell_meta[slot_index],
            regions,
            benefit,
            call_tax,
            params,
            &mut slot_dps[slot_pos],
        );
    }

    let parent_selection = collections::vec![false; cell_meta.len()];
    extract_feasible_states(
        0,
        &parent_selection,
        &slots,
        regions,
        cell_meta,
        &capacities,
        &slot_dps,
        selected,
    );
}

fn compute_slot_dp(
    slot_index: usize,
    meta: CellMeta,
    regions: &RegionTree,
    benefit: &[collections::Vec<f64>],
    call_tax: &[collections::Vec<f64>],
    params: &Algorithm4Params,
    dp: &mut SlotDp,
) {
    let units = meta.units as f64;
    for region_id in (0..regions.nodes.len()).rev() {
        let region = &regions.nodes[region_id];
        let reward = benefit[region_id][slot_index] * params.benefit_scale()
            - call_tax[region_id][slot_index] * params.call_tax_scale;

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

fn extract_feasible_states(
    region_id: usize,
    parent_selection: &[bool],
    bank_slots: &[usize],
    regions: &RegionTree,
    cell_meta: &[CellMeta],
    capacities: &[usize],
    slot_dps: &[SlotDp],
    selected: &mut [collections::Vec<bool>],
) {
    let cap = capacities[region_id];
    let neg_inf = f64::NEG_INFINITY;
    let mut values = collections::vec![collections::vec![neg_inf; cap + 1]; bank_slots.len() + 1];
    let mut take = collections::vec![collections::vec![false; cap + 1]; bank_slots.len() + 1];

    values[0][0] = 0.0;

    // Knapsack ties break toward the earliest item (`candidate > best` is
    // strict), and a slot whose benefit sits deep in the subtree has the same
    // resident-vs-absent gain here as a weaker sibling (the deep benefit
    // appears in both branches). Order items by descending resident-subtree
    // potential so a tie commits continuity to the slot with the most to lose
    // downstream, then by slot index for determinism.
    let parent_state_for = |slot_index: usize| {
        if region_id == 0 {
            0
        } else {
            usize::from(parent_selection[slot_index])
        }
    };
    let mut order = (0..bank_slots.len()).collect::<collections::Vec<_>>();
    order.sort_by(|&a, &b| {
        let pot_a = slot_dps[a].force_value[region_id][parent_state_for(bank_slots[a])][1];
        let pot_b = slot_dps[b].force_value[region_id][parent_state_for(bank_slots[b])][1];
        pot_b
            .partial_cmp(&pot_a)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(bank_slots[a].cmp(&bank_slots[b]))
    });

    for (row, &item_index) in order.iter().enumerate() {
        let slot_index = bank_slots[item_index];
        let weight = cell_meta[slot_index].units;
        let parent_state = parent_state_for(slot_index);
        let absent = slot_dps[item_index].force_value[region_id][parent_state][0];
        let resident = slot_dps[item_index].force_value[region_id][parent_state][1];

        for used in 0..=cap {
            let mut best = values[row][used] + absent;
            let mut choose_resident = false;
            if used >= weight {
                let candidate = values[row][used - weight] + resident;
                if candidate > best {
                    best = candidate;
                    choose_resident = true;
                }
            }
            values[row + 1][used] = best;
            take[row + 1][used] = choose_resident;
        }
    }

    let mut used = 0usize;
    for candidate in 1..=cap {
        if values[bank_slots.len()][candidate] > values[bank_slots.len()][used] {
            used = candidate;
        }
    }

    for row in (0..order.len()).rev() {
        let slot_index = bank_slots[order[row]];
        let chose_resident = take[row + 1][used];
        selected[region_id][slot_index] = chose_resident;
        if chose_resident {
            used = used.saturating_sub(cell_meta[slot_index].units);
        }
    }

    for &child in &regions.nodes[region_id].children {
        let parent_snapshot = selected[region_id].clone();
        extract_feasible_states(
            child,
            &parent_snapshot,
            bank_slots,
            regions,
            cell_meta,
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

fn build_cell_meta(
    local_types: &[ValueType],
    local_count: usize,
    gp_unit_bytes: u8,
) -> collections::Vec<CellMeta> {
    (0..local_count)
        .map(|slot_index| {
            let ty = local_types
                .get(slot_index)
                .copied()
                .unwrap_or(ValueType::I64);
            if ty.is_fp() {
                CellMeta {
                    bank: Bank::Fp,
                    units: 1,
                    is_ref: false,
                }
            } else {
                CellMeta {
                    bank: Bank::Gp,
                    units: gp_value_budget_units(ty, gp_unit_bytes),
                    is_ref: ty.is_ref(),
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

/// Per-block call counts split by cache-survival class: `survivable` counts
/// direct calls to local JIT functions (the only calls the machine can carry a
/// preserved-lane cache across), `barrier` counts everything else (non-local
/// direct, indirect, ref) — those still kill every cached resident.
struct BlockCallCounts {
    survivable: collections::Vec<u32>,
    barrier: collections::Vec<u32>,
}

fn compute_block_call_counts(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    is_local_func: &[bool],
) -> BlockCallCounts {
    let mut survivable = collections::vec![0u32; cfg.blocks.len()];
    let mut barrier = collections::vec![0u32; cfg.blocks.len()];
    for (block_index, block) in cfg.blocks.iter().enumerate() {
        for semantic_index in block.range.clone() {
            match semantic.ops[semantic_index].kind {
                SemanticOpKind::CallDirect { callee, .. }
                    if is_local_func.get(callee as usize).copied().unwrap_or(false) =>
                {
                    survivable[block_index] += 1;
                }
                SemanticOpKind::CallDirect { .. }
                | SemanticOpKind::CallIndirect { .. }
                | SemanticOpKind::CallRef { .. } => {
                    barrier[block_index] += 1;
                }
                _ => {}
            }
        }
    }
    BlockCallCounts {
        survivable,
        barrier,
    }
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
        let cell_meta = collections::vec![
            CellMeta {
                bank: Bank::Gp,
                units: 1,
                is_ref: false,
            },
            CellMeta {
                bank: Bank::Gp,
                units: 1,
                is_ref: false,
            },
            CellMeta {
                bank: Bank::Gp,
                units: 1,
                is_ref: false,
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
            &cell_meta,
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

    fn nomination_config(gp_preserved: u8, overhead: u8) -> BackendConfig {
        BackendConfig::with_volatility(8, 8, gp_preserved, 1, 8, 0, 4, 4, false, 3)
            .with_preserved_lane_save_overhead(overhead)
    }

    #[test]
    fn nomination_requires_relief_to_amortize_the_lane_overhead() {
        let meta = collections::vec![
            CellMeta {
                bank: Bank::Gp,
                units: 1,
                is_ref: false,
            };
            2
        ];
        // One region; slot0 has benefit there, slot1 does not.
        let benefit = collections::vec![collections::vec![10.0, 0.0]];
        let params = Algorithm4Params::default();
        // Relief = weight * (1 - pkeep) = 8 * 0.5 = 4 > overhead 3: nominated.
        let nominated =
            nominate_preserved(&meta, &benefit, &[8.0], nomination_config(6, 3), &params);
        assert_eq!(nominated, collections::vec![true, false]);
        // Relief 4 <= overhead 4: not nominated.
        let nominated =
            nominate_preserved(&meta, &benefit, &[8.0], nomination_config(6, 4), &params);
        assert_eq!(nominated, collections::vec![false, false]);
        // No preserved capacity: the class is inert.
        let nominated =
            nominate_preserved(&meta, &benefit, &[8.0], nomination_config(0, 0), &params);
        assert_eq!(nominated, collections::vec![false, false]);
    }

    #[test]
    fn nomination_clamps_to_preserved_capacity_by_descending_relief() {
        let meta = collections::vec![
            CellMeta {
                bank: Bank::Gp,
                units: 1,
                is_ref: false,
            };
            3
        ];
        // Two regions with survivable calls; slot1 benefits in both (double
        // relief), slot0 and slot2 in one each.
        let benefit = collections::vec![
            collections::vec![10.0, 10.0, 0.0],
            collections::vec![0.0, 10.0, 10.0],
        ];
        let params = Algorithm4Params::default();
        let nominated = nominate_preserved(
            &meta,
            &benefit,
            &[8.0, 8.0],
            nomination_config(2, 3),
            &params,
        );
        // slot1 (relief 8) wins first; the remaining lane goes to slot0 by the
        // slot-index tiebreak against slot2 (both relief 4).
        assert_eq!(nominated, collections::vec![true, true, false]);
    }

    #[test]
    fn nomination_never_selects_ref_locals() {
        let meta = collections::vec![CellMeta {
            bank: Bank::Gp,
            units: 1,
            is_ref: true,
        }];
        let benefit = collections::vec![collections::vec![100.0]];
        let params = Algorithm4Params::default();
        let nominated =
            nominate_preserved(&meta, &benefit, &[64.0], nomination_config(6, 0), &params);
        assert_eq!(nominated, collections::vec![false]);
    }
}
