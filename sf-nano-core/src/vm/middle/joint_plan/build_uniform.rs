//! Uniform-boundary joint plan builder.
//!
//! Implements `STREAMING_MIDDLE.md` Milestone 1: every block has the same
//! cached-local set and the same fixed split of the dynamic register
//! budget between transient values and cache. No region tree, no
//! per-block adaptation, no benefit/Lagrangian solver. Lower codegen
//! quality than `build::build_plan` but enables block-by-block
//! downstream streaming because no edge needs cache repair.
//!
//! Bring-up policy (per the doc):
//! - cache only function params, by index, until cache capacity is
//!   filled (no hot-local analysis)
//! - skip non-param locals to avoid the entry-cache-vs-zero-init
//!   ordering issue
//! - reuse the existing lightweight stack simulation's `peak_gp` /
//!   `peak_fp` per-block maxima as the function operation floor

use crate::collections;

use crate::{
    error::WasmError,
    value_type::ValueType,
    vm::{
        backend::BackendConfig,
        middle::{
            budget::gp_value_budget_units,
            cfg::SemanticCfg,
            frame::{FrameLayoutPlan, FrameSlot},
        },
        wasm::semantic_ir::SemanticProgram,
    },
};

use super::{
    build::compute_lightweight_plan,
    facts::{BlockPlan, FunctionPlan},
};

/// Build a uniform-boundary `FunctionPlan`.
pub(super) fn build_uniform_plan(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    _frame: FrameLayoutPlan,
    config: BackendConfig,
) -> Result<FunctionPlan, WasmError> {
    let raw_gp_total = config.allocatable_gp_dynamic_budget();
    let raw_fp_total = config.fp_dynamic_budget;
    let gp_unit_bytes = config.gp_unit_bytes;

    // Reuse the lightweight stack simulation. Its peak_gp/peak_fp per
    // block already give us the per-block transient pressure in budget
    // units; the function-wide max is the operation floor.
    let lightweight = compute_lightweight_plan(semantic, cfg, gp_unit_bytes);

    let gp_op_floor = lightweight.peak_gp.iter().copied().max().unwrap_or(0);
    let fp_op_floor = lightweight.peak_fp.iter().copied().max().unwrap_or(0);

    // Backend floors per `STREAMING_MIDDLE.md`: cover the worst-case
    // legal Wasm op shape (notably `i64.select` on 32-bit GP needs 5
    // GP units, `i64` binary needs 4) so the streaming budget never
    // makes a legal op illegal.
    let backend_gp_floor: u8 = if gp_unit_bytes >= 8 { 3 } else { 5 };
    let backend_fp_floor: u8 = if raw_fp_total > 0 { 3 } else { 0 };

    let gp_floor = backend_gp_floor.max(saturate_to_u8(gp_op_floor));
    let fp_floor = backend_fp_floor.max(saturate_to_u8(fp_op_floor));

    // Slack: 1 unit of headroom past the bare floor so the lowering
    // has scratch for one extra value beyond the in-flight live set.
    // Tuning knob.
    const GP_SLACK: u8 = 1;
    const FP_SLACK: u8 = 1;

    // The plan keeps the FULL allocatable budget. The "cache capacity"
    // is just budget − transient_reserve; we use it only when picking
    // the cached-locals set. The downstream validator still checks
    // `live + cache <= total` which holds because cache_units ≤
    // total − max_peak.
    let gp_transient_reserve = (gp_floor.saturating_add(GP_SLACK)).min(raw_gp_total);
    let fp_transient_reserve = (fp_floor.saturating_add(FP_SLACK)).min(raw_fp_total);
    let gp_cache_units = raw_gp_total.saturating_sub(gp_transient_reserve);
    let fp_cache_units = raw_fp_total.saturating_sub(fp_transient_reserve);

    // Pick cached locals by index. Wasm locals are `params ++
    // non_param_locals`; bring-up caches only params for now (per
    // STREAMING_MIDDLE.md §5: avoids the entry-cache vs zero-init
    // ordering issue).
    let cached_locals =
        select_param_cache_slots(semantic, gp_unit_bytes, gp_cache_units, fp_cache_units);

    let blocks: collections::Vec<BlockPlan> = lightweight
        .block_entries
        .into_iter()
        .map(|entry| BlockPlan {
            entry,
            tentative_entry_cached_locals: cached_locals.clone(),
        })
        .collect();

    Ok(FunctionPlan {
        gp_unit_bytes,
        gp_dynamic_budget: raw_gp_total,
        fp_dynamic_budget: raw_fp_total,
        blocks,
        uniform_boundary: true,
    })
}

fn saturate_to_u8(x: usize) -> u8 {
    if x > u8::MAX as usize {
        u8::MAX
    } else {
        x as u8
    }
}

/// Select cached-local slots in local-index order. Caches function
/// params only (bring-up); skips non-param locals to avoid the
/// entry-cache vs zero-init ordering issue documented in
/// `STREAMING_MIDDLE.md` §5.
fn select_param_cache_slots(
    semantic: &SemanticProgram,
    gp_unit_bytes: u8,
    gp_cache_capacity: u8,
    fp_cache_capacity: u8,
) -> collections::Vec<FrameSlot> {
    let param_count = semantic.params as usize;
    if param_count == 0 || semantic.local_types.is_empty() {
        return collections::Vec::new();
    }

    let mut gp_remaining = gp_cache_capacity as usize;
    let mut fp_remaining = fp_cache_capacity as usize;
    let mut selected = collections::Vec::new();

    for (idx, ty) in semantic.local_types.iter().enumerate().take(param_count) {
        let is_fp = matches!(ty, ValueType::F32 | ValueType::F64 | ValueType::V128);
        if is_fp {
            if fp_remaining == 0 {
                continue;
            }
            fp_remaining -= 1;
            selected.push(FrameSlot(idx as u16));
        } else {
            let units = gp_value_budget_units(*ty, gp_unit_bytes);
            if gp_remaining < units {
                continue;
            }
            gp_remaining -= units;
            selected.push(FrameSlot(idx as u16));
        }
    }

    selected
}
