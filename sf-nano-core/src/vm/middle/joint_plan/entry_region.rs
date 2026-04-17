//! Block-local entry-region analysis.
//!
//! The planner uses cheap block-local scans to summarize:
//! - local hotness for boundary admission and later eviction
//! - entry-stack values touched before the first hard barrier
//! - whole-block transient-symbol hotness for mid-block spill choices
//!
//! Stack values and locals do not yet share one full mid-block pressure engine,
//! but block-open uses the same scoring shape for both so the boundary policy
//! stays coherent.

use crate::collections;

use crate::vm::{
    middle::{
        cfg::SemanticCfg,
        frame::FrameLayoutPlan,
        joint_plan::facts::{
            BlockEntryStackRegion, BlockLocalInfo, BlockLocalSummary, BlockStackValueInfo,
            BlockTransientRegion, EntryState, FirstAccessKind, LocalSlotScore, TransientSymbolInfo,
        },
    },
    wasm::{
        primitive_op::{self, PrimitiveOpKind},
        semantic_ir::{SemanticOpKind, SemanticProgram},
    },
};

const EARLY_USE_BONUS: i32 = 256;
const REUSE_BONUS: i32 = 64;
const WRITE_FIRST_BOUNDARY_BONUS: i32 = 48;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EntrySymbol {
    Entry(u16),
    Temp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockSymbol {
    Entry(u16),
    Temp(u16),
}

pub(crate) fn analyze_block_entry_regions(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    frame: FrameLayoutPlan,
    entry_states: &[EntryState],
) -> (
    collections::Vec<BlockLocalSummary>,
    collections::Vec<BlockEntryStackRegion>,
) {
    let mut local_summaries = collections::Vec::with_capacity(cfg.blocks.len());
    let mut stack_regions = collections::Vec::with_capacity(cfg.blocks.len());
    let mut scratch = collections::vec![None::<BlockLocalInfo>; semantic.local_count as usize];

    for block in &cfg.blocks {
        analyze_block_locals(semantic, cfg, frame, block.range.clone(), &mut scratch);
        local_summaries.push(summarize_block_locals(&scratch));
        for entry in scratch.iter_mut() {
            *entry = None;
        }

        let entry = &entry_states[block.range.start];
        stack_regions.push(analyze_entry_stack_region(
            block.range.clone(),
            semantic,
            entry,
        ));
    }

    (local_summaries, stack_regions)
}

/// Populate the scratch buffer with block-local analysis for one block.
fn analyze_block_locals(
    semantic: &SemanticProgram,
    _cfg: &SemanticCfg,
    frame: FrameLayoutPlan,
    block_range: core::ops::Range<usize>,
    scratch: &mut [Option<BlockLocalInfo>],
) {
    // Pass 1: whole-block access scan.
    for (block_offset, semantic_index) in block_range.clone().enumerate() {
        let Some((local_idx, access_kind)) =
            classify_local_access(&semantic.ops[semantic_index].kind)
        else {
            continue;
        };
        let slot = frame.local_slot(local_idx);
        let info = scratch[local_idx as usize].get_or_insert_with(|| BlockLocalInfo {
            slot,
            ..Default::default()
        });
        record_local_access(info, access_kind, block_offset as u16, false);
    }

    // Pass 2: entry-region scan (up to first barrier).
    for (block_offset, semantic_index) in block_range.enumerate() {
        if clears_cache_region(&semantic.ops[semantic_index].kind) {
            break;
        }
        let Some((local_idx, access_kind)) =
            classify_local_access(&semantic.ops[semantic_index].kind)
        else {
            continue;
        };
        let slot = frame.local_slot(local_idx);
        let info = scratch[local_idx as usize].get_or_insert_with(|| BlockLocalInfo {
            slot,
            ..Default::default()
        });
        record_local_access(info, access_kind, block_offset as u16, true);
    }
}

/// Extract a compact summary from a populated scratch buffer.
fn summarize_block_locals(scratch: &[Option<BlockLocalInfo>]) -> BlockLocalSummary {
    let mut ranked = collections::Vec::new();
    let mut slot_scores = collections::Vec::new();
    for info in scratch.iter().flatten() {
        let hot_score = score_local(info);
        let entry_hot_score = score_entry_local(info);
        ranked.push((info.slot, hot_score));
        slot_scores.push(LocalSlotScore {
            slot: info.slot,
            entry_hot_score,
            entry_first_access_kind: info.entry_first_access_kind,
            used_anywhere: info.first_access_kind.is_some(),
            read_count: info.read_count,
            write_count: info.write_count,
        });
    }
    ranked.sort_by_key(|(slot, score)| (core::cmp::Reverse(*score), slot.0));
    BlockLocalSummary {
        ranked_slots: ranked.into_iter().map(|(slot, _)| slot).collect(),
        slot_scores,
    }
}

fn record_local_access(
    info: &mut BlockLocalInfo,
    access_kind: FirstAccessKind,
    block_offset: u16,
    entry_region: bool,
) {
    if entry_region {
        match access_kind {
            FirstAccessKind::ReadFirst => {
                info.entry_read_count = info.entry_read_count.saturating_add(1);
                info.entry_first_read_distance.get_or_insert(block_offset);
            }
            FirstAccessKind::WriteFirst => {
                info.entry_write_count = info.entry_write_count.saturating_add(1);
                info.entry_first_write_distance.get_or_insert(block_offset);
            }
        }
        if info.entry_first_access_kind.is_none() {
            info.entry_first_access_kind = Some(access_kind);
        }
        return;
    }

    match access_kind {
        FirstAccessKind::ReadFirst => {
            info.read_count = info.read_count.saturating_add(1);
            info.first_read_distance.get_or_insert(block_offset);
        }
        FirstAccessKind::WriteFirst => {
            info.write_count = info.write_count.saturating_add(1);
            info.first_write_distance.get_or_insert(block_offset);
        }
    }
    if info.first_access_kind.is_none() {
        info.first_access_kind = Some(access_kind);
    }
}

fn clears_cache_region(kind: &SemanticOpKind) -> bool {
    matches!(
        kind,
        SemanticOpKind::Loop { .. }
            | SemanticOpKind::If { .. }
            | SemanticOpKind::Else { .. }
            | SemanticOpKind::End
            | SemanticOpKind::Br { .. }
            | SemanticOpKind::BrIf { .. }
            | SemanticOpKind::BrOnNull { .. }
            | SemanticOpKind::BrOnNonNull { .. }
            | SemanticOpKind::BrOnCast { .. }
            | SemanticOpKind::BrOnCastFail { .. }
            | SemanticOpKind::BrTable { .. }
            | SemanticOpKind::CallDirect { .. }
            | SemanticOpKind::CallIndirect { .. }
            | SemanticOpKind::CallRef { .. }
            | SemanticOpKind::ReturnVoid
            | SemanticOpKind::ReturnOne
            | SemanticOpKind::Return { .. }
            | SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable)
    )
}

/// Summarize whole-block transient-symbol usage for each CFG block.
///
/// CFG blocks already solve stack joins for us, so this analysis can restart
/// symbol numbering at every block entry:
/// - symbols `0..entry_height` are the original entry stack values
/// - fresh block-local temps get monotonically increasing symbol ids
///
/// The builder later combines these facts with the *current* block-local
/// symbol stack to decide whether pressure should evict a cached local or spill
/// the current bottom live transient.
pub(crate) fn analyze_block_transient_regions(
    semantic: &SemanticProgram,
    cfg: &SemanticCfg,
    entry_states: &[EntryState],
) -> collections::Vec<BlockTransientRegion> {
    let mut regions = collections::Vec::with_capacity(cfg.blocks.len());

    for block in &cfg.blocks {
        let entry = &entry_states[block.range.start];
        let mut stack = (0..entry.stack_height)
            .map(BlockSymbol::Entry)
            .collect::<collections::Vec<_>>();
        let mut infos =
            collections::vec![TransientSymbolInfo::default(); entry.stack_height as usize];
        let mut next_symbol = entry.stack_height;

        for (block_offset, semantic_index) in block.range.clone().enumerate() {
            apply_transient_analysis_effect(
                &semantic.ops[semantic_index].kind,
                block_offset as u16,
                &mut stack,
                &mut infos,
                &mut next_symbol,
            );
        }

        for info in &mut infos {
            info.hot_score = score_transient_symbol(info);
        }

        regions.push(BlockTransientRegion {
            entry_stack_height: entry.stack_height,
            symbols: infos,
        });
    }

    regions
}

fn analyze_entry_stack_region(
    semantic_range: core::ops::Range<usize>,
    semantic: &SemanticProgram,
    entry: &EntryState,
) -> BlockEntryStackRegion {
    let mut values = entry
        .stack_types
        .iter()
        .copied()
        .enumerate()
        .map(|(index, ty)| BlockStackValueInfo {
            stack_index: index as u16,
            ty,
            touched_before_barrier: false,
            first_touch_distance: None,
            touch_count: 0,
            hot_score: 0,
        })
        .collect::<collections::Vec<_>>();

    let mut stack = (0..entry.stack_height)
        .map(EntrySymbol::Entry)
        .collect::<collections::Vec<_>>();

    for (block_offset, semantic_index) in semantic_range.enumerate() {
        let kind = &semantic.ops[semantic_index].kind;
        if clears_cache_region(kind) {
            break;
        }

        match kind {
            SemanticOpKind::Primitive(primitive) => {
                let (pop, push) = primitive_op::stack_effect(primitive);
                if matches!(primitive, primitive_op::PrimitiveOpKind::Drop) {
                    // A pure drop is not a reason to keep an entry-stack value
                    // resident. The value is dead, and spilling it is just as
                    // good or better than carrying it hot to the drop site.
                    for _ in 0..pop {
                        let _ = pop_one_symbol(&mut stack);
                    }
                } else {
                    record_stack_touches(
                        &mut stack,
                        pop as usize,
                        block_offset as u16,
                        &mut values,
                    );
                }
                for _ in 0..push {
                    stack.push(EntrySymbol::Temp);
                }
            }
            SemanticOpKind::LocalGet { .. } => stack.push(EntrySymbol::Temp),
            SemanticOpKind::LocalSet { .. } => {
                record_stack_touches(&mut stack, 1, block_offset as u16, &mut values);
            }
            SemanticOpKind::LocalTee { .. } => {
                let symbol = pop_one_symbol(&mut stack);
                if let Some(symbol) = symbol {
                    touch_symbol(symbol, block_offset as u16, &mut values);
                    stack.push(symbol);
                }
            }
            SemanticOpKind::Block { .. } | SemanticOpKind::Loop { .. } | SemanticOpKind::End => {}
            SemanticOpKind::If { .. }
            | SemanticOpKind::Else { .. }
            | SemanticOpKind::Br { .. }
            | SemanticOpKind::BrIf { .. }
            | SemanticOpKind::BrOnNull { .. }
            | SemanticOpKind::BrOnNonNull { .. }
            | SemanticOpKind::BrOnCast { .. }
            | SemanticOpKind::BrOnCastFail { .. }
            | SemanticOpKind::BrTable { .. }
            | SemanticOpKind::CallDirect { .. }
            | SemanticOpKind::CallIndirect { .. }
            | SemanticOpKind::CallRef { .. }
            | SemanticOpKind::ReturnVoid
            | SemanticOpKind::ReturnOne
            | SemanticOpKind::Return { .. } => break,
        }
    }

    for info in &mut values {
        info.hot_score = score_stack_value(info);
    }

    BlockEntryStackRegion {
        entry_stack_height: entry.stack_height,
        values,
    }
}

#[inline]
fn classify_local_access(kind: &SemanticOpKind) -> Option<(u16, FirstAccessKind)> {
    match *kind {
        SemanticOpKind::LocalGet { idx } => Some((idx, FirstAccessKind::ReadFirst)),
        SemanticOpKind::LocalSet { idx } | SemanticOpKind::LocalTee { idx } => {
            Some((idx, FirstAccessKind::WriteFirst))
        }
        _ => None,
    }
}

fn record_stack_touches(
    stack: &mut collections::Vec<EntrySymbol>,
    pop_count: usize,
    block_offset: u16,
    values: &mut [BlockStackValueInfo],
) {
    for _ in 0..pop_count {
        if let Some(symbol) = pop_one_symbol(stack) {
            touch_symbol(symbol, block_offset, values);
        }
    }
}

#[inline]
fn pop_one_symbol(stack: &mut collections::Vec<EntrySymbol>) -> Option<EntrySymbol> {
    stack.pop()
}

fn touch_symbol(symbol: EntrySymbol, block_offset: u16, values: &mut [BlockStackValueInfo]) {
    let EntrySymbol::Entry(index) = symbol else {
        return;
    };
    let Some(info) = values.get_mut(index as usize) else {
        return;
    };
    info.touched_before_barrier = true;
    info.first_touch_distance.get_or_insert(block_offset);
    info.touch_count = info.touch_count.saturating_add(1);
}

fn score_local(info: &BlockLocalInfo) -> i32 {
    match info.first_access_kind {
        None => 0,
        Some(FirstAccessKind::ReadFirst) => {
            let first = info.first_read_distance.unwrap_or(u16::MAX) as i32;
            (EARLY_USE_BONUS / (1 + first))
                + REUSE_BONUS * i32::from(info.read_count)
                + (REUSE_BONUS / 2) * i32::from(info.write_count)
        }
        Some(FirstAccessKind::WriteFirst) => {
            let first = info.first_write_distance.unwrap_or(u16::MAX) as i32;
            (EARLY_USE_BONUS / (1 + first))
                + WRITE_FIRST_BOUNDARY_BONUS
                + REUSE_BONUS * i32::from(info.write_count)
                + (REUSE_BONUS / 2) * i32::from(info.read_count)
        }
    }
}

fn score_entry_local(info: &BlockLocalInfo) -> i32 {
    match info.entry_first_access_kind {
        None => 0,
        Some(FirstAccessKind::ReadFirst) => {
            let first = info.entry_first_read_distance.unwrap_or(u16::MAX) as i32;
            (EARLY_USE_BONUS / (1 + first))
                + REUSE_BONUS * i32::from(info.entry_read_count)
                + (REUSE_BONUS / 2) * i32::from(info.entry_write_count)
        }
        Some(FirstAccessKind::WriteFirst) => {
            let first = info.entry_first_write_distance.unwrap_or(u16::MAX) as i32;
            (EARLY_USE_BONUS / (1 + first))
                + WRITE_FIRST_BOUNDARY_BONUS
                + REUSE_BONUS * i32::from(info.entry_write_count)
                + (REUSE_BONUS / 2) * i32::from(info.entry_read_count)
        }
    }
}

fn score_stack_value(info: &BlockStackValueInfo) -> i32 {
    if !info.touched_before_barrier {
        return 0;
    }
    let first = info.first_touch_distance.unwrap_or(u16::MAX) as i32;
    (EARLY_USE_BONUS / (1 + first)) + REUSE_BONUS * i32::from(info.touch_count)
}

fn apply_transient_analysis_effect(
    kind: &SemanticOpKind,
    block_offset: u16,
    stack: &mut collections::Vec<BlockSymbol>,
    infos: &mut collections::Vec<TransientSymbolInfo>,
    next_symbol: &mut u16,
) {
    match kind {
        SemanticOpKind::Primitive(primitive) => {
            if matches!(primitive, primitive_op::PrimitiveOpKind::Unreachable) {
                return;
            }
            let (pop, push) = primitive_op::stack_effect(primitive);
            record_transient_touches(stack, pop as usize, block_offset, infos);
            push_fresh_transient_symbols(stack, infos, next_symbol, push as usize);
        }
        SemanticOpKind::LocalGet { .. } => {
            push_fresh_transient_symbols(stack, infos, next_symbol, 1);
        }
        SemanticOpKind::LocalSet { .. } => {
            record_transient_touches(stack, 1, block_offset, infos);
        }
        SemanticOpKind::LocalTee { .. } => {
            if let Some(symbol) = stack.last().copied() {
                touch_transient_symbol(symbol, block_offset, infos);
            }
        }
        SemanticOpKind::If { .. } => {
            record_transient_touches(stack, 1, block_offset, infos);
        }
        SemanticOpKind::BrIf { arity, .. } => {
            touch_transient_top(stack, arity.saturating_add(1) as usize, block_offset, infos);
        }
        SemanticOpKind::BrOnNull { arity, .. } => {
            touch_transient_top(stack, arity.saturating_add(1) as usize, block_offset, infos);
        }
        SemanticOpKind::BrOnNonNull { arity, .. } => {
            touch_transient_top(stack, *arity as usize, block_offset, infos);
        }
        SemanticOpKind::BrOnCast { arity, .. } | SemanticOpKind::BrOnCastFail { arity, .. } => {
            touch_transient_top(stack, *arity as usize, block_offset, infos);
        }
        SemanticOpKind::Br { arity, .. } => {
            touch_transient_top(stack, *arity as usize, block_offset, infos);
        }
        SemanticOpKind::BrTable { entries } => {
            let arity = entries.first().map(|entry| entry.arity).unwrap_or(0);
            touch_transient_top(stack, arity.saturating_add(1) as usize, block_offset, infos);
        }
        SemanticOpKind::CallDirect {
            params, results, ..
        } => {
            record_transient_touches(stack, *params as usize, block_offset, infos);
            push_fresh_transient_symbols(stack, infos, next_symbol, *results as usize);
        }
        SemanticOpKind::CallIndirect {
            params, results, ..
        } => {
            record_transient_touches(
                stack,
                params.saturating_add(1) as usize,
                block_offset,
                infos,
            );
            push_fresh_transient_symbols(stack, infos, next_symbol, *results as usize);
        }
        SemanticOpKind::CallRef {
            params, results, ..
        } => {
            record_transient_touches(
                stack,
                params.saturating_add(1) as usize,
                block_offset,
                infos,
            );
            push_fresh_transient_symbols(stack, infos, next_symbol, *results as usize);
        }
        SemanticOpKind::ReturnVoid => {}
        SemanticOpKind::ReturnOne => {
            touch_transient_top(stack, 1, block_offset, infos);
        }
        SemanticOpKind::Return { arity } => {
            touch_transient_top(stack, *arity as usize, block_offset, infos);
        }
        // CFG block boundaries already isolate real control-flow joins, so
        // these structural markers do not need special local transient logic.
        SemanticOpKind::Block { .. }
        | SemanticOpKind::Loop { .. }
        | SemanticOpKind::Else { .. }
        | SemanticOpKind::End => {}
    }
}

fn record_transient_touches(
    stack: &mut collections::Vec<BlockSymbol>,
    pop_count: usize,
    block_offset: u16,
    infos: &mut [TransientSymbolInfo],
) {
    for _ in 0..pop_count {
        if let Some(symbol) = stack.pop() {
            touch_transient_symbol(symbol, block_offset, infos);
        }
    }
}

fn touch_transient_top(
    stack: &[BlockSymbol],
    count: usize,
    block_offset: u16,
    infos: &mut [TransientSymbolInfo],
) {
    let start = stack.len().saturating_sub(count);
    for &symbol in &stack[start..] {
        touch_transient_symbol(symbol, block_offset, infos);
    }
}

fn push_fresh_transient_symbols(
    stack: &mut collections::Vec<BlockSymbol>,
    infos: &mut collections::Vec<TransientSymbolInfo>,
    next_symbol: &mut u16,
    count: usize,
) {
    for _ in 0..count {
        let symbol = *next_symbol;
        *next_symbol = next_symbol.saturating_add(1);
        if infos.len() <= symbol as usize {
            infos.resize(symbol as usize + 1, TransientSymbolInfo::default());
        }
        stack.push(BlockSymbol::Temp(symbol));
    }
}

fn touch_transient_symbol(
    symbol: BlockSymbol,
    block_offset: u16,
    infos: &mut [TransientSymbolInfo],
) {
    let index = match symbol {
        BlockSymbol::Entry(index) | BlockSymbol::Temp(index) => index,
    };
    let Some(info) = infos.get_mut(index as usize) else {
        return;
    };
    info.first_touch_distance.get_or_insert(block_offset);
    info.touch_count = info.touch_count.saturating_add(1);
}

fn score_transient_symbol(info: &TransientSymbolInfo) -> i32 {
    if info.touch_count == 0 {
        return 0;
    }
    let first = info.first_touch_distance.unwrap_or(u16::MAX) as i32;
    (EARLY_USE_BONUS / (1 + first)) + REUSE_BONUS * i32::from(info.touch_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        value_type::ValueType,
        vm::{
            middle::joint_plan::facts::EntryState,
            wasm::{
                primitive_op::PrimitiveOpKind,
                semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
            },
        },
    };

    #[test]
    fn stack_region_tracks_only_values_touched_before_first_barrier() {
        let semantic = SemanticProgram {
            ops: collections::vec![
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Add),
                },
                SemanticOp {
                    kind: SemanticOpKind::CallDirect {
                        callee: 0,
                        params: 0,
                        results: 0,
                    },
                },
            ],
            ..Default::default()
        };
        let entry = EntryState {
            stack_height: 2,
            spill_depth: 0,
            stack_types: collections::vec![ValueType::I32, ValueType::I32],
            live_types: collections::vec![ValueType::I32, ValueType::I32],
        };

        let region = analyze_entry_stack_region(0..semantic.ops.len(), &semantic, &entry);
        assert_eq!(region.entry_stack_height, 2);
        assert!(!region.values[0].touched_before_barrier);
        assert!(region.values[1].touched_before_barrier);
        assert_eq!(region.values[1].touch_count, 1);
    }

    #[test]
    fn local_tee_keeps_the_same_entry_symbol_live() {
        let semantic = SemanticProgram {
            ops: collections::vec![
                SemanticOp {
                    kind: SemanticOpKind::LocalTee { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalTee { idx: 0 },
                },
            ],
            ..Default::default()
        };
        let entry = EntryState {
            stack_height: 1,
            spill_depth: 0,
            stack_types: collections::vec![ValueType::I32],
            live_types: collections::vec![ValueType::I32],
        };

        let region = analyze_entry_stack_region(0..semantic.ops.len(), &semantic, &entry);
        assert_eq!(region.values[0].touch_count, 2);
    }

    #[test]
    fn stack_region_does_not_treat_pure_drop_as_hot_use() {
        let semantic = SemanticProgram {
            ops: collections::vec![SemanticOp {
                kind: SemanticOpKind::Primitive(PrimitiveOpKind::Drop),
            }],
            ..Default::default()
        };
        let entry = EntryState {
            stack_height: 1,
            spill_depth: 0,
            stack_types: collections::vec![ValueType::I32],
            live_types: collections::vec![ValueType::I32],
        };

        let region = analyze_entry_stack_region(0..semantic.ops.len(), &semantic, &entry);
        assert!(!region.values[0].touched_before_barrier);
        assert_eq!(region.values[0].touch_count, 0);
        assert_eq!(region.values[0].hot_score, 0);
    }

    #[test]
    fn transient_region_tracks_entry_and_temp_symbol_uses_across_block() {
        let semantic = SemanticProgram {
            ops: collections::vec![
                SemanticOp {
                    kind: SemanticOpKind::LocalGet { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::I32Add),
                },
                SemanticOp {
                    kind: SemanticOpKind::LocalTee { idx: 0 },
                },
                SemanticOp {
                    kind: SemanticOpKind::Primitive(PrimitiveOpKind::Drop),
                },
            ],
            local_count: 1,
            local_types: collections::vec![ValueType::I32],
            ..Default::default()
        };
        let cfg = crate::vm::middle::cfg::build_semantic_cfg(&semantic);
        let entry_states = collections::vec![EntryState {
            stack_height: 1,
            spill_depth: 0,
            stack_types: collections::vec![ValueType::I32],
            live_types: collections::vec![ValueType::I32],
        }];

        let regions = analyze_block_transient_regions(&semantic, &cfg, &entry_states);
        let region = &regions[0];
        assert_eq!(region.entry_stack_height, 1);
        assert_eq!(
            region.symbols.get(0).unwrap().touch_count,
            1,
            "entry symbol is read by add"
        );
        assert_eq!(
            region.symbols.get(1).unwrap().touch_count,
            1,
            "the local.get temp is consumed by add exactly once",
        );
        assert_eq!(
            region.symbols.get(2).unwrap().touch_count,
            2,
            "add result is touched by tee and drop"
        );
    }
}
