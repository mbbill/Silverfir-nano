//! Compile-time IR dump for debugging.
//!
//! Gated by `#[cfg(feature = "ir-dump")]`. Dumps IR ops after lowering
//! and resolved instructions after backend resolution.
//!
//! Filter with `SF_TRACE_FUNC=42` or `SF_TRACE_FUNC=5,12,47` env var.
//! Unset = dump all functions.

use super::super::resolved::ResolvedInst;
use crate::vm::{lowered::{IrOp, IrOpKind}, planner::{HOT_LOCAL_COUNT, hot_local}};
use std::sync::OnceLock;
use std::collections::HashSet;
use alloc::{vec, vec::Vec};

macro_rules! dump {
    ($($arg:tt)*) => { std::eprintln!($($arg)*) }
}

fn func_filter() -> &'static Option<HashSet<u32>> {
    static FILTER: OnceLock<Option<HashSet<u32>>> = OnceLock::new();
    FILTER.get_or_init(|| {
        std::env::var("SF_TRACE_FUNC").ok().map(|val| {
            val.split(',')
                .filter_map(|s| s.trim().parse::<u32>().ok())
                .collect()
        })
    })
}

fn should_dump(func_idx: u32) -> bool {
    match func_filter() {
        None => true,
        Some(set) => set.contains(&func_idx),
    }
}

fn fmt_ir(i: usize, op: &IrOp) -> alloc::string::String {
    let target_str = match op.alt_target {
        Some(t) => alloc::format!(" -> ir[{}]", t.as_usize()),
        None => alloc::string::String::new(),
    };
    alloc::format!(
        "  ir[{:4}] D{} h={:3} {:?}{}",
        i, op.variant, op.pre_height, op.kind, target_str,
    )
}

#[derive(Clone, Copy, Default)]
struct LocalOpCounts {
    get_hot: u32,
    set_hot: u32,
    tee_hot: u32,
    get_frame: u32,
    set_frame: u32,
    tee_frame: u32,
    frame_reuse_4: u32,
    frame_reuse_8: u32,
}

impl LocalOpCounts {
    fn hot_total(self) -> u32 {
        self.get_hot + self.set_hot + self.tee_hot
    }

    fn frame_total(self) -> u32 {
        self.get_frame + self.set_frame + self.tee_frame
    }

    fn total(self) -> u32 {
        self.hot_total() + self.frame_total()
    }
}

fn remap_local(hot_locals: [Option<u32>; HOT_LOCAL_COUNT], idx: u32) -> u32 {
    let mut pos = idx;
    for (slot, k) in hot_locals.iter().enumerate() {
        if let Some(k) = *k {
            let slot = slot as u32;
            if pos == k {
                pos = slot;
            } else if pos == slot {
                pos = k;
            }
        }
    }
    pos
}

fn inverse_remap(hot_locals: [Option<u32>; HOT_LOCAL_COUNT], frame_size: usize) -> Vec<u32> {
    let mut inverse = vec![0u32; frame_size];
    for original in 0..frame_size as u32 {
        let remapped = remap_local(hot_locals, original) as usize;
        if remapped < frame_size {
            inverse[remapped] = original;
        }
    }
    inverse
}

fn dump_local_pressure(
    func_idx: u32,
    code: &[u8],
    ir_ops: &[IrOp],
    frame_size: usize,
    raw_hot_locals: [Option<u32>; HOT_LOCAL_COUNT],
    hot_locals: [Option<u32>; HOT_LOCAL_COUNT],
) {
    if frame_size == 0 {
        dump!("--- LOCAL PRESSURE func[{}] frame_size=0", func_idx);
        return;
    }

    let weights = hot_local::local_weights(code, frame_size);
    let mut ranked: Vec<(u32, u64)> = weights
        .iter()
        .enumerate()
        .filter_map(|(idx, &weight)| (weight != 0).then_some((idx as u32, weight)))
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut rank_by_local = vec![0usize; frame_size];
    for (rank, (idx, _)) in ranked.iter().enumerate() {
        rank_by_local[*idx as usize] = rank + 1;
    }

    let inverse = inverse_remap(hot_locals, frame_size);
    let mut counts = vec![LocalOpCounts::default(); frame_size];
    let mut last_frame_access: Vec<Option<usize>> = vec![None; frame_size];

    let mut hot_ops = 0u32;
    let mut frame_ops = 0u32;
    let mut frame_locals = 0usize;

    for (ir_idx, op) in ir_ops.iter().enumerate() {
        match op.kind {
            IrOpKind::LocalGetHot { reg } => {
                if let Some(orig) = raw_hot_locals.get(reg as usize).copied().flatten() {
                    counts[orig as usize].get_hot += 1;
                    hot_ops += 1;
                }
            }
            IrOpKind::LocalSetHot { reg } => {
                if let Some(orig) = raw_hot_locals.get(reg as usize).copied().flatten() {
                    counts[orig as usize].set_hot += 1;
                    hot_ops += 1;
                }
            }
            IrOpKind::LocalTeeHot { reg } => {
                if let Some(orig) = raw_hot_locals.get(reg as usize).copied().flatten() {
                    counts[orig as usize].tee_hot += 1;
                    hot_ops += 1;
                }
            }
            IrOpKind::LocalGetFrame { idx } => {
                let orig = inverse[idx as usize] as usize;
                counts[orig].get_frame += 1;
                frame_ops += 1;
                if let Some(prev) = last_frame_access[orig] {
                    let distance = ir_idx - prev;
                    if distance <= 4 {
                        counts[orig].frame_reuse_4 += 1;
                    }
                    if distance <= 8 {
                        counts[orig].frame_reuse_8 += 1;
                    }
                } else {
                    frame_locals += 1;
                }
                last_frame_access[orig] = Some(ir_idx);
            }
            IrOpKind::LocalSetFrame { idx } => {
                let orig = inverse[idx as usize] as usize;
                counts[orig].set_frame += 1;
                frame_ops += 1;
                if let Some(prev) = last_frame_access[orig] {
                    let distance = ir_idx - prev;
                    if distance <= 4 {
                        counts[orig].frame_reuse_4 += 1;
                    }
                    if distance <= 8 {
                        counts[orig].frame_reuse_8 += 1;
                    }
                } else {
                    frame_locals += 1;
                }
                last_frame_access[orig] = Some(ir_idx);
            }
            IrOpKind::LocalTeeFrame { idx } => {
                let orig = inverse[idx as usize] as usize;
                counts[orig].tee_frame += 1;
                frame_ops += 1;
                if let Some(prev) = last_frame_access[orig] {
                    let distance = ir_idx - prev;
                    if distance <= 4 {
                        counts[orig].frame_reuse_4 += 1;
                    }
                    if distance <= 8 {
                        counts[orig].frame_reuse_8 += 1;
                    }
                } else {
                    frame_locals += 1;
                }
                last_frame_access[orig] = Some(ir_idx);
            }
            _ => {}
        }
    }

    let cached = raw_hot_locals
        .iter()
        .enumerate()
        .filter_map(|(slot, orig)| {
            orig.map(|orig| {
                alloc::format!(
                    "l{}=local[{}]/rank{}->slot{}",
                    slot,
                    orig,
                    rank_by_local[orig as usize],
                    slot
                )
            })
        })
        .collect::<Vec<_>>()
        .join(", ");

    let top_static = ranked
        .iter()
        .take(8)
        .map(|(orig, weight)| {
            let cached = raw_hot_locals.contains(&Some(*orig));
            alloc::format!(
                "local[{}] rank{} w={} {}",
                orig,
                rank_by_local[*orig as usize],
                weight,
                if cached { "cached" } else { "frame" }
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");

    let mut frame_ranked: Vec<(u32, LocalOpCounts)> = counts
        .iter()
        .enumerate()
        .filter_map(|(orig, counts)| (counts.frame_total() != 0).then_some((orig as u32, *counts)))
        .collect();
    frame_ranked.sort_by(|a, b| {
        b.1.frame_total()
            .cmp(&a.1.frame_total())
            .then_with(|| a.0.cmp(&b.0))
    });
    let frame_summary = frame_ranked
        .iter()
        .take(8)
        .map(|(orig, counts)| {
            alloc::format!(
                "local[{}] rank{} remap={} frame={} (g{} s{} t{}) reuse<=4:{} reuse<=8:{}",
                orig,
                rank_by_local[*orig as usize],
                remap_local(hot_locals, *orig),
                counts.frame_total(),
                counts.get_frame,
                counts.set_frame,
                counts.tee_frame,
                counts.frame_reuse_4,
                counts.frame_reuse_8,
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");

    let mut overall_ranked: Vec<(u32, LocalOpCounts)> = counts
        .iter()
        .enumerate()
        .filter_map(|(orig, counts)| (counts.total() != 0).then_some((orig as u32, *counts)))
        .collect();
    overall_ranked.sort_by(|a, b| {
        b.1.total()
            .cmp(&a.1.total())
            .then_with(|| a.0.cmp(&b.0))
    });
    let ir_summary = overall_ranked
        .iter()
        .take(8)
        .map(|(orig, counts)| {
            alloc::format!(
                "local[{}] rank{} ir={} (hot{} frame{})",
                orig,
                rank_by_local[*orig as usize],
                counts.total(),
                counts.hot_total(),
                counts.frame_total(),
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");

    dump!(
        "--- LOCAL PRESSURE func[{}] frame_size={} cached=[{}]",
        func_idx,
        frame_size,
        cached,
    );
    dump!("    top-static: {}", top_static);
    dump!(
        "    ir-local-ops: hot={} frame={} distinct-frame-locals={}",
        hot_ops,
        frame_ops,
        frame_locals,
    );
    if !ir_summary.is_empty() {
        dump!("    top-ir: {}", ir_summary);
    }
    if !frame_summary.is_empty() {
        dump!("    top-frame: {}", frame_summary);
    }
}

/// Dump IR ops after lowering.
pub fn dump_ir(
    func_idx: u32,
    code: &[u8],
    frame_size: usize,
    ir_ops: &[IrOp],
    raw_hot_locals: [Option<u32>; HOT_LOCAL_COUNT],
    hot_locals: [Option<u32>; HOT_LOCAL_COUNT],
) {
    if !should_dump(func_idx) {
        return;
    }

    dump!(
        "=== IR func[{}] hot=[{},{},{}] {} ops ===",
        func_idx,
        fmt_opt(hot_locals[0]),
        fmt_opt(hot_locals[1]),
        fmt_opt(hot_locals[2]),
        ir_ops.len(),
    );

    dump_local_pressure(func_idx, code, ir_ops, frame_size, raw_hot_locals, hot_locals);

    for (i, op) in ir_ops.iter().enumerate() {
        dump!("{}", fmt_ir(i, op));
    }
}

/// Dump resolved instructions after backend resolution.
///
/// Cross-references with original IR ops to annotate JIT groups.
pub fn dump_resolved(func_idx: u32, resolved: &[ResolvedInst], ir_ops: &[IrOp]) {
    if !should_dump(func_idx) {
        return;
    }

    dump!("=== RESOLVED func[{}] {} insts ===", func_idx, resolved.len());

    let mut i = 0;
    while i < resolved.len() {
        let inst = &resolved[i];

        if !inst.is_removed() && matches!(inst.kind, IrOpKind::Data { imm0: 0, imm1: 0, imm2: 0 }) {
            // JIT group: find extent
            let start = i;
            let mut end = i + 1;
            while end < resolved.len() && resolved[end].is_internal_only()
                && matches!(resolved[end].kind, IrOpKind::Nop) {
                end += 1;
            }
            dump!("  [JIT group start]");
            for idx in start..end.min(ir_ops.len()) {
                dump!("{}", fmt_ir(idx, &ir_ops[idx]));
            }
            dump!("  [JIT group end]");
            i = end;
        } else if inst.is_removed() {
            match &inst.kind {
                IrOpKind::Block | IrOpKind::Loop | IrOpKind::End => {
                    dump!("  res[{:4}] ({:?})", i, inst.kind);
                }
                _ => {} // skip structural Nops outside JIT groups
            }
            i += 1;
        } else {
            let target_str = match inst.alt_target {
                Some(t) => alloc::format!(" -> ir[{}]", t.as_usize()),
                None => alloc::string::String::new(),
            };
            dump!("  res[{:4}] {:?}{}", i, inst.kind, target_str);
            i += 1;
        }
    }
}

fn fmt_opt(v: Option<u32>) -> alloc::string::String {
    match v {
        Some(n) => alloc::format!("{}", n),
        None => alloc::string::String::from("-"),
    }
}
