//! Compile-time lowered-IR dump for debugging.
//!
//! Gated by `#[cfg(feature = "ir-dump")]`.
//! Filter with `SF_TRACE_FUNC=42` or `SF_TRACE_FUNC=5,12,47`.

use crate::vm::{
    debug::dump_layout as dump,
    lir::{IrOp, IrOpKind},
    plan::{hot_local, HOT_LOCAL_COUNT},
};
use alloc::{string::String, vec, vec::Vec};

fn fmt_ir(i: usize, op: &IrOp) -> alloc::string::String {
    let target_str = match op.alt_target {
        Some(t) => alloc::format!(" -> ir[{}]", t.as_usize()),
        None => alloc::string::String::new(),
    };
    alloc::format!(
        "  ir[{:4}] w={} {:?}{}",
        i, op.window, op.kind, target_str,
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

fn local_pressure_lines(
    func_idx: u32,
    code: &[u8],
    ir_ops: &[IrOp],
    frame_size: usize,
    raw_hot_locals: [Option<u32>; HOT_LOCAL_COUNT],
    hot_locals: [Option<u32>; HOT_LOCAL_COUNT],
) -> Vec<String> {
    if frame_size == 0 {
        return vec![alloc::format!("--- LOCAL PRESSURE func[{}] frame_size=0", func_idx)];
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

    let mut lines = vec![alloc::format!(
        "--- LOCAL PRESSURE func[{}] frame_size={} cached=[{}]",
        func_idx,
        frame_size,
        cached,
    )];
    lines.push(alloc::format!("    top-static: {}", top_static));
    lines.push(alloc::format!(
        "    ir-local-ops: hot={} frame={} distinct-frame-locals={}",
        hot_ops,
        frame_ops,
        frame_locals,
    ));
    if !ir_summary.is_empty() {
        lines.push(alloc::format!("    top-ir: {}", ir_summary));
    }
    if !frame_summary.is_empty() {
        lines.push(alloc::format!("    top-frame: {}", frame_summary));
    }
    lines
}

pub fn dump_ir(
    module_name: &str,
    func_idx: u32,
    code: &[u8],
    frame_size: usize,
    ir_ops: &[IrOp],
    raw_hot_locals: [Option<u32>; HOT_LOCAL_COUNT],
    hot_locals: [Option<u32>; HOT_LOCAL_COUNT],
) {
    if !dump::should_dump_func(func_idx) {
        return;
    }

    let mut lines = vec![alloc::format!(
        "=== IR func[{}] hot=[{},{},{}] {} ops ===",
        func_idx,
        fmt_opt(hot_locals[0]),
        fmt_opt(hot_locals[1]),
        fmt_opt(hot_locals[2]),
        ir_ops.len(),
    )];

    lines.extend(local_pressure_lines(
        func_idx,
        code,
        ir_ops,
        frame_size,
        raw_hot_locals,
        hot_locals,
    ));

    for (i, op) in ir_ops.iter().enumerate() {
        lines.push(fmt_ir(i, op));
    }

    let text = lines.join("\n") + "\n";
    if let Some(dir) = dump::function_dir(module_name, func_idx) {
        let path = dir.join("lowered_ir.txt");
        let _ = dump::write_text(&path, &text);
    } else {
        std::eprint!("{}", text);
    }
}

fn fmt_opt(v: Option<u32>) -> alloc::string::String {
    match v {
        Some(n) => alloc::format!("{}", n),
        None => alloc::string::String::from("-"),
    }
}
