//! Compile-time IR dump for debugging.
//!
//! Gated by `#[cfg(feature = "ir-dump")]`. Dumps IR ops after lowering
//! and resolved instructions after backend resolution.
//!
//! Filter with `SF_TRACE_FUNC=42` or `SF_TRACE_FUNC=5,12,47` env var.
//! Unset = dump all functions.

use super::backend::ResolvedInst;
use super::ir::{IrOp, IrOpKind};
use super::stack::HOT_LOCAL_COUNT;
use std::sync::OnceLock;
use std::collections::HashSet;

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
        Some(t) => alloc::format!(" -> ir[{}]", t),
        None => alloc::string::String::new(),
    };
    alloc::format!(
        "  ir[{:4}] D{} h={:3} {:?}{}",
        i, op.variant, op.pre_height, op.kind, target_str,
    )
}

/// Dump IR ops after lowering.
pub fn dump_ir(
    func_idx: u32,
    ir_ops: &[IrOp],
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

        if !inst.structural && matches!(inst.kind, IrOpKind::Data { imm0: 0, imm1: 0, imm2: 0 }) {
            // JIT group: find extent
            let start = i;
            let mut end = i + 1;
            while end < resolved.len() && resolved[end].structural
                && matches!(resolved[end].kind, IrOpKind::Nop) {
                end += 1;
            }
            dump!("  [JIT group start]");
            for idx in start..end.min(ir_ops.len()) {
                dump!("{}", fmt_ir(idx, &ir_ops[idx]));
            }
            dump!("  [JIT group end]");
            i = end;
        } else if inst.structural {
            match &inst.kind {
                IrOpKind::Block | IrOpKind::Loop | IrOpKind::End => {
                    dump!("  res[{:4}] ({:?})", i, inst.kind);
                }
                _ => {} // skip structural Nops outside JIT groups
            }
            i += 1;
        } else {
            let target_str = match inst.alt_target {
                Some(t) => alloc::format!(" -> ir[{}]", t),
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
