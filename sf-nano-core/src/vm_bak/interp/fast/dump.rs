//! Fast-backend resolved-instruction dump for debugging.
//!
//! Gated by `#[cfg(feature = "ir-dump")]`. This stays backend-specific because
//! it dumps the fast interpreter's resolved instruction stream, not shared IR.

use super::resolved::ResolvedInst;
use crate::vm::{debug::dump_layout as dump, lir::{IrOp, IrOpKind}};
use alloc::vec;

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

pub fn dump_resolved(module_name: &str, func_idx: u32, resolved: &[ResolvedInst], ir_ops: &[IrOp]) {
    if !dump::should_dump_func(func_idx) {
        return;
    }

    let mut lines = vec![alloc::format!(
        "=== RESOLVED func[{}] {} insts ===",
        func_idx,
        resolved.len()
    )];

    let mut i = 0;
    while i < resolved.len() {
        let inst = &resolved[i];

        if !inst.is_removed() && matches!(inst.kind, IrOpKind::Data { imm0: 0, imm1: 0, imm2: 0 }) {
            let start = i;
            let mut end = i + 1;
            while end < resolved.len() && resolved[end].is_internal_only()
                && matches!(resolved[end].kind, IrOpKind::Nop)
            {
                end += 1;
            }
            lines.push("  [JIT group start]".into());
            for idx in start..end.min(ir_ops.len()) {
                lines.push(fmt_ir(idx, &ir_ops[idx]));
            }
            lines.push("  [JIT group end]".into());
            i = end;
        } else if inst.is_removed() {
            match &inst.kind {
                IrOpKind::Block | IrOpKind::Loop | IrOpKind::End => {
                    lines.push(alloc::format!("  res[{:4}] ({:?})", i, inst.kind));
                }
                _ => {}
            }
            i += 1;
        } else {
            let target_str = match inst.alt_target {
                Some(t) => alloc::format!(" -> ir[{}]", t.as_usize()),
                None => alloc::string::String::new(),
            };
            lines.push(alloc::format!("  res[{:4}] {:?}{}", i, inst.kind, target_str));
            i += 1;
        }
    }

    let text = lines.join("\n") + "\n";
    if let Some(dir) = dump::function_dir(module_name, func_idx) {
        let _ = dump::write_text(&dir.join("fast_resolved.txt"), &text);
    } else {
        std::eprint!("{}", text);
    }
}
