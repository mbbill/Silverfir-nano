use alloc::{
    format,
    string::{String, ToString},
};

use crate::vm::{
    lir::leaf::LirLeafOp,
    native::ir::{NativeHelperEffect, NativeHelperInst, NativeInstKind, NativeProgram, NativeTerminator},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeSymbolStats {
    pub blocks: u32,
    pub ops: u32,
    pub br_table: u32,
    pub call_local: u32,
    pub call_indirect: u32,
    pub call_external: u32,
    pub transparent_helpers: u32,
    pub control_transfer_helpers: u32,
    pub loads: u32,
    pub stores: u32,
}

pub fn render_symbol_name(module_name: &str, func_idx: u32, segment: &str) -> String {
    format!(
        "jit::{}::func{}::{}",
        sanitize_token(module_name),
        func_idx,
        sanitize_token(segment),
    )
}

pub fn summarize_program(program: &NativeProgram) -> NativeSymbolStats {
    let mut stats = NativeSymbolStats {
        blocks: program.blocks.len() as u32,
        ..NativeSymbolStats::default()
    };

    for block in &program.blocks {
        stats.ops += block.ops.len() as u32;

        if matches!(block.terminator, NativeTerminator::BrTable { .. }) {
            stats.br_table += 1;
        }

        for inst in &block.ops {
            match &inst.kind {
                NativeInstKind::Move(_) => {}
                NativeInstKind::Leaf { op, .. } => {
                    accumulate_leaf(&mut stats, op);
                }
                NativeInstKind::Helper(NativeHelperInst::Leaf { effect, op, .. }) => {
                    match effect {
                        NativeHelperEffect::Transparent => stats.transparent_helpers += 1,
                        NativeHelperEffect::ControlTransfer => stats.control_transfer_helpers += 1,
                    }
                    accumulate_leaf(&mut stats, op);
                }
                NativeInstKind::CallExternal { .. } => stats.call_external += 1,
                NativeInstKind::CallLocal { .. } => stats.call_local += 1,
                NativeInstKind::CallIndirect { .. } => stats.call_indirect += 1,
            }
        }
    }

    stats
}

pub fn sanitize_token(raw: &str) -> String {
    raw.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' | '.' => c,
            _ => '_',
        })
        .collect()
}

fn accumulate_leaf(stats: &mut NativeSymbolStats, op: &LirLeafOp) {
    if is_memory_load(op) {
        stats.loads += 1;
    }
    if is_memory_store(op) {
        stats.stores += 1;
    }
}

fn is_memory_load(op: &LirLeafOp) -> bool {
    matches!(
        op,
        LirLeafOp::I32Load { .. }
            | LirLeafOp::I64Load { .. }
            | LirLeafOp::F32Load { .. }
            | LirLeafOp::F64Load { .. }
            | LirLeafOp::I32Load8S { .. }
            | LirLeafOp::I32Load8U { .. }
            | LirLeafOp::I32Load16S { .. }
            | LirLeafOp::I32Load16U { .. }
            | LirLeafOp::I64Load8S { .. }
            | LirLeafOp::I64Load8U { .. }
            | LirLeafOp::I64Load16S { .. }
            | LirLeafOp::I64Load16U { .. }
            | LirLeafOp::I64Load32S { .. }
            | LirLeafOp::I64Load32U { .. }
    )
}

fn is_memory_store(op: &LirLeafOp) -> bool {
    matches!(
        op,
        LirLeafOp::I32Store { .. }
            | LirLeafOp::I64Store { .. }
            | LirLeafOp::F32Store { .. }
            | LirLeafOp::F64Store { .. }
            | LirLeafOp::I32Store8 { .. }
            | LirLeafOp::I32Store16 { .. }
            | LirLeafOp::I64Store8 { .. }
            | LirLeafOp::I64Store16 { .. }
            | LirLeafOp::I64Store32 { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::{render_symbol_name, sanitize_token};

    #[test]
    fn render_symbol_name_sanitizes_tokens() {
        assert_eq!(
            render_symbol_name("main module", 8, "body path"),
            "jit::main_module::func8::body_path"
        );
    }

    #[test]
    fn sanitize_token_replaces_non_identifier_chars() {
        assert_eq!(sanitize_token("a b/c:d"), "a_b_c_d");
    }
}
