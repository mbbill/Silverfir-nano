use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use crate::vm::{
    lir::leaf::LirLeafOp,
    native::{
        code::{NativeDebugRegion, NativeDebugRegionKind},
        ir::{
            NativeHelperEffect, NativeHelperInst, NativeInstKind, NativeProgram, NativeTerminator,
        },
    },
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

pub fn render_region_symbol_name(
    module_name: &str,
    func_idx: u32,
    program: &NativeProgram,
    region: &NativeDebugRegion,
) -> String {
    render_symbol_name(
        module_name,
        func_idx,
        &render_region_label(program, region),
    )
}

pub fn render_region_label(program: &NativeProgram, region: &NativeDebugRegion) -> String {
    match &region.kind {
        NativeDebugRegionKind::SharedEntry => "shared_entry".into(),
        NativeDebugRegionKind::PublicEntry => "public_entry".into(),
        NativeDebugRegionKind::InternalEntry => "internal_entry".into(),
        NativeDebugRegionKind::RootExit => "root_exit".into(),
        NativeDebugRegionKind::Block { block } => {
            let Some(block) = program.blocks.get(*block as usize) else {
                return alloc::format!("b{block:02}");
            };
            alloc::format!("b{:02}__{}", block.id.0, summarize_block(block))
        }
        NativeDebugRegionKind::CallLocalContinuation {
            block,
            inst_index,
            callee,
        } => alloc::format!("b{block:02}_call{inst_index}_cont_f{callee}"),
    }
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

fn summarize_block(block: &crate::vm::native::ir::NativeBlock) -> String {
    const MAX_PARTS: usize = 3;

    let mut parts = Vec::new();
    for inst in block.ops.iter().take(MAX_PARTS) {
        parts.push(inst_tag(inst));
    }
    parts.push(terminator_tag(&block.terminator));
    sanitize_token(&parts.join("_"))
}

fn inst_tag(inst: &crate::vm::native::ir::NativeInst) -> String {
    match &inst.kind {
        NativeInstKind::Move(_) => "move".into(),
        NativeInstKind::Leaf { op, .. } => leaf_tag(op),
        NativeInstKind::Helper(NativeHelperInst::Leaf { effect, op, .. }) => alloc::format!(
            "{}_{}",
            match effect {
                NativeHelperEffect::Transparent => "helper_t",
                NativeHelperEffect::ControlTransfer => "helper_ct",
            },
            leaf_tag(op)
        ),
        NativeInstKind::CallExternal { func_idx, .. } => alloc::format!("call_external_f{func_idx}"),
        NativeInstKind::CallLocal { callee, .. } => alloc::format!("call_local_f{callee}"),
        NativeInstKind::CallIndirect { type_idx, .. } => alloc::format!("call_indirect_t{type_idx}"),
    }
}

fn leaf_tag(op: &LirLeafOp) -> String {
    let debug = alloc::format!("{:?}", op);
    let end = debug
        .find(|c: char| c == ' ' || c == '{')
        .unwrap_or(debug.len());
    sanitize_token(&debug[..end]).to_ascii_lowercase()
}

fn terminator_tag(term: &NativeTerminator) -> String {
    match term {
        NativeTerminator::Goto(edge) => alloc::format!("goto_b{}", edge.target.0),
        NativeTerminator::Branch { .. } => "branch".into(),
        NativeTerminator::BrTable { .. } => "br_table".into(),
        NativeTerminator::Return { .. } => "return".into(),
        NativeTerminator::TrapUnreachable => "trap".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::{render_region_label, render_symbol_name, sanitize_token};
    use crate::vm::{
        native::{
            abi::BlockAbi,
            code::{NativeDebugRegion, NativeDebugRegionKind},
            ir::{NativeBlock, NativeBlockId, NativeProgram, NativeTerminator},
        },
        plan::group::GroupPlan,
    };

    #[test]
    fn render_symbol_name_sanitizes_tokens() {
        assert_eq!(
            render_symbol_name("main module", 8, "body path"),
            "jit::main_module::func8::body_path"
        );
    }

    #[test]
    fn render_region_label_summarizes_block_shape() {
        let program = NativeProgram {
            blocks: alloc::vec![NativeBlock {
                id: NativeBlockId(3),
                entry_abi: BlockAbi { tos_width: 0 },
                ops: alloc::vec![],
                terminator: NativeTerminator::TrapUnreachable,
            }],
            groups: GroupPlan::default(),
            ..NativeProgram::default()
        };
        let region = NativeDebugRegion {
            offset: 0,
            len: 4,
            kind: NativeDebugRegionKind::Block { block: 0 },
        };

        assert_eq!(render_region_label(&program, &region), "b03__trap");
    }

    #[test]
    fn sanitize_token_replaces_non_identifier_chars() {
        assert_eq!(sanitize_token("a b/c:d"), "a_b_c_d");
    }
}
