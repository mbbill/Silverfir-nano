use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use crate::vm::compile::ir::{IrOp, IrOpKind, OpIndex};

pub fn render_symbol_name(
    module_name: &str,
    func_idx: u32,
    ir_start: usize,
    group: &[IrOp],
    terminator: Option<&'static str>,
    branch_target: Option<OpIndex>,
) -> String {
    let ir_end = ir_start + group.len();
    let target = branch_target
        .map(|idx| format!("target{}", idx.as_usize()))
        .unwrap_or_else(|| "target-".to_string());

    format!(
        "jit::{}::func{}::ir{}_{}::{}::{}::{}",
        sanitize_token(module_name),
        func_idx,
        ir_start,
        ir_end,
        terminator.unwrap_or("fallthrough"),
        target,
        summarize_group(group),
    )
}

pub fn summarize_group(group: &[IrOp]) -> String {
    const MAX_OPS: usize = 6;

    let mut parts = Vec::new();
    for op in group.iter().take(MAX_OPS) {
        parts.push(kind_tag(&op.kind));
    }
    if group.len() > MAX_OPS {
        parts.push(format!("plus{}", group.len() - MAX_OPS));
    }
    parts.join("_")
}

pub fn sanitize_token(raw: &str) -> String {
    raw.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' | '.' => c,
            _ => '_',
        })
        .collect()
}

fn kind_tag(kind: &IrOpKind) -> String {
    let debug = format!("{:?}", kind);
    let end = debug
        .find(|c: char| c == ' ' || c == '{')
        .unwrap_or(debug.len());
    sanitize_token(&debug[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_op(kind: IrOpKind) -> IrOp {
        IrOp {
            kind,
            variant: 0,
            pre_height: 0,
            fallthrough: None,
            alt_target: None,
            has_target: false,
        }
    }

    #[test]
    fn test_render_symbol_name_includes_core_metadata() {
        let group = [
            make_op(IrOpKind::F64Mul),
            make_op(IrOpKind::F64Add),
            make_op(IrOpKind::If),
        ];
        let name = render_symbol_name(
            "main module",
            8,
            154,
            &group,
            Some("if"),
            Some(OpIndex::from(200)),
        );

        assert_eq!(
            name,
            "jit::main_module::func8::ir154_157::if::target200::F64Mul_F64Add_If"
        );
    }
}
