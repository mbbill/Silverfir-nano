//! Optional JIT address-to-group map dumping.
//!
//! Enable by setting `SF_JIT_MAP=/path/to/map.txt` before running.
//! Each emitted group appends one tab-separated line with:
//! start-address, end-address, module, function index, IR range,
//! terminator, branch target, and a compact op summary.

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use crate::vm::interp::fast::builder::ir::{IrOp, IrOpKind, OpIndex};

#[cfg(any(feature = "wasi", feature = "std", test))]
use std::{
    env,
    fs::{File, OpenOptions},
    io::Write,
    sync::{Mutex, OnceLock},
};

#[cfg(any(feature = "wasi", feature = "std", test))]
static MAP_FILE: OnceLock<Option<Mutex<File>>> = OnceLock::new();

#[cfg(any(feature = "wasi", feature = "std", test))]
fn writer() -> Option<&'static Mutex<File>> {
    MAP_FILE
        .get_or_init(|| {
            let path = env::var("SF_JIT_MAP").ok()?;
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(path)
                .ok()?;
            if writeln!(file, "# start\tend\tmodule\tfunc\tir\tterm\ttarget\tops").is_err() {
                return None;
            }
            Some(Mutex::new(file))
        })
        .as_ref()
}

pub fn record_group(
    code_base: *const u8,
    code_start: usize,
    code_len: usize,
    module_name: &str,
    func_idx: u32,
    ir_start: usize,
    group: &[IrOp],
    terminator: Option<&'static str>,
    branch_target: Option<OpIndex>,
) {
    #[cfg(any(feature = "wasi", feature = "std", test))]
    {
        let Some(writer) = writer() else {
            return;
        };
        let line = render_line(
            code_base,
            code_start,
            code_len,
            module_name,
            func_idx,
            ir_start,
            group,
            terminator,
            branch_target,
        );
        if let Ok(mut file) = writer.lock() {
            let _ = file.write_all(line.as_bytes());
        }
    }

    #[cfg(not(any(feature = "wasi", feature = "std", test)))]
    let _ = (
        code_base,
        code_start,
        code_len,
        module_name,
        func_idx,
        ir_start,
        group,
        terminator,
        branch_target,
    );
}

fn render_line(
    code_base: *const u8,
    code_start: usize,
    code_len: usize,
    module_name: &str,
    func_idx: u32,
    ir_start: usize,
    group: &[IrOp],
    terminator: Option<&'static str>,
    branch_target: Option<OpIndex>,
) -> String {
    let start = code_base as usize + code_start;
    let end = start + code_len;
    let ir_end = ir_start + group.len();
    let target = branch_target
        .map(|idx| idx.as_usize().to_string())
        .unwrap_or_else(|| "-".to_string());

    format!(
        "0x{start:016x}\t0x{end:016x}\tmodule={}\tfunc={}\tir={}..{}\tterm={}\ttarget={}\tops={}\n",
        sanitize_token(module_name),
        func_idx,
        ir_start,
        ir_end,
        terminator.unwrap_or("-"),
        target,
        summarize_group(group),
    )
}

fn summarize_group(group: &[IrOp]) -> String {
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

fn kind_tag(kind: &IrOpKind) -> String {
    let debug = format!("{:?}", kind);
    let end = debug
        .find(|c: char| c == ' ' || c == '{')
        .unwrap_or(debug.len());
    sanitize_token(&debug[..end])
}

fn sanitize_token(raw: &str) -> String {
    raw.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' | '.' => c,
            _ => '_',
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::interp::fast::builder::ir::IrOp;

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
    fn test_render_line_includes_core_metadata() {
        let group = [
            make_op(IrOpKind::F64Mul),
            make_op(IrOpKind::F64Add),
            make_op(IrOpKind::If),
        ];
        let line = render_line(
            0x1000 as *const u8,
            0x40,
            0x18,
            "main module",
            8,
            154,
            &group,
            Some("if"),
            Some(OpIndex::from(200)),
        );

        assert!(line.contains("0x0000000000001040"));
        assert!(line.contains("0x0000000000001058"));
        assert!(line.contains("module=main_module"));
        assert!(line.contains("func=8"));
        assert!(line.contains("ir=154..157"));
        assert!(line.contains("term=if"));
        assert!(line.contains("target=200"));
        assert!(line.contains("ops=F64Mul_F64Add_If"));
    }
}
