//! Optional native static dump.
//!
//! Enable by setting `SF_NATIVE_DUMP_DIR=/path/to/output_dir`.
//! The dump writes exactly two files for the current module compile:
//! - `native_index.txt`: function/region metadata plus per-function IR
//! - `native_code.bin`: concatenated emitted machine code bytes

use alloc::{
    collections::BTreeMap,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::fmt::Write as _;

use crate::{
    error::WasmError,
    vm::{
        entities::ModuleInst,
        lir::ir::{LirEdge, LirInstKind, LirProgram, LirTerminator, LirValue},
        native::{
            abi::{NativeLocation, NativePlace, NativeValue},
            code::{NativeDebugRegion, NativeDebugRegionKind},
            ir::{
                NativeBranchCond, NativeCompareKind, NativeEdge, NativeHelperEffect,
                NativeHelperInst, NativeInstKind, NativeProgram, NativeTerminator,
            },
            symbols,
        },
        plan::{frame::FrameSpan, group::GroupTerminator},
    },
};

#[cfg(any(feature = "std", feature = "wasi", test))]
use std::{
    env,
    fs::{self, File},
    io::Write,
    path::PathBuf,
};

pub struct NativeDumpFunction<'a> {
    pub func_idx: usize,
    pub lir: &'a LirProgram,
}

pub fn dump_enabled() -> bool {
    #[cfg(any(feature = "std", feature = "wasi", test))]
    {
        env::var_os("SF_NATIVE_DUMP_DIR").is_some()
    }

    #[cfg(not(any(feature = "std", feature = "wasi", test)))]
    {
        false
    }
}

pub fn write_module_dump(
    module: &ModuleInst,
    lir_inputs: &[NativeDumpFunction<'_>],
) -> Result<(), WasmError> {
    #[cfg(any(feature = "std", feature = "wasi", test))]
    {
        if !dump_enabled() {
            return Ok(());
        }
        write_module_dump_impl(module, lir_inputs)
            .map_err(|err| WasmError::internal(format!("failed to write native dump: {err}")))
    }

    #[cfg(not(any(feature = "std", feature = "wasi", test)))]
    {
        let _ = (module, lir_inputs);
        Ok(())
    }
}

#[cfg(any(feature = "std", feature = "wasi", test))]
fn write_module_dump_impl(
    module: &ModuleInst,
    lir_inputs: &[NativeDumpFunction<'_>],
) -> Result<(), std::io::Error> {
    let root = env::var_os("SF_NATIVE_DUMP_DIR")
        .map(PathBuf::from)
        .expect("dump path checked by dump_enabled");
    fs::create_dir_all(&root)?;

    let mut code_file = File::create(root.join("native_code.bin"))?;
    let mut index = String::new();
    let mut region_lines = String::new();
    let mut function_sections = String::new();
    let lir_by_func: BTreeMap<usize, &LirProgram> =
        lir_inputs.iter().map(|entry| (entry.func_idx, entry.lir)).collect();

    let _ = writeln!(index, "[module]");
    let _ = writeln!(index, "name={}", module.name);
    let _ = writeln!(index, "function_count={}", module.functions.len());
    let _ = writeln!(index);
    let _ = writeln!(index, "[regions]");

    let mut code_file_offset = 0u64;
    for (func_idx, func) in module
        .functions
        .iter()
        .enumerate()
        .filter(|(_, func)| !func.is_external())
    {
        let Some(spec) = func.spec() else {
            continue;
        };
        let Some(code) = spec.get_native_code() else {
            continue;
        };

        code_file.write_all(&code.text)?;
        let file_start = code_file_offset;
        code_file_offset += code.text.len() as u64;

        let runtime_base = code
            .region()
            .zip(code.executable.as_ref())
            .map(|(region, executable)| unsafe {
                executable.base_ptr().add(region.start as usize) as usize as u64
            });
        let runtime_end = runtime_base.map(|start| start + code.text.len() as u64);
        let fallback_program = code.program();
        let stats = fallback_program.map(symbols::summarize_program);
        let debug_regions = if code.debug_regions().is_empty() && !code.text.is_empty() {
            vec![NativeDebugRegion {
                offset: 0,
                len: code.text.len() as u32,
                kind: NativeDebugRegionKind::SharedEntry,
            }]
        } else {
            code.debug_regions().to_vec()
        };

        for (region_id, region) in debug_regions.iter().enumerate() {
            let symbol = fallback_program
                .map(|program| symbols::render_region_symbol_name(&module.name, func_idx as u32, program, region))
                .unwrap_or_else(|| {
                    symbols::render_symbol_name(
                        &module.name,
                        func_idx as u32,
                        &region.summary_label(),
                    )
                });
            let (block, inst, callee) = region_detail(region);
            let file_off = file_start + region.offset as u64;
            let runtime_start = runtime_base.map(|base| base + region.offset as u64);
            let runtime_end = runtime_start.map(|start| start + region.len as u64);
            let summary = fallback_program
                .map(|program| symbols::render_region_label(program, region))
                .unwrap_or_else(|| region.summary_label());
            let _ = writeln!(
                region_lines,
                "symbol={symbol}\tfunc={func_idx}\tregion={region_id}\tkind={}\tblock={}\tinst={}\tcallee={}\tfile_off=0x{file_off:08x}\tfile_end=0x{:08x}\truntime_start={}\truntime_end={}\tsummary={summary}",
                region_kind_name(region),
                fmt_opt_u32(block),
                fmt_opt_u16(inst),
                fmt_opt_u32(callee),
                file_off + region.len as u64,
                fmt_opt_u64(runtime_start),
                fmt_opt_u64(runtime_end),
            );
        }

        let _ = write_function_section(
            &mut function_sections,
            module,
            func_idx,
            file_start,
            code.text.len() as u64,
            runtime_base,
            runtime_end,
            lir_by_func.get(&func_idx).copied(),
            fallback_program,
            stats,
        );
    }

    index.push_str(&region_lines);
    index.push('\n');
    index.push_str(&function_sections);
    fs::write(root.join("native_index.txt"), index)?;
    Ok(())
}

fn write_function_section(
    out: &mut String,
    module: &ModuleInst,
    func_idx: usize,
    file_start: u64,
    code_size: u64,
    runtime_base: Option<u64>,
    runtime_end: Option<u64>,
    lir: Option<&LirProgram>,
    program: Option<&NativeProgram>,
    stats: Option<symbols::NativeSymbolStats>,
) -> Result<(), core::fmt::Error> {
    writeln!(out, "[function {func_idx}]")?;
    writeln!(out, "module={}", module.name)?;
    writeln!(out, "code_file_off=0x{file_start:08x}")?;
    writeln!(out, "code_size=0x{code_size:08x}")?;
    writeln!(out, "runtime_start={}", fmt_opt_u64(runtime_base))?;
    writeln!(out, "runtime_end={}", fmt_opt_u64(runtime_end))?;
    if let Some(stats) = stats {
        writeln!(
            out,
            "stats=blocks:{} ops:{} call_local:{} call_indirect:{} call_external:{} helpers_t:{} helpers_ct:{} br_table:{} loads:{} stores:{}",
            stats.blocks,
            stats.ops,
            stats.call_local,
            stats.call_indirect,
            stats.call_external,
            stats.transparent_helpers,
            stats.control_transfer_helpers,
            stats.br_table,
            stats.loads,
            stats.stores,
        )?;
    }
    writeln!(out)?;
    writeln!(out, "groups:")?;
    if let Some(program) = program {
        out.push_str(&render_group_plan(program));
    } else {
        writeln!(out, "  <unavailable>")?;
    }
    writeln!(out)?;
    writeln!(out, "lir:")?;
    if let Some(lir) = lir {
        out.push_str(&render_lir_program(lir));
    } else {
        writeln!(out, "  <unavailable>")?;
    }
    writeln!(out)?;
    writeln!(out, "native_ir:")?;
    if let Some(program) = program {
        out.push_str(&render_native_program(program));
    } else {
        writeln!(out, "  <unavailable>")?;
    }
    writeln!(out)?;
    Ok(())
}

fn render_group_plan(program: &NativeProgram) -> String {
    let mut out = String::new();
    if program.groups.groups.is_empty() {
        out.push_str("  <none>\n");
        return out;
    }
    for group in &program.groups.groups {
        let term = match group.terminator {
            Some(GroupTerminator::Br) => "br",
            Some(GroupTerminator::BrIf) => "br_if",
            Some(GroupTerminator::BrTable) => "br_table",
            Some(GroupTerminator::If) => "if",
            Some(GroupTerminator::ReturnVoid) => "return_void",
            Some(GroupTerminator::ReturnOne) => "return_one",
            Some(GroupTerminator::Return) => "return",
            Some(GroupTerminator::Unreachable) => "unreachable",
            None => "-",
        };
        let _ = writeln!(
            out,
            "  g{}: ops[{}..{}) term={term}",
            group.id.as_u32(),
            group.start,
            group.end,
        );
    }
    out
}

fn render_lir_program(program: &LirProgram) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "  abi: tos={} hot={}",
        program.abi.tos_register_count,
        program.abi.hot_local_count,
    );
    let _ = writeln!(out, "  hot_locals: {}", render_hot_locals(program.hot_locals.as_ref().map(|plan| plan.mapping())));
    for block in &program.blocks {
        let _ = writeln!(
            out,
            "  block b{} params=[{}]",
            block.id.as_u32(),
            block.params
                .tos
                .iter()
                .map(|value| render_lir_value(*value))
                .collect::<Vec<_>>()
                .join(", "),
        );
        for (inst_index, inst) in block.ops.iter().enumerate() {
            let _ = writeln!(out, "    {inst_index:02}: {}", render_lir_inst(&inst.kind));
        }
        let _ = writeln!(out, "    term: {}", render_lir_terminator(&block.terminator));
    }
    out
}

fn render_lir_inst(inst: &LirInstKind) -> String {
    match inst {
        LirInstKind::Leaf { op, args, results } => format!(
            "leaf {:?} args=[{}] results=[{}]",
            op,
            render_lir_values(args),
            render_lir_values(results),
        ),
        LirInstKind::WriteOperandSlot { slot, src } => {
            format!("write_operand fp[{}] <- {}", slot.0, render_lir_value(*src))
        }
        LirInstKind::ReadOperandSlot { slot, dst } => {
            format!("read_operand {} <- fp[{}]", render_lir_value(*dst), slot.0)
        }
        LirInstKind::ReadHotLocal { reg, dst } => {
            format!("read_hot {} <- h{reg}", render_lir_value(*dst))
        }
        LirInstKind::WriteHotLocal { reg, src } => {
            format!("write_hot h{reg} <- {}", render_lir_value(*src))
        }
        LirInstKind::ReadFrameLocal { frame_slot, dst } => {
            format!("read_frame {} <- fp[{}]", render_lir_value(*dst), frame_slot.0)
        }
        LirInstKind::WriteFrameLocal { frame_slot, src } => {
            format!("write_frame fp[{}] <- {}", frame_slot.0, render_lir_value(*src))
        }
        LirInstKind::CallExternal {
            func_idx,
            args,
            results,
        } => format!(
            "call_external f{func_idx} args=[{}] results=[{}]",
            render_lir_values(args),
            render_lir_values(results),
        ),
        LirInstKind::CallInternal {
            callee,
            args,
            results,
        } => format!(
            "call_internal f{callee} args=[{}] results=[{}]",
            render_lir_values(args),
            render_lir_values(results),
        ),
        LirInstKind::CallIndirect {
            type_idx,
            table_idx,
            index,
            args,
            results,
        } => format!(
            "call_indirect type={type_idx} table={table_idx} index={} args=[{}] results=[{}]",
            render_lir_value(*index),
            render_lir_values(args),
            render_lir_values(results),
        ),
    }
}

fn render_lir_terminator(term: &LirTerminator) -> String {
    match term {
        LirTerminator::Goto(edge) => format!("goto {}", render_lir_edge(edge)),
        LirTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => format!(
            "branch {} then {} else {}",
            render_lir_value(*cond),
            render_lir_edge(then_edge),
            render_lir_edge(else_edge),
        ),
        LirTerminator::BrTable { index, entries } => format!(
            "br_table index={} entries=[{}]",
            render_lir_value(*index),
            entries
                .iter()
                .map(render_lir_edge)
                .collect::<Vec<_>>()
                .join(", "),
        ),
        LirTerminator::Return { values } => format!("return [{}]", render_lir_values(values)),
        LirTerminator::TrapUnreachable => "trap_unreachable".into(),
    }
}

fn render_lir_edge(edge: &LirEdge) -> String {
    format!(
        "b{} [{}]",
        edge.target.as_u32(),
        edge.tos
            .iter()
            .map(|value| render_lir_value(*value))
            .collect::<Vec<_>>()
            .join(", "),
    )
}

fn render_lir_value(value: LirValue) -> String {
    format!("v{}", value.0)
}

fn render_lir_values(values: &[LirValue]) -> String {
    values
        .iter()
        .map(|value| render_lir_value(*value))
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_native_program(program: &NativeProgram) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "  abi: ctx={} fp={} tmp={} hot={} tos={}",
        program.abi.ctx_register_count,
        program.abi.fp_register_count,
        program.abi.tmp_register_count,
        program.abi.hot_local_count,
        program.abi.tos_register_count,
    );
    let _ = writeln!(
        out,
        "  frame: locals={} operands={} call_scratch={} backend_reserved={} frame_size={} spills={}",
        render_frame_span(program.frame.locals),
        render_frame_span(program.frame.operands),
        render_opt_frame_span(program.frame.call_scratch),
        render_opt_frame_span(program.frame.backend_reserved),
        program.frame.frame_size,
        program.spill_slots,
    );
    let _ = writeln!(out, "  hot_locals: {}", render_hot_locals(program.hot_locals.as_ref().map(|plan| plan.mapping())));
    for block in &program.blocks {
        let _ = writeln!(out, "  block b{} entry_tos={}", block.id.0, block.entry_abi.tos_width);
        for (inst_index, inst) in block.ops.iter().enumerate() {
            let _ = writeln!(out, "    {inst_index:02}: {}", render_native_inst(&inst.kind));
        }
        let _ = writeln!(out, "    term: {}", render_native_terminator(&block.terminator));
    }
    out
}

fn render_native_inst(inst: &NativeInstKind) -> String {
    match inst {
        NativeInstKind::Move(mov) => format!(
            "move {} <- {}",
            render_native_place(mov.dst),
            render_native_value(mov.src),
        ),
        NativeInstKind::Leaf { op, args, results } => format!(
            "leaf {:?} args=[{}] results=[{}]",
            op,
            render_native_values(args),
            render_native_places(results),
        ),
        NativeInstKind::Helper(NativeHelperInst::Leaf {
            effect,
            op,
            args,
            results,
        }) => format!(
            "helper {:?} {:?} args=[{}] results=[{}]",
            effect,
            op,
            render_native_values(args),
            render_native_places(results),
        ),
        NativeInstKind::CallExternal {
            func_idx,
            args,
            results,
        } => format!(
            "call_external f{func_idx} args=[{}] results=[{}]",
            render_native_values(args),
            render_native_places(results),
        ),
        NativeInstKind::CallLocal {
            callee,
            callee_params_count,
            callee_frame_size,
            args,
            results,
        } => format!(
            "call_local f{callee} params={} frame_size={} args=[{}] results=[{}]",
            callee_params_count,
            callee_frame_size,
            render_native_values(args),
            render_native_places(results),
        ),
        NativeInstKind::CallIndirect {
            type_idx,
            table_idx,
            index,
            args,
            results,
        } => format!(
            "call_indirect type={type_idx} table={table_idx} index={} args=[{}] results=[{}]",
            render_native_value(*index),
            render_native_values(args),
            render_native_places(results),
        ),
    }
}

fn render_native_terminator(term: &NativeTerminator) -> String {
    match term {
        NativeTerminator::Goto(edge) => format!("goto {}", render_native_edge(edge)),
        NativeTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => format!(
            "branch {} then {} else {}",
            render_branch_cond(cond),
            render_native_edge(then_edge),
            render_native_edge(else_edge),
        ),
        NativeTerminator::BrTable { index, entries } => format!(
            "br_table index={} entries=[{}]",
            render_native_value(*index),
            entries
                .iter()
                .map(render_native_edge)
                .collect::<Vec<_>>()
                .join(", "),
        ),
        NativeTerminator::Return { values } => {
            format!("return [{}]", render_native_values(values))
        }
        NativeTerminator::TrapUnreachable => "trap_unreachable".into(),
    }
}

fn render_native_edge(edge: &NativeEdge) -> String {
    if edge.copies.is_empty() {
        return format!("b{}", edge.target.0);
    }
    format!(
        "b{} [{}]",
        edge.target.0,
        edge.copies
            .iter()
            .map(|mov| format!("{}<-{}", render_native_place(mov.dst), render_native_value(mov.src)))
            .collect::<Vec<_>>()
            .join(", "),
    )
}

fn render_branch_cond(cond: &NativeBranchCond) -> String {
    match cond {
        NativeBranchCond::Value(value) => render_native_value(*value),
        NativeBranchCond::I32Eqz(value) => format!("i32.eqz {}", render_native_value(*value)),
        NativeBranchCond::I64Eqz(value) => format!("i64.eqz {}", render_native_value(*value)),
        NativeBranchCond::I32Compare { kind, lhs, rhs } => {
            format!("i32.{} {} {}", render_compare_kind(*kind), render_native_value(*lhs), render_native_value(*rhs))
        }
        NativeBranchCond::I64Compare { kind, lhs, rhs } => {
            format!("i64.{} {} {}", render_compare_kind(*kind), render_native_value(*lhs), render_native_value(*rhs))
        }
        NativeBranchCond::F32Compare { kind, lhs, rhs } => {
            format!("f32.{} {} {}", render_compare_kind(*kind), render_native_value(*lhs), render_native_value(*rhs))
        }
        NativeBranchCond::F64Compare { kind, lhs, rhs } => {
            format!("f64.{} {} {}", render_compare_kind(*kind), render_native_value(*lhs), render_native_value(*rhs))
        }
    }
}

fn render_compare_kind(kind: NativeCompareKind) -> &'static str {
    match kind {
        NativeCompareKind::Eq => "eq",
        NativeCompareKind::Ne => "ne",
        NativeCompareKind::LtS => "lts",
        NativeCompareKind::LtU => "ltu",
        NativeCompareKind::GtS => "gts",
        NativeCompareKind::GtU => "gtu",
        NativeCompareKind::LeS => "les",
        NativeCompareKind::LeU => "leu",
        NativeCompareKind::GeS => "ges",
        NativeCompareKind::GeU => "geu",
        NativeCompareKind::Lt => "lt",
        NativeCompareKind::Gt => "gt",
        NativeCompareKind::Le => "le",
        NativeCompareKind::Ge => "ge",
    }
}

fn render_native_place(place: NativePlace) -> String {
    match place {
        NativePlace::Location(NativeLocation::Ctx) => "ctx".into(),
        NativePlace::Location(NativeLocation::Fp) => "fp".into(),
        NativePlace::Location(NativeLocation::Hot(reg)) => format!("h{reg}"),
        NativePlace::Location(NativeLocation::Tos(reg)) => format!("t{reg}"),
        NativePlace::Location(NativeLocation::Tmp(reg)) => format!("tmp{reg}"),
        NativePlace::Frame(slot) => format!("fp[{}]", slot.0),
        NativePlace::Spill(slot) => format!("spill[{}]", slot.0),
    }
}

fn render_native_value(value: NativeValue) -> String {
    match value {
        NativeValue::Place(place) => render_native_place(place),
        NativeValue::Imm64(value) => format!("0x{value:x}"),
    }
}

fn render_native_values(values: &[NativeValue]) -> String {
    values
        .iter()
        .map(|value| render_native_value(*value))
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_native_places(values: &[NativePlace]) -> String {
    values
        .iter()
        .map(|value| render_native_place(*value))
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_hot_locals(mapping: Option<&[Option<u32>]>) -> String {
    let Some(mapping) = mapping else {
        return "-".into();
    };
    mapping
        .iter()
        .enumerate()
        .map(|(slot, local)| match local {
            Some(local) => format!("h{slot}=local{local}"),
            None => format!("h{slot}=-"),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_frame_span(span: FrameSpan) -> String {
    format!("[{}..{})", span.start.0, span.end().0)
}

fn render_opt_frame_span(span: Option<FrameSpan>) -> String {
    span.map(render_frame_span).unwrap_or_else(|| "-".into())
}

fn region_kind_name(region: &NativeDebugRegion) -> &'static str {
    match region.kind {
        NativeDebugRegionKind::SharedEntry => "shared_entry",
        NativeDebugRegionKind::PublicEntry => "public_entry",
        NativeDebugRegionKind::InternalEntry => "internal_entry",
        NativeDebugRegionKind::Block { .. } => "block",
        NativeDebugRegionKind::CallLocalContinuation { .. } => "call_local_cont",
        NativeDebugRegionKind::RootExit => "root_exit",
    }
}

fn region_detail(region: &NativeDebugRegion) -> (Option<u32>, Option<u16>, Option<u32>) {
    match region.kind {
        NativeDebugRegionKind::Block { block } => (Some(block), None, None),
        NativeDebugRegionKind::CallLocalContinuation {
            block,
            inst_index,
            callee,
        } => (Some(block), Some(inst_index), Some(callee)),
        _ => (None, None, None),
    }
}

fn fmt_opt_u64(value: Option<u64>) -> String {
    value
        .map(|value| format!("0x{value:016x}"))
        .unwrap_or_else(|| "-".into())
}

fn fmt_opt_u32(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".into())
}

fn fmt_opt_u16(value: Option<u16>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".into())
}
