//! Optional native static dump.
//!
//! Enable by setting `SF_NATIVE_DUMP_DIR=/path/to/output_dir`.
//! The dump writes exactly two files for the current module compile:
//! - `native_index.txt`: function/region metadata, SSA-IR, MachineIR, runtime contract
//! - `native_code.bin`: concatenated emitted native machine code bytes

use alloc::{format, string::String, vec::Vec};
use core::fmt::Write as _;

use crate::{
    error::WasmError,
    vm::{
        machine::machine_ir::{
            MachineAddr, MachineBranchCond, MachineFloatWidth, MachineFunction,
            MachineInstKind, MachineIntWidth,
            MachineLoadExtension, MachineMemWidth, MachineModule, MachineSign,
            MachineRuntimeContract, MachineStorageType, MachineTerminator,
            MachineValue,
        },
        middle::{
            ssa_ir::ir::{SsaBlock, SsaBoundaryOp, SsaInstKind, SsaProgram, SsaTerminator, SsaValue},
        },
    },
};

#[cfg(any(feature = "std", feature = "wasi", test))]
use std::{
    env,
    fs::{self, File},
    io::Write,
    path::PathBuf,
};

/// One debug region within a compiled function (for profiler symbols).
#[derive(Clone, Debug)]
pub(crate) struct DebugRegion {
    /// Byte offset within the function text.
    pub offset: usize,
    /// Byte length of this region.
    pub len: usize,
    /// Human-readable label (e.g. "b0", "edge_3", "prologue", "return_ok").
    pub label: String,
}

/// Per-function SSA-IR data for the dump.
pub(crate) struct DumpFunctionLir<'a> {
    pub func_idx: u32,
    pub ssa: &'a SsaProgram,
}

/// Per-function debug regions from compilation.
pub(crate) struct DumpFunctionRegions {
    pub func_idx: u32,
    pub regions: Vec<DebugRegion>,
}

pub(crate) fn dump_enabled() -> bool {
    #[cfg(any(feature = "std", feature = "wasi", test))]
    {
        env::var_os("SF_NATIVE_DUMP_DIR").is_some()
    }

    #[cfg(not(any(feature = "std", feature = "wasi", test)))]
    {
        false
    }
}

/// Write the full native dump (native_index.txt + native_code.bin).
///
/// `code_slices` contains `(func_idx, code_bytes)` for each compiled function.
/// `debug_regions_by_func` provides per-block code region metadata.
pub(crate) fn write_module_dump(
    module_name: &str,
    function_count: usize,
    lir_inputs: &[DumpFunctionLir<'_>],
    machine_module: &MachineModule,
    runtime: &MachineRuntimeContract,
    code_slices: &[(u32, &[u8])],
    debug_regions_by_func: &[DumpFunctionRegions],
) -> Result<(), WasmError> {
    #[cfg(any(feature = "std", feature = "wasi", test))]
    {
        if !dump_enabled() {
            return Ok(());
        }
        write_dump_impl(
            module_name,
            function_count,
            lir_inputs,
            machine_module,
            runtime,
            code_slices,
            debug_regions_by_func,
        )
        .map_err(|err| WasmError::internal(format!("failed to write native dump: {err}")))
    }

    #[cfg(not(any(feature = "std", feature = "wasi", test)))]
    {
        let _ = (
            module_name,
            function_count,
            lir_inputs,
            machine_module,
            runtime,
            code_slices,
            debug_regions_by_func,
        );
        Ok(())
    }
}

#[cfg(any(feature = "std", feature = "wasi", test))]
fn write_dump_impl(
    module_name: &str,
    function_count: usize,
    lir_inputs: &[DumpFunctionLir<'_>],
    machine_module: &MachineModule,
    runtime: &MachineRuntimeContract,
    code_slices: &[(u32, &[u8])],
    debug_regions_by_func: &[DumpFunctionRegions],
) -> Result<(), std::io::Error> {
    let root = env::var_os("SF_NATIVE_DUMP_DIR")
        .map(PathBuf::from)
        .expect("dump path checked by dump_enabled");
    fs::create_dir_all(&root)?;

    // Write native_code.bin
    let mut code_file = File::create(root.join("native_code.bin"))?;
    let mut code_offsets: Vec<(u32, u64, usize)> = Vec::new(); // (func_idx, file_offset, len)
    let mut file_offset = 0u64;
    for (func_idx, code_bytes) in code_slices {
        code_file.write_all(code_bytes)?;
        code_offsets.push((*func_idx, file_offset, code_bytes.len()));
        file_offset += code_bytes.len() as u64;
    }

    // Build native_index.txt
    let mut index = String::new();
    let _ = writeln!(index, "[module]");
    let _ = writeln!(index, "name={module_name}");
    let _ = writeln!(index, "function_count={function_count}");
    let _ = writeln!(index);

    // Regions table (per-block granularity)
    let _ = writeln!(index, "[regions]");
    let regions_by_func: alloc::collections::BTreeMap<u32, &[DebugRegion]> = debug_regions_by_func
        .iter()
        .map(|d| (d.func_idx, d.regions.as_slice()))
        .collect();
    for (func_idx, file_off, len) in &code_offsets {
        if let Some(regions) = regions_by_func.get(func_idx) {
            for region in *regions {
                let region_file_off = file_off + region.offset as u64;
                let _ = writeln!(
                    index,
                    "symbol=jit::{module_name}::func{func_idx}::{}\tfunc={func_idx}\tregion={}\tfile_off=0x{region_file_off:08x}\tfile_end=0x{:08x}\tcode_size={}",
                    region.label, region.label, region_file_off + region.len as u64, region.len,
                );
            }
        } else {
            let _ = writeln!(
                index,
                "symbol=jit::{module_name}::func{func_idx}\tfunc={func_idx}\tfile_off=0x{file_off:08x}\tfile_end=0x{:08x}\tcode_size={len}",
                file_off + *len as u64,
            );
        }
    }
    let _ = writeln!(index);

    // Call-link layout
    let _ = writeln!(index, "[call_link]");
    let _ = writeln!(index, "slot_count={}", runtime.call_link.slot_count);
    let _ = writeln!(
        index,
        "continuation_offset={}",
        runtime.call_link.continuation_offset
    );
    let _ = writeln!(
        index,
        "caller_frame_offset={}",
        runtime.call_link.caller_frame_offset
    );
    let _ = writeln!(
        index,
        "caller_result_base_offset={}",
        runtime.call_link.caller_result_base_offset
    );
    let _ = writeln!(index);

    // Per-function sections
    let lir_by_func: alloc::collections::BTreeMap<u32, &SsaProgram> = lir_inputs
        .iter()
        .map(|entry| (entry.func_idx, entry.ssa))
        .collect();

    for func in &machine_module.functions {
        let func_idx = func.id.0;
        let _ = writeln!(index, "[function {func_idx}]");

        // Runtime info
        if let Some(rt) = runtime.functions.get(func_idx as usize) {
            let _ = writeln!(index, "frame_prefix_slots={}", rt.frame_prefix_slots);
            let _ = writeln!(index, "total_frame_slots={}", rt.total_frame_slots);
            if let Some(cs) = rt.call_scratch {
                let _ = writeln!(
                    index,
                    "call_scratch=base:{} slots:{}",
                    cs.base_slot, cs.slots
                );
            }
            if let Some(rr) = rt.return_results {
                let _ = writeln!(
                    index,
                    "return_results=base:{} slots:{}",
                    rr.base_slot, rr.slots
                );
            }
        }

        // Code size
        if let Some((_, file_off, len)) = code_offsets.iter().find(|(idx, _, _)| *idx == func_idx) {
            let _ = writeln!(index, "code_file_off=0x{file_off:08x}");
            let _ = writeln!(index, "code_size={len}");
        }
        let _ = writeln!(index);

        // SSA-IR
        let _ = writeln!(index, "ssa_ir:");
        if let Some(lir) = lir_by_func.get(&func_idx) {
            render_lir_program(&mut index, lir);
        } else {
            let _ = writeln!(index, "  <unavailable>");
        }
        let _ = writeln!(index);

        // MachineIR
        let _ = writeln!(index, "machine_ir:");
        render_machine_function(&mut index, func);
        let _ = writeln!(index);
    }

    fs::write(root.join("native_index.txt"), index)?;
    Ok(())
}

// ---- SSA-IR rendering ----

fn render_lir_program(out: &mut String, program: &SsaProgram) {
    let _ = writeln!(out, "  entry=b{}", program.entry.0);
    let _ = writeln!(
        out,
        "  gp_local_cache=[{}] fp_local_cache=[{}]",
        program
            .local_cache
            .gp_preferred_slots
            .iter()
            .map(|s| format!("fp[{}]", s.0))
            .collect::<Vec<_>>()
            .join(", "),
        program
            .local_cache
            .fp_preferred_slots
            .iter()
            .map(|s| format!("fp[{}]", s.0))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for block in &program.blocks {
        render_lir_block(out, block);
    }
}

fn render_lir_block(out: &mut String, block: &SsaBlock) {
    let _ = writeln!(
        out,
        "  block b{} params=[{}]",
        block.id.0,
        block
            .params
            .iter()
            .map(|v| format!("v{}", v.0))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for (i, inst) in block.ops.iter().enumerate() {
        let _ = writeln!(out, "    {i:02}: {}", render_lir_inst(&inst.kind));
    }
    let _ = writeln!(
        out,
        "    term: {}",
        render_lir_terminator(&block.terminator)
    );
}

fn render_lir_inst(kind: &SsaInstKind) -> String {
    match kind {
        SsaInstKind::Value { op, args, results } => format!(
            "leaf {:?} args=[{}] results=[{}]",
            op,
            operands(args),
            vals(results),
        ),
        SsaInstKind::LoadSlot { slot, dst } => {
            format!("load_slot v{} <- fp[{}]", dst.0, slot.0)
        }
        SsaInstKind::StoreSlot { slot, src } => {
            format!("store_slot fp[{}] <- v{}", slot.0, src.0)
        }
        SsaInstKind::Boundary(bop) => render_boundary(bop),
    }
}

fn render_boundary(bop: &SsaBoundaryOp) -> String {
    match bop {
        SsaBoundaryOp::CallInternal {
            callee,
            args,
            results,
            ..
        } => format!(
            "call_internal f{callee} args=fp[{}..{}) results=fp[{}..{})",
            args.start.0,
            args.start.0 + args.count,
            results.start.0,
            results.start.0 + results.count,
        ),
        SsaBoundaryOp::CallExternal {
            func_idx,
            args,
            results,
            ..
        } => format!(
            "call_external f{func_idx} args=fp[{}..{}) results=fp[{}..{})",
            args.start.0,
            args.start.0 + args.count,
            results.start.0,
            results.start.0 + results.count,
        ),
        SsaBoundaryOp::CallIndirect {
            type_idx,
            table_idx,
            index_slot,
            args,
            results,
            ..
        } => format!(
            "call_indirect type={type_idx} table={table_idx} index=fp[{}] args=fp[{}..{}) results=fp[{}..{})",
            index_slot.0,
            args.start.0,
            args.start.0 + args.count,
            results.start.0,
            results.start.0 + results.count,
        ),
        SsaBoundaryOp::MemoryGrow { mem_idx, io } => {
            format!("memory.grow mem={mem_idx} io=fp[{}..{})", io.start.0, io.start.0 + io.count)
        }
        SsaBoundaryOp::MemoryFill { mem_idx, args } => {
            format!("memory.fill mem={mem_idx} args=fp[{}..{})", args.start.0, args.start.0 + args.count)
        }
        SsaBoundaryOp::MemoryCopy { dst_mem_idx, src_mem_idx, args } => {
            format!("memory.copy dst={dst_mem_idx} src={src_mem_idx} args=fp[{}..{})", args.start.0, args.start.0 + args.count)
        }
        SsaBoundaryOp::MemoryInit { data_idx, mem_idx, args } => {
            format!("memory.init data={data_idx} mem={mem_idx} args=fp[{}..{})", args.start.0, args.start.0 + args.count)
        }
        SsaBoundaryOp::DataDrop { data_idx } => format!("data.drop {data_idx}"),
        SsaBoundaryOp::TableGrow { table_idx, args, results } => {
            format!("table.grow table={table_idx} args=fp[{}..{}) results=fp[{}..{})", args.start.0, args.start.0 + args.count, results.start.0, results.start.0 + results.count)
        }
        SsaBoundaryOp::TableFill { table_idx, args } => {
            format!("table.fill table={table_idx} args=fp[{}..{})", args.start.0, args.start.0 + args.count)
        }
        SsaBoundaryOp::TableCopy { dst_table_idx, src_table_idx, args } => {
            format!("table.copy dst={dst_table_idx} src={src_table_idx} args=fp[{}..{})", args.start.0, args.start.0 + args.count)
        }
        SsaBoundaryOp::TableInit { elem_idx, table_idx, args } => {
            format!("table.init elem={elem_idx} table={table_idx} args=fp[{}..{})", args.start.0, args.start.0 + args.count)
        }
        SsaBoundaryOp::ElemDrop { elem_idx } => format!("elem.drop {elem_idx}"),
    }
}

fn render_lir_terminator(term: &SsaTerminator) -> String {
    match term {
        SsaTerminator::Goto(edge) => {
            format!(
                "goto b{} [{}]",
                edge.target.0,
                render_lir_bindings(&edge.bindings)
            )
        }
        SsaTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => format!(
            "branch v{} then b{} [{}] else b{} [{}]",
            cond.0,
            then_edge.target.0,
            render_lir_bindings(&then_edge.bindings),
            else_edge.target.0,
            render_lir_bindings(&else_edge.bindings),
        ),
        SsaTerminator::BrTable { index, entries } => {
            let targets: Vec<String> = entries.iter().map(|e| format!("b{}", e.target.0)).collect();
            format!("br_table v{} [{}]", index.0, targets.join(", "))
        }
        SsaTerminator::Return { results } => match results {
            Some(span) => format!("return fp[{}..{})", span.start.0, span.start.0 + span.count),
            None => "return void".into(),
        },
        SsaTerminator::TrapUnreachable => "trap_unreachable".into(),
    }
}

fn render_lir_bindings(bindings: &[crate::vm::middle::ssa_ir::ir::SsaBinding]) -> String {
    bindings
        .iter()
        .map(|b| format!("v{}=v{}", b.param.0, b.value.0))
        .collect::<Vec<_>>()
        .join(", ")
}

fn vals(vs: &[SsaValue]) -> String {
    vs.iter()
        .map(|v| format!("v{}", v.0))
        .collect::<Vec<_>>()
        .join(", ")
}

fn operands(ops: &[crate::vm::middle::ssa_ir::ir::SsaOperand]) -> String {
    ops.iter()
        .map(|op| match op {
            crate::vm::middle::ssa_ir::ir::SsaOperand::Value(v) => format!("v{}", v.0),
            crate::vm::middle::ssa_ir::ir::SsaOperand::Const(bits) => format!("#{bits}"),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

// ---- MachineIR rendering ----

fn render_machine_function(out: &mut String, func: &MachineFunction) {
    let p = &func.program;
    let _ = writeln!(out, "  entry=b{} reg_count={}", p.entry.0, p.reg_count);
    for block in &p.blocks {
        let _ = writeln!(
            out,
            "  block b{} params=[{}]",
            block.id.0,
            block
                .params
                .iter()
                .map(|param| format!("r{}:{}", param.reg.0, sty(&param.ty)))
                .collect::<Vec<_>>()
                .join(", ")
        );
        for (i, inst) in block.ops.iter().enumerate() {
            let _ = writeln!(out, "    {i:02}: {}", render_machine_inst(&inst.kind));
        }
        let _ = writeln!(out, "    term: {}", render_machine_term(&block.terminator));
    }
}

fn render_machine_inst(kind: &MachineInstKind) -> String {
    match kind {
        MachineInstKind::Move { ty, dst, src } => {
            format!("move.{} r{} <- {}", sty(ty), dst.0, mval(src))
        }
        MachineInstKind::FloatConst { width, dst, bits } => {
            format!("{}.const r{} <- 0x{:x}", fw(width), dst.0, bits)
        }
        MachineInstKind::Load {
            ty,
            dst,
            addr,
            width,
            extension,
        } => {
            format!(
                "load.{}.{}{} r{} <- [{}]",
                sty(ty),
                mwidth(width),
                mext(extension),
                dst.0,
                maddr(addr)
            )
        }
        MachineInstKind::Store {
            ty,
            addr,
            width,
            src,
        } => {
            format!(
                "store.{}.{} [{}] <- {}",
                sty(ty),
                mwidth(width),
                maddr(addr),
                mval(src)
            )
        }
        MachineInstKind::IntUnary {
            width,
            op,
            dst,
            src,
        } => {
            format!("{}.{:?} r{} <- {}", iw(width), op, dst.0, mval(src))
        }
        MachineInstKind::IntBinary {
            width,
            op,
            dst,
            lhs,
            rhs,
        } => {
            format!(
                "{}.{:?} r{} <- {} {}",
                iw(width),
                op,
                dst.0,
                mval(lhs),
                mval(rhs)
            )
        }
        MachineInstKind::Int64PairBinary {
            op,
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
        } => {
            format!(
                "i64pair.{:?} r{},r{} <- ({}, {}) ({}, {})",
                op,
                dst_lo.0,
                dst_hi.0,
                mval(lhs_lo),
                mval(lhs_hi),
                mval(rhs_lo),
                mval(rhs_hi)
            )
        }
        MachineInstKind::Int64PairDivRem {
            sign,
            rem,
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
        } => {
            format!(
                "i64pair.{}.{} r{},r{} <- ({}, {}) ({}, {})",
                if *rem { "rem" } else { "div" },
                match sign {
                    MachineSign::Signed => "s",
                    MachineSign::Unsigned => "u",
                },
                dst_lo.0,
                dst_hi.0,
                mval(lhs_lo),
                mval(lhs_hi),
                mval(rhs_lo),
                mval(rhs_hi)
            )
        }
        MachineInstKind::Int64PairUnary {
            op,
            dst_lo,
            dst_hi,
            src_lo,
            src_hi,
        } => {
            format!(
                "i64pair.{:?} r{},r{} <- ({}, {})",
                op,
                dst_lo.0,
                dst_hi.0,
                mval(src_lo),
                mval(src_hi)
            )
        }
        MachineInstKind::Int64PairShift {
            op,
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs,
        } => {
            format!(
                "i64pair.{:?} r{},r{} <- ({}, {}) {}",
                op,
                dst_lo.0,
                dst_hi.0,
                mval(lhs_lo),
                mval(lhs_hi),
                mval(rhs)
            )
        }
        MachineInstKind::Int64PairCompare {
            kind,
            sign,
            dst,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
        } => {
            format!(
                "i64pair.cmp.{:?}.{:?} r{} <- ({}, {}) ({}, {})",
                kind,
                sign,
                dst.0,
                mval(lhs_lo),
                mval(lhs_hi),
                mval(rhs_lo),
                mval(rhs_hi)
            )
        }
        MachineInstKind::IntCompare {
            width,
            kind,
            sign,
            dst,
            lhs,
            rhs,
        } => {
            format!(
                "{}.cmp.{:?}.{:?} r{} <- {} {}",
                iw(width),
                kind,
                sign,
                dst.0,
                mval(lhs),
                mval(rhs)
            )
        }
        MachineInstKind::FloatUnary {
            width,
            op,
            dst,
            src,
        } => {
            format!("{}.{:?} r{} <- {}", fw(width), op, dst.0, mval(src))
        }
        MachineInstKind::FloatBinary {
            width,
            op,
            dst,
            lhs,
            rhs,
        } => {
            format!(
                "{}.{:?} r{} <- {} {}",
                fw(width),
                op,
                dst.0,
                mval(lhs),
                mval(rhs)
            )
        }
        MachineInstKind::FloatCompare {
            width,
            kind,
            dst,
            lhs,
            rhs,
        } => {
            format!(
                "{}.cmp.{:?} r{} <- {} {}",
                fw(width),
                kind,
                dst.0,
                mval(lhs),
                mval(rhs)
            )
        }
        MachineInstKind::Convert { op, dst, src } => {
            format!("cvt.{:?} r{} <- {}", op, dst.0, mval(src))
        }
        MachineInstKind::ConvertI64PairToFloat {
            width,
            sign,
            dst,
            src_lo,
            src_hi,
        } => {
            format!(
                "cvt.{:?}.{:?} r{} <- ({}, {})",
                width,
                sign,
                dst.0,
                mval(src_lo),
                mval(src_hi)
            )
        }
        MachineInstKind::ConvertFloatToI64Pair {
            op,
            dst_lo,
            dst_hi,
            src,
        } => {
            format!("cvt.{:?} r{},r{} <- {}", op, dst_lo.0, dst_hi.0, mval(src))
        }
        MachineInstKind::ReinterpretF64ToI64Pair {
            dst_lo,
            dst_hi,
            src,
        } => {
            format!(
                "reinterpret.f64->i64pair r{},r{} <- {}",
                dst_lo.0,
                dst_hi.0,
                mval(src)
            )
        }
        MachineInstKind::ReinterpretI64PairToF64 {
            dst,
            src_lo,
            src_hi,
        } => {
            format!(
                "reinterpret.i64pair->f64 r{} <- ({}, {})",
                dst.0,
                mval(src_lo),
                mval(src_hi)
            )
        }
        MachineInstKind::Select {
            ty,
            dst,
            on_true,
            on_false,
            cond,
        } => {
            format!(
                "select.{} r{} <- {} ? {} : {}",
                sty(ty),
                dst.0,
                mval(cond),
                mval(on_true),
                mval(on_false)
            )
        }
        MachineInstKind::TrapIf { kind, cond } => {
            format!("trap_if {:?} {}", kind, render_branch_cond(cond))
        }
        MachineInstKind::CallHelper(call) => {
            format!(
                "call_helper extern={} const={}",
                call.target.0, call.metadata.0
            )
        }
    }
}

fn render_machine_term(term: &MachineTerminator) -> String {
    match term {
        MachineTerminator::Jump(edge) => {
            format!("jump b{} [{}]", edge.target.0, medge_args(&edge.args))
        }
        MachineTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => format!(
            "branch {} then b{} [{}] else b{} [{}]",
            render_branch_cond(cond),
            then_edge.target.0,
            medge_args(&then_edge.args),
            else_edge.target.0,
            medge_args(&else_edge.args),
        ),
        MachineTerminator::JumpTable { index, entries } => {
            let targets: Vec<String> = entries.iter().map(|e| format!("b{}", e.target.0)).collect();
            format!("jump_table {} [{}]", mval(index), targets.join(", "))
        }
        MachineTerminator::CallDirect {
            callee,
            callee_frame_base,
            continuation,
        } => {
            format!(
                "call_direct f{} frame_base=r{} cont=b{}",
                callee.0, callee_frame_base.0, continuation.0
            )
        }
        MachineTerminator::CallIndirect {
            callee_target,
            callee_frame_base,
            arg_slots,
            caller_result_base,
            continuation,
        } => {
            format!(
                "call_indirect target={} frame_base=r{} args={} result_base={} cont=b{}",
                mval(callee_target),
                callee_frame_base.0,
                arg_slots,
                caller_result_base,
                continuation.0
            )
        }
        MachineTerminator::Return => "return".into(),
        MachineTerminator::Trap { kind } => format!("trap {:?}", kind),
    }
}

fn render_branch_cond(cond: &MachineBranchCond) -> String {
    match cond {
        MachineBranchCond::Value(v) => mval(v),
        MachineBranchCond::IntCompare {
            width,
            kind,
            sign,
            lhs,
            rhs,
        } => format!(
            "{}.cmp.{:?}.{:?} {} {}",
            iw(width),
            kind,
            sign,
            mval(lhs),
            mval(rhs)
        ),
    }
}

// ---- formatting helpers ----

fn mval(v: &MachineValue) -> String {
    match v {
        MachineValue::Reg(r) => format!("r{}", r.0),
        MachineValue::Imm64(i) => {
            if *i <= 0xffff {
                format!("{i}")
            } else {
                format!("0x{i:x}")
            }
        }
    }
}

fn maddr(a: &MachineAddr) -> String {
    if a.offset == 0 {
        format!("r{}", a.base.0)
    } else {
        format!("r{}+{}", a.base.0, a.offset)
    }
}

fn medge_args(args: &[MachineValue]) -> String {
    args.iter().map(mval).collect::<Vec<_>>().join(", ")
}

fn mwidth(w: &MachineMemWidth) -> &'static str {
    match w {
        MachineMemWidth::U8 => "u8",
        MachineMemWidth::U16 => "u16",
        MachineMemWidth::U32 => "u32",
        MachineMemWidth::U64 => "u64",
    }
}

fn mext(e: &MachineLoadExtension) -> &'static str {
    match e {
        MachineLoadExtension::None => "",
        MachineLoadExtension::SignExtend => ".sx",
        MachineLoadExtension::ZeroExtend => ".zx",
    }
}

fn iw(w: &MachineIntWidth) -> &'static str {
    match w {
        MachineIntWidth::I32 => "i32",
        MachineIntWidth::I64 => "i64",
    }
}

fn fw(w: &MachineFloatWidth) -> &'static str {
    match w {
        MachineFloatWidth::F32 => "f32",
        MachineFloatWidth::F64 => "f64",
    }
}

fn sty(ty: &MachineStorageType) -> &'static str {
    match ty {
        MachineStorageType::GpWord => "gp",
        MachineStorageType::GpI64 => "i64",
        MachineStorageType::Fp32 => "f32",
        MachineStorageType::Fp64 => "f64",
    }
}
