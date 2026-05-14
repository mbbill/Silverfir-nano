//! Optional native static dump.
//!
//! Enable by setting `SF_NATIVE_DUMP_DIR=/path/to/output_dir`.
//! The dump writes exactly two files for the current module compile:
//! - `native_index.txt`: function/region metadata, SSA-IR, MachineIR, runtime contract
//! - `native_code.bin`: concatenated emitted native machine code bytes

use crate::collections;

use core::fmt::Write as _;
use tracked_alloc::{format, string::String};

use crate::{
    error::WasmError,
    vm::{
        arch::common::types::DebugRegion,
        machine::machine_ir::{
            MachineAddr, MachineArgSrc, MachineArgSrcPair, MachineBranchCond, MachineCallArgs,
            MachineCallLaneArg, MachineCallTarget, MachineFloatWidth, MachineFunction,
            MachineInstKind, MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineModule,
            MachineModuleAbi, MachineParamLoc, MachineRegOwner, MachineSign, MachineStorageType,
            MachineTerminator, MachineValue,
        },
        middle::frame::{FrameSlot, FrameSpan},
        middle::ssa_ir::ir::{
            DecodedOperand, SsaBinding, SsaBlock, SsaCallOp, SsaInst, SsaInstView, SsaOperand,
            SsaProgram, SsaTerminator,
        },
    },
};

#[cfg(any(sf_has_std, test))]
use std::{
    env,
    fs::{self, File},
    io::Write,
    path::PathBuf,
};

/// Per-function SSA-IR data for the dump.
///
/// This is intentionally owned so debug dumping can preserve exactly the data
/// it needs without forcing the production compile pipeline to keep SSA alive.
pub(crate) struct DumpFunctionLir {
    pub func_idx: u32,
    pub ssa: SsaProgram,
}

/// Per-function debug regions from compilation.
pub(crate) struct DumpFunctionRegions {
    pub func_idx: u32,
    pub regions: collections::Vec<DebugRegion>,
}

pub(crate) fn dump_enabled() -> bool {
    #[cfg(any(sf_has_std, test))]
    {
        env::var_os("SF_NATIVE_DUMP_DIR").is_some()
    }

    #[cfg(not(any(sf_has_std, test)))]
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
    lir_inputs: &[DumpFunctionLir],
    machine_module: &MachineModule,
    runtime: &MachineModuleAbi,
    code_slices: &[(u32, &[u8])],
    debug_regions_by_func: &[DumpFunctionRegions],
) -> Result<(), WasmError> {
    #[cfg(any(sf_has_std, test))]
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
        .map_err(|_err| WasmError::internal("failed to write native dump"))
    }

    #[cfg(not(any(sf_has_std, test)))]
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

#[cfg(any(sf_has_std, test))]
fn write_dump_impl(
    module_name: &str,
    function_count: usize,
    lir_inputs: &[DumpFunctionLir],
    machine_module: &MachineModule,
    runtime: &MachineModuleAbi,
    code_slices: &[(u32, &[u8])],
    debug_regions_by_func: &[DumpFunctionRegions],
) -> Result<(), std::io::Error> {
    let root = env::var_os("SF_NATIVE_DUMP_DIR")
        .map(PathBuf::from)
        .expect("dump path checked by dump_enabled");
    fs::create_dir_all(&root)?;

    // Write native_code.bin
    let mut code_file = File::create(root.join("native_code.bin"))?;
    let mut code_offsets: collections::Vec<(u32, u64, usize)> = collections::Vec::new(); // (func_idx, file_offset, len)
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
    let regions_by_func: tracked_alloc::collections::BTreeMap<u32, &[DebugRegion]> =
        debug_regions_by_func
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

    // (No call_link layout under the new local-call ABI — see
    // docs/ABI_PLAN.md §6. The dead software call-link record has been
    // replaced by a backend-private host-stack call record.)

    // Per-function sections
    let lir_by_func: tracked_alloc::collections::BTreeMap<u32, &SsaProgram> = lir_inputs
        .iter()
        .map(|entry| (entry.func_idx, &entry.ssa))
        .collect();

    for func in &machine_module.functions {
        let func_idx = func.id.0;
        let _ = writeln!(index, "[function {func_idx}]");

        // Runtime info
        if let Some(rt) = runtime.functions.get(func_idx as usize) {
            let _ = writeln!(index, "frame_prefix_slots={}", rt.frame_prefix_slots);
            let _ = writeln!(index, "total_frame_slots={}", rt.total_frame_slots);
            if !rt.param_locs.is_empty() {
                let _ = write!(index, "param_locs=");
                render_param_locs(&mut index, &rt.param_locs);
                let _ = writeln!(index);
            }
            if let Some(hs) = rt.helper_scratch {
                let _ = writeln!(
                    index,
                    "helper_scratch=base:{} slots:{}",
                    hs.base_slot, hs.slots
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
    let _ = writeln!(out, "  local_slots={}", program.local_slot_types.len());
    // Emit local types so analysis tools can compute GP/FP pressure split.
    if !program.local_slot_types.is_empty() {
        let _ = write!(out, "  local_types=[");
        for (i, ty) in program.local_slot_types.iter().enumerate() {
            if i > 0 {
                let _ = write!(out, ", ");
            }
            let _ = write!(out, "{ty}");
        }
        let _ = writeln!(out, "]");
    }
    for block in &program.blocks {
        let cached = program
            .block_entry_cached_slots
            .get(block.id.0 as usize)
            .cloned()
            .unwrap_or_default();
        render_lir_block(out, block, &cached, program);
    }
}

fn render_param_locs(out: &mut String, locs: &[MachineParamLoc]) {
    let _ = write!(out, "[");
    for (index, loc) in locs.iter().enumerate() {
        if index > 0 {
            let _ = write!(out, ", ");
        }
        match *loc {
            MachineParamLoc::Frame { param_index, slot } => {
                let _ = write!(out, "p{param_index}=fp[{}]", slot.0);
            }
            MachineParamLoc::GpArg {
                param_index,
                lane,
                ty,
            } => {
                let _ = write!(out, "p{param_index}=gp{lane}:{ty:?}");
            }
            MachineParamLoc::GpArgPair {
                param_index,
                lo_lane,
                hi_lane,
            } => {
                let _ = write!(out, "p{param_index}=gp{lo_lane}/gp{hi_lane}:i64");
            }
            MachineParamLoc::FpArg {
                param_index,
                lane,
                ty,
            } => {
                let _ = write!(out, "p{param_index}=fpreg{lane}:{ty:?}");
            }
        }
    }
    let _ = write!(out, "]");
}

fn render_lir_block(
    out: &mut String,
    block: &SsaBlock,
    cached: &[FrameSlot],
    program: &SsaProgram,
) {
    if cached.is_empty() {
        let _ = writeln!(
            out,
            "  block b{} params=[{}]",
            block.id.0,
            block
                .params
                .iter()
                .map(|v| format!("v{}", v.0))
                .collect::<collections::Vec<_>>()
                .join(", ")
        );
    } else {
        let _ = writeln!(
            out,
            "  block b{} params=[{}] cached=[{}]",
            block.id.0,
            block
                .params
                .iter()
                .map(|v| format!("v{}", v.0))
                .collect::<collections::Vec<_>>()
                .join(", "),
            cached
                .iter()
                .map(|s| format!("fp[{}]", s.0))
                .collect::<collections::Vec<_>>()
                .join(", ")
        );
    }
    for (i, inst) in block.ops.iter().enumerate() {
        let _ = writeln!(out, "    {i:02}: {}", render_lir_inst(inst, block, program));
    }
    let _ = writeln!(
        out,
        "    term: {}",
        render_lir_terminator(&block.terminator)
    );
}

fn render_lir_inst(inst: &SsaInst, block: &SsaBlock, program: &SsaProgram) -> String {
    match inst.view(program, block) {
        SsaInstView::Value { op, result, args } => {
            let arg_vec: collections::Vec<SsaOperand> = args.iter().collect();
            let result_str = if result.is_some() {
                format!("v{}", result.0)
            } else {
                String::new()
            };
            format!(
                "leaf {:?} args=[{}] results=[{}]",
                op,
                operands(&arg_vec, program),
                result_str,
            )
        }
        SsaInstView::LocalGetSlot { slot, dst } => {
            format!("local.get_slot v{} <- fp[{}]", dst.0, slot.0)
        }
        SsaInstView::LocalGetCache { slot, dst } => {
            format!("local.get_cache v{} <- fp[{}]", dst.0, slot.0)
        }
        SsaInstView::Fill { slot, dst } => {
            format!("fill v{} <- fp[{}]", dst.0, slot.0)
        }
        SsaInstView::LocalSetSlot { slot, src } => {
            format!("local.set_slot fp[{}] <- v{}", slot.0, src.0)
        }
        SsaInstView::LocalSetCache { slot, src } => {
            format!("local.set_cache fp[{}] <- v{}", slot.0, src.0)
        }
        SsaInstView::LocalEnsureCache { slot } => {
            format!("local.ensure_cache fp[{}]", slot.0)
        }
        SsaInstView::LocalReserveCache { slot } => {
            format!("local.reserve_cache fp[{}]", slot.0)
        }
        SsaInstView::LocalDropCache { slot } => {
            format!("local.drop_cache fp[{}]", slot.0)
        }
        SsaInstView::Spill { slot, src } => {
            format!("spill fp[{}] <- v{}", slot.0, src.0)
        }
        SsaInstView::Call(bop) => render_call(bop),
    }
}

fn render_call(bop: &SsaCallOp) -> String {
    match bop {
        SsaCallOp::CallDirect {
            callee,
            args,
            results,
            ..
        } => format!(
            "call_direct f{callee} args={} results=fp[{}..{})",
            render_ssa_call_args(args),
            results.start.0,
            results.start.0 + results.count,
        ),
        SsaCallOp::CallIndirect {
            type_idx,
            table_idx,
            index,
            args,
            results,
            ..
        } => format!(
            "call_indirect type={type_idx} table={table_idx} index={} args={} results=fp[{}..{})",
            render_ssa_call_operand_loc(index),
            render_ssa_call_args(args),
            results.start.0,
            results.start.0 + results.count,
        ),
        SsaCallOp::CallRef {
            type_idx,
            callee_ref,
            args,
            results,
            ..
        } => format!(
            "call_ref type={type_idx} ref={} args={} results=fp[{}..{})",
            render_ssa_call_operand_loc(callee_ref),
            render_ssa_call_args(args),
            results.start.0,
            results.start.0 + results.count,
        ),
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
            let targets: collections::Vec<String> =
                entries.iter().map(|e| format!("b{}", e.target.0)).collect();
            format!("br_table v{} [{}]", index.0, targets.join(", "))
        }
        SsaTerminator::Return { results } => match results {
            Some(span) => format!("return fp[{}..{})", span.start.0, span.start.0 + span.count),
            None => "return void".into(),
        },
        SsaTerminator::ReturnScalar { result, results } => format!(
            "return_scalar {} -> fp[{}..{})",
            render_ssa_scalar_result_loc(result),
            results.start.0,
            results.start.0 + results.count,
        ),
        SsaTerminator::TailCallDirect {
            callee,
            args,
            return_results,
        } => format!(
            "tail_call f{callee} args={} results={}",
            render_ssa_call_args(args),
            render_frame_span_or_void(*return_results),
        ),
        SsaTerminator::TailCallIndirect {
            type_idx,
            table_idx,
            index,
            args,
            return_results,
        } => format!(
            "tail_call_indirect type={type_idx} table={table_idx} index={} args={} results={}",
            render_ssa_call_operand_loc(index),
            render_ssa_call_args(args),
            render_frame_span_or_void(*return_results),
        ),
        SsaTerminator::TailCallRef {
            type_idx,
            callee_ref,
            args,
            return_results,
        } => format!(
            "tail_call_ref type={type_idx} ref={} args={} results={}",
            render_ssa_call_operand_loc(callee_ref),
            render_ssa_call_args(args),
            render_frame_span_or_void(*return_results),
        ),
        SsaTerminator::TrapUnreachable => "trap_unreachable".into(),
        SsaTerminator::EhThrow { tag_idx, args } => format!(
            "eh_throw tag={} args=[{}..{})",
            tag_idx,
            args.start.0,
            args.start.0 + args.count,
        ),
        SsaTerminator::EhThrowRef { exnref_slot } => {
            format!("eh_throw_ref exnref=s{}", exnref_slot.0)
        }
    }
}

fn render_ssa_call_args(args: &crate::vm::middle::ssa_ir::ir::SsaCallArgs) -> String {
    let suffix = args
        .live_suffix
        .iter()
        .map(|arg| {
            format!(
                "p{}=v{}@fp[{}]",
                arg.param_index, arg.value.0, arg.frame_slot.0
            )
        })
        .collect::<collections::Vec<_>>()
        .join(",");
    format!(
        "fp[{}..{}) prefix={} live=[{}]",
        args.frame_base.0,
        args.frame_base.0 + args.total_params,
        args.stack_prefix_count,
        suffix
    )
}

fn render_ssa_call_operand_loc(loc: &crate::vm::middle::ssa_ir::ir::SsaCallOperandLoc) -> String {
    match loc {
        crate::vm::middle::ssa_ir::ir::SsaCallOperandLoc::Stack { slot } => {
            format!("fp[{}]", slot.0)
        }
        crate::vm::middle::ssa_ir::ir::SsaCallOperandLoc::Live { value, slot, .. } => {
            format!("v{}@fp[{}]", value.0, slot.0)
        }
    }
}

fn render_ssa_scalar_result_loc(loc: &crate::vm::middle::ssa_ir::ir::SsaScalarResultLoc) -> String {
    match loc {
        crate::vm::middle::ssa_ir::ir::SsaScalarResultLoc::Stack { slot, .. } => {
            format!("fp[{}]", slot.0)
        }
        crate::vm::middle::ssa_ir::ir::SsaScalarResultLoc::Live { value, slot, .. } => {
            format!("v{}@fp[{}]", value.0, slot.0)
        }
    }
}

fn render_frame_span_or_void(span: Option<FrameSpan>) -> String {
    match span {
        Some(span) => format!("fp[{}..{})", span.start.0, span.start.0 + span.count),
        None => "void".into(),
    }
}

fn render_lir_bindings(bindings: &[SsaBinding]) -> String {
    bindings
        .iter()
        .map(|b| format!("v{}=v{}", b.param.0, b.value.0))
        .collect::<collections::Vec<_>>()
        .join(", ")
        .into()
}

fn operands(ops: &[SsaOperand], program: &SsaProgram) -> String {
    ops.iter()
        .map(|op| match op.decode() {
            DecodedOperand::None => String::from("_"),
            DecodedOperand::Value(v) => format!("v{}", v.0),
            DecodedOperand::Const(idx) => {
                let bits = program.const_pool.get(idx as usize).copied().unwrap_or(0);
                format!("#{bits}")
            }
        })
        .collect::<collections::Vec<_>>()
        .join(", ")
        .into()
}

// ---- MachineIR rendering ----

fn render_machine_function(out: &mut String, func: &MachineFunction) {
    let p = &func.program;
    let _ = writeln!(out, "  entry=b{}", p.entry.0);
    for block in &p.blocks {
        let _ = writeln!(
            out,
            "  block b{} params=[{}]",
            block.id.0,
            block
                .params
                .iter()
                .map(|param| format!(
                    "r{}:{}:{}",
                    param.reg.0,
                    owner_tag(param.owner),
                    sty(&param.ty)
                ))
                .collect::<collections::Vec<_>>()
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
        MachineInstKind::Move {
            owner,
            ty,
            dst,
            src,
            ..
        } => {
            format!(
                "move.{}.{} r{} <- {}",
                owner_tag(*owner),
                sty(ty),
                dst.0,
                mval(src)
            )
        }
        MachineInstKind::FloatConst { width, dst, bits } => {
            format!("{}.const r{} <- 0x{:x}", fw(width), dst.0, bits)
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::V128Const { dst, bytes } => {
            format!("v128.const r{} <- {:02x?}", dst.0, bytes)
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::V128FromRaw { dst, raw, .. } => {
            format!("v128.from_raw r{} <- {}", dst.0, mval(raw))
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::V128ToRaw { dst, src } => {
            format!("v128.to_raw r{} <- {}", dst.0, mval(src))
        }
        MachineInstKind::Load {
            owner,
            ty,
            dst,
            addr,
            width,
            extension,
        } => {
            format!(
                "load.{}.{}.{}{} r{} <- [{}]",
                owner_tag(*owner),
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
        MachineInstKind::IndexedLoad {
            dst,
            base,
            index,
            index_extend,
            offset,
            width,
            extension,
        } => {
            format!(
                "indexed_load.{}{} r{} <- [r{} + r{}({:?}) + {}]",
                mwidth(width),
                mext(extension),
                dst.0,
                base.0,
                index.0,
                index_extend,
                offset
            )
        }
        MachineInstKind::IndexedStore {
            base,
            index,
            index_extend,
            offset,
            width,
            src,
        } => {
            format!(
                "indexed_store.{} [r{} + r{}({:?}) + {}] <- {}",
                mwidth(width),
                base.0,
                index.0,
                index_extend,
                offset,
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
        MachineInstKind::BitfieldExtractU {
            width,
            dst,
            src,
            lsb,
            bits,
        } => {
            format!(
                "ubfx.{} r{} <- r{}, #{}, #{}",
                iw(width),
                dst.0,
                src.0,
                lsb,
                bits
            )
        }
        MachineInstKind::IntBinaryShifted {
            width,
            op,
            dst,
            lhs,
            rhs,
            shift,
            amount,
        } => {
            format!(
                "{}.{:?} r{} <- r{}, r{} {:?} #{}",
                iw(width),
                op,
                dst.0,
                lhs.0,
                rhs.0,
                shift,
                amount
            )
        }
        MachineInstKind::TestBits {
            width,
            kind,
            dst,
            src,
            mask,
        } => {
            format!(
                "tst.{:?}.{} r{} <- r{}, {}",
                kind,
                iw(width),
                dst.0,
                src.0,
                mval(mask)
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
        MachineInstKind::Int64MulFromSignExt32 {
            dst_lo,
            dst_hi,
            lhs,
            rhs,
        } => {
            format!(
                "i32x32_to_i64.smul r{},r{} <- {} {}",
                dst_lo.0,
                dst_hi.0,
                mval(lhs),
                mval(rhs)
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
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdUnary {
            opcode,
            dst_ty,
            dst,
            src,
        } => format!(
            "simd.unary[0x{opcode:x}].{} r{} <- {}",
            sty(dst_ty),
            dst.0,
            mval(src)
        ),
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdBinary {
            opcode,
            dst_ty,
            dst,
            lhs,
            rhs,
        } => format!(
            "simd.binary[0x{opcode:x}].{} r{} <- {} {}",
            sty(dst_ty),
            dst.0,
            mval(lhs),
            mval(rhs)
        ),
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdTernary {
            opcode,
            dst_ty,
            dst,
            a,
            b,
            c,
        } => format!(
            "simd.ternary[0x{opcode:x}].{} r{} <- {} {} {}",
            sty(dst_ty),
            dst.0,
            mval(a),
            mval(b),
            mval(c)
        ),
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdShift {
            opcode,
            dst,
            vector,
            shift,
        } => format!(
            "simd.shift[0x{opcode:x}] r{} <- {} {}",
            dst.0,
            mval(vector),
            mval(shift)
        ),
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdExtractLane {
            opcode,
            lane,
            dst_ty,
            dst,
            src,
        } => format!(
            "simd.extract_lane[0x{opcode:x},{}].{} r{} <- {}",
            lane,
            sty(dst_ty),
            dst.0,
            mval(src)
        ),
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdReplaceLane {
            opcode,
            lane,
            dst,
            vector,
            scalar,
        } => format!(
            "simd.replace_lane[0x{opcode:x},{}] r{} <- {} {}",
            lane,
            dst.0,
            mval(vector),
            mval(scalar)
        ),
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdShuffle {
            dst,
            lhs,
            rhs,
            lanes,
        } => format!(
            "simd.shuffle r{} <- {} {} {:02x?}",
            dst.0,
            mval(lhs),
            mval(rhs),
            lanes
        ),
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdLoad { opcode, dst, addr } => {
            format!("simd.load[0x{opcode:x}] r{} <- [{}]", dst.0, maddr(addr))
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdStore { opcode, addr, src } => {
            format!(
                "simd.store[0x{opcode:x}] [{}] <- {}",
                maddr(addr),
                mval(src)
            )
        }
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdLoadLane {
            opcode,
            lane,
            dst,
            addr,
            vector,
        } => format!(
            "simd.load_lane[0x{opcode:x},{}] r{} <- [{}] {}",
            lane,
            dst.0,
            maddr(addr),
            mval(vector)
        ),
        #[cfg(sf_has_simd)]
        MachineInstKind::SimdStoreLane {
            opcode,
            lane,
            addr,
            vector,
        } => format!(
            "simd.store_lane[0x{opcode:x},{}] [{}] <- {}",
            lane,
            maddr(addr),
            mval(vector)
        ),
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
        MachineInstKind::CallRuntime(call) => {
            format!("call_runtime const={}", call.metadata.0)
        }
        MachineInstKind::EhThrow { tag_idx, args } => {
            format!(
                "eh.throw tag={} frame[{}..+{}]",
                tag_idx, args.start.0, args.count
            )
        }
        MachineInstKind::EhThrowRef { exnref_slot } => {
            format!("eh.throw_ref frame[{}]", exnref_slot.0)
        }
        MachineInstKind::EhAllocExnRef { tag_idx, dst } => {
            format!("eh.alloc_exn_ref tag={} -> r{}", tag_idx, dst.0)
        }
        MachineInstKind::MemoryGrow {
            mem_idx,
            dst,
            delta,
        } => {
            format!("memory.grow mem={} {} -> r{}", mem_idx, mval(delta), dst.0)
        }
        MachineInstKind::MemoryFill {
            mem_idx,
            dest,
            val,
            len,
        } => {
            format!(
                "memory.fill mem={} {} {} {}",
                mem_idx,
                mval(dest),
                mval(val),
                mval(len)
            )
        }
        MachineInstKind::MemoryCopy {
            dst_mem,
            src_mem,
            dest,
            src,
            len,
        } => {
            format!(
                "memory.copy dst={} src={} {} {} {}",
                dst_mem,
                src_mem,
                mval(dest),
                mval(src),
                mval(len)
            )
        }
        MachineInstKind::MemoryInit {
            mem_idx,
            data_idx,
            dest,
            src,
            len,
        } => {
            format!(
                "memory.init mem={} data={} {} {} {}",
                mem_idx,
                data_idx,
                mval(dest),
                mval(src),
                mval(len)
            )
        }
        MachineInstKind::DataDrop { data_idx } => {
            format!("data.drop data={}", data_idx)
        }
        MachineInstKind::TableGrow {
            table_idx,
            dst,
            init_val,
            delta,
        } => {
            format!(
                "table.grow tbl={} {} {} -> r{}",
                table_idx,
                mval(init_val),
                mval(delta),
                dst.0
            )
        }
        MachineInstKind::TableFill {
            table_idx,
            start,
            val,
            len,
        } => {
            format!(
                "table.fill tbl={} {} {} {}",
                table_idx,
                mval(start),
                mval(val),
                mval(len)
            )
        }
        MachineInstKind::TableCopy {
            dst_tbl,
            src_tbl,
            dest,
            src,
            len,
        } => {
            format!(
                "table.copy dst={} src={} {} {} {}",
                dst_tbl,
                src_tbl,
                mval(dest),
                mval(src),
                mval(len)
            )
        }
        MachineInstKind::TableInit {
            table_idx,
            elem_idx,
            dest,
            src,
            len,
        } => {
            format!(
                "table.init tbl={} elem={} {} {} {}",
                table_idx,
                elem_idx,
                mval(dest),
                mval(src),
                mval(len)
            )
        }
        MachineInstKind::ElemDrop { elem_idx } => {
            format!("elem.drop elem={}", elem_idx)
        }
        MachineInstKind::RefFunc { func_idx, dst } => {
            format!("ref.func func={} -> r{}", func_idx, dst.0)
        }
        MachineInstKind::RefAsNonNull { src, dst } => {
            format!("ref.as_non_null {} -> r{}", mval(src), dst.0)
        }
        MachineInstKind::RefEq { lhs, rhs, dst } => {
            format!("ref.eq {} {} -> r{}", mval(lhs), mval(rhs), dst.0)
        }
        MachineInstKind::RefI31 { src, dst } => {
            format!("ref.i31 {} -> r{}", mval(src), dst.0)
        }
        MachineInstKind::I31GetS { src, dst } => {
            format!("i31.get_s {} -> r{}", mval(src), dst.0)
        }
        MachineInstKind::I31GetU { src, dst } => {
            format!("i31.get_u {} -> r{}", mval(src), dst.0)
        }
        MachineInstKind::AnyConvertExtern { src, dst } => {
            format!("any.convert_extern {} -> r{}", mval(src), dst.0)
        }
        MachineInstKind::ExternConvertAny { src, dst } => {
            format!("extern.convert_any {} -> r{}", mval(src), dst.0)
        }
        MachineInstKind::RefTest { ref_type, src, dst } => {
            format!("ref.test {} {} -> r{}", ref_type, mval(src), dst.0)
        }
        MachineInstKind::RefCast { ref_type, src, dst } => {
            format!("ref.cast {} {} -> r{}", ref_type, mval(src), dst.0)
        }
        MachineInstKind::StructNew {
            type_idx,
            fields,
            dst,
        } => {
            let args = fields
                .iter()
                .map(|(value_lo, value_hi)| {
                    if let Some(value_hi) = value_hi {
                        format!("{}, {}", mval(value_lo), mval(value_hi))
                    } else {
                        mval(value_lo)
                    }
                })
                .collect::<collections::Vec<_>>();
            format!(
                "struct.new type={} [{}] -> r{}",
                type_idx,
                args.join(", "),
                dst.0
            )
        }
        MachineInstKind::StructNewDefault { type_idx, dst } => {
            format!("struct.new_default type={} -> r{}", type_idx, dst.0)
        }
        MachineInstKind::StructGet {
            type_idx,
            field_idx,
            signed,
            ty,
            src,
            dst,
            dst_hi,
        } => {
            let op = match signed {
                None => "struct.get",
                Some(true) => "struct.get_s",
                Some(false) => "struct.get_u",
            };
            let result = if let Some(dst_hi) = dst_hi {
                format!("r{},r{}", dst.0, dst_hi.0)
            } else {
                format!("r{}", dst.0)
            };
            format!(
                "{} type={} field={} {:?} {} -> {}",
                op,
                type_idx,
                field_idx,
                ty,
                mval(src),
                result
            )
        }
        MachineInstKind::StructSet {
            type_idx,
            field_idx,
            ref_src,
            value_lo,
            value_hi,
        } => {
            let value = if let Some(value_hi) = value_hi {
                format!("{}, {}", mval(value_lo), mval(value_hi))
            } else {
                mval(value_lo)
            };
            format!(
                "struct.set type={} field={} {} {}",
                type_idx,
                field_idx,
                mval(ref_src),
                value
            )
        }
        MachineInstKind::ArrayNewDefault {
            type_idx,
            length,
            dst,
        } => {
            format!(
                "array.new_default type={} {} -> r{}",
                type_idx,
                mval(length),
                dst.0
            )
        }
        MachineInstKind::ArrayNew {
            type_idx,
            init_lo,
            init_hi,
            length,
            dst,
        } => {
            let init = if let Some(init_hi) = init_hi {
                format!("{}, {}", mval(init_lo), mval(init_hi))
            } else {
                mval(init_lo)
            };
            format!(
                "array.new type={} {} {} -> r{}",
                type_idx,
                init,
                mval(length),
                dst.0
            )
        }
        MachineInstKind::ArrayNewFixed {
            type_idx,
            elements,
            dst,
        } => {
            let elems = elements
                .iter()
                .map(|(value_lo, value_hi)| {
                    if let Some(value_hi) = value_hi {
                        format!("{}, {}", mval(value_lo), mval(value_hi))
                    } else {
                        mval(value_lo)
                    }
                })
                .collect::<collections::Vec<_>>();
            format!(
                "array.new_fixed type={} [{}] -> r{}",
                type_idx,
                elems.join(", "),
                dst.0
            )
        }
        MachineInstKind::ArrayNewData {
            type_idx,
            data_idx,
            src,
            len,
            dst,
        } => {
            format!(
                "array.new_data type={} data={} {} {} -> r{}",
                type_idx,
                data_idx,
                mval(src),
                mval(len),
                dst.0
            )
        }
        MachineInstKind::ArrayNewElem {
            type_idx,
            elem_idx,
            src,
            len,
            dst,
        } => {
            format!(
                "array.new_elem type={} elem={} {} {} -> r{}",
                type_idx,
                elem_idx,
                mval(src),
                mval(len),
                dst.0
            )
        }
        MachineInstKind::ArrayGet {
            type_idx,
            signed,
            ty,
            ref_src,
            index,
            dst,
            dst_hi,
        } => {
            let op = match signed {
                None => "array.get",
                Some(true) => "array.get_s",
                Some(false) => "array.get_u",
            };
            let result = if let Some(dst_hi) = dst_hi {
                format!("r{},r{}", dst.0, dst_hi.0)
            } else {
                format!("r{}", dst.0)
            };
            format!(
                "{} type={} {:?} {} {} -> {}",
                op,
                type_idx,
                ty,
                mval(ref_src),
                mval(index),
                result
            )
        }
        MachineInstKind::ArraySet {
            type_idx,
            ref_src,
            index,
            value_lo,
            value_hi,
        } => {
            let value = if let Some(value_hi) = value_hi {
                format!("{}, {}", mval(value_lo), mval(value_hi))
            } else {
                mval(value_lo)
            };
            format!(
                "array.set type={} {} {} {}",
                type_idx,
                mval(ref_src),
                mval(index),
                value
            )
        }
        MachineInstKind::ArrayFill {
            type_idx,
            ref_src,
            index,
            value_lo,
            value_hi,
            len,
        } => {
            let value = if let Some(value_hi) = value_hi {
                format!("{}, {}", mval(value_lo), mval(value_hi))
            } else {
                mval(value_lo)
            };
            format!(
                "array.fill type={} {} {} {} {}",
                type_idx,
                mval(ref_src),
                mval(index),
                value,
                mval(len)
            )
        }
        MachineInstKind::ArrayCopy {
            dst_type_idx,
            src_type_idx,
            dst_ref,
            dst_index,
            src_ref,
            src_index,
            len,
        } => {
            format!(
                "array.copy dst_type={} src_type={} {} {} {} {} {}",
                dst_type_idx,
                src_type_idx,
                mval(dst_ref),
                mval(dst_index),
                mval(src_ref),
                mval(src_index),
                mval(len)
            )
        }
        MachineInstKind::ArrayInitData {
            type_idx,
            data_idx,
            ref_src,
            dst_index,
            src_index,
            len,
        } => {
            format!(
                "array.init_data type={} data={} {} {} {} {}",
                type_idx,
                data_idx,
                mval(ref_src),
                mval(dst_index),
                mval(src_index),
                mval(len)
            )
        }
        MachineInstKind::ArrayInitElem {
            type_idx,
            elem_idx,
            ref_src,
            dst_index,
            src_index,
            len,
        } => {
            format!(
                "array.init_elem type={} elem={} {} {} {} {}",
                type_idx,
                elem_idx,
                mval(ref_src),
                mval(dst_index),
                mval(src_index),
                mval(len)
            )
        }
        MachineInstKind::ArrayLen { src, dst } => {
            format!("array.len {} -> r{}", mval(src), dst.0)
        }
    }
}

fn owner_tag(owner: MachineRegOwner) -> &'static str {
    match owner {
        MachineRegOwner::LinearValue => "linear",
        MachineRegOwner::CachedLocal => "cache",
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
            let targets: collections::Vec<String> =
                entries.iter().map(|e| format!("b{}", e.target.0)).collect();
            format!("jump_table {} [{}]", mval(index), targets.join(", "))
        }
        MachineTerminator::Call {
            target,
            frame_delta,
            args,
            success,
            ..
        } => match target {
            MachineCallTarget::Direct(callee) => format!(
                "call f{} frame_delta={} frame_params={} lane_args=[{}] success=b{} [{}]",
                callee.0,
                frame_delta,
                args.frame_params.slots,
                render_machine_call_args(args),
                success.target.0,
                medge_args(&success.args)
            ),
            MachineCallTarget::Indirect {
                callee_target,
                callee_entry,
            } => format!(
                "call target=r{} entry=r{} frame_delta={} frame_params={} lane_args=[{}] success=b{} [{}]",
                callee_target.0,
                callee_entry.0,
                frame_delta,
                args.frame_params.slots,
                render_machine_call_args(args),
                success.target.0,
                medge_args(&success.args)
            ),
        },
        MachineTerminator::TailCall { target, args } => match target {
            MachineCallTarget::Direct(callee) => {
                format!(
                    "tail_call f{} frame_params={} lane_args=[{}]",
                    callee.0,
                    args.frame_params.slots,
                    render_machine_call_args(args)
                )
            }
            MachineCallTarget::Indirect {
                callee_target,
                callee_entry,
            } => format!(
                "tail_call target=r{} entry=r{} frame_params={} lane_args=[{}]",
                callee_target.0,
                callee_entry.0,
                args.frame_params.slots,
                render_machine_call_args(args)
            ),
        },
        MachineTerminator::Return => "return".into(),
        MachineTerminator::ReturnScalar { value } => {
            format!("return_scalar {}", render_machine_return_value(value))
        }
        MachineTerminator::Trap { kind } => format!("trap {:?}", kind),
    }
}

fn render_machine_return_value(
    value: &crate::vm::machine::machine_ir::MachineReturnValue,
) -> String {
    use crate::vm::machine::machine_ir::MachineReturnValue;
    match value {
        MachineReturnValue::ScalarGp { src, ty } => {
            format!("gp({:?})={}", ty, render_machine_result_src(*src))
        }
        MachineReturnValue::ScalarGpPair { lo, hi } => format!(
            "gppair=({}, {})",
            render_machine_result_src(*lo),
            render_machine_result_src(*hi)
        ),
        MachineReturnValue::ScalarFp { src, ty } => {
            format!("fp({:?})={}", ty, render_machine_result_src(*src))
        }
    }
}

fn render_machine_result_src(src: crate::vm::machine::machine_ir::MachineResultSrc) -> String {
    use crate::vm::machine::machine_ir::MachineResultSrc;
    match src {
        MachineResultSrc::Reg(reg) => format!("r{}", reg.0),
        MachineResultSrc::FrameSlot(slot) => format!("fp[{}]", slot.0),
        MachineResultSrc::FrameSlotOffset { slot, byte_offset } => {
            format!("fp[{}]+{}", slot.0, byte_offset)
        }
    }
}

fn render_machine_call_args(args: &MachineCallArgs) -> String {
    args.lane_args
        .iter()
        .map(|arg| match arg {
            MachineCallLaneArg::Gp {
                param_index,
                lane,
                src,
                ..
            } => format!("p{}=gp{}:{}", param_index, lane, render_arg_src(src)),
            MachineCallLaneArg::GpPair {
                param_index,
                lo_lane,
                hi_lane,
                src,
            } => format!(
                "p{}=gp{}+gp{}:{}",
                param_index,
                lo_lane,
                hi_lane,
                render_arg_src_pair(src)
            ),
            MachineCallLaneArg::Fp {
                param_index,
                lane,
                src,
                ..
            } => format!("p{}=fp{}:{}", param_index, lane, render_arg_src(src)),
        })
        .collect::<collections::Vec<_>>()
        .join(", ")
        .into()
}

fn render_arg_src_pair(src: &MachineArgSrcPair) -> String {
    format!("{}+{}", render_arg_src(&src.lo), render_arg_src(&src.hi))
}

fn render_arg_src(src: &MachineArgSrc) -> String {
    match src {
        MachineArgSrc::Reg(reg) => format!("r{}", reg.0),
        MachineArgSrc::FrameSlot(slot) => format!("fp[{}]", slot.0),
        MachineArgSrc::FrameSlotOffset { slot, byte_offset } => {
            format!("fp[{}+{}]", slot.0, byte_offset)
        }
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
        MachineBranchCond::TestBits {
            width,
            kind,
            src,
            mask,
        } => format!("tst.{:?}.{} {}, {}", kind, iw(width), mval(src), mval(mask)),
    }
}

// ---- formatting helpers ----

fn mval(v: &MachineValue) -> String {
    match v {
        MachineValue::Reg(r) => format!("r{}", r.0),
        MachineValue::ReservedReg(r) => format!("reserve(r{})", r.0),
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
    args.iter()
        .map(mval)
        .collect::<collections::Vec<_>>()
        .join(", ")
        .into()
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
        MachineStorageType::V128 => "v128",
    }
}
