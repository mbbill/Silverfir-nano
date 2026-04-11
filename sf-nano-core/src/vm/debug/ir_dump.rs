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
            MachineAddr, MachineBranchCond, MachineFloatWidth, MachineFunction, MachineInstKind,
            MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineModule,
            MachineModuleAbi, MachineSign, MachineStorageType, MachineTerminator, MachineValue,
        },
        middle::ssa_ir::ir::{
            SsaBlock, SsaCallOp, SsaInstKind, SsaProgram, SsaTerminator, SsaValue,
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
        render_lir_block(out, block, &cached);
    }
}

fn render_lir_block(
    out: &mut String,
    block: &SsaBlock,
    cached: &[crate::vm::middle::frame::FrameSlot],
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
        SsaInstKind::LocalGetSlot { slot, dst } => {
            format!("local.get_slot v{} <- fp[{}]", dst.0, slot.0)
        }
        SsaInstKind::LocalGetCache { slot, dst } => {
            format!("local.get_cache v{} <- fp[{}]", dst.0, slot.0)
        }
        SsaInstKind::Fill { slot, dst } => {
            format!("fill v{} <- fp[{}]", dst.0, slot.0)
        }
        SsaInstKind::LocalSetSlot { slot, src } => {
            format!("local.set_slot fp[{}] <- v{}", slot.0, src.0)
        }
        SsaInstKind::LocalSetCache { slot, src } => {
            format!("local.set_cache fp[{}] <- v{}", slot.0, src.0)
        }
        SsaInstKind::LocalEnsureCache { slot } => {
            format!("local.ensure_cache fp[{}]", slot.0)
        }
        SsaInstKind::LocalReserveCache { slot } => {
            format!("local.reserve_cache fp[{}]", slot.0)
        }
        SsaInstKind::LocalDropCache { slot } => {
            format!("local.drop_cache fp[{}]", slot.0)
        }
        SsaInstKind::Spill { slot, src } => {
            format!("spill fp[{}] <- v{}", slot.0, src.0)
        }
        SsaInstKind::Call(bop) => render_call(bop),
    }
}

fn render_call(bop: &SsaCallOp) -> String {
    match bop {
        SsaCallOp::CallDirect {
            callee,
            args,
            results,
        } => format!(
            "call_direct f{callee} args=fp[{}..{}) results=fp[{}..{})",
            args.start.0,
            args.start.0 + args.count,
            results.start.0,
            results.start.0 + results.count,
        ),
        SsaCallOp::CallIndirect {
            type_idx,
            table_idx,
            index_slot,
            args,
            results,
        } => format!(
            "call_indirect type={type_idx} table={table_idx} index=fp[{}] args=fp[{}..{}) results=fp[{}..{})",
            index_slot.0,
            args.start.0,
            args.start.0 + args.count,
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
        SsaTerminator::TrapUnreachable => "trap_unreachable".into(),
    }
}

fn render_lir_bindings(bindings: &[crate::vm::middle::ssa_ir::ir::SsaBinding]) -> String {
    bindings
        .iter()
        .map(|b| format!("v{}=v{}", b.param.0, b.value.0))
        .collect::<collections::Vec<_>>()
        .join(", ")
}

fn vals(vs: &[SsaValue]) -> String {
    vs.iter()
        .map(|v| format!("v{}", v.0))
        .collect::<collections::Vec<_>>()
        .join(", ")
}

fn operands(ops: &[crate::vm::middle::ssa_ir::ir::SsaOperand]) -> String {
    ops.iter()
        .map(|op| match op {
            crate::vm::middle::ssa_ir::ir::SsaOperand::Value(v) => format!("v{}", v.0),
            crate::vm::middle::ssa_ir::ir::SsaOperand::Const(bits) => format!("#{bits}"),
        })
        .collect::<collections::Vec<_>>()
        .join(", ")
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
        MachineInstKind::CallExternal(call) => {
            format!("call_external const={}", call.metadata.0)
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
    }
}

fn owner_tag(owner: crate::vm::machine::machine_ir::MachineRegOwner) -> &'static str {
    match owner {
        crate::vm::machine::machine_ir::MachineRegOwner::LinearValue => "linear",
        crate::vm::machine::machine_ir::MachineRegOwner::CachedLocal => "cache",
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
        MachineTerminator::CallDirect {
            callee,
            callee_frame_base,
            caller_result_base,
            continuation,
        } => {
            format!(
                "call_direct f{} frame_base=r{} caller_result_base=r{} cont=b{}",
                callee.0, callee_frame_base.0, caller_result_base.0, continuation.0
            )
        }
        MachineTerminator::CallIndirect {
            callee_target,
            callee_entry,
            callee_frame_base,
            caller_result_base,
            continuation,
        } => {
            format!(
                "call_indirect target=r{} entry=r{} frame_base=r{} caller_result_base=r{} cont=b{}",
                callee_target.0,
                callee_entry.0,
                callee_frame_base.0,
                caller_result_base.0,
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
