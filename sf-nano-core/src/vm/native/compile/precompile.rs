//! Native module/function precompile entry.

use alloc::{vec, vec::Vec};

use crate::error::WasmError;
use crate::vm::{
    lir::ir::LirProgram,
    lir::leaf::LirLeafOp,
    native::{
    arch,
    code::NativeCode,
    dump::{self, NativeDumpFunction},
    ir::{NativeHelperInst, NativeInstKind, NativeProgram},
    jitdump,
    lower,
    map,
    symbols,
    },
};
use crate::vm::store::Store;

use super::{
    build::{build_native_function_for_spec, NativeBuildBundle},
    resolve::{self, ResolvedNativeEntry},
};

#[cfg(target_arch = "aarch64")]
use crate::vm::native::{
    arch::arm64::{
        compile::{compile_program_with_mode, Arm64CompileMode},
        enc,
        lower::supports_direct_lowering_base,
    },
    runtime::layout,
};

/// Native precompile result.
#[derive(Debug, Default)]
pub struct NativePrecompiled {
    pub ir: Option<NativeProgram>,
    pub code: Option<NativeCode>,
}

#[derive(Debug)]
struct PendingNativeFunction {
    func_index: usize,
    params_len: u16,
    locals_len: u16,
    results_len: u16,
    lir: LirProgram,
    ir: NativeProgram,
    resolved: Vec<ResolvedNativeEntry>,
}

pub fn precompile_native(bundle: NativeBuildBundle) -> Result<NativePrecompiled, WasmError> {
    let ir = lower::lower_native(
        &bundle.lir,
        &bundle.planned,
        bundle.backend_config,
        &bundle.local_call_contracts,
    )?;
    ir.validate()?;
    let resolved = resolve::resolve_native(&ir);
    let code = arch::compile_native(&ir, &resolved)?;

    Ok(NativePrecompiled {
        ir: Some(ir),
        code: Some(code),
    })
}

pub fn precompile_module(store: &Store) -> Result<(), WasmError> {
    let module = store.module();

    let all_compiled = module
        .functions
        .iter()
        .filter(|func| !func.is_external())
        .all(|func| {
            func.spec()
                .map(|spec| spec.has_native_code())
                .unwrap_or(true)
        });
    if all_compiled {
        if dump::dump_enabled() {
            dump::write_module_dump(module, &[])?;
        }
        return Ok(());
    }

    let mut pending = Vec::new();
    for (func_index, func) in module
        .functions
        .iter()
        .enumerate()
        .filter(|(_, func)| !func.is_external())
    {
        let Some(spec) = func.spec() else {
            continue;
        };
        if spec.has_native_code() {
            continue;
        }

        let params_len = spec.func_type().params().len() as u16;
        let locals_len = spec.locals().len() as u16;
        let results_len = spec.func_type().results().len() as u16;
        let bundle = build_native_function_for_spec(
            spec.code(),
            store,
            module,
            params_len,
            params_len.saturating_add(locals_len),
            results_len,
        )
        .map_err(|err| {
            WasmError::internal(alloc::format!(
                "native build failed for function {}: {}",
                func_index,
                err
            ))
        })?;
        let ir = lower::lower_native(
            &bundle.lir,
            &bundle.planned,
            bundle.backend_config,
            &bundle.local_call_contracts,
        )
        .map_err(|err| {
            WasmError::internal(alloc::format!(
                "native lowering failed for function {}: {}",
                func_index,
                err
            ))
        })?;
        ir.validate().map_err(|err| {
            WasmError::internal(alloc::format!(
                "native IR validation failed for function {}: {}",
                func_index,
                err
            ))
        })?;
        pending.push(PendingNativeFunction {
            func_index,
            params_len,
            locals_len,
            results_len,
            lir: bundle.lir,
            resolved: resolve::resolve_native(&ir),
            ir,
        });
    }

    #[cfg(target_arch = "aarch64")]
    let direct_arm64 = compute_direct_arm64(module, &pending);
    #[cfg(target_arch = "aarch64")]
    let callee_may_change_views = compute_callee_view_change_flags(module, &pending);

    for pending_func in &pending {
        #[cfg(target_arch = "aarch64")]
        let code = {
            let mode = if direct_arm64
                .get(pending_func.func_index)
                .copied()
                .unwrap_or(false)
            {
                Arm64CompileMode::Direct
            } else {
                Arm64CompileMode::SharedEntry
            };
            compile_program_with_mode(
                &pending_func.ir,
                &pending_func.resolved,
                mode,
                Some(&callee_may_change_views),
            )
        };

        #[cfg(not(target_arch = "aarch64"))]
        let code = arch::compile_native(&pending_func.ir, &pending_func.resolved);

        let code = code.map_err(|err| {
            WasmError::internal(alloc::format!(
                "native codegen failed for function {}: {}",
                pending_func.func_index,
                err
            ))
        })?;
        let cache = code.build_cache(
            pending_func.params_len as usize,
            pending_func.locals_len as usize,
            pending_func.results_len as usize,
        );
        let spec = module.functions[pending_func.func_index]
            .spec()
            .ok_or_else(|| WasmError::internal("local function is missing spec".into()))?;
        spec.set_native_code(code, cache);
    }

    #[cfg(target_arch = "aarch64")]
    finalize_arm64_direct_calls(module)?;

    record_native_symbols(module);
    let dump_functions = pending
        .iter()
        .map(|pending_func| NativeDumpFunction {
            func_idx: pending_func.func_index,
            lir: &pending_func.lir,
        })
        .collect::<Vec<_>>();
    dump::write_module_dump(module, &dump_functions)?;

    Ok(())
}

#[cfg(target_arch = "aarch64")]
fn compute_direct_arm64(module: &crate::vm::entities::ModuleInst, pending: &[PendingNativeFunction]) -> Vec<bool> {
    let mut direct = vec![false; module.functions.len()];

    for (index, func) in module.functions.iter().enumerate() {
        if let Some(spec) = func.spec() {
            if spec.has_native_code() {
                direct[index] = spec.native_cache().internal_entry().is_some();
            }
        }
    }

    for pending_func in pending {
        direct[pending_func.func_index] = supports_direct_lowering_base(&pending_func.ir);
    }

    loop {
        let mut changed = false;
        for pending_func in pending {
            if !direct[pending_func.func_index] {
                continue;
            }
            if local_callees(&pending_func.ir)
                .into_iter()
                .any(|callee| !direct.get(callee as usize).copied().unwrap_or(false))
            {
                direct[pending_func.func_index] = false;
                changed = true;
            }
        }
        if !changed {
            return direct;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn local_callees(program: &NativeProgram) -> Vec<u32> {
    let mut out = Vec::new();
    for block in &program.blocks {
        for inst in &block.ops {
            if let NativeInstKind::CallLocal { callee, .. } = inst.kind {
                if !out.contains(&callee) {
                    out.push(callee);
                }
            }
        }
    }
    out
}

#[cfg(target_arch = "aarch64")]
fn compute_callee_view_change_flags(
    module: &crate::vm::entities::ModuleInst,
    pending: &[PendingNativeFunction],
) -> Vec<bool> {
    let mut flags = vec![false; module.functions.len()];
    let mut programs: Vec<Option<&NativeProgram>> = vec![None; module.functions.len()];

    for (index, func) in module.functions.iter().enumerate() {
        let Some(spec) = func.spec() else {
            continue;
        };
        let Some(code) = spec.get_native_code() else {
            continue;
        };
        let Some(program) = code.program() else {
            continue;
        };
        flags[index] = program_self_changes_views(program);
        programs[index] = Some(program);
    }

    for pending_func in pending {
        flags[pending_func.func_index] = program_self_changes_views(&pending_func.ir);
        programs[pending_func.func_index] = Some(&pending_func.ir);
    }

    loop {
        let mut changed = false;
        for (index, program) in programs.iter().enumerate() {
            let Some(program) = program else {
                continue;
            };
            if flags[index] {
                continue;
            }
            if local_callees(program)
                .into_iter()
                .any(|callee| flags.get(callee as usize).copied().unwrap_or(true))
            {
                flags[index] = true;
                changed = true;
            }
        }
        if !changed {
            return flags;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn program_self_changes_views(program: &NativeProgram) -> bool {
    for block in &program.blocks {
        for inst in &block.ops {
            match &inst.kind {
                NativeInstKind::Leaf { op, .. } => {
                    if leaf_changes_views(op) {
                        return true;
                    }
                }
                NativeInstKind::Helper(NativeHelperInst::Leaf { op, .. }) => {
                    if leaf_changes_views(op) {
                        return true;
                    }
                }
                NativeInstKind::CallIndirect { .. } => return true,
                NativeInstKind::Move(_)
                | NativeInstKind::CallExternal { .. }
                | NativeInstKind::CallLocal { .. } => {}
            }
        }
    }
    false
}

#[cfg(target_arch = "aarch64")]
fn leaf_changes_views(op: &LirLeafOp) -> bool {
    matches!(op, LirLeafOp::MemoryGrow { .. })
}

#[cfg(target_arch = "aarch64")]
fn finalize_arm64_direct_calls(module: &crate::vm::entities::ModuleInst) -> Result<(), WasmError> {
    for func in module.functions.iter().filter(|func| !func.is_external()) {
        let Some(spec) = func.spec() else {
            continue;
        };

        if let Some(result) = spec.with_native_code_mut(|code| -> Result<(), WasmError> {
            let patches = code.direct_call_patches().to_vec();
            if patches.is_empty() {
                return Ok(());
            }
            let Some(executable) = code.executable.as_mut() else {
                return Ok(());
            };

            let mut patch_start = usize::MAX;
            let mut patch_end = 0usize;
            executable.begin_write();
            for patch in &patches {
                let callee = module
                    .functions
                    .get(patch.callee_func_idx as usize)
                    .ok_or_else(|| {
                        WasmError::internal(alloc::format!(
                            "direct call patch references missing callee {}",
                            patch.callee_func_idx
                        ))
                    })?;
                let callee_spec = callee.spec().ok_or_else(|| {
                    WasmError::internal("direct call patch targets external callee".into())
                })?;
                let target = callee_spec.native_cache().internal_entry().ok_or_else(|| {
                    WasmError::internal(alloc::format!(
                        "direct call patch targets non-direct callee {}",
                        patch.callee_func_idx
                    ))
                })?;
                let callee_code = callee_spec.get_native_code().ok_or_else(|| {
                    WasmError::internal("direct callee is missing native code".into())
                })?;
                let callee_program = callee_code.program().ok_or_else(|| {
                    WasmError::internal("direct callee is missing native program".into())
                })?;
                let branch_site =
                    unsafe { executable.base_ptr().add(patch.entry_branch_inst_offset as usize) as usize };
                let delta_bytes = target as isize - branch_site as isize;
                let delta_words = delta_bytes / 4;
                if delta_bytes % 4 == 0
                    && (-(1 << 25)..(1 << 25)).contains(&delta_words)
                {
                    executable.patch_u32(
                        patch.entry_branch_inst_offset as usize,
                        enc::b(delta_words as i32),
                    );
                    executable.patch_u32(
                        patch.entry_branch_reg_offset as usize,
                        enc::nop(),
                    );
                    patch_start = patch_start.min(
                        patch.entry_branch_inst_offset
                            .min(patch.entry_branch_reg_offset)
                            .min(patch.frame_slots_literal_offset) as usize,
                    );
                    patch_end = patch_end.max(
                        patch.entry_branch_inst_offset
                            .max(patch.entry_branch_reg_offset)
                            .max(patch.frame_slots_literal_offset) as usize
                            + 8,
                    );
                } else {
                    executable.patch_u64(patch.entry_literal_offset as usize, target as usize as u64);
                    patch_start = patch_start.min(
                        patch.entry_literal_offset.min(patch.frame_slots_literal_offset) as usize,
                    );
                    patch_end = patch_end.max(
                        patch.entry_literal_offset.max(patch.frame_slots_literal_offset) as usize + 8,
                    );
                }
                executable.patch_u64(
                    patch.frame_slots_literal_offset as usize,
                    layout::frame_slots_used(callee_program) as u64,
                );
            }
            if patch_start == usize::MAX {
                executable.finish_write(0, 0);
            } else {
                executable.finish_write(patch_start, patch_end - patch_start);
            }
            Ok(())
        }) {
            result?;
        }
    }

    Ok(())
}

fn record_native_symbols(module: &crate::vm::entities::ModuleInst) {
    for (func_index, func) in module
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
        let Some(program) = code.program() else {
            continue;
        };

        let Some(region) = code.region() else {
            continue;
        };
        let Some(base) = code.executable.as_ref().map(|executable| executable.base_ptr()) else {
            continue;
        };
        let code_start = unsafe { base.add(region.start as usize) };
        let len = region.len as usize;

        let stats = symbols::summarize_program(program);
        if code.debug_regions().is_empty() {
            record_native_symbol_segment(
                code_start,
                0,
                len,
                &module.name,
                func_index as u32,
                &symbols::render_symbol_name(&module.name, func_index as u32, "function"),
                "function",
                false,
                stats,
            );
        } else {
            for region in code.debug_regions() {
                let symbol_name =
                    symbols::render_region_symbol_name(&module.name, func_index as u32, program, region);
                let segment = symbols::render_region_label(program, region);
                record_native_symbol_segment(
                    code_start,
                    region.offset as usize,
                    region.len as usize,
                    &module.name,
                    func_index as u32,
                    &symbol_name,
                    &segment,
                    code.internal_entry().is_some(),
                    stats,
                );
            }
        }
    }
}

fn record_native_symbol_segment(
    base: *const u8,
    offset: usize,
    len: usize,
    module_name: &str,
    func_idx: u32,
    symbol_name: &str,
    segment: &str,
    direct: bool,
    stats: symbols::NativeSymbolStats,
) {
    if len == 0 {
        return;
    }

    let code_start = unsafe { base.add(offset) };
    let code_bytes = unsafe { core::slice::from_raw_parts(code_start, len) };

    map::record_function(
        code_start,
        len,
        module_name,
        func_idx,
        segment,
        direct,
        stats,
    );
    jitdump::record_function(code_start, code_bytes, symbol_name);
}
