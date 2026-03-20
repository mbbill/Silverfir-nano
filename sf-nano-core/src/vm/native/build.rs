use alloc::{rc::Rc, vec::Vec};

use crate::{
    error::WasmError,
    vm::{
        native::{
            arch,
            code::{CompiledNativeModule, NativeCode, NativeCodeCache},
            ir::machine::MachineFuncId,
            ir_dump,
            lower::{lower_module, LowerFunctionInput, LowerModuleInput},
        },
        plan::{config::PlanConfig, prepare::PrepareInput, prepare_function, PreparedFunction},
        store::Store,
        wasm::{context::CompileContext, decode, inline, semantic_ir::SemanticProgram},
    },
};

#[inline]
pub const fn native_plan_config(backend: crate::vm::backend::BackendConfig) -> PlanConfig {
    PlanConfig::from_backend_config(backend, 3)
}

pub fn ensure_module_compiled(store: &Store) -> Result<(), WasmError> {
    let active_backend = arch::active_native_backend()
        .map_err(|err| WasmError::invalid(alloc::format!("native backend unavailable: {err}")))?;
    let backend = arch::active_backend_config()
        .map_err(|err| WasmError::invalid(alloc::format!("native backend unavailable: {err}")))?;
    let module = store.module();
    let all_compiled = module
        .functions
        .iter()
        .filter_map(|func| func.spec())
        .all(|spec| {
            spec.get_native_code()
                .map(|code| {
                    code.compiled().backend_kind() == active_backend
                        && code.compiled().backend() == backend
                })
                .unwrap_or(false)
        });
    if all_compiled {
        return Ok(());
    }

    // Phase 1: Decode all functions to semantic IR.
    let mut semantics: Vec<Option<SemanticProgram>> = Vec::with_capacity(module.functions.len());
    for (func_idx, func) in module.functions.iter().enumerate() {
        let Some(spec) = func.spec() else {
            semantics.push(None);
            continue;
        };
        let params = spec.func_type().params().len() as u16;
        let local_count = params.saturating_add(spec.locals().len() as u16);
        let results = spec.func_type().results().len() as u16;
        let mut local_types = Vec::with_capacity(local_count as usize);
        local_types.extend_from_slice(spec.func_type().params());
        local_types.extend_from_slice(spec.locals());
        let semantic = decode::decode_to_semantic_ir(
            spec.code(),
            CompileContext::with_value_types(
                &module.types,
                store,
                module,
                params,
                local_count,
                results,
                &local_types,
                spec.func_type().results(),
            ),
        )
        .map_err(|err| {
            WasmError::internal(alloc::format!(
                "native decode failed for function {}: {}",
                func_idx,
                err
            ))
        })?;
        semantics.push(Some(semantic));
    }

    // Phase 2: Inline small leaf callees into their callers.
    // Iterate until fixed-point so that transitive chains (A→B→C) are fully
    // resolved regardless of function index ordering.
    loop {
        let mut any_inlined = false;
        for func_idx in 0..semantics.len() {
            if semantics[func_idx].is_none() {
                continue;
            }
            let mut caller = semantics[func_idx].take().unwrap();
            if inline::inline_calls_in_function(&mut caller, func_idx as u32, &semantics) {
                any_inlined = true;
            }
            semantics[func_idx] = Some(caller);
        }
        if !any_inlined {
            break;
        }
    }

    // Phase 3: Prepare all functions (frame layout + LIR lowering).
    let mut lowered_inputs = Vec::new();
    let mut prepared_functions = Vec::new();
    for (func_idx, func) in module.functions.iter().enumerate() {
        let Some(spec) = func.spec() else {
            continue;
        };
        let semantic = semantics[func_idx].as_ref().unwrap();
        let prepared =
            prepare_function(PrepareInput { config: native_plan_config(backend) }, semantic).map_err(
                |err| {
                    WasmError::internal(alloc::format!(
                        "native prepare failed for function {} type_idx={} params={} results={} max_stack={} ops={}: {}",
                        func_idx,
                        spec.type_index(),
                        spec.func_type().params().len(),
                        spec.func_type().results().len(),
                        semantic.max_stack_height,
                        semantic.ops.len(),
                        err
                    ))
                },
            )?;
        prepared_functions.push((MachineFuncId(func_idx as u32), prepared));
    }

    for (func_idx, (id, prepared)) in prepared_functions.iter().enumerate() {
        let result_count = module
            .functions
            .get(id.0 as usize)
            .and_then(|f| f.spec())
            .map(|s| s.func_type().results().len() as u16)
            .unwrap_or(0);
        lowered_inputs.push(LowerFunctionInput {
            id: *id,
            frame: prepared.frame,
            lir: &prepared.lir,
            result_count,
        });
    }

    #[cfg(has_guard_pages)]
    let use_guard_pages = module
        .memories
        .first()
        .map(|m| m.has_guard_pages())
        .unwrap_or(false);
    #[cfg(has_guard_pages)]
    let use_guard_pages = use_guard_pages && backend.gp_unit_bytes == 8;
    let mut lowered = lower_module(LowerModuleInput {
        backend,
        functions: &lowered_inputs,
        #[cfg(has_guard_pages)]
        use_guard_pages,
    })?;
    let first_transient = crate::vm::native::ir::machine::MACHINE_FIXED_REG_COUNT
        + backend.gp_local_cache_budget as u16;
    lowered
        .module
        .optimize(first_transient, backend.gp_unit_bytes);
    lowered.module.legalize(backend.gp_unit_bytes)?;
    if backend.needs_32bit_gp_legalization() {
        let max_gp_regs = crate::vm::native::ir::machine::MACHINE_FIXED_REG_COUNT
            + backend.gp_local_cache_budget as u16
            + backend.gp_transient_budget as u16;
        lowered
            .module
            .finalize_32bit_gp_legalization(first_transient, max_gp_regs)?;
    }

    // Collect LIR for dump before moving lowered data
    let dump_lir_inputs: Vec<ir_dump::DumpFunctionLir<'_>> = prepared_functions
        .iter()
        .map(|(id, prepared)| ir_dump::DumpFunctionLir {
            func_idx: id.0,
            lir: &prepared.lir,
        })
        .collect();

    let compiled = Rc::new(CompiledNativeModule::new(
        active_backend,
        backend,
        lowered.module,
        lowered.runtime,
    )?);
    #[cfg(target_arch = "aarch64")]
    let arm64_entries = match active_backend {
        arch::NativeBackend::Arm64 => {
            Some(arch::arm64::compile::compile_module(module, &compiled)?)
        }
        _ => None,
    };
    #[cfg(target_arch = "arm")]
    let armv7a_entries = match active_backend {
        arch::NativeBackend::Armv7a => {
            Some(arch::armv7a::compile::compile_module(module, &compiled)?)
        }
        _ => None,
    };
    #[cfg(target_arch = "x86_64")]
    let x86_64_entries = match active_backend {
        arch::NativeBackend::X86_64 => {
            Some(arch::x86_64::compile::compile_module(module, &compiled)?)
        }
        _ => None,
    };

    // Record compile stats.
    {
        let groups = prepared_functions.len();
        let ops: usize = prepared_functions
            .iter()
            .map(|(_, p)| p.lir.blocks.iter().map(|b| b.ops.len()).sum::<usize>())
            .sum();
        let mut bytes = 0usize;
        #[cfg(target_arch = "aarch64")]
        if let Some(ref entries) = arm64_entries {
            bytes = entries
                .iter()
                .filter_map(|e| e.as_ref().map(|e| e.text_len))
                .sum();
        }
        #[cfg(target_arch = "arm")]
        if let Some(ref entries) = armv7a_entries {
            bytes = entries
                .iter()
                .filter_map(|e| e.as_ref().map(|e| e.text_len))
                .sum();
        }
        #[cfg(target_arch = "x86_64")]
        if let Some(ref entries) = x86_64_entries {
            bytes = entries
                .iter()
                .filter_map(|e| e.as_ref().map(|e| e.text_len))
                .sum();
        }
        crate::vm::native::set_native_stats(groups, ops, bytes);
    }

    // Write dump if SF_NATIVE_DUMP_DIR is set
    if ir_dump::dump_enabled() {
        #[cfg(target_arch = "aarch64")]
        let code_slices: Vec<(u32, &[u8])> = arm64_entries
            .as_ref()
            .map(|entries| {
                entries
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, entry)| {
                        entry.as_ref().map(|e| {
                            let ptr = e.entry as *const u8;
                            let len = e.text_len;
                            (idx as u32, unsafe { core::slice::from_raw_parts(ptr, len) })
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        #[cfg(target_arch = "arm")]
        let code_slices: Vec<(u32, &[u8])> = armv7a_entries
            .as_ref()
            .map(|entries| {
                entries
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, entry)| {
                        entry.as_ref().map(|e| {
                            let ptr = e.entry as *const u8;
                            let len = e.text_len;
                            (idx as u32, unsafe { core::slice::from_raw_parts(ptr, len) })
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        #[cfg(target_arch = "x86_64")]
        let code_slices: Vec<(u32, &[u8])> = x86_64_entries
            .as_ref()
            .map(|entries| {
                entries
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, entry)| {
                        entry.as_ref().map(|e| {
                            let ptr = e.entry as *const u8;
                            let len = e.text_len;
                            (idx as u32, unsafe { core::slice::from_raw_parts(ptr, len) })
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        #[cfg(not(any(target_arch = "aarch64", target_arch = "arm", target_arch = "x86_64")))]
        let code_slices: Vec<(u32, &[u8])> = Vec::new();
        #[cfg(target_arch = "aarch64")]
        let dump_regions: Vec<ir_dump::DumpFunctionRegions> = arm64_entries
            .as_ref()
            .map(|entries| {
                entries
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, entry)| {
                        entry.as_ref().map(|e| ir_dump::DumpFunctionRegions {
                            func_idx: idx as u32,
                            regions: e.debug_regions.clone(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        #[cfg(target_arch = "arm")]
        let dump_regions: Vec<ir_dump::DumpFunctionRegions> = armv7a_entries
            .as_ref()
            .map(|entries| {
                entries
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, entry)| {
                        entry.as_ref().map(|e| ir_dump::DumpFunctionRegions {
                            func_idx: idx as u32,
                            regions: e.debug_regions.clone(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        #[cfg(target_arch = "x86_64")]
        let dump_regions: Vec<ir_dump::DumpFunctionRegions> = x86_64_entries
            .as_ref()
            .map(|entries| {
                entries
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, entry)| {
                        entry.as_ref().map(|e| ir_dump::DumpFunctionRegions {
                            func_idx: idx as u32,
                            regions: e.debug_regions.clone(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        #[cfg(not(any(target_arch = "aarch64", target_arch = "arm", target_arch = "x86_64")))]
        let dump_regions: Vec<ir_dump::DumpFunctionRegions> = Vec::new();
        let _ = ir_dump::write_module_dump(
            &module.name,
            module.functions.len(),
            &dump_lir_inputs,
            compiled.module(),
            compiled.runtime(),
            &code_slices,
            &dump_regions,
        );
    }

    for (func_idx, func) in module.functions.iter().enumerate() {
        let Some(spec) = func.spec() else {
            continue;
        };
        let mut code = NativeCode::new(Rc::clone(&compiled), MachineFuncId(func_idx as u32));
        #[cfg(target_arch = "aarch64")]
        {
            let arm64_entry = arm64_entries
                .as_ref()
                .and_then(|entries| entries.get(func_idx).and_then(|e| e.as_ref()));
            code = code.with_arm64_entry(
                arm64_entry.map(|entry| entry.entry),
                arm64_entry.map(|entry| entry.root_return),
            );
        }
        #[cfg(target_arch = "arm")]
        {
            let armv7a_entry = armv7a_entries
                .as_ref()
                .and_then(|entries| entries.get(func_idx).and_then(|e| e.as_ref()));
            code = code.with_armv7a_entry(
                armv7a_entry.map(|entry| entry.entry),
                armv7a_entry.map(|entry| entry.root_return),
            );
        }
        #[cfg(target_arch = "x86_64")]
        {
            let x86_64_entry = x86_64_entries
                .as_ref()
                .and_then(|entries| entries.get(func_idx).and_then(|e| e.as_ref()));
            code = code.with_x86_64_entry(
                x86_64_entry.map(|entry| entry.entry),
                x86_64_entry.map(|entry| entry.root_return),
            );
        }
        spec.set_native_code(code, NativeCodeCache::compiled());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, format, rc::Rc, string::String, vec, vec::Vec};

    use super::{ensure_module_compiled, native_plan_config};
    use crate::{
        module::{
            entities::{FunctionDef, FunctionSpec},
            type_context::TypeContext,
            type_defs::FunctionType,
            Module,
        },
        utils::limits::Limitable,
        value_type::{RefType, ValueType},
        vm::{
            entities::{FunctionInst, ModuleInst},
            instance::{Import, Instance},
            lir::ir::{LirBoundaryOp, LirInstKind, LirProgram, LirValue},
            native::{
                arch::{active_backend_config, backend_mode_test_lock, set_reference_backend_mode},
                ir::machine::{
                    debug_describe_gp_compaction, MachineBranchCond, MachineFuncId,
                    MachineInstKind, MachineProgram, MachineReg, MachineTerminator, MachineValue,
                },
                lower::{lower_module, LowerFunctionInput, LowerModuleInput},
            },
            plan::{prepare::PrepareInput, prepare_function},
            store::Store,
            wasm::{context::CompileContext, decode, inline, semantic_ir::SemanticProgram},
        },
        ReferenceBackendMode,
    };
    use std::{eprintln, fs, path::PathBuf};

    struct ReferenceBackendGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl Drop for ReferenceBackendGuard {
        fn drop(&mut self) {
            set_reference_backend_mode(ReferenceBackendMode::Disabled)
                .expect("reset reference backend mode");
        }
    }

    fn enable_reference_backend_mode(mode: ReferenceBackendMode) -> ReferenceBackendGuard {
        let lock = backend_mode_test_lock()
            .lock()
            .expect("backend mode test lock");
        set_reference_backend_mode(mode).expect("enable reference backend mode");
        ReferenceBackendGuard { _lock: lock }
    }

    fn encode_u32_leb(mut value: u32, out: &mut Vec<u8>) {
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                break;
            }
        }
    }

    fn local_get_code(index: u32) -> Vec<u8> {
        let mut code = vec![0x20];
        encode_u32_leb(index, &mut code);
        code.push(0x0b);
        code
    }

    fn long_argument_caller_code(helper_params: &[ValueType]) -> Vec<u8> {
        let mut code = Vec::new();
        for (index, ty) in helper_params.iter().copied().enumerate() {
            if index + 1 == helper_params.len() {
                code.extend_from_slice(&[0x20, 0x00]);
                continue;
            }
            match ty {
                ValueType::I32 => code.extend_from_slice(&[0x41, 0x00]),
                ValueType::I64 => code.extend_from_slice(&[0x42, 0x00]),
                ValueType::F32 => code.extend_from_slice(&[0x43, 0x00, 0x00, 0x00, 0x00]),
                ValueType::F64 => {
                    code.extend_from_slice(&[0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
                }
                other => panic!("unexpected helper param type in regression: {:?}", other),
            }
        }
        code.extend_from_slice(&[0x10, 0x00, 0x0b]);
        code
    }

    fn max_lir_live_bank_usage(program: &LirProgram, gp_unit_bytes: u8) -> (usize, usize) {
        let mut live = Vec::<LirValue>::new();
        let mut max_gp = 0usize;
        let mut max_fp = 0usize;
        for block in &program.blocks {
            live.clear();
            for inst in &block.ops {
                match &inst.kind {
                    LirInstKind::LoadSlot { dst, .. } => live.push(*dst),
                    LirInstKind::Value { args, results, .. } => {
                        for arg in args {
                            if let Some(index) = live.iter().position(|value| *value == *arg) {
                                live.remove(index);
                            }
                        }
                        live.extend(results.iter().copied());
                    }
                    LirInstKind::StoreSlot { src, .. } => {
                        if let Some(index) = live.iter().position(|value| *value == *src) {
                            live.remove(index);
                        }
                    }
                    LirInstKind::Boundary(boundary) => match boundary {
                        LirBoundaryOp::CallInternal { .. }
                        | LirBoundaryOp::CallExternal { .. }
                        | LirBoundaryOp::CallIndirect { .. }
                        | LirBoundaryOp::MemoryGrow { .. }
                        | LirBoundaryOp::MemoryFill { .. }
                        | LirBoundaryOp::MemoryCopy { .. }
                        | LirBoundaryOp::TableGrow { .. }
                        | LirBoundaryOp::TableFill { .. }
                        | LirBoundaryOp::TableCopy { .. }
                        | LirBoundaryOp::MemoryInit { .. }
                        | LirBoundaryOp::DataDrop { .. }
                        | LirBoundaryOp::TableInit { .. }
                        | LirBoundaryOp::ElemDrop { .. } => {
                            live.clear();
                        }
                    },
                }
                let (gp_live, fp_live) = live.iter().fold((0usize, 0usize), |mut acc, value| {
                    match program.value_types.get(value.0 as usize).copied() {
                        Some(ValueType::F32) | Some(ValueType::F64) => acc.1 += 1,
                        Some(ty) => {
                            let units = if matches!(ty, ValueType::I64) && gp_unit_bytes == 4 {
                                2
                            } else {
                                1
                            };
                            acc.0 += units;
                        }
                        None => acc.0 += 1,
                    }
                    acc
                });
                max_gp = max_gp.max(gp_live);
                max_fp = max_fp.max(fp_live);
            }
        }
        (max_gp, max_fp)
    }

    fn push_gp_use(uses: &mut Vec<MachineReg>, reg: MachineReg, first_fp_reg: u16) {
        if reg.0 < first_fp_reg {
            uses.push(reg);
        }
    }

    fn push_gp_value_use(uses: &mut Vec<MachineReg>, value: MachineValue, first_fp_reg: u16) {
        if let MachineValue::Reg(reg) = value {
            push_gp_use(uses, reg, first_fp_reg);
        }
    }

    fn collect_branch_cond_gp_uses_for_debug(
        cond: &MachineBranchCond,
        first_fp_reg: u16,
        uses: &mut Vec<MachineReg>,
    ) {
        match cond {
            MachineBranchCond::Value(value) => push_gp_value_use(uses, *value, first_fp_reg),
            MachineBranchCond::IntCompare { lhs, rhs, .. }
            | MachineBranchCond::FloatCompare { lhs, rhs, .. } => {
                push_gp_value_use(uses, *lhs, first_fp_reg);
                push_gp_value_use(uses, *rhs, first_fp_reg);
            }
        }
    }

    fn collect_inst_gp_uses_defs_for_debug(
        inst: &MachineInstKind,
        first_fp_reg: u16,
        uses: &mut Vec<MachineReg>,
        defs: &mut Vec<MachineReg>,
    ) {
        match inst {
            MachineInstKind::Move { dst, src, .. } => {
                push_gp_use(defs, *dst, first_fp_reg);
                push_gp_value_use(uses, *src, first_fp_reg);
            }
            MachineInstKind::FloatConst { dst, .. } => {
                push_gp_use(defs, *dst, first_fp_reg);
            }
            MachineInstKind::Load { dst, addr, .. } => {
                push_gp_use(defs, *dst, first_fp_reg);
                push_gp_use(uses, addr.base, first_fp_reg);
            }
            MachineInstKind::Store { addr, src, .. } => {
                push_gp_use(uses, addr.base, first_fp_reg);
                push_gp_value_use(uses, *src, first_fp_reg);
            }
            MachineInstKind::IntBinary { dst, lhs, rhs, .. } => {
                push_gp_use(defs, *dst, first_fp_reg);
                push_gp_value_use(uses, *lhs, first_fp_reg);
                push_gp_value_use(uses, *rhs, first_fp_reg);
            }
            MachineInstKind::TrapIf { cond, .. } => {
                collect_branch_cond_gp_uses_for_debug(cond, first_fp_reg, uses);
            }
            other => panic!("unexpected machine inst in long-arg debug: {:?}", other),
        }
    }

    fn collect_term_gp_uses_for_debug(
        term: &MachineTerminator,
        first_fp_reg: u16,
        uses: &mut Vec<MachineReg>,
    ) {
        match term {
            MachineTerminator::Return | MachineTerminator::Trap { .. } => {}
            MachineTerminator::Jump(edge) => {
                for arg in &edge.args {
                    push_gp_value_use(uses, *arg, first_fp_reg);
                }
            }
            MachineTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                collect_branch_cond_gp_uses_for_debug(cond, first_fp_reg, uses);
                for arg in &then_edge.args {
                    push_gp_value_use(uses, *arg, first_fp_reg);
                }
                for arg in &else_edge.args {
                    push_gp_value_use(uses, *arg, first_fp_reg);
                }
            }
            MachineTerminator::JumpTable { index, entries } => {
                push_gp_value_use(uses, *index, first_fp_reg);
                for edge in entries {
                    for arg in &edge.args {
                        push_gp_value_use(uses, *arg, first_fp_reg);
                    }
                }
            }
            MachineTerminator::CallDirect {
                callee_frame_base, ..
            } => {
                push_gp_use(uses, *callee_frame_base, first_fp_reg);
            }
            MachineTerminator::CallIndirect {
                callee_target,
                callee_frame_base,
                ..
            } => {
                push_gp_value_use(uses, *callee_target, first_fp_reg);
                push_gp_use(uses, *callee_frame_base, first_fp_reg);
            }
        }
    }

    fn max_machine_gp_live_for_debug(program: &MachineProgram) -> usize {
        let gp_reg_count = program.first_fp_reg as usize;
        let block_count = program.blocks.len();
        let mut block_use = vec![vec![false; gp_reg_count]; block_count];
        let mut block_def = vec![vec![false; gp_reg_count]; block_count];

        for block in &program.blocks {
            let block_index = block.id.as_usize();
            for param in &block.params {
                if param.reg.0 < program.first_fp_reg {
                    block_def[block_index][param.reg.0 as usize] = true;
                }
            }
            for inst in &block.ops {
                let mut uses = Vec::new();
                let mut defs = Vec::new();
                collect_inst_gp_uses_defs_for_debug(
                    &inst.kind,
                    program.first_fp_reg,
                    &mut uses,
                    &mut defs,
                );
                for reg in uses {
                    let index = reg.0 as usize;
                    if !block_def[block_index][index] {
                        block_use[block_index][index] = true;
                    }
                }
                for reg in defs {
                    block_def[block_index][reg.0 as usize] = true;
                }
            }
            let mut term_uses = Vec::new();
            collect_term_gp_uses_for_debug(&block.terminator, program.first_fp_reg, &mut term_uses);
            for reg in term_uses {
                let index = reg.0 as usize;
                if !block_def[block_index][index] {
                    block_use[block_index][index] = true;
                }
            }
        }

        let mut live_in = vec![vec![false; gp_reg_count]; block_count];
        let mut live_out = vec![vec![false; gp_reg_count]; block_count];
        let mut changed = true;
        while changed {
            changed = false;
            for block in program.blocks.iter().rev() {
                let block_index = block.id.as_usize();
                let mut new_out = vec![false; gp_reg_count];
                match &block.terminator {
                    MachineTerminator::Jump(edge) => {
                        for (index, live) in
                            live_in[edge.target.as_usize()].iter().copied().enumerate()
                        {
                            new_out[index] |= live;
                        }
                    }
                    MachineTerminator::CallDirect { continuation, .. }
                    | MachineTerminator::CallIndirect { continuation, .. } => {
                        for (index, live) in
                            live_in[continuation.as_usize()].iter().copied().enumerate()
                        {
                            new_out[index] |= live;
                        }
                    }
                    _ => {}
                }
                let mut new_in = block_use[block_index].clone();
                for index in 0..gp_reg_count {
                    new_in[index] |= new_out[index] && !block_def[block_index][index];
                }
                if new_in != live_in[block_index] || new_out != live_out[block_index] {
                    live_in[block_index] = new_in;
                    live_out[block_index] = new_out;
                    changed = true;
                }
            }
        }

        let mut max_live = 0usize;
        for block in &program.blocks {
            let block_index = block.id.as_usize();
            let mut live = live_out[block_index].clone();
            max_live = max_live.max(live.iter().filter(|live| **live).count());

            let mut term_uses = Vec::new();
            collect_term_gp_uses_for_debug(&block.terminator, program.first_fp_reg, &mut term_uses);
            for reg in term_uses {
                live[reg.0 as usize] = true;
            }
            max_live = max_live.max(live.iter().filter(|live| **live).count());

            for inst in block.ops.iter().rev() {
                let mut uses = Vec::new();
                let mut defs = Vec::new();
                collect_inst_gp_uses_defs_for_debug(
                    &inst.kind,
                    program.first_fp_reg,
                    &mut uses,
                    &mut defs,
                );
                for reg in defs {
                    live[reg.0 as usize] = false;
                }
                for reg in uses {
                    live[reg.0 as usize] = true;
                }
                max_live = max_live.max(live.iter().filter(|live| **live).count());
            }
        }

        max_live
    }

    #[test]
    fn compiles_all_local_functions_once() {
        let types = TypeContext::new(vec![
            Rc::new(FunctionType::new(vec![], vec![])),
            Rc::new(FunctionType::new(vec![ValueType::I32], vec![])),
        ]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec0 = FunctionSpec::new(Rc::new(FunctionType::new(vec![], vec![])), 0);
        spec0.set_code((&[0x0b][..]).into());
        let mut spec1 =
            FunctionSpec::new(Rc::new(FunctionType::new(vec![ValueType::I32], vec![])), 1);
        spec1.set_code((&[0x20, 0x00, 0x1a, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: spec0,
            type_index: 0,
        });
        module.functions.push(FunctionInst::Local {
            spec: spec1,
            type_index: 1,
        });
        let store = Box::new(Store::new(module));

        ensure_module_compiled(&store).expect("native compile should succeed");

        let first = store.module().functions[0]
            .spec()
            .and_then(|spec| spec.get_native_code())
            .expect("first native code");
        let second = store.module().functions[1]
            .spec()
            .and_then(|spec| spec.get_native_code())
            .expect("second native code");

        assert!(Rc::ptr_eq(first.compiled_rc(), second.compiled_rc()));
        assert_eq!(first.func_id().0, 0);
        assert_eq!(second.func_id().0, 1);
    }

    #[test]
    fn compiles_function_with_f64_local() {
        // (func (param f64) (result f64) (local.get 0))
        // Bytecode: local.get 0, end
        let types = TypeContext::new(vec![Rc::new(FunctionType::new(
            vec![ValueType::F64],
            vec![ValueType::F64],
        ))]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(
            Rc::new(FunctionType::new(
                vec![ValueType::F64],
                vec![ValueType::F64],
            )),
            0,
        );
        spec.set_code((&[0x20, 0x00, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        let store = Box::new(Store::new(module));

        ensure_module_compiled(&store).expect("f64 local function should compile");
    }

    #[test]
    fn compiles_function_with_f32_local_and_add() {
        // (func (param f32 f32) (result f32) (f32.add (local.get 0) (local.get 1)))
        // Bytecode: local.get 0, local.get 1, f32.add, end
        let types = TypeContext::new(vec![Rc::new(FunctionType::new(
            vec![ValueType::F32, ValueType::F32],
            vec![ValueType::F32],
        ))]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(
            Rc::new(FunctionType::new(
                vec![ValueType::F32, ValueType::F32],
                vec![ValueType::F32],
            )),
            0,
        );
        spec.set_code((&[0x20, 0x00, 0x20, 0x01, 0x92, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        let store = Box::new(Store::new(module));

        ensure_module_compiled(&store).expect("f32 add function should compile");
    }

    #[test]
    fn compiles_function_with_f32_local_in_if() {
        // (func (param f32) (param i32) (result f32)
        //   (if (result f32) (local.get 1)
        //     (then (local.get 0))
        //     (else (f32.const 0))
        //   ))
        // Bytecode:
        //   local.get 1     ;; 0x20 0x01
        //   if (result f32)  ;; 0x04 0x7d
        //   local.get 0     ;; 0x20 0x00
        //   else             ;; 0x05
        //   f32.const 0     ;; 0x43 0x00 0x00 0x00 0x00
        //   end              ;; 0x0b
        //   end              ;; 0x0b
        let types = TypeContext::new(vec![Rc::new(FunctionType::new(
            vec![ValueType::F32, ValueType::I32],
            vec![ValueType::F32],
        ))]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(
            Rc::new(FunctionType::new(
                vec![ValueType::F32, ValueType::I32],
                vec![ValueType::F32],
            )),
            0,
        );
        spec.set_code(
            (&[
                0x20, 0x01, // local.get 1
                0x04, 0x7d, // if (result f32)
                0x20, 0x00, // local.get 0
                0x05, // else
                0x43, 0x00, 0x00, 0x00, 0x00, // f32.const 0
                0x0b, // end
                0x0b, // end (function)
            ][..])
                .into(),
        );
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        let store = Box::new(Store::new(module));

        ensure_module_compiled(&store).expect("f32 if/else function should compile");
    }

    #[test]
    fn emu32_compiles_long_argument_list_internal_call() {
        let _guard = enable_reference_backend_mode(ReferenceBackendMode::Emu32);

        let helper_params = vec![
            ValueType::F32,
            ValueType::I32,
            ValueType::I32,
            ValueType::F64,
            ValueType::F32,
            ValueType::F32,
            ValueType::F32,
            ValueType::F64,
            ValueType::F32,
            ValueType::I32,
            ValueType::I32,
            ValueType::F32,
            ValueType::F64,
            ValueType::I64,
            ValueType::I64,
            ValueType::I32,
            ValueType::I64,
            ValueType::I64,
            ValueType::F32,
            ValueType::I64,
            ValueType::I64,
            ValueType::I64,
            ValueType::I32,
            ValueType::F32,
            ValueType::F32,
            ValueType::F32,
            ValueType::F64,
            ValueType::F32,
            ValueType::I32,
            ValueType::I64,
            ValueType::F32,
            ValueType::F64,
            ValueType::F64,
            ValueType::F32,
            ValueType::I32,
            ValueType::F32,
            ValueType::F32,
            ValueType::F64,
            ValueType::I64,
            ValueType::F64,
            ValueType::I32,
            ValueType::I64,
            ValueType::F32,
            ValueType::F64,
            ValueType::I32,
            ValueType::I32,
            ValueType::I32,
            ValueType::I64,
            ValueType::F64,
            ValueType::I32,
            ValueType::I64,
            ValueType::I64,
            ValueType::F64,
            ValueType::F64,
            ValueType::F64,
            ValueType::F64,
            ValueType::F64,
            ValueType::F64,
            ValueType::I32,
            ValueType::F32,
            ValueType::F64,
            ValueType::F64,
            ValueType::I32,
            ValueType::I64,
            ValueType::F32,
            ValueType::F32,
            ValueType::F32,
            ValueType::I32,
            ValueType::F64,
            ValueType::F64,
            ValueType::F64,
            ValueType::F64,
            ValueType::F64,
            ValueType::F32,
            ValueType::I64,
            ValueType::I64,
            ValueType::I32,
            ValueType::I32,
            ValueType::I32,
            ValueType::F32,
            ValueType::F64,
            ValueType::I32,
            ValueType::I64,
            ValueType::F32,
            ValueType::F32,
            ValueType::F32,
            ValueType::I32,
            ValueType::I32,
            ValueType::F32,
            ValueType::F64,
            ValueType::I64,
            ValueType::F32,
            ValueType::F64,
            ValueType::F32,
            ValueType::F32,
            ValueType::F32,
            ValueType::I32,
            ValueType::F32,
            ValueType::I64,
            ValueType::I32,
        ];
        let helper_type = Rc::new(FunctionType::new(
            helper_params.clone(),
            vec![ValueType::I32],
        ));
        let caller_type = Rc::new(FunctionType::new(
            vec![ValueType::I32],
            vec![ValueType::I32],
        ));
        let types = TypeContext::new(vec![Rc::clone(&helper_type), Rc::clone(&caller_type)]);
        let mut module = ModuleInst::new(String::from("m"), types);

        let helper_code = local_get_code((helper_params.len() - 1) as u32);
        let mut helper_spec = FunctionSpec::new(Rc::clone(&helper_type), 0);
        helper_spec.set_code(helper_code.as_slice().into());
        module.functions.push(FunctionInst::Local {
            spec: helper_spec,
            type_index: 0,
        });

        let caller_code = long_argument_caller_code(&helper_params);
        let mut caller_spec = FunctionSpec::new(Rc::clone(&caller_type), 1);
        caller_spec.set_code(caller_code.as_slice().into());
        module.functions.push(FunctionInst::Local {
            spec: caller_spec,
            type_index: 1,
        });

        let store = Box::new(Store::new(module));
        let backend = active_backend_config().expect("active backend config");
        let module_inst = store.module();

        let mut semantics: Vec<Option<SemanticProgram>> =
            Vec::with_capacity(module_inst.functions.len());
        for func in module_inst.functions.iter() {
            let Some(spec) = func.spec() else {
                semantics.push(None);
                continue;
            };
            let params = spec.func_type().params().len() as u16;
            let local_count = params.saturating_add(spec.locals().len() as u16);
            let results = spec.func_type().results().len() as u16;
            let mut local_types = Vec::with_capacity(local_count as usize);
            local_types.extend_from_slice(spec.func_type().params());
            local_types.extend_from_slice(spec.locals());
            let semantic = decode::decode_to_semantic_ir(
                spec.code(),
                CompileContext::with_value_types(
                    &module_inst.types,
                    &store,
                    module_inst,
                    params,
                    local_count,
                    results,
                    &local_types,
                    spec.func_type().results(),
                ),
            )
            .expect("decode long-argument module");
            semantics.push(Some(semantic));
        }

        loop {
            let mut any_inlined = false;
            for func_idx in 0..semantics.len() {
                if semantics[func_idx].is_none() {
                    continue;
                }
                let mut caller = semantics[func_idx].take().unwrap();
                if inline::inline_calls_in_function(&mut caller, func_idx as u32, &semantics) {
                    any_inlined = true;
                }
                semantics[func_idx] = Some(caller);
            }
            if !any_inlined {
                break;
            }
        }

        let mut prepared_functions = Vec::new();
        for (func_idx, func) in module_inst.functions.iter().enumerate() {
            let Some(spec) = func.spec() else {
                continue;
            };
            let semantic = semantics[func_idx].as_ref().expect("decoded semantic");
            let prepared = prepare_function(
                PrepareInput {
                    config: native_plan_config(backend),
                },
                semantic,
            )
            .expect("prepare long-argument module");
            prepared_functions.push((MachineFuncId(func_idx as u32), prepared));
            let _ = spec;
        }

        let lowered_inputs = prepared_functions
            .iter()
            .map(|(id, prepared)| LowerFunctionInput {
                id: *id,
                frame: prepared.frame,
                lir: &prepared.lir,
                result_count: module_inst.functions[id.0 as usize]
                    .spec()
                    .map(|s| s.func_type().results().len() as u16)
                    .unwrap_or(0),
            })
            .collect::<Vec<_>>();

        let mut lowered = lower_module(LowerModuleInput {
            backend,
            functions: &lowered_inputs,
            #[cfg(has_guard_pages)]
            use_guard_pages: false,
        })
        .expect("lower long-argument module");
        let first_transient = crate::vm::native::ir::machine::MACHINE_FIXED_REG_COUNT
            + backend.gp_local_cache_budget as u16;
        lowered
            .module
            .optimize(first_transient, backend.gp_unit_bytes);
        lowered
            .module
            .legalize(backend.gp_unit_bytes)
            .expect("legalize long-argument module");

        if backend.needs_32bit_gp_legalization() {
            let max_gp_regs = crate::vm::native::ir::machine::MACHINE_FIXED_REG_COUNT
                + backend.gp_local_cache_budget as u16
                + backend.gp_transient_budget as u16;
            if let Err(err) = lowered
                .module
                .finalize_32bit_gp_legalization(first_transient, max_gp_regs)
            {
                let caller_prepared = &prepared_functions[1].1;
                let caller_program = &lowered.module.functions[1].program;
                let (max_lir_gp, max_lir_fp) =
                    max_lir_live_bank_usage(&caller_prepared.lir, backend.gp_unit_bytes);
                let max_machine_gp_live = max_machine_gp_live_for_debug(caller_program);
                panic!(
                    "long-argument caller failed 32-bit finalization: {err}\nmax_lir_gp_units={max_lir_gp}\nmax_lir_fp_live={max_lir_fp}\nmax_machine_gp_live={max_machine_gp_live}\nprepared_lir={:#?}\nmachine_program={:#?}",
                    caller_prepared.lir,
                    caller_program
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn debug_emu32_sha256_compaction_pressure() {
        let _guard = enable_reference_backend_mode(ReferenceBackendMode::Emu32);

        let wasm = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("benchmarks")
            .join("wasi")
            .join("sha256")
            .join("sha256.wasm");
        let bytes = fs::read(&wasm).expect("read sha256.wasm");
        let module = Module::new("sha256", &bytes).expect("parse sha256 module");
        fn dummy_host(
            _caller: &mut crate::vm::entities::Caller,
            _args: &[crate::vm::value::Value],
            _results: &mut [crate::vm::value::Value],
        ) -> Result<(), crate::error::WasmError> {
            Ok(())
        }
        let mut imports = Vec::new();
        for func in module.functions() {
            let FunctionDef::Import {
                module,
                name,
                func_type,
                ..
            } = func.def()
            else {
                continue;
            };
            imports.push(Import::func_typed(
                module,
                name,
                dummy_host,
                (**func_type).clone(),
            ));
        }
        for table in module.tables() {
            let crate::module::entities::TableDef::Import { module, name, .. } = table.def() else {
                continue;
            };
            imports.push(Import::table(
                module,
                name,
                table.limits().min(),
                table.limits().max(),
            ));
        }
        for memory in module.memories() {
            let crate::module::entities::MemoryDef::Import { module, name, .. } = memory.def()
            else {
                continue;
            };
            imports.push(Import::memory(
                module,
                name,
                memory.limits().min(),
                memory.limits().max(),
            ));
        }
        for global in module.globals() {
            let crate::module::entities::GlobalDef::Import {
                module,
                name,
                value_type,
                mutable,
                ..
            } = global.def()
            else {
                continue;
            };
            let zero = match value_type {
                ValueType::I32 => crate::vm::value::Value::I32(0),
                ValueType::I64 => crate::vm::value::Value::I64(0),
                ValueType::F32 => crate::vm::value::Value::F32(0.0),
                ValueType::F64 => crate::vm::value::Value::F64(0.0),
                ValueType::Ref(ref_ty) if *ref_ty == RefType::funcref() => {
                    crate::vm::value::Value::Ref(
                        crate::vm::value::RefHandle::null(),
                        RefType::funcref(),
                    )
                }
                ValueType::Ref(ref_ty) if *ref_ty == RefType::externref() => {
                    crate::vm::value::Value::Ref(
                        crate::vm::value::RefHandle::null(),
                        RefType::externref(),
                    )
                }
                other => panic!("unsupported imported global type in probe: {other:?}"),
            };
            imports.push(Import::global(module, name, zero, *mutable));
        }
        let instance = Instance::from_module(module, &imports).expect("instantiate sha256");
        let store = instance.store();
        let module_inst = store.module();
        let backend = active_backend_config().expect("backend config");

        let mut semantics: Vec<Option<SemanticProgram>> =
            Vec::with_capacity(module_inst.functions.len());
        for (func_idx, func) in module_inst.functions.iter().enumerate() {
            let Some(spec) = func.spec() else {
                semantics.push(None);
                continue;
            };
            let params = spec.func_type().params().len() as u16;
            let local_count = params.saturating_add(spec.locals().len() as u16);
            let results = spec.func_type().results().len() as u16;
            let mut local_types = Vec::with_capacity(local_count as usize);
            local_types.extend_from_slice(spec.func_type().params());
            local_types.extend_from_slice(spec.locals());
            let semantic = decode::decode_to_semantic_ir(
                spec.code(),
                CompileContext::with_value_types(
                    &module_inst.types,
                    store,
                    module_inst,
                    params,
                    local_count,
                    results,
                    &local_types,
                    spec.func_type().results(),
                ),
            )
            .unwrap_or_else(|err| panic!("decode failed for function {func_idx}: {err}"));
            semantics.push(Some(semantic));
        }

        loop {
            let mut any_inlined = false;
            for func_idx in 0..semantics.len() {
                if semantics[func_idx].is_none() {
                    continue;
                }
                let mut caller = semantics[func_idx].take().unwrap();
                if inline::inline_calls_in_function(&mut caller, func_idx as u32, &semantics) {
                    any_inlined = true;
                }
                semantics[func_idx] = Some(caller);
            }
            if !any_inlined {
                break;
            }
        }

        let mut prepared_functions = Vec::new();
        for (func_idx, func) in module_inst.functions.iter().enumerate() {
            let Some(spec) = func.spec() else {
                continue;
            };
            let prepared = prepare_function(
                PrepareInput {
                    config: native_plan_config(backend),
                },
                semantics[func_idx]
                    .as_ref()
                    .expect("semantic program for local function"),
            )
            .unwrap_or_else(|err| panic!("prepare failed for function {func_idx}: {err}"));
            prepared_functions.push((MachineFuncId(func_idx as u32), prepared));
            let _ = spec;
        }

        let lowered_inputs = prepared_functions
            .iter()
            .map(|(id, prepared)| LowerFunctionInput {
                id: *id,
                frame: prepared.frame,
                lir: &prepared.lir,
                result_count: module_inst.functions[id.0 as usize]
                    .spec()
                    .map(|s| s.func_type().results().len() as u16)
                    .unwrap_or(0),
            })
            .collect::<Vec<_>>();

        let mut lowered = lower_module(LowerModuleInput {
            backend,
            functions: &lowered_inputs,
            #[cfg(has_guard_pages)]
            use_guard_pages: false,
        })
        .expect("lower sha256 module");
        let first_transient = crate::vm::native::ir::machine::MACHINE_FIXED_REG_COUNT
            + backend.gp_local_cache_budget as u16;
        lowered
            .module
            .optimize(first_transient, backend.gp_unit_bytes);
        lowered
            .module
            .legalize(backend.gp_unit_bytes)
            .expect("legalize sha256 module");

        let func_idx = 11usize;
        let func = &lowered.module.functions[func_idx];
        let program = &func.program;
        let (max_lir_gp, max_lir_fp) =
            max_lir_live_bank_usage(&prepared_functions[func_idx].1.lir, backend.gp_unit_bytes);
        let max_gp_regs = crate::vm::native::ir::machine::MACHINE_FIXED_REG_COUNT
            + backend.gp_local_cache_budget as u16
            + backend.gp_transient_budget as u16;
        let compaction_debug =
            debug_describe_gp_compaction(program, max_gp_regs).expect("describe sha256 compaction");
        eprintln!(
            "func={} first_fp_reg={} reg_count={} blocks={} used_gp={} used_gp_regs={:?} used_degrees={:?} peak_live_gp={} peak_live_point=block{} {} live={:?} estimated_pressure={} extra_regs={:#?} max_lir_gp_units={} max_lir_fp_live={}",
            func.id.0,
            program.first_fp_reg,
            program.reg_count,
            program.blocks.len(),
            compaction_debug.used_gp_regs.len(),
            compaction_debug.used_gp_regs,
            compaction_debug.used_degrees,
            compaction_debug.peak_live_regs,
            compaction_debug.peak_live_point.block.0,
            compaction_debug.peak_live_point.stage,
            compaction_debug.peak_live_point.live_regs,
            compaction_debug.estimated_pressure,
            compaction_debug.extra_regs,
            max_lir_gp,
            max_lir_fp,
        );
        for block in &program.blocks {
            let gp_params = block
                .params
                .iter()
                .filter(|param| param.reg.0 < program.first_fp_reg)
                .count();
            if gp_params >= 4 || block.ops.len() >= 20 {
                eprintln!(
                    "  block {} gp_params={} ops={} terminator={:?}",
                    block.id.0,
                    gp_params,
                    block.ops.len(),
                    block.terminator
                );
            }
        }

        let mut cloned = program.clone();
        let err = cloned
            .compact_32bit_gp_bank(max_gp_regs)
            .expect_err("sha256 function 11 should fail compaction");
        panic!(
            "sha256 function 11 compaction failure: {err}\nprogram={:#?}",
            program
        );
    }
}
