use crate::collections::{self, phase_span, phase_span_with_function};
use tracked_alloc::rc::Rc;

use core::sync::atomic::{AtomicUsize, Ordering};

use crate::module::entities::FunctionSpec;

/// Minimal native stats surface for CLI/debug output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeStatsSnapshot {
    pub groups: usize,
    pub ssa_ops: usize,
    pub mir_ops: usize,
    pub bytes_emitted: usize,
    pub groups_skipped: usize,
    pub ops_skipped: usize,
}

static STATS_GROUPS: AtomicUsize = AtomicUsize::new(0);
static STATS_SSA_OPS: AtomicUsize = AtomicUsize::new(0);
static STATS_MIR_OPS: AtomicUsize = AtomicUsize::new(0);
static STATS_BYTES: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn set_native_stats(
    groups: usize,
    ssa_ops: usize,
    mir_ops: usize,
    bytes_emitted: usize,
) {
    STATS_GROUPS.store(groups, Ordering::Relaxed);
    STATS_SSA_OPS.store(ssa_ops, Ordering::Relaxed);
    STATS_MIR_OPS.store(mir_ops, Ordering::Relaxed);
    STATS_BYTES.store(bytes_emitted, Ordering::Relaxed);
}

#[inline]
pub fn native_stats_snapshot() -> NativeStatsSnapshot {
    NativeStatsSnapshot {
        groups: STATS_GROUPS.load(Ordering::Relaxed),
        ssa_ops: STATS_SSA_OPS.load(Ordering::Relaxed),
        mir_ops: STATS_MIR_OPS.load(Ordering::Relaxed),
        bytes_emitted: STATS_BYTES.load(Ordering::Relaxed),
        groups_skipped: 0,
        ops_skipped: 0,
    }
}

#[inline]
pub fn native_stats() -> (usize, usize) {
    (
        STATS_GROUPS.load(Ordering::Relaxed),
        STATS_SSA_OPS.load(Ordering::Relaxed),
    )
}

#[inline]
pub const fn native_capacity_skips() -> (usize, usize) {
    (0, 0)
}

#[cfg(sf_ir_dump)]
use crate::vm::debug::ir_dump;
use crate::vm::{backend::BackendConfig, entities::ModuleInst};
use crate::{
    error::WasmError,
    vm::{
        arch,
        machine::{
            lower_module, machine_ir::MachineFuncId, optimize_module, LowerFunctionInput,
            LowerModuleInput, LoweredMachineModule,
        },
        middle::{prepare_function, PrepareInput},
        runtime::code::{CompiledNativeModule, NativeCode, NativeCodeCache},
        store::Store,
        wasm::{context::CompileContext, decode, inline, semantic_ir::SemanticProgram},
    },
};

fn decode_function_semantic(
    module: &ModuleInst,
    store: &Store,
    spec: &FunctionSpec,
) -> Result<SemanticProgram, WasmError> {
    let params = spec.func_type().params().len() as u16;
    let local_count = params.saturating_add(spec.locals().len() as u16);
    let results = spec.func_type().results().len() as u16;
    let mut local_types = collections::Vec::with_capacity(local_count as usize);
    local_types.extend_from_slice(spec.func_type().params());
    local_types.extend_from_slice(spec.locals());
    decode::decode_to_semantic_ir(
        spec.code(),
        CompileContext::with_value_types(
            &module.types,
            store,
            params,
            local_count,
            results,
            &local_types,
            spec.func_type().results(),
        ),
    )
    .map_err(|_err| WasmError::internal("native decode failed for function"))
}

fn finish_native_compile(
    active_backend: arch::NativeBackend,
    backend: BackendConfig,
    module: &ModuleInst,
    groups: usize,
    ssa_ops: usize,
    lowered: LoweredMachineModule,
    #[cfg(sf_ir_dump)] dump_lir_inputs: Option<&[ir_dump::DumpFunctionLir]>,
) -> Result<(), WasmError> {
    let mir_ops: usize = lowered
        .module
        .functions
        .iter()
        .map(|f| f.program.blocks.iter().map(|b| b.ops.len()).sum::<usize>())
        .sum();
    let arch_lower_phase = phase_span("arch_lower");
    let mut compiled = Rc::new(CompiledNativeModule::new(
        active_backend,
        backend,
        lowered.module,
        lowered.abi,
    )?);
    let entries = arch::dispatch_compile_module(active_backend, module, &compiled)?;
    drop(arch_lower_phase);

    let bytes: usize = entries
        .iter()
        .filter_map(|e| e.as_ref().map(|e| e.text_len))
        .sum();
    set_native_stats(groups, ssa_ops, mir_ops, bytes);

    #[cfg(sf_ir_dump)]
    if ir_dump::dump_enabled() {
        let code_slices: collections::Vec<(u32, &[u8])> = entries
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| {
                entry.as_ref().map(|e| {
                    let ptr = e.entry as *const u8;
                    let len = e.text_len;
                    (idx as u32, unsafe { core::slice::from_raw_parts(ptr, len) })
                })
            })
            .collect();
        let dump_regions: collections::Vec<ir_dump::DumpFunctionRegions> = entries
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| {
                entry.as_ref().map(|e| ir_dump::DumpFunctionRegions {
                    func_idx: idx as u32,
                    regions: e.debug_regions.clone(),
                })
            })
            .collect();
        let _ = ir_dump::write_module_dump(
            &module.name,
            module.functions.len(),
            dump_lir_inputs.unwrap_or(&[]),
            compiled.module(),
            compiled.abi(),
            &code_slices,
            &dump_regions,
        );
    }

    #[cfg(sf_emulator)]
    let keep_machine_ir = matches!(active_backend, arch::NativeBackend::Reference);
    #[cfg(not(sf_emulator))]
    let keep_machine_ir = false;
    if !keep_machine_ir {
        if let Some(compiled_mut) = Rc::get_mut(&mut compiled) {
            compiled_mut.strip_machine_ir_for_runtime();
        }
    }

    for (func_idx, func) in module.functions.iter().enumerate() {
        let Some(spec) = func.spec() else {
            continue;
        };
        let mut code = NativeCode::new(Rc::clone(&compiled), MachineFuncId(func_idx as u32));
        let native_entry = entries.get(func_idx).and_then(|e| e.as_ref());
        code = code.with_entry(native_entry.map(|e| e.entry));
        spec.set_native_code(code, NativeCodeCache::compiled());
    }

    Ok(())
}

pub(crate) fn ensure_module_compiled(store: &Store) -> Result<(), WasmError> {
    let active_backend = arch::active_native_backend()
        .map_err(|_err| WasmError::invalid("native backend unavailable"))?;
    let backend = arch::active_backend_config()
        .map_err(|_err| WasmError::invalid("native backend unavailable"))?;
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

    // Phase 1: Scan for tiny leaf semantic callees worth retaining as inline seeds.
    let mut inline_candidates: collections::Vec<Option<SemanticProgram>> =
        collections::Vec::with_capacity(module.functions.len());
    for (func_idx, func) in module.functions.iter().enumerate() {
        let Some(spec) = func.spec() else {
            inline_candidates.push(None);
            continue;
        };
        let sem_scan_function_phase = phase_span_with_function("sem_scan", Some(func_idx as u32));
        let semantic = decode_function_semantic(module, store, spec)?;
        inline_candidates.push(inline::retain_inline_candidate(&semantic).then_some(semantic));
        drop(sem_scan_function_phase);
    }

    // Phase 2: Decode each caller, inline retained leaf callees, and lower immediately.
    let mut groups = 0usize;
    let mut ssa_ops = 0usize;
    let mut prepared_functions: collections::Vec<LowerFunctionInput> = collections::Vec::new();
    for (func_idx, func) in module.functions.iter().enumerate() {
        let Some(spec) = func.spec() else {
            continue;
        };
        let sem_decode_function_phase =
            phase_span_with_function("sem_decode", Some(func_idx as u32));
        let mut semantic = decode_function_semantic(module, store, spec)?;
        drop(sem_decode_function_phase);

        let sem_inline_function_phase =
            phase_span_with_function("sem_inline", Some(func_idx as u32));
        inline::inline_calls_in_function(&mut semantic, func_idx as u32, &inline_candidates);
        drop(sem_inline_function_phase);

        let ssa_lower_function_phase = phase_span_with_function("ssa_lower", Some(func_idx as u32));
        let prepared = prepare_function(
            PrepareInput {
                config: backend,
                function_index: Some(func_idx as u32),
            },
            &semantic,
        )
        .map_err(|_err| {
            WasmError::internal(
                "native prepare failed for function type_idx= params= results= max_stack= ops=",
            )
        })?;
        drop(ssa_lower_function_phase);
        groups += 1;
        ssa_ops += prepared
            .ssa
            .blocks
            .iter()
            .map(|block| block.ops.len())
            .sum::<usize>();
        let func_id = MachineFuncId(func_idx as u32);
        let result_count = spec.func_type().results().len() as u16;
        prepared_functions.push(LowerFunctionInput {
            id: func_id,
            frame: prepared.frame,
            ssa: prepared.ssa,
            result_count,
        });
    }
    drop(inline_candidates);

    #[cfg(sf_ir_dump)]
    let dump_lir_inputs = if ir_dump::dump_enabled() {
        Some(
            prepared_functions
                .iter()
                .map(|prepared| ir_dump::DumpFunctionLir {
                    func_idx: prepared.id.0,
                    ssa: prepared.ssa.clone(),
                })
                .collect::<collections::Vec<_>>(),
        )
    } else {
        None
    };

    #[cfg(sf_has_guard_pages)]
    let use_guard_pages = module
        .memories
        .first()
        .map(|m| m.has_guard_pages())
        .unwrap_or(false);
    #[cfg(sf_has_guard_pages)]
    let use_guard_pages = use_guard_pages && backend.gp_unit_bytes == 8;
    let mut lowered = lower_module(LowerModuleInput {
        backend,
        functions: prepared_functions,
        #[cfg(sf_has_guard_pages)]
        use_guard_pages,
    })?;
    let module_opt_phase = phase_span("module_opt");
    optimize_module(&mut lowered.module);
    if backend.is_32bit_gp_target() {
        lowered
            .module
            .validate_32bit_gp_target(backend.first_fp_reg())?;
    }
    drop(module_opt_phase);

    finish_native_compile(
        active_backend,
        backend,
        module,
        groups,
        ssa_ops,
        lowered,
        #[cfg(sf_ir_dump)]
        dump_lir_inputs.as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use tracked_alloc::{boxed::Box, rc::Rc, string::String};

    use super::ensure_module_compiled;
    use crate::collections;
    use crate::{
        module::{entities::FunctionSpec, type_context::TypeContext, type_defs::FunctionType},
        utils::limits::Limits,
        value_type::ValueType,
        vm::{
            arch::{backend_mode_test_lock, set_reference_backend_mode},
            entities::{FunctionInst, MemInst, ModuleInst},
            store::Store,
        },
        ReferenceBackendMode,
    };

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

    fn encode_u32_leb(mut value: u32, out: &mut collections::Vec<u8>) {
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

    fn local_get_code(index: u32) -> collections::Vec<u8> {
        let mut code = collections::vec![0x20];
        encode_u32_leb(index, &mut code);
        code.push(0x0b);
        code
    }

    fn long_argument_caller_code(helper_params: &[ValueType]) -> collections::Vec<u8> {
        let mut code = collections::Vec::new();
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

    #[test]
    fn compiles_all_local_functions_once() {
        let types = TypeContext::new(collections::vec![
            Rc::new(FunctionType::new(collections::vec![], collections::vec![])),
            Rc::new(FunctionType::new(
                collections::vec![ValueType::I32],
                collections::vec![]
            )),
        ]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec0 = FunctionSpec::new(
            Rc::new(FunctionType::new(collections::vec![], collections::vec![])),
            0,
        );
        spec0.set_code((&[0x0b][..]).into());
        let mut spec1 = FunctionSpec::new(
            Rc::new(FunctionType::new(
                collections::vec![ValueType::I32],
                collections::vec![],
            )),
            1,
        );
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
    fn emu64_compiled_module_publishes_local_call_info_table() {
        let _guard = enable_reference_backend_mode(ReferenceBackendMode::Emu64);

        let types = TypeContext::new(collections::vec![
            Rc::new(FunctionType::new(collections::vec![], collections::vec![])),
            Rc::new(FunctionType::new(
                collections::vec![ValueType::I32],
                collections::vec![ValueType::I32],
            )),
        ]);
        let mut module = ModuleInst::new(String::from("m"), types);

        let mut spec0 = FunctionSpec::new(
            Rc::new(FunctionType::new(collections::vec![], collections::vec![])),
            0,
        );
        spec0.set_code((&[0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: spec0,
            type_index: 0,
        });

        let mut spec1 = FunctionSpec::new(
            Rc::new(FunctionType::new(
                collections::vec![ValueType::I32],
                collections::vec![ValueType::I32],
            )),
            1,
        );
        spec1.set_code((&[0x20, 0x00, 0x0b][..]).into());
        module.functions.push(FunctionInst::Local {
            spec: spec1,
            type_index: 1,
        });

        let store = Box::new(Store::new(module));
        ensure_module_compiled(&store).expect("emu64 native compile should succeed");

        let compiled = store.module().functions[0]
            .spec()
            .and_then(|spec| spec.get_native_code())
            .expect("first native code")
            .compiled();
        let local_call_infos = compiled.dispatch_metadata().local_call_infos();
        assert_eq!(local_call_infos.len(), 2);
        assert!(
            !local_call_infos.base().is_null(),
            "reference-backend compiled module must publish local call infos for indirect local calls"
        );
    }

    #[test]
    fn compiles_function_with_f64_local() {
        // (func (param f64) (result f64) (local.get 0))
        // Bytecode: local.get 0, end
        let types = TypeContext::new(collections::vec![Rc::new(FunctionType::new(
            collections::vec![ValueType::F64],
            collections::vec![ValueType::F64],
        ))]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(
            Rc::new(FunctionType::new(
                collections::vec![ValueType::F64],
                collections::vec![ValueType::F64],
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
        let types = TypeContext::new(collections::vec![Rc::new(FunctionType::new(
            collections::vec![ValueType::F32, ValueType::F32],
            collections::vec![ValueType::F32],
        ))]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(
            Rc::new(FunctionType::new(
                collections::vec![ValueType::F32, ValueType::F32],
                collections::vec![ValueType::F32],
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
        let types = TypeContext::new(collections::vec![Rc::new(FunctionType::new(
            collections::vec![ValueType::F32, ValueType::I32],
            collections::vec![ValueType::F32],
        ))]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(
            Rc::new(FunctionType::new(
                collections::vec![ValueType::F32, ValueType::I32],
                collections::vec![ValueType::F32],
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
    fn compiles_f32_kahan_sum_style_loop() {
        // Reduced from `float_exprs.wast` `f32.kahan_sum`.
        //
        // The important shape is:
        // - three F32 locals
        // - nested `local.tee` on F32 values
        // - a loop backedge guarded by `br_if`
        // - a live F32 value carried through the block right up to the
        //   terminator
        //
        // Native ARM64 compilation was previously failing here with:
        // "missing float-width tracking for machine reg ... at bN terminator".
        let ty = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32, ValueType::I32],
            collections::vec![ValueType::F32],
        ));
        let types = TypeContext::new(collections::vec![Rc::clone(&ty)]);
        let mut module = ModuleInst::new(String::from("m"), types);
        let mut spec = FunctionSpec::new(Rc::clone(&ty), 0);
        spec.set_locals(collections::vec![
            ValueType::F32,
            ValueType::F32,
            ValueType::F32
        ]);
        spec.set_code(
            (&[
                0x02, 0x40, // block
                0x03, 0x40, // loop
                0x20, 0x00, // local.get 0
                0x2a, 0x02, 0x00, // f32.load align=2 offset=0
                0x20, 0x04, // local.get 4
                0x93, // f32.sub
                0x22, 0x04, // local.tee 4
                0x20, 0x03, // local.get 3
                0x92, // f32.add
                0x22, 0x02, // local.tee 2
                0x20, 0x03, // local.get 3
                0x93, // f32.sub
                0x20, 0x04, // local.get 4
                0x93, // f32.sub
                0x21, 0x04, // local.set 4
                0x20, 0x00, // local.get 0
                0x41, 0x04, // i32.const 4
                0x6a, // i32.add
                0x21, 0x00, // local.set 0
                0x20, 0x02, // local.get 2
                0x21, 0x03, // local.set 3
                0x20, 0x01, // local.get 1
                0x41, 0x7f, // i32.const -1
                0x6a, // i32.add
                0x22, 0x01, // local.tee 1
                0x0d, 0x00, // br_if 0
                0x0b, // end loop
                0x0b, // end block
                0x20, 0x02, // local.get 2
                0x0b, // end
            ][..])
                .into(),
        );
        module.functions.push(FunctionInst::Local {
            spec,
            type_index: 0,
        });
        module
            .memories
            .push(MemInst::new(Limits::new(1, Some(1)).unwrap()));
        let store = Box::new(Store::new(module));

        ensure_module_compiled(&store).expect("f32 kahan-style loop should compile");
    }

    #[test]
    fn emu32_compiles_long_argument_list_internal_call() {
        let _guard = enable_reference_backend_mode(ReferenceBackendMode::Emu32);

        let helper_params = collections::vec![
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
            collections::vec![ValueType::I32],
        ));
        let caller_type = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32],
            collections::vec![ValueType::I32],
        ));
        let types = TypeContext::new(collections::vec![
            Rc::clone(&helper_type),
            Rc::clone(&caller_type)
        ]);
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
        ensure_module_compiled(&store).expect(
            "long-argument internal call should compile through the direct 32-bit lowering path",
        );
    }
}
