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
        #[cfg(not(target_arch = "aarch64"))]
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
        #[cfg(not(target_arch = "aarch64"))]
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
    use alloc::{boxed::Box, rc::Rc, string::String, vec};

    use super::ensure_module_compiled;
    use crate::{
        module::{entities::FunctionSpec, type_context::TypeContext, type_defs::FunctionType},
        value_type::ValueType,
        vm::{
            entities::{FunctionInst, ModuleInst},
            store::Store,
        },
    };

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
}
