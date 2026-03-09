use alloc::rc::Rc;

use crate::error::WasmError;
use crate::module::entities::FunctionSpec;
use crate::module::type_defs::FunctionType;
use crate::vm::{
    compile::{CompileContext, StackTracker},
    entities::{FunctionInst, ModuleInst},
    lowered,
    planner::CompilePlan,
    store::Store,
};

use super::code::create_native_code_with_patches;
use super::finalizer;
#[cfg(feature = "lockstep-debug")]
use super::arm64::emit;

pub fn build_for_function(
    function: &FunctionSpec,
    types: Option<&[Rc<FunctionType>]>,
    store: &Store,
    module: &ModuleInst,
    func_idx: u32,
) -> Result<super::instruction::NativeEntry, WasmError> {
    let code = function.code();
    let func_type = function.func_type();
    let params_count = func_type.params().len();
    let results_count = func_type.results().len();
    let locals_count = function.locals().len();
    let compile_config = crate::vm::backend::BackendKind::Native.compile_config();

    let frame_size = params_count + locals_count;
    let compile_plan = CompilePlan::for_config(code, frame_size, compile_config);
    let hot_local_plan = compile_plan.hot_locals();
    let ctx = CompileContext::new(types, store, module, results_count);
    let mut stack = StackTracker::new(
        compile_plan.config(),
        params_count,
        locals_count,
        results_count,
    );
    let ir_ops = lowered::lower_to_ir(code, &ctx, &mut stack, hot_local_plan)?;
    let hot_mask = hot_local_plan.hot_mask();

    let resolved = super::resolve_backend(
        &ir_ops,
        stack.operand_base(),
        compile_plan.config().tos_register_count,
        module,
        hot_mask,
        func_idx,
    )
        .map_err(|err| WasmError::invalid(alloc::format!("native backend unavailable: {}", err)))?;

    let mut buf = module
        .native_code_buffer()
        .map_err(|err| WasmError::invalid(err.into()))?;
    let (code_box, helper_metadata, direct_call_entry_patches) =
        finalizer::finalize(resolved, &mut stack, &mut buf, &module.name, func_idx);
    drop(buf);
    let (native_code, native_cache) =
        create_native_code_with_patches(
            code_box,
            helper_metadata,
            direct_call_entry_patches,
            params_count,
            locals_count,
            results_count,
        );
    let entry = native_cache.entry();
    function.set_native_code(native_code, native_cache);
    Ok(entry)
}

#[cfg(feature = "lockstep-debug")]
pub fn build_for_function_with_checkpoints(
    function: &FunctionSpec,
    types: Option<&[Rc<FunctionType>]>,
    store: &Store,
    module: &ModuleInst,
    func_idx: u32,
    checkpoint_plan: &crate::vm::lockstep::CheckpointPlan,
) -> Result<super::instruction::NativeEntry, WasmError> {
    let code = function.code();
    let func_type = function.func_type();
    let params_count = func_type.params().len();
    let results_count = func_type.results().len();
    let locals_count = function.locals().len();
    let compile_config = crate::vm::backend::BackendKind::Native.compile_config();

    let frame_size = params_count + locals_count;
    let compile_plan = CompilePlan::for_config(code, frame_size, compile_config);
    let hot_local_plan = compile_plan.hot_locals();
    let ctx = CompileContext::new(types, store, module, results_count);
    let mut stack = StackTracker::new(
        compile_plan.config(),
        params_count,
        locals_count,
        results_count,
    );
    let ir_ops = lowered::lower_to_ir(code, &ctx, &mut stack, hot_local_plan)?;
    let hot_mask = hot_local_plan.hot_mask();

    let mut buf = module
        .native_code_buffer()
        .map_err(|err| WasmError::invalid(err.into()))?;
    let resolved = super::resolve_native_with_context_and_checkpoints(
        &ir_ops,
        &mut buf,
        hot_mask,
        &module.name,
        func_idx,
        checkpoint_plan,
    )
    .map_err(|err| WasmError::invalid(alloc::format!("native backend unavailable: {}", err)))?;
    let (code_box, helper_metadata, direct_call_entry_patches) =
        finalizer::finalize_with_checkpoints(
            resolved,
            &mut stack,
            &mut buf,
            &module.name,
            func_idx,
            checkpoint_plan,
        );
    drop(buf);
    let (native_code, mut native_cache) = create_native_code_with_patches(
        code_box,
        helper_metadata,
        direct_call_entry_patches,
        params_count,
        locals_count,
        results_count,
    );
    if let Some(entry_site) = checkpoint_plan.entry_site() {
        let raw_entry_inst = native_code.entry_ptr() as *const super::instruction::NativeInst;
        let mut buf = module
            .native_code_buffer()
            .map_err(|err| WasmError::invalid(err.into()))?;
        buf.begin_write();
        let start = buf.len();
        let (_, len) =
            emit::emit_checkpoint_stop_wrapper(&mut buf, raw_entry_inst, entry_site.id.ordinal as u64);
        let wrapper_entry = unsafe { buf.fn_ptr(start) };
        buf.finish_write(start, len);
        native_cache.set_entry_override(wrapper_entry);
    }
    let entry = native_cache.entry();
    function.set_native_code(native_code, native_cache);
    Ok(entry)
}

#[cfg(feature = "lockstep-debug")]
pub fn build_module_with_checkpoints(
    store: &mut Store,
    checkpoint_plan: &crate::vm::lockstep::ProgramCheckpointPlan,
) -> Result<(), WasmError> {
    let module = store.module();
    for (func_idx, func_inst) in module.functions.iter().enumerate() {
        let Some(func_plan) = checkpoint_plan.function_plan(func_idx as u32) else {
            continue;
        };
        let FunctionInst::Local { spec, .. } = func_inst else {
            continue;
        };
        build_for_function_with_checkpoints(
            spec,
            Some(&module.types),
            store,
            module,
            func_idx as u32,
            func_plan,
        )?;
    }
    Ok(())
}
