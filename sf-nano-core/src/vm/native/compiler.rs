use alloc::rc::Rc;

use crate::error::WasmError;
use crate::module::entities::FunctionSpec;
use crate::module::type_defs::FunctionType;
use crate::vm::{
    compile::{CompileContext, StackTracker},
    entities::ModuleInst,
    lowered,
    planner::CompilePlan,
    store::Store,
};

use super::code::create_native_code_with_patches;
use super::finalizer;

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
