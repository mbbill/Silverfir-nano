use crate::collections;
use crate::vm::jit::entities::FunctionInst;
use crate::{
    error::WasmError,
    module::type_defs::FunctionType,
    vm::{
        entities::{Caller, HostCallback, MemInst},
        jit::arch,
        jit::build,
        jit::runtime::{code::NativeCode, StoreAccess},
        jit::value_encoding::absolutize,
        link::{value_matches_type, FuncEntry, InstanceBackref, RefTypeOwner},
        value::Value,
    },
};
use tracked_alloc::rc::Rc;

enum EvalTarget {
    Host {
        func_type: Rc<FunctionType>,
        callback: HostCallback,
        memory: Option<MemInst>,
    },
    Linked {
        entry: FuncEntry,
        instance_backref: InstanceBackref,
    },
    Local {
        func_type: Rc<FunctionType>,
        code: NativeCode,
    },
}

pub(super) fn eval(
    access: &mut StoreAccess<'_>,
    local_index: u32,
    args: &[Value],
) -> Result<collections::Vec<Value>, WasmError> {
    let memory_is_borrowed = access.with_store(|store| {
        store
            .module()
            .memories
            .iter()
            .any(MemInst::host_callback_borrowed)
    })?;
    if memory_is_borrowed {
        return Err(WasmError::trap(
            "linear memory is borrowed by a host callback",
        ));
    }

    let active_config = arch::backend_config();
    let needs_compile = access.with_store(|store| {
        let func = store
            .module()
            .functions
            .get(local_index as usize)
            .ok_or_else(|| WasmError::internal("function index out of range"))?;
        let params = func.func_type().params();
        if args.len() != params.len() {
            return Err(WasmError::invalid("invalid argument count"));
        }
        if !args
            .iter()
            .zip(params)
            .all(|(value, ty)| value_matches_type(value, *ty, RefTypeOwner::Jit(store)))
        {
            return Err(WasmError::invalid("argument type mismatch"));
        }
        Ok::<_, WasmError>(match func {
            FunctionInst::Local { spec, .. } => spec
                .get_native_code()
                .map(|code| code.compiled().backend() != active_config)
                .unwrap_or(true),
            FunctionInst::Host { .. } | FunctionInst::Linked { .. } => false,
        })
    })??;
    if needs_compile {
        access.with_store(build::ensure_module_compiled)??;
    }

    let target = access.with_store(|store| -> Result<EvalTarget, WasmError> {
        let func = store
            .module()
            .functions
            .get(local_index as usize)
            .ok_or_else(|| WasmError::internal("function index out of range"))?;
        match func {
            FunctionInst::Host {
                func_type,
                callback,
                ..
            } => {
                let memory = store.module().memories.first().cloned();
                Ok(EvalTarget::Host {
                    func_type: Rc::clone(func_type),
                    callback: callback.clone(),
                    memory,
                })
            }
            FunctionInst::Linked { handle, .. } => {
                let absolute = absolutize(store, *handle);
                let entry = store
                    .function_entry_for_handle(absolute)
                    .ok_or_else(|| WasmError::internal("linked function handle not found"))?;
                Ok(EvalTarget::Linked {
                    entry,
                    instance_backref: store.instance_backref().clone(),
                })
            }
            FunctionInst::Local { spec, .. } => {
                let code = spec.get_native_code().cloned().ok_or_else(|| {
                    WasmError::internal("native runtime is missing compiled machine code")
                })?;
                Ok(EvalTarget::Local {
                    func_type: spec.func_type_rc(),
                    code,
                })
            }
        }
    })??;

    match target {
        EvalTarget::Host {
            func_type,
            callback,
            memory,
        } => {
            let mut returns = collections::vec![Value::default(); func_type.results().len()];
            let mut caller = Caller::from_shared_memory(memory, access);
            callback.call(&mut caller, args, &mut returns)?;
            caller.validate_results(&returns, func_type.results())?;
            Ok(returns)
        }
        EvalTarget::Linked {
            entry,
            instance_backref,
        } => {
            if entry.owner == access.id() {
                return eval(access, entry.local_index, args);
            }
            let owner = instance_backref
                .checkout(entry.owner)
                .ok_or_else(|| WasmError::internal("linked function owner no longer available"))?;
            let mut owner_access = StoreAccess::checked_out(owner);
            eval(&mut owner_access, entry.local_index, args)
        }
        EvalTarget::Local { func_type, code } => {
            arch::dispatch_eval(access, local_index, &func_type, &code, args)
        }
    }
}
