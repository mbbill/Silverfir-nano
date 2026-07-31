use tracked_alloc::rc::Rc;

use crate::collections;

use crate::{
    error::WasmError,
    module::type_context::{check_function_types_equivalent, concrete_type_matches_cross_context},
    vm::{
        jit::arch::backend_config,
        jit::runtime::{
            self,
            common::{
                internal_error, run_frame_call_with_status, trap_error, value_matches_value_type,
                NativeCallStatus,
            },
            context::{NativeContext, PendingEscape},
            StoreAccess,
        },
        jit::store::Store,
        jit::value_encoding::{
            localize, try_machine_raw_to_value_in_store, value_to_machine_raw_in_store,
        },
        tag::TagIdentity,
        value::{machine_raw_to_ref, RefValue, Value},
    },
};

#[cfg(test)]
use crate::vm::entities::{Caller, FunctionInst};

use super::abi::{
    RuntimeCallFrameRegion, RuntimeCallMeta, RuntimeCallTargetKind, RuntimeCallTypeCheckKind,
};

/// Uniform entrypoint used by MachineIR runtime calls.
pub(crate) type RuntimeCallEntry =
    unsafe extern "C" fn(ctx: *mut NativeContext, frame: *mut u64, metadata: *const u8) -> u32;

#[inline]
pub(crate) fn call_runtime_entry_ptr() -> RuntimeCallEntry {
    runtime_call_entry
}

#[inline]
fn active_gp_unit_bytes() -> u8 {
    backend_config().gp_unit_bytes
}

unsafe extern "C" fn runtime_call_entry(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
) -> u32 {
    run_frame_call_with_status(ctx, frame, metadata, invoke_runtime_call)
}

#[inline]
unsafe fn frame_read(frame: *mut u64, slot: u16) -> u64 {
    unsafe { *frame.add(slot as usize) }
}

#[inline]
unsafe fn frame_write(frame: *mut u64, slot: u16, value: u64) {
    unsafe {
        *frame.add(slot as usize) = value;
    }
}

#[inline]
fn require_region_slots(
    region: RuntimeCallFrameRegion,
    expected: u16,
    label: &'static str,
) -> Result<(), WasmError> {
    if region.slots != expected {
        return Err(internal_error(label));
    }
    Ok(())
}

#[inline]
fn region_slot(
    region: RuntimeCallFrameRegion,
    index: u16,
    label: &'static str,
) -> Result<u16, WasmError> {
    if index >= region.slots {
        return Err(internal_error(label));
    }
    region
        .base_slot
        .checked_add(index)
        .ok_or_else(|| internal_error("runtime-call frame slot overflow"))
}

#[inline]
unsafe fn region_read(
    frame: *mut u64,
    region: RuntimeCallFrameRegion,
    index: u16,
    label: &'static str,
) -> Result<u64, WasmError> {
    Ok(unsafe { frame_read(frame, region_slot(region, index, label)?) })
}

#[inline]
unsafe fn region_write(
    frame: *mut u64,
    region: RuntimeCallFrameRegion,
    index: u16,
    value: u64,
    label: &'static str,
) -> Result<(), WasmError> {
    unsafe {
        frame_write(frame, region_slot(region, index, label)?, value);
    }
    Ok(())
}

fn invoke_runtime_call(
    ctx: &mut NativeContext,
    frame: *mut u64,
    meta: &RuntimeCallMeta,
) -> Result<NativeCallStatus, WasmError> {
    match meta.target_kind()? {
        RuntimeCallTargetKind::Immediate => {
            call_runtime_by_local_index(ctx, frame, meta.func_idx_source, meta.args, meta.results)
        }
        RuntimeCallTargetKind::FrameSlot => {
            let func_idx_slot = u16::try_from(meta.func_idx_source)
                .map_err(|_| internal_error("runtime-call func_idx source slot exceeds u16"))?;
            call_runtime_by_handle(
                ctx,
                frame,
                machine_raw_to_ref(
                    unsafe { frame_read(frame, func_idx_slot) },
                    active_gp_unit_bytes(),
                ),
                meta.expected_type_idx,
                meta.type_check_kind()?,
                meta.args,
                meta.results,
            )
        }
    }
}

fn call_runtime_by_local_index(
    ctx: &mut NativeContext,
    frame: *mut u64,
    func_idx: u32,
    args_region: RuntimeCallFrameRegion,
    results_region: RuntimeCallFrameRegion,
) -> Result<NativeCallStatus, WasmError> {
    let mut target = runtime::current_store_access(ctx)?;
    invoke_runtime_target(
        ctx,
        frame,
        &mut target,
        func_idx,
        args_region,
        results_region,
    )
}

fn call_runtime_by_handle(
    ctx: &mut NativeContext,
    frame: *mut u64,
    handle: RefValue,
    expected_type_idx: u32,
    type_check_kind: RuntimeCallTypeCheckKind,
    args_region: RuntimeCallFrameRegion,
    results_region: RuntimeCallFrameRegion,
) -> Result<NativeCallStatus, WasmError> {
    if handle.is_null() {
        return Err(trap_error("null function reference"));
    }
    let (caller_handle, caller_id, owner_id, local_index) = {
        let caller_store = ctx
            .store()
            .ok_or_else(|| internal_error("runtime-call context is missing store"))?;
        let caller_id = caller_store.instance_backref().self_id();
        let localized = localize(caller_store, handle);
        if !localized.is_special() && localized.encoded() < caller_store.module().functions.len() {
            let local_index = u32::try_from(localized.encoded())
                .map_err(|_| internal_error("runtime-call local function index exceeds u32"))?;
            (
                caller_store.instance_backref().clone(),
                caller_id,
                caller_id,
                local_index,
            )
        } else {
            let entry = caller_store
                .function_entry_for_handle(localized)
                .ok_or_else(|| trap_error("invalid function reference"))?;
            (
                caller_store.instance_backref().clone(),
                caller_id,
                entry.owner,
                entry.local_index,
            )
        }
    };

    let mut target = if owner_id == caller_id {
        runtime::current_store_access(ctx)?
    } else {
        let token = caller_handle
            .checkout(owner_id)
            .ok_or_else(|| internal_error("runtime-call referenced dead function owner"))?;
        StoreAccess::checked_out(token)
    };

    if target.id() == caller_id {
        target.with_store(|store| {
            validate_runtime_target_type(
                store,
                local_index,
                store,
                expected_type_idx,
                type_check_kind,
            )
        })??;
    } else {
        target.with_store(|owner_store| {
            let caller_store = ctx
                .store()
                .ok_or_else(|| internal_error("runtime-call context is missing store"))?;
            validate_runtime_target_type(
                owner_store,
                local_index,
                caller_store,
                expected_type_idx,
                type_check_kind,
            )
        })??;
    }

    invoke_runtime_target(
        ctx,
        frame,
        &mut target,
        local_index,
        args_region,
        results_region,
    )
}

fn validate_runtime_target_type(
    owner_store: &Store,
    local_index: u32,
    caller_store: &Store,
    expected_type_idx: u32,
    type_check_kind: RuntimeCallTypeCheckKind,
) -> Result<(), WasmError> {
    if expected_type_idx == u32::MAX {
        return Ok(());
    }
    let callee = owner_store
        .module()
        .functions
        .get(local_index as usize)
        .ok_or_else(|| internal_error("runtime-call referenced invalid function index"))?;
    let actual_type_idx = callee.type_index();
    let compatible = if actual_type_idx != u32::MAX {
        concrete_type_matches_cross_context(
            &owner_store.module().types,
            actual_type_idx,
            &caller_store.module().types,
            expected_type_idx,
        )
    } else {
        let expected_type = caller_store
            .module()
            .get_type(expected_type_idx)
            .ok_or_else(|| internal_error("runtime-call expected type index is out of range"))?;
        expected_type.as_ref() == callee.func_type()
            || check_function_types_equivalent(
                expected_type.as_ref(),
                callee.func_type(),
                &caller_store.module().types,
            )
    };
    if compatible {
        return Ok(());
    }
    let trap = match type_check_kind {
        RuntimeCallTypeCheckKind::CallRef => "call_ref type mismatch",
        RuntimeCallTypeCheckKind::IndirectCall => "indirect call type mismatch",
        RuntimeCallTypeCheckKind::None => "call_ref type mismatch",
    };
    Err(trap_error(trap))
}

fn invoke_runtime_target(
    ctx: &mut NativeContext,
    frame: *mut u64,
    target: &mut StoreAccess<'_>,
    local_index: u32,
    args_region: RuntimeCallFrameRegion,
    results_region: RuntimeCallFrameRegion,
) -> Result<NativeCallStatus, WasmError> {
    let func_type = target.with_store(|store| {
        let callee = store
            .module()
            .functions
            .get(local_index as usize)
            .ok_or_else(|| internal_error("runtime-call referenced invalid function index"))?;
        Ok::<_, WasmError>(Rc::new(callee.func_type().clone()))
    })??;

    require_region_slots(
        args_region,
        func_type.params().len() as u16,
        "runtime-call args span does not match function arity",
    )?;
    require_region_slots(
        results_region,
        func_type.results().len() as u16,
        "runtime-call result span does not match function arity",
    )?;

    let args: collections::Vec<Value> = {
        let frame_owner = ctx
            .store()
            .ok_or_else(|| internal_error("runtime-call context is missing frame owner"))?;
        func_type
            .params()
            .iter()
            .enumerate()
            .map(|(index, ty)| unsafe {
                region_read(
                    frame,
                    args_region,
                    index as u16,
                    "runtime-call arg slot is out of bounds",
                )
                .and_then(|raw| {
                    try_machine_raw_to_value_in_store(raw, *ty, active_gp_unit_bytes(), frame_owner)
                })
            })
            .collect::<Result<_, _>>()?
    };
    let ret_vals = match runtime::eval_access(target, local_index, &args) {
        Ok(values) => values,
        Err(WasmError::HostThrow {
            tag,
            args: exn_args,
        }) => {
            if !validate_host_throw_payload(tag, &exn_args, ctx) {
                return Err(trap_error("host threw mistyped exception"));
            }
            let handle = target.with_store_mut(|store| store.alloc_exn(tag, exn_args))?;
            ctx.pending_escape = PendingEscape::Throw { exn: handle, tag };
            return Ok(NativeCallStatus::Thrown);
        }
        Err(other) => return Err(other),
    };

    if ret_vals.len() != func_type.results().len() {
        return Err(internal_error(
            "runtime-call target returned an unexpected result count",
        ));
    }

    ctx.refresh_cached_views_if_stale();

    {
        let frame_owner = ctx
            .store_mut()
            .ok_or_else(|| internal_error("runtime-call context is missing frame owner"))?;
        for (index, value) in ret_vals.into_iter().enumerate() {
            let raw = value_to_machine_raw_in_store(value, active_gp_unit_bytes(), frame_owner);
            unsafe {
                region_write(
                    frame,
                    results_region,
                    index as u16,
                    raw,
                    "runtime-call result slot is out of bounds",
                )?;
            }
        }
    }

    Ok(NativeCallStatus::Ok)
}

/// Validate a host-throw payload against the tag signature. A mismatch means
/// the embedder built a malformed throw and should trap rather than propagate.
fn validate_host_throw_payload(tag: TagIdentity, payload: &[Value], ctx: &NativeContext) -> bool {
    let Some(store) = ctx.store() else {
        return false;
    };
    let Some(tag_inst) = store.module().tags.iter().find(|t| t.handle == tag) else {
        return false;
    };
    let Some(ty) = store.module().types.get_function_type(tag_inst.type_index) else {
        return false;
    };
    let params = ty.params();
    if payload.len() != params.len() {
        return false;
    }
    for (value, expected) in payload.iter().zip(params.iter()) {
        if !value_matches_value_type(value, *expected) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    #[cfg(any(sf_ir_dump, sf_jitdump))]
    use tracked_alloc::string::String;
    use tracked_alloc::{boxed::Box, rc::Rc};

    use super::*;
    use crate::{
        module::{type_context::TypeContext, type_defs::FunctionType},
        utils::limits::Limits,
        value_type::ValueType,
        vm::{
            entities::{HostFn, MemInst},
            jit::entities::ModuleInst,
            jit::runtime::{common::NativeCallStatus, context::NativeContextBox},
            jit::store::{tests::store as test_store, Store},
        },
    };

    fn test_context(module: ModuleInst) -> (Box<Store>, NativeContextBox) {
        let mut store = test_store(module);
        let n_globals = store.module().globals.len();
        let ctx = NativeContext::new(
            (&mut *store) as *mut Store,
            core::ptr::null_mut(),
            n_globals,
        );
        (store, ctx)
    }

    fn call_runtime<T: Copy>(ctx: &mut NativeContext, frame: &mut [u64], meta: &T) -> u32 {
        let entry = call_runtime_entry_ptr();
        unsafe { entry(ctx, frame.as_mut_ptr(), (meta as *const T).cast::<u8>()) }
    }

    #[test]
    fn runtime_call_marshals_frame_slots() {
        fn host_add(
            _caller: &mut Caller,
            args: &[Value],
            results: &mut [Value],
        ) -> Result<(), WasmError> {
            let lhs = match args[0] {
                Value::I32(value) => value,
                _ => panic!("unexpected arg"),
            };
            let rhs = match args[1] {
                Value::I32(value) => value,
                _ => panic!("unexpected arg"),
            };
            results[0] = Value::I32(lhs + rhs);
            Ok(())
        }

        let func_type = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32, ValueType::I32],
            collections::vec![ValueType::I32],
        ));
        let mut module = ModuleInst::new(
            crate::config::Config::new(),
            #[cfg(any(sf_ir_dump, sf_jitdump))]
            String::from("m"),
            TypeContext::empty(),
        );
        module.functions.push(FunctionInst::Host {
            func_type,
            callback: crate::vm::entities::HostCallback::new(host_add as HostFn),
        });
        module.memories.push(
            MemInst::new(
                &crate::config::Config::new(),
                Limits::new(1, Some(1)).unwrap(),
            )
            .expect("test memory within runtime limits"),
        );
        let (_store, mut ctx) = test_context(module);
        let meta = RuntimeCallMeta {
            func_idx_source: 0,
            func_idx_source_kind: RuntimeCallTargetKind::Immediate as u32,
            expected_type_idx: u32::MAX,
            type_check_kind: RuntimeCallTypeCheckKind::None as u32,
            args: RuntimeCallFrameRegion {
                base_slot: 0,
                slots: 2,
            },
            results: RuntimeCallFrameRegion {
                base_slot: 0,
                slots: 1,
            },
        };
        let mut frame = [7, 5, 99];

        let status = call_runtime(&mut ctx, &mut frame, &meta);

        assert_eq!(status, NativeCallStatus::Ok as u32);
        assert_eq!(frame[0], 12);
        assert!(ctx.error.is_none());
    }

    #[test]
    fn runtime_call_reads_dynamic_target_slot() {
        fn host_add(
            _caller: &mut Caller,
            args: &[Value],
            results: &mut [Value],
        ) -> Result<(), WasmError> {
            let lhs = match args[0] {
                Value::I32(value) => value,
                _ => panic!("unexpected arg"),
            };
            let rhs = match args[1] {
                Value::I32(value) => value,
                _ => panic!("unexpected arg"),
            };
            results[0] = Value::I32(lhs + rhs);
            Ok(())
        }

        let func_type = Rc::new(FunctionType::new(
            collections::vec![ValueType::I32, ValueType::I32],
            collections::vec![ValueType::I32],
        ));
        let mut module = ModuleInst::new(
            crate::config::Config::new(),
            #[cfg(any(sf_ir_dump, sf_jitdump))]
            String::from("m"),
            TypeContext::empty(),
        );
        module.functions.push(FunctionInst::Host {
            func_type,
            callback: crate::vm::entities::HostCallback::new(host_add as HostFn),
        });
        module.memories.push(
            MemInst::new(
                &crate::config::Config::new(),
                Limits::new(1, Some(1)).unwrap(),
            )
            .expect("test memory within runtime limits"),
        );
        let (mut store, mut ctx) = test_context(module);
        let _ = store.register_local_function(0);
        let meta = RuntimeCallMeta {
            func_idx_source: 2,
            func_idx_source_kind: RuntimeCallTargetKind::FrameSlot as u32,
            expected_type_idx: u32::MAX,
            type_check_kind: RuntimeCallTypeCheckKind::None as u32,
            args: RuntimeCallFrameRegion {
                base_slot: 0,
                slots: 2,
            },
            results: RuntimeCallFrameRegion {
                base_slot: 0,
                slots: 1,
            },
        };
        let mut frame = [7, 5, 0];

        let status = call_runtime(&mut ctx, &mut frame, &meta);

        assert_eq!(status, NativeCallStatus::Ok as u32);
        assert_eq!(frame[0], 12);
        assert!(ctx.error.is_none());
    }
}
