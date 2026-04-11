use alloc::rc::Rc;

use crate::collections;

use crate::{
    error::WasmError,
    vm::{
        entities::{Caller, FunctionInst},
        raw_value::{raw_to_value, value_to_raw},
        runtime::{
            common::{internal_error, run_frame_call},
            context::NativeContext,
        },
        value::Value,
    },
};

use super::abi::{ExternalCallFrameRegion, ExternalCallMeta, ExternalCallTargetKind};

/// Uniform entrypoint used by MachineIR external calls.
pub(crate) type ExternalCallEntry =
    unsafe extern "C" fn(ctx: *mut NativeContext, frame: *mut u64, metadata: *const u8) -> u32;

#[inline]
pub(crate) fn call_external_entry_ptr() -> ExternalCallEntry {
    external_call_entry
}

unsafe extern "C" fn external_call_entry(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
) -> u32 {
    run_frame_call(ctx, frame, metadata, invoke_external_call)
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
    region: ExternalCallFrameRegion,
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
    region: ExternalCallFrameRegion,
    index: u16,
    label: &'static str,
) -> Result<u16, WasmError> {
    if index >= region.slots {
        return Err(internal_error(label));
    }
    region
        .base_slot
        .checked_add(index)
        .ok_or_else(|| internal_error("external-call frame slot overflow"))
}

#[inline]
unsafe fn region_read(
    frame: *mut u64,
    region: ExternalCallFrameRegion,
    index: u16,
    label: &'static str,
) -> Result<u64, WasmError> {
    Ok(unsafe { frame_read(frame, region_slot(region, index, label)?) })
}

#[inline]
unsafe fn region_write(
    frame: *mut u64,
    region: ExternalCallFrameRegion,
    index: u16,
    value: u64,
    label: &'static str,
) -> Result<(), WasmError> {
    unsafe {
        frame_write(frame, region_slot(region, index, label)?, value);
    }
    Ok(())
}

fn invoke_external_call(
    ctx: &mut NativeContext,
    frame: *mut u64,
    meta: &ExternalCallMeta,
) -> Result<(), WasmError> {
    let func_idx = match meta.target_kind()? {
        ExternalCallTargetKind::Immediate => meta.func_idx_source,
        ExternalCallTargetKind::FrameSlot => {
            let func_idx_slot = u16::try_from(meta.func_idx_source)
                .map_err(|_| internal_error("external-call func_idx source slot exceeds u16"))?;
            unsafe { frame_read(frame, func_idx_slot) as u32 }
        }
    };
    call_external_by_index(ctx, frame, func_idx, meta.args, meta.results)
}

fn call_external_by_index(
    ctx: &mut NativeContext,
    frame: *mut u64,
    func_idx: u32,
    args_region: ExternalCallFrameRegion,
    results_region: ExternalCallFrameRegion,
) -> Result<(), WasmError> {
    let (func_type, callback) = {
        let store = ctx
            .store()
            .ok_or_else(|| internal_error("external-call context is missing store"))?;
        let callee = store
            .module()
            .functions
            .get(func_idx as usize)
            .ok_or_else(|| internal_error("external-call referenced invalid function index"))?;
        match callee {
            FunctionInst::External {
                func_type,
                callback,
            } => (Rc::clone(func_type), *callback),
            FunctionInst::Local { .. } => {
                return Err(internal_error(
                    "external-call entry received a local function target",
                ))
            }
        }
    };

    require_region_slots(
        args_region,
        func_type.params().len() as u16,
        "external-call args span does not match function arity",
    )?;
    require_region_slots(
        results_region,
        func_type.results().len() as u16,
        "external-call result span does not match function arity",
    )?;

    let args: collections::Vec<Value> = func_type
        .params()
        .iter()
        .enumerate()
        .map(|(index, ty)| unsafe {
            region_read(
                frame,
                args_region,
                index as u16,
                "external-call arg slot is out of bounds",
            )
            .map(|raw| raw_to_value(raw, *ty))
        })
        .collect::<Result<_, _>>()?;
    let mut ret_vals = collections::vec![Value::default(); func_type.results().len()];

    let mem_slice = {
        let store = ctx
            .store_mut()
            .ok_or_else(|| internal_error("external-call context is missing store"))?;
        if store.module().memories.is_empty() {
            None
        } else {
            let mem = store.memory_mut(0);
            let ptr = mem.memory_ptr();
            let len = mem.memory_len();
            if len == 0 {
                None
            } else {
                Some(unsafe { core::slice::from_raw_parts_mut(ptr, len) })
            }
        }
    };
    let mut caller = Caller::new(mem_slice);
    callback(&mut caller, &args, &mut ret_vals)?;

    if ret_vals.len() != func_type.results().len() {
        return Err(internal_error(
            "external callback returned an unexpected result count",
        ));
    }

    for (index, value) in ret_vals.into_iter().enumerate() {
        unsafe {
            region_write(
                frame,
                results_region,
                index as u16,
                value_to_raw(value),
                "external-call result slot is out of bounds",
            )?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, rc::Rc, string::String};

    use super::*;
    use crate::{
        module::{type_context::TypeContext, type_defs::FunctionType},
        utils::limits::Limits,
        value_type::ValueType,
        vm::{
            entities::{ExternalFn, MemInst, ModuleInst},
            runtime::common::NativeCallStatus,
            store::Store,
        },
    };

    fn test_context(module: ModuleInst) -> (Box<Store>, NativeContext) {
        let mut store = Box::new(Store::new(module));
        let ctx = NativeContext::new((&mut *store) as *mut Store, core::ptr::null_mut());
        (store, ctx)
    }

    fn call_external<T: Copy>(ctx: &mut NativeContext, frame: &mut [u64], meta: &T) -> u32 {
        let entry = call_external_entry_ptr();
        unsafe { entry(ctx, frame.as_mut_ptr(), (meta as *const T).cast::<u8>()) }
    }

    #[test]
    fn external_call_marshals_frame_slots() {
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
        let mut module = ModuleInst::new(String::from("m"), TypeContext::empty());
        module.functions.push(FunctionInst::External {
            func_type,
            callback: host_add as ExternalFn,
        });
        module
            .memories
            .push(MemInst::new(Limits::new(1, Some(1)).unwrap()));
        let (_store, mut ctx) = test_context(module);
        let meta = ExternalCallMeta {
            func_idx_source: 0,
            func_idx_source_kind: ExternalCallTargetKind::Immediate as u32,
            args: ExternalCallFrameRegion {
                base_slot: 0,
                slots: 2,
            },
            results: ExternalCallFrameRegion {
                base_slot: 0,
                slots: 1,
            },
        };
        let mut frame = [7, 5, 99];

        let status = call_external(&mut ctx, &mut frame, &meta);

        assert_eq!(status, NativeCallStatus::Ok as u32);
        assert_eq!(frame[0], 12);
        assert!(ctx.error.is_none());
    }

    #[test]
    fn external_call_reads_dynamic_target_slot() {
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
        let mut module = ModuleInst::new(String::from("m"), TypeContext::empty());
        module.functions.push(FunctionInst::External {
            func_type,
            callback: host_add as ExternalFn,
        });
        module
            .memories
            .push(MemInst::new(Limits::new(1, Some(1)).unwrap()));
        let (_store, mut ctx) = test_context(module);
        let meta = ExternalCallMeta {
            func_idx_source: 2,
            func_idx_source_kind: ExternalCallTargetKind::FrameSlot as u32,
            args: ExternalCallFrameRegion {
                base_slot: 0,
                slots: 2,
            },
            results: ExternalCallFrameRegion {
                base_slot: 0,
                slots: 1,
            },
        };
        let mut frame = [7, 5, 0];

        let status = call_external(&mut ctx, &mut frame, &meta);

        assert_eq!(status, NativeCallStatus::Ok as u32);
        assert_eq!(frame[0], 12);
        assert!(ctx.error.is_none());
    }
}
