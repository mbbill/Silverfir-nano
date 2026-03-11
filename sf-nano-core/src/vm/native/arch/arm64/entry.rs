//! ARM64 native entry metadata.
//!
//! This file should stay limited to ARM64 entry/patch representation, not
//! frontend semantics.

use alloc::{rc::Rc, vec::Vec};
use core::slice;

use crate::{
    constants::MAX_CALL_STACK_DEPTH,
    error::WasmError,
    module::type_defs::FunctionType,
    vm::{
        entities::{Caller, FunctionInst},
        native::{
            code::NativeCode,
            context::NativeContext,
            ir::{NativeBlockId, NativeProgram},
            machine,
            runtime::layout,
        },
        value::Value,
    },
};

/// One unresolved ARM64 block-entry patch site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arm64EntryPatch {
    pub code_offset: u32,
    pub target: NativeBlockId,
}

const MAX_ARM64_HELPER_CALL_DEPTH: u64 = if MAX_CALL_STACK_DEPTH < 300 {
    MAX_CALL_STACK_DEPTH as u64
} else {
    300
};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct Arm64CallExternalRequest {
    pub func_idx: u64,
    pub args_ptr: *const u64,
    pub args_len: u64,
    pub results_ptr: *mut u64,
    pub results_len: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct Arm64CallIndirectRequest {
    pub caller_frame_slots: u64,
    pub type_idx: u64,
    pub table_idx: u64,
    pub index: u64,
    pub args_ptr: *const u64,
    pub args_len: u64,
    pub results_ptr: *mut u64,
    pub results_len: u64,
}

enum Arm64ResolvedCallee {
    Local {
        code: *const NativeCode,
        program: *const NativeProgram,
        params_len: usize,
        results_len: usize,
    },
    External {
        func_type: Rc<FunctionType>,
        callback: crate::vm::entities::ExternalFn,
    },
}

pub(super) unsafe extern "C" fn arm64_raise_exhaustion(ctx: *mut NativeContext, kind: u64) {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return;
    };
    let message = match kind {
        0 => "stack overflow",
        1 => "call stack exhausted",
        _ => "native execution failed",
    };
    ctx.error = Some(WasmError::exhaustion(message.into()));
}

pub(super) unsafe extern "C" fn arm64_raise_trap(ctx: *mut NativeContext, kind: u64) {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return;
    };
    let error = match kind {
        0 => WasmError::trap("out of bounds memory access".into()),
        1 => WasmError::invalid("global.set: immutable global".into()),
        2 => WasmError::invalid("native reached unreachable".into()),
        3 => WasmError::trap("integer divide by zero".into()),
        4 => WasmError::trap("integer overflow".into()),
        _ => WasmError::trap("native execution trapped".into()),
    };
    ctx.error = Some(error);
}

pub(super) unsafe extern "C" fn arm64_f64_add(lhs: u64, rhs: u64) -> u64 {
    f64::to_bits(f64::from_bits(lhs) + f64::from_bits(rhs))
}

pub(super) unsafe extern "C" fn arm64_f64_sub(lhs: u64, rhs: u64) -> u64 {
    f64::to_bits(f64::from_bits(lhs) - f64::from_bits(rhs))
}

pub(super) unsafe extern "C" fn arm64_f64_mul(lhs: u64, rhs: u64) -> u64 {
    f64::to_bits(f64::from_bits(lhs) * f64::from_bits(rhs))
}

pub(super) unsafe extern "C" fn arm64_f64_div(lhs: u64, rhs: u64) -> u64 {
    f64::to_bits(f64::from_bits(lhs) / f64::from_bits(rhs))
}

pub(super) unsafe extern "C" fn arm64_f64_eq(lhs: u64, rhs: u64) -> u64 {
    u64::from((f64::from_bits(lhs) == f64::from_bits(rhs)) as u32)
}

pub(super) unsafe extern "C" fn arm64_f64_ne(lhs: u64, rhs: u64) -> u64 {
    u64::from((f64::from_bits(lhs) != f64::from_bits(rhs)) as u32)
}

pub(super) unsafe extern "C" fn arm64_f64_lt(lhs: u64, rhs: u64) -> u64 {
    u64::from((f64::from_bits(lhs) < f64::from_bits(rhs)) as u32)
}

pub(super) unsafe extern "C" fn arm64_f64_gt(lhs: u64, rhs: u64) -> u64 {
    u64::from((f64::from_bits(lhs) > f64::from_bits(rhs)) as u32)
}

pub(super) unsafe extern "C" fn arm64_f64_neg(value: u64) -> u64 {
    f64::to_bits(-f64::from_bits(value))
}

pub(super) unsafe extern "C" fn arm64_f64_convert_i32_s(value: u64) -> u64 {
    f64::to_bits((value as i32) as f64)
}

pub(super) unsafe extern "C" fn arm64_f64_convert_i32_u(value: u64) -> u64 {
    f64::to_bits((value as u32) as f64)
}

pub(super) unsafe extern "C" fn arm64_i32_trunc_sat_f64_s(value: u64) -> u64 {
    let value = f64::from_bits(value);
    let out = if value.is_nan() {
        0
    } else if value >= i32::MAX as f64 {
        i32::MAX
    } else if value <= i32::MIN as f64 {
        i32::MIN
    } else {
        value.trunc() as i32
    };
    u64::from(out as u32)
}

pub(super) unsafe extern "C" fn arm64_i32_trunc_sat_f64_u(value: u64) -> u64 {
    let value = f64::from_bits(value);
    let out = if value.is_nan() || value <= 0.0 {
        0
    } else if value >= u32::MAX as f64 {
        u32::MAX
    } else {
        value.trunc() as u32
    };
    u64::from(out)
}

pub(super) unsafe extern "C" fn arm64_memory_fill_helper(
    ctx: *mut NativeContext,
    dst: u64,
    value: u64,
    size: u64,
    mem_idx: u64,
) -> u64 {
    let result = (|| -> Result<(), WasmError> {
        let ctx = unsafe { ctx.as_mut() }
            .ok_or_else(|| WasmError::internal("memory.fill helper missing context".into()))?;
        let store = unsafe { ctx.store.as_mut() }
            .ok_or_else(|| WasmError::internal("memory.fill helper missing store".into()))?;
        let memory = store.memory_mut(mem_idx as usize);
        let dst = dst as usize;
        let size = size as usize;
        if dst.saturating_add(size) > memory.data.len() {
            return Err(WasmError::trap("out of bounds memory access".into()));
        }
        memory.data[dst..dst + size].fill(value as u8);
        ctx.refresh_cached_views();
        Ok(())
    })();
    match result {
        Ok(()) => 0,
        Err(error) => {
            if let Some(ctx) = unsafe { ctx.as_mut() } {
                ctx.error = Some(error);
            }
            1
        }
    }
}

pub(super) unsafe extern "C" fn arm64_memory_copy_helper(
    ctx: *mut NativeContext,
    dst: u64,
    src: u64,
    size: u64,
    dst_idx: u64,
    src_idx: u64,
) -> u64 {
    let result = (|| -> Result<(), WasmError> {
        let ctx = unsafe { ctx.as_mut() }
            .ok_or_else(|| WasmError::internal("memory.copy helper missing context".into()))?;
        let store = unsafe { ctx.store.as_mut() }
            .ok_or_else(|| WasmError::internal("memory.copy helper missing store".into()))?;
        let dst_idx = dst_idx as usize;
        let src_idx = src_idx as usize;
        let dst = dst as usize;
        let src = src as usize;
        let size = size as usize;

        if dst_idx == src_idx {
            let memory = store.memory_mut(dst_idx);
            if src.saturating_add(size) > memory.data.len()
                || dst.saturating_add(size) > memory.data.len()
            {
                return Err(WasmError::trap("out of bounds memory access".into()));
            }
            memory.data.copy_within(src..src + size, dst);
            ctx.refresh_cached_views();
            return Ok(());
        }

        let module = store.module_mut();
        let (src_mem, dst_mem) = if src_idx < dst_idx {
            let (left, right) = module.memories.split_at_mut(dst_idx);
            (&left[src_idx], &mut right[0])
        } else {
            let (left, right) = module.memories.split_at_mut(src_idx);
            (&right[0], &mut left[dst_idx])
        };
        if src.saturating_add(size) > src_mem.data.len()
            || dst.saturating_add(size) > dst_mem.data.len()
        {
            return Err(WasmError::trap("out of bounds memory access".into()));
        }
        dst_mem.data[dst..dst + size].copy_from_slice(&src_mem.data[src..src + size]);
        ctx.refresh_cached_views();
        Ok(())
    })();
    match result {
        Ok(()) => 0,
        Err(error) => {
            if let Some(ctx) = unsafe { ctx.as_mut() } {
                ctx.error = Some(error);
            }
            1
        }
    }
}

pub(super) unsafe extern "C" fn arm64_call_external_helper(
    ctx: *mut NativeContext,
    _fp: *mut u64,
    req: *const Arm64CallExternalRequest,
) -> u64 {
    let result = (|| -> Result<(), WasmError> {
        let ctx = unsafe { ctx.as_mut() }
            .ok_or_else(|| WasmError::internal("call_external helper missing context".into()))?;
        let req = unsafe { req.as_ref() }
            .ok_or_else(|| WasmError::internal("call_external helper missing request".into()))?;
        let args = unsafe { slice::from_raw_parts(req.args_ptr, req.args_len as usize) };
        let results = unsafe { slice::from_raw_parts_mut(req.results_ptr, req.results_len as usize) };

        let (func_type, callback) = {
            let store = unsafe { ctx.store.as_ref() }.ok_or_else(|| {
                WasmError::internal("call_external helper missing store".into())
            })?;
            match store.function(req.func_idx as usize) {
                FunctionInst::External {
                    func_type,
                    callback,
                } => (func_type.clone(), *callback),
                FunctionInst::Local { .. } => {
                    return Err(WasmError::internal(
                        "call_external helper resolved local callee".into(),
                    ))
                }
            }
        };

        invoke_external_callback(ctx, func_type, callback, args, results)
    })();
    match result {
        Ok(()) => 0,
        Err(error) => {
            if let Some(ctx) = unsafe { ctx.as_mut() } {
                ctx.error = Some(error);
            }
            1
        }
    }
}

pub(super) unsafe extern "C" fn arm64_call_indirect_helper(
    ctx: *mut NativeContext,
    fp: *mut u64,
    req: *const Arm64CallIndirectRequest,
) -> u64 {
    let result = (|| -> Result<(), WasmError> {
        let ctx = unsafe { ctx.as_mut() }
            .ok_or_else(|| WasmError::internal("call_indirect helper missing context".into()))?;
        let req = unsafe { req.as_ref() }
            .ok_or_else(|| WasmError::internal("call_indirect helper missing request".into()))?;
        let args = unsafe { slice::from_raw_parts(req.args_ptr, req.args_len as usize) };
        let results = unsafe { slice::from_raw_parts_mut(req.results_ptr, req.results_len as usize) };

        let func_idx = {
            let store = unsafe { ctx.store.as_ref() }.ok_or_else(|| {
                WasmError::internal("call_indirect helper missing store".into())
            })?;
            let table = store.table(req.table_idx as usize);
            let handle = table.elements.get(req.index as usize).ok_or_else(|| {
                WasmError::invalid("call_indirect: table index out of range".into())
            })?;
            if handle.is_null() {
                return Err(WasmError::invalid(
                    "call_indirect: null function reference".into(),
                ));
            }
            handle.raw_value() as u32
        };

        {
            let store = unsafe { ctx.store.as_ref() }.ok_or_else(|| {
                WasmError::internal("call_indirect helper missing store".into())
            })?;
            let expected = store
                .module()
                .get_type(req.type_idx as u32)
                .ok_or_else(|| WasmError::invalid("call_indirect: type index out of range".into()))?;
            let actual = store.function(func_idx as usize).func_type();
            if expected.as_ref() != actual {
                return Err(WasmError::invalid(
                    "call_indirect: signature mismatch".into(),
                ));
            }
        }

        match resolve_callee(
            unsafe { ctx.store.as_ref() }
                .ok_or_else(|| WasmError::internal("call_indirect helper missing store".into()))?,
            func_idx,
        )? {
            Arm64ResolvedCallee::Local {
                code,
                program,
                params_len,
                results_len,
            } => invoke_local_callee(
                ctx,
                fp,
                req.caller_frame_slots as usize,
                unsafe { &*code },
                unsafe { &*program },
                params_len,
                results_len,
                args,
                results,
            ),
            Arm64ResolvedCallee::External {
                func_type,
                callback,
            } => invoke_external_callback(ctx, func_type, callback, args, results),
        }
    })();
    match result {
        Ok(()) => 0,
        Err(error) => {
            if let Some(ctx) = unsafe { ctx.as_mut() } {
                ctx.error = Some(error);
            }
            1
        }
    }
}

/// Shared Rust target for the bring-up ARM64 entry trampoline.
///
/// The generated ARM64 stub tail-branches here with the native entry ABI
/// intact. This keeps the ISA layer real while the per-op ARM64 emitter is
/// still being brought up.
pub(super) unsafe extern "C" fn shared_native_entry(
    ctx: *mut NativeContext,
    fp: *mut u64,
    _l0: u64,
    _l1: u64,
    _l2: u64,
    _t0: u64,
    _t1: u64,
    _t2: u64,
    _t3: u64,
) {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return;
    };
    let Some(program) = current_program(ctx as *mut NativeContext) else {
        return;
    };
    let program = unsafe { &*program };

    let result = machine::execute_program(ctx, program, fp);
    ctx.refresh_cached_views();
    if let Err(error) = result {
        ctx.error = Some(error);
    }
}

fn current_program(ctx: *mut NativeContext) -> Option<*const NativeProgram> {
    let Some(ctx_ref) = (unsafe { ctx.as_mut() }) else {
        return None;
    };
    let Some(code) = (unsafe { ctx_ref.current_code.as_ref() }) else {
        ctx_ref.error = Some(WasmError::internal(
            "native entry called without current code".into(),
        ));
        return None;
    };
    let Some(program) = code.program() else {
        ctx_ref.error = Some(WasmError::internal(
            "native code is missing finalized program".into(),
        ));
        return None;
    };
    Some(program as *const NativeProgram)
}

fn resolve_callee(store: &crate::vm::store::Store, func_idx: u32) -> Result<Arm64ResolvedCallee, WasmError> {
    let func = store.function(func_idx as usize);
    Ok(match func {
        FunctionInst::Local { spec, .. } => {
            let code = spec.get_native_code().ok_or_else(|| {
                WasmError::internal("local callee is missing native code".into())
            })?;
            let program = code.program().ok_or_else(|| {
                WasmError::internal("local callee is missing finalized native program".into())
            })?;
            Arm64ResolvedCallee::Local {
                code: code as *const NativeCode,
                program: program as *const NativeProgram,
                params_len: func.func_type().params().len(),
                results_len: func.func_type().results().len(),
            }
        }
        FunctionInst::External {
            func_type,
            callback,
        } => Arm64ResolvedCallee::External {
            func_type: func_type.clone(),
            callback: *callback,
        },
    })
}

fn invoke_external_callback(
    ctx: &mut NativeContext,
    func_type: Rc<FunctionType>,
    callback: crate::vm::entities::ExternalFn,
    args: &[u64],
    results: &mut [u64],
) -> Result<(), WasmError> {
    if func_type.params().len() != args.len() || func_type.results().len() != results.len() {
        return Err(WasmError::internal(
            "native helper arity does not match function type".into(),
        ));
    }

    let wasm_args = args
        .iter()
        .zip(func_type.params().iter().copied())
        .map(|(raw, ty)| Value::from_raw(*raw, ty))
        .collect::<Vec<_>>();
    let mut returns = func_type
        .results()
        .iter()
        .copied()
        .map(Value::default_for_type)
        .collect::<Vec<_>>();

    let store = unsafe { ctx.store.as_mut() }
        .ok_or_else(|| WasmError::internal("native helper missing store".into()))?;
    let mem_slice = if store.module().memories.is_empty() {
        None
    } else {
        Some(store.module_mut().memories[0].data.as_mut_slice())
    };
    let mut caller = Caller::new(mem_slice);
    callback(&mut caller, &wasm_args, &mut returns)?;

    for (slot, value) in results.iter_mut().zip(returns.into_iter()) {
        *slot = value.to_raw();
    }
    ctx.refresh_cached_views();
    Ok(())
}

fn invoke_local_callee(
    ctx: &mut NativeContext,
    caller_fp: *mut u64,
    caller_frame_slots: usize,
    code: &NativeCode,
    program: &NativeProgram,
    params_len: usize,
    results_len: usize,
    args: &[u64],
    results: &mut [u64],
) -> Result<(), WasmError> {
    if params_len != args.len() || results_len != results.len() {
        return Err(WasmError::internal(
            "native helper local-call arity does not match function type".into(),
        ));
    }
    let callee_slots = layout::frame_slots_used(program);
    let callee_fp = unsafe { caller_fp.add(caller_frame_slots) };
    let callee_end = unsafe { callee_fp.add(callee_slots) };
    if callee_end > ctx.stack_end {
        return Err(WasmError::exhaustion("stack overflow".into()));
    }
    if ctx.call_depth >= MAX_ARM64_HELPER_CALL_DEPTH {
        return Err(WasmError::exhaustion("call stack exhausted".into()));
    }

    unsafe {
        for (index, value) in args.iter().copied().enumerate() {
            *callee_fp.add(index) = value;
        }
        let frame_prefix_slots = program.frame.frame_size as usize;
        if frame_prefix_slots > params_len {
            core::ptr::write_bytes(
                callee_fp.add(params_len),
                0,
                frame_prefix_slots - params_len,
            );
        }
    }

    let saved_code = ctx.current_code;
    ctx.current_code = code as *const NativeCode;
    ctx.call_depth = ctx.call_depth.saturating_add(1);
    let result = if code.internal_entry().is_some() {
        let entry = code
            .entry
            .ok_or_else(|| WasmError::internal("direct local callee is missing entry".into()))?;
        unsafe {
            entry(ctx, callee_fp, 0, 0, 0, 0, 0, 0, 0);
        }
        match ctx.error.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    } else {
        machine::execute_program(ctx, program, callee_fp)
    };
    ctx.current_code = saved_code;
    ctx.call_depth = ctx.call_depth.saturating_sub(1);
    result?;

    for (index, slot) in results.iter_mut().enumerate() {
        *slot = unsafe { *callee_fp.add(index) };
    }
    ctx.refresh_cached_views();
    Ok(())
}
