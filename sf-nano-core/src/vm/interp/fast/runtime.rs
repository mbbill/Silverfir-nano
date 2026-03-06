//! Fast interpreter runtime entry point.
//!
//! Uses an owned stack (no thread_local!) for no_std compatibility.

use alloc::string::ToString;
use alloc::vec;

use crate::error::WasmError;
use crate::vm::entities::{FunctionInst, MemInst, ModuleInst};
use crate::vm::interp::fast::context::Context;
use crate::vm::interp::fast::handlers::run_trampoline;
use crate::vm::interp::stack::InterpreterStack;
use crate::vm::store::Store;
use crate::vm::value::Value;

/// Maximum number of u64 slots in the fast interpreter stack.
const MAX_SLOTS: usize = crate::constants::MAX_STACK_SIZE / core::mem::size_of::<u64>();

/// Evaluate a function with the given arguments.
pub fn eval(
    func_inst: &FunctionInst,
    store: &mut Store,
    args: &[Value],
) -> Result<InterpreterStack, WasmError> {
    // Handle external functions directly (host callbacks, imported functions)
    if let FunctionInst::External { func_type, callback } = func_inst {
        let params = func_type.params();
        let results = func_type.results();
        if args.len() != params.len() {
            return Err(WasmError::invalid(alloc::format!(
                "invalid argument count: got {}, expected {}",
                args.len(),
                params.len()
            )));
        }

        let mut ret_vals = alloc::vec![Value::default(); results.len()];
        let mem_slice = if !store.module().memories.is_empty() {
            let mem = &store.module().memories[0] as *const MemInst as *mut MemInst;
            unsafe { Some((*mem).data.as_mut_slice()) }
        } else {
            None
        };
        let mut caller = crate::vm::entities::Caller::new(mem_slice);
        callback(&mut caller, args, &mut ret_vals)?;

        let mut out = InterpreterStack::with_exact_capacity(results.len());
        for v in &ret_vals {
            out.push(v.to_raw());
        }
        return Ok(out);
    }

    let ft = match func_inst {
        FunctionInst::Local { spec, .. } => spec.func_type(),
        _ => unreachable!(),
    };
    let params_len = ft.params().len();
    if args.len() != params_len {
        return Err(WasmError::invalid(alloc::format!(
            "invalid argument count: got {}, expected {}",
            args.len(),
            params_len
        )));
    }

    // Allocate stack
    let mut stack = vec![0u64; MAX_SLOTS];
    let stack_base = stack.as_mut_ptr();
    let stack_end = unsafe { stack_base.add(MAX_SLOTS) };

    // Write args to stack
    unsafe {
        for (i, a) in args.iter().enumerate() {
            core::ptr::write(stack_base.add(i), a.to_raw());
        }
    }

    // Run
    internal_eval(func_inst, store, stack_base, stack_end, args.len())?;

    // Read results
    let results_len = ft.results().len();
    let mut out = InterpreterStack::with_exact_capacity(results_len);
    unsafe {
        for i in 0..results_len {
            out.push(core::ptr::read(stack_base.add(i)));
        }
    }
    Ok(out)
}

/// Internal evaluation entry point.
pub fn internal_eval(
    func_inst: &FunctionInst,
    store: &mut Store,
    stack_base: *mut u64,
    stack_end: *mut u64,
    sp_offset: usize,
) -> Result<(), WasmError> {
    let spec = match func_inst {
        FunctionInst::Local { spec, .. } => spec,
        FunctionInst::External { .. } => {
            return Err(WasmError::internal(
                "external functions should not reach interpreter".into(),
            ));
        }
    };

    // Ensure fast IR is built
    if !spec.has_fast_code() {
        crate::vm::interp::fast::precompile::precompile_module_two_pass(store)?;
    }

    // Compute frame pointer
    let ft = spec.func_type();
    let params_len = ft.params().len();
    if sp_offset < params_len {
        return Err(WasmError::internal("invalid stack size".into()));
    }
    let fp_index = sp_offset - params_len;
    let fp = unsafe { stack_base.add(fp_index) };

    let locals_len = spec.locals().len();

    // Zero the locals
    if locals_len > 0 {
        unsafe { core::ptr::write_bytes(fp.add(params_len), 0, locals_len) };
    }

    // Get memory 0 base/size
    let (heap_base, heap_size) = if !store.module().memories.is_empty() {
        let m = &store.module().memories[0];
        (m.data.as_ptr() as *mut u8, m.data.len())
    } else {
        (core::ptr::null_mut(), 0usize)
    };

    // Build context
    let module_ptr = store.module() as *const ModuleInst;
    let store_ptr = store as *mut Store;
    let mut ctx = Context::new(store_ptr, module_ptr, stack_end, heap_base, heap_size as u64);

    // Set up sentinel metadata for entry frame
    let frame_size = params_len + locals_len;
    unsafe {
        *fp.add(frame_size) = 0; // return_pc = NULL (sentinel)
        *fp.add(frame_size + 1) = 0; // saved_fp = NULL
        *fp.add(frame_size + 2) = 0; // saved_module = 0
    }

    let entry = spec.fast_cache().entry();
    debug_assert!(!entry.is_null());

    // Cache TERM_INST pointer in context
    ctx.hot.term_inst = super::handlers::term();

    unsafe {
        let nh: super::handlers::NextHandler = core::mem::transmute((*entry.add(1)).handler);
        run_trampoline(&mut ctx, entry, fp, 0, 0, 0, 0, 0, 0, 0, nh);
    }

    // Convert deferred C trap message to WasmError
    if !ctx.hot.trap_message.is_null() && ctx.error.is_none() {
        let msg = unsafe {
            core::ffi::CStr::from_ptr(ctx.hot.trap_message)
                .to_str()
                .unwrap_or("trap")
        };
        ctx.error = Some(WasmError::trap(msg.to_string()));
    }

    if let Some(error) = ctx.error {
        return Err(error);
    }

    Ok(())
}
