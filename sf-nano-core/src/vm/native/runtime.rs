//! Native runtime entry point.

use alloc::string::ToString;
use alloc::vec;
use core::arch::global_asm;

use crate::error::WasmError;
use crate::vm::entities::{FunctionInst, MemInst, ModuleInst};
use crate::vm::interp::stack::InterpreterStack;
use crate::vm::store::Store;
use crate::vm::value::Value;

use super::context::Context;
use super::instruction::{NativeEntry, NativeInst};

const MAX_SLOTS: usize = crate::constants::MAX_STACK_SIZE / core::mem::size_of::<u64>();

#[allow(improper_ctypes)]
unsafe extern "C" {
    fn native_run_entry(
        ctx: *mut Context,
        pc: *mut NativeInst,
        fp: *mut u64,
        l0: u64,
        l1: u64,
        l2: u64,
        t0: u64,
        t1: u64,
        t2: u64,
        t3: u64,
    );

    fn native_term();
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    #[test]
    fn test_native_run_entry_can_enter_terminal_instruction() {
        let mut stack = [0u64; 16];
        let stack_end = stack.as_mut_ptr().wrapping_add(stack.len());
        let mut ctx = Context::new(
            core::ptr::null_mut(),
            core::ptr::null(),
            stack_end,
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_pc = term();

        let mut insts = [
            NativeInst::new_entry_only(term_entry()),
            NativeInst::new_entry_only(term_entry()),
        ];

        unsafe {
            native_run_entry(
                &mut ctx,
                insts.as_mut_ptr(),
                stack.as_mut_ptr(),
                0, 0, 0, 0, 0, 0, 0,
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
global_asm!(
    r#"
    .text
    .p2align 2
    .global _native_run_entry
_native_run_entry:
    sub sp, sp, #80
    stp x19, x20, [sp, #0]
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    stp x27, x28, [sp, #64]
    mov x19, x0
    mov x20, x1
    mov x21, x2
    mov x22, x3
    mov x23, x4
    mov x24, x5
    mov x25, x6
    mov x26, x7
    ldr x27, [sp, #80]
    ldr x28, [sp, #88]
    ldr x16, [x20]
    br x16

    .p2align 2
    .global _native_term
_native_term:
    ldp x19, x20, [sp, #0]
    ldp x21, x22, [sp, #16]
    ldp x23, x24, [sp, #32]
    ldp x25, x26, [sp, #48]
    ldp x27, x28, [sp, #64]
    add sp, sp, #80
    ret
"#
);

pub fn term_entry() -> NativeEntry {
    native_term
}

static mut TERM_INST: [NativeInst; 2] = [
    NativeInst {
        entry: native_term,
        imm0: 0,
        imm1: 0,
        imm2: 0,
    },
    NativeInst {
        entry: native_term,
        imm0: 0,
        imm1: 0,
        imm2: 0,
    },
];

#[inline]
pub fn term() -> *mut NativeInst {
    unsafe { core::ptr::addr_of_mut!(TERM_INST[0]) }
}

pub fn eval(
    func_inst: &FunctionInst,
    store: &mut Store,
    args: &[Value],
) -> Result<InterpreterStack, WasmError> {
    let FunctionInst::Local { spec, .. } = func_inst else {
        return Err(WasmError::invalid(
            "native runtime only supports local functions".into(),
        ));
    };

    let ft = spec.func_type();
    let params_len = ft.params().len();
    if args.len() != params_len {
        return Err(WasmError::invalid(alloc::format!(
            "invalid argument count: got {}, expected {}",
            args.len(),
            params_len
        )));
    }

    let mut stack = vec![0u64; MAX_SLOTS];
    let stack_base = stack.as_mut_ptr();
    let stack_end = unsafe { stack_base.add(MAX_SLOTS) };

    unsafe {
        for (i, a) in args.iter().enumerate() {
            core::ptr::write(stack_base.add(i), a.to_raw());
        }
    }

    internal_eval(func_inst, store, stack_base, stack_end, args.len())?;

    let results_len = ft.results().len();
    let mut out = InterpreterStack::with_exact_capacity(results_len);
    unsafe {
        for i in 0..results_len {
            out.push(core::ptr::read(stack_base.add(i)));
        }
    }
    Ok(out)
}

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
            return Err(WasmError::internal("external functions should not reach native runtime".into()))
        }
    };

    if !spec.has_native_code() {
        crate::vm::native::precompile::precompile_module(store)?;
    }
    if !spec.has_native_code() {
        return Err(WasmError::invalid("native backend unavailable for function".into()));
    }

    let ft = spec.func_type();
    let params_len = ft.params().len();
    if sp_offset < params_len {
        return Err(WasmError::internal("invalid stack size".into()));
    }
    let fp_index = sp_offset - params_len;
    let fp = unsafe { stack_base.add(fp_index) };

    let locals_len = spec.locals().len();
    if locals_len > 0 {
        unsafe { core::ptr::write_bytes(fp.add(params_len), 0, locals_len) };
    }

    let (heap_base, heap_size) = if !store.module().memories.is_empty() {
        let m = &store.module().memories[0];
        (m.data.as_ptr() as *mut u8, m.data.len())
    } else {
        (core::ptr::null_mut(), 0usize)
    };

    let module_ptr = store.module() as *const ModuleInst;
    let store_ptr = store as *mut Store;
    let mut ctx = Context::new(store_ptr, module_ptr, stack_end, heap_base, heap_size as u64);

    let frame_size = params_len + locals_len;
    unsafe {
        *fp.add(frame_size) = 0;
        *fp.add(frame_size + 1) = 0;
        *fp.add(frame_size + 2) = 0;
    }

    let entry = spec.native_cache().entry();
    debug_assert!(!entry.is_null());
    ctx.hot.term_pc = term();

    unsafe {
        native_run_entry(&mut ctx, entry, fp, 0, 0, 0, 0, 0, 0, 0);
    }

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
