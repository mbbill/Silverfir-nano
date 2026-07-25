//! Windows x86_64 trap handler using a Vectored Exception Handler (VEH).
//!
//! Registered first in the VEH chain via `AddVectoredExceptionHandler(1, ...)`.
//! For every access violation we ask the generic trap table whether the
//! faulting RIP belongs to a JIT region. If it does, we patch RAX/RIP and
//! resume; otherwise we hand the exception back to the rest of the chain
//! (debugger, OS unhandled-exception filter) by returning
//! `EXCEPTION_CONTINUE_SEARCH` instead of aborting — VEH sees every
//! exception in the process and aborting on non-JIT faults would break
//! legitimate exception flows in the host.

use crate::vm::jit::runtime::trap_signal;

const EXCEPTION_ACCESS_VIOLATION: u32 = 0xC000_0005;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;

#[repr(C)]
struct ExceptionRecord {
    exception_code: u32,
    exception_flags: u32,
    exception_record: *mut ExceptionRecord,
    exception_address: *mut u8,
    number_parameters: u32,
    _align: u32,
    exception_information: [usize; 15],
}

#[repr(C)]
struct ExceptionPointers {
    exception_record: *mut ExceptionRecord,
    context_record: *mut u8, // CONTEXT*; fields read via raw offsets below
}

// Offsets into Windows x64 CONTEXT (winnt.h). The struct is large and the
// integer GP block sits in the middle, so we use raw byte offsets rather
// than a full repr(C) mirror.
const CTX_RAX: usize = 0x78;
const CTX_RBX: usize = 0x90;
const CTX_RIP: usize = 0xF8;

unsafe extern "system" {
    fn AddVectoredExceptionHandler(
        first: u32,
        handler: unsafe extern "system" fn(*mut ExceptionPointers) -> i32,
    ) -> *mut u8;
}

unsafe extern "system" fn veh(info: *mut ExceptionPointers) -> i32 {
    let er = unsafe { (*info).exception_record };
    if unsafe { (*er).exception_code } != EXCEPTION_ACCESS_VIOLATION {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    let ctx = unsafe { (*info).context_record };
    let pc = unsafe { *(ctx.add(CTX_RIP) as *const u64) } as usize;

    let Some(resolution) = (unsafe { trap_signal::try_resolve_trap(pc) }) else {
        // Not a JIT fault — let other handlers / the OS deal with it.
        return EXCEPTION_CONTINUE_SEARCH;
    };

    // Storm guard, only counted for JIT-attributed faults. Aborts the
    // process the same way the POSIX handlers do.
    if trap_signal::signal_count_inc_and_check() {
        std::process::abort();
    }

    // RBX = MACHINE_CTX_REG (NativeContext pointer) in our x86_64 mapping.
    let ctx_ptr = unsafe { *(ctx.add(CTX_RBX) as *const u64) } as *mut u8;
    if resolution.trap_kind_offset > 0 {
        let fault_addr = unsafe { (*er).exception_information[1] };
        let trap_kind = unsafe { trap_signal::classify_trap_kind(ctx_ptr, fault_addr, resolution) };
        let trap_kind_ptr = unsafe { ctx_ptr.add(resolution.trap_kind_offset) as *mut u32 };
        unsafe { *trap_kind_ptr = trap_kind };
    }

    // RAX = 1 (error status for eval()), RIP = function's return_error_label.
    unsafe {
        *(ctx.add(CTX_RAX) as *mut u64) = 1;
        *(ctx.add(CTX_RIP) as *mut u64) = resolution.error_ret as u64;
    }

    EXCEPTION_CONTINUE_EXECUTION
}

pub(in crate::vm::jit::runtime) unsafe fn install_platform_handler() {
    // First = 1 inserts the handler at the head of the VEH chain so we get
    // first crack at access violations before any user-installed handler.
    let _ = unsafe { AddVectoredExceptionHandler(1, veh) };
}
