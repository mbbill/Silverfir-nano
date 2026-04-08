//! macOS ARM64 signal handler using Darwin's ucontext_t layout.

use crate::vm::runtime::trap_signal;

const SIGSEGV: i32 = 11;
const SIGBUS: i32 = 10;
const SA_SIGINFO: i32 = 0x0040;

#[repr(C)]
struct sigaction {
    // On Darwin ARM64, __sigaction_u is a union; with SA_SIGINFO the
    // sa_sigaction field is used. We declare the whole union as one
    // function-pointer field since both variants have the same size.
    sa_sigaction: unsafe extern "C" fn(i32, *mut u8, *mut u8),
    sa_mask: u32,
    sa_flags: i32,
}

unsafe extern "C" {
    fn sigaction(sig: i32, act: *const sigaction, oldact: *mut sigaction) -> i32;
}

// Darwin ARM64 mcontext layout (from <mach/arm/_structs.h>).
// We only need the exception-state and thread-state portions.
#[repr(C)]
struct Arm64ThreadState {
    x: [u64; 29], // X0-X28
    fp: u64,      // X29
    lr: u64,      // X30
    sp: u64,
    pc: u64,
    cpsr: u32,
    _pad: u32,
}

/// Offsets into Darwin's `ucontext_t` to reach the thread state.
/// On ARM64 macOS the layout is:
///   ucontext_t.uc_mcontext → pointer to __darwin_mcontext64
///   __darwin_mcontext64.__es (exception state, 16 bytes: far:u64 + esr:u32 + exception:u32)
///   __darwin_mcontext64.__ss (thread state = Arm64ThreadState)
const UCONTEXT_MCONTEXT_OFFSET: usize = 48; // uc_mcontext field offset
const MCONTEXT_SS_OFFSET: usize = 16; // skip __es (exception state)

unsafe fn thread_state(ucontext: *mut u8) -> *mut Arm64ThreadState {
    let mctx_ptr = unsafe { *(ucontext.add(UCONTEXT_MCONTEXT_OFFSET) as *const *mut u8) };
    unsafe { mctx_ptr.add(MCONTEXT_SS_OFFSET) as *mut Arm64ThreadState }
}

unsafe extern "C" fn signal_handler(_sig: i32, _info: *mut u8, ucontext: *mut u8) {
    if trap_signal::signal_count_inc_and_check() {
        std::process::abort();
    }

    let ts = unsafe { thread_state(ucontext) };
    let pc = unsafe { (*ts).pc as usize };

    let Some((error_ret, trap_kind_offset)) = (unsafe { trap_signal::try_resolve_trap(pc) }) else {
        // Not in JIT code — abort (we can't chain easily without libc).
        std::process::abort();
    };

    // Read X19 (NativeContext pointer) from the faulting thread state.
    let ctx_ptr = unsafe { (*ts).x[19] } as *mut u8;

    // Set ctx.trap_kind = 1 (MemoryOutOfBounds).
    if trap_kind_offset > 0 {
        let trap_kind_ptr = unsafe { ctx_ptr.add(trap_kind_offset) as *mut u32 };
        unsafe { *trap_kind_ptr = 1 };
    }

    // Set X0 = 1 (error status for eval())
    unsafe { (*ts).x[0] = 1 };

    // Redirect PC to the function's return_error_label
    unsafe { (*ts).pc = error_ret as u64 };
}

pub(in crate::vm::runtime) unsafe fn install_platform_handler() {
    let act = sigaction {
        sa_sigaction: signal_handler,
        sa_mask: 0,
        sa_flags: SA_SIGINFO,
    };
    unsafe {
        sigaction(SIGSEGV, &act, core::ptr::null_mut());
        sigaction(SIGBUS, &act, core::ptr::null_mut());
    }
}
