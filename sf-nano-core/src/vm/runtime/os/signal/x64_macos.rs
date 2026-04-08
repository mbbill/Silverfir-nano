//! macOS x86_64 signal handler using Darwin's ucontext_t layout.

use crate::vm::runtime::trap_signal;

const SIGSEGV: i32 = 11;
const SIGBUS: i32 = 10;
const SA_SIGINFO: i32 = 0x0040;

#[repr(C)]
struct sigaction {
    sa_sigaction: unsafe extern "C" fn(i32, *mut u8, *mut u8),
    sa_mask: u32,
    sa_flags: i32,
}

unsafe extern "C" {
    fn sigaction(sig: i32, act: *const sigaction, oldact: *mut sigaction) -> i32;
}

// Darwin x86_64 mcontext layout (from <i386/_structs.h>):
//   __darwin_mcontext64.__es (exception state, 16 bytes)
//   __darwin_mcontext64.__ss (thread state = x86_thread_state64_t)
// x86_thread_state64_t layout:
//   rax, rbx, rcx, rdx, rdi, rsi, rbp, rsp, r8-r15, rip, rflags, cs, fs, gs
const UCONTEXT_MCONTEXT_OFFSET: usize = 48;
const MCONTEXT_SS_OFFSET: usize = 16; // skip __es (exception state)

#[repr(C)]
struct X86ThreadState64 {
    rax: u64,
    rbx: u64,
    rcx: u64,
    rdx: u64,
    rdi: u64,
    rsi: u64,
    rbp: u64,
    rsp: u64,
    r8: u64,
    r9: u64,
    r10: u64,
    r11: u64,
    r12: u64,
    r13: u64,
    r14: u64,
    r15: u64,
    rip: u64,
    rflags: u64,
    cs: u64,
    fs: u64,
    gs: u64,
}

unsafe fn thread_state(ucontext: *mut u8) -> *mut X86ThreadState64 {
    let mctx_ptr = unsafe { *(ucontext.add(UCONTEXT_MCONTEXT_OFFSET) as *const *mut u8) };
    unsafe { mctx_ptr.add(MCONTEXT_SS_OFFSET) as *mut X86ThreadState64 }
}

unsafe extern "C" fn signal_handler(_sig: i32, _info: *mut u8, ucontext: *mut u8) {
    if trap_signal::signal_count_inc_and_check() {
        std::process::abort();
    }

    let ts = unsafe { thread_state(ucontext) };
    let pc = unsafe { (*ts).rip as usize };

    let Some((error_ret, trap_kind_offset)) = (unsafe { trap_signal::try_resolve_trap(pc) }) else {
        std::process::abort();
    };

    // RBX = MACHINE_CTX_REG (NativeContext pointer) in our x86_64 mapping.
    let ctx_ptr = unsafe { (*ts).rbx } as *mut u8;

    if trap_kind_offset > 0 {
        let trap_kind_ptr = unsafe { ctx_ptr.add(trap_kind_offset) as *mut u32 };
        unsafe { *trap_kind_ptr = 1 };
    }

    // Set RAX = 1 (error status for eval())
    unsafe { (*ts).rax = 1 };

    // Redirect RIP to the function's return_error_label
    unsafe { (*ts).rip = error_ret as u64 };
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
