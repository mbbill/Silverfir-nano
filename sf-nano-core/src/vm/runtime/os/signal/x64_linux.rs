//! Linux x86_64 signal handler using the glibc/musl ucontext_t layout.

use crate::vm::runtime::trap_signal;

const SIGSEGV: i32 = 11;
const SIGBUS: i32 = 7;
const SA_SIGINFO: i32 = 4;
const SIGINFO_SI_ADDR_OFFSET: usize = 16;

#[repr(C)]
struct SigSet {
    __val: [u64; 16],
}

#[repr(C)]
struct libc_sigaction {
    sa_sigaction: unsafe extern "C" fn(i32, *mut u8, *mut u8),
    sa_mask: SigSet,
    sa_flags: i32,
    sa_restorer: usize,
}

unsafe extern "C" {
    fn sigaction(sig: i32, act: *const libc_sigaction, oldact: *mut libc_sigaction) -> i32;
}

unsafe fn siginfo_fault_addr(info: *mut u8) -> usize {
    unsafe { *(info.add(SIGINFO_SI_ADDR_OFFSET) as *const usize) }
}

// Linux x86_64 ucontext layout (glibc / musl sys/ucontext.h):
//   uc_flags           u64   @ 0
//   uc_link            ptr   @ 8
//   uc_stack           24    @ 16
//   uc_mcontext.gregs  23*u64@ 40   (gregs first inside mcontext_t)
//   ...
//
// On this platform `uc_sigmask` sits *after* `uc_mcontext`, not before,
// so the mcontext offset is a flat 40 bytes from the start of ucontext_t.
//
// gregs index constants from Linux sys/ucontext.h:
//   REG_R8=0, R9=1, R10=2, R11=3, R12=4, R13=5, R14=6, R15=7,
//   REG_RDI=8, RSI=9, RBP=10, RBX=11, RDX=12, RAX=13, RCX=14, RSP=15, RIP=16
const UCONTEXT_GREGS_OFFSET: usize = 40;
const REG_RAX: usize = 13;
const REG_RBX: usize = 11;
const REG_RIP: usize = 16;

unsafe extern "C" fn signal_handler(_sig: i32, info: *mut u8, ucontext: *mut u8) {
    if trap_signal::signal_count_inc_and_check() {
        std::process::abort();
    }

    let gregs = unsafe { ucontext.add(UCONTEXT_GREGS_OFFSET) as *mut u64 };
    let pc = unsafe { *gregs.add(REG_RIP) } as usize;

    let Some(resolution) = (unsafe { trap_signal::try_resolve_trap(pc) }) else {
        std::process::abort();
    };

    // RBX = MACHINE_CTX_REG (NativeContext pointer) in our x86_64 mapping.
    let ctx_ptr = unsafe { *gregs.add(REG_RBX) } as *mut u8;
    if resolution.trap_kind_offset > 0 {
        let fault_addr = unsafe { siginfo_fault_addr(info) };
        let trap_kind = unsafe { trap_signal::classify_trap_kind(ctx_ptr, fault_addr, resolution) };
        let trap_kind_ptr = unsafe { ctx_ptr.add(resolution.trap_kind_offset) as *mut u32 };
        unsafe { *trap_kind_ptr = trap_kind };
    }

    unsafe {
        *gregs.add(REG_RAX) = 1;
        *gregs.add(REG_RIP) = resolution.error_ret as u64;
    }
}

pub(in crate::vm::runtime) unsafe fn install_platform_handler() {
    let act = libc_sigaction {
        sa_sigaction: signal_handler,
        sa_mask: SigSet { __val: [0; 16] },
        sa_flags: SA_SIGINFO,
        sa_restorer: 0,
    };
    unsafe {
        sigaction(SIGSEGV, &act, core::ptr::null_mut());
        sigaction(SIGBUS, &act, core::ptr::null_mut());
    }
}

#[cfg(test)]
mod tests {
    use super::{libc_sigaction, SigSet};
    use core::mem::{offset_of, size_of};

    #[test]
    fn glibc_x64_sigaction_layout_matches_expected_abi() {
        assert_eq!(size_of::<SigSet>(), 128);
        assert_eq!(offset_of!(libc_sigaction, sa_sigaction), 0);
        assert_eq!(offset_of!(libc_sigaction, sa_mask), 8);
        assert_eq!(offset_of!(libc_sigaction, sa_flags), 136);
        assert_eq!(offset_of!(libc_sigaction, sa_restorer), 144);
        assert_eq!(size_of::<libc_sigaction>(), 152);
    }
}
