//! Linux ARM64 signal handler using the glibc/musl ucontext_t layout.

use crate::vm::jit::runtime::trap_signal;

const SIGSEGV: i32 = 11;
const SIGBUS: i32 = 7;
const SA_SIGINFO: i32 = 4;
const SIGINFO_SI_ADDR_OFFSET: usize = 16;

// The userspace libc `struct sigaction` on Linux uses a 128-byte sigset_t
// (`_SIGSET_NWORDS * sizeof(unsigned long)` = 16 * 8 on 64-bit) and lays the
// fields out as: sa_handler/sa_sigaction, sa_mask, sa_flags, sa_restorer.
// The kernel-ABI `rt_sigaction` struct is different (8-byte sa_mask, fields
// in a different order) and must not be passed to libc's wrapper.
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

// Linux aarch64 ucontext layout:
//   ucontext.uc_mcontext.regs[0..31] = X0-X30
//   ucontext.uc_mcontext.sp
//   ucontext.uc_mcontext.pc
const UCONTEXT_MCONTEXT_REGS_OFFSET: usize = 184; // offsetof(ucontext_t, uc_mcontext.regs)

#[repr(C)]
struct McontextRegs {
    regs: [u64; 31],
    sp: u64,
    pc: u64,
    pstate: u64,
}

unsafe extern "C" fn signal_handler(_sig: i32, info: *mut u8, ucontext: *mut u8) {
    if trap_signal::signal_count_inc_and_check() {
        std::process::abort();
    }

    let mregs = unsafe { &mut *(ucontext.add(UCONTEXT_MCONTEXT_REGS_OFFSET) as *mut McontextRegs) };
    let pc = mregs.pc as usize;

    let Some(resolution) = (unsafe { trap_signal::try_resolve_trap(pc) }) else {
        std::process::abort();
    };

    let ctx_ptr = mregs.regs[19] as *mut u8;
    if resolution.trap_kind_offset > 0 {
        let fault_addr = unsafe { siginfo_fault_addr(info) };
        let trap_kind = unsafe { trap_signal::classify_trap_kind(ctx_ptr, fault_addr, resolution) };
        let trap_kind_ptr = unsafe { ctx_ptr.add(resolution.trap_kind_offset) as *mut u32 };
        unsafe { *trap_kind_ptr = trap_kind };
    }

    mregs.regs[0] = 1;
    mregs.pc = resolution.error_ret as u64;
}

pub(in crate::vm::jit::runtime) unsafe fn install_platform_handler() {
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
    fn glibc_arm64_sigaction_layout_matches_expected_abi() {
        assert_eq!(size_of::<SigSet>(), 128);
        assert_eq!(offset_of!(libc_sigaction, sa_sigaction), 0);
        assert_eq!(offset_of!(libc_sigaction, sa_mask), 8);
        assert_eq!(offset_of!(libc_sigaction, sa_flags), 136);
        assert_eq!(offset_of!(libc_sigaction, sa_restorer), 144);
        assert_eq!(size_of::<libc_sigaction>(), 152);
    }
}
