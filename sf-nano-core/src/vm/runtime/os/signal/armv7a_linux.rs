//! Linux ARM32 (armv7a) signal handler using the kernel's sigcontext layout.

use crate::vm::runtime::trap_signal;

const SIGSEGV: i32 = 11;
const SIGBUS: i32 = 7;
const SA_SIGINFO: i32 = 4;

#[repr(C)]
struct kernel_sigaction {
    sa_sigaction: unsafe extern "C" fn(i32, *mut u8, *mut u8),
    sa_flags: u32,
    sa_restorer: usize,
    sa_mask: u64,
}

unsafe extern "C" {
    fn sigaction(sig: i32, act: *const kernel_sigaction, oldact: *mut kernel_sigaction) -> i32;
}

// Linux arm32 ucontext layout:
//   ucontext.uc_mcontext is a struct sigcontext at a fixed offset.
//   sigcontext fields: trap_no, error_code, oldmask, arm_r0..arm_r10,
//   arm_fp, arm_ip, arm_sp, arm_lr, arm_pc, arm_cpsr, fault_address
const UCONTEXT_MCONTEXT_OFFSET: usize = 32; // offsetof(ucontext_t, uc_mcontext) on arm32

#[repr(C)]
struct ArmSigcontext {
    trap_no: u32,
    error_code: u32,
    oldmask: u32,
    arm_r0: u32,
    arm_r1: u32,
    arm_r2: u32,
    arm_r3: u32,
    arm_r4: u32,
    arm_r5: u32,
    arm_r6: u32,
    arm_r7: u32,
    arm_r8: u32,
    arm_r9: u32,
    arm_r10: u32,
    arm_fp: u32, // r11
    arm_ip: u32, // r12
    arm_sp: u32, // r13
    arm_lr: u32, // r14
    arm_pc: u32, // r15
    arm_cpsr: u32,
    fault_address: u32,
}

unsafe extern "C" fn signal_handler(_sig: i32, _info: *mut u8, ucontext: *mut u8) {
    if trap_signal::signal_count_inc_and_check() {
        std::process::abort();
    }

    let sc = unsafe { &mut *(ucontext.add(UCONTEXT_MCONTEXT_OFFSET) as *mut ArmSigcontext) };
    let pc = sc.arm_pc as usize;

    let Some((error_ret, trap_kind_offset)) = (unsafe { trap_signal::try_resolve_trap(pc) }) else {
        std::process::abort();
    };

    // R9 = MACHINE_CTX_REG (NativeContext pointer) on armv7a.
    let ctx_ptr = sc.arm_r9 as *mut u8;
    if trap_kind_offset > 0 {
        let trap_kind_ptr = unsafe { ctx_ptr.add(trap_kind_offset) as *mut u32 };
        unsafe { *trap_kind_ptr = 1 };
    }

    sc.arm_r0 = 1; // error status
    sc.arm_pc = error_ret as u32;
}

pub(in crate::vm::runtime) unsafe fn install_platform_handler() {
    let act = kernel_sigaction {
        sa_sigaction: signal_handler,
        sa_flags: SA_SIGINFO as u32,
        sa_restorer: 0,
        sa_mask: 0,
    };
    unsafe {
        sigaction(SIGSEGV, &act, core::ptr::null_mut());
        sigaction(SIGBUS, &act, core::ptr::null_mut());
    }
}
