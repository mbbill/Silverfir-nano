//! Signal-based trap handler for guard-page memory.
//!
//! Installs a SIGSEGV/SIGBUS handler that catches faults in JIT code regions
//! and converts them to wasm traps by redirecting execution to the function's
//! error-return path.
//!
//! The handler is async-signal-safe: it does not allocate. It sets a `trap_kind`
//! flag in `NativeContext` (reachable via X19) and rewrites the signal frame's
//! PC to the function's `return_error_label`. The Rust caller (`eval()`) reads
//! `trap_kind` after JIT returns and creates the `WasmError`.
//!
//! This module is gated on `#[cfg(feature = "guard-pages")]`.

use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use alloc::vec::Vec;

/// Debug counter: aborts after too many consecutive signals (infinite loop detection).
static SIGNAL_COUNT: AtomicUsize = AtomicUsize::new(0);

/// A registered JIT code range with its error-return address.
#[derive(Clone, Copy)]
struct JitCodeRange {
    code_start: usize,
    code_end: usize,
    return_error: usize,
}

/// Global trap table. Sorted by `code_start` for binary search.
///
/// Access is synchronized by a simple spinlock. The signal handler acquires
/// a read-only view; registration happens during compilation (rare).
static mut TRAP_TABLE: Vec<JitCodeRange> = Vec::new();
static TRAP_TABLE_LOCK: AtomicBool = AtomicBool::new(false);
static HANDLER_INSTALLED: AtomicBool = AtomicBool::new(false);

fn lock_trap_table() {
    while TRAP_TABLE_LOCK
        .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        core::hint::spin_loop();
    }
}

fn unlock_trap_table() {
    TRAP_TABLE_LOCK.store(false, Ordering::Release);
}

/// Register JIT code ranges in the global trap table.
pub fn register_jit_ranges(ranges: &[(usize, usize, usize)]) {
    lock_trap_table();
    unsafe {
        let table = &raw mut TRAP_TABLE;
        for &(start, end, error_ret) in ranges {
            (*table).push(JitCodeRange {
                code_start: start,
                code_end: end,
                return_error: error_ret,
            });
        }
        (*table).sort_unstable_by_key(|r| r.code_start);
    }
    unlock_trap_table();
}

/// Look up the error-return address for a faulting PC.
/// Must be called with the trap table lock held (or from the signal handler
/// where concurrent modification is not expected on the same thread).
unsafe fn lookup_return_error(pc: usize) -> Option<usize> {
    let table = unsafe { &*(&raw const TRAP_TABLE) };
    let idx = table.partition_point(|r| r.code_start <= pc);
    if idx == 0 {
        return None;
    }
    let entry = &table[idx - 1];
    if pc < entry.code_end {
        Some(entry.return_error)
    } else {
        None
    }
}

/// Install the signal handler (idempotent).
pub fn install_signal_handler() {
    if HANDLER_INSTALLED.load(Ordering::Relaxed) {
        return;
    }
    if HANDLER_INSTALLED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
        .is_err()
    {
        return;
    }
    unsafe {
        install_platform_handler();
    }
}

/// Reset debug state tracked by the signal handler.
///
/// The infinite-loop detector is only meaningful within a single native entry:
/// one wasm invocation should either complete or trap after the first fault.
/// Resetting here prevents expected trapping test cases from accumulating
/// counts across many independent invocations.
pub fn reset_debug_state() {
    SIGNAL_COUNT.store(0, Ordering::Relaxed);
}

/// Drop all registered JIT ranges.
///
/// Callers must only use this when no compiled native frames from the old
/// ranges can still fault, otherwise a later trap would be unable to resolve
/// back to its owning function.
pub fn clear_registered_jit_ranges() {
    lock_trap_table();
    unsafe {
        let table = &raw mut TRAP_TABLE;
        (*table).clear();
    }
    unlock_trap_table();
}

// ─── macOS ARM64 ────────────────────────────────────────────────────────────

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod platform {
    //! macOS ARM64 signal handler using Darwin's ucontext_t layout.

    use super::*;

    // Signal constants
    const SIGSEGV: i32 = 11;
    const SIGBUS: i32 = 10;
    const SA_SIGINFO: i32 = 0x0040;

    // libc types (minimal, layout-compatible)
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
        fp: u64,       // X29
        lr: u64,       // X30
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
        let mctx_ptr = *(ucontext.add(UCONTEXT_MCONTEXT_OFFSET) as *const *mut u8);
        mctx_ptr.add(MCONTEXT_SS_OFFSET) as *mut Arm64ThreadState
    }

    unsafe extern "C" fn signal_handler(_sig: i32, _info: *mut u8, ucontext: *mut u8) {
        let count = SIGNAL_COUNT.fetch_add(1, Ordering::Relaxed);
        if count > 100 {
            std::process::abort();
        }

        let ts = unsafe { thread_state(ucontext) };
        let pc = unsafe { (*ts).pc as usize };

        let error_ret = unsafe { lookup_return_error(pc) };
        let Some(error_ret) = error_ret else {
            // Not in JIT code — abort (we can't chain easily without libc).
            std::process::abort();
        };

        // Read X19 (NativeContext pointer) from the faulting thread state.
        let ctx_ptr = unsafe { (*ts).x[19] } as *mut u8;

        // Set ctx.trap_kind = 1 (MemoryOutOfBounds).
        let trap_kind_offset = TRAP_KIND_OFFSET.load(Ordering::Relaxed);
        if trap_kind_offset > 0 {
            let trap_kind_ptr = ctx_ptr.add(trap_kind_offset) as *mut u32;
            unsafe { *trap_kind_ptr = 1 };
        }

        // Set X0 = 1 (error status for eval())
        unsafe { (*ts).x[0] = 1 };

        // Redirect PC to the function's return_error_label
        unsafe { (*ts).pc = error_ret as u64 };
    }

    pub(super) unsafe fn install_platform_handler() {
        let act = sigaction {
            sa_sigaction: signal_handler,
            sa_mask: 0,
            sa_flags: SA_SIGINFO,
        };
        sigaction(SIGSEGV, &act, core::ptr::null_mut());
        sigaction(SIGBUS, &act, core::ptr::null_mut());
    }
}

// ─── Linux ARM64 ────────────────────────────────────────────────────────────

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
mod platform {
    use super::*;

    const SIGSEGV: i32 = 11;
    const SIGBUS: i32 = 7;
    const SA_SIGINFO: i32 = 4;

    #[repr(C)]
    struct kernel_sigaction {
        sa_sigaction: unsafe extern "C" fn(i32, *mut u8, *mut u8),
        sa_flags: u64,
        sa_restorer: usize,
        sa_mask: u64,
    }

    unsafe extern "C" {
        fn sigaction(sig: i32, act: *const kernel_sigaction, oldact: *mut kernel_sigaction) -> i32;
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

    unsafe extern "C" fn signal_handler(_sig: i32, _info: *mut u8, ucontext: *mut u8) {
        let mregs = unsafe {
            &mut *(ucontext.add(UCONTEXT_MCONTEXT_REGS_OFFSET) as *mut McontextRegs)
        };
        let pc = mregs.pc as usize;

        let error_ret = unsafe { lookup_return_error(pc) };
        let Some(error_ret) = error_ret else {
            std::process::abort();
        };

        let ctx_ptr = mregs.regs[19] as *mut u8;
        let trap_kind_offset = TRAP_KIND_OFFSET.load(Ordering::Relaxed);
        if trap_kind_offset > 0 {
            let trap_kind_ptr = ctx_ptr.add(trap_kind_offset) as *mut u32;
            unsafe { *trap_kind_ptr = 1 };
        }

        mregs.regs[0] = 1;
        mregs.pc = error_ret as u64;
    }

    pub(super) unsafe fn install_platform_handler() {
        let act = kernel_sigaction {
            sa_sigaction: signal_handler,
            sa_flags: SA_SIGINFO as u64,
            sa_restorer: 0,
            sa_mask: 0,
        };
        sigaction(SIGSEGV, &act, core::ptr::null_mut());
        sigaction(SIGBUS, &act, core::ptr::null_mut());
    }
}

// ─── macOS x86_64 ───────────────────────────────────────────────────────────

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
mod platform {
    //! macOS x86_64 signal handler using Darwin's ucontext_t layout.

    use super::*;

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
        let mctx_ptr = *(ucontext.add(UCONTEXT_MCONTEXT_OFFSET) as *const *mut u8);
        mctx_ptr.add(MCONTEXT_SS_OFFSET) as *mut X86ThreadState64
    }

    unsafe extern "C" fn signal_handler(_sig: i32, _info: *mut u8, ucontext: *mut u8) {
        let ts = unsafe { thread_state(ucontext) };
        let pc = unsafe { (*ts).rip as usize };

        let error_ret = unsafe { lookup_return_error(pc) };
        let Some(error_ret) = error_ret else {
            std::process::abort();
        };

        // RBX = MACHINE_CTX_REG (NativeContext pointer) in our x86_64 mapping.
        let ctx_ptr = unsafe { (*ts).rbx } as *mut u8;

        let trap_kind_offset = TRAP_KIND_OFFSET.load(Ordering::Relaxed);
        if trap_kind_offset > 0 {
            let trap_kind_ptr = ctx_ptr.add(trap_kind_offset) as *mut u32;
            unsafe { *trap_kind_ptr = 1 };
        }

        // Set RAX = 1 (error status for eval())
        unsafe { (*ts).rax = 1 };

        // Redirect RIP to the function's return_error_label
        unsafe { (*ts).rip = error_ret as u64 };
    }

    pub(super) unsafe fn install_platform_handler() {
        let act = sigaction {
            sa_sigaction: signal_handler,
            sa_mask: 0,
            sa_flags: SA_SIGINFO,
        };
        sigaction(SIGSEGV, &act, core::ptr::null_mut());
        sigaction(SIGBUS, &act, core::ptr::null_mut());
    }
}

// ─── Linux ARM32 ────────────────────────────────────────────────────────────

#[cfg(all(target_os = "linux", target_arch = "arm"))]
mod platform {
    use super::*;

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
        arm_fp: u32,  // r11
        arm_ip: u32,  // r12
        arm_sp: u32,  // r13
        arm_lr: u32,  // r14
        arm_pc: u32,  // r15
        arm_cpsr: u32,
        fault_address: u32,
    }

    unsafe extern "C" fn signal_handler(_sig: i32, _info: *mut u8, ucontext: *mut u8) {
        let sc = unsafe {
            &mut *(ucontext.add(UCONTEXT_MCONTEXT_OFFSET) as *mut ArmSigcontext)
        };
        let pc = sc.arm_pc as usize;

        let error_ret = unsafe { lookup_return_error(pc) };
        let Some(error_ret) = error_ret else {
            std::process::abort();
        };

        // R9 = MACHINE_CTX_REG (NativeContext pointer) on armv7a.
        let ctx_ptr = sc.arm_r9 as *mut u8;
        let trap_kind_offset = TRAP_KIND_OFFSET.load(Ordering::Relaxed);
        if trap_kind_offset > 0 {
            let trap_kind_ptr = ctx_ptr.add(trap_kind_offset) as *mut u32;
            unsafe { *trap_kind_ptr = 1 };
        }

        sc.arm_r0 = 1; // error status
        sc.arm_pc = error_ret as u32;
    }

    pub(super) unsafe fn install_platform_handler() {
        let act = kernel_sigaction {
            sa_sigaction: signal_handler,
            sa_flags: SA_SIGINFO as u32,
            sa_restorer: 0,
            sa_mask: 0,
        };
        sigaction(SIGSEGV, &act, core::ptr::null_mut());
        sigaction(SIGBUS, &act, core::ptr::null_mut());
    }
}

use platform::install_platform_handler;

/// Offset of the `trap_kind` field within `NativeContext`, set once at init.
static TRAP_KIND_OFFSET: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

/// Set the byte offset of `NativeContext::trap_kind` so the signal handler
/// can write it without knowing the struct layout at compile time.
pub fn set_trap_kind_offset(offset: usize) {
    TRAP_KIND_OFFSET.store(offset, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_registered_jit_ranges_drops_stale_entries() {
        clear_registered_jit_ranges();
        register_jit_ranges(&[(0x1000, 0x1100, 0x2000)]);

        unsafe {
            assert_eq!(lookup_return_error(0x1080), Some(0x2000));
        }

        clear_registered_jit_ranges();

        unsafe {
            assert_eq!(lookup_return_error(0x1080), None);
        }
    }

    #[test]
    fn reset_debug_state_clears_signal_counter() {
        SIGNAL_COUNT.store(17, Ordering::Relaxed);
        reset_debug_state();
        assert_eq!(SIGNAL_COUNT.load(Ordering::Relaxed), 0);
    }
}
