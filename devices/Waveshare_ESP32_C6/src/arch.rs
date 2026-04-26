//! Small RISC-V barriers used by the JIT executable-memory shim.

#[inline(always)]
pub fn dsb() {
    unsafe { core::arch::asm!("fence rw, rw", options(nostack)) };
}

#[inline(always)]
pub fn isb() {
    unsafe { core::arch::asm!("fence.i", options(nostack)) };
}
