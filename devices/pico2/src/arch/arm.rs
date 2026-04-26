//! Cortex-M33 backend.

/// `dsb sy` — drain in-flight stores so the I-fetch path sees written
/// instructions.
#[inline(always)]
pub fn dsb() {
    cortex_m::asm::dsb();
}

/// `isb sy` — flush the M33's prefetch buffer so stale instructions
/// are discarded.
#[inline(always)]
pub fn isb() {
    cortex_m::asm::isb();
}

/// Spin for at least `cycles` CPU cycles. Caller calibrates against
/// SYSCLK.
#[inline(always)]
pub fn delay_cycles(cycles: u32) {
    cortex_m::asm::delay(cycles);
}

/// Wait-for-interrupt — used by halt loops.
#[inline(always)]
pub fn wfi() {
    cortex_m::asm::wfi();
}
