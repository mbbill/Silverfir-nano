//! Hazard3 RV32IMAC backend.
//!
//! Also defines the `#[panic_handler]` for RV builds: ARM gets the
//! defmt-aware `panic-probe` crate (Cortex-M only), and RV gets this
//! hand-rolled equivalent — log via defmt, then `ebreak` so probe-rs
//! catches the halt the same way it does for Cortex-M's `bkpt`.

/// `fence rw, rw` — RISC-V analogue of dsb sy. Drains pending stores so
/// the I-fetch path can see freshly emitted instructions.
#[inline(always)]
pub fn dsb() {
    riscv::asm::fence();
}

/// `fence.i` — invalidate the local hart's instruction stream cache so
/// the M33-equivalent isb step sees the new bytes. Hazard3 is in-order
/// without an I-cache, but `fence.i` is still the architecturally
/// required barrier per the Zifencei extension.
#[inline(always)]
pub fn isb() {
    riscv::asm::fence_i();
}

/// Spin for at least `cycles` CPU cycles. Backed by `riscv::asm::delay`,
/// which is a tight `addi`/`bnez` loop calibrated for ~2 cycles per
/// iteration in hardware (matching cortex-m-rt's `cortex_m::asm::delay`
/// contract).
#[inline(always)]
pub fn delay_cycles(cycles: u32) {
    riscv::asm::delay(cycles);
}

/// Wait-for-interrupt — used by halt loops.
#[inline(always)]
pub fn wfi() {
    riscv::asm::wfi();
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    if let Some(loc) = info.location() {
        defmt::error!("PANIC at {=str}:{=u32}", loc.file(), loc.line(),);
    } else {
        defmt::error!("PANIC (no location)");
    }
    loop {
        unsafe { riscv::asm::ebreak() };
    }
}
