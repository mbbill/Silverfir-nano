//! Q16.16 Mandelbrot kernel, compiled to wasm32. Phase 3 feeds this
//! module into sf-nano-core's JIT under qemu-arm so we can dump the
//! emitted Thumb-2 and diff against the native reference in
//! `bench_render_i64.thumb2.asm`.
//!
//! Exports:
//! - `run(frame: i32) -> i32` — renders one frame, returns the
//!   framebuffer offset in linear memory. The host may (or may not)
//!   actually invoke it — all we need for the dump is for the JIT to
//!   compile this function.
//!
//! No imports — a bare wasm module so it can be JIT-compiled without a
//! host-side import shim.

#![no_std]

#[path = "../../src/kernel_i64.rs"]
mod kernel;

use kernel::FB_BYTES;

#[unsafe(no_mangle)]
static mut FRAMEBUFFER: [u8; FB_BYTES] = [0u8; FB_BYTES];

#[unsafe(no_mangle)]
pub extern "C" fn run(frame: i32) -> i32 {
    let bytes: &mut [u8] = unsafe { &mut *core::ptr::addr_of_mut!(FRAMEBUFFER) };
    kernel::render(bytes, frame as u32);
    core::ptr::addr_of!(FRAMEBUFFER) as i32
}

/// Entry point for sf-nano-cli (WASI-shaped host). Renders `BENCH_FRAMES`
/// frames so the kernel work dominates startup overhead when timing
/// the JIT — see README, "Runtime comparison" section. Without this
/// loop, `time qemu-arm-static sf-nano-cli …` measures mostly wasm
/// parse + JIT compile, not the actual kernel.
///
/// `black_box` on both sides of `run` stops LLVM from collapsing 30
/// identical frame renders into a single call.
const BENCH_FRAMES: i32 = 300;

#[unsafe(no_mangle)]
pub extern "C" fn _start() {
    let mut f = 0i32;
    while f < BENCH_FRAMES {
        let fb_off = run(core::hint::black_box(f));
        core::hint::black_box(fb_off);
        f += 1;
    }
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
