//! Mandelbrot kernel compiled to Wasm.
//!
//! Same compute as the native `mandelbrot_native` binary — the Rust
//! source is pulled in verbatim from the parent firmware crate via
//! `#[path]`, so there is a single source of truth for the algorithm.
//! Only the "plumbing" differs: here we write pixels into a static
//! framebuffer in the module's own linear memory, then call the
//! imported `env.push_frame` host function to hand that region to the
//! firmware's DMA pipeline.
//!
//! Exports:
//! - `run(frame: i32)` — compute + present one frame. The host drives
//!   the loop by invoking `run` once per frame.
//! - `framebuffer_offset() -> i32` — debug helper for inspecting where
//!   in linear memory the framebuffer lands.
//!
//! Imports:
//! - `env.push_frame(offset: i32, len: i32)` — firmware-side DMA push
//!   into ST7735 RAM. Blocks until the frame is on the panel. The host
//!   side patches an fps overlay into the byte range before the DMA,
//!   so this kernel stays focused on the mandelbrot math alone.

#![no_std]

#[path = "../../src/mandelbrot_kernel.rs"]
mod kernel;

use kernel::FB_BYTES;

#[unsafe(no_mangle)]
static mut FRAMEBUFFER: [u8; FB_BYTES] = [0u8; FB_BYTES];

unsafe extern "C" {
    // Firmware-provided: DMA `len` bytes starting at `offset` (within
    // this module's linear memory) to the ST7735 via SPI1. Blocks.
    fn push_frame(offset: i32, len: i32);
}

/// Compute one Mandelbrot frame and hand it to the host to display.
/// The host drives the loop — it invokes this once per frame so the
/// invoke-side DWT bracket doubles as the fps clock. The fps overlay
/// is patched in on the host side inside `push_frame`, so the Wasm
/// kernel stays focused on compute.
#[unsafe(no_mangle)]
pub extern "C" fn run(frame: i32) {
    // SAFETY: single-threaded wasm, we're the only writer of
    // FRAMEBUFFER, and we don't retain aliases after kernel::render
    // returns.
    let bytes: &mut [u8] = unsafe { &mut *core::ptr::addr_of_mut!(FRAMEBUFFER) };
    kernel::render(bytes, frame as u32);

    let offset = core::ptr::addr_of!(FRAMEBUFFER) as i32;
    let len = FB_BYTES as i32;
    unsafe { push_frame(offset, len) };
}

/// Debug helper: let the host verify where in linear memory the
/// framebuffer sits. Not used on the critical path — `run` already
/// passes the offset to `push_frame` directly.
#[unsafe(no_mangle)]
pub extern "C" fn framebuffer_offset() -> i32 {
    core::ptr::addr_of!(FRAMEBUFFER) as i32
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // In Wasm a panic turns into a trap. With panic = "abort" in the
    // release profile this handler is unreachable in practice, but
    // having it keeps `#![no_std]` happy.
    loop {}
}
