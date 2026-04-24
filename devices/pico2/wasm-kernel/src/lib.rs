//! Wasm kernel driver: framebuffer, entry points, panic handler.
//!
//! All rendering lives in dedicated demo modules, each exposing a
//! uniform `pub fn render(bytes, frame)` + `pub const FB_BYTES`
//! interface. To switch demos, comment/uncomment the two `use … as
//! demo;` lines below — one line edit, no Cargo flags.

#![no_std]

#[path = "../../src/mandelbrot_kernel.rs"]
mod mandelbrot;

#[allow(dead_code)] // inactive when mandelbrot is the selected demo
mod cube;

// ── Active demo ─────────────────────────────────────────────────────
// Swap which line is commented to switch between demos.
use mandelbrot as demo;
// use cube as demo;

pub const FB_BYTES: usize = demo::FB_BYTES;

#[unsafe(no_mangle)]
static mut FRAMEBUFFER: [u8; FB_BYTES] = [0u8; FB_BYTES];

unsafe extern "C" {
    fn push_frame(offset: i32, len: i32);
}

#[unsafe(no_mangle)]
pub extern "C" fn run(frame: i32) {
    let bytes: &mut [u8] = unsafe { &mut *core::ptr::addr_of_mut!(FRAMEBUFFER) };
    demo::render(bytes, frame as u32);
    let offset = core::ptr::addr_of!(FRAMEBUFFER) as i32;
    let len = FB_BYTES as i32;
    unsafe { push_frame(offset, len) };
}

#[unsafe(no_mangle)]
pub extern "C" fn framebuffer_offset() -> i32 {
    core::ptr::addr_of!(FRAMEBUFFER) as i32
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
