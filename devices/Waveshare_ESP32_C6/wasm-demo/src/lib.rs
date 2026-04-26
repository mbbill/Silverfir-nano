//! Wasm demo guest: render into linear memory, host presents it.

#![no_std]

#[cfg(all(feature = "demo-mandelbrot", feature = "demo-cube"))]
compile_error!("select exactly one demo feature");
#[cfg(not(any(feature = "demo-mandelbrot", feature = "demo-cube")))]
compile_error!("select one demo feature");

#[cfg(feature = "demo-cube")]
#[path = "../../src/kernels/cube.rs"]
mod cube;
#[cfg(feature = "demo-mandelbrot")]
#[path = "../../src/kernels/mandelbrot.rs"]
mod mandelbrot;

#[cfg(feature = "demo-cube")]
use cube as demo;
#[cfg(feature = "demo-mandelbrot")]
use mandelbrot as demo;

pub const FB_BYTES: usize = demo::FB_BYTES;

#[unsafe(no_mangle)]
static mut FRAMEBUFFER: [u8; FB_BYTES] = [0u8; FB_BYTES];

#[unsafe(no_mangle)]
pub extern "C" fn run(frame: i32) {
    let bytes: &mut [u8] = unsafe { &mut *core::ptr::addr_of_mut!(FRAMEBUFFER) };
    demo::render(bytes, frame as u32);
}

#[unsafe(no_mangle)]
pub extern "C" fn framebuffer_offset() -> i32 {
    core::ptr::addr_of!(FRAMEBUFFER) as i32
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
