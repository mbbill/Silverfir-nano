//! Wasm Mandelbrot guest: render into linear memory, host presents it.

#![no_std]

#[cfg(not(feature = "demo-mandelbrot"))]
compile_error!("select one demo feature");

#[path = "../../src/kernels/mandelbrot.rs"]
mod mandelbrot;

pub const FB_BYTES: usize = mandelbrot::FB_BYTES;

#[unsafe(no_mangle)]
static mut FRAMEBUFFER: [u8; FB_BYTES] = [0u8; FB_BYTES];

#[unsafe(no_mangle)]
pub extern "C" fn run(frame: i32) {
    let bytes: &mut [u8] = unsafe { &mut *core::ptr::addr_of_mut!(FRAMEBUFFER) };
    mandelbrot::render(bytes, frame as u32);
}

#[unsafe(no_mangle)]
pub extern "C" fn framebuffer_offset() -> i32 {
    core::ptr::addr_of!(FRAMEBUFFER) as i32
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
