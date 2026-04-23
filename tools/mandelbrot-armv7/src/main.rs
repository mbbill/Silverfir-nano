//! Native armv7 Mandelbrot benchmark. Two kernel variants compiled
//! side-by-side:
//!   - `i64` (default): Q16.16 fixed-point, 32×32→64 multiplies. The
//!     kernel Phase 4 optimizes against — this is the "what good i64
//!     Thumb-2 looks like" reference.
//!   - `i32`: Q17.14 fixed-point, pure i32 multiplies. The baseline
//!     that sidesteps the JIT's slow i64 path (what the on-device demo
//!     ships today).
//!
//! Both `bench_render_i64` and `bench_render_i32` are `#[inline(never)]`
//! so each gets its own disassembly symbol.

mod kernel_i32;
mod kernel_i64;

use std::env;
use std::hint::black_box;
use std::time::Instant;

#[inline(never)]
#[no_mangle]
pub fn bench_render_i64(bytes: &mut [u8], frame: u32) {
    kernel_i64::render(bytes, frame);
}

#[inline(never)]
#[no_mangle]
pub fn bench_render_i32(bytes: &mut [u8], frame: u32) {
    kernel_i32::render(bytes, frame);
}

fn fnv1a(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for b in bytes {
        h ^= *b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}

fn main() {
    let mut kernel = "i64";
    let mut frames: u32 = 60;
    for arg in env::args().skip(1) {
        if let Some(v) = arg.strip_prefix("--kernel=") {
            kernel = match v {
                "i32" => "i32",
                "i64" => "i64",
                other => panic!("--kernel must be 'i32' or 'i64', got: {other}"),
            };
        } else if let Ok(n) = arg.parse::<u32>() {
            frames = n;
        } else {
            panic!("unrecognized argument: {arg}");
        }
    }

    assert_eq!(kernel_i32::FB_BYTES, kernel_i64::FB_BYTES);
    let mut fb = vec![0u8; kernel_i64::FB_BYTES];
    let bench: fn(&mut [u8], u32) = match kernel {
        "i32" => bench_render_i32,
        "i64" => bench_render_i64,
        _ => unreachable!(),
    };

    // Warmup — caches, ASLR resolution, page faults outside the timed region.
    bench(&mut fb, 0);

    let start = Instant::now();
    for f in 0..frames {
        bench(&mut fb, black_box(f));
        black_box(&fb);
    }
    let elapsed = start.elapsed();

    let ns_per_frame = elapsed.as_nanos() / (frames as u128);
    let fps = 1.0e9 / (ns_per_frame as f64);
    let checksum = fnv1a(&fb);

    println!("kernel:    {}", kernel);
    println!("frames:    {}", frames);
    println!("total:     {:.3} s", elapsed.as_secs_f64());
    println!("per frame: {} ns  ({:.3} ms)", ns_per_frame, (ns_per_frame as f64) / 1e6);
    println!("fps:       {:.2}", fps);
    println!("checksum:  0x{:08x}  (final frame, frame_idx={})", checksum, frames - 1);
}
