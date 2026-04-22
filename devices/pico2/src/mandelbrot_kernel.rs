//! Shared integer Mandelbrot kernel.
//!
//! Same source compiled twice: once natively as part of this crate
//! (used by `mandelbrot_native`), and once to `wasm32-unknown-unknown`
//! via a sub-crate (used by `mandelbrot_wasm` through sf-nano-core's
//! JIT). Whichever renders faster wins.
//!
//! Q17.14 fixed-point arithmetic — no floats anywhere, so the same
//! code runs unchanged on the MCU-target JIT which emits no FP code.
//! Products stay in i32: with |val| < 2·2^14 = 32768 the full
//! product is below 2^30, well within i32. Trading Q16.16's 1/65536
//! precision for 1/16384 is invisible at pixel scale (pixel step is
//! ~0.019, precision is ~6e-5) and avoids every i64 op in the hot
//! loop — so the JIT benchmark measures the i32 path end-to-end.

/// Physical panel dimensions. The 40-KiB framebuffer is 2 bytes per
/// pixel (RGB565 big-endian, matching ST7735's wire format).
pub const WIDTH: usize = 160;
pub const HEIGHT: usize = 128;
pub const FB_BYTES: usize = WIDTH * HEIGHT * 2;

/// Escape iterations. Interior pixels hit this cap; colorful edge
/// pixels bail much earlier (single digits). Averages out to maybe
/// 20–30 iterations/pixel across the whole view.
const MAX_ITER: u32 = 64;

/// Fractional bits. K=14 keeps `a*b` in i32 as long as |val| < 2^15.
/// For the Mandelbrot viewport (|z| bailout at 2) that's always true.
const Q_BITS: u32 = 14;

/// Q17.14 fixed-point multiply. i32-only: the full product fits in i32,
/// so no i64 extend/mul/shr pair expansion. The shift is i32.shr_s —
/// one native ASR on every backend.
#[inline(always)]
fn qmul(a: i32, b: i32) -> i32 {
    a.wrapping_mul(b) >> Q_BITS
}

/// 1.0 in Q17.14.
const Q_ONE: i32 = 1 << Q_BITS;

/// 4.0 in Q17.14 — the Mandelbrot escape threshold for `|z|²`.
const Q_FOUR: i32 = 4 << Q_BITS;

/// Viewport: centered at (-0.5, 0), covering the full classic view.
/// Width = 3.0 (so x in [-2.0, 1.0]), height = 2.4 (y in [-1.2, 1.2]).
/// Pixel step is identical on both axes (square pixels).
const VIEW_CENTER_X: i32 = -(Q_ONE >> 1); // -0.5
const VIEW_CENTER_Y: i32 = 0;
const VIEW_WIDTH: i32 = 3 * Q_ONE; // 3.0
const VIEW_HEIGHT: i32 = (3 * Q_ONE * 4) / 5; // 2.4, keeps pixels square

/// Draw the full frame into `bytes` as RGB565 big-endian. `frame` is
/// used only to rotate the palette — the Mandelbrot iteration at each
/// pixel is identical every frame, so the JIT-compiled hot loop runs
/// the same work per frame. Makes the benchmark a steady-state
/// throughput measurement rather than a compile-time one.
pub fn render(bytes: &mut [u8], frame: u32) {
    debug_assert_eq!(bytes.len(), FB_BYTES);

    // Per-pixel step and the top-left corner of the viewport.
    let dx = VIEW_WIDTH / WIDTH as i32;
    let dy = VIEW_HEIGHT / HEIGHT as i32;
    let start_x = VIEW_CENTER_X - VIEW_WIDTH / 2;
    let start_y = VIEW_CENTER_Y - VIEW_HEIGHT / 2;

    let mut cy = start_y;
    let mut idx = 0;
    for _ in 0..HEIGHT {
        let mut cx = start_x;
        for _ in 0..WIDTH {
            let iter = mandelbrot_iter(cx, cy);
            let rgb = palette(iter, frame);
            bytes[idx] = (rgb >> 8) as u8;
            bytes[idx + 1] = rgb as u8;
            idx += 2;
            cx = cx.wrapping_add(dx);
        }
        cy = cy.wrapping_add(dy);
    }
}

/// Mandelbrot escape count for one complex point `(cx, cy)` in Q16.16.
/// Returns `MAX_ITER` if the point stays bounded.
#[inline(always)]
fn mandelbrot_iter(cx: i32, cy: i32) -> u32 {
    let mut zx: i32 = 0;
    let mut zy: i32 = 0;
    let mut iter: u32 = 0;
    while iter < MAX_ITER {
        let zx2 = qmul(zx, zx);
        let zy2 = qmul(zy, zy);
        // wrapping_add is safe here: at Q17.14 with |z| < 2, zx² and zy²
        // each fit in ~17 bits, so their sum is at most ~2^18 — wraparound
        // is unreachable. Keeping this as wrapping_add spares the JIT an
        // overflow-check sequence per iteration.
        if zx2.wrapping_add(zy2) > Q_FOUR {
            break;
        }
        let zxy = qmul(zx, zy);
        zx = zx2.wrapping_sub(zy2).wrapping_add(cx);
        zy = zxy.wrapping_add(zxy).wrapping_add(cy);
        iter += 1;
    }
    iter
}

/// 32-entry rotating rainbow palette. Iteration count determines the
/// base hue, `frame` rotates it — cheap per-pixel color cycling that
/// gives the demo its psychedelic animation.
#[inline(always)]
fn palette(iter: u32, frame: u32) -> u16 {
    if iter >= MAX_ITER {
        return 0x0000; // black — interior of the set
    }
    let hue = (iter.wrapping_add(frame)) & 0x1f; // 0..32
    // 32-step rainbow table, RGB565. Six color segments (red→yellow→
    // green→cyan→blue→magenta→red), a bit over 5 steps each.
    const TABLE: [u16; 32] = [
        0xF800, 0xF880, 0xF900, 0xF980, 0xFA00, 0xFA80, 0xFB00, 0xFB80, 0xFC00, 0xFCC0, 0xFD80,
        0xFE40, 0xFF00, 0xF720, 0xEF60, 0xE780, 0xDFC0, 0xD7E0, 0x87E0, 0x37E0, 0x07E4, 0x07EA,
        0x07EF, 0x07F5, 0x07FB, 0x07FF, 0x04FF, 0x023F, 0x001F, 0x401F, 0x801F, 0xC01F,
    ];
    TABLE[hue as usize]
}
