//! Shared integer Mandelbrot kernel — Q16.16 i64 variant.
//!
//! Same source compiled two ways: directly into `native_demo`,
//! and into the `wasm-demo/` sub-crate that `demo_host` executes
//! through sf-nano-core's JIT. Whichever renders faster wins.
//!
//! Q16.16 fixed-point: `qmul` routes through `i64` so the 32×32 product
//! doesn't overflow. This is the "harder" kernel — the `(a as i64)
//! .wrapping_mul(b as i64) >> 16` pattern exercises sf-nano-core's
//! i64 lowering on the hot path, which is exactly the performance
//! surface we want to measure and optimize. The Q17.14 i32 variant
//! used previously sidestepped i64 entirely by keeping everything in
//! i32; the JIT looked good on that kernel but we were avoiding the
//! problem, not solving it.

/// Physical panel dimensions. The 40-KiB framebuffer is 2 bytes per
/// pixel (RGB565 big-endian, matching ST7735's wire format).
pub const WIDTH: usize = 160;
pub const HEIGHT: usize = 128;
pub const FB_BYTES: usize = WIDTH * HEIGHT * 2;

/// Escape iterations. Most interior pixels are caught by the
/// cardioid/period-2 bulb shortcut before ever entering the loop, so
/// MAX_ITER only bounds the work for the remaining thin-filament
/// interior and the very-slow-escaping boundary. 32 is enough to keep
/// boundary banding visually clean at this viewport; halving from 64
/// trims the worst-case cost on both native and JIT.
const MAX_ITER: u32 = 32;

/// Fractional bits. K=16 gives Q16.16 precision. The i32×i32 product
/// needs i64 intermediate — this is the point of this kernel.
const Q_BITS: u32 = 16;

/// Q16.16 fixed-point multiply. Routes through i64 for the product.
/// On armv7 this lowers to SMULL + LSR/ORR (~5 instructions) in LLVM;
/// sf-nano-core's JIT currently emits a wider UMULL + MLA + MLA sequence
/// for the same wasm `i64.mul` — that's the gap Phase 4 optimizations
/// are trying to close.
#[inline(always)]
fn qmul(a: i32, b: i32) -> i32 {
    ((a as i64).wrapping_mul(b as i64) >> Q_BITS) as i32
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

    // Render the bottom half + the center row (cy = 0 lands on row
    // HEIGHT/2). That's HEIGHT/2+1 rows. The rows above center are
    // byte-for-byte mirrors by vertical symmetry: escape(cx, +cy) ==
    // escape(cx, -cy) for z₀ = 0, so pixel color depends only on |cy|.
    // Halves the compute cost of the kernel at the price of a ~40 KiB
    // copy.
    const HALF: usize = HEIGHT / 2;
    const ROW_BYTES: usize = WIDTH * 2;

    let mut cy = start_y;
    let mut idx = 0;
    for _ in 0..=HALF {
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

    // Mirror rows HALF+1..HEIGHT. Row 0 (cy = -VIEW_HEIGHT/2) is the
    // unpaired bottom edge — cy = +VIEW_HEIGHT/2 is out of range — so
    // it was handled above. For r in HALF+1..HEIGHT, source row is
    // HEIGHT - r (e.g., row 65 ← row 63, row 127 ← row 1).
    for row in (HALF + 1)..HEIGHT {
        let src_off = (HEIGHT - row) * ROW_BYTES;
        let dst_off = row * ROW_BYTES;
        let (head, tail) = bytes.split_at_mut(dst_off);
        tail[..ROW_BYTES].copy_from_slice(&head[src_off..src_off + ROW_BYTES]);
    }
}

/// Mandelbrot escape count for one complex point `(cx, cy)` in Q16.16.
/// Returns `MAX_ITER` if the point stays bounded.
/// Cardioid + period-2 bulb interior shortcut.
///
/// The main cardioid and period-2 bulb together cover most of the
/// filled interior of the set. Any pixel in those regions is
/// guaranteed to stay bounded, so we can skip the whole iteration loop
/// and return `MAX_ITER` directly. Cost is ~4 qmuls vs the ~3×64 qmuls
/// of a full interior-pixel iteration — a lopsided trade that's worth
/// it as long as interior pixels are common (they are, for the classic
/// viewport).
///
/// Formulas (standard):
/// - Period-2 bulb: `(cx + 1)² + cy² < 1/16`
/// - Main cardioid: let `q = (cx - 1/4)² + cy²`; interior iff
///   `q * (q + cx - 1/4) < cy² / 4`
#[inline(always)]
fn in_mandelbrot_interior(cx: i32, cy: i32) -> bool {
    let cy_sq = qmul(cy, cy);

    // Period-2 bulb first — it's a single qmul + two adds + one compare.
    let xp1 = cx.wrapping_add(Q_ONE);
    let xp1_sq = qmul(xp1, xp1);
    if xp1_sq.wrapping_add(cy_sq) < (Q_ONE >> 4) {
        return true;
    }

    // Main cardioid. `xm_quarter` is cx − 1/4 (the non-squared form
    // used in the second factor of the implicit test).
    let xm_quarter = cx.wrapping_sub(Q_ONE >> 2);
    let xm_sq = qmul(xm_quarter, xm_quarter);
    let q = xm_sq.wrapping_add(cy_sq);
    qmul(q, q.wrapping_add(xm_quarter)) < (cy_sq >> 2)
}

#[inline(always)]
fn mandelbrot_iter(cx: i32, cy: i32) -> u32 {
    if in_mandelbrot_interior(cx, cy) {
        return MAX_ITER;
    }

    let mut zx: i32 = 0;
    let mut zy: i32 = 0;
    let mut iter: u32 = 0;
    while iter < MAX_ITER {
        let zx2 = qmul(zx, zx);
        let zy2 = qmul(zy, zy);
        // wrapping_add is safe here: at Q16.16 with |z| < 2, zx² and zy²
        // each fit in ~18 bits, so their sum is at most ~2^19 — wraparound
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
