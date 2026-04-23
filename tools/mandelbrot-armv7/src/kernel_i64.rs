//! Q16.16 integer Mandelbrot kernel — the 64-bit-math variant.
//!
//! Same algorithm as `kernel_i32.rs` (Q17.14) but with the fractional
//! bits bumped to 16, forcing the `a*b` product out of i32 and through
//! i64. `qmul` becomes a 32×32→64 multiply followed by a 64-bit right
//! shift, which is where the JIT's 64-bit lowering hurts.
//!
//! This is the kernel Phase 4 is optimizing against. Keep the algorithm
//! identical to the i32 variant — differences are by design only in
//! `Q_BITS` and the intermediate type.

pub const WIDTH: usize = 160;
pub const HEIGHT: usize = 128;
pub const FB_BYTES: usize = WIDTH * HEIGHT * 2;

const MAX_ITER: u32 = 64;
const Q_BITS: u32 = 16;

/// Q16.16 fixed-point multiply. The i32×i32 product overflows i32 for
/// any |a|,|b| ≥ 2^15.5 — so we route through i64. On armv7 this lowers
/// to a single SMULL + ASR (upper register); on x64 a single 64-bit IMUL.
/// The JIT currently lowers it as an i64.mul helper call — the gap Phase
/// 4 is trying to close.
#[inline(always)]
fn qmul(a: i32, b: i32) -> i32 {
    ((a as i64).wrapping_mul(b as i64) >> Q_BITS) as i32
}

const Q_ONE: i32 = 1 << Q_BITS;
const Q_FOUR: i32 = 4 << Q_BITS;

const VIEW_CENTER_X: i32 = -(Q_ONE >> 1);
const VIEW_CENTER_Y: i32 = 0;
const VIEW_WIDTH: i32 = 3 * Q_ONE;
const VIEW_HEIGHT: i32 = (3 * Q_ONE * 4) / 5;

pub fn render(bytes: &mut [u8], frame: u32) {
    debug_assert_eq!(bytes.len(), FB_BYTES);

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

#[inline(always)]
fn mandelbrot_iter(cx: i32, cy: i32) -> u32 {
    let mut zx: i32 = 0;
    let mut zy: i32 = 0;
    let mut iter: u32 = 0;
    while iter < MAX_ITER {
        let zx2 = qmul(zx, zx);
        let zy2 = qmul(zy, zy);
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

#[inline(always)]
fn palette(iter: u32, frame: u32) -> u16 {
    if iter >= MAX_ITER {
        return 0x0000;
    }
    let hue = (iter.wrapping_add(frame)) & 0x1f;
    const TABLE: [u16; 32] = [
        0xF800, 0xF880, 0xF900, 0xF980, 0xFA00, 0xFA80, 0xFB00, 0xFB80, 0xFC00, 0xFCC0, 0xFD80,
        0xFE40, 0xFF00, 0xF720, 0xEF60, 0xE780, 0xDFC0, 0xD7E0, 0x87E0, 0x37E0, 0x07E4, 0x07EA,
        0x07EF, 0x07F5, 0x07FB, 0x07FF, 0x04FF, 0x023F, 0x001F, 0x401F, 0x801F, 0xC01F,
    ];
    TABLE[hue as usize]
}
