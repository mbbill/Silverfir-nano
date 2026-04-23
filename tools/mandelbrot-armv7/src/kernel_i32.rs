//! Q17.14 integer Mandelbrot kernel. Verbatim copy of
//! `devices/pico2/src/mandelbrot_kernel.rs` — kept as a copy rather than
//! a shared crate to keep this benchmark a minimal, self-contained
//! reference binary. If the kernel changes, both copies must move
//! together or the baseline loses meaning.

pub const WIDTH: usize = 160;
pub const HEIGHT: usize = 128;
pub const FB_BYTES: usize = WIDTH * HEIGHT * 2;

const MAX_ITER: u32 = 64;
const Q_BITS: u32 = 14;

#[inline(always)]
fn qmul(a: i32, b: i32) -> i32 {
    a.wrapping_mul(b) >> Q_BITS
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
