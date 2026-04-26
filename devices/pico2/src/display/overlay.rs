//! On-screen text overlays, shared by all three demo binaries.
//!
//! Used for the FPS readout and the "sf-nano" label. The background
//! rectangle is drawn via a raw byte loop rather than
//! `embedded_graphics::primitives::Rectangle`: in `demo_host`'s
//! `push_frame` host callback the e-g Rectangle path froze the JIT's
//! animation (the panel showed a single frame and stopped updating)
//! while equivalent raw writes did not. The root cause is unexplained
//! and filed for later; the workaround costs ~4 instructions per pixel
//! and keeps the demos on one code path.

use embedded_graphics::{
    mono_font::{ascii::FONT_6X10, MonoTextStyle},
    pixelcolor::Rgb565,
    prelude::*,
    text::{Alignment, Text},
};

use super::framebuffer::Framebuffer;
use super::st7735::PANEL_WIDTH;

/// Format `FPS: N` into `buf`. Returns the `str` view. Callers size
/// `buf` at least 16 bytes; longer fps counts are truncated.
pub fn format_fps<'a>(buf: &'a mut [u8], fps: u32) -> &'a str {
    let prefix = b"FPS: ";
    let mut i = 0;
    for b in prefix {
        if i >= buf.len() {
            break;
        }
        buf[i] = *b;
        i += 1;
    }
    if fps == 0 {
        if i < buf.len() {
            buf[i] = b'0';
            i += 1;
        }
    } else {
        let mut digits = [0u8; 10];
        let mut dcount = 0;
        let mut n = fps;
        while n > 0 {
            digits[dcount] = b'0' + (n % 10) as u8;
            n /= 10;
            dcount += 1;
        }
        while dcount > 0 && i < buf.len() {
            dcount -= 1;
            buf[i] = digits[dcount];
            i += 1;
        }
    }
    // SAFETY: bytes written are all ASCII by construction.
    unsafe { core::str::from_utf8_unchecked(&buf[..i]) }
}

/// Stamp a black rectangle with centered white text into `fb`. `origin`
/// is the top-left corner; `size` is `(width, height)` in pixels.
///
/// Call right before the DMA that ships the frame to the panel so the
/// overlay lands on the current frame, not the next one.
pub fn stamp_text_overlay(fb: &mut [u8], origin: (i32, i32), size: (i32, i32), text: &str) {
    let (ox, oy) = origin;
    let (rect_w, rect_h) = size;

    // Raw black background. See the module doc for why we can't use
    // `embedded_graphics::primitives::Rectangle` here.
    let stride = PANEL_WIDTH;
    for y in oy..(oy + rect_h) {
        if y < 0 {
            continue;
        }
        let y = y as usize;
        for x in ox..(ox + rect_w) {
            if x < 0 {
                continue;
            }
            let x = x as usize;
            let idx = (y * stride + x) * 2;
            if idx + 1 < fb.len() {
                fb[idx] = 0;
                fb[idx + 1] = 0;
            }
        }
    }

    // The `Text::draw` path *is* fine — only the e-g filled Rectangle
    // tripped the JIT freeze. Centered text inside the rect.
    let mut eg_fb = Framebuffer { bytes: fb };
    let text_style = MonoTextStyle::new(&FONT_6X10, Rgb565::WHITE);
    let text_center = Point::new(ox + rect_w / 2, oy + 9);
    Text::with_alignment(text, text_center, text_style, Alignment::Center)
        .draw(&mut eg_fb)
        .unwrap();
}

/// Stamp `FPS: NN` at `origin` using a 55×12 rectangle.
pub fn stamp_fps_overlay(fb: &mut [u8], origin: (i32, i32), fps: u32) {
    let mut msg_buf = [0u8; 16];
    let msg = format_fps(&mut msg_buf, fps);
    stamp_text_overlay(fb, origin, (55, 12), msg);
}

/// Stamp the static "sf-nano" label at `origin` using a 46×12 rectangle.
/// 7 glyphs × 6 px = 42 px of text plus 2 px padding on each side.
pub fn stamp_sf_nano_overlay(fb: &mut [u8], origin: (i32, i32)) {
    stamp_text_overlay(fb, origin, (46, 12), "sf-nano");
}
