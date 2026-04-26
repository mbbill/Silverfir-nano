//! On-screen text overlays matching the Pico2 native demo.

use embedded_graphics::{
    mono_font::{ascii::FONT_6X10, MonoTextStyle},
    pixelcolor::Rgb565,
    prelude::*,
    text::{Alignment, Text},
};

use crate::{framebuffer::Framebuffer, kernels::mandelbrot::WIDTH};

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
    unsafe { core::str::from_utf8_unchecked(&buf[..i]) }
}

pub fn stamp_text_overlay(fb: &mut [u8], origin: (i32, i32), size: (i32, i32), text: &str) {
    let (ox, oy) = origin;
    let (rect_w, rect_h) = size;

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
            let idx = (y * WIDTH + x) * 2;
            if idx + 1 < fb.len() {
                fb[idx] = 0;
                fb[idx + 1] = 0;
            }
        }
    }

    let mut eg_fb = Framebuffer { bytes: fb };
    let text_style = MonoTextStyle::new(&FONT_6X10, Rgb565::WHITE);
    let text_center = Point::new(ox + rect_w / 2, oy + 9);
    Text::with_alignment(text, text_center, text_style, Alignment::Center)
        .draw(&mut eg_fb)
        .unwrap();
}

pub fn stamp_fps_overlay(fb: &mut [u8], origin: (i32, i32), fps: u32) {
    let mut msg_buf = [0u8; 16];
    let msg = format_fps(&mut msg_buf, fps);
    stamp_text_overlay(fb, origin, (55, 12), msg);
}

pub fn stamp_sf_nano_overlay(fb: &mut [u8], origin: (i32, i32)) {
    stamp_text_overlay(fb, origin, (46, 12), "sf-nano");
}
