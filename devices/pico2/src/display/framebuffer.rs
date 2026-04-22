//! Thin embedded-graphics `DrawTarget` wrapper around a raw RGB565-BE
//! byte slice. Used only by the FPS overlay and the `lcd_demo`'s
//! fill-pattern helpers; Mandelbrot writes pixels directly, skipping
//! the trait layer.

use embedded_graphics::{
    pixelcolor::{raw::RawU16, Rgb565},
    prelude::*,
};

use super::st7735::{PANEL_HEIGHT, PANEL_WIDTH};

/// `DrawTarget`-compatible view over a `&mut [u8]` holding a
/// `PANEL_WIDTH × PANEL_HEIGHT` RGB565-BE framebuffer.
pub struct Framebuffer<'a> {
    pub bytes: &'a mut [u8],
}

impl<'a> Framebuffer<'a> {
    /// Plot one RGB565-BE pixel into the framebuffer. Silently clips
    /// against the panel bounds.
    #[inline]
    pub fn set_pixel(&mut self, x: u32, y: u32, raw: u16) {
        if x >= PANEL_WIDTH as u32 || y >= PANEL_HEIGHT as u32 {
            return;
        }
        let idx = ((y * PANEL_WIDTH as u32 + x) * 2) as usize;
        if idx + 1 < self.bytes.len() {
            self.bytes[idx] = (raw >> 8) as u8;
            self.bytes[idx + 1] = raw as u8;
        }
    }
}

impl<'a> DrawTarget for Framebuffer<'a> {
    type Color = Rgb565;
    type Error = core::convert::Infallible;

    fn draw_iter<I>(&mut self, pixels: I) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = Pixel<Self::Color>>,
    {
        for Pixel(point, color) in pixels {
            if point.x < 0 || point.y < 0 {
                continue;
            }
            let raw = RawU16::from(color).into_inner();
            self.set_pixel(point.x as u32, point.y as u32, raw);
        }
        Ok(())
    }
}

impl<'a> OriginDimensions for Framebuffer<'a> {
    fn size(&self) -> Size {
        Size::new(PANEL_WIDTH as u32, PANEL_HEIGHT as u32)
    }
}
