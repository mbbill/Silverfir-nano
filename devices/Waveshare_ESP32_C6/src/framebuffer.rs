//! Thin embedded-graphics `DrawTarget` wrapper around an RGB565-BE framebuffer.

use embedded_graphics::{
    pixelcolor::{raw::RawU16, Rgb565},
    prelude::*,
};

use crate::demo::{HEIGHT, WIDTH};

pub struct Framebuffer<'a> {
    pub bytes: &'a mut [u8],
}

impl<'a> Framebuffer<'a> {
    #[inline]
    pub fn set_pixel(&mut self, x: u32, y: u32, raw: u16) {
        if x >= WIDTH as u32 || y >= HEIGHT as u32 {
            return;
        }
        let idx = ((y * WIDTH as u32 + x) * 2) as usize;
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
        Size::new(WIDTH as u32, HEIGHT as u32)
    }
}
