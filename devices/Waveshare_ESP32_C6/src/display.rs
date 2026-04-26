//! Minimal ST7789 display driver for the Waveshare ESP32-C6 LCD.

use embedded_hal::spi::SpiBus;
use esp_hal::{
    gpio::Output,
    time::{Duration, Instant},
};

use crate::board::{LCD_HEIGHT, LCD_MADCTL, LCD_WIDTH, LCD_X_OFFSET, LCD_Y_OFFSET};

const CMD_SWRESET: u8 = 0x01;
const CMD_SLPOUT: u8 = 0x11;
const CMD_NORON: u8 = 0x13;
const CMD_INVON: u8 = 0x21;
const CMD_CASET: u8 = 0x2A;
const CMD_RASET: u8 = 0x2B;
const CMD_RAMWR: u8 = 0x2C;
const CMD_MADCTL: u8 = 0x36;
const CMD_COLMOD: u8 = 0x3A;
const CMD_DISPON: u8 = 0x29;

pub struct Display<SPI> {
    spi: SPI,
    cs: Output<'static>,
    dc: Output<'static>,
    rst: Output<'static>,
    bl: Output<'static>,
}

impl<SPI> Display<SPI>
where
    SPI: SpiBus<u8>,
{
    pub fn new(
        spi: SPI,
        cs: Output<'static>,
        dc: Output<'static>,
        rst: Output<'static>,
        bl: Output<'static>,
    ) -> Self {
        Self {
            spi,
            cs,
            dc,
            rst,
            bl,
        }
    }

    pub fn init(&mut self) {
        self.cs.set_high();
        self.dc.set_high();
        self.bl.set_high();

        self.rst.set_high();
        wait_ms(20);
        self.rst.set_low();
        wait_ms(20);
        self.rst.set_high();
        wait_ms(120);

        self.command(CMD_SWRESET);
        wait_ms(120);
        self.command(CMD_SLPOUT);
        wait_ms(120);

        // Landscape RGB565. The panel is 172x320 glass centered in ST7789 GRAM;
        // after MV rotation the 34-pixel gap is on Y instead of X.
        self.command_data(CMD_MADCTL, &[LCD_MADCTL]);
        self.command_data(CMD_COLMOD, &[0x55]);

        // The board's ST7789 glass matches the ESP-IDF bring-up when color
        // inversion is enabled.
        self.command(CMD_INVON);
        self.command(CMD_NORON);
        wait_ms(10);
        self.command(CMD_DISPON);
        wait_ms(120);
    }

    pub fn clear(&mut self, color: u16) {
        self.set_window(0, 0, LCD_WIDTH as u16 - 1, LCD_HEIGHT as u16 - 1);

        self.dc.set_high();
        self.cs.set_low();

        let mut line = [0u8; LCD_WIDTH * 2];
        for pixel in line.chunks_exact_mut(2) {
            pixel[0] = (color >> 8) as u8;
            pixel[1] = color as u8;
        }

        for _ in 0..LCD_HEIGHT {
            let _ = self.spi.write(&line);
        }

        let _ = self.spi.flush();
        self.cs.set_high();
    }

    pub fn draw_rgb565_be(&mut self, x: u16, y: u16, width: u16, height: u16, bytes: &[u8]) {
        debug_assert_eq!(bytes.len(), width as usize * height as usize * 2);
        debug_assert!(x as usize + width as usize <= LCD_WIDTH);
        debug_assert!(y as usize + height as usize <= LCD_HEIGHT);

        self.set_window(x, y, x + width - 1, y + height - 1);

        self.dc.set_high();
        self.cs.set_low();

        let _ = self.spi.write(bytes);
        let _ = self.spi.flush();
        self.cs.set_high();
    }

    fn set_window(&mut self, x0: u16, y0: u16, x1: u16, y1: u16) {
        let x0 = x0 + LCD_X_OFFSET;
        let x1 = x1 + LCD_X_OFFSET;
        let y0 = y0 + LCD_Y_OFFSET;
        let y1 = y1 + LCD_Y_OFFSET;

        self.command_data(CMD_CASET, &u16_pair_be(x0, x1));
        self.command_data(CMD_RASET, &u16_pair_be(y0, y1));
        self.command(CMD_RAMWR);
    }

    fn command_data(&mut self, cmd: u8, data: &[u8]) {
        self.command(cmd);
        self.data(data);
    }

    fn command(&mut self, cmd: u8) {
        self.cs.set_high();
        self.dc.set_low();
        self.cs.set_low();
        let _ = self.spi.write(&[cmd]);
        let _ = self.spi.flush();
        self.cs.set_high();
    }

    fn data(&mut self, data: &[u8]) {
        self.cs.set_high();
        self.dc.set_high();
        self.cs.set_low();
        let _ = self.spi.write(data);
        let _ = self.spi.flush();
        self.cs.set_high();
    }
}

fn wait_ms(ms: u64) {
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(ms) {}
}

fn u16_pair_be(a: u16, b: u16) -> [u8; 4] {
    [(a >> 8) as u8, a as u8, (b >> 8) as u8, b as u8]
}
