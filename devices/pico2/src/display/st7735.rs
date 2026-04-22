//! ST7735s panel init + low-level SPI command/data helpers.
//!
//! Copied from Waveshare's Pico-LCD-1.8 MicroPython driver with the
//! quirks HACKING.md §5.6 documents (row-end 0x82, CS-high stability
//! delay, etc.). We skip `mipidsi` because its `ExclusiveDevice`
//! wrapper swallows the raw `Spi` handle we need back for DMA.

use embedded_hal::{delay::DelayNs, digital::OutputPin, spi::SpiBus};

/// Panel physical dimensions (landscape, MADCTL = 0x70). RGB565 at 2
/// bytes per pixel gives a 40 960-byte framebuffer.
pub const PANEL_WIDTH: usize = 160;
pub const PANEL_HEIGHT: usize = 128;

// Visible-window coordinates copied from Waveshare's driver. Tightening
// the row end to 0x81 "to exactly match" 128 rows makes the panel show
// white — the visible area maps to an address range that requires the
// 0x82 boundary. Keep these numbers.
pub const CASET_DATA: [u8; 4] = [0x00, 0x01, 0x00, 0xA0]; // cols 1..=160
pub const RASET_DATA: [u8; 4] = [0x00, 0x02, 0x00, 0x82]; // rows 2..=130

pub const CMD_CASET: u8 = 0x2A;
pub const CMD_RASET: u8 = 0x2B;
pub const CMD_RAMWR: u8 = 0x2C;

/// Tightest possible command write: per-command CS toggling, blocking
/// SPI, flush-before-deassert. All writes in this module follow this
/// shape, matching Waveshare's driver.
pub fn write_cmd<SPI, CS, DC>(spi: &mut SPI, cs: &mut CS, dc: &mut DC, cmd: u8)
where
    SPI: SpiBus<u8>,
    CS: OutputPin,
    DC: OutputPin,
{
    cs.set_high().ok();
    dc.set_low().ok();
    cs.set_low().ok();
    spi.write(&[cmd]).ok();
    spi.flush().ok();
    cs.set_high().ok();
}

pub fn write_data<SPI, CS, DC>(spi: &mut SPI, cs: &mut CS, dc: &mut DC, data: &[u8])
where
    SPI: SpiBus<u8>,
    CS: OutputPin,
    DC: OutputPin,
{
    cs.set_high().ok();
    dc.set_high().ok();
    cs.set_low().ok();
    spi.write(data).ok();
    spi.flush().ok();
    cs.set_high().ok();
}

/// Drive the ST7735s through its full startup sequence: hardware reset,
/// MADCTL / COLMOD / frame rate / gamma / power settings, sleep-out,
/// display-on. Runs once at boot; cost doesn't matter.
pub fn init<SPI, CS, DC, RST, D>(spi: &mut SPI, cs: &mut CS, dc: &mut DC, rst: &mut RST, delay: &mut D)
where
    SPI: SpiBus<u8>,
    CS: OutputPin,
    DC: OutputPin,
    RST: OutputPin,
    D: DelayNs,
{
    // Hardware reset.
    rst.set_high().ok();
    delay.delay_us(20);
    rst.set_low().ok();
    delay.delay_us(20);
    rst.set_high().ok();
    delay.delay_ms(120);

    // MADCTL: row-addr flip + col-addr flip + MV swap → landscape.
    write_cmd(spi, cs, dc, 0x36);
    write_data(spi, cs, dc, &[0x70]);

    // COLMOD: 16-bit/pixel RGB565.
    write_cmd(spi, cs, dc, 0x3A);
    write_data(spi, cs, dc, &[0x05]);

    // Frame rate control.
    write_cmd(spi, cs, dc, 0xB1);
    write_data(spi, cs, dc, &[0x01, 0x2C, 0x2D]);
    write_cmd(spi, cs, dc, 0xB2);
    write_data(spi, cs, dc, &[0x01, 0x2C, 0x2D]);
    write_cmd(spi, cs, dc, 0xB3);
    write_data(spi, cs, dc, &[0x01, 0x2C, 0x2D, 0x01, 0x2C, 0x2D]);

    // Column inversion.
    write_cmd(spi, cs, dc, 0xB4);
    write_data(spi, cs, dc, &[0x07]);

    // Power sequence.
    write_cmd(spi, cs, dc, 0xC0);
    write_data(spi, cs, dc, &[0xA2, 0x02, 0x84]);
    write_cmd(spi, cs, dc, 0xC1);
    write_data(spi, cs, dc, &[0xC5]);
    write_cmd(spi, cs, dc, 0xC2);
    write_data(spi, cs, dc, &[0x0A, 0x00]);
    write_cmd(spi, cs, dc, 0xC3);
    write_data(spi, cs, dc, &[0x8A, 0x2A]);
    write_cmd(spi, cs, dc, 0xC4);
    write_data(spi, cs, dc, &[0x8A, 0xEE]);

    // VCOM.
    write_cmd(spi, cs, dc, 0xC5);
    write_data(spi, cs, dc, &[0x0E]);

    // Gamma.
    write_cmd(spi, cs, dc, 0xE0);
    write_data(
        spi,
        cs,
        dc,
        &[
            0x0F, 0x1A, 0x0F, 0x18, 0x2F, 0x28, 0x20, 0x22, 0x1F, 0x1B, 0x23, 0x37, 0x00, 0x07,
            0x02, 0x10,
        ],
    );
    write_cmd(spi, cs, dc, 0xE1);
    write_data(
        spi,
        cs,
        dc,
        &[
            0x0F, 0x1B, 0x0F, 0x17, 0x33, 0x2C, 0x29, 0x2E, 0x30, 0x30, 0x39, 0x3F, 0x00, 0x07,
            0x03, 0x10,
        ],
    );

    // Enable test / disable RAM power save.
    write_cmd(spi, cs, dc, 0xF0);
    write_data(spi, cs, dc, &[0x01]);
    write_cmd(spi, cs, dc, 0xF6);
    write_data(spi, cs, dc, &[0x00]);

    // Sleep out.
    write_cmd(spi, cs, dc, 0x11);
    delay.delay_ms(120);

    // Display on. Waveshare's driver does not pause here; adding a
    // delay after DISPON empirically leaves the panel stuck on white.
    write_cmd(spi, cs, dc, 0x29);
}
