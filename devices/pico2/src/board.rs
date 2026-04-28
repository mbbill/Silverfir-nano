//! Per-board constants and HAL peripheral wiring shared across
//! `devices/pico2/`'s binaries.
//!
//! Scope:
//! - The Pico 2 / Pico 2 W external crystal frequency.
//! - Concrete GPIO / SPI / DMA type aliases for the Waveshare
//!   Pico-LCD-1.8 shield. Binaries that drive the display reference
//!   these instead of redeclaring them.
//!
//! Not here (per-binary):
//! - The `.start_block` `IMAGE_DEF` boot header — small enough to keep
//!   alongside `fn main`.
//! - The `.bi_entries` picotool metadata — its `rp_cargo_bin_name!`
//!   macro captures the *binary's* name and must therefore expand in
//!   the binary crate.

use rp235x_hal as hal;

use hal::{
    dma::{Channel, CH0},
    gpio::{
        bank0::{Gpio10, Gpio11, Gpio12, Gpio13, Gpio28, Gpio8, Gpio9},
        FunctionSio, FunctionSpi, Pin, PullDown, SioOutput,
    },
    pac::SPI1,
    spi::Enabled,
};

/// Pico 2 / Pico 2 W external crystal.
pub const XTAL_FREQ_HZ: u32 = 12_000_000;

/// Requested LCD SPI clock. With the default 150 MHz peripheral clock, the
/// PL022 divider can only realize 75 MHz; the binaries log the achieved rate.
pub const DISPLAY_SPI_TARGET_HZ: u32 = 80_000_000;

// ── Waveshare Pico-LCD-1.8 pin map ──────────────────────────────────
//
//   | LCD signal | Pico GPIO |
//   |------------|----------:|
//   | SCK        | GP10      |
//   | MOSI (DIN) | GP11      |
//   | MISO (unused) | GP28   |
//   | CS         | GP9       |
//   | DC         | GP8       |
//   | RST        | GP12      |
//   | BL         | GP13      |

pub type MosiPin = Pin<Gpio11, FunctionSpi, PullDown>;
pub type MisoPin = Pin<Gpio28, FunctionSpi, PullDown>;
pub type SclkPin = Pin<Gpio10, FunctionSpi, PullDown>;
pub type CsPin = Pin<Gpio9, FunctionSio<SioOutput>, PullDown>;
pub type DcPin = Pin<Gpio8, FunctionSio<SioOutput>, PullDown>;
pub type RstPin = Pin<Gpio12, FunctionSio<SioOutput>, PullDown>;
pub type BlPin = Pin<Gpio13, FunctionSio<SioOutput>, PullDown>;

/// Initialized SPI1 peripheral bound to the Waveshare shield's pins.
pub type DisplaySpi = hal::spi::Spi<Enabled, SPI1, (MosiPin, MisoPin, SclkPin), 8>;

/// DMA channel used to blit the framebuffer to the panel.
pub type DisplayCh = Channel<CH0>;
