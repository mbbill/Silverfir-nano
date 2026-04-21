//! LCD demo — drives the Waveshare Pico-LCD-1.8 (ST7735S, 160x128,
//! RGB565) over SPI1 and draws a static frame. Independent of the
//! milestone sequence; the main heartbeat firmware lives in `main.rs`.
//!
//! Run with:
//!
//!     cargo run --bin lcd_demo
//!
//! Pin map (fixed by the Waveshare shield):
//!
//!   | LCD signal | Pico GPIO |
//!   |------------|----------:|
//!   | SCK        | GP10      |
//!   | MOSI (DIN) | GP11      |
//!   | CS         | GP9       |
//!   | DC         | GP8       |
//!   | RST        | GP12      |
//!   | BL         | GP13      |

#![no_std]
#![no_main]

use defmt_rtt as _;
use panic_probe as _;
use rp235x_hal as hal;

// Both binaries in this crate share the same dep graph, so sf-nano-core
// is linked here too even though the LCD demo never calls into it. The
// shared library crate provides the `#[global_allocator]` + null
// `sf_os_*` shims needed to satisfy the link.
use sf_nano_core as _;
use sf_nano_pico2 as lib;

use embedded_graphics::{
    mono_font::{ascii::FONT_10X20, MonoTextStyle},
    pixelcolor::Rgb565,
    prelude::*,
    primitives::{PrimitiveStyleBuilder, Rectangle},
    text::{Alignment, Text},
};
use embedded_hal::{delay::DelayNs, digital::OutputPin};
use embedded_hal_bus::spi::ExclusiveDevice;
use hal::{clocks::Clock, fugit::RateExtU32};
use mipidsi::{
    interface::SpiInterface,
    models::ST7735s,
    options::{ColorInversion, ColorOrder, Orientation, Rotation},
    Builder,
};

#[unsafe(link_section = ".start_block")]
#[used]
pub static IMAGE_DEF: hal::block::ImageDef = hal::block::ImageDef::secure_exe();

const XTAL_FREQ_HZ: u32 = 12_000_000;

#[hal::entry]
fn main() -> ! {
    lib::heap::init();

    let mut pac = hal::pac::Peripherals::take().unwrap();
    let mut watchdog = hal::Watchdog::new(pac.WATCHDOG);

    let clocks = hal::clocks::init_clocks_and_plls(
        XTAL_FREQ_HZ,
        pac.XOSC,
        pac.CLOCKS,
        pac.PLL_SYS,
        pac.PLL_USB,
        &mut pac.RESETS,
        &mut watchdog,
    )
    .unwrap();

    let mut timer = hal::Timer::new_timer0(pac.TIMER0, &mut pac.RESETS, &clocks);

    let sio = hal::Sio::new(pac.SIO);
    let pins = hal::gpio::Pins::new(
        pac.IO_BANK0,
        pac.PADS_BANK0,
        sio.gpio_bank0,
        &mut pac.RESETS,
    );

    defmt::info!(
        "lcd_demo: clocks up, sys={} Hz peri={} Hz",
        clocks.system_clock.freq().to_Hz(),
        clocks.peripheral_clock.freq().to_Hz()
    );

    // SPI1 pins per Waveshare shield. GP28 is unconnected on the shield
    // and serves as a dummy MISO — rp235x-hal's SPI builder wants all
    // three SPI pins even when we only transmit.
    let mosi = pins.gpio11.into_function::<hal::gpio::FunctionSpi>();
    let miso = pins.gpio28.into_function::<hal::gpio::FunctionSpi>();
    let sclk = pins.gpio10.into_function::<hal::gpio::FunctionSpi>();

    // Control pins driven manually as push-pull outputs.
    let cs = pins.gpio9.into_push_pull_output();
    let dc = pins.gpio8.into_push_pull_output();
    let rst = pins.gpio12.into_push_pull_output();
    let mut bl = pins.gpio13.into_push_pull_output();

    bl.set_high().unwrap();
    defmt::info!("lcd_demo: backlight on");

    let spi_bus = hal::spi::Spi::<_, _, _, 8>::new(pac.SPI1, (mosi, miso, sclk));
    let spi_bus = spi_bus.init(
        &mut pac.RESETS,
        clocks.peripheral_clock.freq(),
        20u32.MHz(),
        embedded_hal::spi::MODE_0,
    );
    defmt::info!("lcd_demo: SPI1 @ 20 MHz");

    let spi_device = ExclusiveDevice::new_no_delay(spi_bus, cs).unwrap();

    // DCX buffer used by mipidsi's SpiInterface for command/data packing.
    // Sized comfortably above a single framebuffer row-pair (160 px * 2 B).
    let mut dcx_buf = [0u8; 512];
    let di = SpiInterface::new(spi_device, dc, &mut dcx_buf);

    defmt::info!("lcd_demo: initializing ST7735s");
    let mut display = Builder::new(ST7735s, di)
        .reset_pin(rst)
        .display_size(128, 160)
        // Waveshare 1.8" shield is a green-tab ST7735s: 2 px column
        // offset and 1 px row offset in native (portrait) coordinates.
        // Builder::display_offset() takes native-space values before the
        // rotation is applied.
        .display_offset(2, 1)
        .orientation(Orientation::new().rotate(Rotation::Deg90))
        .color_order(ColorOrder::Rgb)
        .invert_colors(ColorInversion::Normal)
        .init(&mut timer)
        .unwrap();
    defmt::info!("lcd_demo: display init complete");

    display.clear(Rgb565::BLUE).unwrap();
    defmt::info!("lcd_demo: cleared to blue");

    // White rectangle as the text backdrop.
    let rect_style = PrimitiveStyleBuilder::new()
        .fill_color(Rgb565::WHITE)
        .build();
    // 13 chars * 10 px = 130 px text width; center at x=80 spans x=15..145.
    // Rectangle spans x=10..150 with 5 px margin on each side of the text.
    Rectangle::new(Point::new(10, 40), Size::new(140, 48))
        .into_styled(rect_style)
        .draw(&mut display)
        .unwrap();

    let text_style = MonoTextStyle::new(&FONT_10X20, Rgb565::BLACK);
    Text::with_alignment(
        "sf-nano-pico2",
        Point::new(80, 68),
        text_style,
        Alignment::Center,
    )
    .draw(&mut display)
    .unwrap();

    defmt::info!("lcd_demo: frame drawn — entering heartbeat");

    let mut tick: u32 = 0;
    loop {
        timer.delay_ms(1_000);
        defmt::info!("tick {=u32}", tick);
        tick = tick.wrapping_add(1);
    }
}

#[unsafe(link_section = ".bi_entries")]
#[used]
pub static PICOTOOL_ENTRIES: [hal::binary_info::EntryAddr; 4] = [
    hal::binary_info::rp_cargo_bin_name!(),
    hal::binary_info::rp_cargo_version!(),
    hal::binary_info::rp_program_description!(c"sf-nano-pico2 LCD demo"),
    hal::binary_info::rp_program_build_attribute!(),
];
