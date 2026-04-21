//! Silverfir-nano Pico 2 / Pico 2 W (RP2350) bring-up firmware — milestone 1.
//!
//! Boots, initializes clocks to the stock 150 MHz SYSCLK, then runs a 1 Hz
//! heartbeat loop that logs a tick counter over defmt-RTT. Proves the
//! toolchain, boot block, memory map, linker scripts, probe-rs flash and
//! RTT paths all work end-to-end — before linking sf-nano-core.

#![no_std]
#![no_main]

use defmt_rtt as _;
use panic_probe as _;
use rp235x_hal as hal;

/// Boot ROM image header. Placed in `.start_block` (first 4 KiB of flash
/// by the linker script) so the RP2350 Boot ROM recognizes this as a
/// valid secure-mode executable.
#[unsafe(link_section = ".start_block")]
#[used]
pub static IMAGE_DEF: hal::block::ImageDef = hal::block::ImageDef::secure_exe();

/// Pico 2 / Pico 2 W external crystal.
const XTAL_FREQ_HZ: u32 = 12_000_000;

#[hal::entry]
fn main() -> ! {
    let mut pac = hal::pac::Peripherals::take().unwrap();
    let mut watchdog = hal::Watchdog::new(pac.WATCHDOG);

    let _clocks = hal::clocks::init_clocks_and_plls(
        XTAL_FREQ_HZ,
        pac.XOSC,
        pac.CLOCKS,
        pac.PLL_SYS,
        pac.PLL_USB,
        &mut pac.RESETS,
        &mut watchdog,
    )
    .unwrap();

    defmt::info!("sf-nano-pico2 alive @ 150 MHz SYSCLK");

    let mut tick: u32 = 0;
    loop {
        cortex_m::asm::delay(150_000_000);
        defmt::info!("tick {=u32}", tick);
        tick = tick.wrapping_add(1);
    }
}

/// Picotool `binary-info` metadata — consumed by `picotool info <elf>` to
/// report program name, version, and build attributes without needing
/// external debug symbols.
#[unsafe(link_section = ".bi_entries")]
#[used]
pub static PICOTOOL_ENTRIES: [hal::binary_info::EntryAddr; 4] = [
    hal::binary_info::rp_cargo_bin_name!(),
    hal::binary_info::rp_cargo_version!(),
    hal::binary_info::rp_program_description!(c"Silverfir-nano Pico 2 bring-up"),
    hal::binary_info::rp_program_build_attribute!(),
];
