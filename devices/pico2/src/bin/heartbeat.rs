//! Bring-up heartbeat — milestone 1.
//!
//! Boots, initializes clocks to the stock 150 MHz SYSCLK, then runs a
//! 1 Hz heartbeat loop that logs a tick counter over defmt-RTT. Proves
//! the toolchain, boot block, memory map, linker scripts, probe-rs
//! flash and RTT paths all work end-to-end — before sf-nano-core
//! actually does any JIT work.
//!
//! `cargo run --bin heartbeat`.

#![no_std]
#![no_main]

use defmt_rtt as _;
use panic_probe as _;
use rp235x_hal as hal;

// Pull sf-nano-core into the dep graph. The shared library crate
// `sf_nano_pico2` carries the `#[global_allocator]` and the null
// `sf_os_*` shims needed to link this in.
use sf_nano_core as _;
use sf_nano_pico2 as lib;

use lib::board::XTAL_FREQ_HZ;

/// Boot ROM image header. Placed in `.start_block` (first 4 KiB of flash
/// by the linker script) so the RP2350 Boot ROM recognizes this as a
/// valid secure-mode executable.
#[unsafe(link_section = ".start_block")]
#[used]
pub static IMAGE_DEF: hal::block::ImageDef = hal::block::ImageDef::secure_exe();

#[hal::entry]
fn main() -> ! {
    lib::init();

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

    defmt::info!("heartbeat: alive @ 150 MHz SYSCLK");

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
    hal::binary_info::rp_program_description!(c"sf-nano-pico2 heartbeat bring-up"),
    hal::binary_info::rp_program_build_attribute!(),
];
