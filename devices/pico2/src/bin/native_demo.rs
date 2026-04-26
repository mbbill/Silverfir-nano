//! Native-CPU demo — same selected render kernel as `demo_host`,
//! compiled straight to the active target by rustc. Serves as the performance
//! ceiling the Wasm JIT is measured against.
//!
//! `cargo run --bin native_demo --release`.

#![no_std]
#![no_main]

#[cfg(all(feature = "demo-mandelbrot", feature = "demo-cube"))]
compile_error!("select exactly one demo feature");

#[cfg(not(any(feature = "demo-mandelbrot", feature = "demo-cube")))]
compile_error!("select one demo feature");

extern crate alloc;

use defmt_rtt as _;
// Panic handler: panic-probe on ARM, sf_nano_pico2::arch::rv on RV (see lib).
#[cfg(target_arch = "arm")]
use panic_probe as _;
use rp235x_hal as hal;

use alloc::boxed::Box;
use embedded_hal::{digital::OutputPin, spi::SpiBus, spi::MODE_0};
use hal::{
    clocks::Clock,
    dma::{single_buffer, DMAExt},
    fugit::RateExtU32,
};

use sf_nano_core as _;
use sf_nano_pico2 as lib;

use lib::board::XTAL_FREQ_HZ;
use lib::display::{self, st7735};
#[cfg(feature = "demo-mandelbrot")]
use lib::kernels::mandelbrot as demo;

#[cfg(feature = "demo-cube")]
use lib::kernels::cube as demo;

use demo::FB_BYTES;

#[unsafe(link_section = ".start_block")]
#[used]
pub static IMAGE_DEF: hal::block::ImageDef = hal::block::ImageDef::secure_exe();

#[hal::entry]
fn main() -> ! {
    lib::init();

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
    let sys_hz = clocks.system_clock.freq().to_Hz();

    let sio = hal::Sio::new(pac.SIO);
    let pins = hal::gpio::Pins::new(
        pac.IO_BANK0,
        pac.PADS_BANK0,
        sio.gpio_bank0,
        &mut pac.RESETS,
    );

    defmt::info!("native_demo: clocks up, sys={} Hz", sys_hz);

    let mosi = pins.gpio11.into_function::<hal::gpio::FunctionSpi>();
    let miso = pins.gpio28.into_function::<hal::gpio::FunctionSpi>();
    let sclk = pins.gpio10.into_function::<hal::gpio::FunctionSpi>();

    let mut cs = pins.gpio9.into_push_pull_output();
    let mut dc = pins.gpio8.into_push_pull_output();
    let mut rst = pins.gpio12.into_push_pull_output();
    let mut bl = pins.gpio13.into_push_pull_output();
    bl.set_high().unwrap();

    let spi_bus = hal::spi::Spi::<_, _, _, 8>::new(pac.SPI1, (mosi, miso, sclk));
    let mut spi_bus = spi_bus.init(
        &mut pac.RESETS,
        clocks.peripheral_clock.freq(),
        40u32.MHz(),
        MODE_0,
    );
    defmt::info!("native_demo: SPI1 @ 40 MHz");

    st7735::init(&mut spi_bus, &mut cs, &mut dc, &mut rst, &mut timer);
    defmt::info!("native_demo: ST7735s init complete");

    let dma = pac.DMA.split(&mut pac.RESETS);

    let boxed: Box<[u8; FB_BYTES]> = Box::new([0u8; FB_BYTES]);
    let fb_bytes: &'static mut [u8; FB_BYTES] = Box::leak(boxed);

    let mut spi_opt = Some(spi_bus);
    let mut ch_opt = Some(dma.ch0);
    let mut fb_opt: Option<&'static mut [u8; FB_BYTES]> = Some(fb_bytes);

    defmt::info!("native_demo: entering render loop");

    let mut frame: u32 = 0;
    let mut compute_accumulator_us: u64 = 0;
    let mut push_accumulator_us: u64 = 0;
    let mut fps_accumulator_us: u64 = 0;
    let mut fps_samples: u32 = 0;
    let mut last_log = timer.get_counter();
    let mut displayed_fps: u32 = 0;

    // Top-left overlay origin — same convention as demo_host.
    const FPS_ORIGIN: (i32, i32) = (2, 2);
    // Top-right "sf-nano" label. Panel is 160 wide; 46×12 rect with a 2px margin.
    const LABEL_ORIGIN: (i32, i32) = (112, 2);

    loop {
        // Timing uses the rp235x always-on peripheral timer (μs resolution).
        // Previous version used DWT::cycle_count, but that is ARM-only and
        // additionally only runs while the debug unit is enabled (i.e.
        // while probe-rs is attached) — disconnecting the probe would
        // freeze DWT and the fps overlay would drop to 0 even though
        // rendering kept going. The peripheral timer reports correctly
        // with or without a probe and works on both ARM and RV32 modes
        // of RP2350.
        let t_frame_start = timer.get_counter();

        // --- Compute + overlay ---
        let t_compute_start = timer.get_counter();
        {
            let bytes: &mut [u8] = fb_opt.as_mut().unwrap().as_mut_slice();
            demo::render(bytes, frame);
            display::stamp_fps_overlay(bytes, FPS_ORIGIN, displayed_fps);
            display::stamp_sf_nano_overlay(bytes, LABEL_ORIGIN);
        }
        let t_compute_end = timer.get_counter();

        // --- Push: address window + DMA ---
        let t_push_start = timer.get_counter();
        let mut spi = spi_opt.take().unwrap();
        let ch = ch_opt.take().unwrap();
        let bytes = fb_opt.take().unwrap();

        st7735::write_cmd(&mut spi, &mut cs, &mut dc, st7735::CMD_CASET);
        st7735::write_data(&mut spi, &mut cs, &mut dc, &st7735::CASET_DATA);
        st7735::write_cmd(&mut spi, &mut cs, &mut dc, st7735::CMD_RASET);
        st7735::write_data(&mut spi, &mut cs, &mut dc, &st7735::RASET_DATA);
        st7735::write_cmd(&mut spi, &mut cs, &mut dc, st7735::CMD_RAMWR);

        cs.set_high().ok();
        dc.set_high().ok();
        cs.set_low().ok();
        let transfer = single_buffer::Config::new(ch, bytes, spi).start();
        let (ch_back, bytes_back, mut spi_back) = transfer.wait();
        SpiBus::<u8>::flush(&mut spi_back).ok();
        lib::arch::delay_cycles(100);
        cs.set_high().ok();

        spi_opt = Some(spi_back);
        ch_opt = Some(ch_back);
        fb_opt = Some(bytes_back);
        let t_push_end = timer.get_counter();

        // --- Timing bookkeeping ---
        let t_frame_end = timer.get_counter();
        let frame_us = (t_frame_end - t_frame_start).to_micros();
        let compute_us = (t_compute_end - t_compute_start).to_micros();
        let push_us = (t_push_end - t_push_start).to_micros();
        fps_accumulator_us = fps_accumulator_us.saturating_add(frame_us);
        compute_accumulator_us = compute_accumulator_us.saturating_add(compute_us);
        push_accumulator_us = push_accumulator_us.saturating_add(push_us);
        fps_samples += 1;

        let now = timer.get_counter();
        if (now - last_log).to_micros() >= 1_000_000 {
            let n = fps_samples.max(1) as u64;
            let avg_us = fps_accumulator_us / n;
            let avg_compute = compute_accumulator_us / n;
            let avg_push = push_accumulator_us / n;
            displayed_fps = if avg_us > 0 {
                (1_000_000 / avg_us) as u32
            } else {
                0
            };
            defmt::info!(
                "frame {}: {} fps  |  compute={} us  push={} us  total={} us  (n={})",
                frame,
                displayed_fps,
                avg_compute,
                avg_push,
                avg_us,
                fps_samples
            );
            fps_accumulator_us = 0;
            compute_accumulator_us = 0;
            push_accumulator_us = 0;
            fps_samples = 0;
            last_log = now;
        }

        frame = frame.wrapping_add(1);
    }
}

#[unsafe(link_section = ".bi_entries")]
#[used]
pub static PICOTOOL_ENTRIES: [hal::binary_info::EntryAddr; 4] = [
    hal::binary_info::rp_cargo_bin_name!(),
    hal::binary_info::rp_cargo_version!(),
    hal::binary_info::rp_program_description!(c"sf-nano-pico2 native demo"),
    hal::binary_info::rp_program_build_attribute!(),
];
