//! Native Pico 2 demo host — embeds the selected Wasm render demo and
//! runs it through sf-nano-core on the device.
//!
//! The guest crate compiles the selected shared render kernel to
//! `wasm32-unknown-unknown` and JIT-executes it with sf-nano-core on
//! the device.
//!
//! Architecture: the Wasm module owns rendering into its own linear
//! memory, then calls the host-imported `env.push_frame` to present.
//! The host `push_frame` handler DMAs directly from the Wasm linear
//! memory (no 40 KiB duplicate framebuffer) via a small custom
//! `ReadTarget` impl that wraps a raw (ptr, len) pair. The unsafe
//! invariant is held by the handler's synchronous DMA-wait: nothing
//! else touches the Wasm memory while DMA is running.
//!
//! `cargo run --bin demo_host --release`

#![no_std]
#![no_main]

extern crate alloc;

use defmt_rtt as _;
// Panic handler: panic-probe on ARM, sf_nano_pico2::arch::rv on RV (see lib).
#[cfg(target_arch = "arm")]
use panic_probe as _;
use rp235x_hal as hal;

use embedded_hal::{digital::OutputPin, spi::SpiBus, spi::MODE_0};
use hal::{
    clocks::Clock,
    dma::{single_buffer, DMAExt, ReadTarget},
    fugit::RateExtU32,
};

use alloc::rc::Rc;
use core::cell::RefCell;
use sf_nano_core::{Import, Instance, Value, WasmError};
use sf_nano_pico2 as lib;

use lib::board::{CsPin, DcPin, DisplayCh, DisplaySpi, DISPLAY_SPI_TARGET_HZ, XTAL_FREQ_HZ};
use lib::display::{self, st7735};

#[unsafe(link_section = ".start_block")]
#[used]
pub static IMAGE_DEF: hal::block::ImageDef = hal::block::ImageDef::secure_exe();

/// The Wasm module compiled by `build.rs` from `wasm-demo/`. The
/// actual demo inside can be swapped by editing the `use … as demo;`
/// line in `wasm-demo/src/lib.rs` — the host side stays the same.
const WASM_DEMO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/demo.wasm"));

/// Peripherals owned by the `push_frame` host function. Populated once
/// in `main` before module instantiation, then `take()`-shuffled on
/// every frame — same pattern used by `native_demo` so the DMA can
/// take the SPI+channel by value
/// and hand them back via `wait()`.
struct DisplayCtx {
    spi: Option<DisplaySpi>,
    ch: Option<DisplayCh>,
    cs: CsPin,
    dc: DcPin,
    /// Latest-computed fps, written by the main loop after each invoke.
    /// `push_frame` reads it and stamps the overlay onto the framebuffer
    /// inside Wasm linear memory just before the DMA, so the readout
    /// lives on-panel without any fps code on the Wasm side.
    fps: u32,
}

// --- Custom DMA source for Wasm linear memory ------------------------

/// A raw (ptr, len) pair that looks like a `ReadTarget` to rp235x-hal's
/// DMA. We only ever construct this inside `push_frame`, where we hold
/// the Caller's `&mut [u8]` borrow of Wasm linear memory — that
/// mutable borrow is in scope for the entire DMA + flush + CS
/// sequence, so the pointer is guaranteed valid and non-aliased for
/// the transfer's lifetime.
struct WasmMemSource {
    ptr: *const u8,
    len: u32,
}

// SAFETY: see struct doc. The `ReadTarget` contract requires the
// pointer range stays valid for the duration of the transfer; we
// uphold that by keeping the Caller's `&mut [u8]` borrow alive across
// the whole synchronous DMA operation in `push_frame`.
unsafe impl ReadTarget for WasmMemSource {
    type ReceivedWord = u8;

    fn rx_treq() -> Option<u8> {
        None
    }

    fn rx_address_count(&self) -> (u32, u32) {
        (self.ptr as u32, self.len)
    }

    fn rx_increment(&self) -> bool {
        true
    }
}

// --- The host-imported `env.push_frame` ------------------------------

fn present_frame(
    ctx: &mut DisplayCtx,
    mem: &mut [u8],
    offset: usize,
    len_u: usize,
) -> Result<(), WasmError> {
    let end = offset
        .checked_add(len_u)
        .ok_or_else(|| WasmError::invalid("push_frame: offset+len overflow"))?;
    if end > mem.len() {
        return Err(WasmError::invalid("push_frame: range out of wasm memory"));
    }

    // Stamp the fps overlay onto the framebuffer *before* the DMA so the
    // readout ships to the panel with this frame. `stamp_fps_overlay`
    // writes the background rectangle via a raw loop and the text via
    // embedded-graphics; see `display::overlay` for why the e-g
    // Rectangle path is avoided here.
    display::stamp_fps_overlay(&mut mem[offset..end], (2, 2), ctx.fps);
    display::stamp_sf_nano_overlay(&mut mem[offset..end], (112, 2));
    let src_ptr = unsafe { mem.as_ptr().add(offset) };

    let mut spi = ctx
        .spi
        .take()
        .ok_or_else(|| WasmError::internal("push_frame: spi already borrowed"))?;
    let ch = ctx
        .ch
        .take()
        .ok_or_else(|| WasmError::internal("push_frame: dma channel already borrowed"))?;

    // Address window + RAMWR via blocking SPI (13 bytes, ~2.6 µs at 40 MHz).
    st7735::write_cmd(&mut spi, &mut ctx.cs, &mut ctx.dc, st7735::CMD_CASET);
    st7735::write_data(&mut spi, &mut ctx.cs, &mut ctx.dc, &st7735::CASET_DATA);
    st7735::write_cmd(&mut spi, &mut ctx.cs, &mut ctx.dc, st7735::CMD_RASET);
    st7735::write_data(&mut spi, &mut ctx.cs, &mut ctx.dc, &st7735::RASET_DATA);
    st7735::write_cmd(&mut spi, &mut ctx.cs, &mut ctx.dc, st7735::CMD_RAMWR);

    // Enter data phase for the DMA.
    ctx.cs.set_high().ok();
    ctx.dc.set_high().ok();
    ctx.cs.set_low().ok();

    let source = WasmMemSource {
        ptr: src_ptr,
        len: len_u as u32,
    };
    let transfer = single_buffer::Config::new(ch, source, spi).start();
    let (ch_back, _source_back, mut spi_back) = transfer.wait();
    SpiBus::<u8>::flush(&mut spi_back).ok();
    lib::arch::delay_cycles(100);
    ctx.cs.set_high().ok();

    ctx.spi = Some(spi_back);
    ctx.ch = Some(ch_back);
    Ok(())
}

/// Bind `env.push_frame` to the display peripherals.
///
/// The handler captures them; nothing here needs a `static mut` or a raw
/// pointer, because an import callback may own state.
fn push_frame_handler(
    ctx: Rc<RefCell<DisplayCtx>>,
) -> impl Fn(&mut sf_nano_core::Caller, &[Value], &mut [Value]) -> Result<(), WasmError> + 'static {
    move |caller: &mut sf_nano_core::Caller,
          args: &[Value],
          _returns: &mut [Value]|
          -> Result<(), WasmError> {
        let (offset, len_u) = match (args.first(), args.get(1)) {
            (Some(Value::I32(offset)), Some(Value::I32(len))) => {
                if *offset < 0 || *len < 0 {
                    return Err(WasmError::invalid("push_frame: negative offset or len"));
                }
                (*offset as usize, *len as usize)
            }
            _ => return Err(WasmError::invalid("push_frame: expected (i32, i32)")),
        };

        let mem = caller
            .memory_mut()
            .ok_or_else(|| WasmError::invalid("push_frame: wasm module has no memory"))?;
        present_frame(&mut ctx.borrow_mut(), mem, offset, len_u)
    }
}

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

    defmt::info!("demo_host: clocks up, sys={} Hz", sys_hz);
    defmt::info!("demo_host: wasm module is {} bytes", WASM_DEMO.len());

    let mosi = pins.gpio11.into_function::<hal::gpio::FunctionSpi>();
    let miso = pins.gpio28.into_function::<hal::gpio::FunctionSpi>();
    let sclk = pins.gpio10.into_function::<hal::gpio::FunctionSpi>();

    let cs = pins.gpio9.into_push_pull_output();
    let dc = pins.gpio8.into_push_pull_output();
    let mut rst = pins.gpio12.into_push_pull_output();
    let mut bl = pins.gpio13.into_push_pull_output();
    bl.set_high().unwrap();

    let spi_bus = hal::spi::Spi::<_, _, _, 8>::new(pac.SPI1, (mosi, miso, sclk));
    let mut spi_bus = spi_bus.init(
        &mut pac.RESETS,
        clocks.peripheral_clock.freq(),
        DISPLAY_SPI_TARGET_HZ.Hz(),
        MODE_0,
    );
    let actual_spi_hz =
        spi_bus.set_baudrate(clocks.peripheral_clock.freq(), DISPLAY_SPI_TARGET_HZ.Hz());
    defmt::info!(
        "demo_host: SPI1 requested={} Hz actual={} Hz",
        DISPLAY_SPI_TARGET_HZ,
        actual_spi_hz.to_Hz()
    );

    // ST7735s init needs the same CS/DC pins as `push_frame`, and we
    // can't share ownership — so initialize through locals and move
    // them into `DISPLAY_CTX` afterwards.
    let mut cs = cs;
    let mut dc = dc;
    st7735::init(&mut spi_bus, &mut cs, &mut dc, &mut rst, &mut timer);
    defmt::info!("demo_host: ST7735s init complete");

    let dma = pac.DMA.split(&mut pac.RESETS);

    let display_ctx = Rc::new(RefCell::new(DisplayCtx {
        spi: Some(spi_bus),
        ch: Some(dma.ch0),
        cs,
        dc,
        fps: 0,
    }));

    // One instantiation for either engine: `Instance` honours whichever
    // one this image was built with, and nothing below this line knows
    // which. sf-nano-core infers `push_frame`'s signature from the
    // module's import declaration, so the handler binds by name.
    defmt::info!("demo_host: instantiating...");
    let imports = [Import::func(
        "env",
        "push_frame",
        push_frame_handler(Rc::clone(&display_ctx)),
    )];
    let engine = lib::config::engine();
    {
        let cfg = engine.config();
        defmt::info!(
            "engine: code_arena={} wasm_max_pages={} wasm_stack={}",
            cfg.get_code_arena_bytes(),
            cfg.get_wasm_memory_max_pages(),
            cfg.get_wasm_stack_bytes()
        );
    }
    let mut instance = match Instance::new(&engine, WASM_DEMO, &imports) {
        Ok(inst) => inst,
        Err(e) => {
            defmt::error!("instantiate failed: {=str}", e.message());
            halt();
        }
    };

    // Resolve `run` once. The render loop then calls it without searching
    // the export list by name or allocating for its (empty) result list
    // on every frame.
    let run = match instance.get_func("run") {
        Some(func) => func,
        None => {
            defmt::error!("module does not export `run`");
            halt();
        }
    };

    defmt::info!("demo_host: instance created; entering render loop");

    // Per-frame host loop: invoke `run` once per frame, bracket with
    // the rp235x timer to measure. We used to use `DWT::cycle_count`,
    // but that only runs while the debug unit is enabled (i.e. while
    // probe-rs is attached) — disconnecting the probe would freeze
    // DWT and the fps overlay would drop to 0 even though rendering
    // kept going. `timer.get_counter()` is the always-on peripheral
    // counter, so it reports accurate fps with or without a probe.
    let mut frame: u32 = 0;
    let mut accum_us: u64 = 0;
    let mut samples: u32 = 0;
    let mut last_log = timer.get_counter();

    loop {
        let t0 = timer.get_counter();
        if let Err(e) = instance.call(&run, &[Value::I32(frame as i32)], &mut []) {
            defmt::error!("invoke run failed: {=str}", e.message());
            halt();
        }
        let t1 = timer.get_counter();
        let frame_us = (t1 - t0).to_micros();
        accum_us = accum_us.saturating_add(frame_us);
        samples += 1;

        let now = timer.get_counter();
        if (now - last_log).to_micros() >= 1_000_000 {
            let n = samples.max(1) as u64;
            let avg_us = accum_us / n;
            let fps = if avg_us > 0 {
                (1_000_000 / avg_us) as u32
            } else {
                0
            };
            display_ctx.borrow_mut().fps = fps;
            defmt::info!(
                "frame {}: {} fps  |  invoke={} us  (n={})",
                frame,
                fps,
                avg_us,
                samples
            );
            accum_us = 0;
            samples = 0;
            last_log = now;
        }

        frame = frame.wrapping_add(1);
    }
}

fn halt() -> ! {
    loop {
        lib::arch::wfi();
    }
}

#[unsafe(link_section = ".bi_entries")]
#[used]
pub static PICOTOOL_ENTRIES: [hal::binary_info::EntryAddr; 4] = [
    hal::binary_info::rp_cargo_bin_name!(),
    hal::binary_info::rp_cargo_version!(),
    hal::binary_info::rp_program_description!(c"sf-nano-pico2 Wasm demo host"),
    hal::binary_info::rp_program_build_attribute!(),
];
