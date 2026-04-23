//! Wasm Mandelbrot demo — same kernel source as `mandelbrot_native`,
//! compiled to `wasm32-unknown-unknown` and JIT-executed by
//! sf-nano-core on the device. The headline number is its fps
//! divided by the native binary's fps: how close is sf-nano-core's
//! M33 JIT lowering to native-compiled Rust on the same CPU?
//!
//! Architecture: the Wasm module owns rendering into its own linear
//! memory, then calls the host-imported `env.push_frame` to present.
//! The host `push_frame` handler DMAs directly from the Wasm linear
//! memory (no 40 KiB duplicate framebuffer) via a small custom
//! `ReadTarget` impl that wraps a raw (ptr, len) pair. The unsafe
//! invariant is held by the handler's synchronous DMA-wait: nothing
//! else touches the Wasm memory while DMA is running.
//!
//! `cargo run --bin mandelbrot_wasm --release`

#![no_std]
#![no_main]

extern crate alloc;

use defmt_rtt as _;
use panic_probe as _;
use rp235x_hal as hal;

use embedded_hal::{digital::OutputPin, spi::MODE_0, spi::SpiBus};
use hal::{
    clocks::Clock,
    dma::{single_buffer, DMAExt, ReadTarget},
    fugit::RateExtU32,
};

use sf_nano_core::{HostFn, Import, Instance, Value, WasmError};
use sf_nano_pico2 as lib;

use lib::board::{CsPin, DcPin, DisplayCh, DisplaySpi, XTAL_FREQ_HZ};
use lib::display::{self, st7735};

#[unsafe(link_section = ".start_block")]
#[used]
pub static IMAGE_DEF: hal::block::ImageDef = hal::block::ImageDef::secure_exe();

/// The Wasm module compiled by `build.rs` from `wasm-kernel/`.
const WASM_MANDELBROT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mandelbrot.wasm"));

/// Peripherals owned by the `push_frame` host function. Populated once
/// in `main` before module instantiation, then `take()`-shuffled on
/// every frame — same pattern we use in `lcd_demo` and
/// `mandelbrot_native` so the DMA can take the SPI+channel by value
/// and hand them back via `wait()`.
struct DisplayCtx {
    spi: Option<DisplaySpi>,
    ch: Option<DisplayCh>,
    cs: CsPin,
    dc: DcPin,
}

/// HostFn can't capture — all state lives here, poked by `push_frame`
/// via `unsafe { addr_of_mut!(...) }`. Single-threaded firmware, so
/// no locking needed; the discipline is that `push_frame` is the only
/// caller after `main` finishes initialization.
static mut DISPLAY_CTX: Option<DisplayCtx> = None;

/// Latest-computed fps, updated by the main loop after each invoke.
/// `push_frame` reads this and stamps the overlay onto the framebuffer
/// inside Wasm linear memory just before the DMA transfer, so the fps
/// readout lives on-panel without any fps-rendering code on the Wasm
/// side — keeps the kernel focused on Mandelbrot math.
static mut DISPLAYED_FPS: u32 = 0;

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

fn push_frame(
    caller: &mut sf_nano_core::Caller,
    args: &[Value],
    _returns: &mut [Value],
) -> Result<(), WasmError> {
    let offset = match args.first() {
        Some(Value::I32(v)) => *v,
        _ => return Err(WasmError::invalid("push_frame: expected i32 offset")),
    };
    let len = match args.get(1) {
        Some(Value::I32(v)) => *v,
        _ => return Err(WasmError::invalid("push_frame: expected i32 len")),
    };
    if offset < 0 || len < 0 {
        return Err(WasmError::invalid("push_frame: negative offset or len"));
    }
    let offset = offset as usize;
    let len_u = len as usize;

    let mem = caller
        .memory_mut()
        .ok_or_else(|| WasmError::invalid("push_frame: wasm module has no memory"))?;
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
    let fps = unsafe { *core::ptr::addr_of!(DISPLAYED_FPS) };
    display::stamp_fps_overlay(&mut mem[offset..end], (2, 2), fps);
    display::stamp_sf_nano_overlay(&mut mem[offset..end], (112, 2));
    let src_ptr = unsafe { mem.as_ptr().add(offset) };

    // SAFETY: single-threaded firmware, main is the only setter and
    // this host function is the only consumer after main.
    let ctx = unsafe { (*core::ptr::addr_of_mut!(DISPLAY_CTX)).as_mut() }
        .ok_or_else(|| WasmError::internal("DISPLAY_CTX not initialized"))?;

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
    cortex_m::asm::delay(100);
    ctx.cs.set_high().ok();

    ctx.spi = Some(spi_back);
    ctx.ch = Some(ch_back);
    Ok(())
}

#[hal::entry]
fn main() -> ! {
    lib::init();

    let mut pac = hal::pac::Peripherals::take().unwrap();
    let mut core_pac = cortex_m::Peripherals::take().unwrap();
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

    core_pac.DCB.enable_trace();
    core_pac.DWT.enable_cycle_counter();

    let sio = hal::Sio::new(pac.SIO);
    let pins = hal::gpio::Pins::new(
        pac.IO_BANK0,
        pac.PADS_BANK0,
        sio.gpio_bank0,
        &mut pac.RESETS,
    );

    defmt::info!("mandelbrot_wasm: clocks up, sys={} Hz", sys_hz);
    defmt::info!("mandelbrot_wasm: wasm module is {} bytes", WASM_MANDELBROT.len());

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
        40u32.MHz(),
        MODE_0,
    );
    defmt::info!("mandelbrot_wasm: SPI1 @ 40 MHz");

    // ST7735s init needs the same CS/DC pins as `push_frame`, and we
    // can't share ownership — so initialize through locals and move
    // them into `DISPLAY_CTX` afterwards.
    let mut cs = cs;
    let mut dc = dc;
    st7735::init(&mut spi_bus, &mut cs, &mut dc, &mut rst, &mut timer);
    defmt::info!("mandelbrot_wasm: ST7735s init complete");

    let dma = pac.DMA.split(&mut pac.RESETS);

    // Stash peripherals for `push_frame`.
    unsafe {
        *core::ptr::addr_of_mut!(DISPLAY_CTX) = Some(DisplayCtx {
            spi: Some(spi_bus),
            ch: Some(dma.ch0),
            cs,
            dc,
        });
    }

    // Instantiate the Wasm module. sf-nano-core infers `push_frame`'s
    // signature from the module's import declaration, so we just bind
    // the handler by name.
    let imports = [Import::func("env", "push_frame", push_frame as HostFn)];

    {
        let rc = sf_nano_core::runtime_config();
        defmt::info!(
            "runtime_config: code_arena={} wasm_max_pages={} wasm_stack={}",
            rc.code_arena_bytes,
            rc.wasm_memory_max_pages,
            rc.wasm_stack_bytes
        );
    }
    defmt::info!("mandelbrot_wasm: instantiating...");
    let mut instance = match Instance::new(WASM_MANDELBROT, &imports) {
        Ok(inst) => inst,
        Err(e) => {
            defmt::error!("instantiate failed: {=str}", e.message());
            halt();
        }
    };
    defmt::info!("mandelbrot_wasm: instance created; entering render loop");

    // Per-frame host loop: invoke `run` once per frame, bracket with
    // DWT to measure. The running fps is published into `DISPLAYED_FPS`
    // which `push_frame` reads to stamp the overlay onto the outgoing
    // framebuffer bytes.
    let mut frame: u32 = 0;
    let mut accum_us: u64 = 0;
    let mut samples: u32 = 0;
    let mut last_log = timer.get_counter();

    loop {
        let t0 = cortex_m::peripheral::DWT::cycle_count();
        if let Err(e) = instance.invoke("run", &[Value::I32(frame as i32)]) {
            defmt::error!("invoke run failed: {=str}", e.message());
            halt();
        }
        let t1 = cortex_m::peripheral::DWT::cycle_count();
        let cycles = t1.wrapping_sub(t0);
        let frame_us = cycles as u64 / (sys_hz as u64 / 1_000_000);
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
            unsafe { *core::ptr::addr_of_mut!(DISPLAYED_FPS) = fps };
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
        cortex_m::asm::wfi();
    }
}

#[unsafe(link_section = ".bi_entries")]
#[used]
pub static PICOTOOL_ENTRIES: [hal::binary_info::EntryAddr; 4] = [
    hal::binary_info::rp_cargo_bin_name!(),
    hal::binary_info::rp_cargo_version!(),
    hal::binary_info::rp_program_description!(c"sf-nano-pico2 Wasm Mandelbrot (JIT)"),
    hal::binary_info::rp_program_build_attribute!(),
];
