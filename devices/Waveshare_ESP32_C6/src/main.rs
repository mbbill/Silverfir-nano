#![no_std]
#![no_main]

extern crate alloc;

#[cfg(all(feature = "mode-native", feature = "mode-wasm"))]
compile_error!("select exactly one of mode-native or mode-wasm");
#[cfg(not(any(feature = "mode-native", feature = "mode-wasm")))]
compile_error!("select one of mode-native or mode-wasm");
#[cfg(all(feature = "demo-mandelbrot", feature = "demo-cube"))]
compile_error!("select exactly one demo feature");
#[cfg(not(any(feature = "demo-mandelbrot", feature = "demo-cube")))]
compile_error!("select one demo feature");

#[cfg(feature = "mode-wasm")]
mod arch;
mod board;
#[cfg(feature = "mode-wasm")]
mod config;
mod display;
mod framebuffer;
mod heap;
mod kernels;
#[cfg(feature = "mode-wasm")]
mod os_shim;
mod overlay;

#[cfg(feature = "mode-native")]
use alloc::{boxed::Box, vec};

use esp_backtrace as _;
use esp_hal::{
    clock::CpuClock,
    dma::{DmaRxBuf, DmaTxBuf},
    gpio::{Level, Output, OutputConfig},
    main,
    peripherals::Peripherals,
    spi::{
        master::{Config as SpiConfig, Spi},
        Mode,
    },
    time::{Duration, Instant, Rate},
    usb_serial_jtag::UsbSerialJtag,
    Blocking,
};
use esp_println::println;
#[cfg(feature = "mode-wasm")]
use sf_nano_core::{Import, Instance, Value};

#[cfg(feature = "demo-cube")]
pub(crate) use kernels::cube as demo;
#[cfg(feature = "demo-mandelbrot")]
pub(crate) use kernels::mandelbrot as demo;

use display::Display;

esp_bootloader_esp_idf::esp_app_desc!();

const DISPLAY_DMA_TX_BYTES: usize = 32_736;
const DISPLAY_SPI_MHZ: u32 = 80;

#[cfg(feature = "demo-cube")]
const DEMO_NAME: &str = "cube";
#[cfg(feature = "demo-mandelbrot")]
const DEMO_NAME: &str = "mandelbrot";

#[cfg(feature = "mode-wasm")]
const WASM_DEMO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/demo.wasm"));

struct SerialConsole<'d> {
    _usb: UsbSerialJtag<'d, Blocking>,
}

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);
    heap::init();

    #[cfg(feature = "mode-wasm")]
    #[cfg(feature = "mode-native")]
    println!("waveshare-esp32-c6: native {} + spi dma", DEMO_NAME);
    #[cfg(feature = "mode-wasm")]
    println!("waveshare-esp32-c6: wasm {} + spi dma", DEMO_NAME);

    println!("heap: {} bytes", heap::HEAP_SIZE);
    println!("display spi clock: {} MHz", DISPLAY_SPI_MHZ);
    println!("display dma tx buffer: {} bytes", DISPLAY_DMA_TX_BYTES);
    #[cfg(feature = "mode-wasm")]
    println!(
        "sf-nano: code_arena={} wasm_max_pages={} wasm_stack={}",
        crate::config::CODE_ARENA_BYTES,
        crate::config::WASM_MEMORY_MAX_PAGES,
        crate::config::WASM_STACK_BYTES
    );
    #[cfg(feature = "mode-wasm")]
    println!("wasm module: {} bytes", WASM_DEMO.len());
    println!(
        "lcd: {}x{}, mosi=gpio{}, sclk=gpio{}, cs=gpio{}, dc=gpio{}, rst=gpio{}, bl=gpio{}",
        board::LCD_WIDTH,
        board::LCD_HEIGHT,
        board::LCD_SPI_MOSI_GPIO,
        board::LCD_SPI_SCLK_GPIO,
        board::LCD_CS_GPIO,
        board::LCD_DC_GPIO,
        board::LCD_RST_GPIO,
        board::LCD_BL_GPIO
    );
    println!(
        "board: rgb_led=gpio{}, tf_cs=gpio{}, tf_miso=gpio{}",
        board::RGB_LED_GPIO,
        board::TF_CARD_CS_GPIO,
        board::TF_CARD_MISO_GPIO
    );

    let (mut display, _serial_console) = init_io(peripherals);
    display.init();
    println!("display initialized");
    display.clear(0x0000);

    #[cfg(feature = "mode-native")]
    run_native(display);
    #[cfg(feature = "mode-wasm")]
    run_wasm(display);
}

fn init_io(
    peripherals: Peripherals,
) -> (
    Display<impl embedded_hal::spi::SpiBus<u8>>,
    SerialConsole<'static>,
) {
    let serial_console = SerialConsole {
        _usb: UsbSerialJtag::new(peripherals.USB_DEVICE),
    };

    let (rx_buffer, rx_descriptors, tx_buffer, tx_descriptors) =
        esp_hal::dma_buffers!(4, DISPLAY_DMA_TX_BYTES);
    let dma_rx_buf = DmaRxBuf::new(rx_descriptors, rx_buffer).unwrap();
    let dma_tx_buf = DmaTxBuf::new(tx_descriptors, tx_buffer).unwrap();

    let spi = Spi::new(
        peripherals.SPI2,
        SpiConfig::default()
            .with_frequency(Rate::from_mhz(DISPLAY_SPI_MHZ))
            .with_mode(Mode::_0),
    )
    .unwrap()
    .with_sck(peripherals.GPIO7)
    .with_mosi(peripherals.GPIO6)
    .with_dma(peripherals.DMA_CH0)
    .with_buffers(dma_rx_buf, dma_tx_buf);

    let cs = Output::new(peripherals.GPIO14, Level::High, OutputConfig::default());
    let dc = Output::new(peripherals.GPIO15, Level::High, OutputConfig::default());
    let rst = Output::new(peripherals.GPIO21, Level::High, OutputConfig::default());
    let bl = Output::new(peripherals.GPIO22, Level::High, OutputConfig::default());

    (Display::new(spi, cs, dc, rst, bl), serial_console)
}

#[cfg(feature = "mode-native")]
fn run_native<SPI>(mut display: Display<SPI>) -> !
where
    SPI: embedded_hal::spi::SpiBus<u8>,
{
    let framebuffer = Box::leak(vec![0u8; demo::FB_BYTES].into_boxed_slice());

    let x = ((board::LCD_WIDTH - demo::WIDTH) / 2) as u16;
    let y = ((board::LCD_HEIGHT - demo::HEIGHT) / 2) as u16;

    println!(
        "native {}: {}x{} centered at {},{}",
        DEMO_NAME,
        demo::WIDTH,
        demo::HEIGHT,
        x,
        y
    );

    let mut frame = 0u32;
    let mut fps_samples = 0u32;
    let mut total_accumulator_us = 0u64;
    let mut render_accumulator_us = 0u64;
    let mut push_accumulator_us = 0u64;
    let mut last_log = Instant::now();
    let mut displayed_fps = 0u32;

    const FPS_ORIGIN: (i32, i32) = (2, 2);
    const LABEL_ORIGIN: (i32, i32) = (demo::WIDTH as i32 - 48, 2);

    loop {
        let frame_start = Instant::now();
        demo::render(framebuffer, frame);
        overlay::stamp_fps_overlay(framebuffer, FPS_ORIGIN, displayed_fps);
        overlay::stamp_sf_nano_overlay(framebuffer, LABEL_ORIGIN);
        let render_done = Instant::now();

        display.draw_rgb565_be(x, y, demo::WIDTH as u16, demo::HEIGHT as u16, framebuffer);
        let frame_done = Instant::now();

        render_accumulator_us += (render_done - frame_start).as_micros();
        push_accumulator_us += (frame_done - render_done).as_micros();
        total_accumulator_us += (frame_done - frame_start).as_micros();
        fps_samples += 1;

        if last_log.elapsed() >= Duration::from_secs(1) {
            let n = fps_samples.max(1) as u64;
            let avg_total = total_accumulator_us / n;
            let avg_render = render_accumulator_us / n;
            let avg_push = push_accumulator_us / n;
            displayed_fps = if avg_total > 0 {
                (1_000_000 / avg_total) as u32
            } else {
                0
            };
            println!(
                "native {} frame {}: {} fps render={}us push={}us total={}us n={}",
                DEMO_NAME, frame, displayed_fps, avg_render, avg_push, avg_total, fps_samples
            );

            fps_samples = 0;
            total_accumulator_us = 0;
            render_accumulator_us = 0;
            push_accumulator_us = 0;
            last_log = Instant::now();
        }

        frame = frame.wrapping_add(1);
    }
}

#[cfg(feature = "mode-wasm")]
fn run_wasm<SPI>(mut display: Display<SPI>) -> !
where
    SPI: embedded_hal::spi::SpiBus<u8>,
{
    let imports: [Import; 0] = [];
    let engine = crate::config::engine();
    let mut instance = match Instance::new(&engine, WASM_DEMO, &imports) {
        Ok(instance) => instance,
        Err(error) => {
            println!("instantiate failed: {}", error.message());
            fatal(&mut display, 0xF800);
        }
    };

    let framebuffer_offset = match instance.invoke("framebuffer_offset", &[]) {
        Ok(values) => match values.first() {
            Some(Value::I32(offset)) if *offset >= 0 => *offset as usize,
            _ => {
                println!("framebuffer_offset returned unexpected value");
                fatal(&mut display, 0xF800);
            }
        },
        Err(error) => {
            println!("framebuffer_offset failed: {}", error.message());
            fatal(&mut display, 0xF800);
        }
    };
    let framebuffer_end = match framebuffer_offset.checked_add(demo::FB_BYTES) {
        Some(end) => end,
        None => {
            println!("framebuffer range overflow");
            fatal(&mut display, 0xF800);
        }
    };

    let memory_len = instance.memory().map(|mem| mem.len()).unwrap_or(0);
    if framebuffer_end > memory_len {
        println!(
            "framebuffer out of wasm memory: offset={} end={} memory={}",
            framebuffer_offset, framebuffer_end, memory_len
        );
        fatal(&mut display, 0xF800);
    }

    let x = ((board::LCD_WIDTH - demo::WIDTH) / 2) as u16;
    let y = ((board::LCD_HEIGHT - demo::HEIGHT) / 2) as u16;

    println!(
        "wasm {}: {}x{} offset={} memory={} centered at {},{}",
        DEMO_NAME,
        demo::WIDTH,
        demo::HEIGHT,
        framebuffer_offset,
        memory_len,
        x,
        y
    );

    let mut frame = 0u32;
    let mut fps_samples = 0u32;
    let mut total_accumulator_us = 0u64;
    let mut invoke_accumulator_us = 0u64;
    let mut push_accumulator_us = 0u64;
    let mut last_log = Instant::now();
    let mut displayed_fps = 0u32;

    const FPS_ORIGIN: (i32, i32) = (2, 2);
    const LABEL_ORIGIN: (i32, i32) = (demo::WIDTH as i32 - 48, 2);

    loop {
        let frame_start = Instant::now();
        if let Err(error) = instance.invoke("run", &[Value::I32(frame as i32)]) {
            println!("invoke run failed: {}", error.message());
            fatal(&mut display, 0xF800);
        }
        let invoke_done = Instant::now();

        {
            let mem = match instance.memory_mut() {
                Some(mem) => mem,
                None => {
                    println!("wasm module has no memory");
                    fatal(&mut display, 0xF800);
                }
            };
            let fb = &mut mem[framebuffer_offset..framebuffer_end];
            overlay::stamp_fps_overlay(fb, FPS_ORIGIN, displayed_fps);
            overlay::stamp_sf_nano_overlay(fb, LABEL_ORIGIN);
            display.draw_rgb565_be(x, y, demo::WIDTH as u16, demo::HEIGHT as u16, fb);
        }
        let frame_done = Instant::now();

        invoke_accumulator_us += (invoke_done - frame_start).as_micros();
        push_accumulator_us += (frame_done - invoke_done).as_micros();
        total_accumulator_us += (frame_done - frame_start).as_micros();
        fps_samples += 1;

        if last_log.elapsed() >= Duration::from_secs(1) {
            let n = fps_samples.max(1) as u64;
            let avg_total = total_accumulator_us / n;
            let avg_invoke = invoke_accumulator_us / n;
            let avg_push = push_accumulator_us / n;
            displayed_fps = if avg_total > 0 {
                (1_000_000 / avg_total) as u32
            } else {
                0
            };
            println!(
                "wasm {} frame {}: {} fps invoke={}us push={}us total={}us n={}",
                DEMO_NAME, frame, displayed_fps, avg_invoke, avg_push, avg_total, fps_samples
            );

            fps_samples = 0;
            total_accumulator_us = 0;
            invoke_accumulator_us = 0;
            push_accumulator_us = 0;
            last_log = Instant::now();
        }

        frame = frame.wrapping_add(1);
    }
}

#[cfg(feature = "mode-wasm")]
fn fatal<SPI>(display: &mut Display<SPI>, color: u16) -> !
where
    SPI: embedded_hal::spi::SpiBus<u8>,
{
    display.clear(color);
    loop {}
}
