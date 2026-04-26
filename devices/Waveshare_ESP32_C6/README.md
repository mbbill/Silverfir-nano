# Waveshare ESP32-C6 LCD 1.47 Rust Firmware

`devices/Waveshare_ESP32_C6` is a standalone Rust `no_std` bring-up crate for
the Waveshare ESP32-C6-LCD-1.47 board. The current firmware initializes the
ST7789 LCD over SPI DMA, embeds a `wasm32-unknown-unknown` Mandelbrot guest,
and runs it through `sf-nano-core`'s RV32 JIT. A native Mandelbrot mode is also
available for display-path testing.

The crate is isolated with its own `[workspace]` stub, matching the Pico2
device pattern so embedded dependencies do not enter the repository root
workspace.

## Hardware

Tested board:

- Waveshare ESP32-C6-LCD-1.47
- ESP32-C6 main CPU, `riscv32imac-unknown-none-elf`
- ST7789 LCD, 320x172 landscape, RGB565
- USB Serial/JTAG over the board USB-C connector

LCD pin map:

| LCD signal | ESP32-C6 GPIO |
| --- | ---: |
| MOSI | GPIO6 |
| SCLK | GPIO7 |
| CS | GPIO14 |
| DC | GPIO15 |
| RST | GPIO21 |
| BL | GPIO22 |

Other board pins captured in `src/board.rs`:

| Signal | ESP32-C6 GPIO |
| --- | ---: |
| RGB LED | GPIO8 |
| TF card CS | GPIO4 |
| TF card MISO | GPIO5 |

## Toolchain

Install once:

```powershell
rustup target add riscv32imac-unknown-none-elf
rustup component add rust-src
cargo install cargo-espflash espflash --locked
```

Or run:

```powershell
.\scripts\setup-rust.cmd
```

The tools installed during bring-up are in the normal Cargo bin directory:

```text
C:\Users\mbbill\.cargo\bin\cargo-espflash.exe
C:\Users\mbbill\.cargo\bin\espflash.exe
```

## Build

Run from this directory:

```powershell
.\scripts\build.cmd
.\scripts\build.cmd --release
```

The default build is the Wasm/JIT Mandelbrot firmware. To build the native
Mandelbrot firmware:

```powershell
.\scripts\build.cmd --release --no-default-features --features mode-native,demo-mandelbrot
```

`.cargo/config.toml` sets the default target to
`riscv32imac-unknown-none-elf`.

## Flash And Monitor

The connected board previously enumerated as `COM4` and reports 8 MB flash.

```powershell
.\scripts\flash.cmd COM4
.\scripts\flash.cmd COM4 --release
```

Flash the native Mandelbrot test firmware:

```powershell
.\scripts\flash.cmd COM4 -NoMonitor --release --no-default-features --features mode-native,demo-mandelbrot
```

The flash script uses:

```text
cargo espflash flash --chip esp32c6 --port COM4 --flash-size 8mb --monitor
```

To monitor without flashing:

```powershell
.\scripts\monitor.cmd COM4
```

## Source Layout

```text
src/main.rs                - firmware entry point, clocks, GPIO, SPI DMA, render loops
src/board.rs               - Waveshare board pin map and LCD geometry
src/config.rs              - sf-nano-core runtime memory limits
src/arch.rs                - RISC-V barriers for JIT code visibility
src/heap.rs                - Pico2-style 320 KiB fixed global heap
src/os_shim.rs             - bare-metal executable-memory hooks for sf-nano-core
src/display.rs             - minimal ST7789 init and RGB565 blitter
src/framebuffer.rs         - embedded-graphics draw target for overlays
src/overlay.rs             - Pico2-style FPS and sf-nano text overlays
src/kernels/mandelbrot.rs  - Q16.16 Mandelbrot kernel shared with wasm-demo
wasm-demo/                 - nested no_std Wasm guest crate
```

The Mandelbrot framebuffer is `160x128`, matching the Pico2 benchmark size so
the ESP32-C6 numbers are directly comparable. It is centered on the `320x172`
LCD at `(80,22)`. The LCD is configured with MADCTL `0x60`, which rotates the
panel into landscape mode and moves the ST7789 34-pixel glass offset to Y.

## Memory Model

The firmware mirrors Pico2's heap setup:

- 320 KiB `embedded-alloc` fixed global heap
- 64 KiB executable JIT code arena in ESP `.rwtext`
- 3-page maximum Wasm linear memory
- 32 KiB Wasm invoke stack

The Wasm guest's full-screen framebuffer lives inside Wasm linear memory. The
host stamps the overlays into that memory slice, then sends it directly to the
LCD.

## Current Performance

The display path uses the ESP32-C6 SPI DMA bus with a 32,736-byte TX staging
buffer and an 80 MHz SPI clock. A Pico2-size RGB565 frame is 40,960 bytes, so
the measured push time is close to the raw SPI wire time.

Native Mandelbrot serial log:

```text
native mandelbrot: 160x128 centered at 80,22
native mandelbrot frame 359: 59 fps render=12446us push=4313us total=16759us n=60
```

Wasm/JIT Mandelbrot serial log:

```text
wasm module: 1224 bytes
wasm mandelbrot: 160x128 offset=16448 memory=65536 centered at 80,22
wasm mandelbrot frame 119: 14 fps invoke=64397us push=4629us total=69026us n=15
```

DMA removes most display-driver overhead, but the current HAL bus write is
still synchronous. Native FPS is now limited mostly by Mandelbrot render time;
Wasm FPS is limited mostly by JIT invoke time. Pico2's RV32 reference numbers
for the same Mandelbrot size are 44 fps native and 13 fps Wasm/JIT.
