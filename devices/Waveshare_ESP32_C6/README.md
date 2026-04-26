# Silverfir-nano on Waveshare ESP32-C6 LCD 1.47

`devices/Waveshare_ESP32_C6` is the real-hardware bring-up target for running
the Silverfir-nano WebAssembly JIT on the Waveshare ESP32-C6-LCD-1.47 board in
a Rust `no_std` firmware image.

It builds a small `wasm32-unknown-unknown` demo guest, JITs it through
`sf-nano-core` on the ESP32-C6 RV32 core, and presents the guest framebuffer on
the onboard 320x172 ST7789 LCD over SPI DMA.

This is not an interpreter demo. The guest program stays as a `.wasm` artifact
until boot time, then the firmware emits and executes native RV32 code.

## Demo Results

Measured on the Waveshare ESP32-C6-LCD-1.47 board, release firmware, 160 MHz
CPU clock, 80 MHz LCD SPI clock, and a 160x128 demo framebuffer centered on the
320x172 panel. FPS includes rendering, overlay stamping, and SPI DMA transfer
to the panel.

| Target / demo | What it stresses | Result |
| --- | --- | ---: |
| Wasm/JIT Cube | all-`i32` geometry, Q15 trig LUT, projection, triangle rasterization, face culling/shading | 96-103 fps |
| Native Cube | Same cube kernel compiled directly by rustc | 156-164 fps |

### Cube

https://github.com/user-attachments/assets/db58cd3f-d4f2-4f34-b6e5-30e568588505

[small MP4](assets/sf-nano-esp32-c6-cube-readme.mp4)

## Build And Run

Run from this directory:

```powershell
.\scripts\build.cmd --release
.\scripts\flash.cmd COM4 --release
```

The default demo is Wasm/JIT Mandelbrot. Select cube or native comparison
firmware with Cargo features:

```powershell
.\scripts\flash.cmd COM4 -NoMonitor --release --no-default-features --features mode-wasm,demo-cube
.\scripts\flash.cmd COM4 -NoMonitor --release --no-default-features --features mode-native,demo-cube
.\scripts\flash.cmd COM4 -NoMonitor --release --no-default-features --features mode-native,demo-mandelbrot
```

To monitor without flashing:

```powershell
.\scripts\monitor.cmd COM4
```

## Hardware

Tested setup:

- Waveshare ESP32-C6-LCD-1.47
- ESP32-C6, 160 MHz RV32IMAC core
- Onboard 320x172 ST7789 LCD, RGB565
- USB Serial/JTAG through the board USB-C connector

The connected board previously enumerated as `COM4` and reports 8 MB flash.

## Toolchain

Install once:

```powershell
rustup target add riscv32imac-unknown-none-elf
rustup target add wasm32-unknown-unknown
rustup component add rust-src
cargo install cargo-espflash espflash --locked
```

Or run:

```powershell
.\scripts\setup-rust.cmd
```

`.cargo/config.toml` sets the default firmware target to
`riscv32imac-unknown-none-elf`. The crate has its own `[workspace]` stub and
lockfile so ESP embedded dependencies stay isolated from the repository root
workspace.
