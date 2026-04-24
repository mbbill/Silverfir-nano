# Silverfir-nano on Raspberry Pi Pico 2

`devices/pico2` is the real-hardware bring-up target for running the
Silverfir-nano WebAssembly JIT on the RP2350 Cortex-M33 in a `no_std`
firmware image. It builds a Wasm kernel, verifies/instantiates it through
`sf-nano-core`, JITs it to Thumb-2 in SRAM, and drives a 160x128 ST7735s LCD
from the Wasm-rendered framebuffer.

This is not an interpreter demo. The guest program stays as a `.wasm` artifact
until boot time, then the Pico 2 emits and executes native Thumb-2 code on the
device.

## Demo Results

Measured on a Pico 2 / Pico 2 W class RP2350 board with the Waveshare
Pico-LCD-1.8 shield, release firmware, 150 MHz system clock, and 40 MHz SPI.
The FPS number includes rendering, host import overhead, overlay stamping, and
DMA transfer to the panel.

| Demo | What it stresses | Result |
| --- | --- | --- |
| [Mandelbrot](https://youtu.be/g3rskqAUEYo) | Q16.16 fixed-point fractal, hot `i64.mul` lowering, symmetry copy, host `push_frame` import | 19 fps |
| [Cube](https://youtu.be/ULs2tGhGPfs) | all-`i32` geometry, Q15 trig LUT, projection, triangle rasterization, face culling/shading | 66 fps |
| Wasm3 Mandelbrot comparison | same Pico 2 display path running through Wasm3 instead of Silverfir-nano's JIT | 3 fps |
| `lcd_demo` | native display pipeline ceiling: CPU gradient fill + overlay + 40 MHz DMA push | about 74 fps |

### Mandelbrot

<video src="artifacts/sf-nano-mandelbrot-readme.mp4" controls width="640"></video>

### Cube

<video src="artifacts/sf-nano-cube-readme.mp4" controls width="640"></video>

The `mandelbrot_wasm` binary is the generic Wasm display host. The active Wasm
demo is selected in `wasm-kernel/src/lib.rs`:

```rust
use mandelbrot as demo;
// use cube as demo;
```

Swap those two lines to run the cube demo through the same host binary.

## Prebuilt Firmware

The `artifacts/` directory also includes flashable RP2350 UF2 binaries:

| File | Purpose |
| --- | --- |
| [`sf-nano-pico2-rp2350.uf2`](artifacts/sf-nano-pico2-rp2350.uf2) | Silverfir-nano Pico 2 firmware. Use this to try the JIT demo without building locally. |
| [`wasm3-pico2-rp2350.uf2`](artifacts/wasm3-pico2-rp2350.uf2) | Wasm3 Pico 2 comparison firmware; the Mandelbrot demo runs at 3 fps. |

To flash one, hold BOOTSEL while plugging in the Pico 2, then copy the UF2
file onto the mounted `RPI-RP2` drive. The board resets into the new firmware
after the copy completes.

## Hardware

Tested setup:

- Raspberry Pi Pico 2 or Pico 2 W target board, RP2350, Cortex-M33.
- SWD probe, usually a Pico 1 flashed with Raspberry Pi `debugprobe` firmware.
- Waveshare Pico-LCD-1.8 shield, 160x128 ST7735s, RGB565.

The Pico 2 W onboard LED is behind the CYW43 chip and is intentionally not part
of this crate. This target is compute/display bring-up only; Wi-Fi should live
in a future sibling crate.

LCD pin map:

| LCD signal | Pico GPIO | Firmware role |
| --- | ---: | --- |
| SCK | GP10 | SPI1 SCLK |
| MOSI / DIN | GP11 | SPI1 TX |
| CS | GP9 | software-controlled output |
| DC | GP8 | software-controlled output |
| RST | GP12 | software-controlled output |
| BL | GP13 | backlight, high = on |
| MISO | GP28 | dummy SPI RX pin, unconnected on the shield |

The HAL SPI type requires TX, RX, and SCK pins even for write-only display
traffic, so GP28 is used only to satisfy the type.

## Toolchain

Install once:

```bash
rustup target add thumbv8m.main-none-eabihf
rustup target add wasm32-unknown-unknown
rustup component add llvm-tools
cargo install probe-rs-tools --locked
cargo install cargo-binutils --locked
cargo install flip-link --locked
```

Optional, for UF2 generation:

```bash
cargo install elf2uf2-rs
# or install Raspberry Pi picotool from https://github.com/raspberrypi/picotool
```

Verify the probe and target:

```bash
probe-rs list
probe-rs info --chip RP235x --protocol swd
```

The correct `probe-rs` chip name is `RP235x`.

## Build And Run

Run from this directory:

```bash
cargo run --bin heartbeat
cargo run --bin lcd_demo --release
cargo run --bin mandelbrot_native --release
cargo run --bin mandelbrot_wasm --release
```

`Cargo.toml` sets `default-run = "mandelbrot_wasm"`, so this is equivalent:

```bash
cargo run --release
```

The runner in `.cargo/config.toml` flashes the ELF with:

```text
probe-rs run --chip RP235x --protocol swd
```

and streams `defmt` logs over RTT until you stop it.

To build a USB-BOOTSEL-flashable UF2:

```bash
./build-uf2.sh                 # default: mandelbrot_wasm
./build-uf2.sh heartbeat
./build-uf2.sh lcd_demo
./build-uf2.sh mandelbrot_native
```

The script builds the release ELF, converts it to RP2350 ARM secure UF2, and
verifies the UF2 family ID (`0xE48BFF59`). Use `picotool` when available, or
`elf2uf2-rs` as a fallback.

## Important Build Details

This crate is intentionally not a member of the repository workspace. It has
its own `[workspace]` stub and lockfile so embedded dependencies and target
configuration do not affect hosted builds.

The firmware target is:

```text
thumbv8m.main-none-eabihf
```

That is the hard-float Cortex-M33 target expected by `rp235x-hal`. This does
not mean the JIT emits floating-point code. On Cortex-M profiles, the available
FPU is single-precision only while WebAssembly requires `f64`, so the Thumb-M
JIT path is integer-only for now.

The `build.rs` file does two jobs:

- Copies `memory.x` into Cargo's output directory so `cortex-m-rt` can link the
  RP2350 flash/RAM layout.
- Builds `wasm-kernel/` as `wasm32-unknown-unknown --release`, then copies the
  resulting `sf_nano_pico2_wasm_kernel.wasm` to `OUT_DIR/kernel.wasm` for
  `include_bytes!`.

The Wasm build forces:

```text
-C link-arg=-zstack-size=16384
```

Without that, `wasm-ld` defaults to a 1 MiB Wasm call stack, which inflates the
module's initial linear memory beyond the Pico runtime quota.

## Runtime And Memory Model

The firmware calls `sf_nano_pico2::init()` at startup before any allocation or
JIT work. That installs:

- A 320 KiB `embedded-alloc` global heap for Rust allocation, Wasm module
  metadata, Wasm linear memory, and the per-invoke operand stack.
- An `sf-nano-core` `RuntimeConfig` with:
  - 128 KiB JIT code arena.
  - 3 Wasm memory pages max per linear memory (192 KiB).
  - 32 KiB Wasm operand/call stack per invoke.
- A single 128 KiB executable SRAM arena exposed through
  `sf_os_alloc_executable`.

RP2350 SRAM is executable without MPU permission changes in this firmware. The
write-finish hook still executes `dsb` + `isb` so newly emitted instructions
are visible to the Cortex-M33 fetch pipeline before the JIT jumps into them.
There is no instruction cache to flush on RP2350.

`flip-link` is used so native stack overflow faults instead of silently
corrupting the large heap/code-arena regions in `.bss`.

## JIT Path

The host binary embeds the Wasm kernel bytes and instantiates them with one
import:

```text
env.push_frame(i32 offset, i32 len)
```

Each frame:

1. The host invokes the Wasm export `run(frame)`.
2. JIT-compiled guest code renders into a static framebuffer in Wasm linear
   memory.
3. The guest calls `env.push_frame(offset, len)`.
4. The host validates the framebuffer range against the guest memory.
5. The host stamps the FPS and `sf-nano` overlays into that same memory.
6. A custom `ReadTarget` wrapper lets RP2350 DMA read directly from Wasm
   linear memory into SPI1.
7. The host waits for DMA completion before returning to the guest.

There is no duplicate 40 KiB host framebuffer for the Wasm path. The
synchronous DMA wait is what makes the raw pointer handed to the DMA engine
valid for the whole transfer.

## Display Pipeline

The panel is driven as a 160x128 RGB565 big-endian framebuffer:

```text
160 * 128 * 2 = 40,960 bytes
```

Per frame, the display side sends the ST7735s address window commands
(`CASET`, `RASET`, `RAMWR`) with blocking SPI writes, then keeps CS low while
DMA streams the framebuffer at 40 MHz over SPI1.

`mipidsi` is not used for the hot path. Its SPI device wrapper owns the bus in
a way that makes it awkward to recover the raw `Spi` object for DMA, so this
crate uses a small hand-rolled ST7735s init sequence and raw SPI/DMA calls.

Panel-specific constants copied from the Waveshare driver:

- MADCTL = `0x70` for landscape orientation.
- CASET = `0x0001..0x00A0`.
- RASET = `0x0002..0x0082`.

Do not "tighten" RASET to exactly 128 rows; this shield expects the `0x82`
bound and otherwise shows a blank or unstable image.

The RP2350 PL022 SPI path also needs conservative CS timing:

- Flush after every command/data SPI write before deasserting CS.
- After framebuffer DMA, call `flush()`, wait about 100 CPU cycles, then raise
  CS.

Without that guard the panel can occasionally shift a few pixels and snap back
on the next frame.
