# `devices/pico2/` — development notes

Working notebook for the Pico 2 / Pico 2 W bring-up. Captures what we
learned while scaffolding milestone 1 so later sessions do not have to
rediscover the gotchas. Parent plan: `docs/RP2350_PICO2_BRINGUP_PLAN.md`.

## 1. Scope

- **Targets both Pico 2 and Pico 2 W boards.** This crate is the
  **compute-only** JIT bring-up harness: no Wi-Fi, no on-board LED (the
  LED is wired to CYW43 GPIO 0 on the W and GP25 on the non-W, so
  there is no board-agnostic blink).
- **Not a place for networked firmware.** When Wi-Fi is needed, add a
  sibling `devices/pico2w/` crate with `embassy-rp` + `cyw43` +
  `embassy-net`. Keep that off the milestone-1 critical path.
- **Not a workspace member.** `devices/pico2/Cargo.toml` carries its own
  `[workspace]` stub so embedded deps stay out of the hosted build graph
  at repo root. Same pattern as `sf-nano-bare-smoke/`.

## 2. Hardware setup

### Probe — Pico 1 flashed with debugprobe firmware

- Firmware source + UF2 releases: <https://github.com/raspberrypi/debugprobe>.
  Download **`debugprobe_on_pico.uf2`** (not `debugprobe.uf2`, which is
  for the official probe board).
- Pin assignments on the probe (from
  `include/board_pico_config.h` in that repo):

  | Function            | Probe GPIO | Physical pin | Notes                            |
  |---------------------|-----------:|-------------:|----------------------------------|
  | SWCLK               | GP2        | 4            | required                         |
  | SWDIO               | GP3        | 5            | required                         |
  | UART1 TX (to tgt RX)| GP4        | 6            | optional — RTT over SWD suffices |
  | UART1 RX (from tgt) | GP5        | 7            | optional                         |
  | GND                 | GND        | 8            | required                         |
  | RESET (to tgt nRST) | GP1        | 2            | skip — `probe-rs` resets via SWD |

- USB connected LED on the probe: GP25 (onboard LED on Pico 1). Lights
  when the probe enumerates.

### Target — Pico 2 W

- 3-pin **JST-SH 1.0 mm** debug connector on the bottom edge:
  SWCLK / GND / SWDIO. Either use a JST-SH cable or solder to the
  through-hole pads alongside.
- No `--connect-under-reset` option in our workflow: we never wired
  the probe's GP1 to the target's RUN pin. `probe-rs` resets over SWD.

### Windows USB

- Windows 11 picks up the probe as **Debugprobe on Pico (CMSIS-DAP)**
  with VID:PID `2e8a:000c` and auto-installs the WinUSB driver via
  the CMSIS-DAP v2 Microsoft OS 2.0 descriptor. No Zadig dance needed
  in our case. If `probe-rs list` ever shows nothing, Zadig → pick the
  CMSIS-DAP interface (**Interface 2**, not the CDC ones) → install
  WinUSB.

## 3. Toolchain prerequisites

Install once per machine:

```
rustup target add thumbv8m.main-none-eabihf
rustup component add llvm-tools
cargo install probe-rs-tools --locked
cargo install cargo-binutils --locked
cargo install flip-link --locked            # optional but recommended
```

Then verify:

```
probe-rs list                               # probe enumerates over USB
probe-rs info --chip RP235x --protocol swd  # SWD link reaches the target
```

`probe-rs info` should enumerate **both** M33 cores (two MemAPs, both
`PARTNO: Cortex-M33`, `VARIANT: 1`, `REVISION: 0`) and the `RP235x
CoreSight ROM` marker. The "Probing target via JTAG … could not be
selected" line is benign: `probe-rs` tries JTAG first and falls back to
SWD, which succeeds.

## 4. Build & run

From this directory:

```
cargo run                   # heartbeat firmware: build, flash, attach to RTT
cargo run --bin lcd_demo    # LCD demo: draws a test frame on the Waveshare shield
```

Equivalent to:

```
cargo build --bin <name>
probe-rs run --chip RP235x --protocol swd target/thumbv8m.main-none-eabihf/debug/<name>
```

The runner is wired in `.cargo/config.toml`. `cargo run` streams defmt
output until you Ctrl-C (both binaries stay in a heartbeat loop
forever).

Other useful commands:

```
cargo size --bin heartbeat -- -A    # section sizes (flash + RAM footprint)
cargo nm   --bin heartbeat          # symbol table (grep _SEGGER_RTT to verify RTT)
cargo objdump --bin heartbeat -- -d --no-show-raw-insn   # disassemble
```

### Binaries

- **`heartbeat`** (`src/bin/heartbeat.rs`) — minimal bring-up (M1).
  Prints a 1 Hz tick counter over RTT. Proves the toolchain, boot
  block, probe-rs flash and RTT paths all work end-to-end.
- **`lcd_demo`** (`src/bin/lcd_demo.rs`) — benchmarks the display
  pipeline. Animated gradient with an FPS overlay, DMA-driven push to
  the Waveshare Pico-LCD-1.8 at 40 MHz SPI. Baseline ≈ **74 fps /
  13.3 ms per frame** for CPU fill + DMA, SPI wire time alone being
  8.2 ms of that.
- **`mandelbrot_native`** (`src/bin/mandelbrot_native.rs`) — same
  Mandelbrot kernel as the Wasm demo, compiled natively. The ceiling
  the JIT is measured against — currently **18 fps / 55 ms per
  frame** on Q17.14 integer math.
- **`mandelbrot_wasm`** (`src/bin/mandelbrot_wasm.rs`) — **the
  headline demo.** Same kernel compiled to `wasm32-unknown-unknown`,
  JIT-executed by sf-nano-core. Currently **9 fps / 110 ms per
  frame**, ~55 % of native (or ~50 % compute-only, subtracting the
  shared 10 ms DMA push). `cargo run` default.

## 5. Key configuration decisions and why

### 5.1 Target: `thumbv8m.main-none-eabihf` (hard-float)

`rp235x-hal 0.4` only declares a `cortex-m` dep for the **hard-float**
triple. Trying to build on `thumbv8m.main-none-eabi` (soft-float)
fails deep in the HAL with `unresolved module cortex_m`. Hard-float is
the only supported configuration; do not fight it.

This is **independent of JIT codegen ABI**. The firmware host ABI
(hard-float) and the JIT-emitted code ABI (integer-only on thumbm —
see `sf-nano-core/build.rs` comment on `sf_fp_dp`) cross the host/JIT
boundary only via integer arguments in `sf_os_*` shims. They do not
have to match.

### 5.2 JIT codegen on M33: no FP at all

Wasm specifies f64. M-profile FPUs (FPv5-SP-D16 in M33) are
**single-precision only** — no double-precision M-profile FPU exists.
Conclusion for this target: JIT emits **zero FP code**. Integer-only
Wasm modules only. `sf-nano-core/build.rs` already leaves `sf_fp_dp`
off for `sf_arch_thumbm`; keep it that way.

### 5.3 `DEFMT_LOG=trace` is mandatory

**This is the single most non-obvious thing about this project.**

`defmt` does *compile-time* log-level filtering via the `DEFMT_LOG`
env var, read by its proc macros at build time. Without it set,
**every `defmt::info!` / `error!` / etc. compiles to a no-op**. The
firmware still runs, but the RTT buffer stays empty forever and
`probe-rs run` shows nothing.

Symptom: `probe-rs run` prints `Finished in 0.7s` after flashing and
then hangs silently. Diagnostic: `probe-rs read --chip RP235x
--protocol swd b32 0x2000002c 1` (the RTT control block's WrOff
field) returns `0`.

Fix: `DEFMT_LOG = "trace"` in `.cargo/config.toml` under `[env]`.
Keep it there permanently; tighten later if build noise becomes an
issue.

### 5.4 Boot block: `ImageDef::secure_exe()` in `.start_block`

The RP2350 Boot ROM requires an `IMAGE_DEF` block in the first 4 KiB
of flash to recognize an image. `rp235x-hal` provides the const via
`hal::block::ImageDef::secure_exe()`; `memory.x` puts `.start_block`
right after `.vector_table`. Do not remove either piece — without a
valid `IMAGE_DEF` the chip will not execute the image at all.

### 5.5 `probe-rs` chip name: `RP235x`

Not `RP2350`, not `RP235x_M33_0`, not cored variants. Confirm with
`probe-rs chip list | grep -i rp235`. The registry has evolved, so
re-check if a future probe-rs version complains.

### 5.6 Waveshare Pico-LCD-1.8 shield (if installed)

The Waveshare 1.8" 160×128 ST7735s shield plugs onto the 40-pin
header and is exercised by `lcd_demo`. Pinout (fixed by the shield
PCB):

| LCD signal | Pico GPIO | Role in firmware            |
|------------|----------:|-----------------------------|
| SCK        | GP10      | SPI1 SCLK (FunctionSpi)     |
| MOSI (DIN) | GP11      | SPI1 TX (FunctionSpi)       |
| CS         | GP9       | push-pull output (software) |
| DC         | GP8       | push-pull output (software) |
| RST        | GP12      | push-pull output (software) |
| BL         | GP13      | push-pull output (high = on)|
| MISO       | —         | unused; GP28 serves as a    |
|            |           | dummy FunctionSpi RX pin    |

The rp235x-hal SPI builder requires all three of `(tx, rx, sck)`
even when only transmitting. GP28 is physically unconnected on the
shield and just holds the "miso" slot in the tuple.

#### Architecture: hand-rolled init + raw DMA

`lcd_demo` **does not use `mipidsi`**. The embedded-hal-bus
`ExclusiveDevice` that `mipidsi`'s `SpiInterface` wants owns its
inner `Spi` bus with no disassembly method — once you've handed the
bus over you can't get it back for DMA. It's less work to inline the
~60-byte ST7735s init sequence (`st7735s_init` in `lcd_demo.rs`,
values copied from Waveshare's MicroPython driver) and own the raw
`Spi<Enabled, SPI1, ..., 8>` bus throughout.

Per-frame pipeline:

1. CPU fills a 40-KiB `Box::leak`-ed framebuffer (RGB565 **big-endian**,
   which is ST7735's wire format — no per-pixel byteswap in the push
   path).
2. CPU draws the FPS overlay into the same framebuffer via an
   `embedded_graphics::DrawTarget` wrapper around the byte slice.
3. CPU sends CASET + RASET + RAMWR as blocking SPI writes
   (13 bytes total, ~2.6 µs at 40 MHz).
4. `rp235x_hal::dma::single_buffer::Config::new(ch, bytes, spi).start()`
   fires the DMA. DMA is gated by the SPI TX-empty DREQ, so it
   auto-throttles to the SPI wire rate.
5. `transfer.wait()` blocks until DMA is done, then `spi.flush()`
   waits for BSY=0, then a short CPU spin (see §5.6.1 below), then
   `cs.set_high()`.

Address-window values are verbatim from Waveshare's driver:
`CASET = 0x01..0xA0` (cols 1..=160) and `RASET = 0x02..0x82` (rows
2..=130). Tightening RASET's end to `0x81` to match the 128-row
framebuffer leaves the panel blank — this particular panel's RAM
mapping expects the full `0x82` bound.

MADCTL = `0x70` produces the landscape orientation this demo uses.
The physical (0, 0) corner ends up at the panel's top-right after
all the MV/MX/ML bits; a bottom-right FPS overlay in logical coords
lands in a physical corner that's not where you'd expect from
reading the code. It works; measure twice, draw once.

#### 5.6.1 SPI flush + CS deassert is racy without guards

**Every tiny command byte has to be explicitly flushed, and the DMA
teardown needs a short CPU-spin cushion after `flush()` before
`cs.set_high()`.** Without both, the display shows intermittent
"content slides a few pixels then snaps back" glitches every few
seconds.

What's going on:

- rp235x-hal's `SpiBus::write` completes one byte before returning
  (it pushes to TX FIFO, then waits for the corresponding RX byte).
  In theory this guarantees the byte is fully clocked out.
- rp235x-hal's `SpiBus::flush` polls the PL022's `BSY` bit until it
  clears. In theory BSY=0 means TX FIFO empty **and** shift register
  idle.
- **In practice**, CS deassert races ambient with BSY-clear for one
  frame in every few hundred. Occasionally a frame's last byte (or a
  CASET/RASET payload byte) gets chopped mid-bit, shifting pixel
  addressing for the affected row. The very next frame's CASET/RASET
  resets the window and things snap back.

Mitigation in `lcd_demo.rs`:

- `write_cmd` and `write_data` **call `spi.flush()` explicitly after
  every `spi.write()`**, before toggling CS. Without this, CASET /
  RASET byte payloads occasionally get chopped, producing the
  horizontal shift symptom.
- After the framebuffer DMA, do `spi.flush()` **and then
  `cortex_m::asm::delay(100)`** (≈ 667 ns at 150 MHz, ~3 bytes at 40
  MHz SPI) before `cs.set_high()`. Empirical — `flush()` alone is
  not quite enough on this HAL version.

If the jiggle comes back on a future hal/rp-hal bump, try doubling
the delay first before chasing a real bug.

#### 5.6.2 Orientation / window quirks (panel-specific)

These values are for this exact Waveshare 1.8" shield:

- **MADCTL `0x70`** — landscape with MV + MX + ML set, RGB order.
- **CASET end = `0xA0`, RASET end = `0x82`** (Waveshare's magic;
  tightening to the framebuffer's 128-row dimension makes the panel
  blank).
- **No color inversion** (no `INVON` / `INVOFF` in init).
- **Red squares drawn at logical (0,0) land at the physical upper
  right** under this MADCTL. Don't fight it — if it matters for a
  later demo, adjust your drawing coordinates instead of tweaking
  MADCTL.

If you swap in a different 1.8" ST7735 variant, start from these and
iterate one value at a time — `fill_pattern` in `lcd_demo.rs` makes
a good diagnostic harness because it animates, so any drift /
tearing shows immediately.

## 6. Diagnosing when things go wrong

### "No RTT output" decision tree

1. **Is the probe connected?** `probe-rs list` — should print
   `Debugprobe on Pico (CMSIS-DAP)` with VID:PID `2e8a:000c`.
2. **Does SWD reach the chip?** `probe-rs info --chip RP235x
   --protocol swd` — should dump the two M33 core CPUIDs and the
   RP235x CoreSight ROM marker.
3. **Does the firmware actually execute?** Add a magic-write as the
   very first line of `main` and read it back:
   ```rust
   unsafe { core::ptr::write_volatile(0x2000_0100 as *mut u32, 0xDEAD_BEEF); }
   ```
   then `probe-rs read --chip RP235x --protocol swd b32 0x20000100 1`
   after flashing and resetting. If the value is not `deadbeef`, the
   firmware crashed or is stuck before reaching `main`.
4. **Is the RTT symbol linked?** `cargo nm --bin heartbeat |
   grep _SEGGER_RTT`. Should show `_SEGGER_RTT` in the `.data`
   section (typically around `0x20000008`).
5. **Is the RTT control block populated?** `probe-rs read --chip
   RP235x --protocol swd b8 0x20000008 16` — the first 16 bytes
   should be `53 45 47 47 45 52 20 52 54 54 00 00 00 00 00 00`
   ("SEGGER RTT" padded). If not, `defmt_rtt` has not initialized
   yet (no `defmt::*!` call has fired, or they are no-ops — see
   DEFMT_LOG).
6. **Has anything been written to RTT?** Read the up-channel WrOff
   at offset `+0x2c` from the control-block base (so `0x20000034`
   for the layout we have):
   `probe-rs read --chip RP235x --protocol swd b32 0x20000034 1`.
   `0` means nothing was logged yet. If WrOff > 0 but `probe-rs run`
   still shows nothing, probe-rs attach failed (try `probe-rs attach
   --rtt-scan-memory`).

### Verbose probe-rs logs

```
RUST_LOG=probe_rs=info probe-rs run --chip RP235x --protocol swd <elf>
```

Useful to confirm the flash sections landed where you expect, and
that probe-rs finished its `reset_and_halt` sequence cleanly.

### Expected flash layout

From `cargo size -- -A` (milestone-1 baseline):

| Section        | Addr         | Notes                                          |
|----------------|-------------:|------------------------------------------------|
| `.vector_table`| `0x10000000` | cortex-m-rt vectors                            |
| `.start_block` | `0x10000110` | `IMAGE_DEF` — must be inside first 4 KiB       |
| `.text`        | `0x10000138` | code                                           |
| `.bi_entries`  | after .text  | picotool metadata                              |
| `.rodata`      | after .bi    | constants                                      |
| `.data`        | `0x20000000` | SRAM start — `_SEGGER_RTT` lives in here       |
| `.bss`         | after .data  | zero-initialized                               |
| `.uninit`      | after .bss   | explicitly uninitialized (1 KiB stack sentinel)|

Milestone-1 footprint: ~15 KiB flash, ~1 KiB RAM. 4 MiB flash /
520 KiB SRAM on the board, so we have an absurd amount of headroom
for sf-nano-core + a code arena.

## 7. Known gotchas / non-obvious things

- **`probe-rs run`'s `Finished in X.XXs` line is flash completion,
  not process exit.** The process keeps running and streams RTT
  until the chip halts or you Ctrl-C. Silence after this line means
  "flash succeeded, but no RTT data" — go to §6.
- **`--connect-under-reset` requires a wired reset line** (probe GP1
  → target RUN). We never wired it. Do not add this flag; attach
  uses SWD-driven reset and works fine.
- **Two cores show up in `probe-rs info`.** Both are M33 on RP2350;
  core 0 is the boot core. Our firmware runs on core 0; core 1 stays
  idle. `probe-rs run` defaults to core 0 — do not pass `--core`
  unless you have a specific reason.
- **rp235x-hal's `defmt` feature is on defmt 0.3**, not 1.0. The
  lockfile may show both `defmt 0.3.x` and `defmt 1.0.x` resolved —
  that is fine (other deps pulling in the 1.0 line). Our direct
  `defmt = "0.3"` is what our code and the HAL's `Format` impls see.
- **Onboard LED is unreachable without more setup.** On Pico 2 W
  the LED is on CYW43 GPIO 0 (requires SPI + CYW43 init); on plain
  Pico 2 it is GP25. No board-agnostic blink demo. Use RTT logs to
  prove life instead.
- **LCD power draw can wedge SWD mid-flash.** Observed during
  `lcd_demo` bring-up: probe-rs errors out with
  `Arm(Dap(NoAcknowledge))` or `InnerTransferBlockRequest ... failed`
  part-way through writing flash. Probable cause is the LCD
  backlight + SPI traffic perturbing the 3V3 rail or SWD signal
  integrity while the previous firmware is still active. Fixes, in
  order: (1) plug the target's own USB cable so it is not being
  powered through the probe's 3V3 pin; (2) hold **BOOTSEL** on the
  target while re-plugging its USB to drop the chip into mass-storage
  mode (no user firmware running, so the SWD link is quiet), then
  `cargo run --bin <name>` again. Power-cycling alone is usually
  enough.
- **SPI `flush()` + CS deassert is racy on RP2350's PL022.** Even
  though `SpiBus::flush` waits for BUSY=0, occasional frames end up
  with the last byte chopped and the addressing shifts by a pixel or
  two until the next frame resets the window. Add `spi.flush()`
  explicitly after every `spi.write()` AND insert
  `cortex_m::asm::delay(100)` between DMA `flush()` and
  `cs.set_high()`. Symptom is distinctive: "content jiggles a few
  pixels right every few seconds, snaps back." See §5.6.1.
- **`probe-rs run` can still hold the USB handle after you Ctrl-C.**
  If the next `cargo run` fails with "Could not determine a suitable
  packet size for this probe" or "Failed to open probe", check for a
  stray `probe-rs.exe` process (`tasklist | grep probe-rs`) and kill
  it. Occasionally requires unplug/replug of the probe USB if the
  Windows USB stack is wedged.

## 8. What is *not* wired up yet

- CYW43 Wi-Fi (future `devices/pico2w/` crate with `embassy-rp` +
  `cyw43` + `embassy-net`).
- LCD drawing from JIT-compiled Wasm. The display pipeline is
  validated by `lcd_demo` and the JIT is validated by the M4 sum
  demo in `main.rs`, but nothing ties the two together yet. The
  planned next demo is a fixed-point Mandelbrot in Wasm, rendered
  via the same DMA pipeline `lcd_demo` uses, plus a parallel native
  Rust Mandelbrot binary for a JIT-vs-native speed comparison.
- `picotool` integration. Not installed by default here; the ELF
  does carry `.bi_entries` metadata, so any future `picotool info`
  invocation will Just Work without firmware changes.
- A release profile that has been exercised on hardware. Only
  `cargo run` (dev) has been verified end-to-end; release should
  work but confirm before relying on it.
- DMA double-buffering. `lcd_demo` uses single-buffered DMA —
  CPU blocks on `transfer.wait()` during each 8.2 ms SPI push. A
  JIT-driven frame producer that takes longer than SPI will
  benefit from Level-3 pipelining (two 40-KiB framebuffers, swap on
  frame boundary). Adds 40 KiB of BSS. See
  `docs/RUNTIME_CONFIG_AND_OS_MEMORY.md` / the pipeline notes in
  the design discussion — not yet implemented.

## 9. Milestone log

- **M1 (completed).** Boot + clocks + defmt-RTT heartbeat. Produces
  `heartbeat: alive @ 150 MHz SYSCLK` followed by `tick N` at ~1 Hz.
- **LCD validation side-quest (completed).** Initial `lcd_demo`
  validated SPI1 + Waveshare Pico-LCD-1.8 via `mipidsi` +
  `embedded-graphics`. Surfaced the green-tab offset / color-order
  tuning and the LCD-backlight-vs-SWD wedging gotcha.
- **M2 (completed).** sf-nano-core linked. Shared lib crate
  `sf-nano-pico2` (`src/lib.rs`) carries the `#[global_allocator]`
  (embedded-alloc) and null `sf_os_*` stubs. Behavior unchanged —
  heartbeat still prints, JIT crate is just in the link graph.
- **RuntimeConfig + linear-memory quota (completed, sf-nano-core
  slice 1).** Added `sf_nano_core::RuntimeConfig` (code arena bytes,
  wasm memory max pages, wasm operand-stack bytes) with a one-shot
  `set_runtime_config`. Fixed the double-`CodeBuffer` allocation in
  `finish_native_compile_streaming` via `install_native_code_buffer`
  — one real alloc per compile instead of alloc-swap-drop. See
  `docs/RUNTIME_CONFIG_AND_OS_MEMORY.md`.
- **M3 (completed).** Real `sf_os_alloc_executable` arena (64 KiB
  static, 16-byte aligned) in `os_shim.rs`. Clean alloc/free cycle
  logged over RTT via a self-test before the heartbeat.
- **M4 (completed).** Hardcoded integer Wasm module (`sum_1_to_10.wat`,
  since removed) baked into flash via the `wat` build-dependency,
  JIT-compiled and invoked at startup. Printed `sum 1..=10 = 55` over
  RTT on real hardware — first end-to-end JIT-on-MCU proof. Superseded
  by `mandelbrot_wasm`, which exercises the same path at real workload
  scale.
- **Aggressive memory budget (completed).** Heap 320 KiB, code arena
  128 KiB, wasm memory max 3 pages (192 KiB), wasm stack 32 KiB,
  native stack ~62 KiB via `flip-link` (stack-overflow trap instead
  of silent BSS corruption). ~520 KiB of SRAM fully accounted for.
- **DMA LCD pipeline side-quest (completed).** `lcd_demo` rewritten
  to bypass `mipidsi` for pixel pushes, hand-rolled ST7735s init,
  single-buffer DMA via `rp235x_hal::dma::single_buffer` at 40 MHz
  SPI. FPS overlay via `embedded-graphics` into the framebuffer
  before each push. Surfaced the PL022 `flush()` + CS-deassert race
  (§5.6.1) — fixed with explicit per-write flushes and a short CPU
  cushion after DMA. Baseline ~74 fps with CPU fill; SPI wire time
  alone is 8.2 ms of that, so JIT-rendered frames have ~5 ms of CPU
  budget to match this number.
- **Next: JIT Mandelbrot demo.** Fixed-point integer Mandelbrot in
  Wasm, rendered via the DMA pipeline. Plus a parallel native Rust
  Mandelbrot binary for a direct JIT-vs-native speed comparison on
  the same hardware path.
