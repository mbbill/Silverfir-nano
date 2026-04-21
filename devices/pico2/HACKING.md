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
cargo size --bin sf-nano-pico2 -- -A   # section sizes (flash + RAM footprint)
cargo nm   --bin sf-nano-pico2         # symbol table (grep _SEGGER_RTT to verify RTT)
cargo objdump --bin sf-nano-pico2 -- -d --no-show-raw-insn   # disassemble
```

### Binaries

- **`sf-nano-pico2`** (`src/main.rs`) — minimal heartbeat firmware.
  The milestone-sequence baseline.
- **`lcd_demo`** (`src/bin/lcd_demo.rs`) — one-off validation that
  drives the Waveshare Pico-LCD-1.8 shield over SPI1. Not part of the
  milestone sequence; kept so we can re-verify the SPI/display stack
  without perturbing `main.rs`.

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
header and is validated by `lcd_demo`. Pinout (fixed by the shield
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

This panel is a green-tab style ST7735s. The mipidsi 0.10 builder
settings that produce correct output:

```rust
Builder::new(ST7735s, di)
    .reset_pin(rst)
    .display_size(128, 160)            // native (portrait) resolution
    .display_offset(2, 1)              // green-tab column/row offset
    .orientation(Orientation::new().rotate(Rotation::Deg90))
    .color_order(ColorOrder::Rgb)
    .invert_colors(ColorInversion::Normal)
    .init(&mut timer)
    .unwrap();
```

How we landed on those: we iterated once each on `color_order`
(`Bgr` produced a red background where we asked for blue →
`Rgb`), `invert_colors` (`Inverted` produced black where we asked
for white → `Normal`), and `display_offset` (the default `(0, 0)`
left a 1-px strip of garbage on the top row and right column of the
landscape view → `(2, 1)` fills the panel cleanly). These are all
panel-specific; if you swap in a different 1.8" ST7735 board, start
from these and flip one flag at a time.

SPI at 20 MHz MODE_0 with `ExclusiveDevice::new_no_delay` works fine
— ST7735 has no CS-to-SCK hold requirement that needs an explicit
delay provider. Avoid wiring `timer` into `ExclusiveDevice` because
mipidsi also needs a `DelayNs` for `init()` and the borrows conflict.

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
4. **Is the RTT symbol linked?** `cargo nm --bin sf-nano-pico2 |
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

## 8. What is *not* wired up yet

- sf-nano-core link (milestone 2+).
- `sf_os_alloc_executable` and friends — bare-metal shim stubs
  (milestone 2 uses nulls from `sf-nano-bare-smoke`; milestone 3
  implements a real static arena).
- CYW43 Wi-Fi (future `devices/pico2w/` crate).
- LCD drawing from sf-nano-core — the display stack is validated
  end-to-end via `lcd_demo`, but nothing links the JIT to it yet.
  Natural fit for a later milestone that computes a Wasm result
  and paints it on screen.
- `picotool` integration. Not installed by default here; the ELF
  does carry `.bi_entries` metadata, so any future `picotool info`
  invocation will Just Work without firmware changes.
- A release profile that has been exercised on hardware. Only
  `cargo run` (dev) has been verified end-to-end; release should
  work but confirm before relying on it.

## 9. Milestone log

- **M1 (completed).** Boot + clocks + defmt-RTT heartbeat. Produces
  `sf-nano-pico2 alive @ 150 MHz SYSCLK` followed by `tick N` at
  ~1 Hz.
- **Side-quest (completed).** `lcd_demo` validates SPI1 + Waveshare
  Pico-LCD-1.8 (ST7735s) end-to-end with mipidsi + embedded-graphics.
  Surfaced the green-tab offset / color-order tuning and the SWD
  wedging gotcha; not on the milestone sequence but the
  configuration values live in §5.6 for reuse.
- **M2 (next).** Link sf-nano-core with null `sf_os_*` stubs. Expect
  no behavior change — firmware still prints heartbeat, but links
  against the JIT crate.
- **M3.** Replace null stubs with a real static RWX arena; first
  `CodeBuffer::with_capacity(N)` should succeed. Will surface the
  12 MiB `CodeBuffer::DEFAULT_CAPACITY` blocker in
  `sf-nano-core/src/vm/runtime/code_buf.rs` — plumb a configurable
  capacity through `CodeBuffer::new()` callers before this works.
- **M4.** Hardcoded integer Wasm (sum 1..=10 = 55) JITs and runs on
  the board; result prints over RTT.
