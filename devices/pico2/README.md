# Silverfir-nano on Raspberry Pi Pico 2 / Pico 2 W (RP2350)

Bring-up firmware and demos for running the sf-nano-core JIT on the
Cortex-M33 of the RP2350.

## Binaries

| Name | Purpose | Latest numbers |
|---|---|---|
| `heartbeat` | Minimal bring-up (clocks + defmt-RTT tick). Proves the toolchain, boot block, probe-rs, RTT all work. | 1 Hz tick |
| `lcd_demo` | SPI + DMA display pipeline test: animated gradient at 40 MHz SPI with an fps overlay. | ~74 fps |
| `mandelbrot_native` | Q17.14 integer Mandelbrot compiled natively. The JIT's performance ceiling. | 18 fps |
| `mandelbrot_wasm` | Same kernel, JIT-executed by sf-nano-core. The headline demo. | 9 fps (≈55 % of native) |

`cargo run` without `--bin` runs **`mandelbrot_wasm`** (`default-run`).

## Quickstart

```
cargo run --bin heartbeat          # smoke test: is the device flashing + streaming RTT?
cargo run --bin lcd_demo           # verifies the LCD shield is wired correctly
cargo run --bin mandelbrot_native  # CPU Mandelbrot for the ceiling number
cargo run                          # = --bin mandelbrot_wasm: the JIT demo
```

## Layout

```
devices/pico2/
├── Cargo.toml, Cargo.lock, build.rs, memory.x, .cargo/config.toml
├── HACKING.md               # bring-up decisions, pin map, debugging notes
├── README.md                # this file
├── src/
│   ├── lib.rs               # shared crate re-exports
│   ├── board.rs             # XTAL, pin type aliases, SPI/DMA types
│   ├── config.rs            # sf-nano-core runtime config for this board
│   ├── heap.rs              # #[global_allocator] (embedded-alloc LlffHeap)
│   ├── os_shim.rs           # sf_os_* bare-metal shims
│   ├── mandelbrot_kernel.rs # shared kernel, Q17.14 i32 math
│   ├── display/
│   │   ├── st7735.rs        # panel init + raw SPI command/data helpers
│   │   ├── framebuffer.rs   # embedded-graphics DrawTarget wrapper
│   │   └── overlay.rs       # FPS text overlay
│   └── bin/                 # the four binaries above
└── wasm-kernel/             # standalone crate compiled to wasm32-unknown-unknown
    └── src/lib.rs           # Mandelbrot kernel + push_frame import
```

Hardware / bring-up details (pin map, ST7735s quirks, `DEFMT_LOG`,
probe-rs gotchas, milestone log) are in `HACKING.md`.
