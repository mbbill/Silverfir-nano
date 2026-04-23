# sf-nano-mandelbrot-armv7

Native ARMv7 benchmark of the Q17.14 Mandelbrot kernel from
`devices/pico2/src/mandelbrot_kernel.rs`. Cross-compiled and run under
`qemu-arm-static` to serve as the "what does *good* Thumb-2 look like"
reference for comparison against sf-nano-core's JIT-emitted code.

## Run

```
./scripts/wsl-mandelbrot-native-armv7.sh            # 60 frames, Thumb-2
./scripts/wsl-mandelbrot-native-armv7.sh 30         # 30 frames
ISA=a32 ./scripts/wsl-mandelbrot-native-armv7.sh    # plain ARM instead
```

## Baseline (WSL Debian 12, qemu-arm-static 7.2, -cpu cortex-a15)

These are host-wall-time numbers under qemu user-mode — not
cycle-accurate and not comparable to Pico 2 wall-clock. They're here to
confirm the binary runs and to produce a stable framebuffer checksum
(`0x172f8cdd`) for cross-checking JIT output.

| Metric      | Value |
|---           | ---:|
| per frame   | ~1.0 ms |
| checksum    | `0x172f8cdd` |

Real Pico 2 native runs this same kernel at 18 fps (~55 ms / frame at
150 MHz). The JIT currently runs it at 9 fps (~109 ms / frame).

## Disassembly — native references

- `bench_render_i64.thumb2.asm` — Q16.16 i64 kernel, **212 B**, hot loop
  ~20 instructions with 3× `SMULL` + paired `LSR`/`ORR` for the >>16.
- `bench_render_i32.thumb2.asm` — Q17.14 i32 kernel (control / floor),
  **192 B**, hot loop ~15 instructions with 2× `MUL` + 3× `ASR #14`.

## Disassembly — JIT emission (Phase 3)

`jit_run_i64.thumb2.asm` is the JIT-emitted Thumb-2 for the
`run(frame)` export in `wasm-kernel-i64/`, captured via
`SF_NATIVE_DUMP_DIR` under qemu and post-processed by
`scripts/postprocess_native_dump.py --arch thumbv7`.

### Key numbers

| | Native `bench_render_i64` | JIT `run` | Ratio |
|---                | ---: | ---: | ---: |
| Total `.text`     | 212 B | 1816 B | **8.6×** |
| Multiplies        | 3× SMULL (inline) | 3× UMULL (inline) | same |
| Inline `i64.mul`? | yes | yes (not a helper call) | — |
| Bounds-check stubs| 0 | 5 | — |

### What the gap is

The 8.6× size gap is **not the multiplies** (both sides inline a 32×32→64 mul).
It comes from:
- wasm-level bounds checks and panic stubs (the `blx` calls at 0xc94,
  0xcfc, 0xe48, 0xf08, 0xfc8 are `panic_bounds_check` / out-of-bounds
  traps, not math)
- block-boundary housekeeping (the JIT emits every wasm basic block as
  a separate region — see per-block entries in
  `/tmp/mandel-i64-dump/native_index.txt`)
- frame-slot spills where LLVM keeps values in registers

These are Phase 4's optimization surface.

## How to regenerate the JIT dump

```bash
# 1. Build the wasm kernel (host, one-time)
cd tools/mandelbrot-armv7/wasm-kernel-i64
cargo build --release --target wasm32-unknown-unknown

# 2. Build sf-nano-cli for armv7 + Thumb-2 (dev profile for ir-dump)
cd ../../..
cargo +1.92.0 build --target armv7-unknown-linux-musleabihf -p sf-nano-cli \
    --no-default-features --features jit,thumb2-test

# 3. Run under qemu with SF_NATIVE_DUMP_DIR
rm -rf /tmp/mandel-i64-dump && mkdir /tmp/mandel-i64-dump
SF_NATIVE_DUMP_DIR=/tmp/mandel-i64-dump qemu-arm-static -cpu cortex-a15 \
    target/armv7-unknown-linux-musleabihf/debug/sf-nano-cli \
    --backend native \
    tools/mandelbrot-armv7/wasm-kernel-i64/target/wasm32-unknown-unknown/release/sf_nano_mandelbrot_wasm_kernel_i64.wasm

# 4. Post-process (specify --arch thumbv7 or default aarch64 fails)
python3 scripts/postprocess_native_dump.py \
    --wasm tools/mandelbrot-armv7/wasm-kernel-i64/target/wasm32-unknown-unknown/release/sf_nano_mandelbrot_wasm_kernel_i64.wasm \
    --dump-dir /tmp/mandel-i64-dump \
    --out-dir /tmp/mandel-i64-pp \
    --arch thumbv7 \
    --function 2   # func 2 = `run`
```
