Single-function compiler stress fixture.

This benchmark is intentionally not representative runtime code. It exists to
make memprof phase analysis readable by collapsing the compile pipeline onto one
large function instead of hundreds of interleaved small ones.

Files:
- `gen_single_fn_wat.py`: regenerates the WAT source
- `single_fn_200k.wat`: generated single-function module source
- `single_fn_200k.wasm`: assembled benchmark module (currently about 205 KiB)

Local toolchain used here:
- WAT assembly: `wasm-as`

Regenerate:

```sh
python3 benchmarks/wasi/single-fn/gen_single_fn_wat.py
wasm-as \
  benchmarks/wasi/single-fn/single_fn_200k.wat \
  -o benchmarks/wasi/single-fn/single_fn_200k.wasm
```

Memprof:

```sh
cargo run --release -p sf-nano-cli --features memprof -- \
  --backend native \
  --memprof \
  --memprof-report /tmp/single-fn.html \
  benchmarks/wasi/single-fn/single_fn_200k.wasm
```
