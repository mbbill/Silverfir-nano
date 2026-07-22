- Hosted std builds eagerly compile sufficiently large modules with a bounded
  worker pool. Each worker owns an ABI view, constant pool, and ordinary
  per-function streaming pipeline; results are reassembled in function-index
  order before instantiation returns (`compile_full_functions_parallel`).

- The low-memory/template/no_std path stays strictly single-stream, and small
  modules avoid worker startup entirely (`parallel_eager_worker_count`).

## Facts

- 2026-07-22 measurement: current FFmpeg compilation with workers disabled was
  6.78 s median versus 1.68 s with eight workers, about 4.0x lower wall time but
  with higher aggregate CPU work; details are in
  [[compiler.fact/startup-campaign-2026-07-22]] (sourced).

## Moves

- 2026-07-22 (4454fa83) replaced [[serial-streaming]]:
  independent eager function compilation left multicore capacity idle; bounded
  workers preserve eager completion and per-worker streaming while reducing
  hosted wall time (code).
