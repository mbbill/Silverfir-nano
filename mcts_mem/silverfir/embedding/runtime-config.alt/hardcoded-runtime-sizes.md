- The default executable code-arena size is a compile-time constant (12 MiB on
  32-bit, 16 MiB on 64-bit targets).

- The maximum linear-memory page count is fixed at the wasm32 ceiling and is not
  embedder-tunable.

## Moves

- 2026-04-21 (b206d2aa) replaced by [[runtime-config]]: the bare-metal target
  cannot fit the hosted defaults (12-16 MiB code arena, wasm32 page ceiling)
  into a few hundred KiB of SRAM, so the fixed constants are replaced by a
  write-once global the embedder installs before any instance, with hosted
  defaults preserving the old numbers and the bare-metal default zeroed to force
  a clean error (code).
