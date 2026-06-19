- NativeCode carries a separate cfg-gated entry pointer and root-return pointer
  per target architecture (arm64_entry/arm64_root_return,
  armv7a_entry/armv7a_root_return, x86_64_entry/x86_64_root_return), each with
  its own per-arch typedef and its own `with_<arch>_entry` setter.

## Moves

- 2026-03-24 (b4808682) replaced by [[backends]]: every architecture's native
  entry has the identical extern-C signature (NativeContext*, u64*) -> u32, so
  the three cfg-gated per-arch typedef + entry/return field pairs and their
  per-arch with_*_entry setters were redundant and collapse to a single
  NativeRootEntry/NativeCodePtr and one with_entry (code).
