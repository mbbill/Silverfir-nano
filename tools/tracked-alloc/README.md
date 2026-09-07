# sf-nano-tracked-alloc

Internal allocation helpers for [Silverfir-nano](https://github.com/mbbill/Silverfir-nano).
Applications embedding WebAssembly should depend on `sf-nano-core`.

Containers are direct re-exports of standard `alloc` types in every feature
configuration. The default build is `no_std` and has no dependencies.

The `memprof` feature requires `std` and adds process-wide allocation counters,
explicit runtime-buffer counters and bounded phase records. An embedding
executable must install `TrackingAllocator` around its allocator to collect
ordinary heap statistics. These diagnostic counters are intended for local
development; container type attribution and allocation backtraces are not
provided.

This is a support package, versioned with Silverfir-nano. Its low-level helper
functions are not the engine's supported embedding API.

Licensed under either MIT or Apache-2.0, at your option; see `LICENSE-MIT` and
`LICENSE-APACHE`.
