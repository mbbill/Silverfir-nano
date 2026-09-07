# Benchmark validation audit — 2026-09-07

Scope: the pinned wasmi-benchmarks checkout
`16a3d7c8fdb05506c116a9451175732d1ac77099`, its resolved dependencies,
and Nano's existing comparison scripts. This is a source audit, not a new
competitor measurement or a claim that startup regressions have passed.

## Validation and timing

| Runtime/configuration | Input validation | Startup timing |
| --- | --- | --- |
| Wasmtime/Cranelift, resolved 47.0.2 | Adapter calls `wasmtime::Module::new`; validation is automatic. | Module construction and instantiation are inside the timed operation. |
| V8, Rust crate resolved 150.4.0 | Adapter calls `WasmModuleObject::compile`; the pinned V8 implementation calls `SyncCompile`, which decodes with function-body validation enabled. | Compilation and instantiation are inside the timed operation. |
| wasmi v1/v2 `eager.checked` | Checked module construction; eager validation and translation. | Both happen during timed instantiation. |
| wasmi v1/v2 `lazy-translation.checked` | Eager validation; translation on first use. | Validation is included; unused functions need not be translated. |
| wasmi v1/v2 `lazy.checked` | Function-body validation and translation are deferred until first use. | Not equivalent to completing full validation at startup. |
| wasmi v1/v2 `lazy.unchecked` | Adapter calls unsafe `Module::new_unchecked`. | Semantic validation is omitted. |

The v2 adapter explicitly enables its `validate` Cargo feature. Both v1 and
v2 adapters allow execute tests only for `eager.checked`. Their other registered
modes participate in startup tests. The startup fixtures do not call exports
and have no start section invoking imports; this matters for deferred work.

`benches/criterion/startup.rs:152` times `rt.instantiate`, with runtime setup
and host-function registration outside the loop. `execute.rs:48` constructs
the module before `b.iter`, which times calls. Thus enabling Nano's loader
validation directly changes startup cost, not a per-call validation cost in
the steady-state execute benchmark. This does not establish identical cache,
compilation-tier or cold-process behavior across engines.

## Existing Nano comparison policy

`ci/x64_standings_report.py:153` requires `wasmi-v2.eager.checked` in the
interpreter execution field. `ci/wasmi_startup_ranking.py:32` selects
`wasmi-v1.eager.checked` as its explicit reference and excludes lazy modes
from the primary non-lazy ranking. It does not use `lazy.unchecked` as that
reference. Wasmtime/Cranelift skips FFmpeg startup in the pinned adapter;
comparisons must retain the coverage distinction.

On Nano main `0983d9e4`, the existing standalone validator is optional:
`Instance` invokes it only under `sf_module_validator`, enabled by the
`validator` feature. Neither the default core features nor the pinned
benchmark adapter's JIT/interpreter feature sets enable it. Parsing and other
engine checks still run. The release candidate makes that existing validator
unconditional in safe `Module::new` and removes the repeated instance-stage
validator call. It does not introduce a second validator. Changes inside the
existing validator include a `ref.eq` correctness fix and allocation/borrowing
optimizations; “new checks” previously referred imprecisely to checks newly
executed by the ordinary benchmark loading path.

Consequently the old/new Nano startup regression includes a change in work
performed. That is a real elapsed-time regression, but it is not evidence of
competitors disabling validation. Keep full-validation and unchecked/deferred
startup results explicitly distinguished.

## Dependency evidence

- [Wasmtime 47.0.2 Module documentation](https://docs.rs/wasmtime/47.0.2/wasmtime/struct.Module.html#method.validate).
- [rusty_v8 v150.4.0 V8 submodule](https://github.com/denoland/rusty_v8/tree/v150.4.0/v8) resolves to `ac1e23989121713ca642f6650b34deff7b686896`.
- [Pinned V8 Compile entry](https://github.com/denoland/v8/blob/ac1e23989121713ca642f6650b34deff7b686896/src/api/api.cc#L8914) and [SyncCompile validation](https://github.com/denoland/v8/blob/ac1e23989121713ca642f6650b34deff7b686896/src/wasm/wasm-engine.cc#L672).
- wasmi v2 `2.0.0-beta.8` registry source: `src/engine/config.rs:31` documents the three compilation modes; `src/module/mod.rs:234` and `:261` distinguish checked and unchecked construction. The pinned suite resolves v1 to `1.1.0` and v2 to `2.0.0-beta.8`; adapter call sites are `runtimes/wasmi-v{1,2}/lib.rs:134`.
