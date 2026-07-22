- Imported host functions are stored as clonable `HostCallback` values backed by
  single-threaded shared `dyn Fn` ownership. `Import::func`, typed variants, and
  tag-aware variants accept capturing `'static` closures; the `HostFn` alias is
  retained for embedders that name or cast a plain function pointer
  (`HostCallback::new`).

- The invocation ABI remains `Caller + &[Value] + &mut [Value] -> Result`: the
  callback may access guest memory through Caller and writes multi-value results
  into caller-owned storage.

## Facts

- 2026-07-22 measurement: the wasmi-benchmarks adapter previously discarded its
  real linked callback and installed an inert function because nano accepted
  only `fn`. After `9ae4eb4`, the adapter forwarded the capturing `clock_ms`
  closure and the dedicated CoreMark binary genuinely executed at 38,540.60,
  narrowly ahead of both matched Cranelift integrations (sourced).

- 2026-07-22 pitfall: startup/coremark and execution CoreMark are different
  measurements. The startup case can instantiate with an inert callback and
  measured 3.323 ms; only the dedicated score runner invokes `clock_ms` and
  proves host-callback integration (sourced).

## Moves

- 2026-07-22 (9ae4eb43) replaced [[bare-function-pointer]]: stateful adapters
  and imports were not representable as plain `fn` values; shared capturing
  callbacks retain the typed ABI and plain-function compatibility (code).
