- Runtime state shared across an invocation lives in a store-rooted structure
  that is backend-agnostic: module instances, function/global/memory/table
  instances, and the cross-store registries that resolve raw handles back to
  their values (`Store`).

- Frame slots are the canonical source of truth for locals, spilled deep-stack
  values, call arguments and results, and call-link records; registers are
  execution caches over that frame state, which is what makes the cache,
  local-call frame reuse, and boundary rules safe.

- Generated native code is written into executable code buffers whose backing
  is requested from an OS-abstracted executable-memory allocator; an unreachable
  module's whole executable arena is freed wholesale on drop, with its stale
  per-buffer state purged; the OS may reuse the virtual address.

- Runtime sizing is a value the embedder builds and hands to an engine,
  which copies it into every instance it creates ([[runtime-config]]).
  Hosted targets default it to the pre-config numbers; the bare-metal
  target defaults to zeros that engine construction rejects by name.

- The Wasm operand/call stack and native dispatch context are cached per Store
  and reused across exported invocations. `eval` temporarily takes ownership;
  exact module/table/function-registry revisions control cached dispatch-view
  refresh (`take_native_stack_cache`, `prepare_for_invocation`).

- Host/imported functions are clonable, potentially capturing `Fn` callbacks
  bound at instantiation and stored behind single-threaded shared ownership;
  plain `HostFn` pointers remain source-compatible inputs. Callbacks receive a
  `Caller`, typed params, and a caller-owned multi-value result buffer
  (`HostCallback`, `Import::func`).

## Facts

- 2026-07-22 measurement: forwarding the benchmark's real capturing
  `clock_ms` callback through `HostCallback` enabled genuine CoreMark execution;
  matched runs scored sf-nano 38,540.60, Wasmtime Cranelift 37,814.33, Wasmer
  Cranelift 37,674.24, and V8 34,736.29. This is distinct from the 3.323 ms
  startup/coremark instantiation benchmark, which does not exercise the host
  callback (sourced).

- 2026-07-22 measurement: reusing the native stack/context and validating
  cached views by revision reduced regex-redux from 30.393 to 19.199 us (36.8%)
  because short exported calls no longer allocate/free the stack or rebuild
  type, memory, table, and function dispatch views each time; see
  [[runtime/invocation-cache]] (sourced).

- 2026-04-09 (c329abab) rationale: the native artifact strips MachineIR after
  emission because nothing at runtime reads it on the native backends; the one
  exception is the emulator (Reference) backend, which interprets MachineIR
  directly and so keeps it, and the ir-dump path, consulted before the strip
  while MachineIR is still resident (code).

- 2026-04-26 (8dc01387) pitfall: dropping a `Store` leaves dangling entries in
  the shared cross-store function registry; `Store::drop` tombstones its own
  slots (nulling the owner pointer) and dispatch-view refresh emits an INVALID
  function view for a dead slot rather than filtering it out, so surviving
  handles keep their registry-index alignment (code).

## Moves

- 2026-02-14 (a8528504) replaced [[runtime-store]]: the store is a single-module
  model — the module instance owns its memories, tables, globals and function
  specs directly, dropping Rc<RefCell<>> reference counting and borrow overhead,
  inter-module linking (LinkableData/LinkableInstance), and the per-lookup
  HashMap (Vec + linear scan, since modules have few imports and this avoids a
  hashbrown dependency) (code).

- 2026-02-14 (a8528504) replaced [[runtime-store/host-functions]]: the
  host-function hook is a bare `fn` pointer rather than a `dyn` trait object
  because that is zero-alloc and no_std-friendly and carries multi-value results
  through a caller buffer; WASI and other host capabilities are provided by an
  external crate passing function pointers in, not built into the core (code).
