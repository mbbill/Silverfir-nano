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

- A single global runtime configuration is installed write-once by the embedder
  before any instance is created; hosted targets default it to the pre-config
  numbers, while the bare-metal target defaults to zeros that an unconfigured
  embedder hits as a clean error (`RuntimeConfig`).

- The Wasm operand/call stack that backs each invoke is a single embedder-sized
  buffer allocated per invocation, indexed in 64-bit slots.

- Host/imported functions are supplied to the engine as bare `fn` pointers
  bound at instantiation (`fn(&mut Caller, &[Value], &mut [Value])`), with
  multi-value results written into a caller-owned buffer (`HostFn`).

## Facts

- 2026-04-09 (c329abab) rationale: the native artifact strips MachineIR after
  emission because nothing at runtime reads it on the native backends; the one
  exception is the emulator (Reference) backend, which interprets MachineIR
  directly and so keeps it, and the ir-dump path, consulted before the strip
  while MachineIR is still resident (diff).

- 2026-04-26 (8dc01387) pitfall: dropping a `Store` leaves dangling entries in
  the shared cross-store function registry; `Store::drop` tombstones its own
  slots (nulling the owner pointer) and dispatch-view refresh emits an INVALID
  function view for a dead slot rather than filtering it out, so surviving
  handles keep their registry-index alignment (diff).

## Moves

- 2026-02-14 (a8528504) replaced [[runtime-store]]: the store is a single-module
  model — the module instance owns its memories, tables, globals and function
  specs directly, dropping Rc<RefCell<>> reference counting and borrow overhead,
  inter-module linking (LinkableData/LinkableInstance), and the per-lookup
  HashMap (Vec + linear scan, since modules have few imports and this avoids a
  hashbrown dependency) (diff).

- 2026-02-14 (a8528504) replaced [[runtime-store/host-functions]]: the
  host-function hook is a bare `fn` pointer rather than a `dyn` trait object
  because that is zero-alloc and no_std-friendly and carries multi-value results
  through a caller buffer; WASI and other host capabilities are provided by an
  external crate passing function pointers in, not built into the core (diff).
