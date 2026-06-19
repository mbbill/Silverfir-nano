- GC objects (structs and arrays) live in a per-store arena indexed by a
  `Copy` index handle (`GcRef`) rather than by raw pointer. The arena is
  bump-allocated: objects are appended to a vector and never freed or
  relocated; there is no collector, and a `GcRef` index stays valid for the
  store's whole lifetime.

- A GC object records its module type index alongside its field or element
  values; the runtime cast/test checks compare against the module's declared
  types.

- Exception instances live in a separate per-store arena, also indexed by a
  `Copy` index (`ExnRef`), with the exception's field payload stored out of
  line from the registry entry.

- A raw pooled reference is resolved back to its value through a per-store
  reference registry whose entries carry an i31 scalar inline and a GC
  reference as the originating store pointer plus its index handle
  (`RefRegistryEntry`); resolution against this registry is what reattaches a
  `Copy` handle to its store-local value after a runtime-boundary crossing.

- `ref.test` and `ref.cast` are decided at runtime, not at compile time, by a
  reference type-check that dispatches on the handle's class (null, extern,
  host, pooled) and, for pooled i31/GC references, walks the structural and
  declared subtype relation in the store's type context (`check_ref_type_match`).

- A v128 value's 16-byte payload is interned into a per-store SIMD registry at
  every runtime-boundary crossing, with the operand/frame slot carrying the
  registry index in place of the payload; the bytes are recovered by
  de-interning on the way back out (`SharedSimdRegistry`).

- A thrown Wasm exception that propagates past every active `try_table` handler
  in an invocation surfaces to the embedder as a distinct error variant
  carrying the exception reference, its tag, and the module tag name; a host
  callback signals a Wasm-catchable throw through a separate VM-internal
  inbound variant that never reaches the embedder.

- Reference-typed cached locals must be frame-visible (rooted in the frame)
  before any call or runtime-helper boundary where a callee could need root
  visibility; they are never carried across a local-call safepoint only in a
  register.

## Facts

- 2025-10-04 (80817fc6) rationale: there is no collector at all — no refcount,
  no mark-sweep, no copying GC; allocation is a push onto the arena and nothing
  is ever reclaimed, which is what makes a heap index stay valid for the store's
  whole lifetime (code).

- 2026-04-18 (f98d3458) rationale: the per-store SIMD registry is append-only
  with linear-scan dedup, so unique v128 values grow store memory
  monotonically and repeated inserts are O(n) (code).

- 2026-04-18 (f98d3458) rationale: the append-only linear-scan SIMD registry was
  chosen to keep SIMD bring-up simple, with the author noting it should be
  replaced by a reclaimed/deduplicated representation once the native SIMD
  backend surface settles (sourced).

- 2026-04-22 (71cffdae) rationale: a thrown Wasm exception does not use a
  separate stack unwinder — the EH runtime helper records the exception on the
  native context as a pending escape and returns it through the same in-band
  runtime-call status channel (`NativeCallStatus::Thrown`) other escapes use,
  and the throw / throw_ref native instructions are terminated by an
  Unreachable trap so control never falls through (code).

- 2026-04-22 (71cffdae) pitfall: a host callback signals a Wasm-catchable
  exception by returning a host-throw error carrying a tag and args; the
  runtime-call boundary validates that payload against the tag's
  function-type signature before converting it to a thrown status, and a
  mistyped host throw traps ("host threw mistyped exception") rather than
  propagating an ill-formed exception into Wasm (code).

- 2026-04-22 (71cffdae) pitfall: like a call, a throw is a frame-publish
  boundary — the throw payload or the exnref operand of throw_ref must be
  published to frame slots before the throw, and native lowering asserts no
  live linear SSA values remain at the throw terminator (code).
