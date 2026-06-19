- Each compiled instruction is a TempInst carrying its handler function pointer
  plus logical pattern-data field values, with encoding into the immediate slots
  deferred to the finalizer.

- Static fusion is applied at decode time by an OpFuser that pattern-matches raw
  Wasm opcodes with lookahead, gated by the fusion feature.

- Each backend (interpreter dispatch, static fusion, JIT grouping) independently
  tracks TOS height, computes depth variants, and classifies stack effects; the
  dispatch driver, instruction emitter, and finalizer form three separate stages
  over the TempInst stream, with the finalizer as the sole place encoding
  happens.

## Moves

- 2026-03-05 (2c7ce3f3) replaced by [[neutral-ir]]: stack-state management was
  triplicated across the interpreter builder, static fusion, and the JIT, and
  the handler-coupled TempInst could not serve as a neutral representation;
  lowering Wasm to one neutral IR resolves stack management once and lets all
  three backends share a single pipeline with graceful degradation (code).
