- One streaming decode pass per function body broadcasts each decoded op to
  registered handlers; the validator (when enabled) is one such handler.
- Handlers may also consume the stream in pull mode (`on_stream`), taking
  one or many ops per step — the hook multi-op consumers (IR builders,
  fusion matching) build on; the default implementation forwards op-by-op.
