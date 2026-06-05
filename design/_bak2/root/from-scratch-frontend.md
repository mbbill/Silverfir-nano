- The parser, decoder, and validator are written in-tree, from scratch; no
  external Wasm frontend crates anywhere in the decode path.
- Function bodies are decoded by a single streaming pass (see
  `streaming-decode`).
- Validation is feature-gated and off by default: production builds trust
  their modules; the validator performs pure validation and computes nothing
  for execution.
