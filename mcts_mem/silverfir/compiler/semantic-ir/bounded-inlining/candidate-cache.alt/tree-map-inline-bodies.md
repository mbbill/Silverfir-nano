- A bounded tree map associates each decoded callee key with an optional inline body. Negative entries use the same body-sized map value representation as positive entries.

## Moves

- 2026-09-06 (fc26a6a9) replaced by [[compiler/semantic-ir/bounded-inlining/candidate-cache]]: A fixed key/index table separates negative results from body-sized map storage while preserving the same eight-candidate budget, resolution order and generated code (code).
