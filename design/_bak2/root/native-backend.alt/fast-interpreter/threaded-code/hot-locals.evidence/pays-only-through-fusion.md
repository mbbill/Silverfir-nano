---
commit: 5b1d2e6
---
The A/B test that defined this mechanism's real value: with fusion OFF,
CoreMark for l0+l1 / l0-only / neither measured 3358 / 3394 / 3331 — all
within noise. The register cache alone is worth ~0%. But l0-aware fusion
patterns measured ~10% over non-l0 fusion. The cache pays only through
fusion: a standalone local_get_l0 still costs a dispatch, and out-of-order
execution hides the saved memory access; inside a fused handler the l0
access folds into instruction operands and disappears entirely. The hot
local cache is a fusion enabler, not a standalone optimization.
