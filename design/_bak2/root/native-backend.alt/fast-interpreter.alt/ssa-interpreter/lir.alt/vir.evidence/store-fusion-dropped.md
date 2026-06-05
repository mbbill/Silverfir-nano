---
commit: 240ed673
---
Store-root tiles from the birth design (ST_BIN — a store computing its RHS
inside one dispatch) did not survive contact with implementation: disabled
for an incomplete lowering pass (13cde314), then deleted from the VIR
instruction set as unused (240ed673). Madd/Shladd fusion stayed.
