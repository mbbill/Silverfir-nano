---
commit: a061476a
---
Author (2026-06-04): tagged vs untagged is simple — tags make the type bigger.
The slot must hold a 64-bit value and *stay* 64 bits; with a tag, alignment
pads the slot and wastes space.
