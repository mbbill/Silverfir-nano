---
commit: b3210288
---
Author (2026-06-04): a match-based eval loop relies purely on compiler
optimization — dispatch compiles to whatever rustc decides. With tail calls
and the tricks they enable (register-resident ToS lanes, each handler ending
in its own jump to the next), the dispatch overhead itself becomes directly
optimizable instead of being left to the compiler.
