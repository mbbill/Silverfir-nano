---
commit: b54132f9
---
Author (2026-06-04): the December 2025 fast-interpreter rework was not a
competition with the XIR backend — the fast interpreter's handler system
was simply not as refined as XIR's, and the two basically work the same
way. The rework brought the matured techniques (C handler bodies,
generated handler mappings, the trampoline pattern) over to the fast
interpreter.
