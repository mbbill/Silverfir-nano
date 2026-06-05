---
commit: 63de36fd
---
Author (2026-06-04): the linker came before the interpreter could execute
because spectest required it — the suite needs imports, module registration,
and the `spectest` host module before any test can run.
