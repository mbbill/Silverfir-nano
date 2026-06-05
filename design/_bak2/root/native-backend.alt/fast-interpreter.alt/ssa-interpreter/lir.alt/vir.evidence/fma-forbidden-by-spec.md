---
commit: f7ad56a9
---
Float Madd fusion was disabled as a spec violation: contracting mul+add into
an FMA changes rounding (single vs double rounding), and WebAssembly's IEEE
754 semantics are deterministic — no contraction allowed. A fusion that
looks free can be semantically illegal; the spectest gate caught it.
