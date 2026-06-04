---
commit: 4367f50
---
The fast-interpreter strategy's stated value was a near-JIT-speed pure
interpreter that needed no JIT, which bought two things a JIT could not match at
the time: portability to any C compilation target (the hot handler chain being
generated C), and a tiny embedded footprint — the early README advertised a
~200 KB stripped `no_std` core with zero runtime dependencies, and the goal of
building "likely the fastest pure interpreter in the world." This is the
footprint/portability half of why the interpreter was chosen first; the
speed-among-interpreters half is recorded separately
(interpreter-beats-all-interpreters-approaches-cranelift-brotli-fails). It is an
asserted design target / measured binary size, not a runtime benchmark. The JIT
strategy that replaced it re-honored the same two goals differently — portability
via a per-arch backend behind shared MachineIR, and a few-hundred-KB `no_std`
JIT-only binary still able to run on a ~520 KB MCU — rather than abandoning them.
