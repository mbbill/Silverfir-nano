---
commit: 2fecfa6, 61b3fac
---
Once the native pipeline reached a usable state and passed spectest, the default
backend mode was switched to Native (default features enabled in Cargo.toml) and
then the interpreter CLI feature was disabled entirely. The handler interpreter was
demoted from primary execution path to fallback / ground-truth oracle. This is the
concrete inflection point where the project's *primary* execution path became
native code rather than interpretation — the operational half of the
interpreter→JIT pivot, preceding the later full deletion of the interpreter+fusion
build system. An engineering-state fact (a default flipped, a feature turned off),
driven by native reaching parity, not a benchmark.
