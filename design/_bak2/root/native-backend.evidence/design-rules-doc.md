---
commit: c410200
---
NATIVE_DESIGN.md lays down sixteen rules — the layering constitution. The
boundary test, verbatim: "A concept belongs below NativeIR only if it truly
differs by ISA." VM semantics lower above; the arch backend "should only:
choose machine registers; emit loads, stores, arithmetic, and branches;
implement the concrete native ABI and continuation protocol; honor explicit
helper and trap boundaries." Enter native once, exit once; reference
backend is debug-only; mixed-mode execution is forbidden; cache
invalidation policy is decided above the ISA layer. The companion
typed-residency proposal states the register philosophy: "The target is
not a general register allocator" — bounded GP/FP transient banks plus
canonical frame homes, explicit slot traffic.
