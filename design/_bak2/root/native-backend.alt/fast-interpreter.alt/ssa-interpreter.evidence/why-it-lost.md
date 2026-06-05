---
commit: 51c76585
---
Author (2026-06-04), the end-of-rs verdict: the XIR pipeline reached ~90%
of wasm3's CoreMark — and was still the wrong design. (1) Size: the whole
compiler pipeline is way too big. (2) Combinatorics: XIR's registers are
true registers — allocation makes their ordering arbitrary — so handlers
need permutations over register orderings: more than ~10k handlers at 8
registers even with optimizations. 8 is simultaneously too few (the ideal
— locals living in registers so local.get/set become register moves —
needs more) and too many (permutation count forbids going higher). (3)
Dispatch count: ParCopy kept the instruction count from shrinking, and for
an interpreter the number of dispatches — jumping from instruction to
instruction — is an even bigger performance contributor than memory
access.
