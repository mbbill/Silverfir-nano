---
commit: 9cbc3e50
---
The register-count and value-passing design was decided in an isolated
dispatch laboratory (benchmarks/btb/, ten benchmark versions) before
touching the interpreter. Measured on Apple Silicon: Perm3 baseline (3
regs as args, permutation-baked) 0.5ns/op; Arr8 (pointer to register
array, dynamic indexing) 1.7ns; passing 8 registers as preserve_none args
is nearly free (Reg8-NoOp 0.6ns) — but *dynamic* selection over them
collapses (local array 7ns = 19 memory ops per instruction, pointer array
17.6ns, nested-ternary cmov 3.4ns). Conclusion: scale to 8 registers as
arguments, keep selection static via permuted handlers, and control the
wrapper explosion (calc_perms.py) by generating only used permutations.
