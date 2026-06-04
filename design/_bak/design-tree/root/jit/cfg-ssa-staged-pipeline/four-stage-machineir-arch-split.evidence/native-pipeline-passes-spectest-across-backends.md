---
commit: 4ae8509, db7df58
---
The full native pipeline (SSA-IR → MachineIR → arch) passes spectest end-to-end,
first on arm64, then validated across all four early targets — arm64, x86_64,
armv7a/emu32, emu64 — by end of March 2026. x86_64 was a full second ISA (~3460-line
backend) that dropped in behind MachineIR and passed CoreMark then full spectest
without changing the middle end. The bring-up order was deliberate: prove the new
CFG-based LIR on the *simplest* consumer (the base interpreter passes spectest
first) before debugging native on top of it.

This is the correctness fact that cleared the CFG+SSA+MachineIR structure to
*replace* the micro-JIT and proved the MachineIR + per-arch-backend split is
genuinely portable — a new architecture is a new backend, not a codegen rewrite.
Binary pass/fail against the spec suite, repeated per backend.
