//! Micro-JIT: runtime fusion via micro-assembly.
//!
//! ARM64-only. Replaces static fusion on JIT-capable platforms.
//! See docs/MICRO_JIT.md for architecture details.

pub mod reg;
pub mod arm64_enc;
