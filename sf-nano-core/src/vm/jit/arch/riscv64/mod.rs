//! RV64 backend.
//!
//! The backend follows the same shared 64-bit pipeline shape as ARM64. RV64GC
//! scalar integer lowering and preserved-helper runtime calls are handled here;
//! scalar FP and SIMD are layered in separately.

pub(super) mod abi;
pub(crate) mod backend;
mod preserved;
mod template;
