//! Shared render kernels.
//!
//! These modules are used two ways:
//! - directly by firmware binaries in this crate, such as `native_demo`;
//! - by `wasm-demo/`, which includes the same source files into its
//!   `wasm32-unknown-unknown` build.

pub mod cube;
pub mod mandelbrot;
