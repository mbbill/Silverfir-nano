//! Render kernels shared with the Pico2 bring-up style.
//!
//! One demo is selected per build, so each kernel is declared under the same
//! feature that selects it in `main.rs`. Declaring both would compile the
//! unselected one into a binary crate that has no way to reach it, which is
//! dead by construction rather than by accident.

#[cfg(feature = "demo-cube")]
pub mod cube;
#[cfg(feature = "demo-mandelbrot")]
pub mod mandelbrot;
