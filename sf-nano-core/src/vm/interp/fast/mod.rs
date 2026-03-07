//! Fast interpreter — minimal no_std port from sf-core.
//!
//! Layout:
//! - `runtime`: Entry point and stack management.
//! - `instruction`: 32-byte instruction header & arena.
//! - `context`: Hot state + opaque context container.
//! - `fast_code`: FastCode storage and FastCodeCache.
//! - `handlers/`: Handler implementations organized by category.
//! - `resolved`: Shared resolved instruction form for base/fusion/native.
//! - `compiler/`: Fast-instruction assembly and finalization.
//! - `encoding`: Generated instruction encoding/decoding.

/// Number of TOS (Top-of-Stack) registers in the fast interpreter.
pub const TOS_REGISTER_COUNT: usize = 4;

pub mod compiler;
pub mod context;
pub mod encoding;
pub mod fast_code;
pub mod frame_layout;
pub mod handlers;

/// Generated handler variant lookup tables.
#[allow(dead_code)]
pub mod handler_lookup {
    include!(concat!(env!("OUT_DIR"), "/fast_interp/fast_handler_lookup.rs"));
}

pub mod instruction;
pub mod precompile;
pub mod runtime;

pub mod fusion;
pub mod resolved;
