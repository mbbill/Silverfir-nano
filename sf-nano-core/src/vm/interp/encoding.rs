//! Generated instruction encoding/decoding.
//!
//! The interpreter still uses generated operand encoding from `handlers.toml`.
//! This is interpreter-specific and does not affect the backend-facing LIR
//! boundary.

#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/fast_interp/fast_encoding.rs"));
