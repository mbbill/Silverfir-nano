//! Generated instruction encoding/decoding.
//!
//! Provides type-safe encode/decode functions generated from handlers.toml.

#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/fast_interp/fast_encoding.rs"));
