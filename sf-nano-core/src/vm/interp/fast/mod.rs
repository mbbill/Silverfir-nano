//! Fast interpreter family: base + fusion.
//!
//! This subtree may keep handler/trampoline concepts. It must not dictate the
//! architecture of `native/`.

pub mod build;
pub mod context;
pub mod dump;
pub mod encoding;
pub mod fast_code;
pub mod finalizer;
pub mod frame_layout;
pub mod fusion;
pub mod handlers;
pub mod handlers_c;
pub mod instruction;
pub mod precompile;
pub mod resolve;
pub mod resolved;
pub mod runtime;
pub mod trampoline;

/// Number of rotating-window registers used by the fast interpreter family.
pub const TOS_REGISTER_COUNT: usize = 4;

/// Generated handler lookup tables.
#[allow(dead_code)]
pub mod handler_lookup {
    include!(concat!(env!("OUT_DIR"), "/fast_interp/fast_handler_lookup.rs"));
}
