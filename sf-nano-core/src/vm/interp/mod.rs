//! Handler-based interpreter family.
//!
//! This subtree keeps the interpreter worldview. It is a sibling backend to
//! `native/`, not the owner of the VM architecture.

pub mod build;
pub mod context;
pub mod dump;
pub mod encoding;
pub mod fast_code;
pub mod finalizer;
pub mod frame_layout;
pub mod handlers;
pub mod handlers_c;
pub mod instruction;
pub mod precompile;
pub mod raw_value;
pub mod resolve;
pub mod resolved;
pub mod runtime;
pub mod stack;
pub mod trampoline;

/// Number of rotating-window registers used by the interpreter handler family.
pub const TOS_REGISTER_COUNT: usize = 4;

/// Generated handler lookup tables.
#[allow(dead_code)]
pub mod handler_lookup {
    include!(concat!(
        env!("OUT_DIR"),
        "/fast_interp/fast_handler_lookup.rs"
    ));
}
