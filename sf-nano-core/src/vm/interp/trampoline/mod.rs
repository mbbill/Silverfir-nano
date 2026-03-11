//! Interpreter trampoline integration.
//!
//! The live interpreter uses the same C trampoline implementation as the
//! historical `fast_bak` path. The build script compiles the C sources from
//! this module's directory; Rust code reaches the entry points through
//! `handlers::run_trampoline`.
