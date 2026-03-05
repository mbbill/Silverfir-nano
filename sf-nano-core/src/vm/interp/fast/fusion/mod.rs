//! Fusion subsystem for the fast interpreter.
//!
//! - `resolve`: IR-level fusion pattern matching (runtime, feature = "fusion")
//! - `profiler`: Instruction sequence capture (feature = "profile")
//! - `pattern_trie`: N-gram trie from profiler stats (feature = "profile")
//! - `discovery`: Automatic fusion candidate selection (feature = "profile")

#[cfg(feature = "fusion")]
pub mod resolve;

#[cfg(feature = "profile")]
pub mod profiler;
#[cfg(feature = "profile")]
pub mod pattern_trie;
#[cfg(feature = "profile")]
pub mod discovery;
