//! Fusion backend implementation.
//!
//! Fusion consumes planning-stage grouping and emits handler-stream groups.
//!
//! The crucial architectural change is that grouping is no longer discovered
//! here. Planning consumes the prebuilt fusion table and hands fast/fusion a
//! grouped LIR program to emit.

pub mod policy;
pub mod table;
