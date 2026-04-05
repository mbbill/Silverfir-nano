//! End-to-end middle-end tests.
//!
//! These tests intentionally exercise the current planner/rewrite pipeline from
//! semantic IR down to prepared SSA-IR. The goal is not just helper coverage:
//! we want concrete scenarios from `PLAN.md` to stay pinned by observable
//! middle-end behavior.

mod boundary;
mod calls;
mod cleanup;
mod const_fold;
mod entry_state;
mod helpers;
mod local_access;
mod loops;
mod pressure;
mod typed_blocks;
