//! Value/state classes visible at the LIR boundary.
//!
//! LIR does not expose rotating-window recovery or concrete machine register
//! names. It only distinguishes the classes that survive the CFG/SSA
//! conversion.

/// LIR-visible value/state classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegisterClass {
    TosLane,
    HotLocal,
}
