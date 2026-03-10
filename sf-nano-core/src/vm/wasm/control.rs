//! Structured control-flow analysis helpers.
//!
//! This is the right place for block/loop/if stacks and semantic target
//! resolution. It is not the right place for spill/fill or grouping.

use super::common::{SemanticIndex, SemanticTarget};

/// Structured Wasm block kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    Block,
    Loop,
    If,
}

/// One semantic control-frame entry during decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlFrame {
    pub kind: BlockKind,
    pub entry: SemanticIndex,
    pub else_target: Option<SemanticTarget>,
    pub end_target: SemanticTarget,
    pub params: u16,
    pub results: u16,
}

/// Semantic control stack.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ControlStack {
    pub frames: alloc::vec::Vec<ControlFrame>,
}

impl ControlStack {
    #[inline]
    pub fn push(&mut self, frame: ControlFrame) {
        self.frames.push(frame);
    }

    #[inline]
    pub fn pop(&mut self) -> Option<ControlFrame> {
        self.frames.pop()
    }

    #[inline]
    pub fn current(&self) -> Option<&ControlFrame> {
        self.frames.last()
    }

    #[inline]
    pub fn current_mut(&mut self) -> Option<&mut ControlFrame> {
        self.frames.last_mut()
    }

    pub fn branch_target(&self, depth: u32) -> Option<SemanticTarget> {
        let idx = self.frames.len().checked_sub(depth as usize + 1)?;
        let frame = self.frames.get(idx)?;
        match frame.kind {
            BlockKind::Loop => Some(SemanticTarget::new(frame.entry.as_usize())),
            BlockKind::Block | BlockKind::If => frame.else_target.or(Some(frame.end_target)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_branch_targets_entry() {
        let mut stack = ControlStack::default();
        stack.push(ControlFrame {
            kind: BlockKind::Loop,
            entry: SemanticIndex::new(7),
            else_target: None,
            end_target: SemanticTarget::new(12),
            params: 0,
            results: 0,
        });

        assert_eq!(stack.branch_target(0), Some(SemanticTarget::new(7)));
    }
}
