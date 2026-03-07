//! Stack tracking for SP-based fast interpreter compilation.
//!
//! Tracks compile-time stack height and control flow structure.
//! In SP-based model, we don't track individual slot types - just height.

use super::config::{CompileConfig, MAX_HOT_LOCAL_COUNT};
use super::lowered_ir::OpIndex;

use alloc::vec::Vec;

/// Number of hot local register slots currently representable in lowered IR.
pub const HOT_LOCAL_COUNT: usize = MAX_HOT_LOCAL_COUNT;

/// Control frame kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    Block,
    Loop,
    If,
    Function,
}

/// Pending branch fixup: needs alt patched to END.
#[derive(Debug, Clone)]
pub struct BranchFixup {
    /// Instruction index of the branch
    pub inst_idx: OpIndex,
    /// Stack offset for the branch
    pub stack_offset: usize,
    /// Branch arity
    pub arity: usize,
    /// For BR_TABLE: which entry needs patching (None for BR/BR_IF)
    pub br_table_entry: Option<usize>,
}

/// Control frame for tracking block structure.
#[derive(Debug, Clone)]
pub struct ControlFrame {
    pub kind: BlockKind,
    /// Stack height at block entry (before params)
    pub start_height: usize,
    /// Block parameters
    pub param_count: usize,
    /// Block results
    pub result_count: usize,
    /// Instruction index at block start (loop target, or block start for tracking)
    pub start_inst_idx: OpIndex,
    /// IF instruction index (for patching at ELSE/END)
    pub if_inst_idx: Option<OpIndex>,
    /// ELSE instruction index (for patching at END)
    pub else_inst_idx: Option<OpIndex>,
    /// Pending forward branches to this block's END
    pub pending_fixups: Vec<BranchFixup>,
}

/// Compile-time stack height tracker for SP-based model.
///
/// In SP-based model, we only track stack height, not individual slot contents.
/// Values are always at sp-relative positions, accessed via sp[-1], sp[-2], etc.
pub struct StackTracker {
    config: CompileConfig,
    // Frame layout
    params_count: usize,
    locals_count: usize,
    results_count: usize,
    /// Effective hot local indices after sequential swaps.
    /// hot_locals[0] = l0's effective index, hot_locals[1] = l1's effective index, etc.
    /// None means that register slot is unused for this function.
    hot_locals: [Option<u32>; HOT_LOCAL_COUNT],

    // Stack height tracking
    height: usize,
    max_height: usize,

    /// Number of stack values currently in memory (not in TOS registers)
    /// Invariant: spill_depth <= height
    /// Invariant: height - spill_depth <= config.tos_register_count
    spill_depth: usize,

    // Control flow
    control_stack: Vec<ControlFrame>,
    unreachable: bool,
}

impl StackTracker {
    pub fn new(
        config: CompileConfig,
        params_count: usize,
        locals_count: usize,
        results_count: usize,
        hot_locals: [Option<u32>; HOT_LOCAL_COUNT],
    ) -> Self {
        let mut tracker = Self {
            config,
            params_count,
            locals_count,
            results_count,
            hot_locals,
            height: 0,
            max_height: 0,
            spill_depth: 0,
            control_stack: Vec::with_capacity(16),
            unreachable: false,
        };

        // Push implicit function-level control frame
        tracker.control_stack.push(ControlFrame {
            kind: BlockKind::Function,
            start_height: 0,
            param_count: 0,
            result_count: results_count,
            start_inst_idx: OpIndex::new(0),
            if_inst_idx: None,
            else_inst_idx: None,
            pending_fixups: Vec::new(),
        });

        tracker
    }

    #[inline]
    pub fn config(&self) -> CompileConfig {
        self.config
    }

    // =========================================================================
    // Stack Height Operations
    // =========================================================================

    /// Push one value onto the compile-time stack.
    #[inline]
    pub fn push(&mut self) {
        if !self.unreachable {
            self.height += 1;
            self.max_height = self.max_height.max(self.height);
            self.assert_invariants();
        }
    }

    /// Push n values onto the compile-time stack.
    #[inline]
    pub fn push_n(&mut self, n: usize) {
        if !self.unreachable {
            self.height += n;
            self.max_height = self.max_height.max(self.height);
            self.assert_invariants();
        }
    }

    /// Pop one value from the compile-time stack.
    #[inline]
    pub fn pop(&mut self) {
        if !self.unreachable {
            self.height = self.height.saturating_sub(1);
            // Adjust spill_depth if it exceeds height
            if self.spill_depth > self.height {
                self.spill_depth = self.height;
            }
            self.assert_invariants();
        }
    }

    /// Pop n values from the compile-time stack.
    #[inline]
    pub fn pop_n(&mut self, n: usize) {
        if !self.unreachable {
            self.height = self.height.saturating_sub(n);
            // Adjust spill_depth if it exceeds height
            if self.spill_depth > self.height {
                self.spill_depth = self.height;
            }
            self.assert_invariants();
        }
    }

    /// Current stack height.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Maximum stack height seen.
    #[inline]
    pub fn max_height(&self) -> usize {
        self.max_height
    }

    // =========================================================================
    // TOS Register Cache Tracking (Phase 2)
    // =========================================================================

    /// Current stack depth (alias for height).
    #[inline]
    pub fn depth(&self) -> usize {
        self.height
    }

    /// Number of stack values currently spilled to memory.
    #[inline]
    pub fn spill_depth(&self) -> usize {
        self.spill_depth
    }

    /// Minimum spill depth: height - tos_register_count (clamped to 0).
    #[inline]
    pub fn min_spill_depth(&self) -> usize {
        self.height.saturating_sub(self.config.tos_register_count)
    }

    /// Number of values currently in TOS registers.
    #[inline]
    pub fn tos_count(&self) -> usize {
        self.height - self.spill_depth
    }

    /// Returns depth variant (1-tos_register_count) for handler selection.
    ///
    /// This matches the generated wrapper mapping in `build/fast_interp/gen_c_wrappers.rs`:
    /// - depth == 0 => D1
    /// - otherwise: ((depth - 1) % TOS_REGISTER_COUNT) + 1
    #[inline]
    pub fn depth_variant(&self) -> u8 {
        if self.height == 0 {
            1
        } else {
            (((self.height - 1) % self.config.tos_register_count) + 1) as u8
        }
    }

    /// Returns true if TOS cache is full and a spill is needed before push.
    #[inline]
    pub fn needs_spill_before_push(&self) -> bool {
        self.tos_count() >= self.config.tos_register_count
    }

    /// Returns true if spill_depth > min_spill_depth (needs fill before control flow).
    #[inline]
    pub fn needs_fill_before_control_flow(&self) -> bool {
        self.spill_depth > self.min_spill_depth()
    }

    /// Record that `count` values were spilled from TOS to memory.
    #[inline]
    pub fn record_spill(&mut self, count: usize) {
        self.spill_depth += count;
        self.assert_invariants();
    }

    /// Record that `count` values were filled from memory to TOS.
    #[inline]
    pub fn record_fill(&mut self, count: usize) {
        self.spill_depth = self.spill_depth.saturating_sub(count);
        self.assert_invariants();
    }

    /// Reset TOS state: all values are in memory (spill_depth = height).
    /// Used after calls, control flow merges, etc.
    #[inline]
    pub fn reset_tos_state(&mut self) {
        self.spill_depth = self.height;
        self.assert_invariants();
    }

    /// Normalize for control flow: reduce spill_depth to minimum.
    /// Returns the number of fills needed.
    #[inline]
    pub fn normalize_for_control_flow(&mut self) -> usize {
        let min = self.min_spill_depth();
        let fill_count = self.spill_depth.saturating_sub(min);
        self.spill_depth = min;
        self.assert_invariants();
        fill_count
    }

    /// Debug assertions for TOS invariants.
    #[inline]
    pub fn assert_invariants(&self) {
        debug_assert!(
            self.spill_depth <= self.height,
            "spill_depth ({}) must be <= height ({})",
            self.spill_depth,
            self.height
        );
        debug_assert!(
            self.height - self.spill_depth <= self.config.tos_register_count,
            "TOS invariant violated: tos_count ({}) > TOS_REGISTER_COUNT ({})",
            self.height - self.spill_depth,
            self.config.tos_register_count
        );
    }

    // =========================================================================
    // Frame Layout
    // =========================================================================

    /// Number of parameters in this function.
    #[inline]
    pub fn params_count(&self) -> usize {
        self.params_count
    }

    /// Number of locals (excluding params) in this function.
    #[inline]
    pub fn locals_count(&self) -> usize {
        self.locals_count
    }

    /// Frame size (params + locals), NOT including metadata.
    /// Use this for encoding frame_size in instructions (e.g., return).
    #[inline]
    pub fn frame_size(&self) -> usize {
        self.params_count + self.locals_count
    }

    /// Base of operand stack region (slot index).
    /// Operand stack starts AFTER params, locals, AND metadata slots.
    /// operand[0] is at fp[operand_base()]
    #[inline]
    pub fn operand_base(&self) -> usize {
        self.config.operand_stack_base(self.frame_size())
    }

    /// End of the entire frame (slot index).
    /// frame_end = operand_base + max_height
    #[inline]
    pub fn frame_end(&self) -> usize {
        self.operand_base() + self.max_height
    }

    // =========================================================================
    // Hot Local Register Cache
    // =========================================================================

    /// Whether hot local register `n` is enabled for this function (0=l0, 1=l1, ...).
    #[inline]
    pub fn has_hot_local(&self, n: usize) -> bool {
        n < self.config.hot_local_count && n < HOT_LOCAL_COUNT && self.hot_locals[n].is_some()
    }

    /// Convenience aliases.
    #[inline]
    pub fn has_l0(&self) -> bool { self.has_hot_local(0) }
    #[inline]
    pub fn has_l1(&self) -> bool { self.has_hot_local(1) }

    /// Remap a local index through sequential transpositions: swap(0,K0), swap(1,K1_eff), ...
    ///
    /// Each init_lN swaps fp[N]↔fp[KN_eff], so a Wasm local index maps to
    /// the physical slot by applying each transposition in order.
    #[inline]
    pub fn remap_local(&self, idx: u32) -> u32 {
        let mut pos = idx;
        for (slot, k) in self.hot_locals.iter().enumerate() {
            if let Some(k) = *k {
                let slot = slot as u32;
                if pos == k {
                    pos = slot;
                } else if pos == slot {
                    pos = k;
                }
            }
        }
        pos
    }

    // =========================================================================
    // Control Flow
    // =========================================================================

    /// Enter a new block.
    pub fn enter_block(
        &mut self,
        kind: BlockKind,
        param_count: usize,
        result_count: usize,
        start_inst_idx: OpIndex,
    ) {
        let start_height = self.height.saturating_sub(param_count);

        self.control_stack.push(ControlFrame {
            kind,
            start_height,
            param_count,
            result_count,
            start_inst_idx,
            if_inst_idx: None,
            else_inst_idx: None,
            pending_fixups: Vec::new(),
        });
    }

    /// Set IF instruction index for current frame.
    pub fn set_if_inst(&mut self, idx: OpIndex) {
        if let Some(frame) = self.control_stack.last_mut() {
            frame.if_inst_idx = Some(idx);
        }
    }

    /// Set ELSE instruction index for current frame.
    pub fn set_else_inst(&mut self, idx: OpIndex) {
        if let Some(frame) = self.control_stack.last_mut() {
            frame.else_inst_idx = Some(idx);
        }
    }

    /// Handle ELSE: reset stack to block entry state.
    pub fn enter_else(&mut self) {
        if let Some(frame) = self.control_stack.last() {
            // Reset height to start + params
            self.height = frame.start_height + frame.param_count;
            // Reset TOS state: all values are in memory after control flow merge
            self.spill_depth = self.height;
            self.unreachable = false;
        }
    }

    /// Exit a block (at END), returns the frame for patching.
    ///
    /// Also returns `can_preserve_tos`: true if TOS registers remain valid after this block.
    /// TOS can be preserved when:
    /// - All forward branches to this END have stack_offset = 0 (no stack realignment)
    /// - Height doesn't change (always true for valid WASM)
    pub fn exit_block(&mut self) -> Option<(ControlFrame, bool)> {
        let frame = self.control_stack.pop()?;

        // Check if any forward branch has stack_offset > 0 (would misalign TOS)
        let has_stack_drop = frame.pending_fixups.iter().any(|f| f.stack_offset > 0);

        // After END: height = start_height + result_count
        let new_height = frame.start_height + frame.result_count;

        // TOS can be preserved if:
        // 1. No forward branches with stack drops (TOS registers remain aligned)
        // 2. Height doesn't change (for valid WASM, height should be equal)
        let can_preserve = !has_stack_drop && new_height == self.height;

        // Calculate preserved TOS depth: min(new_height, current_tos_depth)
        let current_tos_depth = self
            .height
            .saturating_sub(self.spill_depth)
            .min(self.config.tos_register_count);
        let preserved_tos_depth = if can_preserve {
            current_tos_depth.min(new_height)
        } else {
            0
        };

        self.height = new_height;
        self.spill_depth = new_height.saturating_sub(preserved_tos_depth);
        self.unreachable = false;
        self.assert_invariants();

        Some((frame, can_preserve))
    }

    /// Get current control frame (immutable).
    pub fn current_frame(&self) -> Option<&ControlFrame> {
        self.control_stack.last()
    }

    /// Get control frame at label depth.
    pub fn frame_at_depth(&self, depth: u32) -> Option<&ControlFrame> {
        let idx = self.control_stack.len().checked_sub(1 + depth as usize)?;
        self.control_stack.get(idx)
    }

    /// Get mutable control frame at label depth.
    pub fn frame_at_depth_mut(&mut self, depth: u32) -> Option<&mut ControlFrame> {
        let idx = self.control_stack.len().checked_sub(1 + depth as usize)?;
        self.control_stack.get_mut(idx)
    }

    /// Get branch arity for a label.
    pub fn branch_arity(&self, label: u32) -> usize {
        self.frame_at_depth(label)
            .map(|f| {
                if f.kind == BlockKind::Loop {
                    f.param_count
                } else {
                    f.result_count
                }
            })
            .unwrap_or(0)
    }

    /// Get branch info: (stack_offset, target_idx or None for forward ref).
    pub fn branch_info(&self, label: u32) -> (usize, Option<OpIndex>) {
        let Some(frame) = self.frame_at_depth(label) else {
            return (0, None);
        };

        let target_height = frame.start_height;
        let arity = if frame.kind == BlockKind::Loop {
            frame.param_count
        } else {
            frame.result_count
        };
        let stack_offset = self.height.saturating_sub(target_height + arity);

        let target = if frame.kind == BlockKind::Loop {
            Some(frame.start_inst_idx)
        } else {
            None // Forward ref, patch at END
        };

        (stack_offset, target)
    }

    /// Register a forward branch fixup.
    pub fn register_forward_branch(&mut self, label: u32, inst_idx: OpIndex, br_table_entry: Option<usize>) {
        let (stack_offset, _) = self.branch_info(label);
        let arity = self.branch_arity(label);

        if let Some(frame) = self.frame_at_depth_mut(label) {
            frame.pending_fixups.push(BranchFixup {
                inst_idx,
                stack_offset,
                arity,
                br_table_entry,
            });
        }
    }

    /// Get return info: (stack_offset, arity).
    pub fn return_info(&self) -> (usize, usize) {
        let arity = self.results_count;
        let stack_offset = self.height.saturating_sub(arity);
        (stack_offset, arity)
    }

    /// Mark current code as unreachable.
    pub fn set_unreachable(&mut self) {
        self.unreachable = true;
    }

    /// Check if current code is unreachable.
    pub fn is_unreachable(&self) -> bool {
        self.unreachable
    }

    // =========================================================================
    // Call Handling
    // =========================================================================

    /// Check if a br_if at the given label would be "simple" (arity=0, no stack drop).
    ///
    /// `height_at_brif` is the stack height when the condition value is on top
    /// (before br_if pops it).
    pub fn is_br_if_simple_at(&self, label: u32, height_at_brif: usize) -> bool {
        let Some(frame) = self.frame_at_depth(label) else {
            return false;
        };
        let arity = if frame.kind == BlockKind::Loop {
            frame.param_count
        } else {
            frame.result_count
        };
        if arity != 0 {
            return false;
        }
        let height_after_pop = height_at_brif.saturating_sub(1);
        height_after_pop == frame.start_height
    }

    /// Apply call effect: pop params, push results.
    pub fn apply_call(&mut self, params: usize, results: usize) {
        self.pop_n(params);
        self.push_n(results);
    }
}
