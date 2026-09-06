//! MachineIR peephole optimization pass.
//!
//! Runs within a single block. Current optimizations:
//!
//! 1. **Constant deduplication** (`deduplicate_constants`): Replaces duplicate
//!    constant materializations with register copies from the first.
//!
//! 2. **Copy propagation** (`copy_propagate`): Rewrites uses of
//!    `move rTmp <- rSrc` to reference `rSrc` directly.
//!
//! 3. **Store-to-load forwarding** (`forward_stored_values`): Replaces
//!    `load` after a matching `store` with a register move.
//!
//! 4. **Load-to-load reuse** (`reuse_loaded_values`): Replaces a second
//!    identical load with a register copy from the first.
//!
//! 5. **Indexed memory fusion** (`fuse_indexed_memory`): Fuses address
//!    computation + load/store into `IndexedLoad`/`IndexedStore`.
//!
//! 6. **Compare-and-branch fusion** (`fuse_compare_branch`): Fuses
//!    `IntCompare + Branch` into `Branch { IntCompare { ... } }`.
//!
//! 7. **SMULL sign-extension fusion** (`fuse_smull_sign_ext`): Replaces
//!    `Int64PairBinary{Mul}` whose operands are both `i64.extend_i32_s`
//!    with `Int64MulFromSignExt32`, a single signed 32x32 -> 64 multiply.
//!    Only fires on 32-bit GP backends (where the pair form exists).
//!
//! 8. **Index-extend relaxation** (`relax_index_extends`): Drops the
//!    `ZeroExtend32` obligation from indexed loads/stores whose index was
//!    defined by a 32-bit-form instruction, including facts proven across
//!    every incoming CFG edge, on backends where those definitions already
//!    zero-extend (`gp32_defs_zero_extend`).

mod cache_loop_frame_words;
mod copy_propagate;
mod deduplicate_constants;
mod eliminate_dead_params;
mod eliminate_overwritten_frame_stores;
mod fold_induction_offsets;
mod forward_stored_values;
mod fuse_compare_branch;
mod fuse_indexed_memory;
mod fuse_isel;
mod fuse_smull_sign_ext;
pub(crate) mod helpers;
mod hoist_loop_address_bases;
mod promote_self_loop_globals;
mod recognize_memmove;
mod relax_index_extends;
mod reuse_loaded_values;
mod reuse_loop_context_loads;
mod reuse_loop_frame_values;
mod simplify_demanded_bits;

use crate::vm::jit::backend::BackendConfig;
use crate::vm::jit::machine::machine_ir::{
    MachineAddr, MachineBlock, MachineInstKind, MachineIntBinaryOp, MachineLoadExtension,
    MachineMemWidth, MachineProgram, MachineReg, MachineStorageType, MachineValue,
};

#[derive(Clone, Copy, Default)]
struct BlockFeatures {
    // Conservative precursor facts only: a pass may run unnecessarily, but
    // must never be skipped when it could rewrite the block. The debug/test
    // oracle at the end of `optimize_block` guards this contract as passes
    // gain new patterns.
    load_count: usize,
    has_store: bool,
    has_move: bool,
    has_address_add: bool,
    may_fuse_isel: bool,
    has_bitwise: bool,
}

impl BlockFeatures {
    fn observe(&mut self, kind: &MachineInstKind) {
        match kind {
            MachineInstKind::Load { .. } => self.load_count += 1,
            MachineInstKind::Store { .. } => self.has_store = true,
            MachineInstKind::Move { .. } => self.has_move = true,
            MachineInstKind::IntBinary { op, .. } => {
                self.has_bitwise |= matches!(
                    op,
                    MachineIntBinaryOp::And | MachineIntBinaryOp::Or | MachineIntBinaryOp::Xor
                );
                self.has_address_add |= *op == MachineIntBinaryOp::Add;
                self.may_fuse_isel |= matches!(
                    op,
                    MachineIntBinaryOp::And
                        | MachineIntBinaryOp::Shl
                        | MachineIntBinaryOp::ShrS
                        | MachineIntBinaryOp::ShrU
                        | MachineIntBinaryOp::Rotl
                        | MachineIntBinaryOp::Rotr
                );
            }
            _ => {}
        }
    }
}

#[derive(Clone, Copy)]
struct TrackedStore {
    addr: MachineAddr,
    src: MachineValue,
    width: MachineMemWidth,
}

#[derive(Clone, Copy)]
struct TrackedLoad {
    addr: MachineAddr,
    ty: MachineStorageType,
    width: MachineMemWidth,
    extension: MachineLoadExtension,
    reg: MachineReg,
}

/// Reusable context for running the block-local peepholes. Holds the immutable
/// config plus the copy-propagation scratch buffer, which is sized by register
/// count and amortised across blocks. Construct once per function (or once per
/// streamed block sequence) and reuse with `optimize_block`.
pub(crate) struct BlockOptCtx {
    pub config: BackendConfig,
    defer_index_extend_relaxation: bool,
    first_fp_reg: u16,
    total_reg_count: usize,
    cp_scratch: copy_propagate::CopyPropagateScratch,
    tracked_stores: crate::collections::Vec<TrackedStore>,
    tracked_loads: crate::collections::Vec<TrackedLoad>,
    bit_scratch: crate::collections::Vec<u64>,
}

impl BlockOptCtx {
    pub(crate) fn new(config: BackendConfig) -> Self {
        let total_reg_count = config.total_reg_count() as usize;
        Self {
            config,
            defer_index_extend_relaxation: false,
            first_fp_reg: config.first_fp_reg(),
            total_reg_count,
            cp_scratch: copy_propagate::CopyPropagateScratch::new(total_reg_count),
            tracked_stores: crate::collections::Vec::new(),
            tracked_loads: crate::collections::Vec::new(),
            bit_scratch: crate::collections::Vec::new(),
        }
    }
}

/// Run all block-local peepholes on a single block.
///
/// This is the unit that the streaming-mode pipeline can invoke per block as
/// the producer streams blocks out, without materializing a full
/// `MachineProgram`. Whole-program passes (`fuse_compare_branch`, the 32-bit
/// `fuse_smull_sign_ext_across_edges`) are not run here; they live in
/// `optimize` and execute after the per-block pass loop.
pub(crate) fn optimize_block(ctx: &mut BlockOptCtx, block: &mut MachineBlock) {
    #[cfg(any(debug_assertions, test))]
    let mut unconditional_oracle = block.clone();

    let features = deduplicate_constants::deduplicate_constants(block, ctx.first_fp_reg);
    let may_forward_store = features.has_store && features.load_count != 0;
    let may_reuse_load = features.load_count > 1;
    let may_fuse_indexed =
        features.has_address_add && (features.has_store || features.load_count != 0);

    if may_forward_store {
        forward_stored_values::forward_stored_values(block, ctx.config, &mut ctx.tracked_stores);
    }
    if may_reuse_load {
        reuse_loaded_values::reuse_loaded_values(block, ctx.config, &mut ctx.tracked_loads);
    }
    if may_fuse_indexed {
        fuse_indexed_memory::fuse_indexed_memory(block);
    }
    if may_reuse_load {
        reuse_loaded_values::reuse_loaded_values(block, ctx.config, &mut ctx.tracked_loads);
    }
    if features.has_move || may_forward_store || may_reuse_load {
        copy_propagate::copy_propagate(block, ctx.config, &mut ctx.cp_scratch);
    }
    // Copy propagation can make previously distinct address bases or stored
    // values identical. Re-run forwarding so those newly exposed store/load
    // pairs do not survive into code emission.
    if may_forward_store {
        forward_stored_values::forward_stored_values(block, ctx.config, &mut ctx.tracked_stores);
    }
    if features.may_fuse_isel {
        fuse_isel::fuse_isel(block, ctx.config);
    }
    if features.has_store {
        eliminate_overwritten_frame_stores::eliminate_overwritten_frame_stores(
            block,
            ctx.config.gp_unit_bytes,
        );
    }
    if features.has_bitwise {
        simplify_demanded_bits::simplify_demanded_bits(block, ctx.config, &mut ctx.bit_scratch);
    }
    if ctx.config.is_32bit_gp_target() {
        fuse_smull_sign_ext::fuse_smull_sign_ext(block, ctx.total_reg_count);
    }
    if ctx.config.gp32_defs_zero_extend && !ctx.defer_index_extend_relaxation {
        relax_index_extends::relax_index_extends_with_policy(
            block,
            ctx.config.relax_index_extends_in_profitable_blocks_only,
        );
    }

    // Keep pass scheduling honest as transformation patterns evolve. Release
    // builds pay nothing; debug builds and tests compare against the original
    // unconditional sequence after every block.
    #[cfg(any(debug_assertions, test))]
    {
        let _ = deduplicate_constants::deduplicate_constants(
            &mut unconditional_oracle,
            ctx.first_fp_reg,
        );
        forward_stored_values::forward_stored_values(
            &mut unconditional_oracle,
            ctx.config,
            &mut ctx.tracked_stores,
        );
        reuse_loaded_values::reuse_loaded_values(
            &mut unconditional_oracle,
            ctx.config,
            &mut ctx.tracked_loads,
        );
        fuse_indexed_memory::fuse_indexed_memory(&mut unconditional_oracle);
        reuse_loaded_values::reuse_loaded_values(
            &mut unconditional_oracle,
            ctx.config,
            &mut ctx.tracked_loads,
        );
        copy_propagate::copy_propagate(&mut unconditional_oracle, ctx.config, &mut ctx.cp_scratch);
        forward_stored_values::forward_stored_values(
            &mut unconditional_oracle,
            ctx.config,
            &mut ctx.tracked_stores,
        );
        fuse_isel::fuse_isel(&mut unconditional_oracle, ctx.config);
        eliminate_overwritten_frame_stores::eliminate_overwritten_frame_stores(
            &mut unconditional_oracle,
            ctx.config.gp_unit_bytes,
        );
        simplify_demanded_bits::simplify_demanded_bits(
            &mut unconditional_oracle,
            ctx.config,
            &mut ctx.bit_scratch,
        );
        if ctx.config.is_32bit_gp_target() {
            fuse_smull_sign_ext::fuse_smull_sign_ext(
                &mut unconditional_oracle,
                ctx.total_reg_count,
            );
        }
        if ctx.config.gp32_defs_zero_extend && !ctx.defer_index_extend_relaxation {
            relax_index_extends::relax_index_extends_with_policy(
                &mut unconditional_oracle,
                ctx.config.relax_index_extends_in_profitable_blocks_only,
            );
        }
        assert_eq!(
            block, &unconditional_oracle,
            "feature-gated peephole schedule diverged from unconditional passes"
        );
    }
}

/// Run peephole optimizations on all blocks in a program.
///
/// `config` still defines physical register banks and bank compatibility, but
/// semantic linear-value versus cached-local ownership now comes from explicit
/// MachineIR metadata, not from register-number layout.
pub(crate) fn optimize(program: &mut MachineProgram, config: BackendConfig) {
    let mut ctx = BlockOptCtx::new(config);
    // The materialized pipeline performs whole-program rewrites after its
    // block-local phase. Defer this irreversible relaxation until those
    // rewrites have finished: a later pass can replace a proven-clean index
    // definition or edge argument, and `None` carries no obligation that the
    // final analysis could restore. Standalone streaming block optimization
    // has no later MachineIR rewrites and retains its local relaxation.
    ctx.defer_index_extend_relaxation = true;
    for block in &mut program.blocks {
        optimize_block(&mut ctx, block);
    }
    if config.is_32bit_gp_target() {
        fuse_smull_sign_ext::fuse_smull_sign_ext_across_edges(program, ctx.total_reg_count);
    }
    // Address hoisting changes instructions, block parameters, and edge
    // arguments, but not CFG targets, so frame-value reuse can consume the
    // same predecessor/dominance analysis.
    let entry = program.entry;
    let loop_graph = hoist_loop_address_bases::analyze_loop_graph(&program.blocks, entry);
    hoist_loop_address_bases::hoist_loop_address_bases(program, config, &loop_graph);
    reuse_loop_frame_values::reuse_loop_frame_values(&mut program.blocks, &loop_graph, entry);
    reuse_loop_context_loads::reuse_loop_context_loads(&mut program.blocks, entry);
    promote_self_loop_globals::promote_self_loop_globals(
        &mut program.blocks,
        &loop_graph,
        entry,
        config,
    );
    eliminate_dead_params::eliminate_dead_params(&mut program.blocks);
    // Dead cached-local parameters can hide otherwise unused physical lanes.
    // Reuse frame words only after those parameters and edge arguments vanish.
    cache_loop_frame_words::cache_loop_frame_words(
        &mut program.blocks,
        &loop_graph,
        entry,
        &mut ctx,
    );
    fuse_compare_branch::fuse_compare_branch(&mut program.blocks, config.gp_unit_bytes, config);
    // After compare-branch fusion: the fold reads loop bounds from
    // `Branch { IntCompare }` latch terminators. The passes since
    // `analyze_loop_graph` rewrite instructions and conditions but never
    // CFG targets, so the loop graph is still valid.
    fold_induction_offsets::fold_induction_offsets(program, &loop_graph);
    // Memmove recognition deliberately matches the still-explicit
    // ZeroExtend32 memory sequence, so it must precede the irreversible
    // relaxation below.
    recognize_memmove::recognize_memmove(program, config);
    // Run this exactly once after every materialized MachineIR rewrite. The
    // fold may have emitted new ZeroExtend32 forms, and clean block parameters
    // (including loop-carried values) can now use the direct indexed form.
    if config.gp32_defs_zero_extend {
        relax_index_extends::relax_index_extends_program_with_policy(
            program,
            config.relax_index_extends_in_profitable_blocks_only,
        );
    }
}
