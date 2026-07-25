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

mod copy_propagate;
mod deduplicate_constants;
mod eliminate_dead_params;
mod forward_stored_values;
mod fuse_compare_branch;
mod fuse_indexed_memory;
mod fuse_isel;
mod fuse_smull_sign_ext;
pub(crate) mod helpers;
mod hoist_loop_address_bases;
mod recognize_memmove;
mod reuse_loaded_values;
mod reuse_loop_context_loads;
mod reuse_loop_frame_values;

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
}

impl BlockFeatures {
    fn observe(&mut self, kind: &MachineInstKind) {
        match kind {
            MachineInstKind::Load { .. } => self.load_count += 1,
            MachineInstKind::Store { .. } => self.has_store = true,
            MachineInstKind::Move { .. } => self.has_move = true,
            MachineInstKind::IntBinary { op, .. } => {
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
    first_fp_reg: u16,
    total_reg_count: usize,
    cp_scratch: copy_propagate::CopyPropagateScratch,
    tracked_stores: crate::collections::Vec<TrackedStore>,
    tracked_loads: crate::collections::Vec<TrackedLoad>,
}

impl BlockOptCtx {
    pub(crate) fn new(config: BackendConfig) -> Self {
        let total_reg_count = config.total_reg_count() as usize;
        Self {
            config,
            first_fp_reg: config.first_fp_reg(),
            total_reg_count,
            cp_scratch: copy_propagate::CopyPropagateScratch::new(total_reg_count),
            tracked_stores: crate::collections::Vec::new(),
            tracked_loads: crate::collections::Vec::new(),
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
    if ctx.config.is_32bit_gp_target() {
        fuse_smull_sign_ext::fuse_smull_sign_ext(block, ctx.total_reg_count);
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
        if ctx.config.is_32bit_gp_target() {
            fuse_smull_sign_ext::fuse_smull_sign_ext(
                &mut unconditional_oracle,
                ctx.total_reg_count,
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
    for block in &mut program.blocks {
        optimize_block(&mut ctx, block);
    }
    if config.is_32bit_gp_target() {
        fuse_smull_sign_ext::fuse_smull_sign_ext_across_edges(program, ctx.total_reg_count);
    }
    // Address hoisting changes instructions, block parameters, and edge
    // arguments, but not CFG targets, so frame-value reuse can consume the
    // same predecessor/dominance analysis.
    let loop_graph = hoist_loop_address_bases::analyze_loop_graph(&program.blocks, program.entry);
    hoist_loop_address_bases::hoist_loop_address_bases(program, config, &loop_graph);
    reuse_loop_frame_values::reuse_loop_frame_values(&mut program.blocks, &loop_graph);
    reuse_loop_context_loads::reuse_loop_context_loads(&mut program.blocks);
    eliminate_dead_params::eliminate_dead_params(&mut program.blocks);
    fuse_compare_branch::fuse_compare_branch(&mut program.blocks, config.gp_unit_bytes, config);
    recognize_memmove::recognize_memmove(program);
}
