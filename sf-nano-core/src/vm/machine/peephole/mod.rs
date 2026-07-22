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
mod reuse_loaded_values;

use crate::vm::backend::BackendConfig;
use crate::vm::machine::machine_ir::{
    MachineAddr, MachineBlock, MachineLoadExtension, MachineMemWidth, MachineProgram, MachineReg,
    MachineStorageType, MachineValue,
};

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
}

impl BlockOptCtx {
    pub(crate) fn new(config: BackendConfig) -> Self {
        let total_reg_count = config.total_reg_count() as usize;
        Self {
            config,
            first_fp_reg: config.first_fp_reg(),
            total_reg_count,
            cp_scratch: copy_propagate::CopyPropagateScratch::new(total_reg_count),
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
    deduplicate_constants::deduplicate_constants(block, ctx.first_fp_reg);
    forward_stored_values::forward_stored_values(block, ctx.config);
    reuse_loaded_values::reuse_loaded_values(block, ctx.config);
    fuse_indexed_memory::fuse_indexed_memory(block);
    reuse_loaded_values::reuse_loaded_values(block, ctx.config);
    copy_propagate::copy_propagate(block, ctx.config, &mut ctx.cp_scratch);
    // Copy propagation can make previously distinct address bases or stored
    // values identical. Re-run forwarding so those newly exposed store/load
    // pairs do not survive into code emission.
    forward_stored_values::forward_stored_values(block, ctx.config);
    fuse_isel::fuse_isel(block, ctx.config);
    fuse_smull_sign_ext::fuse_smull_sign_ext(block, ctx.total_reg_count);
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
    eliminate_dead_params::eliminate_dead_params(&mut program.blocks);
    fuse_compare_branch::fuse_compare_branch(&mut program.blocks, config.gp_unit_bytes, config);
}
