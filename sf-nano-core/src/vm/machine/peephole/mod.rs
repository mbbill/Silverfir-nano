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

mod copy_propagate;
mod deduplicate_constants;
mod forward_stored_values;
mod fuse_compare_branch;
mod fuse_indexed_memory;
mod helpers;
mod reuse_loaded_values;

use crate::vm::backend::BackendConfig;
use crate::vm::machine::machine_ir::{
    MachineAddr, MachineLoadExtension, MachineMemWidth, MachineProgram, MachineReg,
    MachineStorageType, MachineValue,
};

pub(crate) use fuse_compare_branch::reg_dead_at_block_entry;

#[derive(Clone, Copy)]
struct TrackedStore {
    addr: MachineAddr,
    src: MachineValue,
}

#[derive(Clone, Copy)]
struct TrackedLoad {
    addr: MachineAddr,
    ty: MachineStorageType,
    width: MachineMemWidth,
    extension: MachineLoadExtension,
    reg: MachineReg,
}

/// Run peephole optimizations on all blocks in a program.
///
/// Register classification is derived from `config` — the single source of
/// truth for the register layout.
pub(crate) fn optimize(program: &mut MachineProgram, config: BackendConfig) {
    let first_fp_reg = config.first_fp_reg();
    let gp_reg_width = config.gp_unit_bytes;
    let mut cp_scratch = copy_propagate::CopyPropagateScratch::new(config.total_reg_count() as usize);
    for block in &mut program.blocks {
        deduplicate_constants::deduplicate_constants(block, first_fp_reg);
        forward_stored_values::forward_stored_values(block, config);
        reuse_loaded_values::reuse_loaded_values(block, config);
        fuse_indexed_memory::fuse_indexed_memory(block);
        copy_propagate::copy_propagate(block, config, &mut cp_scratch);
    }
    fuse_compare_branch::fuse_compare_branch(&mut program.blocks, gp_reg_width, config);
}
