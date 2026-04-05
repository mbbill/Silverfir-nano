//! Store-to-load forwarding pass.
//!
//! Forward exact `store.u64` values into later exact `load.u64` instructions
//! within a block when no intervening instruction can change the address or
//! the stored source value.

use alloc::vec::Vec;

use crate::vm::backend::BackendConfig;
use crate::vm::machine::machine_ir::{
    MachineBlock, MachineInstKind, MachineLoadExtension, MachineMemWidth, MachineValue,
};

use super::helpers::{
    addrs_overlap, for_each_defined_reg, kill_tracked_stores_by_reg, rewrite_move_storage_type,
};
use super::TrackedStore;

pub(super) fn forward_stored_values(block: &mut MachineBlock, config: BackendConfig) {
    let mut tracked = Vec::<TrackedStore>::new();
    let mut rewritten = Vec::with_capacity(block.ops.len());

    for mut inst in block.ops.drain(..) {
        let mut keep_inst = true;

        match &mut inst.kind {
            MachineInstKind::Load {
                owner,
                ty,
                dst,
                addr,
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
            } => {
                if let Some(src) = tracked
                    .iter()
                    .rev()
                    .find(|entry| entry.addr == *addr)
                    .map(|entry| entry.src)
                {
                    if matches!(src, MachineValue::Reg(src_reg) if src_reg == *dst) {
                        keep_inst = false;
                    } else if let Some(move_ty) = rewrite_move_storage_type(*dst, src, *ty, config)
                    {
                        inst.kind = MachineInstKind::Move {
                            owner: *owner,
                            ty: move_ty,
                            dst: *dst,
                            src,
                        };
                    }
                }
            }
            MachineInstKind::Store {
                addr, width, src, ..
            } => {
                tracked.retain(|entry| {
                    !addrs_overlap(entry.addr, MachineMemWidth::U64, *addr, *width)
                });
                if *width == MachineMemWidth::U64 {
                    tracked.push(TrackedStore {
                        addr: *addr,
                        src: *src,
                    });
                }
            }
            MachineInstKind::CallExternal(_) => {
                tracked.clear();
            }
            _ => {}
        }

        if keep_inst {
            for_each_defined_reg(&inst.kind, |dst| {
                kill_tracked_stores_by_reg(&mut tracked, dst);
            });
            rewritten.push(inst);
        }
    }

    block.ops = rewritten;
}
