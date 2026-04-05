//! Load-to-load reuse pass.
//!
//! When the same address is loaded twice with no intervening store that could
//! alias, replaces the second load with a register copy from the first.

use alloc::vec::Vec;

use crate::vm::backend::BackendConfig;
use crate::vm::machine::machine_ir::{MachineBlock, MachineInstKind, MachineValue};

use super::helpers::{
    addrs_overlap, for_each_defined_reg, kill_tracked_loads_by_reg, rewrite_move_storage_type,
};
use super::TrackedLoad;

pub(super) fn reuse_loaded_values(block: &mut MachineBlock, config: BackendConfig) {
    let mut tracked = Vec::<TrackedLoad>::new();
    let mut rewritten = Vec::with_capacity(block.ops.len());

    for mut inst in block.ops.drain(..) {
        let mut keep_inst = true;
        let mut produced_load = None;
        let mut rewrite_load = None;

        match &inst.kind {
            MachineInstKind::Load {
                owner,
                ty,
                dst,
                addr,
                width,
                extension,
            } => {
                if let Some(src_reg) = tracked
                    .iter()
                    .rev()
                    .find(|entry| {
                        entry.addr == *addr
                            && entry.ty == *ty
                            && entry.width == *width
                            && entry.extension == *extension
                    })
                    .map(|entry| entry.reg)
                {
                    if src_reg == *dst {
                        keep_inst = false;
                    } else if let Some(move_ty) =
                        rewrite_move_storage_type(*dst, MachineValue::Reg(src_reg), *ty, config)
                    {
                        rewrite_load = Some((*owner, *dst, src_reg, move_ty));
                        produced_load = Some(TrackedLoad {
                            addr: *addr,
                            ty: *ty,
                            width: *width,
                            extension: *extension,
                            reg: *dst,
                        });
                    }
                } else {
                    produced_load = Some(TrackedLoad {
                        addr: *addr,
                        ty: *ty,
                        width: *width,
                        extension: *extension,
                        reg: *dst,
                    });
                }
            }
            MachineInstKind::Store { addr, width, .. } => {
                tracked.retain(|entry| !addrs_overlap(entry.addr, entry.width, *addr, *width));
            }
            MachineInstKind::CallExternal(_) => {
                tracked.clear();
            }
            _ => {}
        }

        if keep_inst {
            if let Some((owner, dst, src_reg, ty)) = rewrite_load {
                inst.kind = MachineInstKind::Move {
                    owner,
                    ty,
                    dst,
                    src: MachineValue::Reg(src_reg),
                };
            }
            for_each_defined_reg(&inst.kind, |dst| {
                kill_tracked_loads_by_reg(&mut tracked, dst);
            });
            if let Some(load) = produced_load {
                tracked.push(load);
            }
            rewritten.push(inst);
        }
    }

    block.ops = rewritten;
}
