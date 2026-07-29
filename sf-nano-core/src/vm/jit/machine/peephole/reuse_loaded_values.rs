//! Load-to-load reuse pass.
//!
//! When the same address is loaded twice with no intervening store that could
//! alias, replaces the second load with a register copy from the first.

use crate::collections;

use crate::vm::jit::backend::BackendConfig;
use crate::vm::jit::machine::machine_ir::{MachineBlock, MachineInstKind, MachineValue};

use super::helpers::{
    kill_tracked_loads_by_reg, rewrite_move_storage_type, store_may_alias, unknown_store_may_alias,
};
use super::TrackedLoad;

pub(super) fn reuse_loaded_values(
    block: &mut MachineBlock,
    config: BackendConfig,
    tracked: &mut collections::Vec<TrackedLoad>,
) {
    if block.ops.is_empty() {
        return;
    }

    tracked.clear();

    block.ops.retain_mut(|inst| {
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
                        if addr.base != *dst {
                            produced_load = Some(TrackedLoad {
                                addr: *addr,
                                ty: *ty,
                                width: *width,
                                extension: *extension,
                                reg: *dst,
                            });
                        }
                    }
                } else {
                    if addr.base != *dst {
                        produced_load = Some(TrackedLoad {
                            addr: *addr,
                            ty: *ty,
                            width: *width,
                            extension: *extension,
                            reg: *dst,
                        });
                    }
                }
            }
            MachineInstKind::Store { addr, width, .. } => {
                tracked.retain(|entry| !store_may_alias(entry.addr, entry.width, *addr, *width));
            }
            MachineInstKind::IndexedStore { .. }
            | MachineInstKind::MemoryFill { .. }
            | MachineInstKind::MemoryCopy { .. }
            | MachineInstKind::MemoryInit { .. }
            | MachineInstKind::TableFill { .. }
            | MachineInstKind::TableCopy { .. }
            | MachineInstKind::TableInit { .. }
            | MachineInstKind::TableGrow { .. } => {
                tracked.retain(|entry| !unknown_store_may_alias(entry.addr.base));
            }
            MachineInstKind::CallRuntime(_)
            | MachineInstKind::RefFunc { .. }
            | MachineInstKind::RefAsNonNull { .. }
            | MachineInstKind::RefAbsolutize { .. }
            | MachineInstKind::RefEq { .. }
            | MachineInstKind::RefI31 { .. }
            | MachineInstKind::I31GetS { .. }
            | MachineInstKind::I31GetU { .. }
            | MachineInstKind::AnyConvertExtern { .. }
            | MachineInstKind::ExternConvertAny { .. }
            | MachineInstKind::RefTest { .. }
            | MachineInstKind::RefCast { .. }
            | MachineInstKind::StructNew { .. }
            | MachineInstKind::StructNewDefault { .. }
            | MachineInstKind::StructGet { .. }
            | MachineInstKind::StructSet { .. }
            | MachineInstKind::ArrayNew { .. }
            | MachineInstKind::ArrayNewDefault { .. }
            | MachineInstKind::ArrayGet { .. }
            | MachineInstKind::ArraySet { .. }
            | MachineInstKind::ArrayLen { .. }
            | MachineInstKind::EhAllocExnRef { .. } => {
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
            inst.kind.for_each_defined_reg(|dst| {
                kill_tracked_loads_by_reg(tracked, dst);
            });
            if let Some(load) = produced_load {
                tracked.push(load);
            }
        }

        keep_inst
    });
}
