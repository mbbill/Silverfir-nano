//! Store-to-load forwarding pass.
//!
//! Forward exact `store` values into later exact `load` instructions within a
//! block when no intervening instruction can change the address or the stored
//! source value. Same-width matches only: a `U32` store forwards into a `U32`
//! load, a `U64` store forwards into a `U64` load. We do not synthesize a
//! narrower load from a wider store.
//!
//! Tracking both widths is required on 32-bit GP backends, where an i64
//! wasm local spill/fill lowers to a *pair* of `U32` store/load ops — the
//! `U64`-only tracker from earlier versions was structurally blind to this
//! shape, so the self-reload pattern around gp32 i64 muls survived to the
//! emitted code. See `docs/jit_cycle_cost_study.md` §"Evidence-based causal
//! chain for P1".

use crate::collections;

use crate::vm::backend::BackendConfig;
use crate::vm::machine::machine_ir::{
    MachineBlock, MachineInstKind, MachineLoadExtension, MachineMemWidth, MachineValue,
};

use super::helpers::{
    addrs_overlap, for_each_defined_reg, kill_tracked_stores_by_reg, rewrite_move_storage_type,
};
use super::TrackedStore;

#[inline]
fn is_forwardable_width(width: MachineMemWidth) -> bool {
    matches!(width, MachineMemWidth::U32 | MachineMemWidth::U64)
}

pub(super) fn forward_stored_values(block: &mut MachineBlock, config: BackendConfig) {
    let mut tracked = collections::Vec::<TrackedStore>::new();
    let mut rewritten = collections::Vec::with_capacity(block.ops.len());

    for mut inst in block.ops.drain(..) {
        let mut keep_inst = true;

        match &mut inst.kind {
            MachineInstKind::Load {
                owner,
                ty,
                dst,
                addr,
                width,
                extension: MachineLoadExtension::None,
            } if is_forwardable_width(*width) => {
                if let Some(src) = tracked
                    .iter()
                    .rev()
                    .find(|entry| entry.addr == *addr && entry.width == *width)
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
                    !addrs_overlap(entry.addr, entry.width, *addr, *width)
                });
                if is_forwardable_width(*width) {
                    tracked.push(TrackedStore {
                        addr: *addr,
                        src: *src,
                        width: *width,
                    });
                }
            }
            MachineInstKind::CallRuntime(_)
            | MachineInstKind::RefFunc { .. }
            | MachineInstKind::RefAsNonNull { .. }
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
            | MachineInstKind::ArrayLen { .. } => {
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
