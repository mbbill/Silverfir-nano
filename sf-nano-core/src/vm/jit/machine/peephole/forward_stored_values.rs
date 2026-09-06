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

use crate::vm::jit::backend::BackendConfig;
use crate::vm::jit::machine::machine_ir::{
    MachineBlock, MachineInstKind, MachineLoadExtension, MachineMemWidth, MachineStorageType,
    MachineValue, MACHINE_FP_REG,
};

use super::helpers::{
    inst_defines, inst_uses_value, kill_tracked_stores_by_reg, rewrite_move_storage_type,
    store_may_alias, unknown_store_may_alias,
};
use super::TrackedStore;

#[inline]
fn is_forwardable_width(width: MachineMemWidth) -> bool {
    matches!(width, MachineMemWidth::U32 | MachineMemWidth::U64)
}

pub(super) fn forward_stored_values(
    block: &mut MachineBlock,
    config: BackendConfig,
    tracked: &mut collections::Vec<TrackedStore>,
) {
    if block.ops.is_empty() {
        return;
    }

    preserve_frame_value_before_clobber(block, config);
    tracked.clear();

    block.ops.retain_mut(|inst| {
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
                tracked.retain(|entry| !store_may_alias(entry.addr, entry.width, *addr, *width));
                if is_forwardable_width(*width) {
                    tracked.push(TrackedStore {
                        addr: *addr,
                        src: *src,
                        width: *width,
                    });
                }
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
            | MachineInstKind::ArrayLen { .. } => {
                tracked.clear();
            }
            _ => {}
        }

        if keep_inst {
            inst.kind.for_each_defined_reg(|dst| {
                kill_tracked_stores_by_reg(tracked, dst);
            });
        }

        keep_inst
    });
}

/// A spill immediately followed by destructive arithmetic and its reload can
/// keep the original value in the reload destination instead. Move that value
/// before the arithmetic only when the destination is neither read nor written
/// by it. Keep the frame store: traps and later blocks still observe the same
/// published value. Only native-width GP frame slots have an exact GP-word
/// move representation on every backend.
fn preserve_frame_value_before_clobber(block: &mut MachineBlock, config: BackendConfig) {
    for index in 2..block.ops.len() {
        let MachineInstKind::Load {
            owner,
            ty: MachineStorageType::GpWord,
            dst,
            addr,
            width,
            extension: MachineLoadExtension::None,
        } = block.ops[index].kind
        else {
            continue;
        };
        if addr.base != MACHINE_FP_REG || width.bytes() != u32::from(config.gp_unit_bytes) {
            continue;
        }
        let MachineInstKind::Store {
            ty: MachineStorageType::GpWord,
            addr: store_addr,
            width: store_width,
            src: MachineValue::Reg(src),
        } = block.ops[index - 2].kind
        else {
            continue;
        };
        let arithmetic = &block.ops[index - 1].kind;
        if addr != store_addr
            || width != store_width
            || src == dst
            || !matches!(arithmetic, MachineInstKind::IntBinary { .. })
            || !inst_defines(arithmetic, src)
            || inst_defines(arithmetic, addr.base)
            || inst_defines(arithmetic, dst)
            || inst_uses_value(arithmetic, dst)
            || rewrite_move_storage_type(
                dst,
                MachineValue::Reg(src),
                MachineStorageType::GpWord,
                config,
            ) != Some(MachineStorageType::GpWord)
        {
            continue;
        }
        block.ops[index].kind = MachineInstKind::Move {
            owner,
            ty: MachineStorageType::GpWord,
            dst,
            src: MachineValue::Reg(src),
        };
        block.ops.swap(index - 1, index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::jit::machine::machine_ir::{
        MachineAddr, MachineBlockId, MachineInst, MachineIntBinaryOp, MachineIntWidth, MachineReg,
        MachineRegOwner, MachineTerminator,
    };

    fn spill(config: BackendConfig) -> MachineBlock {
        let width = if config.gp_unit_bytes == 8 {
            MachineMemWidth::U64
        } else {
            MachineMemWidth::U32
        };
        let addr = MachineAddr {
            base: MACHINE_FP_REG,
            offset: 24,
        };
        MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops: collections::vec![
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr,
                        width,
                        src: MachineValue::Reg(MachineReg(4)),
                    }
                },
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I32,
                        op: MachineIntBinaryOp::Xor,
                        dst: MachineReg(4),
                        lhs: MachineValue::Reg(MachineReg(4)),
                        rhs: MachineValue::Imm64(17),
                    }
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        owner: MachineRegOwner::LinearValue,
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(5),
                        addr,
                        width,
                        extension: MachineLoadExtension::None,
                    }
                },
            ],
            terminator: MachineTerminator::Return,
        }
    }

    #[test]
    fn saves_native_frame_word_before_destructive_arithmetic() {
        for bytes in [4, 8] {
            let config = BackendConfig::new(bytes, 6, 0, 0);
            let mut block = spill(config);
            let original = block.clone();
            preserve_frame_value_before_clobber(&mut block, config);
            assert_eq!(block.ops[0], original.ops[0], "published frame value stays");
            assert_eq!(block.ops[2], original.ops[1], "arithmetic stays unchanged");
            assert!(matches!(
                block.ops[1].kind,
                MachineInstKind::Move {
                    ty: MachineStorageType::GpWord,
                    dst: MachineReg(5),
                    src: MachineValue::Reg(MachineReg(4)),
                    ..
                }
            ));
        }
    }

    #[test]
    fn keeps_reload_when_early_copy_would_change_an_operand_or_representation() {
        let config = BackendConfig::new(8, 6, 0, 0);
        for case in 0..9 {
            let mut block = spill(config);
            match case {
                0 => {
                    if let MachineInstKind::Load { dst, .. } = &mut block.ops[2].kind {
                        *dst = MachineReg(4);
                    }
                }
                1 => {
                    if let MachineInstKind::IntBinary { rhs, .. } = &mut block.ops[1].kind {
                        *rhs = MachineValue::Reg(MachineReg(5));
                    }
                }
                2 => {
                    if let MachineInstKind::IntBinary { dst, .. } = &mut block.ops[1].kind {
                        *dst = MachineReg(5);
                    }
                }
                3 => {
                    if let MachineInstKind::Load { width, .. } = &mut block.ops[2].kind {
                        *width = MachineMemWidth::U32;
                    }
                }
                4 => {
                    if let MachineInstKind::Load { addr, .. } = &mut block.ops[2].kind {
                        addr.offset += 8;
                    }
                }
                5 => {
                    if let MachineInstKind::Load { extension, .. } = &mut block.ops[2].kind {
                        *extension = MachineLoadExtension::SignExtend;
                    }
                }
                6 => {
                    for inst in &mut block.ops {
                        match &mut inst.kind {
                            MachineInstKind::Load { addr, .. }
                            | MachineInstKind::Store { addr, .. } => addr.base = MachineReg(6),
                            _ => {}
                        }
                    }
                }
                7 => {
                    if let MachineInstKind::Store { width, .. } = &mut block.ops[0].kind {
                        *width = MachineMemWidth::U32;
                    }
                }
                8 => {
                    if let MachineInstKind::Store { src, .. } = &mut block.ops[0].kind {
                        *src = MachineValue::Reg(MACHINE_FP_REG);
                    }
                    if let MachineInstKind::IntBinary { dst, lhs, .. } = &mut block.ops[1].kind {
                        *dst = MACHINE_FP_REG;
                        *lhs = MachineValue::Reg(MACHINE_FP_REG);
                    }
                }
                _ => unreachable!(),
            }
            let original = block.clone();
            preserve_frame_value_before_clobber(&mut block, config);
            assert_eq!(block, original, "unsafe case {case}");
        }
    }
}
