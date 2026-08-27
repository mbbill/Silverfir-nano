//! Promote cached direct-call frame reloads into preserved GP lanes.
//!
//! A cache-heavy caller can already occupy every preserved lane. At an
//! individual call, however, some of those cached values have just been
//! published to their frame homes and are dead on the success edge. Such a
//! lane can carry a still-live volatile cache through that call. This second
//! form keeps every frame store, follows only a unique identity-jump chain,
//! and replaces the first exact cached reload before any overlapping access.

use crate::vm::jit::{
    backend::BackendConfig,
    machine::machine_ir::{
        is_dynamic_reg, is_gp_reg, is_preserved_dynamic_reg, MachineAddr, MachineArgSrc,
        MachineBlock, MachineBlockId, MachineBlockParam, MachineBranchCond, MachineCallArgs,
        MachineCallLaneArg, MachineCallResults, MachineCallTarget, MachineCompareKind, MachineInst,
        MachineInstKind, MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineProgram,
        MachineReg, MachineRegOwner, MachineResultDst, MachineSign, MachineStorageType,
        MachineTerminator, MachineTrapKind, MachineValue, NonRefCachedBinding, MACHINE_FP_REG,
    },
};

use super::{copy_propagate, helpers};

struct CachedCallPromotion {
    source: usize,
    chain: crate::collections::Vec<usize>,
    carries: crate::collections::Vec<CachedCarry>,
}

#[derive(Clone, Copy)]
struct CachedCarry {
    spilled: MachineReg,
    carry: MachineReg,
    ty: MachineStorageType,
    reload: MachineReg,
    reload_op: usize,
}

#[derive(Clone, Copy)]
struct CachedSnapshot {
    addr: MachineAddr,
    spilled: MachineReg,
    ty: MachineStorageType,
}

#[derive(Clone, Copy)]
struct CachedReload {
    snapshot: CachedSnapshot,
    reload: MachineReg,
    reload_op: usize,
}

const MAX_CACHED_CALL_CHAIN_BLOCKS: usize = 4;

fn promote_cached_spills_with_dead_preserved_lanes(
    program: &mut MachineProgram,
    config: BackendConfig,
    non_ref_cached_bindings: &[NonRefCachedBinding],
) {
    let mut promotions = crate::collections::Vec::new();
    for source in 0..program.blocks.len() {
        if let Some(promotion) =
            find_cached_call_promotion(program, config, non_ref_cached_bindings, source)
        {
            promotions.push(promotion);
        }
    }
    for promotion in promotions {
        apply_cached_call_promotion(program, config, promotion);
    }
}

fn find_cached_call_promotion(
    program: &MachineProgram,
    config: BackendConfig,
    non_ref_cached_bindings: &[NonRefCachedBinding],
    source_index: usize,
) -> Option<CachedCallPromotion> {
    let source = program.blocks.get(source_index)?;
    let MachineTerminator::Call {
        target: MachineCallTarget::Direct(_),
        frame_delta,
        args,
        results,
        success,
    } = &source.terminator
    else {
        return None;
    };
    if !call_args_are_register_only(args)
        || !call_results_are_register_only(results)
        || helpers::terminator_uses_reg(&source.terminator, MACHINE_FP_REG)
    {
        return None;
    }

    let success_index = success.target.as_usize();
    let success_block = program.blocks.get(success_index)?;
    if success_index == source_index
        || success.target == program.entry
        || success_block.id != success.target
        || incoming_edge_count(program, success.target) != 1
        || !is_identity_edge(success, success_block)
    {
        return None;
    }

    let snapshots =
        cached_snapshots_before_call(source, *frame_delta, config, non_ref_cached_bindings);
    if snapshots.is_empty() {
        return None;
    }
    let (chain, reloads) = find_cached_reload_chain(program, config, success.target, snapshots)?;

    let mut carriers = source
        .params
        .iter()
        .filter_map(|param| {
            let binding = unique_cached_binding(non_ref_cached_bindings, source.id, param.reg)?;
            (matches!(param.owner, MachineRegOwner::CachedCell)
                && matches!(
                    param.ty,
                    MachineStorageType::GpWord | MachineStorageType::GpI64
                )
                && binding.ty == param.ty
                && is_preserved_dynamic_reg(param.reg, config)
                && is_gp_reg(param.reg, config)
                && !helpers::terminator_uses_reg(&source.terminator, param.reg)
                && !terminator_defines_reg(results, param.reg)
                && has_final_cached_snapshot(source, binding, *frame_delta, config)
                && chain.iter().all(|&block_index| {
                    !block_mentions_reg(&program.blocks[block_index], param.reg)
                }))
            .then_some(param.reg)
        })
        .collect::<crate::collections::Vec<_>>();
    carriers.sort_by_key(|reg| reg.0);
    carriers.dedup();
    if carriers.is_empty() {
        return None;
    }

    let mut carries = crate::collections::Vec::new();
    for reload in reloads {
        let carrier_index = carriers
            .iter()
            .position(|&carrier| carrier != reload.snapshot.spilled && carrier != reload.reload);
        let Some(carrier_index) = carrier_index else {
            continue;
        };
        let carry = carriers.remove(carrier_index);
        carries.push(CachedCarry {
            spilled: reload.snapshot.spilled,
            carry,
            ty: reload.snapshot.ty,
            reload: reload.reload,
            reload_op: reload.reload_op,
        });
        if carriers.is_empty() {
            break;
        }
    }
    (!carries.is_empty()).then_some(CachedCallPromotion {
        source: source_index,
        chain,
        carries,
    })
}

fn cached_snapshots_before_call(
    source: &MachineBlock,
    frame_delta: u32,
    config: BackendConfig,
    non_ref_cached_bindings: &[NonRefCachedBinding],
) -> crate::collections::Vec<CachedSnapshot> {
    let mut snapshots = crate::collections::Vec::new();
    for (op_index, inst) in source.ops.iter().enumerate() {
        let MachineInstKind::Store {
            ty,
            addr,
            width: MachineMemWidth::U64,
            src: MachineValue::Reg(spilled),
        } = &inst.kind
        else {
            continue;
        };
        if !cached_snapshot_is_non_ref(
            source,
            op_index,
            *ty,
            *addr,
            *spilled,
            non_ref_cached_bindings,
        ) || addr.base != MACHINE_FP_REG
            || !is_dynamic_reg(*spilled, config)
            || !is_gp_reg(*spilled, config)
            || is_preserved_dynamic_reg(*spilled, config)
            || owner_before_op(source, *spilled, op_index) != Some(MachineRegOwner::CachedCell)
            || !frame_slot_is_private(*addr, frame_delta)
            || source.ops[op_index + 1..].iter().any(|later| {
                helpers::inst_defines(&later.kind, *spilled)
                    || cached_chain_inst_is_opaque(&later.kind, config)
                    || inst_may_touch_frame_slot(&later.kind, *addr, config)
            })
            || snapshots
                .iter()
                .any(|snapshot: &CachedSnapshot| snapshot.addr == *addr)
        {
            continue;
        }
        snapshots.push(CachedSnapshot {
            addr: *addr,
            spilled: *spilled,
            ty: *ty,
        });
    }
    snapshots
}

fn find_cached_reload_chain(
    program: &MachineProgram,
    config: BackendConfig,
    start: MachineBlockId,
    mut active: crate::collections::Vec<CachedSnapshot>,
) -> Option<(
    crate::collections::Vec<usize>,
    crate::collections::Vec<CachedReload>,
)> {
    let mut chain = crate::collections::Vec::new();
    let mut current = start;
    for _ in 0..MAX_CACHED_CALL_CHAIN_BLOCKS {
        let current_index = current.as_usize();
        let block = program.blocks.get(current_index)?;
        if current == program.entry
            || block.id != current
            || incoming_edge_count(program, current) != 1
            || chain.contains(&current_index)
        {
            return None;
        }
        chain.push(current_index);

        let mut reloads = crate::collections::Vec::new();
        for (op_index, inst) in block.ops.iter().enumerate() {
            if cached_chain_inst_is_opaque(&inst.kind, config) {
                return None;
            }
            let mut snapshot_index = 0;
            while snapshot_index < active.len() {
                let snapshot = active[snapshot_index];
                if !inst_may_touch_frame_slot(&inst.kind, snapshot.addr, config) {
                    snapshot_index += 1;
                    continue;
                }
                if let MachineInstKind::Load {
                    owner: MachineRegOwner::CachedCell,
                    ty,
                    dst,
                    addr,
                    width: MachineMemWidth::U64,
                    extension: MachineLoadExtension::None,
                } = &inst.kind
                {
                    if *ty == snapshot.ty
                        && *addr == snapshot.addr
                        && is_dynamic_reg(*dst, config)
                        && is_gp_reg(*dst, config)
                        && *dst != snapshot.spilled
                        && reload_has_no_live_in_alias(block, op_index, *dst)
                        && helpers::reg_live_after(
                            &block.ops[op_index + 1..],
                            &block.terminator,
                            *dst,
                        )
                    {
                        reloads.push(CachedReload {
                            snapshot,
                            reload: *dst,
                            reload_op: op_index,
                        });
                    }
                }
                active.remove(snapshot_index);
            }
        }
        if !reloads.is_empty() {
            if !matches!(block.terminator, MachineTerminator::Jump(_)) {
                return None;
            }
            return Some((chain, reloads));
        }
        if active.is_empty() {
            return None;
        }

        let MachineTerminator::Jump(edge) = &block.terminator else {
            return None;
        };
        let next_index = edge.target.as_usize();
        let next = program.blocks.get(next_index)?;
        if edge.target == program.entry
            || next.id != edge.target
            || incoming_edge_count(program, edge.target) != 1
            || !is_identity_edge(edge, next)
        {
            return None;
        }
        current = edge.target;
    }
    None
}

fn apply_cached_call_promotion(
    program: &mut MachineProgram,
    config: BackendConfig,
    promotion: CachedCallPromotion,
) {
    let source = &mut program.blocks[promotion.source];
    for carry in &promotion.carries {
        source.ops.push(MachineInst {
            kind: MachineInstKind::Move {
                owner: MachineRegOwner::CachedCell,
                ty: carry.ty,
                dst: carry.carry,
                src: MachineValue::Reg(carry.spilled),
            },
        });
    }
    let MachineTerminator::Call { success, .. } = &mut source.terminator else {
        unreachable!("cached call promotion source must remain a call");
    };
    success.args.extend(
        promotion
            .carries
            .iter()
            .map(|carry| MachineValue::Reg(carry.carry)),
    );

    for (chain_position, &block_index) in promotion.chain.iter().enumerate() {
        let block = &mut program.blocks[block_index];
        block
            .params
            .extend(promotion.carries.iter().map(|carry| MachineBlockParam {
                reg: carry.carry,
                ty: carry.ty,
                owner: MachineRegOwner::CachedCell,
            }));
        if chain_position + 1 < promotion.chain.len() {
            let MachineTerminator::Jump(edge) = &mut block.terminator else {
                unreachable!("cached call promotion chain must contain only jumps");
            };
            edge.args.extend(
                promotion
                    .carries
                    .iter()
                    .map(|carry| MachineValue::Reg(carry.carry)),
            );
        }
    }

    let reload_block_index = *promotion
        .chain
        .last()
        .expect("cached call promotion must have a reload block");
    let reload_block = &mut program.blocks[reload_block_index];
    let mut carries = promotion.carries;
    carries.sort_by(|lhs, rhs| rhs.reload_op.cmp(&lhs.reload_op));
    for carry in carries {
        remove_cached_reload(reload_block, carry, config);
    }
}

fn remove_cached_reload(block: &mut MachineBlock, carry: CachedCarry, config: BackendConfig) {
    let removed = block.ops.remove(carry.reload_op);
    debug_assert!(matches!(
        removed.kind,
        MachineInstKind::Load {
            owner: MachineRegOwner::CachedCell,
            dst,
            width: MachineMemWidth::U64,
            ..
        } if dst == carry.reload
    ));

    let mut reaches_terminator = true;
    let mut op_has_use = false;
    for inst in &block.ops {
        op_has_use |= helpers::inst_uses_value(&inst.kind, carry.reload);
        if helpers::inst_defines(&inst.kind, carry.reload) {
            reaches_terminator = false;
            break;
        }
    }
    let mut edge_rewrites = 0usize;
    if reaches_terminator {
        let MachineTerminator::Jump(edge) = &mut block.terminator else {
            unreachable!("cached call promotion reload block must end in a jump");
        };
        for arg in &mut edge.args {
            if matches!(arg, MachineValue::Reg(reg) if *reg == carry.reload) {
                *arg = MachineValue::Reg(carry.carry);
                edge_rewrites += 1;
            }
        }
    }
    let rewrote_ops = op_has_use
        && copy_propagate::replace_initial_reg_uses(
            block,
            carry.reload,
            carry.carry,
            config.total_reg_count() as usize,
        );
    debug_assert!(rewrote_ops || edge_rewrites != 0);
}

fn reload_has_no_live_in_alias(block: &MachineBlock, reload_op: usize, reg: MachineReg) -> bool {
    !block.params.iter().any(|param| param.reg == reg)
        && block.ops[..reload_op].iter().all(|inst| {
            !helpers::inst_defines(&inst.kind, reg) && !helpers::inst_uses_value(&inst.kind, reg)
        })
}

fn has_final_cached_snapshot(
    block: &MachineBlock,
    binding: NonRefCachedBinding,
    frame_delta: u32,
    config: BackendConfig,
) -> bool {
    let reg = binding.reg;
    let addr = cached_home_addr(binding.home);
    for (op_index, inst) in block.ops.iter().enumerate().rev() {
        if let MachineInstKind::Store {
            ty,
            addr: store_addr,
            width: MachineMemWidth::U64,
            src: MachineValue::Reg(src),
        } = inst.kind
        {
            if src == reg
                && ty == binding.ty
                && store_addr == addr
                && frame_slot_is_private(addr, frame_delta)
                && owner_before_op(block, reg, op_index) == Some(MachineRegOwner::CachedCell)
                && is_dynamic_reg(reg, config)
            {
                return block.ops[..op_index]
                    .iter()
                    .all(|earlier| !helpers::inst_defines(&earlier.kind, reg))
                    && block.ops[op_index + 1..].iter().all(|later| {
                        !helpers::inst_defines(&later.kind, reg)
                            && !cached_chain_inst_is_opaque(&later.kind, config)
                            && !inst_may_touch_frame_slot(&later.kind, addr, config)
                    });
            }
        }
    }
    false
}

fn cached_snapshot_is_non_ref(
    source: &MachineBlock,
    op_index: usize,
    ty: MachineStorageType,
    addr: MachineAddr,
    reg: MachineReg,
    non_ref_cached_bindings: &[NonRefCachedBinding],
) -> bool {
    match ty {
        MachineStorageType::GpI64 => true,
        MachineStorageType::GpWord => {
            source.ops[..op_index]
                .iter()
                .all(|earlier| !helpers::inst_defines(&earlier.kind, reg))
                && unique_cached_binding(non_ref_cached_bindings, source.id, reg).is_some_and(
                    |binding| binding.ty == ty && cached_home_addr(binding.home) == addr,
                )
        }
        MachineStorageType::Fp32 | MachineStorageType::Fp64 | MachineStorageType::V128 => false,
    }
}

fn unique_cached_binding(
    bindings: &[NonRefCachedBinding],
    block: MachineBlockId,
    reg: MachineReg,
) -> Option<NonRefCachedBinding> {
    let key = (block.as_usize(), reg.0);
    let start = bindings.partition_point(|binding| (binding.block.as_usize(), binding.reg.0) < key);
    let end = bindings.partition_point(|binding| (binding.block.as_usize(), binding.reg.0) <= key);
    (end == start + 1).then(|| bindings[start])
}

fn cached_home_addr(home: crate::vm::jit::middle::frame::FrameSlot) -> MachineAddr {
    MachineAddr {
        base: MACHINE_FP_REG,
        offset: i32::from(home.0) * 8,
    }
}

fn frame_slot_is_private(addr: MachineAddr, frame_delta: u32) -> bool {
    let Ok(offset) = u32::try_from(addr.offset) else {
        return false;
    };
    offset
        .checked_add(8)
        .is_some_and(|slot_end| slot_end <= frame_delta)
}

fn inst_may_touch_frame_slot(
    kind: &MachineInstKind,
    slot: MachineAddr,
    config: BackendConfig,
) -> bool {
    match kind {
        MachineInstKind::Load { addr, width, .. } | MachineInstKind::Store { addr, width, .. } => {
            helpers::addrs_overlap(slot, MachineMemWidth::U64, *addr, *width)
        }
        _ => helpers::inst_uses_value(kind, MACHINE_FP_REG) && !is_native_stack_guard(kind, config),
    }
}

fn cached_chain_inst_is_opaque(kind: &MachineInstKind, config: BackendConfig) -> bool {
    if helpers::inst_defines(kind, MACHINE_FP_REG)
        || matches!(
            kind,
            MachineInstKind::Store {
                src: MachineValue::Reg(MACHINE_FP_REG),
                ..
            } | MachineInstKind::CallRuntime(_)
                | MachineInstKind::EhThrow { .. }
                | MachineInstKind::EhThrowRef { .. }
                | MachineInstKind::EhAllocExnRef { .. }
        )
    {
        return true;
    }
    match kind {
        MachineInstKind::Load { .. } | MachineInstKind::Store { .. } => false,
        _ => helpers::inst_uses_value(kind, MACHINE_FP_REG) && !is_native_stack_guard(kind, config),
    }
}

fn owner_before_op(
    block: &MachineBlock,
    reg: MachineReg,
    op_index: usize,
) -> Option<MachineRegOwner> {
    let mut owner = block
        .params
        .iter()
        .find(|param| param.reg == reg)
        .map(|param| param.owner);
    for inst in &block.ops[..op_index] {
        if helpers::inst_defines(&inst.kind, reg) {
            owner = inst.kind.def_owner();
        }
    }
    owner
}

pub(super) fn promote_call_frame_spills_with_non_ref_cached_bindings(
    program: &mut MachineProgram,
    config: BackendConfig,
    non_ref_cached_bindings: &[NonRefCachedBinding],
) {
    if config.gp_unit_bytes != 8 {
        return;
    }

    promote_cached_spills_with_dead_preserved_lanes(program, config, non_ref_cached_bindings);
}

fn is_native_stack_guard(kind: &MachineInstKind, config: BackendConfig) -> bool {
    let gp_width = if config.gp_unit_bytes == 4 {
        MachineIntWidth::I32
    } else {
        MachineIntWidth::I64
    };
    matches!(
        kind,
        MachineInstKind::TrapIf {
            kind: MachineTrapKind::StackOverflow,
            cond: MachineBranchCond::IntCompare {
                width,
                kind: MachineCompareKind::Gt,
                sign: MachineSign::Unsigned,
                lhs: MachineValue::Reg(MACHINE_FP_REG),
                rhs: MachineValue::Reg(limit),
            },
        } if *width == gp_width && is_dynamic_reg(*limit, config) && is_gp_reg(*limit, config)
    )
}

fn call_args_are_register_only(args: &MachineCallArgs) -> bool {
    args.frame_params.slots == 0
        && args.lane_args.iter().all(|arg| match arg {
            MachineCallLaneArg::Gp { src, .. } | MachineCallLaneArg::Fp { src, .. } => {
                matches!(src, MachineArgSrc::Reg(_))
            }
            MachineCallLaneArg::GpPair { src, .. } => {
                matches!(src.lo, MachineArgSrc::Reg(_)) && matches!(src.hi, MachineArgSrc::Reg(_))
            }
        })
}

fn call_results_are_register_only(results: &MachineCallResults) -> bool {
    match results {
        MachineCallResults::None => true,
        MachineCallResults::ScalarGp { dst, .. } | MachineCallResults::ScalarFp { dst, .. } => {
            matches!(dst, MachineResultDst::Reg(_))
        }
        MachineCallResults::ScalarGpPair { lo, hi } => {
            matches!(lo, MachineResultDst::Reg(_)) && matches!(hi, MachineResultDst::Reg(_))
        }
        MachineCallResults::FrameFallback { .. } => false,
    }
}

fn is_identity_edge(
    edge: &crate::vm::jit::machine::machine_ir::MachineEdge,
    target: &MachineBlock,
) -> bool {
    edge.args.len() == target.params.len()
        && edge
            .args
            .iter()
            .zip(&target.params)
            .all(|(arg, param)| matches!(arg, MachineValue::Reg(reg) if *reg == param.reg))
}

fn incoming_edge_count(program: &MachineProgram, target: MachineBlockId) -> usize {
    program
        .blocks
        .iter()
        .map(|block| terminator_edge_count(&block.terminator, target))
        .sum()
}

fn terminator_edge_count(terminator: &MachineTerminator, target: MachineBlockId) -> usize {
    match terminator {
        MachineTerminator::Jump(edge) => usize::from(edge.target == target),
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => usize::from(then_edge.target == target) + usize::from(else_edge.target == target),
        MachineTerminator::JumpTable { entries, .. } => {
            entries.iter().filter(|edge| edge.target == target).count()
        }
        MachineTerminator::Call { success, .. } => usize::from(success.target == target),
        MachineTerminator::TailCall { .. }
        | MachineTerminator::Return
        | MachineTerminator::ReturnScalar { .. }
        | MachineTerminator::Trap { .. } => 0,
    }
}

fn block_mentions_reg(block: &MachineBlock, reg: MachineReg) -> bool {
    if block.params.iter().any(|param| param.reg == reg) {
        return true;
    }
    for inst in &block.ops {
        if helpers::inst_defines(&inst.kind, reg) || helpers::inst_uses_value(&inst.kind, reg) {
            return true;
        }
    }
    helpers::terminator_uses_reg(&block.terminator, reg)
        || match &block.terminator {
            MachineTerminator::Call { results, .. } => terminator_defines_reg(results, reg),
            _ => false,
        }
}

fn terminator_defines_reg(results: &MachineCallResults, reg: MachineReg) -> bool {
    match results {
        MachineCallResults::ScalarGp { dst, .. } | MachineCallResults::ScalarFp { dst, .. } => {
            matches!(dst, MachineResultDst::Reg(dst) if *dst == reg)
        }
        MachineCallResults::ScalarGpPair { lo, hi } => {
            matches!(lo, MachineResultDst::Reg(dst) if *dst == reg)
                || matches!(hi, MachineResultDst::Reg(dst) if *dst == reg)
        }
        MachineCallResults::None | MachineCallResults::FrameFallback { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        collections,
        vm::jit::{
            machine::machine_ir::{
                MachineCallLaneArg, MachineCallRuntime, MachineConstId, MachineEdge, MachineFuncId,
                MachineIntBinaryOp, MachineIntWidth, MachineResultSrc, MachineReturnValue,
            },
            middle::frame::FrameSlot,
        },
    };

    const VALUE: MachineReg = MachineReg(4);
    const TMP: MachineReg = MachineReg(5);
    const ARG: MachineReg = MachineReg(6);
    const CARRY: MachineReg = MachineReg(7);
    const CARRY2: MachineReg = MachineReg(8);

    fn config() -> BackendConfig {
        BackendConfig::with_volatility(8, 3, 1, 0, 0, 0, 1, 0, true, 3)
    }

    fn config_two_preserved() -> BackendConfig {
        BackendConfig::with_volatility(8, 3, 2, 0, 0, 0, 1, 0, true, 3)
    }

    fn cached_binding(reg: MachineReg, home: u16, ty: MachineStorageType) -> NonRefCachedBinding {
        NonRefCachedBinding {
            block: MachineBlockId(0),
            reg,
            home: FrameSlot(home),
            ty,
        }
    }

    fn promote_dead_preserved_chain(program: &mut MachineProgram, config: BackendConfig) {
        promote_call_frame_spills_with_non_ref_cached_bindings(
            program,
            config,
            &[cached_binding(CARRY, 1, MachineStorageType::GpI64)],
        );
    }

    fn promote_two_dead_preserved_carriers(program: &mut MachineProgram, config: BackendConfig) {
        promote_call_frame_spills_with_non_ref_cached_bindings(
            program,
            config,
            &[
                cached_binding(CARRY, 4, MachineStorageType::GpI64),
                cached_binding(CARRY2, 5, MachineStorageType::GpI64),
            ],
        );
    }

    fn addr(offset: i32) -> MachineAddr {
        MachineAddr {
            base: MACHINE_FP_REG,
            offset,
        }
    }

    fn store(offset: i32, src: MachineReg) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpI64,
                addr: addr(offset),
                width: MachineMemWidth::U64,
                src: MachineValue::Reg(src),
            },
        }
    }

    fn load(offset: i32, dst: MachineReg, owner: MachineRegOwner) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::Load {
                owner,
                ty: MachineStorageType::GpI64,
                dst,
                addr: addr(offset),
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
            },
        }
    }

    fn binary(
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: MachineReg,
        rhs: MachineValue,
    ) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: MachineIntWidth::I64,
                op,
                dst,
                lhs: MachineValue::Reg(lhs),
                rhs,
            },
        }
    }

    fn direct_call(
        frame_delta: u32,
        arg: MachineReg,
        result: Option<MachineReg>,
        success_target: u32,
        success_args: collections::Vec<MachineValue>,
    ) -> MachineTerminator {
        MachineTerminator::Call {
            target: MachineCallTarget::Direct(MachineFuncId(0)),
            frame_delta,
            args: MachineCallArgs {
                frame_params: Default::default(),
                lane_args: collections::vec![MachineCallLaneArg::Gp {
                    param_index: 0,
                    lane: 0,
                    src: MachineArgSrc::Reg(arg),
                    ty: MachineStorageType::GpI64,
                }],
            },
            results: result.map_or(MachineCallResults::None, |dst| {
                MachineCallResults::ScalarGp {
                    dst: MachineResultDst::Reg(dst),
                    ty: MachineStorageType::GpI64,
                }
            }),
            success: MachineEdge {
                target: MachineBlockId(success_target),
                args: success_args,
            },
        }
    }

    fn scalar_return(reg: MachineReg) -> MachineTerminator {
        MachineTerminator::ReturnScalar {
            value: MachineReturnValue::ScalarGp {
                src: MachineResultSrc::Reg(reg),
                ty: MachineStorageType::GpI64,
            },
        }
    }

    fn dead_preserved_chain() -> MachineProgram {
        MachineProgram {
            entry: MachineBlockId(0),
            fp_reg_init_widths: collections::Vec::new(),
            blocks: collections::vec![
                MachineBlock {
                    id: MachineBlockId(0),
                    params: collections::vec![
                        MachineBlockParam::gp_i64(VALUE).with_owner(MachineRegOwner::CachedCell),
                        MachineBlockParam::gp_i64(CARRY).with_owner(MachineRegOwner::CachedCell),
                    ],
                    ops: collections::vec![store(8, CARRY), store(0, VALUE)],
                    terminator: direct_call(32, VALUE, None, 1, collections::Vec::new()),
                },
                MachineBlock {
                    id: MachineBlockId(1),
                    params: collections::Vec::new(),
                    ops: collections::Vec::new(),
                    terminator: MachineTerminator::Jump(MachineEdge {
                        target: MachineBlockId(2),
                        args: collections::Vec::new(),
                    }),
                },
                MachineBlock {
                    id: MachineBlockId(2),
                    params: collections::Vec::new(),
                    ops: collections::vec![load(0, TMP, MachineRegOwner::CachedCell)],
                    terminator: MachineTerminator::Jump(MachineEdge {
                        target: MachineBlockId(3),
                        args: collections::vec![MachineValue::Reg(TMP)],
                    }),
                },
                MachineBlock {
                    id: MachineBlockId(3),
                    params: collections::vec![MachineBlockParam::gp_i64(VALUE)],
                    ops: collections::Vec::new(),
                    terminator: scalar_return(VALUE),
                },
            ],
        }
    }

    fn gp_word_dead_preserved_chain() -> MachineProgram {
        let mut program = dead_preserved_chain();
        program.blocks[0].params[0].ty = MachineStorageType::GpWord;
        program.blocks[0].params[1].ty = MachineStorageType::GpWord;
        let MachineInstKind::Store { ty, .. } = &mut program.blocks[0].ops[0].kind else {
            panic!("expected carrier publication store");
        };
        *ty = MachineStorageType::GpWord;
        let MachineInstKind::Store { ty, .. } = &mut program.blocks[0].ops[1].kind else {
            panic!("expected cached snapshot store");
        };
        *ty = MachineStorageType::GpWord;
        let MachineInstKind::Load { ty, .. } = &mut program.blocks[2].ops[0].kind else {
            panic!("expected cached reload");
        };
        *ty = MachineStorageType::GpWord;
        program.blocks[3].params[0].ty = MachineStorageType::GpWord;
        program
    }

    fn two_dead_preserved_carriers() -> MachineProgram {
        MachineProgram {
            entry: MachineBlockId(0),
            fp_reg_init_widths: collections::Vec::new(),
            blocks: collections::vec![
                MachineBlock {
                    id: MachineBlockId(0),
                    params: collections::vec![
                        MachineBlockParam::gp_i64(VALUE).with_owner(MachineRegOwner::CachedCell),
                        MachineBlockParam::gp_i64(TMP).with_owner(MachineRegOwner::CachedCell),
                        MachineBlockParam::gp_i64(CARRY).with_owner(MachineRegOwner::CachedCell),
                        MachineBlockParam::gp_i64(CARRY2).with_owner(MachineRegOwner::CachedCell),
                    ],
                    ops: collections::vec![
                        store(32, CARRY),
                        store(40, CARRY2),
                        store(0, VALUE),
                        store(16, TMP),
                    ],
                    terminator: direct_call(64, VALUE, None, 1, collections::Vec::new()),
                },
                MachineBlock {
                    id: MachineBlockId(1),
                    params: collections::Vec::new(),
                    ops: collections::Vec::new(),
                    terminator: MachineTerminator::Jump(MachineEdge {
                        target: MachineBlockId(2),
                        args: collections::Vec::new(),
                    }),
                },
                MachineBlock {
                    id: MachineBlockId(2),
                    params: collections::Vec::new(),
                    ops: collections::vec![
                        load(0, TMP, MachineRegOwner::CachedCell),
                        load(16, ARG, MachineRegOwner::CachedCell),
                    ],
                    terminator: MachineTerminator::Jump(MachineEdge {
                        target: MachineBlockId(3),
                        args: collections::vec![MachineValue::Reg(TMP), MachineValue::Reg(ARG),],
                    }),
                },
                MachineBlock {
                    id: MachineBlockId(3),
                    params: collections::vec![
                        MachineBlockParam::gp_i64(VALUE),
                        MachineBlockParam::gp_i64(TMP),
                    ],
                    ops: collections::vec![binary(
                        MachineIntBinaryOp::Add,
                        VALUE,
                        VALUE,
                        MachineValue::Reg(TMP),
                    )],
                    terminator: scalar_return(VALUE),
                },
            ],
        }
    }

    fn two_cached_calls_reusing_one_carrier() -> MachineProgram {
        MachineProgram {
            entry: MachineBlockId(0),
            fp_reg_init_widths: collections::Vec::new(),
            blocks: collections::vec![
                MachineBlock {
                    id: MachineBlockId(0),
                    params: collections::vec![
                        MachineBlockParam::gp_i64(VALUE).with_owner(MachineRegOwner::CachedCell),
                        MachineBlockParam::gp_i64(CARRY).with_owner(MachineRegOwner::CachedCell),
                    ],
                    ops: collections::vec![store(8, CARRY), store(0, VALUE)],
                    terminator: direct_call(
                        32,
                        ARG,
                        None,
                        1,
                        collections::vec![MachineValue::Reg(VALUE)],
                    ),
                },
                MachineBlock {
                    id: MachineBlockId(1),
                    params: collections::vec![
                        MachineBlockParam::gp_i64(VALUE).with_owner(MachineRegOwner::CachedCell),
                    ],
                    ops: collections::vec![load(0, TMP, MachineRegOwner::CachedCell)],
                    terminator: MachineTerminator::Jump(MachineEdge {
                        target: MachineBlockId(2),
                        args: collections::vec![MachineValue::Reg(VALUE), MachineValue::Reg(TMP),],
                    }),
                },
                MachineBlock {
                    id: MachineBlockId(2),
                    params: collections::vec![
                        MachineBlockParam::gp_i64(VALUE).with_owner(MachineRegOwner::CachedCell),
                        MachineBlockParam::gp_i64(CARRY).with_owner(MachineRegOwner::CachedCell),
                    ],
                    ops: collections::vec![store(0, CARRY), store(16, VALUE)],
                    terminator: direct_call(32, ARG, None, 3, collections::Vec::new()),
                },
                MachineBlock {
                    id: MachineBlockId(3),
                    params: collections::Vec::new(),
                    ops: collections::vec![load(16, TMP, MachineRegOwner::CachedCell)],
                    terminator: MachineTerminator::Jump(MachineEdge {
                        target: MachineBlockId(4),
                        args: collections::vec![MachineValue::Reg(TMP)],
                    }),
                },
                MachineBlock {
                    id: MachineBlockId(4),
                    params: collections::vec![MachineBlockParam::gp_i64(VALUE)],
                    ops: collections::Vec::new(),
                    terminator: scalar_return(VALUE),
                },
            ],
        }
    }

    #[test]
    fn reuses_a_dead_preserved_cache_across_a_two_jump_continuation() {
        let mut program = dead_preserved_chain();

        promote_dead_preserved_chain(&mut program, config());

        assert!(matches!(
            program.blocks[0].ops.as_slice(),
            [
                MachineInst {
                    kind: MachineInstKind::Store { .. }
                },
                MachineInst {
                    kind: MachineInstKind::Store { .. }
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        owner: MachineRegOwner::CachedCell,
                        dst: CARRY,
                        src: MachineValue::Reg(VALUE),
                        ..
                    }
                }
            ]
        ));
        let MachineTerminator::Call { success, .. } = &program.blocks[0].terminator else {
            panic!("expected direct call");
        };
        assert_eq!(success.args, collections::vec![MachineValue::Reg(CARRY)]);
        assert_eq!(program.blocks[1].params[0].reg, CARRY);
        let MachineTerminator::Jump(first_jump) = &program.blocks[1].terminator else {
            panic!("expected first continuation jump");
        };
        assert_eq!(first_jump.args, collections::vec![MachineValue::Reg(CARRY)]);
        assert_eq!(program.blocks[2].params[0].reg, CARRY);
        assert!(program.blocks[2].ops.is_empty());
        let MachineTerminator::Jump(reload_jump) = &program.blocks[2].terminator else {
            panic!("expected reload continuation jump");
        };
        assert_eq!(
            reload_jump.args,
            collections::vec![MachineValue::Reg(CARRY)]
        );
    }

    #[test]
    fn rewrites_a_cached_reload_used_by_an_ordinary_op() {
        let mut program = dead_preserved_chain();
        program.blocks[2].ops.push(binary(
            MachineIntBinaryOp::Add,
            ARG,
            TMP,
            MachineValue::Imm64(1),
        ));
        let MachineTerminator::Jump(edge) = &mut program.blocks[2].terminator else {
            panic!("expected reload continuation jump");
        };
        edge.args[0] = MachineValue::Reg(ARG);

        promote_dead_preserved_chain(&mut program, config());

        assert_eq!(program.blocks[2].ops.len(), 1);
        assert!(matches!(
            program.blocks[2].ops[0].kind,
            MachineInstKind::IntBinary {
                dst: ARG,
                lhs: MachineValue::Reg(CARRY),
                ..
            }
        ));
        let MachineTerminator::Jump(edge) = &program.blocks[2].terminator else {
            panic!("expected reload continuation jump");
        };
        assert_eq!(edge.args, collections::vec![MachineValue::Reg(ARG)]);
    }

    #[test]
    fn rewrites_a_cached_reload_used_by_an_op_and_jump_edge() {
        let mut program = dead_preserved_chain();
        program.blocks[2].ops.push(binary(
            MachineIntBinaryOp::Add,
            ARG,
            TMP,
            MachineValue::Imm64(1),
        ));

        promote_dead_preserved_chain(&mut program, config());

        assert_eq!(program.blocks[2].ops.len(), 1);
        assert!(matches!(
            program.blocks[2].ops[0].kind,
            MachineInstKind::IntBinary {
                dst: ARG,
                lhs: MachineValue::Reg(CARRY),
                ..
            }
        ));
        let MachineTerminator::Jump(edge) = &program.blocks[2].terminator else {
            panic!("expected reload continuation jump");
        };
        assert_eq!(edge.args, collections::vec![MachineValue::Reg(CARRY)]);
    }

    #[test]
    fn stops_cached_reload_rewrites_after_the_reload_reg_is_redefined() {
        let mut program = dead_preserved_chain();
        program.blocks[2].ops.extend([
            binary(MachineIntBinaryOp::Add, TMP, TMP, MachineValue::Imm64(1)),
            binary(MachineIntBinaryOp::Add, ARG, TMP, MachineValue::Imm64(2)),
        ]);

        promote_dead_preserved_chain(&mut program, config());

        assert_eq!(program.blocks[2].ops.len(), 2);
        assert!(matches!(
            program.blocks[2].ops[0].kind,
            MachineInstKind::IntBinary {
                dst: TMP,
                lhs: MachineValue::Reg(CARRY),
                ..
            }
        ));
        assert!(matches!(
            program.blocks[2].ops[1].kind,
            MachineInstKind::IntBinary {
                dst: ARG,
                lhs: MachineValue::Reg(TMP),
                ..
            }
        ));
        let MachineTerminator::Jump(edge) = &program.blocks[2].terminator else {
            panic!("expected reload continuation jump");
        };
        assert_eq!(edge.args, collections::vec![MachineValue::Reg(TMP)]);
    }

    #[test]
    fn reuses_two_dead_preserved_caches_for_two_cached_reloads() {
        let mut program = two_dead_preserved_carriers();

        promote_two_dead_preserved_carriers(&mut program, config_two_preserved());

        assert_eq!(program.blocks[0].ops.len(), 6);
        assert!(matches!(
            program.blocks[0].ops[4].kind,
            MachineInstKind::Move {
                dst: CARRY,
                src: MachineValue::Reg(VALUE),
                ..
            }
        ));
        assert!(matches!(
            program.blocks[0].ops[5].kind,
            MachineInstKind::Move {
                dst: CARRY2,
                src: MachineValue::Reg(TMP),
                ..
            }
        ));
        assert!(program.blocks[2].ops.is_empty());
        let MachineTerminator::Jump(edge) = &program.blocks[2].terminator else {
            panic!("expected reload continuation jump");
        };
        assert_eq!(
            edge.args,
            collections::vec![MachineValue::Reg(CARRY), MachineValue::Reg(CARRY2)]
        );
    }

    #[test]
    fn reuses_one_dead_preserved_cache_across_two_cached_call_promotions() {
        let mut program = two_cached_calls_reusing_one_carrier();

        promote_call_frame_spills_with_non_ref_cached_bindings(
            &mut program,
            config(),
            &[
                cached_binding(CARRY, 1, MachineStorageType::GpI64),
                NonRefCachedBinding {
                    block: MachineBlockId(2),
                    reg: CARRY,
                    home: FrameSlot(0),
                    ty: MachineStorageType::GpI64,
                },
            ],
        );

        let carry_moves = [0usize, 2]
            .into_iter()
            .filter(|&block| {
                matches!(
                    program.blocks[block].ops.last().map(|inst| &inst.kind),
                    Some(MachineInstKind::Move {
                        owner: MachineRegOwner::CachedCell,
                        dst: CARRY,
                        ..
                    })
                )
            })
            .count();
        assert_eq!(carry_moves, 2);
        assert!(program.blocks[1].ops.is_empty());
        assert!(program.blocks[3].ops.is_empty());
    }

    #[test]
    fn keeps_cached_reload_when_the_only_continuation_path_branches() {
        let mut program = dead_preserved_chain();
        program.blocks.push(MachineBlock {
            id: MachineBlockId(4),
            params: collections::Vec::new(),
            ops: collections::Vec::new(),
            terminator: MachineTerminator::Trap {
                kind: MachineTrapKind::Unreachable,
            },
        });
        program.blocks[1].terminator = MachineTerminator::Branch {
            cond: MachineBranchCond::Value(MachineValue::Imm64(1)),
            then_edge: MachineEdge {
                target: MachineBlockId(2),
                args: collections::Vec::new(),
            },
            else_edge: MachineEdge {
                target: MachineBlockId(4),
                args: collections::Vec::new(),
            },
        };
        let original = program.clone();

        promote_dead_preserved_chain(&mut program, config());

        assert_eq!(program, original);
    }

    #[test]
    fn keeps_cached_reload_when_the_chain_derives_a_frame_pointer_alias() {
        let mut program = dead_preserved_chain();
        program.blocks[1].ops.push(MachineInst {
            kind: MachineInstKind::Move {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpI64,
                dst: ARG,
                src: MachineValue::Reg(MACHINE_FP_REG),
            },
        });
        let original = program.clone();

        promote_dead_preserved_chain(&mut program, config());

        assert_eq!(program, original);
    }

    #[test]
    fn keeps_cached_reload_without_a_dead_preserved_carrier() {
        let mut program = dead_preserved_chain();
        let MachineTerminator::Call { success, .. } = &mut program.blocks[0].terminator else {
            panic!("expected direct call");
        };
        success.args.push(MachineValue::Reg(CARRY));
        program.blocks[1]
            .params
            .push(MachineBlockParam::gp_i64(CARRY).with_owner(MachineRegOwner::CachedCell));
        let original = program.clone();

        promote_dead_preserved_chain(&mut program, config());

        assert_eq!(program, original);
    }

    #[test]
    fn keeps_cached_reload_when_the_dead_cache_was_not_published() {
        let mut program = dead_preserved_chain();
        program.blocks[0].ops.remove(0);
        let original = program.clone();

        promote_dead_preserved_chain(&mut program, config());

        assert_eq!(program, original);
    }

    #[test]
    fn keeps_cached_reload_when_the_local_call_receives_the_frame_pointer() {
        let mut program = dead_preserved_chain();
        let MachineTerminator::Call { args, .. } = &mut program.blocks[0].terminator else {
            panic!("expected direct call");
        };
        let MachineCallLaneArg::Gp { src, .. } = &mut args.lane_args[0] else {
            panic!("expected scalar GP argument");
        };
        *src = MachineArgSrc::Reg(MACHINE_FP_REG);
        let original = program.clone();

        promote_dead_preserved_chain(&mut program, config());

        assert_eq!(program, original);
    }

    #[test]
    fn keeps_cached_reload_across_a_runtime_call_barrier_before_the_local_call() {
        let mut program = dead_preserved_chain();
        program.blocks[0].ops.push(MachineInst {
            kind: MachineInstKind::CallRuntime(MachineCallRuntime {
                metadata: MachineConstId(0),
            }),
        });
        let original = program.clone();

        promote_dead_preserved_chain(&mut program, config());

        assert_eq!(program, original);
    }

    #[test]
    fn keeps_cached_reload_when_the_carrier_publication_uses_the_wrong_home() {
        let mut program = dead_preserved_chain();
        let MachineInstKind::Store { addr, .. } = &mut program.blocks[0].ops[0].kind else {
            panic!("expected carrier publication store");
        };
        addr.offset = 16;
        let original = program.clone();

        promote_dead_preserved_chain(&mut program, config());

        assert_eq!(program, original);
    }

    #[test]
    fn keeps_cached_reload_when_the_carrier_home_is_overwritten_after_publication() {
        let mut program = dead_preserved_chain();
        program.blocks[0].ops.insert(1, store(8, ARG));
        let original = program.clone();

        promote_dead_preserved_chain(&mut program, config());

        assert_eq!(program, original);
    }

    #[test]
    fn keeps_cached_reload_when_the_carrier_was_redefined_before_publication() {
        let mut program = dead_preserved_chain();
        program.blocks[0].ops.insert(
            0,
            MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::CachedCell,
                    ty: MachineStorageType::GpI64,
                    dst: CARRY,
                    src: MachineValue::Reg(ARG),
                },
            },
        );
        let original = program.clone();

        promote_dead_preserved_chain(&mut program, config());

        assert_eq!(program, original);
    }

    #[test]
    fn keeps_cached_reload_for_wrong_or_conflicting_carrier_bindings() {
        let mut wrong_block = cached_binding(CARRY, 1, MachineStorageType::GpI64);
        wrong_block.block = MachineBlockId(1);
        let wrong_type = cached_binding(CARRY, 1, MachineStorageType::GpWord);
        let conflicting = collections::vec![
            cached_binding(CARRY, 1, MachineStorageType::GpI64),
            cached_binding(CARRY, 2, MachineStorageType::GpI64),
        ];
        for bindings in [
            collections::vec![wrong_block],
            collections::vec![wrong_type],
            conflicting,
        ] {
            let mut program = dead_preserved_chain();
            let original = program.clone();

            promote_call_frame_spills_with_non_ref_cached_bindings(
                &mut program,
                config(),
                &bindings,
            );

            assert_eq!(program, original);
        }
    }

    #[test]
    fn reuses_proven_numeric_gp_word_caches() {
        let mut program = gp_word_dead_preserved_chain();

        promote_call_frame_spills_with_non_ref_cached_bindings(
            &mut program,
            config(),
            &[
                cached_binding(VALUE, 0, MachineStorageType::GpWord),
                cached_binding(CARRY, 1, MachineStorageType::GpWord),
            ],
        );

        assert!(matches!(
            program.blocks[0].ops.last().map(|inst| &inst.kind),
            Some(MachineInstKind::Move {
                owner: MachineRegOwner::CachedCell,
                ty: MachineStorageType::GpWord,
                dst: CARRY,
                src: MachineValue::Reg(VALUE),
            })
        ));
        assert!(program.blocks[2].ops.is_empty());
    }

    #[test]
    fn keeps_gp_word_caches_without_non_ref_provenance() {
        let mut program = gp_word_dead_preserved_chain();
        let original = program.clone();

        promote_call_frame_spills_with_non_ref_cached_bindings(&mut program, config(), &[]);

        assert_eq!(program, original);
    }

    #[test]
    fn keeps_gp_word_cache_when_the_reloaded_home_is_not_proven() {
        let mut program = gp_word_dead_preserved_chain();
        let original = program.clone();

        promote_call_frame_spills_with_non_ref_cached_bindings(
            &mut program,
            config(),
            &[cached_binding(CARRY, 1, MachineStorageType::GpWord)],
        );

        assert_eq!(program, original);
    }

    #[test]
    fn keeps_gp_word_cache_when_the_carrier_home_is_not_proven() {
        let mut program = gp_word_dead_preserved_chain();
        let original = program.clone();

        promote_call_frame_spills_with_non_ref_cached_bindings(
            &mut program,
            config(),
            &[cached_binding(VALUE, 0, MachineStorageType::GpWord)],
        );

        assert_eq!(program, original);
    }

    #[test]
    fn keeps_gp_word_cache_when_the_frame_home_is_unaligned() {
        let mut program = gp_word_dead_preserved_chain();
        let MachineInstKind::Store { addr, .. } = &mut program.blocks[0].ops[1].kind else {
            panic!("expected cached snapshot store");
        };
        addr.offset = 4;
        let MachineInstKind::Load { addr, .. } = &mut program.blocks[2].ops[0].kind else {
            panic!("expected cached reload");
        };
        addr.offset = 4;
        let original = program.clone();

        promote_call_frame_spills_with_non_ref_cached_bindings(
            &mut program,
            config(),
            &[
                cached_binding(VALUE, 0, MachineStorageType::GpWord),
                cached_binding(CARRY, 1, MachineStorageType::GpWord),
            ],
        );

        assert_eq!(program, original);
    }

    #[test]
    fn keeps_gp_word_cache_when_the_bound_value_was_redefined_before_snapshot() {
        let mut program = gp_word_dead_preserved_chain();
        program.blocks[0].ops.insert(
            1,
            MachineInst {
                kind: MachineInstKind::Move {
                    owner: MachineRegOwner::CachedCell,
                    ty: MachineStorageType::GpWord,
                    dst: VALUE,
                    src: MachineValue::Reg(ARG),
                },
            },
        );
        let original = program.clone();

        promote_call_frame_spills_with_non_ref_cached_bindings(
            &mut program,
            config(),
            &[
                cached_binding(VALUE, 0, MachineStorageType::GpWord),
                cached_binding(CARRY, 1, MachineStorageType::GpWord),
            ],
        );

        assert_eq!(program, original);
    }
}
