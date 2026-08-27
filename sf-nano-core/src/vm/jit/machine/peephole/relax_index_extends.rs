//! Elide provably redundant index zero-extensions on indexed memory ops.
//!
//! `IndexedLoad`/`IndexedStore` with `MachineIndexExtend::ZeroExtend32`
//! obligate the backend to zero-extend the 32-bit index before use. On
//! backends where every 32-bit integer instruction already writes a
//! zero-extended destination (x86_64 r32 and AArch64 W-register writes clear
//! bits 63:32),
//! that obligation is vacuous whenever the index register's most recent
//! in-block definition is such an instruction — the extend can be relaxed
//! to `None` and the backend indexes the register directly, saving one
//! `mov` per memory access (three to four per iteration in the stream
//! kernels).
//!
//! Gated on `BackendConfig::gp32_defs_zero_extend`; backends whose 32-bit
//! ops sign-extend (riscv64) leave it off. AArch64 also benefits when an
//! indexed access has a non-zero offset: its register-offset memory encoding
//! cannot carry that offset, so proving the index clean avoids a separate
//! zero-extension before the adjusted-index address calculation.
//!
//! Soundness: a register is tracked as clean only from a whitelisted
//! 32-bit-form definition to its next redefinition. At block boundaries a
//! parameter is clean only when every reachable incoming edge supplies a
//! clean register. The whole-program analysis is a greatest fixed point over
//! reachable block parameters, which lets an inductive loop fact survive its
//! backedge while unknown entry values and mixed merges remain untrusted.

use crate::collections;
use crate::vm::jit::machine::machine_ir::{
    MachineBlock, MachineCallResults, MachineIndexExtend, MachineInstKind, MachineIntUnaryOp,
    MachineIntWidth, MachineLoadExtension, MachineMemWidth, MachineProgram, MachineReg,
    MachineResultDst, MachineStorageType, MachineTerminator, MachineValue,
};

use super::hoist_loop_address_bases::{block_index_for_id, visit_edges};

// Keep the cross-block proof proportional to the function being compiled.
// Local facts are still relaxed above these limits; larger functions retain
// explicit index extensions at block boundaries instead of paying for the
// whole-CFG fixed point during startup.
const MAX_CFG_INDEX_PROOF_BLOCKS: usize = 128;
const MAX_CFG_INDEX_PROOF_OPS: usize = 1200;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CleanRegs {
    // Every production GP register fits here (x86_64 currently uses ids
    // 0..12). Keep an exact sparse overflow for synthetic tests and future
    // register plans rather than folding high ids onto low bits.
    low: u64,
    overflow: collections::Vec<MachineReg>,
}

impl CleanRegs {
    fn block_param_top(block: &MachineBlock) -> Self {
        let mut result = Self::default();
        for param in &block.params {
            if matches!(
                param.ty,
                MachineStorageType::GpWord | MachineStorageType::GpI64
            ) {
                result.insert(param.reg);
            }
        }
        result
    }

    #[inline]
    fn contains(&self, reg: MachineReg) -> bool {
        if let Some(bit) = Self::low_bit(reg) {
            self.low & bit != 0
        } else {
            self.overflow.contains(&reg)
        }
    }

    fn insert(&mut self, reg: MachineReg) {
        if let Some(bit) = Self::low_bit(reg) {
            self.low |= bit;
        } else if !self.overflow.contains(&reg) {
            self.overflow.push(reg);
        }
    }

    fn remove(&mut self, reg: MachineReg) -> bool {
        if let Some(bit) = Self::low_bit(reg) {
            let was_present = self.low & bit != 0;
            self.low &= !bit;
            was_present
        } else if let Some(index) = self.overflow.iter().position(|candidate| *candidate == reg) {
            self.overflow.remove(index);
            true
        } else {
            false
        }
    }

    fn clear(&mut self) {
        self.low = 0;
        self.overflow.clear();
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.low == 0 && self.overflow.is_empty()
    }

    #[inline]
    fn low_bit(reg: MachineReg) -> Option<u64> {
        let shift = u32::from(reg.0);
        (shift < u64::BITS).then(|| 1u64 << shift)
    }
}

/// True when this instruction's destination is written by a 32-bit-form
/// instruction on a `gp32_defs_zero_extend` backend, leaving the upper
/// half zero.
fn def_zero_extends(kind: &MachineInstKind) -> Option<MachineReg> {
    match kind {
        MachineInstKind::IntBinary {
            width: MachineIntWidth::I32,
            dst,
            ..
        }
        | MachineInstKind::IntUnary {
            width: MachineIntWidth::I32,
            op:
                MachineIntUnaryOp::Clz
                | MachineIntUnaryOp::Ctz
                | MachineIntUnaryOp::Popcnt
                | MachineIntUnaryOp::Extend8S
                | MachineIntUnaryOp::Extend16S,
            dst,
            ..
        }
        | MachineInstKind::BitfieldExtractU {
            width: MachineIntWidth::I32,
            dst,
            ..
        }
        | MachineInstKind::IntBinaryShifted {
            width: MachineIntWidth::I32,
            dst,
            ..
        } => Some(*dst),
        MachineInstKind::IntCompare { dst, .. } | MachineInstKind::TestBits { dst, .. } => {
            Some(*dst)
        }
        MachineInstKind::Load {
            dst,
            width: MachineMemWidth::U8 | MachineMemWidth::U16 | MachineMemWidth::U32,
            extension: MachineLoadExtension::None | MachineLoadExtension::ZeroExtend,
            ..
        }
        | MachineInstKind::IndexedLoad {
            dst,
            width: MachineMemWidth::U8 | MachineMemWidth::U16 | MachineMemWidth::U32,
            extension: MachineLoadExtension::None | MachineLoadExtension::ZeroExtend,
            ..
        } => Some(*dst),
        _ => None,
    }
}

fn clean_preserving_reg_move(kind: &MachineInstKind) -> Option<(MachineReg, MachineReg)> {
    match kind {
        MachineInstKind::Move {
            ty: MachineStorageType::GpWord | MachineStorageType::GpI64,
            dst,
            src: MachineValue::Reg(src),
            ..
        } => Some((*dst, *src)),
        _ => None,
    }
}

fn advance_clean_state(kind: &MachineInstKind, clean: &mut CleanRegs) {
    // Runtime calls are semantically opaque here. Native backends preserve
    // live MachineIR values around them, but clearing the proof is the safer
    // contract for this address-specific optimization.
    if matches!(kind, MachineInstKind::CallRuntime(_)) {
        clean.clear();
        return;
    }

    // A full-width move preserves an already-zero upper half. Capture this
    // before killing the destination so self-copies remain clean.
    let clean_move = clean_preserving_reg_move(kind)
        .and_then(|(dst, src)| clean.contains(src).then_some(dst))
        .or_else(|| match kind {
            MachineInstKind::Move {
                ty: MachineStorageType::GpWord | MachineStorageType::GpI64,
                dst,
                src: MachineValue::Imm64(value),
                ..
            } if *value <= u64::from(u32::MAX) => Some(*dst),
            _ => None,
        });
    let clean_def = def_zero_extends(kind).or(clean_move);

    kind.for_each_defined_reg(|reg| {
        clean.remove(reg);
    });
    if let Some(dst) = clean_def {
        clean.insert(dst);
    }
}

fn advance_entry_dependence(kind: &MachineInstKind, dependent: &mut CleanRegs) {
    // The clean-state transfer forgets every incoming fact at a runtime call,
    // so no later obligation can depend on a block-entry fact through it.
    if matches!(kind, MachineInstKind::CallRuntime(_)) {
        dependent.clear();
        return;
    }

    // Only a full-width register move can make an instruction's cleanliness
    // depend on a different block-entry register. Capture self-copies before
    // killing the destination, exactly as in `advance_clean_state`.
    let dependent_move = clean_preserving_reg_move(kind)
        .and_then(|(dst, src)| dependent.contains(src).then_some(dst));
    kind.for_each_defined_reg(|reg| {
        dependent.remove(reg);
    });
    if let Some(dst) = dependent_move {
        dependent.insert(dst);
    }
}

fn exit_clean_state(block: &MachineBlock, entry: &CleanRegs) -> CleanRegs {
    let mut clean = entry.clone();
    for inst in &block.ops {
        advance_clean_state(&inst.kind, &mut clean);
    }
    clean
}

fn kill_call_results(results: &MachineCallResults, clean: &mut CleanRegs) {
    let mut kill = |dst: MachineResultDst| {
        if let MachineResultDst::Reg(reg) = dst {
            clean.remove(reg);
        }
    };
    match results {
        MachineCallResults::ScalarGp { dst, .. } | MachineCallResults::ScalarFp { dst, .. } => {
            kill(*dst);
        }
        MachineCallResults::ScalarGpPair { lo, hi } => {
            kill(*lo);
            kill(*hi);
        }
        MachineCallResults::None | MachineCallResults::FrameFallback { .. } => {}
    }
}

fn reachable_blocks(program: &MachineProgram) -> collections::Vec<bool> {
    let mut reachable = collections::vec![false; program.blocks.len()];
    let Some(entry) = block_index_for_id(&program.blocks, program.entry) else {
        return reachable;
    };
    reachable[entry] = true;
    let mut worklist = collections::vec![entry];
    while let Some(source) = worklist.pop() {
        visit_edges(&program.blocks[source].terminator, |edge| {
            let Some(target) = block_index_for_id(&program.blocks, edge.target) else {
                return;
            };
            if !reachable[target] {
                reachable[target] = true;
                worklist.push(target);
            }
        });
    }
    reachable
}

fn block_entry_clean_states(
    program: &MachineProgram,
    reachable: &[bool],
) -> collections::Vec<CleanRegs> {
    let entry = block_index_for_id(&program.blocks, program.entry);

    // Must analyses start reachable non-entry parameters at top and remove a
    // fact as soon as one incoming edge cannot prove it. This is what lets a
    // clean induction value remain clean around a loop: the preheader proves
    // the seed and the backedge proves the inductive step.
    let mut states: collections::Vec<CleanRegs> = program
        .blocks
        .iter()
        .enumerate()
        .map(|(index, block)| {
            if reachable[index] && Some(index) != entry {
                CleanRegs::block_param_top(block)
            } else {
                CleanRegs::default()
            }
        })
        .collect();

    // Facts only ever move from known to unknown. Process every reachable
    // source once at top, then revisit a block only when one of its entry
    // facts was actually removed. This avoids the quadratic convergence of a
    // synchronous fixed point on long entry-unknown chains.
    let mut queued: collections::Vec<bool> = reachable.iter().copied().collect();
    let mut worklist: collections::Vec<usize> = reachable
        .iter()
        .enumerate()
        .filter_map(|(index, is_reachable)| is_reachable.then_some(index))
        .collect();
    while let Some(source) = worklist.pop() {
        queued[source] = false;
        let mut source_clean = exit_clean_state(&program.blocks[source], &states[source]);
        if let MachineTerminator::Call { results, .. } = &program.blocks[source].terminator {
            // Success-edge result registers are defined by the call, not by
            // the pre-call instruction stream. Other explicit survivor
            // arguments retain their proven value.
            kill_call_results(results, &mut source_clean);
        }
        visit_edges(&program.blocks[source].terminator, |edge| {
            let Some(target) = block_index_for_id(&program.blocks, edge.target) else {
                return;
            };
            if !reachable[target] || Some(target) == entry {
                return;
            }
            let mut changed = false;
            for (param_index, param) in program.blocks[target].params.iter().enumerate() {
                let clean_arg = matches!(
                    edge.args.get(param_index),
                    Some(MachineValue::Reg(reg)) if source_clean.contains(*reg)
                ) || matches!(
                    edge.args.get(param_index),
                    Some(MachineValue::Imm64(value)) if *value <= u64::from(u32::MAX)
                );
                if !clean_arg {
                    changed |= states[target].remove(param.reg);
                }
            }
            if changed && !queued[target] {
                queued[target] = true;
                worklist.push(target);
            }
        });
    }
    states
}

fn zero_extend_index(kind: &mut MachineInstKind) -> Option<(MachineReg, &mut MachineIndexExtend)> {
    match kind {
        MachineInstKind::IndexedLoad {
            index,
            index_extend: index_extend @ MachineIndexExtend::ZeroExtend32,
            ..
        }
        | MachineInstKind::IndexedStore {
            index,
            index_extend: index_extend @ MachineIndexExtend::ZeroExtend32,
            ..
        } => Some((*index, index_extend)),
        _ => None,
    }
}

fn has_zero_extend_index(kind: &MachineInstKind) -> bool {
    matches!(
        kind,
        MachineInstKind::IndexedLoad {
            index_extend: MachineIndexExtend::ZeroExtend32,
            ..
        } | MachineInstKind::IndexedStore {
            index_extend: MachineIndexExtend::ZeroExtend32,
            ..
        }
    )
}

/// Relax facts established within this block and report whether any remaining
/// obligation could change when whole-CFG block-entry facts are supplied.
fn relax_index_extends_locally(block: &mut MachineBlock, seed_entry_params: bool) -> bool {
    if !block
        .ops
        .iter()
        .any(|inst| has_zero_extend_index(&inst.kind))
    {
        return false;
    }

    let mut clean = CleanRegs::default();
    let mut entry_dependent = if seed_entry_params {
        CleanRegs::block_param_top(block)
    } else {
        CleanRegs::default()
    };
    let mut needs_cfg = false;

    for inst in &mut block.ops {
        if let Some((index, index_extend)) = zero_extend_index(&mut inst.kind) {
            if clean.contains(index) {
                *index_extend = MachineIndexExtend::None;
            } else if entry_dependent.contains(index) {
                needs_cfg = true;
            }
        }

        advance_clean_state(&inst.kind, &mut clean);
        if !needs_cfg && !entry_dependent.is_empty() {
            advance_entry_dependence(&inst.kind, &mut entry_dependent);
        }
    }

    needs_cfg
}

/// Relax index extends using facts proven across the complete MachineIR CFG.
pub(super) fn relax_index_extends_program(program: &mut MachineProgram) {
    // Most functions either have no indexed zero-extend obligations, can
    // discharge them from in-block definitions, or leave obligations whose
    // index cannot depend on any block-entry fact. Handle those cases before
    // allocating and solving whole-CFG entry states. Mutating an index
    // obligation does not affect the clean-state transfer function, so the
    // entry-dependent blocks can safely continue through the full analysis.
    let mut unresolved_blocks = collections::Vec::new();
    let mut op_count = 0usize;
    let entry = program.entry;
    for (index, block) in program.blocks.iter_mut().enumerate() {
        op_count = op_count.saturating_add(block.ops.len());
        // The public-entry parameters are unknown by definition; only a
        // non-entry block can receive a clean parameter from a CFG edge.
        let seed_entry_params = block.id != entry;
        if relax_index_extends_locally(block, seed_entry_params) {
            unresolved_blocks.push(index);
        }
    }
    if unresolved_blocks.is_empty() {
        return;
    }
    if program.blocks.len() > MAX_CFG_INDEX_PROOF_BLOCKS || op_count > MAX_CFG_INDEX_PROOF_OPS {
        return;
    }

    // A malformed or otherwise unreachable block has no incoming proof. Do
    // this graph walk only after finding a potentially relevant non-entry
    // candidate, then reuse it in the fixed-point solver.
    let reachable = reachable_blocks(program);
    unresolved_blocks.retain(|index| reachable[*index]);
    if unresolved_blocks.is_empty() {
        return;
    }

    let entry_states = block_entry_clean_states(program, &reachable);
    for index in unresolved_blocks {
        relax_index_extends_with_entry(&mut program.blocks[index], &entry_states[index]);
    }
}

pub(super) fn relax_index_extends(block: &mut MachineBlock) {
    // Standalone block optimization has no predecessor facts available.
    relax_index_extends_locally(block, false);
}

fn relax_index_extends_with_entry(block: &mut MachineBlock, entry: &CleanRegs) {
    let mut clean = entry.clone();

    for inst in &mut block.ops {
        // Uses first: relax this op's own index against the state built
        // by earlier ops.
        if let Some((index, index_extend)) = zero_extend_index(&mut inst.kind) {
            if clean.contains(index) {
                *index_extend = MachineIndexExtend::None;
            }
        }

        // Defs second: every redefinition invalidates; whitelisted 32-bit
        // definitions and clean-preserving moves re-establish cleanliness.
        advance_clean_state(&inst.kind, &mut clean);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collections;
    use crate::vm::jit::backend::BackendConfig;
    use crate::vm::jit::machine::machine_ir::{
        MachineAddr, MachineBlockId, MachineBlockParam, MachineBranchCond, MachineCallArgs,
        MachineCallRuntime, MachineCallTarget, MachineConstId, MachineEdge, MachineFuncId,
        MachineInst, MachineIntBinaryOp, MachineRegOwner, MachineResultDst, MachineStorageType,
        MachineTerminator, MachineValue,
    };

    fn block(ops: collections::Vec<MachineInst>) -> MachineBlock {
        MachineBlock {
            id: MachineBlockId(0),
            params: collections::Vec::new(),
            ops,
            terminator: MachineTerminator::Return,
        }
    }

    fn edge(target: u32, args: collections::Vec<MachineValue>) -> MachineEdge {
        MachineEdge {
            target: MachineBlockId(target),
            args,
        }
    }

    fn cfg_block(
        id: u32,
        params: &[MachineReg],
        ops: collections::Vec<MachineInst>,
        terminator: MachineTerminator,
    ) -> MachineBlock {
        MachineBlock {
            id: MachineBlockId(id),
            params: params
                .iter()
                .copied()
                .map(MachineBlockParam::gp_word)
                .collect(),
            ops,
            terminator,
        }
    }

    fn program(blocks: collections::Vec<MachineBlock>) -> MachineProgram {
        MachineProgram {
            entry: MachineBlockId(0),
            fp_reg_init_widths: collections::Vec::new(),
            blocks,
        }
    }

    fn clean_cfg_chain(block_count: usize) -> MachineProgram {
        assert!(block_count >= 2);
        let mut blocks = collections::Vec::new();
        blocks.push(cfg_block(
            0,
            &[],
            collections::vec![add32(4, 4)],
            MachineTerminator::Jump(edge(1, collections::vec![MachineValue::Reg(MachineReg(4))])),
        ));
        for index in 1..block_count {
            let id = u32::try_from(index).expect("test block count fits u32");
            let last = index + 1 == block_count;
            blocks.push(cfg_block(
                id,
                &[MachineReg(5)],
                if last {
                    collections::vec![indexed_load(5)]
                } else {
                    collections::Vec::new()
                },
                if last {
                    MachineTerminator::Return
                } else {
                    MachineTerminator::Jump(edge(
                        id + 1,
                        collections::vec![MachineValue::Reg(MachineReg(5))],
                    ))
                },
            ));
        }
        program(blocks)
    }

    fn add32(dst: u16, lhs: u16) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: MachineIntWidth::I32,
                op: MachineIntBinaryOp::Add,
                dst: MachineReg(dst),
                lhs: MachineValue::Reg(MachineReg(lhs)),
                rhs: MachineValue::Imm64(8),
            },
        }
    }

    fn add64(dst: u16, lhs: u16) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: MachineIntWidth::I64,
                op: MachineIntBinaryOp::Add,
                dst: MachineReg(dst),
                lhs: MachineValue::Reg(MachineReg(lhs)),
                rhs: MachineValue::Imm64(8),
            },
        }
    }

    fn move_gp(dst: u16, src: u16) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::Move {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: MachineReg(dst),
                src: MachineValue::Reg(MachineReg(src)),
            },
        }
    }

    fn extend32s_i32(dst: u16, src: u16) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IntUnary {
                width: MachineIntWidth::I32,
                op: MachineIntUnaryOp::Extend32S,
                dst: MachineReg(dst),
                src: MachineValue::Reg(MachineReg(src)),
            },
        }
    }

    fn indexed_load(index: u16) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IndexedLoad {
                dst: MachineReg(9),
                base: MachineReg(2),
                index: MachineReg(index),
                index_extend: MachineIndexExtend::ZeroExtend32,
                offset: 0,
                width: MachineMemWidth::U32,
                extension: MachineLoadExtension::None,
            },
        }
    }

    fn extend_of(block: &MachineBlock, index: usize) -> MachineIndexExtend {
        match &block.ops[index].kind {
            MachineInstKind::IndexedLoad { index_extend, .. }
            | MachineInstKind::IndexedStore { index_extend, .. } => *index_extend,
            _ => panic!("not an indexed memory op"),
        }
    }

    fn store_addr_dummy() -> MachineInst {
        MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: MachineAddr {
                    base: MachineReg(2),
                    offset: 0,
                },
                width: MachineMemWidth::U64,
                src: MachineValue::Reg(MachineReg(9)),
            },
        }
    }

    #[test]
    fn clean_regs_keep_inline_boundary_and_overflow_exact() {
        let mut clean = CleanRegs::default();
        assert!(clean.is_empty());
        clean.insert(MachineReg(63));
        clean.insert(MachineReg(64));
        clean.insert(MachineReg(130));
        assert!(!clean.is_empty());

        assert!(clean.contains(MachineReg(63)));
        assert!(clean.contains(MachineReg(64)));
        assert!(clean.contains(MachineReg(130)));
        assert!(!clean.contains(MachineReg(0)));
        assert!(!clean.contains(MachineReg(66)));

        assert!(clean.remove(MachineReg(63)));
        assert!(clean.remove(MachineReg(64)));
        assert!(!clean.remove(MachineReg(64)));
        assert!(clean.contains(MachineReg(130)));

        clean.clear();
        assert!(clean.is_empty());
        assert!(!clean.contains(MachineReg(130)));
    }

    #[test]
    fn relaxes_after_32bit_def() {
        let mut b = block(collections::vec![add32(5, 5), indexed_load(5)]);
        relax_index_extends(&mut b);
        assert_eq!(extend_of(&b, 1), MachineIndexExtend::None);
    }

    #[test]
    fn keeps_extend_without_in_block_def() {
        let mut b = block(collections::vec![indexed_load(5)]);
        relax_index_extends(&mut b);
        assert_eq!(extend_of(&b, 0), MachineIndexExtend::ZeroExtend32);
    }

    #[test]
    fn keeps_extend_after_64bit_def() {
        let mut b = block(collections::vec![add64(5, 5), indexed_load(5)]);
        relax_index_extends(&mut b);
        assert_eq!(extend_of(&b, 1), MachineIndexExtend::ZeroExtend32);
    }

    #[test]
    fn redefinition_invalidates_cleanliness() {
        let mut b = block(collections::vec![add32(5, 5), add64(5, 5), indexed_load(5),]);
        relax_index_extends(&mut b);
        assert_eq!(extend_of(&b, 2), MachineIndexExtend::ZeroExtend32);
    }

    #[test]
    fn i32_extend32s_noop_does_not_establish_cleanliness() {
        let mut b = block(collections::vec![extend32s_i32(5, 5), indexed_load(5)]);
        relax_index_extends(&mut b);
        assert_eq!(extend_of(&b, 1), MachineIndexExtend::ZeroExtend32);
    }

    #[test]
    fn cleanliness_survives_unrelated_ops() {
        let mut b = block(collections::vec![
            add32(5, 5),
            store_addr_dummy(),
            indexed_load(5),
        ]);
        relax_index_extends(&mut b);
        assert_eq!(extend_of(&b, 2), MachineIndexExtend::None);
    }

    #[test]
    fn clean_load_result_relaxes_following_index_use() {
        let mut b = block(collections::vec![indexed_load(5), {
            MachineInst {
                kind: MachineInstKind::IndexedLoad {
                    dst: MachineReg(10),
                    base: MachineReg(2),
                    index: MachineReg(9),
                    index_extend: MachineIndexExtend::ZeroExtend32,
                    offset: 0,
                    width: MachineMemWidth::U32,
                    extension: MachineLoadExtension::None,
                },
            }
        }]);
        relax_index_extends(&mut b);
        // First load's index has no in-block def; second indexes the
        // first's zero-extending U32 result.
        assert_eq!(extend_of(&b, 0), MachineIndexExtend::ZeroExtend32);
        assert_eq!(extend_of(&b, 1), MachineIndexExtend::None);
    }

    #[test]
    fn local_prepass_reports_when_all_obligations_are_resolved() {
        let mut b = block(collections::vec![add32(5, 5), indexed_load(5)]);

        let needs_cfg = relax_index_extends_locally(&mut b, false);

        assert!(!needs_cfg);
        assert_eq!(extend_of(&b, 1), MachineIndexExtend::None);
    }

    #[test]
    fn local_prepass_skips_block_without_index_obligations() {
        let mut b = block(collections::vec![add32(5, 5)]);

        assert!(!relax_index_extends_locally(&mut b, false));
        assert_eq!(b.ops.len(), 1);
    }

    #[test]
    fn local_prepass_skips_cfg_for_nonparam_unknown_index() {
        let mut b = block(collections::vec![indexed_load(5)]);

        let needs_cfg = relax_index_extends_locally(&mut b, false);

        assert!(!needs_cfg);
        assert_eq!(extend_of(&b, 0), MachineIndexExtend::ZeroExtend32);
    }

    #[test]
    fn entry_block_param_is_unknown_and_does_not_request_cfg() {
        let mut entry = cfg_block(
            0,
            &[MachineReg(5)],
            collections::vec![indexed_load(5)],
            MachineTerminator::Return,
        );

        let needs_cfg = relax_index_extends_locally(&mut entry, false);
        assert!(!needs_cfg);
        assert_eq!(extend_of(&entry, 0), MachineIndexExtend::ZeroExtend32);

        let mut p = program(collections::vec![entry]);
        relax_index_extends_program(&mut p);
        assert_eq!(extend_of(&p.blocks[0], 0), MachineIndexExtend::ZeroExtend32);
    }

    #[test]
    fn local_prepass_skips_cfg_after_param_is_overwritten_dirty() {
        let mut b = cfg_block(
            0,
            &[MachineReg(5)],
            collections::vec![add64(5, 5), indexed_load(5)],
            MachineTerminator::Return,
        );

        let needs_cfg = relax_index_extends_locally(&mut b, true);

        assert!(!needs_cfg);
        assert_eq!(extend_of(&b, 1), MachineIndexExtend::ZeroExtend32);
    }

    #[test]
    fn relaxes_clean_block_parameter_across_edge() {
        let mut p = program(collections::vec![
            cfg_block(
                0,
                &[],
                collections::vec![add32(5, 4)],
                MachineTerminator::Jump(edge(
                    1,
                    collections::vec![MachineValue::Reg(MachineReg(5))],
                )),
            ),
            cfg_block(
                1,
                &[MachineReg(6)],
                collections::vec![indexed_load(6)],
                MachineTerminator::Return,
            ),
        ]);

        relax_index_extends_program(&mut p);
        assert_eq!(extend_of(&p.blocks[1], 0), MachineIndexExtend::None);
    }

    #[test]
    fn local_prepass_keeps_cross_edge_work_for_cfg_analysis() {
        let mut p = program(collections::vec![
            cfg_block(
                0,
                &[],
                collections::vec![add32(5, 4)],
                MachineTerminator::Jump(edge(
                    1,
                    collections::vec![MachineValue::Reg(MachineReg(5))],
                )),
            ),
            cfg_block(
                1,
                &[MachineReg(6)],
                collections::vec![add32(7, 7), indexed_load(7), indexed_load(6)],
                MachineTerminator::Return,
            ),
        ]);

        let needs_cfg = relax_index_extends_locally(&mut p.blocks[1], true);
        assert!(needs_cfg);
        assert_eq!(extend_of(&p.blocks[1], 1), MachineIndexExtend::None);
        assert_eq!(extend_of(&p.blocks[1], 2), MachineIndexExtend::ZeroExtend32);

        relax_index_extends_program(&mut p);
        assert_eq!(extend_of(&p.blocks[1], 1), MachineIndexExtend::None);
        assert_eq!(extend_of(&p.blocks[1], 2), MachineIndexExtend::None);
    }

    #[test]
    fn local_prepass_follows_entry_dependence_through_move_chain() {
        let mut p = program(collections::vec![
            cfg_block(
                0,
                &[],
                collections::vec![add32(4, 4)],
                MachineTerminator::Jump(edge(
                    1,
                    collections::vec![MachineValue::Reg(MachineReg(4))],
                )),
            ),
            cfg_block(
                1,
                &[MachineReg(5)],
                collections::vec![move_gp(6, 5), move_gp(7, 6), indexed_load(7)],
                MachineTerminator::Return,
            ),
        ]);

        let needs_cfg = relax_index_extends_locally(&mut p.blocks[1], true);
        assert!(needs_cfg);
        assert_eq!(extend_of(&p.blocks[1], 2), MachineIndexExtend::ZeroExtend32);

        relax_index_extends_program(&mut p);
        assert_eq!(extend_of(&p.blocks[1], 2), MachineIndexExtend::None);
    }

    #[test]
    fn cfg_proof_runs_at_op_budget() {
        let mut entry_ops = collections::Vec::new();
        for _ in 0..MAX_CFG_INDEX_PROOF_OPS - 1 {
            entry_ops.push(add32(4, 4));
        }
        let mut p = program(collections::vec![
            cfg_block(
                0,
                &[],
                entry_ops,
                MachineTerminator::Jump(edge(
                    1,
                    collections::vec![MachineValue::Reg(MachineReg(4))],
                )),
            ),
            cfg_block(
                1,
                &[MachineReg(5)],
                collections::vec![indexed_load(5)],
                MachineTerminator::Return,
            ),
        ]);

        relax_index_extends_program(&mut p);
        assert_eq!(extend_of(&p.blocks[1], 0), MachineIndexExtend::None);
    }

    #[test]
    fn op_budget_keeps_cross_block_extend_and_local_relaxation() {
        let mut entry_ops = collections::Vec::new();
        for _ in 0..MAX_CFG_INDEX_PROOF_OPS - 2 {
            entry_ops.push(add32(4, 4));
        }
        let mut p = program(collections::vec![
            cfg_block(
                0,
                &[],
                entry_ops,
                MachineTerminator::Jump(edge(
                    1,
                    collections::vec![MachineValue::Reg(MachineReg(4))],
                )),
            ),
            cfg_block(
                1,
                &[MachineReg(5)],
                collections::vec![add32(6, 6), indexed_load(6), indexed_load(5)],
                MachineTerminator::Return,
            ),
        ]);

        relax_index_extends_program(&mut p);
        assert_eq!(extend_of(&p.blocks[1], 1), MachineIndexExtend::None);
        assert_eq!(extend_of(&p.blocks[1], 2), MachineIndexExtend::ZeroExtend32);
    }

    #[test]
    fn cfg_proof_runs_at_block_budget() {
        let mut p = clean_cfg_chain(MAX_CFG_INDEX_PROOF_BLOCKS);

        relax_index_extends_program(&mut p);
        assert_eq!(
            extend_of(p.blocks.last().expect("chain has a tail"), 0),
            MachineIndexExtend::None
        );
    }

    #[test]
    fn block_budget_keeps_cross_block_extend() {
        let mut p = clean_cfg_chain(MAX_CFG_INDEX_PROOF_BLOCKS + 1);

        relax_index_extends_program(&mut p);
        assert_eq!(
            extend_of(p.blocks.last().expect("chain has a tail"), 0),
            MachineIndexExtend::ZeroExtend32
        );
    }

    #[test]
    fn mixed_predecessor_keeps_extend() {
        let mut p = program(collections::vec![
            cfg_block(
                0,
                &[],
                collections::vec![add32(5, 4)],
                MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(MachineValue::Imm64(1)),
                    then_edge: edge(1, collections::vec![MachineValue::Reg(MachineReg(5))],),
                    else_edge: edge(1, collections::vec![MachineValue::Reg(MachineReg(4))],),
                },
            ),
            cfg_block(
                1,
                &[MachineReg(6)],
                collections::vec![indexed_load(6)],
                MachineTerminator::Return,
            ),
        ]);

        relax_index_extends_program(&mut p);
        assert_eq!(extend_of(&p.blocks[1], 0), MachineIndexExtend::ZeroExtend32);
    }

    #[test]
    fn clean_loop_parameter_reaches_greatest_fixed_point() {
        let mut p = program(collections::vec![
            cfg_block(
                0,
                &[],
                collections::vec![add32(4, 4)],
                MachineTerminator::Jump(edge(
                    1,
                    collections::vec![MachineValue::Reg(MachineReg(4))],
                )),
            ),
            cfg_block(
                1,
                &[MachineReg(5)],
                collections::vec![indexed_load(5), add32(6, 5)],
                MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(MachineValue::Imm64(1)),
                    then_edge: edge(1, collections::vec![MachineValue::Reg(MachineReg(6))],),
                    else_edge: edge(2, collections::Vec::new()),
                },
            ),
            cfg_block(2, &[], collections::Vec::new(), MachineTerminator::Return,),
        ]);

        relax_index_extends_program(&mut p);
        assert_eq!(extend_of(&p.blocks[1], 0), MachineIndexExtend::None);
    }

    #[test]
    fn dirty_loop_backedge_removes_tentative_cleanliness() {
        let mut p = program(collections::vec![
            cfg_block(
                0,
                &[],
                collections::vec![add32(4, 4)],
                MachineTerminator::Jump(edge(
                    1,
                    collections::vec![MachineValue::Reg(MachineReg(4))],
                )),
            ),
            cfg_block(
                1,
                &[MachineReg(5)],
                collections::vec![indexed_load(5), add64(6, 5)],
                MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(MachineValue::Imm64(1)),
                    then_edge: edge(1, collections::vec![MachineValue::Reg(MachineReg(6))]),
                    else_edge: edge(2, collections::Vec::new()),
                },
            ),
            cfg_block(2, &[], collections::Vec::new(), MachineTerminator::Return),
        ]);

        relax_index_extends_program(&mut p);
        assert_eq!(extend_of(&p.blocks[1], 0), MachineIndexExtend::ZeroExtend32);
    }

    #[test]
    fn unknown_entry_fact_propagates_through_long_chain() {
        const BLOCK_COUNT: u32 = 256;
        let mut blocks = collections::Vec::new();
        blocks.push(cfg_block(
            0,
            &[MachineReg(4)],
            collections::Vec::new(),
            MachineTerminator::Jump(edge(1, collections::vec![MachineValue::Reg(MachineReg(4))])),
        ));
        for id in 1..BLOCK_COUNT {
            let last = id + 1 == BLOCK_COUNT;
            blocks.push(cfg_block(
                id,
                &[MachineReg(5)],
                if last {
                    collections::vec![indexed_load(5)]
                } else {
                    collections::Vec::new()
                },
                if last {
                    MachineTerminator::Return
                } else {
                    MachineTerminator::Jump(edge(
                        id + 1,
                        collections::vec![MachineValue::Reg(MachineReg(5))],
                    ))
                },
            ));
        }
        let mut p = program(blocks);

        relax_index_extends_program(&mut p);
        assert_eq!(
            extend_of(p.blocks.last().expect("long chain has a tail"), 0),
            MachineIndexExtend::ZeroExtend32
        );
    }

    #[test]
    fn call_result_is_unknown_but_clean_survivor_propagates() {
        let mut p = program(collections::vec![
            cfg_block(
                0,
                &[],
                collections::vec![add32(4, 4), add32(5, 5)],
                MachineTerminator::Call {
                    target: MachineCallTarget::Direct(MachineFuncId(1)),
                    frame_delta: 0,
                    args: MachineCallArgs::default(),
                    results: MachineCallResults::ScalarGp {
                        dst: MachineResultDst::Reg(MachineReg(4)),
                        ty: MachineStorageType::GpWord,
                    },
                    success: edge(
                        1,
                        collections::vec![
                            MachineValue::Reg(MachineReg(4)),
                            MachineValue::Reg(MachineReg(5)),
                        ],
                    ),
                },
            ),
            cfg_block(
                1,
                &[MachineReg(6), MachineReg(7)],
                collections::vec![indexed_load(6), indexed_load(7)],
                MachineTerminator::Return,
            ),
        ]);

        relax_index_extends_program(&mut p);
        assert_eq!(extend_of(&p.blocks[1], 0), MachineIndexExtend::ZeroExtend32);
        assert_eq!(extend_of(&p.blocks[1], 1), MachineIndexExtend::None);
    }

    #[test]
    fn unreachable_cycle_cannot_self_prove_cleanliness() {
        let mut p = program(collections::vec![
            cfg_block(0, &[], collections::Vec::new(), MachineTerminator::Return,),
            cfg_block(
                1,
                &[MachineReg(5)],
                collections::vec![indexed_load(5)],
                MachineTerminator::Jump(edge(
                    1,
                    collections::vec![MachineValue::Reg(MachineReg(5))],
                )),
            ),
        ]);

        relax_index_extends_program(&mut p);
        assert_eq!(extend_of(&p.blocks[1], 0), MachineIndexExtend::ZeroExtend32);
    }

    #[test]
    fn entry_self_backedge_does_not_prove_entry_argument() {
        let mut p = program(collections::vec![cfg_block(
            0,
            &[MachineReg(5)],
            collections::vec![indexed_load(5)],
            MachineTerminator::Jump(edge(0, collections::vec![MachineValue::Reg(MachineReg(5))],)),
        )]);

        relax_index_extends_program(&mut p);
        assert_eq!(extend_of(&p.blocks[0], 0), MachineIndexExtend::ZeroExtend32);
    }

    #[test]
    fn malformed_or_noncanonical_edge_values_stay_unknown() {
        let mut p = program(collections::vec![
            cfg_block(
                0,
                &[],
                collections::Vec::new(),
                MachineTerminator::JumpTable {
                    index: MachineValue::Imm64(0),
                    entries: collections::vec![
                        edge(1, collections::Vec::new()),
                        edge(
                            2,
                            collections::vec![MachineValue::ReservedReg(MachineReg(4))],
                        ),
                        edge(
                            3,
                            collections::vec![MachineValue::Imm64(u64::from(u32::MAX) + 1,)],
                        ),
                    ],
                },
            ),
            cfg_block(
                1,
                &[MachineReg(5)],
                collections::vec![indexed_load(5)],
                MachineTerminator::Return,
            ),
            cfg_block(
                2,
                &[MachineReg(6)],
                collections::vec![indexed_load(6)],
                MachineTerminator::Return,
            ),
            cfg_block(
                3,
                &[MachineReg(7)],
                collections::vec![indexed_load(7)],
                MachineTerminator::Return,
            ),
        ]);

        relax_index_extends_program(&mut p);
        for block in &p.blocks[1..] {
            assert_eq!(extend_of(block, 0), MachineIndexExtend::ZeroExtend32);
        }
    }

    #[test]
    fn runtime_call_clears_clean_facts() {
        let mut b = block(collections::vec![
            add32(5, 5),
            MachineInst {
                kind: MachineInstKind::CallRuntime(MachineCallRuntime {
                    metadata: MachineConstId(0),
                }),
            },
            indexed_load(5),
        ]);

        relax_index_extends(&mut b);
        assert_eq!(extend_of(&b, 2), MachineIndexExtend::ZeroExtend32);
    }

    #[test]
    fn high_register_does_not_alias_low_register_fact() {
        let mut b = block(collections::vec![add32(130, 130), indexed_load(2)]);
        relax_index_extends(&mut b);
        assert_eq!(extend_of(&b, 1), MachineIndexExtend::ZeroExtend32);

        let mut p = program(collections::vec![
            cfg_block(
                0,
                &[],
                collections::vec![add32(130, 4)],
                MachineTerminator::Jump(edge(
                    1,
                    collections::vec![MachineValue::Reg(MachineReg(130))],
                )),
            ),
            cfg_block(
                1,
                &[MachineReg(258)],
                collections::vec![indexed_load(258)],
                MachineTerminator::Return,
            ),
        ]);
        relax_index_extends_program(&mut p);
        assert_eq!(extend_of(&p.blocks[1], 0), MachineIndexExtend::None);
    }

    #[test]
    fn optimizer_gates_relaxation_on_gp32_zero_extending_config() {
        let make_program = || {
            program(collections::vec![cfg_block(
                0,
                &[],
                collections::vec![add32(5, 5), indexed_load(5)],
                MachineTerminator::Return,
            )])
        };

        let mut disabled = make_program();
        super::super::optimize(&mut disabled, BackendConfig::new(8, 8, 0, 0));
        assert_eq!(
            extend_of(&disabled.blocks[0], 1),
            MachineIndexExtend::ZeroExtend32
        );

        let mut enabled = make_program();
        super::super::optimize(
            &mut enabled,
            BackendConfig::new(8, 8, 0, 0).with_gp32_zero_extending_defs(),
        );
        assert_eq!(extend_of(&enabled.blocks[0], 1), MachineIndexExtend::None);
    }
}
