//! Recover signed 32x32 -> 64 multiply from sign-extended i32 operands.
//!
//! On 32-bit GP targets, `i64.mul` normally lowers to a generic pair multiply.
//! When both operands came from `i64.extend_i32_s`, the product is exactly a
//! signed widening multiply, represented as `Int64MulFromSignExt32`.
//!
//! This pass intentionally runs at MachineIR: the fact being recovered is
//! target-independent, while each backend can decide whether it has a native
//! widening multiply or should lower the generic pair form.

use crate::collections;

use crate::vm::jit::machine::machine_ir::{
    MachineAddr, MachineBlock, MachineBlockId, MachineEdge, MachineInstKind, MachineIntBinaryOp,
    MachineIntUnaryOp, MachineMemWidth, MachineProgram, MachineReg, MachineTerminator,
    MachineValue,
};

#[derive(Clone, Copy)]
struct TrackedSpill {
    addr: MachineAddr,
    stored_reg: MachineReg,
    sign_ext_of: Option<ValueId>,
    value_id: ValueId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ValueId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SignExtPair {
    lo: MachineReg,
    hi: MachineReg,
}

struct SignExtState {
    value_of: collections::Vec<ValueId>,
    sign_ext_of: collections::Vec<Option<ValueId>>,
    spills: collections::Vec<TrackedSpill>,
    next_value: usize,
}

pub(super) fn fuse_smull_sign_ext(block: &mut MachineBlock, total_reg_count: usize) {
    process_block(block, total_reg_count, &[], true);
}

pub(super) fn fuse_smull_sign_ext_across_edges(
    program: &mut MachineProgram,
    total_reg_count: usize,
) {
    if !program.blocks.iter().any(block_has_pair_mul) {
        return;
    }

    let entry_pairs = compute_entry_pairs(program, total_reg_count);
    for (block_idx, block) in program.blocks.iter_mut().enumerate() {
        let Some(pairs) = entry_pairs.get(block_idx) else {
            continue;
        };
        if !pairs.is_empty() {
            process_block(block, total_reg_count, pairs, true);
        }
    }
}

fn process_block(
    block: &mut MachineBlock,
    total_reg_count: usize,
    entry_pairs: &[SignExtPair],
    rewrite: bool,
) -> collections::Vec<SignExtPair> {
    let mut state = SignExtState::new(total_reg_count, entry_pairs);

    for inst in &mut block.ops {
        if rewrite && state.try_rewrite_mul(&mut inst.kind) {
            continue;
        }
        state.apply_inst(&inst.kind);
    }

    state.output_pairs()
}

fn block_has_pair_mul(block: &MachineBlock) -> bool {
    block.ops.iter().any(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::Int64PairBinary {
                op: MachineIntBinaryOp::Mul,
                ..
            }
        )
    })
}

impl SignExtState {
    fn new(total_reg_count: usize, entry_pairs: &[SignExtPair]) -> Self {
        let mut state = Self {
            value_of: (0..total_reg_count).map(ValueId).collect(),
            sign_ext_of: collections::vec![None; total_reg_count],
            spills: collections::Vec::new(),
            next_value: total_reg_count,
        };
        state.seed_entry_pairs(entry_pairs);
        state
    }

    fn seed_entry_pairs(&mut self, entry_pairs: &[SignExtPair]) {
        for pair in entry_pairs {
            let hi_idx = pair.hi.0 as usize;
            let lo_idx = pair.lo.0 as usize;
            if hi_idx >= self.sign_ext_of.len() || lo_idx >= self.value_of.len() {
                continue;
            }
            let value = self.sign_ext_of[hi_idx].unwrap_or(self.value_of[lo_idx]);
            self.value_of[lo_idx] = value;
            self.sign_ext_of[hi_idx] = Some(value);
        }
    }

    fn new_value(&mut self) -> ValueId {
        let value = ValueId(self.next_value);
        self.next_value += 1;
        value
    }

    fn value_id(&self, reg: MachineReg) -> Option<ValueId> {
        self.value_of.get(reg.0 as usize).copied()
    }

    fn sign_ext_value(&self, reg: MachineReg) -> Option<ValueId> {
        self.sign_ext_of.get(reg.0 as usize).copied().flatten()
    }

    fn set_unknown_value(&mut self, reg: MachineReg) {
        let idx = reg.0 as usize;
        if idx >= self.value_of.len() {
            return;
        }
        let value = self.new_value();
        self.value_of[idx] = value;
        self.sign_ext_of[idx] = None;
    }

    fn set_value(&mut self, reg: MachineReg, value: ValueId, sign_ext_of: Option<ValueId>) {
        let idx = reg.0 as usize;
        if idx >= self.value_of.len() {
            return;
        }
        self.value_of[idx] = value;
        self.sign_ext_of[idx] = sign_ext_of;
    }

    fn reg_for_value(&self, value: ValueId) -> Option<MachineReg> {
        self.value_of
            .iter()
            .position(|candidate| *candidate == value)
            .map(|idx| MachineReg(idx as u16))
    }

    fn try_rewrite_mul(&mut self, kind: &mut MachineInstKind) -> bool {
        let MachineInstKind::Int64PairBinary {
            op: MachineIntBinaryOp::Mul,
            dst_lo,
            dst_hi,
            lhs_lo: MachineValue::Reg(lhs_lo),
            lhs_hi: MachineValue::Reg(lhs_hi),
            rhs_lo: MachineValue::Reg(rhs_lo),
            rhs_hi: MachineValue::Reg(rhs_hi),
        } = *kind
        else {
            return false;
        };

        let Some(lhs_value) = self.value_id(lhs_lo) else {
            return false;
        };
        let Some(rhs_value) = self.value_id(rhs_lo) else {
            return false;
        };
        if self.sign_ext_value(lhs_hi) != Some(lhs_value)
            || self.sign_ext_value(rhs_hi) != Some(rhs_value)
        {
            return false;
        }

        let lhs = self.reg_for_value(lhs_value).unwrap_or(lhs_lo);
        let rhs = self.reg_for_value(rhs_value).unwrap_or(rhs_lo);
        *kind = MachineInstKind::Int64MulFromSignExt32 {
            dst_lo,
            dst_hi,
            lhs: MachineValue::Reg(lhs),
            rhs: MachineValue::Reg(rhs),
        };
        self.set_unknown_value(dst_lo);
        self.set_unknown_value(dst_hi);
        true
    }

    fn apply_inst(&mut self, kind: &MachineInstKind) {
        match kind {
            MachineInstKind::IntBinary {
                op: MachineIntBinaryOp::ShrS,
                dst,
                lhs: MachineValue::Reg(src),
                rhs: MachineValue::Imm64(31),
                ..
            } => {
                let Some(src_value) = self.value_id(*src) else {
                    self.set_unknown_value(*dst);
                    return;
                };
                let dst_value = self.new_value();
                self.set_value(*dst, dst_value, Some(src_value));
            }
            MachineInstKind::Int64PairUnary {
                op: MachineIntUnaryOp::Extend32S,
                dst_lo,
                dst_hi,
                src_lo: MachineValue::Reg(src_lo),
                ..
            } => {
                let Some(src_value) = self.value_id(*src_lo) else {
                    self.set_unknown_value(*dst_lo);
                    self.set_unknown_value(*dst_hi);
                    return;
                };
                self.set_value(*dst_lo, src_value, None);
                let hi_value = self.new_value();
                self.set_value(*dst_hi, hi_value, Some(src_value));
            }
            MachineInstKind::Move {
                dst,
                src: MachineValue::Reg(src),
                ..
            } => {
                let Some(src_value) = self.value_id(*src) else {
                    self.set_unknown_value(*dst);
                    return;
                };
                self.set_value(*dst, src_value, self.sign_ext_value(*src));
            }
            MachineInstKind::Store {
                addr,
                src: MachineValue::Reg(src_reg),
                width: MachineMemWidth::U32,
                ..
            } => {
                self.kill_spills_at(*addr);
                if let Some(value_id) = self.value_id(*src_reg) {
                    self.spills.push(TrackedSpill {
                        addr: *addr,
                        stored_reg: *src_reg,
                        sign_ext_of: self.sign_ext_value(*src_reg),
                        value_id,
                    });
                }
            }
            MachineInstKind::Store { addr, .. } => {
                self.kill_spills_at(*addr);
            }
            MachineInstKind::Load {
                dst,
                addr,
                width: MachineMemWidth::U32,
                ..
            } => {
                let recovered = self.spills.iter().rev().find(|entry| {
                    entry.addr.base == addr.base
                        && entry.addr.offset == addr.offset
                        && entry.stored_reg == *dst
                });
                if let Some(spill) = recovered {
                    self.set_value(*dst, spill.value_id, spill.sign_ext_of);
                } else {
                    self.set_unknown_value(*dst);
                }
            }
            other => {
                other.for_each_defined_reg(|reg| {
                    self.set_unknown_value(reg);
                });
            }
        }
    }

    fn kill_spills_at(&mut self, addr: MachineAddr) {
        self.spills
            .retain(|entry| !(entry.addr.base == addr.base && entry.addr.offset == addr.offset));
    }

    fn output_pairs(&self) -> collections::Vec<SignExtPair> {
        let mut pairs = collections::Vec::new();
        for (hi_idx, sign_value) in self.sign_ext_of.iter().enumerate() {
            let Some(sign_value) = *sign_value else {
                continue;
            };
            for (lo_idx, value) in self.value_of.iter().enumerate() {
                if *value == sign_value {
                    pairs.push(SignExtPair {
                        lo: MachineReg(lo_idx as u16),
                        hi: MachineReg(hi_idx as u16),
                    });
                }
            }
        }
        normalize_pairs(&mut pairs);
        pairs
    }
}

fn compute_entry_pairs(
    program: &MachineProgram,
    total_reg_count: usize,
) -> collections::Vec<collections::Vec<SignExtPair>> {
    let block_count = program.blocks.len();
    let entry_idx = block_index_for_id(&program.blocks, program.entry);
    let mut entry: collections::Vec<Option<collections::Vec<SignExtPair>>> =
        collections::vec![None; block_count];
    if let Some(entry_idx) = entry_idx {
        entry[entry_idx] = Some(collections::Vec::new());
    }

    loop {
        let mut exit: collections::Vec<Option<collections::Vec<SignExtPair>>> =
            collections::vec![None; block_count];
        for (idx, block) in program.blocks.iter().enumerate() {
            let Some(entry_pairs) = entry[idx].as_ref() else {
                continue;
            };
            let mut block = block.clone();
            exit[idx] = Some(process_block(
                &mut block,
                total_reg_count,
                entry_pairs,
                false,
            ));
        }

        let mut next: collections::Vec<Option<collections::Vec<SignExtPair>>> =
            collections::vec![None; block_count];
        for (pred_idx, block) in program.blocks.iter().enumerate() {
            let Some(exit_pairs) = exit[pred_idx].as_ref() else {
                continue;
            };
            visit_edges(&block.terminator, |edge| {
                let Some(target_idx) = block_index_for_id(&program.blocks, edge.target) else {
                    return;
                };
                let incoming =
                    transfer_pairs_across_edge(&program.blocks[target_idx], edge, exit_pairs);
                meet_pairs(&mut next[target_idx], incoming);
            });
        }
        if let Some(entry_idx) = entry_idx {
            next[entry_idx] = Some(collections::Vec::new());
        }

        if next == entry {
            return next
                .into_iter()
                .map(|pairs| pairs.unwrap_or_default())
                .collect();
        }
        entry = next;
    }
}

fn transfer_pairs_across_edge(
    target: &MachineBlock,
    edge: &MachineEdge,
    pairs: &[SignExtPair],
) -> collections::Vec<SignExtPair> {
    let mut mapped = collections::Vec::new();
    for pair in pairs {
        let Some(lo) = target_param_for_incoming(target, &edge.args, pair.lo) else {
            continue;
        };
        let Some(hi) = target_param_for_incoming(target, &edge.args, pair.hi) else {
            continue;
        };
        mapped.push(SignExtPair { lo, hi });
    }
    normalize_pairs(&mut mapped);
    mapped
}

fn target_param_for_incoming(
    target: &MachineBlock,
    args: &[MachineValue],
    incoming: MachineReg,
) -> Option<MachineReg> {
    target
        .params
        .iter()
        .zip(args.iter())
        .find_map(|(param, arg)| match arg {
            MachineValue::Reg(src) if *src == incoming => Some(param.reg),
            _ => None,
        })
}

fn meet_pairs(
    slot: &mut Option<collections::Vec<SignExtPair>>,
    mut incoming: collections::Vec<SignExtPair>,
) {
    normalize_pairs(&mut incoming);
    let Some(current) = slot else {
        *slot = Some(incoming);
        return;
    };
    current.retain(|pair| incoming.contains(pair));
}

fn normalize_pairs(pairs: &mut collections::Vec<SignExtPair>) {
    pairs.sort();
    pairs.dedup();
}

fn block_index_for_id(blocks: &[MachineBlock], id: MachineBlockId) -> Option<usize> {
    let idx = id.as_usize();
    if idx < blocks.len() && blocks[idx].id == id {
        return Some(idx);
    }
    blocks.iter().position(|block| block.id == id)
}

fn visit_edges(terminator: &MachineTerminator, mut f: impl FnMut(&MachineEdge)) {
    match terminator {
        MachineTerminator::Jump(edge) => f(edge),
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            f(then_edge);
            f(else_edge);
        }
        MachineTerminator::JumpTable { entries, .. } => {
            for edge in entries {
                f(edge);
            }
        }
        MachineTerminator::Call { .. }
        | MachineTerminator::TailCall { .. }
        | MachineTerminator::Return
        | MachineTerminator::ReturnScalar { .. }
        | MachineTerminator::Trap { .. } => {}
    }
}
