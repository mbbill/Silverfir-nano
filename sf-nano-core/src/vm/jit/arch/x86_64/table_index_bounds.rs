//! A bounded backward proof for dense `br_table` indices.
//!
//! Follow copies, selects and block arguments to every reaching definition.
//! Constants and masks bound the possible low 32 bits. Cycles of copies do
//! not introduce bits: their incoming definitions are still visited, including
//! every entry into a loop. Unknown inputs, calls and exhausted analysis
//! budgets keep the ordinary unsigned clamp.

use crate::collections;
use crate::vm::jit::machine::machine_ir::{
    MachineBlockId, MachineEdge, MachineInstKind, MachineIntBinaryOp, MachineProgram, MachineReg,
    MachineStorageType, MachineTerminator, MachineValue,
};

const MAX_QUERIES: usize = 256;
const MAX_SCAN: usize = 8192;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Query {
    block: MachineBlockId,
    before: usize,
    reg: MachineReg,
}

struct Proof {
    pending: collections::Vec<Query>,
    seen: collections::Vec<Query>,
    possible_bits: u32,
    entries: usize,
    scanned: usize,
}

impl Proof {
    fn include_bits(&mut self, bits: u32) -> bool {
        self.possible_bits |= bits;
        (self.possible_bits as usize) < self.entries
    }

    fn include_value(&mut self, block: MachineBlockId, before: usize, value: MachineValue) -> bool {
        match value {
            MachineValue::Imm64(value) => self.include_bits(value as u32),
            MachineValue::Reg(reg) => {
                let query = Query { block, before, reg };
                if !self.seen.contains(&query) {
                    if self.seen.len() == MAX_QUERIES {
                        return false;
                    }
                    self.seen.push(query);
                    self.pending.push(query);
                }
                true
            }
            MachineValue::ReservedReg(_) => false,
        }
    }

    fn scan(&mut self) -> bool {
        self.scanned += 1;
        self.scanned <= MAX_SCAN
    }
}

pub(super) fn index_is_in_bounds(
    program: &MachineProgram,
    block: MachineBlockId,
    index: MachineValue,
    entries: usize,
) -> bool {
    let Some(initial) = program.blocks.get(block.as_usize()) else {
        return false;
    };
    let mut proof = Proof {
        pending: collections::Vec::new(),
        seen: collections::Vec::new(),
        possible_bits: 0,
        entries,
        scanned: 0,
    };
    if entries == 0 || !proof.include_value(block, initial.ops.len(), index) {
        return false;
    }
    while let Some(query) = proof.pending.pop() {
        let Some(block) = program.blocks.get(query.block.as_usize()) else {
            return false;
        };
        let mut definition = None;
        for (position, inst) in block.ops[..query.before].iter().enumerate().rev() {
            if !proof.scan() || matches!(inst.kind, MachineInstKind::CallRuntime(_)) {
                return false;
            }
            let mut defines = false;
            inst.kind
                .for_each_defined_reg(|reg| defines |= reg == query.reg);
            if defines {
                definition = Some((position, &inst.kind));
                break;
            }
        }
        if let Some((position, kind)) = definition {
            let bounded = match *kind {
                MachineInstKind::Move {
                    ty: MachineStorageType::GpWord | MachineStorageType::GpI64,
                    src,
                    ..
                } => proof.include_value(query.block, position, src),
                MachineInstKind::Select {
                    ty: MachineStorageType::GpWord | MachineStorageType::GpI64,
                    on_true,
                    on_false,
                    ..
                } => {
                    proof.include_value(query.block, position, on_true)
                        && proof.include_value(query.block, position, on_false)
                }
                MachineInstKind::IntBinary {
                    op: MachineIntBinaryOp::And,
                    lhs: MachineValue::Imm64(mask),
                    ..
                }
                | MachineInstKind::IntBinary {
                    op: MachineIntBinaryOp::And,
                    rhs: MachineValue::Imm64(mask),
                    ..
                } => proof.include_bits(mask as u32),
                MachineInstKind::IntCompare { .. } | MachineInstKind::TestBits { .. } => {
                    proof.include_bits(1)
                }
                _ => false,
            };
            if !bounded {
                return false;
            }
            continue;
        }
        // The function entry also has an implicit incoming ABI edge. Its
        // register arguments are unknown even if a loop jumps back to it.
        if query.block == program.entry {
            return false;
        }
        let Some(position) = block.params.iter().position(|param| param.reg == query.reg) else {
            return false;
        };
        let mut incoming = false;
        for predecessor in &program.blocks {
            if !proof.scan() {
                return false;
            }
            // Call-result registers are defined by the terminator, not its
            // preceding ops. Do not mistake one for the pre-call value.
            if matches!(&predecessor.terminator, MachineTerminator::Call { success, .. }
                if success.target == query.block)
            {
                return false;
            }
            if !all_edges(&predecessor.terminator, |edge| {
                if !proof.scan() {
                    return false;
                }
                if edge.target != query.block {
                    return true;
                }
                incoming = true;
                edge.args.get(position).is_some_and(|arg| {
                    proof.include_value(predecessor.id, predecessor.ops.len(), *arg)
                })
            }) {
                return false;
            }
        }
        if !incoming {
            return false;
        }
    }
    true
}

fn all_edges(term: &MachineTerminator, mut visit: impl FnMut(&MachineEdge) -> bool) -> bool {
    match term {
        MachineTerminator::Jump(edge) => visit(edge),
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => visit(then_edge) && visit(else_edge),
        MachineTerminator::JumpTable { entries, .. } => entries.iter().all(visit),
        MachineTerminator::Call { success, .. } => visit(success),
        MachineTerminator::Return
        | MachineTerminator::ReturnScalar { .. }
        | MachineTerminator::TailCall { .. }
        | MachineTerminator::Trap { .. } => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::jit::machine::machine_ir::{
        MachineBlock, MachineBlockParam, MachineBranchCond, MachineCallRuntime, MachineConstId,
        MachineInst, MachineIntWidth, MachineRegOwner,
    };

    fn edge(target: u32, value: MachineValue) -> MachineEdge {
        MachineEdge {
            target: MachineBlockId(target),
            args: collections::vec![value],
        }
    }

    fn loop_program(initial: MachineValue) -> MachineProgram {
        let param = MachineBlockParam {
            reg: MachineReg(4),
            ty: MachineStorageType::GpWord,
            owner: MachineRegOwner::CachedCell,
        };
        MachineProgram {
            entry: MachineBlockId(0),
            fp_reg_init_widths: collections::Vec::new(),
            blocks: collections::vec![
                MachineBlock {
                    id: MachineBlockId(0),
                    params: collections::Vec::new(),
                    ops: collections::Vec::new(),
                    terminator: MachineTerminator::Jump(edge(1, initial)),
                },
                MachineBlock {
                    id: MachineBlockId(1),
                    params: collections::vec![param.clone()],
                    ops: collections::Vec::new(),
                    terminator: MachineTerminator::Branch {
                        cond: MachineBranchCond::Value(MachineValue::Reg(MachineReg(5))),
                        then_edge: edge(2, MachineValue::Reg(MachineReg(4))),
                        else_edge: edge(3, MachineValue::Reg(MachineReg(4))),
                    },
                },
                MachineBlock {
                    id: MachineBlockId(2),
                    params: collections::vec![param.clone()],
                    ops: collections::vec![MachineInst {
                        kind: MachineInstKind::Select {
                            ty: MachineStorageType::GpWord,
                            dst: MachineReg(4),
                            cond: MachineValue::Reg(MachineReg(5)),
                            on_true: MachineValue::Imm64(7),
                            on_false: MachineValue::Reg(MachineReg(4)),
                        }
                    }],
                    terminator: MachineTerminator::Jump(edge(1, MachineValue::Reg(MachineReg(4)))),
                },
                MachineBlock {
                    id: MachineBlockId(3),
                    params: collections::vec![param],
                    ops: collections::Vec::new(),
                    terminator: MachineTerminator::Return,
                },
            ],
        }
    }

    fn bounded(program: &MachineProgram) -> bool {
        index_is_in_bounds(
            program,
            MachineBlockId(1),
            MachineValue::Reg(MachineReg(4)),
            8,
        )
    }

    #[test]
    fn follows_every_loop_entry_and_copy_cycle_using_i32_indices() {
        for initial in [0, 3, 7, 0x1234_5678_0000_0003, 0xffff_ffff_0000_0007] {
            assert!(bounded(&loop_program(MachineValue::Imm64(initial))));
        }
        for initial in [8, 0x8000_0000, u64::MAX] {
            assert!(!bounded(&loop_program(MachineValue::Imm64(initial))));
        }
        assert!(!bounded(&loop_program(MachineValue::Reg(MachineReg(4)))));
        let mut program = loop_program(MachineValue::Imm64(0));
        program.entry = MachineBlockId(1);
        assert!(!bounded(&program));
    }

    #[test]
    fn masks_bound_unknown_inputs_but_arithmetic_and_late_clobbers_do_not() {
        let mut program = loop_program(MachineValue::Reg(MachineReg(4)));
        program.blocks[0].ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: MachineIntWidth::I32,
                op: MachineIntBinaryOp::And,
                dst: MachineReg(4),
                lhs: MachineValue::Reg(MachineReg(5)),
                rhs: MachineValue::Imm64(7),
            },
        });
        assert!(bounded(&program));
        if let MachineInstKind::IntBinary { op, .. } = &mut program.blocks[0].ops[0].kind {
            *op = MachineIntBinaryOp::Add;
        }
        assert!(!bounded(&program));
        program.blocks[0].ops[0].kind = MachineInstKind::Move {
            owner: MachineRegOwner::LinearValue,
            ty: MachineStorageType::GpWord,
            dst: MachineReg(4),
            src: MachineValue::Imm64(3),
        };
        assert!(bounded(&program));
        program.blocks[0].ops.push(MachineInst {
            kind: MachineInstKind::CallRuntime(MachineCallRuntime {
                metadata: MachineConstId(0),
            }),
        });
        assert!(!bounded(&program));
    }

    #[test]
    fn shared_cycle_cannot_hide_an_unknown_incoming_definition() {
        let mut program = loop_program(MachineValue::Imm64(0));
        program.blocks[3].terminator =
            MachineTerminator::Jump(edge(2, MachineValue::Reg(MachineReg(5))));
        assert!(!bounded(&program));
        program.blocks[3].terminator = MachineTerminator::Jump(edge(2, MachineValue::Imm64(15)));
        assert!(!bounded(&program));
    }

    #[test]
    fn call_results_do_not_inherit_the_pre_call_register_bound() {
        use crate::vm::jit::machine::machine_ir::{
            MachineCallArgs, MachineCallResults, MachineCallTarget, MachineFuncId, MachineResultDst,
        };
        let mut program = loop_program(MachineValue::Imm64(0));
        program.blocks[0].ops.push(MachineInst {
            kind: MachineInstKind::Move {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: MachineReg(4),
                src: MachineValue::Imm64(0),
            },
        });
        program.blocks[0].terminator = MachineTerminator::Call {
            target: MachineCallTarget::Direct(MachineFuncId(0)),
            frame_delta: 0,
            args: MachineCallArgs::default(),
            results: MachineCallResults::ScalarGp {
                dst: MachineResultDst::Reg(MachineReg(4)),
                ty: MachineStorageType::GpWord,
            },
            success: edge(1, MachineValue::Reg(MachineReg(4))),
        };
        assert!(!bounded(&program));
    }

    #[test]
    fn scan_budget_exhaustion_keeps_the_clamp() {
        let mut program = loop_program(MachineValue::Reg(MachineReg(4)));
        program.blocks[0].ops = collections::vec![MachineInst { kind: MachineInstKind::Move {
            owner: MachineRegOwner::LinearValue, ty: MachineStorageType::GpWord,
            dst: MachineReg(5), src: MachineValue::Imm64(0),
        }}; MAX_SCAN + 1];
        assert!(!bounded(&program));
    }
}
