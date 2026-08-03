//! Fold loop-bounded constant index adds into indexed-access displacements.
//!
//! Wasm compilers address `array[i]` through a constant base as
//! `i32.add(i, C)` feeding a zero-extended memory index with displacement
//! zero. The add wraps mod 2^32, so folding `C` into the displacement
//! blindly is unsound: when `i + C >= 2^32` the wrapped form addresses low
//! linear memory while the folded form faults in the guard region. When
//! `i` is an induction variable of a natural loop whose latch compare
//! bounds it, the no-wrap proof is available and the fold is exact —
//! `zext32(i + C) == zext32(i) + C` for every value `i` takes — so trap
//! behavior is bit-for-bit unchanged:
//!
//! ```text
//! i32.Add       t <- i, #C
//! indexed_load  dst <- [base + t(ZeroExtend32) + F]
//! ```
//! becomes
//! ```text
//! indexed_load  dst <- [base + i(ZeroExtend32) + F+C]
//! ```
//!
//! and the add dies with its last use. The folded displacement stays well
//! inside the 8 GiB guard reservation (u32 index max plus an i32
//! displacement). Only same-block windows are rewritten: cross-block
//! values travel exclusively through edge args in this MachineIR, so
//! `reg_live_after` decides liveness exactly, as in the sibling passes.

use crate::collections;
use crate::vm::jit::machine::machine_ir::{
    MachineBlock, MachineBlockId, MachineBranchCond, MachineCompareKind, MachineIndexExtend,
    MachineInstKind, MachineIntBinaryOp, MachineIntWidth, MachineProgram, MachineReg, MachineSign,
    MachineTerminator, MachineValue,
};

use super::helpers::{inst_defines, inst_uses_value, reg_live_after};
use super::hoist_loop_address_bases::{natural_loop_nodes, visit_edges, LoopGraph};

pub(super) fn fold_induction_offsets(program: &mut MachineProgram, loop_graph: &LoopGraph) {
    for header in 0..program.blocks.len() {
        let latches = &loop_graph.latches_by_header[header];
        if latches.len() != 1 {
            continue;
        }
        let latch = latches[0];
        let loop_nodes = natural_loop_nodes(header, latches, &loop_graph.predecessors);
        let inductions = find_inductions(
            &program.blocks,
            header,
            latch,
            &loop_nodes,
            &loop_graph.predecessors,
        );
        for induction in inductions {
            for &node in &loop_nodes {
                let limit = match induction.increment {
                    Some((inc_block, inc_op)) if inc_block == node => inc_op,
                    _ => usize::MAX,
                };
                fold_in_block(&mut program.blocks[node], &induction, limit);
            }
        }
    }
}

/// A header param proven to stay in `[0, max_value]` at every qualifying
/// candidate site of the loop body.
struct Induction {
    reg: MachineReg,
    max_value: u64,
    /// The in-place `reg += stride` site when the counter mutates in
    /// place; candidates at or after this op in that block see the
    /// post-increment value and never qualify.
    increment: Option<(usize, usize)>,
}

fn find_inductions(
    blocks: &[MachineBlock],
    header: usize,
    latch: usize,
    loop_nodes: &[usize],
    predecessors: &[collections::Vec<usize>],
) -> collections::Vec<Induction> {
    let mut found = collections::Vec::new();
    let header_id = blocks[header].id;
    for (position, param) in blocks[header].params.iter().enumerate() {
        let candidate = param.reg;
        let Some(latch_arg) = edge_arg_for(&blocks[latch].terminator, header_id, position) else {
            continue;
        };
        let MachineValue::Reg(step_reg) = latch_arg else {
            continue;
        };
        // Two counter shapes exist. Distinct step register:
        //     step <- i32.Add candidate, #S    ; latch carries step
        // and the in-place mutation the lowerer usually emits:
        //     candidate <- i32.Add candidate, #S ; latch carries candidate
        // In-place counters bound every candidate site BEFORE the
        // increment; the compare still tests the post-increment value.
        let in_place = step_reg == candidate;
        let Some((stride, increment)) =
            single_loop_def_as_const_add(blocks, loop_nodes, header, step_reg, candidate)
        else {
            continue;
        };
        if in_place {
            // The increment must live in the latch, and the latch's only
            // in-loop successor must be the header, so no loop block ever
            // observes the post-increment value.
            if increment.0 != latch
                || !latch_exits_or_reenters_only(blocks, loop_nodes, latch, header_id)
            {
                continue;
            }
        } else {
            // The candidate itself must stay unwritten inside the loop.
            if reg_defined_in_loop(blocks, loop_nodes, candidate, header) {
                continue;
            }
        }
        // Every entry edge must supply a compile-time constant.
        let Some(entry_max) =
            max_entry_constant(blocks, loop_nodes, predecessors, header, position)
        else {
            continue;
        };
        // The latch compare must bound the post-increment value.
        let Some(max_value) = bounded_max(
            &blocks[latch].terminator,
            header_id,
            step_reg,
            stride,
            entry_max,
        ) else {
            continue;
        };
        found.push(Induction {
            reg: candidate,
            max_value,
            increment: in_place.then_some(increment),
        });
    }
    found
}

/// Return the terminator's edge argument at `position` for the edge
/// targeting `target`, if there is exactly one such edge.
fn edge_arg_for(
    term: &MachineTerminator,
    target: MachineBlockId,
    position: usize,
) -> Option<MachineValue> {
    let mut result = None;
    let mut count = 0usize;
    visit_edges(term, |edge| {
        if edge.target == target {
            count += 1;
            result = edge.args.get(position).copied();
        }
    });
    if count == 1 {
        result
    } else {
        None
    }
}

/// If `reg` has exactly one definition inside the loop and it is
/// `i32.Add reg <- base, #stride`, return the stride and the site.
/// A header param definition of `reg` is expected for the in-place form
/// (`reg == base`) and does not count against single-def tracking; any
/// other param definition disqualifies.
fn single_loop_def_as_const_add(
    blocks: &[MachineBlock],
    loop_nodes: &[usize],
    header: usize,
    reg: MachineReg,
    base: MachineReg,
) -> Option<(u64, (usize, usize))> {
    let mut stride = None;
    let mut site = None;
    let mut defs = 0usize;
    for &node in loop_nodes {
        for param in &blocks[node].params {
            if param.reg == reg && !(reg == base && node == header) {
                defs += 2; // a non-header param def disqualifies tracking
            }
        }
        for (op_index, inst) in blocks[node].ops.iter().enumerate() {
            if !inst_defines(&inst.kind, reg) {
                continue;
            }
            defs += 1;
            if let MachineInstKind::IntBinary {
                width: MachineIntWidth::I32,
                op: MachineIntBinaryOp::Add,
                dst,
                lhs,
                rhs,
            } = &inst.kind
            {
                if *dst == reg {
                    stride = const_add_operand(*lhs, *rhs, base);
                    site = Some((node, op_index));
                }
            }
        }
    }
    if defs == 1 {
        Some((stride.filter(|&s| s > 0)?, site?))
    } else {
        None
    }
}

/// True when every edge of the latch terminator either re-enters the
/// header or leaves the loop entirely.
fn latch_exits_or_reenters_only(
    blocks: &[MachineBlock],
    loop_nodes: &[usize],
    latch: usize,
    header_id: MachineBlockId,
) -> bool {
    let mut ok = true;
    visit_edges(&blocks[latch].terminator, |edge| {
        if edge.target == header_id {
            return;
        }
        let inside = loop_nodes
            .iter()
            .any(|&node| blocks[node].id == edge.target);
        if inside {
            ok = false;
        }
    });
    ok
}

/// For `add lhs, rhs` with one operand `Reg(base)` and the other a u32
/// immediate, return the immediate.
fn const_add_operand(lhs: MachineValue, rhs: MachineValue, base: MachineReg) -> Option<u64> {
    let imm = match (lhs, rhs) {
        (MachineValue::Reg(reg), MachineValue::Imm64(imm)) if reg == base => imm,
        (MachineValue::Imm64(imm), MachineValue::Reg(reg)) if reg == base => imm,
        _ => return None,
    };
    (imm <= u64::from(u32::MAX)).then_some(imm)
}

fn reg_defined_in_loop(
    blocks: &[MachineBlock],
    loop_nodes: &[usize],
    reg: MachineReg,
    header: usize,
) -> bool {
    for &node in loop_nodes {
        if node != header && blocks[node].params.iter().any(|param| param.reg == reg) {
            return true;
        }
        if blocks[node]
            .ops
            .iter()
            .any(|inst| inst_defines(&inst.kind, reg))
        {
            return true;
        }
    }
    false
}

/// The largest constant any loop-entry edge supplies for the header param
/// at `position`; `None` when any entry value is not a compile-time
/// constant. Accepts immediate edge args and args whose final definition
/// in the entry block is a `Move` from an immediate.
fn max_entry_constant(
    blocks: &[MachineBlock],
    loop_nodes: &[usize],
    predecessors: &[collections::Vec<usize>],
    header: usize,
    position: usize,
) -> Option<u64> {
    let header_id = blocks[header].id;
    let mut max = None;
    for &pred in &predecessors[header] {
        if loop_nodes.contains(&pred) {
            continue;
        }
        let mut ok = true;
        visit_edges(&blocks[pred].terminator, |edge| {
            if edge.target != header_id {
                return;
            }
            let constant = match edge.args.get(position) {
                Some(MachineValue::Imm64(imm)) => Some(*imm),
                Some(MachineValue::Reg(reg)) => final_const_move(&blocks[pred], *reg),
                _ => None,
            };
            match constant {
                Some(value) if value <= u64::from(u32::MAX) => {
                    max = Some(max.map_or(value, |m: u64| m.max(value)));
                }
                _ => ok = false,
            }
        });
        if !ok {
            return None;
        }
    }
    max
}

/// The value of `reg` at the end of `block` if its final definition there
/// is a move from an immediate.
fn final_const_move(block: &MachineBlock, reg: MachineReg) -> Option<u64> {
    if block.params.iter().any(|param| param.reg == reg) && block.ops.is_empty() {
        return None;
    }
    let mut value = None;
    for inst in &block.ops {
        if inst_defines(&inst.kind, reg) {
            value = match &inst.kind {
                MachineInstKind::Move {
                    src: MachineValue::Imm64(imm),
                    ..
                } => Some(*imm),
                _ => None,
            };
        }
    }
    value
}

/// Derive the maximum body value of the induction from the latch's
/// bounding compare on `step_reg` (the post-increment value), or `None`
/// when the shape is not a recognized bound.
fn bounded_max(
    term: &MachineTerminator,
    header_id: MachineBlockId,
    step_reg: MachineReg,
    stride: u64,
    entry_max: u64,
) -> Option<u64> {
    let MachineTerminator::Branch {
        cond:
            MachineBranchCond::IntCompare {
                width: MachineIntWidth::I32,
                kind,
                sign: MachineSign::Unsigned,
                lhs: MachineValue::Reg(cmp_reg),
                rhs: MachineValue::Imm64(bound),
            },
        then_edge,
        else_edge,
    } = term
    else {
        return None;
    };
    if *cmp_reg != step_reg || *bound > u64::from(u32::MAX) {
        return None;
    }
    let continue_on_true = then_edge.target == header_id;
    if !continue_on_true && else_edge.target != header_id {
        return None;
    }
    // Normalize to the condition under which the loop CONTINUES.
    let continue_kind = if continue_on_true {
        *kind
    } else {
        match kind {
            MachineCompareKind::Eq => MachineCompareKind::Ne,
            MachineCompareKind::Ne => MachineCompareKind::Eq,
            MachineCompareKind::Ge => MachineCompareKind::Lt,
            MachineCompareKind::Lt => MachineCompareKind::Ge,
            MachineCompareKind::Gt => MachineCompareKind::Le,
            MachineCompareKind::Le => MachineCompareKind::Gt,
        }
    };
    match continue_kind {
        // Continue while `d != bound`: every re-entering value satisfies
        // `d != bound`, but the range claim needs stride reachability so
        // the counter cannot step over the bound and wrap.
        MachineCompareKind::Ne => {
            if *bound < entry_max || (*bound - entry_max) % stride != 0 {
                return None;
            }
            if *bound == entry_max {
                Some(entry_max)
            } else {
                Some(entry_max.max(*bound - stride))
            }
        }
        // Continue while `d < bound`: every re-entering value is below
        // the bound; the first iteration sees the entry constant.
        MachineCompareKind::Lt => Some(entry_max.max(bound.checked_sub(1)?)),
        _ => None,
    }
}

/// Rewrite every provable `add + indexed access` window in one block.
/// Candidates at or after `limit` (the in-place increment site) see the
/// post-increment counter and never qualify.
fn fold_in_block(block: &mut MachineBlock, induction: &Induction, mut limit: usize) {
    let mut index = 0;
    while index < block.ops.len() && index < limit {
        let Some((temp, constant)) = fold_candidate(&block.ops[index].kind, induction) else {
            index += 1;
            continue;
        };
        if let Some(uses) = conforming_use_window(block, index, temp, constant) {
            for &use_index in &uses {
                rewrite_access(
                    &mut block.ops[use_index].kind,
                    temp,
                    induction.reg,
                    constant,
                );
            }
            block.ops.remove(index);
            limit = limit.saturating_sub(1);
            // Do not advance: the next instruction shifted into `index`.
            continue;
        }
        index += 1;
    }
}

/// Is this `i32.Add temp <- induction, #C` with a provably wrap-free sum?
fn fold_candidate(kind: &MachineInstKind, induction: &Induction) -> Option<(MachineReg, u64)> {
    let MachineInstKind::IntBinary {
        width: MachineIntWidth::I32,
        op: MachineIntBinaryOp::Add,
        dst,
        lhs,
        rhs,
    } = kind
    else {
        return None;
    };
    if *dst == induction.reg {
        return None;
    }
    let constant = const_add_operand(*lhs, *rhs, induction.reg)?;
    (induction.max_value + constant <= u64::from(u32::MAX)).then_some((*dst, constant))
}

/// Collect the uses of `temp` from just past the add to its redefinition
/// or the block end. All uses must be zero-extended index positions of
/// indexed accesses whose displacement can absorb `constant`; the window
/// must end in a redefinition or with `temp` dead.
fn conforming_use_window(
    block: &MachineBlock,
    add_index: usize,
    temp: MachineReg,
    constant: u64,
) -> Option<collections::Vec<usize>> {
    let mut uses = collections::Vec::new();
    let mut cursor = add_index + 1;
    while cursor < block.ops.len() {
        let kind = &block.ops[cursor].kind;
        if inst_uses_value(kind, temp) {
            if !is_foldable_index_use(kind, temp, constant) {
                return None;
            }
            uses.push(cursor);
        }
        if inst_defines(kind, temp) {
            return (!uses.is_empty()).then_some(uses);
        }
        cursor += 1;
    }
    if uses.is_empty() {
        return None;
    }
    let after_last = uses.last().copied()? + 1;
    if reg_live_after(&block.ops[after_last..], &block.terminator, temp) {
        return None;
    }
    Some(uses)
}

fn is_foldable_index_use(kind: &MachineInstKind, temp: MachineReg, constant: u64) -> bool {
    let (base, index, index_extend, offset, uses_temp_elsewhere) = match kind {
        MachineInstKind::IndexedLoad {
            base,
            index,
            index_extend,
            offset,
            ..
        } => (*base, *index, *index_extend, *offset, false),
        MachineInstKind::IndexedStore {
            base,
            index,
            index_extend,
            offset,
            src,
            ..
        } => (
            *base,
            *index,
            *index_extend,
            *offset,
            matches!(src, MachineValue::Reg(reg) if *reg == temp),
        ),
        _ => return false,
    };
    index == temp
        && base != temp
        && !uses_temp_elsewhere
        && index_extend == MachineIndexExtend::ZeroExtend32
        && offset >= 0
        && i64::from(offset) + constant as i64 <= i64::from(i32::MAX)
}

fn rewrite_access(
    kind: &mut MachineInstKind,
    temp: MachineReg,
    induction_reg: MachineReg,
    constant: u64,
) {
    match kind {
        MachineInstKind::IndexedLoad { index, offset, .. }
        | MachineInstKind::IndexedStore { index, offset, .. }
            if *index == temp =>
        {
            *index = induction_reg;
            *offset += constant as i32;
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::jit::machine::machine_ir::{
        MachineBlockParam, MachineEdge, MachineInst, MachineLoadExtension, MachineMemWidth,
        MachineRegOwner, MachineStorageType,
    };

    const MEM: MachineReg = MachineReg(2);
    const COUNTER: MachineReg = MachineReg(4);
    const TEMP: MachineReg = MachineReg(5);
    const DATA: MachineReg = MachineReg(12);

    fn add32(dst: MachineReg, src: MachineReg, imm: u64) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IntBinary {
                width: MachineIntWidth::I32,
                op: MachineIntBinaryOp::Add,
                dst,
                lhs: MachineValue::Reg(src),
                rhs: MachineValue::Imm64(imm),
            },
        }
    }

    fn load_idx(dst: MachineReg, index: MachineReg, offset: i32) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::IndexedLoad {
                dst,
                base: MEM,
                index,
                index_extend: MachineIndexExtend::ZeroExtend32,
                offset,
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
            },
        }
    }

    fn counter_loop(
        bound: u64,
        stride: u64,
        body: collections::Vec<MachineInst>,
    ) -> MachineProgram {
        let mut ops = body;
        ops.push(add32(COUNTER, COUNTER, stride));
        MachineProgram {
            entry: MachineBlockId(2),
            fp_reg_init_widths: collections::Vec::new(),
            blocks: collections::vec![
                MachineBlock {
                    id: MachineBlockId(0),
                    params: collections::vec![MachineBlockParam::gp_word(COUNTER)],
                    ops,
                    terminator: MachineTerminator::Branch {
                        cond: MachineBranchCond::IntCompare {
                            width: MachineIntWidth::I32,
                            kind: MachineCompareKind::Ne,
                            sign: MachineSign::Unsigned,
                            lhs: MachineValue::Reg(COUNTER),
                            rhs: MachineValue::Imm64(bound),
                        },
                        then_edge: MachineEdge {
                            target: MachineBlockId(0),
                            args: collections::vec![MachineValue::Reg(COUNTER)],
                        },
                        else_edge: MachineEdge {
                            target: MachineBlockId(1),
                            args: collections::Vec::new(),
                        },
                    },
                },
                MachineBlock {
                    id: MachineBlockId(1),
                    params: collections::Vec::new(),
                    ops: collections::Vec::new(),
                    terminator: MachineTerminator::Return,
                },
                MachineBlock {
                    id: MachineBlockId(2),
                    params: collections::Vec::new(),
                    ops: collections::vec![MachineInst {
                        kind: MachineInstKind::Move {
                            owner: MachineRegOwner::LinearValue,
                            ty: MachineStorageType::GpWord,
                            dst: COUNTER,
                            src: MachineValue::Imm64(0),
                        },
                    }],
                    terminator: MachineTerminator::Jump(MachineEdge {
                        target: MachineBlockId(0),
                        args: collections::vec![MachineValue::Reg(COUNTER)],
                    }),
                },
            ],
        }
    }

    fn run(program: &mut MachineProgram) {
        let graph = super::super::hoist_loop_address_bases::analyze_loop_graph(
            &program.blocks,
            program.entry,
        );
        fold_induction_offsets(program, &graph);
    }

    #[test]
    fn folds_bounded_in_place_counter_add_into_displacement() {
        let mut program = counter_loop(
            0x200000,
            32,
            collections::vec![add32(TEMP, COUNTER, 0x1000), load_idx(DATA, TEMP, 0)],
        );
        run(&mut program);
        let ops = &program.blocks[0].ops;
        assert_eq!(ops.len(), 2, "the candidate add must be deleted");
        assert_eq!(
            ops[0].kind,
            load_idx(DATA, COUNTER, 0x1000).kind,
            "the load must index the counter with the folded displacement"
        );
    }

    #[test]
    fn no_fold_when_the_sum_may_wrap() {
        let body = collections::vec![
            add32(TEMP, COUNTER, u64::from(u32::MAX) - 0x100),
            load_idx(DATA, TEMP, 0),
        ];
        let mut program = counter_loop(0x200000, 32, body.clone());
        run(&mut program);
        assert_eq!(
            program.blocks[0].ops[..2],
            body[..],
            "wrap risk must block the fold"
        );
    }

    #[test]
    fn no_fold_without_stride_reachability_of_the_bound() {
        // stride 32 cannot hit the odd bound: the counter would step over
        // it and keep running, so no range is provable.
        let body = collections::vec![add32(TEMP, COUNTER, 0x1000), load_idx(DATA, TEMP, 0)];
        let mut program = counter_loop(0x200001, 32, body.clone());
        run(&mut program);
        assert_eq!(program.blocks[0].ops[..2], body[..]);
    }

    #[test]
    fn no_fold_after_the_increment() {
        // A candidate placed after `counter += stride` sees the
        // post-increment value; the proof does not cover it.
        let mut program = counter_loop(0x200000, 32, collections::Vec::new());
        program.blocks[0].ops.push(add32(TEMP, COUNTER, 0x1000));
        program.blocks[0].ops.push(load_idx(DATA, TEMP, 0));
        run(&mut program);
        assert_eq!(program.blocks[0].ops.len(), 3, "nothing may be rewritten");
    }

    #[test]
    fn no_fold_when_the_temp_escapes_as_a_store_source() {
        let mut program = counter_loop(
            0x200000,
            32,
            collections::vec![
                add32(TEMP, COUNTER, 0x1000),
                MachineInst {
                    kind: MachineInstKind::IndexedStore {
                        base: MEM,
                        index: TEMP,
                        index_extend: MachineIndexExtend::ZeroExtend32,
                        offset: 0,
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(TEMP),
                    },
                },
            ],
        );
        run(&mut program);
        assert_eq!(
            program.blocks[0].ops.len(),
            3,
            "a non-index use of the temp must block the fold"
        );
    }
}
