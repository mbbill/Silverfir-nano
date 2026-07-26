//! The folding predecoder: wasm bytecode → folded instruction stream.
//!
//! Single forward pass with a compile-time symbolic stack. Routing opcodes
//! (`local.get/set/tee`, consts) emit nothing; semantic ops emit one
//! instruction with the routing folded into operand/destination fields.
//! See `mcts_mem/silverfir/interpreter/dispatch.md` for the model and the
//! soundness rules implemented here; the rules were adversarially reviewed
//! on the `foldsim` reference model before this port.
//!
//! v1 (stage A1) coverage: i32/i64 ALU, compares, conversions, locals,
//! consts, `block`/`loop`/`if`/`else`/`end`/`br`/`br_if`, `select`,
//! `call`/`call_indirect`, `return`, `unreachable`. Everything else returns
//! a clean "unsupported" error.
//!
//! Frame model: `[ params | locals | temps by height ]`, all u64 slots. The
//! executor (stage A2) zero-initializes non-param locals; the predecoder
//! only computes the frame size.

use crate::collections::{vec, Vec};
use crate::error::WasmError;
use crate::module::type_context::TypeContext;
use crate::module::Module;
use crate::op_decoder::{BlockType, Decoder, Immediate, OpStream, OpcodeHandler};
use crate::opcodes::{Opcode, OpcodeFC, WasmOpcode};
use crate::utils::limits::Limitable;

use super::instr::{
    operand_is_float, result_is_float, Instr, Op, FLAG_ADDR64, FLAG_A_ACC, FLAG_A_CONST,
    FLAG_B_ACC, FLAG_B_CONST, FLAG_DST_ACC, FLAG_FUSED,
};

/// Producer index meaning "no patchable producer" (call results, block
/// results arriving over a merge).
const NO_DEF: u32 = u32::MAX;
/// Branch-target placeholder while the target's `end` is still ahead.
const FIXUP: u64 = u64::MAX;
/// Null funcref representation (function indices are table/ref values).
pub(super) const NULL_FUNCREF: u64 = u64::MAX;

pub(crate) struct PredecodedFunction {
    pub code: Vec<Instr>,
    /// Side tables for `BrTable`: resolved instruction indices, the last
    /// entry is the default target.
    pub br_tables: Vec<Vec<u32>>,
    /// Total frame slots: params + locals + max temp height.
    pub frame_slots: u32,
    /// params + locals (== the base slot index of temps).
    pub n_locals: u32,
    pub n_params: u32,
    pub n_results: u32,
}

/// Predecode one local (non-import) function of a parsed module.
pub(crate) fn predecode_function(
    module: &Module,
    func_index: usize,
) -> Result<PredecodedFunction, WasmError> {
    let func = module
        .functions()
        .get(func_index)
        .ok_or(WasmError::invalid("interp: function index out of range"))?;
    let spec = func
        .spec()
        .ok_or(WasmError::invalid("interp: cannot predecode an import"))?;
    let n_params = func.func_type().params().len() as u32;
    let n_results = func.func_type().results().len() as u32;
    let n_locals = n_params + spec.locals().len() as u32;

    let mut p = Predecoder {
        types: module.types(),
        module,
        code: Vec::new(),
        stack: Vec::new(),
        frames: Vec::new(),
        dead: false,
        region: 0,
        last_read: vec![0u32; n_locals as usize],
        last_write: vec![0u32; n_locals as usize],
        last_mat_mov: NO_DEF,
        last_mat_region: 0,
        last_write_region: vec![0u32; n_locals as usize],
        br_tables: Vec::new(),
        max_height: 0,
        n_locals,
        n_results,
        last_call_idx: NO_DEF,
        last_call_height: 0,
        last_call_region: 0,
    };
    {
        let mut decoder = Decoder::new(spec.code());
        decoder.add_handler(&mut p);
        decoder.decode_function()?;
    }
    Ok(PredecodedFunction {
        frame_slots: n_locals + p.max_height,
        code: p.code,
        br_tables: p.br_tables,
        n_locals,
        n_params,
        n_results,
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Desc {
    /// A pending `local.get` (slot index < n_locals).
    Local(u32),
    /// A pending constant.
    ConstV(u64),
    /// A materialized value in the temp slot for `height`, produced by
    /// emitted instruction `def` in control region `region`.
    Temp { height: u32, def: u32, region: u32 },
}

#[derive(Clone, Copy)]
enum Fixup {
    /// Patch `code[i].c`.
    InstrC(u32),
    /// Patch `br_tables[tbl][entry]`.
    Table { tbl: u32, entry: u32 },
}

struct CtlFrame {
    /// Stack height at entry, params excluded (branch results land here).
    base: u32,
    params: u32,
    results: u32,
    is_loop: bool,
    is_if: bool,
    dead_entry: bool,
    /// A br targets this block's end (never set for loops: a br to a loop
    /// targets its header).
    end_targeted: bool,
    saw_else: bool,
    then_fell_live: bool,
    /// Loop header instruction index (loops only).
    header: u32,
    /// The `if`'s BrIfNot awaiting its else/end target.
    else_fixup: Option<u32>,
    /// Sites whose branch target awaits this block's end.
    fixups: Vec<Fixup>,
}

struct Predecoder<'m> {
    types: &'m TypeContext,
    module: &'m Module,
    code: Vec<Instr>,
    stack: Vec<Desc>,
    frames: Vec<CtlFrame>,
    dead: bool,
    region: u32,
    /// Per local: 1 + index of the last emitted instruction reading /
    /// writing it (0 = never). Used by the dst-folding soundness rules.
    last_read: Vec<u32>,
    last_write: Vec<u32>,
    /// Index of an immediately preceding materialization MovSlot that a
    /// next one may merge with (NO_DEF = none), and its region.
    last_mat_mov: u32,
    last_mat_region: u32,
    /// Control region of the last write per local (acc write-through:
    /// a consumer may read the accumulator only when the write happened
    /// in the same region — a merge between them means the consumer can
    /// be reached with a stale accumulator).
    last_write_region: Vec<u32>,
    br_tables: Vec<Vec<u32>>,
    max_height: u32,
    n_locals: u32,
    n_results: u32,
    /// Cell index / result height / region of the last emitted
    /// SINGLE-result call (u32::MAX = none). An adjacent consumer of that
    /// result reads the accumulator: the native Return leaves result 0
    /// in it, and the driver carries it across activation boundaries.
    last_call_idx: u32,
    last_call_height: u32,
    last_call_region: u32,
}

fn block_arity(types: &TypeContext, bt: &BlockType) -> Result<(u32, u32), WasmError> {
    match bt {
        BlockType::Empty => Ok((0, 0)),
        BlockType::ValueType(_) => Ok((0, 1)),
        BlockType::TypeIndex(idx) => types
            .get_function_type(*idx as u32)
            .map(|ft| (ft.params().len() as u32, ft.results().len() as u32))
            .ok_or(WasmError::invalid("interp: bad block type index")),
    }
}

fn desync() -> WasmError {
    // The input is validated wasm; reaching this means a predecoder bug,
    // reported as a clean error rather than a silent mis-compile.
    WasmError::invalid("interp: predecoder stack desync")
}

/// The fused compare-and-branch op for a condition producer: the
/// branch-if-true sense, or the inverted sense for `br_if_not`-style
/// guards. `I32_Eqz` folds to the plain conditional branches. (`I64_Eqz`
/// must NOT fold: `BrIf` tests only the low 32 bits of the condition.)
fn fuse_cmp_br(op: Op, invert: bool) -> Option<Op> {
    use Op::*;
    let (taken, inverted) = match op {
        I32_Eq => (I32_BrEq, I32_BrNe),
        I32_Ne => (I32_BrNe, I32_BrEq),
        I32_LtS => (I32_BrLtS, I32_BrGeS),
        I32_LtU => (I32_BrLtU, I32_BrGeU),
        I32_GtS => (I32_BrGtS, I32_BrLeS),
        I32_GtU => (I32_BrGtU, I32_BrLeU),
        I32_LeS => (I32_BrLeS, I32_BrGtS),
        I32_LeU => (I32_BrLeU, I32_BrGtU),
        I32_GeS => (I32_BrGeS, I32_BrLtS),
        I32_GeU => (I32_BrGeU, I32_BrLtU),
        I64_Eq => (I64_BrEq, I64_BrNe),
        I64_Ne => (I64_BrNe, I64_BrEq),
        I64_LtS => (I64_BrLtS, I64_BrGeS),
        I64_LtU => (I64_BrLtU, I64_BrGeU),
        I64_GtS => (I64_BrGtS, I64_BrLeS),
        I64_GtU => (I64_BrGtU, I64_BrLeU),
        I64_LeS => (I64_BrLeS, I64_BrGtS),
        I64_LeU => (I64_BrLeU, I64_BrGtU),
        I64_GeS => (I64_BrGeS, I64_BrLtS),
        I64_GeU => (I64_BrGeU, I64_BrLtU),
        I32_Eqz => (BrIfNot, BrIf),
        I32_And => (I32_BrAnd, I32_BrAndNot),
        _ => return None,
    };
    Some(if invert { inverted } else { taken })
}

/// The inverted compare, for folding `i32.eqz` over a compare result.
fn invert_cmp(op: Op) -> Option<Op> {
    use Op::*;
    Some(match op {
        I32_Eq => I32_Ne,
        I32_Ne => I32_Eq,
        I32_LtS => I32_GeS,
        I32_LtU => I32_GeU,
        I32_GtS => I32_LeS,
        I32_GtU => I32_LeU,
        I32_LeS => I32_GtS,
        I32_LeU => I32_GtU,
        I32_GeS => I32_LtS,
        I32_GeU => I32_LtU,
        I64_Eq => I64_Ne,
        I64_Ne => I64_Eq,
        I64_LtS => I64_GeS,
        I64_LtU => I64_GeU,
        I64_GtS => I64_LeS,
        I64_GtU => I64_LeU,
        I64_LeS => I64_GtS,
        I64_LeU => I64_GtU,
        I64_GeS => I64_LtS,
        I64_GeU => I64_LtU,
        _ => return None,
    })
}

/// An opcode this engine does not implement.
///
/// Named by family, because "not yet supported" on its own cannot tell
/// SIMD from GC from a tail call -- and deciding what the interpreter
/// should implement next is exactly the question that needs answering.
fn unsupported_opcode(op: WasmOpcode) -> WasmError {
    match op {
        WasmOpcode::FD(_) => WasmError::invalid("interp: SIMD is not supported"),
        WasmOpcode::FB(_) => WasmError::invalid("interp: GC is not supported"),
        _ => WasmError::invalid("interp: opcode not yet supported by the interpreter"),
    }
}

/// A shape the folded stack machine cannot express, as opposed to an
/// opcode it has never heard of.
fn unsupported() -> WasmError {
    WasmError::invalid("interp: opcode not yet supported by the interpreter")
}

impl<'m> Predecoder<'m> {
    /// Whether the memory at `idx` is 64-bit addressed.
    ///
    /// Only the memory's own declaration decides this, so an access is marked
    /// once at predecode and never re-checked on the hot path.
    /// Whether the table at `idx` is 64-bit indexed.
    fn table_is_64(&self, idx: u64) -> bool {
        self.module
            .tables()
            .get(idx as usize)
            .is_some_and(|t| t.limits().is64)
    }

    fn memory_is_64(&self, idx: u64) -> bool {
        self.module
            .memories()
            .get(idx as usize)
            .is_some_and(|m| m.limits().is64)
    }

    fn height(&self) -> u32 {
        self.stack.len() as u32
    }

    fn temp_slot(&self, height: u32) -> u64 {
        (self.n_locals + height) as u64
    }

    /// Like `temp_slot`, but records the slot as actually used (read or
    /// written). Frame sizing counts only used slots: a dst-folded
    /// producer's original temp slot is never touched at run time.
    fn temp_slot_used(&mut self, height: u32) -> u64 {
        self.max_height = self.max_height.max(height + 1);
        (self.n_locals + height) as u64
    }

    fn emit(&mut self, op: Op, flags: u16, a: u64, b: u64, c: u64) -> u32 {
        let idx = self.code.len() as u32;
        self.code.push(Instr::new(op, flags, a, b, c));
        idx
    }

    fn bump_region(&mut self) {
        self.region += 1;
    }

    /// Resolve one popped descriptor into an operand field, recording local
    /// reads for the dst-folding rules. `at` is the index the consuming
    /// instruction will get.
    fn operand(&mut self, d: Desc, at: u32) -> (u64, bool) {
        match d {
            Desc::Local(i) => {
                self.last_read[i as usize] = at + 1;
                (i as u64, false)
            }
            Desc::ConstV(k) => (k, true),
            Desc::Temp { height, .. } => (self.temp_slot_used(height), false),
        }
    }

    // The result temp's slot is NOT marked used here: if the set that
    // consumes it dst-folds, the slot never exists at run time. Every
    // reader/writer path goes through `temp_slot_used`.
    fn push_result_temp(&mut self, def: u32) {
        let height = self.height();
        self.stack.push(Desc::Temp {
            height,
            def,
            region: self.region,
        });
    }

    fn push_unknown_temps(&mut self, n: u32) {
        for _ in 0..n {
            let height = self.height();
            self.max_height = self.max_height.max(height + 1);
            self.stack.push(Desc::Temp {
                height,
                def: NO_DEF,
                region: self.region,
            });
        }
    }

    /// Materialize the stack entry at position `i` into its canonical temp
    /// slot, if it is still a pending descriptor.
    fn materialize_at(&mut self, i: usize) {
        let height = i as u32;
        let d = self.stack[i];
        let idx = match d {
            Desc::Local(l) => {
                let slot = self.temp_slot_used(height);
                let at = self.code.len() as u32;
                self.last_read[l as usize] = at + 1;
                // Adjacent staging movs merge into one MovPair dispatch
                // (strict in-order copies, so src2 == dst1 stays exact).
                if self.last_mat_mov != NO_DEF
                    && self.last_mat_mov + 1 == at
                    && self.last_mat_region == self.region
                    && self.code[self.last_mat_mov as usize].op == Op::MovSlot
                    && self.code[self.last_mat_mov as usize].flags == 0
                {
                    let prev = self.last_mat_mov;
                    let pm = self.code[prev as usize];
                    self.code[prev as usize] = Instr {
                        op: Op::MovPair,
                        flags: 0,
                        a: pm.a,
                        b: l as u64,
                        c: pm.c << 32 | slot,
                    };
                    self.last_mat_mov = NO_DEF; // pairs never re-pair
                    prev
                } else {
                    let idx = self.emit(Op::MovSlot, 0, l as u64, 0, slot);
                    self.last_mat_mov = idx;
                    self.last_mat_region = self.region;
                    idx
                }
            }
            Desc::ConstV(k) => {
                let slot = self.temp_slot_used(height);
                self.emit(Op::MovConst, FLAG_A_CONST, k, 0, slot)
            }
            Desc::Temp { .. } => return,
        };
        self.stack[i] = Desc::Temp {
            height,
            def: idx,
            region: self.region,
        };
    }

    fn materialize_all(&mut self) {
        for i in 0..self.stack.len() {
            self.materialize_at(i);
        }
    }

    fn pop(&mut self) -> Result<Desc, WasmError> {
        self.stack.pop().ok_or_else(desync)
    }

    /// A condition descriptor whose producer can be rewritten in place:
    /// the immediately preceding instruction, same control region, with its
    /// dst still pointing at the descriptor's own temp slot (not folded
    /// into a local). Returns the producer index.
    fn rewritable_producer(&self, cond: Desc) -> Option<u32> {
        let Desc::Temp {
            def,
            region,
            height,
        } = cond
        else {
            return None;
        };
        if def == NO_DEF || def + 1 != self.code.len() as u32 || region != self.region {
            return None;
        }
        let ins = &self.code[def as usize];
        let cdst = if ins.op == Op::Select || ins.op == Op::MovPair || ins.flags & FLAG_FUSED != 0 {
            ins.c & 0xffff_ffff
        } else {
            ins.c
        };
        if cdst != self.temp_slot(height) {
            return None;
        }
        Some(def)
    }

    /// Try to mark the acc edge for a just-popped operand: if its producer
    /// is rewritable (immediately preceding, same region, dst unfolded),
    /// the producer keeps the value in the accumulator and the consumer —
    /// the cell the caller is about to emit — reads it from there. Returns
    /// the operand's acc flag, or 0. At most one operand of a consumer can
    /// match (only one instruction sits at `len-1`), and slot fields stay
    /// valid either way: the hints are droppable by construction.
    fn acc_operand(&mut self, d: Desc, which_flag: u16, want_float: bool) -> u16 {
        match d {
            Desc::Temp { height, def, .. } => {
                if let Some(def) = self.rewritable_producer(d) {
                    // Domain agreement: the producer's result rides the
                    // accumulator of ITS domain; a consumer reading the
                    // other domain's register must fall back to the slot.
                    if result_is_float(self.code[def as usize].op) != want_float {
                        return 0;
                    }
                    self.code[def as usize].flags |= FLAG_DST_ACC;
                    which_flag
                } else if def == NO_DEF
                    && !want_float
                    && self.last_call_idx != NO_DEF
                    && self.last_call_idx + 1 == self.code.len() as u32
                    && height == self.last_call_height
                    && self.last_call_region == self.region
                {
                    // Adjacent consumer of a single-result call: the native
                    // Return leaves result 0 in the INTEGER accumulator (it
                    // is a raw-bits copy), and the driver carries it across
                    // activation boundaries. No producer-side mark exists;
                    // float-domain consumers read the slot instead.
                    which_flag
                } else {
                    0
                }
            }
            // Write-through residency: every native value handler leaves
            // its result in the accumulator, so a local written by the
            // immediately preceding instruction (folded dst or mov) can
            // be read from the accumulator — the slot stays written, no
            // producer-side mark exists. Both operands of one consumer
            // may read the same just-written local.
            Desc::Local(x) => {
                // last_write is 1 + writer index (0 = never written), so
                // equality with the code length means "written by the
                // immediately preceding instruction" — and the nonzero
                // gate keeps never-written locals at function start from
                // colliding with an empty stream.
                if self.last_write[x as usize] > 0
                    && self.last_write[x as usize] == self.code.len() as u32
                    && self.last_write_region[x as usize] == self.region
                    && result_is_float(self.code[self.last_write[x as usize] as usize - 1].op)
                        == want_float
                {
                    which_flag
                } else {
                    0
                }
            }
            Desc::ConstV(_) => 0,
        }
    }

    /// Emit a value-producing instruction consuming `n` operands (1 or 2)
    /// and pushing its result temp.
    fn value_op(&mut self, op: Op, n: u32) -> Result<(), WasmError> {
        // cmp+eqz combo: `i32.eqz` over a just-emitted, unfolded compare
        // becomes the inverted compare in place (no new instruction; the
        // operand descriptor stays valid — same slot, same producer).
        if op == Op::I32_Eqz {
            if let Some(&cond) = self.stack.last() {
                if let Some(def) = self.rewritable_producer(cond) {
                    if let Some(inv) = invert_cmp(self.code[def as usize].op) {
                        self.code[def as usize].op = inv;
                        return Ok(());
                    }
                }
            }
        }
        let at = self.code.len() as u32;
        let mut flags = 0u16;
        let (b, b_const) = if n == 2 {
            let d = self.pop()?;
            let r = self.operand(d, at);
            flags |= self.acc_operand(d, FLAG_B_ACC, operand_is_float(op, true));
            r
        } else {
            (0, false)
        };
        let d = self.pop()?;
        let (a, a_const) = self.operand(d, at);
        flags |= self.acc_operand(d, FLAG_A_ACC, operand_is_float(op, false));
        if a_const {
            flags |= FLAG_A_CONST;
        }
        if b_const {
            flags |= FLAG_B_CONST;
        }
        // The dst slot is marked used at EMIT time: even if the consuming
        // set later dst-folds this producer, control flow may discard the
        // descriptor (else/end resets) while the write still executes, so
        // the slot must exist. Folding leaves at most a one-slot surplus.
        let dst = self.temp_slot_used(self.height());
        let idx = self.emit(op, flags, a, b, dst);
        self.push_result_temp(idx);
        Ok(())
    }

    /// Retro-patch the destination field of a producer instruction.
    fn patch_dst(&mut self, def: u32, slot: u64) {
        let instr = &mut self.code[def as usize];
        if instr.op == Op::Select || instr.flags & FLAG_FUSED != 0 {
            instr.c = (instr.c & !0xffff_ffff) | slot;
        } else {
            instr.c = slot;
        }
    }

    fn local_set(&mut self, idx: u32, is_tee: bool) -> Result<(), WasmError> {
        // Hazard flush: pending reads of this local captured the OLD value.
        let mut flushed = false;
        if self.stack.len() > 1 {
            for i in 0..self.stack.len() - 1 {
                if self.stack[i] == Desc::Local(idx) {
                    self.materialize_at(i);
                    flushed = true;
                }
            }
        }
        let top = self.pop()?;

        // dst-folding soundness (design doc §4.2): known producer, same
        // control region, target local neither read nor written since the
        // producer, and no hazard flush for this set.
        let fold_def = match top {
            Desc::Temp { def, region, .. }
                if def != NO_DEF
                    && region == self.region
                    && !flushed
                    && self.code[def as usize].op != Op::MovPair
                    && self.last_read[idx as usize] <= def + 1
                    && self.last_write[idx as usize] <= def + 1 =>
            {
                Some(def)
            }
            _ => None,
        };

        if let Some(def) = fold_def {
            self.patch_dst(def, idx as u64);
            self.last_write[idx as usize] = def + 1;
            self.last_write_region[idx as usize] = self.region;
        } else {
            let at = self.code.len() as u32;
            let (op, flags, a) = match top {
                Desc::Local(src) => {
                    self.last_read[src as usize] = at + 1;
                    let acc = self.acc_operand(top, FLAG_A_ACC, false);
                    (Op::MovSlot, acc, src as u64)
                }
                Desc::ConstV(k) => (Op::MovConst, FLAG_A_CONST, k),
                Desc::Temp { height, .. } => {
                    let acc = self.acc_operand(top, FLAG_A_ACC, false);
                    (Op::MovSlot, acc, self.temp_slot_used(height))
                }
            };
            self.emit(op, flags, a, 0, idx as u64);
            self.last_write[idx as usize] = at + 1;
            self.last_write_region[idx as usize] = self.region;
        }
        if is_tee {
            self.stack.push(Desc::Local(idx));
        }
        Ok(())
    }

    /// A br/br_if/br_table to a non-loop label makes that block's end a
    /// real merge target (keeps code after it reachable).
    fn mark_branch_target(&mut self, depth: u32) {
        let n = self.frames.len();
        if (depth as usize) < n {
            let f = &mut self.frames[n - 1 - depth as usize];
            if !f.is_loop {
                f.end_targeted = true;
            }
        }
    }

    /// Emit the value moves a taken branch needs: the top `arity` values
    /// move from the current heights down to the target's base heights.
    /// Caller must have materialized the stack first. Returns whether any
    /// move was needed.
    fn branch_value_moves(&mut self, depth: u32) -> Result<bool, WasmError> {
        let n = self.frames.len();
        if (depth as usize) >= n {
            // br to the function label == return; handled by the caller.
            return Ok(false);
        }
        let f = &self.frames[n - 1 - depth as usize];
        let arity = if f.is_loop { f.params } else { f.results };
        let target_base = f.base;
        let h = self.height();
        if arity == 0 || h == target_base + arity {
            return Ok(false);
        }
        if h < target_base + arity {
            return Err(desync());
        }
        for i in 0..arity {
            let src = self.temp_slot_used(h - arity + i);
            let dst = self.temp_slot_used(target_base + i);
            self.emit(Op::MovSlot, 0, src, 0, dst);
        }
        Ok(true)
    }

    /// Emit the branch instruction for label `depth`: direct to a loop
    /// header, or fixed up at the target block's end.
    fn emit_branch(&mut self, op: Op, flags: u16, a: u64, depth: u32) -> Result<(), WasmError> {
        let n = self.frames.len();
        if (depth as usize) >= n {
            // Branch to the function label: a return — conditional ops
            // guard it with their inverted sense.
            match op {
                Op::Br => self.emit_return(),
                Op::BrIf | Op::BrIfNot => {
                    let guard = if op == Op::BrIf {
                        Op::BrIfNot
                    } else {
                        Op::BrIf
                    };
                    let skip = self.emit(guard, flags, a, 0, FIXUP);
                    self.emit_return();
                    let here = self.code.len() as u64;
                    self.code[skip as usize].c = here;
                }
                _ => return Err(desync()),
            }
            return Ok(());
        }
        let i = n - 1 - depth as usize;
        if self.frames[i].is_loop {
            let header = self.frames[i].header as u64;
            self.emit(op, flags, a, 0, header);
        } else {
            let idx = self.emit(op, flags, a, 0, FIXUP);
            self.frames[i].fixups.push(Fixup::InstrC(idx));
        }
        Ok(())
    }

    /// Point an already-emitted instruction's `c` at the branch target for
    /// `depth` — the in-place fused compare-branch path. The caller must
    /// have checked `depth < frames.len()`.
    fn retarget_branch(&mut self, def: u32, depth: u32) {
        let n = self.frames.len();
        let i = n - 1 - depth as usize;
        if self.frames[i].is_loop {
            self.code[def as usize].c = self.frames[i].header as u64;
        } else {
            self.code[def as usize].c = FIXUP;
            self.frames[i].fixups.push(Fixup::InstrC(def));
        }
    }

    fn emit_return(&mut self) {
        let r = self.n_results;
        let base = self.height().saturating_sub(r);
        if r > 0 {
            self.max_height = self.max_height.max(base + r);
        }
        self.emit(Op::Return, 0, self.temp_slot(base), r as u64, 0);
    }

    fn call_boundary(
        &mut self,
        func_idx_field: Option<u64>,
        params: u32,
        results: u32,
        indirect: Option<(u64, bool, u64)>,
    ) -> Result<(), WasmError> {
        // A call is not a merge point: only the arguments need staging
        // (v1: one mov per still-pending argument; the stage_args batch
        // combo replaces this later).
        let h = self.height() as usize;
        if h < params as usize {
            return Err(desync());
        }
        for i in h - params as usize..h {
            self.materialize_at(i);
        }
        let arg_base = self.temp_slot(self.height() - params);
        match indirect {
            None => {
                let fidx = func_idx_field.unwrap_or(0);
                self.emit(Op::Call, 0, fidx, arg_base, 0);
            }
            Some((target, target_const, type_idx)) => {
                let mut flags = if target_const { FLAG_A_CONST } else { 0 };
                // The native handler bounds-checks the index with a 32-bit
                // compare, so a 2^32-aligned 64-bit index would alias entry 0
                // instead of trapping. Deny it the handler.
                if self.table_is_64(type_idx >> 32) {
                    flags |= FLAG_ADDR64;
                }
                self.emit(Op::CallIndirect, flags, target, arg_base, type_idx);
            }
        }
        for _ in 0..params {
            let _ = self.pop()?;
        }
        // Results are written by the callee into the argument overlap area:
        // they are already materialized at their canonical heights.
        self.push_unknown_temps(results);
        if results == 1 {
            self.last_call_idx = self.code.len() as u32 - 1;
            self.last_call_height = self.height() - 1;
            self.last_call_region = self.region;
        }
        Ok(())
    }

    fn fc_op(&mut self, fc: OpcodeFC, imm: &Immediate) -> Result<(), WasmError> {
        use OpcodeFC::*;
        match fc {
            I32_TRUNC_SAT_F32_S => self.value_op(Op::I32_TruncSatF32S, 1),
            I32_TRUNC_SAT_F32_U => self.value_op(Op::I32_TruncSatF32U, 1),
            I32_TRUNC_SAT_F64_S => self.value_op(Op::I32_TruncSatF64S, 1),
            I32_TRUNC_SAT_F64_U => self.value_op(Op::I32_TruncSatF64U, 1),
            I64_TRUNC_SAT_F32_S => self.value_op(Op::I64_TruncSatF32S, 1),
            I64_TRUNC_SAT_F32_U => self.value_op(Op::I64_TruncSatF32U, 1),
            I64_TRUNC_SAT_F64_S => self.value_op(Op::I64_TruncSatF64S, 1),
            I64_TRUNC_SAT_F64_U => self.value_op(Op::I64_TruncSatF64U, 1),
            MEMORY_FILL | MEMORY_COPY | MEMORY_INIT => {
                let h = self.stack.len();
                if h < 3 {
                    return Err(desync());
                }
                for i in h - 3..h {
                    self.materialize_at(i);
                }
                let base = self.temp_slot_used(self.height() - 3);
                for _ in 0..3 {
                    let _ = self.pop()?;
                }
                let (op, b) = match (fc, imm) {
                    (MEMORY_FILL, Immediate::MemoryIndex(m)) => (Op::MemoryFill, *m as u64),
                    (MEMORY_FILL, _) => (Op::MemoryFill, 0),
                    (MEMORY_COPY, Immediate::MemoryCopyArgs { dstidx, srcidx }) => {
                        (Op::MemoryCopy, ((*dstidx as u64) << 32) | *srcidx as u64)
                    }
                    (MEMORY_COPY, _) => (Op::MemoryCopy, 0),
                    (MEMORY_INIT, Immediate::MemoryInitArgs { dataidx, memidx }) => {
                        (Op::MemoryInit, ((*memidx as u64) << 32) | *dataidx as u64)
                    }
                    _ => (Op::MemoryInit, 0),
                };
                self.emit(op, 0, base, b, 0);
                Ok(())
            }
            DATA_DROP => {
                let seg = match imm {
                    Immediate::DataIndex(i) => *i as u64,
                    _ => 0,
                };
                self.emit(Op::DataDrop, 0, seg, 0, 0);
                Ok(())
            }
            ELEM_DROP => {
                let seg = match imm {
                    Immediate::ElementIndex(i) => *i as u64,
                    _ => 0,
                };
                self.emit(Op::ElemDrop, 0, seg, 0, 0);
                Ok(())
            }
            TABLE_SIZE => {
                let t = match imm {
                    Immediate::TableIndex(t) => *t as u64,
                    _ => 0,
                };
                let dst = self.temp_slot_used(self.height());
                let idx = self.emit(Op::TableSize, 0, 0, t, dst);
                self.push_result_temp(idx);
                Ok(())
            }
            TABLE_GROW => {
                let t = match imm {
                    Immediate::TableIndex(t) => *t as u64,
                    _ => 0,
                };
                let at = self.code.len() as u32;
                let delta = self.pop()?;
                let (b, b_const) = self.operand(delta, at);
                let init = self.pop()?;
                let (a, a_const) = self.operand(init, at);
                let mut flags = 0u16;
                if a_const {
                    flags |= FLAG_A_CONST;
                }
                if b_const {
                    flags |= FLAG_B_CONST;
                }
                let dst = self.temp_slot_used(self.height());
                let idx = self.emit(Op::TableGrow, flags, a, b, (t << 32) | dst);
                self.push_result_temp(idx);
                Ok(())
            }
            TABLE_FILL | TABLE_COPY | TABLE_INIT => {
                let h = self.stack.len();
                if h < 3 {
                    return Err(desync());
                }
                for i in h - 3..h {
                    self.materialize_at(i);
                }
                let base = self.temp_slot_used(self.height() - 3);
                for _ in 0..3 {
                    let _ = self.pop()?;
                }
                let (op, b) = match (fc, imm) {
                    (TABLE_FILL, Immediate::TableIndex(t)) => (Op::TableFill, *t as u64),
                    (TABLE_COPY, Immediate::TableCopyArgs { dstidx, srcidx }) => {
                        (Op::TableCopy, ((*dstidx as u64) << 32) | *srcidx as u64)
                    }
                    (TABLE_INIT, Immediate::TableInitArgs { elemidx, tableidx }) => {
                        (Op::TableInit, ((*tableidx as u64) << 32) | *elemidx as u64)
                    }
                    _ => return Err(desync()),
                };
                self.emit(op, 0, base, b, 0);
                Ok(())
            }
        }
    }

    fn patch_fixups_to_here(&mut self, fixups: &[Fixup]) {
        let here = self.code.len() as u32;
        for &f in fixups {
            match f {
                Fixup::InstrC(i) => self.code[i as usize].c = here as u64,
                Fixup::Table { tbl, entry } => {
                    self.br_tables[tbl as usize][entry as usize] = here;
                }
            }
        }
    }

    fn patch_instr_to_here(&mut self, i: u32) {
        self.code[i as usize].c = self.code.len() as u64;
    }
}

fn wasm_binop(o: Opcode) -> Option<Op> {
    use Opcode::*;
    Some(match o {
        I32_ADD => Op::I32_Add,
        I32_SUB => Op::I32_Sub,
        I32_MUL => Op::I32_Mul,
        I32_DIV_S => Op::I32_DivS,
        I32_DIV_U => Op::I32_DivU,
        I32_REM_S => Op::I32_RemS,
        I32_REM_U => Op::I32_RemU,
        I32_AND => Op::I32_And,
        I32_OR => Op::I32_Or,
        I32_XOR => Op::I32_Xor,
        I32_SHL => Op::I32_Shl,
        I32_SHR_S => Op::I32_ShrS,
        I32_SHR_U => Op::I32_ShrU,
        I32_ROTL => Op::I32_Rotl,
        I32_ROTR => Op::I32_Rotr,
        I32_EQ => Op::I32_Eq,
        I32_NE => Op::I32_Ne,
        I32_LT_S => Op::I32_LtS,
        I32_LT_U => Op::I32_LtU,
        I32_GT_S => Op::I32_GtS,
        I32_GT_U => Op::I32_GtU,
        I32_LE_S => Op::I32_LeS,
        I32_LE_U => Op::I32_LeU,
        I32_GE_S => Op::I32_GeS,
        I32_GE_U => Op::I32_GeU,
        I64_ADD => Op::I64_Add,
        I64_SUB => Op::I64_Sub,
        I64_MUL => Op::I64_Mul,
        I64_DIV_S => Op::I64_DivS,
        I64_DIV_U => Op::I64_DivU,
        I64_REM_S => Op::I64_RemS,
        I64_REM_U => Op::I64_RemU,
        I64_AND => Op::I64_And,
        I64_OR => Op::I64_Or,
        I64_XOR => Op::I64_Xor,
        I64_SHL => Op::I64_Shl,
        I64_SHR_S => Op::I64_ShrS,
        I64_SHR_U => Op::I64_ShrU,
        I64_ROTL => Op::I64_Rotl,
        I64_ROTR => Op::I64_Rotr,
        I64_EQ => Op::I64_Eq,
        I64_NE => Op::I64_Ne,
        I64_LT_S => Op::I64_LtS,
        I64_LT_U => Op::I64_LtU,
        I64_GT_S => Op::I64_GtS,
        I64_GT_U => Op::I64_GtU,
        I64_LE_S => Op::I64_LeS,
        I64_LE_U => Op::I64_LeU,
        I64_GE_S => Op::I64_GeS,
        I64_GE_U => Op::I64_GeU,
        _ => return None,
    })
}

fn wasm_unop(o: Opcode) -> Option<Op> {
    use Opcode::*;
    Some(match o {
        I32_CLZ => Op::I32_Clz,
        I32_CTZ => Op::I32_Ctz,
        I32_POPCNT => Op::I32_Popcnt,
        I32_EXTEND8_S => Op::I32_Extend8S,
        I32_EXTEND16_S => Op::I32_Extend16S,
        I32_EQZ => Op::I32_Eqz,
        I64_CLZ => Op::I64_Clz,
        I64_CTZ => Op::I64_Ctz,
        I64_POPCNT => Op::I64_Popcnt,
        I64_EXTEND8_S => Op::I64_Extend8S,
        I64_EXTEND16_S => Op::I64_Extend16S,
        I64_EXTEND32_S => Op::I64_Extend32S,
        I64_EQZ => Op::I64_Eqz,
        I32_WRAP_I64 => Op::I32_WrapI64,
        I64_EXTEND_I32_S => Op::I64_ExtendI32S,
        I64_EXTEND_I32_U => Op::I64_ExtendI32U,
        F32_ABS => Op::F32_Abs,
        F32_NEG => Op::F32_Neg,
        F32_CEIL => Op::F32_Ceil,
        F32_FLOOR => Op::F32_Floor,
        F32_TRUNC => Op::F32_Trunc,
        F32_NEAREST => Op::F32_Nearest,
        F32_SQRT => Op::F32_Sqrt,
        F64_ABS => Op::F64_Abs,
        F64_NEG => Op::F64_Neg,
        F64_CEIL => Op::F64_Ceil,
        F64_FLOOR => Op::F64_Floor,
        F64_TRUNC => Op::F64_Trunc,
        F64_NEAREST => Op::F64_Nearest,
        F64_SQRT => Op::F64_Sqrt,
        I32_TRUNC_F32_S => Op::I32_TruncF32S,
        I32_TRUNC_F32_U => Op::I32_TruncF32U,
        I32_TRUNC_F64_S => Op::I32_TruncF64S,
        I32_TRUNC_F64_U => Op::I32_TruncF64U,
        I64_TRUNC_F32_S => Op::I64_TruncF32S,
        I64_TRUNC_F32_U => Op::I64_TruncF32U,
        I64_TRUNC_F64_S => Op::I64_TruncF64S,
        I64_TRUNC_F64_U => Op::I64_TruncF64U,
        F32_CONVERT_I32_S => Op::F32_ConvertI32S,
        F32_CONVERT_I32_U => Op::F32_ConvertI32U,
        F32_CONVERT_I64_S => Op::F32_ConvertI64S,
        F32_CONVERT_I64_U => Op::F32_ConvertI64U,
        F32_DEMOTE_F64 => Op::F32_DemoteF64,
        F64_CONVERT_I32_S => Op::F64_ConvertI32S,
        F64_CONVERT_I32_U => Op::F64_ConvertI32U,
        F64_CONVERT_I64_S => Op::F64_ConvertI64S,
        F64_CONVERT_I64_U => Op::F64_ConvertI64U,
        F64_PROMOTE_F32 => Op::F64_PromoteF32,
        I32_REINTERPRET_F32 => Op::I32_ReinterpretF32,
        I64_REINTERPRET_F64 => Op::I64_ReinterpretF64,
        F32_REINTERPRET_I32 => Op::F32_ReinterpretI32,
        F64_REINTERPRET_I64 => Op::F64_ReinterpretI64,
        _ => return None,
    })
}

fn wasm_fbinop(o: Opcode) -> Option<Op> {
    use Opcode::*;
    Some(match o {
        F32_ADD => Op::F32_Add,
        F32_SUB => Op::F32_Sub,
        F32_MUL => Op::F32_Mul,
        F32_DIV => Op::F32_Div,
        F32_MIN => Op::F32_Min,
        F32_MAX => Op::F32_Max,
        F32_COPYSIGN => Op::F32_Copysign,
        F32_EQ => Op::F32_Eq,
        F32_NE => Op::F32_Ne,
        F32_LT => Op::F32_Lt,
        F32_GT => Op::F32_Gt,
        F32_LE => Op::F32_Le,
        F32_GE => Op::F32_Ge,
        F64_ADD => Op::F64_Add,
        F64_SUB => Op::F64_Sub,
        F64_MUL => Op::F64_Mul,
        F64_DIV => Op::F64_Div,
        F64_MIN => Op::F64_Min,
        F64_MAX => Op::F64_Max,
        F64_COPYSIGN => Op::F64_Copysign,
        F64_EQ => Op::F64_Eq,
        F64_NE => Op::F64_Ne,
        F64_LT => Op::F64_Lt,
        F64_GT => Op::F64_Gt,
        F64_LE => Op::F64_Le,
        F64_GE => Op::F64_Ge,
        _ => return None,
    })
}

fn wasm_load(o: Opcode) -> Option<Op> {
    use Opcode::*;
    Some(match o {
        I32_LOAD => Op::I32_Load,
        I64_LOAD => Op::I64_Load,
        F32_LOAD => Op::F32_Load,
        F64_LOAD => Op::F64_Load,
        I32_LOAD8_S => Op::I32_Load8S,
        I32_LOAD8_U => Op::I32_Load8U,
        I32_LOAD16_S => Op::I32_Load16S,
        I32_LOAD16_U => Op::I32_Load16U,
        I64_LOAD8_S => Op::I64_Load8S,
        I64_LOAD8_U => Op::I64_Load8U,
        I64_LOAD16_S => Op::I64_Load16S,
        I64_LOAD16_U => Op::I64_Load16U,
        I64_LOAD32_S => Op::I64_Load32S,
        I64_LOAD32_U => Op::I64_Load32U,
        _ => return None,
    })
}

fn wasm_store(o: Opcode) -> Option<Op> {
    use Opcode::*;
    Some(match o {
        I32_STORE => Op::I32_Store,
        I64_STORE => Op::I64_Store,
        F32_STORE => Op::F32_Store,
        F64_STORE => Op::F64_Store,
        I32_STORE8 => Op::I32_Store8,
        I32_STORE16 => Op::I32_Store16,
        I64_STORE8 => Op::I64_Store8,
        I64_STORE16 => Op::I64_Store16,
        I64_STORE32 => Op::I64_Store32,
        _ => return None,
    })
}

impl<'m> OpcodeHandler for Predecoder<'m> {
    fn on_decode_begin(&mut self) -> Result<(), WasmError> {
        Ok(())
    }

    fn on_stream<'x, 'y, 'z>(
        &mut self,
        stream: &mut OpStream<'x, 'y, 'z>,
    ) -> Result<(), WasmError> {
        while let Some(op) = stream.next()? {
            let o = match op.wasm_op {
                WasmOpcode::OP(o) => o,
                WasmOpcode::FC(fc) => {
                    if self.dead {
                        continue;
                    }
                    self.fc_op(fc, &op.imm.clone())?;
                    continue;
                }
                other => return Err(unsupported_opcode(other)),
            };
            let imm = op.imm.clone();

            // Dead-code handling: skip, but keep frame nesting and merge
            // reachability bookkeeping.
            if self.dead {
                match o {
                    Opcode::BLOCK | Opcode::LOOP | Opcode::IF => {
                        let (p, r) = match &imm {
                            Immediate::Block(bt) => block_arity(self.types, bt)?,
                            _ => (0, 0),
                        };
                        let base = self.height().saturating_sub(p);
                        self.frames.push(CtlFrame {
                            base,
                            params: p,
                            results: r,
                            is_loop: o == Opcode::LOOP,
                            is_if: o == Opcode::IF,
                            dead_entry: true,
                            end_targeted: false,
                            saw_else: false,
                            then_fell_live: false,
                            header: 0,
                            else_fixup: None,
                            fixups: Vec::new(),
                        });
                    }
                    Opcode::ELSE => {
                        if let Some(f) = self.frames.last_mut() {
                            f.saw_else = true;
                            let (dead_entry, base, params, else_fixup) =
                                (f.dead_entry, f.base, f.params, f.else_fixup.take());
                            self.dead = dead_entry;
                            self.stack.truncate(base as usize);
                            self.push_unknown_temps(params);
                            if let Some(fx) = else_fixup {
                                self.patch_instr_to_here(fx);
                            }
                        }
                    }
                    Opcode::END => {
                        if let Some(f) = self.frames.pop() {
                            let live_after = !f.dead_entry
                                && (f.end_targeted || f.then_fell_live || (f.is_if && !f.saw_else));
                            self.stack.truncate(f.base as usize);
                            self.push_unknown_temps(f.results);
                            if let Some(fx) = f.else_fixup {
                                self.patch_instr_to_here(fx);
                            }
                            self.patch_fixups_to_here(&f.fixups);
                            self.dead = !live_after;
                            if live_after {
                                self.bump_region();
                            }
                        }
                        // Function-level end while dead: the tail is
                        // genuinely unreachable; nothing to emit.
                    }
                    _ => {}
                }
                continue;
            }

            match o {
                Opcode::NOP => {}
                Opcode::UNREACHABLE => {
                    self.emit(Op::Unreachable, 0, 0, 0, 0);
                    self.dead = true;
                }
                Opcode::BLOCK | Opcode::LOOP => {
                    let (p, r) = match &imm {
                        Immediate::Block(bt) => block_arity(self.types, bt)?,
                        _ => (0, 0),
                    };
                    if o == Opcode::LOOP {
                        // Loop header is a merge point (back edges arrive).
                        self.materialize_all();
                        self.bump_region();
                    }
                    let base = self.height().checked_sub(p).ok_or_else(desync)?;
                    let header = self.code.len() as u32;
                    self.frames.push(CtlFrame {
                        base,
                        params: p,
                        results: r,
                        is_loop: o == Opcode::LOOP,
                        is_if: false,
                        dead_entry: false,
                        end_targeted: false,
                        saw_else: false,
                        then_fell_live: false,
                        header,
                        else_fixup: None,
                        fixups: Vec::new(),
                    });
                }
                Opcode::IF => {
                    let (p, r) = match &imm {
                        Immediate::Block(bt) => block_arity(self.types, bt)?,
                        _ => (0, 0),
                    };
                    let cond = self.pop()?;
                    self.materialize_all();
                    // Fixed combo: fuse the guard into a fusible condition
                    // producer (inverted sense: skip the then-branch when
                    // the compare fails).
                    let fx = if let Some((def, op)) =
                        self.rewritable_producer(cond).and_then(|def| {
                            fuse_cmp_br(self.code[def as usize].op, true).map(|op| (def, op))
                        }) {
                        self.code[def as usize].op = op;
                        self.code[def as usize].c = FIXUP;
                        def
                    } else {
                        let at = self.code.len() as u32;
                        let (a, a_const) = self.operand(cond, at);
                        let mut flags = self.acc_operand(cond, FLAG_A_ACC, false);
                        if a_const {
                            flags |= FLAG_A_CONST;
                        }
                        self.emit(Op::BrIfNot, flags, a, 0, FIXUP)
                    };
                    self.bump_region();
                    let base = self.height().checked_sub(p).ok_or_else(desync)?;
                    self.frames.push(CtlFrame {
                        base,
                        params: p,
                        results: r,
                        is_loop: false,
                        is_if: true,
                        dead_entry: false,
                        end_targeted: false,
                        saw_else: false,
                        then_fell_live: false,
                        header: 0,
                        else_fixup: Some(fx),
                        fixups: Vec::new(),
                    });
                }
                Opcode::ELSE => {
                    self.materialize_all();
                    let f = self.frames.last_mut().ok_or_else(desync)?;
                    f.saw_else = true;
                    f.then_fell_live = true;
                    let (base, params, else_fixup) = (f.base, f.params, f.else_fixup.take());
                    // Jump from the live then-arm over the else-arm.
                    let jump = self.emit(Op::Br, 0, 0, 0, FIXUP);
                    self.frames
                        .last_mut()
                        .unwrap()
                        .fixups
                        .push(Fixup::InstrC(jump));
                    if let Some(fx) = else_fixup {
                        self.patch_instr_to_here(fx);
                    }
                    self.stack.truncate(base as usize);
                    self.push_unknown_temps(params);
                    self.bump_region();
                }
                Opcode::END => {
                    self.materialize_all();
                    let f = self.frames.pop();
                    match f {
                        Some(f) => {
                            if self.height() != f.base + f.results {
                                return Err(desync());
                            }
                            if let Some(fx) = f.else_fixup {
                                self.patch_instr_to_here(fx);
                            }
                            self.patch_fixups_to_here(&f.fixups);
                            // Fell through live; code after a live end is live.
                            self.bump_region();
                        }
                        None => {
                            // Function-level end: implicit return.
                            if self.height() != self.n_results {
                                return Err(desync());
                            }
                            self.emit_return();
                        }
                    }
                }
                Opcode::BR => {
                    let d = match imm {
                        Immediate::LabelIndex(d) => d,
                        _ => return Err(desync()),
                    };
                    self.mark_branch_target(d);
                    self.materialize_all();
                    self.branch_value_moves(d)?;
                    self.emit_branch(Op::Br, 0, 0, d)?;
                    self.bump_region();
                    self.dead = true;
                }
                Opcode::BR_IF => {
                    let d = match imm {
                        Immediate::LabelIndex(d) => d,
                        _ => return Err(desync()),
                    };
                    self.mark_branch_target(d);
                    let cond = self.pop()?;
                    self.materialize_all();

                    // Simple form: the taken path needs no value moves.
                    let needs_moves = {
                        let n = self.frames.len();
                        if (d as usize) < n {
                            let f = &self.frames[n - 1 - d as usize];
                            let arity = if f.is_loop { f.params } else { f.results };
                            arity != 0 && self.height() != f.base + arity
                        } else {
                            self.n_results != 0
                        }
                    };

                    // Fixed combo: a fusible condition producer becomes the
                    // branch itself (the guard form uses the inverted
                    // sense). A branch to the function label is a return —
                    // not a branch target — so it stays unfused.
                    let fused = if !needs_moves && (d as usize) >= self.frames.len() {
                        None
                    } else {
                        self.rewritable_producer(cond).and_then(|def| {
                            fuse_cmp_br(self.code[def as usize].op, needs_moves).map(|op| (def, op))
                        })
                    };

                    if let Some((def, op)) = fused {
                        self.code[def as usize].op = op;
                        if !needs_moves {
                            self.retarget_branch(def, d);
                        } else {
                            // Guard form: the fused inverted branch skips
                            // the taken path's moves + jump.
                            self.code[def as usize].c = FIXUP;
                            self.branch_value_moves(d)?;
                            self.emit_branch(Op::Br, 0, 0, d)?;
                            let here = self.code.len() as u64;
                            self.code[def as usize].c = here;
                        }
                    } else {
                        let at = self.code.len() as u32;
                        let (a, a_const) = self.operand(cond, at);
                        let mut flags = self.acc_operand(cond, FLAG_A_ACC, false);
                        if a_const {
                            flags |= FLAG_A_CONST;
                        }
                        if !needs_moves {
                            self.emit_branch(Op::BrIf, flags, a, d)?;
                        } else {
                            // General form: guard, moves, jump.
                            let skip = self.emit(Op::BrIfNot, flags, a, 0, FIXUP);
                            self.branch_value_moves(d)?;
                            self.emit_branch(Op::Br, 0, 0, d)?;
                            let here = self.code.len() as u64;
                            self.code[skip as usize].c = here;
                        }
                    }
                    self.bump_region();
                }
                Opcode::RETURN => {
                    self.materialize_all();
                    self.emit_return();
                    self.bump_region();
                    self.dead = true;
                }
                Opcode::CALL => {
                    let fidx = match imm {
                        Immediate::FunctionIndex(i) => i,
                        _ => return Err(desync()),
                    };
                    let ft = self
                        .module
                        .functions()
                        .get(fidx as usize)
                        .map(|f| f.func_type_rc())
                        .ok_or(WasmError::invalid("interp: bad call target"))?;
                    self.call_boundary(
                        Some(fidx as u64),
                        ft.params().len() as u32,
                        ft.results().len() as u32,
                        None,
                    )?;
                }
                Opcode::CALL_INDIRECT => {
                    let (tidx, table) = match imm {
                        Immediate::CallIndirectArgs { typeidx, tableidx } => (typeidx, tableidx),
                        _ => return Err(desync()),
                    };
                    let ft = self
                        .types
                        .get_function_type(tidx)
                        .ok_or(WasmError::invalid("interp: bad call_indirect type"))?
                        .clone();
                    let at = self.code.len() as u32;
                    let target = self.pop()?;
                    let (t, t_const) = self.operand(target, at);
                    self.call_boundary(
                        None,
                        ft.params().len() as u32,
                        ft.results().len() as u32,
                        Some((t, t_const, ((table as u64) << 32) | tidx as u64)),
                    )?;
                }
                Opcode::DROP => {
                    // A dropped temp's slot was still written by its producer.
                    if let Desc::Temp { height, def, .. } = self.pop()? {
                        if def != NO_DEF {
                            let _ = self.temp_slot_used(height);
                        }
                    }
                }
                Opcode::SELECT | Opcode::SELECT_T => {
                    let at = self.code.len() as u32;
                    // Condition is always materialized (Select packs its
                    // slot into c alongside the destination).
                    let top = self.stack.len().checked_sub(1).ok_or_else(desync)?;
                    self.materialize_at(top);
                    let cond = match self.pop()? {
                        Desc::Temp { height, .. } => self.temp_slot_used(height),
                        _ => return Err(desync()),
                    };
                    let v2 = self.pop()?;
                    let (b, b_const) = self.operand(v2, at);
                    let mut flags = self.acc_operand(v2, FLAG_B_ACC, false);
                    let v1 = self.pop()?;
                    let (a, a_const) = self.operand(v1, at);
                    flags |= self.acc_operand(v1, FLAG_A_ACC, false);
                    if a_const {
                        flags |= FLAG_A_CONST;
                    }
                    if b_const {
                        flags |= FLAG_B_CONST;
                    }
                    let dst = self.temp_slot_used(self.height());
                    let idx = self.emit(Op::Select, flags, a, b, (cond << 32) | dst);
                    self.push_result_temp(idx);
                }
                Opcode::LOCAL_GET => {
                    if let Immediate::LocalIndex(i) = imm {
                        self.stack.push(Desc::Local(i));
                    }
                }
                Opcode::LOCAL_SET => {
                    if let Immediate::LocalIndex(i) = imm {
                        self.local_set(i, false)?;
                    }
                }
                Opcode::LOCAL_TEE => {
                    if let Immediate::LocalIndex(i) = imm {
                        self.local_set(i, true)?;
                    }
                }
                Opcode::I32_CONST => {
                    if let Immediate::I32(v) = imm {
                        self.stack.push(Desc::ConstV(v as u32 as u64));
                    }
                }
                Opcode::I64_CONST => {
                    if let Immediate::I64(v) = imm {
                        self.stack.push(Desc::ConstV(v as u64));
                    }
                }
                Opcode::REF_NULL => {
                    self.stack.push(Desc::ConstV(NULL_FUNCREF));
                }
                Opcode::REF_FUNC => {
                    if let Immediate::FunctionIndex(i) = imm {
                        self.stack.push(Desc::ConstV(i as u64));
                    }
                }
                Opcode::REF_IS_NULL => {
                    self.value_op(Op::RefIsNull, 1)?;
                }
                Opcode::TABLE_GET => {
                    let t = match imm {
                        Immediate::TableIndex(t) => t,
                        _ => return Err(desync()),
                    };
                    let at = self.code.len() as u32;
                    let d = self.pop()?;
                    let (a, a_const) = self.operand(d, at);
                    let flags = if a_const { FLAG_A_CONST } else { 0 };
                    let dst = self.temp_slot_used(self.height());
                    let idx = self.emit(Op::TableGet, flags, a, t as u64, dst);
                    self.push_result_temp(idx);
                }
                Opcode::TABLE_SET => {
                    let t = match imm {
                        Immediate::TableIndex(t) => t,
                        _ => return Err(desync()),
                    };
                    let at = self.code.len() as u32;
                    let v = self.pop()?;
                    let (b, b_const) = self.operand(v, at);
                    let i = self.pop()?;
                    let (a, a_const) = self.operand(i, at);
                    let mut flags = 0u16;
                    if a_const {
                        flags |= FLAG_A_CONST;
                    }
                    if b_const {
                        flags |= FLAG_B_CONST;
                    }
                    self.emit(Op::TableSet, flags, a, b, t as u64);
                }
                Opcode::F32_CONST => {
                    if let Immediate::F32(v) = imm {
                        self.stack.push(Desc::ConstV(v.to_bits() as u64));
                    }
                }
                Opcode::F64_CONST => {
                    if let Immediate::F64(v) = imm {
                        self.stack.push(Desc::ConstV(v.to_bits()));
                    }
                }
                Opcode::GLOBAL_GET => {
                    let g = match imm {
                        Immediate::GlobalIndex(g) => g,
                        _ => return Err(desync()),
                    };
                    let dst = self.temp_slot_used(self.height());
                    let idx = self.emit(Op::GlobalGet, 0, g as u64, 0, dst);
                    self.push_result_temp(idx);
                }
                Opcode::GLOBAL_SET => {
                    let g = match imm {
                        Immediate::GlobalIndex(g) => g,
                        _ => return Err(desync()),
                    };
                    let at = self.code.len() as u32;
                    let d = self.pop()?;
                    let (a, a_const) = self.operand(d, at);
                    let mut flags = self.acc_operand(d, FLAG_A_ACC, false);
                    if a_const {
                        flags |= FLAG_A_CONST;
                    }
                    self.emit(Op::GlobalSet, flags, a, 0, g as u64);
                }
                Opcode::MEMORY_SIZE => {
                    let m = match imm {
                        Immediate::MemoryIndex(m) => m as u64,
                        _ => 0,
                    };
                    let dst = self.temp_slot_used(self.height());
                    let idx = self.emit(Op::MemorySize, 0, 0, m, dst);
                    self.push_result_temp(idx);
                }
                Opcode::MEMORY_GROW => {
                    let at = self.code.len() as u32;
                    let d = self.pop()?;
                    let (a, a_const) = self.operand(d, at);
                    let flags = if a_const { FLAG_A_CONST } else { 0 };
                    let m = match imm {
                        Immediate::MemoryIndex(m) => m as u64,
                        _ => 0,
                    };
                    let dst = self.temp_slot_used(self.height());
                    let idx = self.emit(Op::MemoryGrow, flags, a, m, dst);
                    self.push_result_temp(idx);
                }
                Opcode::BR_TABLE => {
                    let (labels, default) = match &imm {
                        Immediate::BrLabels(labels, default) => (labels.clone(), *default),
                        _ => return Err(desync()),
                    };
                    for &d in labels.iter() {
                        self.mark_branch_target(d);
                    }
                    self.mark_branch_target(default);
                    let at = self.code.len() as u32;
                    let d0 = self.pop()?;
                    let (a, a_const) = self.operand(d0, at);
                    let mut flags = if a_const { FLAG_A_CONST } else { 0 };
                    self.materialize_all();
                    // Acc marking AFTER materialization: the adjacency
                    // guard re-evaluates against the post-materialization
                    // length, so the mark only lands when no mov sits
                    // between the producer and the BrTable cell (movs
                    // clobber the accumulator — every value handler
                    // computes into it).
                    flags |= self.acc_operand(d0, FLAG_A_ACC, false);
                    // v1 restriction: every target must need no value moves
                    // (LLVM switch tables are arity-0). Reject otherwise.
                    for &d in labels.iter().chain(core::iter::once(&default)) {
                        let n = self.frames.len();
                        if (d as usize) >= n {
                            if self.n_results != 0 {
                                return Err(WasmError::invalid(
                                    "interp: branch to a function-level label with results",
                                ));
                            }
                            continue;
                        }
                        let f = &self.frames[n - 1 - d as usize];
                        let arity = if f.is_loop { f.params } else { f.results };
                        if arity != 0 && self.height() != f.base + arity {
                            return Err(WasmError::invalid(
                                "interp: branch with operands above the target's arity",
                            ));
                        }
                    }
                    let tbl = self.br_tables.len() as u32;
                    let mut entries = Vec::new();
                    for &d in labels.iter().chain(core::iter::once(&default)) {
                        let n = self.frames.len();
                        if (d as usize) >= n {
                            // function label: lower as a jump to a shared
                            // return we emit right after the table dispatch
                            entries.push(u32::MAX);
                            continue;
                        }
                        let i = n - 1 - d as usize;
                        if self.frames[i].is_loop {
                            entries.push(self.frames[i].header);
                        } else {
                            let entry = entries.len() as u32;
                            self.frames[i].fixups.push(Fixup::Table { tbl, entry });
                            entries.push(u32::MAX);
                        }
                    }
                    self.emit(Op::BrTable, flags, a, 0, tbl as u64);
                    // Shared return landing pad for function-label entries.
                    if entries.iter().any(|&e| e == u32::MAX) {
                        let here = self.code.len() as u32;
                        let mut needs_ret = false;
                        for (i, e) in entries.iter_mut().enumerate() {
                            // Only unresolved FUNCTION-label entries point at
                            // the pad; block fixups will overwrite theirs.
                            let d = labels.get(i).copied().unwrap_or(default);
                            if *e == u32::MAX && (d as usize) >= self.frames.len() {
                                *e = here;
                                needs_ret = true;
                            }
                        }
                        if needs_ret {
                            self.emit_return();
                        }
                    }
                    self.br_tables.push(entries);
                    self.bump_region();
                    self.dead = true;
                }
                o => {
                    if let Some(vop) = wasm_binop(o).or_else(|| wasm_fbinop(o)) {
                        self.value_op(vop, 2)?;
                    } else if let Some(vop) = wasm_unop(o) {
                        self.value_op(vop, 1)?;
                    } else if let Some(lop) = wasm_load(o) {
                        let offset = match &imm {
                            Immediate::MemArg { offset, memidx, .. } => {
                                if *offset >= 1u64 << 48 {
                                    return Err(WasmError::invalid(
                                        "interp: static memory offset does not fit the packed form",
                                    ));
                                }
                                ((*memidx as u64) << 48) | *offset
                            }
                            _ => return Err(desync()),
                        };
                        let addr64 = self.memory_is_64(offset >> 48);
                        let at = self.code.len() as u32;
                        let d = self.pop()?;
                        let (a, a_const) = self.operand(d, at);
                        // Address-add fusion: a single-use, just-emitted
                        // i32.add producing this address folds into the
                        // load (the corpus-universal base+index pattern).
                        let mut fused = false;
                        if let Some(def) = self.rewritable_producer(d) {
                            let add = self.code[def as usize];
                            let dst = self.temp_slot_used(self.height());
                            if add.op == Op::I32_Add
                                && !addr64
                                && add.flags & (FLAG_A_CONST | FLAG_B_CONST) == 0
                                && offset >> 48 == 0
                                && add.a < 1 << 16
                                && add.b < 1 << 16
                                && dst < 1 << 16
                            {
                                let (a1, a2, afl) = if add.flags & FLAG_B_ACC != 0 {
                                    (add.b, add.a, FLAG_A_ACC)
                                } else {
                                    (add.a, add.b, add.flags & FLAG_A_ACC)
                                };
                                self.code[def as usize] = Instr {
                                    op: lop,
                                    flags: afl | FLAG_FUSED,
                                    a: a1,
                                    b: offset,
                                    c: a2 << 32 | dst,
                                };
                                self.push_result_temp(def);
                                fused = true;
                            }
                        }
                        if !fused {
                            let mut flags = self.acc_operand(d, FLAG_A_ACC, false);
                            if a_const {
                                flags |= FLAG_A_CONST;
                            }
                            if addr64 {
                                flags |= FLAG_ADDR64;
                            }
                            let dst = self.temp_slot_used(self.height());
                            let idx = self.emit(lop, flags, a, offset, dst);
                            self.push_result_temp(idx);
                        }
                    } else if let Some(sop) = wasm_store(o) {
                        let offset = match &imm {
                            Immediate::MemArg { offset, memidx, .. } => {
                                if *offset >= 1u64 << 48 {
                                    return Err(WasmError::invalid(
                                        "interp: static memory offset does not fit the packed form",
                                    ));
                                }
                                ((*memidx as u64) << 48) | *offset
                            }
                            _ => return Err(desync()),
                        };
                        let addr64 = self.memory_is_64(offset >> 48);
                        let at = self.code.len() as u32;
                        let v = self.pop()?;
                        let (b, b_const) = self.operand(v, at);
                        let mut flags =
                            self.acc_operand(v, FLAG_B_ACC, operand_is_float(sop, true));
                        if b_const {
                            flags |= FLAG_B_CONST;
                        }
                        let ad = self.pop()?;
                        let (a, a_const) = self.operand(ad, at);
                        // Address-add fusion (see the load arm). When it
                        // fires the value was necessarily folded (the add
                        // is the last instruction), so the value's flags
                        // carry no adjacency marks.
                        let mut fused = false;
                        if let Some(def) = self.rewritable_producer(ad) {
                            let add = self.code[def as usize];
                            if add.op == Op::I32_Add
                                && !addr64
                                && add.flags & (FLAG_A_CONST | FLAG_B_CONST) == 0
                                && offset >> 32 == 0
                                && add.a < 1 << 16
                                && add.b < 1 << 16
                            {
                                let (a1, a2, afl) = if add.flags & FLAG_B_ACC != 0 {
                                    (add.b, add.a, FLAG_A_ACC)
                                } else {
                                    (add.a, add.b, add.flags & FLAG_A_ACC)
                                };
                                self.code[def as usize] = Instr {
                                    op: sop,
                                    flags: flags | afl | FLAG_FUSED,
                                    a: a1,
                                    b,
                                    c: a2 << 32 | offset,
                                };
                                fused = true;
                            }
                        }
                        if !fused {
                            flags |= self.acc_operand(ad, FLAG_A_ACC, false);
                            if a_const {
                                flags |= FLAG_A_CONST;
                            }
                            if addr64 {
                                flags |= FLAG_ADDR64;
                            }
                            self.emit(sop, flags, a, b, offset);
                        }
                    } else {
                        return Err(unsupported());
                    }
                }
            }
        }
        Ok(())
    }

    fn on_decode_end(&mut self) -> Result<(), WasmError> {
        if !self.frames.is_empty() {
            return Err(desync());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::Module;
    use std::vec::Vec as StdVec;

    fn predecode_wat(src: &str, func: usize) -> PredecodedFunction {
        let bin: StdVec<u8> = wat::parse_str(src).expect("wat");
        let module = Module::new("t", &bin).expect("module");
        predecode_function(&module, func).expect("predecode")
    }

    fn ops(f: &PredecodedFunction) -> StdVec<Op> {
        f.code.iter().map(|i| i.op).collect()
    }

    #[test]
    fn folds_get_get_add_set_to_one_instruction() {
        let f = predecode_wat(
            r#"(module (func (local i32 i32 i32)
                local.get 0
                local.get 1
                i32.add
                local.set 2))"#,
            0,
        );
        assert_eq!(ops(&f), [Op::I32_Add, Op::Return]);
        let add = &f.code[0];
        assert_eq!(add.flags, 0);
        assert_eq!((add.a, add.b, add.c), (0, 1, 2)); // dst-folded to local 2
                                                      // one conservative slot for the (folded-away) producer dst
        assert_eq!(f.frame_slots, 4);
    }

    #[test]
    fn folds_const_operand_inline() {
        let f = predecode_wat(
            r#"(module (func (param i32) (result i32)
                local.get 0
                i32.const 41
                i32.add))"#,
            0,
        );
        assert_eq!(ops(&f), [Op::I32_Add, Op::Return]);
        let add = &f.code[0];
        assert_eq!(add.flags, FLAG_B_CONST);
        assert_eq!((add.a, add.b), (0, 41));
        assert_eq!(add.c, 1); // result at temp slot (n_locals=1, height 0)
        assert_eq!(f.code[1].a, 1); // return reads the temp slot
        assert_eq!(f.code[1].b, 1); // one result
    }

    #[test]
    fn hazard_flush_blocks_dst_folding() {
        // Stack holds a pending read of local 0 when local 0 is overwritten:
        // the old value must be materialized first, and the set must not
        // retro-patch the add (the flush mov executes after it).
        let f = predecode_wat(
            r#"(module (func (param i32) (result i32)
                local.get 0
                local.get 0
                i32.const 1
                i32.add
                local.set 0
                ))"#,
            0,
        );
        // The flush mov is emitted at set-processing time, i.e. AFTER the
        // add — which is still correct: the fold is blocked, so the add
        // writes a temp and local 0 keeps its old value until the set mov.
        assert_eq!(ops(&f), [Op::I32_Add, Op::MovSlot, Op::MovSlot, Op::Return]);
        // add writes a temp, NOT local 0
        assert_ne!(f.code[0].c, 0);
        // flush: old local 0 -> its canonical temp slot 1 (n_locals = 1)
        assert_eq!((f.code[1].a, f.code[1].c), (0, 1));
        // unfolded set copies the add result into local 0
        assert_eq!(f.code[2].c, 0);
        // the returned value is the flushed OLD param value
        assert_eq!((f.code[3].a, f.code[3].b), (1, 1));
    }

    #[test]
    fn loop_back_edge_targets_header() {
        let f = predecode_wat(
            r#"(module (func (local i32)
                (loop $l
                    local.get 0
                    i32.const 1
                    i32.add
                    local.set 0
                    local.get 0
                    i32.const 10
                    i32.lt_u
                    br_if $l)))"#,
            0,
        );
        // add (dst-folded to local 0), fused compare-branch, return
        assert_eq!(ops(&f), [Op::I32_Add, Op::I32_BrLtU, Op::Return]);
        assert_eq!(f.code[0].c, 0); // induction update folded into the add
        assert_eq!(f.code[1].c, 0); // fused back edge targets the loop header
    }

    #[test]
    fn if_else_produces_guard_and_join() {
        let f = predecode_wat(
            r#"(module (func (param i32) (result i32)
                local.get 0
                (if (result i32) (then i32.const 1) (else i32.const 2))))"#,
            0,
        );
        assert_eq!(
            ops(&f),
            [Op::BrIfNot, Op::MovConst, Op::Br, Op::MovConst, Op::Return]
        );
        assert_eq!(f.code[0].c, 3); // guard jumps to the else arm
        assert_eq!(f.code[2].c, 4); // then-arm jump lands after the else arm
                                    // both arms materialize their constant into the same join slot
        assert_eq!(f.code[1].c, f.code[3].c);
    }

    #[test]
    fn rejects_unsupported_opcodes_cleanly() {
        // Tail calls are outside the current op set.
        let bin: StdVec<u8> =
            wat::parse_str(r#"(module (func $f) (func return_call $f))"#).expect("wat");
        let module = Module::new("t", &bin).expect("module");
        assert!(predecode_function(&module, 1).is_err());
    }
}
