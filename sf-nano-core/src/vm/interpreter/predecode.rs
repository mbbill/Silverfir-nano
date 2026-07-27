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
use crate::module::entities::GlobalDef;
use crate::module::type_context::TypeContext;
use crate::module::Module;
use crate::op_decoder::{BlockType, Decoder, Immediate, OpStream, OpcodeHandler};
use crate::op_decoder::{CatchClause, CatchClauseKind};
use crate::opcodes::{Opcode, OpcodeFC, WasmOpcode};
use crate::utils::limits::Limitable;
use crate::vm::tag::TagHandle;

/// Marks a packed memarg field as a `wide_memargs` index rather than an
/// inline `memidx << 48 | offset`. Bit 63 is free in the inline form, whose
/// index occupies bits 48..63.
pub(super) const WIDE_MEMARG: u64 = 1 << 63;

use super::instr::{
    operand_is_float, result_is_float, Instr, Op, FLAG_ADDR64, FLAG_A_ACC, FLAG_A_CONST,
    FLAG_B_ACC, FLAG_B_CONST, FLAG_DST_ACC, FLAG_FUSED, FLAG_SHARED_GLOBAL, FLAG_SHARED_TABLE,
};

/// Producer index meaning "no patchable producer" (call results, block
/// results arriving over a merge).
const NO_DEF: u32 = u32::MAX;
/// Branch-target placeholder while the target's `end` is still ahead.
const FIXUP: u64 = u64::MAX;
/// Internal placeholder for a handler targeting the implicit function label.
/// Finalization replaces it with a dedicated `Return` landing cell.
const EH_FUNCTION_TARGET_FIXUP: u32 = u32::MAX;
/// Internal placeholder for a handler whose enclosing block has not ended.
const EH_TARGET_FIXUP: u32 = u32::MAX - 1;
/// Null funcref representation (function indices are table/ref values).
pub(super) const NULL_FUNCREF: u64 = u64::MAX;

/// One resolved `try_table` clause at a potentially-throwing instruction.
///
/// `tag = None` is a `catch_all[_ref]`. Typed catches carry the runtime tag
/// identity rather than their module-local tag index, so imported aliases
/// compare correctly. A `_ref` clause receives the exception reference after
/// any typed payload fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExceptionHandler {
    pub(crate) tag: Option<TagHandle>,
    pub(crate) payload_arity: u32,
    pub(crate) forwards_exn: bool,
    pub(crate) target: u32,
    pub(crate) target_base: u32,
}

/// A range in `PredecodedFunction::exception_handlers` for one exact cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExceptionSite {
    pub(crate) pc: u32,
    handlers_start: u32,
    handlers_len: u32,
}

pub(crate) struct PredecodedFunction {
    pub code: Vec<Instr>,
    /// Side tables for `BrTable`: resolved instruction indices, the last
    /// entry is the default target.
    pub br_tables: Vec<Vec<u32>>,
    /// `(memory index, static offset)` for accesses whose offset does not fit
    /// the packed `memidx << 48 | offset` form. Memory64 permits a full
    /// 64-bit offset, which leaves no room to pack an index beside it, so
    /// those cells carry an index into this table instead. Only `FLAG_ADDR64`
    /// cells use it, and those never reach a native handler.
    pub wide_memargs: Vec<(u32, u64)>,
    /// Total frame slots: params + locals + max temp height.
    pub frame_slots: u32,
    /// params + locals (== the base slot index of temps).
    pub n_locals: u32,
    pub n_params: u32,
    pub n_results: u32,
    /// A synthetic `Return` cell used when a slow-path tail callee (for
    /// example an imported host function) has produced this function's
    /// results. Re-entering the native chain here lets the ordinary return
    /// records route through native-only callers instead of making Rust
    /// guess at their layout.
    pub(crate) slow_tail_return: Option<u32>,
    /// Sorted by exact instruction index. Only instructions executing under
    /// one or more active `try_table` clauses have an entry.
    exception_sites: Vec<ExceptionSite>,
    /// Flattened handler chains referenced by `exception_sites`.
    exception_handlers: Vec<ExceptionHandler>,
}

impl PredecodedFunction {
    /// The active handler chain at `pc`, innermost try first and in source
    /// clause order within each try.
    pub(crate) fn exception_handlers_at(&self, pc: u32) -> &[ExceptionHandler] {
        let Ok(i) = self
            .exception_sites
            .binary_search_by_key(&pc, |site| site.pc)
        else {
            return &[];
        };
        let site = self.exception_sites[i];
        let start = site.handlers_start as usize;
        &self.exception_handlers[start..start + site.handlers_len as usize]
    }

    /// Whether `pc` must remain a slow-path boundary so an escaping exception
    /// can be matched in this activation.
    pub(crate) fn has_exception_handlers_at(&self, pc: u32) -> bool {
        self.exception_sites
            .binary_search_by_key(&pc, |site| site.pc)
            .is_ok()
    }
}

/// Predecode one local (non-import) function of a parsed module.
pub(crate) fn predecode_function(
    module: &Module,
    tag_handles: &[TagHandle],
    func_index: usize,
) -> Result<PredecodedFunction, WasmError> {
    if tag_handles.len() != module.tags().len() {
        return Err(WasmError::invalid(
            "interp: runtime tag table does not match module",
        ));
    }
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
        tag_handles,
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
        wide_memargs: Vec::new(),
        exception_sites: Vec::new(),
        exception_handlers: Vec::new(),
        needs_slow_tail_return: false,
        slow_tail_return: None,
        max_height: 0,
        func_index: func_index as u32,
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
        wide_memargs: p.wide_memargs,
        n_locals,
        n_params,
        n_results,
        slow_tail_return: p.slow_tail_return,
        exception_sites: p.exception_sites,
        exception_handlers: p.exception_handlers,
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
    /// Patch `exception_handlers[handler].target`.
    ExceptionTarget(u32),
}

#[derive(Clone, Copy)]
enum PendingExceptionTarget {
    Function,
    Instruction(u32),
    /// Index of an outer non-loop control frame whose end is still ahead.
    FrameEnd(u32),
}

/// A `try_table` clause resolved while its labels still refer directly to the
/// outer control stack. Copies are materialized into exact-PC handler chains
/// only at instructions that can throw.
#[derive(Clone, Copy)]
struct ActiveExceptionHandler {
    tag: Option<TagHandle>,
    payload_arity: u32,
    forwards_exn: bool,
    target: PendingExceptionTarget,
    target_base: u32,
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
    /// `try_table` catch clauses, innermost frame first when searching.
    /// Empty for every other block kind.
    catches: Vec<ActiveExceptionHandler>,
}

struct Predecoder<'m> {
    types: &'m TypeContext,
    module: &'m Module,
    /// Runtime identities resolved by the linker. Module tag indices are
    /// aliases for these handles, not identities themselves: two imports may
    /// name one tag, while two same-signature tags remain distinct.
    tag_handles: &'m [TagHandle],
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
    wide_memargs: Vec<(u32, u64)>,
    exception_sites: Vec<ExceptionSite>,
    exception_handlers: Vec<ExceptionHandler>,
    needs_slow_tail_return: bool,
    slow_tail_return: Option<u32>,
    max_height: u32,
    /// Function being decoded. A direct tail call back to this function can
    /// be lowered to parameter moves plus a branch, keeping the whole loop
    /// inside the native dispatch chain.
    func_index: u32,
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

    /// Whether the table at `idx` is reachable from another instance, and so
    /// is held as the shared entity rather than a private array.
    /// Whether the global at `idx` is reachable from another instance, and so
    /// lives in a shared cell rather than this instance's array.
    fn global_is_shared(&self, idx: u64) -> bool {
        self.module.globals().get(idx as usize).is_some_and(|g| {
            matches!(g.def(), GlobalDef::Import { .. }) || !g.export_names().is_empty()
        })
    }

    /// Resolve catch labels before pushing the `try_table` frame. At this
    /// point each label index names an outer control frame directly; retaining
    /// that resolved target avoids depth arithmetic after nested blocks have
    /// been entered.
    fn resolve_try_catches(
        &self,
        catches: &[CatchClause],
    ) -> Result<Vec<ActiveExceptionHandler>, WasmError> {
        let mut resolved = Vec::with_capacity(catches.len());
        for clause in catches {
            let (tag, payload_arity) = match clause.kind {
                CatchClauseKind::Catch | CatchClauseKind::CatchRef => {
                    let tag_idx = clause
                        .tag_idx
                        .ok_or(WasmError::invalid("interp: typed catch has no tag"))?;
                    let tag = self
                        .tag_handles
                        .get(tag_idx as usize)
                        .copied()
                        .ok_or(WasmError::invalid("interp: bad catch tag"))?;
                    let payload_arity = self
                        .module
                        .tags()
                        .get(tag_idx as usize)
                        .map(|t| t.func_type().params().len() as u32)
                        .ok_or(WasmError::invalid("interp: bad catch tag"))?;
                    (Some(tag), payload_arity)
                }
                CatchClauseKind::CatchAll | CatchClauseKind::CatchAllRef => (None, 0),
            };
            let forwards_exn = matches!(
                clause.kind,
                CatchClauseKind::CatchRef | CatchClauseKind::CatchAllRef
            );

            let depth = clause.label_idx as usize;
            let (target, target_base) = if depth < self.frames.len() {
                let frame_idx = self.frames.len() - 1 - depth;
                let frame = &self.frames[frame_idx];
                let target_base = self
                    .n_locals
                    .checked_add(frame.base)
                    .ok_or(WasmError::invalid("interp: exception target slot overflow"))?;
                let target = if frame.is_loop {
                    PendingExceptionTarget::Instruction(frame.header)
                } else {
                    PendingExceptionTarget::FrameEnd(frame_idx as u32)
                };
                (target, target_base)
            } else if depth == self.frames.len() {
                // The implicit function label has logical result base zero.
                // Finalization gives it a dedicated Return landing cell.
                (PendingExceptionTarget::Function, 0)
            } else {
                return Err(WasmError::invalid(
                    "interp: exception target label out of range",
                ));
            };

            resolved.push(ActiveExceptionHandler {
                tag,
                payload_arity,
                forwards_exn,
                target,
                target_base,
            });
        }
        Ok(resolved)
    }

    /// Snapshot all active catch clauses at one exact instruction cell.
    /// Frames are walked in reverse so runtime matching naturally implements
    /// innermost-try precedence, while each frame's source order is retained.
    fn record_exception_site(&mut self, pc: u32) {
        let active: Vec<ActiveExceptionHandler> = self
            .frames
            .iter()
            .rev()
            .flat_map(|frame| frame.catches.iter().copied())
            .collect();
        if active.is_empty() {
            return;
        }

        debug_assert!(self.exception_sites.last().is_none_or(|site| site.pc < pc));
        let handlers_start = self.exception_handlers.len() as u32;
        for handler in active {
            let target = match handler.target {
                PendingExceptionTarget::Function => EH_FUNCTION_TARGET_FIXUP,
                PendingExceptionTarget::Instruction(pc) => pc,
                PendingExceptionTarget::FrameEnd(frame_idx) => {
                    let handler_idx = self.exception_handlers.len() as u32;
                    self.frames[frame_idx as usize]
                        .fixups
                        .push(Fixup::ExceptionTarget(handler_idx));
                    EH_TARGET_FIXUP
                }
            };
            self.exception_handlers.push(ExceptionHandler {
                tag: handler.tag,
                payload_arity: handler.payload_arity,
                forwards_exn: handler.forwards_exn,
                target,
                target_base: handler.target_base,
            });
        }
        self.exception_sites.push(ExceptionSite {
            pc,
            handlers_start,
            handlers_len: self.exception_handlers.len() as u32 - handlers_start,
        });
    }

    fn has_active_exception_handlers(&self) -> bool {
        self.frames.iter().any(|frame| !frame.catches.is_empty())
    }

    /// Give function-label catches and slow-path tail callees one ordinary
    /// native `Return` target. In both cases results already occupy the
    /// function label's canonical base (zero); executing the cell is what
    /// consumes the real native return record and preserves native callers.
    fn finish_return_landing(&mut self) {
        let has_function_catch = self
            .exception_handlers
            .iter()
            .any(|handler| handler.target == EH_FUNCTION_TARGET_FIXUP);
        if !has_function_catch && !self.needs_slow_tail_return {
            return;
        }
        // Function-label results are installed from slot zero, deliberately
        // overlapping params/locals. A `_ref` catch can produce more values
        // than the throwing instruction itself materialized, so reserve any
        // portion that extends beyond the local area explicitly.
        self.max_height = self
            .max_height
            .max(self.n_results.saturating_sub(self.n_locals));
        let landing = self.emit(Op::Return, 0, 0, self.n_results as u64, 0);
        if self.needs_slow_tail_return {
            self.slow_tail_return = Some(landing);
        }
        for handler in &mut self.exception_handlers {
            if handler.target == EH_FUNCTION_TARGET_FIXUP {
                handler.target = landing;
            }
        }
    }

    fn table_is_shared(&self, idx: u64) -> bool {
        self.module
            .tables()
            .get(idx as usize)
            .is_some_and(|t| t.is_import() || !t.export_names().is_empty())
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

    /// Reserve the overlap area where a slow tail-call boundary stages its
    /// results before moving them to the current function's result slots.
    ///
    /// Ordinary calls account for this space when their result descriptors
    /// are pushed. A tail call has no continuation and therefore pushes
    /// nothing, but imported and cross-instance callees still write results
    /// at `arg_base` first. Keep that runtime write inside the predecoded
    /// frame even when locals occupy every otherwise-reserved slot.
    fn reserve_tail_results(&mut self, params: u32, results: u32) -> Result<(), WasmError> {
        let base = self.height().checked_sub(params).ok_or_else(desync)?;
        let end = base
            .checked_add(results)
            .ok_or(WasmError::invalid("interp: tail-call result area overflow"))?;
        self.max_height = self.max_height.max(end);
        Ok(())
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

    /// A `call_ref`: same staging as any call, but the callee comes from a
    /// reference operand, so there is no table index and no runtime type
    /// check -- the reference already names the function.
    fn call_boundary_ref(
        &mut self,
        params: u32,
        results: u32,
        type_idx: u32,
        target: u64,
        target_const: bool,
        tail: bool,
    ) -> Result<(), WasmError> {
        let h = self.height() as usize;
        if h < params as usize {
            return Err(desync());
        }
        if tail {
            self.reserve_tail_results(params, results)?;
        }
        if !tail && self.has_active_exception_handlers() {
            // An exceptional edge is a control-flow merge just like a
            // branch: values below the call arguments that survive at the
            // catch target must already occupy their canonical slots.
            self.materialize_all();
        } else {
            for i in h - params as usize..h {
                self.materialize_at(i);
            }
        }
        let arg_base = self.temp_slot(self.height() - params);
        let flags = if target_const { FLAG_A_CONST } else { 0 };
        let op = if tail { Op::ReturnCallRef } else { Op::CallRef };
        let call_pc = self.emit(op, flags, target, arg_base, type_idx as u64);
        if !tail {
            self.record_exception_site(call_pc);
        }
        for _ in 0..params {
            let _ = self.pop()?;
        }
        if tail {
            self.needs_slow_tail_return = true;
            self.bump_region();
            self.dead = true;
            return Ok(());
        }
        self.push_unknown_temps(results);
        if results == 1 {
            self.last_call_idx = self.code.len() as u32 - 1;
            self.last_call_height = self.height() - 1;
            self.last_call_region = self.region;
        }
        Ok(())
    }

    fn call_boundary_tail(
        &mut self,
        func_idx_field: Option<u64>,
        params: u32,
        results: u32,
        indirect: Option<(u64, bool, u64)>,
        tail: bool,
    ) -> Result<(), WasmError> {
        // A call is not a merge point: only the arguments need staging
        // (v1: one mov per still-pending argument; the stage_args batch
        // combo replaces this later).
        let h = self.height() as usize;
        if h < params as usize {
            return Err(desync());
        }
        if tail {
            self.reserve_tail_results(params, results)?;
        }
        if !tail && self.has_active_exception_handlers() {
            self.materialize_all();
        } else {
            for i in h - params as usize..h {
                self.materialize_at(i);
            }
        }
        let arg_base = self.temp_slot(self.height() - params);
        let call_pc = match indirect {
            None => {
                let fidx = func_idx_field.unwrap_or(0);
                let op = if tail { Op::ReturnCall } else { Op::Call };
                self.emit(op, 0, fidx, arg_base, 0)
            }
            Some((target, target_const, type_idx)) => {
                let mut flags = if target_const { FLAG_A_CONST } else { 0 };
                // The native handler bounds-checks the index with a 32-bit
                // compare, so a 2^32-aligned 64-bit index would alias entry 0
                // instead of trapping. Deny it the handler.
                if self.table_is_64(type_idx >> 32) {
                    flags |= FLAG_ADDR64;
                }
                if self.table_is_shared(type_idx >> 32) {
                    flags |= FLAG_SHARED_TABLE;
                }
                let op = if tail {
                    Op::ReturnCallIndirect
                } else {
                    Op::CallIndirect
                };
                self.emit(op, flags, target, arg_base, type_idx)
            }
        };
        if !tail {
            self.record_exception_site(call_pc);
        }
        for _ in 0..params {
            let _ = self.pop()?;
        }
        // A tail call does not return here: the callee's results are this
        // function's results, so nothing lands on the operand stack and the
        // rest of the block is unreachable.
        if tail {
            self.needs_slow_tail_return = true;
            self.bump_region();
            self.dead = true;
            return Ok(());
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

    /// Lower a direct self tail call to a loop backedge.
    ///
    /// The regular call boundary already stages every argument in fresh
    /// temporary slots above the locals. That makes the parameter update a
    /// simple non-overlapping copy, after which resetting the remaining
    /// locals and branching to cell zero is exactly a fresh invocation of
    /// this same function. Unlike `ReturnCall`, this form never exits native
    /// dispatch merely so the Rust driver can replace an activation with an
    /// identical one.
    fn self_tail_call(&mut self, params: u32) -> Result<(), WasmError> {
        let h = self.height() as usize;
        if h < params as usize {
            return Err(desync());
        }
        for i in h - params as usize..h {
            self.materialize_at(i);
        }
        let arg_base = self.temp_slot_used(self.height() - params);
        for _ in 0..params {
            let _ = self.pop()?;
        }
        for i in 0..params {
            self.emit(Op::MovSlot, 0, arg_base + i as u64, 0, i as u64);
        }
        for i in params..self.n_locals {
            self.emit(Op::MovConst, FLAG_A_CONST, 0, 0, i as u64);
        }
        self.emit(Op::Br, 0, 0, 0, 0);
        self.bump_region();
        self.dead = true;
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
                // The native fill/copy handlers bound-check in 32 bits, so a
                // 64-bit memory takes the shared executor -- the same reason
                // loads and stores do.
                let touched = match op {
                    Op::MemoryCopy => (b >> 32) | (b & 0xffff_ffff),
                    Op::MemoryFill => b,
                    _ => b >> 32,
                };
                let flags = if self.memory_is_64(touched) {
                    FLAG_ADDR64
                } else {
                    0
                };
                self.emit(op, flags, base, b, 0);
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
                Fixup::ExceptionTarget(handler) => {
                    self.exception_handlers[handler as usize].target = here;
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
                            catches: Vec::new(),
                        });
                    }
                    Opcode::TRY_TABLE => {
                        let (bt, catches) = match &imm {
                            Immediate::TryTable {
                                block_type,
                                catches,
                            } => (block_type, catches),
                            _ => return Err(desync()),
                        };
                        let (p, r) = block_arity(self.types, bt)?;
                        let catches = self.resolve_try_catches(catches)?;
                        let base = self.height().saturating_sub(p);
                        self.frames.push(CtlFrame {
                            base,
                            params: p,
                            results: r,
                            is_loop: false,
                            is_if: false,
                            dead_entry: true,
                            end_targeted: false,
                            saw_else: false,
                            then_fell_live: false,
                            header: 0,
                            else_fixup: None,
                            fixups: Vec::new(),
                            catches,
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
                        catches: Vec::new(),
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
                        catches: Vec::new(),
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
                Opcode::CALL | Opcode::RETURN_CALL => {
                    let tail = o == Opcode::RETURN_CALL;
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
                    if tail && fidx == self.func_index {
                        self.self_tail_call(ft.params().len() as u32)?;
                    } else {
                        self.call_boundary_tail(
                            Some(fidx as u64),
                            ft.params().len() as u32,
                            ft.results().len() as u32,
                            None,
                            tail,
                        )?;
                    }
                }
                Opcode::CALL_INDIRECT | Opcode::RETURN_CALL_INDIRECT => {
                    let tail = o == Opcode::RETURN_CALL_INDIRECT;
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
                    self.call_boundary_tail(
                        None,
                        ft.params().len() as u32,
                        ft.results().len() as u32,
                        Some((t, t_const, ((table as u64) << 32) | tidx as u64)),
                        tail,
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
                // Named rather than left to the generic fallthrough: this is
                // a whole feature the engine lacks, not one stray opcode, and
                // saying so is what tells a reader where the boundary is.
                Opcode::TRY_TABLE => {
                    let (bt, catches) = match &imm {
                        Immediate::TryTable {
                            block_type,
                            catches,
                        } => (block_type, catches),
                        _ => return Err(desync()),
                    };
                    let (p, r) = block_arity(self.types, bt)?;
                    // A try_table is a block that also remembers where a
                    // matching `throw` should land. The body is ordinary
                    // code; only the catch search distinguishes it.
                    for &d in catches.iter().map(|c| &c.label_idx) {
                        self.mark_branch_target(d);
                    }
                    let catches = self.resolve_try_catches(catches)?;
                    let base = self.height().saturating_sub(p);
                    self.frames.push(CtlFrame {
                        base,
                        params: p,
                        results: r,
                        is_loop: false,
                        is_if: false,
                        // The body is entered from live code. If it branches
                        // to this try_table's label, its end revives control
                        // just like an ordinary block end.
                        dead_entry: false,
                        end_targeted: false,
                        saw_else: false,
                        then_fell_live: false,
                        header: 0,
                        else_fixup: None,
                        fixups: Vec::new(),
                        catches,
                    });
                }
                Opcode::THROW => {
                    let tag_idx = match imm {
                        Immediate::TagIndex(t) => t,
                        _ => return Err(desync()),
                    };
                    let params = self
                        .module
                        .tags()
                        .get(tag_idx as usize)
                        .map(|t| t.func_type().params().len() as u32)
                        .ok_or(WasmError::invalid("interp: bad throw tag"))?;
                    self.materialize_all();
                    let h = self.height();
                    if h < params {
                        return Err(desync());
                    }
                    let base = self.temp_slot(h - params);
                    let throw_pc = self.emit(Op::Throw, 0, tag_idx as u64, base, 0);
                    self.record_exception_site(throw_pc);
                    for _ in 0..params {
                        let _ = self.pop()?;
                    }
                    self.bump_region();
                    self.dead = true;
                }
                Opcode::THROW_REF => {
                    self.materialize_all();
                    let at = self.code.len() as u32;
                    let exn = self.pop()?;
                    let (a, a_const) = self.operand(exn, at);
                    let flags = if a_const { FLAG_A_CONST } else { 0 };
                    let throw_pc = self.emit(Op::ThrowRef, flags, a, 0, 0);
                    self.record_exception_site(throw_pc);
                    self.bump_region();
                    self.dead = true;
                }
                Opcode::BR_ON_NULL | Opcode::BR_ON_NON_NULL => {
                    let d = match imm {
                        Immediate::LabelIndex(d) => d,
                        _ => return Err(desync()),
                    };
                    let on_null = o == Opcode::BR_ON_NULL;
                    self.mark_branch_target(d);
                    self.materialize_all();
                    // `br_on_null` branches WITHOUT the reference and keeps it
                    // on fall-through; `br_on_non_null` branches WITH it and
                    // drops it on fall-through. So the reference is off the
                    // stack across the branch in one case and on it in the
                    // other, which is the only difference between them.
                    let r = self.pop()?;
                    let at = self.code.len() as u32;
                    let (a, a_const) = self.operand(r, at);
                    // The scratch for the null test must sit ABOVE the
                    // reference's own slot: with the reference popped, the
                    // next temp height IS its slot, and the test would
                    // clobber the value the fall-through path still needs.
                    self.stack.push(r);
                    let cond = self.temp_slot_used(self.height());
                    let flags = if a_const { FLAG_A_CONST } else { 0 };
                    self.emit(Op::RefIsNull, flags, a, 0, cond);
                    if on_null {
                        let _ = self.pop()?;
                    }
                    // Guard, moves, jump. The guard skips the branch on the
                    // sense that does NOT take it.
                    let guard = if on_null { Op::BrIfNot } else { Op::BrIf };
                    let skip = self.emit(guard, 0, cond, 0, FIXUP);
                    self.branch_value_moves(d)?;
                    self.emit_branch(Op::Br, 0, 0, d)?;
                    let here = self.code.len() as u64;
                    self.code[skip as usize].c = here;
                    if on_null {
                        self.stack.push(r);
                    } else {
                        let _ = self.pop()?;
                    }
                    self.bump_region();
                }
                Opcode::REF_AS_NON_NULL => {
                    self.value_op(Op::RefAsNonNull, 1)?;
                }
                // Nominally a GC opcode, but it is reference identity and
                // nothing more: non-GC tests use it, and a slot comparison
                // answers it exactly.
                Opcode::REF_EQ => {
                    self.value_op(Op::RefEq, 2)?;
                }
                Opcode::CALL_REF | Opcode::RETURN_CALL_REF => {
                    let tail = o == Opcode::RETURN_CALL_REF;
                    let tidx = match imm {
                        Immediate::TypeIndex(t) => t,
                        _ => return Err(desync()),
                    };
                    let ft = self
                        .types
                        .get_function_type(tidx)
                        .ok_or(WasmError::invalid("interp: bad call_ref type"))?
                        .clone();
                    let at = self.code.len() as u32;
                    let target = self.pop()?;
                    let (t, t_const) = self.operand(target, at);
                    self.call_boundary_ref(
                        ft.params().len() as u32,
                        ft.results().len() as u32,
                        tidx,
                        t,
                        t_const,
                        tail,
                    )?;
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
                    let flags = if self.global_is_shared(g as u64) {
                        FLAG_SHARED_GLOBAL
                    } else {
                        0
                    };
                    let idx = self.emit(Op::GlobalGet, flags, g as u64, 0, dst);
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
                    if self.global_is_shared(g as u64) {
                        flags |= FLAG_SHARED_GLOBAL;
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
                    let tbl = self.br_tables.len() as u32;
                    // A target whose arity leaves operands above it needs the
                    // values moved down, and a table has nowhere to put those
                    // moves -- so such a target gets its own landing pad after
                    // the dispatch: moves, then an ordinary branch. Targets
                    // that need none still point straight at the block.
                    let mut pads: Vec<Option<u32>> = Vec::new();
                    let mut entries = Vec::new();
                    for &d in labels.iter().chain(core::iter::once(&default)) {
                        let n = self.frames.len();
                        if (d as usize) >= n {
                            // function label: lower as a jump to a shared
                            // return we emit right after the table dispatch
                            entries.push(u32::MAX);
                            pads.push(None);
                            continue;
                        }
                        let i = n - 1 - d as usize;
                        let f = &self.frames[i];
                        let arity = if f.is_loop { f.params } else { f.results };
                        if arity != 0 && self.height() != f.base + arity {
                            entries.push(u32::MAX);
                            pads.push(Some(d));
                            continue;
                        }
                        pads.push(None);
                        if self.frames[i].is_loop {
                            entries.push(self.frames[i].header);
                        } else {
                            let entry = entries.len() as u32;
                            self.frames[i].fixups.push(Fixup::Table { tbl, entry });
                            entries.push(u32::MAX);
                        }
                    }
                    self.emit(Op::BrTable, flags, a, 0, tbl as u64);
                    // Per-target move pads.
                    for i in 0..pads.len() {
                        let Some(d) = pads[i] else { continue };
                        let pad = self.code.len() as u32;
                        self.branch_value_moves(d)?;
                        self.emit_branch(Op::Br, 0, 0, d)?;
                        entries[i] = pad;
                    }
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
                            Immediate::MemArg { offset, memidx, .. } => (*memidx, *offset),
                            _ => return Err(desync()),
                        };
                        let (memidx, raw_offset) = offset;
                        let addr64 = self.memory_is_64(memidx as u64);
                        // A memory64 offset can use all 64 bits, leaving no
                        // room for the index beside it; such a cell carries a
                        // side-table index instead. It is always a slow cell,
                        // so nothing native reads the packed form.
                        let offset = if raw_offset >= 1u64 << 48 {
                            if !addr64 {
                                return Err(WasmError::invalid(
                                    "interp: static memory offset does not fit the packed form",
                                ));
                            }
                            let at = self.wide_memargs.len() as u64;
                            self.wide_memargs.push((memidx, raw_offset));
                            WIDE_MEMARG | at
                        } else {
                            ((memidx as u64) << 48) | raw_offset
                        };
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
                            Immediate::MemArg { offset, memidx, .. } => (*memidx, *offset),
                            _ => return Err(desync()),
                        };
                        let (memidx, raw_offset) = offset;
                        let addr64 = self.memory_is_64(memidx as u64);
                        // A memory64 offset can use all 64 bits, leaving no
                        // room for the index beside it; such a cell carries a
                        // side-table index instead. It is always a slow cell,
                        // so nothing native reads the packed form.
                        let offset = if raw_offset >= 1u64 << 48 {
                            if !addr64 {
                                return Err(WasmError::invalid(
                                    "interp: static memory offset does not fit the packed form",
                                ));
                            }
                            let at = self.wide_memargs.len() as u64;
                            self.wide_memargs.push((memidx, raw_offset));
                            WIDE_MEMARG | at
                        } else {
                            ((memidx as u64) << 48) | raw_offset
                        };
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
        self.finish_return_landing();
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
        let tag_handles: StdVec<TagHandle> = module
            .tags()
            .iter()
            .map(|_| TagHandle::mint_fresh())
            .collect();
        predecode_function(&module, &tag_handles, func).expect("predecode")
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
    fn catchless_try_table_branch_to_own_label_revives_its_end() {
        let f = predecode_wat(
            r#"(module
                (func (result i32)
                    (try_table (result i32)
                        (br 0 (i32.const 7)))))"#,
            0,
        );

        assert_eq!(ops(&f), [Op::MovConst, Op::Br, Op::Return]);
        assert_eq!(f.code[1].c, 2, "br 0 must land after the try_table");
        assert_eq!((f.code[2].a, f.code[2].b), (0, 1));
    }

    #[test]
    fn throw_keeps_runtime_handler_metadata_and_payload_layout() {
        let f = predecode_wat(
            r#"(module
                (tag $e (param i32))
                (func (result i32)
                    (block $h (result i32)
                        (try_table (result i32) (catch $e $h)
                            (throw $e (i32.const 7))
                            (i32.const 2))
                        (return))
                    (return)))"#,
            0,
        );

        let throw_pc = f
            .code
            .iter()
            .position(|ins| ins.op == Op::Throw)
            .expect("throw cell") as u32;
        let throw = f.code[throw_pc as usize];
        assert_eq!(throw.a, 0, "throw keeps its module tag index");
        assert_eq!(throw.b, 0, "payload starts in the canonical temp slot");

        let handlers = f.exception_handlers_at(throw_pc);
        assert_eq!(handlers.len(), 1);
        assert!(handlers[0].tag.is_some());
        assert_eq!(handlers[0].payload_arity, 1);
        assert!(!handlers[0].forwards_exn);
        assert_eq!(handlers[0].target_base, 0);
        assert_eq!(f.code[handlers[0].target as usize].op, Op::Return);
    }

    #[test]
    fn handler_chain_is_inner_first_and_tail_calls_have_no_site() {
        let f = predecode_wat(
            r#"(module
                (tag $a)
                (tag $b)
                (func $callee)
                (func
                    (block $outer
                        (try_table (catch_all $outer)
                            (block $inner
                                (try_table
                                    (catch $a $inner)
                                    (catch $b $inner)
                                    (call $callee)
                                    (return_call $callee)))))))"#,
            1,
        );

        let call_pc = f
            .code
            .iter()
            .position(|ins| ins.op == Op::Call)
            .expect("call cell") as u32;
        let tail_pc = f
            .code
            .iter()
            .position(|ins| ins.op == Op::ReturnCall)
            .expect("tail-call cell") as u32;
        let handlers = f.exception_handlers_at(call_pc);
        assert_eq!(handlers.len(), 3);
        assert!(handlers[0].tag.is_some());
        assert!(handlers[1].tag.is_some());
        assert_ne!(handlers[0].tag, handlers[1].tag);
        assert_eq!(handlers[2].tag, None);
        assert!(f.exception_handlers_at(tail_pc).is_empty());
    }

    #[test]
    fn direct_self_tail_call_becomes_a_native_loop() {
        let f = predecode_wat(
            r#"(module
                (func (param i64 i64 i64) (result i64)
                    local.get 0
                    i64.eqz
                    if
                        local.get 1
                        return
                    end
                    local.get 0
                    i64.const 1
                    i64.sub
                    local.get 2
                    local.get 1
                    local.get 2
                    i64.add
                    return_call 0))"#,
            0,
        );

        assert!(
            f.code.iter().all(|ins| ins.op != Op::ReturnCall),
            "a self tail call must not cross into the Rust activation driver"
        );
        assert_eq!(f.code.iter().filter(|ins| ins.op == Op::Br).count(), 1);
        let backedge = f.code.iter().find(|ins| ins.op == Op::Br).unwrap();
        assert_eq!(backedge.c, 0);
        assert_eq!(
            f.code
                .iter()
                .filter(|ins| ins.op == Op::MovSlot && ins.c < 3)
                .count(),
            3
        );
    }

    #[test]
    fn slow_tail_boundaries_reserve_their_result_staging_area() {
        let wat = r#"(module
            (type $pair (func (result i32 i64)))
            (import "host" "pair" (func $imported (type $pair)))
            (func $local (type $pair)
                i32.const 7
                i64.const 9)
            (table 1 funcref)
            (elem (i32.const 0) func $local)
            (func (type $pair) (local i64)
                (return_call $imported))
            (func (type $pair) (local i64)
                (return_call_ref $pair (ref.func $local)))
            (func (type $pair) (local i64)
                (return_call_indirect 0 (type $pair) (i32.const 0))))"#;

        // The import occupies function index 0 and `$local` index 1.
        for (func_index, expected_op) in [
            (2, Op::ReturnCall),
            (3, Op::ReturnCallRef),
            (4, Op::ReturnCallIndirect),
        ] {
            let f = predecode_wat(wat, func_index);
            let tail = f
                .code
                .iter()
                .find(|ins| ins.op == expected_op)
                .expect("tail-call cell");
            let result_end = tail.b as u32 + 2;
            assert!(
                f.frame_slots >= result_end,
                "{expected_op:?} must reserve both result slots at arg_base"
            );
        }
    }

    #[test]
    fn catch_ref_metadata_forwards_the_exception_reference() {
        let f = predecode_wat(
            r#"(module
                (tag $e)
                (func
                    (block $h (result exnref)
                        (try_table (catch_ref $e $h) (throw $e))
                        (unreachable))
                    (drop)))"#,
            0,
        );
        let throw_pc = f
            .code
            .iter()
            .position(|ins| ins.op == Op::Throw)
            .expect("throw cell") as u32;
        let handlers = f.exception_handlers_at(throw_pc);
        assert_eq!(handlers.len(), 1);
        assert_eq!(handlers[0].payload_arity, 0);
        assert!(handlers[0].forwards_exn);
    }

    #[test]
    fn function_catch_ref_reserves_space_for_the_forwarded_reference() {
        let f = predecode_wat(
            r#"(module
                (tag $e)
                (func (result exnref)
                    (try_table (result exnref) (catch_ref $e 0)
                        (throw $e))))"#,
            0,
        );
        let throw_pc = f
            .code
            .iter()
            .position(|ins| ins.op == Op::Throw)
            .expect("throw cell") as u32;
        let handler = f.exception_handlers_at(throw_pc)[0];
        assert!(handler.forwards_exn);
        assert!(f.frame_slots >= 1);
        assert_eq!(f.code[handler.target as usize].op, Op::Return);
    }

    #[test]
    fn throw_ref_is_materialized_and_function_catches_use_a_return_landing() {
        let f = predecode_wat(
            r#"(module
                (func (param exnref)
                    (try_table (catch_all 0)
                        (throw_ref (local.get 0)))))"#,
            0,
        );
        let throw_ref_pc = f
            .code
            .iter()
            .position(|ins| ins.op == Op::ThrowRef)
            .expect("throw_ref cell") as u32;
        let throw_ref = f.code[throw_ref_pc as usize];
        assert_eq!(throw_ref.flags & FLAG_A_CONST, 0);

        let handlers = f.exception_handlers_at(throw_ref_pc);
        assert_eq!(handlers.len(), 1);
        assert_eq!(handlers[0].target_base, 0);
        let landing = f.code[handlers[0].target as usize];
        assert_eq!(landing.op, Op::Return);
        assert_eq!((landing.a, landing.b), (0, 0));
    }

    #[test]
    fn rejects_unsupported_opcodes_cleanly() {
        // SIMD is excluded by design: a v128 lane is a representation change,
        // not more handlers. The point is that it fails predecode with a
        // message naming the family rather than desyncing or trapping later.
        let bin: StdVec<u8> =
            wat::parse_str(r#"(module (func (result v128) v128.const i32x4 0 0 0 0))"#)
                .expect("wat");
        let module = Module::new("t", &bin).expect("module");
        match predecode_function(&module, &[], 0) {
            Ok(_) => panic!("SIMD must be refused"),
            Err(err) => assert!(
                std::format!("{err:?}").contains("SIMD"),
                "the error should name the family, got {err:?}"
            ),
        }
    }
}
