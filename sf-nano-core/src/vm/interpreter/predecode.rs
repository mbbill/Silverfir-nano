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
use crate::opcodes::{Opcode, OpcodeFB, OpcodeFC, WasmOpcode};
use crate::utils::{leb128, limits::Limitable};
use crate::value_type::ValueType;
use crate::vm::tag::TagIdentity;
use crate::vm::value::{ref_to_machine_raw, RefValue};
use core::mem;
#[cfg(test)]
use core::ops::Deref;
use core::ops::{Index, IndexMut, Range};
use tracked_alloc::rc::Rc;

/// Marks a packed memarg field as a `wide_memargs` index rather than an
/// inline `memidx << 48 | offset`. Bit 63 is free in the inline form, whose
/// index occupies bits 48..63.
pub(super) const WIDE_MEMARG: u64 = 1 << 63;

use super::instr::{
    operand_is_f32, operand_is_float, result_is_f32, result_is_float, Instr, Op, FLAG_ADDR64,
    FLAG_A_ACC, FLAG_A_CONST, FLAG_B_ACC, FLAG_B_CONST, FLAG_DST_ACC, FLAG_FUSED, FLAG_NO_NATIVE,
    FLAG_SHARED_GLOBAL, FLAG_SHARED_TABLE,
};
use super::layout::{slot_fields, Pinned};
use super::SLOT_GP_UNIT_BYTES;

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
/// One resolved `try_table` clause at a potentially-throwing instruction.
///
/// `tag = None` is a `catch_all[_ref]`. Typed catches carry the runtime tag
/// identity rather than their module-local tag index, so imported aliases
/// compare correctly. A `_ref` clause receives the exception reference after
/// any typed payload fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExceptionHandler {
    pub(crate) tag: Option<TagIdentity>,
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

/// Test-only immutable view into the pre-link instruction stream.
///
/// Production links from one reusable function scratch and clears it for the
/// next body. Tests retain one module image so the predecoder and old in-place
/// linker oracles can inspect the exact stage-A stream after publication.
#[cfg(test)]
pub(crate) struct FunctionCode {
    arena: Rc<Vec<Instr>>,
    range: Range<usize>,
}

#[cfg(test)]
impl Deref for FunctionCode {
    type Target = [Instr];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.arena[self.range.clone()]
    }
}

/// Four-byte persistent form of [`Pinned`]. Valid pinned locals are below
/// slot 8192 because call records encode their byte offsets in 16 bits, so
/// each slot needs 14 bits including an explicit none value. The final two
/// bits carry the integer/float register-file choice.
#[derive(Clone, Copy)]
struct PackedPinned(u32);

impl PackedPinned {
    const SLOT_BITS: u32 = 14;
    const SLOT_MASK: u32 = (1 << Self::SLOT_BITS) - 1;
    const NONE: u32 = Self::SLOT_MASK;

    fn new(pin: Pinned) -> Self {
        let slot = |value: u64| {
            if value == u64::MAX {
                Self::NONE
            } else {
                debug_assert!(value < 8192);
                value as u32
            }
        };
        Self(
            slot(pin.l0)
                | slot(pin.l1) << Self::SLOT_BITS
                | u32::from(pin.l0_float) << 28
                | u32::from(pin.l1_float) << 29,
        )
    }

    fn expand(self) -> Pinned {
        let slot = |value: u32| {
            if value == Self::NONE {
                u64::MAX
            } else {
                value as u64
            }
        };
        Pinned {
            l0: slot(self.0 & Self::SLOT_MASK),
            l1: slot((self.0 >> Self::SLOT_BITS) & Self::SLOT_MASK),
            l0_float: self.0 & (1 << 28) != 0,
            l1_float: self.0 & (1 << 29) != 0,
        }
    }
}

pub(crate) struct PredecodedFunction {
    #[cfg(test)]
    pub code: FunctionCode,
    /// Final pinned-local choice, maintained alongside instruction emission
    /// and mutation so link does not rescan the instruction stream.
    #[cfg(test)]
    pinned: PackedPinned,
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

/// Mutable function-relative view over the reusable current-function scratch.
///
/// All indices stored by the predecoder are relative to `start`; the wrapper
/// translates only actual vector access. A safety re-decode truncates the
/// arena back to `start`, so the optimistic loop-home pass remains exactly
/// rollbackable even though it no longer owns a per-function vector.
struct FunctionCodeBuilder<'a> {
    arena: &'a mut Vec<Instr>,
    start: usize,
}

impl FunctionCodeBuilder<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.arena.len() - self.start
    }

    #[inline]
    fn push(&mut self, instr: Instr) {
        self.arena.push(instr);
    }

    #[inline]
    fn pop(&mut self) -> Option<Instr> {
        if self.arena.len() == self.start {
            None
        } else {
            self.arena.pop()
        }
    }

    #[inline]
    fn range(&self) -> Range<usize> {
        self.start..self.arena.len()
    }
}

impl Index<usize> for FunctionCodeBuilder<'_> {
    type Output = Instr;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.arena[self.start + index]
    }
}

impl IndexMut<usize> for FunctionCodeBuilder<'_> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.arena[self.start + index]
    }
}

impl PredecodedFunction {
    #[cfg(test)]
    pub(super) fn pinned(&self) -> Pinned {
        self.pinned.expand()
    }

    /// Resolved branch-table targets for this function. The last entry in
    /// the selected table is the default target.
    #[cfg(test)]
    #[inline]
    pub(crate) fn br_table(&self, index: usize) -> Option<&[u32]> {
        self.br_tables.get(index).map(Vec::as_slice)
    }

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
    #[cfg(test)]
    pub(crate) fn has_exception_handlers_at(&self, pc: u32) -> bool {
        self.exception_sites
            .binary_search_by_key(&pc, |site| site.pc)
            .is_ok()
    }
}

/// Minimal constructor for linker unit tests.
///
/// Production functions are always built by `predecode_functions`; keeping
/// this here lets the sibling linker module exercise deliberately malformed
/// ACC hint streams without opening the representation in non-test builds.
#[cfg(test)]
pub(super) fn linker_test_function(
    code: Vec<Instr>,
    br_tables: Vec<Vec<u32>>,
    n_locals: u32,
) -> PredecodedFunction {
    let code_len = code.len();
    let pinned = PackedPinned::new(super::engine::select_pinned_reference(&code, n_locals));
    PredecodedFunction {
        frame_slots: n_locals,
        code: FunctionCode {
            arena: Rc::new(code),
            range: 0..code_len,
        },
        pinned,
        br_tables,
        wide_memargs: Vec::new(),
        n_locals,
        n_params: 0,
        n_results: 0,
        slow_tail_return: None,
        exception_sites: Vec::new(),
        exception_handlers: Vec::new(),
    }
}

/// Predecode one local (non-import) function of a parsed module.
#[cfg(test)]
pub(crate) fn predecode_function(
    module: &Module,
    tag_identities: &[TagIdentity],
    function_handles: &[RefValue],
    func_index: usize,
) -> Result<PredecodedFunction, WasmError> {
    let mut code = Vec::new();
    let mut scratch = PredecodeScratch::default();
    let parts = predecode_function_into(
        module,
        tag_identities,
        function_handles,
        func_index,
        false,
        &mut code,
        &mut scratch,
    )?;
    let code = Rc::new(code);
    Ok(parts.finish(Some(&code)))
}

/// Predecode every local body through one reusable function-sized scratch.
/// `finish_function` runs before that scratch is cleared, letting the caller
/// consume and link the hot instruction stream immediately. Imported
/// functions retain their index-preserving `None` entries in both results.
///
/// Tests additionally retain one immutable module image for the old
/// whole-module in-place linker oracle. Production never allocates or fills
/// that image.
pub(crate) fn predecode_functions<T>(
    module: &Module,
    tag_identities: &[TagIdentity],
    function_handles: &[RefValue],
    mut finish_function: impl FnMut(usize, &UnlinkedFunction<'_>, &[Instr]) -> T,
) -> Result<(UnlinkedPredecodedFunctions, Vec<Option<T>>), WasmError> {
    let local_count = module.functions().iter().filter(|f| !f.is_import()).count();
    if local_count == 0 {
        let mut funcs = Vec::with_capacity(module.functions().len());
        funcs.resize_with(module.functions().len(), || None);
        let mut linked = Vec::with_capacity(module.functions().len());
        linked.resize_with(module.functions().len(), || None);
        return Ok((
            UnlinkedPredecodedFunctions {
                #[cfg(test)]
                code: Vec::new(),
                parts: funcs,
            },
            linked,
        ));
    }
    let body_bytes = module
        .functions()
        .iter()
        .filter_map(|f| f.spec())
        .fold(0usize, |sum, spec| sum.saturating_add(spec.code().len()));
    let average_body_bytes = body_bytes / local_count;
    let mut code = Vec::with_capacity((average_body_bytes / 4).max(1));
    #[cfg(test)]
    let mut test_code = Vec::with_capacity(
        (body_bytes / 4)
            .max(local_count)
            .saturating_add(local_count),
    );
    // These vectors are compile-time state, not part of any published
    // function. Keep their allocations for the next body rather than
    // allocating and freeing the same transient shapes per local function.
    let mut scratch = PredecodeScratch::default();
    let mut parts = Vec::with_capacity(module.functions().len());
    let mut linked = Vec::with_capacity(module.functions().len());
    for (func_index, func) in module.functions().iter().enumerate() {
        if func.is_import() {
            parts.push(None);
            linked.push(None);
        } else {
            code.clear();
            let decoded = predecode_function_into(
                module,
                tag_identities,
                function_handles,
                func_index,
                false,
                &mut code,
                &mut scratch,
            )?;
            #[cfg(test)]
            let mut decoded = decoded;
            debug_assert_eq!(decoded.code_range, 0..code.len());
            linked.push(Some(finish_function(
                func_index,
                &UnlinkedFunction { parts: &decoded },
                &code,
            )));
            #[cfg(test)]
            {
                let start = test_code.len();
                test_code.extend_from_slice(&code);
                decoded.code_range = start..test_code.len();
                // Preserve the old exact stage-A arena, including its
                // prefetch pad, solely for the in-place differential oracle.
                test_code.push(Instr::new(Op::Unreachable, 0, 0, 0, 0));
            }
            parts.push(Some(decoded));
        }
    }

    Ok((
        UnlinkedPredecodedFunctions {
            #[cfg(test)]
            code: test_code,
            parts,
        },
        linked,
    ))
}

struct PredecodedFunctionParts {
    code_range: Range<usize>,
    pinned: PackedPinned,
    br_tables: Vec<Vec<u32>>,
    wide_memargs: Vec<(u32, u64)>,
    frame_slots: u32,
    n_locals: u32,
    n_params: u32,
    n_results: u32,
    slow_tail_return: Option<u32>,
    exception_sites: Vec<ExceptionSite>,
    exception_handlers: Vec<ExceptionHandler>,
}

pub(crate) trait LinkFunction {
    fn code_start(&self) -> usize;
    fn code_len(&self) -> usize;
    fn pinned(&self) -> Pinned;
    fn br_table_count(&self) -> usize;
    fn br_table(&self, index: usize) -> Option<&[u32]>;
    fn has_exception_handlers_at(&self, pc: u32) -> bool;
}

pub(crate) struct UnlinkedFunction<'a> {
    parts: &'a PredecodedFunctionParts,
}

impl LinkFunction for UnlinkedFunction<'_> {
    fn code_start(&self) -> usize {
        self.parts.code_range.start
    }

    fn code_len(&self) -> usize {
        self.parts.code_range.len()
    }

    fn pinned(&self) -> Pinned {
        self.parts.pinned.expand()
    }

    fn br_table_count(&self) -> usize {
        self.parts.br_tables.len()
    }

    fn br_table(&self, index: usize) -> Option<&[u32]> {
        self.parts.br_tables.get(index).map(Vec::as_slice)
    }

    fn has_exception_handlers_at(&self, pc: u32) -> bool {
        self.parts
            .exception_sites
            .binary_search_by_key(&pc, |site| site.pc)
            .is_ok()
    }
}

impl UnlinkedFunction<'_> {
    pub(crate) fn frame_slots(&self) -> u32 {
        self.parts.frame_slots
    }

    pub(crate) fn n_locals(&self) -> u32 {
        self.parts.n_locals
    }

    pub(crate) fn n_params(&self) -> u32 {
        self.parts.n_params
    }
}

pub(crate) struct UnlinkedPredecodedFunctions {
    #[cfg(test)]
    code: Vec<Instr>,
    parts: Vec<Option<PredecodedFunctionParts>>,
}

impl UnlinkedPredecodedFunctions {
    pub(crate) fn len(&self) -> usize {
        self.parts.len()
    }

    pub(crate) fn function(&self, index: usize) -> Option<UnlinkedFunction<'_>> {
        Some(UnlinkedFunction {
            parts: self.parts.get(index)?.as_ref()?,
        })
    }

    #[cfg(test)]
    pub(crate) fn clone_code_for_oracle(&self) -> Vec<Instr> {
        self.code.clone()
    }

    pub(crate) fn publish(
        self,
        test_code: Option<Vec<Instr>>,
    ) -> Vec<Option<Rc<PredecodedFunction>>> {
        let test_code = test_code.map(Rc::new);
        self.parts
            .into_iter()
            .map(|parts| parts.map(|parts| Rc::new(parts.finish(test_code.as_ref()))))
            .collect()
    }

    #[cfg(test)]
    fn publish_for_test(self) -> Vec<Option<Rc<PredecodedFunction>>> {
        let test_code = Some(self.code.clone());
        self.publish(test_code)
    }
}

#[cfg(test)]
impl LinkFunction for PredecodedFunction {
    fn code_start(&self) -> usize {
        0
    }

    fn code_len(&self) -> usize {
        self.code.len()
    }

    fn pinned(&self) -> Pinned {
        self.pinned.expand()
    }

    fn br_table_count(&self) -> usize {
        self.br_tables.len()
    }

    fn br_table(&self, index: usize) -> Option<&[u32]> {
        PredecodedFunction::br_table(self, index)
    }

    fn has_exception_handlers_at(&self, pc: u32) -> bool {
        PredecodedFunction::has_exception_handlers_at(self, pc)
    }
}

impl PredecodedFunctionParts {
    fn finish(self, test_code: Option<&Rc<Vec<Instr>>>) -> PredecodedFunction {
        #[cfg(not(test))]
        let _ = test_code;
        PredecodedFunction {
            #[cfg(test)]
            code: FunctionCode {
                arena: test_code
                    .expect("test publication retains stage-A code")
                    .clone(),
                range: self.code_range,
            },
            #[cfg(test)]
            pinned: self.pinned,
            br_tables: self.br_tables,
            wide_memargs: self.wide_memargs,
            frame_slots: self.frame_slots,
            n_locals: self.n_locals,
            n_params: self.n_params,
            n_results: self.n_results,
            slow_tail_return: self.slow_tail_return,
            exception_sites: self.exception_sites,
            exception_handlers: self.exception_handlers,
        }
    }
}

fn predecode_function_into(
    module: &Module,
    tag_identities: &[TagIdentity],
    function_handles: &[RefValue],
    func_index: usize,
    _disable_fast: bool,
    code: &mut Vec<Instr>,
    scratch: &mut PredecodeScratch,
) -> Result<PredecodedFunctionParts, WasmError> {
    if tag_identities.len() != module.tags().len() {
        return Err(WasmError::invalid(
            "interp: runtime tag table does not match module",
        ));
    }
    if function_handles.len() != module.functions().len() {
        return Err(WasmError::invalid(
            "interp: runtime function table does not match module",
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

    // Optimistically coalesce unique loop parameters with their source
    // locals. If a decoded backedge would have to synthesize a write into
    // such a local, repeat the pass with that loop's parameters canonical.
    // Loop ordinals are structural and therefore stable across passes.
    let code_start = code.len();
    // Move the vector headers out of the module scratch while a Predecoder
    // owns them. `mem::take` itself does not allocate. Every retry and every
    // following function receives the capacities grown by the prior pass.
    let mut stack = mem::take(&mut scratch.stack);
    let mut frames = mem::take(&mut scratch.frames);
    let mut locals = mem::take(&mut scratch.locals);
    let mut canonical_loop_homes = mem::take(&mut scratch.canonical_loop_homes);
    let mut unsafe_loop_homes = mem::take(&mut scratch.unsafe_loop_homes);
    let mut side_scratch = mem::take(&mut scratch.side);
    canonical_loop_homes.clear();

    // These are persistent function outputs. They can reuse a failed
    // optimistic pass, but the successful buffers move into the published
    // function and therefore cannot be shared with the next body.
    let mut br_tables = Vec::new();
    let mut wide_memargs = Vec::new();
    let mut exception_sites = Vec::new();
    let mut exception_handlers = Vec::new();
    let p = loop {
        // Roll back an optimistic pass to its starting cursor. Production's
        // reusable scratch starts at zero; test callers may retain a prefix.
        code.truncate(code_start);
        stack.clear();
        frames.clear();
        locals.clear();
        locals.resize(n_locals as usize, LocalState::default());
        unsafe_loop_homes.clear();
        side_scratch.clear();
        br_tables.clear();
        wide_memargs.clear();
        exception_sites.clear();
        exception_handlers.clear();
        let mut p = Predecoder {
            types: module.types(),
            module,
            tag_identities,
            function_handles,
            code: FunctionCodeBuilder {
                arena: code,
                start: code_start,
            },
            stack,
            frames,
            dead: false,
            region: 0,
            locals,
            last_mat_mov: NO_DEF,
            last_mat_region: 0,
            br_tables,
            wide_memargs,
            exception_sites,
            exception_handlers,
            needs_slow_tail_return: false,
            slow_tail_return: None,
            max_height: 0,
            func_index: func_index as u32,
            n_locals,
            n_results,
            last_call_idx: NO_DEF,
            last_call_height: 0,
            last_call_region: 0,
            pending_fill: None,
            canonical_loop_homes,
            unsafe_loop_homes,
            next_loop_id: 0,
            force_canonical_loop_homes: false,
            side_scratch,
        };
        let mut decoder = Decoder::new(spec.code());
        #[cfg(test)]
        if _disable_fast {
            decoder.disable_predecode_fast_for_test();
        }
        decoder.add_handler(&mut p);
        decoder.decode_function()?;
        drop(decoder);
        let mut learned_unsafe_home = false;
        p.canonical_loop_homes
            .resize(p.next_loop_id as usize, false);
        if p.force_canonical_loop_homes {
            for canonical in &mut p.canonical_loop_homes {
                if !*canonical {
                    learned_unsafe_home = true;
                    *canonical = true;
                }
            }
        } else {
            for (loop_id, &unsafe_home) in p.unsafe_loop_homes.iter().enumerate() {
                if unsafe_home && !p.canonical_loop_homes[loop_id] {
                    learned_unsafe_home = true;
                    p.canonical_loop_homes[loop_id] = true;
                }
            }
        }
        if !learned_unsafe_home {
            break p;
        }

        // Reclaim every buffer before retrying. In particular, a safety
        // re-decode no longer repeats the local-state and control-stack
        // allocations that prompted the retry.
        stack = mem::take(&mut p.stack);
        frames = mem::take(&mut p.frames);
        locals = mem::take(&mut p.locals);
        canonical_loop_homes = mem::take(&mut p.canonical_loop_homes);
        unsafe_loop_homes = mem::take(&mut p.unsafe_loop_homes);
        side_scratch = mem::take(&mut p.side_scratch);
        br_tables = mem::take(&mut p.br_tables);
        wide_memargs = mem::take(&mut p.wide_memargs);
        exception_sites = mem::take(&mut p.exception_sites);
        exception_handlers = mem::take(&mut p.exception_handlers);
    };
    let code_range = p.code.range();
    let pinned = PackedPinned::new(select_pinned(&p.locals));
    #[cfg(test)]
    assert_eq!(
        pinned.expand(),
        super::engine::select_pinned_reference(&p.code.arena[code_range.clone()], n_locals,),
        "incremental pinned-local census disagrees with the original full-stream scan"
    );
    let mut p = p;
    let parts = PredecodedFunctionParts {
        code_range,
        pinned,
        frame_slots: n_locals + p.max_height,
        br_tables: mem::take(&mut p.br_tables),
        wide_memargs: mem::take(&mut p.wide_memargs),
        n_locals,
        n_params,
        n_results,
        slow_tail_return: p.slow_tail_return,
        exception_sites: mem::take(&mut p.exception_sites),
        exception_handlers: mem::take(&mut p.exception_handlers),
    };
    scratch.stack = mem::take(&mut p.stack);
    scratch.frames = mem::take(&mut p.frames);
    scratch.locals = mem::take(&mut p.locals);
    scratch.canonical_loop_homes = mem::take(&mut p.canonical_loop_homes);
    scratch.unsafe_loop_homes = mem::take(&mut p.unsafe_loop_homes);
    scratch.side = mem::take(&mut p.side_scratch);
    Ok(parts)
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

/// One stable source-to-home copy performed on a taken branch.
///
/// Sources are always canonical temp slots by the time a plan is built.
/// Destinations use local indices directly and stack heights for temp homes,
/// so frame-slot accounting stays centralized in `temp_slot_used`.
#[derive(Clone, Copy)]
struct BranchCopy {
    source_height: u32,
    destination: BranchHome,
}

#[derive(Clone, Copy)]
enum BranchHome {
    Local(u32),
    Temp(u32),
}

/// Everything target-specific about one structured branch transfer.
///
/// `br_table` builds one plan per target depth, so repeated entries share one
/// landing pad while copy-free loop/block targets remain directly addressable.
struct BranchPlan {
    depth: u32,
    /// Range in the module-scoped `PredecodeScratch::branch_copies` buffer.
    /// Copies are consumed before another branch opcode can reset it.
    copies: Range<usize>,
}

#[derive(Clone, Copy)]
struct PendingMemoryFill {
    operands: [Desc; 3],
    base_height: u32,
    memory: u32,
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
    tag: Option<TagIdentity>,
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
    /// Per loop parameter, the unique local whose slot is also the
    /// parameter's runtime home. `NO_DEF` means the canonical temp slot.
    /// Empty for non-loop frames.
    param_locals: Vec<u32>,
    /// Structural ordinal of this loop, stable across safety-redecode
    /// passes. `NO_DEF` for non-loop frames.
    loop_id: u32,
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

/// Module-scoped capacity cache for state that exists only while translating
/// one function body.
///
/// Contents are cleared between passes and functions; only allocations are
/// retained. None of these values is referenced by a `PredecodedFunction`,
/// so sharing the scratch cannot affect published code, side tables,
/// exception metadata, or activation/re-entry lifetimes.
#[derive(Default)]
struct PredecodeScratch {
    stack: Vec<Desc>,
    frames: Vec<CtlFrame>,
    locals: Vec<LocalState>,
    canonical_loop_homes: Vec<bool>,
    unsafe_loop_homes: Vec<bool>,
    side: PredecodeSideScratch,
}

/// Opcode-scoped workspaces, grouped so ownership can move between the
/// module scratch and a `Predecoder` as one value rather than shuffling ten
/// independent vector headers on every function and safety retry.
#[derive(Default)]
struct PredecodeSideScratch {
    /// Short-lived side workspaces. Unlike `br_tables` and exception output,
    /// none of these buffers is reachable from the finished function.
    loop_candidates: Vec<(u32, usize)>,
    branch_copies: Vec<BranchCopy>,
    branch_plans: Vec<BranchPlan>,
    br_depths: Vec<u32>,
    br_plan_for_target: Vec<Option<u32>>,
    br_entry_plans: Vec<u32>,
    br_pad_plans: Vec<u32>,
    br_pad_for_plan: Vec<Option<u32>>,
    br_entry_pads: Vec<Option<u32>>,
    br_pad_addresses: Vec<u32>,
}

impl PredecodeSideScratch {
    fn clear(&mut self) {
        self.loop_candidates.clear();
        self.branch_copies.clear();
        self.branch_plans.clear();
        self.br_depths.clear();
        self.br_plan_for_target.clear();
        self.br_entry_plans.clear();
        self.br_pad_plans.clear();
        self.br_pad_for_plan.clear();
        self.br_entry_pads.clear();
        self.br_pad_addresses.clear();
    }
}

#[cfg(test)]
impl PredecodeScratch {
    /// Capacities of every allocation-owning transient vector. Persistent
    /// function output (`Instr`, branch tables, wide memargs, exceptions) is
    /// intentionally absent from this census.
    fn capacities(&self) -> [usize; 15] {
        [
            self.stack.capacity(),
            self.frames.capacity(),
            self.locals.capacity(),
            self.canonical_loop_homes.capacity(),
            self.unsafe_loop_homes.capacity(),
            self.side.loop_candidates.capacity(),
            self.side.branch_copies.capacity(),
            self.side.branch_plans.capacity(),
            self.side.br_depths.capacity(),
            self.side.br_plan_for_target.capacity(),
            self.side.br_entry_plans.capacity(),
            self.side.br_pad_plans.capacity(),
            self.side.br_pad_for_plan.capacity(),
            self.side.br_entry_pads.capacity(),
            self.side.br_pad_addresses.capacity(),
        ]
    }
}

struct Predecoder<'m, 'code> {
    types: &'m TypeContext,
    module: &'m Module,
    /// Runtime identities resolved by the linker. Module tag indices are
    /// aliases for these handles, not identities themselves: two imports may
    /// name one tag, while two same-signature tags remain distinct.
    tag_identities: &'m [TagIdentity],
    /// Frame-form identities for `ref.func`: local indices for this
    /// instance's functions and absolute handles for linked imports.
    function_handles: &'m [RefValue],
    code: FunctionCodeBuilder<'code>,
    stack: Vec<Desc>,
    frames: Vec<CtlFrame>,
    dead: bool,
    region: u32,
    /// Per-local dataflow state and the live pinned-local census. Keeping the
    /// two together avoids adding an allocation to each decoded function.
    locals: Vec<LocalState>,
    /// Index of an immediately preceding plain `MovSlot` that another
    /// ordered copy may merge with (NO_DEF = none), and its region.
    ///
    /// This covers both stack materialization and ordinary local
    /// assignments. Keeping the pairing rule here avoids teaching any
    /// particular Wasm source pattern about `MovPair`.
    last_mat_mov: u32,
    last_mat_region: u32,
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
    /// A fill whose only following Wasm operations have so far been
    /// side-effect-free operand pushes. Delaying it across local.get/const
    /// exposes the common adjacent fill+copy pair without moving it across
    /// any instruction that can trap or mutate state.
    pending_fill: Option<PendingMemoryFill>,
    /// Loops proven unsafe for local-backed parameter homes by an earlier
    /// optimistic pass.
    canonical_loop_homes: Vec<bool>,
    /// Local-backed homes that would require a synthetic Wasm-local write
    /// on some backedge in this pass.
    unsafe_loop_homes: Vec<bool>,
    /// Next structural loop ordinal.
    next_loop_id: u32,
    /// Exception landing metadata currently names canonical temp spans, so
    /// an actual decoded `try_table` requires all loop homes to stay there.
    force_canonical_loop_homes: bool,
    /// Reusable workspaces borrowed from `PredecodeScratch`. They are reset
    /// at each opcode that owns their contents and never enter published
    /// metadata.
    side_scratch: PredecodeSideScratch,
}

/// Predecode state for one WebAssembly local.
///
/// The first three fields drive destination and accumulator folding. The
/// remaining counters describe the final instruction stream exactly enough
/// to choose its two pinned locals without a later full-stream scan. Domain
/// facts use counters rather than sticky bits because load/store fusion and
/// the i64 subtraction-branch rewrite can remove an already emitted access.
#[derive(Clone, Copy, Default)]
struct LocalState {
    /// 1 + index of the last emitted instruction reading this local.
    last_read: u32,
    /// 1 + index of the last emitted instruction writing this local.
    last_write: u32,
    /// Control region of the last write. Accumulator write-through is safe
    /// only when the writer and reader remain in the same region.
    last_write_region: u32,
    pin_count: u32,
    pin_r_int: u32,
    pin_r_float: u32,
    pin_w_int: u32,
    pin_w_float: u32,
    pin_f32: u32,
}

#[inline]
fn adjust(counter: &mut u32, add: bool) {
    if add {
        *counter += 1;
    } else {
        debug_assert!(*counter != 0);
        *counter -= 1;
    }
}

#[inline]
fn census_read(locals: &mut [LocalState], slot: u64, is_float: bool, is_f32: bool, add: bool) {
    let stat = &mut locals[slot as usize];
    adjust(&mut stat.pin_count, add);
    if is_float {
        adjust(&mut stat.pin_r_float, add);
        if is_f32 {
            adjust(&mut stat.pin_f32, add);
        }
    } else {
        adjust(&mut stat.pin_r_int, add);
    }
}

#[inline]
fn census_write(
    locals: &mut [LocalState],
    slot: u64,
    is_float: bool,
    is_f32: bool,
    counted: bool,
    add: bool,
) {
    let stat = &mut locals[slot as usize];
    if counted {
        adjust(&mut stat.pin_count, add);
    }
    if is_float {
        adjust(&mut stat.pin_w_float, add);
        if is_f32 {
            adjust(&mut stat.pin_f32, add);
        }
    } else {
        adjust(&mut stat.pin_w_int, add);
    }
}

/// Apply or remove one instruction's contribution using the original
/// link-time census rules. Keeping this function field-for-field equivalent
/// to the test-only oracle makes forged instruction streams checkable too.
fn update_pin_census(locals: &mut [LocalState], ins: Instr, add: bool) {
    let n = locals.len() as u64;
    let (a_s, b_s, c_d) = slot_fields(ins.op);
    if a_s && ins.flags & FLAG_A_CONST == 0 && ins.a < n {
        census_read(
            locals,
            ins.a,
            operand_is_float(ins.op, false),
            operand_is_f32(ins.op, false),
            add,
        );
    }
    if b_s && ins.flags & FLAG_B_CONST == 0 && ins.b < n {
        census_read(
            locals,
            ins.b,
            operand_is_float(ins.op, true),
            operand_is_f32(ins.op, true),
            add,
        );
    }
    if c_d && ins.c < n {
        census_write(
            locals,
            ins.c,
            result_is_float(ins.op),
            result_is_f32(ins.op),
            true,
            add,
        );
    }
    if matches!(ins.op, Op::I32_SubBrIf | Op::I64_SubBrIf) && ins.a < n {
        census_write(locals, ins.a, false, false, true, add);
    }
    if ins.op == Op::Select {
        let dst = ins.c & 0xffff_ffff;
        if dst < n {
            census_write(locals, dst, false, false, false, add);
        }
    }
    if ins.op == Op::MovPair {
        for dst in [ins.c >> 32, ins.c & 0xffff_ffff] {
            if dst < n {
                census_write(locals, dst, false, false, true, add);
            }
        }
    }
}

/// Select the same top two locals as the old linker pass from the live
/// counters maintained above.
fn select_pinned(locals: &[LocalState]) -> Pinned {
    if locals.is_empty() {
        return Pinned::NONE;
    }
    let (has_l1, has_float_regs, float_pin_f32) = super::engine::PIN_CAPS;
    let float_ok = |i: usize| has_float_regs && (float_pin_f32 || locals[i].pin_f32 == 0);
    let mut best = (usize::MAX, 0u32);
    let mut second = (usize::MAX, 0u32);
    for (i, stat) in locals.iter().enumerate() {
        let wdom = u8::from(stat.pin_w_int != 0) | (u8::from(stat.pin_w_float != 0) << 1);
        if wdom == 3 || (wdom & 2 != 0 && !float_ok(i)) {
            continue;
        }
        if stat.pin_count > best.1 {
            second = best;
            best = (i, stat.pin_count);
        } else if stat.pin_count > second.1 {
            second = (i, stat.pin_count);
        }
    }
    let ok = |(i, count): (usize, u32)| count > 0 && i * 8 < 1 << 16;
    let mode = |i: usize| {
        let no_writers = locals[i].pin_w_int == 0 && locals[i].pin_w_float == 0;
        float_ok(i)
            && (locals[i].pin_w_float != 0
                || (no_writers && locals[i].pin_r_float != 0 && locals[i].pin_r_int == 0))
    };
    let (l0, l0_float) = if ok(best) {
        (best.0 as u64, mode(best.0))
    } else {
        (u64::MAX, false)
    };
    let (l1, l1_float) = if has_l1 && l0 != u64::MAX && ok(second) {
        (second.0 as u64, mode(second.0))
    } else {
        (u64::MAX, false)
    };
    Pinned {
        l0,
        l1,
        l0_float,
        l1_float,
    }
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

#[derive(Clone, Copy)]
struct BrIfFusion {
    op: Op,
    b_const: Option<u64>,
}

#[derive(Clone, Copy)]
struct SubBrIfFusion {
    def: u32,
    op: Op,
    b: u64,
    /// The i64 form consumes an intervening `i64.ne local, 0` cell.
    remove_condition: bool,
    local: u32,
}

/// Extend ordinary compare/branch fusion with a full-width lowering for
/// `i64.eqz`. Plain `BrIf` cannot represent this because it tests only the
/// low 32 bits; comparing the original i64 operand with constant zero can.
fn fuse_br_if(op: Op, invert: bool) -> Option<BrIfFusion> {
    let op = match op {
        Op::I64_Eqz if invert => Op::I64_BrNe,
        Op::I64_Eqz => Op::I64_BrEq,
        op => {
            return fuse_cmp_br(op, invert).map(|op| BrIfFusion { op, b_const: None });
        }
    };
    Some(BrIfFusion {
        op,
        b_const: Some(0),
    })
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

impl<'m, 'code> Predecoder<'m, 'code> {
    fn allocate_loop_id(&mut self) -> u32 {
        let loop_id = self.next_loop_id;
        self.next_loop_id = self
            .next_loop_id
            .checked_add(1)
            .expect("loop ordinal overflow");
        self.unsafe_loop_homes.push(false);
        loop_id
    }

    fn loop_uses_canonical_homes(&self, loop_id: u32) -> bool {
        self.canonical_loop_homes
            .get(loop_id as usize)
            .copied()
            .unwrap_or(false)
    }

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
                        .tag_identities
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
        if !self.has_active_exception_handlers() {
            return;
        }

        debug_assert!(self.exception_sites.last().is_none_or(|site| site.pc < pc));
        let handlers_start = self.exception_handlers.len() as u32;
        // Copy one clause before mutating the target frame. This preserves
        // the old innermost-first flattened order without allocating a fresh
        // `Vec` for every potentially throwing cell.
        for active_frame in (0..self.frames.len()).rev() {
            let catch_count = self.frames[active_frame].catches.len();
            for catch_index in 0..catch_count {
                let handler = self.frames[active_frame].catches[catch_index];
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
        let ins = Instr::new(op, flags, a, b, c);
        update_pin_census(&mut self.locals, ins, true);
        self.code.push(ins);
        idx
    }

    /// Replace an emitted cell while keeping the live pinned-local census in
    /// step. Every in-place fusion and destination patch goes through here.
    fn replace_instr(&mut self, index: u32, ins: Instr) {
        let old = self.code[index as usize];
        update_pin_census(&mut self.locals, old, false);
        update_pin_census(&mut self.locals, ins, true);
        self.code[index as usize] = ins;
    }

    fn edit_instr(&mut self, index: u32, edit: impl FnOnce(&mut Instr)) {
        let mut ins = self.code[index as usize];
        edit(&mut ins);
        self.replace_instr(index, ins);
    }

    fn pop_instr(&mut self) -> Option<Instr> {
        let ins = self.code.pop()?;
        update_pin_census(&mut self.locals, ins, false);
        Some(ins)
    }

    /// Emit one ordered slot copy, merging two adjacent plain copies into
    /// `MovPair`. The pair deliberately preserves program order: copy 1 is
    /// committed before copy 2 reads, so `src2 == dst1` remains exact.
    ///
    /// Returns the cell that owns the copy. When a pair is formed, both
    /// copies share the preceding cell and no new cell is appended.
    fn emit_ordered_mov_slot(&mut self, flags: u16, src: u64, dst: u64) -> u32 {
        let at = self.code.len() as u32;
        if flags == 0
            && self.last_mat_mov != NO_DEF
            && self.last_mat_mov + 1 == at
            && self.last_mat_region == self.region
            && self.code[self.last_mat_mov as usize].op == Op::MovSlot
            && self.code[self.last_mat_mov as usize].flags == 0
            && self.code[self.last_mat_mov as usize].c != dst
        {
            let prev = self.last_mat_mov;
            let pm = self.code[prev as usize];
            debug_assert!(pm.c <= u32::MAX as u64 && dst <= u32::MAX as u64);
            self.replace_instr(
                prev,
                Instr {
                    op: Op::MovPair,
                    flags: 0,
                    head_pad: 0,
                    a: pm.a,
                    b: src,
                    c: pm.c << 32 | dst,
                },
            );
            self.last_mat_mov = NO_DEF; // pairs never re-pair
            prev
        } else {
            let idx = self.emit(Op::MovSlot, flags, src, 0, dst);
            if flags == 0 {
                self.last_mat_mov = idx;
                self.last_mat_region = self.region;
            }
            idx
        }
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
                self.locals[i as usize].last_read = at + 1;
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
                let idx = self.emit_ordered_mov_slot(0, l as u64, slot);
                self.locals[l as usize].last_read = idx + 1;
                idx
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

    /// Stage a descriptor into an explicitly chosen temporary slot.
    ///
    /// Pending bulk fusion needs six values live at once even though the
    /// Wasm operand stack reuses the same three logical heights for the
    /// second operation. Unlike `materialize_at`, this helper does not
    /// change the virtual stack descriptor.
    fn stage_desc_at(&mut self, d: Desc, height: u32) {
        let dst = self.temp_slot_used(height);
        match d {
            Desc::Local(l) => {
                let idx = self.emit_ordered_mov_slot(0, l as u64, dst);
                self.locals[l as usize].last_read = idx + 1;
            }
            Desc::ConstV(k) => {
                self.emit(Op::MovConst, FLAG_A_CONST, k, 0, dst);
            }
            Desc::Temp {
                height: src_height, ..
            } => {
                let src = self.temp_slot_used(src_height);
                if src != dst {
                    self.emit_ordered_mov_slot(0, src, dst);
                }
            }
        }
    }

    fn stage_three(&mut self, operands: [Desc; 3], base_height: u32) {
        for (i, operand) in operands.into_iter().enumerate() {
            self.stage_desc_at(operand, base_height + i as u32);
        }
    }

    fn defer_memory_fill(&mut self, memory: u32) -> Result<(), WasmError> {
        let h = self.stack.len();
        if h < 3 {
            return Err(desync());
        }
        let base_height = (h - 3) as u32;
        let operands = [self.stack[h - 3], self.stack[h - 2], self.stack[h - 1]];
        self.stack.truncate(h - 3);
        self.pending_fill = Some(PendingMemoryFill {
            operands,
            base_height,
            memory,
        });
        Ok(())
    }

    fn flush_pending_fill(&mut self) {
        let Some(fill) = self.pending_fill.take() else {
            return;
        };
        self.stage_three(fill.operands, fill.base_height);
        let base = self.temp_slot_used(fill.base_height);
        let mut flags = if self.memory_is_64(fill.memory as u64) {
            FLAG_ADDR64
        } else {
            0
        };
        if fill.memory != 0 {
            flags |= FLAG_NO_NATIVE;
        }
        self.emit(Op::MemoryFill, flags, base, fill.memory as u64, 0);
    }

    /// Fuse only the exact straight-line shape exposed by delaying a fill
    /// across local.get/const pushes. No arithmetic, state mutation, or
    /// potentially trapping instruction may move across the pair.
    fn try_fuse_pending_fill_copy(
        &mut self,
        dst_memory: u32,
        src_memory: u32,
    ) -> Result<bool, WasmError> {
        let Some(fill) = self.pending_fill else {
            return Ok(false);
        };
        if fill.memory != dst_memory
            || fill.memory != src_memory
            || self.memory_is_64(fill.memory as u64)
        {
            return Ok(false);
        }
        let base = fill.base_height as usize;
        if self.stack.len() != base + 3
            || self.stack[base..]
                .iter()
                .any(|d| matches!(d, Desc::Temp { .. }))
        {
            return Ok(false);
        }
        let copy = [self.stack[base], self.stack[base + 1], self.stack[base + 2]];
        self.stack.truncate(base);
        self.pending_fill = None;
        self.stage_three(fill.operands, fill.base_height);
        self.stage_three(copy, fill.base_height + 3);
        let fill_base = self.temp_slot_used(fill.base_height);
        let copy_base = self.temp_slot_used(fill.base_height + 3);
        let flags = if fill.memory == 0 { 0 } else { FLAG_NO_NATIVE };
        self.emit(
            Op::MemoryFillCopy,
            flags,
            fill_base,
            copy_base,
            fill.memory as u64,
        );
        Ok(true)
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
                    // MovPair always writes both destinations through and
                    // leaves only its SECOND ordered copy in the integer
                    // accumulator. Its low packed destination is exactly
                    // the temp accepted by `rewritable_producer` above.
                    // Keep this paired with the local-destination gate
                    // below and the native-handler invariant documented on
                    // `Op::MovPair`.
                    if self.code[def as usize].op != Op::MovPair {
                        self.edit_instr(def, |ins| ins.flags |= FLAG_DST_ACC);
                    }
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
                let write = self.locals[x as usize].last_write;
                if write > 0
                    && write == self.code.len() as u32
                    && self.locals[x as usize].last_write_region == self.region
                    && result_is_float(self.code[write as usize - 1].op) == want_float
                    // Both packed destinations are written by one MovPair
                    // cell, but only destination 2 is its accumulator
                    // result. Let destination 1 fall back to its frame slot.
                    // Keep this paired with the temp case above.
                    && (self.code[write as usize - 1].op != Op::MovPair
                        || self.code[write as usize - 1].c & 0xffff_ffff == x as u64)
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
                        self.edit_instr(def, |ins| ins.op = inv);
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
        self.edit_instr(def, |instr| {
            if instr.op == Op::Select || instr.flags & FLAG_FUSED != 0 {
                instr.c = (instr.c & !0xffff_ffff) | slot;
            } else {
                instr.c = slot;
            }
        });
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
                    && self.locals[idx as usize].last_read <= def + 1
                    && self.locals[idx as usize].last_write <= def + 1 =>
            {
                Some(def)
            }
            _ => None,
        };

        if let Some(def) = fold_def {
            self.patch_dst(def, idx as u64);
            self.locals[idx as usize].last_write = def + 1;
            self.locals[idx as usize].last_write_region = self.region;
        } else {
            let (op, flags, a, read_local) = match top {
                Desc::Local(src) => {
                    let acc = self.acc_operand(top, FLAG_A_ACC, false);
                    (Op::MovSlot, acc, src as u64, Some(src))
                }
                Desc::ConstV(k) => (Op::MovConst, FLAG_A_CONST, k, None),
                Desc::Temp { height, .. } => {
                    let acc = self.acc_operand(top, FLAG_A_ACC, false);
                    (Op::MovSlot, acc, self.temp_slot_used(height), None)
                }
            };
            let at = if op == Op::MovSlot {
                self.emit_ordered_mov_slot(flags, a, idx as u64)
            } else {
                self.emit(op, flags, a, 0, idx as u64)
            };
            if let Some(src) = read_local {
                self.locals[src as usize].last_read = at + 1;
            }
            self.locals[idx as usize].last_write = at + 1;
            self.locals[idx as usize].last_write_region = self.region;
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

    fn branch_arity(&self, depth: u32) -> u32 {
        let n = self.frames.len();
        if (depth as usize) >= n {
            return self.n_results;
        }
        let f = &self.frames[n - 1 - depth as usize];
        if f.is_loop {
            f.params
        } else {
            f.results
        }
    }

    fn branch_param_local(&self, depth: u32, param: u32) -> Option<u32> {
        let n = self.frames.len();
        if (depth as usize) >= n {
            return None;
        }
        let f = &self.frames[n - 1 - depth as usize];
        if !f.is_loop {
            return None;
        }
        let local = *f.param_locals.get(param as usize)?;
        (local != NO_DEF).then_some(local)
    }

    /// Prepare a direct or conditional branch for transfer planning.
    ///
    /// `br_if` may fall through, so every pending stack descriptor must stay
    /// valid if the taken path writes an aliased local. Exact loop
    /// parameter/local pairs are the only values that can safely remain
    /// pending; every mismatch is staged before any destination write.
    fn prepare_branch(&mut self, depth: u32) -> Result<(), WasmError> {
        let n = self.frames.len();
        if (depth as usize) >= n {
            self.materialize_all();
            return Ok(());
        }
        let frame_index = n - 1 - depth as usize;
        let f = &self.frames[frame_index];
        let (is_loop, target_base, arity) = (
            f.is_loop,
            f.base,
            if f.is_loop { f.params } else { f.results },
        );
        let h = self.height();
        let target_end = target_base.checked_add(arity).ok_or_else(desync)?;
        if h < target_end {
            return Err(desync());
        }
        let source_base = h - arity;
        for i in 0..self.stack.len() {
            let source_param = (i as u32).checked_sub(source_base);
            let preserve = is_loop
                && source_param.is_some_and(|p| {
                    p < arity && {
                        let local = self.frames[frame_index].param_locals[p as usize];
                        local != NO_DEF && self.stack[i] == Desc::Local(local)
                    }
                });
            if !preserve {
                self.materialize_at(i);
            }
        }
        Ok(())
    }

    /// Prepare the values shared by every `br_table` target.
    ///
    /// A table never falls through, so values between the deepest surviving
    /// target prefix and the common branch tuple are dead on every outcome
    /// and need no materialization. A pending tuple local can remain in place
    /// only when every target aliases that exact local at the same parameter.
    fn prepare_br_table(&mut self, depths: &[u32]) -> Result<(), WasmError> {
        let Some(&first) = depths.first() else {
            return Err(desync());
        };
        let arity = self.branch_arity(first);
        let h = self.height();
        if h < arity {
            return Err(desync());
        }

        let mut live_prefix = 0;
        for &depth in depths {
            if self.branch_arity(depth) != arity {
                return Err(desync());
            }
            let n = self.frames.len();
            if (depth as usize) >= n {
                continue;
            }
            let f = &self.frames[n - 1 - depth as usize];
            let target_end = f.base.checked_add(arity).ok_or_else(desync)?;
            if h < target_end {
                return Err(desync());
            }
            live_prefix = live_prefix.max(f.base);
        }

        let source_base = h - arity;
        if live_prefix > source_base {
            return Err(desync());
        }
        for i in 0..live_prefix as usize {
            self.materialize_at(i);
        }
        for param in 0..arity {
            let at = (source_base + param) as usize;
            let preserve = match self.stack[at] {
                Desc::Local(local) => depths
                    .iter()
                    .all(|&depth| self.branch_param_local(depth, param) == Some(local)),
                _ => false,
            };
            if !preserve {
                self.materialize_at(at);
            }
        }
        Ok(())
    }

    /// Build the target-specific transfer after branch sources have been
    /// prepared. The returned copies all read stable temp slots, so emitting
    /// them in parameter order preserves parallel-copy semantics.
    fn plan_branch(&mut self, depth: u32) -> Result<BranchPlan, WasmError> {
        let n = self.frames.len();
        let arity = self.branch_arity(depth);
        let h = self.height();
        if h < arity {
            return Err(desync());
        }
        let source_base = h - arity;
        let copies_start = self.side_scratch.branch_copies.len();

        if (depth as usize) >= n {
            for i in 0..arity {
                if !matches!(
                    self.stack[(source_base + i) as usize],
                    Desc::Temp { height, .. } if height == source_base + i
                ) {
                    return Err(desync());
                }
            }
            return Ok(BranchPlan {
                depth,
                copies: copies_start..copies_start,
            });
        }

        let frame_index = n - 1 - depth as usize;
        let (target_base, is_loop, loop_id) = {
            let f = &self.frames[frame_index];
            (f.base, f.is_loop, f.loop_id)
        };
        let target_end = target_base.checked_add(arity).ok_or_else(desync)?;
        if h < target_end {
            return Err(desync());
        }
        for i in 0..arity {
            let source = self.stack[(source_base + i) as usize];
            let param_local = if is_loop {
                self.frames[frame_index].param_locals[i as usize]
            } else {
                NO_DEF
            };
            let destination = if param_local != NO_DEF {
                BranchHome::Local(param_local)
            } else {
                BranchHome::Temp(target_base.checked_add(i).ok_or_else(desync)?)
            };

            match (source, destination) {
                (Desc::Local(source), BranchHome::Local(destination)) if source == destination => {}
                (Desc::Temp { height, .. }, BranchHome::Temp(destination))
                    if height == destination => {}
                (Desc::Temp { height, .. }, destination) => {
                    if matches!(destination, BranchHome::Local(_)) {
                        self.unsafe_loop_homes[loop_id as usize] = true;
                    }
                    self.side_scratch.branch_copies.push(BranchCopy {
                        source_height: height,
                        destination,
                    });
                }
                _ => return Err(desync()),
            }
        }
        Ok(BranchPlan {
            depth,
            copies: copies_start..self.side_scratch.branch_copies.len(),
        })
    }

    fn emit_branch_copies(&mut self, plan: &BranchPlan) {
        for copy_index in plan.copies.clone() {
            let copy = self.side_scratch.branch_copies[copy_index];
            let src = self.temp_slot_used(copy.source_height);
            let (dst, dst_local) = match copy.destination {
                BranchHome::Local(local) => (local as u64, Some(local)),
                BranchHome::Temp(height) => (self.temp_slot_used(height), None),
            };
            let at = self.emit(Op::MovSlot, 0, src, 0, dst);
            if let Some(local) = dst_local {
                self.locals[local as usize].last_write = at + 1;
                self.locals[local as usize].last_write_region = self.region;
            }
        }
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
                    self.edit_instr(skip, |ins| ins.c = here);
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
            let target = self.frames[i].header as u64;
            self.edit_instr(def, |ins| ins.c = target);
        } else {
            self.edit_instr(def, |ins| ins.c = FIXUP);
            self.frames[i].fixups.push(Fixup::InstrC(def));
        }
    }

    /// Recognize an in-place arithmetic update that can own a following
    /// nonzero branch. `local` is both the first operand and destination.
    fn sub_br_if_update(&self, def: u32, local: u32, i64_width: bool) -> Option<SubBrIfFusion> {
        let ins = self.code[def as usize];
        if ins.a != local as u64 || ins.c != local as u64 || ins.flags & FLAG_A_CONST != 0 {
            return None;
        }

        let (op, b) = match (i64_width, ins.op) {
            (false, Op::I32_Sub) => (Op::I32_SubBrIf, ins.b),
            (false, Op::I32_Add) if ins.flags & FLAG_B_CONST != 0 => {
                // Keep paired with the I64_Add arm below: both encode
                // `x + k` as `x - wrapping_neg(k)`. The explicit i32 width
                // is intentional, especially for i32::MIN; this is why
                // there is no separate I32_AddBrIf opcode.
                (Op::I32_SubBrIf, 0u32.wrapping_sub(ins.b as u32) as u64)
            }
            (true, Op::I64_Sub) => (Op::I64_SubBrIf, ins.b),
            (true, Op::I64_Add) if ins.flags & FLAG_B_CONST != 0 => {
                // Keep paired with the I32_Add arm above: use the same
                // add-via-sub identity at i64 width. `wrapping_sub` keeps
                // i64::MIN and every other bit pattern exact, without a
                // separate I64_AddBrIf opcode.
                (Op::I64_SubBrIf, 0u64.wrapping_sub(ins.b))
            }
            _ => return None,
        };
        Some(SubBrIfFusion {
            def,
            op,
            b,
            remove_condition: false,
            local,
        })
    }

    /// Match the ordinary i32 condition form, where the updated local is
    /// itself a valid `br_if` condition.
    fn direct_sub_br_if(&self, cond: Desc) -> Option<SubBrIfFusion> {
        let Desc::Local(local) = cond else {
            return None;
        };
        let write = self.locals[local as usize].last_write;
        if write == 0
            || write != self.code.len() as u32
            || self.locals[local as usize].last_write_region != self.region
        {
            return None;
        }
        self.sub_br_if_update(write - 1, local, false)
    }

    /// Match the full-width nonzero test required after an i64 update.
    /// Wasm's `br_if` condition is i32, so an i64 local reaches it through
    /// an immediately preceding `i64.ne local, 0` comparison.
    fn i64_sub_br_if(&self, cond: Desc) -> Option<SubBrIfFusion> {
        let cmp_def = self.rewritable_producer(cond)?;
        let cmp = self.code[cmp_def as usize];
        if cmp.op != Op::I64_Ne {
            return None;
        }
        let local = if cmp.flags & FLAG_B_CONST != 0
            && cmp.b == 0
            && cmp.flags & FLAG_A_CONST == 0
            && cmp.a < self.n_locals as u64
        {
            cmp.a as u32
        } else if cmp.flags & FLAG_A_CONST != 0
            && cmp.a == 0
            && cmp.flags & FLAG_B_CONST == 0
            && cmp.b < self.n_locals as u64
        {
            cmp.b as u32
        } else {
            return None;
        };

        let write = self.locals[local as usize].last_write;
        if write == 0
            || write != cmp_def
            || self.locals[local as usize].last_write_region != self.region
        {
            return None;
        }
        let mut fusion = self.sub_br_if_update(write - 1, local, true)?;
        fusion.remove_condition = true;
        Some(fusion)
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

    fn emit_materialized_bulk(
        &mut self,
        op: Op,
        packed_indices: u64,
        address_is_64: bool,
    ) -> Result<(), WasmError> {
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
        let mut flags = if address_is_64 { FLAG_ADDR64 } else { 0 };
        let memory_is_zero = match op {
            Op::MemoryFill => packed_indices == 0,
            // `MemoryCopy::b` packs `dst << 32 | src`, so zero proves both
            // memory indices at once.
            Op::MemoryCopy => packed_indices == 0,
            _ => true,
        };
        if !memory_is_zero {
            flags |= FLAG_NO_NATIVE;
        }
        self.emit(op, flags, base, packed_indices, 0);
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
            MEMORY_FILL => {
                let memory = match imm {
                    Immediate::MemoryIndex(m) => *m,
                    _ => 0,
                };
                // A memory32 fill may wait across the local.get/const pushes
                // that prepare an immediately following copy. Memory64 stays
                // on the ordinary shared-executor path.
                if !self.memory_is_64(memory as u64) {
                    self.defer_memory_fill(memory)
                } else {
                    self.emit_materialized_bulk(Op::MemoryFill, memory as u64, true)
                }
            }
            MEMORY_COPY => {
                let (dst_memory, src_memory) = match imm {
                    Immediate::MemoryCopyArgs { dstidx, srcidx } => (*dstidx, *srcidx),
                    _ => (0, 0),
                };
                if self.try_fuse_pending_fill_copy(dst_memory, src_memory)? {
                    return Ok(());
                }
                self.flush_pending_fill();
                let packed = ((dst_memory as u64) << 32) | src_memory as u64;
                self.emit_materialized_bulk(
                    Op::MemoryCopy,
                    packed,
                    self.memory_is_64(dst_memory as u64) || self.memory_is_64(src_memory as u64),
                )
            }
            MEMORY_INIT => {
                let (data, memory) = match imm {
                    Immediate::MemoryInitArgs { dataidx, memidx } => (*dataidx, *memidx),
                    _ => (0, 0),
                };
                let packed = ((memory as u64) << 32) | data as u64;
                self.emit_materialized_bulk(
                    Op::MemoryInit,
                    packed,
                    self.memory_is_64(memory as u64),
                )
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

    fn fb_op(&mut self, fb: OpcodeFB, imm: &Immediate) -> Result<(), WasmError> {
        use OpcodeFB::*;
        let op = match fb {
            REF_TEST | REF_TEST_NULL => Op::RefTest,
            REF_CAST | REF_CAST_NULL => Op::RefCast,
            _ => return Err(unsupported_opcode(WasmOpcode::FB(fb))),
        };
        let ref_type = match imm {
            Immediate::RefType(ValueType::Ref(ref_type)) => *ref_type,
            _ => return Err(desync()),
        };

        let at = self.code.len() as u32;
        let source = self.pop()?;
        let (a, a_const) = self.operand(source, at);
        let flags = if a_const { FLAG_A_CONST } else { 0 };
        let dst = self.temp_slot_used(self.height());
        let index = self.emit(op, flags, a, ref_type.encode_to_u64(), dst);
        self.push_result_temp(index);
        Ok(())
    }

    fn patch_fixups_to_here(&mut self, fixups: &[Fixup]) {
        let here = self.code.len() as u32;
        for &f in fixups {
            match f {
                Fixup::InstrC(i) => self.edit_instr(i, |ins| ins.c = here as u64),
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
        let here = self.code.len() as u64;
        self.edit_instr(i, |ins| ins.c = here);
    }

    /// Lower a decoded load. Kept out of the byte-dispatch loop so the same
    /// implementation serves both its common-op fast lane and generic decode
    /// fallback without duplicating this comparatively large fusion logic.
    #[inline(never)]
    fn lower_load(&mut self, lop: Op, memidx: u32, raw_offset: u64) -> Result<(), WasmError> {
        let addr64 = self.memory_is_64(memidx as u64);
        // A memory64 offset can use all 64 bits, leaving no room for the
        // index beside it; such a cell carries a side-table index instead.
        // It is always a slow cell, so nothing native reads the packed form.
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
        // Address-add fusion: a single-use, just-emitted i32.add producing
        // this address folds into the load.
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
                self.replace_instr(
                    def,
                    Instr {
                        op: lop,
                        flags: afl | FLAG_FUSED,
                        head_pad: 0,
                        a: a1,
                        b: offset,
                        c: a2 << 32 | dst,
                    },
                );
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
            if memidx != 0 {
                flags |= FLAG_NO_NATIVE;
            }
            let dst = self.temp_slot_used(self.height());
            let idx = self.emit(lop, flags, a, offset, dst);
            self.push_result_temp(idx);
        }
        Ok(())
    }

    /// Lower a decoded store; see [`Predecoder::lower_load`].
    #[inline(never)]
    fn lower_store(&mut self, sop: Op, memidx: u32, raw_offset: u64) -> Result<(), WasmError> {
        let addr64 = self.memory_is_64(memidx as u64);
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
        let mut flags = self.acc_operand(v, FLAG_B_ACC, operand_is_float(sop, true));
        if b_const {
            flags |= FLAG_B_CONST;
        }
        let ad = self.pop()?;
        let (a, a_const) = self.operand(ad, at);
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
                self.replace_instr(
                    def,
                    Instr {
                        op: sop,
                        flags: flags | afl | FLAG_FUSED,
                        head_pad: 0,
                        a: a1,
                        b,
                        c: a2 << 32 | offset,
                    },
                );
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
            if memidx != 0 {
                flags |= FLAG_NO_NATIVE;
            }
            self.emit(sop, flags, a, b, offset);
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum FastLowering {
    Fallback,
    LocalGet,
    LocalSet,
    LocalTee,
    I32Const,
    I64Const,
    F32Const,
    F64Const,
    Value1(Op),
    Value2(Op),
    Load(Op),
    Store(Op),
}

macro_rules! set_simple_lowerings {
    ($table:ident, $kind:ident, $($wasm:ident => $lowered:ident),+ $(,)?) => {
        $($table[Opcode::$wasm as usize] = FastLowering::$kind(Op::$lowered);)+
    };
}

/// Byte-indexed lowering for the common routing, numeric and memory opcodes.
/// The predecoder probes this before constructing a generic `DecodedOp`; the
/// same table remains the generic fallback's authoritative opcode-to-op map.
const FAST_LOWERINGS: [FastLowering; 256] = {
    let mut table = [FastLowering::Fallback; 256];
    table[Opcode::LOCAL_GET as usize] = FastLowering::LocalGet;
    table[Opcode::LOCAL_SET as usize] = FastLowering::LocalSet;
    table[Opcode::LOCAL_TEE as usize] = FastLowering::LocalTee;
    table[Opcode::I32_CONST as usize] = FastLowering::I32Const;
    table[Opcode::I64_CONST as usize] = FastLowering::I64Const;
    table[Opcode::F32_CONST as usize] = FastLowering::F32Const;
    table[Opcode::F64_CONST as usize] = FastLowering::F64Const;
    set_simple_lowerings!(table, Value2,
        I32_ADD => I32_Add, I32_SUB => I32_Sub, I32_MUL => I32_Mul,
        I32_DIV_S => I32_DivS, I32_DIV_U => I32_DivU, I32_REM_S => I32_RemS,
        I32_REM_U => I32_RemU, I32_AND => I32_And, I32_OR => I32_Or,
        I32_XOR => I32_Xor, I32_SHL => I32_Shl, I32_SHR_S => I32_ShrS,
        I32_SHR_U => I32_ShrU, I32_ROTL => I32_Rotl, I32_ROTR => I32_Rotr,
        I32_EQ => I32_Eq, I32_NE => I32_Ne, I32_LT_S => I32_LtS,
        I32_LT_U => I32_LtU, I32_GT_S => I32_GtS, I32_GT_U => I32_GtU,
        I32_LE_S => I32_LeS, I32_LE_U => I32_LeU, I32_GE_S => I32_GeS,
        I32_GE_U => I32_GeU, I64_ADD => I64_Add, I64_SUB => I64_Sub,
        I64_MUL => I64_Mul, I64_DIV_S => I64_DivS, I64_DIV_U => I64_DivU,
        I64_REM_S => I64_RemS, I64_REM_U => I64_RemU, I64_AND => I64_And,
        I64_OR => I64_Or, I64_XOR => I64_Xor, I64_SHL => I64_Shl,
        I64_SHR_S => I64_ShrS, I64_SHR_U => I64_ShrU, I64_ROTL => I64_Rotl,
        I64_ROTR => I64_Rotr, I64_EQ => I64_Eq, I64_NE => I64_Ne,
        I64_LT_S => I64_LtS, I64_LT_U => I64_LtU, I64_GT_S => I64_GtS,
        I64_GT_U => I64_GtU, I64_LE_S => I64_LeS, I64_LE_U => I64_LeU,
        I64_GE_S => I64_GeS, I64_GE_U => I64_GeU,
        F32_ADD => F32_Add, F32_SUB => F32_Sub, F32_MUL => F32_Mul,
        F32_DIV => F32_Div, F32_MIN => F32_Min, F32_MAX => F32_Max,
        F32_COPYSIGN => F32_Copysign, F32_EQ => F32_Eq, F32_NE => F32_Ne,
        F32_LT => F32_Lt, F32_GT => F32_Gt, F32_LE => F32_Le,
        F32_GE => F32_Ge, F64_ADD => F64_Add, F64_SUB => F64_Sub,
        F64_MUL => F64_Mul, F64_DIV => F64_Div, F64_MIN => F64_Min,
        F64_MAX => F64_Max, F64_COPYSIGN => F64_Copysign, F64_EQ => F64_Eq,
        F64_NE => F64_Ne, F64_LT => F64_Lt, F64_GT => F64_Gt,
        F64_LE => F64_Le, F64_GE => F64_Ge,
    );
    set_simple_lowerings!(table, Value1,
        I32_CLZ => I32_Clz, I32_CTZ => I32_Ctz, I32_POPCNT => I32_Popcnt,
        I32_EXTEND8_S => I32_Extend8S, I32_EXTEND16_S => I32_Extend16S,
        I32_EQZ => I32_Eqz, I64_CLZ => I64_Clz, I64_CTZ => I64_Ctz,
        I64_POPCNT => I64_Popcnt, I64_EXTEND8_S => I64_Extend8S,
        I64_EXTEND16_S => I64_Extend16S, I64_EXTEND32_S => I64_Extend32S,
        I64_EQZ => I64_Eqz, I32_WRAP_I64 => I32_WrapI64,
        I64_EXTEND_I32_S => I64_ExtendI32S, I64_EXTEND_I32_U => I64_ExtendI32U,
        F32_ABS => F32_Abs, F32_NEG => F32_Neg, F32_CEIL => F32_Ceil,
        F32_FLOOR => F32_Floor, F32_TRUNC => F32_Trunc,
        F32_NEAREST => F32_Nearest, F32_SQRT => F32_Sqrt,
        F64_ABS => F64_Abs, F64_NEG => F64_Neg, F64_CEIL => F64_Ceil,
        F64_FLOOR => F64_Floor, F64_TRUNC => F64_Trunc,
        F64_NEAREST => F64_Nearest, F64_SQRT => F64_Sqrt,
        I32_TRUNC_F32_S => I32_TruncF32S, I32_TRUNC_F32_U => I32_TruncF32U,
        I32_TRUNC_F64_S => I32_TruncF64S, I32_TRUNC_F64_U => I32_TruncF64U,
        I64_TRUNC_F32_S => I64_TruncF32S, I64_TRUNC_F32_U => I64_TruncF32U,
        I64_TRUNC_F64_S => I64_TruncF64S, I64_TRUNC_F64_U => I64_TruncF64U,
        F32_CONVERT_I32_S => F32_ConvertI32S, F32_CONVERT_I32_U => F32_ConvertI32U,
        F32_CONVERT_I64_S => F32_ConvertI64S, F32_CONVERT_I64_U => F32_ConvertI64U,
        F32_DEMOTE_F64 => F32_DemoteF64, F64_CONVERT_I32_S => F64_ConvertI32S,
        F64_CONVERT_I32_U => F64_ConvertI32U, F64_CONVERT_I64_S => F64_ConvertI64S,
        F64_CONVERT_I64_U => F64_ConvertI64U, F64_PROMOTE_F32 => F64_PromoteF32,
        I32_REINTERPRET_F32 => I32_ReinterpretF32,
        I64_REINTERPRET_F64 => I64_ReinterpretF64,
        F32_REINTERPRET_I32 => F32_ReinterpretI32,
        F64_REINTERPRET_I64 => F64_ReinterpretI64,
    );
    set_simple_lowerings!(table, Load,
        I32_LOAD => I32_Load, I64_LOAD => I64_Load, F32_LOAD => F32_Load,
        F64_LOAD => F64_Load, I32_LOAD8_S => I32_Load8S,
        I32_LOAD8_U => I32_Load8U, I32_LOAD16_S => I32_Load16S,
        I32_LOAD16_U => I32_Load16U, I64_LOAD8_S => I64_Load8S,
        I64_LOAD8_U => I64_Load8U, I64_LOAD16_S => I64_Load16S,
        I64_LOAD16_U => I64_Load16U, I64_LOAD32_S => I64_Load32S,
        I64_LOAD32_U => I64_Load32U,
    );
    set_simple_lowerings!(table, Store,
        I32_STORE => I32_Store, I64_STORE => I64_Store, F32_STORE => F32_Store,
        F64_STORE => F64_Store, I32_STORE8 => I32_Store8,
        I32_STORE16 => I32_Store16, I64_STORE8 => I64_Store8,
        I64_STORE16 => I64_Store16, I64_STORE32 => I64_Store32,
    );
    table
};

/// Trivially-copy result of a successful common-op probe. `imm0` is the
/// scalar immediate (or raw memory offset), and `imm1` is the memory index.
/// No field owns memory, so overwriting this slot never runs drop glue.
#[derive(Clone, Copy)]
struct FastDecoded {
    lowering: FastLowering,
    imm0: u64,
    imm1: u32,
    consumed: usize,
}

impl FastDecoded {
    const EMPTY: Self = Self {
        lowering: FastLowering::Fallback,
        imm0: 0,
        imm1: 0,
        consumed: 0,
    };
}

#[inline]
fn probe_u32(bytes: &[u8], cursor: &mut usize) -> Option<u32> {
    let tail = bytes.get(*cursor..)?;
    let &first = tail.first()?;
    if first & 0x80 == 0 {
        *cursor += 1;
        return Some(first as u32);
    }
    let (value, consumed) = leb128::read_leb128_u32(tail).ok()?;
    *cursor += consumed;
    Some(value)
}

#[inline]
fn probe_u64(bytes: &[u8], cursor: &mut usize) -> Option<u64> {
    let tail = bytes.get(*cursor..)?;
    let &first = tail.first()?;
    if first & 0x80 == 0 {
        *cursor += 1;
        return Some(first as u64);
    }
    let (value, consumed) = leb128::read_leb128_u64(tail).ok()?;
    *cursor += consumed;
    Some(value)
}

#[inline]
fn probe_i32(bytes: &[u8], cursor: &mut usize) -> Option<i32> {
    let tail = bytes.get(*cursor..)?;
    let &first = tail.first()?;
    if first & 0x80 == 0 {
        *cursor += 1;
        let value = if first & 0x40 != 0 {
            (first as i32) | !0x7f
        } else {
            first as i32
        };
        return Some(value);
    }
    let (value, consumed) = leb128::read_leb128_i32(tail).ok()?;
    *cursor += consumed;
    Some(value)
}

#[inline]
fn probe_i64(bytes: &[u8], cursor: &mut usize) -> Option<i64> {
    let tail = bytes.get(*cursor..)?;
    let &first = tail.first()?;
    if first & 0x80 == 0 {
        *cursor += 1;
        let value = if first & 0x40 != 0 {
            (first as i64) | !0x7f
        } else {
            first as i64
        };
        return Some(value);
    }
    let (value, consumed) = leb128::read_leb128_i64(tail).ok()?;
    *cursor += consumed;
    Some(value)
}

#[inline]
fn probe_fixed<const N: usize>(bytes: &[u8], cursor: &mut usize) -> Option<[u8; N]> {
    let end = cursor.checked_add(N)?;
    let value = bytes.get(*cursor..end)?.try_into().ok()?;
    *cursor = end;
    Some(value)
}

/// Probe one common opcode without mutating the decoder cursor. A malformed
/// immediate deliberately reports a miss: the unchanged generic fallback then
/// produces the same public error it did before this fast lane existed.
#[inline(never)]
fn probe_fast(bytes: &[u8], decoded: &mut FastDecoded) -> bool {
    let Some(&opcode) = bytes.first() else {
        return false;
    };
    let lowering = FAST_LOWERINGS[opcode as usize];
    let mut cursor = 1;
    let (imm0, imm1) = match lowering {
        FastLowering::Fallback => return false,
        FastLowering::LocalGet | FastLowering::LocalSet | FastLowering::LocalTee => {
            let Some(index) = probe_u32(bytes, &mut cursor) else {
                return false;
            };
            (index as u64, 0)
        }
        FastLowering::I32Const => {
            let Some(value) = probe_i32(bytes, &mut cursor) else {
                return false;
            };
            (value as u32 as u64, 0)
        }
        FastLowering::I64Const => {
            let Some(value) = probe_i64(bytes, &mut cursor) else {
                return false;
            };
            (value as u64, 0)
        }
        FastLowering::F32Const => {
            let Some(value) = probe_fixed::<4>(bytes, &mut cursor) else {
                return false;
            };
            (u32::from_le_bytes(value) as u64, 0)
        }
        FastLowering::F64Const => {
            let Some(value) = probe_fixed::<8>(bytes, &mut cursor) else {
                return false;
            };
            (u64::from_le_bytes(value), 0)
        }
        FastLowering::Load(_) | FastLowering::Store(_) => {
            let Some(align_flag) = probe_u32(bytes, &mut cursor) else {
                return false;
            };
            let (align, memidx) = if align_flag < 64 {
                (align_flag, 0)
            } else {
                let Some(memidx) = probe_u32(bytes, &mut cursor) else {
                    return false;
                };
                (align_flag - 64, memidx)
            };
            if align >= 32 {
                return false;
            }
            let Some(offset) = probe_u64(bytes, &mut cursor) else {
                return false;
            };
            (offset, memidx)
        }
        FastLowering::Value1(_) | FastLowering::Value2(_) => (0, 0),
    };
    *decoded = FastDecoded {
        lowering,
        imm0,
        imm1,
        consumed: cursor,
    };
    true
}

impl OpcodeHandler for Predecoder<'_, '_> {
    fn on_decode_begin(&mut self) -> Result<(), WasmError> {
        Ok(())
    }

    fn on_stream<'x, 'y, 'z>(
        &mut self,
        stream: &mut OpStream<'x, 'y, 'z>,
    ) -> Result<(), WasmError> {
        let mut fast = FastDecoded::EMPTY;
        loop {
            if probe_fast(stream.predecode_bytes(), &mut fast) {
                // Probe first, commit second: every miss leaves the generic
                // decoder at the exact byte where it started.
                stream.consume_predecoded(fast.consumed);
                if self.pending_fill.is_some()
                    && !matches!(
                        fast.lowering,
                        FastLowering::LocalGet | FastLowering::I32Const | FastLowering::I64Const
                    )
                {
                    self.flush_pending_fill();
                }
                if self.dead {
                    continue;
                }
                match fast.lowering {
                    FastLowering::LocalGet => {
                        self.stack.push(Desc::Local(fast.imm0 as u32));
                    }
                    FastLowering::LocalSet => self.local_set(fast.imm0 as u32, false)?,
                    FastLowering::LocalTee => self.local_set(fast.imm0 as u32, true)?,
                    FastLowering::I32Const
                    | FastLowering::I64Const
                    | FastLowering::F32Const
                    | FastLowering::F64Const => self.stack.push(Desc::ConstV(fast.imm0)),
                    FastLowering::Value1(op) => self.value_op(op, 1)?,
                    FastLowering::Value2(op) => self.value_op(op, 2)?,
                    FastLowering::Load(op) => {
                        self.lower_load(op, fast.imm1, fast.imm0)?;
                    }
                    FastLowering::Store(op) => {
                        self.lower_store(op, fast.imm1, fast.imm0)?;
                    }
                    FastLowering::Fallback => unreachable!(),
                }
                continue;
            }

            let Some(op) = stream.next()? else {
                break;
            };
            if self.pending_fill.is_some()
                && !matches!(
                    op.wasm_op,
                    WasmOpcode::OP(Opcode::LOCAL_GET | Opcode::I32_CONST | Opcode::I64_CONST)
                        | WasmOpcode::FC(OpcodeFC::MEMORY_COPY)
                )
            {
                self.flush_pending_fill();
            }
            let o = match op.wasm_op {
                WasmOpcode::OP(o) => o,
                WasmOpcode::FC(fc) => {
                    if self.dead {
                        continue;
                    }
                    self.fc_op(fc, &op.imm)?;
                    continue;
                }
                WasmOpcode::FB(fb) => {
                    if self.dead {
                        continue;
                    }
                    self.fb_op(fb, &op.imm)?;
                    continue;
                }
                other => return Err(unsupported_opcode(other)),
            };
            // Borrowed, not cloned: the decoded op outlives the iteration
            // and nothing in the body touches the stream again.
            let imm = &op.imm;
            if o == Opcode::TRY_TABLE {
                self.force_canonical_loop_homes = true;
            }

            // Dead-code handling: skip, but keep frame nesting and merge
            // reachability bookkeeping.
            if self.dead {
                match o {
                    Opcode::BLOCK | Opcode::LOOP | Opcode::IF => {
                        let (p, r) = match imm {
                            Immediate::Block(bt) => block_arity(self.types, bt)?,
                            _ => (0, 0),
                        };
                        let base = self.height().saturating_sub(p);
                        let is_loop = o == Opcode::LOOP;
                        let loop_id = if is_loop {
                            self.allocate_loop_id()
                        } else {
                            NO_DEF
                        };
                        self.frames.push(CtlFrame {
                            base,
                            params: p,
                            results: r,
                            is_loop,
                            param_locals: Vec::new(),
                            loop_id,
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
                        let (bt, catches) = match imm {
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
                            param_locals: Vec::new(),
                            loop_id: NO_DEF,
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
                    let (p, r) = match imm {
                        Immediate::Block(bt) => block_arity(self.types, bt)?,
                        _ => (0, 0),
                    };
                    let base = self.height().checked_sub(p).ok_or_else(desync)?;
                    let is_loop = o == Opcode::LOOP;
                    let loop_id = if is_loop {
                        self.allocate_loop_id()
                    } else {
                        NO_DEF
                    };
                    let mut param_locals = Vec::new();
                    if is_loop {
                        // Loop header is a merge point (back edges arrive).
                        // Values below the parameters must be canonical
                        // before the first iteration: a local write in the
                        // loop cannot be allowed to overwrite an outer
                        // pending `local.get` on later iterations.
                        for i in 0..base as usize {
                            self.materialize_at(i);
                        }
                        param_locals = vec![NO_DEF; p as usize];
                        if !self.loop_uses_canonical_homes(loop_id) {
                            // A unique pending local can be the loop
                            // parameter's home directly. Duplicates stay in
                            // separate temp slots because their back-edge
                            // values are allowed to diverge.
                            self.side_scratch.loop_candidates.clear();
                            for i in 0..p as usize {
                                if let Desc::Local(local) = self.stack[base as usize + i] {
                                    self.side_scratch.loop_candidates.push((local, i));
                                }
                            }
                            self.side_scratch
                                .loop_candidates
                                .sort_unstable_by_key(|&(local, _)| local);
                            let mut first = 0;
                            while first < self.side_scratch.loop_candidates.len() {
                                let mut end = first + 1;
                                while end < self.side_scratch.loop_candidates.len()
                                    && self.side_scratch.loop_candidates[end].0
                                        == self.side_scratch.loop_candidates[first].0
                                {
                                    end += 1;
                                }
                                if end == first + 1 {
                                    let (local, param) = self.side_scratch.loop_candidates[first];
                                    param_locals[param] = local;
                                }
                                first = end;
                            }
                        }
                        for i in 0..p as usize {
                            if param_locals[i] == NO_DEF {
                                self.materialize_at(base as usize + i);
                            }
                        }
                        self.bump_region();
                    }
                    let header = self.code.len() as u32;
                    self.frames.push(CtlFrame {
                        base,
                        params: p,
                        results: r,
                        is_loop,
                        param_locals,
                        loop_id,
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
                    let (p, r) = match imm {
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
                        self.edit_instr(def, |ins| {
                            ins.op = op;
                            ins.c = FIXUP;
                        });
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
                        param_locals: Vec::new(),
                        loop_id: NO_DEF,
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
                    let d = match *imm {
                        Immediate::LabelIndex(d) => d,
                        _ => return Err(desync()),
                    };
                    self.mark_branch_target(d);
                    self.prepare_branch(d)?;
                    self.side_scratch.branch_copies.clear();
                    let plan = self.plan_branch(d)?;
                    self.emit_branch_copies(&plan);
                    self.emit_branch(Op::Br, 0, 0, d)?;
                    self.bump_region();
                    self.dead = true;
                }
                Opcode::BR_IF => {
                    let d = match *imm {
                        Immediate::LabelIndex(d) => d,
                        _ => return Err(desync()),
                    };
                    self.mark_branch_target(d);
                    let cond = self.pop()?;
                    self.prepare_branch(d)?;

                    self.side_scratch.branch_copies.clear();
                    let plan = self.plan_branch(d)?;
                    let needs_moves = !plan.copies.is_empty();

                    // A loop commonly updates an induction local and uses
                    // the new value as its condition:
                    //
                    //     local.get $n; i32.const 1; i32.sub
                    //     local.tee $n; local.get $n; br_if $loop
                    //
                    // Destination folding has already made the update read
                    // and write `$n`. When it is still the immediately
                    // preceding writer, make that one cell perform the
                    // write-through and branch as well. The i64 form looks
                    // through the required `i64.ne $n, 0` condition cell.
                    let sub_branch = if !needs_moves && (d as usize) < self.frames.len() {
                        self.direct_sub_br_if(cond)
                            .or_else(|| self.i64_sub_br_if(cond))
                    } else {
                        None
                    };

                    // Fixed combo: a fusible condition producer becomes the
                    // branch itself (the guard form uses the inverted
                    // sense). A branch to the function label is a return —
                    // not a branch target — so it stays unfused.
                    let fused = if !needs_moves && (d as usize) >= self.frames.len() {
                        None
                    } else {
                        self.rewritable_producer(cond).and_then(|def| {
                            fuse_br_if(self.code[def as usize].op, needs_moves)
                                .map(|fusion| (def, fusion))
                        })
                    };

                    if let Some(fusion) = sub_branch {
                        if fusion.remove_condition {
                            debug_assert_eq!(
                                self.code.len() as u32,
                                fusion.def + 2,
                                "the i64 zero comparison must immediately follow its update"
                            );
                            self.pop_instr();
                            self.locals[fusion.local as usize].last_read = fusion.def + 1;
                        }
                        self.edit_instr(fusion.def, |ins| {
                            ins.op = fusion.op;
                            ins.b = fusion.b;
                            // `a` remains the authoritative destination slot;
                            // do not make the fused control handler depend on
                            // transient accumulator residency for that operand.
                            ins.flags &= !(FLAG_A_ACC | FLAG_DST_ACC);
                        });
                        self.retarget_branch(fusion.def, d);
                    } else if let Some((def, fusion)) = fused {
                        self.edit_instr(def, |ins| {
                            ins.op = fusion.op;
                            if let Some(b) = fusion.b_const {
                                ins.b = b;
                                ins.flags = (ins.flags & !FLAG_B_ACC) | FLAG_B_CONST;
                            }
                        });
                        if !needs_moves {
                            self.retarget_branch(def, d);
                        } else {
                            // Guard form: the fused inverted branch skips
                            // the taken path's moves + jump.
                            self.edit_instr(def, |ins| ins.c = FIXUP);
                            self.emit_branch_copies(&plan);
                            self.emit_branch(Op::Br, 0, 0, d)?;
                            let here = self.code.len() as u64;
                            self.edit_instr(def, |ins| ins.c = here);
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
                            self.emit_branch_copies(&plan);
                            self.emit_branch(Op::Br, 0, 0, d)?;
                            let here = self.code.len() as u64;
                            self.edit_instr(skip, |ins| ins.c = here);
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
                    let fidx = match *imm {
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
                    let (tidx, table) = match *imm {
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
                    if let Immediate::LocalIndex(i) = *imm {
                        self.stack.push(Desc::Local(i));
                    }
                }
                Opcode::LOCAL_SET => {
                    if let Immediate::LocalIndex(i) = *imm {
                        self.local_set(i, false)?;
                    }
                }
                Opcode::LOCAL_TEE => {
                    if let Immediate::LocalIndex(i) = *imm {
                        self.local_set(i, true)?;
                    }
                }
                Opcode::I32_CONST => {
                    if let Immediate::I32(v) = *imm {
                        self.stack.push(Desc::ConstV(v as u32 as u64));
                    }
                }
                Opcode::I64_CONST => {
                    if let Immediate::I64(v) = *imm {
                        self.stack.push(Desc::ConstV(v as u64));
                    }
                }
                Opcode::REF_NULL => {
                    self.stack.push(Desc::ConstV(ref_to_machine_raw(
                        RefValue::null(),
                        SLOT_GP_UNIT_BYTES,
                    )));
                }
                Opcode::REF_FUNC => {
                    if let Immediate::FunctionIndex(i) = *imm {
                        let handle =
                            self.function_handles
                                .get(i as usize)
                                .copied()
                                .ok_or_else(|| {
                                    WasmError::invalid("ref.func: function identity missing")
                                })?;
                        self.stack
                            .push(Desc::ConstV(ref_to_machine_raw(handle, SLOT_GP_UNIT_BYTES)));
                    }
                }
                Opcode::REF_IS_NULL => {
                    self.value_op(Op::RefIsNull, 1)?;
                }
                // Named rather than left to the generic fallthrough: this is
                // a whole feature the engine lacks, not one stray opcode, and
                // saying so is what tells a reader where the boundary is.
                Opcode::TRY_TABLE => {
                    let (bt, catches) = match imm {
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
                        param_locals: Vec::new(),
                        loop_id: NO_DEF,
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
                    let tag_idx = match *imm {
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
                    let d = match *imm {
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
                    self.side_scratch.branch_copies.clear();
                    let plan = self.plan_branch(d)?;
                    // Guard, moves, jump. The guard skips the branch on the
                    // sense that does NOT take it.
                    let guard = if on_null { Op::BrIfNot } else { Op::BrIf };
                    let skip = self.emit(guard, 0, cond, 0, FIXUP);
                    self.emit_branch_copies(&plan);
                    self.emit_branch(Op::Br, 0, 0, d)?;
                    let here = self.code.len() as u64;
                    self.edit_instr(skip, |ins| ins.c = here);
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
                    let tidx = match *imm {
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
                    let t = match *imm {
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
                    let t = match *imm {
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
                    if let Immediate::F32(v) = *imm {
                        self.stack.push(Desc::ConstV(v.to_bits() as u64));
                    }
                }
                Opcode::F64_CONST => {
                    if let Immediate::F64(v) = *imm {
                        self.stack.push(Desc::ConstV(v.to_bits()));
                    }
                }
                Opcode::GLOBAL_GET => {
                    let g = match *imm {
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
                    let g = match *imm {
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
                    let m = match *imm {
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
                    let m = match *imm {
                        Immediate::MemoryIndex(m) => m as u64,
                        _ => 0,
                    };
                    let dst = self.temp_slot_used(self.height());
                    let idx = self.emit(Op::MemoryGrow, flags, a, m, dst);
                    self.push_result_temp(idx);
                }
                Opcode::BR_TABLE => {
                    let (labels, default) = match imm {
                        Immediate::BrLabels(labels, default) => (labels.as_slice(), *default),
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
                    let mut depths = mem::take(&mut self.side_scratch.br_depths);
                    depths.clear();
                    depths.extend_from_slice(labels);
                    depths.push(default);
                    self.prepare_br_table(&depths)?;
                    // Acc marking AFTER materialization: the adjacency
                    // guard re-evaluates against the post-materialization
                    // length, so the mark only lands when no mov sits
                    // between the producer and the BrTable cell (movs
                    // clobber the accumulator — every value handler
                    // computes into it).
                    flags |= self.acc_operand(d0, FLAG_A_ACC, false);
                    let tbl = self.br_tables.len() as u32;

                    // Build one transfer per target depth from the shared
                    // source tuple. Copy-free frame targets stay direct.
                    // Function returns and targets with copies use one
                    // landing pad per target, so repeated table entries share
                    // both planning work and the emitted move sequence.
                    let n = self.frames.len();
                    self.side_scratch.branch_copies.clear();
                    let mut plans = mem::take(&mut self.side_scratch.branch_plans);
                    plans.clear();
                    let mut plan_for_target = mem::take(&mut self.side_scratch.br_plan_for_target);
                    plan_for_target.clear();
                    plan_for_target.resize(n + 1, None);
                    let mut entry_plans = mem::take(&mut self.side_scratch.br_entry_plans);
                    entry_plans.clear();
                    for &depth in depths.iter() {
                        let target = (depth as usize).min(n);
                        let plan = if let Some(plan) = plan_for_target[target] {
                            plan
                        } else {
                            let plan = plans.len() as u32;
                            plans.push(self.plan_branch(depth)?);
                            plan_for_target[target] = Some(plan);
                            plan
                        };
                        entry_plans.push(plan);
                    }

                    let mut pad_plans = mem::take(&mut self.side_scratch.br_pad_plans);
                    pad_plans.clear();
                    let mut pad_for_plan = mem::take(&mut self.side_scratch.br_pad_for_plan);
                    pad_for_plan.clear();
                    pad_for_plan.resize(plans.len(), None);
                    let mut entry_pads = mem::take(&mut self.side_scratch.br_entry_pads);
                    entry_pads.clear();
                    let mut entries = Vec::with_capacity(entry_plans.len());
                    for &plan_index in &entry_plans {
                        let plan = &plans[plan_index as usize];
                        let needs_pad = (plan.depth as usize) >= n || !plan.copies.is_empty();
                        if needs_pad {
                            let pad = if let Some(pad) = pad_for_plan[plan_index as usize] {
                                pad
                            } else {
                                let pad = pad_plans.len() as u32;
                                pad_plans.push(plan_index);
                                pad_for_plan[plan_index as usize] = Some(pad);
                                pad
                            };
                            entries.push(u32::MAX);
                            entry_pads.push(Some(pad));
                            continue;
                        }
                        entry_pads.push(None);
                        let i = n - 1 - plan.depth as usize;
                        if self.frames[i].is_loop {
                            entries.push(self.frames[i].header);
                        } else {
                            let entry = entries.len() as u32;
                            self.frames[i].fixups.push(Fixup::Table { tbl, entry });
                            entries.push(u32::MAX);
                        }
                    }
                    self.emit(Op::BrTable, flags, a, 0, tbl as u64);

                    let mut pad_addresses = mem::take(&mut self.side_scratch.br_pad_addresses);
                    pad_addresses.clear();
                    for &plan_index in &pad_plans {
                        let pad = self.code.len() as u32;
                        pad_addresses.push(pad);
                        let plan = BranchPlan {
                            depth: plans[plan_index as usize].depth,
                            copies: plans[plan_index as usize].copies.clone(),
                        };
                        self.emit_branch_copies(&plan);
                        self.emit_branch(Op::Br, 0, 0, plan.depth)?;
                    }
                    for (entry, entry_pad) in entries.iter_mut().zip(entry_pads.iter().copied()) {
                        if let Some(pad) = entry_pad {
                            *entry = pad_addresses[pad as usize];
                        }
                    }

                    self.br_tables.push(entries);
                    self.side_scratch.br_depths = depths;
                    self.side_scratch.branch_plans = plans;
                    self.side_scratch.br_plan_for_target = plan_for_target;
                    self.side_scratch.br_entry_plans = entry_plans;
                    self.side_scratch.br_pad_plans = pad_plans;
                    self.side_scratch.br_pad_for_plan = pad_for_plan;
                    self.side_scratch.br_entry_pads = entry_pads;
                    self.side_scratch.br_pad_addresses = pad_addresses;
                    self.bump_region();
                    self.dead = true;
                }
                o => match FAST_LOWERINGS[o as usize] {
                    FastLowering::Value2(vop) => self.value_op(vop, 2)?,
                    FastLowering::Value1(vop) => self.value_op(vop, 1)?,
                    FastLowering::Load(lop) => match imm {
                        Immediate::MemArg { offset, memidx, .. } => {
                            self.lower_load(lop, *memidx, *offset)?;
                        }
                        _ => return Err(desync()),
                    },
                    FastLowering::Store(sop) => match imm {
                        Immediate::MemArg { offset, memidx, .. } => {
                            self.lower_store(sop, *memidx, *offset)?;
                        }
                        _ => return Err(desync()),
                    },
                    FastLowering::Fallback
                    | FastLowering::LocalGet
                    | FastLowering::LocalSet
                    | FastLowering::LocalTee
                    | FastLowering::I32Const
                    | FastLowering::I64Const
                    | FastLowering::F32Const
                    | FastLowering::F64Const => return Err(unsupported()),
                },
            }
        }
        Ok(())
    }

    fn on_decode_end(&mut self) -> Result<(), WasmError> {
        if !self.frames.is_empty() {
            return Err(desync());
        }
        self.flush_pending_fill();
        self.finish_return_landing();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::layout::native_guard;
    use super::*;
    use crate::module::Module;
    use crate::vm::interpreter::instr::{op_from_index, N_OPS};
    use std::fmt::Write as _;
    use std::format;
    use std::string::String as StdString;
    use std::vec::Vec as StdVec;

    fn assert_pinned_fields_match(code: &[Instr], locals: &[LocalState], context: &str) {
        let actual = PackedPinned::new(select_pinned(locals)).expand();
        let expected = super::super::engine::select_pinned_reference(code, locals.len() as u32);
        assert_eq!(actual.l0, expected.l0, "{context}: l0");
        assert_eq!(actual.l1, expected.l1, "{context}: l1");
        assert_eq!(actual.l0_float, expected.l0_float, "{context}: l0_float");
        assert_eq!(actual.l1_float, expected.l1_float, "{context}: l1_float");
    }

    fn generated_field(state: u64, n_locals: u32) -> u64 {
        match state & 7 {
            0 => 0,
            1 => n_locals.saturating_sub(1) as u64,
            2 => n_locals as u64,
            3 => 8191,
            4 => 8192,
            _ => state.rotate_left(17) & 0x3fff,
        }
    }

    #[test]
    fn incremental_pinned_census_matches_reference_for_generated_mutations() {
        // Begin with the layouts which bypass ordinary slot_fields handling,
        // plus float readers/writers and the 16-bit call-record boundary.
        let n_locals = 8194u32;
        let mut locals = vec![LocalState::default(); n_locals as usize];
        let mut code = StdVec::new();
        let edge_cells = [
            Instr::new(Op::MovPair, 0, 0, 1, 8191u64 << 32 | 8192),
            Instr::new(Op::Select, 0, 2, 3, 4u64 << 32 | 5),
            Instr::new(Op::I32_SubBrIf, FLAG_B_CONST, 6, 1, 0),
            Instr::new(Op::I64_SubBrIf, 0, 7, 8, 0),
            Instr::new(Op::F32_Add, 0, 9, 10, 11),
            Instr::new(Op::F64_Store, 0, 12, 13, 0),
            Instr::new(Op::I32_Load, FLAG_FUSED, 14, 0, 15u64 << 32 | 16),
            Instr::new(Op::I32_Store, FLAG_FUSED, 17, 18, 19u64 << 32),
        ];
        for (i, ins) in edge_cells.into_iter().enumerate() {
            update_pin_census(&mut locals, ins, true);
            code.push(ins);
            assert_pinned_fields_match(&code, &locals, &format!("edge push {i}"));
        }

        // Exercise the exact edit protocol: remove the old contribution,
        // install the replacement, then delete a cell entirely.
        let replacements = [
            Instr::new(Op::MovSlot, 0, 8192, 0, 8191),
            Instr::new(Op::F32_Load, FLAG_FUSED, 10, 0, 9u64 << 32 | 11),
            Instr::new(Op::Select, FLAG_A_CONST, 0, 13, 7u64 << 32 | 6),
        ];
        for (i, replacement) in replacements.into_iter().enumerate() {
            let old = code[i];
            update_pin_census(&mut locals, old, false);
            update_pin_census(&mut locals, replacement, true);
            code[i] = replacement;
            assert_pinned_fields_match(&code, &locals, &format!("edge replace {i}"));
        }
        let removed = code.pop().unwrap();
        update_pin_census(&mut locals, removed, false);
        assert_pinned_fields_match(&code, &locals, "edge pop");

        // The generated streams include forged field/flag combinations on
        // purpose. The old linker accepted any Instr payload, so metadata
        // maintenance must preserve its behavior outside normal predecode
        // shapes as well as for valid Wasm.
        let local_counts = [0u32, 1, 2, 7, 33, 8194];
        let mut random = 0x4d59_5df4_d0f3_3173u64;
        for stream in 0..256usize {
            let n_locals = local_counts[stream % local_counts.len()];
            let mut locals = vec![LocalState::default(); n_locals as usize];
            let mut code = StdVec::new();
            for step in 0..96usize {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                let op = op_from_index((random as usize) % N_OPS);
                let flags = ((random >> 19) as u16) & ((FLAG_NO_NATIVE << 1) - 1);
                let a = generated_field(random, n_locals);
                let b = generated_field(random.rotate_left(11), n_locals);
                let lo = generated_field(random.rotate_left(23), n_locals) & 0xffff_ffff;
                let hi = generated_field(random.rotate_left(37), n_locals) & 0xffff_ffff;
                let c = if matches!(op, Op::MovPair | Op::Select) || flags & FLAG_FUSED != 0 {
                    hi << 32 | lo
                } else {
                    generated_field(random.rotate_left(29), n_locals)
                };
                let ins = Instr::new(op, flags, a, b, c);
                match (random >> 61) as u8 {
                    0 if !code.is_empty() => {
                        let index = (random as usize >> 8) % code.len();
                        let old = code[index];
                        update_pin_census(&mut locals, old, false);
                        update_pin_census(&mut locals, ins, true);
                        code[index] = ins;
                    }
                    1 if !code.is_empty() => {
                        let old = code.pop().unwrap();
                        update_pin_census(&mut locals, old, false);
                    }
                    _ => {
                        update_pin_census(&mut locals, ins, true);
                        code.push(ins);
                    }
                }
                if step == 95 {
                    assert_pinned_fields_match(
                        &code,
                        &locals,
                        &format!("generated stream {stream}"),
                    );
                }
            }
        }
    }

    fn predecode_wat_mode(src: &str, func: usize, disable_fast: bool) -> PredecodedFunction {
        let bin: StdVec<u8> = wat::parse_str(src).expect("wat");
        let module = Module::new("t", &bin).expect("module");
        let tag_identities: StdVec<TagIdentity> = module
            .tags()
            .iter()
            .map(|_| TagIdentity::mint_fresh())
            .collect();
        let function_handles: StdVec<RefValue> =
            (0..module.functions().len()).map(RefValue::new).collect();
        let mut code = Vec::new();
        let mut scratch = PredecodeScratch::default();
        let parts = predecode_function_into(
            &module,
            &tag_identities,
            &function_handles,
            func,
            disable_fast,
            &mut code,
            &mut scratch,
        )
        .expect("predecode");
        let code = Rc::new(code);
        parts.finish(Some(&code))
    }

    fn predecode_wat(src: &str, func: usize) -> PredecodedFunction {
        predecode_wat_mode(src, func, false)
    }

    fn predecode_module_wat(src: &str) -> Vec<Option<Rc<PredecodedFunction>>> {
        let bin: StdVec<u8> = wat::parse_str(src).expect("wat");
        let module = Module::new("t", &bin).expect("module");
        let tag_identities: StdVec<TagIdentity> = module
            .tags()
            .iter()
            .map(|_| TagIdentity::mint_fresh())
            .collect();
        let function_handles: StdVec<RefValue> =
            (0..module.functions().len()).map(RefValue::new).collect();
        predecode_functions(&module, &tag_identities, &function_handles, |_, _, _| ())
            .expect("predecode module")
            .0
            .publish_for_test()
    }

    fn ops(f: &PredecodedFunction) -> StdVec<Op> {
        f.code.iter().map(|i| i.op).collect()
    }

    fn assert_same_predecode(fast: &PredecodedFunction, generic: &PredecodedFunction) {
        assert_eq!(fast.code.len(), generic.code.len(), "instruction count");
        for (i, (a, b)) in fast.code.iter().zip(generic.code.iter()).enumerate() {
            assert_eq!(
                (a.op, a.flags, a.a, a.b, a.c),
                (b.op, b.flags, b.a, b.b, b.c),
                "instruction {i}"
            );
        }
        assert_eq!(fast.br_tables, generic.br_tables, "branch tables");
        assert_eq!(fast.wide_memargs, generic.wide_memargs, "wide memargs");
        assert_eq!(fast.frame_slots, generic.frame_slots, "frame slots");
        assert_eq!(fast.n_locals, generic.n_locals, "local count");
        assert_eq!(fast.n_params, generic.n_params, "parameter count");
        assert_eq!(fast.n_results, generic.n_results, "result count");
        assert_eq!(fast.pinned(), generic.pinned(), "pinned locals");
        assert_eq!(
            fast.slow_tail_return, generic.slow_tail_return,
            "slow tail landing"
        );
        assert_eq!(
            fast.exception_sites, generic.exception_sites,
            "exception sites"
        );
        assert_eq!(
            fast.exception_handlers, generic.exception_handlers,
            "exception handlers"
        );
    }

    #[test]
    fn module_predecode_has_one_instruction_backing_for_many_functions() {
        let mut wat = StdString::from("(module");
        for value in 0..64 {
            write!(&mut wat, " (func (result i32) i32.const {value})").unwrap();
        }
        wat.push(')');

        let funcs = predecode_module_wat(&wat);
        let first = funcs[0].as_ref().expect("defined function");
        let arena = Rc::as_ptr(&first.code.arena);
        let arena_base = first.code.arena.as_ptr();
        let mut cursor = 0usize;
        for func in funcs.iter().flatten() {
            assert_eq!(
                Rc::as_ptr(&func.code.arena),
                arena,
                "every defined function must share one Vec<Instr> owner"
            );
            assert_eq!(
                func.code.range.start, cursor,
                "code ranges follow their pads"
            );
            assert_eq!(
                func.code.as_ptr(),
                unsafe { arena_base.add(func.code.range.start) },
                "the published slice points into the original arena buffer"
            );
            assert_eq!(
                func.code.arena[func.code.range.end].op,
                Op::Unreachable,
                "every body owns one prefetch pad"
            );
            cursor = func.code.range.end + 1;
        }
        assert_eq!(cursor, first.code.arena.len());
    }

    #[test]
    fn linked_publication_keeps_metadata_in_per_function_rcs() {
        let funcs = predecode_module_wat(
            r#"(module
                (func (param i32)
                    (block $exit
                        local.get 0
                        br_table $exit $exit))
                (func (param i32)
                    (block $exit
                        local.get 0
                        br_table $exit $exit)))"#,
        );
        let first = funcs[0].as_ref().expect("first defined function");
        let second = funcs[1].as_ref().expect("second defined function");

        assert!(
            !Rc::ptr_eq(first, second),
            "function metadata must retain one Rc allocation per body"
        );
        assert_eq!(first.br_tables.len(), 1);
        assert_eq!(second.br_tables.len(), 1);
        assert_ne!(
            first.br_tables.as_ptr(),
            second.br_tables.as_ptr(),
            "branch-table vectors must remain function-owned"
        );
    }

    #[test]
    fn module_arena_rollback_preserves_neighboring_functions() {
        let wat = r#"(module
            (func (result i32) i32.const 7)
            (func (param $x i32) (param $n i32) (result i32)
                local.get $x
                (loop $unsafe (param i32) (result i32)
                    drop
                    i32.const 42
                    local.get $n
                    i32.const 1
                    i32.sub
                    local.tee $n
                    br_if $unsafe)
                drop
                local.get $x)
            (func (result i64) i64.const 99))"#;
        let funcs = predecode_module_wat(wat);
        for (index, shared) in funcs.iter().enumerate() {
            let shared = shared.as_ref().expect("defined function");
            let standalone = predecode_wat(wat, index);
            assert_same_predecode(shared, &standalone);
        }
        assert_eq!(funcs[0].as_ref().unwrap().code[0].a, 7);
        assert_eq!(funcs[2].as_ref().unwrap().code[0].a, 99);
    }

    #[test]
    fn shared_scratch_matches_fresh_scratch_for_complex_side_metadata() {
        // Production predecode exercises one scratch across all four
        // functions. The comparison path constructs a fresh scratch for each
        // body, preserving the old ownership model as a cfg(test) oracle.
        // Cover rollback, branch tables, exceptions, and self-tail re-entry.
        let wat = r#"(module
            (tag $e (param i32))
            (func (param $x i32) (param $n i32) (result i32)
                local.get $x
                (loop $unsafe (param i32) (result i32)
                    drop
                    i32.const 42
                    local.get $n
                    i32.const 1
                    i32.sub
                    local.tee $n
                    br_if $unsafe)
                drop
                local.get $x)
            (func (param i32) (result i32)
                (block $exit (result i32)
                    local.get 0
                    (loop $again (param i32) (result i32)
                        local.get 0
                        br_table $again $exit)))
            (func (result i32)
                (try_table (result i32) (catch $e 0)
                    i32.const 7
                    throw $e))
            (func $reenter (param i32) (result i32)
                local.get 0
                return_call $reenter))"#;
        let bin: StdVec<u8> = wat::parse_str(wat).expect("wat");
        let module = Module::new("scratch-oracle", &bin).expect("module");
        let tag_identities: StdVec<TagIdentity> = module
            .tags()
            .iter()
            .map(|_| TagIdentity::mint_fresh())
            .collect();
        let function_handles: StdVec<RefValue> =
            (0..module.functions().len()).map(RefValue::new).collect();
        let shared = predecode_functions(&module, &tag_identities, &function_handles, |_, _, _| ())
            .expect("shared predecode")
            .0
            .publish_for_test();
        for (index, shared) in shared.iter().enumerate() {
            let mut code = Vec::new();
            let mut fresh_scratch = PredecodeScratch::default();
            let fresh = predecode_function_into(
                &module,
                &tag_identities,
                &function_handles,
                index,
                false,
                &mut code,
                &mut fresh_scratch,
            )
            .expect("fresh predecode");
            let code = Rc::new(code);
            let fresh = fresh.finish(Some(&code));
            assert_same_predecode(shared.as_ref().expect("defined function"), &fresh);
        }
    }

    #[test]
    fn fourteen_thousand_functions_share_transient_allocation_owners() {
        const FUNCTION_COUNT: usize = 14_000;
        let mut wat = StdString::from("(module");
        for _ in 0..FUNCTION_COUNT {
            // Exercise every core scratch vector and all br_table workspaces.
            // The bodies are deliberately identical, so any capacity growth
            // after body zero would prove that state was not actually reused.
            wat.push_str(
                r#" (func (param i32) (result i32)
                    (block $exit (result i32)
                        local.get 0
                        (loop $again (param i32) (result i32)
                            local.get 0
                            br_table $again $exit)))"#,
            );
        }
        wat.push(')');

        let bin: StdVec<u8> = wat::parse_str(&wat).expect("wat");
        let module = Module::new("scratch-census", &bin).expect("module");
        let function_handles: StdVec<RefValue> =
            (0..module.functions().len()).map(RefValue::new).collect();
        let mut code = Vec::new();
        let mut scratch = PredecodeScratch::default();
        let mut prior = scratch.capacities();
        let mut shared_growth_owners = 0usize;
        let mut legacy_fresh_owners = 0usize;
        let mut owner_count = None;

        for func_index in 0..FUNCTION_COUNT {
            let parts = predecode_function_into(
                &module,
                &[],
                &function_handles,
                func_index,
                false,
                &mut code,
                &mut scratch,
            )
            .expect("predecode");
            drop(parts);

            let capacities = scratch.capacities();
            shared_growth_owners += capacities
                .iter()
                .zip(prior.iter())
                .filter(|(after, before)| after > before)
                .count();
            let allocated_owners = capacities.iter().filter(|&&capacity| capacity != 0).count();
            assert_eq!(
                *owner_count.get_or_insert(allocated_owners),
                allocated_owners,
                "identical bodies must exercise the same transient owners"
            );
            // The pre-change path created these allocation-owning Vecs anew
            // for each body. This is a static owner census, independent of
            // allocator implementation and benchmark timing.
            legacy_fresh_owners += allocated_owners;
            prior = capacities;
        }

        let owner_count = owner_count.expect("at least one function");
        assert_eq!(owner_count, 15, "the fixture must cover every scratch Vec");
        assert_eq!(
            (legacy_fresh_owners, shared_growth_owners),
            (210_000, 15),
            "14k fresh-scratch owner allocations must collapse to one module-scoped growth per owner"
        );
    }

    #[test]
    fn common_byte_fast_lane_matches_generic_decoder() {
        let wat = r#"(module
            (memory 1)
            (func (param i32 i64 f32 f64) (result i32) (local i32)
                local.get 0
                i32.const 123456
                i32.add
                local.tee 4
                drop
                i32.const 0
                local.get 4
                i32.store offset=70000
                i32.const 0
                i32.load offset=70000
                drop
                local.get 1
                i64.const -987654321
                i64.xor
                drop
                local.get 2
                f32.const 1.25
                f32.add
                drop
                local.get 3
                f64.const -2.5
                f64.mul
                drop
                block
                    local.get 4
                    i32.eqz
                    br_if 0
                end
                local.get 4))"#;
        let fast = predecode_wat_mode(wat, 0, false);
        let generic = predecode_wat_mode(wat, 0, true);
        assert_same_predecode(&fast, &generic);
    }

    #[test]
    fn malformed_fast_immediate_is_a_non_consuming_miss() {
        let mut decoded = FastDecoded::EMPTY;
        assert!(!probe_fast(&[Opcode::LOCAL_GET as u8, 0x80], &mut decoded));
        assert!(!probe_fast(&[Opcode::I64_CONST as u8, 0x80], &mut decoded));
        assert_eq!(decoded.consumed, 0);
    }

    #[test]
    fn predecode_caches_every_memory_guard_failure() {
        let wat = r#"(module
            (memory $m0 1)
            (memory $m1 1)
            (func (param i32 i32)
                local.get 0
                i32.load $m0
                drop
                local.get 0
                i32.load $m1
                drop
                local.get 0
                local.get 1
                i32.store $m0
                local.get 0
                local.get 1
                i32.store $m1)
            (func (param i32 i32 i32)
                local.get 0 local.get 1 local.get 2 memory.fill $m0
                local.get 0 local.get 1 local.get 2 memory.fill $m1)
            (func (param i32 i32 i32)
                local.get 0 local.get 1 local.get 2 memory.copy $m0 $m0
                local.get 0 local.get 1 local.get 2 memory.copy $m1 $m1)
            (func (param i32 i32 i32 i32 i32 i32)
                local.get 0 local.get 1 local.get 2 memory.fill $m0
                local.get 3 local.get 4 local.get 5 memory.copy $m0 $m0
                local.get 0 local.get 1 local.get 2 memory.fill $m1
                local.get 3 local.get 4 local.get 5 memory.copy $m1 $m1))"#;

        for func in 0..4 {
            let f = predecode_wat(wat, func);
            for ins in f.code.iter().filter(|ins| {
                matches!(
                    super::super::layout::family(ins.op),
                    super::super::layout::Fam::Load | super::super::layout::Fam::Store
                ) || matches!(ins.op, Op::MemoryFill | Op::MemoryCopy | Op::MemoryFillCopy)
            }) {
                assert_eq!(
                    ins.flags & FLAG_NO_NATIVE != 0,
                    !native_guard(ins),
                    "op={:?} flags={:#x} b={:#x} c={:#x}",
                    ins.op,
                    ins.flags,
                    ins.b,
                    ins.c
                );
            }
        }
    }

    #[test]
    fn fuses_adjacent_fill_and_copy_on_the_same_memory() {
        let f = predecode_wat(
            r#"(module
                (memory 1)
                (func (param i32 i32 i32 i32 i32 i32)
                    (memory.fill
                        (local.get 0) (local.get 1) (local.get 2))
                    (memory.copy
                        (local.get 3) (local.get 4) (local.get 5))))"#,
            0,
        );
        let pair = f
            .code
            .iter()
            .find(|ins| ins.op == Op::MemoryFillCopy)
            .expect("adjacent pair must fuse");
        assert_eq!(pair.b, pair.a + 3);
        assert_eq!(pair.c, 0);
        assert!(!ops(&f).contains(&Op::MemoryFill));
        assert!(!ops(&f).contains(&Op::MemoryCopy));
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
        // The flush copy is emitted at set-processing time, i.e. AFTER the
        // add. It remains ordered first inside MovPair, so the old value is
        // captured before the unfolded set overwrites local 0.
        assert_eq!(ops(&f), [Op::I32_Add, Op::MovPair, Op::Return]);
        // add writes a temp, NOT local 0
        assert_ne!(f.code[0].c, 0);
        let pair = f.code[1];
        // copy 1: old local 0 -> canonical temp slot 1; copy 2: the add's
        // result slot 2 -> local 0.
        assert_eq!((pair.a, pair.b, pair.c), (0, 2, 1u64 << 32));
        // the returned value is the flushed OLD param value
        assert_eq!((f.code[2].a, f.code[2].b), (1, 1));
    }

    #[test]
    fn mov_pair_accumulator_belongs_only_to_second_destination() {
        for (result_local, expect_acc) in [("$x", false), ("$y", true)] {
            let f = predecode_wat(
                &format!(
                    r#"(module
                        (func (param $a i64) (param $b i64)
                              (result i64) (local $x i64) (local $y i64)
                            local.get $a
                            local.set $x
                            local.get $b
                            local.set $y
                            local.get {result_local}
                            i64.const 1
                            i64.add))"#
                ),
                0,
            );

            assert_eq!(ops(&f), [Op::MovPair, Op::I64_Add, Op::Return]);
            let pair = f.code[0];
            assert_eq!((pair.a, pair.b, pair.c), (0, 1, 2u64 << 32 | 3));
            assert_eq!(
                pair.flags & FLAG_DST_ACC,
                0,
                "MovPair must remain write-through even when destination 2 is forwarded"
            );
            let add_reads_acc = f.code[1].flags & FLAG_A_ACC != 0;
            assert_eq!(
                add_reads_acc, expect_acc,
                "MovPair leaves only destination 2 (`$y`) in the accumulator"
            );
        }
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
    fn typed_loop_counter_reuses_local_and_fuses_decrement_branch() {
        let f = predecode_wat(
            r#"(module
                (func (param $n i32) (result i32)
                    (local.get $n)
                    (loop $continue (param i32) (result i32)
                        (i32.const 1)
                        (i32.sub)
                        (local.tee $n)
                        (local.get $n)
                        (br_if $continue))))"#,
            0,
        );

        assert_eq!(
            ops(&f),
            [Op::I32_SubBrIf, Op::MovSlot, Op::Return],
            "the loop body must contain one dispatch cell"
        );
        let hot = f.code[0];
        assert_eq!(hot.flags, FLAG_B_CONST);
        assert_eq!((hot.a, hot.b, hot.c), (0, 1, 0));
        assert_eq!((f.code[1].a, f.code[1].c), (0, 1));
        assert_eq!((f.code[2].a, f.code[2].b), (1, 1));
    }

    #[test]
    fn unrelated_try_table_opcode_byte_does_not_disable_loop_aliases() {
        let f = predecode_wat(
            r#"(module
                (func (param $n i32) (result i32)
                    i32.const 31
                    drop
                    local.get $n
                    (loop $continue (param i32) (result i32)
                        i32.const 1
                        i32.sub
                        local.tee $n
                        local.get $n
                        br_if $continue)))"#,
            0,
        );

        assert_eq!(
            ops(&f),
            [Op::I32_SubBrIf, Op::MovSlot, Op::Return],
            "an immediate byte equal to the try_table opcode is not a try_table"
        );
    }

    #[test]
    fn unsafe_loop_home_does_not_deopt_an_independent_safe_loop() {
        let f = predecode_wat(
            r#"(module
                (func (param $x i32) (param $n i32) (result i32)
                    local.get $x
                    (loop $unsafe (param i32) (result i32)
                        drop
                        i32.const 42
                        local.get $n
                        i32.const 1
                        i32.sub
                        local.tee $n
                        br_if $unsafe)
                    drop
                    i32.const 3
                    local.set $n
                    local.get $n
                    (loop $safe (param i32) (result i32)
                        i32.const 1
                        i32.sub
                        local.tee $n
                        local.get $n
                        br_if $safe)
                    drop
                    local.get $x))"#,
            0,
        );

        assert!(
            f.code.iter().any(|ins| ins.op == Op::I32_SubBrIf),
            "canonicalizing one unsafe loop must retain an independent safe loop's hot fusion"
        );
    }

    #[test]
    fn typed_loop_constant_add_one_reuses_sub_branch_handler() {
        let f = predecode_wat(
            r#"(module
                (func (param $n i32) (result i32)
                    local.get $n
                    (loop $continue (param i32) (result i32)
                        i32.const 1
                        i32.add
                        local.tee $n
                        local.get $n
                        br_if $continue)))"#,
            0,
        );

        assert_eq!(
            ops(&f),
            [Op::I32_SubBrIf, Op::MovSlot, Op::Return],
            "constant add and branch must reuse the subtraction handler"
        );
        let hot = f.code[0];
        assert_eq!(hot.flags, FLAG_B_CONST);
        assert_eq!(hot.b, u32::MAX as u64);
    }

    #[test]
    fn typed_loop_constant_add_negates_rhs_with_i32_wrapping() {
        for (constant, expected_rhs) in [("-1", 1u32), ("-2147483648", 0x8000_0000u32), ("0", 0u32)]
        {
            let wat = format!(
                r#"(module
                    (func (param $n i32) (result i32)
                        local.get $n
                        (loop $continue (param i32) (result i32)
                            i32.const {constant}
                            i32.add
                            local.tee $n
                            local.get $n
                            br_if $continue)))"#
            );
            let f = predecode_wat(&wat, 0);
            assert_eq!(f.code[0].op, Op::I32_SubBrIf);
            assert_ne!(f.code[0].flags & FLAG_B_CONST, 0);
            assert_eq!(f.code[0].b, expected_rhs as u64, "constant {constant}");
        }
    }

    #[test]
    fn typed_loop_variable_rhs_add_stays_unfused() {
        let f = predecode_wat(
            r#"(module
                (func (param $n i32) (param $step i32) (result i32)
                    local.get $n
                    (loop $continue (param i32) (result i32)
                        local.get $step
                        i32.add
                        local.tee $n
                        local.get $n
                        br_if $continue)))"#,
            0,
        );

        assert_eq!(ops(&f), [Op::I32_Add, Op::BrIf, Op::MovSlot, Op::Return]);
        assert!(f.code.iter().all(|ins| ins.op != Op::I32_SubBrIf));
    }

    #[test]
    fn i64_sub_nonzero_branch_removes_the_zero_comparison() {
        let f = predecode_wat(
            r#"(module
                (func (param $n i64) (result i64)
                    (loop $continue
                        local.get $n
                        i64.const 1
                        i64.sub
                        local.set $n
                        local.get $n
                        i64.const 0
                        i64.ne
                        br_if $continue)
                    local.get $n))"#,
            0,
        );

        assert_eq!(ops(&f), [Op::I64_SubBrIf, Op::MovSlot, Op::Return]);
        let hot = f.code[0];
        assert_eq!(hot.flags, FLAG_B_CONST);
        assert_eq!((hot.a, hot.b, hot.c), (0, 1, 0));
        assert!(f.code.iter().all(|ins| ins.op != Op::I64_Ne));
    }

    #[test]
    fn fibonacci_i64_countdown_uses_the_generic_sub_branch() {
        let f = predecode_wat(
            r#"(module
                (func (export "run") (param $n i64) (result i64)
                    (local $a i64)
                    (local $b i64)
                    (local $i i64)
                    (local.set $a (i64.const 0))
                    (local.set $b (i64.const 1))
                    (local.set $i (local.get $n))
                    (block $break
                        (br_if $break (i64.eqz (local.get $i)))
                        (loop $continue
                            (i64.add (local.get $a) (local.get $b))
                            (local.set $a (local.get $b))
                            (local.set $b)
                            (local.set $i (i64.sub (local.get $i) (i64.const 1)))
                            (br_if $continue
                                (i64.ne (local.get $i) (i64.const 0)))))
                    (local.get $a)))"#,
            0,
        );

        assert_eq!(
            ops(&f),
            [
                Op::MovConst,
                Op::MovConst,
                Op::MovSlot,
                Op::I64_BrEq,
                Op::I64_Add,
                Op::MovPair,
                Op::I64_SubBrIf,
                Op::MovSlot,
                Op::Return,
            ]
        );
        let pair = f.code[5];
        assert_eq!((pair.a, pair.b, pair.c), (2, 4, 1u64 << 32 | 2));
        let hot = f.code[6];
        assert_eq!(hot.flags, FLAG_B_CONST);
        assert_eq!((hot.a, hot.b, hot.c), (3, 1, 4));
        assert!(f.code.iter().all(|ins| ins.op != Op::I64_Ne));
    }

    #[test]
    fn i64_constant_add_negates_rhs_with_i64_wrapping() {
        for (constant, expected_rhs) in [
            ("1", u64::MAX),
            ("-1", 1),
            ("-9223372036854775808", 0x8000_0000_0000_0000),
            ("0", 0),
        ] {
            let wat = format!(
                r#"(module
                    (func (param $n i64) (result i64)
                        (loop $continue
                            local.get $n
                            i64.const {constant}
                            i64.add
                            local.set $n
                            local.get $n
                            i64.const 0
                            i64.ne
                            br_if $continue)
                        local.get $n))"#
            );
            let f = predecode_wat(&wat, 0);
            assert_eq!(f.code[0].op, Op::I64_SubBrIf);
            assert_ne!(f.code[0].flags & FLAG_B_CONST, 0);
            assert_eq!(f.code[0].b, expected_rhs, "constant {constant}");
        }
    }

    #[test]
    fn i64_variable_rhs_add_stays_unfused() {
        let f = predecode_wat(
            r#"(module
                (func (param $n i64) (param $step i64) (result i64)
                    (loop $continue
                        local.get $n
                        local.get $step
                        i64.add
                        local.set $n
                        local.get $n
                        i64.const 0
                        i64.ne
                        br_if $continue)
                    local.get $n))"#,
            0,
        );

        assert!(f.code.iter().any(|ins| ins.op == Op::I64_Add));
        assert!(f.code.iter().all(|ins| ins.op != Op::I64_SubBrIf));
    }

    #[test]
    fn i64_sub_branch_keeps_a_safe_accumulator_rhs() {
        let f = predecode_wat(
            r#"(module
                (func (param $n i64) (param $step i64) (result i64)
                    (loop $continue
                        local.get $n
                        local.get $step
                        i64.const 0
                        i64.add
                        i64.sub
                        local.set $n
                        local.get $n
                        i64.const 0
                        i64.ne
                        br_if $continue)
                    local.get $n))"#,
            0,
        );

        let branch = f
            .code
            .iter()
            .find(|ins| ins.op == Op::I64_SubBrIf)
            .expect("fused i64 subtraction branch");
        assert_eq!(branch.flags & (FLAG_A_ACC | FLAG_DST_ACC), 0);
        assert_ne!(branch.flags & FLAG_B_ACC, 0);
    }

    #[test]
    fn i64_sub_branch_rejects_other_comparisons_and_branch_moves() {
        for compare in ["i64.const 0 i64.eq", "i64.const 1 i64.ne"] {
            let wat = format!(
                r#"(module
                    (func (param $n i64) (result i64)
                        (loop $continue
                            local.get $n
                            i64.const 1
                            i64.sub
                            local.set $n
                            local.get $n
                            {compare}
                            br_if $continue)
                        local.get $n))"#
            );
            let f = predecode_wat(&wat, 0);
            assert!(
                f.code.iter().all(|ins| ins.op != Op::I64_SubBrIf),
                "comparison {compare}"
            );
        }

        let with_moves = predecode_wat(
            r#"(module
                (func (param $value i64) (param $n i64) (result i64)
                    (block $exit (result i64)
                        local.get $value
                        local.get $n
                        i64.const 1
                        i64.sub
                        local.set $n
                        local.get $n
                        i64.const 0
                        i64.ne
                        br_if $exit
                        drop
                        i64.const 7)))"#,
            0,
        );
        assert!(
            with_moves.code.iter().all(|ins| ins.op != Op::I64_SubBrIf),
            "a taken-path value move must keep the update and branch separate"
        );
    }

    #[test]
    fn i64_eqz_br_if_uses_full_width_compare_with_constant_zero() {
        let f = predecode_wat(
            r#"(module
                (func (param $x i64) (result i32)
                    (block $zero
                        local.get $x
                        i64.eqz
                        br_if $zero
                        i32.const 0
                        return)
                    i32.const 1))"#,
            0,
        );

        let branch = f
            .code
            .iter()
            .find(|ins| ins.op == Op::I64_BrEq)
            .expect("full-width zero comparison branch");
        assert_eq!(branch.b, 0);
        assert_ne!(branch.flags & FLAG_B_CONST, 0);
        assert!(f.code.iter().all(|ins| ins.op != Op::I64_Eqz));
    }

    #[test]
    fn i64_eqz_br_if_move_guard_uses_inverted_full_width_compare() {
        let f = predecode_wat(
            r#"(module
                (func (param $x i64) (result i32)
                    (block $exit (result i32)
                        i32.const 10
                        i32.const 1
                        i32.add
                        i32.const 40
                        i32.const 2
                        i32.add
                        local.get $x
                        i64.eqz
                        br_if $exit
                        drop)))"#,
            0,
        );

        let guard = f
            .code
            .iter()
            .position(|ins| ins.op == Op::I64_BrNe)
            .expect("inverted full-width zero comparison guard");
        assert_eq!(f.code[guard].b, 0);
        assert_ne!(f.code[guard].flags & FLAG_B_CONST, 0);
        assert!(
            f.code[guard + 1..].iter().any(|ins| ins.op == Op::MovSlot),
            "the inverted guard must skip a taken-path branch-value move"
        );
        assert!(f.code.iter().all(|ins| ins.op != Op::I64_Eqz));
    }

    #[test]
    fn br_table_repeated_transfer_plan_shares_one_landing_pad() {
        let f = predecode_wat(
            r#"(module
                (func (param $discarded i32) (param $value i32)
                      (param $selector i32) (result i32)
                    (block $exit (result i32)
                        local.get $discarded
                        local.get $value
                        local.get $selector
                        br_table $exit $exit $exit)))"#,
            0,
        );

        assert_eq!(f.br_tables.len(), 1);
        let entries = &f.br_tables[0];
        assert_eq!(entries.len(), 3);
        assert!(
            entries.iter().all(|&entry| entry == entries[0]),
            "repeated labels with one transfer plan must share a landing pad"
        );
        let table = f
            .code
            .iter()
            .position(|ins| ins.op == Op::BrTable)
            .expect("br_table instruction");
        assert!(
            entries[0] as usize > table,
            "the shared entry must name a landing pad after br_table"
        );
        assert_eq!(f.code[entries[0] as usize].op, Op::MovSlot);
        assert_eq!(
            f.code[table + 1..]
                .iter()
                .filter(|ins| ins.op == Op::MovSlot)
                .count(),
            1,
            "the repeated entries must share one post-table copy"
        );
    }

    #[test]
    fn br_table_keeps_a_common_exact_loop_alias_direct() {
        let f = predecode_wat(
            r#"(module
                (func (param $x i32) (param $selector i32) (result i32)
                    local.get $x
                    (loop $l (param i32) (result i32)
                        local.get $selector
                        br_table $l $l)))"#,
            0,
        );

        assert_eq!(ops(&f), [Op::BrTable]);
        assert_eq!(f.br_tables.len(), 1);
        assert_eq!(f.br_tables[0].len(), 2);
        assert!(f.br_tables[0].iter().all(|&target| target == 0));
    }

    #[test]
    fn br_table_does_not_materialize_values_discarded_by_every_target() {
        let f = predecode_wat(
            r#"(module
                (func (param $discarded i32) (param $value i32)
                      (param $selector i32) (result i32)
                    (block $exit (result i32)
                        local.get $discarded
                        local.get $value
                        local.get $selector
                        br_table $exit $exit)))"#,
            0,
        );

        assert_eq!(
            ops(&f),
            [Op::MovSlot, Op::BrTable, Op::MovSlot, Op::Br, Op::Return]
        );
        assert_eq!(
            f.code[0].a, 1,
            "only the carried `$value` tuple member should be staged"
        );
        assert!(
            f.code.iter().all(|ins| ins.op != Op::MovPair),
            "the discarded local must not be included in table staging"
        );
    }

    #[test]
    fn fused_sub_branch_keeps_safe_accumulator_rhs() {
        let f = predecode_wat(
            r#"(module
                (func (param $n i32) (param $step i32) (result i32)
                    local.get $n
                    (loop $continue (param i32) (result i32)
                        local.get $step
                        i32.const 0
                        i32.add
                        i32.sub
                        local.tee $n
                        local.get $n
                        br_if $continue)))"#,
            0,
        );

        let branch = f
            .code
            .iter()
            .find(|ins| ins.op == Op::I32_SubBrIf)
            .expect("fused subtraction branch");
        assert_eq!(
            branch.flags & (FLAG_A_ACC | FLAG_DST_ACC),
            0,
            "the in-place destination must not depend on accumulator residency"
        );
        assert_ne!(
            branch.flags & FLAG_B_ACC,
            0,
            "the distinct rhs accumulator register remains a valid fast path"
        );
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
        let function_handles = [RefValue::new(0)];
        match predecode_function(&module, &[], &function_handles, 0) {
            Ok(_) => panic!("SIMD must be refused"),
            Err(err) => assert!(
                std::format!("{err:?}").contains("SIMD"),
                "the error should name the family, got {err:?}"
            ),
        }
    }
}
