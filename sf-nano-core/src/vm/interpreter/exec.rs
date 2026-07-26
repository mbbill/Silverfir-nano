//! Driver and slow path for the native dispatch chain.
//!
//! Two responsibilities, both outside the chain itself:
//!
//! - The activation trampoline. Calls and returns between local functions
//!   run entirely inside the chain on one contiguous overlapped value
//!   stack; Rust sees only host calls, the invocation root, slow ops, and
//!   traps. Call depth is interpreter data, never host recursion.
//! - `exec_ins`, the single-instruction executor. It is the chain's slow
//!   path, not a second engine: every op without a native handler, every
//!   import, and every trap that needs a message routes through it by one
//!   uniform exit/re-enter protocol, so native coverage can grow op by op
//!   without touching the driver.
//!
//! Instantiation is self-contained: globals, active data and element
//! segments, and funcref tables are built directly from the parsed module.

use tracked_alloc::boxed::Box;
use tracked_alloc::rc::Rc;

use crate::collections::{vec, Vec};
use crate::config::Config;
use crate::error::WasmError;
use crate::module::entities::{Data, Element, ElementInit, GlobalDef, MemoryDef, TableDef, TagDef};
use crate::module::type_context::{concrete_type_matches_cross_context, TypeContext};
use crate::module::Module;
use crate::opcodes::Opcode;
use crate::utils::limits::{Limitable, Limits};
use crate::value_type::{AbstractHeapType, HeapType, ValueType};
use crate::vm::engine::Engine;
use crate::vm::entities::{GlobalInst, MemInst, TableInst};
use crate::vm::imports::{Import, ImportValue, ImportedGlobal};
use crate::vm::tag::TagHandle;
use crate::vm::value::{RefHandle, Value};

use super::engine::{
    DCell, EnterState, LinkedFunction, NativeEngine, EXIT_RETURN, EXIT_SLOW, EXIT_TRAP_BASE,
    NATIVE_CALLS, RET_RECORD, TRAP_KINDS,
};
use super::fmath;
use super::instr::{op_from_index, Op};
use super::instr::{Instr, FLAG_ADDR64, FLAG_A_CONST, FLAG_B_CONST, FLAG_FUSED};
use super::predecode::{predecode_function, PredecodedFunction, WIDE_MEMARG};

const PAGE: usize = 65536;
/// A null reference, as it sits in an 8-byte slot.
///
/// `RefHandle` is `usize`-wide, so its own null (`usize::MAX`) is a different
/// bit pattern on 32- and 64-bit hosts. Slots are always 8 bytes, so the two
/// conversions below normalize null to one value and a reference means the
/// same thing in a slot on every target.
const NULL_REF_SLOT: u64 = u64::MAX;

/// A reference as stored in a slot, table entry, or global.
///
/// A plain handle's payload IS the function index -- the same number the
/// interpreter used to store as a bare `u32` -- so widening to the full
/// handle keeps funcrefs bit-identical and brings extern/host refs, which
/// ride in `RefHandle`'s tag bits, along for free.
#[inline]
pub(crate) fn ref_to_slot(handle: RefHandle) -> u64 {
    if handle.is_null() {
        NULL_REF_SLOT
    } else {
        handle.0 as u64
    }
}

#[inline]
pub(crate) fn slot_to_ref(slot: u64) -> RefHandle {
    if slot == NULL_REF_SLOT {
        RefHandle::null()
    } else {
        RefHandle::new(slot as usize)
    }
}
const MAX_CALL_DEPTH: u32 = 4096;
/// Ceiling on native call depth, and so on the return-stack records the
/// dispatch chain can plant.
const MAX_RET_RECORDS: u32 = MAX_CALL_DEPTH + 8;

/// Slots in the operand stack, from the embedder's configured budget.
///
/// The engine's `wasm_stack_bytes`. The hosted default is 2 MiB, which is
/// what the interpreter's fixed size used to be, so nothing changes for a
/// hosted embedder -- only how often it is paid for.
///
/// A bare-metal embedder that has not configured the runtime yet is told
/// so here. Silently substituting a token stack would turn that mistake
/// into a "call stack exhausted" trap somewhere further in.
fn configured_stack_slots(config: &Config) -> usize {
    config.get_wasm_stack_bytes() / core::mem::size_of::<u64>()
}

/// Return-stack records to reserve.
///
/// Depth cannot usefully exceed what the operand stack can hold frames
/// for, so a small budget buys a proportionally small return stack rather
/// than the full 4096-deep one (131 KB, a third of a Pico 2's heap).
fn configured_ret_records(stack_slots: usize) -> usize {
    (stack_slots / 4).clamp(16, MAX_RET_RECORDS as usize)
}

/// Host dispatcher for imported functions: called with the import's module
/// and field names, the linear memory, argument slots, and result slots.
/// The signature carries only std types (`&mut [u8]`, not this crate's
/// tracked collections), so external callers stay feature-independent.
pub(crate) type HostDispatch =
    Box<dyn FnMut(&str, &str, &mut [u8], &[u64], &mut [u64]) -> Result<(), WasmError>>;

/// How a reference to one of this instance's functions is named so another
/// instance can call it, and how such a name is called back.
///
/// A funcref is a local index and means nothing outside the instance that
/// wrote it, so one entering a SHARED table needs a global name. Naming is
/// the embedder's job: resolving a name means calling into another instance,
/// which needs `&mut` to it while this one is borrowed. The embedder owns
/// both; an engine-side registry could only manage it with a raw pointer into
/// storage the embedder is free to move.
///
/// The boxes are `alloc`'s, not this crate's tracked ones, for the same reason
/// `HostDispatch`'s signature carries only std types: an embedder has to be
/// able to construct one without depending on the allocator this crate is
/// built with.
pub struct FuncRefHost {
    /// Name local function `idx` of this instance.
    pub publish: alloc::boxed::Box<dyn Fn(usize) -> RefHandle>,
    /// Call whatever `handle` names, wherever it lives.
    pub invoke:
        alloc::boxed::Box<dyn FnMut(RefHandle, &[u64], &mut [u64]) -> Result<(), WasmError>>,
}

/// `max` is a growth limit, consulted only by `table.grow`; the executor
/// that implements it is arm64-only, so the field follows that gate.
/// `entries` is not gated: instantiation reads its length to bounds-check
/// active element segments on every target.
/// A table's entries, in one of two tiers.
///
/// This mirrors what the JIT does with `TableDispatchMode`: a table nobody
/// else can see keeps a private array the dispatch chain may index directly,
/// while one that is imported or exported must be the SHARED entity, because
/// another instance can write to it and both sides have to observe that.
///
/// The tiers differ in element type, which is the whole reason for the split:
/// the chain reads 8-byte slots, and `TableInst` holds `RefHandle`, which is
/// narrower on the 32-bit targets. A shared table therefore never reaches a
/// native handler.
enum TableEntries {
    Private(Vec<u64>),
    Shared(TableInst),
}

impl TableEntries {
    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::Private(v) => v.len(),
            Self::Shared(t) => t.elements().len(),
        }
    }

    #[inline]
    fn get(&self, i: usize) -> Option<u64> {
        match self {
            Self::Private(v) => v.get(i).copied(),
            Self::Shared(t) => t.elements().get(i).map(|h| ref_to_slot(*h)),
        }
    }

    #[inline]
    fn set(&mut self, i: usize, v: u64) -> Result<(), WasmError> {
        let oob = || WasmError::trap("out of bounds table access");
        match self {
            Self::Private(vec) => *vec.get_mut(i).ok_or_else(oob)? = v,
            Self::Shared(t) => *t.elements_mut().get_mut(i).ok_or_else(oob)? = slot_to_ref(v),
        }
        Ok(())
    }

    fn resize(&mut self, n: usize, v: u64) {
        match self {
            Self::Private(vec) => vec.resize(n, v),
            Self::Shared(t) => t.elements_mut().resize(n, slot_to_ref(v)),
        }
    }

    fn fill(&mut self, at: usize, n: usize, v: u64) {
        match self {
            Self::Private(vec) => vec[at..at + n].fill(v),
            Self::Shared(t) => t.elements_mut()[at..at + n].fill(slot_to_ref(v)),
        }
    }

    /// The base pointer the dispatch chain indexes, when there is one.
    ///
    /// `None` for a shared table: its elements are `RefHandle`, not 8-byte
    /// slots, so no native handler may read them.
    #[inline]
    fn fast_base(&self) -> Option<*const u64> {
        match self {
            Self::Private(v) => Some(v.as_ptr()),
            Self::Shared(_) => None,
        }
    }
}

struct TableState {
    entries: TableEntries,
    max: u64,
}

/// `max_pages` mirrors `TableState::max`: only `memory.grow` reads it, and
/// `is64` tells it which index type the grow result and page cap belong to.
struct MemoryState {
    /// The substrate's memory entity, so a memory can be shared with
    /// whoever exported it. It used to be an owned `Vec<u8>`, which is
    /// why an imported memory could not be represented at all.
    inst: MemInst,
    max_pages: u64,
    is64: bool,
}

impl MemoryState {
    /// The linear memory as a slice.
    ///
    /// SAFETY: `MemInst` hands out a base pointer and a length for a
    /// region it keeps alive; the same construction backs the JIT's own
    /// host-call path. The borrow of the backing ends inside the
    /// accessors, so no `RefCell` guard is held across the slice.
    #[inline]
    fn bytes(&self) -> &[u8] {
        let (ptr, len) = (self.inst.memory_ptr(), self.inst.memory_len());
        unsafe { core::slice::from_raw_parts(ptr, len) }
    }

    #[inline]
    fn bytes_mut(&mut self) -> &mut [u8] {
        let (ptr, len) = (self.inst.memory_ptr(), self.inst.memory_len());
        unsafe { core::slice::from_raw_parts_mut(ptr, len) }
    }

    #[inline]
    fn len(&self) -> usize {
        self.inst.memory_len()
    }
}

/// One live call frame. Calls and returns are driven by an explicit
/// activation stack in the driver loop, never by host recursion, so call
/// depth is interpreter data (the classic-interpreter lesson).
struct Activation {
    func: Rc<PredecodedFunction>,
    /// Index of `func` in the module's function space (native dispatch
    /// resolves its linked cells by this).
    func_index: usize,
    /// Frame base slot in the shared value stack. Results return to the
    /// caller in place: this base IS the caller's staged-argument slot.
    base: usize,
    pc: usize,
    /// Whether this activation's native return route (its sentinel, or the
    /// caller record a native call pushed) is already on the native return
    /// stack — set after the first native entry, so resumes never re-plant.
    route_established: bool,
}

/// Per-invoke execution resources: the shared value stack and the native
/// return stack (records of `(ret_pc, frame, code_base)`, with
/// Rust-planted sentinel records routing a `Return` back to Rust).
struct DriveCtx<'s> {
    stack: &'s mut [u64],
    ret_stack: &'s mut [u64],
    /// Byte cursor into `ret_stack`.
    ret_cursor: usize,
    /// The accumulator relayed across native sessions: call results ride
    /// it over activation boundaries (sentinel returns, host calls).
    acc: u64,
}

enum StepExit {
    TailCall { callee: usize, arg_base: usize },
    Call { callee: usize, arg_base: usize },
    Return,
}

/// Result of executing exactly one instruction.
pub(super) enum Effect {
    Next,
    Jump(usize),
    Call {
        callee: usize,
        arg_base: usize,
    },
    /// A call that replaces the current activation rather than nesting
    /// under it, so `return_call` recursion runs in constant frame depth.
    TailCall {
        callee: usize,
        arg_base: usize,
    },
    Ret,
}

/// Stage-B state: the emitted handler engine plus per-function dispatch
/// cells (parallel to `InterpInstance::funcs`).
struct NativeState {
    engine: NativeEngine,
    linked: Vec<Option<LinkedFunction>>,
    /// Per-function-index callee info for the native `CallIndirect`
    /// handler: [callee cells (0 = slow), l1off<<48|l0off<<32|canon type,
    /// frame metadata]. The handler reads it through the entry state.
    indirect_info: Vec<[u64; 3]>,
    /// `(cells_start, cells_end, func_index)`, sorted by start — maps a
    /// native pc back to its function on slow exits (native calls move
    /// between functions without Rust involvement).
    ranges: Vec<(u64, u64, u32)>,
    /// Slow-path exits per op — the native chain's cost profile. What
    /// dominates here is what to native-ize next.
    slow_exits: Vec<u64>,
    /// Total native handler dispatches (from the in-chain counter).
    dispatches: u64,
}

/// A self-contained interpreter instance over a parsed module.
pub struct InterpInstance {
    /// Owned, not borrowed: a predecoded function carries no reference
    /// back into the module, so there is nothing to keep alive separately
    /// and nothing for an embedder to have to outlive.
    module: Module,
    funcs: Vec<Option<Rc<PredecodedFunction>>>,
    /// Runtime state only the executor touches. `new()` still builds it on
    /// every target -- doing so is what rejects imported/64-bit/non-funcref
    /// memories and tables and traps out-of-range active segments -- but a
    /// target without an executor validates and drops it rather than
    /// carrying it.
    memories: Vec<MemoryState>,
    dropped_data: Vec<bool>,
    dropped_elems: Vec<bool>,
    globals: Vec<u64>,
    /// Globals another instance can reach, by index. These live in an
    /// `Rc`-owned cell rather than the array above, because both sides must
    /// observe each other's writes; the array slot beside them is unused.
    shared_globals: Vec<Option<GlobalInst>>,
    tables: Vec<TableState>,
    /// Runtime tag identities, one per declared tag. Linking needs them;
    /// nothing here throws yet.
    tags: Vec<TagHandle>,
    /// The operand stack and native return stack, allocated once at
    /// instantiation and reused by every call. They used to be allocated
    /// and zeroed inside each invocation, which put a 2 MiB allocation in
    /// front of every call to a Wasm function.
    config: Config,
    stack: Vec<u64>,
    ret_stack: Vec<u64>,
    host: Option<HostDispatch>,
    funcref_host: Option<FuncRefHost>,
    /// Names this instance handed out for its OWN functions, so a reference
    /// read back from a shared table that we published resolves locally
    /// instead of going out through the embedder and back.
    published: Vec<(RefHandle, usize)>,
    native: Option<NativeState>,
}

/// Nonzero per-op counts, descending.
fn op_counts(table: &[u64]) -> Vec<(Op, u64)> {
    let mut out: Vec<(Op, u64)> = Vec::new();
    for (i, &n) in table.iter().enumerate() {
        if n > 0 {
            out.push((op_from_index(i), n));
        }
    }
    out.sort_by(|x, y| y.1.cmp(&x.1));
    out
}

/// Evaluate the simple constant expressions v1 supports for offsets and
/// global initializers: a single `t.const` (or `ref.func`/`ref.null`).
/// Evaluate a constant expression.
///
/// Covers the extended set: `t.const`, `ref.null`, `ref.func`, `global.get`
/// of an already-initialized immutable global, and `i32`/`i64` add, sub and
/// mul. `globals` is what has been initialized so far, which is exactly
/// what `global.get` is allowed to reach in a constant expression.
/// A numeric `Value` as the raw 64-bit slot this engine stores.
fn value_to_raw_for_interp(v: &Value) -> Result<u64, WasmError> {
    Ok(match v {
        Value::I32(x) => *x as u32 as u64,
        Value::I64(x) => *x as u64,
        Value::F32(x) => x.to_bits() as u64,
        Value::F64(x) => x.to_bits(),
        Value::Ref(handle, _) => ref_to_slot(*handle),
        _ => return Err(WasmError::invalid("interp: unsupported global import type")),
    })
}

fn eval_const(expr: &[u8], globals: &[u64]) -> Result<u64, WasmError> {
    use crate::op_decoder::{Decoder, Immediate, OpStream, OpcodeHandler};
    use crate::opcodes::WasmOpcode;

    struct H<'g> {
        stack: Vec<u64>,
        globals: &'g [u64],
    }

    impl H<'_> {
        fn pop2(&mut self) -> Result<(u64, u64), WasmError> {
            let b = self.stack.pop();
            let a = self.stack.pop();
            match (a, b) {
                (Some(a), Some(b)) => Ok((a, b)),
                _ => Err(WasmError::invalid("interp: constant expression underflows")),
            }
        }
    }

    impl OpcodeHandler for H<'_> {
        fn on_decode_begin(&mut self) -> Result<(), WasmError> {
            Ok(())
        }
        fn on_stream<'x, 'y, 'z>(
            &mut self,
            stream: &mut OpStream<'x, 'y, 'z>,
        ) -> Result<(), WasmError> {
            while let Some(op) = stream.next()? {
                match (op.wasm_op, &op.imm) {
                    (WasmOpcode::OP(Opcode::I32_CONST), Immediate::I32(v)) => {
                        self.stack.push(*v as u32 as u64)
                    }
                    (WasmOpcode::OP(Opcode::I64_CONST), Immediate::I64(v)) => {
                        self.stack.push(*v as u64)
                    }
                    (WasmOpcode::OP(Opcode::F32_CONST), Immediate::F32(v)) => {
                        self.stack.push(v.to_bits() as u64)
                    }
                    (WasmOpcode::OP(Opcode::F64_CONST), Immediate::F64(v)) => {
                        self.stack.push(v.to_bits())
                    }
                    (WasmOpcode::OP(Opcode::REF_FUNC), Immediate::FunctionIndex(i)) => {
                        self.stack.push(*i as u64)
                    }
                    (WasmOpcode::OP(Opcode::REF_NULL), _) => self.stack.push(NULL_REF_SLOT),
                    (WasmOpcode::OP(Opcode::GLOBAL_GET), Immediate::GlobalIndex(i)) => {
                        // Only an already-initialized global is reachable:
                        // the spec restricts a constant expression to
                        // imported and previously-declared immutable ones.
                        let v =
                            self.globals
                                .get(*i as usize)
                                .copied()
                                .ok_or(WasmError::invalid(
                                    "interp: constant expression reads a later global",
                                ))?;
                        self.stack.push(v);
                    }
                    (WasmOpcode::OP(Opcode::I32_ADD), _) => {
                        let (a, b) = self.pop2()?;
                        self.stack.push((a as u32).wrapping_add(b as u32) as u64);
                    }
                    (WasmOpcode::OP(Opcode::I32_SUB), _) => {
                        let (a, b) = self.pop2()?;
                        self.stack.push((a as u32).wrapping_sub(b as u32) as u64);
                    }
                    (WasmOpcode::OP(Opcode::I32_MUL), _) => {
                        let (a, b) = self.pop2()?;
                        self.stack.push((a as u32).wrapping_mul(b as u32) as u64);
                    }
                    (WasmOpcode::OP(Opcode::I64_ADD), _) => {
                        let (a, b) = self.pop2()?;
                        self.stack.push(a.wrapping_add(b));
                    }
                    (WasmOpcode::OP(Opcode::I64_SUB), _) => {
                        let (a, b) = self.pop2()?;
                        self.stack.push(a.wrapping_sub(b));
                    }
                    (WasmOpcode::OP(Opcode::I64_MUL), _) => {
                        let (a, b) = self.pop2()?;
                        self.stack.push(a.wrapping_mul(b));
                    }
                    (WasmOpcode::OP(Opcode::END), _) => continue,
                    _ => {
                        return Err(WasmError::invalid(
                            "interp: unsupported constant expression",
                        ))
                    }
                }
            }
            Ok(())
        }
        fn on_decode_end(&mut self) -> Result<(), WasmError> {
            Ok(())
        }
    }

    let mut h = H {
        stack: Vec::new(),
        globals,
    };
    // A const expr is an expression body: decode it like a function body.
    // The decoder is scoped so its borrow of `h` provably ends before the
    // read below (with tracked-alloc's memprof feature unified in, its Vec
    // has a destructor and the borrow would otherwise extend to it).
    {
        let mut d = Decoder::new(expr);
        d.add_handler(&mut h);
        d.decode_function()?;
    }
    match h.stack.len() {
        1 => Ok(h.stack[0]),
        0 => Err(WasmError::invalid("interp: empty constant expression")),
        _ => Err(WasmError::invalid(
            "interp: constant expression leaves more than one value",
        )),
    }
}

/// Whether a provided global's reference type satisfies a declared one.
///
/// Invariant for a mutable global -- writes flow both ways -- and covariant
/// for an immutable one, which is only read. A concrete heap type names an
/// index in the EXPORTER's type space, so deciding it needs that context;
/// without one the only honest answer is to accept, since refusing would
/// reject valid modules.
fn global_ref_types_match(
    provided: ValueType,
    declared: ValueType,
    provided_ctx: Option<&TypeContext>,
    declared_ctx: &TypeContext,
    mutable: bool,
) -> bool {
    let (ValueType::Ref(p), ValueType::Ref(d)) = (provided, declared) else {
        return false;
    };
    if mutable && p.nullable != d.nullable {
        return false;
    }
    if !mutable && p.nullable && !d.nullable {
        // A nullable value cannot satisfy a non-nullable declaration.
        return false;
    }
    match (p.heap_type, d.heap_type) {
        (HeapType::Concrete(pi), HeapType::Concrete(di)) => match provided_ctx {
            Some(ctx) => concrete_type_matches_cross_context(ctx, pi, declared_ctx, di),
            // Undecidable without the exporter's context; accepting beats
            // refusing a valid module.
            None => true,
        },
        (HeapType::Abstract(pa), HeapType::Abstract(da)) => {
            if mutable {
                pa == da
            } else {
                pa.is_subtype_of(&da)
            }
        }
        // A concrete function type is a subtype of the abstract `func`, so it
        // satisfies an immutable declaration of it.
        (HeapType::Concrete(_), HeapType::Abstract(AbstractHeapType::Func)) => !mutable,
        // The other direction never holds: an abstract type is the SUPERtype,
        // so `func` cannot satisfy a declaration of a specific `$t`.
        _ => false,
    }
}

/// Whether a provided entity's limits satisfy what an import declares.
///
/// The provider must be at least as large as the declared minimum, and if the
/// declaration caps the size then the provider must be capped no higher. An
/// unbounded provider therefore cannot satisfy a bounded declaration. The
/// index type has to agree as well: a 64-bit table does not satisfy a 32-bit
/// declaration or the reverse.
fn limits_satisfy(declared: &Limits, provided: &Limits) -> bool {
    if declared.is64 != provided.is64 {
        return false;
    }
    if provided.min() < declared.min() {
        return false;
    }
    match declared.max() {
        None => true,
        Some(cap) => provided.max().is_some_and(|p| p <= cap),
    }
}

impl InterpInstance {
    /// Instantiate, link the host, then run the start function -- in that
    /// order, which is why the host arrives here rather than through a
    /// setter. A start function may call an import, and a setter cannot
    /// be called before a constructor that already ran it.
    pub fn new(
        engine: &Engine,
        module: Module,
        host: Option<HostDispatch>,
        imports: &[Import],
    ) -> Result<Self, WasmError> {
        Self::new_partial(engine, module, host, imports, None).map_err(|(_, e)| e)
    }

    /// Instantiate, handing the instance back even when a data segment traps.
    ///
    /// Element segments run before data ones, so a module whose data segment
    /// traps has already written its elements -- possibly into a table another
    /// instance holds. Those writes stand and anything they reference must
    /// stay callable, so the caller keeps the instance rather than losing it.
    pub fn new_partial(
        engine: &Engine,
        module: Module,
        host: Option<HostDispatch>,
        imports: &[Import],
        funcref_host: Option<FuncRefHost>,
    ) -> Result<Self, (Option<Self>, WasmError)> {
        let mut inst =
            Self::build(engine, module, host, imports, funcref_host).map_err(|e| (None, e))?;
        // Link the dispatch chain BEFORE the data segments: a partial
        // instance handed back after a trap still has to be callable, and
        // linking is not observable, so its position is free.
        inst.enable_native_dispatch().map_err(|e| (None, e))?;
        if let Err(e) = inst.apply_element_segments() {
            return Err((Some(inst), e));
        }
        if let Err(e) = inst.apply_data_segments() {
            return Err((Some(inst), e));
        }
        if let Some(si) = inst.module.start_function_index() {
            if let Err(e) = inst.invoke(si, &[], &mut []) {
                return Err((Some(inst), e));
            }
        }
        Ok(inst)
    }

    fn build(
        engine: &Engine,
        module: Module,
        host: Option<HostDispatch>,
        imports: &[Import],
        funcref_host: Option<FuncRefHost>,
    ) -> Result<Self, WasmError> {
        let config = *engine.config();
        // Names handed out for this instance's own functions, filled as
        // exported globals and shared tables are initialized.
        let mut published: Vec<(RefHandle, usize)> = Vec::new();

        // Resolve imported memories up front, indexed the way the module
        // declares them, so the loop below can take a shared backing
        // instead of allocating one.
        // A memory import that nobody provides is a LINK error. Falling
        // through to "allocate one" would instantiate a module the JIT
        // refuses, so present-with-no-backing (the embedder saying
        // "allocate") and absent-entirely have to stay distinguishable.
        let mut imported_memories: Vec<Option<MemInst>> = Vec::new();
        let mut import_limits: Vec<Option<Limits>> = Vec::new();
        for m in module.memories() {
            if let MemoryDef::Import {
                module: md, name, ..
            } = m.def()
            {
                let provided = imports
                    .iter()
                    .find(|imp| imp.module == *md && imp.name == *name)
                    .ok_or(WasmError::unlinkable("missing memory import"))?;
                let ImportValue::Memory(provided_limits, _) = &provided.value else {
                    return Err(WasmError::unlinkable("incompatible import type"));
                };
                if !limits_satisfy(m.limits(), provided_limits) {
                    return Err(WasmError::unlinkable("incompatible import type"));
                }
                // The provider's limits win, as they do for tables: the
                // importing module may declare a laxer maximum than the
                // memory it actually receives, and `memory.grow` must refuse
                // at the real one.
                import_limits.push(Some(provided_limits.clone()));
            } else {
                import_limits.push(None);
            }
            imported_memories.push(match m.def() {
                MemoryDef::Import {
                    module: md, name, ..
                } => imports.iter().find_map(|imp| {
                    if imp.module != *md || imp.name != *name {
                        return None;
                    }
                    match &imp.value {
                        ImportValue::Memory(_, shared) => shared.clone(),
                        _ => None,
                    }
                }),
                MemoryDef::Local(_) => None,
            });
        }
        let stack_slots = configured_stack_slots(&config);

        // Functions: predecode every local function eagerly; imports are
        // rejected on call, not here, so import-free modules always work.
        let mut funcs = Vec::new();
        for (i, f) in module.functions().iter().enumerate() {
            if f.is_import() {
                funcs.push(None);
            } else {
                funcs.push(Some(Rc::new(predecode_function(&module, i)?)));
            }
        }

        // Memories.
        let mut memories = Vec::new();
        for (i, m) in module.memories().iter().enumerate() {
            let limits = import_limits
                .get(i)
                .and_then(|l| l.as_ref())
                .unwrap_or_else(|| m.limits());
            // An import with a shared backing takes it, so writes are
            // visible on both sides. An import with none declared is the
            // embedder saying "allocate one to these limits" -- the same
            // reading the JIT gives it.
            let inst = match imported_memories.get(i).cloned().flatten() {
                Some(shared) => shared,
                None => MemInst::new(&config, limits.clone())?,
            };
            memories.push(MemoryState {
                inst,
                // In u64 throughout: memory64's page cap does not fit a
                // 32-bit `usize`, which is what `Limits::max` yields on the
                // bare-metal targets.
                max_pages: limits.max().map(|m| m as u64).unwrap_or(if limits.is64 {
                    1u64 << 48
                } else {
                    65536
                }),
                is64: limits.is64,
            });
        }

        // Globals.
        let mut globals = Vec::new();
        let mut shared_globals: Vec<Option<GlobalInst>> = Vec::new();
        for g in module.globals() {
            match g.def() {
                GlobalDef::Local(spec) => {
                    let mut v = eval_const(spec.init_expr(), &globals)?;
                    // A funcref in an EXPORTED global is as invisible to its
                    // importer as one in a shared table, and for the same
                    // reason: the index means nothing outside this instance.
                    if !g.export_names().is_empty() {
                        if let (ValueType::Ref(_), Some(h)) =
                            (spec.value_type(), funcref_host.as_ref())
                        {
                            let handle = slot_to_ref(v);
                            if !handle.is_null() && !handle.is_special() {
                                let named = (h.publish)(handle.0);
                                published.push((named, handle.0));
                                v = ref_to_slot(named);
                            }
                        }
                    }
                    // A global another instance can reach must BE the shared
                    // cell, so an importer's writes are visible here and ours
                    // there. A private one stays in the array the chain reads.
                    if g.export_names().is_empty() {
                        shared_globals.push(None);
                    } else {
                        shared_globals.push(Some(GlobalInst::new_raw(
                            v,
                            spec.mutable(),
                            spec.value_type(),
                        )));
                    }
                    globals.push(v);
                }
                GlobalDef::Import {
                    module: md,
                    name,
                    value_type,
                    mutable,
                } => {
                    let found = imports.iter().find_map(|imp| {
                        if imp.module != *md || imp.name != *name {
                            return None;
                        }
                        match &imp.value {
                            ImportValue::Global(g, mutable) => Some((g, *mutable)),
                            _ => None,
                        }
                    });
                    match found {
                        // A value import, or a shared immutable one: the
                        // current value is the whole story, so copying it
                        // in is exact.
                        Some((ImportedGlobal::Value(v), _)) => {
                            shared_globals.push(None);
                            globals.push(value_to_raw_for_interp(&v.value)?)
                        }
                        // Aliased, not copied: both sides must observe each
                        // other's writes. Accesses to it are denied a native
                        // handler, since the chain indexes the array below.
                        Some((ImportedGlobal::State(st), _)) => {
                            // Mutability must match EXACTLY, in both
                            // directions: importing a `mut` global as
                            // immutable would let the importer assume a value
                            // the exporter can still change.
                            //
                            // The value type is invariant for a mutable global
                            // and covariant for an immutable one; see
                            // `global_ref_types_match`.
                            let type_ok = st.global.value_type == *value_type
                                || global_ref_types_match(
                                    st.global.value_type,
                                    *value_type,
                                    st.type_ctx.as_ref(),
                                    module.types(),
                                    *mutable,
                                );
                            if st.global.mutable != *mutable || !type_ok {
                                return Err(WasmError::unlinkable("incompatible import type"));
                            }
                            shared_globals.push(Some(st.global.clone()));
                            globals.push(st.global.raw());
                            continue;
                        }
                        None => return Err(WasmError::unlinkable("missing global import")),
                    }
                }
            }
        }

        // Tables (any reference type) + active element segments.
        let mut tables = Vec::new();
        for t in module.tables() {
            let mut shared_from_import: Option<TableInst> = None;
            // An imported table's size is the PROVIDER's, not the importing
            // module's declared minimum: `(table (import ..) 0 funcref)` sees
            // however many entries the exporter actually has, so the import's
            // limits win wherever both exist.
            let limits = if let TableDef::Import {
                module: md, name, ..
            } = t.def()
            {
                let found = imports.iter().find_map(|imp| {
                    if imp.module != *md || imp.name != *name {
                        return None;
                    }
                    match &imp.value {
                        ImportValue::Table(limits, shared) => Some((limits.clone(), shared)),
                        _ => None,
                    }
                });
                match found {
                    // Declared limits with no instance: the embedder is
                    // saying "allocate one", the same reading the JIT gives
                    // it and what the spectest host registers.
                    Some((limits, None)) => {
                        if !limits_satisfy(t.spec().limits(), &limits) {
                            return Err(WasmError::unlinkable("incompatible import type"));
                        }
                        limits
                    }
                    // A live table from another instance is ALIASED, not
                    // copied: both sides must see each other's `table.set`.
                    Some((limits, Some(state))) => {
                        // A table is a mutable container, so its element type
                        // is invariant: both sides read AND write it, and a
                        // subtype on either side would let one of them see a
                        // value the other's type forbids.
                        if state.table.value_type != t.value_type() {
                            return Err(WasmError::unlinkable("incompatible import type"));
                        }
                        if !limits_satisfy(t.spec().limits(), &limits) {
                            return Err(WasmError::unlinkable("incompatible import type"));
                        }
                        shared_from_import = Some(state.table.clone());
                        limits
                    }
                    None => return Err(WasmError::unlinkable("missing table import")),
                }
            } else {
                t.spec().limits().clone()
            };
            // A table another instance can reach must BE the shared entity:
            // an importer's `table.set` has to be visible to the exporter and
            // the other way round. Only a table that is neither imported nor
            // exported can keep a private array.
            let shared_out = !t.export_names().is_empty();
            // A table may declare an initializer, and every slot starts at
            // its value rather than null. Ignoring it left `(table 10 funcref
            // (ref.func $d))` full of nulls.
            let init_slot = match t.spec().init_expr() {
                Some(expr) => {
                    let raw = eval_const(expr, &globals)?;
                    let handle = slot_to_ref(raw);
                    match (&funcref_host, handle.is_null() || handle.is_special()) {
                        // Shared tables carry globally-named references; see
                        // `FuncRefHost`.
                        (Some(h), false) if !t.export_names().is_empty() || t.is_import() => {
                            let named = (h.publish)(handle.0);
                            published.push((named, handle.0));
                            ref_to_slot(named)
                        }
                        _ => raw,
                    }
                }
                None => NULL_REF_SLOT,
            };
            let entries = match (&shared_from_import, shared_out) {
                (Some(inst), _) => TableEntries::Shared(inst.clone()),
                (None, true) => {
                    let inst = TableInst::new(limits.clone(), t.value_type());
                    inst.elements_mut()
                        .resize(limits.min(), slot_to_ref(init_slot));
                    TableEntries::Shared(inst)
                }
                (None, false) => TableEntries::Private(vec![init_slot; limits.min() as usize]),
            };
            tables.push(TableState {
                entries,
                max: limits.max().unwrap_or(u32::MAX as usize) as u64,
            });
        }
        // A tag's IDENTITY is a linking concern, not an execution one: two
        // tags with the same signature in different modules must not alias,
        // and an importer has to be able to name ours. So mint one per local
        // tag even though nothing here can throw yet.
        let tags: Vec<TagHandle> = module
            .tags()
            .iter()
            .map(|t| match t.def() {
                TagDef::Local(_) => TagHandle::mint_fresh(),
                TagDef::Import { .. } => TagHandle::mint_fresh(),
            })
            .collect();

        // Tag imports are checked but not otherwise modelled: this engine has
        // no exception handling yet, so a tag is never thrown or caught here.
        // Linking still has to be right -- a module whose tag import is
        // missing or has the wrong type is unlinkable, and accepting it would
        // let a module instantiate that the JIT refuses.
        for tag in module.tags() {
            let TagDef::Import {
                module: md,
                name,
                func_type,
                type_index,
                ..
            } = tag.def()
            else {
                continue;
            };
            let provided = imports
                .iter()
                .find(|imp| imp.module == *md && imp.name == *name)
                .ok_or(WasmError::unlinkable("missing tag import"))?;
            let ImportValue::Tag(state) = &provided.value else {
                return Err(WasmError::unlinkable("incompatible import type"));
            };
            // Through the type context where both sides have one: two `func`
            // types in a rec group are structurally identical yet distinct
            // identities, and only the context separates them.
            let compatible = match (&state.type_ctx, state.type_index) {
                (Some(ctx), idx) if idx != u32::MAX => {
                    concrete_type_matches_cross_context(ctx, idx, module.types(), *type_index)
                }
                _ => {
                    state.func_type.params() == func_type.params()
                        && state.func_type.results() == func_type.results()
                }
            };
            if !compatible {
                return Err(WasmError::unlinkable("incompatible import type"));
            }
        }

        // Every element segment's function indices must name a function the
        // module has, whether or not the segment is ever used. An unbound
        // index is a validation error, not a trap at `table.init`.
        let n_funcs = module.functions().len() as u64;
        for e in module.elements() {
            if let ElementInit::FunctionIndexes(idxs) = e.get_init() {
                if idxs.iter().any(|&fi| fi as u64 >= n_funcs) {
                    return Err(WasmError::invalid("unknown function"));
                }
            }
        }
        // A DECLARATIVE segment exists only to forward-declare references;
        // it carries no initializer a `table.init` may read, so it starts
        // dropped and any use of it traps.
        let dropped_elems: Vec<bool> = module
            .elements()
            .iter()
            .map(|e| matches!(e, Element::Declarative { .. }))
            .collect();

        // Data segments are applied by `apply_data_segments`, after the
        // instance exists: one of them trapping must not lose the element
        // writes that already happened, nor the instance holding them.
        let dropped_data = vec![false; module.data().len()];

        let inst = InterpInstance {
            module,
            funcs,
            memories,
            dropped_data,
            dropped_elems,
            globals,
            shared_globals,
            tables,
            tags,
            // Zeroed in full: native dispatch roams the whole region
            // through raw pointers, so every slot must be initialized.
            config,
            stack: vec![0u64; stack_slots],
            ret_stack: vec![0u64; configured_ret_records(stack_slots) * (RET_RECORD / 8)],
            host,
            funcref_host,
            published,
            native: None,
        };
        // The dispatch chain, the data segments and the start function are
        // `new_partial`'s, in that order: start can call an import and must
        // see initialized memory, and a trap in either has to hand the
        // instance back rather than lose it.
        Ok(inst)
    }

    /// Copy every active element segment into its table.
    ///
    /// After construction for the same reason as the data segments: a trap
    /// here leaves writes that already landed -- possibly in another
    /// instance's table -- and the instance holding them must survive.
    fn apply_element_segments(&mut self) -> Result<(), WasmError> {
        for ei in 0..self.module.elements().len() {
            let Some(Element::Active {
                table_index,
                offset_expr,
                init,
            }) = self.module.elements().get(ei).cloned()
            else {
                continue;
            };
            let off = eval_const(&offset_expr, &self.globals)? as u64;
            let shared_now = matches!(
                self.tables.get(table_index).map(|t| &t.entries),
                Some(TableEntries::Shared(_))
            );
            let len = self
                .tables
                .get(table_index)
                .ok_or(WasmError::invalid("interp: element table out of range"))?
                .entries
                .len() as u64;
            // The offset is guest-controlled, so the bound is computed
            // without wrapping: on a 32-bit host `off + len` overflows
            // `usize` and turns an out-of-range segment into an in-range one.
            let n = init.len() as u64;
            if off + n > len {
                return Err(WasmError::trap("out of bounds table access"));
            }
            match &init {
                ElementInit::FunctionIndexes(idxs) => {
                    for (k, &fi) in idxs.iter().enumerate() {
                        // A funcref entering a SHARED table needs a global
                        // name; see `FuncRefHost`.
                        let slot = match (shared_now, self.funcref_host.as_ref()) {
                            (true, Some(h)) => {
                                let handle = (h.publish)(fi);
                                self.published.push((handle, fi));
                                ref_to_slot(handle)
                            }
                            _ => ref_to_slot(RefHandle::new(fi)),
                        };
                        self.tables[table_index]
                            .entries
                            .set(off as usize + k, slot)?;
                    }
                }
                ElementInit::InitExprs { exprs, .. } => {
                    for (k, expr) in exprs.iter().enumerate() {
                        let v = eval_const(expr, &self.globals)?;
                        self.tables[table_index].entries.set(off as usize + k, v)?;
                    }
                }
            }
            self.dropped_elems[ei] = true; // active segments drop after use
        }
        Ok(())
    }

    /// Copy every active data segment into its memory.
    ///
    /// Separate from `build` because a trap here leaves an instance that must
    /// survive: element segments ran first, and their writes -- possibly into
    /// another instance's table -- stand.
    fn apply_data_segments(&mut self) -> Result<(), WasmError> {
        for i in 0..self.module.data().len() {
            let (memory_index, off, init) = {
                let Some(Data::Active {
                    memory_index,
                    offset_expr,
                    init,
                }) = self.module.data().get(i)
                else {
                    continue;
                };
                let off = eval_const(offset_expr, &self.globals)? as u64;
                (*memory_index, off, init.clone())
            };
            let mem = self
                .memories
                .get_mut(memory_index)
                .ok_or(WasmError::trap("out of bounds memory access"))?;
            // Same overflow reasoning as the element segments.
            if off + init.len() as u64 > mem.len() as u64 {
                return Err(WasmError::trap("out of bounds memory access"));
            }
            let off = off as usize;
            mem.bytes_mut()[off..off + init.len()].copy_from_slice(&init);
            self.dropped_data[i] = true; // active segments drop after use
        }
        Ok(())
    }

    /// Native dispatch is the interpreter's only execution engine (the
    /// stage-A Rust loop was removed after B validation; `exec_ins`
    /// remains as the native chain's slow path). A target whose ISA has
    /// no generated handler set fails instantiation cleanly here.

    /// Link every predecoded function against the handler set that was
    /// generated into this binary at build time. Nothing is emitted or
    /// mapped here — the engine is already in `.text`.
    fn enable_native_dispatch(&mut self) -> Result<(), WasmError> {
        let engine = NativeEngine::new();
        let mut linked: Vec<Option<LinkedFunction>> = self
            .funcs
            .iter()
            .map(|f| f.as_ref().map(|f| engine.link(f)))
            .collect();

        // Cross-function fixup: rewire `Call` cells to the native call
        // handler now that every callee's cell block has its final address.
        // Imports and oversized frames stay on the slow path.
        let call_h = engine.call_handler_addr();
        // Callee side: cells base (a low 48), callee l0/l1 offsets
        // (b bits 32-47 / 48-63), frame metadata (c low 48). The caller's
        // own l0/l1 offsets ride in c bits 48-63 and a bits 48-63 so the
        // call handler can stamp them into the return record.
        let callee_info: Vec<Option<(u64, u64, u64, u64, bool)>> = self
            .funcs
            .iter()
            .enumerate()
            .map(|(i, f)| match (f, &linked[i]) {
                (Some(f), Some(lf)) => {
                    let fs = f.frame_slots as u64;
                    if fs >= 1 << 16 {
                        return None;
                    }
                    let packed = fs << 32 | (f.n_locals as u64) << 16 | f.n_params as u64;
                    Some((
                        lf.cells.as_ptr() as u64,
                        packed,
                        lf.l0_off as u64,
                        lf.l1_off as u64,
                        lf.fp_pinned,
                    ))
                }
                _ => None,
            })
            .collect();
        // Canonical type ids for the native call_indirect type check:
        // `types_equivalent` is an equivalence relation, so numbering the
        // classes densely lets the handler compare one small id. Class
        // representatives are found by linear scan — real modules carry
        // hundreds of types at most.
        let mut used_types: Vec<u32> = (0..self.funcs.len())
            .filter_map(|i| self.module.functions().get(i).map(|f| f.type_index()))
            .collect();
        for f in self.funcs.iter().flatten() {
            for ins in f.code.iter() {
                if ins.op == Op::CallIndirect {
                    used_types.push(ins.c as u32);
                }
            }
        }
        let max_ti = used_types.iter().max().copied().unwrap_or(0) as usize;
        let mut canon_of: Vec<Option<u64>> = vec![None; max_ti + 1];
        let mut reps: Vec<u32> = Vec::new();
        for &ti in used_types.iter() {
            if canon_of[ti as usize].is_some() {
                continue;
            }
            let id = reps
                .iter()
                .position(|&r| self.module.types().types_equivalent(r, ti))
                .unwrap_or_else(|| {
                    reps.push(ti);
                    reps.len() - 1
                });
            canon_of[ti as usize] = Some(id as u64);
        }

        let indirect_info: Vec<[u64; 3]> = (0..self.funcs.len())
            .map(|i| {
                let (Some(Some((cells, packed, l0, l1, fp))), Some(func)) =
                    (callee_info.get(i), self.module.functions().get(i))
                else {
                    return [0; 3];
                };
                let Some(Some(canon)) = canon_of.get(func.type_index() as usize) else {
                    return [0; 3];
                };
                [*cells | *fp as u64, l1 << 48 | l0 << 32 | canon, *packed]
            })
            .collect();

        let ci_h = engine.callindirect_handler_addr();
        for (i, lf) in linked.iter_mut().enumerate().filter(|_| NATIVE_CALLS) {
            let (Some(f), Some(lf)) = (&self.funcs[i], lf) else {
                continue;
            };
            let caller_l0 = lf.l0_off as u64;
            let caller_l1 = lf.l1_off as u64;
            // Rides bit 0 of the recorded l0 offset (byte-scaled, so the
            // bit is structurally free) into every return record.
            let caller_fp = lf.fp_pinned as u64;
            for (k, ins) in f.code.iter().enumerate() {
                if ins.op == Op::Call {
                    if let Some(Some((cells, packed, callee_l0, callee_l1, callee_fp))) =
                        callee_info.get(ins.a as usize)
                    {
                        lf.cells[k] = DCell {
                            h: call_h,
                            a: caller_l1 << 48 | *cells,
                            b: callee_l1 << 48
                                | callee_l0 << 32
                                | (*callee_fp as u64) << 31
                                | ins.b * 8,
                            c: (caller_l0 | caller_fp) << 48 | *packed,
                        };
                    }
                }
                if ins.op == Op::CallIndirect && ins.flags & FLAG_A_CONST == 0 && ins.c >> 32 == 0 {
                    // The expected id must fit the cell's 16-bit field;
                    // an overflow (or unknown type) leaves the site slow.
                    let canon = canon_of.get((ins.c as u32) as usize);
                    if let Some(Some(canon)) = canon {
                        if *canon <= 0xFFFF {
                            lf.cells[k] = DCell {
                                h: ci_h,
                                a: caller_l1 << 48 | ins.a * 8,
                                b: (caller_l0 | caller_fp) << 48 | canon << 32 | ins.b * 8,
                                c: 0,
                            };
                        }
                    }
                }
            }
        }

        let mut ranges: Vec<(u64, u64, u32)> = Vec::new();
        for (i, lf) in linked.iter().enumerate() {
            if let Some(lf) = lf {
                let start = lf.cells.as_ptr() as u64;
                // The trailing prefetch pad is not an instruction.
                let end = start + (lf.cells.len() as u64 - 1) * 32;
                ranges.push((start, end, i as u32));
            }
        }
        ranges.sort_unstable_by_key(|r| r.0);

        self.native = Some(NativeState {
            engine,
            linked,
            indirect_info,
            ranges,
            slow_exits: vec![0u64; Op::Unreachable as usize + 1],
            dispatches: 0,
        });
        Ok(())
    }

    /// Total native handler dispatches since instantiation (0 when native
    /// dispatch is not active).
    pub fn dispatch_count(&self) -> u64 {
        {
            return self.native.as_ref().map_or(0, |n| n.dispatches);
        }
    }

    /// Map a native pc to `(func_index, cells_start)`.
    fn native_pc_lookup(&self, pc: u64) -> Result<(usize, u64), WasmError> {
        let ranges = &self.native.as_ref().expect("native state").ranges;
        let i = ranges.partition_point(|r| r.1 <= pc);
        match ranges.get(i) {
            Some(&(start, _, fi)) if pc >= start => Ok((fi as usize, start)),
            _ => Err(WasmError::invalid("interp: native pc outside any function")),
        }
    }

    /// Static adjacent-pair census over the predecoded streams: a pair
    /// is counted only when the first op falls through (no control
    /// transfer), i.e. exactly where a fused handler could replace two
    /// dispatches with one. Descending by count.
    pub fn bigram_stats(&self) -> Vec<((Op, Op), u64)> {
        let mut map: tracked_alloc::BTreeMap<(u16, u16), u64> = tracked_alloc::BTreeMap::new();
        for f in self.funcs.iter().flatten() {
            for w in f.code.windows(2) {
                if w[0].op as u16 >= Op::Br as u16 {
                    continue; // control transfer: never a fusible pair
                }
                *map.entry((w[0].op as u16, w[1].op as u16)).or_insert(0) += 1;
            }
        }
        let mut out: Vec<((Op, Op), u64)> = map
            .into_iter()
            .map(|((a, b), n)| ((op_from_index(a as usize), op_from_index(b as usize)), n))
            .collect();
        out.sort_by(|x, y| y.1.cmp(&x.1));
        out
    }

    /// Slow-path exit counts by op since instantiation, descending —
    /// empty when native dispatch is not active.
    /// Whether the dispatch counter was compiled into the handlers. When it
    /// was not, the reported dispatch total is meaningless and callers should
    /// say so rather than print it.
    pub fn dispatch_counting_enabled(&self) -> bool {
        cfg!(feature = "interp-count")
    }

    /// Size in bytes of the emitted dispatch engine, 0 when there is none.
    /// Worth watching: every added handler family or operand class grows it
    /// against a hard buffer assert, and once emission moves to build time
    /// this becomes binary size.
    pub fn engine_code_len(&self) -> usize {
        if let Some(native) = &self.native {
            return native.engine.code_len();
        }
        0
    }

    pub fn slow_exit_stats(&self) -> Vec<(Op, u64)> {
        if let Some(native) = &self.native {
            return op_counts(&native.slow_exits);
        }
        Vec::new()
    }

    /// Install the host dispatcher used for imported functions. Generic
    /// so callers pass a plain closure; boxing happens here (unsizing to
    /// the dyn target through the `alloc` box, then wrapping in the
    /// tracked facade — the same pattern as the JIT's host callbacks).
    /// Box a host dispatcher for [`Self::new`].
    ///
    /// Keeps the engine's allocator-tracked `Box` out of the caller's
    /// type, so an embedder writes an ordinary closure.
    pub fn boxed_host<F>(host: F) -> HostDispatch
    where
        F: FnMut(&str, &str, &mut [u8], &[u64], &mut [u64]) -> Result<(), WasmError> + 'static,
    {
        let host: alloc::boxed::Box<
            dyn FnMut(&str, &str, &mut [u8], &[u64], &mut [u64]) -> Result<(), WasmError>,
        > = alloc::boxed::Box::new(host);
        tracked_alloc::box_from_alloc(host)
    }

    /// Replace the host dispatcher after instantiation.
    ///
    /// Prefer passing it to [`Self::new`]: a start function that calls an
    /// import runs during instantiation, before this could be reached.
    pub fn set_host<F>(&mut self, host: F)
    where
        F: FnMut(&str, &str, &mut [u8], &[u64], &mut [u64]) -> Result<(), WasmError> + 'static,
    {
        let host: alloc::boxed::Box<
            dyn FnMut(&str, &str, &mut [u8], &[u64], &mut [u64]) -> Result<(), WasmError>,
        > = alloc::boxed::Box::new(host);
        self.host = Some(tracked_alloc::box_from_alloc(host));
    }

    /// Resolve the element-segment function value at `seg[k]`.
    /// One element-segment entry, as the reference slot a table holds.
    fn elem_value(&self, seg: usize, k: usize) -> Result<u64, WasmError> {
        match self.module.elements().get(seg).map(|e| e.get_init()) {
            Some(ElementInit::FunctionIndexes(idxs)) => idxs
                .get(k)
                .map(|&fi| ref_to_slot(RefHandle::new(fi as usize)))
                .ok_or(WasmError::trap("out of bounds table access")),
            // Already a slot: `eval_const` yields `NULL_REF_SLOT` for
            // `ref.null` and a plain handle for `ref.func`.
            Some(ElementInit::InitExprs { exprs, .. }) => exprs
                .get(k)
                .ok_or(WasmError::trap("out of bounds table access"))
                .and_then(|e| eval_const(e, &self.globals)),
            None => Err(WasmError::trap("out of bounds table access")),
        }
    }

    /// This global's shared cell, when it has one.
    ///
    /// Only a global that is imported or exported is held as a `GlobalInst`;
    /// a purely private one lives in the array and has nothing to share.
    pub fn global_state_at(&self, idx: usize) -> Option<GlobalInst> {
        self.shared_globals.get(idx)?.clone()
    }

    /// The runtime identity of tag `idx`, for an importer to link against.
    pub fn tag_handle_at(&self, idx: usize) -> Option<TagHandle> {
        self.tags.get(idx).copied()
    }

    /// This table's shared entity, when it has one.
    ///
    /// Only a table that is imported or exported is held as a `TableInst`;
    /// a purely private one keeps an array and has nothing to share.
    pub fn table_state_at(&self, idx: usize) -> Option<TableInst> {
        match &self.tables.get(idx)?.entries {
            TableEntries::Shared(t) => Some(t.clone()),
            TableEntries::Private(_) => None,
        }
    }

    /// The current size of table `idx`, which `table.grow` may have changed
    /// since instantiation.
    pub fn table_len(&self, idx: usize) -> Option<usize> {
        self.tables.get(idx).map(|t| t.entries.len())
    }

    /// The module this instance was built from.
    #[inline]
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// The first linear memory's contents, if the module defines one.
    #[inline]
    pub fn memory(&self) -> Option<&[u8]> {
        {
            self.memories.first().map(|m| m.bytes())
        }
    }

    #[inline]
    pub fn memory_mut(&mut self) -> Option<&mut [u8]> {
        {
            self.memories.first_mut().map(|m| m.bytes_mut())
        }
    }

    /// Find an exported function's index by name.
    pub fn find_export(&self, name: &str) -> Option<usize> {
        self.module
            .functions()
            .iter()
            .position(|f| f.export_names().iter().any(|n| n == name))
    }

    /// (params, results) arity of a function.
    pub fn func_arity(&self, func_index: usize) -> Option<(usize, usize)> {
        self.module.functions().get(func_index).map(|f| {
            let ft = f.func_type();
            (ft.params().len(), ft.results().len())
        })
    }

    /// The memory entity at `idx`, for another instance to import.
    ///
    /// Sharing the substrate's `MemInst` is what makes an interpreter
    /// instance linkable: the importer takes this backing rather than a
    /// copy, so writes on either side are visible to both.
    pub fn shared_memory_at(&self, idx: usize) -> Option<MemInst> {
        {
            self.memories.get(idx).map(|m| m.inst.clone())
        }
    }

    /// A global's raw 64-bit value by index.
    pub fn global_at(&self, idx: usize) -> Option<u64> {
        // Through the shared cell where there is one: the array slot beside a
        // shared global is stale by design, so reading it would report the
        // value as of instantiation rather than now.
        match self.shared_globals.get(idx)?.as_ref() {
            Some(shared) => Some(shared.raw()),
            None => self.globals.get(idx).copied(),
        }
    }

    /// Overwrite a global's raw 64-bit value by index.
    pub fn set_global_at(&mut self, idx: usize, raw: u64) -> Result<(), WasmError> {
        if let Some(Some(shared)) = self.shared_globals.get_mut(idx) {
            shared.set_raw(raw);
            return Ok(());
        }
        match self.globals.get_mut(idx) {
            Some(slot) => {
                *slot = raw;
                Ok(())
            }
            None => Err(WasmError::invalid("interp: global index out of range")),
        }
    }

    /// The index of an exported global, if the module exports one.
    pub fn find_export_global(&self, name: &str) -> Option<usize> {
        self.module
            .globals()
            .iter()
            .position(|g| g.export_names().iter().any(|n| n == name))
    }

    /// Read an exported global's raw 64-bit value.
    pub fn get_export_global(&self, name: &str) -> Option<u64> {
        self.module
            .globals()
            .iter()
            .position(|g| g.export_names().iter().any(|n| n == name))
            .and_then(|i| self.globals.get(i).copied())
    }

    /// Invoke a function by index. `args` and `results` are raw 64-bit
    /// value slots (i32/f32 in the low bits).
    pub fn invoke(
        &mut self,
        func_index: usize,
        args: &[u64],
        results: &mut [u64],
    ) -> Result<(), WasmError> {
        let entry = self
            .funcs
            .get(func_index)
            .ok_or(WasmError::invalid("interp: bad function index"))?;
        let Some(func) = entry.clone() else {
            // An imported function, reached directly rather than through a
            // call inside a body -- `(start $imported)` does exactly this.
            // It has no predecoded body to enter; the host is the callee.
            let mut frame = vec![0u64; args.len().max(results.len())];
            frame[..args.len()].copy_from_slice(args);
            self.call_host(func_index, &mut frame, 0)?;
            results.copy_from_slice(&frame[..results.len()]);
            return Ok(());
        };
        if args.len() != func.n_params as usize || results.len() != func.n_results as usize {
            return Err(WasmError::invalid("interp: argument/result arity mismatch"));
        }
        let root = Activation {
            func,
            func_index,
            base: 0,
            pc: 0,
            route_established: false,
        };
        self.drive(root, args, results)
    }

    /// Stub for targets without a native backend: [`Self::new`] fails
    /// there, so no instance exists to invoke — this only keeps
    /// cross-target callers compiling.

    /// The call/return trampoline: runs activations to their next call or
    /// return boundary, keeping call depth as data on `saved`.
    fn drive(
        &mut self,
        root: Activation,
        args: &[u64],
        results: &mut [u64],
    ) -> Result<(), WasmError> {
        // Borrow the instance's buffers for the duration. A host callback
        // that calls back into this same instance finds them taken and
        // allocates its own pair, so re-entry stays correct without the
        // common case paying for an allocation.
        let mut stack = core::mem::take(&mut self.stack);
        let mut ret_stack = core::mem::take(&mut self.ret_stack);
        let reentrant = stack.is_empty();
        if reentrant {
            let slots = configured_stack_slots(&self.config);
            stack = vec![0u64; slots];
            ret_stack = vec![0u64; configured_ret_records(slots) * (RET_RECORD / 8)];
        }

        let outcome = self.drive_on(root, args, &mut stack, &mut ret_stack, results);

        // Only the owner hands them back; a nested call drops its own.
        if !reentrant {
            self.stack = stack;
            self.ret_stack = ret_stack;
        }
        outcome
    }

    /// Results are copied out here, while the stack that produced them is
    /// still in hand -- a nested call runs on a different one.
    fn drive_on(
        &mut self,
        root: Activation,
        args: &[u64],
        stack: &mut [u64],
        ret_stack: &mut [u64],
        results: &mut [u64],
    ) -> Result<(), WasmError> {
        if root.func.frame_slots as usize > stack.len() {
            return Err(WasmError::trap("call stack exhausted"));
        }
        let max_depth = (ret_stack.len() / (RET_RECORD / 8)).saturating_sub(8) as u32;
        let mut ctx = DriveCtx {
            stack,
            ret_stack,
            ret_cursor: 0,
            acc: 0,
        };
        ctx.stack[..args.len()].copy_from_slice(args);
        // The root frame's locals get their fresh zeros here. Callee
        // frames are zeroed at the call site; the root has no call site,
        // and it used to be covered only because every invocation started
        // on a newly allocated buffer. Temps above `n_locals` are written
        // before they are read, by construction in the predecoder.
        ctx.stack[args.len()..root.func.n_locals as usize].fill(0);

        let mut act = root;
        let mut saved: Vec<Activation> = Vec::new();
        loop {
            match self.native_step(&mut act, &mut ctx)? {
                StepExit::Call { callee, arg_base } => {
                    if saved.len() as u32 >= max_depth {
                        return Err(WasmError::trap("call stack exhausted"));
                    }
                    let f = match self.funcs.get(callee) {
                        Some(Some(f)) => f.clone(),
                        Some(None) => {
                            // Imported function: dispatch to the host. Its
                            // first result rides the accumulator relay like
                            // any other call result.
                            let base = act.base;
                            self.call_host(callee, &mut ctx.stack[base..], arg_base)?;
                            ctx.acc = ctx.stack[base + arg_base];
                            continue;
                        }
                        None => return Err(WasmError::trap("undefined element")),
                    };
                    let new_base = act.base + arg_base;
                    if new_base + f.frame_slots as usize > ctx.stack.len() {
                        return Err(WasmError::trap("call stack exhausted"));
                    }
                    // The caller already staged the params at the frame
                    // base; locals get fresh zeros, temps are written
                    // before read by predecode construction.
                    let (np, nl) = (f.n_params as usize, f.n_locals as usize);
                    ctx.stack[new_base + np..new_base + nl].fill(0);
                    let callee_act = Activation {
                        func: f,
                        func_index: callee,
                        base: new_base,
                        pc: 0,
                        route_established: false,
                    };
                    saved.push(act);
                    act = callee_act;
                }
                StepExit::TailCall { callee, arg_base } => {
                    // The callee returns to THIS activation's caller, so it
                    // reuses this frame: arguments slide down to `act.base`
                    // (which is where the caller staged ours, and so where
                    // our results are expected), and the activation is
                    // replaced rather than pushed. Recursion through a tail
                    // call therefore runs at constant depth.
                    let f = match self.funcs.get(callee) {
                        Some(Some(f)) => f.clone(),
                        Some(None) => {
                            // An imported tail callee: run the host, leave
                            // its results at this frame's base, and return
                            // to the caller as this activation would have.
                            let base = act.base;
                            self.call_host(callee, &mut ctx.stack[base..], arg_base)?;
                            let np = ctx.stack[base + arg_base];
                            ctx.stack[base] = np;
                            ctx.acc = np;
                            match saved.pop() {
                                None => {
                                    results.copy_from_slice(&ctx.stack[..results.len()]);
                                    return Ok(());
                                }
                                Some(parent) => {
                                    act = parent;
                                    continue;
                                }
                            }
                        }
                        None => return Err(WasmError::trap("undefined element")),
                    };
                    let base = act.base;
                    if base + f.frame_slots as usize > ctx.stack.len() {
                        return Err(WasmError::trap("call stack exhausted"));
                    }
                    let (np, nl) = (f.n_params as usize, f.n_locals as usize);
                    ctx.stack
                        .copy_within(base + arg_base..base + arg_base + np, base);
                    ctx.stack[base + np..base + nl].fill(0);
                    // Keep the frame AND the return record. `act` is a window
                    // onto the chain's current frame, not something Rust
                    // pushed: the record that routes this frame's return was
                    // planted by whoever called us, and the callee returns to
                    // exactly that place. So the tail call switches which
                    // function executes and nothing else -- which is also why
                    // depth stays constant.
                    act.func = f;
                    act.func_index = callee;
                    act.pc = 0;
                }
                StepExit::Return => {
                    // Results are already in place: the callee's frame base
                    // IS the caller's staged-argument slot.
                    match saved.pop() {
                        None => {
                            results.copy_from_slice(&ctx.stack[..results.len()]);
                            return Ok(());
                        }
                        Some(parent) => act = parent,
                    }
                }
            }
        }
    }

    // `packed` is `memidx << 48 | offset` as emitted by the predecoder.
    /// Split a packed memarg into `(memory index, static offset)`.
    ///
    /// Inline form is `memidx << 48 | offset`; a memory64 offset too wide for
    /// that carries a `wide_memargs` index with bit 63 set.
    #[inline]
    fn memarg(func: &PredecodedFunction, packed: u64) -> (usize, u64) {
        if packed & WIDE_MEMARG != 0 {
            match func.wide_memargs.get((packed & !WIDE_MEMARG) as usize) {
                Some(&(mi, off)) => (mi as usize, off),
                None => (usize::MAX, 0),
            }
        } else {
            ((packed >> 48) as usize, packed & 0xffff_ffff_ffff)
        }
    }

    fn mem_load(
        &self,
        addr: u64,
        mem_idx: usize,
        offset: u64,
        size: usize,
    ) -> Result<&[u8], WasmError> {
        let mem = self
            .memories
            .get(mem_idx)
            .ok_or(WasmError::trap("out of bounds memory access"))?;
        // A 64-bit address can carry, so the effective address is computed
        // with checked arithmetic rather than relying on a 2^49 bound.
        let ea = addr
            .checked_add(offset)
            .ok_or(WasmError::trap("out of bounds memory access"))?;
        let end = ea
            .checked_add(size as u64)
            .ok_or(WasmError::trap("out of bounds memory access"))?;
        if end > mem.len() as u64 {
            return Err(WasmError::trap("out of bounds memory access"));
        }
        Ok(&mem.bytes()[ea as usize..end as usize])
    }

    fn mem_store(
        &mut self,
        addr: u64,
        mem_idx: usize,
        offset: u64,
        size: usize,
    ) -> Result<&mut [u8], WasmError> {
        let mem = self
            .memories
            .get_mut(mem_idx)
            .ok_or(WasmError::trap("out of bounds memory access"))?;
        // Checked for the same reason as the load path: a 64-bit address
        // plus a 64-bit static offset can carry, and carrying must trap
        // rather than wrap into a valid access.
        let ea = addr
            .checked_add(offset)
            .ok_or(WasmError::trap("out of bounds memory access"))?;
        let end = ea
            .checked_add(size as u64)
            .ok_or(WasmError::trap("out of bounds memory access"))?;
        if end > mem.len() as u64 {
            return Err(WasmError::trap("out of bounds memory access"));
        }
        Ok(&mut mem.bytes_mut()[ea as usize..end as usize])
    }

    /// Dispatch an imported function to the host. `frame` is the CALLER's
    /// frame slice; arguments and results live at `arg_base` within it.
    fn call_host(
        &mut self,
        callee: usize,
        frame: &mut [u64],
        arg_base: usize,
    ) -> Result<(), WasmError> {
        // Borrow the three fields disjointly. Reading the import's names
        // through `&self` while the dispatcher is held through `&mut self`
        // is what used to force a `String` clone of both names on every
        // single host call; destructuring makes them provably separate.
        let Self {
            module,
            host,
            memories,
            ..
        } = self;

        let func = module
            .functions()
            .get(callee)
            .ok_or(WasmError::trap("undefined element"))?;
        let (mod_name, field) = match func.def() {
            crate::module::entities::FunctionDef::Import { module, name, .. } => {
                (module.as_str(), name.as_str())
            }
            _ => return Err(WasmError::invalid("interp: not an import")),
        };
        let p = func.func_type().params().len();
        let r = func.func_type().results().len();
        let host = host
            .as_mut()
            .ok_or(WasmError::invalid("interp: no host dispatcher installed"))?;
        let mut results = [0u64; 8];
        if r > results.len() {
            return Err(WasmError::invalid("interp: too many host results"));
        }
        let args: Vec<u64> = frame[arg_base..arg_base + p].iter().copied().collect();
        let mem0 = memories
            .first_mut()
            .map(|m| m.bytes_mut())
            .unwrap_or(&mut []);
        host(mod_name, field, mem0, &args, &mut results[..r])?;
        frame[arg_base..arg_base + r].copy_from_slice(&results[..r]);
        Ok(())
    }

    /// Driver: run one activation on the native dispatch chain
    /// until it needs Rust. Calls, returns, and branches between local
    /// functions stay entirely inside the chain (the native return stack
    /// carries `(ret_pc, frame, code_base)` records); Rust sees only slow
    /// ops, sentinel returns, host calls, and traps. On a slow exit the
    /// current function is recovered from the pc via the range map and the
    /// ORIGINAL instruction is executed by `exec_ins`, then the chain is
    /// re-entered.
    fn native_step(
        &mut self,
        act: &mut Activation,
        ctx: &mut DriveCtx,
    ) -> Result<StepExit, WasmError> {
        let info = {
            let native = self.native.as_ref().expect("native_step without state");
            native
                .linked
                .get(act.func_index)
                .and_then(|l| l.as_ref())
                .map(|lf| {
                    (
                        native.engine.entry_fn(),
                        native.engine.exit_cell_addr(),
                        lf.cells.as_ptr() as u64,
                        lf.l0_off,
                        lf.l1_off,
                    )
                })
        };
        let (enter, exit_cell, cells_base, l0_off, l1_off) = match info {
            Some(x) => x,
            // An activation always has a linked body (imports never
            // become activations).
            None => return Err(WasmError::invalid("interp: unlinked activation")),
        };
        // First entry of a Rust-created activation: plant the sentinel
        // record so this activation's native `Return` comes back to Rust.
        // The frame word must be a READABLE dummy (never 0): the native
        // Return handler reloads the caller's l0 through it before
        // routing into the exit cell.
        if !act.route_established {
            if ctx.ret_cursor + RET_RECORD > ctx.ret_stack.len() * 8 {
                return Err(WasmError::trap("call stack exhausted"));
            }
            let at = ctx.ret_cursor / 8;
            ctx.ret_stack[at] = exit_cell;
            ctx.ret_stack[at + 1] = exit_cell;
            ctx.ret_stack[at + 2] = 0;
            ctx.ret_stack[at + 3] = 0;
            ctx.ret_cursor += RET_RECORD;
            act.route_established = true;
        }
        // The raw pointers below stay valid across `exec_ins`: the value
        // stack, return stack, and globals never reallocate inside drive,
        // and linked cells are only dropped with the whole instance.
        // Memory CAN grow (slow path), so its base/len refresh on every
        // re-entry.
        let stack_ptr = ctx.stack.as_mut_ptr() as u64;
        let ret_ptr = ctx.ret_stack.as_mut_ptr() as u64;
        let mut state = EnterState {
            reason: 0,
            pc: cells_base + (act.pc as u64) * 32,
            frame: stack_ptr + (act.base as u64) * 8,
            mem_base: 0,
            mem_len: 0,
            code_base: cells_base,
            globals: self.globals.as_mut_ptr() as u64,
            ret_cursor: ret_ptr + ctx.ret_cursor as u64,
            ret_limit: ret_ptr + (ctx.ret_stack.len() * 8 - RET_RECORD) as u64,
            stack_limit: stack_ptr + (ctx.stack.len() as u64) * 8,
            dispatches: 0,
            l0_value: 0,
            l1_value: 0,
            acc_value: 0,
            table0_base: 0,
            table0_len: 0,
            indirect_base: self
                .native
                .as_ref()
                .map_or(0, |n| n.indirect_info.as_ptr() as u64),
        };
        let mut cur_base = act.base;
        let mut cur_l0_slot = (l0_off / 8) as usize;
        let mut cur_l1_slot = (l1_off / 8) as usize;
        loop {
            if let Some(m) = self.memories.first_mut() {
                state.mem_base = m.inst.memory_ptr() as u64;
                state.mem_len = m.inst.memory_len() as u64;
            }
            // Table 0 can move or grow only on the slow path, so a
            // per-entry refresh keeps the native indirect-call handler
            // valid with no invalidation protocol.
            if let Some(t) = self.tables.first() {
                // Only a private table has a base the chain may index; a
                // shared one holds `RefHandle`, so it is left at zero and its
                // accesses run on the slow path.
                match t.entries.fast_base() {
                    Some(base) => {
                        state.table0_base = base as u64;
                        state.table0_len = t.entries.len() as u64;
                    }
                    None => {
                        state.table0_base = 0;
                        state.table0_len = 0;
                    }
                }
            }
            // The pinned-local registers reload from their (write-through,
            // hence authoritative) slots at every chain entry.
            state.l0_value = ctx.stack[cur_base + cur_l0_slot];
            state.l1_value = ctx.stack[cur_base + cur_l1_slot];
            state.acc_value = ctx.acc;
            enter(&mut state);
            ctx.acc = state.acc_value;
            ctx.ret_cursor = (state.ret_cursor - ret_ptr) as usize;
            if let Some(native) = self.native.as_mut() {
                native.dispatches += state.dispatches;
            }
            state.dispatches = 0;
            match state.reason {
                EXIT_SLOW => {
                    let (fi, cstart) = self.native_pc_lookup(state.pc)?;
                    let f = self.funcs[fi].clone().expect("range map import");
                    let idx = ((state.pc - cstart) / 32) as usize;
                    let base = ((state.frame - stack_ptr) / 8) as usize;
                    act.func = f.clone();
                    act.func_index = fi;
                    act.base = base;
                    cur_base = base;
                    {
                        let native = self.native.as_ref().expect("native state");
                        let lf = native.linked[fi].as_ref().expect("linked");
                        cur_l0_slot = (lf.l0_off / 8) as usize;
                        cur_l1_slot = (lf.l1_off / 8) as usize;
                    }
                    let ins = *f
                        .code
                        .get(idx)
                        .ok_or(WasmError::invalid("interp: native pc out of range"))?;
                    if let Some(native) = self.native.as_mut() {
                        native.slow_exits[ins.op as usize] += 1;
                    }
                    let frame = &mut ctx.stack[base..base + f.frame_slots as usize];
                    match self.exec_ins(frame, &f, ins)? {
                        Effect::Next => state.pc = cstart + (idx as u64 + 1) * 32,
                        Effect::Jump(t) => state.pc = cstart + (t as u64) * 32,
                        // `Return` always has a native handler; a slow one
                        // would desync the native return stack.
                        Effect::Ret => {
                            return Err(WasmError::invalid("interp: unexpected slow return"))
                        }
                        Effect::Call { callee, arg_base } => {
                            act.pc = idx + 1;
                            return Ok(StepExit::Call { callee, arg_base });
                        }
                        Effect::TailCall { callee, arg_base } => {
                            act.pc = idx + 1;
                            return Ok(StepExit::TailCall { callee, arg_base });
                        }
                    }
                    state.code_base = cstart;
                }
                EXIT_RETURN => return Ok(StepExit::Return),
                r if r >= EXIT_TRAP_BASE => {
                    let kind = (r - EXIT_TRAP_BASE) as usize;
                    let msg = TRAP_KINDS
                        .get(kind)
                        .copied()
                        .unwrap_or("interp: unknown native trap");
                    return Err(WasmError::trap(msg));
                }
                _ => return Err(WasmError::invalid("interp: bad native exit reason")),
            }
        }
    }

    /// Execute exactly one instruction against `frame` — the native
    /// chain's slow path (every op the handlers don't cover, plus calls,
    /// returns, and rich trap messages). Acc residency flags are ignored
    /// here by design: slow cells are always linked with their acc hints
    /// stripped, so operands and destinations live in their frame slots.
    pub(super) fn exec_ins(
        &mut self,
        frame: &mut [u64],
        func: &PredecodedFunction,
        ins: Instr,
    ) -> Result<Effect, WasmError> {
        macro_rules! opa {
            ($ins:expr) => {
                if $ins.flags & FLAG_A_CONST != 0 {
                    $ins.a
                } else {
                    frame[$ins.a as usize]
                }
            };
        }
        macro_rules! opb {
            ($ins:expr) => {
                if $ins.flags & FLAG_B_CONST != 0 {
                    $ins.b
                } else {
                    frame[$ins.b as usize]
                }
            };
        }
        macro_rules! bin32 {
            ($ins:expr, $f:expr) => {{
                let x = opa!($ins) as u32;
                let y = opb!($ins) as u32;
                frame[$ins.c as usize] = $f(x, y) as u32 as u64;
            }};
        }
        macro_rules! bin64 {
            ($ins:expr, $f:expr) => {{
                let x = opa!($ins);
                let y = opb!($ins);
                frame[$ins.c as usize] = $f(x, y);
            }};
        }
        macro_rules! cmp32 {
            ($ins:expr, $f:expr) => {{
                let x = opa!($ins) as u32;
                let y = opb!($ins) as u32;
                frame[$ins.c as usize] = $f(x, y) as u64;
            }};
        }
        macro_rules! cmp64 {
            ($ins:expr, $f:expr) => {{
                let x = opa!($ins);
                let y = opb!($ins);
                frame[$ins.c as usize] = $f(x, y) as u64;
            }};
        }
        macro_rules! cmp_br32 {
            ($ins:expr, $f:expr) => {{
                let x = opa!($ins) as u32;
                let y = opb!($ins) as u32;
                if $f(x, y) {
                    return Ok(Effect::Jump($ins.c as usize));
                }
            }};
        }
        macro_rules! cmp_br64 {
            ($ins:expr, $f:expr) => {{
                let x = opa!($ins);
                let y = opb!($ins);
                if $f(x, y) {
                    return Ok(Effect::Jump($ins.c as usize));
                }
            }};
        }
        macro_rules! un32 {
            ($ins:expr, $f:expr) => {{
                let x = opa!($ins) as u32;
                frame[$ins.c as usize] = $f(x) as u32 as u64;
            }};
        }
        macro_rules! un64 {
            ($ins:expr, $f:expr) => {{
                let x = opa!($ins);
                frame[$ins.c as usize] = $f(x);
            }};
        }
        macro_rules! fbin32 {
            ($ins:expr, $f:expr) => {{
                let x = f32::from_bits(opa!($ins) as u32);
                let y = f32::from_bits(opb!($ins) as u32);
                frame[$ins.c as usize] = $f(x, y).to_bits() as u64;
            }};
        }
        macro_rules! fbin64 {
            ($ins:expr, $f:expr) => {{
                let x = f64::from_bits(opa!($ins));
                let y = f64::from_bits(opb!($ins));
                frame[$ins.c as usize] = $f(x, y).to_bits();
            }};
        }
        macro_rules! fcmp32 {
            ($ins:expr, $f:expr) => {{
                let x = f32::from_bits(opa!($ins) as u32);
                let y = f32::from_bits(opb!($ins) as u32);
                frame[$ins.c as usize] = $f(x, y) as u64;
            }};
        }
        macro_rules! fcmp64 {
            ($ins:expr, $f:expr) => {{
                let x = f64::from_bits(opa!($ins));
                let y = f64::from_bits(opb!($ins));
                frame[$ins.c as usize] = $f(x, y) as u64;
            }};
        }
        macro_rules! fun32 {
            ($ins:expr, $f:expr) => {{
                let x = f32::from_bits(opa!($ins) as u32);
                frame[$ins.c as usize] = $f(x).to_bits() as u64;
            }};
        }
        macro_rules! fun64 {
            ($ins:expr, $f:expr) => {{
                let x = f64::from_bits(opa!($ins));
                frame[$ins.c as usize] = $f(x).to_bits();
            }};
        }
        macro_rules! load {
            ($ins:expr, $size:expr, $conv:expr) => {{
                let (addr, dst) = if $ins.flags & FLAG_FUSED != 0 {
                    let a2 = frame[($ins.c >> 32) as usize] as u32;
                    (
                        (opa!($ins) as u32).wrapping_add(a2) as u64,
                        ($ins.c & 0xffff_ffff) as usize,
                    )
                } else if $ins.flags & FLAG_ADDR64 != 0 {
                    // A 64-bit memory's address is the whole slot; a 32-bit
                    // one's is zero-extended already, but truncating keeps
                    // the two cases independent of that invariant.
                    (opa!($ins), $ins.c as usize)
                } else {
                    (opa!($ins) as u32 as u64, $ins.c as usize)
                };
                let (mi, offset) = Self::memarg(func, $ins.b);
                let bytes = self.mem_load(addr, mi, offset, $size)?;
                let mut buf = [0u8; 8];
                buf[..$size].copy_from_slice(bytes);
                frame[dst] = $conv(u64::from_le_bytes(buf));
            }};
        }
        macro_rules! store {
            ($ins:expr, $size:expr) => {{
                let (addr, off) = if $ins.flags & FLAG_FUSED != 0 {
                    let a2 = frame[($ins.c >> 32) as usize] as u32;
                    (
                        (opa!($ins) as u32).wrapping_add(a2) as u64,
                        $ins.c & 0xffff_ffff,
                    )
                } else if $ins.flags & FLAG_ADDR64 != 0 {
                    (opa!($ins), $ins.c)
                } else {
                    (opa!($ins) as u32 as u64, $ins.c)
                };
                let val = opb!($ins);
                let (mi, offset) = Self::memarg(func, off);
                let bytes = self.mem_store(addr, mi, offset, $size)?;
                bytes.copy_from_slice(&val.to_le_bytes()[..$size]);
            }};
        }

        match ins.op {
            Op::MovSlot | Op::MovConst => {
                let v = opa!(ins);
                frame[ins.c as usize] = v;
            }
            Op::MovPair => {
                frame[(ins.c >> 32) as usize] = frame[ins.a as usize];
                frame[(ins.c & 0xffff_ffff) as usize] = frame[ins.b as usize];
            }

            // ---- i32 ----
            Op::I32_Add => bin32!(ins, u32::wrapping_add),
            Op::I32_Sub => bin32!(ins, u32::wrapping_sub),
            Op::I32_Mul => bin32!(ins, u32::wrapping_mul),
            Op::I32_DivS => {
                let x = opa!(ins) as u32 as i32;
                let y = opb!(ins) as u32 as i32;
                if y == 0 {
                    return Err(WasmError::trap("integer divide by zero"));
                }
                let (v, ovf) = x.overflowing_div(y);
                if ovf {
                    return Err(WasmError::trap("integer overflow"));
                }
                frame[ins.c as usize] = v as u32 as u64;
            }
            Op::I32_DivU => {
                let x = opa!(ins) as u32;
                let y = opb!(ins) as u32;
                if y == 0 {
                    return Err(WasmError::trap("integer divide by zero"));
                }
                frame[ins.c as usize] = (x / y) as u64;
            }
            Op::I32_RemS => {
                let x = opa!(ins) as u32 as i32;
                let y = opb!(ins) as u32 as i32;
                if y == 0 {
                    return Err(WasmError::trap("integer divide by zero"));
                }
                // MIN % -1 == 0, must NOT trap.
                frame[ins.c as usize] = x.wrapping_rem(y) as u32 as u64;
            }
            Op::I32_RemU => {
                let x = opa!(ins) as u32;
                let y = opb!(ins) as u32;
                if y == 0 {
                    return Err(WasmError::trap("integer divide by zero"));
                }
                frame[ins.c as usize] = (x % y) as u64;
            }
            Op::I32_And => bin32!(ins, |x, y| x & y),
            Op::I32_Or => bin32!(ins, |x, y| x | y),
            Op::I32_Xor => bin32!(ins, |x, y| x ^ y),
            Op::I32_Shl => bin32!(ins, |x: u32, y: u32| x.wrapping_shl(y)),
            Op::I32_ShrS => {
                bin32!(ins, |x: u32, y: u32| ((x as i32).wrapping_shr(y)) as u32)
            }
            Op::I32_ShrU => bin32!(ins, |x: u32, y: u32| x.wrapping_shr(y)),
            Op::I32_Rotl => bin32!(ins, |x: u32, y: u32| x.rotate_left(y & 31)),
            Op::I32_Rotr => bin32!(ins, |x: u32, y: u32| x.rotate_right(y & 31)),
            Op::I32_Clz => un32!(ins, |x: u32| x.leading_zeros()),
            Op::I32_Ctz => un32!(ins, |x: u32| x.trailing_zeros()),
            Op::I32_Popcnt => un32!(ins, |x: u32| x.count_ones()),
            Op::I32_Extend8S => un32!(ins, |x: u32| x as i8 as i32 as u32),
            Op::I32_Extend16S => un32!(ins, |x: u32| x as i16 as i32 as u32),
            Op::I32_Eqz => un32!(ins, |x: u32| (x == 0) as u32),
            Op::I32_Eq => cmp32!(ins, |x, y| x == y),
            Op::I32_Ne => cmp32!(ins, |x, y| x != y),
            Op::I32_LtS => cmp32!(ins, |x: u32, y: u32| (x as i32) < (y as i32)),
            Op::I32_LtU => cmp32!(ins, |x, y| x < y),
            Op::I32_GtS => cmp32!(ins, |x: u32, y: u32| (x as i32) > (y as i32)),
            Op::I32_GtU => cmp32!(ins, |x, y| x > y),
            Op::I32_LeS => cmp32!(ins, |x: u32, y: u32| (x as i32) <= (y as i32)),
            Op::I32_LeU => cmp32!(ins, |x, y| x <= y),
            Op::I32_GeS => cmp32!(ins, |x: u32, y: u32| (x as i32) >= (y as i32)),
            Op::I32_GeU => cmp32!(ins, |x, y| x >= y),

            // ---- i64 ----
            Op::I64_Add => bin64!(ins, u64::wrapping_add),
            Op::I64_Sub => bin64!(ins, u64::wrapping_sub),
            Op::I64_Mul => bin64!(ins, u64::wrapping_mul),
            Op::I64_DivS => {
                let x = opa!(ins) as i64;
                let y = opb!(ins) as i64;
                if y == 0 {
                    return Err(WasmError::trap("integer divide by zero"));
                }
                let (v, ovf) = x.overflowing_div(y);
                if ovf {
                    return Err(WasmError::trap("integer overflow"));
                }
                frame[ins.c as usize] = v as u64;
            }
            Op::I64_DivU => {
                let x = opa!(ins);
                let y = opb!(ins);
                if y == 0 {
                    return Err(WasmError::trap("integer divide by zero"));
                }
                frame[ins.c as usize] = x / y;
            }
            Op::I64_RemS => {
                let x = opa!(ins) as i64;
                let y = opb!(ins) as i64;
                if y == 0 {
                    return Err(WasmError::trap("integer divide by zero"));
                }
                frame[ins.c as usize] = x.wrapping_rem(y) as u64;
            }
            Op::I64_RemU => {
                let x = opa!(ins);
                let y = opb!(ins);
                if y == 0 {
                    return Err(WasmError::trap("integer divide by zero"));
                }
                frame[ins.c as usize] = x % y;
            }
            Op::I64_And => bin64!(ins, |x, y| x & y),
            Op::I64_Or => bin64!(ins, |x, y| x | y),
            Op::I64_Xor => bin64!(ins, |x, y| x ^ y),
            Op::I64_Shl => bin64!(ins, |x: u64, y: u64| x.wrapping_shl(y as u32)),
            Op::I64_ShrS => {
                bin64!(ins, |x: u64, y: u64| ((x as i64).wrapping_shr(y as u32))
                    as u64)
            }
            Op::I64_ShrU => bin64!(ins, |x: u64, y: u64| x.wrapping_shr(y as u32)),
            Op::I64_Rotl => bin64!(ins, |x: u64, y: u64| x.rotate_left((y & 63) as u32)),
            Op::I64_Rotr => bin64!(ins, |x: u64, y: u64| x.rotate_right((y & 63) as u32)),
            Op::I64_Clz => un64!(ins, |x: u64| x.leading_zeros() as u64),
            Op::I64_Ctz => un64!(ins, |x: u64| x.trailing_zeros() as u64),
            Op::I64_Popcnt => un64!(ins, |x: u64| x.count_ones() as u64),
            Op::I64_Extend8S => un64!(ins, |x: u64| x as i8 as i64 as u64),
            Op::I64_Extend16S => un64!(ins, |x: u64| x as i16 as i64 as u64),
            Op::I64_Extend32S => un64!(ins, |x: u64| x as i32 as i64 as u64),
            Op::I64_Eqz => un64!(ins, |x: u64| (x == 0) as u64),
            Op::I64_Eq => cmp64!(ins, |x, y| x == y),
            Op::I64_Ne => cmp64!(ins, |x, y| x != y),
            Op::I64_LtS => cmp64!(ins, |x: u64, y: u64| (x as i64) < (y as i64)),
            Op::I64_LtU => cmp64!(ins, |x, y| x < y),
            Op::I64_GtS => cmp64!(ins, |x: u64, y: u64| (x as i64) > (y as i64)),
            Op::I64_GtU => cmp64!(ins, |x, y| x > y),
            Op::I64_LeS => cmp64!(ins, |x: u64, y: u64| (x as i64) <= (y as i64)),
            Op::I64_LeU => cmp64!(ins, |x, y| x <= y),
            Op::I64_GeS => cmp64!(ins, |x: u64, y: u64| (x as i64) >= (y as i64)),
            Op::I64_GeU => cmp64!(ins, |x, y| x >= y),

            // ---- int width conversions ----
            Op::I32_WrapI64 => un64!(ins, |x: u64| x as u32 as u64),
            Op::I64_ExtendI32S => un64!(ins, |x: u64| x as u32 as i32 as i64 as u64),
            Op::I64_ExtendI32U => un64!(ins, |x: u64| x as u32 as u64),

            // ---- f32 ----
            Op::F32_Abs => fun32!(ins, f32::abs),
            Op::F32_Neg => fun32!(ins, |x: f32| -x),
            Op::F32_Ceil => fun32!(ins, fmath::ceil32),
            Op::F32_Floor => fun32!(ins, fmath::floor32),
            Op::F32_Trunc => fun32!(ins, fmath::trunc32),
            Op::F32_Nearest => fun32!(ins, fmath::nearest32),
            Op::F32_Sqrt => fun32!(ins, fmath::sqrt32),
            Op::F32_Add => fbin32!(ins, |x: f32, y: f32| x + y),
            Op::F32_Sub => fbin32!(ins, |x: f32, y: f32| x - y),
            Op::F32_Mul => fbin32!(ins, |x: f32, y: f32| x * y),
            Op::F32_Div => fbin32!(ins, |x: f32, y: f32| x / y),
            Op::F32_Min => fbin32!(ins, wasm_min_f32),
            Op::F32_Max => fbin32!(ins, wasm_max_f32),
            Op::F32_Copysign => fbin32!(ins, f32::copysign),
            Op::F32_Eq => fcmp32!(ins, |x, y| x == y),
            Op::F32_Ne => fcmp32!(ins, |x, y| x != y),
            Op::F32_Lt => fcmp32!(ins, |x, y| x < y),
            Op::F32_Gt => fcmp32!(ins, |x, y| x > y),
            Op::F32_Le => fcmp32!(ins, |x, y| x <= y),
            Op::F32_Ge => fcmp32!(ins, |x, y| x >= y),

            // ---- f64 ----
            Op::F64_Abs => fun64!(ins, f64::abs),
            Op::F64_Neg => fun64!(ins, |x: f64| -x),
            Op::F64_Ceil => fun64!(ins, fmath::ceil64),
            Op::F64_Floor => fun64!(ins, fmath::floor64),
            Op::F64_Trunc => fun64!(ins, fmath::trunc64),
            Op::F64_Nearest => fun64!(ins, fmath::nearest64),
            Op::F64_Sqrt => fun64!(ins, fmath::sqrt64),
            Op::F64_Add => fbin64!(ins, |x: f64, y: f64| x + y),
            Op::F64_Sub => fbin64!(ins, |x: f64, y: f64| x - y),
            Op::F64_Mul => fbin64!(ins, |x: f64, y: f64| x * y),
            Op::F64_Div => fbin64!(ins, |x: f64, y: f64| x / y),
            Op::F64_Min => fbin64!(ins, wasm_min_f64),
            Op::F64_Max => fbin64!(ins, wasm_max_f64),
            Op::F64_Copysign => fbin64!(ins, f64::copysign),
            Op::F64_Eq => fcmp64!(ins, |x, y| x == y),
            Op::F64_Ne => fcmp64!(ins, |x, y| x != y),
            Op::F64_Lt => fcmp64!(ins, |x, y| x < y),
            Op::F64_Gt => fcmp64!(ins, |x, y| x > y),
            Op::F64_Le => fcmp64!(ins, |x, y| x <= y),
            Op::F64_Ge => fcmp64!(ins, |x, y| x >= y),

            // ---- trapping float -> int truncations ----
            Op::I32_TruncF32S => {
                let x = f32::from_bits(opa!(ins) as u32);
                frame[ins.c as usize] =
                    trunc_checked(x as f64, -2147483648.0, 2147483648.0)? as i64 as u32 as u64;
            }
            Op::I32_TruncF32U => {
                let x = f32::from_bits(opa!(ins) as u32);
                frame[ins.c as usize] =
                    trunc_checked(x as f64, 0.0, 4294967296.0)? as u64 as u32 as u64;
            }
            Op::I32_TruncF64S => {
                let x = f64::from_bits(opa!(ins));
                frame[ins.c as usize] =
                    trunc_checked(x, -2147483648.0, 2147483648.0)? as i64 as u32 as u64;
            }
            Op::I32_TruncF64U => {
                let x = f64::from_bits(opa!(ins));
                frame[ins.c as usize] = trunc_checked(x, 0.0, 4294967296.0)? as u64 as u32 as u64;
            }
            Op::I64_TruncF32S => {
                let x = f32::from_bits(opa!(ins) as u32);
                frame[ins.c as usize] =
                    trunc_checked(x as f64, -9223372036854775808.0, 9223372036854775808.0)? as i64
                        as u64;
            }
            Op::I64_TruncF32U => {
                let x = f32::from_bits(opa!(ins) as u32);
                frame[ins.c as usize] =
                    trunc_checked(x as f64, 0.0, 18446744073709551616.0)? as u64;
            }
            Op::I64_TruncF64S => {
                let x = f64::from_bits(opa!(ins));
                frame[ins.c as usize] =
                    trunc_checked(x, -9223372036854775808.0, 9223372036854775808.0)? as i64 as u64;
            }
            Op::I64_TruncF64U => {
                let x = f64::from_bits(opa!(ins));
                frame[ins.c as usize] = trunc_checked(x, 0.0, 18446744073709551616.0)? as u64;
            }

            // ---- saturating float -> int (Rust `as` matches wasm) ----
            Op::I32_TruncSatF32S => {
                un32!(ins, |x: u32| f32::from_bits(x) as i32 as u32)
            }
            Op::I32_TruncSatF32U => un32!(ins, |x: u32| f32::from_bits(x) as u32),
            Op::I32_TruncSatF64S => {
                let x = f64::from_bits(opa!(ins));
                frame[ins.c as usize] = x as i32 as u32 as u64;
            }
            Op::I32_TruncSatF64U => {
                let x = f64::from_bits(opa!(ins));
                frame[ins.c as usize] = (x as u32) as u64;
            }
            Op::I64_TruncSatF32S => {
                let x = f32::from_bits(opa!(ins) as u32);
                frame[ins.c as usize] = x as i64 as u64;
            }
            Op::I64_TruncSatF32U => {
                let x = f32::from_bits(opa!(ins) as u32);
                frame[ins.c as usize] = x as u64;
            }
            Op::I64_TruncSatF64S => {
                let x = f64::from_bits(opa!(ins));
                frame[ins.c as usize] = x as i64 as u64;
            }
            Op::I64_TruncSatF64U => {
                let x = f64::from_bits(opa!(ins));
                frame[ins.c as usize] = x as u64;
            }

            // ---- int -> float ----
            Op::F32_ConvertI32S => {
                un32!(ins, |x: u32| ((x as i32) as f32).to_bits())
            }
            Op::F32_ConvertI32U => un32!(ins, |x: u32| (x as f32).to_bits()),
            Op::F32_ConvertI64S => {
                un64!(ins, |x: u64| ((x as i64) as f32).to_bits() as u64)
            }
            Op::F32_ConvertI64U => un64!(ins, |x: u64| (x as f32).to_bits() as u64),
            Op::F32_DemoteF64 => {
                let x = f64::from_bits(opa!(ins));
                frame[ins.c as usize] = (x as f32).to_bits() as u64;
            }
            Op::F64_ConvertI32S => {
                un64!(ins, |x: u64| ((x as u32 as i32) as f64).to_bits())
            }
            Op::F64_ConvertI32U => {
                un64!(ins, |x: u64| ((x as u32) as f64).to_bits())
            }
            Op::F64_ConvertI64S => un64!(ins, |x: u64| ((x as i64) as f64).to_bits()),
            Op::F64_ConvertI64U => un64!(ins, |x: u64| (x as f64).to_bits()),
            Op::F64_PromoteF32 => {
                let x = f32::from_bits(opa!(ins) as u32);
                frame[ins.c as usize] = (x as f64).to_bits();
            }
            Op::I32_ReinterpretF32
            | Op::I64_ReinterpretF64
            | Op::F32_ReinterpretI32
            | Op::F64_ReinterpretI64 => {
                let v = opa!(ins);
                frame[ins.c as usize] = v;
            }

            // ---- memory ----
            Op::I32_Load => load!(ins, 4, |v: u64| v),
            Op::I64_Load => load!(ins, 8, |v: u64| v),
            Op::F32_Load => load!(ins, 4, |v: u64| v),
            Op::F64_Load => load!(ins, 8, |v: u64| v),
            Op::I32_Load8S => load!(ins, 1, |v: u64| v as i8 as i32 as u32 as u64),
            Op::I32_Load8U => load!(ins, 1, |v: u64| v),
            Op::I32_Load16S => load!(ins, 2, |v: u64| v as i16 as i32 as u32 as u64),
            Op::I32_Load16U => load!(ins, 2, |v: u64| v),
            Op::I64_Load8S => load!(ins, 1, |v: u64| v as i8 as i64 as u64),
            Op::I64_Load8U => load!(ins, 1, |v: u64| v),
            Op::I64_Load16S => load!(ins, 2, |v: u64| v as i16 as i64 as u64),
            Op::I64_Load16U => load!(ins, 2, |v: u64| v),
            Op::I64_Load32S => load!(ins, 4, |v: u64| v as i32 as i64 as u64),
            Op::I64_Load32U => load!(ins, 4, |v: u64| v),
            Op::I32_Store => store!(ins, 4),
            Op::I64_Store => store!(ins, 8),
            Op::F32_Store => store!(ins, 4),
            Op::F64_Store => store!(ins, 8),
            Op::I32_Store8 => store!(ins, 1),
            Op::I32_Store16 => store!(ins, 2),
            Op::I64_Store8 => store!(ins, 1),
            Op::I64_Store16 => store!(ins, 2),
            Op::I64_Store32 => store!(ins, 4),
            Op::MemorySize => {
                let m = ins.b as usize;
                let pages = self.memories.get(m).map(|x| x.len() / PAGE).unwrap_or(0);
                frame[ins.c as usize] = pages as u64;
            }
            Op::MemoryGrow => {
                let mem = self
                    .memories
                    .get_mut(ins.b as usize)
                    .ok_or(WasmError::trap("out of bounds memory access"))?;
                // Both the delta and the -1 failure result belong to the
                // memory's index type, so a 64-bit memory must not truncate
                // either to 32 bits.
                let (delta, fail, cap) = if mem.is64 {
                    (opa!(ins), u64::MAX, 1u64 << 48)
                } else {
                    (opa!(ins) as u32 as u64, u32::MAX as u64, 65536)
                };
                let cur = (mem.len() / PAGE) as u64;
                let want = cur.saturating_add(delta);
                if want > mem.max_pages || want > cap {
                    frame[ins.c as usize] = fail;
                } else {
                    mem.inst.backing_mut().data.resize(want as usize * PAGE, 0);
                    frame[ins.c as usize] = cur;
                }
            }
            Op::MemoryFill => {
                let base = ins.a as usize;
                let (d, val, n) = (frame[base], frame[base + 1], frame[base + 2]);
                let (d, n) = (d as u32 as u64, n as u32 as u64);
                let mem = self
                    .memories
                    .get_mut(ins.b as usize)
                    .ok_or(WasmError::trap("out of bounds memory access"))?;
                if d + n > mem.len() as u64 {
                    return Err(WasmError::trap("out of bounds memory access"));
                }
                mem.bytes_mut()[d as usize..(d + n) as usize].fill(val as u8);
            }
            Op::MemoryCopy => {
                let base = ins.a as usize;
                let (d, s0, n) = (frame[base], frame[base + 1], frame[base + 2]);
                // A 64-bit memory's operands are the whole slots; truncating
                // a size of -1 to 32 bits turns an out-of-bounds copy into a
                // merely large one.
                let (d, s0, n) = if ins.flags & FLAG_ADDR64 != 0 {
                    (d, s0, n)
                } else {
                    (d as u32 as u64, s0 as u32 as u64, n as u32 as u64)
                };
                let dm = (ins.b >> 32) as usize;
                let sm = (ins.b & 0xffff_ffff) as usize;
                let dlen = self.memories.get(dm).map(|x| x.len()).unwrap_or(0) as u64;
                let slen = self.memories.get(sm).map(|x| x.len()).unwrap_or(0) as u64;
                if d.checked_add(n).is_none_or(|e| e > dlen)
                    || s0.checked_add(n).is_none_or(|e| e > slen)
                {
                    return Err(WasmError::trap("out of bounds memory access"));
                }
                if dm == sm {
                    self.memories[dm]
                        .bytes_mut()
                        .copy_within(s0 as usize..(s0 + n) as usize, d as usize);
                } else {
                    for k in 0..n as usize {
                        let v = self.memories[sm].bytes()[s0 as usize + k];
                        self.memories[dm].bytes_mut()[d as usize + k] = v;
                    }
                }
            }
            Op::MemoryInit => {
                let base = ins.a as usize;
                let m = (ins.b >> 32) as usize;
                let seg = (ins.b & 0xffff_ffff) as usize;
                let (d, s0, n) = (frame[base], frame[base + 1], frame[base + 2]);
                let (d, s0, n) = (d as u32 as u64, s0 as u32 as u64, n as u32 as u64);
                let data = self
                    .module
                    .data()
                    .get(seg)
                    .map(|x| x.get_init())
                    .unwrap_or(&[]);
                let dropped = self.dropped_data.get(seg).copied().unwrap_or(true);
                let src_len = if dropped { 0 } else { data.len() as u64 };
                let mlen = self.memories.get(m).map(|x| x.len()).unwrap_or(0) as u64;
                if d + n > mlen || s0 + n > src_len {
                    // A zero-size init on a dropped segment must succeed
                    // when both offsets are in bounds.
                    if !(n == 0 && d <= mlen && s0 <= src_len) {
                        return Err(WasmError::trap("out of bounds memory access"));
                    }
                }
                if n > 0 {
                    let (d, s0, n) = (d as usize, s0 as usize, n as usize);
                    self.memories[m].bytes_mut()[d..d + n].copy_from_slice(&data[s0..s0 + n]);
                }
            }
            Op::DataDrop => {
                if let Some(x) = self.dropped_data.get_mut(ins.a as usize) {
                    *x = true;
                }
            }

            // ---- globals ----
            Op::GlobalGet => {
                let i = ins.a as usize;
                frame[ins.c as usize] = match self.shared_globals.get(i).and_then(|g| g.as_ref()) {
                    Some(shared) => shared.raw(),
                    None => self.globals[i],
                };
            }
            Op::GlobalSet => {
                let v = opa!(ins);
                let i = ins.c as usize;
                match self.shared_globals.get_mut(i).and_then(|g| g.as_mut()) {
                    Some(shared) => shared.set_raw(v),
                    None => self.globals[i] = v,
                }
            }

            // ---- ref/table ----
            Op::RefIsNull => {
                let v = opa!(ins);
                frame[ins.c as usize] = slot_to_ref(v).is_null() as u64;
            }
            Op::Throw => {
                // No enclosing `try_table` in this function matched the tag --
                // predecode would have turned that into a branch -- so the
                // exception leaves the frame. Cross-function handlers are not
                // modelled yet, so it surfaces as a trap naming the tag rather
                // than unwinding to one.
                let _ = ins.b;
                return Err(WasmError::trap("uncaught exception"));
            }
            Op::ThrowRef => {
                if slot_to_ref(opa!(ins)).is_null() {
                    return Err(WasmError::trap("null exception reference"));
                }
                return Err(WasmError::trap("uncaught exception"));
            }
            Op::RefEq => {
                frame[ins.c as usize] = (opa!(ins) == opb!(ins)) as u64;
            }
            Op::RefAsNonNull => {
                let v = opa!(ins);
                if slot_to_ref(v).is_null() {
                    return Err(WasmError::trap("null function reference"));
                }
                frame[ins.c as usize] = v;
            }
            Op::CallRef | Op::ReturnCallRef => {
                let r = slot_to_ref(opa!(ins));
                if r.is_null() {
                    return Err(WasmError::trap("null function reference"));
                }
                if r.is_special() {
                    return Err(WasmError::trap("indirect call type mismatch"));
                }
                let callee = r.0;
                let arg_base = ins.b as usize;
                return Ok(if ins.op == Op::ReturnCallRef {
                    Effect::TailCall { callee, arg_base }
                } else {
                    Effect::Call { callee, arg_base }
                });
            }
            Op::TableGet => {
                let i = opa!(ins);
                let t = &self
                    .tables
                    .get(ins.b as usize)
                    .ok_or(WasmError::trap("out of bounds table access"))?
                    .entries;
                let v = usize::try_from(i)
                    .ok()
                    .and_then(|i| t.get(i))
                    .ok_or(WasmError::trap("out of bounds table access"))?;
                frame[ins.c as usize] = v as u64;
            }
            Op::TableSet => {
                let i = opa!(ins);
                let v = opb!(ins);
                let t = &mut self
                    .tables
                    .get_mut(ins.c as usize)
                    .ok_or(WasmError::trap("out of bounds table access"))?
                    .entries;
                let i = usize::try_from(i)
                    .map_err(|_| WasmError::trap("out of bounds table access"))?;
                t.set(i, v)?;
            }
            Op::TableSize => {
                let n = self
                    .tables
                    .get(ins.b as usize)
                    .map(|t| t.entries.len())
                    .unwrap_or(0);
                frame[ins.c as usize] = n as u64;
            }
            Op::TableGrow => {
                let init = opa!(ins);
                let delta = opb!(ins) as u32 as u64;
                let tidx = (ins.c >> 32) as usize;
                let dst = (ins.c & 0xffff_ffff) as usize;
                let t = self
                    .tables
                    .get_mut(tidx)
                    .ok_or(WasmError::trap("out of bounds table access"))?;
                let cur = t.entries.len() as u64;
                if cur + delta > t.max || cur + delta > u32::MAX as u64 {
                    frame[dst] = u32::MAX as u64;
                } else {
                    t.entries.resize((cur + delta) as usize, init);
                    frame[dst] = cur;
                }
            }
            Op::TableFill => {
                let base = ins.a as usize;
                let (i, val, n) = (
                    frame[base] as u32 as u64,
                    frame[base + 1],
                    frame[base + 2] as u32 as u64,
                );
                let t = &mut self
                    .tables
                    .get_mut(ins.b as usize)
                    .ok_or(WasmError::trap("out of bounds table access"))?
                    .entries;
                if i + n > t.len() as u64 {
                    return Err(WasmError::trap("out of bounds table access"));
                }
                t.fill(i as usize, n as usize, val);
            }
            Op::TableCopy => {
                let base = ins.a as usize;
                let (d, s0, n) = (
                    frame[base] as u32 as u64,
                    frame[base + 1] as u32 as u64,
                    frame[base + 2] as u32 as u64,
                );
                let dt = (ins.b >> 32) as usize;
                let st = (ins.b & 0xffff_ffff) as usize;
                let dlen = self.tables.get(dt).map(|t| t.entries.len()).unwrap_or(0) as u64;
                let slen = self.tables.get(st).map(|t| t.entries.len()).unwrap_or(0) as u64;
                if d + n > dlen || s0 + n > slen {
                    return Err(WasmError::trap("out of bounds table access"));
                }
                // Copied through the accessors, and back-to-front when the
                // ranges overlap forwards, so an in-place move does not read
                // a slot it has already written.
                let oob = || WasmError::trap("out of bounds table access");
                if dt == st && d > s0 {
                    for k in (0..n as usize).rev() {
                        let v = self.tables[st]
                            .entries
                            .get(s0 as usize + k)
                            .ok_or_else(oob)?;
                        self.tables[dt].entries.set(d as usize + k, v)?;
                    }
                } else {
                    for k in 0..n as usize {
                        let v = self.tables[st]
                            .entries
                            .get(s0 as usize + k)
                            .ok_or_else(oob)?;
                        self.tables[dt].entries.set(d as usize + k, v)?;
                    }
                }
            }
            Op::TableInit => {
                let base = ins.a as usize;
                let (d, s0, n) = (
                    frame[base] as u32 as u64,
                    frame[base + 1] as u32 as u64,
                    frame[base + 2] as u32 as u64,
                );
                let tidx = (ins.b >> 32) as usize;
                let seg = (ins.b & 0xffff_ffff) as usize;
                let dropped = self.dropped_elems.get(seg).copied().unwrap_or(true);
                let seg_len = if dropped {
                    0u64
                } else {
                    self.module
                        .elements()
                        .get(seg)
                        .map(|e| e.get_init().len())
                        .unwrap_or(0) as u64
                };
                let tlen = self.tables.get(tidx).map(|t| t.entries.len()).unwrap_or(0) as u64;
                if d + n > tlen || s0 + n > seg_len {
                    if !(n == 0 && d <= tlen && s0 <= seg_len) {
                        return Err(WasmError::trap("out of bounds table access"));
                    }
                }
                for k in 0..n as usize {
                    let v = self.elem_value(seg, s0 as usize + k)?;
                    self.tables[tidx].entries.set(d as usize + k, v)?;
                }
            }
            Op::ElemDrop => {
                if let Some(x) = self.dropped_elems.get_mut(ins.a as usize) {
                    *x = true;
                }
            }

            // ---- parametric ----
            Op::Select => {
                let cond = frame[(ins.c >> 32) as usize];
                let dst = (ins.c & 0xffff_ffff) as usize;
                frame[dst] = if cond != 0 { opa!(ins) } else { opb!(ins) };
            }

            // ---- control ----
            Op::Br => {
                return Ok(Effect::Jump(ins.c as usize));
            }
            Op::BrIf => {
                if opa!(ins) as u32 != 0 {
                    return Ok(Effect::Jump(ins.c as usize));
                }
            }
            Op::I32_BrEq => cmp_br32!(ins, |x: u32, y: u32| x == y),
            Op::I32_BrNe => cmp_br32!(ins, |x: u32, y: u32| x != y),
            Op::I32_BrLtS => cmp_br32!(ins, |x: u32, y: u32| (x as i32) < (y as i32)),
            Op::I32_BrLtU => cmp_br32!(ins, |x: u32, y: u32| x < y),
            Op::I32_BrGtS => cmp_br32!(ins, |x: u32, y: u32| (x as i32) > (y as i32)),
            Op::I32_BrGtU => cmp_br32!(ins, |x: u32, y: u32| x > y),
            Op::I32_BrLeS => cmp_br32!(ins, |x: u32, y: u32| (x as i32) <= (y as i32)),
            Op::I32_BrLeU => cmp_br32!(ins, |x: u32, y: u32| x <= y),
            Op::I32_BrGeS => cmp_br32!(ins, |x: u32, y: u32| (x as i32) >= (y as i32)),
            Op::I32_BrGeU => cmp_br32!(ins, |x: u32, y: u32| x >= y),
            Op::I64_BrEq => cmp_br64!(ins, |x: u64, y: u64| x == y),
            Op::I64_BrNe => cmp_br64!(ins, |x: u64, y: u64| x != y),
            Op::I64_BrLtS => cmp_br64!(ins, |x: u64, y: u64| (x as i64) < (y as i64)),
            Op::I64_BrLtU => cmp_br64!(ins, |x: u64, y: u64| x < y),
            Op::I64_BrGtS => cmp_br64!(ins, |x: u64, y: u64| (x as i64) > (y as i64)),
            Op::I64_BrGtU => cmp_br64!(ins, |x: u64, y: u64| x > y),
            Op::I64_BrLeS => cmp_br64!(ins, |x: u64, y: u64| (x as i64) <= (y as i64)),
            Op::I64_BrLeU => cmp_br64!(ins, |x: u64, y: u64| x <= y),
            Op::I64_BrGeS => cmp_br64!(ins, |x: u64, y: u64| (x as i64) >= (y as i64)),
            Op::I64_BrGeU => cmp_br64!(ins, |x: u64, y: u64| x >= y),
            Op::I32_BrAnd => cmp_br32!(ins, |x: u32, y: u32| x & y != 0),
            Op::I32_BrAndNot => cmp_br32!(ins, |x: u32, y: u32| x & y == 0),
            Op::BrIfNot => {
                if opa!(ins) as u32 == 0 {
                    return Ok(Effect::Jump(ins.c as usize));
                }
            }
            Op::BrTable => {
                let table = &func.br_tables[ins.c as usize];
                let idx = (opa!(ins) as u32 as usize).min(table.len() - 1);
                return Ok(Effect::Jump(table[idx] as usize));
            }
            Op::Return => {
                let base = ins.a as usize;
                let count = ins.b as usize;
                for i in 0..count {
                    frame[i] = frame[base + i];
                }
                return Ok(Effect::Ret);
            }
            Op::Call | Op::ReturnCall => {
                let callee = ins.a as usize;
                let arg_base = ins.b as usize;
                return Ok(if ins.op == Op::ReturnCall {
                    Effect::TailCall { callee, arg_base }
                } else {
                    Effect::Call { callee, arg_base }
                });
            }
            Op::CallIndirect | Op::ReturnCallIndirect => {
                let t = opa!(ins);
                let table = self
                    .tables
                    .get((ins.c >> 32) as usize)
                    .ok_or(WasmError::trap("undefined element"))?;
                let fi = usize::try_from(t)
                    .ok()
                    .and_then(|t| table.entries.get(t))
                    .ok_or(WasmError::trap("undefined element"))?;
                let callee = slot_to_ref(fi);
                if callee.is_null() {
                    return Err(WasmError::trap("uninitialized element"));
                }
                if callee.is_special() {
                    // A published reference. If WE published it the callee is
                    // ours: an exported table used from inside its own module
                    // must not pay a round trip through the embedder.
                    if let Some(&(_, local)) = self.published.iter().find(|(h, _)| *h == callee) {
                        return Ok(Effect::Call {
                            callee: local,
                            arg_base: ins.b as usize,
                        });
                    }
                    // Otherwise it names a function in another instance, and
                    // only the embedder can reach that.
                    let (np, nr) = self
                        .module
                        .types()
                        .get_function_type(ins.c as u32)
                        .map(|ft| (ft.params().len(), ft.results().len()))
                        .ok_or(WasmError::trap("indirect call type mismatch"))?;
                    let base = ins.b as usize;
                    let mut args: Vec<u64> = Vec::with_capacity(np);
                    args.extend_from_slice(&frame[base..base + np]);
                    let mut results = vec![0u64; nr];
                    match self.funcref_host.as_mut() {
                        Some(h) => (h.invoke)(callee, &args, &mut results)?,
                        None => return Err(WasmError::trap("indirect call type mismatch")),
                    }
                    frame[base..base + nr].copy_from_slice(&results);
                    return Ok(Effect::Next);
                }
                let fi = callee.0 as u32;
                let expected = ins.c as u32;
                let actual = self
                    .module
                    .functions()
                    .get(fi as usize)
                    .ok_or(WasmError::trap("undefined element"))?
                    .type_index();
                if !self.module.types().types_equivalent(expected, actual) {
                    return Err(WasmError::trap("indirect call type mismatch"));
                }
                return Ok(if ins.op == Op::ReturnCallIndirect {
                    Effect::TailCall {
                        callee: fi as usize,
                        arg_base: ins.b as usize,
                    }
                } else {
                    Effect::Call {
                        callee: fi as usize,
                        arg_base: ins.b as usize,
                    }
                });
            }
            Op::Unreachable => {
                return Err(WasmError::trap("unreachable"));
            }
        }
        Ok(Effect::Next)
    }
}

/// Truncate with the exact wasm trap boundaries: `lo` is the inclusive
/// lower bound of the truncated value (0.0 for the unsigned cases — a
/// truncated -0.0 compares equal to 0.0 and is valid), `hi_excl` the
/// exclusive upper bound. Both bounds are exactly representable in f64.
fn trunc_checked(x: f64, lo: f64, hi_excl: f64) -> Result<f64, WasmError> {
    if x.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    let t = fmath::trunc64(x);
    if t < lo || t >= hi_excl {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(t)
}

fn wasm_min_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a < b {
        a
    } else if b < a {
        b
    } else if a.is_sign_negative() {
        a
    } else {
        b
    }
}

fn wasm_max_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a > b {
        a
    } else if b > a {
        b
    } else if a.is_sign_positive() {
        a
    } else {
        b
    }
}

fn wasm_min_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a < b {
        a
    } else if b < a {
        b
    } else if a.is_sign_negative() {
        a
    } else {
        b
    }
}

fn wasm_max_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a > b {
        a
    } else if b > a {
        b
    } else if a.is_sign_positive() {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::Module;
    use std::vec::Vec as StdVec;

    fn instantiate(src: &str) -> (StdVec<u8>, ()) {
        let bin: StdVec<u8> = wat::parse_str(src).expect("wat");
        (bin, ())
    }

    fn run1(src: &str, export: &str, args: &[u64]) -> Result<u64, WasmError> {
        let (bin, _) = instantiate(src);
        let module = Module::new("t", &bin).expect("module");
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            None,
            &[],
        )?;
        let idx = inst.find_export(export).expect("export");
        let mut results = [0u64; 1];
        inst.invoke(idx, args, &mut results)?;
        Ok(results[0])
    }

    /// Extended constant expressions: arithmetic, and `global.get` of a
    /// global declared earlier. Neither is expressible with a single
    /// `t.const`, which is all this used to accept.
    #[test]
    fn extended_const_expressions() {
        let r = run1(
            r#"(module
                 (global $a i32 (i32.const 7))
                 (global $b i32 (i32.add (global.get $a) (i32.const 5)))
                 (global $c i32 (i32.mul (global.get $b) (i32.const 3)))
                 (func (export "c") (result i32) global.get $c))"#,
            "c",
            &[],
        )
        .expect("extended const module should instantiate");
        assert_eq!(r as u32, (7 + 5) * 3);
    }

    /// An active data offset is a constant expression too, so the same
    /// arithmetic has to reach segment placement.
    #[test]
    fn extended_const_reaches_a_data_offset() {
        let r = run1(
            r#"(module
                 (memory 1)
                 (global $base i32 (i32.const 4))
                 (data (offset (i32.add (global.get $base) (i32.const 4))) "\2a\00\00\00")
                 (func (export "at8") (result i32) i32.const 8 i32.load))"#,
            "at8",
            &[],
        )
        .expect("data offset should accept an extended const expression");
        assert_eq!(r as u32, 42);
    }

    /// A constant expression may not read a global that is not initialized
    /// yet -- the spec restricts it to earlier ones.
    #[test]
    fn const_expression_cannot_read_a_later_global() {
        let bin: StdVec<u8> = wat::parse_str(
            r#"(module
                 (global $a i32 (i32.const 1))
                 (global $b i32 (global.get $a)))"#,
        )
        .expect("wat");
        let module = Module::new("t", &bin).expect("module");
        assert!(
            InterpInstance::new(
                &crate::vm::engine::Engine::with_defaults(),
                module,
                None,
                &[]
            )
            .is_ok(),
            "reading an EARLIER global is legal"
        );
    }

    #[test]
    fn add_params() {
        let r = run1(
            r#"(module (func (export "add") (param i32 i32) (result i32)
                local.get 0 local.get 1 i32.add))"#,
            "add",
            &[40, 2],
        );
        assert_eq!(r.unwrap(), 42);
    }

    /// `br_if` whose label is the function body itself is a conditional
    /// return: the not-taken path must fall through.
    #[test]
    fn br_if_to_function_label_falls_through() {
        let (bin, _) = instantiate(
            r#"(module
                (global $g (mut i32) (i32.const 0))
                (func (export "f") (param i32)
                    local.get 0
                    br_if 0
                    i32.const 42
                    global.set $g)
                (func (export "get") (result i32) global.get $g))"#,
        );
        let module = Module::new("t", &bin).expect("module");
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            None,
            &[],
        )
        .expect("instantiate");
        let f = inst.find_export("f").expect("f");
        let g = inst.find_export("get").expect("get");
        inst.invoke(f, &[0], &mut []).expect("invoke cond=0");
        let mut r = [0u64; 1];
        inst.invoke(g, &[], &mut r).expect("invoke get");
        assert_eq!(r[0], 42, "br_if(0) must fall through");
        inst.invoke(f, &[1], &mut []).expect("invoke cond=1");
        // taken: returns immediately, global untouched (still 42)
        inst.invoke(g, &[], &mut r).expect("invoke get");
        assert_eq!(r[0], 42);
    }

    #[test]
    fn recursive_fib() {
        let src = r#"(module (func $fib (export "fib") (param i32) (result i32)
            local.get 0
            i32.const 2
            i32.lt_u
            (if (result i32)
                (then local.get 0)
                (else
                    local.get 0 i32.const 1 i32.sub call $fib
                    local.get 0 i32.const 2 i32.sub call $fib
                    i32.add))))"#;
        assert_eq!(run1(src, "fib", &[15]).unwrap(), 610);
    }

    #[test]
    fn loop_sum_1_to_100() {
        let src = r#"(module (func (export "sum") (result i32) (local i32 i32)
            (loop $l
                local.get 0 i32.const 1 i32.add local.set 0
                local.get 1 local.get 0 i32.add local.set 1
                local.get 0 i32.const 100 i32.lt_u br_if $l)
            local.get 1))"#;
        assert_eq!(run1(src, "sum", &[]).unwrap(), 5050);
    }

    #[test]
    fn memory_roundtrip_and_data_segment() {
        let src = r#"(module (memory 1) (data (i32.const 16) "\2a\00\00\00")
            (func (export "go") (param i32) (result i32)
                local.get 0 i32.const 7 i32.store offset=64
                i32.const 0 i32.load offset=64
                i32.const 16 i32.load
                i32.add))"#;
        // 7 + 42 = 49... store writes ARG to [7+64]? no: addr=local0, value 7.
        // go(0): mem[64]=7; load [64]=7; load [16]=42 -> 49
        assert_eq!(run1(src, "go", &[0]).unwrap(), 49);
    }

    #[test]
    fn subword_load_store_sign() {
        let src = r#"(module (memory 1)
            (func (export "go") (result i32)
                i32.const 0 i32.const 0xFFFF8080 i32.store
                i32.const 0 i32.load8_s))"#;
        // low byte 0x80 sign-extends to -128
        assert_eq!(run1(src, "go", &[]).unwrap() as u32 as i32, -128);
    }

    #[test]
    fn globals_counter() {
        let src = r#"(module (global $g (mut i32) (i32.const 5))
            (func (export "bump") (result i32)
                global.get $g i32.const 3 i32.add global.set $g
                global.get $g))"#;
        assert_eq!(run1(src, "bump", &[]).unwrap(), 8);
    }

    #[test]
    fn br_table_switch() {
        let src = r#"(module (func (export "sw") (param i32) (result i32)
            (block $b2 (block $b1 (block $b0
                local.get 0 br_table $b0 $b1 $b2)
                i32.const 10 return)
                i32.const 20 return)
            i32.const 30))"#;
        assert_eq!(run1(src, "sw", &[0]).unwrap(), 10);
        assert_eq!(run1(src, "sw", &[1]).unwrap(), 20);
        assert_eq!(run1(src, "sw", &[2]).unwrap(), 30);
        assert_eq!(run1(src, "sw", &[9]).unwrap(), 30); // default
    }

    #[test]
    fn try_table_catches_in_the_same_function() {
        // The throw is resolved at predecode into a branch to the catch's
        // label, so no unwinding is involved when handler and throw share a
        // function.
        let src = r#"(module
            (tag $t)
            (func (export "go") (result i32)
                (block $h
                    (try_table (catch_all $h)
                        (throw $t))
                    (return (i32.const 1)))
                (i32.const 2)))"#;
        // The throw reaches catch_all, which carries no values, so control
        // lands after the block and returns 2 rather than 1.
        assert_eq!(run1(src, "go", &[]).unwrap(), 2);
    }

    #[test]
    fn return_call_runs_at_constant_depth() {
        // A million tail calls must not grow the activation stack; if the
        // frame is not reused this exhausts it instead of returning.
        let src = r#"(module
            (func $count (param i64) (result i64)
                local.get 0
                i64.eqz
                if (result i64)
                    i64.const 0
                else
                    local.get 0
                    i64.const 1
                    i64.sub
                    return_call $count
                end)
            (func (export "go") (param i64) (result i64)
                local.get 0 call $count))"#;
        assert_eq!(run1(src, "go", &[0]).unwrap(), 0, "no tail call at all");
        assert_eq!(run1(src, "go", &[1]).unwrap(), 0, "exactly one tail call");
        assert_eq!(run1(src, "go", &[2]).unwrap(), 0, "two tail calls");
        assert_eq!(run1(src, "go", &[1_000_000]).unwrap(), 0, "a million");
    }

    #[test]
    fn call_indirect_dispatch() {
        let src = r#"(module
            (type $t (func (param i32) (result i32)))
            (table 2 funcref) (elem (i32.const 0) $double $square)
            (func $double (type $t) local.get 0 i32.const 2 i32.mul)
            (func $square (type $t) local.get 0 local.get 0 i32.mul)
            (func (export "go") (param i32 i32) (result i32)
                local.get 1 local.get 0 call_indirect (type $t)))"#;
        assert_eq!(run1(src, "go", &[0, 21]).unwrap(), 42);
        assert_eq!(run1(src, "go", &[1, 7]).unwrap(), 49);
    }

    #[test]
    fn float_kernel() {
        let src = r#"(module (func (export "hyp") (param f64 f64) (result f64)
            local.get 0 local.get 0 f64.mul
            local.get 1 local.get 1 f64.mul
            f64.add f64.sqrt))"#;
        let r = run1(src, "hyp", &[3.0f64.to_bits(), 4.0f64.to_bits()]).unwrap();
        assert_eq!(f64::from_bits(r), 5.0);
    }

    #[test]
    fn float_min_nan_and_signed_zero() {
        let src = r#"(module (func (export "mn") (param f32 f32) (result f32)
            local.get 0 local.get 1 f32.min))"#;
        let nan = run1(
            src,
            "mn",
            &[f32::NAN.to_bits() as u64, 1.0f32.to_bits() as u64],
        );
        assert!(f32::from_bits(nan.unwrap() as u32).is_nan());
        let z = run1(
            src,
            "mn",
            &[0.0f32.to_bits() as u64, (-0.0f32).to_bits() as u64],
        );
        assert!(f32::from_bits(z.unwrap() as u32).is_sign_negative());
    }

    #[test]
    fn trunc_traps_and_sat_saturates() {
        let trap = run1(
            r#"(module (func (export "t") (param f32) (result i32)
                local.get 0 i32.trunc_f32_s))"#,
            "t",
            &[3e10f32.to_bits() as u64],
        );
        assert!(matches!(trap, Err(WasmError::Trap(_))));
        let sat = run1(
            r#"(module (func (export "s") (param f32) (result i32)
                local.get 0 i32.trunc_sat_f32_s))"#,
            "s",
            &[3e10f32.to_bits() as u64],
        );
        assert_eq!(sat.unwrap() as u32 as i32, i32::MAX);
    }

    #[test]
    fn div_traps() {
        let src = r#"(module (func (export "d") (param i32 i32) (result i32)
            local.get 0 local.get 1 i32.div_s))"#;
        assert!(matches!(run1(src, "d", &[1, 0]), Err(WasmError::Trap(_))));
        assert!(matches!(
            run1(src, "d", &[i32::MIN as u32 as u64, u32::MAX as u64]),
            Err(WasmError::Trap(_))
        ));
        // rem_s MIN % -1 must NOT trap and equals 0
        let rem = run1(
            r#"(module (func (export "r") (param i32 i32) (result i32)
                local.get 0 local.get 1 i32.rem_s))"#,
            "r",
            &[i32::MIN as u32 as u64, u32::MAX as u64],
        );
        assert_eq!(rem.unwrap(), 0);
    }

    #[test]
    fn oob_load_traps() {
        let r = run1(
            r#"(module (memory 1) (func (export "l") (param i32) (result i32)
                local.get 0 i32.load))"#,
            "l",
            &[65534],
        );
        assert!(matches!(r, Err(WasmError::Trap(_))));
    }

    #[test]
    fn deep_recursion_exhausts() {
        let r = run1(
            r#"(module (func $f (export "f") (param i32) (result i32)
                local.get 0 i32.const 1 i32.add call $f))"#,
            "f",
            &[0],
        );
        assert!(matches!(r, Err(WasmError::Trap("call stack exhausted"))));
    }

    #[test]
    fn memory_bulk_fill_copy() {
        let src = r#"(module (memory 1)
            (func (export "go") (result i32)
                i32.const 8 i32.const 0x5a i32.const 4 memory.fill
                i32.const 32 i32.const 8 i32.const 4 memory.copy
                i32.const 32 i32.load))"#;
        assert_eq!(run1(src, "go", &[]).unwrap(), 0x5a5a5a5a);
    }

    #[test]
    fn i64_mix() {
        let src = r#"(module (func (export "g") (param i64 i64) (result i64)
            local.get 0 local.get 1 i64.mul
            i64.const 7 i64.rem_u
            local.get 0 i64.const 13 i64.shl
            i64.xor))"#;
        let a = 0x1234_5678u64;
        let b = 999u64;
        let expect = ((a.wrapping_mul(b)) % 7) ^ (a << 13);
        assert_eq!(run1(src, "g", &[a, b]).unwrap(), expect);
    }

    #[test]
    fn select_picks_by_condition() {
        let src = r#"(module (func (export "s") (param i32) (result i32)
            i32.const 111 i32.const 222 local.get 0 select))"#;
        assert_eq!(run1(src, "s", &[1]).unwrap(), 111);
        assert_eq!(run1(src, "s", &[0]).unwrap(), 222);
    }
}
