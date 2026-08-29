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
use tracked_alloc::string::String;

use crate::collections::{vec, Vec};
use crate::config::Config;
use crate::error::WasmError;
use crate::module::entities::{Data, Element, ElementInit, GlobalDef, MemoryDef, TableDef, TagDef};
use crate::module::type_context::{concrete_type_matches_cross_context, TypeContext};
use crate::module::Module;
use crate::utils::limits::{Limitable, Limits};
use crate::value_type::{AbstractHeapType, HeapType, RefType, ValueType};
use crate::vm::engine::Engine;
use crate::vm::entities::{Caller, GlobalInst, MemInst, TableInst};
use crate::vm::imports::{Import, ImportValue, ImportedFunction, ImportedGlobal};
use crate::vm::link::{
    ref_type_matches, FuncEntry, InstanceBackref, InstanceId, InstanceLease, InstanceToken,
    LinkArenas, LinkRegistry, RefTypeOwner,
};
use crate::vm::tag::TagIdentity;
use crate::vm::value::{machine_raw_to_ref, ref_to_machine_raw, RefValue, Value};
use core::cell::RefCell;
use core::marker::PhantomData;
use core::ptr::NonNull;

#[cfg(sf_module_validator)]
use super::baseline_artifact::BaselineArtifact;
#[cfg(all(test, sf_module_validator))]
use super::baseline_exec::{
    raw_stepper_supports_function, RawFrameState, RawSlots, RawStepExit, RawStepper,
};
#[cfg(sf_module_validator)]
use super::baseline_function_plan::FunctionPlanKind;
use super::engine::{
    CallFixup, DCell, EnterState, LinkPlan, LinkScratch, LinkedFunction, NativeEngine, EXIT_RETURN,
    EXIT_SLOW, EXIT_TRAP_BASE, RET_RECORD, TRAP_KINDS,
};
use super::fmath;
use super::instr::{op_from_index, Op};
use super::instr::{Instr, FLAG_ADDR64, FLAG_A_CONST, FLAG_B_CONST, FLAG_FUSED};
use super::predecode::{
    predecode_functions, PredecodedFunction, UnlinkedPredecodedFunctions, WIDE_MEMARG,
};
use super::SLOT_GP_UNIT_BYTES;

const PAGE: usize = 65536;
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
/// Funcref-typed argument and result slots use absolute world identities at
/// this boundary; the interpreter converts against the caller's frame
/// immediately before and after the callback.
///
/// The signature carries only std types (`&mut [u8]`, not this crate's
/// tracked collections), so external callers stay feature-independent.
type DynHostDispatch =
    dyn for<'a> FnMut(&str, &str, &mut Caller<'a>, &[u64], &mut [u64]) -> Result<(), WasmError>;

#[derive(Clone)]
pub(crate) struct HostDispatch {
    callback: Rc<RefCell<Box<DynHostDispatch>>>,
}

impl HostDispatch {
    fn new(callback: Box<DynHostDispatch>) -> Self {
        Self {
            callback: Rc::new(RefCell::new(callback)),
        }
    }

    fn invoke(
        &self,
        module: &str,
        name: &str,
        caller: &mut Caller<'_>,
        args: &[u64],
        results: &mut [u64],
    ) -> Result<(), WasmError> {
        let mut callback = self
            .callback
            .try_borrow_mut()
            .map_err(|_| WasmError::trap("interp: host dispatcher is already active"))?;
        callback(module, name, caller, args, results)
    }
}

/// Embedder linking hook for function references the interpreter should
/// delegate outside its runtime world.
///
/// Both engines use one runtime-world function identity. Registry-known
/// interpreter functions can be entered directly, but an installed hook keeps
/// precedence so embedders may override that routing. The hook is also the
/// escape hatch for functions owned outside this world and for cross-engine
/// calls, since a runtime world itself is single-tier.
///
/// The boxes are `alloc`'s, not this crate's tracked ones, for the same reason
/// `HostDispatch`'s signature carries only std types: an embedder has to be
/// able to construct one without depending on the allocator this crate is
/// built with.
pub struct FuncRefHost {
    /// Call whatever absolute world identity `handle` names.
    ///
    /// Funcref-typed argument and result slots are absolute at this boundary.
    pub invoke: alloc::boxed::Box<dyn FnMut(RefValue, &[u64], &mut [u64]) -> Result<(), WasmError>>,
}

type SharedFuncRefHost = Rc<RefCell<FuncRefHost>>;

#[inline]
fn value_type_is_function_ref(module: &Module, value_type: ValueType) -> bool {
    let ValueType::Ref(ref_type) = value_type else {
        return false;
    };
    matches!(
        ref_type.heap_type.top_type(module.types()),
        HeapType::Abstract(AbstractHeapType::Func)
    )
}

fn absolutize_ref_with(
    module: &Module,
    function_identities: &[RefValue],
    handle: RefValue,
) -> RefValue {
    if handle.is_null() || handle.is_special() {
        return handle;
    }
    let local = handle.payload();
    if module.functions().get(local).is_none() {
        return handle;
    }
    function_identities
        .get(local)
        .copied()
        .filter(|identity| !identity.is_null())
        .unwrap_or(handle)
}

fn localize_ref_with(
    module: &Module,
    instance_backref: &InstanceBackref,
    link_registry: &LinkArenas,
    handle: RefValue,
) -> RefValue {
    if handle.is_null() || handle.is_special() {
        return handle;
    }
    if handle.encoded() < module.functions().len() {
        return handle;
    }
    let Some(entry) = link_registry.functions.entry_for_handle(handle) else {
        return handle;
    };
    if entry.owner == instance_backref.self_id() {
        RefValue::new(entry.local_index as usize)
    } else {
        handle
    }
}

#[inline]
fn absolutize_slot_with(module: &Module, function_identities: &[RefValue], slot: u64) -> u64 {
    ref_to_machine_raw(
        absolutize_ref_with(
            module,
            function_identities,
            machine_raw_to_ref(slot, SLOT_GP_UNIT_BYTES),
        ),
        SLOT_GP_UNIT_BYTES,
    )
}

#[inline]
fn localize_slot_with(
    module: &Module,
    instance_backref: &InstanceBackref,
    link_registry: &LinkArenas,
    slot: u64,
) -> u64 {
    ref_to_machine_raw(
        localize_ref_with(
            module,
            instance_backref,
            link_registry,
            machine_raw_to_ref(slot, SLOT_GP_UNIT_BYTES),
        ),
        SLOT_GP_UNIT_BYTES,
    )
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
/// the chain reads 8-byte slots, and `TableInst` holds `RefValue`, which is
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
            Self::Shared(t) => t
                .elements()
                .get(i)
                .map(|handle| ref_to_machine_raw(*handle, SLOT_GP_UNIT_BYTES)),
        }
    }

    #[inline]
    fn set(&mut self, i: usize, v: u64) -> Result<(), WasmError> {
        let oob = || WasmError::trap("out of bounds table access");
        match self {
            Self::Private(vec) => *vec.get_mut(i).ok_or_else(oob)? = v,
            Self::Shared(t) => {
                *t.elements_mut().get_mut(i).ok_or_else(oob)? =
                    machine_raw_to_ref(v, SLOT_GP_UNIT_BYTES)
            }
        }
        Ok(())
    }

    /// Fallible grow, mirroring the JIT's `do_table_grow`: a failed host
    /// allocation becomes the caller's -1 result, not an abort.
    fn try_resize(&mut self, n: usize, v: u64) -> Result<(), ()> {
        match self {
            Self::Private(vec) => {
                let additional = n.saturating_sub(vec.len());
                if vec.try_reserve(additional).is_err() {
                    return Err(());
                }
                vec.resize(n, v);
            }
            Self::Shared(t) => {
                let mut elements = t.elements_mut();
                let additional = n.saturating_sub(elements.len());
                if elements.try_reserve(additional).is_err() {
                    return Err(());
                }
                elements.resize(n, machine_raw_to_ref(v, SLOT_GP_UNIT_BYTES));
            }
        }
        Ok(())
    }

    fn fill(&mut self, at: usize, n: usize, v: u64) {
        match self {
            Self::Private(vec) => vec[at..at + n].fill(v),
            Self::Shared(t) => {
                t.elements_mut()[at..at + n].fill(machine_raw_to_ref(v, SLOT_GP_UNIT_BYTES))
            }
        }
    }

    /// The base pointer the dispatch chain indexes, when there is one.
    ///
    /// `None` for a shared table: its elements are `RefValue`, not 8-byte
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
    #[cfg(all(test, sf_module_validator))]
    raw: Option<RawFrameState>,
}

/// A caller suspended at a Rust-owned call boundary.
///
/// Calls covered by an exception handler deliberately stay on the slow path,
/// so every frame that can catch an exception from a callee has one of these
/// checkpoints. Restoring the cursor discards the callee's sentinel together
/// with any native-only descendant records below it.
// The activation precedes the return cursor so the hot save is a few wide
// paired moves instead of a field-by-field shuffle. Measured on AArch64
// (Cobalt-100) and x86-64; see cross-instance-identity in mcts_mem.
#[repr(C)]
struct SavedActivation {
    activation: Activation,
    ret_cursor: usize,
    #[cfg(all(test, sf_module_validator))]
    raw_return_top: Option<usize>,
}

#[cold]
#[inline(never)]
fn grow_saved_activations(saved: &mut Vec<SavedActivation>) {
    saved.reserve(1);
}

#[derive(Clone, Copy)]
pub(super) struct PendingException {
    exn: RefValue,
    tag: TagIdentity,
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

enum ExternalCallTarget {
    Host {
        dispatch: HostDispatch,
        names: Rc<(String, String)>,
        memory: Option<MemInst>,
    },
    FuncRef {
        host: Option<SharedFuncRefHost>,
        handle: RefValue,
        world: Option<WorldFuncTarget>,
    },
}

struct WorldFuncTarget {
    entry: FuncEntry,
    checkout: InstanceBackref,
}

pub(super) struct PreparedCall {
    target: ExternalCallTarget,
    args: Vec<u64>,
    result_types: Vec<ValueType>,
}

pub(super) enum ResolvedIndirectCall {
    Local(usize),
    External(Box<PreparedCall>),
}

impl PreparedCall {
    fn invoke(&self) -> Result<Vec<u64>, WasmError> {
        let mut results = vec![0u64; self.result_types.len()];
        match &self.target {
            ExternalCallTarget::Host {
                dispatch,
                names,
                memory,
            } => {
                let mut caller = Caller::from_shared_memory(memory.clone());
                dispatch.invoke(
                    names.0.as_str(),
                    names.1.as_str(),
                    &mut caller,
                    &self.args,
                    &mut results,
                )?;
            }
            ExternalCallTarget::FuncRef {
                host,
                handle,
                world,
            } => match host {
                Some(host) => {
                    let mut host = host.try_borrow_mut().map_err(|_| {
                        WasmError::trap("interp: function-reference host is already active")
                    })?;
                    (host.invoke)(*handle, &self.args, &mut results)?;
                }
                None => {
                    let world = world
                        .as_ref()
                        .ok_or_else(|| WasmError::trap(super::EXTERNAL_FUNCREF_HOST_REQUIRED))?;
                    results = InterpInstance::invoke_world_target(
                        world,
                        &self.args,
                        self.result_types.len(),
                    )?;
                }
            },
        }
        Ok(results)
    }
}

/// Variant order is performance-sensitive. `Call` is the common exit from
/// `native_step`; keeping it at discriminant zero removes a taken branch from
/// the AArch64 local-call dispatch and preserves the measured hot-loop layout.
enum StepExit {
    Call {
        callee: usize,
        arg_base: usize,
    },
    TailCall {
        callee: usize,
        arg_base: usize,
    },
    Throw {
        pending: PendingException,
        search_current: bool,
    },
    ExternalCall {
        /// Boxed on purpose. `StepExit` and [`Effect`] are returned by value
        /// from the slow path, so an inline `PreparedCall` sets the size of
        /// every exit -- including the common ones that carry two words. The
        /// indirection is paid only when a host call actually happens.
        call: Box<PreparedCall>,
        result_base: usize,
        site: usize,
        search_current: bool,
    },
    Return,
}

/// Why a materialized run of the local activation driver ended.
///
/// `ExternalCall` owns everything needed to run outside the current instance
/// materialization, so the token-derived `&mut InterpInstance` can end before
/// the call is invoked.
enum MaterializedRunExit {
    ExternalCall {
        call: Box<PreparedCall>,
        result_base: usize,
        error_site: Option<usize>,
        resume_tail_return: bool,
    },
    Complete,
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
    ExternalCall {
        /// Boxed for the same reason as `StepExit::ExternalCall`: this enum is
        /// `exec_ins`'s by-value return, so the largest variant is copied on
        /// every slow exit, not only on the external-call one.
        call: Box<PreparedCall>,
        arg_base: usize,
        tail_target: Option<usize>,
    },
    Throw(PendingException),
    Ret,
}

/// Stage-B state: the emitted handler engine plus per-function dispatch
/// cells (parallel to `InterpInstance::funcs`).
struct NativeState {
    engine: NativeEngine,
    /// One arena owns every function's dispatch cells and flattened branch
    /// tables. `linked` stores stable ranges into it rather than one vector
    /// allocation per function. Function entry addresses are cached after
    /// linking, so this field is ownership-only once initialization ends.
    plan: LinkPlan,
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
    /// Published only after the shared instruction allocation has been
    /// transferred into dispatch cells. Each defined body keeps the original
    /// per-function `Rc` metadata and side-table vectors.
    funcs: Option<Vec<Option<Rc<PredecodedFunction>>>>,
    /// Owned stage-A arena. It exists only between `build` and the first
    /// step of `initialize`, where link consumes it in place and publishes
    /// immutable function metadata.
    unlinked_funcs: Option<UnlinkedPredecodedFunctions>,
    /// Runtime state only the executor touches. Instance construction still
    /// builds it on every target -- doing so is what rejects
    /// imported/64-bit/non-funcref memories and tables and traps out-of-range
    /// active segments -- but a target without an executor validates and
    /// drops it rather than carrying it.
    memories: Vec<MemoryState>,
    dropped_data: Vec<bool>,
    dropped_elems: Vec<bool>,
    globals: Vec<u64>,
    /// Globals backed by an aliased cell, by index. These live in an
    /// `Rc`-owned cell rather than the array above because both sides must
    /// observe each other's writes; the array slot beside them is unused.
    /// Reachability itself is recorded separately below.
    shared_globals: Vec<Option<GlobalInst>>,
    /// Whether each global can be observed by another instance. This is a
    /// semantic fact, not a proxy for whether this engine uses a shared cell.
    global_reachable: Vec<bool>,
    tables: Vec<TableState>,
    /// Whether each table can be observed by another instance. Imported
    /// tables without a supplied backing can still use a private vector, so
    /// `TableEntries::Shared` is deliberately not this fact.
    table_reachable: Vec<bool>,
    /// Runtime tag identities, one per declared tag. Linking needs them;
    /// catches compare these handles rather than module-local indices.
    tags: Vec<TagIdentity>,
    /// Shared reference arena used by exception objects. Linked instances
    /// clone one registry so an exception retains both its payload and its
    /// identity while crossing an imported-function boundary.
    instance_backref: InstanceBackref,
    link_registry: LinkArenas,
    /// The operand stack and native return stack, allocated on the first
    /// guest call and then reused by every call. `Some(empty)` is the
    /// never-entered instance state; `None` means an outer call currently
    /// owns the buffer and a re-entry must use a temporary pair.
    ///
    /// Keeping the owner slot present but empty lets instantiation avoid
    /// allocating and zeroing 2 MiB of runtime-only storage. Predecode and
    /// native linking remain eager; only memory that cannot be observed
    /// before a call is deferred.
    config: Config,
    stack: Option<Vec<u64>>,
    ret_stack: Option<Vec<u64>>,
    host: Option<HostDispatch>,
    funcref_host: Option<SharedFuncRefHost>,
    /// Stable import names for calls that leave an instance materialization.
    ///
    /// The driver clones one `Rc` into an owned call request, then passes the
    /// borrowed strings to the host only after the instance borrow has ended.
    import_names: Vec<Option<Rc<(String, String)>>>,
    /// Frame-relative handles: local/host functions use their local index;
    /// linked imports retain the provider's absolute world identity.
    function_handles: Vec<RefValue>,
    /// Absolute world identities indexed by this module's function index.
    /// Non-escapable local functions keep the null sentinel and consume no
    /// world address.
    function_identities: Vec<RefValue>,
    /// Reverse of `function_identities`, restricted to this instance's OWN
    /// functions: absolute encoded value minus `self_abs_base` -> local index,
    /// `u32::MAX` for a gap. Built once at instantiation.
    ///
    /// Without it, localizing a self-owned funcref read out of a reachable
    /// table falls through to the shared arena on every indirect call, which
    /// costs a `RefCell` borrow pair and a dependent load chain. The local
    /// early-out cannot catch these: `link.rs` keeps the absolute range
    /// disjoint from the local one, so a self-owned absolute handle never
    /// satisfies it.
    self_abs_base: usize,
    self_local_by_abs: Vec<u32>,
    /// Single-decode validation output retained for future tier selection.
    /// Production execution does not consult it yet.
    #[cfg(sf_module_validator)]
    baseline_artifact: BaselineArtifact,
    #[cfg(sf_module_validator)]
    baseline_plan: Vec<FunctionPlanKind>,
    #[cfg(all(test, sf_module_validator))]
    whole_function_plan: Option<Vec<FunctionPlanKind>>,
    #[cfg(all(test, sf_module_validator))]
    whole_function_artifact: Option<Rc<BaselineArtifact>>,
    #[cfg(all(test, sf_module_validator))]
    whole_function_saved: Option<Vec<SavedActivation>>,
    native: Option<NativeState>,
}

/// Scoped access to an interpreter instance while it is being evaluated.
///
/// Occupied instances carry their checkout token across guest and host calls.
/// A standalone or initializing instance instead carries its caller-owned,
/// stable address. Neither form exposes an instance reference beyond a
/// `with_instance` closure.
pub(crate) enum InterpInstanceAccess<'a> {
    CheckedOut(InstanceToken),
    Borrowed {
        pointer: NonNull<InterpInstance>,
        lifetime: PhantomData<&'a mut InterpInstance>,
    },
}

impl InterpInstanceAccess<'static> {
    #[inline]
    pub(crate) fn checked_out(token: InstanceToken) -> Self {
        Self::CheckedOut(token)
    }
}

impl<'a> InterpInstanceAccess<'a> {
    #[inline]
    pub(super) fn borrowed(instance: &'a mut InterpInstance) -> Self {
        Self::Borrowed {
            pointer: NonNull::from(instance),
            lifetime: PhantomData,
        }
    }

    /// Materialize for one non-reentrant read scope.
    #[inline]
    pub(crate) fn with_instance<R>(
        &self,
        use_instance: impl for<'instance> FnOnce(&'instance InterpInstance) -> R,
    ) -> Result<R, WasmError> {
        let instance = match self {
            Self::CheckedOut(token) => token
                .interp()
                .ok_or_else(|| WasmError::internal("instance is not an interpreter"))?,
            Self::Borrowed { pointer, .. } => unsafe { pointer.as_ref() },
        };
        Ok(use_instance(instance))
    }

    /// Materialize for one non-reentrant mutation scope.
    #[inline]
    pub(crate) fn with_instance_mut<R>(
        &mut self,
        use_instance: impl for<'instance> FnOnce(&'instance mut InterpInstance) -> R,
    ) -> Result<R, WasmError> {
        let instance = match self {
            Self::CheckedOut(token) => token
                .interp_mut()
                .ok_or_else(|| WasmError::internal("instance is not an interpreter"))?,
            Self::Borrowed { pointer, .. } => unsafe { pointer.as_mut() },
        };
        Ok(use_instance(instance))
    }
}

pub(crate) struct InterpInstanceLease {
    lease: InstanceLease,
}

impl InterpInstanceLease {
    #[inline]
    pub(crate) fn instance_id(&self) -> InstanceId {
        self.lease.id()
    }

    #[inline]
    pub(crate) fn has_exclusive_lease(&self) -> bool {
        self.lease.is_exclusive()
    }

    #[inline]
    pub(crate) fn into_occupied_id(self) -> InstanceId {
        self.lease.into_occupied_id()
    }

    #[inline]
    pub(crate) fn with_instance<R>(
        &self,
        use_instance: impl for<'instance> FnOnce(&'instance InterpInstance) -> R,
    ) -> R {
        let instance = self
            .lease
            .token()
            .interp()
            .expect("interpreter lease must resolve to an InterpInstance");
        use_instance(instance)
    }

    #[inline]
    pub(crate) fn with_instance_mut<R>(
        &mut self,
        use_instance: impl for<'instance> FnOnce(&'instance mut InterpInstance) -> R,
    ) -> R {
        let instance = self
            .lease
            .token_mut()
            .interp_mut()
            .expect("interpreter lease must resolve to an InterpInstance");
        use_instance(instance)
    }

    pub(crate) fn checkout_for_invocation(&self) -> Result<InstanceToken, WasmError> {
        self.lease
            .checkout_again()
            .ok_or_else(|| WasmError::internal("interpreter instance is no longer available"))
    }

    pub(crate) fn memory(&self) -> Option<&[u8]> {
        self.lease
            .token()
            .interp()
            .expect("interpreter lease must resolve to an InterpInstance")
            .memory()
    }

    pub(crate) fn memory_mut(&mut self) -> Option<&mut [u8]> {
        self.lease
            .token_mut()
            .interp_mut()
            .expect("interpreter lease must resolve to an InterpInstance")
            .memory_mut()
    }
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

/// A numeric `Value` as the raw 64-bit slot this engine stores.
pub(crate) fn value_to_raw_for_interp(v: &Value) -> Result<u64, WasmError> {
    Ok(match v {
        Value::I32(x) => *x as u32 as u64,
        Value::I64(x) => *x as u64,
        Value::F32(x) => x.to_bits() as u64,
        Value::F64(x) => x.to_bits(),
        Value::Ref(handle, _) => ref_to_machine_raw(*handle, SLOT_GP_UNIT_BYTES),
        _ => return Err(WasmError::invalid("interp: unsupported value type")),
    })
}

pub(crate) fn raw_to_value_for_interp(raw: u64, ty: ValueType) -> Result<Value, WasmError> {
    Ok(match ty {
        ValueType::I32 => Value::I32(raw as u32 as i32),
        ValueType::I64 => Value::I64(raw as i64),
        ValueType::F32 => Value::F32(f32::from_bits(raw as u32)),
        ValueType::F64 => Value::F64(f64::from_bits(raw)),
        ValueType::Ref(ref_ty) => Value::Ref(machine_raw_to_ref(raw, SLOT_GP_UNIT_BYTES), ref_ty),
        ValueType::V128 | ValueType::Unknown => {
            return Err(WasmError::invalid("interp: unsupported value type"))
        }
    })
}

/// Evaluate a constant expression to the raw 64-bit slot this engine stores.
///
/// The grammar lives in `vm::const_eval`; this resolver reads
/// already-initialized globals from the flat array (`globals` is what has
/// been initialized so far, which is exactly what `global.get` may reach),
/// resolves `ref.func` through this instance's frame-handle table, and refuses
/// the GC constructors this engine does not support.
fn eval_const(
    module: &Module,
    function_handles: &[RefValue],
    expr: &[u8],
    globals: &[u64],
) -> Result<u64, WasmError> {
    use crate::vm::const_eval::{self, ConstResolver};

    struct R<'a> {
        module: &'a Module,
        function_handles: &'a [RefValue],
        globals: &'a [u64],
    }

    impl ConstResolver for R<'_> {
        fn func_ref(&mut self, func_idx: u32) -> Result<Value, WasmError> {
            let function = self
                .module
                .functions()
                .get(func_idx as usize)
                .ok_or_else(|| WasmError::invalid("ref.func: function index out of range"))?;
            let handle = self
                .function_handles
                .get(func_idx as usize)
                .copied()
                .ok_or_else(|| WasmError::invalid("ref.func: function identity missing"))?;
            Ok(Value::Ref(
                handle,
                crate::value_type::RefType::new(
                    false,
                    crate::value_type::HeapType::Concrete(function.type_index()),
                ),
            ))
        }

        fn global_get(&mut self, global_idx: u32) -> Result<Value, WasmError> {
            // Only an already-initialized global is reachable: the spec
            // restricts a constant expression to imported and
            // previously-declared ones.
            let raw = self
                .globals
                .get(global_idx as usize)
                .copied()
                .ok_or_else(|| {
                    WasmError::invalid("interp: constant expression reads a later global")
                })?;
            let ty = self
                .module
                .globals()
                .get(global_idx as usize)
                .map(|g| g.value_type())
                .ok_or_else(|| WasmError::invalid("global.get: index out of range"))?;
            raw_to_value_for_interp(raw, ty)
        }

        fn ref_i31(&mut self, _value: i32) -> Result<Value, WasmError> {
            Err(WasmError::invalid("interp: GC is not supported"))
        }

        fn alloc_struct(
            &mut self,
            _type_idx: u32,
            _fields: crate::collections::Vec<Value>,
        ) -> Result<Value, WasmError> {
            Err(WasmError::invalid("interp: GC is not supported"))
        }

        fn alloc_array(
            &mut self,
            _type_idx: u32,
            _elements: crate::collections::Vec<Value>,
        ) -> Result<Value, WasmError> {
            Err(WasmError::invalid("interp: GC is not supported"))
        }
    }

    let value = const_eval::eval_const_expr(
        expr,
        module.types(),
        &mut R {
            module,
            function_handles,
            globals,
        },
    )?;
    value_to_raw_for_interp(&value)
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
    /// Registry-aware partial instantiation used by linkers that keep several
    /// interpreter instances in one runtime graph.
    pub(crate) fn new_partial_with_registry(
        engine: &Engine,
        module: Module,
        #[cfg(sf_module_validator)] baseline: super::ValidatedBaselinePlan,
        host: Option<HostDispatch>,
        imports: &[Import],
        funcref_host: Option<FuncRefHost>,
        registry: &LinkRegistry,
    ) -> Result<InterpInstanceLease, (Option<InterpInstanceLease>, WasmError)> {
        let (instance_id, instance_backref) = registry.reserve_instance();
        let lease_handle = instance_backref.clone();
        let inst = match Self::build(
            engine,
            module,
            #[cfg(sf_module_validator)]
            baseline,
            host,
            imports,
            funcref_host,
            registry.arenas(),
            instance_backref,
        ) {
            Ok(inst) => inst,
            Err(error) => {
                registry
                    .instance_table()
                    .abandon(instance_id)
                    .expect("fresh interpreter reservation");
                return Err((None, error));
            }
        };
        match Self::initialize(inst) {
            Ok(inst) => Self::lease_in(registry, instance_id, lease_handle, inst)
                .map_err(|error| (None, error)),
            Err((Some(inst), error)) => {
                let leased = Self::lease_in(registry, instance_id, lease_handle, inst)
                    .map_err(|lease_error| (None, lease_error))?;
                Err((Some(leased), error))
            }
            Err((None, error)) => {
                registry
                    .instance_table()
                    .abandon(instance_id)
                    .expect("unoccupied interpreter reservation");
                Err((None, error))
            }
        }
    }

    pub(in crate::vm) fn initialize(mut inst: Self) -> Result<Self, (Option<Self>, WasmError)> {
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

    fn lease_in(
        registry: &LinkRegistry,
        instance_id: InstanceId,
        lease_handle: InstanceBackref,
        inst: Self,
    ) -> Result<InterpInstanceLease, WasmError> {
        registry
            .instance_table()
            .occupy_interp(instance_id, Box::new(inst))
            .map_err(|_| WasmError::invalid("interpreter instance slot is unavailable"))?;
        let lease = InstanceLease::checkout(&lease_handle)
            .ok_or_else(|| WasmError::invalid("interpreter instance checkout failed"))?;
        Ok(InterpInstanceLease { lease })
    }

    pub(in crate::vm) fn build(
        engine: &Engine,
        module: Module,
        #[cfg(sf_module_validator)] baseline: super::ValidatedBaselinePlan,
        host: Option<HostDispatch>,
        imports: &[Import],
        funcref_host: Option<FuncRefHost>,
        link_registry: LinkArenas,
        instance_backref: InstanceBackref,
    ) -> Result<Self, WasmError> {
        #[cfg(sf_module_validator)]
        let (baseline_artifact, baseline_plan) = baseline.into_parts();
        let config = *engine.config();
        let global_reachable: Vec<bool> = module
            .globals()
            .iter()
            .map(|global| global.is_import() || !global.export_names().is_empty())
            .collect();
        let table_reachable: Vec<bool> = module
            .tables()
            .iter()
            .map(|table| table.is_import() || !table.export_names().is_empty())
            .collect();
        let escapable_functions = module.escapable_functions()?;

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
        // Resolve tag identities before predecode. Catch folding must compare
        // the entities selected by linking, not their module-local indices:
        // two imports can alias one tag, and equal signatures do not make two
        // independently provided tags identical.
        let mut tags: Vec<TagIdentity> = Vec::with_capacity(module.tags().len());
        for tag in module.tags() {
            match tag.def() {
                TagDef::Local(_) => tags.push(TagIdentity::mint_fresh()),
                TagDef::Import {
                    module: md,
                    name,
                    func_type,
                    type_index,
                    ..
                } => {
                    let provided = imports
                        .iter()
                        .find(|imp| imp.module == *md && imp.name == *name)
                        .ok_or(WasmError::unlinkable("missing tag import"))?;
                    let ImportValue::Tag(state) = &provided.value else {
                        return Err(WasmError::unlinkable("incompatible import type"));
                    };
                    // Through the type context where both sides have one: two
                    // `func` types in a rec group are structurally identical
                    // yet distinct identities, and only the context separates
                    // them.
                    let compatible = match (&state.type_ctx, state.type_index) {
                        (Some(ctx), idx) if idx != u32::MAX => concrete_type_matches_cross_context(
                            ctx,
                            idx,
                            module.types(),
                            *type_index,
                        ),
                        _ => {
                            state.func_type.params() == func_type.params()
                                && state.func_type.results() == func_type.results()
                        }
                    };
                    if !compatible {
                        return Err(WasmError::unlinkable("incompatible import type"));
                    }
                    tags.push(state.handle);
                }
            }
        }

        // Functions join the world's one address space before any `ref.func`
        // can be evaluated. A linked import keeps the provider's identity;
        // local functions and host imports name this instance's function
        // slot and retain the local-index frame form.
        link_registry
            .functions
            .observe_local_function_count(module.functions().len());
        let mut function_handles = Vec::with_capacity(module.functions().len());
        let mut function_identities = Vec::with_capacity(module.functions().len());
        for (func_idx, func) in module.functions().iter().enumerate() {
            let linked_handle = match func.def() {
                crate::module::entities::FunctionDef::Import {
                    module: import_module,
                    name,
                    ..
                } => imports.iter().find_map(|import| {
                    if import.module != *import_module || import.name != *name {
                        return None;
                    }
                    match &import.value {
                        ImportValue::Func(ImportedFunction::Linked { handle, .. }) => Some(*handle),
                        _ => None,
                    }
                }),
                crate::module::entities::FunctionDef::Local(_) => None,
            };
            if let Some(handle) = linked_handle {
                if link_registry.functions.entry_for_handle(handle).is_none() {
                    return Err(WasmError::unlinkable(
                        "interpreter linked function identity is unavailable",
                    ));
                }
                function_handles.push(handle);
                function_identities.push(if escapable_functions[func_idx] {
                    handle
                } else {
                    RefValue::null()
                });
                continue;
            }

            let local = RefValue::new(func_idx);
            let absolute = if escapable_functions[func_idx] {
                link_registry.functions.absolute_handle(FuncEntry {
                    owner: instance_backref.self_id(),
                    local_index: u32::try_from(func_idx).map_err(|_| {
                        WasmError::invalid("interpreter function index is too large")
                    })?,
                })
            } else {
                RefValue::null()
            };
            function_handles.push(local);
            function_identities.push(absolute);
        }

        // Reverse map for this instance's own escapable functions. Imports
        // carry a peer's identity and must NOT localize through it.
        let mut self_abs_base = usize::MAX;
        let mut self_abs_hi = 0usize;
        for (idx, func) in module.functions().iter().enumerate() {
            if func.is_import() {
                continue;
            }
            let identity = function_identities[idx];
            if identity.is_null() || identity.is_special() {
                continue;
            }
            let encoded = identity.encoded();
            self_abs_base = self_abs_base.min(encoded);
            self_abs_hi = self_abs_hi.max(encoded);
        }
        let mut self_local_by_abs = Vec::new();
        if self_abs_base != usize::MAX {
            self_local_by_abs.resize(self_abs_hi - self_abs_base + 1, u32::MAX);
            for (idx, func) in module.functions().iter().enumerate() {
                if func.is_import() {
                    continue;
                }
                let identity = function_identities[idx];
                if identity.is_null() || identity.is_special() {
                    continue;
                }
                self_local_by_abs[identity.encoded() - self_abs_base] = idx as u32;
            }
        }

        // Predecode every local function eagerly; imports are dispatched at
        // the slow boundary, so import-free modules remain entirely local.
        let funcs = predecode_functions(&module, &tags, &function_handles)?;
        let import_names = module
            .functions()
            .iter()
            .map(|func| match func.def() {
                crate::module::entities::FunctionDef::Import { module, name, .. } => {
                    Some(Rc::new((module.clone(), name.clone())))
                }
                crate::module::entities::FunctionDef::Local(_) => None,
            })
            .collect();

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
        for (global_idx, g) in module.globals().iter().enumerate() {
            match g.def() {
                GlobalDef::Local(spec) => {
                    let mut v = eval_const(&module, &function_handles, spec.init_expr(), &globals)?;
                    if value_type_is_function_ref(&module, spec.value_type()) {
                        v = if global_reachable[global_idx] {
                            absolutize_slot_with(&module, &function_identities, v)
                        } else {
                            localize_slot_with(&module, &instance_backref, &link_registry, v)
                        };
                    }
                    // An exported local global must BE the shared cell, so an
                    // importer's writes are visible here and ours there. A
                    // private one stays in the array the chain reads.
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
                            let mut raw = value_to_raw_for_interp(&v.value)?;
                            if value_type_is_function_ref(&module, *value_type) {
                                raw = if global_reachable[global_idx] {
                                    absolutize_slot_with(&module, &function_identities, raw)
                                } else {
                                    localize_slot_with(
                                        &module,
                                        &instance_backref,
                                        &link_registry,
                                        raw,
                                    )
                                };
                            }
                            shared_globals.push(None);
                            globals.push(raw)
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
                            let mut raw = st.global.raw();
                            if value_type_is_function_ref(&module, *value_type) {
                                raw = if global_reachable[global_idx] {
                                    absolutize_slot_with(&module, &function_identities, raw)
                                } else {
                                    localize_slot_with(
                                        &module,
                                        &instance_backref,
                                        &link_registry,
                                        raw,
                                    )
                                };
                            }
                            globals.push(raw);
                            continue;
                        }
                        None => return Err(WasmError::unlinkable("missing global import")),
                    }
                }
            }
        }

        // Tables (any reference type) + active element segments.
        let mut tables = Vec::new();
        for (table_idx, t) in module.tables().iter().enumerate() {
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
                    let raw = eval_const(&module, &function_handles, expr, &globals)?;
                    if value_type_is_function_ref(&module, t.value_type()) {
                        if table_reachable[table_idx] {
                            absolutize_slot_with(&module, &function_identities, raw)
                        } else {
                            localize_slot_with(&module, &instance_backref, &link_registry, raw)
                        }
                    } else {
                        raw
                    }
                }
                None => ref_to_machine_raw(RefValue::null(), SLOT_GP_UNIT_BYTES),
            };
            let entries = match (&shared_from_import, shared_out) {
                (Some(inst), _) => TableEntries::Shared(inst.clone()),
                (None, true) => {
                    let inst = TableInst::new(limits.clone(), t.value_type());
                    inst.elements_mut().resize(
                        limits.min(),
                        machine_raw_to_ref(init_slot, SLOT_GP_UNIT_BYTES),
                    );
                    TableEntries::Shared(inst)
                }
                (None, false) => TableEntries::Private(vec![init_slot; limits.min() as usize]),
            };
            tables.push(TableState {
                entries,
                max: limits.max().unwrap_or(u32::MAX as usize) as u64,
            });
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
            funcs: None,
            unlinked_funcs: Some(funcs),
            memories,
            dropped_data,
            dropped_elems,
            globals,
            shared_globals,
            global_reachable,
            tables,
            table_reachable,
            tags,
            instance_backref,
            link_registry,
            config,
            // `drive` materializes the pair immediately before native code
            // can observe it. `Some(empty)` is distinct from the `None`
            // left while an outer call owns the reusable buffers.
            stack: Some(Vec::new()),
            ret_stack: Some(Vec::new()),
            host,
            funcref_host: funcref_host.map(|host| Rc::new(RefCell::new(host))),
            import_names,
            function_handles,
            self_abs_base,
            self_local_by_abs,
            #[cfg(sf_module_validator)]
            baseline_artifact,
            #[cfg(sf_module_validator)]
            baseline_plan,
            #[cfg(all(test, sf_module_validator))]
            whole_function_plan: None,
            #[cfg(all(test, sf_module_validator))]
            whole_function_artifact: None,
            #[cfg(all(test, sf_module_validator))]
            whole_function_saved: Some(Vec::new()),
            function_identities,
            native: None,
        };
        // `initialize` adds the dispatch chain, data segments, and start
        // function in that order: start can call an import and must see
        // initialized memory, and a trap in either has to hand the instance
        // back rather than lose it.
        Ok(inst)
    }

    /// Copy every active element segment into its table.
    ///
    /// After construction for the same reason as the data segments: a trap
    /// here leaves writes that already landed -- possibly in another
    /// instance's table -- and the instance holding them must survive.
    pub(super) fn apply_element_segments(&mut self) -> Result<(), WasmError> {
        for ei in 0..self.module.elements().len() {
            let Some(Element::Active {
                table_index,
                offset_expr,
                init,
            }) = self.module.elements().get(ei).cloned()
            else {
                continue;
            };
            let off = eval_const(
                &self.module,
                &self.function_handles,
                &offset_expr,
                &self.globals,
            )? as u64;
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
                        let handle = self
                            .function_handles
                            .get(fi)
                            .copied()
                            .ok_or(WasmError::invalid("element function identity missing"))?;
                        let slot = self.table_slot_for_storage(
                            table_index,
                            ref_to_machine_raw(handle, SLOT_GP_UNIT_BYTES),
                        );
                        self.tables[table_index]
                            .entries
                            .set(off as usize + k, slot)?;
                    }
                }
                ElementInit::InitExprs { exprs, .. } => {
                    for (k, expr) in exprs.iter().enumerate() {
                        let v = self.table_slot_for_storage(
                            table_index,
                            eval_const(&self.module, &self.function_handles, expr, &self.globals)?,
                        );
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
                let off = eval_const(
                    &self.module,
                    &self.function_handles,
                    offset_expr,
                    &self.globals,
                )? as u64;
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
    fn functions(&self) -> &[Option<Rc<PredecodedFunction>>] {
        self.funcs
            .as_deref()
            .expect("interpreter functions must be published after link")
    }

    #[cfg(all(test, sf_module_validator))]
    pub(super) fn set_whole_function_plan_for_test(
        &mut self,
        artifact: BaselineArtifact,
        mut plan: Vec<FunctionPlanKind>,
    ) -> Result<(), WasmError> {
        if plan.len() != self.module.functions().len()
            || artifact.functions.len() != self.module.functions().len()
        {
            return Err(WasmError::invalid("mixed function plan length mismatch"));
        }
        for (index, (mode, function)) in plan.iter_mut().zip(self.module.functions()).enumerate() {
            *mode = if function.spec().is_none() {
                FunctionPlanKind::Import
            } else {
                match *mode {
                    FunctionPlanKind::Raw => {
                        if raw_stepper_supports_function(&self.module, &artifact, index) {
                            FunctionPlanKind::Raw
                        } else {
                            FunctionPlanKind::FullFold
                        }
                    }
                    FunctionPlanKind::FullFold | FunctionPlanKind::Hybrid => {
                        FunctionPlanKind::FullFold
                    }
                    FunctionPlanKind::Import => {
                        return Err(WasmError::invalid("local function planned as import"));
                    }
                }
            };
        }
        self.whole_function_plan = Some(plan);
        self.whole_function_artifact = Some(Rc::new(artifact));
        Ok(())
    }

    #[cfg(all(test, sf_module_validator))]
    #[inline]
    pub(super) fn whole_function_mode(&self, function: usize) -> FunctionPlanKind {
        self.whole_function_plan
            .as_ref()
            .and_then(|plan| plan.get(function))
            .copied()
            .unwrap_or(FunctionPlanKind::FullFold)
    }

    #[cfg(all(test, sf_module_validator))]
    pub(super) fn whole_direct_call_is_slow_for_test(&self, caller: usize, callee: usize) -> bool {
        let Some(function) = self.functions().get(caller).and_then(Option::as_ref) else {
            return false;
        };
        let Some(pc) = function
            .code
            .iter()
            .position(|instruction| instruction.op == Op::Call && instruction.a as usize == callee)
        else {
            return false;
        };
        let Some(native) = self.native.as_ref() else {
            return false;
        };
        let Some(linked) = native.linked.get(caller).and_then(Option::as_ref) else {
            return false;
        };
        native.plan.cells(linked)[pc].h != native.engine.call_handler_addr()
    }

    #[cfg(all(test, sf_module_validator))]
    pub(crate) fn baseline_plan_label_for_test(&self, function: usize) -> Option<&'static str> {
        self.baseline_plan.get(function).map(|kind| match kind {
            FunctionPlanKind::Raw => "raw",
            FunctionPlanKind::Hybrid => "hybrid",
            FunctionPlanKind::FullFold => "full-fold",
            FunctionPlanKind::Import => "import",
        })
    }

    #[cfg(all(test, sf_module_validator))]
    pub(crate) fn baseline_artifact_has_function_for_test(&self, function: usize) -> Option<bool> {
        self.baseline_artifact
            .functions
            .get(function)
            .map(Option::is_some)
    }

    #[cfg(all(test, sf_module_validator))]
    pub(crate) fn predecode_is_complete_for_test(&self) -> bool {
        let functions = self.functions();
        functions.len() == self.module.functions().len()
            && functions
                .iter()
                .zip(self.module.functions())
                .all(|(decoded, function)| decoded.is_some() == function.spec().is_some())
    }

    fn enable_native_dispatch(&mut self) -> Result<(), WasmError> {
        #[cfg(sf_module_validator)]
        if self.baseline_artifact.functions.len() != self.module.functions().len()
            || self.baseline_plan.len() != self.module.functions().len()
        {
            return Err(WasmError::invalid(
                "interpreter baseline metadata function count mismatch",
            ));
        }
        let engine = NativeEngine::new();
        let mut scratch = LinkScratch::default();
        let mut unlinked = self
            .unlinked_funcs
            .take()
            .ok_or_else(|| WasmError::internal("interpreter functions already linked"))?;
        #[cfg(test)]
        let test_code = Some(unlinked.clone_code_for_oracle());
        #[cfg(not(test))]
        let test_code = None;
        let function_count = unlinked.defined_count();
        let br_entry_count = unlinked.br_entry_count();
        let instrs = unlinked.take_code();
        let mut plan = LinkPlan::from_instr_arena(instrs, function_count, br_entry_count);
        // Seeded with every defined function's own type, then extended by
        // the link pass with each `call_indirect`'s type as it goes.
        let mut used_types: Vec<u32> = (0..unlinked.len())
            .filter_map(|i| self.module.functions().get(i).map(|f| f.type_index()))
            .collect();
        let linked: Vec<Option<LinkedFunction>> = (0..unlinked.len())
            .map(|i| {
                unlinked.function(i).map(|function| {
                    engine.link_in_place(&function, i, &mut plan, &mut scratch, &mut used_types)
                })
            })
            .collect();
        plan.finish_layout();

        // Cross-function fixup: rewire `Call` cells to the native call
        // handler now that every callee's cell block has its final address.
        // Imports and oversized frames stay on the slow path.
        let call_h = engine.call_handler_addr();
        // Callee side: cells base (a low 48), callee l0/l1 offsets
        // (b bits 32-47 / 48-63), frame metadata (c low 48). The caller's
        // own l0/l1 offsets ride in c bits 48-63 and a bits 48-63 so the
        // call handler can stamp them into the return record.
        let callee_info: Vec<Option<(u64, u64, u64, u64, bool)>> = self
            .module
            .functions()
            .iter()
            .enumerate()
            .map(|(i, _)| {
                #[cfg(all(test, sf_module_validator))]
                if self.whole_function_mode(i) == FunctionPlanKind::Raw {
                    return None;
                }
                match (unlinked.function(i), &linked[i]) {
                    (Some(f), Some(lf)) => {
                        let fs = f.frame_slots() as u64;
                        if fs >= 1 << 16 {
                            return None;
                        }
                        let packed = fs << 32 | (f.n_locals() as u64) << 16 | f.n_params() as u64;
                        Some((
                            lf.cell_base(),
                            packed,
                            lf.l0_off as u64,
                            lf.l1_off as u64,
                            lf.fp_pinned,
                        ))
                    }
                    _ => None,
                }
            })
            .collect();
        // Canonical type ids for the native call_indirect type check:
        // `types_equivalent` is an equivalence relation, so numbering the
        // classes densely lets the handler compare one small id. Class
        // representatives are found by linear scan — real modules carry
        // hundreds of types at most.
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

        let indirect_info: Vec<[u64; 3]> = (0..unlinked.len())
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
        // Only call sites recorded by the first link pass are visited here.
        // This replaces the old second sweep over every instruction and its
        // per-cell exception-site binary search.
        for fixup in plan.take_call_fixups() {
            let caller = match fixup {
                CallFixup::Direct { caller, .. } | CallFixup::Indirect { caller, .. } => caller,
            };
            let Some(lf) = linked.get(caller).and_then(|lf| lf.as_ref()) else {
                continue;
            };
            let caller_l0 = lf.l0_off as u64;
            let caller_l1 = lf.l1_off as u64;
            // Rides bit 0 of the recorded l0 offset (byte-scaled, so the
            // bit is structurally free) into every return record.
            let caller_fp = lf.fp_pinned as u64;
            match fixup {
                CallFixup::Direct {
                    cell,
                    callee,
                    arg_base,
                    ..
                } => {
                    if let Some(Some((cells, packed, callee_l0, callee_l1, callee_fp))) =
                        callee_info.get(callee)
                    {
                        *plan.cell_mut(cell) = DCell {
                            h: call_h,
                            a: caller_l1 << 48 | *cells,
                            b: callee_l1 << 48
                                | callee_l0 << 32
                                | (*callee_fp as u64) << 31
                                | arg_base * 8,
                            c: (caller_l0 | caller_fp) << 48 | *packed,
                        };
                    }
                }
                CallFixup::Indirect {
                    cell,
                    table_slot,
                    arg_base,
                    expected_type,
                    ..
                } => {
                    // The expected id must fit the cell's 16-bit field;
                    // an overflow (or unknown type) leaves the site slow.
                    let canon = canon_of.get(expected_type as usize);
                    if let Some(Some(canon)) = canon {
                        if *canon <= 0xFFFF {
                            *plan.cell_mut(cell) = DCell {
                                h: ci_h,
                                a: caller_l1 << 48 | table_slot * 8,
                                b: (caller_l0 | caller_fp) << 48 | canon << 32 | arg_base * 8,
                                // The native handler consumes only a/b. Keep
                                // the original type id here so a guard failure
                                // can rebuild the slow instruction directly.
                                c: expected_type as u64,
                            };
                        }
                    }
                }
            }
        }

        let mut ranges: Vec<(u64, u64, u32)> = Vec::with_capacity(linked.len());
        for (i, lf) in linked.iter().enumerate() {
            if let Some(lf) = lf {
                let start = lf.cell_base();
                // The trailing prefetch pad is not an instruction.
                let end = start + plan.instruction_len(lf) as u64 * 32;
                debug_assert!(ranges.last().is_none_or(|r| r.0 < start));
                ranges.push((start, end, i as u32));
            }
        }

        self.funcs = Some(unlinked.publish(test_code));

        self.native = Some(NativeState {
            engine,
            plan,
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
        let native = self.native.as_ref().expect("native state");
        for function in native.linked.iter().flatten() {
            for w in native.plan.instruction_heads(function).windows(2) {
                let first = (w[0] & 0xffff) as u16;
                let second = (w[1] & 0xffff) as u16;
                if first >= Op::Br as u16 {
                    continue; // control transfer: never a fusible pair
                }
                *map.entry((first, second)).or_insert(0) += 1;
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

    pub(crate) fn boxed_caller_host<F>(host: F) -> HostDispatch
    where
        F: for<'a> FnMut(&str, &str, &mut Caller<'a>, &[u64], &mut [u64]) -> Result<(), WasmError>
            + 'static,
    {
        let host: alloc::boxed::Box<DynHostDispatch> = alloc::boxed::Box::new(host);
        HostDispatch::new(tracked_alloc::box_from_alloc(host))
    }

    /// Resolve the element-segment function value at `seg[k]`.
    ///
    /// Passive segments have no statically private destination, so their
    /// funcrefs are always absolute. `table.init` localizes only when its
    /// eventual destination is private.
    fn elem_value(&self, seg: usize, k: usize) -> Result<u64, WasmError> {
        let element = self
            .module
            .elements()
            .get(seg)
            .ok_or(WasmError::trap("out of bounds table access"))?;
        let raw = match element.get_init() {
            ElementInit::FunctionIndexes(idxs) => idxs
                .get(k)
                .and_then(|&fi| self.function_handles.get(fi as usize))
                .copied()
                .map(|handle| ref_to_machine_raw(handle, SLOT_GP_UNIT_BYTES))
                .ok_or(WasmError::trap("out of bounds table access")),
            // Already in the shared slot encoding: `eval_const` yields the
            // target-width null wire form for `ref.null` and a plain handle
            // for `ref.func`.
            ElementInit::InitExprs { exprs, .. } => exprs
                .get(k)
                .ok_or(WasmError::trap("out of bounds table access"))
                .and_then(|e| eval_const(&self.module, &self.function_handles, e, &self.globals)),
        }?;
        Ok(self.absolutize_slot_for_type(raw, element.value_type()))
    }

    /// This global's shared cell, when it has one.
    ///
    /// Only a global that is imported or exported is held as a `GlobalInst`;
    /// a purely private one lives in the array and has nothing to share.
    pub(crate) fn global_state_at(&self, idx: usize) -> Option<GlobalInst> {
        if !self.global_reachable.get(idx).copied().unwrap_or(true) {
            return None;
        }
        self.shared_globals.get(idx)?.clone()
    }

    /// The runtime identity of tag `idx`, for an importer to link against.
    pub(crate) fn tag_identity_at(&self, idx: usize) -> Option<TagIdentity> {
        self.tags.get(idx).copied()
    }

    fn tag_params_for_handle(&self, handle: TagIdentity) -> Option<&[ValueType]> {
        let idx = self
            .tags
            .iter()
            .position(|candidate| *candidate == handle)?;
        self.module
            .tags()
            .get(idx)
            .map(|tag| tag.func_type().params())
    }

    fn host_value_matches_type(&self, value: &Value, expected: ValueType) -> bool {
        match (value, expected) {
            (Value::I32(_), ValueType::I32)
            | (Value::I64(_), ValueType::I64)
            | (Value::F32(_), ValueType::F32)
            | (Value::F64(_), ValueType::F64) => true,
            #[cfg(sf_has_simd)]
            (Value::V128(_), ValueType::V128) => true,
            (Value::Ref(handle, actual), ValueType::Ref(expected)) => {
                if handle.is_null() {
                    // A null has no dynamic object to inspect; its annotated
                    // bottom/concrete family is therefore part of the value.
                    actual.nullable
                        && expected.nullable
                        && actual
                            .heap_type
                            .is_subtype_of(&expected.heap_type, self.module.types())
                } else {
                    ref_type_matches(*handle, &expected.heap_type, RefTypeOwner::Interp(self))
                        .unwrap_or(false)
                }
            }
            _ => false,
        }
    }

    #[inline]
    fn absolutize_ref(&self, handle: RefValue) -> RefValue {
        absolutize_ref_with(&self.module, &self.function_identities, handle)
    }

    #[inline]
    fn localize_ref(&self, handle: RefValue) -> RefValue {
        // Self-owned absolute handles resolve from an owner-private array.
        // Correct by construction: only non-import functions populate it, so a
        // peer's identity can never hit and still falls through below.
        let encoded = handle.encoded();
        if encoded >= self.self_abs_base {
            if let Some(&local) = self.self_local_by_abs.get(encoded - self.self_abs_base) {
                if local != u32::MAX {
                    return RefValue::new(local as usize);
                }
            }
        }
        localize_ref_with(
            &self.module,
            &self.instance_backref,
            &self.link_registry,
            handle,
        )
    }

    #[inline]
    fn absolutize_slot_for_type(&self, slot: u64, value_type: ValueType) -> u64 {
        if value_type_is_function_ref(&self.module, value_type) {
            ref_to_machine_raw(
                self.absolutize_ref(machine_raw_to_ref(slot, SLOT_GP_UNIT_BYTES)),
                SLOT_GP_UNIT_BYTES,
            )
        } else {
            slot
        }
    }

    #[inline]
    fn localize_slot_for_type(&self, slot: u64, value_type: ValueType) -> u64 {
        if value_type_is_function_ref(&self.module, value_type) {
            ref_to_machine_raw(
                self.localize_ref(machine_raw_to_ref(slot, SLOT_GP_UNIT_BYTES)),
                SLOT_GP_UNIT_BYTES,
            )
        } else {
            slot
        }
    }

    #[inline]
    fn table_slot_for_storage(&self, table_idx: usize, slot: u64) -> u64 {
        let Some(table) = self.module.tables().get(table_idx) else {
            return slot;
        };
        if !value_type_is_function_ref(&self.module, table.value_type()) {
            return slot;
        }
        if self.table_reachable.get(table_idx).copied().unwrap_or(true) {
            ref_to_machine_raw(
                self.absolutize_ref(machine_raw_to_ref(slot, SLOT_GP_UNIT_BYTES)),
                SLOT_GP_UNIT_BYTES,
            )
        } else {
            ref_to_machine_raw(
                self.localize_ref(machine_raw_to_ref(slot, SLOT_GP_UNIT_BYTES)),
                SLOT_GP_UNIT_BYTES,
            )
        }
    }

    #[inline]
    fn table_slot_for_frame(&self, table_idx: usize, slot: u64) -> u64 {
        let Some(table) = self.module.tables().get(table_idx) else {
            return slot;
        };
        self.localize_slot_for_type(slot, table.value_type())
    }

    #[inline]
    fn global_slot_for_storage(&self, global_idx: usize, slot: u64) -> u64 {
        let Some(global) = self.module.globals().get(global_idx) else {
            return slot;
        };
        if !value_type_is_function_ref(&self.module, global.value_type()) {
            return slot;
        }
        if self
            .global_reachable
            .get(global_idx)
            .copied()
            .unwrap_or(true)
        {
            ref_to_machine_raw(
                self.absolutize_ref(machine_raw_to_ref(slot, SLOT_GP_UNIT_BYTES)),
                SLOT_GP_UNIT_BYTES,
            )
        } else {
            ref_to_machine_raw(
                self.localize_ref(machine_raw_to_ref(slot, SLOT_GP_UNIT_BYTES)),
                SLOT_GP_UNIT_BYTES,
            )
        }
    }

    #[inline]
    fn global_slot_for_frame(&self, global_idx: usize, slot: u64) -> u64 {
        let Some(global) = self.module.globals().get(global_idx) else {
            return slot;
        };
        self.localize_slot_for_type(slot, global.value_type())
    }

    /// Read one global in the raw representation used by an interpreter frame.
    ///
    /// Reachable globals store function references in absolute world form, so
    /// callers executing guest code must not read their backing cell directly.
    /// This is the common boundary used by both instruction representations.
    #[inline]
    pub(super) fn global_get_for_frame(&self, global_idx: usize) -> u64 {
        let value = match self
            .shared_globals
            .get(global_idx)
            .and_then(|global| global.as_ref())
        {
            Some(shared) => shared.raw(),
            None => self.globals[global_idx],
        };
        self.global_slot_for_frame(global_idx, value)
    }

    /// Store one raw frame slot into a global's canonical backing form.
    ///
    /// Private function-reference globals keep module-local handles, while
    /// reachable globals keep absolute world identities. The module validator
    /// has already established mutability and the value type for guest writes.
    #[inline]
    pub(super) fn global_set_from_frame(&mut self, global_idx: usize, value: u64) {
        let value = self.global_slot_for_storage(global_idx, value);
        match self
            .shared_globals
            .get_mut(global_idx)
            .and_then(|global| global.as_mut())
        {
            Some(shared) => shared.set_raw(value),
            None => self.globals[global_idx] = value,
        }
    }

    pub(crate) fn absolutize_value_for_type(&self, value: Value, value_type: ValueType) -> Value {
        let Value::Ref(handle, ref_type) = value else {
            return value;
        };
        if value_type_is_function_ref(&self.module, value_type) {
            Value::Ref(self.absolutize_ref(handle), ref_type)
        } else {
            value
        }
    }

    pub(crate) fn localize_value_for_type(&self, value: Value, value_type: ValueType) -> Value {
        let Value::Ref(handle, ref_type) = value else {
            return value;
        };
        if value_type_is_function_ref(&self.module, value_type) {
            Value::Ref(self.localize_ref(handle), ref_type)
        } else {
            value
        }
    }

    /// Convert module-local function indices in an exception payload into
    /// names meaningful to every linked interpreter instance. Other
    /// references are already backend/global handles and remain untouched.
    fn canonicalize_exception_fields(
        &self,
        mut fields: Vec<Value>,
        params: &[ValueType],
    ) -> Result<Vec<Value>, WasmError> {
        for (field, &param) in fields.iter_mut().zip(params) {
            let (Value::Ref(handle, _), ValueType::Ref(ref_type)) = (*field, param) else {
                continue;
            };
            if handle.is_null()
                || handle.is_special()
                || !value_type_is_function_ref(&self.module, param)
            {
                *field = Value::Ref(handle, ref_type);
                continue;
            }
            let handle = if self.module.functions().get(handle.payload()).is_some() {
                self.absolutize_ref(handle)
            } else if self
                .link_registry
                .functions
                .entry_for_handle(handle)
                .is_some()
            {
                handle
            } else {
                return Err(WasmError::trap("host threw mistyped exception"));
            };
            *field = Value::Ref(handle, ref_type);
        }
        Ok(fields)
    }

    /// Translate one of this instance's own absolute function identities back
    /// to its canonical local slot representation. The exception object
    /// remains globally named; only the value installed in the catching frame
    /// is localized, preserving `ref.eq` with a fresh `ref.func` in the
    /// source instance while a different instance keeps the global name.
    fn localize_exception_field(&self, value: Value) -> Value {
        let Value::Ref(handle, ref_type) = value else {
            return value;
        };
        Value::Ref(self.localize_ref(handle), ref_type)
    }

    fn alloc_exception_from_frame(
        &mut self,
        tag_idx: usize,
        frame: &[u64],
        payload_base: usize,
    ) -> Result<PendingException, WasmError> {
        let tag = self
            .tags
            .get(tag_idx)
            .copied()
            .ok_or(WasmError::invalid("interp: bad throw tag"))?;
        let params: Vec<ValueType> = self
            .module
            .tags()
            .get(tag_idx)
            .map(|tag| tag.func_type().params().iter().copied().collect())
            .ok_or(WasmError::invalid("interp: bad throw tag"))?;
        let end = payload_base
            .checked_add(params.len())
            .ok_or(WasmError::invalid("interp: throw payload overflows frame"))?;
        let raw = frame
            .get(payload_base..end)
            .ok_or(WasmError::invalid("interp: throw payload outside frame"))?;
        let mut fields = Vec::with_capacity(params.len());
        for (&value, &ty) in raw.iter().zip(&params) {
            fields.push(raw_to_value_for_interp(value, ty)?);
        }
        let fields = self.canonicalize_exception_fields(fields, &params)?;
        let exn = self.link_registry.alloc_exn(tag, fields);
        Ok(PendingException { exn, tag })
    }

    fn exception_from_ref(&self, exn: RefValue) -> Result<PendingException, WasmError> {
        if exn.is_null() {
            return Err(WasmError::trap("null reference"));
        }
        let instance = self.link_registry.resolve_exn(exn).ok_or(WasmError::trap(
            "throw_ref operand is not an exception reference",
        ))?;
        Ok(PendingException {
            exn,
            tag: instance.tag,
        })
    }

    /// Turn the two catchable error channels at an imported-call boundary
    /// into the interpreter's explicit unwind value. Ordinary traps and
    /// runtime errors remain errors and are never considered by catch_all.
    fn pending_from_error(&mut self, error: WasmError) -> Result<PendingException, WasmError> {
        match error {
            WasmError::Exception { exn, tag, .. } => {
                let resolved = self
                    .link_registry
                    .resolve_exn(exn)
                    .ok_or(WasmError::trap("invalid exception reference"))?;
                if resolved.tag != tag {
                    return Err(WasmError::trap("invalid exception reference"));
                }
                Ok(PendingException { exn, tag })
            }
            WasmError::HostThrow { tag, args } => {
                let Some(params) = self
                    .tag_params_for_handle(tag)
                    .map(|params| params.iter().copied().collect::<Vec<_>>())
                else {
                    return Err(WasmError::trap("host threw mistyped exception"));
                };
                if args.len() != params.len()
                    || !args
                        .iter()
                        .zip(&params)
                        .all(|(value, ty)| self.host_value_matches_type(value, *ty))
                {
                    return Err(WasmError::trap("host threw mistyped exception"));
                }
                let args = self.canonicalize_exception_fields(args, &params)?;
                let exn = self.link_registry.alloc_exn(tag, args);
                Ok(PendingException { exn, tag })
            }
            other => Err(other),
        }
    }

    #[inline]
    fn uncaught_exception(pending: PendingException) -> WasmError {
        WasmError::Exception {
            exn: pending.exn,
            tag: pending.tag,
            module_tag_name: None,
        }
    }

    /// This table's shared entity, when it has one.
    ///
    /// Only a table that is imported or exported is held as a `TableInst`;
    /// a purely private one keeps an array and has nothing to share.
    pub(crate) fn table_state_at(&self, idx: usize) -> Option<TableInst> {
        if !self.table_reachable.get(idx).copied().unwrap_or(true) {
            return None;
        }
        match &self.tables.get(idx)?.entries {
            TableEntries::Shared(t) => Some(t.clone()),
            TableEntries::Private(_) => None,
        }
    }

    /// The current size of table `idx`, which `table.grow` may have changed
    /// since instantiation.
    pub(crate) fn table_len(&self, idx: usize) -> Option<usize> {
        self.tables.get(idx).map(|t| t.entries.len())
    }

    /// The module this instance was built from.
    #[inline]
    pub(crate) fn module(&self) -> &Module {
        &self.module
    }

    #[inline]
    pub(crate) fn instance_backref(&self) -> &InstanceBackref {
        &self.instance_backref
    }

    #[inline]
    pub(crate) fn link_arenas(&self) -> &LinkArenas {
        &self.link_registry
    }

    /// The first linear memory's contents, if the module defines one.
    #[inline]
    pub(crate) fn memory(&self) -> Option<&[u8]> {
        {
            self.memories.first().map(|m| m.bytes())
        }
    }

    #[inline]
    pub(crate) fn memory_mut(&mut self) -> Option<&mut [u8]> {
        {
            self.memories.first_mut().map(|m| m.bytes_mut())
        }
    }

    /// Find an exported function's index by name.
    pub(crate) fn find_export(&self, name: &str) -> Option<usize> {
        self.module
            .functions()
            .iter()
            .position(|f| f.export_names().iter().any(|n| n == name))
    }

    /// (params, results) arity of a function.
    pub(crate) fn func_arity(&self, func_index: usize) -> Option<(usize, usize)> {
        self.module.functions().get(func_index).map(|f| {
            let ft = f.func_type();
            (ft.params().len(), ft.results().len())
        })
    }

    /// The function's absolute world identity for an external boundary.
    pub(crate) fn function_handle_at(&self, func_index: usize) -> Option<RefValue> {
        self.function_identities
            .get(func_index)
            .copied()
            .filter(|handle| !handle.is_null())
    }

    /// The memory entity at `idx`, for another instance to import.
    ///
    /// Sharing the substrate's `MemInst` is what makes an interpreter
    /// instance linkable: the importer takes this backing rather than a
    /// copy, so writes on either side are visible to both.
    pub(crate) fn shared_memory_at(&self, idx: usize) -> Option<MemInst> {
        {
            self.memories.get(idx).map(|m| m.inst.clone())
        }
    }

    /// A global's raw 64-bit value by index.
    pub(crate) fn global_at(&self, idx: usize) -> Option<u64> {
        // Through the shared cell where there is one: the array slot beside a
        // shared global is stale by design, so reading it would report the
        // value as of instantiation rather than now.
        match self.shared_globals.get(idx)?.as_ref() {
            Some(shared) => Some(shared.raw()),
            None => self.globals.get(idx).copied(),
        }
    }

    /// Overwrite a global's raw 64-bit value by index.
    pub(crate) fn set_global_at(&mut self, idx: usize, raw: u64) -> Result<(), WasmError> {
        let raw = self.global_slot_for_storage(idx, raw);
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
    pub(crate) fn find_export_global(&self, name: &str) -> Option<usize> {
        self.module
            .globals()
            .iter()
            .position(|g| g.export_names().iter().any(|n| n == name))
    }

    /// Invoke a function by index. `args` and `results` are this instance's
    /// raw frame slots (i32/f32 in the low bits, own funcrefs local).
    ///
    /// The typed `Instance` adapter converts its absolute [`Value`]s at this
    /// boundary; direct raw callers already speak the
    /// interpreter frame form.
    pub(crate) fn invoke(
        &mut self,
        func_index: usize,
        args: &[u64],
        results: &mut [u64],
    ) -> Result<(), WasmError> {
        let mut access = InterpInstanceAccess::borrowed(self);
        Self::invoke_access(&mut access, func_index, args, results)
    }

    pub(crate) fn invoke_access(
        access: &mut InterpInstanceAccess<'_>,
        func_index: usize,
        args: &[u64],
        results: &mut [u64],
    ) -> Result<(), WasmError> {
        let entry =
            access.with_instance(|instance| instance.functions().get(func_index).cloned())?;
        let Some(entry) = entry else {
            return Err(WasmError::invalid("interp: bad function index"));
        };
        let Some(func) = entry else {
            // An imported function, reached directly rather than through a
            // call inside a body -- `(start $imported)` does exactly this.
            // It has no predecoded body to enter; the host is the callee.
            let frame_len = args.len().max(results.len());
            let mut frame = vec![0u64; frame_len];
            frame[..args.len()].copy_from_slice(args);
            let call = access
                .with_instance(|instance| instance.prepare_import_call(func_index, &frame, 0))??;
            match Self::run_external_call(access, &call) {
                Ok(returned) => frame[..returned.len()].copy_from_slice(&returned),
                Err(error) => {
                    let pending = access
                        .with_instance_mut(|instance| instance.pending_from_error(error))??;
                    return Err(Self::uncaught_exception(pending));
                }
            }
            results.copy_from_slice(&frame[..results.len()]);
            return Ok(());
        };
        if args.len() != func.n_params as usize || results.len() != func.n_results as usize {
            return Err(WasmError::invalid("interp: argument/result arity mismatch"));
        }
        #[cfg(all(test, sf_module_validator))]
        let raw = access.with_instance(|instance| {
            (instance.whole_function_mode(func_index) == FunctionPlanKind::Raw).then_some(
                RawFrameState {
                    pc: 0,
                    stp: 0,
                    top: func.n_locals as usize,
                    operand_base: func.n_locals as usize,
                    return_base: 0,
                },
            )
        })?;
        let root = Activation {
            func,
            func_index,
            base: 0,
            pc: 0,
            route_established: false,
            #[cfg(all(test, sf_module_validator))]
            raw,
        };
        Self::drive(access, root, args, results)
    }

    /// Stub for targets without a native backend: [`Self::new`] fails
    /// there, so no instance exists to invoke — this only keeps
    /// cross-target callers compiling.

    /// The call/return trampoline: runs activations to their next call or
    /// return boundary, keeping call depth as data on `saved`.
    fn drive(
        access: &mut InterpInstanceAccess<'_>,
        root: Activation,
        args: &[u64],
        results: &mut [u64],
    ) -> Result<(), WasmError> {
        let (mut stack, mut ret_stack, reentrant) = access.with_instance_mut(|instance| {
            if instance
                .memories
                .iter()
                .any(|memory| memory.inst.host_callback_borrowed())
            {
                return Err(WasmError::trap(
                    "linear memory is borrowed by a host callback",
                ));
            }

            // Take the instance's buffers before the first guest step. A
            // re-entry sees `None` and allocates its own pair. The owner of
            // a never-entered instance sees `Some(empty)`, materializes the
            // reusable pair once, and hands it back after the call.
            let stack = instance.stack.take();
            let ret_stack = instance.ret_stack.take();
            let reentrant = stack.is_none();
            debug_assert_eq!(reentrant, ret_stack.is_none());
            let mut stack = stack.unwrap_or_default();
            let mut ret_stack = ret_stack.unwrap_or_default();
            if stack.is_empty() {
                let slots = configured_stack_slots(&instance.config);
                // Zeroed in full: native dispatch roams the whole region
                // through raw pointers, so every slot must be initialized.
                stack = vec![0u64; slots];
                ret_stack = vec![0u64; configured_ret_records(slots) * (RET_RECORD / 8)];
            }
            Ok((stack, ret_stack, reentrant))
        })??;

        let outcome = Self::drive_on(access, root, args, &mut stack, &mut ret_stack, results);

        // Only the owner hands them back; a nested call drops its own.
        if !reentrant {
            access.with_instance_mut(|instance| {
                instance.stack = Some(stack);
                instance.ret_stack = Some(ret_stack);
            })?;
        }
        outcome
    }

    /// Results are copied out here, while the stack that produced them is
    /// still in hand -- a nested call runs on a different one.
    fn drive_on(
        access: &mut InterpInstanceAccess<'_>,
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
        #[cfg(all(test, sf_module_validator))]
        let (mut saved, reuse_saved) = access.with_instance_mut(|instance| {
            let reuse = instance.whole_function_plan.is_some();
            let saved = if reuse {
                instance.whole_function_saved.take().unwrap_or_default()
            } else {
                Vec::new()
            };
            (saved, reuse)
        })?;
        #[cfg(not(all(test, sf_module_validator)))]
        let mut saved: Vec<SavedActivation> = Vec::new();
        let outcome = (|| -> Result<(), WasmError> {
            loop {
                // Local dispatch cannot re-enter the instance table. Keep one
                // materialization through native sessions, slow instructions,
                // local calls, returns, and exception handling. The helper can
                // leave this scope only with an owned external-call request.
                let exit = access.with_instance_mut(|instance| {
                    instance.run_materialized(&mut act, &mut saved, &mut ctx, max_depth, results)
                })??;
                let MaterializedRunExit::ExternalCall {
                    call,
                    result_base,
                    error_site,
                    resume_tail_return,
                } = exit
                else {
                    return Ok(());
                };

                // This is the re-entry barrier. Host callbacks, installed
                // funcref hooks, and foreign instances may all check out this
                // same slot, so no token-derived instance reference is live.
                match Self::run_external_call(access, &call) {
                    Ok(returned) => {
                        ctx.stack[result_base..result_base + returned.len()]
                            .copy_from_slice(&returned);
                        ctx.acc = returned.first().copied().unwrap_or(0);
                        if resume_tail_return {
                            // Preserve the imported-tail ordering: the host call
                            // and its result writes happen before this internal
                            // landing invariant is validated.
                            act.pc = act.func.slow_tail_return.ok_or(WasmError::invalid(
                                "interp: tail call has no return landing",
                            ))? as usize;
                        }
                    }
                    Err(error) => {
                        let pending = access
                            .with_instance_mut(|instance| instance.pending_from_error(error))??;
                        access.with_instance(|instance| {
                            instance.unwind_exception(
                                pending, &mut act, &mut saved, &mut ctx, error_site,
                            )
                        })??;
                    }
                }
            }
        })();
        #[cfg(all(test, sf_module_validator))]
        if reuse_saved {
            saved.clear();
            access.with_instance_mut(|instance| instance.whole_function_saved = Some(saved))?;
        }
        outcome
    }

    /// Run only work that cannot execute guest or host code outside this
    /// instance. Imported and external targets are prepared here, but their
    /// owned requests are returned to [`Self::drive_on`] before invocation.
    // One call enters this loop per materialization; local calls and returns
    // stay inside it. Keeping that scheduling region separate prevents the
    // token boundary's slow paths from forcing its state to spill.
    #[inline(never)]
    fn run_materialized(
        &mut self,
        act: &mut Activation,
        saved: &mut Vec<SavedActivation>,
        ctx: &mut DriveCtx<'_>,
        max_depth: u32,
        results: &mut [u64],
    ) -> Result<MaterializedRunExit, WasmError> {
        loop {
            #[cfg(all(test, sf_module_validator))]
            let step = if act.raw.is_some() {
                let artifact = Rc::clone(
                    self.whole_function_artifact
                        .as_ref()
                        .ok_or_else(|| WasmError::internal("mixed Raw artifact is missing"))?,
                );
                let raw = act.raw.as_mut().expect("checked mixed Raw activation");
                let mut top = raw.top;
                let exit = RawStepper {
                    instance: self,
                    artifact: &artifact,
                    function_index: act.func_index,
                    frame_base: act.base,
                    state: raw,
                    slots: RawSlots::Fixed {
                        slots: ctx.stack,
                        top: &mut top,
                    },
                    acc: &mut ctx.acc,
                }
                .step()
                .map_err(|error| match error {
                    super::baseline_exec::BaselineExecError::Wasm(error) => error,
                    super::baseline_exec::BaselineExecError::Unsupported { .. } => {
                        WasmError::invalid("mixed Raw plan reached an unsupported opcode")
                    }
                })?;
                raw.top = top;
                match exit {
                    RawStepExit::Continue => continue,
                    RawStepExit::Call { callee, arg_base } => StepExit::Call { callee, arg_base },
                    RawStepExit::Return => StepExit::Return,
                }
            } else {
                self.native_step(act, ctx)?
            };
            #[cfg(not(all(test, sf_module_validator)))]
            let step = self.native_step(act, ctx)?;
            match step {
                StepExit::Call { callee, arg_base } => {
                    if saved.len() as u32 >= max_depth {
                        return Err(WasmError::trap("call stack exhausted"));
                    }
                    let f = match self.functions().get(callee).cloned() {
                        Some(Some(f)) => f,
                        Some(None) => {
                            // Imported function: dispatch to the host. Its
                            // first result rides the accumulator relay like
                            // any other call result.
                            let base = act.base;
                            let call =
                                self.prepare_import_call(callee, &ctx.stack[base..], arg_base)?;
                            return Ok(MaterializedRunExit::ExternalCall {
                                call: Box::new(call),
                                result_base: base + arg_base,
                                error_site: act.pc.checked_sub(1),
                                resume_tail_return: false,
                            });
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
                    #[cfg(all(test, sf_module_validator))]
                    let raw = if self.whole_function_mode(callee) == FunctionPlanKind::Raw {
                        let artifact = self
                            .whole_function_artifact
                            .as_ref()
                            .and_then(|artifact| artifact.functions.get(callee))
                            .and_then(Option::as_ref)
                            .ok_or_else(|| {
                                WasmError::invalid("mixed Raw callee artifact is missing")
                            })?;
                        let operand_base = new_base + nl;
                        let required = operand_base
                            .checked_add(artifact.max_operand_height as usize)
                            .ok_or_else(|| WasmError::trap("call stack exhausted"))?;
                        if required > ctx.stack.len() {
                            return Err(WasmError::trap("call stack exhausted"));
                        }
                        Some(RawFrameState {
                            pc: 0,
                            stp: 0,
                            top: operand_base,
                            operand_base,
                            return_base: new_base,
                        })
                    } else {
                        None
                    };
                    #[cfg(all(test, sf_module_validator))]
                    let raw_return_top = act.raw.as_ref().map(|_| new_base + f.n_results as usize);
                    let callee_act = Activation {
                        func: f,
                        func_index: callee,
                        base: new_base,
                        pc: 0,
                        route_established: false,
                        #[cfg(all(test, sf_module_validator))]
                        raw,
                    };
                    // Dominate `push`'s capacity check so the hot path can
                    // copy the caller straight into the spare Vec slot.
                    while saved.len() == saved.capacity() {
                        grow_saved_activations(saved);
                    }
                    saved.push(SavedActivation {
                        activation: core::mem::replace(act, callee_act),
                        ret_cursor: ctx.ret_cursor,
                        #[cfg(all(test, sf_module_validator))]
                        raw_return_top,
                    });
                }
                StepExit::TailCall { callee, arg_base } => {
                    // The callee returns to THIS activation's caller, so it
                    // reuses this frame: arguments slide down to `act.base`
                    // (which is where the caller staged ours, and so where
                    // our results are expected), and the activation is
                    // replaced rather than pushed. Recursion through a tail
                    // call therefore runs at constant depth.
                    let f = match self.functions().get(callee).cloned() {
                        Some(Some(f)) => f,
                        Some(None) => {
                            // An imported tail callee: run the host, leave
                            // its results at this frame's base, and return
                            // to the caller as this activation would have.
                            let base = act.base;
                            let call =
                                self.prepare_import_call(callee, &ctx.stack[base..], arg_base)?;
                            return Ok(MaterializedRunExit::ExternalCall {
                                call: Box::new(call),
                                result_base: base,
                                // A tail call retires this activation before
                                // entering the callee, so its try scopes are
                                // not candidates.
                                error_site: None,
                                // Do not synthesize a Rust return. After the
                                // host succeeds, the landing executes an
                                // ordinary native Return and consumes every
                                // native-only caller route before the sentinel.
                                resume_tail_return: true,
                            });
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
                StepExit::Throw {
                    pending,
                    search_current,
                } => {
                    let site = search_current.then_some(act.pc);
                    self.unwind_exception(pending, act, saved, ctx, site)?;
                }
                StepExit::ExternalCall {
                    call,
                    result_base,
                    site,
                    search_current,
                } => {
                    return Ok(MaterializedRunExit::ExternalCall {
                        call,
                        result_base,
                        error_site: search_current.then_some(site),
                        resume_tail_return: false,
                    });
                }
                StepExit::Return => {
                    // Results are already in place: the callee's frame base
                    // IS the caller's staged-argument slot.
                    match saved.pop() {
                        None => {
                            results.copy_from_slice(&ctx.stack[..results.len()]);
                            return Ok(MaterializedRunExit::Complete);
                        }
                        Some(parent) => {
                            // Native Return already popped the callee's
                            // sentinel. Reasserting the saved checkpoint makes
                            // the synchronization invariant explicit and also
                            // covers targets whose backend reports Return
                            // without exposing its cursor mechanics.
                            ctx.ret_cursor = parent.ret_cursor;
                            #[cfg(all(test, sf_module_validator))]
                            let mut activation = parent.activation;
                            #[cfg(not(all(test, sf_module_validator)))]
                            let activation = parent.activation;
                            #[cfg(all(test, sf_module_validator))]
                            if let (Some(raw), Some(top)) =
                                (activation.raw.as_mut(), parent.raw_return_top)
                            {
                                raw.top = top;
                            }
                            *act = activation;
                        }
                    }
                }
            }
        }
    }

    pub(super) fn run_external_call(
        access: &InterpInstanceAccess<'_>,
        call: &PreparedCall,
    ) -> Result<Vec<u64>, WasmError> {
        let mut results = call.invoke()?;
        access.with_instance(|instance| instance.localize_call_results(call, &mut results))?;
        Ok(results)
    }

    /// Match and install one handler in `act`.
    ///
    /// The predecoder has already canonicalized the handler target's stack
    /// base. Runtime work is therefore limited to identity matching and
    /// copying the exception payload (plus the reference for `_ref` clauses)
    /// into those authoritative slots.
    fn try_handle_exception(
        &self,
        pending: PendingException,
        act: &mut Activation,
        ctx: &mut DriveCtx<'_>,
        site: usize,
    ) -> Result<bool, WasmError> {
        let Some(handler) = act
            .func
            .exception_handlers_at(site as u32)
            .iter()
            .copied()
            .find(|handler| handler.tag.is_none_or(|tag| tag == pending.tag))
        else {
            return Ok(false);
        };

        let base = act
            .base
            .checked_add(handler.target_base as usize)
            .ok_or(WasmError::invalid("interp: exception target slot overflow"))?;
        let payload_arity = handler.payload_arity as usize;
        let total = payload_arity + usize::from(handler.forwards_exn);
        let end = base
            .checked_add(total)
            .ok_or(WasmError::invalid("interp: exception target slot overflow"))?;
        let target = ctx
            .stack
            .get_mut(base..end)
            .ok_or(WasmError::invalid("interp: exception target outside frame"))?;

        let resolved = (handler.tag.is_some() || handler.forwards_exn)
            .then(|| self.link_registry.resolve_exn(pending.exn))
            .flatten();
        if (handler.tag.is_some() || handler.forwards_exn) && resolved.is_none() {
            return Err(WasmError::trap("invalid exception reference during catch"));
        }
        if let Some(exn) = resolved {
            if exn.tag != pending.tag
                || (handler.tag.is_some() && exn.fields.len() != payload_arity)
            {
                return Err(WasmError::trap("mistyped exception payload"));
            }
            if handler.tag.is_some() {
                for (dst, value) in target[..payload_arity].iter_mut().zip(&exn.fields) {
                    *dst = value_to_raw_for_interp(&self.localize_exception_field(*value))?;
                }
            }
        }
        if handler.forwards_exn {
            target[payload_arity] = ref_to_machine_raw(pending.exn, SLOT_GP_UNIT_BYTES);
        }

        act.pc = handler.target as usize;
        // A catch target is a merge. Slots are authoritative and the first
        // re-entered cell must not inherit an accumulator edge from the throw.
        ctx.acc = 0;
        Ok(true)
    }

    /// Walk Rust activation checkpoints until `pending` is caught.
    ///
    /// Calls that execute under a `try_table` stay slow, so no native-only
    /// caller can own a handler. Restoring a checkpoint atomically discards
    /// the leaving activation's sentinel and all native descendants while
    /// retaining the selected caller's established return route.
    fn unwind_exception(
        &self,
        pending: PendingException,
        act: &mut Activation,
        saved: &mut Vec<SavedActivation>,
        ctx: &mut DriveCtx<'_>,
        mut current_site: Option<usize>,
    ) -> Result<(), WasmError> {
        loop {
            if let Some(site) = current_site {
                if self.try_handle_exception(pending, act, ctx, site)? {
                    return Ok(());
                }
            }

            let Some(parent) = saved.pop() else {
                return Err(Self::uncaught_exception(pending));
            };
            ctx.ret_cursor = parent.ret_cursor;
            *act = parent.activation;
            current_site = act.pc.checked_sub(1);
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

    pub(super) fn mem_load(
        &self,
        addr: u64,
        mem_idx: usize,
        offset: u64,
        size: usize,
    ) -> Result<u64, WasmError> {
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
        let mut bytes = [0u8; 8];
        if !mem.inst.read_bytes(ea as usize, &mut bytes[..size]) {
            return Err(WasmError::trap("out of bounds memory access"));
        }
        Ok(u64::from_le_bytes(bytes))
    }

    pub(super) fn mem_store(
        &mut self,
        addr: u64,
        mem_idx: usize,
        offset: u64,
        size: usize,
        value: u64,
    ) -> Result<(), WasmError> {
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
        if !mem
            .inst
            .write_bytes(ea as usize, &value.to_le_bytes()[..size])
        {
            return Err(WasmError::trap("out of bounds memory access"));
        }
        Ok(())
    }

    /// Current page count in the memory's index type, represented as a raw
    /// interpreter slot. Invalid static indices have already been rejected by
    /// validation; preserving the old zero fallback keeps the slow path exact.
    #[inline]
    pub(super) fn memory_size(&self, mem_idx: usize) -> u64 {
        self.memories
            .get(mem_idx)
            .map(|memory| memory.len() / PAGE)
            .unwrap_or(0) as u64
    }

    /// Grow a memory and return its previous page count, or the memory's
    /// correctly-sized -1 sentinel when a limit or host allocation refuses it.
    pub(super) fn memory_grow(&mut self, mem_idx: usize, raw_delta: u64) -> Result<u64, WasmError> {
        let runtime_cap = self.config.get_wasm_memory_max_pages() as u64;
        let mem = self
            .memories
            .get_mut(mem_idx)
            .ok_or(WasmError::trap("out of bounds memory access"))?;
        // Both the delta and the -1 failure result belong to the memory's
        // index type, so a 64-bit memory must not truncate either to 32 bits.
        let (delta, fail, cap) = if mem.is64 {
            (raw_delta, u64::MAX, 1u64 << 48)
        } else {
            (raw_delta as u32 as u64, u32::MAX as u64, 65536)
        };
        let cur = (mem.len() / PAGE) as u64;
        let want = cur.saturating_add(delta);
        // The declared maximum is a type-level ceiling; the engine's
        // configured cap applies at grow time (`check_memory_quota` states
        // this split), and a failed host allocation is the spec's -1 result,
        // not an abort. Sizes are computed in u64: `want as usize * PAGE`
        // would truncate a memory64 request on a 32-bit host before any check
        // could reject it.
        let new_len = want.checked_mul(PAGE as u64);
        match new_len {
            Some(new_len)
                if want <= mem.max_pages
                    && want <= cap
                    && want <= runtime_cap
                    && new_len <= usize::MAX as u64 =>
            {
                if mem.inst.try_grow_to(new_len as usize) {
                    Ok(cur)
                } else {
                    Ok(fail)
                }
            }
            _ => Ok(fail),
        }
    }

    /// Prepare an imported call while the caller is materialized.
    ///
    /// `frame` is the caller's frame slice. The returned request owns every
    /// value it needs, so the materialization can end before guest or host
    /// code runs.
    pub(super) fn prepare_import_call(
        &self,
        callee: usize,
        frame: &[u64],
        arg_base: usize,
    ) -> Result<PreparedCall, WasmError> {
        let func = self
            .module
            .functions()
            .get(callee)
            .ok_or(WasmError::trap("undefined element"))?;
        match func.def() {
            crate::module::entities::FunctionDef::Import { .. } => {}
            _ => return Err(WasmError::invalid("interp: not an import")),
        }
        let func_type = func.func_type();
        let p = func_type.params().len();
        let callee_handle =
            self.function_handles
                .get(callee)
                .copied()
                .ok_or(WasmError::invalid(
                    "interp: imported function identity is missing",
                ))?;
        let world = self
            .link_registry
            .functions
            .entry_for_handle(callee_handle)
            .filter(|entry| entry.owner != self.instance_backref.self_id())
            .map(|entry| WorldFuncTarget {
                entry,
                checkout: self.instance_backref.clone(),
            });
        if func_type.results().len() > 8 {
            return Err(WasmError::invalid("interp: too many host results"));
        }
        let mut args: Vec<u64> = frame[arg_base..arg_base + p].iter().copied().collect();
        for (arg, &value_type) in args.iter_mut().zip(func_type.params()) {
            if value_type_is_function_ref(&self.module, value_type) {
                *arg = absolutize_slot_with(&self.module, &self.function_identities, *arg);
            }
        }
        let target = if world.is_some() {
            ExternalCallTarget::FuncRef {
                host: self.funcref_host.clone(),
                handle: callee_handle,
                world,
            }
        } else {
            let dispatch = self
                .host
                .as_ref()
                .ok_or(WasmError::invalid("interp: no host dispatcher installed"))?;
            let names = self
                .import_names
                .get(callee)
                .and_then(|names| names.as_ref())
                .ok_or(WasmError::invalid("interp: import names are missing"))?;
            ExternalCallTarget::Host {
                dispatch: dispatch.clone(),
                names: Rc::clone(names),
                memory: self.memories.first().map(|memory| memory.inst.clone()),
            }
        };
        Ok(PreparedCall {
            target,
            args,
            result_types: func_type.results().iter().copied().collect(),
        })
    }

    fn prepare_funcref_call(
        &self,
        handle: RefValue,
        param_types: Vec<ValueType>,
        result_types: Vec<ValueType>,
        frame: &[u64],
        arg_base: usize,
        entry: Option<FuncEntry>,
    ) -> PreparedCall {
        let mut args = Vec::with_capacity(param_types.len());
        args.extend_from_slice(&frame[arg_base..arg_base + param_types.len()]);
        for (arg, &value_type) in args.iter_mut().zip(&param_types) {
            *arg = self.absolutize_slot_for_type(*arg, value_type);
        }
        PreparedCall {
            target: ExternalCallTarget::FuncRef {
                host: self.funcref_host.clone(),
                handle,
                world: entry.map(|entry| WorldFuncTarget {
                    entry,
                    checkout: self.instance_backref.clone(),
                }),
            },
            args,
            result_types,
        }
    }

    /// Enter a registry-known interpreter function through absolute raw
    /// slots.
    ///
    /// The caller has already ended its materialization and converted every
    /// funcref argument to world form before this function checks out the
    /// owner. The callee is materialized only in short conversion scopes
    /// around the ordinary invocation, so guest execution never overlaps a
    /// token-derived `&mut InterpInstance`.
    fn invoke_world_target(
        target: &WorldFuncTarget,
        args: &[u64],
        expected_results: usize,
    ) -> Result<Vec<u64>, WasmError> {
        let token = target
            .checkout
            .checkout(target.entry.owner)
            .ok_or_else(|| {
                WasmError::internal("interpreter function owner is no longer available")
            })?;
        let mut access = InterpInstanceAccess::checked_out(token);
        let local_index = target.entry.local_index as usize;
        let (local_args, result_types) = access.with_instance(|instance| {
            let func_type = instance
                .module
                .functions()
                .get(local_index)
                .ok_or_else(|| WasmError::internal("interpreter function index is out of range"))?
                .func_type();
            if args.len() != func_type.params().len()
                || expected_results != func_type.results().len()
            {
                return Err(WasmError::internal(
                    "interpreter linked function arity does not match",
                ));
            }

            let local_args = args
                .iter()
                .copied()
                .zip(func_type.params().iter().copied())
                .map(|(arg, value_type)| instance.localize_slot_for_type(arg, value_type))
                .collect::<Vec<_>>();
            let result_types = func_type.results().iter().copied().collect::<Vec<_>>();
            Ok::<_, WasmError>((local_args, result_types))
        })??;

        let mut results = vec![0u64; result_types.len()];
        Self::invoke_access(&mut access, local_index, &local_args, &mut results)?;

        access.with_instance(|instance| {
            for (result, value_type) in results.iter_mut().zip(result_types) {
                *result = instance.absolutize_slot_for_type(*result, value_type);
            }
        })?;
        Ok(results)
    }

    fn localize_call_results(&self, call: &PreparedCall, results: &mut [u64]) {
        for (result, &value_type) in results.iter_mut().zip(&call.result_types) {
            *result = self.localize_slot_for_type(*result, value_type);
        }
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
                        lf.cell_base(),
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
            indirect_len: self
                .native
                .as_ref()
                .map_or(0, |n| n.indirect_info.len() as u64),
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
                // shared one holds `RefValue`, so it is left at zero and its
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
                    let f = self
                        .functions()
                        .get(fi)
                        .cloned()
                        .flatten()
                        .expect("range map import");
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
                    let (cell, fast_indirect) = {
                        let native = self.native.as_ref().expect("native state");
                        let cell = native
                            .plan
                            .cell_index(state.pc)
                            .ok_or(WasmError::invalid("interp: native pc out of range"))?;
                        let payload = native.plan.native_call_indirect_payload(
                            cell,
                            native.engine.callindirect_handler_addr(),
                        );
                        (cell, payload)
                    };
                    let mut restored_ins = None;
                    let slow_op = if fast_indirect.is_some() {
                        Op::CallIndirect
                    } else {
                        let ins = {
                            let native = self.native.as_ref().expect("native state");
                            native
                                .plan
                                .restore_slow_instr(cell, native.engine.slow_stub_addr())
                                .ok_or(WasmError::internal(
                                    "interpreter native slow payload is not recoverable",
                                ))?
                        };
                        let op = ins.op;
                        restored_ins = Some(ins);
                        op
                    };
                    if let Some(native) = self.native.as_mut() {
                        native.slow_exits[slow_op as usize] += 1;
                    }
                    let frame = &mut ctx.stack[base..base + f.frame_slots as usize];
                    let result = match fast_indirect {
                        Some((table_slot, arg_base, expected_type)) => {
                            let table_element = frame[table_slot as usize];
                            self.exec_call_indirect(
                                frame,
                                &f,
                                table_element,
                                arg_base as usize,
                                expected_type as u64,
                                false,
                            )
                        }
                        None => self.exec_ins(
                            frame,
                            &f,
                            restored_ins.expect("generic slow exit has an instruction"),
                        ),
                    };
                    let effect = match result {
                        Ok(effect) => effect,
                        Err(error) => match self.pending_from_error(error) {
                            Ok(pending) => {
                                act.pc = idx;
                                let search_current = !matches!(
                                    slow_op,
                                    Op::ReturnCall | Op::ReturnCallIndirect | Op::ReturnCallRef
                                );
                                return Ok(StepExit::Throw {
                                    pending,
                                    search_current,
                                });
                            }
                            Err(error) => return Err(error),
                        },
                    };
                    match effect {
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
                        Effect::ExternalCall {
                            call,
                            arg_base,
                            tail_target,
                        } => {
                            let tail = tail_target.is_some();
                            act.pc = tail_target.unwrap_or(idx + 1);
                            return Ok(StepExit::ExternalCall {
                                call,
                                result_base: if tail { base } else { base + arg_base },
                                site: idx,
                                search_current: !tail,
                            });
                        }
                        Effect::Throw(pending) => {
                            // Unlike a normal call exit, a throw is handled at
                            // the throwing instruction itself.
                            act.pc = idx;
                            return Ok(StepExit::Throw {
                                pending,
                                search_current: true,
                            });
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

    /// Whether a baseline indirect call can treat this table as closed world.
    /// Imported/exported or growable tables can change outside the baseline
    /// executor and remain on its explicit unsupported boundary for now.
    #[cfg(test)]
    pub(super) fn call_indirect_table_is_closed(&self, table_index: usize) -> bool {
        let Some(table) = self.tables.get(table_index) else {
            return false;
        };
        !self
            .table_reachable
            .get(table_index)
            .copied()
            .unwrap_or(true)
            && matches!(&table.entries, TableEntries::Private(_))
            && table.max == table.entries.len() as u64
    }

    /// Resolve the semantic target shared by the folded and raw baseline
    /// representations. The result owns any request that must leave the
    /// instance materialization; local targets carry only their function index.
    #[inline(always)]
    pub(super) fn resolve_call_indirect(
        &self,
        frame: &[u64],
        table_element: u64,
        arg_base: usize,
        table_index: usize,
        expected_type: u32,
    ) -> Result<ResolvedIndirectCall, WasmError> {
        let table = self
            .tables
            .get(table_index)
            .ok_or(WasmError::trap("undefined element"))?;
        let fi = usize::try_from(table_element)
            .ok()
            .and_then(|element| table.entries.get(element))
            .ok_or(WasmError::trap("undefined element"))?;
        let callee = machine_raw_to_ref(fi, SLOT_GP_UNIT_BYTES);
        if callee.is_null() {
            return Err(WasmError::trap("uninitialized element"));
        }
        if callee.is_special() {
            return Err(WasmError::trap("indirect call type mismatch"));
        }
        let local = self.localize_ref(callee);
        if local.encoded() >= self.module.functions().len() {
            return self.resolve_external_indirect_call(callee, frame, arg_base, expected_type);
        }
        let fi = local.encoded() as u32;
        let actual = self
            .module
            .functions()
            .get(fi as usize)
            .ok_or(WasmError::trap("undefined element"))?
            .type_index();
        if !self.module.types().types_equivalent(expected_type, actual) {
            return Err(WasmError::trap("indirect call type mismatch"));
        }
        Ok(ResolvedIndirectCall::Local(fi as usize))
    }

    /// Foreign identities allocate owned call state and may need cross-context
    /// type matching. Keep all of that off the self-owned table hot path so the
    /// compact resolver above can inline back into folded `exec_call_indirect`.
    #[cold]
    #[inline(never)]
    fn resolve_external_indirect_call(
        &self,
        callee: RefValue,
        frame: &[u64],
        arg_base: usize,
        expected_type: u32,
    ) -> Result<ResolvedIndirectCall, WasmError> {
        let entry = self.link_registry.functions.entry_for_handle(callee);
        if entry.is_some() {
            let expected = RefType::non_nullable_concrete(expected_type);
            if !ref_type_matches(callee, &expected.heap_type, RefTypeOwner::Interp(self))
                .unwrap_or(false)
            {
                return Err(WasmError::trap("indirect call type mismatch"));
            }
        }
        let (param_types, result_types) = self
            .module
            .types()
            .get_function_type(expected_type)
            .map(|ft| {
                (
                    ft.params().iter().copied().collect::<Vec<_>>(),
                    ft.results().iter().copied().collect::<Vec<_>>(),
                )
            })
            .ok_or(WasmError::trap("indirect call type mismatch"))?;
        let call =
            self.prepare_funcref_call(callee, param_types, result_types, frame, arg_base, entry);
        Ok(ResolvedIndirectCall::External(Box::new(call)))
    }

    /// Execute the common semantic slow path for `call_indirect` and
    /// `return_call_indirect` after their operand representation has been
    /// decoded. Native indirect-call guard exits can enter here directly;
    /// static slow cells use the same helper through `exec_ins`.
    fn exec_call_indirect(
        &mut self,
        frame: &mut [u64],
        func: &PredecodedFunction,
        table_element: u64,
        arg_base: usize,
        table_and_type: u64,
        tail: bool,
    ) -> Result<Effect, WasmError> {
        match self.resolve_call_indirect(
            frame,
            table_element,
            arg_base,
            (table_and_type >> 32) as usize,
            table_and_type as u32,
        )? {
            ResolvedIndirectCall::Local(callee) => Ok(if tail {
                Effect::TailCall { callee, arg_base }
            } else {
                Effect::Call { callee, arg_base }
            }),
            ResolvedIndirectCall::External(call) => {
                let tail_target = if tail {
                    Some(func.slow_tail_return.ok_or(WasmError::invalid(
                        "interp: tail call has no return landing",
                    ))? as usize)
                } else {
                    None
                };
                Ok(Effect::ExternalCall {
                    call,
                    arg_base,
                    tail_target,
                })
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
                frame[dst] = $conv(self.mem_load(addr, mi, offset, $size)?);
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
                self.mem_store(addr, mi, offset, $size, val)?;
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
                frame[ins.c as usize] = self.memory_size(ins.b as usize);
            }
            Op::MemoryGrow => {
                frame[ins.c as usize] = self.memory_grow(ins.b as usize, opa!(ins))?;
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
            Op::MemoryFillCopy => {
                let fill_base = ins.a as usize;
                let copy_base = ins.b as usize;
                let (fill_dst, value, fill_len) = (
                    frame[fill_base] as u32 as u64,
                    frame[fill_base + 1],
                    frame[fill_base + 2] as u32 as u64,
                );
                let (copy_dst, copy_src, copy_len) = (
                    frame[copy_base] as u32 as u64,
                    frame[copy_base + 1] as u32 as u64,
                    frame[copy_base + 2] as u32 as u64,
                );
                let mem = self
                    .memories
                    .get_mut(ins.c as usize)
                    .ok_or(WasmError::trap("out of bounds memory access"))?;
                let memory_len = mem.len() as u64;

                // Preserve the proposal's sequential trap effects: a valid
                // fill is committed before copy bounds are checked.
                let fill_end = fill_dst + fill_len;
                if fill_end > memory_len {
                    return Err(WasmError::trap("out of bounds memory access"));
                }
                mem.bytes_mut()[fill_dst as usize..fill_end as usize].fill(value as u8);

                let copy_end = copy_dst + copy_len;
                let source_end = copy_src + copy_len;
                if copy_end > memory_len || source_end > memory_len {
                    return Err(WasmError::trap("out of bounds memory access"));
                }

                // When the copy source was made uniform by the fill, only
                // destination bytes outside the already-filled range need
                // stores. Otherwise execute the ordinary memmove semantics.
                if copy_src >= fill_dst && source_end <= fill_end {
                    if copy_dst < fill_dst {
                        let left_end = copy_end.min(fill_dst);
                        mem.bytes_mut()[copy_dst as usize..left_end as usize].fill(value as u8);
                    }
                    if copy_end > fill_end {
                        let right_start = copy_dst.max(fill_end);
                        mem.bytes_mut()[right_start as usize..copy_end as usize].fill(value as u8);
                    }
                } else {
                    mem.bytes_mut()
                        .copy_within(copy_src as usize..source_end as usize, copy_dst as usize);
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
                frame[ins.c as usize] = self.global_get_for_frame(ins.a as usize);
            }
            Op::GlobalSet => {
                self.global_set_from_frame(ins.c as usize, opa!(ins));
            }

            // ---- ref/table ----
            Op::RefIsNull => {
                let v = opa!(ins);
                frame[ins.c as usize] = machine_raw_to_ref(v, SLOT_GP_UNIT_BYTES).is_null() as u64;
            }
            Op::Throw => {
                let pending =
                    self.alloc_exception_from_frame(ins.a as usize, frame, ins.b as usize)?;
                return Ok(Effect::Throw(pending));
            }
            Op::ThrowRef => {
                let pending =
                    self.exception_from_ref(machine_raw_to_ref(opa!(ins), SLOT_GP_UNIT_BYTES))?;
                return Ok(Effect::Throw(pending));
            }
            Op::RefEq => {
                frame[ins.c as usize] = (opa!(ins) == opb!(ins)) as u64;
            }
            Op::RefAsNonNull => {
                let v = opa!(ins);
                if machine_raw_to_ref(v, SLOT_GP_UNIT_BYTES).is_null() {
                    return Err(WasmError::trap("null function reference"));
                }
                frame[ins.c as usize] = v;
            }
            Op::RefTest => {
                let handle = machine_raw_to_ref(opa!(ins), SLOT_GP_UNIT_BYTES);
                let target = RefType::decode_from_u64(ins.b);
                let is_match = if handle.is_null() {
                    target.nullable
                } else {
                    ref_type_matches(handle, &target.heap_type, RefTypeOwner::Interp(self))?
                };
                frame[ins.c as usize] = u64::from(is_match);
            }
            Op::RefCast => {
                let raw = opa!(ins);
                let handle = machine_raw_to_ref(raw, SLOT_GP_UNIT_BYTES);
                let target = RefType::decode_from_u64(ins.b);
                let is_match = if handle.is_null() {
                    target.nullable
                } else {
                    ref_type_matches(handle, &target.heap_type, RefTypeOwner::Interp(self))?
                };
                if !is_match {
                    return Err(WasmError::trap("cast failure"));
                }
                frame[ins.c as usize] = raw;
            }
            Op::CallRef | Op::ReturnCallRef => {
                let r = machine_raw_to_ref(opa!(ins), SLOT_GP_UNIT_BYTES);
                if r.is_null() {
                    return Err(WasmError::trap("null function reference"));
                }
                if r.is_special() {
                    return Err(WasmError::trap("indirect call type mismatch"));
                }
                let local = self.localize_ref(r);
                if local.encoded() >= self.module.functions().len() {
                    let entry = self.link_registry.functions.entry_for_handle(r);
                    if entry.is_some() {
                        let expected = RefType::non_nullable_concrete(ins.c as u32);
                        if !ref_type_matches(r, &expected.heap_type, RefTypeOwner::Interp(self))
                            .unwrap_or(false)
                        {
                            return Err(WasmError::trap("indirect call type mismatch"));
                        }
                    }
                    let (param_types, result_types) = self
                        .module
                        .types()
                        .get_function_type(ins.c as u32)
                        .map(|ft| {
                            (
                                ft.params().iter().copied().collect::<Vec<_>>(),
                                ft.results().iter().copied().collect::<Vec<_>>(),
                            )
                        })
                        .ok_or(WasmError::trap("indirect call type mismatch"))?;
                    let base = ins.b as usize;
                    let call =
                        self.prepare_funcref_call(r, param_types, result_types, frame, base, entry);
                    let tail_target = if ins.op == Op::ReturnCallRef {
                        Some(func.slow_tail_return.ok_or(WasmError::invalid(
                            "interp: tail call has no return landing",
                        ))? as usize)
                    } else {
                        None
                    };
                    return Ok(Effect::ExternalCall {
                        call: Box::new(call),
                        arg_base: base,
                        tail_target,
                    });
                }
                let callee = local.encoded();
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
                frame[ins.c as usize] = self.table_slot_for_frame(ins.b as usize, v);
            }
            Op::TableSet => {
                let i = opa!(ins);
                let table_idx = ins.c as usize;
                let i = usize::try_from(i)
                    .map_err(|_| WasmError::trap("out of bounds table access"))?;
                let len = self
                    .tables
                    .get(table_idx)
                    .ok_or(WasmError::trap("out of bounds table access"))?
                    .entries
                    .len();
                if i >= len {
                    return Err(WasmError::trap("out of bounds table access"));
                }
                let v = self.table_slot_for_storage(table_idx, opb!(ins));
                self.tables[table_idx].entries.set(i, v)?;
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
                    .get(tidx)
                    .ok_or(WasmError::trap("out of bounds table access"))?;
                let cur = t.entries.len() as u64;
                if cur + delta > t.max || cur + delta > u32::MAX as u64 {
                    frame[dst] = u32::MAX as u64;
                } else if delta > 0 {
                    let init = self.table_slot_for_storage(tidx, init);
                    frame[dst] = match self.tables[tidx]
                        .entries
                        .try_resize((cur + delta) as usize, init)
                    {
                        Ok(()) => cur,
                        Err(()) => u32::MAX as u64,
                    };
                } else {
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
                let table_idx = ins.b as usize;
                let len = self
                    .tables
                    .get(table_idx)
                    .ok_or(WasmError::trap("out of bounds table access"))?
                    .entries
                    .len() as u64;
                if i + n > len {
                    return Err(WasmError::trap("out of bounds table access"));
                }
                if n > 0 {
                    let val = self.table_slot_for_storage(table_idx, val);
                    self.tables[table_idx]
                        .entries
                        .fill(i as usize, n as usize, val);
                }
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
                        let v = self.table_slot_for_storage(dt, v);
                        self.tables[dt].entries.set(d as usize + k, v)?;
                    }
                } else {
                    for k in 0..n as usize {
                        let v = self.tables[st]
                            .entries
                            .get(s0 as usize + k)
                            .ok_or_else(oob)?;
                        let v = self.table_slot_for_storage(dt, v);
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
                    let v = self.table_slot_for_storage(tidx, v);
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
            Op::I32_SubBrIf => {
                let value = (opa!(ins) as u32).wrapping_sub(opb!(ins) as u32);
                frame[ins.a as usize] = value as u64;
                if value != 0 {
                    return Ok(Effect::Jump(ins.c as usize));
                }
            }
            Op::I64_SubBrIf => {
                let value = opa!(ins).wrapping_sub(opb!(ins));
                frame[ins.a as usize] = value;
                if value != 0 {
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
                return self.exec_call_indirect(
                    frame,
                    func,
                    opa!(ins),
                    ins.b as usize,
                    ins.c,
                    ins.op == Op::ReturnCallIndirect,
                );
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
pub(super) fn trunc_checked(x: f64, lo: f64, hi_excl: f64) -> Result<f64, WasmError> {
    if x.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    let t = fmath::trunc64(x);
    if t < lo || t >= hi_excl {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(t)
}

pub(super) fn wasm_min_f32(a: f32, b: f32) -> f32 {
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

pub(super) fn wasm_max_f32(a: f32, b: f32) -> f32 {
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

pub(super) fn wasm_min_f64(a: f64, b: f64) -> f64 {
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

pub(super) fn wasm_max_f64(a: f64, b: f64) -> f64 {
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
    use core::cell::Cell;
    use std::vec::Vec as StdVec;

    trait TestInterpInstance: Sized {
        fn new(
            engine: &Engine,
            module: Module,
            host: Option<HostDispatch>,
            imports: &[Import],
        ) -> Result<Self, WasmError>;

        fn boxed_host<F>(host: F) -> HostDispatch
        where
            F: FnMut(&str, &str, &mut [u8], &[u64], &mut [u64]) -> Result<(), WasmError> + 'static;

        fn set_host<F>(&mut self, host: F)
        where
            F: FnMut(&str, &str, &mut [u8], &[u64], &mut [u64]) -> Result<(), WasmError> + 'static;
    }

    impl TestInterpInstance for InterpInstance {
        fn new(
            engine: &Engine,
            module: Module,
            host: Option<HostDispatch>,
            imports: &[Import],
        ) -> Result<Self, WasmError> {
            let registry = LinkRegistry::new();
            let (_, instance_backref) = registry.reserve_instance();
            #[cfg(sf_module_validator)]
            let baseline = super::super::ValidatedBaselinePlan::validate(&module)?;
            let inst = Self::build(
                engine,
                module,
                #[cfg(sf_module_validator)]
                baseline,
                host,
                imports,
                None,
                registry.arenas(),
                instance_backref,
            )?;
            Self::initialize(inst).map_err(|(_, error)| error)
        }

        fn boxed_host<F>(host: F) -> HostDispatch
        where
            F: FnMut(&str, &str, &mut [u8], &[u64], &mut [u64]) -> Result<(), WasmError> + 'static,
        {
            let mut host = host;
            let wrapped = move |module: &str,
                                name: &str,
                                caller: &mut Caller<'_>,
                                args: &[u64],
                                results: &mut [u64]| {
                let mut empty_memory = [];
                let memory = caller.memory_mut().unwrap_or(&mut empty_memory);
                host(module, name, memory, args, results)
            };
            Self::boxed_caller_host(wrapped)
        }

        fn set_host<F>(&mut self, host: F)
        where
            F: FnMut(&str, &str, &mut [u8], &[u64], &mut [u64]) -> Result<(), WasmError> + 'static,
        {
            self.host = Some(Self::boxed_host(host));
        }
    }

    fn instantiate(src: &str) -> (StdVec<u8>, ()) {
        let bin: StdVec<u8> = wat::parse_str(src).expect("wat");
        (bin, ())
    }

    fn linked_cell_for_op(
        instance: &InterpInstance,
        func_index: usize,
        op: Op,
    ) -> (&DCell, Instr, usize) {
        let function = instance.functions()[func_index]
            .as_ref()
            .expect("defined function");
        let pc = function
            .code
            .iter()
            .position(|ins| ins.op == op)
            .expect("requested linked op");
        let native = instance.native.as_ref().expect("native state");
        let linked = native.linked[func_index].as_ref().expect("linked function");
        let cells = native.plan.cells(linked);
        let cell = &cells[pc];
        let cell_index = native
            .plan
            .cell_index(cell as *const DCell as u64)
            .expect("cell belongs to plan");
        (cell, function.code[pc], cell_index)
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

    fn run2(src: &str, export: &str, args: &[u64]) -> Result<[u64; 2], WasmError> {
        let (bin, _) = instantiate(src);
        let module = Module::new("t", &bin).expect("module");
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            None,
            &[],
        )?;
        let idx = inst.find_export(export).expect("export");
        let mut results = [0u64; 2];
        inst.invoke(idx, args, &mut results)?;
        Ok(results)
    }

    #[test]
    fn call_stacks_materialize_on_first_guest_call_and_are_reused() {
        let (bin, _) = instantiate(
            r#"(module
                (func (export "run") (param $write i32) (result i32)
                    (local $value i32)
                    local.get $write
                    if
                        i32.const 41
                        local.set $value
                    end
                    local.get $value))"#,
        );
        let module = Module::new("lazy-call-stacks", &bin).expect("module");
        let engine = Engine::new(Config::new().wasm_stack_bytes(4096)).expect("engine");
        let mut inst = InterpInstance::new(&engine, module, None, &[]).expect("instantiate");

        assert_eq!(inst.stack.as_ref().expect("owner stack").len(), 0);
        assert_eq!(
            inst.ret_stack.as_ref().expect("owner return stack").len(),
            0
        );

        let run = inst.find_export("run").expect("run");
        let mut result = [u64::MAX];
        inst.invoke(run, &[1], &mut result).expect("first invoke");
        assert_eq!(result, [41]);
        let stack = inst.stack.as_ref().expect("returned owner stack");
        let ret_stack = inst
            .ret_stack
            .as_ref()
            .expect("returned owner return stack");
        assert_eq!(stack.len(), configured_stack_slots(engine.config()));
        assert_eq!(
            ret_stack.len(),
            configured_ret_records(stack.len()) * (RET_RECORD / 8)
        );
        let stack_ptr = stack.as_ptr();
        let ret_stack_ptr = ret_stack.as_ptr();

        // The second call must see a freshly zeroed local without replacing
        // either reusable allocation.
        inst.invoke(run, &[0], &mut result).expect("second invoke");
        assert_eq!(result, [0]);
        assert_eq!(
            inst.stack.as_ref().expect("owner stack").as_ptr(),
            stack_ptr
        );
        assert_eq!(
            inst.ret_stack
                .as_ref()
                .expect("owner return stack")
                .as_ptr(),
            ret_stack_ptr
        );
    }

    #[test]
    fn imported_start_leaves_owner_stacks_unmaterialized() {
        let (bin, _) = instantiate(
            r#"(module
                (import "host" "start" (func $start))
                (start $start))"#,
        );
        let module = Module::new("imported-start-call-stacks", &bin).expect("module");
        let func_type = module.functions()[0].func_type().clone();
        let import = Import::func_typed(
            "host",
            "start",
            |_caller, _args, _results| Ok(()),
            func_type,
        );
        let calls = Rc::new(Cell::new(0usize));
        let host_calls = Rc::clone(&calls);
        let host = InterpInstance::boxed_host(move |_module, _name, _memory, args, results| {
            assert!(args.is_empty());
            assert!(results.is_empty());
            host_calls.set(host_calls.get() + 1);
            Ok(())
        });

        let inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            Some(host),
            &[import],
        )
        .expect("instantiate");

        assert_eq!(calls.get(), 1, "the imported start must still execute");
        assert!(
            inst.stack.as_ref().is_some_and(Vec::is_empty),
            "an imported start has no guest frame and must not allocate the owner stack"
        );
        assert!(
            inst.ret_stack.as_ref().is_some_and(Vec::is_empty),
            "an imported start has no guest frame and must not allocate the owner return stack"
        );
    }

    #[test]
    fn guest_trap_returns_owner_stacks_for_the_next_call() {
        let (bin, _) = instantiate(
            r#"(module
                (func (export "trap")
                    unreachable)
                (func (export "ok") (result i32)
                    i32.const 29))"#,
        );
        let module = Module::new("trap-call-stacks", &bin).expect("module");
        let engine = Engine::new(Config::new().wasm_stack_bytes(4096)).expect("engine");
        let mut inst = InterpInstance::new(&engine, module, None, &[]).expect("instantiate");
        let trap = inst.find_export("trap").expect("trap");
        let ok = inst.find_export("ok").expect("ok");

        assert!(inst.stack.as_ref().is_some_and(Vec::is_empty));
        assert!(inst.ret_stack.as_ref().is_some_and(Vec::is_empty));
        assert!(matches!(
            inst.invoke(trap, &[], &mut []),
            Err(WasmError::Trap("unreachable"))
        ));

        let stack_ptr = inst
            .stack
            .as_ref()
            .filter(|stack| !stack.is_empty())
            .expect("trap returned the materialized owner stack")
            .as_ptr();
        let ret_stack_ptr = inst
            .ret_stack
            .as_ref()
            .filter(|stack| !stack.is_empty())
            .expect("trap returned the materialized owner return stack")
            .as_ptr();

        let mut result = [u64::MAX];
        inst.invoke(ok, &[], &mut result)
            .expect("instance remains callable after a trap");
        assert_eq!(result, [29]);
        assert_eq!(
            inst.stack.as_ref().expect("owner stack").as_ptr(),
            stack_ptr,
            "the call after a trap must reuse the returned owner stack"
        );
        assert_eq!(
            inst.ret_stack
                .as_ref()
                .expect("owner return stack")
                .as_ptr(),
            ret_stack_ptr,
            "the call after a trap must reuse the returned owner return stack"
        );
    }

    #[test]
    fn same_instance_guest_reentry_keeps_the_outer_stacks_owned() {
        const INNER_INDEX: usize = 1;
        const OUTER_MARKER: u64 = 1_229_801_703_532_086_340;
        const INNER_MARKER: u64 = 6_148_933_456_521_300_104;

        fn invoke_lease(
            lease: &InterpInstanceLease,
            func_index: usize,
            args: &[u64],
            results: &mut [u64],
        ) -> Result<(), WasmError> {
            let mut access = InterpInstanceAccess::checked_out(lease.checkout_for_invocation()?);
            InterpInstance::invoke_access(&mut access, func_index, args, results)
        }

        let (bin, _) = instantiate(
            r#"(module
                (import "host" "reenter" (func $reenter (param i64) (result i64)))
                (func (export "inner") (param $seed i64) (result i64)
                    (local $scratch i64)
                    i64.const 6148933456521300104
                    local.set $scratch
                    local.get $seed
                    local.get $scratch
                    i64.xor)
                (func (export "outer") (param $seed i64) (result i64)
                    (local $kept i64)
                    i64.const 1229801703532086340
                    local.set $kept
                    local.get $seed
                    call $reenter
                    drop
                    local.get $kept))"#,
        );
        let module = Module::new("same-instance-stack-reentry", &bin).expect("module");
        let func_type = module.functions()[0].func_type().clone();
        let import = Import::func_typed(
            "host",
            "reenter",
            |_caller, _args, _results| Ok(()),
            func_type,
        );
        let registry = LinkRegistry::new();
        let (id, instance_backref) = registry.reserve_instance();
        let lease_backref = instance_backref.clone();
        let callback_backref = instance_backref.clone();
        let reentered = Rc::new(Cell::new(false));
        let callback_reentered = Rc::clone(&reentered);
        let host = InterpInstance::boxed_host(move |_module, _name, _memory, args, results| {
            let token = callback_backref
                .checkout(id)
                .ok_or_else(|| WasmError::internal("same-instance checkout failed"))?;
            let mut inner_access = InterpInstanceAccess::checked_out(token);
            inner_access.with_instance(|instance| {
                assert!(
                    instance.stack.is_none(),
                    "the active outer guest call must own its operand stack"
                );
                assert!(
                    instance.ret_stack.is_none(),
                    "the active outer guest call must own its return stack"
                );
            })?;

            let mut inner_result = [u64::MAX];
            InterpInstance::invoke_access(&mut inner_access, INNER_INDEX, args, &mut inner_result)?;
            assert_eq!(inner_result, [args[0] ^ INNER_MARKER]);
            inner_access.with_instance(|instance| {
                assert!(
                    instance.stack.is_none(),
                    "a reentrant guest call must drop, not return, its temporary operand stack"
                );
                assert!(
                    instance.ret_stack.is_none(),
                    "a reentrant guest call must drop, not return, its temporary return stack"
                );
            })?;
            results[0] = inner_result[0];
            callback_reentered.set(true);
            Ok(())
        });
        let engine = Engine::new(Config::new().wasm_stack_bytes(4096)).expect("engine");
        #[cfg(sf_module_validator)]
        let baseline =
            super::super::ValidatedBaselinePlan::validate(&module).expect("validated baseline");
        let built = InterpInstance::build(
            &engine,
            module,
            #[cfg(sf_module_validator)]
            baseline,
            Some(host),
            &[import],
            None,
            registry.arenas(),
            instance_backref,
        )
        .expect("build instance");
        let built = InterpInstance::initialize(built)
            .unwrap_or_else(|(_, error)| panic!("initialize instance: {error:?}"));
        let inst =
            InterpInstance::lease_in(&registry, id, lease_backref, built).expect("occupy instance");
        let (inner, outer) = inst.with_instance(|instance| {
            (
                instance.find_export("inner").expect("inner"),
                instance.find_export("outer").expect("outer"),
            )
        });
        assert_eq!(inner, INNER_INDEX);

        // Materialize the reusable owner pair before the reentrant call so
        // both its identity and the outer frame's live value are testable.
        let mut result = [u64::MAX];
        invoke_lease(&inst, inner, &[7], &mut result).expect("warm owner stacks");
        assert_eq!(result, [7 ^ INNER_MARKER]);
        let (stack_ptr, ret_stack_ptr) = inst.with_instance(|instance| {
            (
                instance.stack.as_ref().expect("owner stack").as_ptr(),
                instance
                    .ret_stack
                    .as_ref()
                    .expect("owner return stack")
                    .as_ptr(),
            )
        });

        invoke_lease(&inst, outer, &[11], &mut result).expect("same-instance guest reentry");
        assert!(
            reentered.get(),
            "the host callback must enter the inner guest"
        );
        assert_eq!(
            result,
            [OUTER_MARKER],
            "the inner root writes local slot 1 too, but must not overwrite the outer local"
        );
        inst.with_instance(|instance| {
            assert_eq!(
                instance
                    .stack
                    .as_ref()
                    .expect("returned owner stack")
                    .as_ptr(),
                stack_ptr
            );
            assert_eq!(
                instance
                    .ret_stack
                    .as_ref()
                    .expect("returned owner return stack")
                    .as_ptr(),
                ret_stack_ptr
            );
        });
    }

    #[test]
    fn start_function_materializes_and_returns_the_owner_stacks() {
        let (bin, _) = instantiate(
            r#"(module
                (global $value (mut i32) (i32.const 0))
                (func $start
                    i32.const 73
                    global.set $value)
                (start $start)
                (func (export "get") (result i32)
                    global.get $value))"#,
        );
        let module = Module::new("start-call-stacks", &bin).expect("module");
        let engine = Engine::new(Config::new().wasm_stack_bytes(4096)).expect("engine");
        let mut inst = InterpInstance::new(&engine, module, None, &[]).expect("instantiate");

        assert_eq!(
            inst.stack.as_ref().expect("owner stack").len(),
            configured_stack_slots(engine.config())
        );
        assert!(inst
            .ret_stack
            .as_ref()
            .is_some_and(|return_stack| !return_stack.is_empty()));
        let get = inst.find_export("get").expect("get");
        let mut result = [u64::MAX];
        inst.invoke(get, &[], &mut result).expect("invoke get");
        assert_eq!(result, [73]);
    }

    #[test]
    fn module_link_plan_owns_contiguous_cells_and_branch_tables() {
        let src = r#"(module
            (type $unary (func (param i32) (result i32)))
            (table 2 funcref)
            (elem (i32.const 0) $inc $twice)
            (func $inc (type $unary) (param $x i32) (result i32)
                local.get $x i32.const 1 i32.add)
            (func $twice (type $unary) (param $x i32) (result i32)
                local.get $x i32.const 2 i32.mul)
            (func (export "direct") (param $x i32) (result i32)
                local.get $x call $inc)
            (func (export "indirect") (param $x i32) (param $which i32) (result i32)
                local.get $x local.get $which call_indirect (type $unary))
            (func (export "switch") (param $which i32) (result i32)
                (block $outer (result i32)
                    (block $inner (result i32)
                        i32.const 41
                        local.get $which
                        br_table $inner $outer)
                    i32.const 1
                    i32.add)))"#;
        let (bin, _) = instantiate(src);
        let module = Module::new("module-link-plan", &bin).expect("module");
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            None,
            &[],
        )
        .expect("instantiate");

        {
            let native = inst.native.as_ref().expect("native state");
            assert!(native.plan.call_fixups_are_drained());

            let mut next_cell_base = None;
            let mut range_index = 0usize;
            let mut br_table_cell = None;
            for (func_index, linked) in native.linked.iter().enumerate() {
                let Some(linked) = linked else { continue };
                let base = linked.cell_base();
                if let Some(expected) = next_cell_base {
                    assert_eq!(base, expected, "function cell slices must be contiguous");
                }
                let cells = native.plan.cells(linked);
                assert_eq!(
                    base,
                    cells.as_ptr() as u64,
                    "cached function base must address its arena slice"
                );
                let predecoded = inst.functions()[func_index].as_ref().expect("defined body");
                assert_eq!(cells.len(), predecoded.code.len() + 1);
                next_cell_base = Some(base + (cells.len() * core::mem::size_of::<DCell>()) as u64);

                assert_eq!(
                    native.ranges[range_index],
                    (
                        base,
                        base + (predecoded.code.len() * core::mem::size_of::<DCell>()) as u64,
                        func_index as u32,
                    )
                );
                range_index += 1;

                if br_table_cell.is_none() {
                    br_table_cell = predecoded
                        .code
                        .iter()
                        .position(|ins| ins.op == Op::BrTable)
                        .map(|pc| &cells[pc]);
                }
            }
            assert_eq!(range_index, native.ranges.len());

            let branch_bytes = native.plan.branch_bytes();
            let br_table_cell = br_table_cell.expect("br_table cell");
            assert!(branch_bytes.contains(&br_table_cell.b));
            assert!(br_table_cell.b + 4 <= branch_bytes.end);
        }

        for (export, args, expected) in [
            ("direct", &[40][..], 41),
            ("indirect", &[20, 1][..], 40),
            ("switch", &[0][..], 42),
            ("switch", &[9][..], 41),
        ] {
            let func = inst.find_export(export).expect("export");
            let mut result = [0u64];
            inst.invoke(func, args, &mut result).expect("invoke");
            assert_eq!(result[0], expected, "{export}");
        }
    }

    const ITERATIVE_FIB_WAT: &str = r#"(module
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
            (local.get $a)))"#;

    #[test]
    fn iterative_fibonacci_uses_the_generic_i64_sub_branch() {
        let (bin, _) = instantiate(ITERATIVE_FIB_WAT);
        let module = Module::new("fibonacci-iter", &bin).expect("module");
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            None,
            &[],
        )
        .expect("instantiate");
        let run = inst.find_export("run").expect("run");

        for (n, expected) in [
            (0, 0),
            (1, 1),
            (2, 1),
            (10, 55),
            (93, 12_200_160_415_121_876_738),
        ] {
            let mut result = [u64::MAX];
            inst.invoke(run, &[n], &mut result).expect("invoke");
            assert_eq!(result[0], expected, "fib({n})");
        }
        assert!(inst
            .slow_exit_stats()
            .iter()
            .all(|&(op, exits)| { !matches!(op, Op::MovPair | Op::I64_SubBrIf) || exits == 0 }));
    }

    #[test]
    fn iterative_fibonacci_dispatch_count_tracks_three_hot_cells() {
        let (bin, _) = instantiate(ITERATIVE_FIB_WAT);
        let module = Module::new("fibonacci-iter-count", &bin).expect("module");
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            None,
            &[],
        )
        .expect("instantiate");
        let run = inst.find_export("run").expect("run");
        let mut result = [u64::MAX];
        inst.invoke(run, &[100], &mut result).expect("invoke");

        assert_eq!(result[0], 3_736_710_778_780_434_371);
        if inst.dispatch_counting_enabled() {
            assert_eq!(
                inst.dispatch_count(),
                306,
                "three loop cells per iteration plus six setup/exit cells"
            );
        }
    }

    #[test]
    fn ordered_local_copy_pair_preserves_old_value_and_pinned_write_through() {
        let src = r#"(module
            (func (export "run") (param $x i32) (result i32 i32)
                local.get $x
                local.get $x
                i32.const 1
                i32.add
                local.set $x
                local.get $x))"#;
        let (bin, _) = instantiate(src);
        let module = Module::new("ordered-copy-pair", &bin).expect("module");
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            None,
            &[],
        )
        .expect("instantiate");
        let run = inst.find_export("run").expect("run");
        let mut result = [u64::MAX; 2];
        inst.invoke(run, &[41], &mut result).expect("invoke");

        assert_eq!(result, [41, 42]);
        assert!(inst
            .slow_exit_stats()
            .iter()
            .all(|&(op, exits)| op != Op::MovPair || exits == 0));
    }

    #[test]
    fn ordered_copy_pair_forwards_only_its_second_destination() {
        let src = r#"(module
            (func (export "first") (param $a i64) (param $b i64) (result i64)
                  (local $x i64) (local $y i64)
                local.get $a
                local.set $x
                local.get $b
                local.set $y
                local.get $x
                i64.const 1
                i64.add)
            (func (export "second") (param $a i64) (param $b i64) (result i64)
                  (local $x i64) (local $y i64)
                local.get $a
                local.set $x
                local.get $b
                local.set $y
                local.get $y
                i64.const 1
                i64.add))"#;
        let (bin, _) = instantiate(src);
        let module = Module::new("ordered-copy-pair-acc", &bin).expect("module");
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            None,
            &[],
        )
        .expect("instantiate");
        let first = inst.find_export("first").expect("first");
        let second = inst.find_export("second").expect("second");
        let mut result = [u64::MAX];

        inst.invoke(first, &[7, 99], &mut result).expect("first");
        assert_eq!(result[0], 8, "destination 1 must reload from its slot");
        inst.invoke(second, &[7, 99], &mut result).expect("second");
        assert_eq!(
            result[0], 100,
            "destination 2 must read the MovPair accumulator result"
        );
        assert!(inst
            .slow_exit_stats()
            .iter()
            .all(|&(op, exits)| op != Op::MovPair || exits == 0));
    }

    #[test]
    fn ordered_copy_pair_forwards_an_unpinned_full_width_second_value() {
        let src = r#"(module
            (func (export "run")
                  (param $a i64) (param $b i64)
                  (param $hot0 i64) (param $hot1 i64)
                  (result i64) (local $x i64) (local $y i64)
                ;; Make $hot0/$hot1 the two unambiguous pinned locals so
                ;; neither MovPair destination can outrank its ACC hint.
                local.get $hot0
                i64.const 1
                i64.add
                local.set $hot0
                local.get $hot0
                i64.const 1
                i64.add
                local.set $hot0
                local.get $hot1
                i64.const 1
                i64.add
                local.set $hot1
                local.get $hot1
                i64.const 1
                i64.add
                local.set $hot1
                local.get $a
                local.set $x
                local.get $b
                local.set $y
                local.get $y
                i64.const 1
                i64.add))"#;
        let (bin, _) = instantiate(src);
        let module = Module::new("ordered-copy-pair-unpinned-acc", &bin).expect("module");
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            None,
            &[],
        )
        .expect("instantiate");
        let run = inst.find_export("run").expect("run");
        let pair = inst.functions()[run]
            .as_ref()
            .expect("predecoded body")
            .code
            .iter()
            .find(|ins| ins.op == Op::MovPair)
            .expect("local copy pair");
        assert_eq!((pair.a, pair.b, pair.c), (0, 1, 4u64 << 32 | 5));

        let linked = inst.native.as_ref().expect("native state").linked[run]
            .as_ref()
            .expect("linked body");
        let mut pinned = [linked.l0_off, linked.l1_off];
        pinned.sort_unstable();
        assert_eq!(pinned, [16, 24], "only $hot0/$hot1 should be pinned");

        let full_width = 0x1234_5678_0000_0063;
        let mut result = [u64::MAX];
        inst.invoke(run, &[7, full_width, 10, 20], &mut result)
            .expect("invoke");
        assert_eq!(result[0], full_width + 1);
        assert!(inst
            .slow_exit_stats()
            .iter()
            .all(|&(op, exits)| op != Op::MovPair || exits == 0));
    }

    #[test]
    fn ordered_copy_pair_writes_l1_then_l0_without_slow_exit() {
        let src = r#"(module
            (func (export "run")
                  (param $l0 i64) (param $l1 i64)
                  (param $src0 i64) (param $src1 i64)
                  (result i64 i64)
                local.get $src0
                local.set $l1
                local.get $src1
                local.set $l0
                local.get $l0
                local.get $l1))"#;
        let (bin, _) = instantiate(src);
        let module = Module::new("ordered-copy-pair-l1-l0", &bin).expect("module");
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            None,
            &[],
        )
        .expect("instantiate");
        let run = inst.find_export("run").expect("run");
        let code = &inst.functions()[run]
            .as_ref()
            .expect("predecoded body")
            .code;
        let pair = code
            .iter()
            .find(|ins| ins.op == Op::MovPair && ins.c == 1u64 << 32)
            .expect("ordered local1-then-local0 copy pair");
        assert_eq!((pair.a, pair.b), (2, 3));
        let mut result = [u64::MAX; 2];
        inst.invoke(run, &[41, 99, 7, 8], &mut result)
            .expect("invoke");

        assert_eq!(result, [8, 7]);
        assert!(inst
            .slow_exit_stats()
            .iter()
            .all(|&(op, exits)| op != Op::MovPair || exits == 0));
    }

    #[test]
    fn i64_sub_branch_tests_the_full_result_and_wraps() {
        let src = r#"(module
            (func (export "run") (param $n i64) (result i32)
                (block $taken
                    local.get $n
                    i64.const 1
                    i64.sub
                    local.set $n
                    local.get $n
                    i64.const 0
                    i64.ne
                    br_if $taken
                    i32.const 0
                    return)
                i32.const 1))"#;

        assert_eq!(run1(src, "run", &[1]).unwrap(), 0);
        assert_eq!(run1(src, "run", &[0x1_0000_0001]).unwrap(), 1);
        assert_eq!(run1(src, "run", &[0]).unwrap(), 1);
    }

    #[test]
    fn i64_sub_branch_accepts_an_accumulator_rhs_and_aliasing() {
        let accumulator_rhs = r#"(module
            (func (export "run") (param $n i64) (param $step i64) (result i64)
                (block $done
                    local.get $n
                    local.get $step
                    i64.const 0
                    i64.add
                    i64.sub
                    local.set $n
                    local.get $n
                    i64.const 0
                    i64.ne
                    br_if $done)
                local.get $n))"#;
        assert_eq!(run1(accumulator_rhs, "run", &[10, 3]).unwrap(), 7);
        assert_eq!(run1(accumulator_rhs, "run", &[3, 3]).unwrap(), 0);

        let alias = r#"(module
            (func (export "run") (param $n i64) (result i64)
                (block $done
                    local.get $n
                    local.get $n
                    i64.sub
                    local.set $n
                    local.get $n
                    i64.const 0
                    i64.ne
                    br_if $done)
                local.get $n))"#;
        assert_eq!(run1(alias, "run", &[u64::MAX]).unwrap(), 0);
    }

    #[test]
    fn fused_i64_constant_add_branch_preserves_wrapping() {
        for (constant, input, expected) in [
            ("1", u64::MAX, 0),
            ("-1", 0, u64::MAX),
            ("-9223372036854775808", 0, 0x8000_0000_0000_0000),
            ("0", 0, 0),
        ] {
            let src = std::format!(
                r#"(module
                    (func (export "run") (param $n i64) (result i64)
                        (block $done
                            local.get $n
                            i64.const {constant}
                            i64.add
                            local.set $n
                            local.get $n
                            i64.const 0
                            i64.ne
                            br_if $done)
                        local.get $n))"#
            );
            assert_eq!(
                run1(&src, "run", &[input]).unwrap(),
                expected,
                "constant {constant}"
            );
        }
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
    fn typed_loop_parameter_counter_has_one_hot_dispatch_per_iteration() {
        // This is wasmi-benchmarks' `counter-param` input. Unlike
        // `counter-local`, the induction value is both a typed loop
        // parameter and a local, so a back edge has to preserve both views
        // without materializing extra hot-loop dispatches.
        let (bin, _) = instantiate(
            r#"(module
                (func (export "run") (param $n i32) (result i32)
                    (local.get $n)
                    (loop $continue (param i32) (result i32)
                        (i32.const 1)
                        (i32.sub)
                        (local.tee $n)
                        (local.get $n)
                        (br_if $continue))))"#,
        );
        let module = Module::new("counter-param", &bin).expect("module");
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            None,
            &[],
        )
        .expect("instantiate");
        let run = inst.find_export("run").expect("run");
        let mut result = [u64::MAX; 1];

        inst.invoke(run, &[100], &mut result).expect("invoke");
        assert_eq!(result[0], 0);

        if inst.dispatch_counting_enabled() {
            let dispatches = inst.dispatch_count();
            assert_eq!(
                dispatches, 102,
                "100 loop cells plus exit materialization and return"
            );
        }
    }

    #[test]
    fn typed_loop_parameter_add_counter_has_one_hot_dispatch_per_iteration() {
        let (bin, _) = instantiate(
            r#"(module
                (func (export "run") (param $n i32) (result i32)
                    (local.get $n)
                    (loop $continue (param i32) (result i32)
                        (i32.const 1)
                        (i32.add)
                        (local.tee $n)
                        (local.get $n)
                        (br_if $continue))))"#,
        );
        let module = Module::new("add-counter-param", &bin).expect("module");
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            None,
            &[],
        )
        .expect("instantiate");
        let run = inst.find_export("run").expect("run");
        let mut result = [u64::MAX; 1];

        // Start 100 increments below the i32 wrapping point.
        inst.invoke(run, &[(u32::MAX - 99) as u64], &mut result)
            .expect("invoke");
        assert_eq!(result[0], 0);

        if inst.dispatch_counting_enabled() {
            let dispatches = inst.dispatch_count();
            assert_eq!(
                dispatches, 102,
                "100 fused add-and-branch cells plus exit materialization and return"
            );
        }
    }

    #[test]
    fn loop_parameter_alias_does_not_mutate_its_source_local() {
        let src = r#"(module
            (func (export "run") (param $x i32) (param $n i32) (result i32)
                local.get $x
                (loop $continue (param i32) (result i32)
                    drop
                    i32.const 42
                    local.get $n
                    i32.const 1
                    i32.sub
                    local.tee $n
                    br_if $continue)
                drop
                local.get $x))"#;

        assert_eq!(
            run1(src, "run", &[7, 2]).unwrap(),
            7,
            "a loop parameter is a stack value, not an assignment to the local that sourced it"
        );
    }

    #[test]
    fn br_table_loop_parameter_alias_does_not_mutate_its_source_local() {
        let src = r#"(module
            (func (export "run") (param $x i32) (param $selector i32) (result i32)
                (block $exit (result i32)
                    local.get $x
                    (loop $continue (param i32) (result i32)
                        drop
                        i32.const 42
                        local.get $selector
                        i32.const 1
                        i32.sub
                        local.tee $selector
                        br_table $exit $continue $exit))
                drop
                local.get $x))"#;

        assert_eq!(
            run1(src, "run", &[7, 2]).unwrap(),
            7,
            "a br_table landing pad must not synthesize a write to the loop parameter's source local"
        );
    }

    #[test]
    fn fused_constant_add_branch_preserves_i32_wrapping() {
        let src = r#"(module
            (func (export "add_one") (param $n i32) (result i32)
                local.get $n
                (loop $continue (param i32) (result i32)
                    i32.const 1
                    i32.add
                    local.tee $n
                    local.get $n
                    br_if $continue))
            (func (export "add_neg_one") (param $n i32) (result i32)
                local.get $n
                (loop $continue (param i32) (result i32)
                    i32.const -1
                    i32.add
                    local.tee $n
                    local.get $n
                    br_if $continue))
            (func (export "add_min") (param $n i32) (result i32)
                local.get $n
                (loop $continue (param i32) (result i32)
                    i32.const -2147483648
                    i32.add
                    local.tee $n
                    local.get $n
                    br_if $continue)))"#;

        assert_eq!(
            run1(src, "add_one", &[u32::MAX as u64]).unwrap(),
            0,
            "adding one must wrap -1 to zero"
        );
        assert_eq!(
            run1(src, "add_neg_one", &[1]).unwrap(),
            0,
            "adding -1 must reuse subtraction by +1"
        );
        assert_eq!(
            run1(src, "add_min", &[0x8000_0000]).unwrap(),
            0,
            "the self-negating i32 minimum constant must remain exact"
        );
    }

    #[test]
    fn variable_rhs_add_branch_remains_correct_when_unfused() {
        let src = r#"(module
            (func (export "run") (param $n i32) (param $step i32) (result i32)
                local.get $n
                (loop $continue (param i32) (result i32)
                    local.get $step
                    i32.add
                    local.tee $n
                    local.get $n
                    br_if $continue)))"#;
        assert_eq!(run1(src, "run", &[u32::MAX as u64, 1]).unwrap(), 0);
    }

    #[test]
    fn fused_i64_eqz_branch_tests_all_64_bits() {
        let src = r#"(module
            (func (export "is_zero") (param $x i64) (result i32)
                (block $zero
                    local.get $x
                    i64.eqz
                    br_if $zero
                    i32.const 0
                    return)
                i32.const 1))"#;

        assert_eq!(run1(src, "is_zero", &[0]).unwrap(), 1);
        assert_eq!(run1(src, "is_zero", &[7]).unwrap(), 0);
        assert_eq!(
            run1(src, "is_zero", &[0x1_0000_0000]).unwrap(),
            0,
            "a zero low word must not hide nonzero high i64 bits"
        );
    }

    #[test]
    fn fused_i64_eqz_move_guard_preserves_taken_and_fallthrough_values() {
        let src = r#"(module
            (func (export "choose") (param $x i64) (result i32)
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
                    drop)))"#;

        assert_eq!(run1(src, "choose", &[0]).unwrap(), 42);
        assert_eq!(run1(src, "choose", &[9]).unwrap(), 11);
        assert_eq!(
            run1(src, "choose", &[0x1_0000_0000]).unwrap(),
            11,
            "the inverted guard must compare the complete i64 value"
        );
    }

    #[test]
    fn fused_dynamic_decrement_handles_an_accumulator_rhs_with_slot_destination() {
        // Make two unrelated locals hotter than `$n` in the static pin
        // ranking, forcing the fused branch's destination through an
        // ordinary frame slot. Its dynamic rhs stays in the accumulator;
        // native backends must keep that register distinct from their
        // slot-load scratch while performing the in-place subtraction.
        let src = r#"(module
            (global $iterations (mut i32) (i32.const 0))
            (func (export "run")
                (param $n i32) (param $step i32)
                (param $hot0 i32) (param $hot1 i32)
                (result i32)
                local.get $hot0 local.get $hot0 i32.add drop
                local.get $hot0 local.get $hot0 i32.add drop
                local.get $hot0 local.get $hot0 i32.add drop
                local.get $hot1 local.get $hot1 i32.add drop
                local.get $hot1 local.get $hot1 i32.add drop
                local.get $hot1 local.get $hot1 i32.add drop
                local.get $n
                (loop $continue (param i32) (result i32)
                    global.get $iterations
                    i32.const 1
                    i32.add
                    global.set $iterations
                    local.get $step
                    i32.const 0
                    i32.add
                    i32.sub
                    local.tee $n
                    local.get $n
                    br_if $continue)
                drop
                global.get $iterations))"#;
        assert_eq!(run1(src, "run", &[6, 2, 11, 13]).unwrap(), 3);
    }

    #[test]
    fn duplicate_loop_parameter_locals_remain_independent() {
        let src = r#"(module
            (func (export "duplicate") (param $x i32) (param $n i32)
                (result i32 i32) (local $a i32) (local $b i32)
                local.get $x
                local.get $x
                (loop $l (param i32 i32) (result i32 i32)
                    local.set $b
                    local.set $a
                    local.get $a
                    i32.const 1
                    i32.sub
                    local.set $a
                    local.get $b
                    i32.const 1
                    i32.add
                    local.set $b
                    local.get $a
                    local.get $b
                    local.get $n
                    i32.const 1
                    i32.sub
                    local.tee $n
                    br_if $l)))"#;
        assert_eq!(run2(src, "duplicate", &[10, 2]).unwrap(), [8, 12]);
    }

    #[test]
    fn loop_parameter_back_edges_do_not_swap_their_source_locals() {
        let src = r#"(module
            (func (export "swap") (param $a i32) (param $b i32)
                (param $n i32) (result i32 i32)
                local.get $a
                local.get $b
                (loop $l (param i32 i32) (result i32 i32)
                    drop
                    drop
                    local.get $b
                    local.get $a
                    local.get $n
                    i32.const 1
                    i32.sub
                    local.tee $n
                    br_if $l)))"#;
        assert_eq!(
            run2(src, "swap", &[10, 20, 2]).unwrap(),
            [20, 10],
            "each iteration supplies the unchanged locals `$b, $a`; loop parameters do not assign them"
        );
    }

    #[test]
    fn untaken_loop_branch_does_not_write_aliased_local() {
        let src = r#"(module
            (func (export "not_taken") (param $x i32) (result i32)
                local.get $x
                (loop $l (param i32) (result i32)
                    drop
                    i32.const 42
                    i32.const 0
                    br_if $l)
                drop
                local.get $x))"#;
        assert_eq!(run1(src, "not_taken", &[7]).unwrap(), 7);
    }

    #[test]
    fn br_table_loop_backedge_and_exit_preserve_values() {
        let src = r#"(module
            (func (export "table") (param $x i32) (param $n i32) (result i32)
                (block $exit (result i32)
                    local.get $x
                    (loop $l (param i32) (result i32)
                        i32.const 1
                        i32.add
                        local.get $n
                        i32.const 1
                        i32.sub
                        local.tee $n
                        br_table $l $exit))))"#;
        assert_eq!(run1(src, "table", &[10, 1]).unwrap(), 12);
    }

    #[test]
    fn nested_branch_depth_reaches_loop_parameter() {
        let src = r#"(module
            (func (export "nested") (param $x i32) (param $n i32) (result i32)
                local.get $x
                (loop $l (param i32) (result i32)
                    (block (param i32) (result i32)
                        i32.const 1
                        i32.add
                        local.get $n
                        i32.const 1
                        i32.sub
                        local.tee $n
                        br_if $l))))"#;
        assert_eq!(run1(src, "nested", &[10, 2]).unwrap(), 12);
    }

    #[test]
    fn local_overwrite_materializes_pending_loop_parameter() {
        let src = r#"(module
            (func (export "hazard") (param $x i32) (param $n i32) (result i32)
                local.get $x
                (loop $l (param i32) (result i32)
                    i32.const 999
                    local.set $x
                    local.get $n
                    i32.const 1
                    i32.sub
                    local.tee $n
                    br_if $l)))"#;
        assert_eq!(run1(src, "hazard", &[7, 3]).unwrap(), 7);
    }

    #[test]
    fn exception_edges_keep_canonical_loop_parameter_layout() {
        let src = r#"(module
            (tag $e (param i32))
            (func (export "catch_loop") (param $x i32) (result i32)
                (local $again i32)
                local.get $x
                (loop $l (param i32) (result i32)
                    local.set $x
                    local.get $again
                    (if (result i32)
                        (then local.get $x)
                        (else
                            i32.const 1
                            local.set $again
                            (try_table (result i32) (catch $e $l)
                                local.get $x
                                i32.const 1
                                i32.add
                                throw $e))))))"#;
        assert_eq!(run1(src, "catch_loop", &[10]).unwrap(), 11);
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
    fn frame_memory_primitives_match_folded_memory32() {
        let src = r#"(module
            (memory 1 2)
            (func (export "roundtrip") (param i32 i32) (result i32)
                local.get 0 local.get 1 i32.store offset=4
                local.get 0 i32.load offset=4)
            (func (export "load") (param i32) (result i32)
                local.get 0 i32.load)
            (func (export "size") (result i32) memory.size)
            (func (export "grow") (param i32) (result i32)
                local.get 0 memory.grow))"#;
        let bin: StdVec<u8> = wat::parse_str(src).expect("wat");
        let engine = Engine::with_defaults();
        let mut helper = InterpInstance::new(
            &engine,
            Module::new("memory32-helper", &bin).expect("module"),
            None,
            &[],
        )
        .expect("helper instance");
        let mut folded = InterpInstance::new(
            &engine,
            Module::new("memory32-folded", &bin).expect("module"),
            None,
            &[],
        )
        .expect("folded instance");

        let raw = 0x89ab_cdefu32 as u64;
        helper.mem_store(8, 0, 4, 4, raw).expect("helper store");
        let helper_raw = helper.mem_load(8, 0, 4, 4).expect("helper load");
        let mut folded_result = [0u64; 1];
        folded
            .invoke(
                folded.find_export("roundtrip").expect("roundtrip"),
                &[8, raw],
                &mut folded_result,
            )
            .expect("folded roundtrip");
        assert_eq!(helper_raw, folded_result[0]);

        let helper_oob = helper.mem_load(65_534, 0, 0, 4).expect_err("helper OOB");
        let folded_oob = folded
            .invoke(
                folded.find_export("load").expect("load"),
                &[65_534],
                &mut folded_result,
            )
            .expect_err("folded OOB");
        assert_eq!(helper_oob, folded_oob);

        folded
            .invoke(
                folded.find_export("size").expect("size"),
                &[],
                &mut folded_result,
            )
            .expect("folded size");
        assert_eq!(helper.memory_size(0), folded_result[0]);
        assert_eq!(helper.memory_grow(0, 1).expect("helper grow"), 1);
        folded
            .invoke(
                folded.find_export("grow").expect("grow"),
                &[1],
                &mut folded_result,
            )
            .expect("folded grow");
        assert_eq!(folded_result[0], 1);
        assert_eq!(helper.memory_size(0), 2);
        assert_eq!(
            helper.memory_grow(0, 1).expect("helper refused grow"),
            u32::MAX as u64
        );
        folded
            .invoke(
                folded.find_export("grow").expect("grow"),
                &[1],
                &mut folded_result,
            )
            .expect("folded refused grow");
        assert_eq!(folded_result[0], u32::MAX as u64);
    }

    #[test]
    fn frame_memory_primitives_match_folded_memory64() {
        let src = r#"(module
            (memory i64 1 2)
            (func (export "roundtrip") (param i64 i64) (result i64)
                local.get 0 local.get 1 i64.store offset=8
                local.get 0 i64.load offset=8)
            (func (export "load") (param i64) (result i64)
                local.get 0 i64.load)
            (func (export "grow") (param i64) (result i64)
                local.get 0 memory.grow))"#;
        let bin: StdVec<u8> = wat::parse_str(src).expect("wat");
        let engine = Engine::with_defaults();
        let mut helper = InterpInstance::new(
            &engine,
            Module::new("memory64-helper", &bin).expect("module"),
            None,
            &[],
        )
        .expect("helper instance");
        let mut folded = InterpInstance::new(
            &engine,
            Module::new("memory64-folded", &bin).expect("module"),
            None,
            &[],
        )
        .expect("folded instance");

        let raw = 0x0123_4567_89ab_cdefu64;
        helper.mem_store(16, 0, 8, 8, raw).expect("helper store");
        let helper_raw = helper.mem_load(16, 0, 8, 8).expect("helper load");
        let mut folded_result = [0u64; 1];
        folded
            .invoke(
                folded.find_export("roundtrip").expect("roundtrip"),
                &[16, raw],
                &mut folded_result,
            )
            .expect("folded roundtrip");
        assert_eq!(helper_raw, folded_result[0]);

        let helper_oob = helper.mem_load(65_530, 0, 0, 8).expect_err("helper OOB");
        let folded_oob = folded
            .invoke(
                folded.find_export("load").expect("load"),
                &[65_530],
                &mut folded_result,
            )
            .expect_err("folded OOB");
        assert_eq!(helper_oob, folded_oob);

        assert_eq!(helper.memory_grow(0, 1).expect("helper grow"), 1);
        folded
            .invoke(
                folded.find_export("grow").expect("grow"),
                &[1],
                &mut folded_result,
            )
            .expect("folded grow");
        assert_eq!(folded_result[0], 1);
        assert_eq!(
            helper.memory_grow(0, 1).expect("helper refused grow"),
            u64::MAX
        );
        folded
            .invoke(
                folded.find_export("grow").expect("grow"),
                &[1],
                &mut folded_result,
            )
            .expect("folded refused grow");
        assert_eq!(folded_result[0], u64::MAX);
    }

    #[cfg(all(sf_has_guard_pages, sf_jit))]
    #[test]
    fn interpreter_grows_a_shared_jit_guarded_memory() {
        let provider_wasm = wat::parse_str(
            r#"(module
                (memory (export "memory") 1 3))"#,
        )
        .expect("provider wat");
        let jit =
            Engine::new(Config::new().tier(crate::vm::engine::Tier::Jit)).expect("jit engine");
        let provider_module = Module::new("guard-provider", &provider_wasm).expect("module");
        let provider = crate::Instance::from_module(&jit, provider_module, &[])
            .expect("jit provider instance");
        let shared = provider.shared_memory_at(0).expect("exported memory");
        assert!(shared.has_guard_pages());

        let importer_wasm = wat::parse_str(
            r#"(module
                (import "provider" "memory" (memory 1 3))
                (func (export "grow") (param i32) (result i32)
                    local.get 0 memory.grow)
                (func (export "size") (result i32) memory.size)
                (func (export "roundtrip-new-page") (param i32) (result i32)
                    i32.const 65536 local.get 0 i32.store
                    i32.const 65536 i32.load))"#,
        )
        .expect("importer wat");
        let import = Import::memory_with_state(
            "provider",
            "memory",
            shared.limits.clone(),
            Some(shared.clone()),
        );
        let interp = Engine::new(Config::new().tier(crate::vm::engine::Tier::Interp))
            .expect("interp engine");
        let importer_module = Module::new("guard-importer", &importer_wasm).expect("module");
        let mut importer = crate::Instance::from_module(&interp, importer_module, &[import])
            .expect("interp importer instance");

        assert_eq!(
            importer
                .invoke("grow", &[Value::I32(1)])
                .expect("interpreter grow"),
            crate::collections::vec![Value::I32(1)]
        );
        assert_eq!(provider.memory_pages("memory"), Some(2));
        assert_eq!(shared.current_pages(), 2);
        assert_eq!(
            importer.invoke("size", &[]).expect("interpreter size"),
            crate::collections::vec![Value::I32(2)]
        );
        assert_eq!(
            importer
                .invoke("roundtrip-new-page", &[Value::I32(0x1234_5678)])
                .expect("newly committed page is accessible"),
            crate::collections::vec![Value::I32(0x1234_5678)]
        );

        assert_eq!(
            importer
                .invoke("grow", &[Value::I32(2)])
                .expect("over-limit grow returns failure"),
            crate::collections::vec![Value::I32(-1)]
        );
        assert_eq!(provider.memory_pages("memory"), Some(2));
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
    fn frame_global_primitives_match_private_and_shared_folded_globals() {
        let private_src = r#"(module
            (global $g (mut i64) (i64.const 5))
            (func (export "read") (result i64) global.get $g)
            (func (export "write") (param i64) local.get 0 global.set $g))"#;
        let private_bin: StdVec<u8> = wat::parse_str(private_src).expect("wat");
        let engine = Engine::with_defaults();
        let mut private = InterpInstance::new(
            &engine,
            Module::new("private-global", &private_bin).expect("module"),
            None,
            &[],
        )
        .expect("private instance");
        assert!(private.global_state_at(0).is_none());
        private.global_set_from_frame(0, 17);
        let mut result = [0u64; 1];
        private
            .invoke(private.find_export("read").expect("read"), &[], &mut result)
            .expect("folded read");
        assert_eq!(result[0], 17);
        private
            .invoke(private.find_export("write").expect("write"), &[23], &mut [])
            .expect("folded write");
        assert_eq!(private.global_get_for_frame(0), 23);

        let shared_src = r#"(module
            (global $g (export "state") (mut i64) (i64.const 5))
            (func (export "read") (result i64) global.get $g)
            (func (export "write") (param i64) local.get 0 global.set $g))"#;
        let shared_bin: StdVec<u8> = wat::parse_str(shared_src).expect("wat");
        let mut shared = InterpInstance::new(
            &engine,
            Module::new("shared-global", &shared_bin).expect("module"),
            None,
            &[],
        )
        .expect("shared instance");
        shared.global_set_from_frame(0, 29);
        shared
            .invoke(shared.find_export("read").expect("read"), &[], &mut result)
            .expect("folded read");
        assert_eq!(result[0], 29);
        assert_eq!(shared.global_state_at(0).expect("shared cell").raw(), 29);
        shared
            .invoke(shared.find_export("write").expect("write"), &[31], &mut [])
            .expect("folded write");
        assert_eq!(shared.global_get_for_frame(0), 31);
        assert_eq!(shared.global_state_at(0).expect("shared cell").raw(), 31);
    }

    #[test]
    fn frame_global_funcrefs_localize_and_absolutize_like_folded() {
        let src = r#"(module
            (func $first (export "first"))
            (func $second (export "second"))
            (global $slot (export "slot") (mut funcref) (ref.func $first))
            (func (export "read") (result funcref) global.get $slot)
            (func (export "write") (param funcref) local.get 0 global.set $slot))"#;
        let bin: StdVec<u8> = wat::parse_str(src).expect("wat");
        let engine = Engine::with_defaults();
        let mut inst = InterpInstance::new(
            &engine,
            Module::new("global-funcref", &bin).expect("module"),
            None,
            &[],
        )
        .expect("instance");
        let first_local = ref_to_machine_raw(RefValue::new(0), SLOT_GP_UNIT_BYTES);
        let second_local = ref_to_machine_raw(RefValue::new(1), SLOT_GP_UNIT_BYTES);

        inst.global_set_from_frame(0, second_local);
        let second_absolute = ref_to_machine_raw(
            inst.function_handle_at(1).expect("second identity"),
            SLOT_GP_UNIT_BYTES,
        );
        assert_eq!(
            inst.global_state_at(0).expect("shared global").raw(),
            second_absolute
        );
        let mut result = [0u64; 1];
        inst.invoke(inst.find_export("read").expect("read"), &[], &mut result)
            .expect("folded read");
        assert_eq!(result[0], second_local);

        inst.invoke(
            inst.find_export("write").expect("write"),
            &[first_local],
            &mut [],
        )
        .expect("folded write");
        assert_eq!(inst.global_get_for_frame(0), first_local);
        let first_absolute = ref_to_machine_raw(
            inst.function_handle_at(0).expect("first identity"),
            SLOT_GP_UNIT_BYTES,
        );
        assert_eq!(
            inst.global_state_at(0).expect("shared global").raw(),
            first_absolute
        );
    }

    #[test]
    fn frame_runtime_primitives_keep_entity_borrows_scoped() {
        let src = r#"(module
            (memory 1 2)
            (func $first (export "first"))
            (func $second (export "second"))
            (global $slot (export "slot") (mut funcref) (ref.func $first)))"#;
        let bin: StdVec<u8> = wat::parse_str(src).expect("wat");
        let module = Module::new("scoped-frame-primitives", &bin).expect("module");
        let registry = LinkRegistry::new();
        let (_, instance_backref) = registry.reserve_instance();
        #[cfg(sf_module_validator)]
        let baseline =
            super::super::ValidatedBaselinePlan::validate(&module).expect("validated baseline");
        // Deliberately stop before native linking. This keeps the ownership
        // oracle executable under Miri while exercising the real memory and
        // global entities built for an interpreter instance.
        let mut inst = InterpInstance::build(
            &Engine::with_defaults(),
            module,
            #[cfg(sf_module_validator)]
            baseline,
            None,
            &[],
            None,
            registry.arenas(),
            instance_backref,
        )
        .expect("build instance state");

        inst.mem_store(4, 0, 0, 4, 0x1234_5678).expect("store");
        assert_eq!(inst.mem_load(4, 0, 0, 4).expect("load"), 0x1234_5678);
        assert_eq!(inst.memory_grow(0, 1).expect("grow"), 1);

        let second_local = ref_to_machine_raw(RefValue::new(1), SLOT_GP_UNIT_BYTES);
        inst.global_set_from_frame(0, second_local);
        assert_eq!(inst.global_get_for_frame(0), second_local);
        let second_absolute = ref_to_machine_raw(
            inst.function_handle_at(1).expect("second identity"),
            SLOT_GP_UNIT_BYTES,
        );
        assert_eq!(
            inst.global_state_at(0).expect("shared global").raw(),
            second_absolute
        );
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
        // A same-function throw follows the same runtime exception path as a
        // cross-call throw; only the unwind search stops in the current
        // activation.
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
    fn exception_unwind_crosses_native_descendants_and_reuses_the_instance() {
        // `$middle -> $leaf` remains a native-linked call. Only `$catch`'s
        // protected call to `$middle` is a Rust checkpoint, so this exercises
        // discarding native descendant return records while retaining the
        // catching activation's route.
        let bin: StdVec<u8> = wat::parse_str(
            r#"(module
                (tag $e (param i32))
                (func $leaf (export "uncaught") (param i32)
                    (throw $e (local.get 0)))
                (func $middle (param i32)
                    (call $leaf (local.get 0)))
                (func (export "catch") (param i32) (result i32)
                    (block $h (result i32)
                        (try_table (result i32) (catch $e $h)
                            (call $middle (local.get 0))
                            (i32.const -1))
                        unreachable))
                (func (export "plain") (param i32) (result i32)
                    (i32.add (local.get 0) (i32.const 1))))"#,
        )
        .expect("wat");
        let module = Module::new("unwind", &bin).expect("module");
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            None,
            &[],
        )
        .expect("instance");
        let catch = inst.find_export("catch").expect("catch");
        let uncaught = inst.find_export("uncaught").expect("uncaught");
        let plain = inst.find_export("plain").expect("plain");
        let mut result = [0u64; 1];

        inst.invoke(catch, &[41], &mut result).expect("caught");
        assert_eq!(result[0], 41, "typed payload must retain its order/value");
        inst.invoke(plain, &[7], &mut result)
            .expect("normal call after catch");
        assert_eq!(result[0], 8);

        let err = inst
            .invoke(uncaught, &[9], &mut [])
            .expect_err("uncaught throw must surface");
        assert!(matches!(err, WasmError::Exception { .. }));
        inst.invoke(plain, &[10], &mut result)
            .expect("normal call after uncaught exception");
        assert_eq!(result[0], 11);
    }

    #[test]
    fn throw_ref_reuses_the_caught_exception_object() {
        let bin: StdVec<u8> = wat::parse_str(
            r#"(module
                (tag $e)
                (func (export "go")
                    (block $h (result exnref)
                        (try_table (catch_ref $e $h)
                            (throw $e))
                        unreachable)
                    throw_ref))"#,
        )
        .expect("wat");
        let module = Module::new("throw-ref", &bin).expect("module");
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            None,
            &[],
        )
        .expect("instance");
        let tag = inst.tag_identity_at(0).expect("tag");
        let go = inst.find_export("go").expect("go");
        let err = inst
            .invoke(go, &[], &mut [])
            .expect_err("throw_ref must rethrow");
        let WasmError::Exception {
            exn,
            tag: actual_tag,
            ..
        } = err
        else {
            panic!("expected exception, got {err:?}");
        };
        assert_eq!(actual_tag, tag);
        let exn_instance = inst
            .link_registry
            .resolve_exn(exn)
            .expect("same exception handle remains resolvable");
        assert_eq!(exn_instance.tag, tag);
        assert!(exn_instance.fields.is_empty());
    }

    #[test]
    fn world_funcrefs_use_one_identity_and_drive_foreign_calls() {
        let bin: StdVec<u8> = wat::parse_str(
            r#"(module
                (type $ft (func))
                (tag $e (param (ref $ft)))
                (func $pad)
                (func $dummy (type $ft))
                (elem declare func $dummy)
                (func (export "same") (result i32)
                    (block $h (result (ref $ft))
                        (try_table (catch $e $h)
                            (throw $e (ref.func $dummy)))
                        unreachable)
                    (ref.eq (ref.func $dummy)))
                (func (export "escape")
                    (throw $e (ref.func $dummy))))"#,
        )
        .expect("wat");
        let module = Module::new("world-funcref", &bin).expect("module");
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            None,
            &[],
        )
        .expect("instance");
        let same = inst.find_export("same").expect("same");
        let escape = inst.find_export("escape").expect("escape");
        let mut result = [0u64; 1];

        inst.invoke(same, &[], &mut result).expect("local catch");
        assert_eq!(
            result[0], 1,
            "localizing the shared identity must preserve ref.eq"
        );

        let WasmError::Exception { exn, .. } = inst
            .invoke(escape, &[], &mut [])
            .expect_err("escape exception")
        else {
            panic!("expected exception");
        };
        let field = inst
            .link_registry
            .resolve_exn(exn)
            .and_then(|exn| exn.fields.first().copied())
            .expect("exception funcref field");
        let Value::Ref(handle, _) = field else {
            panic!("expected funcref field");
        };
        assert!(!handle.is_special(), "world funcrefs remain untagged");
        assert_ne!(handle, RefValue::new(1));
        let entry = inst
            .link_registry
            .functions
            .entry_for_handle(handle)
            .expect("world function identity");
        assert_eq!(entry.owner, inst.instance_backref.self_id());
        assert_eq!(entry.local_index, 1);

        exercise_foreign_world_funcref_calls();
    }

    fn exercise_foreign_world_funcref_calls() {
        fn invoke_lease(
            lease: &InterpInstanceLease,
            func_index: usize,
            results: &mut [u64],
        ) -> Result<(), WasmError> {
            let mut access = InterpInstanceAccess::checked_out(lease.checkout_for_invocation()?);
            InterpInstance::invoke_access(&mut access, func_index, &[], results)
        }

        let provider_bin: StdVec<u8> = wat::parse_str(
            r#"(module
                (type $t (func (result i32)))
                (type $identity (func (param funcref) (result funcref)))
                (func (export "f") (type $t) (result i32)
                    i32.const 7)
                (func (export "identity") (type $identity)
                    local.get 0))"#,
        )
        .expect("provider wat");
        let provider_module = Module::new("provider", &provider_bin).expect("provider module");
        let provider_type = provider_module.functions()[0].func_type().clone();
        let provider_type_index = provider_module.functions()[0].type_index();
        let identity_type = provider_module.functions()[1].func_type().clone();
        let identity_type_index = provider_module.functions()[1].type_index();
        let provider_types = provider_module.types().clone();
        let registry = LinkRegistry::new();
        let engine = crate::vm::engine::Engine::with_defaults();
        #[cfg(sf_module_validator)]
        let provider_baseline = super::super::ValidatedBaselinePlan::validate(&provider_module)
            .expect("validated provider baseline");
        let provider = match InterpInstance::new_partial_with_registry(
            &engine,
            provider_module,
            #[cfg(sf_module_validator)]
            provider_baseline,
            None,
            &[],
            None,
            &registry,
        ) {
            Ok(provider) => provider,
            Err((_, error)) => panic!("provider: {error:?}"),
        };
        let handle = provider
            .with_instance(|provider| provider.function_handle_at(0))
            .expect("provider identity");
        let identity_handle = provider
            .with_instance(|provider| provider.function_handle_at(1))
            .expect("provider identity function");

        let consumer_bin: StdVec<u8> = wat::parse_str(
            r#"(module
                (type $t (func (result i32)))
                (type $identity (func (param funcref) (result funcref)))
                (import "provider" "f" (func $f (type $t)))
                (import "provider" "identity" (func $identity (type $identity)))
                (table 1 funcref)
                (elem (i32.const 0) func $f)
                (func $local)
                (elem declare func $local)
                (func (export "direct") (result i32)
                    call $f)
                (func (export "by_ref") (result i32)
                    (call_ref $t (ref.func $f)))
                (func (export "indirect") (result i32)
                    (call_indirect (type $t) (i32.const 0)))
                (func (export "round_trip") (result i32)
                    ref.func $local
                    call $identity
                    ref.func $local
                    ref.eq))"#,
        )
        .expect("consumer wat");
        let make_imports = || {
            [
                Import::linked_func_typed_with_context_and_index(
                    "provider",
                    "f",
                    handle,
                    provider_type.clone(),
                    provider_type_index,
                    provider_types.clone(),
                ),
                Import::linked_func_typed_with_context_and_index(
                    "provider",
                    "identity",
                    identity_handle,
                    identity_type.clone(),
                    identity_type_index,
                    provider_types.clone(),
                ),
            ]
        };
        let hooked_module =
            Module::new("hooked-consumer", &consumer_bin).expect("hooked consumer module");
        #[cfg(sf_module_validator)]
        let hooked_baseline = super::super::ValidatedBaselinePlan::validate(&hooked_module)
            .expect("validated hooked baseline");
        let hooked = match InterpInstance::new_partial_with_registry(
            &engine,
            hooked_module,
            #[cfg(sf_module_validator)]
            hooked_baseline,
            None,
            &make_imports(),
            Some(FuncRefHost {
                invoke: alloc::boxed::Box::new(
                    move |callee: RefValue, args: &[u64], results: &mut [u64]| {
                        if callee == handle {
                            assert!(args.is_empty());
                            results[0] = 13;
                            return Ok(());
                        }
                        assert_eq!(callee, identity_handle);
                        assert_eq!(args.len(), 1);
                        results[0] = args[0];
                        Ok(())
                    },
                ),
            }),
            &registry,
        ) {
            Ok(hooked) => hooked,
            Err((_, error)) => panic!("hooked consumer: {error:?}"),
        };

        for (export, expected) in [
            ("direct", 13),
            ("by_ref", 13),
            ("indirect", 13),
            ("round_trip", 1),
        ] {
            let index = hooked
                .with_instance(|hooked| hooked.find_export(export))
                .expect("hooked consumer export");
            let mut result = [0u64; 1];
            invoke_lease(&hooked, index, &mut result)
                .expect("the installed hook drives the world identity");
            assert_eq!(result[0], expected, "{export}");
        }

        let without_hook_module =
            Module::new("no-hook-consumer", &consumer_bin).expect("no-hook consumer module");
        #[cfg(sf_module_validator)]
        let without_hook_baseline =
            super::super::ValidatedBaselinePlan::validate(&without_hook_module)
                .expect("validated no-hook baseline");
        let without_hook = match InterpInstance::new_partial_with_registry(
            &engine,
            without_hook_module,
            #[cfg(sf_module_validator)]
            without_hook_baseline,
            None,
            &make_imports(),
            None,
            &registry,
        ) {
            Ok(without_hook) => without_hook,
            Err((_, error)) => panic!("no-hook consumer: {error:?}"),
        };

        for (export, expected) in [
            ("direct", 7),
            ("by_ref", 7),
            ("indirect", 7),
            ("round_trip", 1),
        ] {
            let index = without_hook
                .with_instance(|without_hook| without_hook.find_export(export))
                .expect("no-hook consumer export");
            let mut result = [0u64; 1];
            invoke_lease(&without_hook, index, &mut result)
                .expect("the engine-native path drives the world identity");
            assert_eq!(result[0], expected, "{export}");
        }
    }

    #[test]
    fn host_throw_is_validated_and_enters_the_same_unwind_path() {
        let bin: StdVec<u8> = wat::parse_str(
            r#"(module
                (type $exception (func (param i32)))
                (import "host" "exception" (tag $e (type $exception)))
                (import "host" "throw" (func $throw (param i32)))
                (func (export "go") (param i32) (result i32)
                    (block $h (result i32)
                        (try_table (result i32) (catch $e $h)
                            (call $throw (local.get 0))
                            (i32.const -1))
                        unreachable)))"#,
        )
        .expect("wat");
        let module = Module::new("host-throw", &bin).expect("module");
        let tag_type = module.tags()[0].func_type().clone();
        let func_type = module.functions()[0].func_type().clone();
        let (tag_import, tag) = Import::tag_typed_with_handle("host", "exception", tag_type);
        let func_import = Import::func_typed(
            "host",
            "throw",
            |_caller, _args, _results| Ok(()),
            func_type,
        );
        let host = InterpInstance::boxed_host(move |_module, _name, _memory, args, _results| {
            Err(WasmError::HostThrow {
                tag,
                args: vec![Value::I32(args[0] as u32 as i32)],
            })
        });
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            Some(host),
            &[tag_import, func_import],
        )
        .expect("instance");
        let go = inst.find_export("go").expect("go");
        let mut result = [0u64; 1];

        inst.invoke(go, &[37], &mut result)
            .expect("well-typed host throw is catchable");
        assert_eq!(result[0], 37);

        inst.set_host(move |_module, _name, _memory, _args, _results| {
            Err(WasmError::HostThrow { tag, args: vec![] })
        });
        let err = inst
            .invoke(go, &[1], &mut result)
            .expect_err("mistyped host throw must trap");
        assert!(matches!(
            err,
            WasmError::Trap("host threw mistyped exception")
        ));
    }

    #[test]
    fn host_throw_rejects_null_for_a_non_null_reference_payload() {
        let bin: StdVec<u8> = wat::parse_str(
            r#"(module
                (type $ft (func))
                (type $exception (func (param (ref $ft))))
                (import "host" "exception" (tag $e (type $exception)))
                (import "host" "throw" (func $throw))
                (func (export "go") (result i32)
                    (block $h (result (ref $ft))
                        (try_table (catch $e $h)
                            (call $throw)
                            unreachable)
                        unreachable)
                    ref.is_null))"#,
        )
        .expect("wat");
        let module = Module::new("host-ref-throw", &bin).expect("module");
        let tag_type = module.tags()[0].func_type().clone();
        let func_type = module.functions()[0].func_type().clone();
        let (tag_import, tag) = Import::tag_typed_with_handle("host", "exception", tag_type);
        let func_import = Import::func_typed(
            "host",
            "throw",
            |_caller, _args, _results| Ok(()),
            func_type,
        );
        let host = InterpInstance::boxed_host(move |_module, _name, _memory, _args, _results| {
            Err(WasmError::HostThrow {
                tag,
                args: vec![Value::Ref(RefValue::null(), RefType::funcref())],
            })
        });
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            Some(host),
            &[tag_import, func_import],
        )
        .expect("instance");
        let go = inst.find_export("go").expect("go");

        assert!(matches!(
            inst.invoke(go, &[], &mut [0]),
            Err(WasmError::Trap("host threw mistyped exception"))
        ));
    }

    #[test]
    fn catch_all_rejects_a_forged_exception_handle() {
        let bin: StdVec<u8> = wat::parse_str(
            r#"(module
                (type $exception (func))
                (import "host" "exception" (tag $e (type $exception)))
                (import "host" "throw" (func $throw))
                (func (export "go") (result i32)
                    (block $h
                        (try_table (catch_all $h)
                            (call $throw))
                        unreachable)
                    i32.const 1))"#,
        )
        .expect("wat");
        let module = Module::new("forged-exception", &bin).expect("module");
        let tag_type = module.tags()[0].func_type().clone();
        let func_type = module.functions()[0].func_type().clone();
        let (tag_import, tag) = Import::tag_typed_with_handle("host", "exception", tag_type);
        let func_import = Import::func_typed(
            "host",
            "throw",
            |_caller, _args, _results| Ok(()),
            func_type,
        );
        let host = InterpInstance::boxed_host(move |_module, _name, _memory, _args, _results| {
            Err(WasmError::Exception {
                exn: RefValue::new(1_234_567),
                tag,
                module_tag_name: None,
            })
        });
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            Some(host),
            &[tag_import, func_import],
        )
        .expect("instance");
        let go = inst.find_export("go").expect("go");

        assert!(matches!(
            inst.invoke(go, &[], &mut [0]),
            Err(WasmError::Trap("invalid exception reference"))
        ));
    }

    #[test]
    fn imported_tags_match_by_runtime_identity_not_signature() {
        // `$a` and `$b` are different module indices with the same
        // signature. The catch matches only when the linker binds both
        // imports to the same runtime tag. A distinct `$b` is not caught and
        // surfaces as an uncaught wasm exception.
        let bin: StdVec<u8> = wat::parse_str(
            r#"(module
                (type $e (func))
                (import "m" "a" (tag $a (type $e)))
                (import "m" "b" (tag $b (type $e)))
                (func (export "go") (result i32)
                    (block $caught
                        (try_table
                            (catch $a $caught)
                            (throw $b))
                        unreachable)
                    i32.const 1))"#,
        )
        .expect("wat");
        let engine = crate::vm::engine::Engine::with_defaults();

        let module = Module::new("aliased", &bin).expect("module");
        let tag_type = module.tags()[0].func_type().clone();
        let (import_a, shared_handle) = Import::tag_typed_with_handle("m", "a", tag_type.clone());
        let import_b = Import::linked_tag_typed("m", "b", shared_handle, tag_type);
        let mut aliased =
            InterpInstance::new(&engine, module, None, &[import_a, import_b]).expect("aliased");
        let go = aliased.find_export("go").expect("go");
        let mut result = [0u64; 1];
        aliased
            .invoke(go, &[], &mut result)
            .expect("invoke aliased tags");
        assert_eq!(result[0], 1, "aliased imports must match the typed catch");

        let module = Module::new("distinct", &bin).expect("module");
        let tag_type = module.tags()[0].func_type().clone();
        let import_a = Import::tag_typed("m", "a", tag_type.clone());
        let import_b = Import::tag_typed("m", "b", tag_type);
        let mut distinct =
            InterpInstance::new(&engine, module, None, &[import_a, import_b]).expect("distinct");
        let go = distinct.find_export("go").expect("go");
        let err = distinct
            .invoke(go, &[], &mut result)
            .expect_err("same-signature distinct imports must not match");
        assert!(
            matches!(err, WasmError::Exception { .. }),
            "the distinct throw must remain uncaught, got {err:?}"
        );
    }

    #[test]
    fn exported_imported_tag_preserves_the_provided_handle() {
        let bin: StdVec<u8> = wat::parse_str(
            r#"(module
                (type $e (func))
                (import "m" "t" (tag $t (type $e)))
                (export "t" (tag $t)))"#,
        )
        .expect("wat");
        let module = Module::new("reexport", &bin).expect("module");
        let tag_type = module.tags()[0].func_type().clone();
        let (import, handle) = Import::tag_typed_with_handle("m", "t", tag_type);
        let engine = crate::vm::engine::Engine::new(
            crate::config::Config::new().tier(crate::vm::engine::Tier::Interp),
        )
        .expect("engine");
        let instance = crate::vm::instance::Instance::from_module(&engine, module, &[import])
            .expect("instance");

        assert_eq!(
            instance.tag_identity("t"),
            Some(handle),
            "re-exporting an imported tag must not mint a new identity"
        );
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
    fn self_return_call_resets_non_parameter_locals() {
        let src = r#"(module
            (func $count (param $n i64) (param $sum i64) (result i64)
                (local $scratch i64)
                local.get $n
                i64.eqz
                if
                    local.get $sum
                    return
                end
                local.get $n
                i64.const 1
                i64.sub
                local.get $sum
                local.get $scratch
                i64.add
                i64.const 7
                local.set $scratch
                return_call $count)
            (func (export "go") (param i64) (result i64)
                local.get 0
                i64.const 0
                call $count))"#;
        assert_eq!(
            run1(src, "go", &[1000]).unwrap(),
            0,
            "every tail invocation must observe a fresh zeroed scratch local"
        );
    }

    #[test]
    fn imported_return_call_preserves_native_caller_routes() {
        let bin: StdVec<u8> = wat::parse_str(
            r#"(module
                (import "host" "ret" (func $ret (result i32)))
                (func $tail (result i32)
                    (return_call $ret))
                (func $parent (result i32)
                    (i32.add (call $tail) (i32.const 5)))
                (func (export "go") (result i32)
                    (i32.add (call $parent) (i32.const 2))))"#,
        )
        .expect("wat");
        let module = Module::new("host-tail", &bin).expect("module");
        let func_type = module.functions()[0].func_type().clone();
        let import = Import::func_typed(
            "host",
            "ret",
            |_caller, _args, results| {
                results[0] = Value::I32(40);
                Ok(())
            },
            func_type,
        );
        let host = InterpInstance::boxed_host(|_module, _name, _memory, _args, results| {
            results[0] = 40;
            Ok(())
        });
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            Some(host),
            &[import],
        )
        .expect("instance");
        let go = inst.find_export("go").expect("go");
        let mut result = [0u64; 1];

        inst.invoke(go, &[], &mut result).expect("first invoke");
        assert_eq!(result[0], 47);
        inst.invoke(go, &[], &mut result).expect("second invoke");
        assert_eq!(result[0], 47);
    }

    #[test]
    fn exported_table_call_indirect_guard_uses_native_payload_fast_path() {
        let (bin, _) = instantiate(
            r#"(module
                (type $unused (func))
                (type $unary (func (param i32) (result i32)))
                (table (export "table") 2 funcref)
                (elem (i32.const 0) $double $square)
                (func $double (type $unary) (param i32) (result i32)
                    local.get 0 i32.const 2 i32.mul)
                (func $square (type $unary) (param i32) (result i32)
                    local.get 0 local.get 0 i32.mul)
                (func (export "go") (param i32 i32) (result i32)
                    local.get 1 local.get 0 call_indirect (type $unary)))"#,
        );
        let module = Module::new("exported-table-indirect", &bin).expect("module");
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            None,
            &[],
        )
        .expect("instance");
        let go = inst.find_export("go").expect("go");

        {
            let native = inst.native.as_ref().expect("native state");
            assert!(inst.tables[0].entries.fast_base().is_none());
            let (cell, ins, cell_index) = linked_cell_for_op(&inst, go, Op::CallIndirect);
            let ci_h = native.engine.callindirect_handler_addr();
            assert_eq!(cell.h, ci_h);
            assert_eq!(cell.c, ins.c, "native cell retains the module type id");
            assert_eq!(
                native.plan.native_call_indirect_payload(cell_index, ci_h),
                Some((ins.a, ins.b, ins.c as u32))
            );
        }

        let mut result = [u64::MAX];
        inst.invoke(go, &[0, 21], &mut result).expect("double");
        assert_eq!(result, [42]);
        inst.invoke(go, &[1, 7], &mut result).expect("square");
        assert_eq!(result, [49]);
        assert_eq!(
            inst.slow_exit_stats()
                .into_iter()
                .find(|&(op, _)| op == Op::CallIndirect),
            Some((Op::CallIndirect, 2))
        );
    }

    #[test]
    fn static_call_indirect_keeps_generic_slow_restore() {
        let (bin, _) = instantiate(
            r#"(module
                (type $unary (func (param i32) (result i32)))
                (table 1 funcref)
                (elem (i32.const 0) $double)
                (func $double (type $unary) (param i32) (result i32)
                    local.get 0 i32.const 2 i32.mul)
                (func (export "go") (param i32) (result i32)
                    local.get 0 i32.const 0 call_indirect (type $unary)))"#,
        );
        let module = Module::new("static-table-indirect", &bin).expect("module");
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            None,
            &[],
        )
        .expect("instance");
        let go = inst.find_export("go").expect("go");

        {
            let native = inst.native.as_ref().expect("native state");
            let (cell, ins, cell_index) = linked_cell_for_op(&inst, go, Op::CallIndirect);
            assert_eq!(cell.h, native.engine.slow_stub_addr());
            assert_eq!(
                native.plan.native_call_indirect_payload(
                    cell_index,
                    native.engine.callindirect_handler_addr()
                ),
                None
            );
            let restored = native
                .plan
                .restore_slow_instr(cell_index, native.engine.slow_stub_addr())
                .expect("generic slow instruction");
            assert_eq!(
                (
                    restored.op,
                    restored.flags,
                    restored.a,
                    restored.b,
                    restored.c
                ),
                (ins.op, ins.flags, ins.a, ins.b, ins.c)
            );
        }

        let mut result = [u64::MAX];
        inst.invoke(go, &[21], &mut result).expect("invoke");
        assert_eq!(result, [42]);
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
    fn reference_slot_encoding_round_trips_function_range_endpoints() {
        for encoded in [0, crate::vm::value::FUNCADDR_TOP] {
            let handle = RefValue::new(encoded);
            let slot = ref_to_machine_raw(handle, SLOT_GP_UNIT_BYTES);
            assert_eq!(
                machine_raw_to_ref(slot, SLOT_GP_UNIT_BYTES),
                handle,
                "interpreter target-width slot changed a function handle"
            );
        }

        let null_slot = ref_to_machine_raw(RefValue::null(), SLOT_GP_UNIT_BYTES);
        let expected_null_slot = if SLOT_GP_UNIT_BYTES == 4 {
            u32::MAX as u64
        } else {
            u64::MAX
        };
        assert_eq!(null_slot, expected_null_slot);
        assert_eq!(
            machine_raw_to_ref(null_slot, SLOT_GP_UNIT_BYTES),
            RefValue::null()
        );
    }

    #[test]
    fn special_funcref_in_private_table_takes_slow_path() {
        const FUNCADDR_TOP: usize = (1 << 28) - 2;

        // Keep the table index non-constant: constant-index call sites link
        // directly to the slow stub and do not exercise the native guard.
        let bin: StdVec<u8> = wat::parse_str(
            r#"(module
                (type $callee (func (result i32)))
                (import "host" "foreign" (func $foreign (result funcref)))
                (table 1 funcref)
                (func (export "go") (param $table_index i32) (result i32)
                    i32.const 0
                    call $foreign
                    table.set
                    local.get $table_index
                    call_indirect (type $callee)))"#,
        )
        .expect("wat");
        let module = Module::new("special-private-table", &bin).expect("module");
        let special = ref_to_machine_raw(RefValue::hostref(7), SLOT_GP_UNIT_BYTES);
        let unregistered = ref_to_machine_raw(RefValue::new(FUNCADDR_TOP / 2), SLOT_GP_UNIT_BYTES);
        assert_eq!(unregistered >> 32, 0);
        let mut returned_refs = [special, unregistered].into_iter();
        let host = InterpInstance::boxed_host(move |_module, _name, _memory, _args, results| {
            results[0] = returned_refs.next().expect("one reference per invocation");
            Ok(())
        });
        let mut inst = InterpInstance::new(
            &crate::vm::engine::Engine::with_defaults(),
            module,
            Some(host),
            &[],
        )
        .expect("instance");
        let go = inst.find_export("go").expect("go");

        {
            let native = inst.native.as_ref().expect("native state");
            assert!(inst.tables[0].entries.fast_base().is_some());
            let (cell, ins, cell_index) = linked_cell_for_op(&inst, go, Op::CallIndirect);
            let ci_h = native.engine.callindirect_handler_addr();
            assert_eq!(cell.h, ci_h);
            assert_eq!(cell.c, ins.c);
            assert_eq!(
                native.plan.native_call_indirect_payload(cell_index, ci_h),
                Some((ins.a, ins.b, ins.c as u32))
            );
        }

        assert!(matches!(
            inst.invoke(go, &[0], &mut [0]),
            Err(WasmError::Trap("indirect call type mismatch"))
        ));
        assert!(matches!(
            inst.invoke(go, &[0], &mut [0]),
            Err(WasmError::Trap(
                crate::vm::interpreter::EXTERNAL_FUNCREF_HOST_REQUIRED
            ))
        ));
        assert_eq!(
            inst.slow_exit_stats()
                .into_iter()
                .find(|&(op, _)| op == Op::CallIndirect),
            Some((Op::CallIndirect, 2))
        );
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
