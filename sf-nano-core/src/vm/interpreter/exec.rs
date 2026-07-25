//! Stage A2 executor: a correctness-first Rust dispatch loop over the
//! folded instruction stream. Performance is explicitly not a goal of this
//! stage (see `docs/INTERPRETER_V2.md` §7); it exists to pin the predecoder
//! and value semantics before the generated-handler dispatch substrate.
//!
//! v1 scope: single-module, no imports (host functions, imported
//! memories/tables/globals all rejected at instantiation), one linear
//! memory, funcref tables for `call_indirect`.

use tracked_alloc::boxed::Box;
use tracked_alloc::rc::Rc;

use crate::collections::{vec, Vec};
use crate::error::WasmError;
use crate::module::entities::{Data, Element, ElementInit, GlobalDef};
use crate::module::Module;
use crate::opcodes::Opcode;
use crate::utils::limits::Limitable;
use crate::value_type::ValueType;

#[cfg(all(sf_jit, sf_backend_arm64))]
use super::dispatch_arm64::{
    DCell, EnterState, LinkedFunction, NativeEngine, EXIT_RETURN, EXIT_SLOW, EXIT_TRAP_BASE,
    RET_RECORD, TRAP_KINDS,
};
use super::instr::Op;
#[cfg(all(sf_jit, sf_backend_arm64))]
use super::instr::{Instr, FLAG_A_CONST, FLAG_B_CONST, FLAG_FUSED};
use super::predecode::{predecode_function, PredecodedFunction};

const PAGE: usize = 65536;
const NULL_FUNC: u32 = u32::MAX;
#[cfg(all(sf_jit, sf_backend_arm64))]
const MAX_CALL_DEPTH: u32 = 4096;
/// Value-stack budget per invoke. Frames overlap on one contiguous stack
/// (a callee's frame base is its caller's staged-argument slot); running
/// past the budget traps "call stack exhausted" on every backend.
#[cfg(all(sf_jit, sf_backend_arm64))]
const VALUE_STACK_SLOTS: usize = 256 * 1024;

/// Host dispatcher for imported functions: called with the import's module
/// and field names, the linear memory, argument slots, and result slots.
/// The signature carries only std types (`&mut [u8]`, not this crate's
/// tracked collections), so external callers stay feature-independent.
pub type HostDispatch<'h> =
    Box<dyn FnMut(&str, &str, &mut [u8], &[u64], &mut [u64]) -> Result<(), WasmError> + 'h>;

/// `max` is a growth limit, consulted only by `table.grow`; the executor
/// that implements it is arm64-only, so the field follows that gate.
/// `entries` is not gated: instantiation reads its length to bounds-check
/// active element segments on every target.
struct TableState {
    entries: Vec<u32>,
    #[cfg(all(sf_jit, sf_backend_arm64))]
    max: u64,
}

/// `max_pages` mirrors `TableState::max`: only `memory.grow` reads it.
struct MemoryState {
    bytes: Vec<u8>,
    #[cfg(all(sf_jit, sf_backend_arm64))]
    max_pages: u64,
}

/// One live call frame. Calls and returns are driven by an explicit
/// activation stack in the driver loop, never by host recursion, so call
/// depth is interpreter data (the classic-interpreter lesson).
#[cfg(all(sf_jit, sf_backend_arm64))]
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
#[cfg(all(sf_jit, sf_backend_arm64))]
struct DriveCtx {
    stack: Vec<u64>,
    ret_stack: Vec<u64>,
    /// Byte cursor into `ret_stack`.
    ret_cursor: usize,
    /// The accumulator relayed across native sessions: call results ride
    /// it over activation boundaries (sentinel returns, host calls).
    acc: u64,
}

#[cfg(all(sf_jit, sf_backend_arm64))]
enum StepExit {
    Call { callee: usize, arg_base: usize },
    Return,
}

/// Result of executing exactly one instruction.
#[cfg(all(sf_jit, sf_backend_arm64))]
pub(super) enum Effect {
    Next,
    Jump(usize),
    Call { callee: usize, arg_base: usize },
    Ret,
}

/// Stage-B state: the emitted handler engine plus per-function dispatch
/// cells (parallel to `InterpInstance::funcs`).
#[cfg(all(sf_jit, sf_backend_arm64))]
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
pub struct InterpInstance<'m> {
    module: &'m Module,
    funcs: Vec<Option<Rc<PredecodedFunction>>>,
    /// Runtime state only the executor touches. `new()` still builds it on
    /// every target -- doing so is what rejects imported/64-bit/non-funcref
    /// memories and tables and traps out-of-range active segments -- but a
    /// target without an executor validates and drops it rather than
    /// carrying it.
    #[cfg(all(sf_jit, sf_backend_arm64))]
    memories: Vec<MemoryState>,
    #[cfg(all(sf_jit, sf_backend_arm64))]
    dropped_data: Vec<bool>,
    #[cfg(all(sf_jit, sf_backend_arm64))]
    dropped_elems: Vec<bool>,
    globals: Vec<u64>,
    #[cfg(all(sf_jit, sf_backend_arm64))]
    tables: Vec<TableState>,
    host: Option<HostDispatch<'m>>,
    #[cfg(all(sf_jit, sf_backend_arm64))]
    native: Option<NativeState>,
}

/// Recover an `Op` from its dense `#[repr(u16)]` discriminant (the same
/// invariant the dispatch key relies on).
fn op_from_index(i: usize) -> Op {
    debug_assert!(i <= Op::Unreachable as usize);
    unsafe { core::mem::transmute(i as u16) }
}

/// Nonzero per-op counts, descending.
#[cfg(all(sf_jit, sf_backend_arm64))]
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
fn eval_simple_const(expr: &[u8]) -> Result<u64, WasmError> {
    use crate::op_decoder::{Decoder, Immediate, OpStream, OpcodeHandler};
    use crate::opcodes::WasmOpcode;

    struct H {
        value: Option<u64>,
    }
    impl OpcodeHandler for H {
        fn on_decode_begin(&mut self) -> Result<(), WasmError> {
            Ok(())
        }
        fn on_stream<'x, 'y, 'z>(
            &mut self,
            stream: &mut OpStream<'x, 'y, 'z>,
        ) -> Result<(), WasmError> {
            while let Some(op) = stream.next()? {
                let v = match (op.wasm_op, &op.imm) {
                    (WasmOpcode::OP(Opcode::I32_CONST), Immediate::I32(v)) => *v as u32 as u64,
                    (WasmOpcode::OP(Opcode::I64_CONST), Immediate::I64(v)) => *v as u64,
                    (WasmOpcode::OP(Opcode::F32_CONST), Immediate::F32(v)) => v.to_bits() as u64,
                    (WasmOpcode::OP(Opcode::F64_CONST), Immediate::F64(v)) => v.to_bits(),
                    (WasmOpcode::OP(Opcode::REF_FUNC), Immediate::FunctionIndex(i)) => *i as u64,
                    (WasmOpcode::OP(Opcode::REF_NULL), _) => NULL_FUNC as u64,
                    (WasmOpcode::OP(Opcode::END), _) => continue,
                    _ => {
                        return Err(WasmError::invalid(
                            "interp: unsupported constant expression",
                        ))
                    }
                };
                if self.value.replace(v).is_some() {
                    return Err(WasmError::invalid(
                        "interp: unsupported constant expression",
                    ));
                }
            }
            Ok(())
        }
        fn on_decode_end(&mut self) -> Result<(), WasmError> {
            Ok(())
        }
    }

    let mut h = H { value: None };
    // A const expr is an expression body: decode it like a function body.
    // The decoder is scoped so its borrow of `h` provably ends before the
    // read below (with tracked-alloc's memprof feature unified in, its Vec
    // has a destructor and the borrow would otherwise extend to it).
    {
        let mut d = Decoder::new(expr);
        d.add_handler(&mut h);
        d.decode_function()?;
    }
    h.value
        .ok_or(WasmError::invalid("interp: empty constant expression"))
}

impl<'m> InterpInstance<'m> {
    pub fn new(module: &'m Module) -> Result<Self, WasmError> {
        // Functions: predecode every local function eagerly; imports are
        // rejected on call, not here, so import-free modules always work.
        let mut funcs = Vec::new();
        for (i, f) in module.functions().iter().enumerate() {
            if f.is_import() {
                funcs.push(None);
            } else {
                funcs.push(Some(Rc::new(predecode_function(module, i)?)));
            }
        }

        // Memories.
        let mut memories = Vec::new();
        for m in module.memories() {
            if m.is_import() {
                return Err(WasmError::invalid("interp: imported memory unsupported"));
            }
            let limits = m.spec().limits();
            if limits.is64 {
                return Err(WasmError::invalid("interp: memory64 unsupported"));
            }
            memories.push(MemoryState {
                bytes: vec![0u8; limits.min() as usize * PAGE],
                #[cfg(all(sf_jit, sf_backend_arm64))]
                max_pages: limits.max().unwrap_or(65536) as u64,
            });
        }

        // Globals.
        let mut globals = Vec::new();
        for g in module.globals() {
            match g.def() {
                GlobalDef::Local(spec) => globals.push(eval_simple_const(spec.init_expr())?),
                GlobalDef::Import { .. } => {
                    return Err(WasmError::invalid("interp: imported global unsupported"))
                }
            }
        }

        // Tables (funcref only) + active element segments.
        let mut tables = Vec::new();
        for t in module.tables() {
            if t.is_import() {
                return Err(WasmError::invalid("interp: imported table unsupported"));
            }
            if t.value_type() != ValueType::funcref() {
                return Err(WasmError::invalid("interp: non-funcref table unsupported"));
            }
            let limits = t.spec().limits();
            if limits.is64 {
                return Err(WasmError::invalid("interp: table64 unsupported"));
            }
            tables.push(TableState {
                entries: vec![NULL_FUNC; limits.min() as usize],
                #[cfg(all(sf_jit, sf_backend_arm64))]
                max: limits.max().unwrap_or(u32::MAX as usize) as u64,
            });
        }
        let mut dropped_elems = vec![false; module.elements().len()];
        for (ei, e) in module.elements().iter().enumerate() {
            if let Element::Active {
                table_index,
                offset_expr,
                init,
            } = e
            {
                let off = eval_simple_const(offset_expr)? as usize;
                let table = tables
                    .get_mut(*table_index)
                    .ok_or(WasmError::invalid("interp: element table out of range"))?;
                match init {
                    ElementInit::FunctionIndexes(idxs) => {
                        if off + idxs.len() > table.entries.len() {
                            return Err(WasmError::trap("out of bounds table access"));
                        }
                        for (k, &fi) in idxs.iter().enumerate() {
                            table.entries[off + k] = fi as u32;
                        }
                    }
                    ElementInit::InitExprs { exprs, .. } => {
                        if off + exprs.len() > table.entries.len() {
                            return Err(WasmError::trap("out of bounds table access"));
                        }
                        for (k, expr) in exprs.iter().enumerate() {
                            table.entries[off + k] = eval_simple_const(expr)? as u32;
                        }
                    }
                }
                dropped_elems[ei] = true; // active segments drop after use
            }
        }

        // Active data segments.
        let mut dropped_data = vec![false; module.data().len()];
        for (i, d) in module.data().iter().enumerate() {
            if let Data::Active {
                memory_index,
                offset_expr,
                init,
            } = d
            {
                let off = eval_simple_const(offset_expr)? as usize;
                let mem = memories
                    .get_mut(*memory_index)
                    .ok_or(WasmError::trap("out of bounds memory access"))?;
                if off + init.len() > mem.bytes.len() {
                    return Err(WasmError::trap("out of bounds memory access"));
                }
                mem.bytes[off..off + init.len()].copy_from_slice(init);
                dropped_data[i] = true; // active segments drop after use
            }
        }

        let mut inst = InterpInstance {
            module,
            funcs,
            #[cfg(all(sf_jit, sf_backend_arm64))]
            memories,
            #[cfg(all(sf_jit, sf_backend_arm64))]
            dropped_data,
            #[cfg(all(sf_jit, sf_backend_arm64))]
            dropped_elems,
            globals,
            #[cfg(all(sf_jit, sf_backend_arm64))]
            tables,
            host: None,
            #[cfg(all(sf_jit, sf_backend_arm64))]
            native: None,
        };
        inst.enable_native_dispatch()?;
        // Run the module's start function, if any.
        if let Some(si) = module.start_function_index() {
            inst.invoke(si, &[], &mut [])?;
        }
        Ok(inst)
    }

    /// Native dispatch is the interpreter's only execution engine (the
    /// stage-A Rust loop was removed after B validation; `exec_ins`
    /// remains as the native chain's slow path). Targets without a
    /// backend, and hosts without executable memory, fail instantiation
    /// cleanly here — they get an engine again when build-time handler
    /// generation lands.
    #[cfg(not(all(sf_jit, sf_backend_arm64)))]
    fn enable_native_dispatch(&mut self) -> Result<(), WasmError> {
        Err(WasmError::invalid(
            "interp: no native dispatch backend for this target",
        ))
    }

    /// Emit the handler set and link every predecoded function. Failure
    /// (no executable memory on this host) fails instantiation: there is
    /// no interpreter without the native chain.
    #[cfg(all(sf_jit, sf_backend_arm64))]
    fn enable_native_dispatch(&mut self) -> Result<(), WasmError> {
        let engine = NativeEngine::new()?;
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
        for (i, lf) in linked.iter_mut().enumerate() {
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
                let end = start + lf.cells.len() as u64 * 32;
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
        #[cfg(all(sf_jit, sf_backend_arm64))]
        {
            return self.native.as_ref().map_or(0, |n| n.dispatches);
        }
        #[cfg(not(all(sf_jit, sf_backend_arm64)))]
        {
            0
        }
    }

    /// Map a native pc to `(func_index, cells_start)`.
    #[cfg(all(sf_jit, sf_backend_arm64))]
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
    /// Size in bytes of the emitted dispatch engine, 0 when there is none.
    /// Worth watching: every added handler family or operand class grows it
    /// against a hard buffer assert, and once emission moves to build time
    /// this becomes binary size.
    pub fn engine_code_len(&self) -> usize {
        #[cfg(all(sf_jit, sf_backend_arm64))]
        if let Some(native) = &self.native {
            return native.engine.code_len();
        }
        0
    }

    pub fn slow_exit_stats(&self) -> Vec<(Op, u64)> {
        #[cfg(all(sf_jit, sf_backend_arm64))]
        if let Some(native) = &self.native {
            return op_counts(&native.slow_exits);
        }
        Vec::new()
    }

    /// Install the host dispatcher used for imported functions. Generic
    /// so callers pass a plain closure; boxing happens here (unsizing to
    /// the dyn target through the `alloc` box, then wrapping in the
    /// tracked facade — the same pattern as the JIT's host callbacks).
    pub fn set_host<F>(&mut self, host: F)
    where
        F: FnMut(&str, &str, &mut [u8], &[u64], &mut [u64]) -> Result<(), WasmError> + 'm,
    {
        let host: alloc::boxed::Box<
            dyn FnMut(&str, &str, &mut [u8], &[u64], &mut [u64]) -> Result<(), WasmError> + 'm,
        > = alloc::boxed::Box::new(host);
        self.host = Some(tracked_alloc::box_from_alloc(host));
    }

    /// Resolve the element-segment function value at `seg[k]`.
    #[cfg(all(sf_jit, sf_backend_arm64))]
    fn elem_value(&self, seg: usize, k: usize) -> Result<u32, WasmError> {
        match self.module.elements().get(seg).map(|e| e.get_init()) {
            Some(ElementInit::FunctionIndexes(idxs)) => idxs
                .get(k)
                .map(|&fi| fi as u32)
                .ok_or(WasmError::trap("out of bounds table access")),
            Some(ElementInit::InitExprs { exprs, .. }) => exprs
                .get(k)
                .ok_or(WasmError::trap("out of bounds table access"))
                .and_then(|e| eval_simple_const(e).map(|v| v as u32)),
            None => Err(WasmError::trap("out of bounds table access")),
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
    #[cfg(all(sf_jit, sf_backend_arm64))]
    pub fn invoke(
        &mut self,
        func_index: usize,
        args: &[u64],
        results: &mut [u64],
    ) -> Result<(), WasmError> {
        let func = self
            .funcs
            .get(func_index)
            .and_then(|f| f.clone())
            .ok_or(WasmError::invalid("interp: bad function index"))?;
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
        let stack = self.drive(root, args)?;
        results.copy_from_slice(&stack[..results.len()]);
        Ok(())
    }

    /// Stub for targets without a native backend: [`Self::new`] fails
    /// there, so no instance exists to invoke — this only keeps
    /// cross-target callers compiling.
    #[cfg(not(all(sf_jit, sf_backend_arm64)))]
    pub fn invoke(
        &mut self,
        _func_index: usize,
        _args: &[u64],
        _results: &mut [u64],
    ) -> Result<(), WasmError> {
        Err(WasmError::invalid(
            "interp: no native dispatch backend for this target",
        ))
    }

    /// The call/return trampoline: runs activations to their next call or
    /// return boundary, keeping call depth as data on `saved`.
    #[cfg(all(sf_jit, sf_backend_arm64))]
    fn drive(&mut self, root: Activation, args: &[u64]) -> Result<Vec<u64>, WasmError> {
        if root.func.frame_slots as usize > VALUE_STACK_SLOTS {
            return Err(WasmError::trap("call stack exhausted"));
        }
        let mut ctx = DriveCtx {
            // Full-length and zeroed up front: native dispatch roams the
            // whole region through raw pointers, so every slot must be
            // initialized memory (zeroed pages are cheap; only touched
            // ones are ever committed).
            stack: vec![0u64; VALUE_STACK_SLOTS],
            ret_stack: vec![0u64; (MAX_CALL_DEPTH as usize + 8) * (RET_RECORD / 8)],
            ret_cursor: 0,
            acc: 0,
        };
        ctx.stack[..args.len()].copy_from_slice(args);

        let mut act = root;
        let mut saved: Vec<Activation> = Vec::new();
        loop {
            match self.native_step(&mut act, &mut ctx)? {
                StepExit::Call { callee, arg_base } => {
                    if saved.len() as u32 >= MAX_CALL_DEPTH {
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
                    if new_base + f.frame_slots as usize > VALUE_STACK_SLOTS {
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
                StepExit::Return => {
                    // Results are already in place: the callee's frame base
                    // IS the caller's staged-argument slot.
                    match saved.pop() {
                        None => return Ok(ctx.stack),
                        Some(parent) => act = parent,
                    }
                }
            }
        }
    }

    // `packed` is `memidx << 48 | offset` as emitted by the predecoder.
    #[cfg(all(sf_jit, sf_backend_arm64))]
    fn mem_load(&self, addr: u64, packed: u64, size: usize) -> Result<&[u8], WasmError> {
        let mem = self
            .memories
            .get((packed >> 48) as usize)
            .ok_or(WasmError::trap("out of bounds memory access"))?;
        let ea = addr + (packed & 0xffff_ffff_ffff); // both < 2^49, no overflow
        let end = ea + size as u64;
        if end > mem.bytes.len() as u64 {
            return Err(WasmError::trap("out of bounds memory access"));
        }
        Ok(&mem.bytes[ea as usize..end as usize])
    }

    #[cfg(all(sf_jit, sf_backend_arm64))]
    fn mem_store(&mut self, addr: u64, packed: u64, size: usize) -> Result<&mut [u8], WasmError> {
        let mem = self
            .memories
            .get_mut((packed >> 48) as usize)
            .ok_or(WasmError::trap("out of bounds memory access"))?;
        let ea = addr + (packed & 0xffff_ffff_ffff);
        let end = ea + size as u64;
        if end > mem.bytes.len() as u64 {
            return Err(WasmError::trap("out of bounds memory access"));
        }
        Ok(&mut mem.bytes[ea as usize..end as usize])
    }

    /// Dispatch an imported function to the host. `frame` is the CALLER's
    /// frame slice; arguments and results live at `arg_base` within it.
    #[cfg(all(sf_jit, sf_backend_arm64))]
    fn call_host(
        &mut self,
        callee: usize,
        frame: &mut [u64],
        arg_base: usize,
    ) -> Result<(), WasmError> {
        let func = self
            .module
            .functions()
            .get(callee)
            .ok_or(WasmError::trap("undefined element"))?;
        let (mod_name, field) = match func.def() {
            crate::module::entities::FunctionDef::Import { module, name, .. } => {
                (module.clone(), name.clone())
            }
            _ => return Err(WasmError::invalid("interp: not an import")),
        };
        let p = func.func_type().params().len();
        let r = func.func_type().results().len();
        let host = self
            .host
            .as_mut()
            .ok_or(WasmError::invalid("interp: no host dispatcher installed"))?;
        let mut results = [0u64; 8];
        if r > results.len() {
            return Err(WasmError::invalid("interp: too many host results"));
        }
        let args: Vec<u64> = frame[arg_base..arg_base + p].iter().copied().collect();
        let mem0 = self
            .memories
            .first_mut()
            .map(|m| m.bytes.as_mut_slice())
            .unwrap_or(&mut []);
        host(&mod_name, &field, mem0, &args, &mut results[..r])?;
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
    #[cfg(all(sf_jit, sf_backend_arm64))]
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
            stack_limit: stack_ptr + (VALUE_STACK_SLOTS as u64) * 8,
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
                state.mem_base = m.bytes.as_mut_ptr() as u64;
                state.mem_len = m.bytes.len() as u64;
            }
            // Table 0 can move or grow only on the slow path, so a
            // per-entry refresh keeps the native indirect-call handler
            // valid with no invalidation protocol.
            if let Some(t) = self.tables.first() {
                state.table0_base = t.entries.as_ptr() as u64;
                state.table0_len = t.entries.len() as u64;
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
    #[cfg(all(sf_jit, sf_backend_arm64))]
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
                } else {
                    (opa!($ins) as u32 as u64, $ins.c as usize)
                };
                let bytes = self.mem_load(addr, $ins.b, $size)?;
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
                } else {
                    (opa!($ins) as u32 as u64, $ins.c)
                };
                let val = opb!($ins);
                let bytes = self.mem_store(addr, off, $size)?;
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
            Op::F32_Ceil => fun32!(ins, f32::ceil),
            Op::F32_Floor => fun32!(ins, f32::floor),
            Op::F32_Trunc => fun32!(ins, f32::trunc),
            Op::F32_Nearest => fun32!(ins, f32::round_ties_even),
            Op::F32_Sqrt => fun32!(ins, f32::sqrt),
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
            Op::F64_Ceil => fun64!(ins, f64::ceil),
            Op::F64_Floor => fun64!(ins, f64::floor),
            Op::F64_Trunc => fun64!(ins, f64::trunc),
            Op::F64_Nearest => fun64!(ins, f64::round_ties_even),
            Op::F64_Sqrt => fun64!(ins, f64::sqrt),
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
                let pages = self
                    .memories
                    .get(m)
                    .map(|x| x.bytes.len() / PAGE)
                    .unwrap_or(0);
                frame[ins.c as usize] = pages as u64;
            }
            Op::MemoryGrow => {
                let delta = opa!(ins) as u32 as u64;
                let mem = self
                    .memories
                    .get_mut(ins.b as usize)
                    .ok_or(WasmError::trap("out of bounds memory access"))?;
                let cur = (mem.bytes.len() / PAGE) as u64;
                let want = cur + delta;
                if want > mem.max_pages || want > 65536 {
                    frame[ins.c as usize] = u32::MAX as u64;
                } else {
                    mem.bytes.resize(want as usize * PAGE, 0);
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
                if d + n > mem.bytes.len() as u64 {
                    return Err(WasmError::trap("out of bounds memory access"));
                }
                mem.bytes[d as usize..(d + n) as usize].fill(val as u8);
            }
            Op::MemoryCopy => {
                let base = ins.a as usize;
                let (d, s0, n) = (frame[base], frame[base + 1], frame[base + 2]);
                let (d, s0, n) = (d as u32 as u64, s0 as u32 as u64, n as u32 as u64);
                let dm = (ins.b >> 32) as usize;
                let sm = (ins.b & 0xffff_ffff) as usize;
                let dlen = self.memories.get(dm).map(|x| x.bytes.len()).unwrap_or(0) as u64;
                let slen = self.memories.get(sm).map(|x| x.bytes.len()).unwrap_or(0) as u64;
                if d + n > dlen || s0 + n > slen {
                    return Err(WasmError::trap("out of bounds memory access"));
                }
                if dm == sm {
                    self.memories[dm]
                        .bytes
                        .copy_within(s0 as usize..(s0 + n) as usize, d as usize);
                } else {
                    for k in 0..n as usize {
                        let v = self.memories[sm].bytes[s0 as usize + k];
                        self.memories[dm].bytes[d as usize + k] = v;
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
                let mlen = self.memories.get(m).map(|x| x.bytes.len()).unwrap_or(0) as u64;
                if d + n > mlen || s0 + n > src_len {
                    // A zero-size init on a dropped segment must succeed
                    // when both offsets are in bounds.
                    if !(n == 0 && d <= mlen && s0 <= src_len) {
                        return Err(WasmError::trap("out of bounds memory access"));
                    }
                }
                if n > 0 {
                    let (d, s0, n) = (d as usize, s0 as usize, n as usize);
                    self.memories[m].bytes[d..d + n].copy_from_slice(&data[s0..s0 + n]);
                }
            }
            Op::DataDrop => {
                if let Some(x) = self.dropped_data.get_mut(ins.a as usize) {
                    *x = true;
                }
            }

            // ---- globals ----
            Op::GlobalGet => {
                frame[ins.c as usize] = self.globals[ins.a as usize];
            }
            Op::GlobalSet => {
                let v = opa!(ins);
                self.globals[ins.c as usize] = v;
            }

            // ---- ref/table ----
            Op::RefIsNull => {
                let v = opa!(ins);
                frame[ins.c as usize] = (v == NULL_FUNC as u64) as u64;
            }
            Op::TableGet => {
                let i = opa!(ins) as u32 as usize;
                let t = &self
                    .tables
                    .get(ins.b as usize)
                    .ok_or(WasmError::trap("out of bounds table access"))?
                    .entries;
                let v = *t
                    .get(i)
                    .ok_or(WasmError::trap("out of bounds table access"))?;
                frame[ins.c as usize] = v as u64;
            }
            Op::TableSet => {
                let i = opa!(ins) as u32 as usize;
                let v = opb!(ins) as u32;
                let t = &mut self
                    .tables
                    .get_mut(ins.c as usize)
                    .ok_or(WasmError::trap("out of bounds table access"))?
                    .entries;
                *t.get_mut(i)
                    .ok_or(WasmError::trap("out of bounds table access"))? = v;
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
                let init = opa!(ins) as u32;
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
                    frame[base + 1] as u32,
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
                t[i as usize..(i + n) as usize].fill(val);
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
                if dt == st {
                    self.tables[dt]
                        .entries
                        .copy_within(s0 as usize..(s0 + n) as usize, d as usize);
                } else {
                    for k in 0..n as usize {
                        let v = self.tables[st].entries[s0 as usize + k];
                        self.tables[dt].entries[d as usize + k] = v;
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
                    self.tables[tidx].entries[d as usize + k] = v;
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
            Op::Call => {
                return Ok(Effect::Call {
                    callee: ins.a as usize,
                    arg_base: ins.b as usize,
                });
            }
            Op::CallIndirect => {
                let t = opa!(ins) as u32 as usize;
                let table = self
                    .tables
                    .get((ins.c >> 32) as usize)
                    .ok_or(WasmError::trap("undefined element"))?;
                let fi = *table
                    .entries
                    .get(t)
                    .ok_or(WasmError::trap("undefined element"))?;
                if fi == NULL_FUNC {
                    return Err(WasmError::trap("uninitialized element"));
                }
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
                return Ok(Effect::Call {
                    callee: fi as usize,
                    arg_base: ins.b as usize,
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
#[cfg(all(sf_jit, sf_backend_arm64))]
fn trunc_checked(x: f64, lo: f64, hi_excl: f64) -> Result<f64, WasmError> {
    if x.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer"));
    }
    let t = x.trunc();
    if t < lo || t >= hi_excl {
        return Err(WasmError::trap("integer overflow"));
    }
    Ok(t)
}

#[cfg(all(sf_jit, sf_backend_arm64))]
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

#[cfg(all(sf_jit, sf_backend_arm64))]
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

#[cfg(all(sf_jit, sf_backend_arm64))]
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

#[cfg(all(sf_jit, sf_backend_arm64))]
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

#[cfg(all(test, sf_jit, sf_backend_arm64))]
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
        let mut inst = InterpInstance::new(&module)?;
        let idx = inst.find_export(export).expect("export");
        let mut results = [0u64; 1];
        inst.invoke(idx, args, &mut results)?;
        Ok(results[0])
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
        let mut inst = InterpInstance::new(&module).expect("instantiate");
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
