//! Bounded-counter store-to-load forwarding over the final SSA.
//!
//! Compilers commonly keep a loop counter in linear memory (clang's
//! `ctx->len` idiom): every iteration stores `counter+1`, writes one byte at
//! `buffer[counter]`, then reloads the counter from memory to compare against
//! a bound `B`. The reload is redundant — the store dominates it in the same
//! block — but eliminating it requires proving the intervening byte store
//! cannot alias the counter cell, which needs the loop-invariant bound
//! `counter < B`. On Apple M4 the redundant same-address store→load chain is
//! also a measured pipeline hazard: the eagerly-issued reload waits on store
//! data every iteration and the exit branch resolves from it (sha256: −16%
//! throughput, recovered in full by forwarding the stored value).
//!
//! This pass proves the bound with a closed induction and forwards the stored
//! value to the reload:
//!
//! - A *cell* is `(root local R, constant offset, 4 bytes)`, addressed as
//!   `i32.load/store {offset} (local.get R)`. `R` must never be reassigned in
//!   the function.
//! - The candidate window is `store cell, v` … `load cell` within one block
//!   with no call and no unclassifiable memory write in between.
//! - An intervening store is admissible when its address resolves to a
//!   constant offset from `R` whose byte range is disjoint from the cell, or
//!   when it is the counter byte-store `i32.store8 (base + local.get C)`
//!   where `base` resolves to `R + delta` and the *counter protocol* proves
//!   `C < B` with `delta + B <= cell offset`.
//! - Counter protocol (all checked structurally, bail on any mismatch):
//!   every assignment to `C` in the function is either a constant `< B`, or
//!   the result of a candidate cell load that is immediately followed by
//!   `set C` and a block terminator branching on `C != B`, whose `== B`
//!   successor reassigns `C` to a constant `< B` before any use of `C` in a
//!   counter byte-store. Induction: `C` starts `< B` (constant init), each
//!   iteration stores `C+1 <= B` to the cell, the in-window byte store stays
//!   below the cell, the reload therefore returns the stored value, and the
//!   `!= B` guard filters `B` before the value can reach the next byte store.
//!   Values of `C` are locals, so calls cannot alter them; the `== B` path
//!   resets `C` regardless of what a callee writes to memory.
//!
//! The rewrite deletes the load and repoints its uses (including terminator
//! edge bindings) at the stored value, which dominates them from the same
//! block.

use crate::collections;
use tracked_alloc::collections::BTreeMap;

use crate::vm::middle::{
    frame::FrameSlot,
    ssa_ir::ir::{DecodedOperand, SsaInst, SsaOp, SsaOperand, SsaProgram, SsaTerminator, SsaValue},
};
use crate::vm::wasm::primitive_op::PrimitiveOpKind;

pub(crate) fn forward_bounded_counters(program: &mut SsaProgram) {
    let mut rewrites: collections::Vec<Rewrite> = collections::Vec::new();
    for block_index in 0..program.blocks.len() {
        collect_block_rewrites(program, block_index, &mut rewrites);
    }
    if rewrites.is_empty() {
        return;
    }

    // Verify each counter protocol once; a failed protocol drops every
    // rewrite that depends on it. Protocol-free rewrites pass unconditionally.
    let mut verified: BTreeMap<(u16, u32, u16, u64, usize), bool> = BTreeMap::new();
    rewrites.retain(|rewrite| {
        let Some(protocol) = rewrite.protocol else {
            return true;
        };
        let key = (
            rewrite.root.0,
            rewrite.offset,
            protocol.counter.0,
            protocol.bound,
            rewrite.block,
        );
        *verified.entry(key).or_insert_with(|| {
            counter_protocol_holds(
                program,
                rewrite.root,
                rewrite.offset,
                protocol.counter,
                protocol.bound,
                rewrite.block,
                protocol.store8_index,
            )
        })
    });

    // Apply bottom-up so removing an op never shifts a pending index.
    let mut rewrites = rewrites;
    rewrites.sort_by(|a, b| (b.block, b.load_index).cmp(&(a.block, a.load_index)));
    for rewrite in rewrites {
        apply_rewrite(program, &rewrite);
    }
}

struct Rewrite {
    block: usize,
    /// Op index of the redundant load within the block.
    load_index: usize,
    /// The load's result value; its single consumer (a `local.set`) is
    /// located at apply time because earlier rewrites in the same block can
    /// shift op indices.
    loaded: SsaValue,
    /// The stored value's recipe: `local.get inc_slot` (via `inc_get_op`,
    /// preserving the cache/slot access kind) + `inc_const`. The rewrite
    /// recomputes it instead of forwarding, keeping every value single-use.
    inc_slot: FrameSlot,
    inc_get_op: SsaOp,
    inc_add_op: SsaOp,
    inc_const: SsaOperand,
    root: FrameSlot,
    offset: u32,
    /// `None`: every intervening store was provably disjoint — forwarding is
    /// unconditional. `Some`: a counter byte-store was admitted under the
    /// bound; the counter protocol must verify.
    protocol: Option<CounterProtocol>,
}

#[derive(Clone, Copy)]
struct CounterProtocol {
    counter: FrameSlot,
    bound: u64,
    /// Op index of the counter byte-store in the window.
    store8_index: usize,
}

/// Full-width memory-cell store kinds eligible as window openers:
/// (class, offset, width). Class pairs a store with its same-kind load.
fn cell_store(kind: &PrimitiveOpKind) -> Option<(u8, u32, u64)> {
    match kind {
        PrimitiveOpKind::I32Store { offset, memidx: 0 } => Some((0, *offset, 4)),
        PrimitiveOpKind::I64Store { offset, memidx: 0 } => Some((1, *offset, 8)),
        PrimitiveOpKind::F32Store { offset, memidx: 0 } => Some((2, *offset, 4)),
        PrimitiveOpKind::F64Store { offset, memidx: 0 } => Some((3, *offset, 8)),
        _ => None,
    }
}

fn cell_load(kind: &PrimitiveOpKind) -> Option<(u8, u32, u64)> {
    match kind {
        PrimitiveOpKind::I32Load { offset, memidx: 0 } => Some((0, *offset, 4)),
        PrimitiveOpKind::I64Load { offset, memidx: 0 } => Some((1, *offset, 8)),
        PrimitiveOpKind::F32Load { offset, memidx: 0 } => Some((2, *offset, 4)),
        PrimitiveOpKind::F64Load { offset, memidx: 0 } => Some((3, *offset, 8)),
        _ => None,
    }
}

/// Byte range of any store whose address resolves to `root + const`:
/// (start, width). Sub-width stores count too — they close windows they
/// overlap but can be admitted as disjoint.
fn const_store_range(kind: &PrimitiveOpKind) -> Option<(u32, u64)> {
    match kind {
        PrimitiveOpKind::I32Store { offset, memidx: 0 } => Some((*offset, 4)),
        PrimitiveOpKind::I64Store { offset, memidx: 0 } => Some((*offset, 8)),
        PrimitiveOpKind::F32Store { offset, memidx: 0 } => Some((*offset, 4)),
        PrimitiveOpKind::F64Store { offset, memidx: 0 } => Some((*offset, 8)),
        PrimitiveOpKind::I32Store8 { offset, memidx: 0 } => Some((*offset, 1)),
        PrimitiveOpKind::I32Store16 { offset, memidx: 0 } => Some((*offset, 2)),
        PrimitiveOpKind::I64Store8 { offset, memidx: 0 } => Some((*offset, 1)),
        PrimitiveOpKind::I64Store16 { offset, memidx: 0 } => Some((*offset, 2)),
        PrimitiveOpKind::I64Store32 { offset, memidx: 0 } => Some((*offset, 4)),
        _ => None,
    }
}

fn ranges_disjoint(a_start: u64, a_width: u64, b_start: u64, b_width: u64) -> bool {
    a_start + a_width <= b_start || b_start + b_width <= a_start
}

/// Per-block value facts the matcher needs: which local a value was read
/// from, and constant folds of `local.get +` chains.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ValueFact {
    /// `local.get slot`, still valid (no later `local.set slot` seen).
    GetLocal(FrameSlot),
    /// `local.get slot` + constant.
    LocalPlusConst(FrameSlot, u64),
    /// `local.get a` + `local.get b` (either order).
    LocalPlusLocal(FrameSlot, FrameSlot),
    /// Constant.
    Const(u64),
    Opaque,
}

fn collect_block_rewrites(
    program: &SsaProgram,
    block_index: usize,
    rewrites: &mut collections::Vec<Rewrite>,
) {
    let block = &program.blocks[block_index];
    let mut facts: BTreeMap<SsaValue, ValueFact> = BTreeMap::new();

    // Open store windows: (root, offset, kind class) -> (op index, stored
    // operand, width).
    let mut open_stores: BTreeMap<(u16, u32, u8), (usize, SsaOperand, u64)> = BTreeMap::new();

    for (op_index, inst) in block.ops.iter().enumerate() {
        match inst.op {
            SsaOp::LOCAL_GET_SLOT | SsaOp::LOCAL_GET_CACHE => {
                facts.insert(inst.result, ValueFact::GetLocal(FrameSlot(inst.meta)));
            }
            SsaOp::LOCAL_SET_SLOT | SsaOp::LOCAL_SET_CACHE => {
                let slot = FrameSlot(inst.meta);
                // A reassigned root/base invalidates address facts derived
                // from it and any open window rooted at it.
                facts = facts
                    .into_iter()
                    .filter(|(_, fact)| !fact_mentions(fact, slot))
                    .collect();
                open_stores = open_stores
                    .into_iter()
                    .filter(|&((root, _, _), _)| root != slot.0)
                    .collect();
            }
            SsaOp::CALL => {
                open_stores.clear();
            }
            _ => {
                let Some(pool_idx) = inst.op.as_primitive_idx() else {
                    // Fill/Spill/other structural ops: fills read the frame,
                    // spills write frame slots (not linear memory); neither
                    // touches wasm memory. Keep windows open.
                    continue;
                };
                let kind = program.primitive_pool[pool_idx as usize].clone();
                match kind {
                    PrimitiveOpKind::I32Add => {
                        facts.insert(inst.result, add_fact(&program.const_pool, &facts, inst));
                    }
                    PrimitiveOpKind::I32Const { value } => {
                        facts.insert(inst.result, ValueFact::Const(value as u32 as u64));
                    }
                    ref kind if cell_store(kind).is_some() => {
                        let (class, offset, width) = cell_store(kind).expect("checked by guard");
                        if let Some(root) = operand_root(&program.const_pool, &facts, inst.args[0])
                        {
                            // Close every window this store may alias: keep
                            // only same-root windows with provably disjoint
                            // byte ranges (the store's own cell re-opens).
                            let store_root = root.0;
                            open_stores = open_stores
                                .into_iter()
                                .filter(|&((win_root, win_off, _), (_, _, win_width))| {
                                    win_root == store_root
                                        && ranges_disjoint(
                                            u64::from(offset),
                                            width,
                                            u64::from(win_off),
                                            win_width,
                                        )
                                })
                                .collect();
                            open_stores
                                .insert((root.0, offset, class), (op_index, inst.args[1], width));
                        } else {
                            open_stores.clear();
                        }
                    }
                    ref kind if cell_load(kind).is_some() => {
                        let (class, offset, width) = cell_load(kind).expect("checked by guard");
                        if let Some(root) = operand_root(&program.const_pool, &facts, inst.args[0])
                        {
                            if let Some(&(store_index, stored, _)) =
                                open_stores.get(&(root.0, offset, class))
                            {
                                let recipe = increment_recipe(
                                    program,
                                    block,
                                    &facts,
                                    store_index,
                                    op_index,
                                    stored,
                                );
                                let set_index =
                                    single_local_set_use(program, block, op_index, inst.result);
                                if let (
                                    Some((inc_slot, inc_get_op, inc_add_op, inc_const)),
                                    Some(set_index),
                                ) = (recipe, set_index)
                                {
                                    if let Some(protocol) = window_admissible(
                                        program,
                                        block,
                                        block_index,
                                        &facts,
                                        store_index,
                                        op_index,
                                        root,
                                        offset,
                                        width,
                                    ) {
                                        let _ = set_index;
                                        rewrites.push(Rewrite {
                                            block: block_index,
                                            load_index: op_index,
                                            loaded: inst.result,
                                            inc_slot,
                                            inc_get_op,
                                            inc_add_op,
                                            inc_const,
                                            root,
                                            offset,
                                            protocol,
                                        });
                                    }
                                }
                            }
                        }
                    }
                    _ => {
                        if primitive_writes_memory(&kind) {
                            // Classification happens lazily in
                            // `window_admissible`; here we only have to keep
                            // the conservative default for kinds the window
                            // check cannot admit. Windows stay open — the
                            // admissibility walk re-examines every op in the
                            // window and bails on anything unclassified.
                        }
                    }
                }
            }
        }
    }
}

fn fact_mentions(fact: &ValueFact, slot: FrameSlot) -> bool {
    match *fact {
        ValueFact::GetLocal(s) => s == slot,
        ValueFact::LocalPlusConst(s, _) => s == slot,
        ValueFact::LocalPlusLocal(a, b) => a == slot || b == slot,
        _ => false,
    }
}

fn add_fact(
    const_pool: &[u64],
    facts: &BTreeMap<SsaValue, ValueFact>,
    inst: &SsaInst,
) -> ValueFact {
    let a = operand_fact(const_pool, facts, inst.args[0]);
    let b = operand_fact(const_pool, facts, inst.args[1]);
    match (a, b) {
        (ValueFact::GetLocal(s), ValueFact::Const(c))
        | (ValueFact::Const(c), ValueFact::GetLocal(s)) => ValueFact::LocalPlusConst(s, c),
        (ValueFact::GetLocal(a), ValueFact::GetLocal(b)) => ValueFact::LocalPlusLocal(a, b),
        (ValueFact::LocalPlusConst(s, c), ValueFact::Const(c2))
        | (ValueFact::Const(c2), ValueFact::LocalPlusConst(s, c)) => {
            ValueFact::LocalPlusConst(s, c.wrapping_add(c2))
        }
        (ValueFact::Const(a), ValueFact::Const(b)) => ValueFact::Const(a.wrapping_add(b)),
        _ => ValueFact::Opaque,
    }
}

fn operand_fact(
    const_pool: &[u64],
    facts: &BTreeMap<SsaValue, ValueFact>,
    operand: SsaOperand,
) -> ValueFact {
    match operand.decode() {
        DecodedOperand::Value(value) => facts.get(&value).copied().unwrap_or(ValueFact::Opaque),
        DecodedOperand::Const(index) => const_pool
            .get(index as usize)
            .copied()
            .map(ValueFact::Const)
            .unwrap_or(ValueFact::Opaque),
        DecodedOperand::None => ValueFact::Opaque,
    }
}

fn operand_root(
    const_pool: &[u64],
    facts: &BTreeMap<SsaValue, ValueFact>,
    operand: SsaOperand,
) -> Option<FrameSlot> {
    match operand_fact(const_pool, facts, operand) {
        ValueFact::GetLocal(slot) => Some(slot),
        _ => None,
    }
}

fn primitive_writes_memory(kind: &PrimitiveOpKind) -> bool {
    matches!(
        kind,
        PrimitiveOpKind::I32Store { .. }
            | PrimitiveOpKind::I64Store { .. }
            | PrimitiveOpKind::F32Store { .. }
            | PrimitiveOpKind::F64Store { .. }
            | PrimitiveOpKind::I32Store8 { .. }
            | PrimitiveOpKind::I32Store16 { .. }
            | PrimitiveOpKind::I64Store8 { .. }
            | PrimitiveOpKind::I64Store16 { .. }
            | PrimitiveOpKind::I64Store32 { .. }
            | PrimitiveOpKind::V128Store { .. }
            | PrimitiveOpKind::MemoryFill { .. }
            | PrimitiveOpKind::MemoryCopy { .. }
            | PrimitiveOpKind::MemoryInit { .. }
            | PrimitiveOpKind::MemoryGrow { .. }
    )
}

/// Check every op strictly between the store and the load.
///
/// Returns `Some(None)` when every intervening store is provably disjoint
/// (unconditional forwarding), `Some(Some(protocol))` when exactly one
/// counter byte-store was admitted under the bounded-counter argument, and
/// `None` when the window is inadmissible.
#[allow(clippy::too_many_arguments)]
fn window_admissible(
    program: &SsaProgram,
    block: &crate::vm::middle::ssa_ir::ir::SsaBlock,
    block_index: usize,
    facts: &BTreeMap<SsaValue, ValueFact>,
    store_index: usize,
    load_index: usize,
    root: FrameSlot,
    cell_offset: u32,
    cell_width: u64,
) -> Option<Option<CounterProtocol>> {
    let mut protocol: Option<CounterProtocol> = None;

    for (window_offset, inst) in block.ops[store_index + 1..load_index].iter().enumerate() {
        let abs_index = store_index + 1 + window_offset;
        if inst.op == SsaOp::CALL {
            return None;
        }
        let Some(pool_idx) = inst.op.as_primitive_idx() else {
            continue;
        };
        let kind = &program.primitive_pool[pool_idx as usize];
        if !primitive_writes_memory(kind) {
            continue;
        }
        // Constant-addressed store from the same root: admissible iff the
        // byte ranges are disjoint.
        if let Some((offset, width)) = const_store_range(kind) {
            match operand_fact(&program.const_pool, facts, inst.args[0]) {
                ValueFact::GetLocal(slot) if slot == root => {
                    if !ranges_disjoint(
                        u64::from(offset),
                        width,
                        u64::from(cell_offset),
                        cell_width,
                    ) {
                        return None;
                    }
                    continue;
                }
                ValueFact::LocalPlusLocal(a, b)
                    if matches!(kind, PrimitiveOpKind::I32Store8 { .. }) =>
                {
                    // The counter byte-store: address = base + counter where
                    // base is a stable alias of root + delta. At most one per
                    // window, i32 cells only.
                    if protocol.is_some() || cell_width != 4 {
                        return None;
                    }
                    let guard = load_guard(program, block, load_index)?;
                    let mut admitted = false;
                    for (base, counter) in [(a, b), (b, a)] {
                        if counter != guard.counter {
                            continue;
                        }
                        let Some(delta) =
                            local_root_delta(program, base, root, block_index, abs_index)
                        else {
                            continue;
                        };
                        if delta + guard.bound + u64::from(offset) <= u64::from(cell_offset) {
                            protocol = Some(CounterProtocol {
                                counter,
                                bound: guard.bound,
                                store8_index: abs_index,
                            });
                            admitted = true;
                            break;
                        }
                    }
                    if !admitted {
                        return None;
                    }
                    continue;
                }
                _ => return None,
            }
        }
        // Memory fill/copy/grow or unclassifiable store forms.
        return None;
    }
    Some(protocol)
}

#[derive(Clone, Copy)]
struct LoadGuard {
    counter: FrameSlot,
    bound: u64,
}

/// The structure immediately after the load that yields the bound:
/// `set C <- loaded; get C -> v; I32Ne(v, #B) -> cond; branch cond`.
/// (Constant folding has already inlined `#B` into the Ne operand.)
fn load_guard(
    program: &SsaProgram,
    block: &crate::vm::middle::ssa_ir::ir::SsaBlock,
    load_index: usize,
) -> Option<LoadGuard> {
    let loaded = block.ops[load_index].result;
    let set = block.ops.get(load_index + 1)?;
    if !matches!(set.op, SsaOp::LOCAL_SET_SLOT | SsaOp::LOCAL_SET_CACHE) {
        return None;
    }
    if set.args[0].as_value() != Some(loaded) {
        return None;
    }
    let counter = FrameSlot(set.meta);

    // Find the Ne against a constant whose input is a fresh get of `counter`
    // (or the loaded value directly), and require the block terminator to
    // branch on its result.
    let mut ne_bound: Option<(SsaValue, u64)> = None;
    let mut counter_reads: collections::Vec<SsaValue> = collections::vec![loaded];
    for inst in &block.ops[load_index + 2..] {
        match inst.op {
            SsaOp::LOCAL_GET_SLOT | SsaOp::LOCAL_GET_CACHE => {
                if FrameSlot(inst.meta) == counter {
                    counter_reads.push(inst.result);
                }
            }
            SsaOp::LOCAL_SET_SLOT | SsaOp::LOCAL_SET_CACHE => {
                if FrameSlot(inst.meta) == counter {
                    return None;
                }
            }
            _ => {
                let Some(pool_idx) = inst.op.as_primitive_idx() else {
                    continue;
                };
                if let PrimitiveOpKind::I32Ne = &program.primitive_pool[pool_idx as usize] {
                    let input = inst.args[0].as_value();
                    let bound = const_operand(program, inst.args[1]);
                    if let (Some(input), Some(bound)) = (input, bound) {
                        if counter_reads.contains(&input) {
                            ne_bound = Some((inst.result, bound));
                        }
                    }
                }
            }
        }
    }
    let (cond, bound) = ne_bound?;
    if bound == 0 {
        return None;
    }
    match &block.terminator {
        SsaTerminator::Branch { cond: c, .. } if *c == cond => Some(LoadGuard { counter, bound }),
        _ => None,
    }
}

fn const_operand(program: &SsaProgram, operand: SsaOperand) -> Option<u64> {
    match operand.decode() {
        DecodedOperand::Const(index) => program.const_pool.get(index as usize).copied(),
        _ => None,
    }
}

/// `slot` is a stable alias of `root + delta` at `block_index`: the root is
/// assigned at most once (entry block only), and exactly one def of `slot`
/// reaches the block — a `root + const` def (in-block defs before
/// `before_op` take precedence over entry-reaching defs).
fn local_root_delta(
    program: &SsaProgram,
    slot: FrameSlot,
    root: FrameSlot,
    block_index: usize,
    before_op: usize,
) -> Option<u64> {
    // The root may be assigned at most once, and only in the entry block
    // (which dominates every use, so all `get root` values agree).
    let entry = program.entry.as_usize();
    let mut root_sets = 0usize;
    for (bi, block) in program.blocks.iter().enumerate() {
        for inst in &block.ops {
            if matches!(inst.op, SsaOp::LOCAL_SET_SLOT | SsaOp::LOCAL_SET_CACHE)
                && FrameSlot(inst.meta) == root
            {
                root_sets += 1;
                if root_sets > 1 || bi != entry {
                    return None;
                }
            }
        }
    }

    let defs = slot_reaching_defs(program, slot)?;
    // In-block defs before the use win over entry-reaching defs.
    let in_block_def = defs
        .sites
        .iter()
        .filter(|&&(b, op, _)| b == block_index && op < before_op)
        .last();
    if let Some(&(_, _, fact)) = in_block_def {
        return match fact {
            DefFact::RootPlusConst(base, delta) if base == root => Some(delta),
            _ => None,
        };
    }
    let reaching = defs.reach_in.get(block_index).copied()?;
    let mut delta: Option<u64> = None;
    for (site_index, &(_, _, fact)) in defs.sites.iter().enumerate() {
        if reaching & (1u64 << site_index) == 0 {
            continue;
        }
        match fact {
            DefFact::RootPlusConst(base, d) if base == root => {
                if delta.is_some_and(|existing| existing != d) {
                    return None;
                }
                delta = Some(d);
            }
            _ => return None,
        }
    }
    delta
}

/// The counter protocol: every assignment to `counter` in the function is a
/// constant `< bound`, or a guarded cell load (a load of `(root, offset)`
/// immediately stored to `counter` and compared `!= bound` by the block
/// terminator, whose `== bound` successor re-assigns `counter` to a constant
/// `< bound` before any `local.get counter`).
fn counter_protocol_holds(
    program: &SsaProgram,
    root: FrameSlot,
    offset: u32,
    counter: FrameSlot,
    bound: u64,
    window_block: usize,
    window_store8_op: usize,
) -> bool {
    let Some(defs) = slot_reaching_defs(program, counter) else {
        return false;
    };
    let Some(&reaching) = defs.reach_in.get(window_block) else {
        return false;
    };
    // Defs whose value the counter byte-store can observe: everything
    // reaching the block entry, plus in-block defs before the byte store.
    let mut observable: collections::Vec<(usize, usize, DefFact)> = collections::Vec::new();
    for (site_index, &(b, op, fact)) in defs.sites.iter().enumerate() {
        let reaches_entry = reaching & (1u64 << site_index) != 0;
        let in_block_before = b == window_block && op < window_store8_op;
        if reaches_entry || in_block_before {
            observable.push((b, op, fact));
        }
    }
    // An in-block def before the byte store shadows entry-reaching defs, but
    // only on this iteration's path; conservatively require every observable
    // def to conform.
    for (b, op, fact) in observable {
        match fact {
            DefFact::Const(value) if value < bound => {}
            _ => {
                if b == usize::MAX {
                    // Implicit zero-init: conforms iff 0 < bound.
                    if bound == 0 {
                        return false;
                    }
                    continue;
                }
                if !guarded_cell_load_assignment(program, b, op, root, offset, counter, bound) {
                    return false;
                }
            }
        }
    }
    true
}

/// Resolve `src` at `op_index` to a constant defined earlier in the block
/// (either an inline const operand or an `I32Const` producer).
fn same_block_const(
    program: &SsaProgram,
    block: &crate::vm::middle::ssa_ir::ir::SsaBlock,
    op_index: usize,
    src: SsaOperand,
) -> Option<u64> {
    match src.decode() {
        DecodedOperand::Const(index) => program.const_pool.get(index as usize).copied(),
        DecodedOperand::Value(value) => {
            for inst in block.ops[..op_index].iter().rev() {
                if inst.result == value {
                    let pool_idx = inst.op.as_primitive_idx()?;
                    return match program.primitive_pool[pool_idx as usize] {
                        PrimitiveOpKind::I32Const { value } => Some(value as u32 as u64),
                        _ => None,
                    };
                }
            }
            None
        }
        DecodedOperand::None => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn guarded_cell_load_assignment(
    program: &SsaProgram,
    site_block: usize,
    site_op: usize,
    root: FrameSlot,
    offset: u32,
    counter: FrameSlot,
    bound: u64,
) -> bool {
    let block = &program.blocks[site_block];
    if site_op == 0 {
        return false;
    }
    // The set's source must be the immediately preceding cell load.
    let load = &block.ops[site_op - 1];
    let Some(pool_idx) = load.op.as_primitive_idx() else {
        return false;
    };
    let is_cell_load = matches!(
        program.primitive_pool[pool_idx as usize],
        PrimitiveOpKind::I32Load { offset: o, memidx: 0 } if o == offset
    );
    let site_src = block.ops[site_op].args[0];
    if !is_cell_load || site_src.as_value() != Some(load.result) {
        return false;
    }
    // The load's base must be a get of `root` earlier in the block with no
    // intervening set of root (root stability is checked globally by
    // `local_root_delta`, but verify the base here too).
    let base_ok = load.args[0].as_value().is_some_and(|base| {
        block.ops[..site_op - 1]
            .iter()
            .rev()
            .find(|inst| inst.result == base)
            .is_some_and(|inst| {
                matches!(inst.op, SsaOp::LOCAL_GET_SLOT | SsaOp::LOCAL_GET_CACHE)
                    && FrameSlot(inst.meta) == root
            })
    });
    if !base_ok {
        return false;
    }
    // Guard: terminator branches on `counter != bound`.
    let Some(guard) = load_guard(program, block, site_op - 1) else {
        return false;
    };
    if guard.counter != counter || guard.bound != bound {
        return false;
    }
    // The `== bound` successor (branch else edge: Ne false) must reassign the
    // counter to a constant < bound before any read of the counter.
    let SsaTerminator::Branch { else_edge, .. } = &block.terminator else {
        return false;
    };
    let Some(else_block) = program.blocks.get(else_edge.target.as_usize()) else {
        return false;
    };
    for (op_index, inst) in else_block.ops.iter().enumerate() {
        match inst.op {
            SsaOp::LOCAL_GET_SLOT | SsaOp::LOCAL_GET_CACHE => {
                if FrameSlot(inst.meta) == counter {
                    return false;
                }
            }
            SsaOp::LOCAL_SET_SLOT | SsaOp::LOCAL_SET_CACHE => {
                if FrameSlot(inst.meta) == counter {
                    return same_block_const(program, else_block, op_index, inst.args[0])
                        .is_some_and(|value| value < bound);
                }
            }
            _ => {}
        }
    }
    false
}

/// The load result must have exactly one consumer, a `local.set` in the same
/// block, and must not escape into calls, extra args, or the terminator.
/// Returns the set's op index.
fn single_local_set_use(
    program: &SsaProgram,
    block: &crate::vm::middle::ssa_ir::ir::SsaBlock,
    load_index: usize,
    loaded: SsaValue,
) -> Option<usize> {
    let mut set_index: Option<usize> = None;
    for (offset, inst) in block.ops[load_index + 1..].iter().enumerate() {
        let abs_index = load_index + 1 + offset;
        let is_local_set = matches!(inst.op, SsaOp::LOCAL_SET_SLOT | SsaOp::LOCAL_SET_CACHE);
        for arg in inst.args.iter() {
            if arg.as_value() == Some(loaded) {
                if !is_local_set || set_index.is_some() {
                    return None;
                }
                set_index = Some(abs_index);
            }
        }
        if inst.op == SsaOp::CALL {
            if let Some(call) = program.call_ops.get(inst.meta as usize) {
                if call_op_uses_value(call, loaded) {
                    return None;
                }
            }
        }
    }
    if block
        .extra_args
        .iter()
        .any(|extra| extra.as_value() == Some(loaded))
    {
        return None;
    }
    if terminator_uses_value(&block.terminator, loaded) {
        return None;
    }
    if nonbranch_terminator_uses_value(&block.terminator, loaded) {
        return None;
    }
    set_index
}

/// The stored operand's recompute recipe: the stored value must be defined in
/// this block before the store as `I32Add(local.get S, const)`, with no
/// reassignment of `S` between that get and the load (the facts map already
/// guarantees the fact was not invalidated when the store opened the window;
/// re-check the window range explicitly).
fn increment_recipe(
    program: &SsaProgram,
    block: &crate::vm::middle::ssa_ir::ir::SsaBlock,
    facts: &BTreeMap<SsaValue, ValueFact>,
    store_index: usize,
    load_index: usize,
    stored: SsaOperand,
) -> Option<(FrameSlot, SsaOp, SsaOp, SsaOperand)> {
    let stored_value = stored.as_value()?;
    // Find the defining Add before the store.
    let (add_index, add_inst) = block.ops[..store_index]
        .iter()
        .enumerate()
        .rev()
        .find(|(_, inst)| inst.result == stored_value)?;
    let pool_idx = add_inst.op.as_primitive_idx()?;
    if !matches!(
        program.primitive_pool[pool_idx as usize],
        PrimitiveOpKind::I32Add
    ) {
        return None;
    }
    // One operand is a local.get result, the other a constant.
    let mut slot_and_op: Option<(FrameSlot, SsaOp)> = None;
    let mut const_operand: Option<SsaOperand> = None;
    for arg in add_inst.args.iter() {
        match arg.decode() {
            DecodedOperand::Value(value) => {
                let get = block.ops[..add_index]
                    .iter()
                    .rev()
                    .find(|inst| inst.result == value)?;
                if !matches!(get.op, SsaOp::LOCAL_GET_SLOT | SsaOp::LOCAL_GET_CACHE) {
                    return None;
                }
                if slot_and_op.is_some() {
                    return None;
                }
                slot_and_op = Some((FrameSlot(get.meta), get.op));
            }
            DecodedOperand::Const(_) => {
                if const_operand.is_some() {
                    return None;
                }
                const_operand = Some(*arg);
            }
            DecodedOperand::None => {}
        }
    }
    let (slot, get_op) = slot_and_op?;
    let inc_const = const_operand?;
    let add_op = add_inst.op;
    // `slot` must not be reassigned between the get and the load.
    let _ = facts;
    for inst in &block.ops[add_index..load_index] {
        if matches!(inst.op, SsaOp::LOCAL_SET_SLOT | SsaOp::LOCAL_SET_CACHE)
            && FrameSlot(inst.meta) == slot
        {
            return None;
        }
    }
    Some((slot, get_op, add_op, inc_const))
}

/// Block successors of a terminator (edge targets only)./// Block successors of a terminator (edge targets only).
fn successors(terminator: &SsaTerminator) -> collections::Vec<usize> {
    match terminator {
        SsaTerminator::Goto(edge) => collections::vec![edge.target.as_usize()],
        SsaTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => collections::vec![then_edge.target.as_usize(), else_edge.target.as_usize()],
        SsaTerminator::BrTable { entries, .. } => {
            entries.iter().map(|e| e.target.as_usize()).collect()
        }
        _ => collections::Vec::new(),
    }
}

/// How a def site assigns the slot, as far as the matcher can classify it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DefFact {
    Const(u64),
    RootPlusConst(FrameSlot, u64),
    /// Anything else; poisons whichever property is being checked.
    Other,
}

struct SlotDefs {
    /// Def sites of the slot, in (block, op) order. Site 0 is the implicit
    /// wasm zero-init at function entry.
    sites: collections::Vec<(usize, usize, DefFact)>,
    /// Per block: bitmask (by site index) of defs reaching BLOCK ENTRY.
    reach_in: collections::Vec<u64>,
}

/// Compute reaching definitions of `slot` (classified per site). Returns
/// `None` when the slot has more than 63 def sites (bitmask limit) — callers
/// bail conservatively.
fn slot_reaching_defs(program: &SsaProgram, slot: FrameSlot) -> Option<SlotDefs> {
    let mut sites: collections::Vec<(usize, usize, DefFact)> = collections::Vec::new();
    // Implicit zero-init reaches the entry.
    sites.push((usize::MAX, 0, DefFact::Const(0)));
    for (block_index, block) in program.blocks.iter().enumerate() {
        let mut facts: BTreeMap<SsaValue, ValueFact> = BTreeMap::new();
        for (op_index, inst) in block.ops.iter().enumerate() {
            match inst.op {
                SsaOp::LOCAL_GET_SLOT | SsaOp::LOCAL_GET_CACHE => {
                    facts.insert(inst.result, ValueFact::GetLocal(FrameSlot(inst.meta)));
                }
                SsaOp::LOCAL_SET_SLOT | SsaOp::LOCAL_SET_CACHE => {
                    let set_slot = FrameSlot(inst.meta);
                    if set_slot == slot {
                        let fact = match operand_fact(&program.const_pool, &facts, inst.args[0]) {
                            ValueFact::Const(value) => DefFact::Const(value),
                            ValueFact::LocalPlusConst(base, delta) => {
                                DefFact::RootPlusConst(base, delta)
                            }
                            _ => match same_block_const(program, block, op_index, inst.args[0]) {
                                Some(value) => DefFact::Const(value),
                                None => DefFact::Other,
                            },
                        };
                        sites.push((block_index, op_index, fact));
                    }
                    facts = facts
                        .into_iter()
                        .filter(|(_, fact)| !fact_mentions(fact, set_slot))
                        .collect();
                }
                _ => {
                    if let Some(pool_idx) = inst.op.as_primitive_idx() {
                        match program.primitive_pool[pool_idx as usize].clone() {
                            PrimitiveOpKind::I32Add => {
                                facts.insert(
                                    inst.result,
                                    add_fact(&program.const_pool, &facts, inst),
                                );
                            }
                            PrimitiveOpKind::I32Const { value } => {
                                facts.insert(inst.result, ValueFact::Const(value as u32 as u64));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    if sites.len() > 63 {
        return None;
    }

    // Per-block gen/kill: the LAST def in a block generates; any def kills.
    let block_count = program.blocks.len();
    let mut gen: collections::Vec<u64> = collections::vec![0; block_count];
    let mut kills: collections::Vec<bool> = collections::vec![false; block_count];
    for (site_index, &(block_index, _, _)) in sites.iter().enumerate() {
        if block_index == usize::MAX {
            continue;
        }
        kills[block_index] = true;
        gen[block_index] = 1u64 << site_index; // later sites overwrite: sites are in op order per block
    }

    let entry = program.entry.as_usize();
    let mut reach_in: collections::Vec<u64> = collections::vec![0; block_count];
    if entry < block_count {
        reach_in[entry] = 1; // implicit zero-init (site 0)
    }
    // Fixpoint.
    let mut changed = true;
    while changed {
        changed = false;
        for (block_index, block) in program.blocks.iter().enumerate() {
            let out = if kills[block_index] {
                gen[block_index]
            } else {
                reach_in[block_index]
            };
            for succ in successors(&block.terminator) {
                if succ < block_count && (reach_in[succ] | out) != reach_in[succ] {
                    reach_in[succ] |= out;
                    changed = true;
                }
            }
        }
    }
    Some(SlotDefs { sites, reach_in })
}

fn apply_rewrite(program: &mut SsaProgram, rewrite: &Rewrite) {
    // Allocate two fresh values: the re-read of the increment slot and the
    // recomputed sum. The counter is i32 by construction.
    let get_value = SsaValue(program.value_types.len() as u32);
    program.value_types.push(crate::value_type::ValueType::I32);
    program.value_sink_local.push(None);
    let add_value = SsaValue(program.value_types.len() as u32);
    program.value_types.push(crate::value_type::ValueType::I32);
    program.value_sink_local.push(None);

    let block = &mut program.blocks[rewrite.block];
    // Replace the load with the fresh get, insert the Add after it, and
    // repoint the consuming set at the recomputed value.
    block.ops[rewrite.load_index] = SsaInst {
        op: rewrite.inc_get_op,
        meta: rewrite.inc_slot.0,
        result: get_value,
        args: [SsaOperand::NONE, SsaOperand::NONE],
    };
    block.ops.insert(
        rewrite.load_index + 1,
        SsaInst {
            op: rewrite.inc_add_op,
            meta: 0,
            result: add_value,
            args: [SsaOperand::value(get_value), rewrite.inc_const],
        },
    );
    let set = block.ops[rewrite.load_index + 2..]
        .iter_mut()
        .find(|inst| inst.args[0].as_value() == Some(rewrite.loaded))
        .expect("collection guaranteed a single local.set consumer");
    debug_assert!(matches!(
        set.op,
        SsaOp::LOCAL_SET_SLOT | SsaOp::LOCAL_SET_CACHE
    ));
    set.args[0] = SsaOperand::value(add_value);
}

fn call_op_uses_value(call: &crate::vm::middle::ssa_ir::ir::SsaCallOp, value: SsaValue) -> bool {
    use crate::vm::middle::ssa_ir::ir::{SsaCallOp, SsaCallOperandLoc};
    let args_use = |args: &crate::vm::middle::ssa_ir::ir::SsaCallArgs| {
        args.live_suffix.iter().any(|arg| arg.value == value)
    };
    let loc_uses = |loc: &SsaCallOperandLoc| matches!(loc, SsaCallOperandLoc::Live { value: v, .. } if *v == value);
    match call {
        SsaCallOp::CallDirect { args, .. } => args_use(args),
        SsaCallOp::CallIndirect { args, index, .. } => args_use(args) || loc_uses(index),
        SsaCallOp::CallRef {
            args, callee_ref, ..
        } => args_use(args) || loc_uses(callee_ref),
    }
}

fn nonbranch_terminator_uses_value(terminator: &SsaTerminator, value: SsaValue) -> bool {
    use crate::vm::middle::ssa_ir::ir::{SsaCallOperandLoc, SsaScalarResultLoc};
    let loc_uses = |loc: &SsaCallOperandLoc| matches!(loc, SsaCallOperandLoc::Live { value: v, .. } if *v == value);
    let args_use = |args: &crate::vm::middle::ssa_ir::ir::SsaCallArgs| {
        args.live_suffix.iter().any(|arg| arg.value == value)
    };
    match terminator {
        SsaTerminator::ReturnScalar { result, .. } => {
            matches!(result, SsaScalarResultLoc::Live { value: v, .. } if *v == value)
        }
        SsaTerminator::TailCallDirect { args, .. } => args_use(args),
        SsaTerminator::TailCallIndirect { args, index, .. } => args_use(args) || loc_uses(index),
        SsaTerminator::TailCallRef {
            args, callee_ref, ..
        } => args_use(args) || loc_uses(callee_ref),
        _ => false,
    }
}

fn terminator_uses_value(terminator: &SsaTerminator, value: SsaValue) -> bool {
    let edge_uses = |edge: &crate::vm::middle::ssa_ir::ir::SsaEdge| {
        edge.bindings.iter().any(|binding| binding.value == value)
    };
    match terminator {
        SsaTerminator::Goto(edge) => edge_uses(edge),
        SsaTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => *cond == value || edge_uses(then_edge) || edge_uses(else_edge),
        SsaTerminator::BrTable { index, entries } => {
            *index == value || entries.iter().any(edge_uses)
        }
        _ => false,
    }
}
