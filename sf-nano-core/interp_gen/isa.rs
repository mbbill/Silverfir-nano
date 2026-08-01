//! The interface a target backend implements for the handler generator.
//!
//! The generic driver owns everything that must be identical across
//! targets: which `(op, operand-class)` combinations exist, the order they
//! are emitted in, and the packed slot each one is recorded at. A backend
//! owns everything that touches registers or instructions.
//!
//! The split is deliberate. The linker classifies a cell with the same
//! `layout` code the driver enumerates with, so a backend cannot
//! accidentally register a handler under a key the linker will never look
//! up, nor leave a key the linker does look up pointing at a handler that
//! reads its operands from somewhere else.

use super::asm::Asm;
use super::instr::Op;
use super::layout::{Cls, DstCls, PairDstCls};

/// Build-time-only views of [`PairDstCls`]: which destination class the
/// first and second results of a pair op land in. Only the generator asks
/// this question — the runtime library indexes the already-generated
/// handler table and never splits a pair class again.
pub trait PairDstSplit {
    fn first(self) -> Option<DstCls>;
    fn second(self) -> Option<DstCls>;
}

impl PairDstSplit for PairDstCls {
    fn first(self) -> Option<DstCls> {
        match self {
            PairDstCls::FirstL0 | PairDstCls::L0L1 => Some(DstCls::L0),
            PairDstCls::FirstL1 | PairDstCls::L1L0 => Some(DstCls::L1),
            PairDstCls::None | PairDstCls::SecondL0 | PairDstCls::SecondL1 => None,
        }
    }

    fn second(self) -> Option<DstCls> {
        match self {
            PairDstCls::SecondL0 | PairDstCls::L1L0 => Some(DstCls::L0),
            PairDstCls::SecondL1 | PairDstCls::L0L1 => Some(DstCls::L1),
            PairDstCls::None | PairDstCls::FirstL0 | PairDstCls::FirstL1 => None,
        }
    }
}

/// The full operand-class set. A backend narrows it in [`Caps`] when
/// emitted-code size matters more than the last few percent of residency.
pub const CLASSES: [Cls; 5] = [Cls::Slot, Cls::Const, Cls::Acc, Cls::L0, Cls::L1];
pub const DSTS: [DstCls; 4] = [DstCls::Mem, DstCls::Acc, DstCls::L0, DstCls::L1];
/// The reduced set for register- or flash-constrained targets: no second
/// pinned local. Takes a three-operand op from 100 variants to 48.
pub const CLASSES_NO_L1: [Cls; 4] = [Cls::Slot, Cls::Const, Cls::Acc, Cls::L0];
pub const DSTS_NO_L1: [DstCls; 3] = [DstCls::Mem, DstCls::Acc, DstCls::L0];

impl Cls {
    /// Whether the value arrives in a chain-invariant register rather than
    /// being loaded out of the cell.
    pub fn is_reg(self) -> bool {
        matches!(self, Cls::Acc | Cls::L0 | Cls::L1)
    }
}

/// Label names of the shared, non-handler blocks. The driver owns the
/// names; a backend defines the code at them in `emit_prelude` and
/// branches to them from handler bodies.
pub struct Stubs {
    /// Common exit epilogue: reason already staged by the caller.
    pub exit_common: String,
    /// Bail to the single-instruction executor.
    pub slow: String,
    /// A `Return` that popped a sentinel record: back to the driver.
    pub return_exit: String,
    /// Trap: out of bounds memory access.
    pub trap_oob: String,
    /// Trap: call stack exhausted.
    pub trap_exhaust: String,
    /// `extern "C" fn(*mut EnterState)` trampoline.
    pub entry: String,
    /// Native `Call` handler (wired by the cross-function fixup pass).
    pub call: String,
    /// Native `CallIndirect` handler (likewise).
    pub call_indirect: String,
    /// Shared activation-entry tail both call flavours branch into.
    pub call_core: String,
}

/// What a target backend can do. Anything switched off here simply has no
/// handler: the cell links to the slow stub and the shared executor runs
/// it, so a reduced backend is slower, never wrong.
pub struct Caps {
    /// Operand residency classes this backend emits variants for. Trimming
    /// this is the size lever for flash-constrained targets: dropping `L1`
    /// takes a three-operand op from 100 variants to 48.
    pub classes: &'static [Cls],
    pub dsts: &'static [DstCls],
    /// Native pointer width in bytes. 4 means a 64-bit wasm value needs a
    /// register pair.
    pub ptr_bytes: u32,
    /// Whether the second pinned local exists. Must agree with `classes`.
    pub has_l1: bool,
    /// Whether pinned locals and the accumulator have float twins.
    pub has_float_regs: bool,
    /// Whether an f32-domain slot may be float-pinned. False on RISC-V:
    /// a single held in an F register must be NaN-boxed, and the pinned
    /// register reloads with a full-width load from a slot that holds the
    /// value zero-extended, which is not a valid boxed single. f64 slots
    /// are unaffected, so only the 32-bit case is excluded.
    pub float_pin_f32: bool,
    /// Whether the backend implements the native call protocol (its own
    /// `Call`/`CallIndirect` handlers running on the overlapped value
    /// stack). When false the cross-function fixup is skipped, both call
    /// flavours link to the slow stub, and the driver drives every
    /// activation — correct, but two chain crossings per call.
    pub native_calls: bool,
    /// Whether the engine blob starts on its own page. Wide cores predict
    /// the dispatch chain's indirect branches through virtually indexed
    /// structures whose granule exceeds a cache line, so the same handler
    /// bytes at a different page offset measure differently per binary;
    /// pinning the base makes every handler's virtual address a constant
    /// of the blob's content. Costs up to one page of padding, so the
    /// flash-constrained targets leave it off.
    pub page_aligned_blob: bool,
}

/// One handler the driver is asking about.
#[derive(Clone, Copy)]
pub struct Variant {
    pub op: Op,
    pub a: Cls,
    pub b: Cls,
    pub d: DstCls,
    /// Ordered destination residency for `MovPair`; `None` for other ops.
    pub pair_d: PairDstCls,
    /// Address-fused bank (memory ops only).
    pub fused: bool,
    /// Whether this handler bumps the in-chain dispatch counter.
    pub counted: bool,
}

pub trait Isa {
    fn caps(&self) -> Caps;

    /// Entry trampoline, exit stubs, and the two call handlers. Called
    /// once, before any handler, so those blocks keep the low addresses.
    fn emit_prelude(&mut self, a: &mut Asm, st: &Stubs);

    /// Whether this backend has a native form of `v`. Returning false
    /// leaves the slot empty and the cell takes the slow path.
    fn wants(&self, v: &Variant) -> bool;

    /// Emit the handler body. The driver has already placed the entry
    /// label; the body must end in a dispatch tail on every path.
    fn emit_handler(&mut self, a: &mut Asm, st: &Stubs, v: &Variant);
}
