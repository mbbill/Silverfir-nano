//! Prepared backend-facing SSA-IR.
//!
//! This is the frontend/native handoff for the engine's prepared pipeline:
//! - canonical locals and deep stack values live in frame slots
//! - only a bounded set of transient values stays live as SSA values
//! - explicit slot traffic publishes and reloads transient values through
//!   operand slots so the backend never needs general register allocation
//!
//! # Layout
//!
//! `SsaInst` is a flat 16-byte struct (not an enum) so that `SsaBlock.ops`
//! stays cache-dense. Variant dispatch is encoded in `SsaInst.op` (a `SsaOp`):
//! opcodes `0..=9` are reserved for non-primitive variants (Fill, Spill,
//! Local*, Call); opcodes `10..=u16::MAX` are primitive-pool indices shifted
//! by `PRIMITIVE_BASE`. Primitive op payloads (constants, memargs, etc.) stay
//! inside `PrimitiveOpKind` and are kept once per unique op in
//! `SsaProgram.primitive_pool`.
//!
//! Two pools live on the program:
//! - `const_pool: Vec<u64>` backs `SsaOperand::Const` (the 4-byte packed
//!   operand carries only a const-pool index)
//! - `primitive_pool: Vec<PrimitiveOpKind>` is dedup-inserted; `SsaInst.op`
//!   holds its index
//! - `call_ops: Vec<SsaCallOp>` backs the `Call` variant; `SsaInst.meta` is
//!   the index
//!
//! Three-arg primitives (Select + memory/table bulk ops) store their third
//! operand inside `SsaBlock.extra_args`; `SsaInst.meta` holds the index.
//!
//! Use [`SsaInst::view`] or [`SsaBlock::view`] to decode an instruction into
//! a pattern-matchable [`SsaInstView`] enum.

use crate::collections;

use crate::error::WasmError;
use crate::value_type::ValueType;

use super::target::SsaTarget;
use crate::vm::middle::frame::{FrameSlot, FrameSpan};
use crate::vm::wasm::primitive_op::{self, PrimitiveOpKind};

/// One SSA value in prepared SSA-IR. `SsaValue::NONE` is a sentinel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct SsaValue(pub u32);

impl SsaValue {
    /// Sentinel for "no value" (used for missing results or unused fields).
    pub(crate) const NONE: Self = Self(u32::MAX);

    #[inline]
    pub(crate) fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    #[inline]
    pub(crate) fn is_some(self) -> bool {
        self.0 != u32::MAX
    }
}

/// Packed 4-byte operand.
///
/// Encoding:
/// - `u32::MAX` — `NONE` sentinel (unused arg slot)
/// - bit 31 = 0 — `Value` operand; low 31 bits are the `SsaValue` id
/// - bit 31 = 1 — `Const` operand; low 31 bits are an index into
///   `SsaProgram.const_pool`
///
/// Use [`SsaOperand::value`] / [`SsaOperand::const_idx`] to construct and
/// [`SsaOperand::decode`] / [`SsaOperand::as_value`] to read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub(crate) struct SsaOperand(u32);

/// Decoded view of [`SsaOperand`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodedOperand {
    None,
    Value(SsaValue),
    /// Index into `SsaProgram.const_pool`.
    Const(u32),
}

impl SsaOperand {
    const CONST_TAG: u32 = 0x8000_0000;
    const INDEX_MASK: u32 = 0x7FFF_FFFF;

    /// Sentinel for "no operand" (unused arg slot).
    pub(crate) const NONE: Self = Self(u32::MAX);

    #[inline]
    pub(crate) fn value(v: SsaValue) -> Self {
        debug_assert!(v.0 < Self::CONST_TAG, "SsaValue out of range for operand");
        debug_assert!(!v.is_none(), "cannot pack NONE SsaValue as operand");
        Self(v.0)
    }

    #[inline]
    pub(crate) fn const_idx(idx: u32) -> Self {
        debug_assert!(
            idx < Self::CONST_TAG,
            "const pool index out of range for operand"
        );
        Self(Self::CONST_TAG | idx)
    }

    #[inline]
    pub(crate) fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    #[inline]
    pub(crate) fn as_value(self) -> Option<SsaValue> {
        if self.is_none() || (self.0 & Self::CONST_TAG) != 0 {
            None
        } else {
            Some(SsaValue(self.0 & Self::INDEX_MASK))
        }
    }

    /// Convenience: extract the inner `SsaValue`, panicking otherwise.
    #[inline]
    pub(crate) fn unwrap_value(self) -> SsaValue {
        self.as_value()
            .expect("SsaOperand::unwrap_value called on non-Value operand")
    }

    #[inline]
    pub(crate) fn decode(self) -> DecodedOperand {
        if self.is_none() {
            DecodedOperand::None
        } else if (self.0 & Self::CONST_TAG) != 0 {
            DecodedOperand::Const(self.0 & Self::INDEX_MASK)
        } else {
            DecodedOperand::Value(SsaValue(self.0))
        }
    }
}

/// Instruction opcode / variant discriminator.
///
/// Values `0..=9` are reserved for non-primitive variants. Values starting
/// at [`SsaOp::PRIMITIVE_BASE`] (= 10) are primitive pool indices shifted by
/// `PRIMITIVE_BASE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub(crate) struct SsaOp(u16);

impl SsaOp {
    pub(crate) const FILL: Self = Self(0);
    pub(crate) const SPILL: Self = Self(1);
    pub(crate) const LOCAL_GET_SLOT: Self = Self(2);
    pub(crate) const LOCAL_GET_CACHE: Self = Self(3);
    pub(crate) const LOCAL_SET_SLOT: Self = Self(4);
    pub(crate) const LOCAL_SET_CACHE: Self = Self(5);
    pub(crate) const LOCAL_ENSURE_CACHE: Self = Self(6);
    pub(crate) const LOCAL_RESERVE_CACHE: Self = Self(7);
    pub(crate) const LOCAL_DROP_CACHE: Self = Self(8);
    pub(crate) const CALL: Self = Self(9);

    /// First opcode value used for primitive pool indices.
    pub(crate) const PRIMITIVE_BASE: u16 = 10;

    /// Maximum number of primitive-pool entries a single function can have.
    /// `SsaOp` encodes them as `PRIMITIVE_BASE + pool_idx` in a u16, so the
    /// largest encodable pool index is `u16::MAX - PRIMITIVE_BASE`, giving
    /// `u16::MAX - PRIMITIVE_BASE + 1` entries.
    pub(crate) const MAX_PRIMITIVE_POOL: usize =
        (u16::MAX as usize) - (Self::PRIMITIVE_BASE as usize) + 1;

    /// Build a primitive opcode for pool index `pool_idx`. Callers must
    /// guarantee `pool_idx < MAX_PRIMITIVE_POOL`; interning code in
    /// `ProgramBuilder` / `SsaProgram::intern_primitive` returns an error
    /// rather than overflowing.
    #[inline]
    pub(crate) fn primitive(pool_idx: u32) -> Self {
        let code = Self::PRIMITIVE_BASE as u32 + pool_idx;
        debug_assert!(code <= u16::MAX as u32, "primitive op pool overflow");
        Self(code as u16)
    }

    #[inline]
    pub(crate) fn as_primitive_idx(self) -> Option<u32> {
        if self.0 >= Self::PRIMITIVE_BASE {
            Some(self.0 as u32 - Self::PRIMITIVE_BASE as u32)
        } else {
            None
        }
    }

    #[inline]
    pub(crate) fn is_primitive(self) -> bool {
        self.0 >= Self::PRIMITIVE_BASE
    }
}

/// Stable facts about a local slot, carried from preparation to the backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LocalSlotInfo {
    pub is_param: bool,
    pub reads_before_write: bool,
}

/// One SSA operation inside a block body.
///
/// Flat 16-byte layout. See the module doc for the opcode encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SsaInst {
    pub op: SsaOp,
    /// Secondary scalar: slot index for Fill/Spill/Local*, call-op index for
    /// Call, extra-arg index for 3-arg primitives, unused (0) otherwise.
    pub meta: u16,
    /// Destination value; `SsaValue::NONE` when the op has no result.
    pub result: SsaValue,
    /// Inline operand slots; `SsaOperand::NONE` for unused positions.
    pub args: [SsaOperand; 2],
}

impl SsaInst {
    #[inline]
    pub(crate) fn fill(slot: FrameSlot, dst: SsaValue) -> Self {
        Self {
            op: SsaOp::FILL,
            meta: slot.0,
            result: dst,
            args: [SsaOperand::NONE, SsaOperand::NONE],
        }
    }

    #[inline]
    pub(crate) fn spill(slot: FrameSlot, src: SsaValue) -> Self {
        Self {
            op: SsaOp::SPILL,
            meta: slot.0,
            result: SsaValue::NONE,
            args: [SsaOperand::value(src), SsaOperand::NONE],
        }
    }

    #[inline]
    pub(crate) fn local_get_slot(slot: FrameSlot, dst: SsaValue) -> Self {
        Self {
            op: SsaOp::LOCAL_GET_SLOT,
            meta: slot.0,
            result: dst,
            args: [SsaOperand::NONE, SsaOperand::NONE],
        }
    }

    #[inline]
    pub(crate) fn local_get_cache(slot: FrameSlot, dst: SsaValue) -> Self {
        Self {
            op: SsaOp::LOCAL_GET_CACHE,
            meta: slot.0,
            result: dst,
            args: [SsaOperand::NONE, SsaOperand::NONE],
        }
    }

    #[inline]
    pub(crate) fn local_set_slot(slot: FrameSlot, src: SsaValue) -> Self {
        Self {
            op: SsaOp::LOCAL_SET_SLOT,
            meta: slot.0,
            result: SsaValue::NONE,
            args: [SsaOperand::value(src), SsaOperand::NONE],
        }
    }

    #[inline]
    pub(crate) fn local_set_cache(slot: FrameSlot, src: SsaValue) -> Self {
        Self {
            op: SsaOp::LOCAL_SET_CACHE,
            meta: slot.0,
            result: SsaValue::NONE,
            args: [SsaOperand::value(src), SsaOperand::NONE],
        }
    }

    #[inline]
    pub(crate) fn local_ensure_cache(slot: FrameSlot) -> Self {
        Self {
            op: SsaOp::LOCAL_ENSURE_CACHE,
            meta: slot.0,
            result: SsaValue::NONE,
            args: [SsaOperand::NONE, SsaOperand::NONE],
        }
    }

    #[inline]
    pub(crate) fn local_reserve_cache(slot: FrameSlot) -> Self {
        Self {
            op: SsaOp::LOCAL_RESERVE_CACHE,
            meta: slot.0,
            result: SsaValue::NONE,
            args: [SsaOperand::NONE, SsaOperand::NONE],
        }
    }

    #[inline]
    pub(crate) fn local_drop_cache(slot: FrameSlot) -> Self {
        Self {
            op: SsaOp::LOCAL_DROP_CACHE,
            meta: slot.0,
            result: SsaValue::NONE,
            args: [SsaOperand::NONE, SsaOperand::NONE],
        }
    }

    /// Build a Call instruction. Callers must guarantee
    /// `call_idx <= u16::MAX` (the `meta` slot is u16); `ProgramBuilder` /
    /// `SsaProgram::push_call_op` returns an error before reaching this.
    #[inline]
    pub(crate) fn call(call_idx: u32) -> Self {
        debug_assert!(
            call_idx <= u16::MAX as u32,
            "call_ops index exceeds SsaInst.meta (u16) capacity"
        );
        Self {
            op: SsaOp::CALL,
            meta: call_idx as u16,
            result: SsaValue::NONE,
            args: [SsaOperand::NONE, SsaOperand::NONE],
        }
    }

    /// Build a primitive op instruction.
    ///
    /// `args` must have length 0-3; the third arg (if present) must already
    /// have been pushed into the enclosing `SsaBlock.extra_args` and its
    /// index passed as `extra_args_idx`.
    #[inline]
    pub(crate) fn primitive(
        pool_idx: u32,
        result: SsaValue,
        args: [SsaOperand; 2],
        extra_args_idx: u16,
    ) -> Self {
        Self {
            op: SsaOp::primitive(pool_idx),
            meta: extra_args_idx,
            result,
            args,
        }
    }

    /// Decode into a pattern-matchable view. Primitive ops and call ops need
    /// `program` for their pool lookups; non-primitive variants only need
    /// the enclosing block for extra_args.
    #[inline]
    pub(crate) fn view<'a>(&self, program: &'a SsaProgram, block: &'a SsaBlock) -> SsaInstView<'a> {
        decode_view(*self, program, block)
    }
}

/// Type-safe pattern-matchable view over one SsaInst.
///
/// Each variant carries the full decoded payload for a given op kind so
/// callers can destructure directly. A release-build consumer that only
/// needs `Value` / `Call` can fall through non-primitive variants with `_`;
/// debug-mode validation and tests destructure everything. `dead_code` is
/// allowed because validation (which reads most fields) is gated behind
/// `#[cfg(any(debug_assertions, test))]`, leaving some fields produce-only
/// in release.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) enum SsaInstView<'a> {
    Fill {
        slot: FrameSlot,
        dst: SsaValue,
    },
    Spill {
        slot: FrameSlot,
        src: SsaValue,
    },
    LocalGetSlot {
        slot: FrameSlot,
        dst: SsaValue,
    },
    LocalGetCache {
        slot: FrameSlot,
        dst: SsaValue,
    },
    LocalSetSlot {
        slot: FrameSlot,
        src: SsaValue,
    },
    LocalSetCache {
        slot: FrameSlot,
        src: SsaValue,
    },
    LocalEnsureCache {
        slot: FrameSlot,
    },
    LocalReserveCache {
        slot: FrameSlot,
    },
    LocalDropCache {
        slot: FrameSlot,
    },
    Call(&'a SsaCallOp),
    Value {
        op: &'a PrimitiveOpKind,
        result: SsaValue,
        args: SsaArgs<'a>,
    },
}

/// Argument list view for a Value instruction: up to 2 inline slots plus an
/// optional third arg drawn from the enclosing block's `extra_args`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SsaArgs<'a> {
    inline: [SsaOperand; 2],
    extra: Option<SsaOperand>,
    _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a> SsaArgs<'a> {
    #[inline]
    fn from_parts(inline: [SsaOperand; 2], extra: Option<SsaOperand>) -> Self {
        Self {
            inline,
            extra,
            _marker: core::marker::PhantomData,
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        let mut n = 0;
        for a in &self.inline {
            if !a.is_none() {
                n += 1;
            } else {
                break;
            }
        }
        if self.extra.is_some() {
            n += 1;
        }
        n
    }

    #[inline]
    pub(crate) fn get(&self, idx: usize) -> Option<SsaOperand> {
        match idx {
            0 | 1 => {
                let a = self.inline[idx];
                if a.is_none() {
                    None
                } else {
                    Some(a)
                }
            }
            2 => self.extra,
            _ => None,
        }
    }

    /// Iterator over all present operands in order.
    pub(crate) fn iter(&self) -> SsaArgIter<'_> {
        SsaArgIter {
            args: *self,
            index: 0,
        }
    }

    /// Collect operands into a Vec — convenience for rare non-hot paths.
    pub(crate) fn to_vec(&self) -> collections::Vec<SsaOperand> {
        let mut v = collections::Vec::with_capacity(self.len());
        for a in self.iter() {
            v.push(a);
        }
        v
    }
}

pub(crate) struct SsaArgIter<'a> {
    args: SsaArgs<'a>,
    index: usize,
}

impl<'a> Iterator for SsaArgIter<'a> {
    type Item = SsaOperand;

    fn next(&mut self) -> Option<SsaOperand> {
        let v = self.args.get(self.index)?;
        self.index += 1;
        Some(v)
    }
}

fn decode_view<'a>(inst: SsaInst, program: &'a SsaProgram, block: &'a SsaBlock) -> SsaInstView<'a> {
    match inst.op {
        SsaOp::FILL => SsaInstView::Fill {
            slot: FrameSlot(inst.meta),
            dst: inst.result,
        },
        SsaOp::SPILL => SsaInstView::Spill {
            slot: FrameSlot(inst.meta),
            src: inst.args[0]
                .as_value()
                .expect("Spill src must be an SsaValue"),
        },
        SsaOp::LOCAL_GET_SLOT => SsaInstView::LocalGetSlot {
            slot: FrameSlot(inst.meta),
            dst: inst.result,
        },
        SsaOp::LOCAL_GET_CACHE => SsaInstView::LocalGetCache {
            slot: FrameSlot(inst.meta),
            dst: inst.result,
        },
        SsaOp::LOCAL_SET_SLOT => SsaInstView::LocalSetSlot {
            slot: FrameSlot(inst.meta),
            src: inst.args[0]
                .as_value()
                .expect("LocalSetSlot src must be an SsaValue"),
        },
        SsaOp::LOCAL_SET_CACHE => SsaInstView::LocalSetCache {
            slot: FrameSlot(inst.meta),
            src: inst.args[0]
                .as_value()
                .expect("LocalSetCache src must be an SsaValue"),
        },
        SsaOp::LOCAL_ENSURE_CACHE => SsaInstView::LocalEnsureCache {
            slot: FrameSlot(inst.meta),
        },
        SsaOp::LOCAL_RESERVE_CACHE => SsaInstView::LocalReserveCache {
            slot: FrameSlot(inst.meta),
        },
        SsaOp::LOCAL_DROP_CACHE => SsaInstView::LocalDropCache {
            slot: FrameSlot(inst.meta),
        },
        SsaOp::CALL => {
            let call = program
                .call_ops
                .get(inst.meta as usize)
                .expect("Call meta out of range for call_ops");
            SsaInstView::Call(call)
        }
        _ => {
            debug_assert!(inst.op.is_primitive(), "unknown SsaOp");
            let pool_idx = inst.op.as_primitive_idx().expect("primitive op") as usize;
            let kind = program
                .primitive_pool
                .get(pool_idx)
                .expect("primitive op index out of range for primitive_pool");
            let extra = if primitive_op::stack_effect(kind).0 == 3 {
                block.extra_args.get(inst.meta as usize).copied()
            } else {
                None
            };
            SsaInstView::Value {
                op: kind,
                result: inst.result,
                args: SsaArgs::from_parts(inst.args, extra),
            }
        }
    }
}

/// Full prepared SSA-IR program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SsaProgram {
    pub entry: SsaTarget,
    pub blocks: collections::Vec<SsaBlock>,
    pub local_slot_types: collections::Vec<ValueType>,
    pub local_slot_info: collections::Vec<LocalSlotInfo>,
    pub block_entry_cached_slots: collections::Vec<collections::Vec<FrameSlot>>,
    pub block_cfg_origins: collections::Vec<collections::Vec<u32>>,
    pub value_types: collections::Vec<ValueType>,
    pub value_sink_local: collections::Vec<Option<FrameSlot>>,
    /// Backing store for `SsaOperand::Const` — each entry is a 64-bit value.
    pub const_pool: collections::Vec<u64>,
    /// Deduplicated primitive op vocabulary; `SsaOp` carries a pool index.
    pub primitive_pool: collections::Vec<PrimitiveOpKind>,
    /// Backing store for `Call` variants; `SsaInst.meta` is the index.
    pub call_ops: collections::Vec<SsaCallOp>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EntryCacheRequirement {
    Ensure,
    Reserve,
}

impl SsaProgram {
    #[inline]
    pub(crate) fn value_sink(&self, value: SsaValue) -> Option<FrameSlot> {
        self.value_sink_local
            .get(value.0 as usize)
            .copied()
            .flatten()
    }

    /// Append a constant to the const pool and return its operand. No dedup.
    pub(crate) fn intern_const(&mut self, bits: u64) -> SsaOperand {
        let idx = self.const_pool.len() as u32;
        self.const_pool.push(bits);
        SsaOperand::const_idx(idx)
    }

    /// Insert a primitive op into the pool, deduplicating against existing
    /// entries. Returns the pool index, or an error if the pool already
    /// holds the maximum number of entries `SsaOp` can encode
    /// (`SsaOp::MAX_PRIMITIVE_POOL`).
    ///
    /// Production rewrite uses its own `ProgramBuilder`; this variant exists
    /// for later passes (optimize) and test fixtures that mutate an existing
    /// `SsaProgram`.
    pub(crate) fn intern_primitive(&mut self, kind: PrimitiveOpKind) -> Result<u32, WasmError> {
        if let Some(idx) = self.primitive_pool.iter().position(|k| k == &kind) {
            return Ok(idx as u32);
        }
        if self.primitive_pool.len() >= SsaOp::MAX_PRIMITIVE_POOL {
            return Err(WasmError::internal(
                "SSA-IR primitive op pool overflow: function has more distinct primitive ops than a u16 opcode can address",
            ));
        }
        let idx = self.primitive_pool.len() as u32;
        self.primitive_pool.push(kind);
        Ok(idx)
    }

    /// Append a call op; returns its index.
    ///
    /// Test-only: production rewrite pushes through `ProgramBuilder`, which
    /// also propagates overflow errors. Tests that deliberately craft more
    /// than `u16::MAX + 1` calls would be malformed, so `assert!` is
    /// acceptable here.
    #[cfg(test)]
    pub(crate) fn push_call_op(&mut self, call: SsaCallOp) -> u32 {
        assert!(
            self.call_ops.len() <= u16::MAX as usize,
            "call_ops index would exceed SsaInst.meta (u16) capacity"
        );
        let idx = self.call_ops.len() as u32;
        self.call_ops.push(call);
        idx
    }

    #[cfg(test)]
    pub(crate) fn final_block_for_cfg_block(&self, cfg_block: u32) -> Option<SsaTarget> {
        self.block_cfg_origins
            .iter()
            .position(|origins| origins.contains(&cfg_block))
            .map(|index| SsaTarget(index as u32))
    }
}

pub(crate) fn entry_cache_requirement_from_ops(
    ops: &[SsaInst],
    slot: FrameSlot,
) -> Option<EntryCacheRequirement> {
    for inst in ops {
        match inst.op {
            SsaOp::LOCAL_GET_CACHE | SsaOp::LOCAL_ENSURE_CACHE => {
                if FrameSlot(inst.meta) == slot {
                    return Some(EntryCacheRequirement::Ensure);
                }
            }
            SsaOp::LOCAL_SET_CACHE | SsaOp::LOCAL_RESERVE_CACHE => {
                if FrameSlot(inst.meta) == slot {
                    return Some(EntryCacheRequirement::Reserve);
                }
            }
            SsaOp::LOCAL_GET_SLOT | SsaOp::LOCAL_SET_SLOT | SsaOp::LOCAL_DROP_CACHE => {
                if FrameSlot(inst.meta) == slot {
                    return None;
                }
            }
            SsaOp::CALL => return None,
            _ => {}
        }
    }
    None
}

#[inline]
pub(crate) fn entry_cache_requirement(
    ops: &[SsaInst],
    slot: FrameSlot,
    carried_through: bool,
) -> Option<EntryCacheRequirement> {
    entry_cache_requirement_from_ops(ops, slot)
        .or_else(|| carried_through.then_some(EntryCacheRequirement::Ensure))
}

#[inline]
pub(crate) fn block_entry_cache_requirement(
    entry_slots: &[FrameSlot],
    block: &SsaBlock,
    slot: FrameSlot,
) -> Option<EntryCacheRequirement> {
    entry_cache_requirement(&block.ops, slot, entry_slots.contains(&slot))
}

/// One SSA-IR basic block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SsaBlock {
    pub id: SsaTarget,
    pub params: collections::Vec<SsaValue>,
    pub ops: collections::Vec<SsaInst>,
    /// Overflow slot for the third operand of 3-arg primitive ops (Select,
    /// memory/table bulk ops). `SsaInst.meta` indexes this vector.
    pub extra_args: collections::Vec<SsaOperand>,
    pub terminator: SsaTerminator,
}

impl SsaBlock {
    #[inline]
    pub(crate) fn view<'a>(&'a self, inst_idx: usize, program: &'a SsaProgram) -> SsaInstView<'a> {
        decode_view(self.ops[inst_idx], program, self)
    }

    /// Append `extra` to `extra_args` and return its index (cast to u16).
    ///
    /// Convenience for test fixtures — production construction uses the
    /// rewrite-time `ProgramBuilder`.
    #[cfg(test)]
    pub(crate) fn push_extra_arg(&mut self, extra: SsaOperand) -> u16 {
        let idx = self.extra_args.len();
        debug_assert!(idx < u16::MAX as usize, "block extra_args overflow");
        self.extra_args.push(extra);
        idx as u16
    }
}

/// One explicit mapping from a predecessor live-out value to a successor block parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SsaBinding {
    pub param: SsaValue,
    pub value: SsaValue,
}

/// One control-flow edge with explicit live-in bindings for the successor.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SsaEdge {
    pub target: SsaTarget,
    pub bindings: collections::Vec<SsaBinding>,
}

/// Prepared slot-based call operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SsaCallOp {
    CallDirect {
        callee: u32,
        args: FrameSpan,
        results: FrameSpan,
    },
    CallIndirect {
        type_idx: u32,
        table_idx: u32,
        index_slot: FrameSlot,
        args: FrameSpan,
        results: FrameSpan,
    },
}

/// Explicit CFG terminator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SsaTerminator {
    Goto(SsaEdge),
    Branch {
        cond: SsaValue,
        then_edge: SsaEdge,
        else_edge: SsaEdge,
    },
    BrTable {
        index: SsaValue,
        entries: collections::Vec<SsaEdge>,
    },
    Return {
        results: Option<FrameSpan>,
    },
    TrapUnreachable,
}
