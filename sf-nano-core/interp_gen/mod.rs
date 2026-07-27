//! Build-time generator for the interpreter's native dispatch engine.
//!
//! Compiled by `build.rs`, never by the crate. It walks the handler
//! variant space defined in `layout` and asks the selected target backend
//! to emit each one, producing a single assembler source file that
//! `global_asm!` folds into the binary's own `.text`. Nothing is generated
//! at run time, so the interpreter needs no executable memory and its
//! engine size is a link-time (on MCUs, a flash) budget rather than an
//! allocation.
//!
//! Emission ORDER is a first-class parameter, not an implementation
//! detail: relocating a rarely-engaged family between two hot ones has
//! measured ~5% swings three separate times. The order below is the one
//! the arm64 engine was tuned into — hot families first, cold ones
//! (div/rem, bulk memory, float, the reinterpret tail) after them.

pub(crate) use super::instr;
pub(crate) use super::layout;

pub mod arm32;
pub mod arm64;
pub mod asm;
pub mod isa;
pub mod riscv;
pub mod x86_64;

use asm::{Asm, ObjFmt};
use instr::{op_from_index, Op};
use isa::{Isa, Stubs, Variant};
use layout::{build_op_base, family, total_slots, Cls, DstCls, Fam, PairDstCls, N_OPS};

/// One emission group: a run of ops sharing a shape, in the order the
/// engine wants them laid out.
struct Group {
    ops: &'static [Op],
    /// Emit the address-fused bank instead of the plain one.
    fused: bool,
}

const fn g(ops: &'static [Op]) -> Group {
    Group { ops, fused: false }
}
const fn gf(ops: &'static [Op]) -> Group {
    Group { ops, fused: true }
}

use Op::*;

const MEM_OPS: &[Op] = &[
    I32_Load,
    F32_Load,
    I64_Load,
    F64_Load,
    I32_Load8U,
    I64_Load8U,
    I32_Load16U,
    I64_Load16U,
    I32_Load8S,
    I32_Load16S,
    I64_Load8S,
    I64_Load16S,
    I64_Load32S,
    I64_Load32U,
    I32_Store,
    F32_Store,
    I64_Store,
    F64_Store,
    I32_Store8,
    I64_Store8,
    I32_Store16,
    I64_Store16,
    I64_Store32,
];

fn emit_order() -> Vec<Group> {
    vec![
        g(&[Return]),
        g(&[MovSlot, MovConst]),
        // integer ALU, the hottest family
        g(&[
            I32_Add, I32_Sub, I32_And, I32_Or, I32_Xor, I32_Mul, I32_Shl, I32_ShrU, I32_ShrS,
            I32_Rotr, I32_Rotl, I64_Add, I64_Sub, I64_And, I64_Or, I64_Xor, I64_Mul, I64_Shl,
            I64_ShrU, I64_ShrS, I64_Rotr, I64_Rotl,
        ]),
        g(&[
            I32_Eq, I32_Ne, I32_LtS, I32_LtU, I32_GtS, I32_GtU, I32_LeS, I32_LeU, I32_GeS, I32_GeU,
            I64_Eq, I64_Ne, I64_LtS, I64_LtU, I64_GtS, I64_GtU, I64_LeS, I64_LeU, I64_GeS, I64_GeU,
        ]),
        g(&[I32_Eqz, I64_Eqz]),
        g(&[
            I32_Clz,
            I64_Clz,
            I32_Ctz,
            I64_Ctz,
            I32_Extend8S,
            I32_Extend16S,
            I64_Extend8S,
            I64_Extend16S,
            I64_Extend32S,
            I32_WrapI64,
            I64_ExtendI32S,
            I64_ExtendI32U,
        ]),
        g(&[Br]),
        g(&[BrIf, BrIfNot]),
        g(&[
            I32_BrEq,
            I32_BrNe,
            I32_BrLtS,
            I32_BrLtU,
            I32_BrGtS,
            I32_BrGtU,
            I32_BrLeS,
            I32_BrLeU,
            I32_BrGeS,
            I32_BrGeU,
            I64_BrEq,
            I64_BrNe,
            I64_BrLtS,
            I64_BrLtU,
            I64_BrGtS,
            I64_BrGtU,
            I64_BrLeS,
            I64_BrLeU,
            I64_BrGeS,
            I64_BrGeU,
            I32_BrAnd,
            I32_BrAndNot,
            I32_SubBrIf,
            I64_SubBrIf,
        ]),
        g(&[Select]),
        g(&[BrTable]),
        g(&[GlobalGet, GlobalSet]),
        g(MEM_OPS),
        gf(MEM_OPS),
        // cold from here down: never place one of these between two hot
        // families (measured 5.3% on lua fib when div/rem sat between the
        // binop and compare blocks).
        g(&[
            I32_DivS, I32_DivU, I32_RemS, I32_RemU, I64_DivS, I64_DivU, I64_RemS, I64_RemU,
        ]),
        g(&[MemoryFill, MemoryCopy, MemoryFillCopy]),
        g(&[
            F32_Add, F32_Sub, F32_Mul, F32_Div, F32_Min, F32_Max, F64_Add, F64_Sub, F64_Mul,
            F64_Div, F64_Min, F64_Max,
        ]),
        g(&[F32_Copysign, F64_Copysign]),
        g(&[
            F32_Eq, F32_Ne, F32_Lt, F32_Gt, F32_Le, F32_Ge, F64_Eq, F64_Ne, F64_Lt, F64_Gt, F64_Le,
            F64_Ge,
        ]),
        g(&[
            F32_Abs,
            F32_Neg,
            F32_Sqrt,
            F32_Ceil,
            F32_Floor,
            F32_Trunc,
            F32_Nearest,
            F64_Abs,
            F64_Neg,
            F64_Sqrt,
            F64_Ceil,
            F64_Floor,
            F64_Trunc,
            F64_Nearest,
        ]),
        g(&[
            F32_ConvertI32S,
            F32_ConvertI32U,
            F32_ConvertI64S,
            F32_ConvertI64U,
            F64_ConvertI32S,
            F64_ConvertI32U,
            F64_ConvertI64S,
            F64_ConvertI64U,
        ]),
        g(&[F32_DemoteF64, F64_PromoteF32]),
        g(&[
            I32_TruncSatF32S,
            I32_TruncSatF32U,
            I32_TruncSatF64S,
            I32_TruncSatF64U,
            I64_TruncSatF32S,
            I64_TruncSatF32U,
            I64_TruncSatF64S,
            I64_TruncSatF64U,
        ]),
        g(&[
            I32_TruncF32S,
            I32_TruncF32U,
            I32_TruncF64S,
            I32_TruncF64U,
            I64_TruncF32S,
            I64_TruncF32U,
            I64_TruncF64S,
            I64_TruncF64U,
        ]),
        g(&[I32_Popcnt, I64_Popcnt]),
        g(&[
            I32_ReinterpretF32,
            I64_ReinterpretF64,
            F32_ReinterpretI32,
            F64_ReinterpretI64,
        ]),
        // MovPair last: its first position, between the mov and binop
        // families, cost sha256 5%.
        g(&[MovPair]),
    ]
}

/// Diagnostic count-mode for the in-chain dispatch counter: 0 = count
/// every dispatch, 1 = only handlers whose variant involves L0, 2 = only
/// L1. Modes 1 and 2 turn one run into an exact, timing-independent
/// measurement of dynamic pinned-class engagement.
const COUNT_MODE: u8 = 0;

fn counted(counting: bool, a: Cls, b: Cls, d: DstCls, pair_d: PairDstCls) -> bool {
    if !counting {
        return false;
    }
    match COUNT_MODE {
        1 => {
            a == Cls::L0
                || b == Cls::L0
                || d == DstCls::L0
                || pair_d.first() == Some(DstCls::L0)
                || pair_d.second() == Some(DstCls::L0)
        }
        2 => {
            a == Cls::L1
                || b == Cls::L1
                || d == DstCls::L1
                || pair_d.first() == Some(DstCls::L1)
                || pair_d.second() == Some(DstCls::L1)
        }
        _ => true,
    }
}

/// The class combinations a family owns, in emission order.
fn combos(
    fam: Fam,
    classes: &[Cls],
    dsts: &[DstCls],
    has_l1: bool,
) -> Vec<(Cls, Cls, DstCls, PairDstCls)> {
    let mut out = Vec::new();
    let plain = PairDstCls::None;
    match fam {
        Fam::None => {}
        Fam::Fixed => out.push((Cls::Slot, Cls::Slot, DstCls::Mem, plain)),
        Fam::Dst => {
            for &d in dsts {
                out.push((Cls::Slot, Cls::Slot, d, plain));
            }
        }
        Fam::ConstDst => {
            for &d in dsts {
                out.push((Cls::Const, Cls::Slot, d, plain));
            }
        }
        Fam::SrcA => {
            for &a in classes {
                out.push((a, Cls::Slot, DstCls::Mem, plain));
            }
        }
        Fam::SrcADst | Fam::Load => {
            for &a in classes {
                for &d in dsts {
                    out.push((a, Cls::Slot, d, plain));
                }
            }
        }
        Fam::SrcAB | Fam::Store => {
            for &a in classes {
                for &b in classes {
                    out.push((a, b, DstCls::Mem, plain));
                }
            }
        }
        Fam::SrcABPairDst => {
            let pair_dsts = if has_l1 {
                &[
                    PairDstCls::None,
                    PairDstCls::FirstL0,
                    PairDstCls::FirstL1,
                    PairDstCls::SecondL0,
                    PairDstCls::SecondL1,
                    PairDstCls::L0L1,
                    PairDstCls::L1L0,
                ][..]
            } else {
                &[PairDstCls::None, PairDstCls::FirstL0, PairDstCls::SecondL0][..]
            };
            for &a in classes {
                for &b in classes {
                    for &pair_d in pair_dsts {
                        out.push((a, b, DstCls::Mem, pair_d));
                    }
                }
            }
        }
        Fam::SrcABDst => {
            for &a in classes {
                for &b in classes {
                    for &d in dsts {
                        out.push((a, b, d, plain));
                    }
                }
            }
        }
    }
    out
}

fn stubs(a: &Asm) -> Stubs {
    Stubs {
        exit_common: a.local("exit_common"),
        slow: a.local("exit_slow"),
        return_exit: a.local("exit_return"),
        trap_oob: a.local("trap_oob"),
        trap_exhaust: a.local("trap_exhaust"),
        entry: a.local("entry"),
        call: a.local("h_call"),
        call_indirect: a.local("h_call_indirect"),
        call_core: a.local("call_core"),
    }
}

/// Emit the whole engine for one backend. Returns the assembler source.
pub fn generate(isa: &mut dyn Isa, fmt: ObjFmt, thumb: bool, counting: bool) -> String {
    let caps = isa.caps();
    let mut a = Asm::new(fmt, thumb);
    let st = stubs(&a);
    let op_base = build_op_base();
    let n_slots = total_slots();
    let mut slot_label: Vec<Option<String>> = vec![None; n_slots];

    a.raw("\t.text");
    if thumb {
        a.raw("\t.syntax unified");
        a.raw("\t.thumb");
    }
    a.align(4);
    a.global_label("sf_interp_code_base");
    let base = a.base_label();
    a.label(&base);

    isa.emit_prelude(&mut a, &st);

    for group in emit_order() {
        for &op in group.ops {
            let fam = family(op);
            if group.fused && !fam.has_fused_bank() {
                continue;
            }
            for (ac, bc, dc, pair_dc) in combos(fam, caps.classes, caps.dsts, caps.has_l1) {
                let v = Variant {
                    op,
                    a: ac,
                    b: bc,
                    d: dc,
                    pair_d: pair_dc,
                    fused: group.fused,
                    counted: counted(counting, ac, bc, dc, pair_dc),
                };
                if !isa.wants(&v) {
                    continue;
                }
                let Some(idx) = fam.index_of(ac, bc, dc, pair_dc, group.fused) else {
                    continue;
                };
                let slot = op_base[op as usize] as usize + idx;
                assert!(
                    slot_label[slot].is_none(),
                    "duplicate handler slot for {op:?} ({ac:?},{bc:?},{dc:?},{pair_dc:?},fused={})",
                    group.fused
                );
                let label = a.fresh("h");
                a.label(&label);
                isa.emit_handler(&mut a, &st, &v);
                slot_label[slot] = Some(label);
            }
        }
    }

    let end = a.local("sf_end");
    a.label(&end);

    // Tables. Kept in read-only data rather than in `.text`: the runtime
    // reads them, and nothing here needs to be executable.
    a.rodata();
    a.align(2);
    a.global_label("sf_interp_meta");
    a.code_off(&st.entry);
    a.code_off(&st.slow);
    a.code_off(&st.return_exit);
    a.code_off(&st.call);
    a.code_off(&st.call_indirect);
    a.plain_off(&end);
    a.word(n_slots as u32);

    a.align(2);
    a.global_label("sf_interp_handlers");
    for label in slot_label.iter() {
        match label {
            Some(l) => a.code_off(l),
            // Offset 0 is the blob base, which is never a handler, so it
            // doubles as the "no native form" sentinel.
            None => a.word(0),
        }
    }

    // A debug aid, not a contract: which op owns which slot range.
    a.raw("");
    for i in 0..N_OPS {
        let op = op_from_index(i);
        let fam = family(op);
        if fam.slots() > 0 {
            a.comment(&format!(
                "{:?}: slots {}..{}",
                op,
                op_base[i],
                op_base[i] as usize + fam.slots()
            ));
        }
    }

    a.finish()
}
