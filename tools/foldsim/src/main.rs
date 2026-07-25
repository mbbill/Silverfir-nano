// foldsim: simulates the "folded stack machine" predecoder over real wasm
// binaries and reports the statistics the interpreter design discussion needs:
//
//   - raw wasm op mix (routing vs semantic)
//   - predicted dispatch count after local.get/set/tee + const folding
//   - dst-folding (set/tee) hit rates
//   - operand class distribution (T/L/C pairs) for handler-variant sizing
//   - materialized stack depth histogram (TOS window sizing)
//   - temp def->use span histogram (acc coverage)
//   - local read-after-write span histogram (register-local upper bound)
//   - cmp+br / dec-and-branch fixed-combo opportunity counts
//
// Counting model (documented approximations):
//   - Control-flow boundaries (loop entry, if/else, br*, call, return, end)
//     materialize all pending Local/Const descriptors as movs; plain `block`
//     entry does not. `end` materializes conservatively (any block end may be
//     a br target).
//   - Call arguments are materialized to canonical slots (movs), args/results
//     modeled by callee signature.
//   - br with values does not count extra result-shuffle movs (assumed folded
//     into the branch handler).
//   - Dead code (after br/return/unreachable until merge) is not counted.
//   - Weighted counters multiply by 10^loop_depth (capped at 6), the
//     project's historical static-profiling heuristic.
//
// Functions containing unmodeled ops (SIMD, GC, EH, ref call forms) fall back
// to raw counting only and are reported as "bailed".

use std::collections::HashMap;
use std::env;
use std::fmt::Write as _;
use std::fs;

use sf_nano_core::error::WasmError;
use sf_nano_core::module::type_context::TypeContext;
use sf_nano_core::module::Module;
use sf_nano_core::op_decoder::{BlockType, Decoder, Immediate, OpStream, OpcodeHandler};
use sf_nano_core::opcodes::{Opcode, OpcodeFC, WasmOpcode};

const WEIGHT_CAP: u32 = 6;

#[derive(Clone, Copy, Default)]
struct C2 {
    n: u64,
    w: f64,
}

impl C2 {
    fn add(&mut self, w: f64) {
        self.n += 1;
        self.w += w;
    }
    fn merge(&mut self, o: &C2) {
        self.n += o.n;
        self.w += o.w;
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Desc {
    Local(u32),
    Const,
    // def_em: emission index of the producer (u64::MAX = unknown, e.g. call
    // results). region: control-region id at def; dst-folding is sound only
    // within one region (no branch/merge emitted since the producer).
    Temp { def_em: u64, region: u64 },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    T = 0,
    L = 1,
    C = 2,
}

#[derive(Default)]
struct Counters {
    // raw op mix
    raw_total: C2,
    raw_local: C2,     // local.get/set/tee
    raw_const: C2,     // i32/i64/f32/f64.const
    raw_structure: C2, // block/loop/end/else/nop/drop
    raw_control: C2,   // br/br_if/br_table/return/call/call_indirect/if/select
    raw_semantic: C2,  // everything else

    // emitted (dispatched) instructions
    em_semantic: C2,
    em_control: C2,
    em_mov: C2,

    // mov breakdown
    mov_flush: C2,    // local.set hazard flush
    mov_boundary: C2, // control-flow boundary materialization
    mov_callarg: C2,  // call-boundary materialization
    mov_set: C2,      // local.set/tee that could not dst-fold

    // folding outcomes
    get_folded: C2, // Local desc consumed as L operand
    get_mat: C2,    // Local desc materialized (became mov)
    const_folded: C2,
    const_mat: C2,
    set_folded: C2,
    set_unfolded: C2,
    tee_folded: C2,
    tee_unfolded: C2,

    // binop operand classes [a][b], a = deeper operand
    binop_classes: [[C2; 3]; 3],
    // store operand classes [addr][value]
    store_classes: [[C2; 3]; 3],

    // temp def->use span (emitted-instruction distance)
    span1: C2,
    span2_4: C2,
    span_gt4: C2,
    span_unknown: C2, // temps from calls / block results

    // local read-after-write span
    lspan_le2: C2,
    lspan_3_8: C2,
    lspan_gt8: C2,
    lspan_none: C2, // read before any write in this walk (params etc.)

    // materialized temp depth at emission
    depth_hist: [C2; 16],

    // fixed-combo opportunities
    cmp_br: C2,    // compare feeding br_if/if at span 1
    induction: C2, // local.set folding an add/sub of (local, const)
    ind_br: C2,    // induction-update immediately feeding a cmp+br (loop back-edge)
    // why a local.set could not dst-fold -- decides whether the fix is copy
    // propagation (Local), const rematerialization (Const), or hazard-aware
    // folding (Temp blocked by an intervening read/write of the target)
    su_local: C2,
    su_const: C2,
    su_temp_hazard: C2,

    // short-span (<=2) local-read concentration across locals, per function
    sr_total: f64,
    sr_top1: f64,
    sr_top2: f64,
    sr_top3: f64,
}

impl Counters {
    fn merge(&mut self, o: &Counters) {
        macro_rules! m {
            ($($f:ident),*) => { $( self.$f.merge(&o.$f); )* };
        }
        m!(
            raw_total,
            raw_local,
            raw_const,
            raw_structure,
            raw_control,
            raw_semantic,
            em_semantic,
            em_control,
            em_mov,
            mov_flush,
            mov_boundary,
            mov_callarg,
            mov_set,
            get_folded,
            get_mat,
            const_folded,
            const_mat,
            set_folded,
            set_unfolded,
            tee_folded,
            tee_unfolded,
            span1,
            span2_4,
            span_gt4,
            span_unknown,
            lspan_le2,
            lspan_3_8,
            lspan_gt8,
            lspan_none,
            cmp_br,
            induction,
            ind_br,
            su_local,
            su_const,
            su_temp_hazard
        );
        for a in 0..3 {
            for b in 0..3 {
                self.binop_classes[a][b].merge(&o.binop_classes[a][b]);
                self.store_classes[a][b].merge(&o.store_classes[a][b]);
            }
        }
        for i in 0..16 {
            self.depth_hist[i].merge(&o.depth_hist[i]);
        }
        self.sr_total += o.sr_total;
        self.sr_top1 += o.sr_top1;
        self.sr_top2 += o.sr_top2;
        self.sr_top3 += o.sr_top3;
    }
}

struct Frame {
    base: usize, // stack height below params at entry
    params: usize,
    results: usize,
    is_loop: bool,
    is_if: bool,
    dead_entry: bool,
    // merge reachability: set when a br targets this block's end label, when
    // the then-arm falls through live into else, or for an else-less if
    end_targeted: bool,
    saw_else: bool,
    then_fell_live: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum LastEmit {
    Other,
    Compare,
    // add/sub whose operands folded as (Local(idx), Const)
    AddSubLocalConst(u32),
}

struct Sim<'m> {
    types: &'m TypeContext,
    module: &'m Module,
    c: Counters,
    stack: Vec<Desc>,
    frames: Vec<Frame>,
    dead: bool,
    em_idx: u64,
    region: u64,
    last_emit: LastEmit,
    last_ind_em: u64,
    last_write_em: HashMap<u32, u64>,
    last_read_em: HashMap<u32, u64>,
    last_consumed_local: Option<u32>,
    // temps resident at the START of the current wasm op (before operand pops)
    pre_depth: usize,
    // per-loop (by op_offset) locals written inside: back-edge write proxy
    loop_writes: HashMap<usize, Vec<u32>>,
    short_reads: HashMap<u32, f64>,
    bailed: bool,
    desync: bool,
    // debug: last decoded ops (offset, opcode), captured when FOLDSIM_DEBUG=1
    debug: bool,
    hist: Vec<(usize, String)>,
    desync_site: Option<String>,
}

fn block_arity(types: &TypeContext, bt: &BlockType) -> (usize, usize) {
    match bt {
        BlockType::Empty => (0, 0),
        BlockType::ValueType(_) => (0, 1),
        BlockType::TypeIndex(idx) => match types.get_function_type(*idx as u32) {
            Some(ft) => (ft.params().len(), ft.results().len()),
            None => (0, 0),
        },
    }
}

impl<'m> Sim<'m> {
    fn new(module: &'m Module) -> Self {
        Sim {
            types: module.types(),
            module,
            c: Counters::default(),
            stack: Vec::new(),
            frames: Vec::new(),
            dead: false,
            em_idx: 0,
            region: 0,
            last_emit: LastEmit::Other,
            last_ind_em: u64::MAX,
            last_write_em: HashMap::new(),
            last_read_em: HashMap::new(),
            last_consumed_local: None,
            pre_depth: 0,
            loop_writes: HashMap::new(),
            short_reads: HashMap::new(),
            bailed: false,
            desync: false,
            debug: env::var_os("FOLDSIM_DEBUG").is_some(),
            hist: Vec::new(),
            desync_site: None,
        }
    }

    fn mark_desync(&mut self, whence: &str) {
        if !self.desync && self.desync_site.is_none() {
            self.desync_site = Some(format!(
                "{} | stack {} frames {} dead {} | last ops: {:?}",
                whence,
                self.stack.len(),
                self.frames.len(),
                self.dead,
                self.hist
            ));
        }
        self.desync = true;
    }

    fn weight(&self) -> f64 {
        let loops = self.frames.iter().filter(|f| f.is_loop).count() as u32;
        10f64.powi(loops.min(WEIGHT_CAP) as i32)
    }

    // Depth is sampled at SEMANTIC ops only, at wasm-op start (pre_depth), so
    // an instruction's own temp operands count toward the window occupancy it
    // observes; movs/control are excluded so the histogram answers exactly
    // "how many temps must a window hold when real work executes".
    fn record_depth(&mut self, w: f64) {
        self.c.depth_hist[self.pre_depth.min(15)].add(w);
    }

    fn emit_sem(&mut self, w: f64) {
        self.record_depth(w);
        self.c.em_semantic.add(w);
        self.em_idx += 1;
    }

    fn emit_ctl(&mut self, w: f64) {
        self.c.em_control.add(w);
        self.em_idx += 1;
        self.last_emit = LastEmit::Other;
        // a branch/call ends the dst-folding region
        self.region += 1;
    }

    fn emit_mov(&mut self, w: f64, kind: &mut C2) {
        self.c.em_mov.add(w);
        kind.add(w);
        self.em_idx += 1;
        self.last_emit = LastEmit::Other;
    }

    // Materialize one stack slot in place if it is Local/Const.
    fn materialize_slot(&mut self, i: usize, w: f64, boundary_kind: u8) {
        let d = self.stack[i];
        match d {
            Desc::Local(_) | Desc::Const => {
                if let Desc::Local(l) = d {
                    self.c.get_mat.add(w);
                    self.last_read_em.insert(l, self.em_idx);
                } else {
                    self.c.const_mat.add(w);
                }
                self.c.em_mov.add(w);
                match boundary_kind {
                    0 => self.c.mov_boundary.add(w),
                    1 => self.c.mov_flush.add(w),
                    _ => self.c.mov_callarg.add(w),
                }
                self.em_idx += 1;
                self.last_emit = LastEmit::Other;
                self.stack[i] = Desc::Temp {
                    def_em: self.em_idx - 1,
                    region: self.region,
                };
            }
            Desc::Temp { .. } => {}
        }
    }

    fn materialize_all(&mut self, w: f64, boundary_kind: u8) {
        for i in 0..self.stack.len() {
            self.materialize_slot(i, w, boundary_kind);
        }
    }

    // Pop one operand, classify it, record folding/span stats.
    fn consume(&mut self, w: f64) -> Class {
        let d = match self.stack.pop() {
            Some(d) => d,
            None => {
                // simulation desynced from the real stack discipline; report
                // loudly instead of silently fabricating values
                self.mark_desync("consume underflow");
                Desc::Temp {
                    def_em: u64::MAX,
                    region: self.region,
                }
            }
        };
        self.last_consumed_local = None;
        match d {
            Desc::Local(idx) => {
                self.c.get_folded.add(w);
                self.last_read_em.insert(idx, self.em_idx);
                self.last_consumed_local = Some(idx);
                match self.last_write_em.get(&idx) {
                    None => self.c.lspan_none.add(w),
                    Some(&we) => {
                        let span = self.em_idx.saturating_sub(we);
                        if span <= 2 {
                            self.c.lspan_le2.add(w);
                            *self.short_reads.entry(idx).or_insert(0.0) += w;
                        } else if span <= 8 {
                            self.c.lspan_3_8.add(w);
                        } else {
                            self.c.lspan_gt8.add(w);
                        }
                    }
                }
                Class::L
            }
            Desc::Const => {
                self.c.const_folded.add(w);
                Class::C
            }
            Desc::Temp { def_em, .. } => {
                if def_em == u64::MAX {
                    self.c.span_unknown.add(w);
                } else {
                    let span = self.em_idx - def_em;
                    if span <= 1 {
                        self.c.span1.add(w);
                    } else if span <= 4 {
                        self.c.span2_4.add(w);
                    } else {
                        self.c.span_gt4.add(w);
                    }
                }
                Class::T
            }
        }
    }

    fn note_temp_span(&mut self, def_em: u64, w: f64) {
        let span = self.em_idx.saturating_sub(def_em);
        if span <= 1 {
            self.c.span1.add(w);
        } else if span <= 4 {
            self.c.span2_4.add(w);
        } else {
            self.c.span_gt4.add(w);
        }
    }

    fn push_temp(&mut self) {
        self.stack.push(Desc::Temp {
            def_em: self.em_idx.saturating_sub(1),
            region: self.region,
        });
    }

    fn push_unknown_temps(&mut self, n: usize) {
        for _ in 0..n {
            self.stack.push(Desc::Temp {
                def_em: u64::MAX,
                region: self.region,
            });
        }
    }

    fn unop(&mut self, w: f64) {
        let _a = self.consume(w);
        self.emit_sem(w);
        self.last_emit = LastEmit::Other;
        self.push_temp();
    }

    fn binop(&mut self, w: f64, is_compare: bool, op: Opcode) {
        let b = self.consume(w);
        let b_loc = self.last_consumed_local;
        let a = self.consume(w);
        let a_loc = self.last_consumed_local;
        self.c.binop_classes[a as usize][b as usize].add(w);
        self.emit_sem(w);
        self.last_emit = if is_compare {
            LastEmit::Compare
        } else {
            match op {
                Opcode::I32_ADD | Opcode::I32_SUB | Opcode::I64_ADD | Opcode::I64_SUB => {
                    // record dec-and-branch candidate shape: (Local, Const) operands
                    match (a, b, a_loc, b_loc) {
                        (Class::L, Class::C, Some(l), _) | (Class::C, Class::L, _, Some(l)) => {
                            LastEmit::AddSubLocalConst(l)
                        }
                        _ => LastEmit::Other,
                    }
                }
                _ => LastEmit::Other,
            }
        };
        self.push_temp();
    }

    fn local_set(&mut self, idx: u32, w: f64, is_tee: bool) {
        // hazard flush: pending aliases of this local lose their value
        let mut flushed = false;
        for i in 0..self.stack.len().saturating_sub(1) {
            if self.stack[i] == Desc::Local(idx) {
                self.materialize_slot(i, w, 1);
                flushed = true;
            }
        }
        if self.stack.is_empty() {
            self.mark_desync("local.set on empty stack");
        }
        let top = *self.stack.last().unwrap_or(&Desc::Temp {
            def_em: u64::MAX,
            region: self.region,
        });
        // dst-folding is sound when the producer is a known emission in the
        // current control region (no branch/merge between producer and set),
        // AND the target local was neither read nor written since the
        // producer (patching would change what those accesses observe),
        // AND no hazard flush fired for this set (flush movs execute after
        // the producer, so the producer must not already write the local).
        let top_def = match top {
            Desc::Temp { def_em, region } if def_em != u64::MAX && region == self.region => {
                Some(def_em)
            }
            _ => None,
        };
        let folded = !flushed
            && top_def.is_some_and(|d| {
                self.last_read_em.get(&idx).is_none_or(|&r| r <= d)
                    && self.last_write_em.get(&idx).is_none_or(|&wr| wr <= d)
            });
        if folded {
            // retro-patch the producing instruction's dst
            if is_tee {
                self.c.tee_folded.add(w);
            } else {
                self.c.set_folded.add(w);
            }
            let def_em = top_def.unwrap();
            // the value's single use is this set; record its span
            self.note_temp_span(def_em, w);
            // induction detection only valid when the producer is the latest
            // emission (last_emit describes that instruction)
            if def_em + 1 == self.em_idx
                && matches!(self.last_emit, LastEmit::AddSubLocalConst(l) if l == idx)
            {
                self.c.induction.add(w);
                self.last_ind_em = self.em_idx.saturating_sub(1);
            }
            self.stack.pop();
            self.last_write_em.insert(idx, def_em);
        } else {
            // emit a mov (from local/const/older temp)
            match top {
                Desc::Local(y) => {
                    self.c.get_folded.add(w);
                    self.last_read_em.insert(y, self.em_idx);
                    if !is_tee {
                        self.c.su_local.add(w);
                    }
                }
                Desc::Const => {
                    self.c.const_folded.add(w);
                    if !is_tee {
                        self.c.su_const.add(w);
                    }
                }
                Desc::Temp { def_em, .. } if def_em != u64::MAX => {
                    self.note_temp_span(def_em, w);
                    if !is_tee {
                        self.c.su_temp_hazard.add(w);
                    }
                }
                _ => {}
            }
            self.stack.pop();
            if is_tee {
                self.c.tee_unfolded.add(w);
            } else {
                self.c.set_unfolded.add(w);
            }
            let mut k = std::mem::take(&mut self.c.mov_set);
            self.emit_mov(w, &mut k);
            self.c.mov_set = k;
            self.last_write_em.insert(idx, self.em_idx - 1);
        }
        if is_tee {
            self.stack.push(Desc::Local(idx));
        }
    }

    fn branch_condition(&mut self, w: f64) {
        let top_def = match self.stack.last() {
            Some(Desc::Temp { def_em, .. }) => Some(*def_em),
            _ => None,
        };
        let cond = self.consume(w);
        if cond == Class::T
            && self.last_emit == LastEmit::Compare
            && top_def == Some(self.em_idx.saturating_sub(1))
        {
            // the immediately-preceding compare feeds this branch
            self.c.cmp_br.add(w);
            // ...and if the compare itself was immediately preceded by the
            // induction update, this is a counted loop's back-edge: today it
            // costs two dispatches (update, then fused compare-branch).
            if self.em_idx >= 2 && self.last_ind_em == self.em_idx - 2 {
                self.c.ind_br.add(w);
            }
        }
    }

    // A br/br_if/br_table to a non-loop label makes that block's end a real
    // merge target, keeping code after it reachable.
    fn mark_branch_target(&mut self, depth: u32) {
        let n = self.frames.len();
        if (depth as usize) < n {
            let f = &mut self.frames[n - 1 - depth as usize];
            if !f.is_loop {
                f.end_targeted = true;
            }
        }
    }

    fn count_raw(&mut self, wasm_op: WasmOpcode, w: f64) {
        self.c.raw_total.add(w);
        match wasm_op {
            WasmOpcode::OP(o) => match o {
                Opcode::LOCAL_GET | Opcode::LOCAL_SET | Opcode::LOCAL_TEE => {
                    self.c.raw_local.add(w)
                }
                o if is_const(o) => self.c.raw_const.add(w),
                Opcode::BLOCK
                | Opcode::LOOP
                | Opcode::END
                | Opcode::ELSE
                | Opcode::NOP
                | Opcode::DROP => self.c.raw_structure.add(w),
                Opcode::BR
                | Opcode::BR_IF
                | Opcode::BR_TABLE
                | Opcode::RETURN
                | Opcode::CALL
                | Opcode::CALL_INDIRECT
                | Opcode::IF
                | Opcode::SELECT
                | Opcode::SELECT_T
                | Opcode::RETURN_CALL
                | Opcode::RETURN_CALL_INDIRECT
                | Opcode::CALL_REF
                | Opcode::RETURN_CALL_REF
                | Opcode::UNREACHABLE => self.c.raw_control.add(w),
                _ => self.c.raw_semantic.add(w),
            },
            _ => self.c.raw_semantic.add(w),
        }
    }

    // A call is not a merge point: pending descriptors below the argument
    // region stay valid across it (the callee cannot touch caller locals).
    // Only the arguments must be staged to canonical slots.
    fn call_boundary(&mut self, w: f64, params: usize, results: usize) {
        let lo = self.stack.len().saturating_sub(params);
        for i in lo..self.stack.len() {
            self.materialize_slot(i, w, 2);
        }
        if self.stack.len() < params {
            self.mark_desync("call arg underflow");
        }
        for _ in 0..params {
            let _ = self.stack.pop();
        }
        // like emit_ctl but WITHOUT the region bump: a call is not a merge
        // point, and dst-folding a pre-call producer across it is sound (the
        // callee cannot observe caller locals)
        self.c.em_control.add(w);
        self.em_idx += 1;
        self.last_emit = LastEmit::Other;
        self.push_unknown_temps(results);
    }
}

// Classify a plain opcode for the raw-mix pass and drive the simulation.
fn is_load(op: Opcode) -> bool {
    (op as u8) >= 0x28 && (op as u8) <= 0x35
}
fn is_store(op: Opcode) -> bool {
    (op as u8) >= 0x36 && (op as u8) <= 0x3e
}
fn is_const(op: Opcode) -> bool {
    (op as u8) >= 0x41 && (op as u8) <= 0x44
}
fn is_compare_bin(op: Opcode) -> bool {
    let b = op as u8;
    (0x46..=0x4f).contains(&b)
        || (0x51..=0x5a).contains(&b)
        || (0x5b..=0x60).contains(&b)
        || (0x61..=0x66).contains(&b)
}
fn is_eqz(op: Opcode) -> bool {
    matches!(op, Opcode::I32_EQZ | Opcode::I64_EQZ)
}
fn is_numeric_bin(op: Opcode) -> bool {
    let b = op as u8;
    (0x6a..=0x78).contains(&b)
        || (0x7c..=0x8a).contains(&b)
        || (0x92..=0x98).contains(&b)
        || (0xa0..=0xa6).contains(&b)
}
fn is_numeric_un(op: Opcode) -> bool {
    let b = op as u8;
    (0x67..=0x69).contains(&b)
        || (0x79..=0x7b).contains(&b)
        || (0x8b..=0x91).contains(&b)
        || (0x99..=0x9f).contains(&b)
        || (0xa7..=0xc4).contains(&b)
}

// Pre-pass: for every loop (keyed by its op_offset), collect the locals
// written anywhere inside it — the data behind the back-edge write proxy.
#[derive(Default)]
struct LoopWriteScan {
    // stack of open control frames; Some(offset, writes) for loops
    open: Vec<Option<(usize, std::collections::HashSet<u32>)>>,
    map: HashMap<usize, Vec<u32>>,
}

impl OpcodeHandler for LoopWriteScan {
    fn on_decode_begin(&mut self) -> Result<(), WasmError> {
        Ok(())
    }

    fn on_stream<'x, 'y, 'z>(
        &mut self,
        stream: &mut OpStream<'x, 'y, 'z>,
    ) -> Result<(), WasmError> {
        while let Some(op) = stream.next()? {
            match op.wasm_op {
                WasmOpcode::OP(Opcode::BLOCK) | WasmOpcode::OP(Opcode::IF) => {
                    self.open.push(None);
                }
                WasmOpcode::OP(Opcode::LOOP) => {
                    self.open
                        .push(Some((op.op_offset, std::collections::HashSet::new())));
                }
                WasmOpcode::OP(Opcode::END) => {
                    if let Some(Some((off, set))) = self.open.pop() {
                        self.map.insert(off, set.into_iter().collect());
                    }
                }
                WasmOpcode::OP(Opcode::LOCAL_SET) | WasmOpcode::OP(Opcode::LOCAL_TEE) => {
                    if let Immediate::LocalIndex(idx) = op.imm {
                        for f in self.open.iter_mut().flatten() {
                            f.1.insert(idx);
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn on_decode_end(&mut self) -> Result<(), WasmError> {
        Ok(())
    }
}

struct SimHandler<'m> {
    sim: Sim<'m>,
    func_results: usize,
}

impl<'m> OpcodeHandler for SimHandler<'m> {
    fn on_decode_begin(&mut self) -> Result<(), WasmError> {
        Ok(())
    }

    fn on_stream<'x, 'y, 'z>(
        &mut self,
        stream: &mut OpStream<'x, 'y, 'z>,
    ) -> Result<(), WasmError> {
        while let Some(op) = stream.next()? {
            let wasm_op = op.wasm_op;
            let imm = op.imm.clone();
            let s = &mut self.sim;
            if s.debug {
                if s.hist.len() >= 16 {
                    s.hist.remove(0);
                }
                s.hist.push((op.op_offset, format!("{:?}", wasm_op)));
            }
            let w = s.weight();
            s.pre_depth = s
                .stack
                .iter()
                .filter(|d| matches!(d, Desc::Temp { .. }))
                .count();

            // ---------- raw mix (live code only, matching the sim's basis) ----------
            // a loop's END executes once per exit, not per iteration: raw-count
            // it at the post-pop weight, matching the emitted-side accounting
            let w_raw = if matches!(wasm_op, WasmOpcode::OP(Opcode::END))
                && matches!(s.frames.last(), Some(f) if f.is_loop)
            {
                let loops = s.frames.iter().filter(|f| f.is_loop).count() as u32 - 1;
                10f64.powi(loops.min(WEIGHT_CAP) as i32)
            } else {
                w
            };
            if !s.dead {
                s.count_raw(wasm_op, w_raw);
            }

            // ---------- simulation ----------
            let o = match wasm_op {
                WasmOpcode::OP(o) => o,
                WasmOpcode::FC(fc) => {
                    if s.dead {
                        continue;
                    }
                    match fc {
                        // trunc_sat: unary conversions
                        OpcodeFC::I32_TRUNC_SAT_F32_S
                        | OpcodeFC::I32_TRUNC_SAT_F32_U
                        | OpcodeFC::I32_TRUNC_SAT_F64_S
                        | OpcodeFC::I32_TRUNC_SAT_F64_U
                        | OpcodeFC::I64_TRUNC_SAT_F32_S
                        | OpcodeFC::I64_TRUNC_SAT_F32_U
                        | OpcodeFC::I64_TRUNC_SAT_F64_S
                        | OpcodeFC::I64_TRUNC_SAT_F64_U => s.unop(w),
                        OpcodeFC::MEMORY_INIT
                        | OpcodeFC::MEMORY_COPY
                        | OpcodeFC::MEMORY_FILL
                        | OpcodeFC::TABLE_INIT
                        | OpcodeFC::TABLE_COPY
                        | OpcodeFC::TABLE_FILL => {
                            for _ in 0..3 {
                                let _ = s.consume(w);
                            }
                            s.emit_sem(w);
                            s.last_emit = LastEmit::Other;
                        }
                        OpcodeFC::DATA_DROP | OpcodeFC::ELEM_DROP => {
                            s.emit_sem(w);
                            s.last_emit = LastEmit::Other;
                        }
                        OpcodeFC::TABLE_GROW => {
                            let _ = s.consume(w);
                            let _ = s.consume(w);
                            s.emit_sem(w);
                            s.last_emit = LastEmit::Other;
                            s.push_temp();
                        }
                        OpcodeFC::TABLE_SIZE => {
                            s.emit_sem(w);
                            s.last_emit = LastEmit::Other;
                            s.push_temp();
                        }
                    }
                    continue;
                }
                _ => {
                    s.bailed = true;
                    return Err(WasmError::invalid("foldsim: unmodeled prefix op"));
                }
            };

            // dead-code skipping with frame tracking
            if s.dead {
                match o {
                    Opcode::BLOCK | Opcode::LOOP | Opcode::IF => {
                        let (p, r) = block_arity(
                            s.types,
                            match &imm {
                                Immediate::Block(bt) => bt,
                                _ => &BlockType::Empty,
                            },
                        );
                        s.frames.push(Frame {
                            base: s.stack.len().saturating_sub(p),
                            params: p,
                            results: r,
                            is_loop: o == Opcode::LOOP,
                            is_if: o == Opcode::IF,
                            dead_entry: true,
                            end_targeted: false,
                            saw_else: false,
                            then_fell_live: false,
                        });
                    }
                    Opcode::ELSE => {
                        if let Some(f) = s.frames.last_mut() {
                            f.saw_else = true;
                            // then-arm ended dead: no fall-through into end
                            s.dead = f.dead_entry;
                            let base = f.base;
                            let params = f.params;
                            s.stack.truncate(base);
                            let n = params;
                            s.push_unknown_temps(n);
                        }
                    }
                    Opcode::END => {
                        if let Some(f) = s.frames.pop() {
                            // reached the end while dead: code after is live
                            // only if this end is a genuine merge target
                            let live_after = !f.dead_entry
                                && (f.end_targeted || f.then_fell_live || (f.is_if && !f.saw_else));
                            s.dead = !live_after;
                            s.stack.truncate(f.base);
                            s.push_unknown_temps(f.results);
                        }
                        // function-level END while dead: nothing follows, the
                        // function tail is genuinely unreachable — stay dead
                    }
                    _ => {}
                }
                continue;
            }

            match o {
                Opcode::NOP => {}
                Opcode::UNREACHABLE => {
                    s.dead = true;
                }
                Opcode::BLOCK | Opcode::LOOP => {
                    let (p, r) = block_arity(
                        s.types,
                        match &imm {
                            Immediate::Block(bt) => bt,
                            _ => &BlockType::Empty,
                        },
                    );
                    if o == Opcode::LOOP {
                        // loop header is a merge point (back-edges arrive here)
                        s.materialize_all(w, 0);
                        s.region += 1;
                        // back-edge write proxy: locals written inside the loop
                        // are re-written each iteration before any in-loop read
                        // crosses the back edge
                        if let Some(ws) = s.loop_writes.get(&op.op_offset).cloned() {
                            for l in ws {
                                s.last_write_em.insert(l, s.em_idx);
                            }
                        }
                    }
                    s.frames.push(Frame {
                        base: s.stack.len().saturating_sub(p),
                        params: p,
                        results: r,
                        is_loop: o == Opcode::LOOP,
                        is_if: false,
                        dead_entry: false,
                        end_targeted: false,
                        saw_else: false,
                        then_fell_live: false,
                    });
                }
                Opcode::IF => {
                    s.branch_condition(w);
                    s.materialize_all(w, 0);
                    let (p, r) = block_arity(
                        s.types,
                        match &imm {
                            Immediate::Block(bt) => bt,
                            _ => &BlockType::Empty,
                        },
                    );
                    s.emit_ctl(w);
                    s.frames.push(Frame {
                        base: s.stack.len().saturating_sub(p),
                        params: p,
                        results: r,
                        is_loop: false,
                        is_if: true,
                        dead_entry: false,
                        end_targeted: false,
                        saw_else: false,
                        then_fell_live: false,
                    });
                }
                Opcode::ELSE => {
                    // jump over else-arm from end of then-arm
                    s.materialize_all(w, 0);
                    s.emit_ctl(w);
                    if let Some(f) = s.frames.last_mut() {
                        f.saw_else = true;
                        f.then_fell_live = true;
                        s.dead = f.dead_entry;
                        let base = f.base;
                        let params = f.params;
                        s.stack.truncate(base);
                        s.push_unknown_temps(params);
                    }
                    s.region += 1;
                }
                Opcode::END => {
                    if let Some(f) = s.frames.pop() {
                        // end-of-block materialization runs once per exit, not
                        // per loop iteration: weight computed after the pop
                        let w2 = s.weight();
                        s.materialize_all(w2, 0);
                        if s.stack.len() != f.base + f.results {
                            s.mark_desync("END height mismatch");
                            s.stack.truncate(f.base);
                            s.push_unknown_temps(f.results);
                        } else {
                            // keep the materialized result descriptors: their
                            // heights are already canonical
                            s.stack.truncate(f.base + f.results);
                        }
                        s.dead = f.dead_entry;
                    } else {
                        s.materialize_all(w, 0);
                    }
                    s.region += 1;
                }
                Opcode::BR => {
                    if let Immediate::LabelIndex(d) = imm {
                        s.mark_branch_target(d);
                    }
                    s.materialize_all(w, 0);
                    s.emit_ctl(w);
                    s.dead = true;
                }
                Opcode::BR_IF => {
                    if let Immediate::LabelIndex(d) = imm {
                        s.mark_branch_target(d);
                    }
                    s.branch_condition(w);
                    s.materialize_all(w, 0);
                    s.emit_ctl(w);
                }
                Opcode::BR_TABLE => {
                    if let Immediate::BrLabels(ref labels, default) = imm {
                        for &d in labels.iter() {
                            s.mark_branch_target(d);
                        }
                        s.mark_branch_target(default);
                    }
                    let _idx = s.consume(w);
                    s.materialize_all(w, 0);
                    s.emit_ctl(w);
                    s.dead = true;
                }
                Opcode::RETURN => {
                    s.materialize_all(w, 0);
                    s.emit_ctl(w);
                    s.dead = true;
                }
                Opcode::CALL => {
                    let fidx = match imm {
                        Immediate::FunctionIndex(i) => i,
                        _ => 0,
                    };
                    let (p, r) = s
                        .module
                        .functions()
                        .get(fidx as usize)
                        .map(|f| {
                            let ft = f.func_type();
                            (ft.params().len(), ft.results().len())
                        })
                        .unwrap_or((0, 0));
                    s.call_boundary(w, p, r);
                }
                Opcode::CALL_INDIRECT => {
                    let tidx = match imm {
                        Immediate::CallIndirectArgs { typeidx, .. } => typeidx,
                        _ => 0,
                    };
                    let _target = s.consume(w);
                    let (p, r) = s
                        .types
                        .get_function_type(tidx)
                        .map(|ft| (ft.params().len(), ft.results().len()))
                        .unwrap_or((0, 0));
                    s.call_boundary(w, p, r);
                }
                Opcode::RETURN_CALL | Opcode::RETURN_CALL_INDIRECT => {
                    if o == Opcode::RETURN_CALL_INDIRECT {
                        let _ = s.consume(w);
                    }
                    s.materialize_all(w, 2);
                    s.stack.clear();
                    s.emit_ctl(w);
                    s.dead = true;
                }
                Opcode::DROP => match s.stack.pop() {
                    Some(Desc::Local(_)) => s.c.get_folded.add(w),
                    Some(Desc::Const) => s.c.const_folded.add(w),
                    Some(Desc::Temp { def_em, .. }) if def_em != u64::MAX => {
                        s.note_temp_span(def_em, w);
                    }
                    Some(_) => {}
                    None => s.mark_desync("drop on empty stack"),
                },
                Opcode::SELECT | Opcode::SELECT_T => {
                    let _c = s.consume(w);
                    let _b = s.consume(w);
                    let _a = s.consume(w);
                    s.emit_sem(w);
                    s.last_emit = LastEmit::Other;
                    s.push_temp();
                }
                Opcode::LOCAL_GET => {
                    if let Immediate::LocalIndex(i) = imm {
                        s.stack.push(Desc::Local(i));
                    }
                }
                Opcode::LOCAL_SET => {
                    if let Immediate::LocalIndex(i) = imm {
                        s.local_set(i, w, false);
                    }
                }
                Opcode::LOCAL_TEE => {
                    if let Immediate::LocalIndex(i) = imm {
                        s.local_set(i, w, true);
                    }
                }
                Opcode::GLOBAL_GET => {
                    s.emit_sem(w);
                    s.last_emit = LastEmit::Other;
                    s.push_temp();
                }
                Opcode::GLOBAL_SET => {
                    let _ = s.consume(w);
                    s.emit_sem(w);
                    s.last_emit = LastEmit::Other;
                }
                Opcode::TABLE_GET => {
                    let _ = s.consume(w);
                    s.emit_sem(w);
                    s.last_emit = LastEmit::Other;
                    s.push_temp();
                }
                Opcode::TABLE_SET => {
                    let _ = s.consume(w);
                    let _ = s.consume(w);
                    s.emit_sem(w);
                    s.last_emit = LastEmit::Other;
                }
                Opcode::MEMORY_SIZE => {
                    s.emit_sem(w);
                    s.last_emit = LastEmit::Other;
                    s.push_temp();
                }
                Opcode::MEMORY_GROW => {
                    let _ = s.consume(w);
                    s.emit_sem(w);
                    s.last_emit = LastEmit::Other;
                    s.push_temp();
                }
                Opcode::REF_NULL | Opcode::REF_FUNC => {
                    s.stack.push(Desc::Const);
                }
                Opcode::REF_IS_NULL => s.unop(w),
                o if is_const(o) => {
                    s.stack.push(Desc::Const);
                }
                o if is_load(o) => {
                    let _addr = s.consume(w);
                    s.emit_sem(w);
                    s.last_emit = LastEmit::Other;
                    s.push_temp();
                }
                o if is_store(o) => {
                    let b = s.consume(w); // value
                    let a = s.consume(w); // addr
                    s.c.store_classes[a as usize][b as usize].add(w);
                    s.emit_sem(w);
                    s.last_emit = LastEmit::Other;
                }
                o if is_eqz(o) => {
                    let _a = s.consume(w);
                    s.emit_sem(w);
                    s.last_emit = LastEmit::Compare;
                    s.push_temp();
                }
                o if is_compare_bin(o) => s.binop(w, true, o),
                o if is_numeric_bin(o) => s.binop(w, false, o),
                o if is_numeric_un(o) => s.unop(w),
                _ => {
                    s.bailed = true;
                    return Err(WasmError::invalid("foldsim: unmodeled op"));
                }
            }
        }
        Ok(())
    }

    fn on_decode_end(&mut self) -> Result<(), WasmError> {
        // final consistency check: after the implicit function-end, the
        // symbolic stack must hold exactly the function results (unless the
        // tail was dead code)
        let s = &mut self.sim;
        if !s.bailed
            && !s.desync
            && !s.dead
            && s.frames.is_empty()
            && s.stack.len() != self.func_results
        {
            s.mark_desync("function-end height mismatch");
        }
        Ok(())
    }
}

fn pct(n: f64, d: f64) -> f64 {
    if d == 0.0 {
        0.0
    } else {
        n / d * 100.0
    }
}

fn report(name: &str, c: &Counters, bailed: usize, desynced: usize, total_funcs: usize) -> String {
    let mut o = String::new();
    let rt = c.raw_total.w;
    let em_total = c.em_semantic.w + c.em_control.w + c.em_mov.w;
    let em_total_n = c.em_semantic.n + c.em_control.n + c.em_mov.n;
    macro_rules! line {
        ($($t:tt)*) => { let _ = writeln!(o, $($t)*); };
    }
    line!("== {name} ==");
    line!(
        "  functions simulated: {}/{} (bailed {}, desynced {})",
        total_funcs - bailed - desynced,
        total_funcs,
        bailed,
        desynced
    );
    line!(
        "  raw ops        : {:>12} unweighted | weighted mix: local {:.1}% const {:.1}% structure {:.1}% control {:.1}% semantic {:.1}%",
        c.raw_total.n,
        pct(c.raw_local.w, rt),
        pct(c.raw_const.w, rt),
        pct(c.raw_structure.w, rt),
        pct(c.raw_control.w, rt),
        pct(c.raw_semantic.w, rt)
    );
    line!(
        "  dispatches     : {:>12} unweighted | weighted ratio to raw: {:.3} (sem {:.1}% ctl {:.1}% mov {:.1}%)",
        em_total_n,
        em_total / rt.max(1.0),
        pct(c.em_semantic.w, em_total),
        pct(c.em_control.w, em_total),
        pct(c.em_mov.w, em_total)
    );
    line!(
        "  old-basis ratio: {:.3} (denominator excludes structure ops, the basis the old interpreter's dynamic dispatch counts used)",
        em_total / (rt - c.raw_structure.w).max(1.0)
    );
    line!(
        "  movs breakdown : set-unfold {:.1}% boundary {:.1}% call-arg {:.1}% flush {:.1}%  (of all dispatches: {:.2}%)",
        pct(c.mov_set.w, c.em_mov.w),
        pct(c.mov_boundary.w, c.em_mov.w),
        pct(c.mov_callarg.w, c.em_mov.w),
        pct(c.mov_flush.w, c.em_mov.w),
        pct(c.em_mov.w, em_total)
    );
    line!(
        "  local.get fold : {:.1}% folded (unweighted {}/{})",
        pct(c.get_folded.w, c.get_folded.w + c.get_mat.w),
        c.get_folded.n,
        c.get_folded.n + c.get_mat.n
    );
    line!(
        "  local.set fold : {:.1}% dst-folded | tee: {:.1}%",
        pct(c.set_folded.w, c.set_folded.w + c.set_unfolded.w),
        pct(c.tee_folded.w, c.tee_folded.w + c.tee_unfolded.w)
    );
    line!(
        "  const fold     : {:.1}% folded",
        pct(c.const_folded.w, c.const_folded.w + c.const_mat.w)
    );
    let cls_total: f64 = c.binop_classes.iter().flatten().map(|x| x.w).sum();
    let cl = |a: usize, b: usize| pct(c.binop_classes[a][b].w, cls_total);
    line!(
        "  binop classes  : TT {:.1}% TL {:.1}% TC {:.1}% | LT {:.1}% LL {:.1}% LC {:.1}% | CT {:.1}% CL {:.1}% CC {:.1}%",
        cl(0, 0), cl(0, 1), cl(0, 2), cl(1, 0), cl(1, 1), cl(1, 2), cl(2, 0), cl(2, 1), cl(2, 2)
    );
    let st_total: f64 = c.store_classes.iter().flatten().map(|x| x.w).sum();
    let st = |a: usize, b: usize| pct(c.store_classes[a][b].w, st_total);
    line!(
        "  store classes  : [addr][val] TT {:.1}% TL {:.1}% TC {:.1}% | LT {:.1}% LL {:.1}% LC {:.1}% | CT {:.1}% CL {:.1}% CC {:.1}% ({:.1}% of sem)",
        st(0, 0), st(0, 1), st(0, 2), st(1, 0), st(1, 1), st(1, 2), st(2, 0), st(2, 1), st(2, 2),
        pct(st_total, c.em_semantic.w)
    );
    let span_total = c.span1.w + c.span2_4.w + c.span_gt4.w + c.span_unknown.w;
    line!(
        "  temp spans     : 1: {:.1}%  2-4: {:.1}%  >4: {:.1}%  unknown: {:.1}%",
        pct(c.span1.w, span_total),
        pct(c.span2_4.w, span_total),
        pct(c.span_gt4.w, span_total),
        pct(c.span_unknown.w, span_total)
    );
    let lspan_total = c.lspan_le2.w + c.lspan_3_8.w + c.lspan_gt8.w + c.lspan_none.w;
    line!(
        "  local RAW span : <=2: {:.1}%  3-8: {:.1}%  >8: {:.1}%  no-write: {:.1}%",
        pct(c.lspan_le2.w, lspan_total),
        pct(c.lspan_3_8.w, lspan_total),
        pct(c.lspan_gt8.w, lspan_total),
        pct(c.lspan_none.w, lspan_total)
    );
    let dh_total: f64 = c.depth_hist.iter().map(|x| x.w).sum();
    let mut cum = 0.0;
    let mut depth_line = String::new();
    for (i, h) in c.depth_hist.iter().enumerate().take(6) {
        cum += h.w;
        let _ = write!(depth_line, "<={}: {:.1}%  ", i, pct(cum, dh_total));
    }
    line!("  mat depth cum  : {}", depth_line);
    line!(
        "  fixed combos   : cmp+br {:.2}% of dispatches | induction-update {:.2}% | \
back-edge (update+cmp+br) {:.2}%",
        pct(c.cmp_br.w, em_total),
        pct(c.induction.w, em_total),
        pct(c.ind_br.w, em_total)
    );
    line!(
        "  set-unfold why : local-copy {:.2}% of dispatches | const {:.2}% | temp-hazard {:.2}%",
        pct(c.su_local.w, em_total),
        pct(c.su_const.w, em_total),
        pct(c.su_temp_hazard.w, em_total)
    );
    line!(
        "  short-read conc: per-func top1 {:.1}% top2 {:.1}% top3 {:.1}% of short-span reads (short = {:.1}% of consumed local reads)",
        pct(c.sr_top1, c.sr_total),
        pct(c.sr_top2, c.sr_total),
        pct(c.sr_top3, c.sr_total),
        pct(c.sr_total, lspan_total)
    );
    o
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: foldsim <a.wasm> [b.wasm ...]");
        std::process::exit(1);
    }
    let mut agg = Counters::default();
    let mut agg_bailed = 0usize;
    let mut agg_desynced = 0usize;
    let mut agg_funcs = 0usize;
    for path in &args {
        let bytes = match fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{path}: read error: {e}");
                continue;
            }
        };
        let module = match Module::new(path, &bytes) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{path}: parse error: {e:?}");
                continue;
            }
        };
        let mut file_c = Counters::default();
        let mut bailed = 0usize;
        let mut desynced = 0usize;
        let mut funcs = 0usize;
        for f in module.functions() {
            let Some(spec) = f.spec() else { continue };
            if spec.code().is_empty() {
                continue;
            }
            funcs += 1;
            let mut scan = LoopWriteScan::default();
            {
                let mut d = Decoder::new(spec.code());
                d.add_handler(&mut scan);
                let _ = d.decode_function();
            }
            let mut h = SimHandler {
                sim: Sim::new(&module),
                func_results: f.func_type().results().len(),
            };
            h.sim.loop_writes = scan.map;
            let res = {
                let mut d = Decoder::new(spec.code());
                d.add_handler(&mut h);
                d.decode_function()
            };
            if res.is_err() || h.sim.bailed {
                bailed += 1;
                continue;
            }
            if h.sim.desync {
                desynced += 1;
                if h.sim.debug {
                    eprintln!(
                        "DESYNC {} func#{}: {}",
                        path,
                        funcs - 1,
                        h.sim.desync_site.as_deref().unwrap_or("?")
                    );
                }
                continue;
            }
            // fold short-span local-read concentration into the counters
            let mut sr: Vec<f64> = h.sim.short_reads.values().copied().collect();
            sr.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let c = &mut h.sim.c;
            c.sr_total = sr.iter().sum();
            c.sr_top1 = sr.first().copied().unwrap_or(0.0);
            c.sr_top2 = c.sr_top1 + sr.get(1).copied().unwrap_or(0.0);
            c.sr_top3 = c.sr_top2 + sr.get(2).copied().unwrap_or(0.0);
            file_c.merge(&h.sim.c);
        }
        let name = path.rsplit('/').next().unwrap_or(path);
        print!("{}", report(name, &file_c, bailed, desynced, funcs));
        agg.merge(&file_c);
        agg_bailed += bailed;
        agg_desynced += desynced;
        agg_funcs += funcs;
    }
    if args.len() > 1 {
        print!(
            "{}",
            report("AGGREGATE", &agg, agg_bailed, agg_desynced, agg_funcs)
        );
    }
}
