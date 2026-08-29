//! Temporary, feature-gated eager-startup stage profiler.
//!
//! The production crate is `no_std`. This module exists only behind the
//! `startup-profile` feature used by a dedicated hosted CI binary. Keeping a
//! fixed thread-local table avoids allocation and locks inside the measured
//! path; ordinary builds do not compile this module or any of its call sites.

use core::cell::RefCell;
use std::time::{Duration, Instant};

/// Mutually named measurements emitted by the temporary profiler.
///
/// Some stages are deliberately nested. The CI reducer derives exclusive
/// `predecode.lowering` and `link.cell_transform` values by subtracting their
/// named children from the corresponding total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Stage {
    StartupTotal,
    Drop,
    ParserTotal,
    ParserHeader,
    ParserPlan,
    ParserCustom,
    ParserType,
    ParserImport,
    ParserFunction,
    ParserTable,
    ParserMemory,
    ParserTag,
    ParserGlobal,
    ParserExport,
    ParserStart,
    ParserElement,
    ParserDataCount,
    ParserCode,
    ParserData,
    ParserFinalize,
    InstanceBuildTotal,
    InstanceSetup,
    InstanceMemories,
    InstanceGlobals,
    InstanceTables,
    InstanceStackDeferred,
    InstanceElementSegments,
    InstanceDataSegments,
    InstanceLease,
    PredecodeTotal,
    PredecodeDecode,
    PredecodeScratch,
    PinnedCensus,
    LinkTotal,
    LinkCellsTotal,
    LinkHandlerSelection,
    LinkCallFixup,
    LinkFinalize,
}

impl Stage {
    pub const COUNT: usize = Stage::LinkFinalize as usize + 1;

    pub const ALL: [Stage; Self::COUNT] = [
        Stage::StartupTotal,
        Stage::Drop,
        Stage::ParserTotal,
        Stage::ParserHeader,
        Stage::ParserPlan,
        Stage::ParserCustom,
        Stage::ParserType,
        Stage::ParserImport,
        Stage::ParserFunction,
        Stage::ParserTable,
        Stage::ParserMemory,
        Stage::ParserTag,
        Stage::ParserGlobal,
        Stage::ParserExport,
        Stage::ParserStart,
        Stage::ParserElement,
        Stage::ParserDataCount,
        Stage::ParserCode,
        Stage::ParserData,
        Stage::ParserFinalize,
        Stage::InstanceBuildTotal,
        Stage::InstanceSetup,
        Stage::InstanceMemories,
        Stage::InstanceGlobals,
        Stage::InstanceTables,
        Stage::InstanceStackDeferred,
        Stage::InstanceElementSegments,
        Stage::InstanceDataSegments,
        Stage::InstanceLease,
        Stage::PredecodeTotal,
        Stage::PredecodeDecode,
        Stage::PredecodeScratch,
        Stage::PinnedCensus,
        Stage::LinkTotal,
        Stage::LinkCellsTotal,
        Stage::LinkHandlerSelection,
        Stage::LinkCallFixup,
        Stage::LinkFinalize,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::StartupTotal => "startup.total",
            Self::Drop => "drop",
            Self::ParserTotal => "parser.total",
            Self::ParserHeader => "parser.header",
            Self::ParserPlan => "parser.plan",
            Self::ParserCustom => "parser.section.custom",
            Self::ParserType => "parser.section.type",
            Self::ParserImport => "parser.section.import",
            Self::ParserFunction => "parser.section.function",
            Self::ParserTable => "parser.section.table",
            Self::ParserMemory => "parser.section.memory",
            Self::ParserTag => "parser.section.tag",
            Self::ParserGlobal => "parser.section.global",
            Self::ParserExport => "parser.section.export",
            Self::ParserStart => "parser.section.start",
            Self::ParserElement => "parser.section.element",
            Self::ParserDataCount => "parser.section.data_count",
            Self::ParserCode => "parser.section.code",
            Self::ParserData => "parser.section.data",
            Self::ParserFinalize => "parser.finalize",
            Self::InstanceBuildTotal => "instance.build.total",
            Self::InstanceSetup => "instance.setup",
            Self::InstanceMemories => "instance.memories",
            Self::InstanceGlobals => "instance.globals",
            Self::InstanceTables => "instance.tables",
            Self::InstanceStackDeferred => "instance.stack_deferred",
            Self::InstanceElementSegments => "instance.element_segments",
            Self::InstanceDataSegments => "instance.data_segments",
            Self::InstanceLease => "instance.lease",
            Self::PredecodeTotal => "predecode.total",
            Self::PredecodeDecode => "predecode.decode",
            Self::PredecodeScratch => "predecode.scratch",
            Self::PinnedCensus => "predecode.pinned_census",
            Self::LinkTotal => "link.total",
            Self::LinkCellsTotal => "link.cells.total",
            Self::LinkHandlerSelection => "link.handler_selection",
            Self::LinkCallFixup => "link.call_fixup",
            Self::LinkFinalize => "link.finalize",
        }
    }
}

#[derive(Clone)]
struct State {
    nanos: [u64; Stage::COUNT],
    calls: [u64; Stage::COUNT],
}

impl Default for State {
    fn default() -> Self {
        Self {
            nanos: [0; Stage::COUNT],
            calls: [0; Stage::COUNT],
        }
    }
}

std::thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

/// One immutable sample, normally one complete instantiate-and-drop cycle.
#[derive(Clone, Debug)]
pub struct Snapshot {
    nanos: [u64; Stage::COUNT],
    calls: [u64; Stage::COUNT],
}

impl Snapshot {
    pub fn entries(&self) -> impl Iterator<Item = (Stage, u64, u64)> + '_ {
        Stage::ALL.into_iter().map(|stage| {
            let index = stage as usize;
            (stage, self.nanos[index], self.calls[index])
        })
    }
}

/// Reset the current thread before a single measured startup.
pub fn reset() {
    STATE.with(|state| *state.borrow_mut() = State::default());
}

/// Copy the current thread's counters without resetting them.
pub fn snapshot() -> Snapshot {
    STATE.with(|state| {
        let state = state.borrow();
        Snapshot {
            nanos: state.nanos,
            calls: state.calls,
        }
    })
}

/// Add an externally measured interval, used for whole startup and drop.
pub fn record(stage: Stage, elapsed: Duration) {
    let nanos = elapsed.as_nanos().min(u64::MAX as u128) as u64;
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let index = stage as usize;
        state.nanos[index] = state.nanos[index].saturating_add(nanos);
        state.calls[index] = state.calls[index].saturating_add(1);
    });
}

#[inline]
pub(crate) fn measure<T>(stage: Stage, body: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let result = body();
    record(stage, start.elapsed());
    result
}

/// Scope timer used where a borrowed return value cannot cross a closure.
pub(crate) struct Span {
    stage: Stage,
    start: Instant,
}

impl Span {
    #[inline]
    pub(crate) fn new(stage: Stage) -> Self {
        Self {
            stage,
            start: Instant::now(),
        }
    }
}

impl Drop for Span {
    #[inline]
    fn drop(&mut self) {
        record(self.stage, self.start.elapsed());
    }
}
