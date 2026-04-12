use serde::Serialize;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use tracked_alloc::{self, AllocationProfile, GlobalAllocationProfile, ProfileEventKind};

pub struct Session {
    output_path: PathBuf,
    command_line: Vec<String>,
}

impl Session {
    pub fn new(output_path: Option<PathBuf>, command_line: &[String]) -> Self {
        tracked_alloc::reset_tracking();
        tracked_alloc::reset_global_tracking();
        tracked_alloc::set_global_tracking_enabled(true);
        tracked_alloc::set_tracking_enabled(true);
        Self {
            output_path: absolutize_report_path(output_path.unwrap_or_else(default_report_path)),
            command_line: command_line.to_vec(),
        }
    }

    pub fn finish(self) -> Result<PathBuf, String> {
        tracked_alloc::set_global_tracking_enabled(false);
        tracked_alloc::set_tracking_enabled(false);
        let global_profile = tracked_alloc::global_profile();
        let profile = tracked_alloc::profile();
        let document = ReportDocument::from_profile(profile, global_profile, &self.command_line);
        write_html_report(&self.output_path, &document)?;
        Ok(self.output_path)
    }
}

fn default_report_path() -> PathBuf {
    std::env::temp_dir().join(format!("sf-nano-memprof-{}.html", std::process::id()))
}

fn absolutize_report_path(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

#[derive(Serialize)]
struct ReportDocument {
    command_line: Vec<String>,
    summary: ReportSummary,
    initial_index: usize,
    timeline: Vec<ReportPoint>,
    global_timeline: Vec<ReportGlobalPoint>,
    phases: Vec<ReportPhase>,
    stacks: Vec<ReportStack>,
    events: Vec<ReportEvent>,
}

impl ReportDocument {
    fn from_profile(
        profile: AllocationProfile,
        global_profile: GlobalAllocationProfile,
        command_line: &[String],
    ) -> Self {
        let mut peak_index = 0usize;
        let mut peak_bytes = 0usize;
        let mut peak_time_ns = 0u64;
        for (index, point) in profile.timeline.iter().enumerate() {
            if point.total_bytes >= peak_bytes {
                peak_bytes = point.total_bytes;
                peak_time_ns = point.time_ns;
                peak_index = index;
            }
        }
        let initial_index = if profile.timeline.is_empty() {
            0
        } else {
            peak_index.min(profile.timeline.len().saturating_sub(1))
        };
        let summary = ReportSummary {
            peak_bytes,
            peak_time_ns,
            final_bytes: profile.snapshot.total_bytes,
            live_records: profile.snapshot.records.len(),
            event_count: profile.events.len(),
            phase_count: profile.phases.len(),
            final_time_ns: profile.now_ns,
            global_peak_bytes: global_profile.peak_bytes,
            global_final_bytes: global_profile.final_bytes,
            global_final_time_ns: global_profile.now_ns,
        };
        let timeline = profile
            .timeline
            .into_iter()
            .map(|point| ReportPoint {
                time_ns: point.time_ns,
                total_bytes: point.total_bytes,
                live_records: point.live_records,
            })
            .collect();
        let global_timeline = global_profile
            .timeline
            .into_iter()
            .map(|point| ReportGlobalPoint {
                time_ns: point.time_ns,
                live_bytes: point.live_bytes,
            })
            .collect();
        let phases = profile
            .phases
            .into_iter()
            .map(|phase| ReportPhase {
                name: phase.name,
                start_time_ns: phase.start_time_ns,
                end_time_ns: phase.end_time_ns,
                function_index: phase.function_index,
            })
            .collect();
        let stacks = profile
            .stacks
            .into_iter()
            .map(|stack| ReportStack {
                id: stack.id,
                text: stack.text.into(),
            })
            .collect();
        let events = profile
            .events
            .into_iter()
            .map(|event| ReportEvent {
                time_ns: event.time_ns,
                total_bytes: event.total_bytes,
                live_records: event.live_records,
                kind: profile_event_kind_name(event.kind),
                id: event.id,
                owner_kind: event.owner_kind,
                type_name: event.type_name,
                element_type: event.element_type,
                len: event.len,
                capacity: event.capacity,
                size_bytes: event.size_bytes,
                ptr: event.ptr,
                create_stack_id: event.create_stack_id,
                last_update_stack_id: event.last_update_stack_id,
            })
            .collect();
        Self {
            command_line: command_line.to_vec(),
            summary,
            initial_index,
            timeline,
            global_timeline,
            phases,
            stacks,
            events,
        }
    }
}

#[derive(Serialize)]
struct ReportSummary {
    peak_bytes: usize,
    peak_time_ns: u64,
    final_bytes: usize,
    live_records: usize,
    event_count: usize,
    phase_count: usize,
    final_time_ns: u64,
    global_peak_bytes: usize,
    global_final_bytes: usize,
    global_final_time_ns: u64,
}

#[derive(Serialize)]
struct ReportPoint {
    time_ns: u64,
    total_bytes: usize,
    live_records: usize,
}

#[derive(Serialize)]
struct ReportGlobalPoint {
    time_ns: u64,
    live_bytes: usize,
}

#[derive(Serialize)]
struct ReportPhase {
    name: &'static str,
    start_time_ns: u64,
    end_time_ns: u64,
    function_index: Option<u32>,
}

#[derive(Serialize)]
struct ReportStack {
    id: u64,
    text: String,
}

#[derive(Serialize)]
struct ReportEvent {
    time_ns: u64,
    total_bytes: usize,
    live_records: usize,
    kind: &'static str,
    id: u64,
    owner_kind: Option<&'static str>,
    type_name: Option<&'static str>,
    element_type: Option<&'static str>,
    len: Option<usize>,
    capacity: Option<usize>,
    size_bytes: Option<usize>,
    ptr: Option<usize>,
    create_stack_id: Option<u64>,
    last_update_stack_id: Option<u64>,
}

fn profile_event_kind_name(kind: ProfileEventKind) -> &'static str {
    match kind {
        ProfileEventKind::Create => "create",
        ProfileEventKind::Update => "update",
        ProfileEventKind::Remove => "remove",
    }
}

const HTML_PREFIX: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Silverfir-nano Memory Profile</title>
<style>
body {
  margin: 0;
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #0f1720;
  color: #dbe6f3;
}
main {
  max-width: 1280px;
  margin: 0 auto;
  padding: 24px;
}
h1, h2, h3 {
  margin: 0 0 12px;
  font-weight: 600;
}
p {
  margin: 0 0 12px;
  color: #9fb2c9;
}
.section-note {
  margin-top: -4px;
  margin-bottom: 12px;
  font-size: 13px;
  color: #8da3bc;
}
section {
  margin-bottom: 28px;
}
.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}
.stat {
  background: #142030;
  border: 1px solid #223247;
  border-radius: 8px;
  padding: 12px 14px;
}
.stat-label {
  font-size: 12px;
  color: #8da3bc;
  margin-bottom: 6px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.stat-value {
  font-size: 20px;
  font-weight: 600;
}
.curve-wrap {
  background: #142030;
  border: 1px solid #223247;
  border-radius: 8px;
  padding: 12px;
}
.phase-timeline {
  display: grid;
  gap: 6px;
  margin-bottom: 12px;
}
.phase-stage-row {
  display: grid;
  grid-template-columns: 140px minmax(0, 1fr);
  gap: 10px;
  align-items: start;
}
.phase-stage-label {
  appearance: none;
  border: 1px solid #223247;
  background: #0f1720;
  color: #dbe6f3;
  border-radius: 8px;
  padding: 8px 10px;
  text-align: left;
  font: inherit;
  cursor: pointer;
}
.phase-stage-label.is-active {
  border-color: #34d399;
  box-shadow: inset 0 0 0 1px rgba(52, 211, 153, 0.28);
}
.phase-stage-label:focus-visible,
.phase-bar:focus-visible {
  outline: 2px solid rgba(124, 197, 255, 0.9);
  outline-offset: 1px;
}
.phase-stage-label strong {
  display: block;
  font-size: 13px;
  line-height: 1.2;
}
.phase-stage-meta {
  display: block;
  margin-top: 2px;
  font-size: 11px;
  color: #8da3bc;
}
.phase-row {
  position: relative;
  height: 28px;
  background: #0f1720;
  border: 1px solid #223247;
  border-radius: 8px;
  overflow: hidden;
}
.phase-stage-tracks {
  display: grid;
  gap: 6px;
}
.phase-selected-marker {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 1px;
  background: rgba(52, 211, 153, 0.9);
  pointer-events: none;
}
.phase-bar {
  position: absolute;
  top: 4px;
  bottom: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  box-sizing: border-box;
  appearance: none;
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: #dbe6f3;
  font-size: 12px;
  font-weight: 600;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  cursor: pointer;
  user-select: none;
}
.phase-bar:hover,
.phase-stage-label:hover {
  border-color: #3a5678;
}
.phase-bar.is-active {
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.35);
}
#phase-note[hidden],
#phase-timeline[hidden] {
  display: none;
}
#curve {
  width: 100%;
  height: 320px;
  display: block;
  cursor: crosshair;
}
.curve-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 12px;
  color: #9fb2c9;
  font-size: 13px;
}
.curve-legend-item {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  user-select: none;
}
.curve-legend-toggle {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}
.curve-legend-toggle input {
  margin: 0;
}
.curve-legend-swatch {
  width: 14px;
  height: 3px;
  border-radius: 2px;
}
.selection {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin-top: 14px;
}
.selection-item {
  background: #0f1720;
  border: 1px solid #223247;
  border-radius: 8px;
  padding: 10px 12px;
}
.selection-item strong {
  display: block;
  margin-bottom: 4px;
  font-size: 12px;
  color: #8da3bc;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.tables {
  display: grid;
  grid-template-columns: 1fr;
  gap: 24px;
}
.table-panel {
  min-width: 0;
}
table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
  font-size: 13px;
  background: #142030;
  border: 1px solid #223247;
  border-radius: 8px;
  overflow: hidden;
}
thead {
  background: #17263a;
}
th, td {
  padding: 8px 10px;
  border-bottom: 1px solid #223247;
  text-align: left;
  vertical-align: top;
  overflow-wrap: anywhere;
  word-break: break-word;
}
tbody tr:last-child td {
  border-bottom: none;
}
tbody tr.record-row {
  cursor: pointer;
}
tbody tr.record-row:hover {
  background: #18283d;
}
tbody tr.record-row.is-selected {
  background: #1d3048;
}
.table-breakdown th:nth-child(1),
.table-breakdown td:nth-child(1) {
  width: 56%;
}
.table-breakdown th:nth-child(2),
.table-breakdown td:nth-child(2) {
  width: 14%;
  white-space: nowrap;
}
.table-breakdown th:nth-child(3),
.table-breakdown td:nth-child(3) {
  width: 10%;
}
.table-breakdown th:nth-child(4),
.table-breakdown td:nth-child(4) {
  width: 20%;
  white-space: nowrap;
}
.table-records th:nth-child(1),
.table-records td:nth-child(1) {
  width: 58%;
}
.table-records th:nth-child(2),
.table-records td:nth-child(2) {
  width: 12%;
  white-space: nowrap;
}
.table-records th:nth-child(3),
.table-records td:nth-child(3),
.table-records th:nth-child(4),
.table-records td:nth-child(4) {
  width: 7%;
}
.table-records th:nth-child(5),
.table-records td:nth-child(5) {
  width: 16%;
  white-space: nowrap;
}
.inspector-window {
  position: fixed;
  top: 88px;
  right: 24px;
  width: min(560px, calc(100vw - 32px));
  max-height: min(70vh, 720px);
  display: none;
  flex-direction: column;
  background: #142030;
  border: 1px solid #223247;
  border-radius: 8px;
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.35);
  z-index: 20;
  overflow: hidden;
}
.inspector-window.is-open {
  display: flex;
}
.inspector-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px;
  border-bottom: 1px solid #223247;
  cursor: move;
  user-select: none;
}
.inspector-title {
  font-size: 14px;
  font-weight: 600;
  color: #dbe6f3;
}
.inspector-close {
  border: 1px solid #2b435f;
  background: #0f1720;
  color: #dbe6f3;
  border-radius: 6px;
  padding: 4px 10px;
  font: inherit;
  cursor: pointer;
}
.inspector-body {
  padding: 12px;
  overflow: auto;
}
pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: ui-monospace, SFMono-Regular, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  color: #c7d7ea;
}
.muted {
  color: #8da3bc;
}
@media (max-width: 960px) {
  .phase-stage-row {
    grid-template-columns: 1fr;
  }
  .table-breakdown th:nth-child(2),
  .table-breakdown td:nth-child(2),
  .table-breakdown th:nth-child(4),
  .table-breakdown td:nth-child(4),
  .table-records th:nth-child(2),
  .table-records td:nth-child(2),
  .table-records th:nth-child(5),
  .table-records td:nth-child(5) {
    white-space: normal;
  }
}
</style>
</head>
<body>
<main>
  <section>
    <h1>Memory Profile</h1>
    <p id="command-line"></p>
    <div class="stats" id="summary-stats"></div>
  </section>
  <section class="curve-wrap">
    <h2>Total Usage Curve</h2>
    <p class="section-note" id="phase-note">Compiler phase navigator, zoomed to the phase span. Click a stage for that stage's peak, or a segment for a specific span.</p>
    <div class="phase-timeline" id="phase-timeline" hidden></div>
    <canvas id="curve"></canvas>
    <div class="curve-legend" id="curve-legend"></div>
    <div class="selection" id="selection-summary"></div>
  </section>
  <section class="tables">
    <div class="table-panel">
      <h2>Type Summary</h2>
      <p class="section-note">Groups the live allocations at the selected time by type.</p>
      <table class="table-breakdown">
        <thead>
          <tr><th>Type</th><th>Total</th><th>Count</th><th>Largest</th></tr>
        </thead>
        <tbody id="aggregate-body"></tbody>
      </table>
    </div>
    <div class="table-panel">
      <h2>Individual Allocations</h2>
      <p class="section-note">Shows the largest live allocations at the selected time. Click a row to inspect it in a floating window.</p>
      <table class="table-records">
        <thead>
          <tr><th>Type</th><th>Bytes</th><th>Len</th><th>Cap</th><th>Ptr</th></tr>
        </thead>
        <tbody id="records-body"></tbody>
      </table>
      <p class="muted" id="records-note"></p>
    </div>
  </section>
</main>
<div class="inspector-window" id="record-inspector" aria-hidden="true">
  <div class="inspector-header" id="record-inspector-header">
    <div class="inspector-title">Allocation Inspector</div>
    <button type="button" class="inspector-close" id="record-inspector-close">X</button>
  </div>
  <div class="inspector-body">
    <pre id="record-detail">Click an allocation row to inspect its creation site and latest size-change site.</pre>
  </div>
</div>
<script id="report-data" type="application/json">
"#;

const HTML_SUFFIX: &str = r#"
</script>
<script>
const report = JSON.parse(document.getElementById('report-data').textContent);
const stackById = new Map((report.stacks || []).map((stack) => [stack.id, stack.text]));
const canvas = document.getElementById('curve');
const ctx = canvas.getContext('2d');
const curveLegend = document.getElementById('curve-legend');
const summaryStats = document.getElementById('summary-stats');
const commandLine = document.getElementById('command-line');
const phaseNote = document.getElementById('phase-note');
const phaseTimeline = document.getElementById('phase-timeline');
const selectionSummary = document.getElementById('selection-summary');
const aggregateBody = document.getElementById('aggregate-body');
const recordsBody = document.getElementById('records-body');
const recordsNote = document.getElementById('records-note');
const recordInspector = document.getElementById('record-inspector');
const recordInspectorHeader = document.getElementById('record-inspector-header');
const recordInspectorClose = document.getElementById('record-inspector-close');
const recordDetail = document.getElementById('record-detail');
const phaseStageOrder = [
  'sem_decode',
  'sem_inline',
  'ssa_lower',
  'cfg_lower',
  'slot_lower',
  'joint_plan',
  'ssa_rewrite',
  'mir_lower',
  'module_opt',
  'arch_lower',
];

let selectedIndex = Math.max(0, Math.min(report.initial_index, report.timeline.length - 1));
let selectedRecords = [];
let inspectorDrag = null;
const reportPhases = (report.phases || [])
  .map((phase, index) => ({
    ...phase,
    __index: index,
    start_time_ns: Number(phase.start_time_ns) || 0,
    end_time_ns: Number(phase.end_time_ns) || 0,
  }))
  .sort((left, right) => {
    if (left.start_time_ns !== right.start_time_ns) {
      return left.start_time_ns - right.start_time_ns;
    }
    if (left.end_time_ns !== right.end_time_ns) {
      return left.end_time_ns - right.end_time_ns;
    }
    return String(left.name).localeCompare(String(right.name));
  });
const maxTimelineTimeNs = Math.max(
  report.timeline.length ? Number(report.timeline[report.timeline.length - 1].time_ns) : 0,
  report.global_timeline.length
    ? Number(report.global_timeline[report.global_timeline.length - 1].time_ns)
    : 0,
  reportPhases.length
    ? reportPhases.reduce(
        (maxTimeNs, phase) => Math.max(maxTimeNs, Number(phase.end_time_ns) || 0),
        0
      )
    : 0,
  1
);
const maxTimelineBytes = (() => {
  let maxBytes = 1;
  for (const point of report.timeline) {
    const bytes = Number(point.total_bytes) || 0;
    if (bytes > maxBytes) {
      maxBytes = bytes;
    }
  }
  for (const point of report.global_timeline || []) {
    const bytes = Number(point.live_bytes) || 0;
    if (bytes > maxBytes) {
      maxBytes = bytes;
    }
  }
  return maxBytes;
})();
const curvePadding = {
  left: 56,
  right: 14,
  top: 12,
  bottom: 28,
};
const showCodeBufferSeries = report.events.some(
  (event) => event.owner_kind === 'RuntimeMemory' && event.type_name === 'CodeBuffer'
);
const showGuardPageSeries = report.events.some(
  (event) => event.owner_kind === 'RuntimeMemory' && event.type_name === 'GuardPageMemory'
);
const showRuntimeOtherSeries = report.events.some(
  (event) =>
    event.owner_kind === 'RuntimeMemory' &&
    event.type_name !== 'CodeBuffer' &&
    event.type_name !== 'GuardPageMemory'
);
const showGlobalAllocatorSeries =
  (report.global_timeline && report.global_timeline.length > 1) ||
  Number(report.summary.global_peak_bytes || 0) > 0;
const curveSeriesMeta = [
  { key: 'total_bytes', label: 'Tracked total', color: '#7cc5ff', width: 2.5, visible: true },
  {
    key: 'global_bytes',
    label: 'Global allocator',
    color: '#e5e7eb',
    width: 2,
    visible: showGlobalAllocatorSeries,
  },
  {
    key: 'tracked_alloc_bytes',
    label: 'Tracked alloc',
    color: '#f59e0b',
    width: 1.5,
    visible: true,
  },
  {
    key: 'code_buffer_bytes',
    label: 'Code Buffer',
    color: '#34d399',
    width: 1.5,
    visible: showCodeBufferSeries,
  },
  {
    key: 'guard_page_bytes',
    label: 'Guard Pages',
    color: '#f87171',
    width: 1.5,
    visible: showGuardPageSeries,
  },
  {
    key: 'runtime_other_bytes',
    label: 'Runtime other',
    color: '#c084fc',
    width: 1.5,
    visible: showRuntimeOtherSeries,
  },
];
const curveSeriesState = new Map(
  curveSeriesMeta
    .filter((series) => series.visible)
    .map((series) => [series.key, true])
);
let cachedCurveBuckets = null;
let cachedCurveBucketCount = 0;

function curveMetrics() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const padLeft = curvePadding.left;
  const padRight = curvePadding.right;
  const padTop = curvePadding.top;
  const padBottom = curvePadding.bottom;
  const plotWidth = Math.max(1, width - padLeft - padRight);
  const plotHeight = Math.max(1, height - padTop - padBottom);
  return {
    width,
    height,
    padLeft,
    padRight,
    padTop,
    padBottom,
    plotWidth,
    plotHeight,
    plotLeft: padLeft,
    plotRight: padLeft + plotWidth,
  };
}

function allocationCategory(record) {
  if (record.owner_kind === 'RuntimeMemory') {
    if (record.type_name === 'CodeBuffer') {
      return 'code_buffer';
    }
    if (record.type_name === 'GuardPageMemory') {
      return 'guard_page';
    }
    return 'runtime_other';
  }
  return 'tracked_alloc';
}

function emptyBreakdown() {
  return {
    tracked_alloc_bytes: 0,
    code_buffer_bytes: 0,
    guard_page_bytes: 0,
    runtime_other_bytes: 0,
  };
}

function applyBreakdownDelta(breakdown, category, deltaBytes) {
  if (!deltaBytes) {
    return;
  }
  switch (category) {
    case 'code_buffer':
      breakdown.code_buffer_bytes = Math.max(0, breakdown.code_buffer_bytes + deltaBytes);
      break;
    case 'guard_page':
      breakdown.guard_page_bytes = Math.max(0, breakdown.guard_page_bytes + deltaBytes);
      break;
    case 'runtime_other':
      breakdown.runtime_other_bytes = Math.max(0, breakdown.runtime_other_bytes + deltaBytes);
      break;
    default:
      breakdown.tracked_alloc_bytes = Math.max(0, breakdown.tracked_alloc_bytes + deltaBytes);
      break;
  }
}

function breakdownTotal(breakdown) {
  return (
    breakdown.tracked_alloc_bytes +
    breakdown.code_buffer_bytes +
    breakdown.guard_page_bytes +
    breakdown.runtime_other_bytes
  );
}

function breakdownForRecords(records) {
  const breakdown = emptyBreakdown();
  for (const record of records) {
    applyBreakdownDelta(
      breakdown,
      allocationCategory(record),
      Number(record.size_bytes) || 0
    );
  }
  return breakdown;
}

function renderCurveLegend() {
  curveLegend.innerHTML = curveSeriesMeta
    .filter((series) => series.visible)
    .map(
      (series) => `
        <span class="curve-legend-item">
          <label class="curve-legend-toggle">
            <input
              type="checkbox"
              data-series-key="${escapeHtml(series.key)}"
              ${curveSeriesState.get(series.key) ? 'checked' : ''}
            >
            <span class="curve-legend-swatch" style="background:${series.color}"></span>
            <span>${escapeHtml(series.label)}</span>
          </label>
        </span>
      `
    )
    .join('');
  for (const input of curveLegend.querySelectorAll('input[data-series-key]')) {
    input.addEventListener('change', () => {
      const key = input.getAttribute('data-series-key');
      if (!key) {
        return;
      }
      curveSeriesState.set(key, input.checked);
      drawCurve();
    });
  }
}

function snapshotBreakdownPoint(breakdown) {
  return {
    total_bytes: breakdownTotal(breakdown),
    global_bytes: 0,
    tracked_alloc_bytes: breakdown.tracked_alloc_bytes,
    code_buffer_bytes: breakdown.code_buffer_bytes,
    guard_page_bytes: breakdown.guard_page_bytes,
    runtime_other_bytes: breakdown.runtime_other_bytes,
  };
}

function buildTrackedCurveBuckets(bucketCount) {
  const safeBucketCount = Math.max(2, bucketCount);
  const buckets = new Array(safeBucketCount);
  const live = new Map();
  const breakdown = emptyBreakdown();

  buckets[0] = snapshotBreakdownPoint(breakdown);

  for (const event of report.events) {
    if (event.kind === 'create') {
      const category = allocationCategory(event);
      const sizeBytes = Number(event.size_bytes) || 0;
      live.set(event.id, { category, size_bytes: sizeBytes });
      applyBreakdownDelta(breakdown, category, sizeBytes);
    } else if (event.kind === 'update') {
      const liveRecord = live.get(event.id);
      if (liveRecord) {
        const nextSizeBytes = Number(event.size_bytes) || 0;
        applyBreakdownDelta(
          breakdown,
          liveRecord.category,
          nextSizeBytes - liveRecord.size_bytes
        );
        liveRecord.size_bytes = nextSizeBytes;
      }
    } else if (event.kind === 'remove') {
      const liveRecord = live.get(event.id);
      if (liveRecord) {
        applyBreakdownDelta(breakdown, liveRecord.category, -liveRecord.size_bytes);
        live.delete(event.id);
      }
    }

    const bucketIndex = Math.max(
      0,
      Math.min(
        safeBucketCount - 1,
        Math.floor((Number(event.time_ns) / maxTimelineTimeNs) * (safeBucketCount - 1))
      )
    );
    buckets[bucketIndex] = snapshotBreakdownPoint(breakdown);
  }

  let last = buckets[0];
  for (let index = 0; index < safeBucketCount; index += 1) {
    if (buckets[index]) {
      last = buckets[index];
    } else {
      buckets[index] = { ...last };
    }
  }

  return buckets;
}

function buildGlobalCurveBuckets(bucketCount) {
  const safeBucketCount = Math.max(2, bucketCount);
  const buckets = new Array(safeBucketCount).fill(0);
  if (!report.global_timeline.length) {
    return buckets;
  }
  let pointIndex = 0;
  let currentBytes = Number(report.global_timeline[0].live_bytes) || 0;
  for (let index = 0; index < safeBucketCount; index += 1) {
    const timeNs = (index / Math.max(1, safeBucketCount - 1)) * maxTimelineTimeNs;
    while (
      pointIndex + 1 < report.global_timeline.length &&
      Number(report.global_timeline[pointIndex + 1].time_ns) <= timeNs
    ) {
      pointIndex += 1;
      currentBytes = Number(report.global_timeline[pointIndex].live_bytes) || 0;
    }
    buckets[index] = currentBytes;
  }
  return buckets;
}

function globalBytesAtTimeNs(timeNs) {
  if (!report.global_timeline.length) {
    return 0;
  }
  let low = 0;
  let high = report.global_timeline.length - 1;
  while (low < high) {
    const mid = Math.floor((low + high + 1) / 2);
    if (Number(report.global_timeline[mid].time_ns) <= timeNs) {
      low = mid;
    } else {
      high = mid - 1;
    }
  }
  return Number(report.global_timeline[low].live_bytes) || 0;
}

function curveBucketsForPlotWidth(plotWidth) {
  const bucketCount = Math.max(2, Math.floor(plotWidth));
  if (cachedCurveBuckets && cachedCurveBucketCount === bucketCount) {
    return cachedCurveBuckets;
  }
  cachedCurveBuckets = {
    tracked: buildTrackedCurveBuckets(bucketCount),
    global: buildGlobalCurveBuckets(bucketCount),
  };
  cachedCurveBucketCount = bucketCount;
  return cachedCurveBuckets;
}

function seriesIsEnabled(series) {
  return series.visible && curveSeriesState.get(series.key) !== false;
}

function enabledCurveSeries() {
  return curveSeriesMeta.filter((series) => seriesIsEnabled(series));
}

function curveValueAtIndex(series, trackedCurveBuckets, globalCurveBuckets, index) {
  if (series.key === 'global_bytes') {
    return globalCurveBuckets[index] || 0;
  }
  const point = trackedCurveBuckets[index];
  return point ? point[series.key] || 0 : 0;
}

function currentMaxBytes(trackedCurveBuckets, globalCurveBuckets) {
  let maxBytes = 1;
  for (const series of enabledCurveSeries()) {
    const bucketCount =
      series.key === 'global_bytes' ? globalCurveBuckets.length : trackedCurveBuckets.length;
    for (let index = 0; index < bucketCount; index += 1) {
      maxBytes = Math.max(
        maxBytes,
        curveValueAtIndex(series, trackedCurveBuckets, globalCurveBuckets, index)
      );
    }
  }
  return maxBytes;
}

function selectedMarkerValue(point, breakdown, globalBytes) {
  const visibleSeries = enabledCurveSeries();
  if (!visibleSeries.length) {
    return null;
  }
  for (const series of visibleSeries) {
    switch (series.key) {
      case 'global_bytes':
        return globalBytes;
      case 'tracked_alloc_bytes':
        return breakdown.tracked_alloc_bytes;
      case 'code_buffer_bytes':
        return breakdown.code_buffer_bytes;
      case 'guard_page_bytes':
        return breakdown.guard_page_bytes;
      case 'runtime_other_bytes':
        return breakdown.runtime_other_bytes;
      default:
        return point.total_bytes;
    }
  }
  return point.total_bytes;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function formatBytes(bytes) {
  const units = ['B', 'KiB', 'MiB', 'GiB'];
  let value = Number(bytes);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)}\u00a0${units[unit]}`;
}

function formatTime(timeNs) {
  return `${(Number(timeNs) / 1e6).toFixed(3)} ms`;
}

function phaseStageRank(name) {
  const rank = phaseStageOrder.indexOf(name);
  return rank === -1 ? phaseStageOrder.length : rank;
}

function phaseLabel(phase) {
  if (phase.function_index == null) {
    return phase.name;
  }
  return `${phase.name} #${phase.function_index}`;
}

function phaseBarLabel(phase) {
  if (phase.function_index == null) {
    return phase.name;
  }
  return `#${phase.function_index}`;
}

function phaseDurationNs(phase) {
  return Math.max(0, Number(phase.end_time_ns) - Number(phase.start_time_ns));
}

function phaseColor(name) {
  switch (name) {
    case 'sem_decode':
      return '#38bdf8';
    case 'sem_inline':
      return '#818cf8';
    case 'cfg_lower':
      return '#2dd4bf';
    case 'slot_lower':
      return '#22c55e';
    case 'joint_plan':
      return '#f59e0b';
    case 'ssa_lower':
      return '#f97316';
    case 'ssa_rewrite':
      return '#fb923c';
    case 'mir_lower':
      return '#ef4444';
    case 'module_opt':
      return '#f43f5e';
    case 'arch_lower':
      return '#a855f7';
    default:
      return '#64748b';
  }
}

function buildPhaseStageRows() {
  const rowsByName = new Map();
  for (const phase of reportPhases) {
    let row = rowsByName.get(phase.name);
    if (!row) {
      row = {
        name: phase.name,
        spans: [],
      };
      rowsByName.set(phase.name, row);
    }
    row.spans.push(phase);
  }
  return Array.from(rowsByName.values())
    .map((row) => {
      row.spans.sort((left, right) => {
        if (left.start_time_ns !== right.start_time_ns) {
          return left.start_time_ns - right.start_time_ns;
        }
        if (left.end_time_ns !== right.end_time_ns) {
          return left.end_time_ns - right.end_time_ns;
        }
        return (left.function_index ?? -1) - (right.function_index ?? -1);
      });
      row.start_time_ns = row.spans.length ? row.spans[0].start_time_ns : 0;
      row.end_time_ns = row.spans.reduce(
        (endTimeNs, phase) => Math.max(endTimeNs, phase.end_time_ns),
        row.start_time_ns
      );
      row.total_duration_ns = row.spans.reduce(
        (totalDurationNs, phase) => totalDurationNs + phaseDurationNs(phase),
        0
      );
      return row;
    })
    .sort((left, right) => {
      const leftRank = phaseStageRank(left.name);
      const rightRank = phaseStageRank(right.name);
      if (leftRank !== rightRank) {
        return leftRank - rightRank;
      }
      if (left.start_time_ns !== right.start_time_ns) {
        return left.start_time_ns - right.start_time_ns;
      }
      return String(left.name).localeCompare(String(right.name));
    });
}

const phaseStageRows = buildPhaseStageRows();
const phaseDomainStartTimeNs = phaseStageRows.length
  ? phaseStageRows.reduce(
      (startTimeNs, row) => Math.min(startTimeNs, Number(row.start_time_ns) || 0),
      Number.POSITIVE_INFINITY
    )
  : 0;
const phaseDomainEndTimeNs = phaseStageRows.length
  ? phaseStageRows.reduce(
      (endTimeNs, row) => Math.max(endTimeNs, Number(row.end_time_ns) || 0),
      0
    )
  : 0;
const phaseDomainDurationNs = Math.max(1, phaseDomainEndTimeNs - phaseDomainStartTimeNs);

function phasePercent(timeNs) {
  return ((Number(timeNs) - phaseDomainStartTimeNs) / phaseDomainDurationNs) * 100;
}

function phaseTrackLayoutWidthPx() {
  const timelineWidth = Math.max(
    1,
    phaseTimeline.getBoundingClientRect().width || canvas.clientWidth || 1
  );
  if (window.innerWidth <= 960) {
    return timelineWidth;
  }
  return Math.max(1, timelineWidth - 150);
}

function layoutPhaseTracks(spans) {
  const tracks = [];
  for (const phase of spans) {
    const left = Math.max(0, Math.min(100, phasePercent(phase.start_time_ns)));
    const actualWidth = (phaseDurationNs(phase) / phaseDomainDurationNs) * 100;
    const item = {
      phase,
      left,
      actualWidth,
      right: left + actualWidth,
    };
    let placed = false;
    for (const track of tracks) {
      const previous = track[track.length - 1];
      if (item.left >= previous.right) {
        track.push(item);
        placed = true;
        break;
      }
    }
    if (!placed) {
      tracks.push([item]);
    }
  }
  return tracks;
}

function activePhasesAtTimeNs(timeNs) {
  return reportPhases
    .filter((phase) => phase.start_time_ns <= timeNs && timeNs <= phase.end_time_ns)
    .sort((left, right) => {
      if ((left.function_index == null) !== (right.function_index == null)) {
        return left.function_index == null ? -1 : 1;
      }
      const leftDuration = phaseDurationNs(left);
      const rightDuration = phaseDurationNs(right);
      if (leftDuration !== rightDuration) {
        return rightDuration - leftDuration;
      }
      return phaseLabel(left).localeCompare(phaseLabel(right));
    });
}

function phasePeakIndex(phase) {
  return peakIndexForSpans([phase]);
}

function peakIndexForSpans(spans) {
  if (!report.timeline.length) {
    return 0;
  }
  if (!spans.length) {
    return selectedIndex;
  }
  let bestIndex = nearestTimelineIndex(spans[0].start_time_ns);
  let bestBytes = -1;
  let found = false;
  let spanIndex = 0;
  for (let index = 0; index < report.timeline.length; index += 1) {
    const point = report.timeline[index];
    const timeNs = Number(point.time_ns) || 0;
    while (spanIndex < spans.length && timeNs > spans[spanIndex].end_time_ns) {
      spanIndex += 1;
    }
    if (spanIndex >= spans.length) {
      break;
    }
    if (timeNs < spans[spanIndex].start_time_ns) {
      continue;
    }
    const bytes = Number(point.total_bytes) || 0;
    if (!found || bytes >= bestBytes) {
      found = true;
      bestBytes = bytes;
      bestIndex = index;
    }
  }
  return bestIndex;
}

function renderPhaseTimeline() {
  if (!phaseStageRows.length) {
    phaseNote.hidden = true;
    phaseTimeline.hidden = true;
    return;
  }
  phaseNote.hidden = false;
  phaseTimeline.hidden = false;
  const selectedPoint = report.timeline[selectedIndex];
  const selectedTimeNs = selectedPoint ? Number(selectedPoint.time_ns) || 0 : 0;
  const trackWidthPx = phaseTrackLayoutWidthPx();
  const selectedMarkerPercent =
    selectedTimeNs >= phaseDomainStartTimeNs && selectedTimeNs <= phaseDomainEndTimeNs
      ? Math.max(0, Math.min(100, phasePercent(selectedTimeNs)))
      : null;
  phaseTimeline.innerHTML = phaseStageRows
    .map(
      (row) => {
        const rowActive = row.spans.some(
          (phase) => selectedTimeNs >= phase.start_time_ns && selectedTimeNs <= phase.end_time_ns
        );
        const tracks = layoutPhaseTracks(row.spans);
        return `
          <div class="phase-stage-row">
            <button
              type="button"
              class="phase-stage-label ${rowActive ? 'is-active' : ''}"
              data-phase-name="${escapeHtml(row.name)}"
              title="${escapeHtml(
                `${row.name}\nSpans: ${row.spans.length}\nTotal duration: ${formatTime(
                  row.total_duration_ns
                )}\nStart: ${formatTime(row.start_time_ns)}\nEnd: ${formatTime(row.end_time_ns)}`
              )}"
            >
              <strong>${escapeHtml(row.name)}</strong>
              <span class="phase-stage-meta">${escapeHtml(
                `${row.spans.length} span${row.spans.length === 1 ? '' : 's'} · ${formatTime(
                  row.total_duration_ns
                )}`
              )}</span>
            </button>
            <div class="phase-stage-tracks">
              ${tracks
                .map(
                  (track) => `
                    <div class="phase-row">
                      ${
                        selectedMarkerPercent == null
                          ? ''
                          : `<div class="phase-selected-marker" style="left:${selectedMarkerPercent}%"></div>`
                      }
                      ${track
                        .map((item) => {
                          const phase = item.phase;
                          const active =
                            selectedTimeNs >= phase.start_time_ns && selectedTimeNs <= phase.end_time_ns;
                          const label =
                            item.actualWidth * trackWidthPx / 100 >= 34 ? phaseBarLabel(phase) : '';
                          return `
                            <button
                              type="button"
                              class="phase-bar ${active ? 'is-active' : ''}"
                              data-phase-index="${phase.__index}"
                              title="${escapeHtml(
                                `${phaseLabel(phase)}\nStart: ${formatTime(
                                  phase.start_time_ns
                                )}\nEnd: ${formatTime(phase.end_time_ns)}\nDuration: ${formatTime(
                                  phaseDurationNs(phase)
                                )}`
                              )}"
                              style="left:${item.left}%;width:${item.actualWidth}%;background:${phaseColor(
                                phase.name
                              )}"
                            >${escapeHtml(label)}</button>
                          `;
                        })
                        .join('')}
                    </div>
                  `
                )
                .join('')}
            </div>
          </div>
        `;
      }
    )
    .join('');
  for (const button of phaseTimeline.querySelectorAll('button[data-phase-name]')) {
    button.addEventListener('click', () => {
      const phaseName = button.getAttribute('data-phase-name');
      if (!phaseName) {
        return;
      }
      const row = phaseStageRows.find((entry) => entry.name === phaseName);
      if (!row) {
        return;
      }
      renderSelection(peakIndexForSpans(row.spans));
    });
  }
  for (const button of phaseTimeline.querySelectorAll('button[data-phase-index]')) {
    button.addEventListener('click', () => {
      const phaseIndex = Number(button.getAttribute('data-phase-index'));
      const phase = reportPhases.find((entry) => entry.__index === phaseIndex);
      if (!phase) {
        return;
      }
      renderSelection(phasePeakIndex(phase));
    });
  }
}

function displayType(record) {
  if (record.owner_kind === 'String') {
    return 'String';
  }
  if (record.owner_kind === 'Vec' && record.element_type) {
    return `Vec<${record.element_type}>`;
  }
  if (record.owner_kind === 'BTreeSet' && record.element_type) {
    return `BTreeSet<${record.element_type}>`;
  }
  if (record.owner_kind === 'Rc') {
    if (record.type_name === 'str') {
      return 'Rc<str>';
    }
    if (record.element_type && record.type_name && record.type_name.startsWith('[')) {
      return `Rc<[${record.element_type}]>`;
    }
    return record.type_name ? `Rc<${record.type_name}>` : 'Rc';
  }
  if (record.owner_kind === 'Box') {
    if (record.type_name === 'str') {
      return 'Box<str>';
    }
    if (record.element_type && record.type_name && record.type_name.startsWith('[')) {
      return `Box<[${record.element_type}]>`;
    }
    return record.type_name ? `Box<${record.type_name}>` : 'Box';
  }
  return record.type_name || record.owner_kind || 'unknown';
}

function pointerText(ptr) {
  if (!ptr) {
    return '-';
  }
  return `0x${Number(ptr).toString(16)}`;
}

function stackText(stackId) {
  if (stackId == null) {
    return '(none)';
  }
  return stackById.get(stackId) || '(missing stack)';
}

function snapshotAt(timeNs) {
  const live = new Map();
  for (const event of report.events) {
    if (event.time_ns > timeNs) {
      break;
    }
    if (event.kind === 'create') {
      live.set(event.id, {
        id: event.id,
        owner_kind: event.owner_kind,
        type_name: event.type_name,
        element_type: event.element_type,
        len: event.len,
        capacity: event.capacity,
        size_bytes: event.size_bytes || 0,
        ptr: event.ptr || 0,
        create_stack_id: event.create_stack_id ?? null,
        last_update_stack_id: null,
      });
    } else if (event.kind === 'update') {
      const record = live.get(event.id);
      if (!record) {
        continue;
      }
      record.len = event.len;
      record.capacity = event.capacity;
      if (event.size_bytes != null) {
        record.size_bytes = event.size_bytes;
      }
      if (event.ptr != null) {
        record.ptr = event.ptr;
      }
      if (event.last_update_stack_id != null) {
        record.last_update_stack_id = event.last_update_stack_id;
      }
    } else if (event.kind === 'remove') {
      live.delete(event.id);
    }
  }
  return Array.from(live.values()).sort((a, b) => {
    if (b.size_bytes !== a.size_bytes) {
      return b.size_bytes - a.size_bytes;
    }
    return a.id - b.id;
  });
}

function aggregate(records) {
  const byType = new Map();
  for (const record of records) {
    const key = displayType(record);
    let bucket = byType.get(key);
    if (!bucket) {
      bucket = { type: key, total_bytes: 0, count: 0, largest_bytes: 0 };
      byType.set(key, bucket);
    }
    bucket.total_bytes += record.size_bytes;
    bucket.count += 1;
    bucket.largest_bytes = Math.max(bucket.largest_bytes, record.size_bytes);
  }
  return Array.from(byType.values()).sort((a, b) => {
    if (b.total_bytes !== a.total_bytes) {
      return b.total_bytes - a.total_bytes;
    }
    return a.type.localeCompare(b.type);
  });
}

function renderSummaryStats() {
  const stats = [
    ['Peak', formatBytes(report.summary.peak_bytes)],
    ['Peak Time', formatTime(report.summary.peak_time_ns)],
    ['Final', formatBytes(report.summary.final_bytes)],
    ['Global Peak', formatBytes(report.summary.global_peak_bytes)],
    ['Global Final', formatBytes(report.summary.global_final_bytes)],
    ['Live Records', report.summary.live_records],
    ['Events', report.summary.event_count],
    ['Phases', report.summary.phase_count || 0],
    ['End Time', formatTime(report.summary.final_time_ns)],
  ];
  summaryStats.innerHTML = stats
    .map(([label, value]) => `
      <div class="stat">
        <div class="stat-label">${escapeHtml(label)}</div>
        <div class="stat-value">${escapeHtml(value)}</div>
      </div>
    `)
    .join('');
}

function renderRecordDetail(record) {
  if (!record) {
    recordDetail.textContent = 'No live allocations at this point.';
    return;
  }
  const parts = [
    `${displayType(record)}  ${formatBytes(record.size_bytes)}`,
    '',
    'Create site:',
    stackText(record.create_stack_id),
  ];
  if (record.last_update_stack_id != null) {
    parts.push('', 'Last size-change site:', stackText(record.last_update_stack_id));
  }
  recordDetail.textContent = parts.join('\n');
}

function hideInspector() {
  for (const previous of recordsBody.querySelectorAll('tr.record-row.is-selected')) {
    previous.classList.remove('is-selected');
  }
  recordInspector.classList.remove('is-open');
  recordInspector.setAttribute('aria-hidden', 'true');
}

function openInspector() {
  recordInspector.classList.add('is-open');
  recordInspector.setAttribute('aria-hidden', 'false');
}

function selectRecordRow(row, record, options = {}) {
  for (const previous of recordsBody.querySelectorAll('tr.record-row.is-selected')) {
    previous.classList.remove('is-selected');
  }
  if (row) {
    row.classList.add('is-selected');
  }
  renderRecordDetail(record);
  if (record) {
    openInspector();
  }
}

function renderSelection(index) {
  selectedIndex = Math.max(0, Math.min(index, report.timeline.length - 1));
  hideInspector();
  const point = report.timeline[selectedIndex] || { time_ns: 0, total_bytes: 0, live_records: 0 };
  selectedRecords = snapshotAt(point.time_ns);
  const aggregates = aggregate(selectedRecords);
  const breakdown = breakdownForRecords(selectedRecords);
  const globalBytes = globalBytesAtTimeNs(point.time_ns);
  const untrackedGlobalGap = Math.max(0, globalBytes - breakdown.tracked_alloc_bytes);
  const activePhases = activePhasesAtTimeNs(Number(point.time_ns) || 0);

  const selectionItems = [
    ['Selected Time', formatTime(point.time_ns)],
    ['Total Bytes', formatBytes(point.total_bytes)],
    ['Global allocator', formatBytes(globalBytes)],
    ['Tracked alloc', formatBytes(breakdown.tracked_alloc_bytes)],
  ];
  if (showGlobalAllocatorSeries) {
    selectionItems.push(['Untracked heap gap', formatBytes(untrackedGlobalGap)]);
  }
  if (showCodeBufferSeries) {
    selectionItems.push(['Code Buffer', formatBytes(breakdown.code_buffer_bytes)]);
  }
  if (showGuardPageSeries) {
    selectionItems.push(['Guard Pages', formatBytes(breakdown.guard_page_bytes)]);
  }
  if (showRuntimeOtherSeries) {
    selectionItems.push(['Runtime other', formatBytes(breakdown.runtime_other_bytes)]);
  }
  if (activePhases.length) {
    selectionItems.push(['Active phases', activePhases.map(phaseLabel).join(', ')]);
  }
  selectionItems.push(['Live Records', point.live_records], ['Types', aggregates.length]);

  selectionSummary.innerHTML = selectionItems
    .map(([label, value]) => `
      <div class="selection-item">
        <strong>${escapeHtml(label)}</strong>
        <span>${escapeHtml(value)}</span>
      </div>
    `)
    .join('');

  aggregateBody.innerHTML = aggregates.length
    ? aggregates
        .map((entry) => `
          <tr>
            <td>${escapeHtml(entry.type)}</td>
            <td>${escapeHtml(formatBytes(entry.total_bytes))}</td>
            <td>${escapeHtml(entry.count)}</td>
            <td>${escapeHtml(formatBytes(entry.largest_bytes))}</td>
          </tr>
        `)
        .join('')
    : '<tr><td colspan="4" class="muted">No live allocations at this point.</td></tr>';

  const visibleRecords = selectedRecords.slice(0, 500);
  recordsBody.innerHTML = visibleRecords.length
    ? visibleRecords
        .map((record, index) => `
          <tr class="record-row" data-record-index="${index}">
            <td>${escapeHtml(displayType(record))}</td>
            <td>${escapeHtml(formatBytes(record.size_bytes))}</td>
            <td>${escapeHtml(record.len == null ? '-' : record.len)}</td>
            <td>${escapeHtml(record.capacity == null ? '-' : record.capacity)}</td>
            <td>${escapeHtml(pointerText(record.ptr))}</td>
          </tr>
        `)
        .join('')
    : '<tr><td colspan="5" class="muted">No live allocations at this point.</td></tr>';

  const recordsNoteParts = [];
  if (selectedRecords.length > visibleRecords.length) {
    recordsNoteParts.push(
      `Showing the largest ${visibleRecords.length} of ${selectedRecords.length} live allocations.`
    );
  }
  if (visibleRecords.length) {
    recordsNoteParts.push('Click a row to inspect it in the floating window.');
  }
  recordsNote.textContent = recordsNoteParts.join(' ');

  for (const row of recordsBody.querySelectorAll('tr.record-row')) {
    row.addEventListener('click', () => {
      const index = Number(row.getAttribute('data-record-index'));
      const record = visibleRecords[index];
      if (!record) {
        return;
      }
      selectRecordRow(row, record);
    });
  }

  if (!visibleRecords.length) {
    hideInspector();
    renderRecordDetail(null);
  }

  renderPhaseTimeline();
  drawCurve();
}

function nearestTimelineIndex(timeNs) {
  if (!report.timeline.length) {
    return 0;
  }
  let low = 0;
  let high = report.timeline.length - 1;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (report.timeline[mid].time_ns < timeNs) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  const right = low;
  const left = Math.max(0, right - 1);
  const leftDist = Math.abs(report.timeline[left].time_ns - timeNs);
  const rightDist = Math.abs(report.timeline[right].time_ns - timeNs);
  return leftDist <= rightDist ? left : right;
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawCurve() {
  resizeCanvas();
  const { width, height, padLeft, padTop, plotWidth, plotHeight } = curveMetrics();
  const curveBuckets = curveBucketsForPlotWidth(plotWidth);
  const trackedCurveBuckets = curveBuckets.tracked;
  const globalCurveBuckets = curveBuckets.global;
  ctx.clearRect(0, 0, width, height);

  if (!report.timeline.length && !report.global_timeline.length) {
    ctx.fillStyle = '#8da3bc';
    ctx.font = '13px sans-serif';
    ctx.fillText('No timeline data', 12, 24);
    return;
  }

  const maxTime = maxTimelineTimeNs;
  const maxBytes = currentMaxBytes(trackedCurveBuckets, globalCurveBuckets);

  const xFor = (timeNs) => padLeft + (Number(timeNs) / Number(maxTime)) * plotWidth;
  const yFor = (bytes) => padTop + plotHeight - (Number(bytes) / Number(maxBytes)) * plotHeight;

  ctx.strokeStyle = '#223247';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padLeft, padTop);
  ctx.lineTo(padLeft, padTop + plotHeight);
  ctx.lineTo(padLeft + plotWidth, padTop + plotHeight);
  ctx.stroke();

  const drawSeries = (series) => {
    if (!seriesIsEnabled(series)) {
      return;
    }
    const buckets = series.key === 'global_bytes' ? globalCurveBuckets : trackedCurveBuckets;
    let started = false;
    ctx.strokeStyle = series.color;
    ctx.lineWidth = series.width;
    ctx.beginPath();
    for (let index = 0; index < buckets.length; index += 1) {
      const point = buckets[index];
      const x =
        padLeft + (index / Math.max(1, buckets.length - 1)) * plotWidth;
      const y = yFor(series.key === 'global_bytes' ? point : point[series.key]);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  };

  for (const series of enabledCurveSeries()) {
    drawSeries(series);
  }

  const selected = report.timeline[selectedIndex];
  if (selected) {
    const x = xFor(selected.time_ns);
    const breakdown = breakdownForRecords(selectedRecords);
    const globalBytes = globalBytesAtTimeNs(selected.time_ns);
    const markerBytes = selectedMarkerValue(selected, breakdown, globalBytes);
    ctx.strokeStyle = '#34d399';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, padTop);
    ctx.lineTo(x, padTop + plotHeight);
    ctx.stroke();

    if (markerBytes != null) {
      ctx.fillStyle = '#34d399';
      ctx.beginPath();
      ctx.arc(x, yFor(markerBytes), 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.fillStyle = '#8da3bc';
  ctx.font = '12px sans-serif';
  ctx.fillText(formatBytes(maxBytes), 8, padTop + 12);
  ctx.fillText('0 B', 20, padTop + plotHeight);
  ctx.fillText('0 ms', padLeft, height - 8);
  const endLabel = formatTime(maxTime);
  ctx.fillText(endLabel, Math.max(padLeft, width - ctx.measureText(endLabel).width - 8), height - 8);
}

canvas.addEventListener('click', (event) => {
  if (!report.timeline.length && !report.global_timeline.length) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const { plotLeft, plotRight } = curveMetrics();
  const x = event.clientX - rect.left;
  const clampedX = Math.max(plotLeft, Math.min(plotRight, x));
  const ratio = (clampedX - plotLeft) / Math.max(1, plotRight - plotLeft);
  const timeNs = Math.max(0, Math.min(1, ratio)) * maxTimelineTimeNs;
  renderSelection(nearestTimelineIndex(timeNs));
});

recordInspectorClose.addEventListener('click', () => {
  hideInspector();
});

recordInspectorHeader.addEventListener('pointerdown', (event) => {
  if (event.target === recordInspectorClose) {
    return;
  }
  const rect = recordInspector.getBoundingClientRect();
  inspectorDrag = {
    offsetX: event.clientX - rect.left,
    offsetY: event.clientY - rect.top,
  };
  recordInspectorHeader.setPointerCapture(event.pointerId);
});

recordInspectorHeader.addEventListener('pointermove', (event) => {
  if (!inspectorDrag) {
    return;
  }
  const maxLeft = Math.max(0, window.innerWidth - recordInspector.offsetWidth);
  const maxTop = Math.max(0, window.innerHeight - recordInspector.offsetHeight);
  const left = Math.max(0, Math.min(maxLeft, event.clientX - inspectorDrag.offsetX));
  const top = Math.max(0, Math.min(maxTop, event.clientY - inspectorDrag.offsetY));
  recordInspector.style.left = `${left}px`;
  recordInspector.style.top = `${top}px`;
  recordInspector.style.right = 'auto';
});

function endInspectorDrag(event) {
  if (inspectorDrag) {
    inspectorDrag = null;
    recordInspectorHeader.releasePointerCapture(event.pointerId);
  }
}

recordInspectorHeader.addEventListener('pointerup', endInspectorDrag);
recordInspectorHeader.addEventListener('pointercancel', endInspectorDrag);

window.addEventListener('resize', () => drawCurve());

commandLine.textContent = report.command_line.length
  ? report.command_line.join(' ')
  : 'No command line recorded';
renderSummaryStats();
renderCurveLegend();
renderSelection(selectedIndex);
</script>
</body>
</html>
"#;

fn write_html_report(path: &PathBuf, document: &ReportDocument) -> Result<(), String> {
    let file =
        fs::File::create(path).map_err(|err| format!("creating {}: {err}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(HTML_PREFIX.as_bytes())
        .map_err(|err| format!("writing {}: {err}", path.display()))?;
    {
        let mut json_writer = ScriptJsonWriter::new(&mut writer);
        serde_json::to_writer(&mut json_writer, document).map_err(|err| err.to_string())?;
        json_writer
            .flush()
            .map_err(|err| format!("writing {}: {err}", path.display()))?;
    }
    writer
        .write_all(HTML_SUFFIX.as_bytes())
        .map_err(|err| format!("writing {}: {err}", path.display()))?;
    writer
        .flush()
        .map_err(|err| format!("writing {}: {err}", path.display()))
}

struct ScriptJsonWriter<'a, W> {
    inner: &'a mut W,
    prev_was_lt: bool,
}

impl<'a, W> ScriptJsonWriter<'a, W> {
    fn new(inner: &'a mut W) -> Self {
        Self {
            inner,
            prev_was_lt: false,
        }
    }
}

impl<W: Write> Write for ScriptJsonWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.write_all(buf)?;
        Ok(buf.len())
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        for &byte in buf {
            if self.prev_was_lt && byte == b'/' {
                self.inner.write_all(br"\/")?;
                self.prev_was_lt = false;
                continue;
            }
            self.inner.write_all(&[byte])?;
            self.prev_was_lt = byte == b'<';
        }
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
fn render_html(json: &str) -> String {
    let escaped_json = json.replace("</", "<\\/");
    let mut html =
        String::with_capacity(HTML_PREFIX.len() + escaped_json.len() + HTML_SUFFIX.len());
    html.push_str(HTML_PREFIX);
    html.push_str(&escaped_json);
    html.push_str(HTML_SUFFIX);
    html
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_html_embeds_profile_json() {
        let html = render_html(
            "{\"timeline\":[],\"stacks\":[],\"events\":[],\"summary\":{},\"initial_index\":0,\"command_line\":[]}",
        );
        assert!(html.contains("report-data"));
        assert!(html.contains("Memory Profile"));
    }
}
