use serde::Serialize;
use std::collections::BTreeSet;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use tracked_alloc::{self, AggregateSnapshot, AllocationProfile, ProfilePhase, TimelinePoint};

pub struct Session {
    output_path: PathBuf,
    command_line: Vec<String>,
}

impl Session {
    pub fn new(output_path: Option<PathBuf>, command_line: &[String]) -> Self {
        tracked_alloc::reset_tracking();
        tracked_alloc::set_tracking_enabled(true);
        Self {
            output_path: absolutize_report_path(output_path.unwrap_or_else(default_report_path)),
            command_line: command_line.to_vec(),
        }
    }

    pub fn finish(self) -> Result<PathBuf, String> {
        tracked_alloc::set_tracking_enabled(false);
        let profile = tracked_alloc::profile();
        let document = ReportDocument::from_profile(profile, &self.command_line);
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
    phases: Vec<ReportPhase>,
    stacks: Vec<ReportStack>,
    snapshots: Vec<ReportSnapshot>,
}

impl ReportDocument {
    fn from_profile(profile: AllocationProfile, command_line: &[String]) -> Self {
        // Peak is computed from the *tracked* total (heap allocations), not
        // the runtime-memory series. Runtime memory (the 16 MiB native code
        // buffer reservation, guard-page mmaps) is reported as a separate
        // off-by-default series in the HTML.
        let mut raw_peak_index = 0usize;
        let mut peak_bytes = 0usize;
        let mut peak_time_ns = 0u64;
        for (index, point) in profile.timeline.iter().enumerate() {
            if point.total_bytes >= peak_bytes {
                peak_bytes = point.total_bytes;
                peak_time_ns = point.time_ns;
                raw_peak_index = index;
            }
        }
        let phase_peaks = compute_phase_peaks(&profile.timeline, &profile.phases);
        let compact_timeline = compact_timeline(&profile.timeline, raw_peak_index, &phase_peaks);
        let timeline = build_report_timeline(&compact_timeline.points, &profile.snapshots);
        let snapshots = build_report_snapshots(&profile.snapshots, &compact_timeline.points);
        let initial_index = if timeline.is_empty() {
            0
        } else {
            nearest_timeline_index(&timeline, peak_time_ns)
        };
        let summary = ReportSummary {
            peak_bytes,
            peak_time_ns,
            final_bytes: profile.snapshot.total_bytes,
            live_records: profile.snapshot.records.len(),
            snapshot_count: profile.snapshots.len(),
            phase_count: profile.phases.len(),
            final_time_ns: profile.now_ns,
        };
        let phases = profile
            .phases
            .into_iter()
            .zip(phase_peaks.into_iter())
            .map(|phase| ReportPhase {
                name: phase.0.name,
                start_time_ns: phase.0.start_time_ns,
                end_time_ns: phase.0.end_time_ns,
                function_index: phase.0.function_index,
                peak_time_ns: phase.1.peak_time_ns,
                peak_bytes: phase.1.peak_bytes,
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
        Self {
            command_line: command_line.to_vec(),
            summary,
            initial_index,
            timeline,
            phases,
            stacks,
            snapshots,
        }
    }
}

const MAX_REPORT_TIMELINE_POINTS: usize = 1_024;

struct CompactedTimeline {
    points: Vec<TimelinePoint>,
}

#[derive(Clone, Copy)]
struct PhasePeakInfo {
    peak_time_ns: u64,
    peak_bytes: usize,
}

fn compact_timeline(
    timeline: &[TimelinePoint],
    raw_peak_index: usize,
    _phase_peaks: &[PhasePeakInfo],
) -> CompactedTimeline {
    if timeline.is_empty() {
        return CompactedTimeline { points: Vec::new() };
    }
    if timeline.len() <= MAX_REPORT_TIMELINE_POINTS {
        return CompactedTimeline {
            points: timeline.to_vec(),
        };
    }

    let last_index = timeline.len().saturating_sub(1);
    let mut keep = BTreeSet::new();
    keep.insert(0usize);
    keep.insert(last_index);
    keep.insert(raw_peak_index.min(last_index));
    keep.insert(raw_peak_index.saturating_sub(1));
    keep.insert((raw_peak_index + 1).min(last_index));

    if MAX_REPORT_TIMELINE_POINTS > keep.len() {
        let free_slots = MAX_REPORT_TIMELINE_POINTS - keep.len();
        for slot in 0..free_slots {
            let numerator = slot.saturating_mul(last_index);
            let denominator = free_slots.saturating_sub(1).max(1);
            keep.insert(numerator / denominator);
        }
    }

    CompactedTimeline {
        points: keep.into_iter().map(|index| timeline[index]).collect(),
    }
}

fn compute_phase_peaks(timeline: &[TimelinePoint], phases: &[ProfilePhase]) -> Vec<PhasePeakInfo> {
    if timeline.is_empty() {
        return phases
            .iter()
            .map(|_| PhasePeakInfo {
                peak_time_ns: 0,
                peak_bytes: 0,
            })
            .collect();
    }

    let mut phase_order = (0..phases.len()).collect::<Vec<_>>();
    phase_order.sort_by_key(|&index| phases[index].start_time_ns);

    let mut peaks = phases
        .iter()
        .map(|phase| {
            let raw_peak_index = nearest_raw_timeline_index(timeline, phase.start_time_ns);
            PhasePeakInfo {
                peak_time_ns: timeline[raw_peak_index].time_ns,
                peak_bytes: timeline[raw_peak_index].total_bytes,
            }
        })
        .collect::<Vec<_>>();

    let mut active = Vec::<usize>::new();
    let mut next_phase = 0usize;
    for point in timeline {
        while next_phase < phase_order.len()
            && phases[phase_order[next_phase]].start_time_ns <= point.time_ns
        {
            active.push(phase_order[next_phase]);
            next_phase += 1;
        }
        active.retain(|&phase_index| phases[phase_index].end_time_ns >= point.time_ns);
        for &phase_index in &active {
            if point.total_bytes >= peaks[phase_index].peak_bytes {
                peaks[phase_index] = PhasePeakInfo {
                    peak_time_ns: point.time_ns,
                    peak_bytes: point.total_bytes,
                };
            }
        }
    }

    peaks
}

fn build_report_timeline(
    timeline: &[TimelinePoint],
    snapshots: &[AggregateSnapshot],
) -> Vec<ReportPoint> {
    let mut report_timeline = Vec::with_capacity(timeline.len());
    let mut snap_index = 0usize;

    for point in timeline {
        // Advance to nearest snapshot at or before this timeline point.
        while snap_index + 1 < snapshots.len() && snapshots[snap_index + 1].time_ns <= point.time_ns
        {
            snap_index += 1;
        }

        // Find the snapshot_id: index into the report snapshots vec (same order
        // as profile snapshots since we convert 1:1).
        let snapshot_id = if snapshots.is_empty() || snapshots[snap_index].time_ns > point.time_ns {
            0
        } else {
            snap_index
        };

        report_timeline.push(ReportPoint {
            time_ns: point.time_ns,
            total_bytes: point.total_bytes,
            code_buffer_bytes: point.code_buffer_bytes,
            guard_page_bytes: point.guard_page_bytes,
            live_records: point.live_records,
            snapshot_id,
        });
    }

    report_timeline
}

fn build_report_snapshots(
    snapshots: &[AggregateSnapshot],
    _timeline: &[TimelinePoint],
) -> Vec<ReportSnapshot> {
    snapshots
        .iter()
        .map(|snap| {
            let aggregates = snap
                .entries
                .iter()
                .map(|entry| ReportAggregate {
                    type_label: aggregate_type_label(
                        entry.owner_kind,
                        entry.type_name,
                        entry.element_type,
                    ),
                    total_bytes: entry.total_bytes,
                    count: entry.count,
                    largest_bytes: entry.largest_bytes,
                    create_stack_id: entry.create_stack_id,
                })
                .collect();
            ReportSnapshot { aggregates }
        })
        .collect()
}

fn aggregate_type_label(owner_kind: &str, type_name: &str, element_type: Option<&str>) -> String {
    match owner_kind {
        "String" => "String".to_owned(),
        "Vec" => match element_type {
            Some(element_type) => format!("Vec<{element_type}>"),
            None => "Vec".to_owned(),
        },
        "BTreeSet" => match element_type {
            Some(element_type) => format!("BTreeSet<{element_type}>"),
            None => "BTreeSet".to_owned(),
        },
        "BTreeMap" => format!("BTreeMap<{type_name}>"),
        "Rc" => {
            if type_name == "str" {
                "Rc<str>".to_owned()
            } else if type_name.starts_with('[') {
                match element_type {
                    Some(element_type) => format!("Rc<[{element_type}]>"),
                    None => "Rc".to_owned(),
                }
            } else {
                format!("Rc<{type_name}>")
            }
        }
        "Box" => {
            if type_name == "str" {
                "Box<str>".to_owned()
            } else if type_name.starts_with('[') {
                match element_type {
                    Some(element_type) => format!("Box<[{element_type}]>"),
                    None => "Box".to_owned(),
                }
            } else {
                format!("Box<{type_name}>")
            }
        }
        _ => type_name.to_owned(),
    }
}

fn nearest_raw_timeline_index(timeline: &[TimelinePoint], time_ns: u64) -> usize {
    let mut low = 0usize;
    let mut high = timeline.len().saturating_sub(1);
    while low < high {
        let mid = (low + high) / 2;
        if timeline[mid].time_ns < time_ns {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let right = low;
    let left = right.saturating_sub(1);
    let left_dist = timeline[left].time_ns.abs_diff(time_ns);
    let right_dist = timeline[right].time_ns.abs_diff(time_ns);
    if left_dist <= right_dist {
        left
    } else {
        right
    }
}

fn nearest_timeline_index(timeline: &[ReportPoint], time_ns: u64) -> usize {
    let mut low = 0usize;
    let mut high = timeline.len().saturating_sub(1);
    while low < high {
        let mid = (low + high) / 2;
        if timeline[mid].time_ns < time_ns {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let right = low;
    let left = right.saturating_sub(1);
    let left_dist = timeline[left].time_ns.abs_diff(time_ns);
    let right_dist = timeline[right].time_ns.abs_diff(time_ns);
    if left_dist <= right_dist {
        left
    } else {
        right
    }
}

#[derive(Serialize)]
struct ReportSummary {
    peak_bytes: usize,
    peak_time_ns: u64,
    final_bytes: usize,
    live_records: usize,
    snapshot_count: usize,
    phase_count: usize,
    final_time_ns: u64,
}

#[derive(Serialize)]
struct ReportPoint {
    time_ns: u64,
    total_bytes: usize,
    code_buffer_bytes: usize,
    guard_page_bytes: usize,
    live_records: usize,
    snapshot_id: usize,
}

#[derive(Serialize)]
struct ReportPhase {
    name: &'static str,
    start_time_ns: u64,
    end_time_ns: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_index: Option<u32>,
    peak_time_ns: u64,
    peak_bytes: usize,
}

#[derive(Serialize)]
struct ReportStack {
    id: u64,
    text: String,
}

#[derive(Serialize)]
struct ReportSnapshot {
    aggregates: Vec<ReportAggregate>,
}

#[derive(Serialize)]
struct ReportAggregate {
    type_label: String,
    total_bytes: usize,
    count: usize,
    largest_bytes: usize,
    create_stack_id: u64,
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
.group-header td {
  background: #17263a;
  padding: 10px 10px;
  font-size: 13px;
}
.group-entry td:first-child {
  padding-left: 20px;
  color: #9fb2c9;
}
.stack-trace {
  margin-top: 4px;
  padding-left: 12px;
  font-size: 11px;
  color: #6b839e;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  line-height: 1.5;
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
  .table-breakdown td:nth-child(4) {
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
      <p class="section-note">Groups the live allocations at the selected time by creation site.</p>
      <table class="table-breakdown">
        <thead>
          <tr><th>Type</th><th>Total</th><th>Count</th><th>Largest</th></tr>
        </thead>
        <tbody id="aggregate-body"></tbody>
      </table>
    </div>
  </section>
</main>
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
const snapshots = report.snapshots || [];
const phaseStageOrder = [
  'sem_scan',
  'sem_decode',
  'sem_inline',
  'ssa_lower',
  'cfg_lower',
  'slot_lower',
  'joint_plan',
  'ssa_emit',
  'ssa_cleanup',
  'ssa_opt',
  'ssa_sink',
  'ssa_validate',
  'mir_lower',
  'module_opt',
  'arch_lower',
];

let selectedIndex = Math.max(0, Math.min(report.initial_index, report.timeline.length - 1));
const reportPhases = (report.phases || [])
  .map((phase, index) => ({
    ...phase,
    __index: index,
    start_time_ns: Number(phase.start_time_ns) || 0,
    end_time_ns: Number(phase.end_time_ns) || 0,
    peak_time_ns: Number(phase.peak_time_ns) || 0,
    peak_bytes: Number(phase.peak_bytes) || 0,
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
  return maxBytes;
})();
const curvePadding = {
  left: 56,
  right: 14,
  top: 12,
  bottom: 28,
};
const curveSeriesMeta = [
  { key: 'total_bytes', label: 'Tracked total', color: '#f59e0b', width: 2.5, visible: true, defaultOn: true },
  // Code buffer: mmap'd executable region for emitted native code. A single
  // CodeBuffer reserves 16 MiB up front regardless of how much code gets
  // emitted, which would drown out the heap curve — off by default.
  { key: 'code_buffer_bytes', label: 'Code buffer', color: '#a855f7', width: 2, visible: true, defaultOn: false },
  // Guard pages: virtual reservation for each wasm linear memory (~8 GB
  // reserved per memory, a few pages actually committed). Separated so the
  // enormous virtual reservation does not dominate the chart — off by default.
  { key: 'guard_page_bytes', label: 'Guard pages', color: '#22c55e', width: 2, visible: true, defaultOn: false },
];
const curveSeriesState = new Map(
  curveSeriesMeta
    .filter((series) => series.visible)
    .map((series) => [series.key, !!series.defaultOn])
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

function seriesIsEnabled(series) {
  return series.visible && curveSeriesState.get(series.key) !== false;
}

function enabledCurveSeries() {
  return curveSeriesMeta.filter((series) => seriesIsEnabled(series));
}

function curveValueAtPoint(series, point) {
  return Number(point[series.key] || 0);
}

function currentMaxBytes() {
  let maxBytes = 1;
  for (const series of enabledCurveSeries()) {
    for (const point of report.timeline) {
      maxBytes = Math.max(maxBytes, curveValueAtPoint(series, point));
    }
  }
  return maxBytes;
}

function selectedMarkerValue(point) {
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
    case 'sem_scan':
      return '#0ea5e9';
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
    case 'ssa_emit':
      return '#fb923c';
    case 'ssa_cleanup':
      return '#fdba74';
    case 'ssa_opt':
      return '#facc15';
    case 'ssa_sink':
      return '#fbbf24';
    case 'ssa_validate':
      return '#fcd34d';
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
  return nearestTimelineIndex(Number(phase.peak_time_ns) || Number(phase.start_time_ns) || 0);
}

function peakIndexForSpans(spans) {
  if (!report.timeline.length) {
    return 0;
  }
  if (!spans.length) {
    return selectedIndex;
  }
  let bestPhase = spans[0];
  let bestBytes = Number(bestPhase.peak_bytes) || 0;
  for (const phase of spans) {
    const bytes = Number(phase.peak_bytes) || 0;
    if (bytes >= bestBytes) {
      bestPhase = phase;
      bestBytes = bytes;
    }
  }
  return phasePeakIndex(bestPhase);
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

function renderSummaryStats() {
  const stats = [
    ['Peak', formatBytes(report.summary.peak_bytes)],
    ['Peak Time', formatTime(report.summary.peak_time_ns)],
    ['Final', formatBytes(report.summary.final_bytes)],
    ['Live Records', report.summary.live_records],
    ['Snapshots', report.summary.snapshot_count],
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

function renderSelection(index) {
  selectedIndex = Math.max(0, Math.min(index, report.timeline.length - 1));
  const point = report.timeline[selectedIndex] || { time_ns: 0, total_bytes: 0, live_records: 0 };
  const snapshot = snapshots[Number(point.snapshot_id) || 0] || { aggregates: [] };
  const aggregates = snapshot.aggregates || [];
  const activePhases = activePhasesAtTimeNs(Number(point.time_ns) || 0);

  const selectionItems = [
    ['Selected Time', formatTime(point.time_ns)],
    ['Total Bytes', formatBytes(point.total_bytes)],
  ];
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

  // Group aggregates by create_stack_id.
  const groups = new Map();
  for (const entry of aggregates) {
    const key = entry.create_stack_id;
    if (!groups.has(key)) {
      groups.set(key, { entries: [], totalBytes: 0, totalCount: 0 });
    }
    const group = groups.get(key);
    group.entries.push(entry);
    group.totalBytes += entry.total_bytes;
    group.totalCount += entry.count;
  }
  // Sort groups by total bytes descending.
  const sortedGroups = [...groups.entries()].sort((a, b) => b[1].totalBytes - a[1].totalBytes);

  if (!sortedGroups.length) {
    aggregateBody.innerHTML = '<tr><td colspan="4" class="muted">No live allocations at this point.</td></tr>';
  } else {
    let html = '';
    for (const [stackId, group] of sortedGroups) {
      const fullStack = stackText(stackId);
      const stackLines = fullStack.split('\n');
      const siteLine = stackLines[0] || '(unknown)';
      const traceLines = stackLines.slice(1).filter(l => l.trim());
      let headerContent = `<strong>${escapeHtml(formatBytes(group.totalBytes))}</strong> &mdash; <span class="muted">${escapeHtml(siteLine)}</span>`;
      if (traceLines.length) {
        headerContent += `<div class="stack-trace">${traceLines.map(l => escapeHtml(l)).join('<br>')}</div>`;
      }
      html += `<tr class="group-header"><td colspan="4">${headerContent}</td></tr>`;
      for (const entry of group.entries) {
        html += `<tr class="group-entry"><td>&nbsp;&nbsp;${escapeHtml(entry.type_label)}</td><td>${escapeHtml(formatBytes(entry.total_bytes))}</td><td>${escapeHtml(entry.count)}</td><td>${escapeHtml(formatBytes(entry.largest_bytes))}</td></tr>`;
      }
    }
    aggregateBody.innerHTML = html;
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
  ctx.clearRect(0, 0, width, height);

  if (!report.timeline.length) {
    ctx.fillStyle = '#8da3bc';
    ctx.font = '13px sans-serif';
    ctx.fillText('No timeline data', 12, 24);
    return;
  }

  const maxTime = maxTimelineTimeNs;
  const maxBytes = currentMaxBytes();

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
    let started = false;
    ctx.strokeStyle = series.color;
    ctx.lineWidth = series.width;
    ctx.beginPath();
    for (const point of report.timeline) {
      const x = xFor(point.time_ns);
      const y = yFor(curveValueAtPoint(series, point));
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
    const markerBytes = selectedMarkerValue(selected);
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
  if (!report.timeline.length) {
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
