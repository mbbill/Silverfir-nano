use serde::Serialize;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use tracked_alloc::{
    self, AllocationProfile, GlobalAllocationProfile, ProfileEventKind, ProfilePhase, TimelinePoint,
};

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
    phases: Vec<ReportPhase>,
    stacks: Vec<ReportStack>,
    snapshots: Vec<ReportSnapshot>,
}

impl ReportDocument {
    fn from_profile(
        profile: AllocationProfile,
        global_profile: GlobalAllocationProfile,
        command_line: &[String],
    ) -> Self {
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
        let sampled = sample_report_data(
            &profile.events,
            &compact_timeline.points,
            &global_profile.timeline,
        );
        let timeline = sampled.timeline;
        let snapshots = sampled.snapshots;
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
            event_count: profile.events.len(),
            phase_count: profile.phases.len(),
            final_time_ns: profile.now_ns,
            global_peak_bytes: global_profile.peak_bytes,
            global_final_bytes: global_profile.final_bytes,
            global_final_time_ns: global_profile.now_ns,
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
const MAX_RECORDS_PER_SNAPSHOT: usize = 500;

struct CompactedTimeline {
    points: Vec<TimelinePoint>,
}

#[derive(Clone, Copy)]
struct PhasePeakInfo {
    peak_time_ns: u64,
    peak_bytes: usize,
}

#[derive(Clone, Copy, Default)]
struct Breakdown {
    tracked_alloc_bytes: usize,
    code_buffer_bytes: usize,
    guard_page_bytes: usize,
    runtime_other_bytes: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct LiveRecord {
    owner_kind: &'static str,
    type_name: &'static str,
    element_type: Option<&'static str>,
    len: Option<usize>,
    capacity: Option<usize>,
    size_bytes: usize,
    ptr: usize,
    create_stack_id: u64,
    last_update_stack_id: Option<u64>,
}

struct SampledReportData {
    timeline: Vec<ReportPoint>,
    snapshots: Vec<ReportSnapshot>,
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

fn sample_report_data(
    events: &[tracked_alloc::ProfileEvent],
    timeline: &[TimelinePoint],
    global_timeline: &[tracked_alloc::GlobalTimelinePoint],
) -> SampledReportData {
    let mut report_timeline = Vec::with_capacity(timeline.len());
    let mut current_live = HashMap::<u64, LiveRecord>::new();
    let mut breakdown = Breakdown::default();
    let mut event_index = 0usize;
    let mut global_index = 0usize;
    let mut global_bytes = global_timeline
        .first()
        .map(|point| point.live_bytes)
        .unwrap_or(0);
    let mut snapshots = Vec::<ReportSnapshot>::new();
    let mut last_snapshot_id = None;
    let mut snapshot_dirty = true;

    for point in timeline {
        while event_index < events.len() && events[event_index].time_ns <= point.time_ns {
            let event = &events[event_index];
            if apply_raw_event(&mut current_live, &mut breakdown, event) {
                snapshot_dirty = true;
            }
            event_index += 1;
        }

        while global_index + 1 < global_timeline.len()
            && global_timeline[global_index + 1].time_ns <= point.time_ns
        {
            global_index += 1;
            global_bytes = global_timeline[global_index].live_bytes;
        }

        let snapshot_id = if snapshot_dirty || last_snapshot_id.is_none() {
            snapshots.push(build_snapshot(&current_live));
            let snapshot_id = snapshots.len().saturating_sub(1);
            last_snapshot_id = Some(snapshot_id);
            snapshot_dirty = false;
            snapshot_id
        } else {
            last_snapshot_id.unwrap_or(0)
        };

        report_timeline.push(ReportPoint {
            time_ns: point.time_ns,
            total_bytes: point.total_bytes,
            live_records: point.live_records,
            global_bytes,
            tracked_alloc_bytes: breakdown.tracked_alloc_bytes,
            code_buffer_bytes: breakdown.code_buffer_bytes,
            guard_page_bytes: breakdown.guard_page_bytes,
            runtime_other_bytes: breakdown.runtime_other_bytes,
            snapshot_id,
        });
    }

    SampledReportData {
        timeline: report_timeline,
        snapshots,
    }
}

fn apply_raw_event(
    current_live: &mut HashMap<u64, LiveRecord>,
    breakdown: &mut Breakdown,
    event: &tracked_alloc::ProfileEvent,
) -> bool {
    match event.kind {
        ProfileEventKind::Create => {
            let record = LiveRecord {
                owner_kind: event.owner_kind.expect("create owner kind"),
                type_name: event.type_name.expect("create type name"),
                element_type: event.element_type,
                len: event.len,
                capacity: event.capacity,
                size_bytes: event.size_bytes.expect("create size"),
                ptr: event.ptr.expect("create ptr"),
                create_stack_id: event.create_stack_id.expect("create stack"),
                last_update_stack_id: event.last_update_stack_id,
            };
            apply_breakdown_delta(
                breakdown,
                allocation_category(record.owner_kind, record.type_name),
                record.size_bytes as isize,
            );
            current_live.insert(event.id, record);
            true
        }
        ProfileEventKind::Update => {
            let Some(record) = current_live.get_mut(&event.id) else {
                return false;
            };
            let mut changed = false;
            if let Some(size_bytes) = event.size_bytes {
                let delta = size_bytes as isize - record.size_bytes as isize;
                apply_breakdown_delta(
                    breakdown,
                    allocation_category(record.owner_kind, record.type_name),
                    delta,
                );
                record.size_bytes = size_bytes;
                changed = true;
            }
            if record.len != event.len {
                changed = true;
            }
            record.len = event.len;
            if record.capacity != event.capacity {
                changed = true;
            }
            record.capacity = event.capacity;
            if let Some(ptr) = event.ptr {
                if record.ptr != ptr {
                    changed = true;
                }
                record.ptr = ptr;
            }
            if let Some(stack_id) = event.last_update_stack_id {
                if record.last_update_stack_id != Some(stack_id) {
                    changed = true;
                }
                record.last_update_stack_id = Some(stack_id);
            }
            changed
        }
        ProfileEventKind::Remove => {
            let Some(record) = current_live.remove(&event.id) else {
                return false;
            };
            apply_breakdown_delta(
                breakdown,
                allocation_category(record.owner_kind, record.type_name),
                -(record.size_bytes as isize),
            );
            true
        }
    }
}

fn build_snapshot(current_live: &HashMap<u64, LiveRecord>) -> ReportSnapshot {
    ReportSnapshot {
        aggregates: aggregate_types(current_live),
        records: largest_records(current_live),
    }
}

fn allocation_category(owner_kind: &str, type_name: &str) -> &'static str {
    if owner_kind == "RuntimeMemory" {
        if type_name == "CodeBuffer" {
            return "code_buffer";
        }
        if type_name == "GuardPageMemory" {
            return "guard_page";
        }
        return "runtime_other";
    }
    "tracked_alloc"
}

fn apply_breakdown_delta(breakdown: &mut Breakdown, category: &str, delta: isize) {
    match category {
        "code_buffer" => {
            breakdown.code_buffer_bytes = breakdown.code_buffer_bytes.saturating_add_signed(delta);
        }
        "guard_page" => {
            breakdown.guard_page_bytes = breakdown.guard_page_bytes.saturating_add_signed(delta);
        }
        "runtime_other" => {
            breakdown.runtime_other_bytes =
                breakdown.runtime_other_bytes.saturating_add_signed(delta);
        }
        _ => {
            breakdown.tracked_alloc_bytes =
                breakdown.tracked_alloc_bytes.saturating_add_signed(delta);
        }
    }
}

fn aggregate_types(current_live: &HashMap<u64, LiveRecord>) -> Vec<ReportAggregate> {
    let mut by_type = HashMap::<String, ReportAggregate>::new();
    for record in current_live.values() {
        let type_label = display_type_label(record);
        let entry = by_type
            .entry(type_label.clone())
            .or_insert(ReportAggregate {
                type_label,
                total_bytes: 0,
                count: 0,
                largest_bytes: 0,
            });
        entry.total_bytes += record.size_bytes;
        entry.count += 1;
        entry.largest_bytes = entry.largest_bytes.max(record.size_bytes);
    }
    let mut aggregates = by_type.into_values().collect::<Vec<_>>();
    aggregates.sort_by(|left, right| {
        right
            .total_bytes
            .cmp(&left.total_bytes)
            .then_with(|| left.type_label.cmp(&right.type_label))
    });
    aggregates
}

fn largest_records(current_live: &HashMap<u64, LiveRecord>) -> Vec<ReportRecord> {
    let mut records = current_live.iter().collect::<Vec<_>>();
    records.sort_by(|(left_id, left), (right_id, right)| {
        right
            .size_bytes
            .cmp(&left.size_bytes)
            .then_with(|| left_id.cmp(right_id))
    });
    records
        .into_iter()
        .take(MAX_RECORDS_PER_SNAPSHOT)
        .map(|(_, record)| ReportRecord {
            type_label: display_type_label(record),
            len: record.len,
            capacity: record.capacity,
            size_bytes: record.size_bytes,
            ptr: record.ptr,
            create_stack_id: record.create_stack_id,
            last_update_stack_id: record.last_update_stack_id,
        })
        .collect()
}

fn display_type_label(record: &LiveRecord) -> String {
    match record.owner_kind {
        "String" => "String".to_owned(),
        "Vec" => match record.element_type {
            Some(element_type) => format!("Vec<{element_type}>"),
            None => "Vec".to_owned(),
        },
        "BTreeSet" => match record.element_type {
            Some(element_type) => format!("BTreeSet<{element_type}>"),
            None => "BTreeSet".to_owned(),
        },
        "BTreeMap" => format!("BTreeMap<{}>", record.type_name),
        "Rc" => {
            if record.type_name == "str" {
                "Rc<str>".to_owned()
            } else if record.type_name.starts_with('[') {
                match record.element_type {
                    Some(element_type) => format!("Rc<[{element_type}]>"),
                    None => "Rc".to_owned(),
                }
            } else {
                format!("Rc<{}>", record.type_name)
            }
        }
        "Box" => {
            if record.type_name == "str" {
                "Box<str>".to_owned()
            } else if record.type_name.starts_with('[') {
                match record.element_type {
                    Some(element_type) => format!("Box<[{element_type}]>"),
                    None => "Box".to_owned(),
                }
            } else {
                format!("Box<{}>", record.type_name)
            }
        }
        _ => record.type_name.to_owned(),
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
    global_bytes: usize,
    tracked_alloc_bytes: usize,
    code_buffer_bytes: usize,
    guard_page_bytes: usize,
    runtime_other_bytes: usize,
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
    records: Vec<ReportRecord>,
}

#[derive(Serialize)]
struct ReportAggregate {
    type_label: String,
    total_bytes: usize,
    count: usize,
    largest_bytes: usize,
}

#[derive(Serialize)]
struct ReportRecord {
    type_label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    len: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    capacity: Option<usize>,
    size_bytes: usize,
    ptr: usize,
    create_stack_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_update_stack_id: Option<u64>,
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
let selectedRecords = [];
let inspectorDrag = null;
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
const showCodeBufferSeries = report.timeline.some(
  (point) => Number(point.code_buffer_bytes || 0) > 0
);
const showGuardPageSeries = report.timeline.some(
  (point) => Number(point.guard_page_bytes || 0) > 0
);
const showRuntimeOtherSeries = report.timeline.some(
  (point) => Number(point.runtime_other_bytes || 0) > 0
);
const showGlobalAllocatorSeries =
  report.timeline.some((point) => Number(point.global_bytes || 0) > 0) ||
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
  const visibleSeries = enabledCurveSeries();
  if (!visibleSeries.length) {
    return null;
  }
  for (const series of visibleSeries) {
    switch (series.key) {
      case 'global_bytes':
        return point.global_bytes;
      case 'tracked_alloc_bytes':
        return point.tracked_alloc_bytes;
      case 'code_buffer_bytes':
        return point.code_buffer_bytes;
      case 'guard_page_bytes':
        return point.guard_page_bytes;
      case 'runtime_other_bytes':
        return point.runtime_other_bytes;
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
    `${record.type_label || 'unknown'}  ${formatBytes(record.size_bytes)}`,
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
  const snapshot = snapshots[Number(point.snapshot_id) || 0] || { aggregates: [], records: [] };
  selectedRecords = snapshot.records || [];
  const aggregates = snapshot.aggregates || [];
  const globalBytes = Number(point.global_bytes || 0);
  const trackedAllocBytes = Number(point.tracked_alloc_bytes || 0);
  const codeBufferBytes = Number(point.code_buffer_bytes || 0);
  const guardPageBytes = Number(point.guard_page_bytes || 0);
  const runtimeOtherBytes = Number(point.runtime_other_bytes || 0);
  const untrackedGlobalGap = Math.max(0, globalBytes - trackedAllocBytes);
  const activePhases = activePhasesAtTimeNs(Number(point.time_ns) || 0);

  const selectionItems = [
    ['Selected Time', formatTime(point.time_ns)],
    ['Total Bytes', formatBytes(point.total_bytes)],
    ['Global allocator', formatBytes(globalBytes)],
    ['Tracked alloc', formatBytes(trackedAllocBytes)],
  ];
  if (showGlobalAllocatorSeries) {
    selectionItems.push(['Untracked heap gap', formatBytes(untrackedGlobalGap)]);
  }
  if (showCodeBufferSeries) {
    selectionItems.push(['Code Buffer', formatBytes(codeBufferBytes)]);
  }
  if (showGuardPageSeries) {
    selectionItems.push(['Guard Pages', formatBytes(guardPageBytes)]);
  }
  if (showRuntimeOtherSeries) {
    selectionItems.push(['Runtime other', formatBytes(runtimeOtherBytes)]);
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
            <td>${escapeHtml(entry.type_label)}</td>
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
            <td>${escapeHtml(record.type_label || 'unknown')}</td>
            <td>${escapeHtml(formatBytes(record.size_bytes))}</td>
            <td>${escapeHtml(record.len == null ? '-' : record.len)}</td>
            <td>${escapeHtml(record.capacity == null ? '-' : record.capacity)}</td>
            <td>${escapeHtml(pointerText(record.ptr))}</td>
          </tr>
        `)
        .join('')
    : '<tr><td colspan="5" class="muted">No live allocations at this point.</td></tr>';

  const recordsNoteParts = [];
  if (Number(point.live_records || 0) > visibleRecords.length) {
    recordsNoteParts.push(
      `Showing the largest ${visibleRecords.length} of ${point.live_records} live allocations.`
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
