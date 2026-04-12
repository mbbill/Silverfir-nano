#!/usr/bin/env node

const fs = require("fs");

function usage() {
  console.error(
    "usage: node scripts/memprof_phase_delta.js <report.html> [--phase <name>] [--span <index>]"
  );
  process.exit(2);
}

function formatBytes(bytes) {
  const abs = Math.abs(bytes);
  if (abs >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
  }
  if (abs >= 1024) {
    return `${(bytes / 1024).toFixed(2)} KiB`;
  }
  return `${bytes} B`;
}

function formatMs(ns) {
  return `${(ns / 1e6).toFixed(3)} ms`;
}

function parseReport(path) {
  const html = fs.readFileSync(path, "utf8");
  const match = html.match(
    /<script id="report-data" type="application\/json">\n([\s\S]*?)\n<\/script>/
  );
  if (!match) {
    throw new Error(`report json not found in ${path}`);
  }
  return JSON.parse(match[1]);
}

function parseArgs(argv) {
  if (argv.length < 3) {
    usage();
  }
  const args = {
    reportPath: argv[2],
    phase: null,
    spanIndex: null,
  };
  for (let i = 3; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--phase") {
      args.phase = argv[++i] ?? usage();
    } else if (arg === "--span") {
      args.spanIndex = Number(argv[++i]);
      if (!Number.isInteger(args.spanIndex) || args.spanIndex < 0) {
        throw new Error("--span must be a non-negative integer");
      }
    } else {
      usage();
    }
  }
  return args;
}

function nearestTimelineIndex(timeline, targetNs) {
  let low = 0;
  let high = timeline.length - 1;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (Number(timeline[mid].time_ns) < targetNs) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  const right = low;
  const left = Math.max(0, right - 1);
  const leftDist = Math.abs(Number(timeline[left].time_ns) - targetNs);
  const rightDist = Math.abs(Number(timeline[right].time_ns) - targetNs);
  return leftDist <= rightDist ? left : right;
}

function collectPeakIndex(timeline, span) {
  const startIndex = nearestTimelineIndex(timeline, span.start_time_ns);
  const endIndex = nearestTimelineIndex(timeline, span.end_time_ns);
  let bestIndex = startIndex;
  let bestBytes = Number(timeline[startIndex].total_bytes);
  for (let i = startIndex; i <= endIndex; i += 1) {
    const bytes = Number(timeline[i].total_bytes);
    if (bytes > bestBytes) {
      bestBytes = bytes;
      bestIndex = i;
    }
  }
  return bestIndex;
}

function displayType(record) {
  if (record.owner_kind === "String") {
    return "String";
  }
  if (record.owner_kind === "Vec" && record.element_type) {
    return `Vec<${record.element_type}>`;
  }
  if (record.owner_kind === "BTreeSet" && record.element_type) {
    return `BTreeSet<${record.element_type}>`;
  }
  if (record.owner_kind === "Rc") {
    if (record.type_name === "str") {
      return "Rc<str>";
    }
    if (
      record.element_type &&
      record.type_name &&
      record.type_name.startsWith("[")
    ) {
      return `Rc<[${record.element_type}]>`;
    }
    return record.type_name ? `Rc<${record.type_name}>` : "Rc";
  }
  return record.type_name || record.owner_kind || "unknown";
}

function collectLiveStateAt(report, targetNs) {
  const live = new Map();
  for (const event of report.events) {
    const timeNs = Number(event.time_ns);
    if (timeNs > targetNs) {
      break;
    }
    if (event.kind === "create") {
      live.set(event.id, {
        owner_kind: event.owner_kind,
        type_name: event.type_name,
        element_type: event.element_type,
        size_bytes: Number(event.size_bytes || 0),
      });
    } else if (event.kind === "update") {
      const record = live.get(event.id);
      if (record && event.size_bytes != null) {
        record.size_bytes = Number(event.size_bytes);
      }
    } else if (event.kind === "remove") {
      live.delete(event.id);
    }
  }
  return live;
}

function diffByType(before, after) {
  const beforeByType = new Map();
  const afterByType = new Map();
  for (const record of before.values()) {
    const key = displayType(record);
    beforeByType.set(key, (beforeByType.get(key) || 0) + record.size_bytes);
  }
  for (const record of after.values()) {
    const key = displayType(record);
    afterByType.set(key, (afterByType.get(key) || 0) + record.size_bytes);
  }
  const keys = new Set([...beforeByType.keys(), ...afterByType.keys()]);
  return [...keys]
    .map((key) => ({
      type: key,
      delta: (afterByType.get(key) || 0) - (beforeByType.get(key) || 0),
      after: afterByType.get(key) || 0,
    }))
    .filter((item) => item.delta !== 0)
    .sort((a, b) => b.delta - a.delta);
}

function summarizePhase(report, phaseName) {
  const timeline = report.timeline || [];
  const spans = (report.phases || [])
    .filter((phase) => phase.name === phaseName)
    .sort(
      (a, b) =>
        Number(a.start_time_ns) - Number(b.start_time_ns) ||
        Number(a.end_time_ns) - Number(b.end_time_ns)
    );
  const details = spans.map((span, index) => {
    const startIndex = nearestTimelineIndex(timeline, Number(span.start_time_ns));
    const endIndex = nearestTimelineIndex(timeline, Number(span.end_time_ns));
    const peakIndex = collectPeakIndex(timeline, {
      start_time_ns: Number(span.start_time_ns),
      end_time_ns: Number(span.end_time_ns),
    });
    const startBytes = Number(timeline[startIndex].total_bytes);
    const endBytes = Number(timeline[endIndex].total_bytes);
    const peakBytes = Number(timeline[peakIndex].total_bytes);
    return {
      index,
      functionIndex: span.function_index,
      startTimeNs: Number(span.start_time_ns),
      endTimeNs: Number(span.end_time_ns),
      peakTimeNs: Number(timeline[peakIndex].time_ns),
      startBytes,
      endBytes,
      peakBytes,
      endDeltaBytes: endBytes - startBytes,
      peakDeltaBytes: peakBytes - startBytes,
    };
  });
  const maxPeakDelta = details.reduce(
    (best, span) => (span.peakDeltaBytes > best.peakDeltaBytes ? span : best),
    details[0]
  );
  return { name: phaseName, spans: details, maxPeakDelta };
}

function printSummary(report, phaseFilter) {
  const phaseNames = [...new Set((report.phases || []).map((phase) => phase.name))];
  const names = phaseFilter ? phaseNames.filter((name) => name === phaseFilter) : phaseNames;
  if (names.length === 0) {
    throw new Error(`no phases matched${phaseFilter ? ` ${phaseFilter}` : ""}`);
  }
  for (const name of names) {
    const summary = summarizePhase(report, name);
    const span = summary.maxPeakDelta;
    console.log(
      [
        `${name}`,
        `spans=${summary.spans.length}`,
        `max_peak_delta=${formatBytes(span.peakDeltaBytes)}`,
        `end_delta=${formatBytes(span.endDeltaBytes)}`,
        `func=${span.functionIndex == null ? "-" : span.functionIndex}`,
        `span=${span.index}`,
        `start=${formatMs(span.startTimeNs)}`,
        `peak=${formatMs(span.peakTimeNs)}`,
        `end=${formatMs(span.endTimeNs)}`,
      ].join("  ")
    );
  }
}

function printSpanInspect(report, phaseName, spanIndex) {
  const summary = summarizePhase(report, phaseName);
  const span = summary.spans[spanIndex] ?? summary.maxPeakDelta;
  const startLive = collectLiveStateAt(report, span.startTimeNs);
  const peakLive = collectLiveStateAt(report, span.peakTimeNs);
  const deltas = diffByType(startLive, peakLive).filter((item) => item.delta > 0);
  console.log(
    `${phaseName} span=${span.index} func=${span.functionIndex == null ? "-" : span.functionIndex}`
  );
  console.log(
    `start=${formatMs(span.startTimeNs)} peak=${formatMs(span.peakTimeNs)} end=${formatMs(span.endTimeNs)} peak_delta=${formatBytes(span.peakDeltaBytes)}`
  );
  console.log("");
  for (const item of deltas.slice(0, 12)) {
    console.log(
      `${formatBytes(item.delta).padStart(10)}  live_at_peak=${formatBytes(item.after).padStart(10)}  ${item.type}`
    );
  }
}

function main() {
  const args = parseArgs(process.argv);
  const report = parseReport(args.reportPath);
  if (args.phase) {
    printSummary(report, args.phase);
    console.log("");
    printSpanInspect(report, args.phase, args.spanIndex);
  } else {
    printSummary(report, null);
  }
}

main();
