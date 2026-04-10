#!/usr/bin/env python3

from __future__ import annotations

import argparse
import bisect
import json
import html
import http.server
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, TextIO, Tuple


Record = Dict[str, object]


@dataclass
class Totals:
    alloc: int = 0
    exec: int = 0
    guard: int = 0

    @property
    def total(self) -> int:
        return self.alloc + self.exec + self.guard

    def copy(self) -> "Totals":
        return Totals(self.alloc, self.exec, self.guard)


@dataclass
class Bucket:
    start_us: int
    max_time_us: int
    max_totals: Totals
    end_totals: Totals

    @classmethod
    def constant(cls, start_us: int, totals: Totals) -> "Bucket":
        return cls(
            start_us=start_us,
            max_time_us=start_us,
            max_totals=totals.copy(),
            end_totals=totals.copy(),
        )

    def note(self, t_us: int, totals: Totals) -> None:
        self.end_totals = totals.copy()
        if totals.total >= self.max_totals.total:
            self.max_totals = totals.copy()
            self.max_time_us = t_us


@dataclass
class Peak:
    index: int
    time_us: int
    peak_bytes: int
    alloc_bytes: int
    exec_bytes: int
    guard_bytes: int
    rise_bytes: int
    fall_bytes: int
    prominence_bytes: int


@dataclass(frozen=True)
class Image:
    base: int
    size: int
    path: str

    @property
    def end(self) -> int:
        return self.base + self.size


class LiveState:
    def __init__(self) -> None:
        self.allocs: Dict[int, Tuple[int, int, int]] = {}
        self.exec_regions: Dict[int, Tuple[int, int, int]] = {}
        self.guard_regions: Dict[int, Tuple[int, int, int]] = {}
        self.totals = Totals()

    def apply(self, record: Record) -> None:
        record_type = record["type"]
        if record_type == "alloc":
            ptr = parse_hex(record["ptr"])
            size = int(record["size"])
            stack_id = int(record["stack"])
            tag_id = int(record.get("tag", 0))
            self.allocs[ptr] = (size, stack_id, tag_id)
            self.totals.alloc += size
        elif record_type == "free":
            ptr = parse_hex(record["ptr"])
            live = self.allocs.pop(ptr, None)
            if live is not None:
                self.totals.alloc -= live[0]
        elif record_type == "realloc":
            old_ptr = parse_hex(record["old_ptr"])
            live = self.allocs.pop(old_ptr, None)
            if live is not None:
                self.totals.alloc -= live[0]
            new_ptr = parse_hex(record["new_ptr"])
            new_size = int(record["new_size"])
            stack_id = int(record["stack"])
            tag_id = int(record.get("tag", 0))
            self.allocs[new_ptr] = (new_size, stack_id, tag_id)
            self.totals.alloc += new_size
        elif record_type == "exec":
            base = parse_hex(record["base"])
            prev_reserved, prev_used, _prev_tag = self.exec_regions.get(base, (0, 0, 0))
            self.totals.exec -= prev_used
            reserved = int(record["reserved"])
            used = int(record["used"])
            tag_id = int(record.get("tag", 0))
            self.exec_regions[base] = (reserved, used, tag_id)
            self.totals.exec += used
        elif record_type == "exec_drop":
            base = parse_hex(record["base"])
            prev = self.exec_regions.pop(base, None)
            if prev is not None:
                self.totals.exec -= prev[1]
        elif record_type == "guard":
            base = parse_hex(record["base"])
            prev_reserved, prev_committed, prev_tag = self.guard_regions.get(base, (0, 0, 0))
            self.totals.guard -= prev_committed
            reserved = int(record["reserved"])
            committed = int(record["committed"])
            if reserved == 0:
                reserved = prev_reserved
            tag_id = int(record.get("tag", prev_tag))
            self.guard_regions[base] = (reserved, committed, tag_id)
            self.totals.guard += committed
        elif record_type == "guard_drop":
            base = parse_hex(record["base"])
            prev = self.guard_regions.pop(base, None)
            if prev is not None:
                self.totals.guard -= prev[1]


def parse_hex(value: object) -> int:
    if not isinstance(value, str):
        raise ValueError(f"expected hex string, got {value!r}")
    return int(value, 16)


def iter_records(path: Path) -> Iterator[Record]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise RuntimeError(f"{path}:{line_number}: expected object record")
            yield record


def timed_event_us(record: Record) -> Optional[int]:
    value = record.get("t_us")
    if value is None:
        return None
    return int(value)


def ensure_bucket(
    buckets: List[Bucket], target_index: int, bucket_us: int, carry_totals: Totals
) -> None:
    while len(buckets) <= target_index:
        start_us = len(buckets) * bucket_us
        buckets.append(Bucket.constant(start_us, carry_totals))


def build_buckets(path: Path, bucket_us: int) -> List[Bucket]:
    state = LiveState()
    buckets: List[Bucket] = []
    carry = state.totals.copy()

    for record in iter_records(path):
        t_us = timed_event_us(record)
        if t_us is None:
            continue
        bucket_index = t_us // bucket_us
        ensure_bucket(buckets, bucket_index, bucket_us, carry)
        state.apply(record)
        current = state.totals.copy()
        buckets[bucket_index].note(t_us, current)
        carry = current

    return buckets


def is_local_peak(buckets: List[Bucket], index: int) -> bool:
    current = buckets[index].max_totals.total
    prev_value = buckets[index - 1].max_totals.total if index > 0 else -1
    next_value = buckets[index + 1].max_totals.total if index + 1 < len(buckets) else -1
    return (current >= prev_value and current > next_value) or (
        current > prev_value and current >= next_value
    )


def find_spikes_from_buckets(buckets: List[Bucket]) -> List[Peak]:
    if not buckets:
        return []

    peak_indexes = [idx for idx in range(len(buckets)) if is_local_peak(buckets, idx)]
    if not peak_indexes:
        peak_indexes = [max(range(len(buckets)), key=lambda idx: buckets[idx].max_totals.total)]

    peaks: List[Peak] = []
    for position, index in enumerate(peak_indexes):
        bucket = buckets[index]
        left_bound = peak_indexes[position - 1] if position > 0 else 0
        right_bound = peak_indexes[position + 1] if position + 1 < len(peak_indexes) else len(buckets) - 1
        left_valley = min(b.end_totals.total for b in buckets[left_bound : index + 1])
        right_valley = min(b.end_totals.total for b in buckets[index : right_bound + 1])
        rise = max(0, bucket.max_totals.total - left_valley)
        fall = max(0, bucket.max_totals.total - right_valley)
        prominence = min(rise, fall) if rise > 0 and fall > 0 else max(rise, fall)
        peaks.append(
            Peak(
                index=index,
                time_us=bucket.max_time_us,
                peak_bytes=bucket.max_totals.total,
                alloc_bytes=bucket.max_totals.alloc,
                exec_bytes=bucket.max_totals.exec,
                guard_bytes=bucket.max_totals.guard,
                rise_bytes=rise,
                fall_bytes=fall,
                prominence_bytes=prominence,
            )
        )
    peaks.sort(
        key=lambda peak: (
            peak.peak_bytes,
            peak.prominence_bytes,
            peak.rise_bytes,
            -peak.time_us,
        ),
        reverse=True,
    )
    return peaks


def find_spikes(path: Path, bucket_us: int) -> List[Peak]:
    return find_spikes_from_buckets(build_buckets(path, bucket_us))


def write_json_output(payload: object, handle: Optional[TextIO]) -> None:
    out = handle or sys.stdout
    json.dump(payload, out, indent=2)
    out.write("\n")


def timeline_payload(buckets: List[Bucket], bucket_us: int) -> Dict[str, object]:
    return {
        "bucket_us": bucket_us,
        "buckets": [
            {
                "start_us": bucket.start_us,
                "max_time_us": bucket.max_time_us,
                "max_total_live_bytes": bucket.max_totals.total,
                "max_alloc_live_bytes": bucket.max_totals.alloc,
                "max_exec_live_bytes": bucket.max_totals.exec,
                "max_guard_live_bytes": bucket.max_totals.guard,
                "end_total_live_bytes": bucket.end_totals.total,
                "end_alloc_live_bytes": bucket.end_totals.alloc,
                "end_exec_live_bytes": bucket.end_totals.exec,
                "end_guard_live_bytes": bucket.end_totals.guard,
            }
            for bucket in buckets
        ],
    }


def command_spikes(args: argparse.Namespace) -> int:
    peaks = find_spikes(args.trace, args.bucket_us)
    peaks = peaks[: args.top]

    if args.json:
        write_json_output(
            [
                {
                    "time_us": peak.time_us,
                    "peak_bytes": peak.peak_bytes,
                    "alloc_bytes": peak.alloc_bytes,
                    "exec_bytes": peak.exec_bytes,
                    "guard_bytes": peak.guard_bytes,
                    "rise_bytes": peak.rise_bytes,
                    "fall_bytes": peak.fall_bytes,
                    "prominence_bytes": peak.prominence_bytes,
                }
                for peak in peaks
            ],
            args.json,
        )
        return 0

    if not peaks:
        print("No timed events found.")
        return 0

    print(
        "rank  time_us   peak_live  prominence  rise       fall       alloc      exec       guard"
    )
    for index, peak in enumerate(peaks, start=1):
        print(
            f"{index:>4}  {peak.time_us:>7}  "
            f"{format_mib(peak.peak_bytes):>9}  "
            f"{format_mib(peak.prominence_bytes):>10}  "
            f"{format_mib(peak.rise_bytes):>9}  "
            f"{format_mib(peak.fall_bytes):>9}  "
            f"{format_mib(peak.alloc_bytes):>9}  "
            f"{format_mib(peak.exec_bytes):>9}  "
            f"{format_mib(peak.guard_bytes):>9}"
        )
    return 0


def command_timeline(args: argparse.Namespace) -> int:
    buckets = build_buckets(args.trace, args.bucket_us)
    payload = timeline_payload(buckets, args.bucket_us)
    write_json_output(payload, args.json)
    return 0


def command_curve_html(args: argparse.Namespace) -> int:
    buckets = build_buckets(args.trace, args.bucket_us)
    payload = timeline_payload(buckets, args.bucket_us)
    spikes = [
        {
            "time_us": peak.time_us,
            "peak_bytes": peak.peak_bytes,
            "prominence_bytes": peak.prominence_bytes,
        }
        for peak in find_spikes_from_buckets(buckets)[: args.spikes]
    ]
    write_curve_html(
        out_path=args.out,
        trace_path=args.trace,
        timeline=payload,
        spikes=spikes,
    )
    print(args.out)
    return 0


def reconstruct_snapshot(
    path: Path, target_time_us: int
) -> Tuple[LiveState, Dict[int, List[str]], Optional[List[str]], List[Image], Dict[int, str]]:
    state = LiveState()
    stacks: Dict[int, List[str]] = {}
    command_line: Optional[List[str]] = None
    images: List[Image] = []
    tags: Dict[int, str] = {0: "untagged"}

    for record in iter_records(path):
        record_type = record["type"]
        if record_type == "meta":
            raw_command = record.get("command_line", [])
            if isinstance(raw_command, list):
                command_line = [str(value) for value in raw_command]
            continue
        if record_type == "image":
            images.append(
                Image(
                    base=parse_hex(record["base"]),
                    size=int(record.get("size", 0)),
                    path=str(record.get("path", "<unknown>")),
                )
            )
            continue
        if record_type == "tag":
            tags[int(record["id"])] = str(record.get("name", "untagged"))
            continue
        if record_type == "stack":
            stack_id = int(record["id"])
            raw_frames = record.get("frames", [])
            if isinstance(raw_frames, list):
                stacks[stack_id] = [str(frame) for frame in raw_frames]
            continue

        t_us = timed_event_us(record)
        if t_us is None:
            continue
        if t_us > target_time_us:
            break
        state.apply(record)

    images.sort(key=lambda image: image.base)
    return state, stacks, command_line, images, tags


def aggregate_live_allocs(state: LiveState) -> Dict[Tuple[int, int], int]:
    per_stack: Dict[Tuple[int, int], int] = {}
    for size, stack_id, tag_id in state.allocs.values():
        key = (stack_id, tag_id)
        per_stack[key] = per_stack.get(key, 0) + size
    return per_stack


def aggregate_live_tags(state: LiveState) -> Dict[int, Totals]:
    per_tag: Dict[int, Totals] = {}

    def bucket(tag_id: int) -> Totals:
        current = per_tag.get(tag_id)
        if current is None:
            current = Totals()
            per_tag[tag_id] = current
        return current

    for size, _stack_id, tag_id in state.allocs.values():
        bucket(tag_id).alloc += size
    for _reserved, used, tag_id in state.exec_regions.values():
        bucket(tag_id).exec += used
    for _reserved, committed, tag_id in state.guard_regions.values():
        bucket(tag_id).guard += committed

    return per_tag


def summarize_top_tags(
    state: LiveState,
    tag_names: Dict[int, str],
    top: int,
) -> List[Dict[str, object]]:
    per_tag = aggregate_live_tags(state)
    rows: List[Dict[str, object]] = []
    for tag_id, totals in sorted(
        per_tag.items(),
        key=lambda item: item[1].total,
        reverse=True,
    )[:top]:
        rows.append(
            {
                "tag_id": tag_id,
                "tag_name": tag_names.get(tag_id, f"tag:{tag_id}"),
                "total_live_bytes": totals.total,
                "alloc_live_bytes": totals.alloc,
                "exec_live_bytes": totals.exec,
                "guard_live_bytes": totals.guard,
            }
        )
    return rows


def write_collapsed_snapshot(
    path: Path,
    per_stack: Dict[Tuple[int, int], int],
    stack_labels: Dict[int, List[str]],
    tag_names: Dict[int, str],
    state: LiveState,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for (stack_id, tag_id), live_bytes in sorted(per_stack.items(), key=lambda item: item[1], reverse=True):
            if live_bytes <= 0:
                continue
            frames = stack_labels.get(stack_id, [])
            components = [tag_names.get(tag_id, f"tag:{tag_id}"), "[alloc]"]
            if frames:
                components.extend(reversed(frames))
            handle.write(";".join(components))
            handle.write(f" {live_bytes}\n")
        for _base, (_reserved, used, tag_id) in state.exec_regions.items():
            if used > 0:
                handle.write(f"{tag_names.get(tag_id, f'tag:{tag_id}')};[jit-exec] {used}\n")
        for _base, (_reserved, committed, tag_id) in state.guard_regions.items():
            if committed > 0:
                handle.write(
                    f"{tag_names.get(tag_id, f'tag:{tag_id}')};[guard-committed] {committed}\n"
                )


def summarize_top_stacks(
    per_stack: Dict[Tuple[int, int], int],
    stack_labels: Dict[int, List[str]],
    tag_names: Dict[int, str],
    top: int,
) -> List[Tuple[int, int, int, str, List[str]]]:
    rows: List[Tuple[int, int, int, str, List[str]]] = []
    for (stack_id, tag_id), live_bytes in sorted(
        per_stack.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top]:
        rows.append(
            (
                stack_id,
                tag_id,
                live_bytes,
                tag_names.get(tag_id, f"tag:{tag_id}"),
                stack_labels.get(stack_id, []),
            )
        )
    return rows


def build_flamegraph_tree(
    state: LiveState,
    stack_labels: Dict[int, List[str]],
    tag_names: Dict[int, str],
) -> Dict[str, object]:
    root: Dict[str, object] = {
        "name": "root",
        "value": 0,
        "children_map": {},
    }

    def add_path(path: List[str], value: int) -> None:
        node = root
        node["value"] = int(node["value"]) + value
        for name in path:
            children_map = node["children_map"]
            assert isinstance(children_map, dict)
            child = children_map.get(name)
            if child is None:
                child = {"name": name, "value": 0, "children_map": {}}
                children_map[name] = child
            node = child
            node["value"] = int(node["value"]) + value

    for size, stack_id, tag_id in state.allocs.values():
        if size <= 0:
            continue
        frames = stack_labels.get(stack_id, [])
        path = [tag_names.get(tag_id, f"tag:{tag_id}"), "[alloc]"]
        if frames:
            path.extend(reversed(frames))
        add_path(path, size)

    for _base, (_reserved, used, tag_id) in state.exec_regions.items():
        if used > 0:
            add_path([tag_names.get(tag_id, f"tag:{tag_id}"), "[jit-exec]"], used)
    for _base, (_reserved, committed, tag_id) in state.guard_regions.items():
        if committed > 0:
            add_path([tag_names.get(tag_id, f"tag:{tag_id}"), "[guard-committed]"], committed)

    def freeze(node: Dict[str, object]) -> Dict[str, object]:
        children_map = node.get("children_map", {})
        assert isinstance(children_map, dict)
        children = sorted(
            (freeze(child) for child in children_map.values()),
            key=lambda child: int(child["value"]),
            reverse=True,
        )
        return {
            "name": node["name"],
            "value": int(node["value"]),
            "children": children,
        }

    return freeze(root)


def snapshot_payload(
    trace: Path,
    time_us: int,
    top: int,
    symbolize: bool,
    frame_cache: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    state, stacks, command_line, images, tag_names = reconstruct_snapshot(trace, time_us)
    per_stack = aggregate_live_allocs(state)
    live_stacks = {
        stack_id: stacks.get(stack_id, [])
        for stack_id, _tag_id in per_stack.keys()
    }
    stack_labels = symbolize_stacks(
        live_stacks,
        images,
        enabled=symbolize,
        frame_cache=frame_cache,
    )
    top_rows = summarize_top_stacks(per_stack, stack_labels, tag_names, top)
    top_tags = summarize_top_tags(state, tag_names, top)
    return {
        "time_us": time_us,
        "command_line": command_line or [],
        "total_live_bytes": state.totals.total,
        "alloc_live_bytes": state.totals.alloc,
        "exec_live_bytes": state.totals.exec,
        "guard_live_bytes": state.totals.guard,
        "live_alloc_count": len(state.allocs),
        "top_live_tags": top_tags,
        "top_live_stacks": [
            {
                "stack_id": stack_id,
                "tag_id": tag_id,
                "tag_name": tag_name,
                "live_bytes": live_bytes,
                "frames": frames,
            }
            for stack_id, tag_id, live_bytes, tag_name, frames in top_rows
        ],
        "flamegraph": build_flamegraph_tree(
            state,
            stack_labels,
            tag_names,
        ),
    }


def command_snapshot(args: argparse.Namespace) -> int:
    payload = snapshot_payload(
        trace=args.trace,
        time_us=args.time_us,
        top=args.top,
        symbolize=args.symbolize,
    )

    if args.collapsed_out is not None:
        # Rebuild collapsed output from a fresh exact snapshot so the file contains
        # every live stack, not only the top rows shown in the summary.
        state, stacks, _, images, tag_names = reconstruct_snapshot(args.trace, args.time_us)
        per_stack = aggregate_live_allocs(state)
        live_stacks = {
            stack_id: stacks.get(stack_id, [])
            for stack_id, _tag_id in per_stack.keys()
        }
        stack_labels = symbolize_stacks(live_stacks, images, enabled=args.symbolize)
        write_collapsed_snapshot(
            args.collapsed_out,
            per_stack,
            stack_labels,
            tag_names,
            state,
        )

    if args.json is not None:
        write_json_output(payload, args.json)
        return 0

    print(f"time_us: {args.time_us}")
    print(f"total_live_bytes: {payload['total_live_bytes']} ({format_mib(payload['total_live_bytes'])})")
    print(f"alloc_live_bytes: {payload['alloc_live_bytes']} ({format_mib(payload['alloc_live_bytes'])})")
    print(f"exec_live_bytes: {payload['exec_live_bytes']} ({format_mib(payload['exec_live_bytes'])})")
    print(f"guard_live_bytes: {payload['guard_live_bytes']} ({format_mib(payload['guard_live_bytes'])})")
    print(f"live_alloc_count: {payload['live_alloc_count']}")
    if args.collapsed_out is not None:
        print(f"collapsed_out: {args.collapsed_out}")
    print("top_live_tags:")
    for item in payload["top_live_tags"]:
        assert isinstance(item, dict)
        print(
            f"  tag={item['tag_name']} total={item['total_live_bytes']} ({format_mib(int(item['total_live_bytes']))}) "
            f"alloc={format_mib(int(item['alloc_live_bytes']))} "
            f"exec={format_mib(int(item['exec_live_bytes']))} "
            f"guard={format_mib(int(item['guard_live_bytes']))}"
        )
    print("top_live_stacks:")
    for item in payload["top_live_stacks"]:
        assert isinstance(item, dict)
        stack_id = item["stack_id"]
        live_bytes = item["live_bytes"]
        tag_name = item["tag_name"]
        frames = item["frames"]
        preview = " <- ".join(frames[:4]) if frames else "<empty>"
        print(
            f"  tag={tag_name} stack={stack_id} live={live_bytes} ({format_mib(live_bytes)}) frames={preview}"
        )
    return 0


def symbolize_stacks(
    stacks: Dict[int, List[str]],
    images: List[Image],
    enabled: bool,
    frame_cache: Optional[Dict[str, str]] = None,
) -> Dict[int, List[str]]:
    if not enabled or not stacks:
        return stacks

    symbolizer = AtosSymbolizer(images)
    if not symbolizer.available:
        return stacks

    cache = frame_cache if frame_cache is not None else {}
    frame_hexes = {
        frame
        for frames in stacks.values()
        for frame in frames
        if frame not in cache
    }
    resolved = symbolizer.resolve_many(frame_hexes)
    cache.update(resolved)
    return {
        stack_id: [cache.get(frame, frame) for frame in frames]
        for stack_id, frames in stacks.items()
    }


class AtosSymbolizer:
    def __init__(self, images: Sequence[Image]) -> None:
        self.images = sorted(images, key=lambda image: image.base)
        self.bases = [image.base for image in self.images]
        self.atos = shutil.which("atos")

    @property
    def available(self) -> bool:
        return self.atos is not None and bool(self.images)

    def resolve_many(self, frame_hexes: Sequence[str]) -> Dict[str, str]:
        resolved: Dict[str, str] = {}
        if not self.available:
            return resolved

        addresses_by_image: Dict[Image, List[str]] = {}
        for frame_hex in frame_hexes:
            image = self.image_for_frame(parse_hex(frame_hex))
            if image is None:
                resolved[frame_hex] = frame_hex
                continue
            addresses_by_image.setdefault(image, []).append(frame_hex)

        for image, addresses in addresses_by_image.items():
            unique_addresses = sorted(set(addresses))
            for chunk_start in range(0, len(unique_addresses), 256):
                chunk = unique_addresses[chunk_start : chunk_start + 256]
                try:
                    completed = subprocess.run(
                        [self.atos, "-o", image.path, "-l", hex(image.base), *chunk],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                except OSError:
                    return resolved
                if completed.returncode != 0:
                    continue
                lines = completed.stdout.splitlines()
                if len(lines) != len(chunk):
                    continue
                for frame_hex, line in zip(chunk, lines):
                    text = line.strip()
                    if not text or text == frame_hex:
                        continue
                    resolved[frame_hex] = f"{text} [{Path(image.path).name}]"
        return resolved

    def image_for_frame(self, address: int) -> Optional[Image]:
        index = bisect.bisect_right(self.bases, address) - 1
        if index < 0:
            return None
        image = self.images[index]
        if index + 1 < len(self.images) and address >= self.images[index + 1].base:
            return None
        if image.size > 0 and address >= image.end:
            return None
        return image


def format_mib(value: int) -> str:
    mib = value / (1024 * 1024)
    return f"{mib:7.2f} MiB"


def write_curve_html(
    out_path: Path,
    trace_path: Path,
    timeline: Dict[str, object],
    spikes: List[Dict[str, int]],
) -> None:
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>sf-nano memtrace curve</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg: #0f1720;
      --panel: #17212b;
      --panel-2: #1d2a36;
      --text: #e7eef6;
      --muted: #9eb0c2;
      --grid: #2b3a49;
      --total: #7dd3fc;
      --alloc: #60a5fa;
      --exec: #f59e0b;
      --guard: #f97316;
      --spike: #ef4444;
      --selected: #22c55e;
    }
    body {
      margin: 0;
      background: linear-gradient(180deg, #0c141c 0%, var(--bg) 100%);
      color: var(--text);
      font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .wrap {
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }
    .title {
      font-size: 20px;
      margin-bottom: 6px;
    }
    .sub {
      color: var(--muted);
      margin-bottom: 18px;
      word-break: break-all;
    }
    .card {
      background: rgba(23, 33, 43, 0.92);
      border: 1px solid rgba(125, 211, 252, 0.12);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
    }
    .legend {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 12px;
      color: var(--muted);
    }
    .legend span::before {
      content: "";
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 999px;
      margin-right: 8px;
      vertical-align: -1px;
      background: currentColor;
    }
    .chart-shell {
      position: relative;
      height: 520px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0)),
        var(--panel-2);
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.04);
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .tooltip {
      position: absolute;
      pointer-events: none;
      min-width: 260px;
      max-width: 420px;
      background: rgba(13, 21, 28, 0.96);
      border: 1px solid rgba(125, 211, 252, 0.22);
      border-radius: 10px;
      padding: 10px 12px;
      box-shadow: 0 20px 48px rgba(0,0,0,0.35);
      color: var(--text);
      opacity: 0;
      transform: translate(-9999px, -9999px);
      transition: opacity 80ms linear;
      white-space: pre-wrap;
    }
    .details {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }
    .detail {
      background: rgba(255,255,255,0.03);
      border-radius: 10px;
      padding: 12px;
      border: 1px solid rgba(255,255,255,0.04);
    }
    .detail .k {
      color: var(--muted);
      margin-bottom: 4px;
    }
    .detail .v {
      font-size: 15px;
      word-break: break-word;
    }
    .cmd {
      color: #cbd5e1;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title">sf-nano memtrace curve</div>
    <div class="sub">__TRACE_PATH_HTML__</div>
    <div class="card">
      <div class="legend">
        <span style="color: var(--total)">total live</span>
        <span style="color: var(--alloc)">alloc live</span>
        <span style="color: var(--exec)">exec live</span>
        <span style="color: var(--guard)">guard committed</span>
        <span style="color: var(--spike)">detected spike</span>
        <span style="color: var(--selected)">selected bucket</span>
      </div>
      <div class="chart-shell">
        <canvas id="chart"></canvas>
        <div id="tooltip" class="tooltip"></div>
      </div>
      <div class="details">
        <div class="detail">
          <div class="k">Selected Peak Time</div>
          <div class="v" id="selected-time">click the chart</div>
        </div>
        <div class="detail">
          <div class="k">Selected Peak Live</div>
          <div class="v" id="selected-live">-</div>
        </div>
        <div class="detail">
          <div class="k">Snapshot Command</div>
          <div class="v cmd" id="snapshot-cmd">python3 tools/memtrace/analyze.py snapshot TRACE --time-us T --symbolize --collapsed-out /tmp/peak.folded</div>
        </div>
      </div>
    </div>
  </div>
  <script id="timeline-data" type="application/json">__TIMELINE_JSON__</script>
  <script id="spike-data" type="application/json">__SPIKE_JSON__</script>
  <script>
    const timeline = JSON.parse(document.getElementById("timeline-data").textContent);
    const spikes = JSON.parse(document.getElementById("spike-data").textContent);
    const tracePath = __TRACE_PATH_JSON__;
    const buckets = timeline.buckets;
    const canvas = document.getElementById("chart");
    const tooltip = document.getElementById("tooltip");
    const ctx = canvas.getContext("2d");
    const selectedTime = document.getElementById("selected-time");
    const selectedLive = document.getElementById("selected-live");
    const snapshotCmd = document.getElementById("snapshot-cmd");
    let hoverIndex = null;
    let selectedIndex = null;

    function formatBytes(bytes) {
      const mib = bytes / (1024 * 1024);
      return `${mib.toFixed(2)} MiB (${bytes} B)`;
    }

    function formatTime(us) {
      const ms = us / 1000;
      return `${ms.toFixed(3)} ms (${us} us)`;
    }

    function maxOf(key) {
      let max = 0;
      for (const bucket of buckets) {
        if (bucket[key] > max) max = bucket[key];
      }
      return max;
    }

    const maxBytes = Math.max(
      maxOf("max_total_live_bytes"),
      maxOf("max_alloc_live_bytes"),
      maxOf("max_exec_live_bytes"),
      maxOf("max_guard_live_bytes"),
      1
    );

    const dpi = window.devicePixelRatio || 1;
    const pad = { top: 24, right: 24, bottom: 42, left: 72 };

    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * dpi);
      canvas.height = Math.floor(rect.height * dpi);
      ctx.setTransform(dpi, 0, 0, dpi, 0, 0);
      draw();
    }

    function plotWidth() {
      return canvas.clientWidth - pad.left - pad.right;
    }

    function plotHeight() {
      return canvas.clientHeight - pad.top - pad.bottom;
    }

    function xForIndex(index) {
      if (buckets.length <= 1) return pad.left;
      return pad.left + (index / (buckets.length - 1)) * plotWidth();
    }

    function yForBytes(value) {
      const ratio = value / maxBytes;
      return pad.top + (1 - ratio) * plotHeight();
    }

    function bucketAtClientX(clientX) {
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      if (x < pad.left) return 0;
      if (x > rect.width - pad.right) return buckets.length - 1;
      const ratio = (x - pad.left) / Math.max(plotWidth(), 1);
      const index = Math.round(ratio * Math.max(buckets.length - 1, 0));
      return Math.max(0, Math.min(buckets.length - 1, index));
    }

    function strokeSeries(key, color, lineWidth) {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      for (let i = 0; i < buckets.length; i++) {
        const bucket = buckets[i];
        const x = xForIndex(i);
        const y = yForBytes(bucket[key]);
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, yForBytes(buckets[i - 1][key]));
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    function drawAxes() {
      ctx.strokeStyle = "rgba(158,176,194,0.25)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, canvas.clientHeight - pad.bottom);
      ctx.lineTo(canvas.clientWidth - pad.right, canvas.clientHeight - pad.bottom);
      ctx.stroke();

      ctx.fillStyle = "rgba(158,176,194,0.9)";
      ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let i = 0; i <= 4; i++) {
        const value = (maxBytes * i) / 4;
        const y = yForBytes(value);
        ctx.strokeStyle = "rgba(43,58,73,0.75)";
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(canvas.clientWidth - pad.right, y);
        ctx.stroke();
        ctx.fillText(formatBytes(value).replace(/ \\(.*/, ""), pad.left - 10, y);
      }

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const totalTimeUs = buckets.length ? buckets[buckets.length - 1].max_time_us : 0;
      for (let i = 0; i <= 4; i++) {
        const ratio = i / 4;
        const x = pad.left + ratio * plotWidth();
        const us = Math.round(totalTimeUs * ratio);
        ctx.fillStyle = "rgba(158,176,194,0.9)";
        ctx.fillText((us / 1000).toFixed(1) + " ms", x, canvas.clientHeight - pad.bottom + 10);
      }
    }

    function drawSpikes() {
      for (const spike of spikes) {
        const index = buckets.findIndex(bucket => bucket.max_time_us >= spike.time_us);
        if (index < 0) continue;
        const x = xForIndex(index);
        ctx.strokeStyle = "rgba(239,68,68,0.55)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, canvas.clientHeight - pad.bottom);
        ctx.stroke();
      }
    }

    function drawSelection(index, color) {
      if (index == null || index < 0 || index >= buckets.length) return;
      const x = xForIndex(index);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, canvas.clientHeight - pad.bottom);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, yForBytes(buckets[index].max_total_live_bytes), 4, 0, Math.PI * 2);
      ctx.fill();
    }

    function updateSelection(index) {
      const bucket = buckets[index];
      selectedIndex = index;
      selectedTime.textContent = `${formatTime(bucket.max_time_us)} in bucket starting ${formatTime(bucket.start_us)}`;
      selectedLive.textContent = [
        `total: ${formatBytes(bucket.max_total_live_bytes)}`,
        `alloc: ${formatBytes(bucket.max_alloc_live_bytes)}`,
        `exec: ${formatBytes(bucket.max_exec_live_bytes)}`,
        `guard: ${formatBytes(bucket.max_guard_live_bytes)}`
      ].join("\\n");
      snapshotCmd.textContent =
        `python3 tools/memtrace/analyze.py snapshot ${tracePath} --time-us ${bucket.max_time_us} --symbolize --collapsed-out /tmp/peak.folded`;
      draw();
    }

    function showTooltip(index, clientX, clientY) {
      const bucket = buckets[index];
      tooltip.textContent = [
        `bucket: ${index}`,
        `bucket start: ${formatTime(bucket.start_us)}`,
        `peak time: ${formatTime(bucket.max_time_us)}`,
        `peak total: ${formatBytes(bucket.max_total_live_bytes)}`,
        `peak alloc: ${formatBytes(bucket.max_alloc_live_bytes)}`,
        `peak exec: ${formatBytes(bucket.max_exec_live_bytes)}`,
        `peak guard: ${formatBytes(bucket.max_guard_live_bytes)}`,
        `end total: ${formatBytes(bucket.end_total_live_bytes)}`
      ].join("\\n");
      const host = tooltip.parentElement || document.body;
      const rect = host.getBoundingClientRect();
      let x = clientX - rect.left + 14;
      let y = clientY - rect.top + 14;
      const maxX = Math.max(8, rect.width - tooltip.offsetWidth - 8);
      const maxY = Math.max(8, rect.height - tooltip.offsetHeight - 8);
      x = Math.min(Math.max(8, x), maxX);
      y = Math.min(Math.max(8, y), maxY);
      tooltip.style.opacity = "1";
      tooltip.style.transform = `translate(${x}px, ${y}px)`;
    }

    function hideTooltip() {
      tooltip.style.opacity = "0";
      tooltip.style.transform = "translate(-9999px, -9999px)";
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      drawAxes();
      drawSpikes();
      strokeSeries("max_guard_live_bytes", "#f97316", 1.2);
      strokeSeries("max_exec_live_bytes", "#f59e0b", 1.2);
      strokeSeries("max_alloc_live_bytes", "#60a5fa", 1.2);
      strokeSeries("max_total_live_bytes", "#7dd3fc", 2.4);
      drawSelection(selectedIndex, "#22c55e");
      drawSelection(hoverIndex, "rgba(231,238,246,0.8)");
    }

    canvas.addEventListener("mousemove", event => {
      hoverIndex = bucketAtClientX(event.clientX);
      showTooltip(hoverIndex, event.clientX, event.clientY);
      draw();
    });

    canvas.addEventListener("mouseleave", () => {
      hoverIndex = null;
      hideTooltip();
      draw();
    });

    canvas.addEventListener("click", event => {
      updateSelection(bucketAtClientX(event.clientX));
    });

    const bestSpike = spikes.length
      ? spikes.reduce((best, spike) => spike.peak_bytes > best.peak_bytes ? spike : best, spikes[0])
      : null;
    if (bestSpike) {
      const index = buckets.findIndex(bucket => bucket.max_time_us >= bestSpike.time_us);
      if (index >= 0) updateSelection(index);
    } else if (buckets.length) {
      updateSelection(0);
    } else {
      draw();
    }

    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
  </script>
</body>
</html>
"""
    html_text = (
        template.replace("__TRACE_PATH_HTML__", html.escape(str(trace_path)))
        .replace("__TRACE_PATH_JSON__", json.dumps(str(trace_path)))
        .replace("__TIMELINE_JSON__", html.escape(json.dumps(timeline, separators=(",", ":"))))
        .replace("__SPIKE_JSON__", html.escape(json.dumps(spikes, separators=(",", ":"))))
    )
    out_path.write_text(html_text, encoding="utf-8")


class ViewerContext:
    def __init__(
        self,
        trace: Path,
        bucket_us: int,
        spike_count: int,
        top: int,
        symbolize: bool,
    ) -> None:
        self.trace = trace
        self.bucket_us = bucket_us
        self.top = top
        self.symbolize = symbolize
        self._snapshot_cache: Dict[Tuple[int, bool, int], Dict[str, object]] = {}
        self._frame_symbol_cache: Dict[str, str] = {}

        buckets = build_buckets(trace, bucket_us)
        self.timeline = timeline_payload(buckets, bucket_us)
        self.spikes = [
            {
                "time_us": peak.time_us,
                "peak_bytes": peak.peak_bytes,
                "prominence_bytes": peak.prominence_bytes,
            }
            for peak in find_spikes_from_buckets(buckets)[:spike_count]
        ]

    def timeline_payload(self) -> Dict[str, object]:
        return {
            "trace_path": str(self.trace),
            "bucket_us": self.bucket_us,
            "top": self.top,
            "symbolize": self.symbolize,
            "timeline": self.timeline,
            "spikes": self.spikes,
        }

    def snapshot(self, time_us: int, symbolize: Optional[bool] = None) -> Dict[str, object]:
        use_symbolize = self.symbolize if symbolize is None else symbolize
        key = (time_us, use_symbolize, self.top)
        cached = self._snapshot_cache.get(key)
        if cached is not None:
            return cached
        payload = snapshot_payload(
            trace=self.trace,
            time_us=time_us,
            top=self.top,
            symbolize=use_symbolize,
            frame_cache=self._frame_symbol_cache if use_symbolize else None,
        )
        self._snapshot_cache[key] = payload
        return payload


def viewer_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>sf-nano memtrace viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg: #0b1320;
      --panel: rgba(17, 27, 39, 0.94);
      --panel-2: rgba(22, 36, 51, 0.98);
      --text: #e8f1f7;
      --muted: #9db0c1;
      --grid: #274053;
      --border: rgba(125, 211, 252, 0.12);
      --total: #7dd3fc;
      --alloc: #60a5fa;
      --exec: #f59e0b;
      --guard: #f97316;
      --spike: #ef4444;
      --selected: #22c55e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at top, #162334 0%, var(--bg) 45%);
      color: var(--text);
      font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .wrap {
      max-width: 1600px;
      margin: 0 auto;
      padding: 24px;
    }
    .title { font-size: 22px; margin-bottom: 6px; }
    .sub { color: var(--muted); margin-bottom: 18px; word-break: break-all; }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 24px 80px rgba(0,0,0,0.30);
    }
    .legend {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 12px;
      color: var(--muted);
    }
    .legend span::before {
      content: "";
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 999px;
      margin-right: 8px;
      vertical-align: -1px;
      background: currentColor;
    }
    .chart-shell, .flame-shell {
      position: relative;
      background: var(--panel-2);
      border: 1px solid rgba(255,255,255,0.04);
      border-radius: 12px;
      overflow: hidden;
    }
    .chart-shell { height: 420px; }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .tooltip {
      position: absolute;
      top: 0;
      left: 0;
      pointer-events: none;
      min-width: 240px;
      max-width: 460px;
      background: rgba(8, 14, 22, 0.97);
      border: 1px solid rgba(125, 211, 252, 0.2);
      border-radius: 10px;
      padding: 10px 12px;
      box-shadow: 0 20px 48px rgba(0,0,0,0.35);
      color: var(--text);
      opacity: 0;
      transform: translate(-9999px, -9999px);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .details {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }
    .detail {
      background: rgba(255,255,255,0.03);
      border-radius: 10px;
      padding: 12px;
      border: 1px solid rgba(255,255,255,0.04);
      white-space: pre-wrap;
    }
    .detail .k { color: var(--muted); margin-bottom: 6px; }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(360px, 0.65fr);
      gap: 16px;
      margin-top: 16px;
    }
    .flame-shell {
      min-height: 360px;
      padding: 10px;
    }
    .flame-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 8px;
      color: var(--muted);
    }
    #snapshot-status {
      overflow-wrap: anywhere;
      word-break: break-word;
      text-align: right;
      flex: 1 1 320px;
    }
    svg { width: 100%; display: block; }
    .topstacks {
      max-height: 720px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .status {
      color: var(--muted);
      margin-top: 10px;
    }
    @media (max-width: 1100px) {
      .details, .layout { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title">sf-nano memtrace viewer</div>
    <div class="sub" id="trace-path">loading…</div>
    <div class="card">
      <div class="legend">
        <span style="color: var(--total)">total live</span>
        <span style="color: var(--alloc)">alloc live</span>
        <span style="color: var(--exec)">exec live</span>
        <span style="color: var(--guard)">guard committed</span>
        <span style="color: var(--spike)">detected spike</span>
        <span style="color: var(--selected)">selected time</span>
      </div>
      <div class="chart-shell">
        <canvas id="chart"></canvas>
        <div id="curve-tooltip" class="tooltip"></div>
      </div>
      <div class="details">
        <div class="detail">
          <div class="k">Selected Time</div>
          <div id="selected-time">loading…</div>
        </div>
        <div class="detail">
          <div class="k">Selected Memory</div>
          <div id="selected-memory">-</div>
        </div>
        <div class="detail">
          <div class="k">Snapshot Command</div>
          <div id="selected-command">-</div>
        </div>
      </div>
      <div class="layout">
        <div class="flame-shell">
          <div class="flame-header">
            <div>Flamegraph At Selected Time</div>
            <div id="snapshot-status">click the curve to load a snapshot</div>
          </div>
          <svg id="flamegraph"></svg>
          <div id="flame-tooltip" class="tooltip"></div>
        </div>
        <div class="detail topstacks">
          <div class="k">Top Live Tags And Stacks</div>
          <div id="top-stacks">-</div>
        </div>
      </div>
    </div>
  </div>
  <script>
    const canvas = document.getElementById("chart");
    const curveTooltip = document.getElementById("curve-tooltip");
    const flameTooltip = document.getElementById("flame-tooltip");
    const tracePathEl = document.getElementById("trace-path");
    const selectedTimeEl = document.getElementById("selected-time");
    const selectedMemoryEl = document.getElementById("selected-memory");
    const selectedCommandEl = document.getElementById("selected-command");
    const snapshotStatusEl = document.getElementById("snapshot-status");
    const topStacksEl = document.getElementById("top-stacks");
    const flameSvg = document.getElementById("flamegraph");
    const ctx = canvas.getContext("2d");
    const ns = "http://www.w3.org/2000/svg";

    let tracePath = "";
    let buckets = [];
    let spikes = [];
    let hoverIndex = null;
    let selectedIndex = null;
    let chartState = null;
    let currentSnapshotRequest = 0;

    function formatBytes(bytes) {
      const mib = bytes / (1024 * 1024);
      return `${mib.toFixed(2)} MiB (${bytes} B)`;
    }

    function formatTime(us) {
      return `${(us / 1000).toFixed(3)} ms (${us} us)`;
    }

    function colorForName(name) {
      if (name === "[alloc]") return "#60a5fa";
      if (name === "[jit-exec]") return "#f59e0b";
      if (name === "[guard-committed]") return "#f97316";
      let hash = 0;
      for (let i = 0; i < name.length; i++) {
        hash = ((hash << 5) - hash + name.charCodeAt(i)) | 0;
      }
      const hue = Math.abs(hash) % 360;
      return `hsl(${hue} 62% 56%)`;
    }

    function setTooltip(el, lines, clientX, clientY) {
      el.textContent = lines.join("\\n");
      const host = el.parentElement || document.body;
      const rect = host.getBoundingClientRect();
      let x = clientX - rect.left + 12;
      let y = clientY - rect.top + 12;
      const maxX = Math.max(8, rect.width - el.offsetWidth - 8);
      const maxY = Math.max(8, rect.height - el.offsetHeight - 8);
      x = Math.min(Math.max(8, x), maxX);
      y = Math.min(Math.max(8, y), maxY);
      el.style.opacity = "1";
      el.style.transform = `translate(${x}px, ${y}px)`;
    }

    function hideTooltip(el) {
      el.style.opacity = "0";
      el.style.transform = "translate(-9999px, -9999px)";
    }

    function computeChartState() {
      const maxBytes = Math.max(
        ...buckets.map(bucket => Math.max(
          bucket.max_total_live_bytes,
          bucket.max_alloc_live_bytes,
          bucket.max_exec_live_bytes,
          bucket.max_guard_live_bytes
        )),
        1
      );
      return {
        dpi: window.devicePixelRatio || 1,
        pad: { top: 24, right: 24, bottom: 42, left: 80 },
        maxBytes
      };
    }

    function resizeCanvas() {
      if (!chartState) return;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * chartState.dpi);
      canvas.height = Math.floor(rect.height * chartState.dpi);
      ctx.setTransform(chartState.dpi, 0, 0, chartState.dpi, 0, 0);
      drawCurve();
    }

    function plotWidth() { return canvas.clientWidth - chartState.pad.left - chartState.pad.right; }
    function plotHeight() { return canvas.clientHeight - chartState.pad.top - chartState.pad.bottom; }

    function xForIndex(index) {
      if (buckets.length <= 1) return chartState.pad.left;
      return chartState.pad.left + (index / (buckets.length - 1)) * plotWidth();
    }

    function yForBytes(bytes) {
      return chartState.pad.top + (1 - bytes / chartState.maxBytes) * plotHeight();
    }

    function bucketAtClientX(clientX) {
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      if (x <= chartState.pad.left) return 0;
      if (x >= rect.width - chartState.pad.right) return buckets.length - 1;
      const ratio = (x - chartState.pad.left) / Math.max(plotWidth(), 1);
      return Math.max(0, Math.min(buckets.length - 1, Math.round(ratio * (buckets.length - 1))));
    }

    function strokeSeries(key, color, lineWidth) {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      for (let i = 0; i < buckets.length; i++) {
        const x = xForIndex(i);
        const y = yForBytes(buckets[i][key]);
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, yForBytes(buckets[i - 1][key]));
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    function drawAxes() {
      const pad = chartState.pad;
      ctx.strokeStyle = "rgba(157,176,193,0.22)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, canvas.clientHeight - pad.bottom);
      ctx.lineTo(canvas.clientWidth - pad.right, canvas.clientHeight - pad.bottom);
      ctx.stroke();

      ctx.fillStyle = "rgba(157,176,193,0.92)";
      ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let i = 0; i <= 4; i++) {
        const value = (chartState.maxBytes * i) / 4;
        const y = yForBytes(value);
        ctx.strokeStyle = "rgba(39,64,83,0.72)";
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(canvas.clientWidth - pad.right, y);
        ctx.stroke();
        ctx.fillText((value / (1024 * 1024)).toFixed(1) + " MiB", pad.left - 10, y);
      }

      const totalTime = buckets.length ? buckets[buckets.length - 1].max_time_us : 0;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let i = 0; i <= 4; i++) {
        const ratio = i / 4;
        const x = pad.left + ratio * plotWidth();
        const us = Math.round(totalTime * ratio);
        ctx.fillStyle = "rgba(157,176,193,0.92)";
        ctx.fillText((us / 1000).toFixed(1) + " ms", x, canvas.clientHeight - pad.bottom + 10);
      }
    }

    function drawSpikeMarkers() {
      for (const spike of spikes) {
        const index = buckets.findIndex(bucket => bucket.max_time_us >= spike.time_us);
        if (index < 0) continue;
        const x = xForIndex(index);
        ctx.strokeStyle = "rgba(239,68,68,0.45)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, chartState.pad.top);
        ctx.lineTo(x, canvas.clientHeight - chartState.pad.bottom);
        ctx.stroke();
      }
    }

    function drawSelection(index, color) {
      if (index == null || index < 0 || index >= buckets.length) return;
      const x = xForIndex(index);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, chartState.pad.top);
      ctx.lineTo(x, canvas.clientHeight - chartState.pad.bottom);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, yForBytes(buckets[index].max_total_live_bytes), 4, 0, Math.PI * 2);
      ctx.fill();
    }

    function drawCurve() {
      if (!chartState) return;
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      drawAxes();
      drawSpikeMarkers();
      strokeSeries("max_guard_live_bytes", "#f97316", 1.1);
      strokeSeries("max_exec_live_bytes", "#f59e0b", 1.1);
      strokeSeries("max_alloc_live_bytes", "#60a5fa", 1.1);
      strokeSeries("max_total_live_bytes", "#7dd3fc", 2.3);
      drawSelection(selectedIndex, "#22c55e");
      drawSelection(hoverIndex, "rgba(231,241,247,0.78)");
    }

    function updateSelectedBucket(index) {
      selectedIndex = index;
      const bucket = buckets[index];
      selectedTimeEl.textContent = `${formatTime(bucket.max_time_us)} in bucket starting ${formatTime(bucket.start_us)}`;
      selectedMemoryEl.textContent = [
        `total: ${formatBytes(bucket.max_total_live_bytes)}`,
        `alloc: ${formatBytes(bucket.max_alloc_live_bytes)}`,
        `exec: ${formatBytes(bucket.max_exec_live_bytes)}`,
        `guard: ${formatBytes(bucket.max_guard_live_bytes)}`
      ].join("\\n");
      selectedCommandEl.textContent =
        `python3 tools/memtrace/analyze.py snapshot ${tracePath} --time-us ${bucket.max_time_us} --symbolize --collapsed-out /tmp/peak.folded`;
      drawCurve();
      loadSnapshot(bucket.max_time_us);
    }

    async function loadSnapshot(timeUs) {
      const requestId = ++currentSnapshotRequest;
      snapshotStatusEl.textContent = `loading snapshot at ${formatTime(timeUs)}…`;
      topStacksEl.textContent = "loading…";
      while (flameSvg.firstChild) flameSvg.removeChild(flameSvg.firstChild);
      try {
        const response = await fetch(`/api/snapshot?time_us=${timeUs}`);
        if (!response.ok) throw new Error(await response.text());
        const payload = await response.json();
        if (requestId !== currentSnapshotRequest) return;
        renderSnapshot(payload);
      } catch (error) {
        if (requestId !== currentSnapshotRequest) return;
        snapshotStatusEl.textContent = `snapshot failed: ${error}`;
        topStacksEl.textContent = "";
      }
    }

    function renderSnapshot(payload) {
      snapshotStatusEl.textContent =
        `snapshot loaded: ${formatBytes(payload.total_live_bytes)} at ${formatTime(payload.time_us)}`;
      const tagText = (payload.top_live_tags || []).map(item => {
        return [
          `${item.tag_name}: ${formatBytes(item.total_live_bytes)}`,
          `  alloc=${formatBytes(item.alloc_live_bytes)} exec=${formatBytes(item.exec_live_bytes)} guard=${formatBytes(item.guard_live_bytes)}`
        ].join("\\n");
      }).join("\\n\\n");
      const stackText = (payload.top_live_stacks || []).map(item => {
        const preview = (item.frames || []).slice(0, 6).join(" <- ") || "<empty>";
        return `${item.tag_name} | ${formatBytes(item.live_bytes)}\\n${preview}`;
      }).join("\\n\\n");
      topStacksEl.textContent = [
        "Top Tags",
        tagText || "<none>",
        "",
        "Top Stack Sites",
        stackText || "<none>"
      ].join("\\n");
      renderFlamegraph(payload.flamegraph);
    }

    function maxDepth(node) {
      if (!node.children || !node.children.length) return 1;
      return 1 + Math.max(...node.children.map(maxDepth));
    }

    function renderFlamegraph(tree) {
      const children = tree.children || [];
      const barHeight = 24;
      const approxCharWidth = 7.2;
      const depth = Math.max(...children.map(maxDepth), 1);
      const svgWidth = flameSvg.clientWidth || flameSvg.parentElement.clientWidth - 20 || 900;
      const svgHeight = Math.max(180, depth * barHeight + 16);
      flameSvg.setAttribute("viewBox", `0 0 ${svgWidth} ${svgHeight}`);
      flameSvg.setAttribute("height", String(svgHeight));
      while (flameSvg.firstChild) flameSvg.removeChild(flameSvg.firstChild);

      function appendNode(node, x, y, width, path) {
        if (width < 0.9 || node.value <= 0) return;
        const group = document.createElementNS(ns, "g");
        const rect = document.createElementNS(ns, "rect");
        rect.setAttribute("x", String(x));
        rect.setAttribute("y", String(y));
        rect.setAttribute("width", String(width));
        rect.setAttribute("height", String(barHeight - 1));
        rect.setAttribute("fill", colorForName(node.name));
        rect.setAttribute("stroke", "rgba(0,0,0,0.2)");
        rect.dataset.tooltip = [
          node.name,
          formatBytes(node.value),
          path.join(" <- ")
        ].join("\\n");
        rect.addEventListener("mousemove", event => {
          setTooltip(flameTooltip, rect.dataset.tooltip.split("\\n"), event.clientX, event.clientY);
        });
        rect.addEventListener("mouseleave", () => hideTooltip(flameTooltip));
        group.appendChild(rect);

        if (width > 120) {
          const text = document.createElementNS(ns, "text");
          text.setAttribute("x", String(x + 6));
          text.setAttribute("y", String(y + 16));
          text.setAttribute("fill", "rgba(8,14,22,0.88)");
          text.setAttribute("font-size", "12");
          text.setAttribute("font-family", "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace");
          text.setAttribute("pointer-events", "none");
          const maxChars = Math.max(4, Math.floor((width - 12) / approxCharWidth));
          text.textContent =
            node.name.length > maxChars ? node.name.slice(0, maxChars - 1) + "…" : node.name;
          group.appendChild(text);
        }

        flameSvg.appendChild(group);
        let cursor = x;
        for (const child of node.children || []) {
          const childWidth = width * (child.value / node.value);
          appendNode(child, cursor, y + barHeight, childWidth, path.concat(child.name));
          cursor += childWidth;
        }
      }

      let cursor = 0;
      for (const child of children) {
        const childWidth = svgWidth * (child.value / tree.value);
        appendNode(child, cursor, 0, childWidth, [child.name]);
        cursor += childWidth;
      }
    }

    async function loadTimeline() {
      const response = await fetch("/api/timeline");
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      tracePath = payload.trace_path;
      tracePathEl.textContent = tracePath;
      buckets = payload.timeline.buckets;
      spikes = payload.spikes;
      chartState = computeChartState();
      resizeCanvas();
      const bestSpike = spikes.length
        ? spikes.reduce((best, spike) => spike.peak_bytes > best.peak_bytes ? spike : best, spikes[0])
        : null;
      if (bestSpike) {
        const index = buckets.findIndex(bucket => bucket.max_time_us >= bestSpike.time_us);
        updateSelectedBucket(index >= 0 ? index : 0);
      } else if (buckets.length) {
        updateSelectedBucket(0);
      }
    }

    canvas.addEventListener("mousemove", event => {
      if (!buckets.length) return;
      hoverIndex = bucketAtClientX(event.clientX);
      const bucket = buckets[hoverIndex];
      setTooltip(curveTooltip, [
        `bucket: ${hoverIndex}`,
        `peak time: ${formatTime(bucket.max_time_us)}`,
        `peak total: ${formatBytes(bucket.max_total_live_bytes)}`,
        `peak alloc: ${formatBytes(bucket.max_alloc_live_bytes)}`,
        `peak exec: ${formatBytes(bucket.max_exec_live_bytes)}`,
        `peak guard: ${formatBytes(bucket.max_guard_live_bytes)}`
      ], event.clientX, event.clientY);
      drawCurve();
    });

    canvas.addEventListener("mouseleave", () => {
      hoverIndex = null;
      hideTooltip(curveTooltip);
      drawCurve();
    });

    canvas.addEventListener("click", event => {
      if (!buckets.length) return;
      updateSelectedBucket(bucketAtClientX(event.clientX));
    });

    window.addEventListener("resize", resizeCanvas);
    loadTimeline().catch(error => {
      tracePathEl.textContent = `failed to load timeline: ${error}`;
    });
  </script>
</body>
</html>
"""


def make_viewer_handler(context: ViewerContext) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                body = viewer_html().encode("utf-8")
                self.send_body("text/html; charset=utf-8", body)
                return

            if parsed.path == "/api/timeline":
                payload = context.timeline_payload()
                return self.send_json(payload)

            if parsed.path == "/api/snapshot":
                params = urllib.parse.parse_qs(parsed.query)
                try:
                    time_us = int(params.get("time_us", [""])[0])
                except (TypeError, ValueError):
                    return self.send_error(400, "time_us is required")
                symbolize_values = params.get("symbolize")
                symbolize = context.symbolize
                if symbolize_values:
                    symbolize = symbolize_values[0] not in ("0", "false", "False")
                try:
                    payload = context.snapshot(time_us, symbolize=symbolize)
                except Exception as exc:  # pragma: no cover - debug surface
                    return self.send_error(500, f"snapshot failed: {exc}")
                return self.send_json(payload)

            self.send_error(404, "not found")

        def send_json(self, payload: object) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_body("application/json; charset=utf-8", body)

        def send_body(self, content_type: str, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            try:
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def run_viewer_server(
    context: ViewerContext,
    host: str,
    port: int,
    open_browser: bool,
) -> int:
    server = http.server.ThreadingHTTPServer((host, port), make_viewer_handler(context))
    url = f"http://{server.server_address[0]}:{server.server_address[1]}/"
    print(url)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception as exc:
            print(f"warning: failed to open browser: {exc}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nviewer stopped", file=sys.stderr)
    finally:
        server.server_close()
    return 0


def command_serve(args: argparse.Namespace) -> int:
    context = ViewerContext(
        trace=args.trace,
        bucket_us=args.bucket_us,
        spike_count=args.spikes,
        top=args.top,
        symbolize=not args.raw_pcs,
    )
    return run_viewer_server(
        context=context,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


def command_record_view(args: argparse.Namespace) -> int:
    cli_args = list(args.cli_args)
    if cli_args and cli_args[0] == "--":
        cli_args = cli_args[1:]
    if not cli_args:
        print("record-view requires sf-nano-cli arguments after '--'", file=sys.stderr)
        return 2

    trace_path = args.trace_out
    if trace_path is None:
        fd, temp_path = tempfile.mkstemp(prefix="sf-memtrace-", suffix=".jsonl")
        Path(temp_path).unlink(missing_ok=True)
        trace_path = Path(temp_path)
        try:
            import os

            os.close(fd)
        except OSError:
            pass

    command = [
        "cargo",
        "run",
        "--release",
        "-p",
        "sf-nano-cli",
        "--features",
        "memtrace",
        "--bin",
        "sf-nano-cli",
        "--",
        "--memtrace",
        "--memtrace-output",
        str(trace_path),
        *cli_args,
    ]
    completed = subprocess.run(command)
    if completed.returncode != 0:
        return completed.returncode

    context = ViewerContext(
        trace=trace_path,
        bucket_us=args.bucket_us,
        spike_count=args.spikes,
        top=args.top,
        symbolize=not args.raw_pcs,
    )
    return run_viewer_server(
        context=context,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze sf-nano memtrace raw logs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser(
        "serve",
        help="start a local memtrace viewer with curve and click-to-snapshot flamegraph",
    )
    serve.add_argument("trace", type=Path, help="raw memtrace JSONL file")
    serve.add_argument("--bucket-us", type=int, default=1_000, help="timeline bucket size in microseconds")
    serve.add_argument("--spikes", type=int, default=12, help="number of spike markers to include")
    serve.add_argument("--top", type=int, default=20, help="number of live stacks to show in the side panel")
    serve.add_argument("--host", default="127.0.0.1", help="bind host for the local viewer")
    serve.add_argument("--port", type=int, default=0, help="bind port; 0 chooses a free port")
    serve.add_argument("--no-browser", action="store_true", help="do not auto-open the browser")
    serve.add_argument("--raw-pcs", action="store_true", help="skip symbolization and show raw PCs")
    serve.set_defaults(func=command_serve)

    record_view = subparsers.add_parser(
        "record-view",
        help="run sf-nano-cli with memtrace, then open the local viewer",
    )
    record_view.add_argument("--trace-out", type=Path, help="trace path; defaults to a temp file")
    record_view.add_argument("--bucket-us", type=int, default=1_000, help="timeline bucket size in microseconds")
    record_view.add_argument("--spikes", type=int, default=12, help="number of spike markers to include")
    record_view.add_argument("--top", type=int, default=20, help="number of live stacks to show in the side panel")
    record_view.add_argument("--host", default="127.0.0.1", help="bind host for the local viewer")
    record_view.add_argument("--port", type=int, default=0, help="bind port; 0 chooses a free port")
    record_view.add_argument("--no-browser", action="store_true", help="do not auto-open the browser")
    record_view.add_argument("--raw-pcs", action="store_true", help="skip symbolization and show raw PCs")
    record_view.add_argument(
        "cli_args",
        nargs=argparse.REMAINDER,
        help="arguments passed to sf-nano-cli after '--'",
    )
    record_view.set_defaults(func=command_record_view)

    timeline = subparsers.add_parser(
        "timeline",
        help="emit bucketed live-memory data for step-curve visualization",
    )
    timeline.add_argument("trace", type=Path, help="raw memtrace JSONL file")
    timeline.add_argument(
        "--bucket-us",
        type=int,
        default=1_000,
        help="timeline bucket size in microseconds",
    )
    timeline.add_argument("--json", type=argparse.FileType("w"), help="write timeline as JSON")
    timeline.set_defaults(func=command_timeline)

    curve = subparsers.add_parser(
        "curve-html",
        help="write a standalone HTML viewer for the bucketed memory curve",
    )
    curve.add_argument("trace", type=Path, help="raw memtrace JSONL file")
    curve.add_argument(
        "--bucket-us",
        type=int,
        default=1_000,
        help="timeline bucket size in microseconds",
    )
    curve.add_argument(
        "--spikes",
        type=int,
        default=12,
        help="number of detected spikes to mark on the curve",
    )
    curve.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output HTML path",
    )
    curve.set_defaults(func=command_curve_html)

    spikes = subparsers.add_parser("spikes", help="find candidate memory spike timestamps")
    spikes.add_argument("trace", type=Path, help="raw memtrace JSONL file")
    spikes.add_argument("--bucket-us", type=int, default=1_000, help="timeline bucket size in microseconds")
    spikes.add_argument("--top", type=int, default=20, help="number of spikes to print")
    spikes.add_argument("--json", type=argparse.FileType("w"), help="write spikes as JSON")
    spikes.set_defaults(func=command_spikes)

    snapshot = subparsers.add_parser(
        "snapshot",
        help="reconstruct live memory at one timestamp and emit collapsed stacks",
    )
    snapshot.add_argument("trace", type=Path, help="raw memtrace JSONL file")
    snapshot.add_argument("--time-us", type=int, required=True, help="target time in microseconds")
    snapshot.add_argument(
        "--collapsed-out",
        type=Path,
        help="write collapsed stacks suitable for flamegraph tools",
    )
    snapshot.add_argument(
        "--symbolize",
        action="store_true",
        help="symbolize snapshot frames with atos when image records are available",
    )
    snapshot.add_argument("--top", type=int, default=20, help="number of top live stacks to print")
    snapshot.add_argument("--json", type=argparse.FileType("w"), help="write snapshot summary as JSON")
    snapshot.set_defaults(func=command_snapshot)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
