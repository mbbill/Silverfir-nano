#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


LOCAL_COUNT = 64
TARGET_WASM_BYTES = 200 * 1024


def segment_lines(segment_index: int) -> list[str]:
    a = segment_index % LOCAL_COUNT
    b = (segment_index + 7) % LOCAL_COUNT
    c = (segment_index + 19) % LOCAL_COUNT
    d = (segment_index + 31) % LOCAL_COUNT

    delta = 11 + segment_index * 3
    mask = ((segment_index * 13) ^ 0x5A) & 255
    offset_a = (segment_index * 8) % 65520
    offset_b = (segment_index * 8 + 4096) % 65520

    return [
        "    block",
        f"      local.get {a}",
        f"      i32.const {delta}",
        "      i32.add",
        f"      local.tee {a}",
        f"      i32.const {mask}",
        "      i32.and",
        "      if",
        f"        i32.const {offset_a}",
        "        i32.load",
        f"        local.get {b}",
        "        i32.add",
        f"        local.set {c}",
        f"        local.get {c}",
        f"        local.get {d}",
        "        i32.xor",
        f"        local.set {d}",
        f"        i32.const {offset_b}",
        f"        local.get {d}",
        "        i32.store",
        "      else",
        f"        i32.const {offset_b}",
        "        i32.load",
        f"        local.get {c}",
        "        i32.sub",
        f"        local.set {b}",
        f"        local.get {b}",
        f"        local.get {a}",
        "        i32.or",
        f"        local.set {a}",
        f"        i32.const {offset_a}",
        f"        local.get {a}",
        "        i32.store",
        "      end",
        "    end",
    ]


def distribute_segments(segment_count: int, function_count: int) -> list[int]:
    base = segment_count // function_count
    extra = segment_count % function_count
    return [base + (1 if index < extra else 0) for index in range(function_count)]


def append_function(
    lines: list[str],
    *,
    name: str | None,
    export_name: str | None,
    segment_start: int,
    segment_count: int,
    store_offset: int,
) -> None:
    locals_decl = " ".join(["i32"] * LOCAL_COUNT)
    header = "  (func"
    if name is not None:
        header += f" ${name}"
    if export_name is not None:
        header += f' (export "{export_name}")'
    lines.extend(
        [
            header,
            f"    (local {locals_decl})",
        ]
    )
    for index in range(segment_start, segment_start + segment_count):
        lines.extend(segment_lines(index))
    lines.extend(
        [
            f"    i32.const {store_offset}",
            "    local.get 0",
            "    i32.store",
            "  )",
        ]
    )


def build_module(segment_count: int, function_count: int = 1) -> str:
    if function_count < 1:
        raise ValueError("function_count must be at least 1")

    lines = [
        "(module",
        '  (memory (export "memory") 1)',
    ]

    if function_count == 1:
        append_function(
            lines,
            name=None,
            export_name="_start",
            segment_start=0,
            segment_count=segment_count,
            store_offset=0,
        )
    else:
        segment_start = 0
        for function_index, function_segments in enumerate(
            distribute_segments(segment_count, function_count)
        ):
            append_function(
                lines,
                name=f"f{function_index}",
                export_name=None,
                segment_start=segment_start,
                segment_count=function_segments,
                store_offset=function_index * 4,
            )
            segment_start += function_segments
        lines.append('  (func (export "_start")')
        for function_index in range(function_count):
            lines.append(f"    call $f{function_index}")
        lines.append("  )")

    lines.extend(
        [
            ")",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a single-function WAT stress fixture for memprof."
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=2900,
        help="Total number of repeated structured-control-flow segments to emit.",
    )
    parser.add_argument(
        "--functions",
        type=int,
        default=1,
        help="Number of functions to distribute the segments across.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("single_fn_200k.wat"),
        help="Output WAT file path.",
    )
    args = parser.parse_args()

    text = build_module(args.segments, args.functions)
    args.output.write_text(text)
    print(
        f"wrote {args.output} ({args.output.stat().st_size} bytes of WAT, "
        f"segments={args.segments}, functions={args.functions}, "
        f"target_wasm~{TARGET_WASM_BYTES} bytes)"
    )


if __name__ == "__main__":
    main()
