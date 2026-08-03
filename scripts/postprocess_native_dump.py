#!/usr/bin/env python3
"""Post-process sf-nano native dumps into per-function debug folders.

This script stitches together:
- Wasm disassembly from `wasm-objdump`
- native dump metadata
- SSA IR
- Machine IR

Output layout:

  <out-dir>/
    module.json
    function_map.json
    functions/
      0006/
        summary.json
        overview.txt
        wasm_disasm.txt
        wasm_text.wat
        ssa_ir.txt
        machine_ir.txt
        native_disasm.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


FUNC_HEADER_RE = re.compile(r"^\[function (\d+)\]\s*$")
WASM_FUNC_RE = re.compile(r"^[0-9a-f]+ func\[(\d+)\](?: <([^>]+)>)?:\s*$")
WASM_FUNC_DETAIL_RE = re.compile(r"^\s*-\s*func\[(\d+)\]\s+sig=(\d+)(?:\s+<([^>]+)>)?")

# Match real instructions in objdump output. `native_disasm.txt` is objdump-like:
# "   0: d503201f      nop". Keep this local to the native-dump postprocessor so
# policy comparisons can treat final machine instructions as first-class data,
# not a private helper inside compare_llvm.py.
_OBJDUMP_INSN_RE = re.compile(r"^\s+[0-9a-f]+:\s+[0-9a-f]+\s+\S+")
_EMPTY_RE = re.compile(r"^\s*$")
_ASM_COMMENT_RE = re.compile(r"^\s*(//|;|@)")


def count_native_instructions(text: str) -> int:
    count = 0
    for line in text.splitlines():
        if _EMPTY_RE.match(line) or _ASM_COMMENT_RE.match(line):
            continue
        if _OBJDUMP_INSN_RE.match(line):
            count += 1
    return count


@dataclass
class NativeFunctionDump:
    index: int
    meta: str
    ssa_ir: str
    machine_ir: str


@dataclass
class WasmFunctionInfo:
    index: int
    sig: Optional[int]
    name: Optional[str]
    disasm: str
    wat: str


def run_capture(argv: List[str]) -> str:
    result = subprocess.run(argv, check=True, capture_output=True, text=True)
    return result.stdout


def parse_native_index(text: str) -> Dict[int, NativeFunctionDump]:
    functions: Dict[int, NativeFunctionDump] = {}
    current_index: Optional[int] = None
    meta_lines: List[str] = []
    ssa_lines: List[str] = []
    machine_lines: List[str] = []
    section = "meta"

    def flush() -> None:
        nonlocal current_index, meta_lines, ssa_lines, machine_lines, section
        if current_index is None:
            return
        functions[current_index] = NativeFunctionDump(
            index=current_index,
            meta="\n".join(meta_lines).rstrip() + ("\n" if meta_lines else ""),
            ssa_ir="\n".join(ssa_lines).rstrip() + ("\n" if ssa_lines else ""),
            machine_ir="\n".join(machine_lines).rstrip() + ("\n" if machine_lines else ""),
        )
        current_index = None
        meta_lines = []
        ssa_lines = []
        machine_lines = []
        section = "meta"

    for line in text.splitlines():
        match = FUNC_HEADER_RE.match(line)
        if match:
            flush()
            current_index = int(match.group(1))
            continue
        if current_index is None:
            continue
        if line == "ssa_ir:":
            section = "ssa"
            continue
        if line == "machine_ir:":
            section = "machine"
            continue
        if section == "meta":
            meta_lines.append(line)
        elif section == "ssa":
            ssa_lines.append(line)
        else:
            machine_lines.append(line)

    flush()
    return functions


def parse_wasm_function_details(text: str) -> Dict[int, WasmFunctionInfo]:
    details: Dict[int, WasmFunctionInfo] = {}
    for line in text.splitlines():
        match = WASM_FUNC_DETAIL_RE.match(line)
        if not match:
            continue
        index = int(match.group(1))
        sig = int(match.group(2))
        name = match.group(3)
        details[index] = WasmFunctionInfo(
            index=index,
            sig=sig,
            name=name,
            disasm="",
            wat="",
        )
    return details


def parse_wasm_disassembly(text: str) -> Dict[int, str]:
    functions: Dict[int, str] = {}
    current_index: Optional[int] = None
    lines: List[str] = []

    def flush() -> None:
        nonlocal current_index, lines
        if current_index is None:
            return
        functions[current_index] = "\n".join(lines).rstrip() + ("\n" if lines else "")
        current_index = None
        lines = []

    for line in text.splitlines():
        match = WASM_FUNC_RE.match(line)
        if match:
            flush()
            current_index = int(match.group(1))
            lines.append(line)
            continue
        if current_index is not None:
            lines.append(line)

    flush()
    return functions


def parse_wasm_text(text: str) -> Dict[int, str]:
    functions: Dict[int, str] = {}
    current_index: Optional[int] = None
    depth = 0
    lines: List[str] = []

    def flush() -> None:
        nonlocal current_index, depth, lines
        if current_index is None:
            return
        functions[current_index] = "\n".join(lines).rstrip() + ("\n" if lines else "")
        current_index = None
        depth = 0
        lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.lstrip()
        func_match = re.match(r"\(func(?:\s+\$f(\d+))?", stripped)
        if current_index is None and func_match:
            idx = func_match.group(1)
            if idx is None:
                continue
            current_index = int(idx)
            lines = [line]
            depth = line.count("(") - line.count(")")
            if depth <= 0:
                flush()
            continue
        if current_index is not None:
            lines.append(line)
            depth += line.count("(") - line.count(")")
            if depth <= 0:
                flush()
    flush()
    return functions


def parse_int_field(meta: str, key: str) -> Optional[int]:
    match = re.search(rf"^{re.escape(key)}=(\d+)$", meta, re.MULTILINE)
    return int(match.group(1)) if match else None


def section_metrics(text: str, kind: str) -> Dict[str, int]:
    if kind == "ssa":
        return {
            "blocks": text.count("\n  block b"),
            "gotos": text.count("term: goto"),
            "branches": text.count("term: branch"),
            "jump_tables": text.count("term: br_table"),
            "ensure_cache": text.count("cell.ensure_cache"),
            "drop_cache": text.count("cell.drop_cache"),
            "get_cache": text.count("cell.get_cache"),
            "set_cache": text.count("cell.set_cache"),
            "get_slot": text.count("cell.get_slot"),
            "set_slot": text.count("cell.set_slot"),
            "spill": text.count("spill "),
            "fill": text.count("fill "),
        }
    return {
        "blocks": text.count("\n  block b"),
        "jumps": text.count("term: jump"),
        "branches": text.count("term: branch"),
        "jump_tables": text.count("term: jump_table"),
        "call_direct": text.count("term: call_direct") + text.count("call_direct f"),
        "load_gp": text.count("load.gp.u64"),
        "move_gp": text.count("move.gp"),
        "indexed_load_u32": text.count("indexed_load.u32.zx"),
        "indexed_store_u32": text.count("indexed_store.u32"),
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def function_dir_name(index: int) -> str:
    return f"{index:04d}"


def native_code_slice(
    code_blob: bytes, meta: str
) -> tuple[Optional[int], Optional[int], bytes]:
    off_match = re.search(r"^code_file_off=(0x[0-9a-fA-F]+)$", meta, re.MULTILINE)
    size_match = re.search(r"^code_size=(\d+)$", meta, re.MULTILINE)
    if not off_match or not size_match:
        return None, None, b""
    offset = int(off_match.group(1), 16)
    size = int(size_match.group(1))
    return offset, size, code_blob[offset : offset + size]


# Per-architecture settings for the objcopy/objdump pipeline. Add a new
# entry here to support another JIT target; the rest of the script is
# arch-agnostic.
ARCH_CONFIGS: Dict[str, Dict[str, str]] = {
    "aarch64": {
        "objdump_machine": "aarch64",
        "elf_format": "elf64-littleaarch64",
        "llvm_triple": "aarch64",
    },
    "thumbv7": {
        "objdump_machine": "arm",
        "elf_format": "elf32-littlearm",
        "llvm_triple": "thumbv7",
    },
    "x86_64": {
        "objdump_machine": "i386:x86-64",
        "elf_format": "elf64-x86-64",
        "llvm_triple": "x86_64",
    },
}


def available_native_disassembler(arch: str) -> Optional[List[str]]:
    machine = ARCH_CONFIGS[arch]["objdump_machine"]
    if shutil.which("gobjdump"):
        return ["gobjdump", "-D", "-b", "binary", "-m", machine]
    if shutil.which("objdump"):
        probe = subprocess.run(
            ["objdump", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        text = (probe.stdout or "") + (probe.stderr or "")
        if "-b" in text and "--architecture" in text:
            return ["objdump", "-D", "-b", "binary", "-m", machine]
    return None


def find_llvm_tool(tool: str) -> Optional[str]:
    candidates = [
        f"/opt/homebrew/opt/llvm/bin/{tool}",
        shutil.which(tool),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def create_native_code_elf(
    native_code_path: Path,
    workdir: Path,
    arch: str,
) -> Optional[Path]:
    llvm_objcopy = find_llvm_tool("llvm-objcopy")
    if llvm_objcopy is None:
        return None
    elf_path = workdir / "native_code.elf"
    proc = subprocess.run(
        [
            llvm_objcopy,
            "-I",
            "binary",
            "-O",
            ARCH_CONFIGS[arch]["elf_format"],
            "--rename-section",
            ".data=.text,alloc,load,readonly,code",
            str(native_code_path),
            str(elf_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not elf_path.exists():
        return None
    return elf_path


def disassemble_native_range_from_elf(
    elf_path: Path,
    start: int,
    size: int,
    arch: str,
) -> Optional[str]:
    llvm_objdump = find_llvm_tool("llvm-objdump")
    if llvm_objdump is None:
        return None
    stop = start + size
    proc = subprocess.run(
        [
            llvm_objdump,
            "-d",
            f"--triple={ARCH_CONFIGS[arch]['llvm_triple']}",
            f"--start-address=0x{start:x}",
            f"--stop-address=0x{stop:x}",
            str(elf_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def disassemble_native_bytes(code_bytes: bytes, arch: str) -> str:
    if not code_bytes:
        return "unavailable: missing code bytes for function\n"
    base_cmd = available_native_disassembler(arch)
    if base_cmd is None:
        return f"unavailable: install gobjdump or a compatible objdump for raw {arch} disassembly\n"

    with tempfile.NamedTemporaryFile(prefix="sf-native-", suffix=".bin") as tmp:
        tmp.write(code_bytes)
        tmp.flush()
        proc = subprocess.run(
            [*base_cmd, tmp.name],
            capture_output=True,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        if detail:
            return f"unavailable: native disassembly failed: {detail}\n"
        return "unavailable: native disassembly failed\n"
    return proc.stdout


def disassemble_native_function(
    code_off: Optional[int],
    code_size: Optional[int],
    code_bytes: bytes,
    module_elf: Optional[Path],
    arch: str,
) -> str:
    if code_off is not None and code_size is not None and module_elf is not None:
        disasm = disassemble_native_range_from_elf(module_elf, code_off, code_size, arch)
        if disasm:
            return disasm
    return disassemble_native_bytes(code_bytes, arch)


def build_overview(
    index: int,
    wasm: Optional[WasmFunctionInfo],
    native: NativeFunctionDump,
    summary: Dict[str, object],
    native_disasm: str,
) -> str:
    parts = [
        f"Function {index}",
        "",
        "Summary:",
        json.dumps(summary, indent=2, sort_keys=True),
        "",
        "=== Wasm Disassembly ===",
        (wasm.disasm if wasm else "<unavailable>\n").rstrip(),
        "",
        "=== Wasm Text ===",
        (wasm.wat if wasm else "<unavailable>\n").rstrip(),
        "",
        "=== Native Metadata ===",
        native.meta.rstrip() if native.meta else "<unavailable>",
        "",
        "=== SSA IR ===",
        native.ssa_ir.rstrip() if native.ssa_ir else "<unavailable>",
        "",
        "=== Machine IR ===",
        native.machine_ir.rstrip() if native.machine_ir else "<unavailable>",
        "",
        "=== Native Disassembly ===",
        native_disasm.rstrip() if native_disasm else "<unavailable>",
        "",
    ]
    return "\n".join(parts)


###############################################################################
# Pressure analysis
###############################################################################

# Old ARM64 register split (static banks, from commit 40cfc0e)
OLD_GP_LOCAL_CACHE = 13
OLD_GP_TRANSIENT = 9
OLD_FP_LOCAL_CACHE = 19
OLD_FP_TRANSIENT = 10

# New ARM64 unified budget
NEW_GP_DYNAMIC = 22
NEW_FP_DYNAMIC = 29

BLOCK_HEADER_RE = re.compile(
    r"^\s*block b(\d+)\s+params=\[([^\]]*)\](?:\s+cached=\[(.*)\])?\s*$"
)
LOCAL_TYPES_RE = re.compile(r"^\s*cell_types=\[([^\]]*)\]\s*$")
LOCAL_SLOTS_RE = re.compile(r"^\s*cells=(\d+)\s*$")
SSA_VALUE_DEF_RE = re.compile(r"v(\d+)")


@dataclass
class BlockPressure:
    block_id: int
    cached_count: int
    cached_gp: int
    cached_fp: int
    param_count: int
    ensure_count: int
    drop_count: int
    leaf_count: int
    get_cache: int
    set_cache: int
    get_slot: int
    set_slot: int
    peak_live: int  # peak simultaneously live SSA values during block


@dataclass
class FunctionPressure:
    index: int
    code_size: int
    local_count: int
    gp_locals: int
    fp_locals: int
    block_count: int
    blocks: List[BlockPressure]
    total_ensure: int
    total_drop: int


# Regexes for liveness analysis within a block.
_DEF_GET_CACHE_RE = re.compile(r"local\.get_cache (v\d+)")
_DEF_GET_SLOT_RE = re.compile(r"local\.get_slot (v\d+)")
_DEF_FILL_RE = re.compile(r"fill (v\d+)")
_DEF_RESULTS_RE = re.compile(r"results=\[([^\]]*)\]")
_USE_ARGS_RE = re.compile(r"args=\[([^\]]*)\]")
_USE_SET_CACHE_RE = re.compile(r"local\.set_cache fp\[\d+\] <- (v\d+)")
_USE_SET_SLOT_RE = re.compile(r"local\.set_slot fp\[\d+\] <- (v\d+)")
_USE_SPILL_RE = re.compile(r"spill fp\[\d+\] <- (v\d+)")
_TERM_VALS_RE = re.compile(r"v\d+")


def _compute_peak_live(params_str: str, body_lines: List[str]) -> int:
    """Compute peak simultaneously-live SSA values in a block.

    Walk instructions forward. Before each instruction, kill consumed values
    (last use), then add defined values. Track the maximum live set size.
    """
    # Collect all value defs and their last-use positions.
    defs: Dict[str, int] = {}  # value -> instruction index where defined
    last_use: Dict[str, int] = {}  # value -> last instruction index that uses it

    # Params are defined at position -1 (block entry).
    if params_str.strip():
        for p in params_str.split(","):
            p = p.strip()
            if p:
                defs[p] = -1

    for idx, line in enumerate(body_lines):
        line_s = line.strip()
        # Definitions: get_cache, get_slot, fill, leaf results
        for pattern in (_DEF_GET_CACHE_RE, _DEF_GET_SLOT_RE, _DEF_FILL_RE):
            m = pattern.search(line_s)
            if m:
                defs[m.group(1)] = idx

        m = _DEF_RESULTS_RE.search(line_s)
        if m and m.group(1).strip():
            for v in m.group(1).split(","):
                v = v.strip()
                if v:
                    defs[v] = idx

        # Uses: args, set_cache src, set_slot src, spill src, terminator values
        used_here: List[str] = []
        m = _USE_ARGS_RE.search(line_s)
        if m and m.group(1).strip():
            for tok in m.group(1).split(","):
                tok = tok.strip()
                if tok.startswith("v"):
                    used_here.append(tok)

        for pattern in (_USE_SET_CACHE_RE, _USE_SET_SLOT_RE, _USE_SPILL_RE):
            m = pattern.search(line_s)
            if m:
                used_here.append(m.group(1))

        if line_s.startswith("term:"):
            for v in _TERM_VALS_RE.findall(line_s):
                used_here.append(v)

        for v in used_here:
            last_use[v] = idx

    # Walk forward computing live set size.
    # A value is live from its def until its last use (inclusive).
    # At each instruction: live = values defined at or before this point
    # whose last use is at or after this point.
    # For efficiency, track add/remove events.
    if not defs:
        return 0

    max_inst = len(body_lines)
    # Build timeline: (position, +1 for def, -1 for last-use-end)
    events: List[tuple] = []
    for v, d in defs.items():
        lu = last_use.get(v, d)  # if never used, dies at def
        events.append((d, 0, +1, v))      # def: add to live
        events.append((lu, 1, -1, v))      # last use: remove after

    events.sort()

    live = 0
    peak = 0
    for _pos, phase, delta, _v in events:
        live += delta
        if live > peak:
            peak = live

    return peak


def parse_local_types(ssa_text: str) -> List[str]:
    """Extract local types from the SSA IR text."""
    for line in ssa_text.splitlines():
        m = LOCAL_TYPES_RE.match(line)
        if m and m.group(1).strip():
            return [t.strip() for t in m.group(1).split(",")]
    return []


def is_fp_type(ty: str) -> bool:
    return ty in ("f32", "f64")


def analyze_function_pressure(
    native: NativeFunctionDump,
) -> Optional[FunctionPressure]:
    ssa = native.ssa_ir
    if not ssa or ssa.strip() == "<unavailable>":
        return None

    # Parse local_slots
    local_count = 0
    for line in ssa.splitlines():
        m = LOCAL_SLOTS_RE.match(line)
        if m:
            local_count = int(m.group(1))
            break

    if local_count == 0:
        return None

    # Parse local types
    local_types = parse_local_types(ssa)
    gp_locals = sum(1 for t in local_types if not is_fp_type(t))
    fp_locals = sum(1 for t in local_types if is_fp_type(t))

    # Parse code_size from meta
    code_size = parse_int_field(native.meta, "code_size") or 0

    # Split into blocks and analyze each
    blocks: List[BlockPressure] = []
    current_block_id: Optional[int] = None
    current_cached: List[str] = []
    current_params: List[str] = []
    current_lines: List[str] = []

    def flush_block() -> None:
        nonlocal current_block_id, current_cached, current_params, current_lines
        if current_block_id is None:
            return
        body = "\n".join(current_lines)

        # Classify cached cells as GP/FP. Each token leads with c<cell> (the
        # cell id, indexing cell_types) and may carry a `?` (reserve,
        # write-first) marker; legacy dumps used fp[slot]@l<lane> forms, so the
        # old suffix is tolerated and ignored here.
        cached_gp = 0
        cached_fp = 0
        for slot_str in current_cached:
            m = re.match(r"c(\d+)\??$", slot_str.strip())
            if m:
                idx = int(m.group(1))
                if idx < len(local_types) and is_fp_type(local_types[idx]):
                    cached_fp += 1
                else:
                    cached_gp += 1

        params_str = ", ".join(current_params)
        peak_live = _compute_peak_live(params_str, current_lines)

        blocks.append(BlockPressure(
            block_id=current_block_id,
            cached_count=len(current_cached),
            cached_gp=cached_gp,
            cached_fp=cached_fp,
            param_count=len([p for p in current_params if p.strip()]),
            ensure_count=body.count("cell.ensure_cache"),
            drop_count=body.count("cell.drop_cache"),
            leaf_count=body.count("leaf "),
            get_cache=body.count("cell.get_cache"),
            set_cache=body.count("cell.set_cache"),
            get_slot=body.count("cell.get_slot"),
            set_slot=body.count("cell.set_slot"),
            peak_live=peak_live,
        ))
        current_block_id = None
        current_cached = []
        current_params = []
        current_lines = []

    for line in ssa.splitlines():
        m = BLOCK_HEADER_RE.match(line)
        if m:
            flush_block()
            current_block_id = int(m.group(1))
            current_params = [p for p in m.group(2).split(",") if p.strip()]
            cached_str = m.group(3)
            current_cached = (
                [c for c in cached_str.split(",") if c.strip()]
                if cached_str and cached_str.strip()
                else []
            )
            continue
        if current_block_id is not None:
            current_lines.append(line)

    flush_block()

    total_ensure = sum(b.ensure_count for b in blocks)
    total_drop = sum(b.drop_count for b in blocks)

    return FunctionPressure(
        index=native.index,
        code_size=code_size,
        local_count=local_count,
        gp_locals=gp_locals,
        fp_locals=fp_locals,
        block_count=len(blocks),
        blocks=blocks,
        total_ensure=total_ensure,
        total_drop=total_drop,
    )


def build_pressure_report(
    native_functions: Dict[int, NativeFunctionDump],
    indices: List[int],
) -> str:
    parts: List[str] = []
    parts.append("=" * 80)
    parts.append("CACHE PRESSURE ANALYSIS")
    parts.append("=" * 80)
    parts.append("")
    parts.append(
        f"Old ARM64 model: GP cache={OLD_GP_LOCAL_CACHE}, "
        f"GP transient={OLD_GP_TRANSIENT}, "
        f"FP cache={OLD_FP_LOCAL_CACHE}, FP transient={OLD_FP_TRANSIENT}"
    )
    parts.append(
        f"New ARM64 model: GP dynamic={NEW_GP_DYNAMIC}, "
        f"FP dynamic={NEW_FP_DYNAMIC}"
    )
    parts.append("")

    results: List[FunctionPressure] = []
    for idx in indices:
        native = native_functions.get(idx)
        if native is None:
            continue
        fp = analyze_function_pressure(native)
        if fp is not None:
            results.append(fp)

    if not results:
        parts.append("No functions with pressure data found.")
        return "\n".join(parts)

    # --- Module summary table ---
    parts.append("-" * 80)
    parts.append("PER-FUNCTION SUMMARY (sorted by code size)")
    parts.append("-" * 80)
    parts.append(
        f"{'func':>6} {'locals':>6} {'gp':>4} {'fp':>4} "
        f"{'blks':>5} {'code':>7} "
        f"{'ensure':>7} {'drop':>6} {'fix%':>5}  "
        f"{'c_avg':>5} {'c_max':>5}  "
        f"{'t_avg':>5} {'t_max':>5}  "
        f"{'pk_avg':>6} {'pk_max':>6}  "
        f"{'old_c':>5} {'old_t':>5}"
    )

    results.sort(key=lambda r: r.code_size, reverse=True)
    total_code = 0
    total_ensure = 0
    total_drop = 0
    total_blocks = 0

    for fp in results:
        total_code += fp.code_size
        total_ensure += fp.total_ensure
        total_drop += fp.total_drop
        total_blocks += fp.block_count

        cache_sizes = [b.cached_count for b in fp.blocks]
        live_peaks = [b.peak_live for b in fp.blocks]
        combined = [b.cached_count + b.peak_live for b in fp.blocks]

        cache_max = max(cache_sizes) if cache_sizes else 0
        cache_avg = sum(cache_sizes) / len(cache_sizes) if cache_sizes else 0
        live_max = max(live_peaks) if live_peaks else 0
        live_avg = sum(live_peaks) / len(live_peaks) if live_peaks else 0
        comb_max = max(combined) if combined else 0
        comb_avg = sum(combined) / len(combined) if combined else 0

        # Old model: how much of each bank would be used?
        # Cache: min(gp_locals, OLD_GP_LOCAL_CACHE)
        # Transient: peak stack values needed (live_max approximation)
        old_cache_used = min(fp.gp_locals, OLD_GP_LOCAL_CACHE)
        old_trans_need = live_max  # approximation

        # Old model pressure flags
        old_c_flag = "!" if fp.gp_locals > OLD_GP_LOCAL_CACHE else ""
        old_t_flag = "!" if old_trans_need > OLD_GP_TRANSIENT else ""

        # Boundary fixup percentage
        total_ops = sum(
            b.ensure_count + b.drop_count + b.leaf_count
            + b.get_cache + b.set_cache + b.get_slot + b.set_slot
            for b in fp.blocks
        )
        fix_pct = (
            (fp.total_ensure + fp.total_drop) / total_ops * 100
            if total_ops > 0
            else 0
        )

        parts.append(
            f"{fp.index:>6} {fp.local_count:>6} {fp.gp_locals:>4} {fp.fp_locals:>4} "
            f"{fp.block_count:>5} {fp.code_size:>7} "
            f"{fp.total_ensure:>7} {fp.total_drop:>6} {fix_pct:>4.0f}%  "
            f"{cache_avg:>5.1f} {cache_max:>5}  "
            f"{live_avg:>5.1f} {live_max:>5}  "
            f"{comb_avg:>6.1f} {comb_max:>6}  "
            f"{old_cache_used:>4}{old_c_flag} {old_trans_need:>4}{old_t_flag}"
        )

    parts.append("")
    parts.append(
        f"TOTALS: {len(results)} functions, {total_blocks} blocks, "
        f"code={total_code}, ensure={total_ensure}, drop={total_drop}"
    )

    # --- Combined pressure distribution ---
    parts.append("")
    parts.append("-" * 80)
    parts.append(
        f"COMBINED PRESSURE DISTRIBUTION  (cache + transient vs budget={NEW_GP_DYNAMIC})"
    )
    parts.append("-" * 80)

    from collections import Counter

    all_combined: List[int] = []
    all_cache_sizes: List[int] = []
    all_transient: List[int] = []
    for fp in results:
        for b in fp.blocks:
            all_combined.append(b.cached_count + b.peak_live)
            all_cache_sizes.append(b.cached_count)
            all_transient.append(b.peak_live)

    if all_combined:
        dist = Counter(all_combined)
        over_budget = sum(1 for c in all_combined if c > NEW_GP_DYNAMIC)
        parts.append(
            f"{'combined':>8} {'blocks':>8} {'pct':>6}   "
            f"(blocks over budget={NEW_GP_DYNAMIC}: {over_budget})"
        )
        for size in sorted(dist.keys()):
            pct = dist[size] / len(all_combined) * 100
            marker = " <-- budget" if size == NEW_GP_DYNAMIC else ""
            parts.append(f"{size:>8} {dist[size]:>8} {pct:>5.1f}%{marker}")
        parts.append(f"{'total':>8} {len(all_combined):>8}")

    # --- Separate cache and transient distributions ---
    parts.append("")
    parts.append("-" * 80)
    parts.append("CACHE vs TRANSIENT DISTRIBUTIONS")
    parts.append("-" * 80)

    if all_cache_sizes and all_transient:
        cache_dist = Counter(all_cache_sizes)
        trans_dist = Counter(all_transient)
        max_key = max(
            max(cache_dist.keys(), default=0), max(trans_dist.keys(), default=0)
        )
        parts.append(f"{'size':>6}  {'cache_blks':>10} {'cache%':>7}  {'trans_blks':>10} {'trans%':>7}")
        for size in range(max_key + 1):
            cb = cache_dist.get(size, 0)
            tb = trans_dist.get(size, 0)
            if cb == 0 and tb == 0:
                continue
            cp = cb / len(all_cache_sizes) * 100
            tp = tb / len(all_transient) * 100
            parts.append(f"{size:>6}  {cb:>10} {cp:>6.1f}%  {tb:>10} {tp:>6.1f}%")

    # --- Boundary-only block analysis ---
    parts.append("")
    parts.append("-" * 80)
    parts.append("BOUNDARY-ONLY BLOCKS (blocks with ensure/drop but no leaf ops)")
    parts.append("-" * 80)

    boundary_only = 0
    mixed = 0
    pure_logic = 0
    for fp in results:
        for b in fp.blocks:
            has_boundary = b.ensure_count > 0 or b.drop_count > 0
            has_logic = b.leaf_count > 0 or b.get_cache > 0 or b.set_cache > 0
            if has_boundary and not has_logic:
                boundary_only += 1
            elif has_boundary and has_logic:
                mixed += 1
            else:
                pure_logic += 1

    parts.append(f"Pure logic blocks:      {pure_logic:>6}")
    parts.append(f"Boundary-only blocks:   {boundary_only:>6}")
    parts.append(f"Mixed blocks:           {mixed:>6}")
    parts.append(f"Total:                  {pure_logic + boundary_only + mixed:>6}")

    # --- Cache stability analysis ---
    parts.append("")
    parts.append("-" * 80)
    parts.append("CACHE STABILITY (per function: how much does the cached set change?)")
    parts.append("-" * 80)
    parts.append(
        f"{'func':>6} {'locals':>6} {'blks':>5} "
        f"{'unique':>6} {'ratio':>6} "
        f"{'union':>6} {'isect':>6} {'empty':>6}"
    )

    for fp in sorted(results, key=lambda r: r.code_size, reverse=True):
        if not fp.blocks:
            continue
        native = native_functions[fp.index]
        block_sets: List[frozenset] = []
        for line in native.ssa_ir.splitlines():
            m = BLOCK_HEADER_RE.match(line)
            if m and m.group(3) and m.group(3).strip():
                slots = frozenset(
                    s.strip() for s in m.group(3).split(",") if s.strip()
                )
                block_sets.append(slots)
            elif m:
                block_sets.append(frozenset())

        if not block_sets:
            continue
        nonempty = [s for s in block_sets if s]
        empty_count = len(block_sets) - len(nonempty)
        unique_sets = len(set(block_sets))
        union_size = len(frozenset().union(*block_sets)) if block_sets else 0
        intersect_size = (
            len(frozenset.intersection(*nonempty))
            if nonempty
            else 0
        )
        set_ratio = unique_sets / len(block_sets) if block_sets else 0

        parts.append(
            f"{fp.index:>6} {fp.local_count:>6} {len(block_sets):>5} "
            f"{unique_sets:>6} {set_ratio:>5.2f}x "
            f"{union_size:>6} {intersect_size:>6} {empty_count:>6}"
        )

    parts.append("")
    parts.append("=" * 80)
    parts.append("LEGEND")
    parts.append("=" * 80)
    parts.append("locals      : total Wasm locals in the function")
    parts.append("gp/fp       : locals split by register class (integer vs float)")
    parts.append("blks        : SSA-IR block count")
    parts.append("code        : native code size in bytes")
    parts.append("ensure/drop : cache management ops (boundary fixup overhead)")
    parts.append("fix%        : ensure+drop as % of all SSA ops")
    parts.append("c_avg/c_max : cached-local count per block (average / max)")
    parts.append("t_avg/t_max : peak live transient SSA values per block (average / max)")
    parts.append("pk_avg/pk_max : combined (cache + transient) per block")
    parts.append(f"old_c       : locals old system would cache (min(gp, {OLD_GP_LOCAL_CACHE})),"
                 f" ! = exceeds budget")
    parts.append(f"old_t       : peak transient in old model, ! = exceeds {OLD_GP_TRANSIENT}")
    parts.append("unique      : distinct cached-slot sets across blocks")
    parts.append("ratio       : unique / block_count (1.0 = every block different)")
    parts.append("union       : distinct locals cached anywhere in the function")
    parts.append("isect       : locals cached in every non-empty block (stable core)")
    parts.append("empty       : blocks with no cached locals")
    parts.append("")

    return "\n".join(parts)


def selected_indices(
    available: Iterable[int], requested: Optional[List[int]]
) -> List[int]:
    ordered = sorted(set(available))
    if not requested:
        return ordered
    wanted = set(requested)
    return [index for index in ordered if index in wanted]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wasm", required=True, help="Path to the input wasm file")
    parser.add_argument(
        "--dump-dir",
        required=True,
        help="Directory containing native_index.txt/native_code.bin",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--function",
        type=int,
        action="append",
        dest="functions",
        help="Restrict output to specific function index; repeatable",
    )
    parser.add_argument(
        "--arch",
        choices=sorted(ARCH_CONFIGS.keys()),
        default="aarch64",
        help="JIT target architecture for the native dump (default: aarch64)",
    )
    args = parser.parse_args()

    wasm_path = Path(args.wasm).resolve()
    dump_dir = Path(args.dump_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    native_index_path = dump_dir / "native_index.txt"
    native_code_path = dump_dir / "native_code.bin"
    if not wasm_path.exists():
        raise SystemExit(f"missing wasm file: {wasm_path}")
    if not native_index_path.exists():
        raise SystemExit(f"missing native_index.txt: {native_index_path}")

    native_index_text = native_index_path.read_text()
    native_code_blob = native_code_path.read_bytes()
    native_functions = parse_native_index(native_index_text)

    wasm_details_text = run_capture(["wasm-objdump", "-x", str(wasm_path)])
    wasm_disasm_text = run_capture(["wasm-objdump", "-d", str(wasm_path)])
    wasm_wat_text = run_capture(["wasm2wat", "--generate-names", str(wasm_path)])
    wasm_details = parse_wasm_function_details(wasm_details_text)
    wasm_disasm = parse_wasm_disassembly(wasm_disasm_text)
    wasm_wat = parse_wasm_text(wasm_wat_text)

    for index, disasm in wasm_disasm.items():
        info = wasm_details.get(index)
        if info is None:
            wasm_details[index] = WasmFunctionInfo(
                index=index,
                sig=None,
                name=None,
                disasm=disasm,
                wat=wasm_wat.get(index, ""),
            )
        else:
            info.disasm = disasm
            info.wat = wasm_wat.get(index, "")

    indices = selected_indices(native_functions.keys(), args.functions)
    function_map = []
    with tempfile.TemporaryDirectory(prefix="sf-native-post-") as tempdir:
        module_elf = create_native_code_elf(native_code_path, Path(tempdir), args.arch)

        for index in indices:
            native = native_functions[index]
            wasm = wasm_details.get(index)
            code_off, code_size, code_bytes = native_code_slice(native_code_blob, native.meta)
            native_disasm = disassemble_native_function(
                code_off,
                code_size,
                code_bytes,
                module_elf,
                args.arch,
            )
            native_instruction_count = count_native_instructions(native_disasm)
            ssa_metrics = section_metrics(native.ssa_ir, "ssa")
            machine_metrics = section_metrics(native.machine_ir, "machine")
            ssa_ops = sum(ssa_metrics.values())
            machine_ops = sum(machine_metrics.values())
            summary = {
                "function_index": index,
                "wasm_name": wasm.name if wasm else None,
                "wasm_sig": wasm.sig if wasm else None,
                "frame_prefix_slots": parse_int_field(native.meta, "frame_prefix_slots"),
                "total_frame_slots": parse_int_field(native.meta, "total_frame_slots"),
                "code_file_off": f"0x{code_off:08x}" if code_off is not None else None,
                "code_size": code_size,
                "native_metrics": {
                    "instruction_count": native_instruction_count,
                    "code_size_bytes": code_size,
                    "instructions_per_machine_metric": (
                        native_instruction_count / machine_ops if machine_ops else None
                    ),
                    "instructions_per_ssa_metric": (
                        native_instruction_count / ssa_ops if ssa_ops else None
                    ),
                    "machine_metrics_per_ssa_metric": (
                        machine_ops / ssa_ops if ssa_ops else None
                    ),
                },
                "ssa_metrics": ssa_metrics,
                "machine_metrics": machine_metrics,
            }
            rel_dir = Path("functions") / function_dir_name(index)
            function_map.append(
                {
                    "function_index": index,
                    "dir": str(rel_dir),
                    "wasm_name": wasm.name if wasm else None,
                    "code_size": summary["code_size"],
                    "native_instruction_count": native_instruction_count,
                    "summary_path": str((out_dir / rel_dir / "summary.json").resolve()),
                    "native_disasm_path": str((out_dir / rel_dir / "native_disasm.txt").resolve()),
                    "ssa_ir_path": str((out_dir / rel_dir / "ssa_ir.txt").resolve()),
                    "machine_ir_path": str((out_dir / rel_dir / "machine_ir.txt").resolve()),
                }
            )

            function_dir = out_dir / rel_dir
            write_json(function_dir / "summary.json", summary)
            write_text(function_dir / "wasm_disasm.txt", wasm.disasm if wasm else "")
            write_text(function_dir / "wasm_text.wat", wasm.wat if wasm else "")
            write_text(function_dir / "ssa_ir.txt", native.ssa_ir)
            write_text(function_dir / "machine_ir.txt", native.machine_ir)
            write_text(function_dir / "native_disasm.txt", native_disasm)
            write_text(
                function_dir / "overview.txt",
                build_overview(
                    index,
                    wasm,
                    native,
                    summary,
                    native_disasm,
                ),
            )

    module_summary = {
        "wasm": str(wasm_path),
        "dump_dir": str(dump_dir),
        "native_index": str(native_index_path),
        "function_count": len(indices),
    }
    totals = {
        "ssa_metrics": {},
        "machine_metrics": {},
    }
    for item in function_map:
        try:
            summary = json.loads(Path(item["summary_path"]).read_text())
        except (OSError, KeyError, json.JSONDecodeError):
            continue
        for group in ["ssa_metrics", "machine_metrics"]:
            for key, value in summary.get(group, {}).items():
                totals[group][key] = totals[group].get(key, 0) + int(value or 0)

    module_metrics = {
        "wasm": str(wasm_path),
        "dump_dir": str(dump_dir),
        "function_count": len(indices),
        "native_instruction_count": sum(
            item.get("native_instruction_count") or 0 for item in function_map
        ),
        "code_size_bytes": sum(item.get("code_size") or 0 for item in function_map),
        "ssa_metrics": totals["ssa_metrics"],
        "machine_metrics": totals["machine_metrics"],
        "functions": function_map,
    }
    write_json(out_dir / "module.json", module_summary)
    write_json(out_dir / "module_metrics.json", module_metrics)
    write_json(out_dir / "function_map.json", function_map)

    # Pressure analysis report
    pressure_report = build_pressure_report(native_functions, indices)
    write_text(out_dir / "pressure_report.txt", pressure_report)


if __name__ == "__main__":
    main()
