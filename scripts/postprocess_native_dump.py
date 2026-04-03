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
            "ensure_cache": text.count("local.ensure_cache"),
            "drop_cache": text.count("local.drop_cache"),
            "get_cache": text.count("local.get_cache"),
            "set_cache": text.count("local.set_cache"),
            "get_slot": text.count("local.get_slot"),
            "set_slot": text.count("local.set_slot"),
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


def available_native_disassembler() -> Optional[List[str]]:
    if shutil.which("gobjdump"):
        return ["gobjdump", "-D", "-b", "binary", "-m", "aarch64"]
    if shutil.which("objdump"):
        probe = subprocess.run(
            ["objdump", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        text = (probe.stdout or "") + (probe.stderr or "")
        if "-b" in text and "--architecture" in text:
            return ["objdump", "-D", "-b", "binary", "-m", "aarch64"]
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
            "elf64-littleaarch64",
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
) -> Optional[str]:
    llvm_objdump = find_llvm_tool("llvm-objdump")
    if llvm_objdump is None:
        return None
    stop = start + size
    proc = subprocess.run(
        [
            llvm_objdump,
            "-d",
            "--triple=aarch64",
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


def disassemble_native_bytes(code_bytes: bytes) -> str:
    if not code_bytes:
        return "unavailable: missing code bytes for function\n"
    base_cmd = available_native_disassembler()
    if base_cmd is None:
        return "unavailable: install gobjdump or a compatible objdump for raw AArch64 disassembly\n"

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
) -> str:
    if code_off is not None and code_size is not None and module_elf is not None:
        disasm = disassemble_native_range_from_elf(module_elf, code_off, code_size)
        if disasm:
            return disasm
    return disassemble_native_bytes(code_bytes)


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
        module_elf = create_native_code_elf(native_code_path, Path(tempdir))

        for index in indices:
            native = native_functions[index]
            wasm = wasm_details.get(index)
            code_off, code_size, code_bytes = native_code_slice(native_code_blob, native.meta)
            native_disasm = disassemble_native_function(
                code_off,
                code_size,
                code_bytes,
                module_elf,
            )
            summary = {
                "function_index": index,
                "wasm_name": wasm.name if wasm else None,
                "wasm_sig": wasm.sig if wasm else None,
                "frame_prefix_slots": parse_int_field(native.meta, "frame_prefix_slots"),
                "total_frame_slots": parse_int_field(native.meta, "total_frame_slots"),
                "code_file_off": f"0x{code_off:08x}" if code_off is not None else None,
                "code_size": code_size,
                "ssa_metrics": section_metrics(native.ssa_ir, "ssa"),
                "machine_metrics": section_metrics(native.machine_ir, "machine"),
            }
            rel_dir = Path("functions") / function_dir_name(index)
            function_map.append(
                {
                    "function_index": index,
                    "dir": str(rel_dir),
                    "wasm_name": wasm.name if wasm else None,
                    "code_size": summary["code_size"],
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
    write_json(out_dir / "module.json", module_summary)
    write_json(out_dir / "function_map.json", function_map)


if __name__ == "__main__":
    main()
