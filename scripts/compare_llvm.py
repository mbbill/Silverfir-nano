#!/usr/bin/env python3
"""Compare Silverfir native codegen against wasmer LLVM backend output.

Given:
- Silverfir postprocessed output (from postprocess_native_dump.py)
- Wasmer LLVM debug directory (from --compiler-debug-dir)

Produces a summary table of instruction-count ratios (SF / LLVM) sorted
by worst ratio, and optionally a side-by-side assembly comparison for a
single function.

Usage:

  # Summary table (sorted by ratio, worst first)
  python3 scripts/compare_llvm.py \
    --sf-dir /tmp/coremark-sf-pp \
    --llvm-dir /tmp/coremark-llvm-debug

  # Single-function side-by-side
  python3 scripts/compare_llvm.py \
    --sf-dir /tmp/coremark-sf-pp \
    --llvm-dir /tmp/coremark-llvm-debug \
    --function 6

  # Weighted by samply profile hotness
  python3 scripts/compare_llvm.py \
    --sf-dir /tmp/coremark-sf-pp \
    --llvm-dir /tmp/coremark-llvm-debug \
    --profile /tmp/coremark-native-profile.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def load_function_map(sf_dir: Path) -> List[dict]:
    fmap_path = sf_dir / "function_map.json"
    if not fmap_path.exists():
        sys.exit(f"error: {fmap_path} not found")
    with open(fmap_path) as f:
        return json.load(f)


def detect_num_imports(fmap: List[dict]) -> int:
    """Detect number of wasm imports by finding the smallest function_index
    that has a non-zero code_size (i.e. the first defined function).

    Note: wasmer's LLVM debug output uses global wasm indices directly
    (function_5.s for wasm func 5), so no offset is needed by default.
    This value is only used when --num-imports is explicitly set to a
    non-zero value, forcing module-local indexing."""
    defined = [
        e["function_index"]
        for e in fmap
        if e.get("code_size", 0) > 0
    ]
    if not defined:
        return 0
    return min(defined)


def find_llvm_subdir(llvm_dir: Path) -> Optional[Path]:
    """Find the llvm/<hash>/ subdirectory."""
    llvm_sub = llvm_dir / "llvm"
    if not llvm_sub.exists():
        return None
    subdirs = [d for d in llvm_sub.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    if subdirs:
        # Pick the most recently modified
        return max(subdirs, key=lambda d: d.stat().st_mtime)
    return None


def read_file_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text()


# ---------------------------------------------------------------------------
# Instruction counting
# ---------------------------------------------------------------------------

# Match real AArch64 instructions in both objdump and .s formats.
# objdump format: "   0: d503201f      nop"
# .s format:      "	ldr	x0, [x1]" or "	mov	w0, #42"
_OBJDUMP_INSN_RE = re.compile(r"^\s+[0-9a-f]+:\s+[0-9a-f]+\s+\S+")
_ASM_INSN_RE = re.compile(r"^\s+[a-z]\w*")
_ASM_DIRECTIVE_RE = re.compile(r"^\s*\.")
_ASM_LABEL_RE = re.compile(r"^\S+:")
_ASM_COMMENT_RE = re.compile(r"^\s*(//|;|@)")
_EMPTY_RE = re.compile(r"^\s*$")


def count_instructions(text: str) -> int:
    """Count real instructions in assembly text, skipping labels, directives,
    comments, and blank lines."""
    count = 0
    for line in text.splitlines():
        # Skip blank/comment
        if _EMPTY_RE.match(line) or _ASM_COMMENT_RE.match(line):
            continue
        # objdump format (SF native_disasm.txt)
        if _OBJDUMP_INSN_RE.match(line):
            count += 1
            continue
        # .s format (LLVM assembly)
        if _ASM_DIRECTIVE_RE.match(line):
            continue
        if _ASM_LABEL_RE.match(line):
            continue
        if _ASM_INSN_RE.match(line):
            count += 1
    return count


def extract_instruction_lines(text: str) -> List[str]:
    """Extract just the instruction lines from assembly text."""
    result = []
    for line in text.splitlines():
        if _EMPTY_RE.match(line) or _ASM_COMMENT_RE.match(line):
            continue
        if _OBJDUMP_INSN_RE.match(line):
            result.append(line.rstrip())
            continue
        if _ASM_DIRECTIVE_RE.match(line):
            continue
        if _ASM_LABEL_RE.match(line):
            result.append(line.rstrip())
            continue
        if _ASM_INSN_RE.match(line):
            result.append(line.rstrip())
    return result


# ---------------------------------------------------------------------------
# Profile parsing
# ---------------------------------------------------------------------------

@dataclass
class HotspotEntry:
    symbol: str
    self_pct: float
    total_pct: float


def parse_hotspots_from_profile(profile_path: Path, limit: int = 100) -> List[HotspotEntry]:
    """Run samply-for-ai query to get hotspots, return parsed entries."""
    try:
        result = subprocess.run(
            [
                "samply-for-ai", "query",
                "--profile", str(profile_path),
                "hotspots", "--limit", str(limit),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        print("warning: samply-for-ai not found, skipping profile weighting",
              file=sys.stderr)
        return []
    except subprocess.CalledProcessError as e:
        print(f"warning: samply-for-ai failed: {e.stderr}", file=sys.stderr)
        return []

    entries = []
    # samply-for-ai outputs JSON
    try:
        data = json.loads(result.stdout)
        for item in data.get("data", []):
            func = item.get("function", {})
            entries.append(HotspotEntry(
                symbol=func.get("name", ""),
                self_pct=item.get("self_percent", 0.0),
                total_pct=item.get("total_percent", 0.0),
            ))
    except json.JSONDecodeError:
        # Fallback: try text format "  12.3%   15.4%  symbol"
        pct_re = re.compile(r"^\s*([\d.]+)%\s+([\d.]+)%\s+(.+)$")
        for line in result.stdout.splitlines():
            m = pct_re.match(line)
            if m:
                entries.append(HotspotEntry(
                    symbol=m.group(3).strip(),
                    self_pct=float(m.group(1)),
                    total_pct=float(m.group(2)),
                ))
    return entries


def aggregate_hotness_by_function(
    hotspots: List[HotspotEntry],
) -> Dict[int, float]:
    """Map wasm function index -> total self% from JIT symbols."""
    func_re = re.compile(r"jit::.*::func(\d+)::")
    result: Dict[int, float] = {}
    for entry in hotspots:
        m = func_re.search(entry.symbol)
        if m:
            idx = int(m.group(1))
            result[idx] = result.get(idx, 0.0) + entry.self_pct
    return result


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class FunctionComparison:
    wasm_index: int
    wasm_name: Optional[str]
    sf_insn_count: int
    llvm_insn_count: int
    sf_code_size: int
    ratio: float
    hotness: float  # self% from profile, 0 if no profile


def compare_functions(
    sf_dir: Path,
    llvm_hash_dir: Path,
    fmap: List[dict],
    num_imports: int,
    hotness_map: Dict[int, float],
    target_function: Optional[int] = None,
) -> List[FunctionComparison]:
    results = []

    for entry in fmap:
        wasm_idx = entry["function_index"]
        code_size = entry.get("code_size", 0)

        if code_size == 0:
            continue  # skip imports / empty functions

        if target_function is not None and wasm_idx != target_function:
            continue

        func_dir = sf_dir / entry["dir"]
        sf_asm_path = func_dir / "native_disasm.txt"
        sf_asm = read_file_text(sf_asm_path)
        if sf_asm is None or "unavailable" in sf_asm[:50]:
            continue

        # Wasmer LLVM uses global wasm indices by default (function_5.s etc).
        # If --num-imports is explicitly set, assume module-local indexing.
        llvm_idx = wasm_idx if num_imports == 0 else wasm_idx - num_imports
        llvm_asm_path = llvm_hash_dir / f"function_{llvm_idx}.s"
        llvm_asm = read_file_text(llvm_asm_path)
        if llvm_asm is None:
            continue

        sf_count = count_instructions(sf_asm)
        llvm_count = count_instructions(llvm_asm)

        ratio = sf_count / llvm_count if llvm_count > 0 else float("inf")
        hotness = hotness_map.get(wasm_idx, 0.0)

        results.append(FunctionComparison(
            wasm_index=wasm_idx,
            wasm_name=entry.get("wasm_name"),
            sf_insn_count=sf_count,
            llvm_insn_count=llvm_count,
            sf_code_size=code_size,
            ratio=ratio,
            hotness=hotness,
        ))

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary_table(
    comparisons: List[FunctionComparison],
    sort_by_hotness: bool,
) -> None:
    if not comparisons:
        print("No functions matched for comparison.")
        return

    if sort_by_hotness:
        # Sort by hotness-weighted ratio: ratio * hotness, descending
        comparisons.sort(
            key=lambda c: c.ratio * max(c.hotness, 0.01),
            reverse=True,
        )
    else:
        comparisons.sort(key=lambda c: c.ratio, reverse=True)

    total_sf = sum(c.sf_insn_count for c in comparisons)
    total_llvm = sum(c.llvm_insn_count for c in comparisons)
    overall_ratio = total_sf / total_llvm if total_llvm > 0 else float("inf")

    print(f"{'func':>6} {'name':<30} {'SF':>7} {'LLVM':>7} {'ratio':>7} "
          f"{'code_sz':>8} {'hot%':>6}")
    print("-" * 80)

    for c in comparisons:
        name = (c.wasm_name or "")[:30]
        hot_str = f"{c.hotness:.1f}" if c.hotness > 0 else "-"
        print(f"{c.wasm_index:>6} {name:<30} {c.sf_insn_count:>7} "
              f"{c.llvm_insn_count:>7} {c.ratio:>7.2f} "
              f"{c.sf_code_size:>8} {hot_str:>6}")

    print("-" * 80)
    print(f"{'TOTAL':>6} {'':<30} {total_sf:>7} {total_llvm:>7} "
          f"{overall_ratio:>7.2f}")
    print()

    # Highlight top optimization candidates
    candidates = [c for c in comparisons if c.ratio > 1.5]
    if candidates:
        print(f"Optimization candidates (ratio > 1.5): {len(candidates)} functions")
        extra_insns = sum(
            c.sf_insn_count - c.llvm_insn_count
            for c in candidates
        )
        print(f"Total extra instructions vs LLVM: {extra_insns}")


def print_side_by_side(
    sf_dir: Path,
    llvm_hash_dir: Path,
    fmap: List[dict],
    num_imports: int,
    wasm_idx: int,
) -> None:
    entry = None
    for e in fmap:
        if e["function_index"] == wasm_idx:
            entry = e
            break
    if entry is None:
        print(f"error: function {wasm_idx} not found in function_map.json",
              file=sys.stderr)
        return

    func_dir = sf_dir / entry["dir"]

    # Print wasm source
    wasm_disasm = read_file_text(func_dir / "wasm_disasm.txt")
    if wasm_disasm:
        print("=" * 80)
        print(f"WASM DISASSEMBLY (function {wasm_idx})")
        print("=" * 80)
        print(wasm_disasm)

    # Print SSA IR
    ssa_ir = read_file_text(func_dir / "ssa_ir.txt")
    if ssa_ir:
        print("=" * 80)
        print(f"SSA IR (function {wasm_idx})")
        print("=" * 80)
        print(ssa_ir)

    # Print Machine IR
    machine_ir = read_file_text(func_dir / "machine_ir.txt")
    if machine_ir:
        print("=" * 80)
        print(f"MACHINE IR (function {wasm_idx})")
        print("=" * 80)
        print(machine_ir)

    # Print SF assembly
    sf_asm = read_file_text(func_dir / "native_disasm.txt")
    llvm_idx = wasm_idx if num_imports == 0 else wasm_idx - num_imports
    llvm_asm = read_file_text(llvm_hash_dir / f"function_{llvm_idx}.s")

    sf_count = count_instructions(sf_asm) if sf_asm else 0
    llvm_count = count_instructions(llvm_asm) if llvm_asm else 0

    print("=" * 80)
    print(f"SILVERFIR ASSEMBLY  ({sf_count} instructions)")
    print("=" * 80)
    if sf_asm:
        print(sf_asm)
    else:
        print("<unavailable>")

    print("=" * 80)
    print(f"LLVM ASSEMBLY  ({llvm_count} instructions)")
    print("=" * 80)
    if llvm_asm:
        print(llvm_asm)
    else:
        print("<unavailable>")

    # Print LLVM optimized IR if available
    llvm_postopt = read_file_text(llvm_hash_dir / f"function_{llvm_idx}.postopt.ll")
    if llvm_postopt:
        print("=" * 80)
        print(f"LLVM OPTIMIZED IR (function_{llvm_idx}.postopt.ll)")
        print("=" * 80)
        print(llvm_postopt)

    if sf_count > 0 and llvm_count > 0:
        ratio = sf_count / llvm_count
        print("=" * 80)
        print(f"SUMMARY: SF={sf_count} insns, LLVM={llvm_count} insns, "
              f"ratio={ratio:.2f}x, delta=+{sf_count - llvm_count} insns")
        print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Silverfir native codegen vs wasmer LLVM backend",
    )
    parser.add_argument(
        "--sf-dir", required=True, type=Path,
        help="Silverfir postprocessed output directory",
    )
    parser.add_argument(
        "--llvm-dir", required=True, type=Path,
        help="Wasmer LLVM debug directory (--compiler-debug-dir output)",
    )
    parser.add_argument(
        "--num-imports", type=int, default=None,
        help="Number of wasm imports (auto-detected from function_map.json)",
    )
    parser.add_argument(
        "--function", type=int, default=None,
        help="Show side-by-side for one function (global wasm index)",
    )
    parser.add_argument(
        "--profile", type=Path, default=None,
        help="samply-for-ai profile JSON for hotness weighting",
    )
    args = parser.parse_args()

    fmap = load_function_map(args.sf_dir)

    num_imports = args.num_imports
    if num_imports is None:
        # Wasmer LLVM uses global wasm indices (function_5.s = wasm func 5),
        # so default to 0 (no offset). Use --num-imports N to force
        # module-local indexing if the LLVM tool uses a different convention.
        num_imports = 0

    llvm_hash_dir = find_llvm_subdir(args.llvm_dir)
    if llvm_hash_dir is None:
        sys.exit(f"error: no llvm/<hash>/ subdirectory found under {args.llvm_dir}")

    # Load hotness data
    hotness_map: Dict[int, float] = {}
    if args.profile:
        hotspots = parse_hotspots_from_profile(args.profile)
        hotness_map = aggregate_hotness_by_function(hotspots)

    if args.function is not None:
        print_side_by_side(
            args.sf_dir, llvm_hash_dir, fmap, num_imports, args.function,
        )
    else:
        comparisons = compare_functions(
            args.sf_dir, llvm_hash_dir, fmap, num_imports, hotness_map,
        )
        print_summary_table(comparisons, sort_by_hotness=bool(hotness_map))


if __name__ == "__main__":
    main()
