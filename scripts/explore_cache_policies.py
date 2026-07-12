#!/usr/bin/env python3
"""Explore cache-residency policies by compiling a corpus and ranking generated code.

This is an experiment harness, not a benchmark runner. It compares policies by
postprocessed generated-code metrics (especially final native instruction count
from native_disasm.txt), preserving artifacts for cross-layer investigation.

Typical use:

  python3 scripts/explore_cache_policies.py \
    --out-dir target/algo4-experiments/edge-sweep \
    --mode change-and-check \
    --title "ALGORITHM4 edge-cost sweep" \
    --hypothesis "Lower edge cost may improve final native code." \
    --baseline-policy algorithm4 \
    --wasi-fast \
    --policy algorithm4 \
    --policy algorithm4:edge=0.75 \
    --policy algorithm4:edge=0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
WASI_DIR = REPO_ROOT / "benchmarks" / "wasi"
DEFAULT_CLI = REPO_ROOT / "target" / "debug" / "sf-nano-cli"
PRIMARY_WEIGHT = 10.0


def policy_id(spec: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.strip())[:48].strip("_")
    digest = hashlib.sha1(spec.encode("utf-8")).hexdigest()[:8]
    return f"{safe or 'policy'}-{digest}"


def module_id(wasm: Path) -> str:
    try:
        rel = wasm.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        rel = wasm
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(rel))[:80].strip("_")
    digest = hashlib.sha1(str(wasm.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{safe or 'module'}-{digest}"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(args: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state() -> Dict[str, Any]:
    return {
        "branch": git_output(["branch", "--show-current"]),
        "head": git_output(["rev-parse", "HEAD"]),
        "dirty": bool(git_output(["status", "--short"])),
    }


def load_wasi_fast_modules() -> List[Path]:
    sys.path.insert(0, str(WASI_DIR))
    import run_tests  # type: ignore

    modules: List[Path] = []
    seen = set()
    for test in run_tests.TESTS:
        args = test.get("fast", {}).get("args", test.get("args", []))
        if not args:
            continue
        wasm = Path(test["cwd"]) / args[0]
        key = str(wasm.resolve())
        if key not in seen:
            seen.add(key)
            modules.append(wasm)
    return modules


def run(argv: List[str], *, env: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def compile_and_postprocess(cli: Path, wasm: Path, policy: str, out_dir: Path, arch: str) -> Dict[str, Any]:
    pid = policy_id(policy)
    mid = module_id(wasm)
    root = out_dir / "artifacts" / pid / mid
    dump_dir = root / "native-dump"
    post_dir = root / "post"
    shutil.rmtree(root, ignore_errors=True)
    dump_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SF_CACHE_POLICY"] = policy
    env["SF_NATIVE_DUMP_DIR"] = str(dump_dir)

    start = time.perf_counter()
    proc = run([str(cli), "--compile-only", str(wasm)], env=env, cwd=wasm.parent)
    elapsed = time.perf_counter() - start

    result: Dict[str, Any] = {
        "policy": policy,
        "policy_id": pid,
        "wasm": str(wasm),
        "wasm_sha256": file_sha256(wasm),
        "module_id": mid,
        "status": "failed" if proc.returncode else "ok",
        "returncode": proc.returncode,
        "compile_seconds": elapsed,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "artifact_dir": str(root),
    }
    if proc.returncode != 0:
        return result

    post = run([
        sys.executable,
        str(REPO_ROOT / "scripts" / "postprocess_native_dump.py"),
        "--wasm", str(wasm),
        "--dump-dir", str(dump_dir),
        "--out-dir", str(post_dir),
        "--arch", arch,
    ])
    result["postprocess_returncode"] = post.returncode
    result["postprocess_stdout"] = post.stdout[-4000:]
    result["postprocess_stderr"] = post.stderr[-4000:]
    if post.returncode != 0:
        result["status"] = "postprocess_failed"
        return result

    metrics_path = post_dir / "module_metrics.json"
    if not metrics_path.exists():
        result["status"] = "missing_metrics"
        return result
    metrics = load_json(metrics_path)
    result["metrics"] = metrics
    result["native_instruction_count"] = metrics.get("native_instruction_count", 0)
    result["code_size_bytes"] = metrics.get("code_size_bytes", 0)
    result["function_count"] = metrics.get("function_count", 0)
    result["ssa_metrics"] = metrics.get("ssa_metrics", {})
    result["machine_metrics"] = metrics.get("machine_metrics", {})
    return result


def metric_delta(value: float, base: float) -> float:
    return ((value - base) / base * 100.0) if base else 0.0


def summarize_mid_metrics(result: Dict[str, Any]) -> Dict[str, int]:
    ssa = result.get("ssa_metrics", {}) or {}
    machine = result.get("machine_metrics", {}) or {}
    return {
        "ensure_cache": int(ssa.get("ensure_cache", 0) or 0),
        "drop_cache": int(ssa.get("drop_cache", 0) or 0),
        "get_cache": int(ssa.get("get_cache", 0) or 0),
        "set_cache": int(ssa.get("set_cache", 0) or 0),
        "get_slot": int(ssa.get("get_slot", 0) or 0),
        "set_slot": int(ssa.get("set_slot", 0) or 0),
        "spill": int(ssa.get("spill", 0) or 0),
        "fill": int(ssa.get("fill", 0) or 0),
        "ssa_blocks": int(ssa.get("blocks", 0) or 0),
        "machine_blocks": int(machine.get("blocks", 0) or 0),
        "move_gp": int(machine.get("move_gp", 0) or 0),
        "load_gp": int(machine.get("load_gp", 0) or 0),
    }


def rank_results(results: List[Dict[str, Any]], baseline_policy: str) -> Dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]
    by_module: Dict[str, List[Dict[str, Any]]] = {}
    for r in ok:
        by_module.setdefault(r["module_id"], []).append(r)

    module_rankings = []
    aggregate: Dict[str, Dict[str, float]] = {}
    misleading_middle_wins: List[Dict[str, Any]] = []

    for mid, rows in by_module.items():
        best_insn = min(r.get("native_instruction_count", 10**18) for r in rows)
        best_code = min(r.get("code_size_bytes", 10**18) for r in rows)
        baseline = next((r for r in rows if r["policy"] == baseline_policy), rows[0])
        base_insn = baseline.get("native_instruction_count", 0) or 0
        base_code = baseline.get("code_size_bytes", 0) or 0
        base_mid = summarize_mid_metrics(baseline)
        module_rows = []

        for r in rows:
            mid_metrics = summarize_mid_metrics(r)
            insn = r.get("native_instruction_count", 0) or 0
            code = r.get("code_size_bytes", 0) or 0
            insn_best_delta = metric_delta(insn, best_insn)
            code_best_delta = metric_delta(code, best_code)
            insn_base_delta = metric_delta(insn, base_insn)
            code_base_delta = metric_delta(code, base_code)
            score = insn_best_delta * PRIMARY_WEIGHT + code_best_delta
            pid = r["policy_id"]
            agg = aggregate.setdefault(pid, {
                "score": 0.0, "modules": 0, "insn_best": 0.0, "code_best": 0.0,
                "insn_base": 0.0, "code_base": 0.0, "wins_vs_baseline": 0,
                "losses_vs_baseline": 0, "ties_vs_baseline": 0,
            })
            agg["score"] += score
            agg["modules"] += 1
            agg["insn_best"] += insn_best_delta
            agg["code_best"] += code_best_delta
            agg["insn_base"] += insn_base_delta
            agg["code_base"] += code_base_delta
            if insn < base_insn:
                agg["wins_vs_baseline"] += 1
            elif insn > base_insn:
                agg["losses_vs_baseline"] += 1
            else:
                agg["ties_vs_baseline"] += 1

            mid_improved = (
                mid_metrics["ensure_cache"] + mid_metrics["drop_cache"] + mid_metrics["get_slot"] + mid_metrics["set_slot"]
                < base_mid["ensure_cache"] + base_mid["drop_cache"] + base_mid["get_slot"] + base_mid["set_slot"]
            )
            native_regressed = insn > base_insn
            if r["policy"] != baseline_policy and mid_improved and native_regressed:
                misleading_middle_wins.append({
                    "module_id": mid,
                    "wasm": r["wasm"],
                    "policy": r["policy"],
                    "baseline_policy": baseline_policy,
                    "native_instruction_delta_pct_vs_baseline": insn_base_delta,
                    "artifact_dir": r["artifact_dir"],
                })

            module_rows.append({
                "policy": r["policy"],
                "policy_id": pid,
                "native_instruction_count": insn,
                "code_size_bytes": code,
                "instruction_delta_pct_vs_best": insn_best_delta,
                "code_size_delta_pct_vs_best": code_best_delta,
                "instruction_delta_pct_vs_baseline": insn_base_delta,
                "code_size_delta_pct_vs_baseline": code_base_delta,
                "score": score,
                "middle_metrics": mid_metrics,
                "misleading_middle_win": mid_improved and native_regressed,
                "artifact_dir": r["artifact_dir"],
            })
        module_rows.sort(key=lambda x: (x["score"], x["native_instruction_count"], x["code_size_bytes"]))
        module_rankings.append({"module_id": mid, "wasm": rows[0]["wasm"], "baseline_policy": baseline_policy, "policies": module_rows})

    ranking = []
    for pid, agg in aggregate.items():
        modules = max(1, int(agg["modules"]))
        ranking.append({
            "policy_id": pid,
            "avg_score": agg["score"] / modules,
            "avg_instruction_delta_pct_vs_best": agg["insn_best"] / modules,
            "avg_code_size_delta_pct_vs_best": agg["code_best"] / modules,
            "avg_instruction_delta_pct_vs_baseline": agg["insn_base"] / modules,
            "avg_code_size_delta_pct_vs_baseline": agg["code_base"] / modules,
            "wins_vs_baseline": int(agg["wins_vs_baseline"]),
            "losses_vs_baseline": int(agg["losses_vs_baseline"]),
            "ties_vs_baseline": int(agg["ties_vs_baseline"]),
            "modules": modules,
        })
    ranking.sort(key=lambda x: x["avg_score"])
    return {"policies": ranking, "modules": module_rankings, "misleading_middle_wins": misleading_middle_wins}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cli", default=str(DEFAULT_CLI))
    parser.add_argument("--arch", default="aarch64")
    parser.add_argument("--policy", action="append", required=True)
    parser.add_argument("--baseline-policy", default="algorithm4")
    parser.add_argument("--mode", choices=["change-and-check", "remove-and-see", "ablation"], default="change-and-check")
    parser.add_argument("--title", default="ALGORITHM4 policy exploration")
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--experiment-id")
    parser.add_argument("--wasm", action="append", type=Path, dest="wasms")
    parser.add_argument("--wasi-fast", action="store_true", help="Use unique modules from benchmarks/wasi/run_tests.py fast corpus")
    args = parser.parse_args()

    wasms: List[Path] = []
    if args.wasi_fast:
        wasms.extend(load_wasi_fast_modules())
    if args.wasms:
        wasms.extend(args.wasms)
    if not wasms:
        raise SystemExit("provide --wasi-fast or one or more --wasm paths")
    if args.baseline_policy not in args.policy:
        args.policy.insert(0, args.baseline_policy)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cli = Path(args.cli).resolve()
    exp_id = args.experiment_id or f"algo4-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{hashlib.sha1((args.title + ''.join(args.policy)).encode()).hexdigest()[:8]}"
    wasm_entries = [{"path": str(w.resolve()), "sha256": file_sha256(w.resolve())} for w in wasms]

    manifest = {
        "schema_version": 1,
        "experiment_id": exp_id,
        "title": args.title,
        "mode": args.mode,
        "hypothesis": args.hypothesis,
        "git": git_state(),
        "repo": str(REPO_ROOT),
        "cli": str(cli),
        "arch": args.arch,
        "baseline_policy": args.baseline_policy,
        "policies": args.policy,
        "corpus": {"name": "wasi-fast" if args.wasi_fast else "explicit", "modules": wasm_entries},
        "primary_metric": "native_instruction_count",
        "guardrail_metrics": ["code_size_bytes", "ensure_cache", "drop_cache", "spill", "fill", "boundary_fixup_pct"],
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "policies.json", [{"policy": p, "policy_id": policy_id(p)} for p in args.policy])

    results = []
    for policy in args.policy:
        for wasm in wasms:
            print(f"[algo4-explore] policy={policy} wasm={wasm}", flush=True)
            results.append(compile_and_postprocess(cli, wasm.resolve(), policy, out_dir, args.arch))
    write_json(out_dir / "results.json", results)
    ranking = rank_results(results, args.baseline_policy)
    write_json(out_dir / "ranking.json", ranking)

    lines = [args.title, f"mode={args.mode}", f"baseline={args.baseline_policy}", ""]
    for idx, row in enumerate(ranking["policies"], 1):
        lines.append(
            f"{idx:2d}. {row['policy_id']} score={row['avg_score']:.3f} "
            f"insn_vs_best={row['avg_instruction_delta_pct_vs_best']:.3f}% "
            f"insn_vs_base={row['avg_instruction_delta_pct_vs_baseline']:.3f}% "
            f"code_vs_base={row['avg_code_size_delta_pct_vs_baseline']:.3f}% "
            f"W/L/T={row['wins_vs_baseline']}/{row['losses_vs_baseline']}/{row['ties_vs_baseline']}"
        )
    if ranking["misleading_middle_wins"]:
        lines += ["", "Misleading middle wins (middle improved, native regressed):"]
        for item in ranking["misleading_middle_wins"][:20]:
            lines.append(
                f"- {item['policy']} {item['wasm']} native_delta={item['native_instruction_delta_pct_vs_baseline']:.3f}% artifacts={item['artifact_dir']}"
            )
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
