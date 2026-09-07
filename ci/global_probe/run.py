from pathlib import Path
import json, os, shutil, statistics, subprocess
root = Path.cwd()
work = Path(os.environ['RUNNER_TEMP']) / 'global-probe'
work.mkdir()
base = work / 'baseline'
candidate = work / 'candidate'
subprocess.run(['git', 'worktree', 'add', '--detach', str(base), 'f73219f56a70b77028f0d79730c7efca29ba3439'], check=True)
subprocess.run(['git', 'worktree', 'add', '--detach', str(candidate), 'd4948f3e90974e58560a4facb8e3e23bacdbab2e'], check=True)
target = work / 'target'
env = dict(os.environ, CARGO_TARGET_DIR=str(target))
backend = Path('sf-nano-core/interp_gen/x86_64.rs')
generator = Path('sf-nano-core/interp_gen/mod.rs')
original_backend = (candidate / backend).read_text()
original_generator = (candidate / generator).read_text()
variants = ['baseline', 'direct', 'rcx', 'padding5', 'indexed', 'align16', 'align32']
outputs = {}
for variant in variants:
    checkout = base if variant == 'baseline' else candidate
    asm = original_backend
    gen = original_generator
    if variant == 'rcx':
        asm = asm.replace('a.ins("mov rax, [rbx + 16]"); // storage cell address', 'a.ins("mov rcx, [rbx + 16]"); // storage cell address')
        asm = asm.replace('a.ins(&format!("mov {}, [rax]", q(rd)));', 'a.ins(&format!("mov {}, [rcx]", q(rd)));')
    if variant == 'indexed':
        asm = asm.replace('a.ins("mov rax, [rbx + 16]"); // storage cell address', 'a.ins("mov rcx, [rbx + 16]");\n                a.ins("xor eax, eax");')
        asm = asm.replace('a.ins(&format!("mov {}, [rax]", q(rd)));', 'a.ins(&format!("mov {}, [rcx + rax]", q(rd)));')
        asm = asm.replace('a.ins("mov rdx, [rbx + 16]"); // storage cell address', 'a.ins("mov rcx, [rbx + 16]");\n                a.ins("xor edx, edx");')
        asm = asm.replace('a.ins(&format!("mov [rdx], {}", q(ra)));', 'a.ins(&format!("mov [rcx + rdx], {}", q(ra)));')
    if variant == 'padding5':
        gen = gen.replace('                isa.emit_handler(&mut a, &st, &v);', '                isa.emit_handler(&mut a, &st, &v);\n                if matches!(op, Op::GlobalGet | Op::GlobalSet) { a.raw(".fill 5, 1, 0x90"); }')
    if variant.startswith('align'):
        align = '4' if variant == 'align16' else '5'
        gen = gen.replace('                a.label(&label);', f'                if matches!(op, Op::GlobalGet | Op::GlobalSet) {{ a.align({align}); }}\n                a.label(&label);')
    if variant != 'baseline':
        (candidate / backend).write_text(asm)
        (candidate / generator).write_text(gen)
    subprocess.run(['cargo', 'build', '-p', 'sf-nano-core', '--release', '--no-default-features', '--features', 'interp'], cwd=checkout, env=env, check=True)
    output = work / ("probe-" + variant)
    subprocess.run(['rustc', '--edition=2021', '-C', 'opt-level=3', '-C', 'lto', str(root / 'ci/global_probe/probe.rs'), '--extern', f'sf_nano_core={target}/release/libsf_nano_core.rlib', '-L', f'{target}/release/deps', '-o', str(output)], check=True)
    outputs[variant] = output
    print('BUILT', variant, flush=True)
results = {name: [] for name in variants}
for run in range(12):
    order = variants[run % len(variants):] + variants[:run % len(variants)]
    if run % 2: order = order[::-1]
    for name in order:
        value = float(subprocess.check_output([str(outputs[name])], text=True))
        results[name].append(value)
        print('SAMPLE', run, name, value, flush=True)
summary = {name: {'median': statistics.median(values), 'samples': values} for name, values in results.items()}
(root / 'global-probe-results.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as out:
    out.write('| Variant | Median seconds | Baseline / candidate |\n|---|---:|---:|\n')
    for name, values in summary.items(): out.write(f"| {name} | {values['median']:.6f} | {summary['baseline']['median'] / values['median']:.4f} |\n")
