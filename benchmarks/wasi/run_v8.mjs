#!/usr/bin/env node
/**
 * Run WASI benchmarks in Node.js (V8).
 * Usage: node run_v8.mjs
 */
import { WASI } from 'node:wasi';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));

// Mirrors run_tests.py: every benchmark takes the wall-clock target in
// seconds as its LAST argument and sizes its own workload to hit it. The
// patterns and validation strings are kept identical so the two harnesses
// report the same metric for the same run.
const DEFAULT_TARGET = 2.0;
const TARGET = (() => {
  const i = process.argv.indexOf('--time');
  if (i >= 0 && process.argv[i + 1]) {
    const v = parseFloat(process.argv[i + 1]);
    if (v > 0) return v;
  }
  return DEFAULT_TARGET;
})();

const TESTS = [
  // --- Integer / control flow ---
  {
    name: 'coremark/coremark.wasm',
    cwd: path.join(SCRIPT_DIR, 'coremark'),
    args: ['coremark.wasm'],
    pattern: /Iterations\/Sec\s*:\s*(\S+)/,
    source: 'stdout',
  },
  {
    name: 'sha256/sha256.wasm',
    cwd: path.join(SCRIPT_DIR, 'sha256'),
    args: ['sha256.wasm'],
    pattern: /sha256: throughput = (\S+ MB\/s)/,
    source: 'stdout',
  },
  {
    name: 'bzip2/bzip2.wasm',
    cwd: path.join(SCRIPT_DIR, 'bzip2'),
    args: ['bzip2.wasm'],
    pattern: /bzip2: throughput = (\S+ MB\/s)/,
    source: 'stdout',
  },
  {
    name: 'lz4/lz4.wasm',
    cwd: path.join(SCRIPT_DIR, 'lz4'),
    args: ['lz4.wasm'],
    pattern: /(lz4 (?:compress|decompress): throughput = \S+ MB\/s)/g,
    source: 'stdout',
    multi: true,
  },
  // --- Lua ---
  {
    name: 'lua/fib',
    cwd: path.join(SCRIPT_DIR, 'lua'),
    args: ['lua.wasm', 'fib.lua'],
    pattern: /fib: rate = (\S+ fib20\/s)/,
    source: 'stdout',
  },
  {
    name: 'lua/sunfish',
    cwd: path.join(SCRIPT_DIR, 'lua'),
    args: ['lua.wasm', 'sunfish.lua'],
    pattern: /Score:\s+(\S+)/,
    source: 'stdout',
  },
  {
    name: 'lua/json_bench',
    cwd: path.join(SCRIPT_DIR, 'lua'),
    args: ['lua.wasm', 'json_bench.lua'],
    pattern: /Score:\s+(\S+)/,
    source: 'stdout',
  },
  // --- Floating point ---
  {
    name: 'mandelbrot/mandel.wasm',
    cwd: path.join(SCRIPT_DIR, 'mandelbrot'),
    args: ['mandel.wasm'],
    pattern: /mandel: rate = (\S+ Kpixel\/s)/,
    source: 'stdout',
  },
  {
    name: 'c-ray/c-ray.wasm',
    cwd: path.join(SCRIPT_DIR, 'c-ray'),
    args: ['c-ray.wasm'],
    stdin: path.join(SCRIPT_DIR, 'c-ray', 'scene'),
    pattern: /c-ray: rate = (\S+ Kpixel\/s)/,
    source: 'stdout',
  },
  // --- Memory bound ---
  {
    name: 'stream/stream.wasm',
    cwd: path.join(SCRIPT_DIR, 'stream'),
    args: ['stream.wasm'],
    pattern: /(Copy|Scale|Add|Triad):\s+(\S+)/g,
    source: 'stdout',
    multi: true,
  },
  // --- Database ---
  // speedtest1 cannot ramp a batch, so its size is chosen here, exactly as
  // run_tests.py does it: probe at size 10, and if that already covers half
  // the target report the probe itself. Metric is work/second, never the
  // TOTAL line -- once the size is chosen to hit the target, elapsed time is
  // ~the target on every runtime and carries no information.
  {
    name: 'sqlite/speedtest1.wasm',
    cwd: path.join(SCRIPT_DIR, 'sqlite'),
    args: ['speedtest1.wasm', '--memdb', '--nosync', '--journal', 'off',
           '--testset', 'main'],
    sizeArg: true,
    source: 'stdout',
  },
];

const TOTAL_RE = /^\s*TOTAL\.+\s+([\d.]+)s/m;

async function runTest(test, extraArgs) {
  const tmpOut = path.join(os.tmpdir(), `wasi_out_${process.pid}.tmp`);
  const tmpErr = path.join(os.tmpdir(), `wasi_err_${process.pid}.tmp`);

  let outFd, errFd, inFd;
  try {
    outFd = fs.openSync(tmpOut, 'w');
    errFd = fs.openSync(tmpErr, 'w');
    if (test.stdin) inFd = fs.openSync(test.stdin, 'r');

    const wasmPath = path.join(test.cwd, test.args[0]);
    if (!fs.existsSync(wasmPath)) return { status: 'SKIP', metric: 'wasm file not found' };

    const runArgs = extraArgs ? [...test.args, ...extraArgs]
                              : [...test.args, String(TARGET)];

    const wasi = new WASI({
      version: 'preview1',
      args: runArgs,
      preopens: { '.': test.cwd },
      stdin: inFd !== undefined ? inFd : 0,
      stdout: outFd,
      stderr: errFd,
    });

    const wasmBytes = fs.readFileSync(wasmPath);
    const module = await WebAssembly.compile(wasmBytes);
    const instance = await WebAssembly.instantiate(module, wasi.getImportObject());

    const t0 = performance.now();
    let exitCode = 0;
    try {
      wasi.start(instance);
    } catch (e) {
      exitCode = e.exitCode ?? 1;
    }
    const elapsed = (performance.now() - t0) / 1000;

    fs.closeSync(outFd); outFd = null;
    fs.closeSync(errFd); errFd = null;
    if (inFd !== undefined) { fs.closeSync(inFd); inFd = null; }

    const stdout = fs.readFileSync(tmpOut, 'utf-8');
    const stderr = fs.readFileSync(tmpErr, 'utf-8');

    // Extract metric
    if (test.sizeArg) return { status: 'RAW', stdout, elapsed };

    if (!test.pattern) {
      return { status: 'PASS', metric: `${elapsed.toFixed(3)}s (wall clock)`, elapsed };
    }

    const text = test.source === 'stderr' ? stderr : stdout;

    if (test.multi) {
      const matches = [...text.matchAll(test.pattern)];
      if (matches.length > 0) {
        const sep = test.separator || '; ';
        let metric;
        if (matches[0][2] !== undefined) {
          metric = matches.map(m => `${m[1]}: ${m[2]} MB/s`).join(', ');
        } else {
          metric = matches.map(m => m[1].trim()).join(sep);
        }
        return { status: 'PASS', metric, elapsed };
      }
    } else {
      const m = text.match(test.pattern);
      if (m) return { status: 'PASS', metric: m[1].trim(), elapsed };
    }

    if (exitCode === 0) return { status: 'PASS', metric: `${elapsed.toFixed(3)}s (no metric found)`, elapsed };
    return { status: 'FAIL', metric: `exit code ${exitCode}`, elapsed };

  } catch (e) {
    return { status: 'FAIL', metric: e.message };
  } finally {
    if (outFd != null) try { fs.closeSync(outFd); } catch {}
    if (errFd != null) try { fs.closeSync(errFd); } catch {}
    if (inFd != null) try { fs.closeSync(inFd); } catch {}
    try { fs.unlinkSync(tmpOut); } catch {}
    try { fs.unlinkSync(tmpErr); } catch {}
  }
}

async function runSqlite(test) {
  const cal = 10;
  const probe = await runTest(test, ['--size', String(cal)]);
  if (probe.status !== 'RAW') return probe;
  const pm = TOTAL_RE.exec(probe.stdout);
  if (!pm || parseFloat(pm[1]) <= 0) return { status: 'FAIL', metric: 'no TOTAL line' };
  const psecs = parseFloat(pm[1]);
  if (psecs >= TARGET / 2) {
    return { status: 'PASS', elapsed: psecs,
             metric: `${(cal / psecs).toFixed(2)} size/s (size=${cal}, ${psecs.toFixed(3)}s)` };
  }
  let size = Math.round(cal * TARGET / psecs);
  size = Math.max(1, Math.min(size, 1000));
  const real = await runTest(test, ['--size', String(size)]);
  if (real.status !== 'RAW') return real;
  const rm = TOTAL_RE.exec(real.stdout);
  if (!rm || parseFloat(rm[1]) <= 0) return { status: 'FAIL', metric: 'no TOTAL line' };
  const secs = parseFloat(rm[1]);
  return { status: 'PASS', elapsed: secs,
           metric: `${(size / secs).toFixed(2)} size/s (size=${size}, ${secs.toFixed(3)}s)` };
}

async function main() {
  console.log('Runtime: Node.js %s (V8 %s), %ss/benchmark\n',
              process.version, process.versions.v8, TARGET);

  const results = [];
  for (let i = 0; i < TESTS.length; i++) {
    const test = TESTS[i];
    process.stdout.write(`[${i + 1}/${TESTS.length}] ${test.name} ... `);
    const result = test.sizeArg ? await runSqlite(test) : await runTest(test);
    results.push({ name: test.name, ...result });
    console.log(`${result.status}  ${result.metric}`);
  }

  const passed = results.filter(r => r.status === 'PASS').length;
  const failed = results.filter(r => r.status === 'FAIL').length;
  const skipped = results.filter(r => r.status === 'SKIP').length;

  console.log('\n' + '='.repeat(72));
  console.log(`Results: ${passed} passed, ${failed} failed, ${skipped} skipped / ${TESTS.length} total`);
  console.log('='.repeat(72));
  console.log();
  console.log(`${'Test'.padEnd(35)} ${'Status'.padEnd(6)} Metric`);
  console.log('-'.repeat(72));
  for (const r of results) {
    console.log(`${r.name.padEnd(35)} ${r.status.padEnd(6)} ${r.metric}`);
  }

  process.exit(failed > 0 ? 1 : 0);
}

main();
