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

// Mirrors run_tests.py: adjustable benchmarks take the wall-clock target in
// seconds as their LAST argument and calibrate only a repeat count for their
// fixed work unit. CoreMark preserves its official invocation and duration.
// The patterns and validation strings are kept identical so the two harnesses
// report the same normalized rate.
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
    standardDuration: true,
    pattern: /Iterations\/Sec\s*:\s*(\S+)/,
    source: 'stdout',
    contains: [
      'seedcrc          : 0xe9f5',
      '[0]crclist       : 0xe714',
      '[0]crcmatrix     : 0x1fd7',
      '[0]crcstate      : 0x8e3a',
    ],
  },
  {
    name: 'sha256/sha256.wasm',
    cwd: path.join(SCRIPT_DIR, 'sha256'),
    args: ['sha256.wasm'],
    pattern: /sha256: throughput = (\S+ MB\/s)/,
    source: 'stdout',
    contains: 'hash = 5eb4ca70d0ee472b',
  },
  {
    name: 'bzip2/bzip2.wasm',
    cwd: path.join(SCRIPT_DIR, 'bzip2'),
    args: ['bzip2.wasm'],
    pattern: /bzip2: throughput = (\S+ MB\/s)/,
    source: 'stdout',
    contains: '32 KB input -> 3 KB compressed',
  },
  {
    name: 'lz4/lz4.wasm',
    cwd: path.join(SCRIPT_DIR, 'lz4'),
    args: ['lz4.wasm'],
    pattern: /(lz4 (?:compress|decompress): throughput = \S+ MB\/s)/g,
    source: 'stdout',
    multi: true,
    contains: '64 KB input -> 27 KB compressed',
  },
  {
    name: 'funcref/funcref.wasm',
    cwd: path.join(SCRIPT_DIR, 'funcref'),
    args: ['funcref.wasm'],
    pattern: /(funcref (?:exported-table|direct): rate = \S+ calls\/s)/g,
    source: 'stdout',
    multi: true,
    contains: 'funcref validates: table=264 direct=264',
  },
  // --- Lua ---
  {
    name: 'lua/fib',
    cwd: path.join(SCRIPT_DIR, 'lua'),
    args: ['lua.wasm', 'fib.lua'],
    pattern: /fib: rate = (\S+ fib20\/s)/,
    source: 'stdout',
    contains: 'fib(20) = 6765',
  },
  {
    name: 'lua/sunfish',
    cwd: path.join(SCRIPT_DIR, 'lua'),
    args: ['lua.wasm', 'sunfish.lua'],
    pattern: /Score:\s+(\S+)/,
    source: 'stdout',
    contains: 'Result:        b1c3 / 0',
  },
  {
    name: 'lua/json_bench',
    cwd: path.join(SCRIPT_DIR, 'lua'),
    args: ['lua.wasm', 'json_bench.lua'],
    pattern: /Score:\s+(\S+)/,
    source: 'stdout',
    contains: 'JSON roundtrip validates',
  },
  // --- Floating point ---
  {
    name: 'mandelbrot/mandel.wasm',
    cwd: path.join(SCRIPT_DIR, 'mandelbrot'),
    args: ['mandel.wasm'],
    pattern: /mandel: rate = (\S+ Kpixel\/s)/,
    source: 'stdout',
    contains: 'mandel: checksum = 6a0fc6b0',
  },
  {
    name: 'c-ray/c-ray.wasm',
    cwd: path.join(SCRIPT_DIR, 'c-ray'),
    args: ['c-ray.wasm'],
    stdin: path.join(SCRIPT_DIR, 'c-ray', 'scene'),
    pattern: /c-ray: rate = (\S+ Kpixel\/s)/,
    source: 'stdout',
    contains: 'c-ray: checksum = 75700000',
  },
  // --- Memory bound ---
  {
    name: 'stream/stream.wasm',
    cwd: path.join(SCRIPT_DIR, 'stream'),
    args: ['stream.wasm'],
    pattern: /(Copy|Scale|Add|Triad):\s+(\S+)/g,
    source: 'stdout',
    multi: true,
    contains: 'Solution Validates',
  },
  // --- Database ---
  {
    name: 'sqlite/sqlite_bench.wasm',
    cwd: path.join(SCRIPT_DIR, 'sqlite'),
    args: ['sqlite_bench.wasm'],
    pattern: /sqlite: rate = (\S+ iteration\/s)/,
    source: 'stdout',
    contains: 'sqlite: checksum = 524800',
  },
];

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

    const runArgs = extraArgs
      ? [...test.args, ...extraArgs]
      : test.standardDuration
        ? [...test.args]
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

    if (exitCode !== 0) {
      const detail = `${stderr}\n${stdout}`
        .split(/\r?\n/)
        .map(line => line.trim())
        .filter(Boolean)
        .slice(-4)
        .join(' | ')
        .slice(0, 500);
      return {
        status: 'FAIL',
        metric: `exit code ${exitCode}${detail ? `: ${detail}` : ''}`,
        elapsed,
      };
    }

    const expected = Array.isArray(test.contains)
      ? test.contains
      : test.contains ? [test.contains] : [];
    for (const needle of expected) {
      if (!stdout.includes(needle)) {
        return {
          status: 'FAIL',
          metric: `expected stdout to contain '${needle}'`,
          elapsed,
        };
      }
    }

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

    return { status: 'PASS', metric: `${elapsed.toFixed(3)}s (no metric found)`, elapsed };

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

async function main() {
  console.log(
    'Runtime: Node.js %s (V8 %s), %ss/adjustable benchmark; CoreMark official duration\n',
    process.version,
    process.versions.v8,
    TARGET,
  );

  const results = [];
  for (let i = 0; i < TESTS.length; i++) {
    const test = TESTS[i];
    process.stdout.write(`[${i + 1}/${TESTS.length}] ${test.name} ... `);
    const result = await runTest(test);
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
