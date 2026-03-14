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

const TESTS = [
  {
    name: 'coremark/coremark.wasm',
    cwd: path.join(SCRIPT_DIR, 'coremark'),
    args: ['coremark.wasm'],
    pattern: /CoreMark 1\.0 :\s*(\S+)/,
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
  {
    name: 'mandelbrot/mandel.wasm',
    cwd: path.join(SCRIPT_DIR, 'mandelbrot'),
    args: ['mandel.wasm', '128', '6e6'],
    pattern: /Elapsed time:\s*(.+)/,
    source: 'stderr',
  },
  {
    name: 'c-ray/c-ray.wasm',
    cwd: path.join(SCRIPT_DIR, 'c-ray'),
    args: ['c-ray.wasm', '-s', '4000x4000'],
    stdin: path.join(SCRIPT_DIR, 'c-ray', 'scene'),
    pattern: /Rendering took:\s*(.+\))/,
    source: 'stderr',
  },
  {
    name: 'stream/stream.wasm',
    cwd: path.join(SCRIPT_DIR, 'stream'),
    args: ['stream.wasm'],
    pattern: /(Copy|Scale|Add|Triad):\s+(\S+)/g,
    source: 'stdout',
    multi: true,
  },
  {
    name: 'lua/fib',
    cwd: path.join(SCRIPT_DIR, 'lua'),
    args: ['lua.wasm', 'fib.lua'],
    pattern: null,
    source: null,
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
];

async function runTest(test) {
  const tmpOut = path.join(os.tmpdir(), `wasi_out_${process.pid}.tmp`);
  const tmpErr = path.join(os.tmpdir(), `wasi_err_${process.pid}.tmp`);

  let outFd, errFd, inFd;
  try {
    outFd = fs.openSync(tmpOut, 'w');
    errFd = fs.openSync(tmpErr, 'w');
    if (test.stdin) inFd = fs.openSync(test.stdin, 'r');

    const wasmPath = path.join(test.cwd, test.args[0]);
    if (!fs.existsSync(wasmPath)) return { status: 'SKIP', metric: 'wasm file not found' };

    const wasi = new WASI({
      version: 'preview1',
      args: test.args,
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
    if (!test.pattern) {
      return { status: 'PASS', metric: `${elapsed.toFixed(3)}s (wall clock)`, elapsed };
    }

    const text = test.source === 'stderr' ? stderr : stdout;

    if (test.multi) {
      const matches = [...text.matchAll(test.pattern)];
      if (matches.length > 0) {
        const metric = matches[0][0].includes('throughput')
          ? matches.map(m => m[1]).join('; ')
          : matches.map(m => `${m[1]}: ${m[2]} MB/s`).join(', ');
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

async function main() {
  console.log('Runtime: Node.js %s (V8 %s)\n', process.version, process.versions.v8);

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
