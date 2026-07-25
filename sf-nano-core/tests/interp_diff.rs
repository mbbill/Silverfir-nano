//! Differential tests: the interpreter (native dispatch) against the JIT
//! on identical modules. This is the interpreter's whole-pipeline oracle
//! oracle role exercised in reverse: while the interpreter is young, the
//! JIT is the reference.
#![cfg(all(sf_interp_engine, sf_jit))]

use sf_nano_core::module::Module;
use sf_nano_core::{Instance, InterpInstance, Value};

fn value_bits(v: &Value) -> u64 {
    match v {
        Value::I32(x) => *x as u32 as u64,
        Value::I64(x) => *x as u64,
        Value::F32(x) => x.to_bits() as u64,
        Value::F64(x) => x.to_bits(),
        other => panic!("unsupported result value {other:?}"),
    }
}

/// Run one exported function on both engines and compare raw result bits.
fn diff(wasm: &[u8], export: &str, args: &[Value]) {
    let mut jit = Instance::new(&engine(), wasm, &[]).expect("jit instantiation");
    let jit_results = jit.invoke(export, args).expect("jit invoke");
    let jbits: Vec<u64> = jit_results.iter().map(value_bits).collect();

    let module = Module::new("diff", wasm).expect("module parse");
    let iargs: Vec<u64> = args.iter().map(value_bits).collect();

    let mut interp = InterpInstance::new(&engine(), module, None).expect("interp instantiation");
    let idx = interp.find_export(export).expect("export in interp");
    let mut iresults = vec![0u64; jit_results.len()];
    interp
        .invoke(idx, &iargs, &mut iresults)
        .expect("interp invoke");
    assert_eq!(jbits, iresults, "engines disagree on {export}({args:?})");
}

fn diff_wat(src: &str, export: &str, args: &[Value]) {
    let wasm = wat::parse_str(src).expect("wat");
    diff(&wasm, export, args);
}

#[test]
fn diff_recursive_fib() {
    let src = r#"(module (func $fib (export "fib") (param i32) (result i32)
        local.get 0 i32.const 2 i32.lt_u
        (if (result i32)
            (then local.get 0)
            (else
                local.get 0 i32.const 1 i32.sub call $fib
                local.get 0 i32.const 2 i32.sub call $fib
                i32.add))))"#;
    for n in [0, 1, 2, 10, 20] {
        diff_wat(src, "fib", &[Value::I32(n)]);
    }
}

#[test]
fn diff_memory_sieve() {
    // Sieve of Eratosthenes over one memory page; returns prime count < n.
    let src = r#"(module (memory 1)
        (func (export "sieve") (param i32) (result i32)
            (local i32 i32 i32)
            (local.set 1 (i32.const 2))
            (block $done (loop $outer
                (br_if $done (i32.ge_u (local.get 1) (local.get 0)))
                (if (i32.eqz (i32.load8_u (local.get 1)))
                    (then
                        (local.set 3 (i32.add (local.get 3) (i32.const 1)))
                        (local.set 2 (i32.mul (local.get 1) (i32.const 2)))
                        (block $mdone (loop $mark
                            (br_if $mdone (i32.ge_u (local.get 2) (local.get 0)))
                            (i32.store8 (local.get 2) (i32.const 1))
                            (local.set 2 (i32.add (local.get 2) (local.get 1)))
                            (br $mark)))))
                (local.set 1 (i32.add (local.get 1) (i32.const 1)))
                (br $outer)))
            local.get 3))"#;
    diff_wat(src, "sieve", &[Value::I32(1000)]);
    diff_wat(src, "sieve", &[Value::I32(65536)]);
}

#[test]
fn diff_float_iteration() {
    // Mandelbrot-style escape iteration: mixes f64 mul/add/cmp with i32 loop
    // control and i32<->f64 conversions.
    let src = r#"(module (func (export "iter") (param f64 f64) (result i32)
        (local f64 f64 f64 i32)
        (block $out (loop $l
            (br_if $out (i32.ge_u (local.get 5) (i32.const 1000)))
            (local.set 4 (f64.sub (f64.mul (local.get 2) (local.get 2))
                                  (f64.mul (local.get 3) (local.get 3))))
            (local.set 3 (f64.add (f64.mul (f64.const 2) (f64.mul (local.get 2) (local.get 3)))
                                  (local.get 1)))
            (local.set 2 (f64.add (local.get 4) (local.get 0)))
            (local.set 5 (i32.add (local.get 5) (i32.const 1)))
            (br_if $l (f64.lt (f64.add (f64.mul (local.get 2) (local.get 2))
                                       (f64.mul (local.get 3) (local.get 3)))
                              (f64.const 4)))))
        local.get 5))"#;
    diff_wat(src, "iter", &[Value::F64(-0.5), Value::F64(0.5)]);
    diff_wat(src, "iter", &[Value::F64(0.3), Value::F64(0.02)]);
    diff_wat(src, "iter", &[Value::F64(2.0), Value::F64(2.0)]);
}

#[test]
fn diff_br_table_machine() {
    let src = r#"(module (func (export "m") (param i32 i32) (result i32)
        (local i32)
        (local.set 2 (local.get 1))
        (block $b3 (block $b2 (block $b1 (block $b0
            local.get 0 br_table $b0 $b1 $b2 $b3)
            (return (i32.add (local.get 2) (i32.const 100))))
            (return (i32.mul (local.get 2) (i32.const 3))))
            (return (i32.sub (i32.const 0) (local.get 2))))
        i32.const -1))"#;
    for sel in 0..5 {
        diff_wat(src, "m", &[Value::I32(sel), Value::I32(17)]);
    }
}

#[test]
fn diff_call_indirect() {
    let src = r#"(module
        (type $t (func (param i32) (result i32)))
        (table 3 funcref) (elem (i32.const 0) $inc $dec $neg)
        (func $inc (type $t) local.get 0 i32.const 1 i32.add)
        (func $dec (type $t) local.get 0 i32.const 1 i32.sub)
        (func $neg (type $t) i32.const 0 local.get 0 i32.sub)
        (func (export "go") (param i32 i32) (result i32)
            local.get 1 local.get 0 call_indirect (type $t)))"#;
    for f in 0..3 {
        diff_wat(src, "go", &[Value::I32(f), Value::I32(41)]);
    }
}

#[test]
fn diff_multivalue_block() {
    let src = r#"(module (func (export "mv") (param i32) (result i32)
        (local i32 i32)
        (block (result i32 i32)
            local.get 0
            local.get 0 i32.const 3 i32.mul)
        local.set 2
        local.set 1
        (i32.sub (local.get 2) (local.get 1))))"#;
    diff_wat(src, "mv", &[Value::I32(21)]);
}

#[test]
fn diff_i64_and_conversions() {
    let src = r#"(module (func (export "g") (param i64 f64) (result i64)
        local.get 0 i64.const 13 i64.rotl
        local.get 1 i64.trunc_sat_f64_s
        i64.xor
        local.get 0 i64.popcnt
        i64.add))"#;
    diff_wat(
        src,
        "g",
        &[Value::I64(0x0123_4567_89ab_cdef), Value::F64(1e9)],
    );
    diff_wat(src, "g", &[Value::I64(-1), Value::F64(-1e30)]);
}

#[test]
fn diff_globals_and_bulk_memory() {
    let src = r#"(module (memory 1) (global $g (mut i32) (i32.const 0))
        (data (i32.const 8) "\01\02\03\04\05\06\07\08")
        (func (export "go") (result i32)
            (memory.copy (i32.const 100) (i32.const 8) (i32.const 8))
            (memory.fill (i32.const 108) (i32.const 9) (i32.const 4))
            (global.set $g (i32.add (global.get $g) (i32.load (i32.const 104))))
            (i32.add (global.get $g) (i32.load (i32.const 108)))))"#;
    diff_wat(src, "go", &[]);
}

#[test]
fn diff_fib_min_wasm_from_disk() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../benchmarks/wasi/fib/fib_min.wasm"
    );
    let wasm = std::fs::read(path).expect("read fib_min.wasm");
    for n in [1, 5, 10, 20, 25] {
        diff(&wasm, "fib", &[Value::I32(n)]);
    }
}

/// One engine on this target's defaults, for the tests in this file.
fn engine() -> sf_nano_core::Engine {
    sf_nano_core::Engine::with_defaults()
}
