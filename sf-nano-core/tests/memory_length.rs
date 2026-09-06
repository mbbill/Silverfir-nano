//! Memory length stays current across calls, growth and explicit checks.
use sf_nano_core::{Config, Engine, Instance, Value};

#[test]
fn memory_growth_preserves_live_values_and_refreshes_length() {
    for width in [32, 64] {
        let ty = format!("i{width}");
        let memory_type = if width == 64 { "i64 " } else { "" };
        let wasm = wat::parse_str(format!(
            r#"(module
                (memory {memory_type}1 2)
                (func $grow (result {ty})
                    (memory.grow ({ty}.const 1)))
                (func (export "run") (param $seed i64) (result i64 {ty} {ty} i32)
                    (local $a i64) (local $b i64) (local $c i64) (local $d i64)
                    (local $e i64) (local $f i64) (local $g i64) (local $h i64)
                    (local $old {ty}) (local $n i32)
                    (local.set $a (i64.add (local.get $seed) (i64.const 1)))
                    (local.set $b (i64.add (local.get $seed) (i64.const 2)))
                    (local.set $c (i64.add (local.get $seed) (i64.const 3)))
                    (local.set $d (i64.add (local.get $seed) (i64.const 4)))
                    (local.set $e (i64.add (local.get $seed) (i64.const 5)))
                    (local.set $f (i64.add (local.get $seed) (i64.const 6)))
                    (local.set $g (i64.add (local.get $seed) (i64.const 7)))
                    (local.set $h (i64.add (local.get $seed) (i64.const 8)))
                    (local.set $n (i32.const 3))
                    (loop $again
                        (local.set $old (call $grow))
                        (local.set $a (i64.add (local.get $a) (local.get $b)))
                        (local.set $c (i64.add (local.get $c) (local.get $d)))
                        (local.set $e (i64.add (local.get $e) (local.get $f)))
                        (local.set $g (i64.add (local.get $g) (local.get $h)))
                        (br_if $again (local.tee $n (i32.sub (local.get $n) (i32.const 1)))))
                    (memory.fill ({ty}.const 65536) (i32.const 91) ({ty}.const 4))
                    (memory.copy ({ty}.const 131068) ({ty}.const 65536) ({ty}.const 4))
                    (i64.add (i64.add (local.get $a) (local.get $c))
                        (i64.add (local.get $e) (local.get $g)))
                    (local.get $old)
                    (memory.size)
                    (i32.load8_u ({ty}.const 131071)))
                (func (export "read") (param {ty}) (result i32)
                    (i32.load8_u (local.get 0))))"#
        ))
        .unwrap();
        let engine = Engine::new(Config::new()).unwrap();
        let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
        let value = |x| {
            if width == 64 {
                Value::I64(x)
            } else {
                Value::I32(x as i32)
            }
        };
        for seed in [0, 17, -19, i64::MAX] {
            let expected = seed.wrapping_mul(16).wrapping_add(76);
            assert_eq!(
                instance.invoke("run", &[Value::I64(seed)]).unwrap(),
                [Value::I64(expected), value(-1), value(2), Value::I32(91)]
            );
            for address in [-1, 131072, 0x1_0000_0000] {
                if width == 32 && address == 0x1_0000_0000 {
                    continue;
                }
                assert!(
                    instance.invoke("read", &[value(address)]).is_err(),
                    "width={width}, address={address}, seed={seed}"
                );
            }
        }
    }
}

#[test]
fn normal_and_template_memory_checks_preserve_the_access_address() {
    let wasm = wat::parse_str(
        r#"(module
        (memory 1)
        (func (export "read") (param i32) (result i32)
            (i32.load (local.get 0)))
        (func (export "write") (param i32 i32)
            (i32.store (local.get 0) (local.get 1))))"#,
    )
    .unwrap();
    // A one-byte compiler budget forces the supported streaming template path.
    for budget in [u32::MAX, 1] {
        let engine = Engine::new(Config::new().compiler_ram_budget_bytes(budget)).unwrap();
        let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
        for address in [0, 1, 65532] {
            instance
                .invoke("write", &[Value::I32(address), Value::I32(0x12345678)])
                .unwrap();
            assert_eq!(
                instance.invoke("read", &[Value::I32(address)]).unwrap(),
                [Value::I32(0x12345678)],
                "budget={budget}, address={address}"
            );
        }
        for address in [-1, 65533, 65536] {
            assert!(instance.invoke("read", &[Value::I32(address)]).is_err());
            assert!(instance
                .invoke("write", &[Value::I32(address), Value::I32(0)])
                .is_err());
        }
        assert_eq!(
            instance.invoke("read", &[Value::I32(65532)]).unwrap(),
            [Value::I32(0x12345678)]
        );
    }
}

#[test]
fn ordinary_callers_exchange_parameters_and_results_with_template_bodies() {
    let padding = "nop ".repeat(128);
    for ty in ["i32", "i64"] {
        if ty == "i64" && usize::BITS == 32 {
            // The streaming JIT supports i64 only on 64-bit GP backends.
            continue;
        }
        let wasm = wat::parse_str(format!(
            r#"(module
            (func $large (param {ty} {ty}) (result {ty} {ty})
                {padding}
                (local.get 0) ({ty}.add (local.get 1) ({ty}.const 1)))
            (func (export "run") (param {ty} {ty}) (result {ty} {ty})
                (call $large (local.get 0) (local.get 1))))"#
        ))
        .unwrap();
        // The small caller fits this budget, while the padded callee does not.
        let engine = Engine::new(Config::new().compiler_ram_budget_bytes(4_000)).unwrap();
        let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
        let value = |x| {
            if ty == "i64" {
                Value::I64(x)
            } else {
                Value::I32(x as i32)
            }
        };
        assert_eq!(
            instance.invoke("run", &[value(17), value(-4)]).unwrap(),
            [value(17), value(-3)]
        );
    }
}

#[test]
fn memory64_offsets_and_access_ends_cannot_wrap_into_low_memory() {
    let wasm = wat::parse_str(
        r#"(module
        (memory i64 1)
        (memory i64 1)
        (func (export "load0") (param i64) (result i64)
            (i64.load offset=16 (local.get 0)))
        (func (export "load1") (param i64) (result i64)
            (i64.load 1 offset=16 (local.get 0)))
        (func (export "store0") (param i64)
            (i64.store offset=16 (local.get 0) (i64.const 19)))
        (func (export "store1") (param i64)
            (i64.store 1 offset=16 (local.get 0) (i64.const 19))))"#,
    )
    .unwrap();
    let engine = Engine::new(Config::new()).unwrap();
    let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
    for memory in [0, 1] {
        let load = format!("load{memory}");
        let store = format!("store{memory}");
        for address in [0, 65_512] {
            instance.invoke(&store, &[Value::I64(address)]).unwrap();
            assert_eq!(
                instance.invoke(&load, &[Value::I64(address)]).unwrap(),
                [Value::I64(19)]
            );
        }
        for address in [65_513, 0x1_0000_0000, -1, -8, -16, -17, -23] {
            assert!(
                instance.invoke(&load, &[Value::I64(address)]).is_err(),
                "{load} at {address}"
            );
            assert!(
                instance.invoke(&store, &[Value::I64(address)]).is_err(),
                "{store} at {address}"
            );
        }
        assert_eq!(
            instance.invoke(&load, &[Value::I64(0)]).unwrap(),
            [Value::I64(19)]
        );
    }
}
