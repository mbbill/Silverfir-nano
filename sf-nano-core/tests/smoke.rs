//! Basic smoke tests for sf-nano.

use sf_nano_core::{Instance, Value};

/// (module
///   (func $add (export "add") (param i32 i32) (result i32)
///     local.get 0
///     local.get 1
///     i32.add))
const ADD_WASM: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01,
    0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00, 0x0a, 0x09,
    0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
];

#[test]
fn test_add() {
    let mut instance = Instance::new(&engine(), ADD_WASM, &[]).expect("instantiation failed");
    let result = instance
        .invoke("add", &[Value::I32(3), Value::I32(4)])
        .expect("invoke failed");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], Value::I32(7));
}

#[test]
fn test_add_negative() {
    let mut instance = Instance::new(&engine(), ADD_WASM, &[]).expect("instantiation failed");
    let result = instance
        .invoke("add", &[Value::I32(-1), Value::I32(1)])
        .expect("invoke failed");
    assert_eq!(result[0], Value::I32(0));
}

#[test]
fn test_missing_export() {
    let mut instance = Instance::new(&engine(), ADD_WASM, &[]).expect("instantiation failed");
    let result = instance.invoke("nonexistent", &[Value::I32(1), Value::I32(2)]);
    assert!(result.is_err());
}

/// Fibonacci with locals, if/else, loop, br_if
const FIB_WASM: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00, 0x07, 0x07, 0x01, 0x03, 0x66, 0x69, 0x62, 0x00, 0x00, 0x0a, 0x51, 0x01,
    0x4f, 0x01, 0x03, 0x7f, 0x20, 0x00, 0x41, 0x01, 0x4c, 0x04, 0x40, 0x20, 0x00, 0x0f, 0x0b, 0x41,
    0x00, 0x21, 0x01, 0x41, 0x01, 0x21, 0x02, 0x41, 0x01, 0x21, 0x03, 0x02, 0x40, 0x03, 0x40, 0x20,
    0x03, 0x20, 0x00, 0x4e, 0x0d, 0x01, 0x20, 0x01, 0x20, 0x02, 0x6a, 0x21, 0x01, 0x20, 0x01, 0x20,
    0x02, 0x6b, 0x21, 0x01, 0x20, 0x01, 0x20, 0x02, 0x6a, 0x21, 0x02, 0x20, 0x02, 0x20, 0x01, 0x6b,
    0x21, 0x01, 0x20, 0x03, 0x41, 0x01, 0x6a, 0x21, 0x03, 0x0c, 0x00, 0x0b, 0x0b, 0x20, 0x02, 0x0b,
];

#[test]
fn test_fibonacci() {
    let mut instance = Instance::new(&engine(), FIB_WASM, &[]).expect("instantiation failed");
    let cases = [(0, 0), (1, 1), (2, 1), (3, 2), (5, 5), (10, 55), (20, 6765)];
    for (input, expected) in cases {
        let result = instance
            .invoke("fib", &[Value::I32(input)])
            .expect("invoke failed");
        assert_eq!(
            result[0],
            Value::I32(expected),
            "fib({}) should be {}",
            input,
            expected
        );
    }
}

/// A module that outgrows the configured native code arena must fail
/// instantiation with an error, not panic the process — on a device the
/// arena is a few KiB and one large guest function used to abort firmware.
#[cfg(feature = "jit")]
#[test]
fn test_code_arena_exhaustion_is_an_error() {
    let config = sf_nano_core::Config::new().code_arena_bytes(64);
    let engine = sf_nano_core::Engine::new(config).expect("engine config rejected");
    let result = Instance::new(&engine, FIB_WASM, &[]);
    assert!(
        result.is_err(),
        "a 64-byte arena must fail instantiation cleanly"
    );
}

#[test]
fn test_global_counter_loop_writes_back() {
    let wasm = wat::parse_str(
        r#"(module
            (global $count (export "count") (mut i32) (i32.const 0))
            (func (export "run") (param $n i32) (result i32)
                (global.set $count (local.get $n))
                (loop $continue
                    (br_if $continue
                        (global.set $count
                            (i32.sub (global.get $count) (i32.const 1)))
                        (global.get $count)))
                (global.get $count)))"#,
    )
    .expect("counter WAT should parse");
    let mut instance = Instance::new(&engine(), &wasm, &[]).expect("instantiation failed");

    for input in [1, 2, 17, 1_000] {
        let result = instance
            .invoke("run", &[Value::I32(input)])
            .expect("counter invoke failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], Value::I32(0));
        assert_eq!(
            instance.get_global("count").expect("global lookup failed"),
            Some(Value::I32(0))
        );
    }
}

/// One engine on this target's defaults, for the tests in this file.
fn engine() -> sf_nano_core::Engine {
    sf_nano_core::Engine::with_defaults()
}
