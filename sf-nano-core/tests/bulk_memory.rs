//! Regression tests for native bulk-memory helper calls.

use sf_nano_core::{Instance, Value};

fn compile(wat_src: &str) -> Vec<u8> {
    wat::parse_str(wat_src).expect("wat parse failed")
}

#[test]
fn bulk_memory_preserves_c_argument_lanes() {
    let wasm = compile(
        r#"
        (module
          (memory 1)
          (func (export "fill")
            (result i32)
            (local $l0 i32) (local $l1 i32) (local $l2 i32)
            (local $l3 i32) (local $l4 i32) (local $l5 i32)
            (local $l6 i32) (local $l7 i32) (local $l8 i32)
            (local.set $l0 (i32.const 1))
            (local.set $l1 (i32.const 2))
            (local.set $l2 (i32.const 3))
            (local.set $l3 (i32.const 4))
            (local.set $l4 (i32.const 5))
            (local.set $l5 (i32.const 6))
            (local.set $l6 (i32.const 7))
            (local.set $l7 (i32.const 8))
            (local.set $l8 (i32.const 9))
            (memory.fill (i32.const 128) (i32.const 37) (i32.const 32))
            (i32.add
              (i32.load8_u (i32.const 128))
              (i32.add (local.get $l0)
                (i32.add (local.get $l1)
                  (i32.add (local.get $l2)
                    (i32.add (local.get $l3)
                      (i32.add (local.get $l4)
                        (i32.add (local.get $l5)
                          (i32.add (local.get $l6)
                            (i32.add (local.get $l7) (local.get $l8)))))))))))
          (func (export "copy")
            (result i32)
            (memory.fill (i32.const 256) (i32.const 42) (i32.const 32))
            (memory.copy (i32.const 512) (i32.const 256) (i32.const 32))
            (i32.load8_u (i32.const 512))))
        "#,
    );
    let mut instance = Instance::new(&engine(), &wasm, &[]).expect("instantiation failed");
    let fill = instance.invoke("fill", &[]).expect("memory.fill failed");
    assert_eq!(fill.as_slice(), &[Value::I32(82)]);

    let copy = instance.invoke("copy", &[]).expect("memory.copy failed");
    assert_eq!(copy.as_slice(), &[Value::I32(42)]);
}

#[test]
fn bulk_memory_bounds_trap_unwinds_helper_frame() {
    let wasm = compile(
        r#"
        (module
          (memory 1)
          (func (export "fill") (param $dest i32) (param $len i32)
            (memory.fill (local.get $dest) (i32.const 0) (local.get $len))))
        "#,
    );
    let mut instance = Instance::new(&engine(), &wasm, &[]).expect("instantiation failed");
    let err = instance
        .invoke("fill", &[Value::I32(65_520), Value::I32(32)])
        .expect_err("out-of-bounds fill should trap");
    assert!(err.message().contains("out of bounds"), "{err:?}");
}

/// One engine on this target's defaults, for the tests in this file.
fn engine() -> sf_nano_core::Engine {
    sf_nano_core::Engine::with_defaults()
}
