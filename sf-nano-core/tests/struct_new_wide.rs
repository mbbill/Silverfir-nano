//! Regression tests for `struct.new` with more than three fields.
//!
//! The old preserved-helper ABI packed struct fields into a fixed 3-slot
//! inline payload. Wider structs now materialize fields into a stack payload
//! area and pass a pointer + count, mirroring `array.new_fixed`.

use sf_nano_core::{Instance, Value};

fn compile(wat_src: &str) -> Vec<u8> {
    wat::parse_str(wat_src).expect("wat parse failed")
}

fn expect_values(values: impl AsRef<[Value]>, expected: &[Value]) {
    assert_eq!(values.as_ref(), expected);
}

#[test]
fn struct_new_five_i32_fields_roundtrip() {
    let wasm = compile(
        r#"
        (module
          (type $s (struct (field i32) (field i32) (field i32) (field i32) (field i32)))
          (func (export "get0") (result i32)
            (struct.get $s 0
              (struct.new $s (i32.const 100) (i32.const 200) (i32.const 300)
                             (i32.const 400) (i32.const 500))))
          (func (export "get1") (result i32)
            (struct.get $s 1
              (struct.new $s (i32.const 100) (i32.const 200) (i32.const 300)
                             (i32.const 400) (i32.const 500))))
          (func (export "get2") (result i32)
            (struct.get $s 2
              (struct.new $s (i32.const 100) (i32.const 200) (i32.const 300)
                             (i32.const 400) (i32.const 500))))
          (func (export "get3") (result i32)
            (struct.get $s 3
              (struct.new $s (i32.const 100) (i32.const 200) (i32.const 300)
                             (i32.const 400) (i32.const 500))))
          (func (export "get4") (result i32)
            (struct.get $s 4
              (struct.new $s (i32.const 100) (i32.const 200) (i32.const 300)
                             (i32.const 400) (i32.const 500))))
        )
        "#,
    );
    let mut instance = Instance::new(&wasm, &[]).expect("instantiation failed");
    for (i, expected) in [100, 200, 300, 400, 500].iter().enumerate() {
        let name = format!("get{}", i);
        let result = instance.invoke(&name, &[]).expect("invoke failed");
        assert_eq!(result.as_ref(), &[Value::I32(*expected)], "field {}", i);
    }
}

#[test]
fn struct_new_eight_mixed_fields_roundtrip() {
    // Eight fields exercising i32, i64, f32, f64 storage classes so the
    // payload area must correctly handle mixed widths and pair operands.
    let wasm = compile(
        r#"
        (module
          (type $s (struct
            (field i32) (field i64) (field f32) (field f64)
            (field i32) (field i64) (field f32) (field f64)))
          (global $g (ref $s) (struct.new $s
            (i32.const 1) (i64.const 2) (f32.const 3) (f64.const 4)
            (i32.const 5) (i64.const 6) (f32.const 7) (f64.const 8)))
          (func (export "build_and_get_i32") (param $a i32) (param $b i32) (result i32)
            (struct.get $s 4
              (struct.new $s
                (local.get $a) (i64.const 22) (f32.const 33) (f64.const 44)
                (local.get $b) (i64.const 66) (f32.const 77) (f64.const 88))))
          (func (export "build_and_get_i64") (param $a i64) (result i64)
            (struct.get $s 5
              (struct.new $s
                (i32.const 11) (i64.const 22) (f32.const 33) (f64.const 44)
                (i32.const 55) (local.get $a) (f32.const 77) (f64.const 88))))
          (func (export "build_and_get_f64") (param $a f64) (result f64)
            (struct.get $s 7
              (struct.new $s
                (i32.const 11) (i64.const 22) (f32.const 33) (f64.const 44)
                (i32.const 55) (i64.const 66) (f32.const 77) (local.get $a))))
        )
        "#,
    );
    let mut instance = Instance::new(&wasm, &[]).expect("instantiation failed");

    let r = instance
        .invoke("build_and_get_i32", &[Value::I32(1111), Value::I32(2222)])
        .expect("invoke failed");
    expect_values(r, &[Value::I32(2222)]);

    let r = instance
        .invoke("build_and_get_i64", &[Value::I64(9999)])
        .expect("invoke failed");
    expect_values(r, &[Value::I64(9999)]);

    let r = instance
        .invoke("build_and_get_f64", &[Value::F64(3.14)])
        .expect("invoke failed");
    expect_values(r, &[Value::F64(3.14)]);
}
