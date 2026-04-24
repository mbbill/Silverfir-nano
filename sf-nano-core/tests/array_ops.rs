//! Regression tests for array GC ops.

use sf_nano_core::{Instance, Value, WasmError};

fn compile(wat_src: &str) -> Vec<u8> {
    wat::parse_str(wat_src).expect("wat parse failed")
}

fn expect_trap<T: core::fmt::Debug>(result: Result<T, WasmError>, needle: &str) {
    let err = result.expect_err("expected trap, got success");
    let msg = err.message();
    assert!(
        msg.contains(needle),
        "expected trap containing {:?}, got {:?}",
        needle,
        msg
    );
}

fn expect_values(values: impl AsRef<[Value]>, expected: &[Value]) {
    assert_eq!(values.as_ref(), expected);
}

#[test]
fn array_get_oob_traps() {
    let wasm = compile(
        r#"
        (module
          (type $vec (array f32))
          (func (export "get") (param $i i32) (result f32)
            (array.get $vec
              (array.new_default $vec (i32.const 3))
              (local.get $i))))
        "#,
    );
    let mut instance = Instance::new(&wasm, &[]).expect("instantiation failed");
    let ok = instance
        .invoke("get", &[Value::I32(0)])
        .expect("in-bounds get");
    expect_values(ok, &[Value::F32(0.0)]);
    let err = instance.invoke("get", &[Value::I32(10)]);
    expect_trap(err, "out of bounds");
}

#[test]
fn array_get_oob_traps_via_param_ref() {
    // Passing the ref via parameter works correctly.
    let wasm = compile(
        r#"
        (module
          (type $vec (array f32))
          (func (export "mk") (result (ref $vec))
            (array.new_default $vec (i32.const 3)))
          (func (export "get2") (param $v (ref $vec)) (param $i i32) (result f32)
            (array.get $vec (local.get $v) (local.get $i))))
        "#,
    );
    let mut instance = Instance::new(&wasm, &[]).expect("instantiation failed");
    let mk = instance.invoke("mk", &[]).expect("mk");
    let v = mk.into_iter().next().unwrap();
    expect_values(
        instance
            .invoke("get2", &[v.clone(), Value::I32(0)])
            .expect("get2(0)"),
        &[Value::F32(0.0)],
    );
    expect_trap(
        instance.invoke("get2", &[v, Value::I32(10)]),
        "out of bounds",
    );
}

#[test]
fn array_get_oob_traps_after_unrelated_call() {
    let wasm = compile(
        r#"
        (module
          (type $vec (array f32))
          (func $side_effect)
          (func (export "get") (param $i i32) (result f32)
            (call $side_effect)
            (array.get $vec
              (array.new_default $vec (i32.const 3))
              (local.get $i))))
        "#,
    );
    let mut instance = Instance::new(&wasm, &[]).expect("instantiation failed");
    expect_values(
        instance.invoke("get", &[Value::I32(0)]).expect("get0"),
        &[Value::F32(0.0)],
    );
    expect_trap(instance.invoke("get", &[Value::I32(10)]), "out of bounds");
}

#[test]
fn array_get_oob_traps_via_call() {
    // Smaller repro: the OOB-triggering function has a parameter `ref $vec`
    // and calls into it. The original spec test uses `(call $new)` returning
    // the ref. Tests the via-call path vs inline construction.
    let wasm = compile(
        r#"
        (module
          (type $vec (array f32))
          (func $new (result (ref $vec))
            (array.new_default $vec (i32.const 3)))
          (func $get (param $i i32) (param $v (ref $vec)) (result f32)
            (array.get $vec (local.get $v) (local.get $i)))
          (func (export "get") (param $i i32) (result f32)
            (call $get (local.get $i) (call $new))))
        "#,
    );
    let mut instance = Instance::new(&wasm, &[]).expect("instantiation failed");
    expect_values(
        instance.invoke("get", &[Value::I32(0)]).expect("get0"),
        &[Value::F32(0.0)],
    );
    expect_trap(instance.invoke("get", &[Value::I32(10)]), "out of bounds");
}

#[test]
fn array_get_oob_traps_after_many_allocations() {
    // Mirror the spectest scenario: many allocations before the OOB call.
    let wasm = compile(
        r#"
        (module
          (type $vec (array f32))
          (type $mvec (array (mut f32)))

          (global (ref $vec) (array.new $vec (f32.const 1) (i32.const 3)))
          (global (ref $vec) (array.new_default $vec (i32.const 3)))

          (func $new (export "new") (result (ref $vec))
            (array.new_default $vec (i32.const 3)))

          (func $get (param $i i32) (param $v (ref $vec)) (result f32)
            (array.get $vec (local.get $v) (local.get $i)))
          (func (export "get") (param $i i32) (result f32)
            (call $get (local.get $i) (call $new)))

          (func $set_get (param $i i32) (param $v (ref $mvec)) (param $y f32) (result f32)
            (array.set $mvec (local.get $v) (local.get $i) (local.get $y))
            (array.get $mvec (local.get $v) (local.get $i)))
          (func (export "set_get") (param $i i32) (param $y f32) (result f32)
            (call $set_get (local.get $i)
              (array.new_default $mvec (i32.const 3))
              (local.get $y)))

          (func (export "len") (result i32)
            (array.len (call $new))))
        "#,
    );
    let mut instance = Instance::new(&wasm, &[]).expect("instantiation failed");
    let _ = instance.invoke("new", &[]).expect("new");
    let _ = instance.invoke("new", &[]).expect("new");
    expect_values(
        instance.invoke("get", &[Value::I32(0)]).expect("get0"),
        &[Value::F32(0.0)],
    );
    expect_values(
        instance
            .invoke("set_get", &[Value::I32(1), Value::F32(7.0)])
            .expect("set_get"),
        &[Value::F32(7.0)],
    );
    expect_values(instance.invoke("len", &[]).expect("len"), &[Value::I32(3)]);
    expect_trap(instance.invoke("get", &[Value::I32(10)]), "out of bounds");
    expect_trap(
        instance.invoke("set_get", &[Value::I32(10), Value::F32(7.0)]),
        "out of bounds",
    );
}

#[test]
fn array_set_roundtrip() {
    let wasm = compile(
        r#"
        (module
          (type $mvec (array (mut f32)))
          (func (export "set_get") (param $i i32) (param $y f32) (result f32)
            (local $v (ref $mvec))
            (local.set $v (array.new_default $mvec (i32.const 3)))
            (array.set $mvec (local.get $v) (local.get $i) (local.get $y))
            (array.get $mvec (local.get $v) (local.get $i))))
        "#,
    );
    let mut instance = Instance::new(&wasm, &[]).expect("instantiation failed");
    let r = instance
        .invoke("set_get", &[Value::I32(1), Value::F32(7.0)])
        .expect("invoke");
    expect_values(r, &[Value::F32(7.0)]);
}

#[test]
fn array_get_s_u_packed() {
    let wasm = compile(
        r#"
        (module
          (type $v (array i8))
          (data $d "\00\01\02\ff\04")
          (func (export "gets") (param $i i32) (result i32)
            (array.get_s $v (array.new_data $v $d (i32.const 0) (i32.const 5)) (local.get $i)))
          (func (export "getu") (param $i i32) (result i32)
            (array.get_u $v (array.new_data $v $d (i32.const 0) (i32.const 5)) (local.get $i))))
        "#,
    );
    let mut instance = Instance::new(&wasm, &[]).expect("instantiation failed");
    let r = instance.invoke("gets", &[Value::I32(3)]).expect("gets");
    expect_values(r, &[Value::I32(-1)]);
    let r = instance.invoke("getu", &[Value::I32(3)]).expect("getu");
    expect_values(r, &[Value::I32(0xff)]);
}

#[test]
fn array_len_returns_length() {
    let wasm = compile(
        r#"
        (module
          (type $v (array f32))
          (func (export "len") (result i32)
            (array.len (array.new_default $v (i32.const 42)))))
        "#,
    );
    let mut instance = Instance::new(&wasm, &[]).expect("instantiation failed");
    let r = instance.invoke("len", &[]).expect("invoke");
    expect_values(r, &[Value::I32(42)]);
}
