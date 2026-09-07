//! JIT-specific native-code and GC regression fixtures.
use super::*;

#[cfg(target_arch = "aarch64")]
#[test]
fn native_if_mixed_operands_uses_direct_arm64() {
    let mut runner = instantiate_first_module_with_backend("if.wast", Tier::Jit);
    let id = only_instance_id(&runner);
    let ret = runner
        .world
        .invoke(id, "as-mixed-operands", &[Value::I32(0)])
        .expect("invoke export");
    assert_eq!(ret.as_slice(), &[Value::I32(-3)]);

    let instance = runner.world.instance(id).expect("instance");
    assert_eq!(
        instance.function_has_native_code("as-mixed-operands"),
        Some(true),
        "expected native code to be compiled for as-mixed-operands"
    );
}

#[test]
fn native_regress_br_on_cast_with_i31ref() {
    let wasm_bytes = wat::parse_str(
        r#"
        (module
          (table 2 anyref)

          (func (export "init")
            (table.set (i32.const 0) (ref.i31 (i32.const 7)))
            (table.set (i32.const 1) (ref.null any)))

          (func (export "br_on_cast") (param $i i32) (result i32)
            (block $l (result (ref i31))
              (br_on_cast $l anyref (ref i31) (table.get (local.get $i)))
              (return (i32.const -1)))
            (i31.get_u))

          (func (export "br_on_cast_fail") (param $i i32) (result i32)
            (block $l (result anyref)
              (br_on_cast_fail $l anyref (ref i31) (table.get (local.get $i)))
              (return (i31.get_u)))
            (return (i32.const -1))))
        "#,
    )
    .expect("compile wat");

    let mut runner = WastTestRunner::new(engine_for(Tier::Jit));
    let id = runner
        .try_instantiate_temp(&wasm_bytes)
        .expect("instantiate temp module");

    runner.world.invoke(id, "init", &[]).expect("invoke init");

    let cast_hit = runner
        .world
        .invoke(id, "br_on_cast", &[Value::I32(0)])
        .expect("invoke br_on_cast success");
    assert_eq!(cast_hit.as_slice(), &[Value::I32(7)]);

    let cast_miss = runner
        .world
        .invoke(id, "br_on_cast", &[Value::I32(1)])
        .expect("invoke br_on_cast failure");
    assert_eq!(cast_miss.as_slice(), &[Value::I32(-1)]);

    let cast_fail_miss = runner
        .world
        .invoke(id, "br_on_cast_fail", &[Value::I32(0)])
        .expect("invoke br_on_cast_fail success");
    assert_eq!(cast_fail_miss.as_slice(), &[Value::I32(7)]);

    let cast_fail_hit = runner
        .world
        .invoke(id, "br_on_cast_fail", &[Value::I32(1)])
        .expect("invoke br_on_cast_fail branch");
    assert_eq!(cast_fail_hit.as_slice(), &[Value::I32(-1)]);
}

#[test]
fn native_regress_struct_new_and_struct_get() {
    let wasm_bytes = wat::parse_str(
        r#"
        (module
          (type $st (struct (field i16)))

          (func (export "make_field") (result i32)
            i32.const 6
            struct.new $st
            struct.get_s $st 0)
        )
        "#,
    )
    .expect("compile wat");

    let mut runner = WastTestRunner::new(engine_for(Tier::Jit));
    let id = runner
        .try_instantiate_temp(&wasm_bytes)
        .expect("instantiate temp module");

    let field = runner
        .world
        .invoke(id, "make_field", &[])
        .expect("invoke make_field");
    assert_eq!(field.as_slice(), &[Value::I32(6)]);
}

#[test]
fn native_regress_array_set_and_get() {
    let wasm_bytes = wat::parse_str(
        r#"
        (module
          (type $arr (array (mut i32)))

          (func (export "write_then_read") (result i32)
            (local $tmp (ref $arr))
            i32.const 2
            array.new_default $arr
            local.set $tmp
            local.get $tmp
            i32.const 1
            i32.const 7
            array.set $arr
            local.get $tmp
            i32.const 1
            array.get $arr)
        )
        "#,
    )
    .expect("compile wat");

    let mut runner = WastTestRunner::new(engine_for(Tier::Jit));
    let id = runner
        .try_instantiate_temp(&wasm_bytes)
        .expect("instantiate temp module");

    let value = runner
        .world
        .invoke(id, "write_then_read", &[])
        .expect("invoke write_then_read");
    assert_eq!(value.as_slice(), &[Value::I32(7)]);
}

#[test]
fn native_regress_array_new_fixed_fill_and_copy() {
    let wasm_bytes = wat::parse_str(
        r#"
        (module
          (type $arr (array (mut i32)))

          (func (export "fixed_sum") (result i32)
            (local $tmp (ref $arr))
            i32.const 4
            i32.const 5
            i32.const 6
            array.new_fixed $arr 3
            local.set $tmp
            local.get $tmp
            i32.const 0
            array.get $arr
            local.get $tmp
            i32.const 1
            array.get $arr
            i32.add
            local.get $tmp
            i32.const 2
            array.get $arr
            i32.add)

          (func (export "fill_copy") (result i32)
            (local $src (ref $arr))
            (local $dst (ref $arr))
            i32.const 3
            array.new_default $arr
            local.set $src
            local.get $src
            i32.const 0
            i32.const 9
            i32.const 3
            array.fill $arr
            i32.const 3
            array.new_default $arr
            local.set $dst
            local.get $dst
            i32.const 0
            local.get $src
            i32.const 0
            i32.const 3
            array.copy $arr $arr
            local.get $dst
            i32.const 2
            array.get $arr)
        )
        "#,
    )
    .expect("compile wat");

    let mut runner = WastTestRunner::new(engine_for(Tier::Jit));
    let id = runner
        .try_instantiate_temp(&wasm_bytes)
        .expect("instantiate temp module");

    let fixed_sum = runner
        .world
        .invoke(id, "fixed_sum", &[])
        .expect("invoke fixed_sum");
    assert_eq!(fixed_sum.as_slice(), &[Value::I32(15)]);

    let copied = runner
        .world
        .invoke(id, "fill_copy", &[])
        .expect("invoke fill_copy");
    assert_eq!(copied.as_slice(), &[Value::I32(9)]);
}

#[test]
fn native_regress_array_new_fixed_const_expr() {
    let wasm_bytes = wat::parse_str(
        r#"
        (module
          (type $arr (array (mut i32)))
          (global $g (ref $arr)
            i32.const 3
            i32.const 4
            array.new_fixed $arr 2)

          (func (export "read") (result i32)
            global.get $g
            i32.const 1
            array.get $arr)
        )
        "#,
    )
    .expect("compile wat");

    let mut runner = WastTestRunner::new(engine_for(Tier::Jit));
    let id = runner
        .try_instantiate_temp(&wasm_bytes)
        .expect("instantiate temp module");

    let value = runner.world.invoke(id, "read", &[]).expect("invoke read");
    assert_eq!(value.as_slice(), &[Value::I32(4)]);
}

#[test]
fn native_regress_array_new_data_and_init_data() {
    let wasm_bytes = wat::parse_str(
        r#"
        (module
          (type $arr (array (mut i8)))
          (data "ABCD")

          (func (export "new_data_second") (result i32)
            i32.const 0
            i32.const 4
            array.new_data $arr 0
            i32.const 1
            array.get_s $arr)

          (func (export "init_data_third") (result i32)
            (local $tmp (ref $arr))
            i32.const 4
            array.new_default $arr
            local.set $tmp
            local.get $tmp
            i32.const 1
            i32.const 1
            i32.const 2
            array.init_data $arr 0
            local.get $tmp
            i32.const 2
            array.get_s $arr)
        )
        "#,
    )
    .expect("compile wat");

    let mut runner = WastTestRunner::new(engine_for(Tier::Jit));
    let id = runner
        .try_instantiate_temp(&wasm_bytes)
        .expect("instantiate temp module");

    let new_data = runner
        .world
        .invoke(id, "new_data_second", &[])
        .expect("invoke new_data_second");
    assert_eq!(new_data.as_slice(), &[Value::I32(i32::from(b'B'))]);

    let init_data = runner
        .world
        .invoke(id, "init_data_third", &[])
        .expect("invoke init_data_third");
    assert_eq!(init_data.as_slice(), &[Value::I32(i32::from(b'C'))]);
}

#[test]
fn native_regress_array_new_elem_and_init_elem() {
    let wasm_bytes = wat::parse_str(
        r#"
        (module
          (type $arr (array (mut funcref)))
          (func $f0)
          (func $f1)
          (elem func $f0 $f1)

          (func (export "new_elem_non_null") (result i32)
            (local $tmp (ref $arr))
            i32.const 0
            i32.const 2
            array.new_elem $arr 0
            local.set $tmp
            local.get $tmp
            i32.const 1
            array.get $arr
            ref.is_null
            i32.eqz)

          (func (export "init_elem_non_null") (result i32)
            (local $tmp (ref $arr))
            i32.const 2
            array.new_default $arr
            local.set $tmp
            local.get $tmp
            i32.const 0
            i32.const 0
            i32.const 2
            array.init_elem $arr 0
            local.get $tmp
            i32.const 0
            array.get $arr
            ref.is_null
            i32.eqz)
        )
        "#,
    )
    .expect("compile wat");

    let mut runner = WastTestRunner::new(engine_for(Tier::Jit));
    let id = runner
        .try_instantiate_temp(&wasm_bytes)
        .expect("instantiate temp module");

    let new_elem = runner
        .world
        .invoke(id, "new_elem_non_null", &[])
        .expect("invoke new_elem_non_null");
    assert_eq!(new_elem.as_slice(), &[Value::I32(1)]);

    let init_elem = runner
        .world
        .invoke(id, "init_elem_non_null", &[])
        .expect("invoke init_elem_non_null");
    assert_eq!(init_elem.as_slice(), &[Value::I32(1)]);
}

#[test]
fn native_cross_instance_gc_array_funcref_call_preserves_identity() {
    let mut runner = WastTestRunner::new(engine_for(Tier::Jit));
    runner
        .execute_wast_content(include_str!("../../../tests/runtime_world_gc_funcref.wast"))
        .expect("cross-instance GC funcref array");
}
