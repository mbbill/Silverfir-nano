use sf_nano_core::{Config, Engine, Instance, Module, RuntimeWorld, Tier, Value};

#[test]
fn safe_loading_rejects_invalid_code_before_instantiation() {
    let cases = [
        ("wrong result", "(module (func (result i32) f32.const 1))"),
        ("missing operand", "(module (func i32.add drop))"),
        ("unknown local", "(module (func local.get 0 drop))"),
        ("final supertype", "(module (type $f (func)) (type (sub $f (func))))"),
        ("function equality", "(module (type $f (func)) (func (param (ref null $f)) local.get 0 local.get 0 ref.eq drop))"),
        ("function equality after bottom", "(module (type $f (func)) (func (param (ref null $f)) unreachable local.get 0 ref.eq drop))"),
        ("invalid constant", "(module (global i32 (f32.const 0)))"),
    ];
    for (name, wat) in cases {
        // WAT encoding deliberately does not establish semantic validity.
        let bytes = wat::parse_str(wat).expect("encodable fixture");
        assert!(Module::new(name, &bytes).is_err(), "{name}");
        for &tier in Tier::ALL {
            let engine = Engine::new(Config::new().tier(tier)).unwrap();
            assert!(
                Instance::new(&engine, &bytes, &[]).is_err(),
                "{tier:?}: {name}"
            );
        }
    }
}

#[test]
fn reference_equality_accepts_gc_types_and_polymorphic_stack() {
    for wat in [
        "(module (type $s (struct)) (func (param (ref null $s)) (result i32) local.get 0 local.get 0 ref.eq))",
        "(module (type $a (array i32)) (func (param (ref null $a)) (result i32) local.get 0 local.get 0 ref.eq))",
        "(module (func (result i32) unreachable ref.eq))",
    ] {
        let bytes = wat::parse_str(wat).unwrap();
        Module::new("valid equality", &bytes).expect("valid reference equality");
    }
}

#[test]
fn prevalidated_loading_owns_input_and_keeps_link_checks() {
    let mut bytes = wat::parse_str(
        r#"(module
            (import "host" "required" (func))
            (func (export "answer") (result i32) i32.const 42))"#,
    )
    .unwrap();
    Module::new("validate", &bytes).expect("establish validity of these exact bytes");
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        // SAFETY: Module::new validated these same bytes above. They have not
        // been modified, and semantic validity does not depend on the engine.
        let module = unsafe { Module::new_unchecked("trusted", &bytes) }.unwrap();
        let mut world = RuntimeWorld::new();
        let error = world.instantiate(&engine, module, &[]).unwrap_err();
        assert!(error.error().is_unlinkable());
    }

    bytes =
        wat::parse_str("(module (func (export \"answer\") (result i32) i32.const 42))").unwrap();
    let module = Module::new("owned input", &bytes).unwrap();
    bytes.fill(0);
    drop(bytes);
    let mut instance = Instance::from_module(&Engine::with_defaults(), module, &[]).unwrap();
    assert_eq!(
        instance.invoke("answer", &[]).unwrap(),
        vec![Value::I32(42)]
    );
}
