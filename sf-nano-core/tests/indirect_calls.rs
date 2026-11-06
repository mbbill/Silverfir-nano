use sf_nano_core::Module;
use sf_nano_core::{Config, Engine, RuntimeWorld, Tier, Value};

fn i32s(values: Vec<Value>) -> Vec<i32> {
    values
        .into_iter()
        .map(|value| match value {
            Value::I32(v) => v,
            other => panic!("expected i32, got {other:?}"),
        })
        .collect()
}

/// Every dispatch shape a table can take: fixed and private, growable and
/// private, exported, and a 64-bit one whose index keeps all its bits.
fn dispatch_shapes() -> Vec<u8> {
    wat::parse_str(
        r#"(module
        (type $unary (func (param i32) (result i32)))
        (table $fixed 2 2 funcref)
        (table $growable 2 funcref)
        (table $shared (export "table") 2 funcref)
        (table $wide i64 2 funcref)
        (elem (table $fixed) (i32.const 0) func $double)
        (elem (table $growable) (i32.const 0) func $double)
        (elem (table $shared) (i32.const 0) func $double)
        (elem (table $wide) (i64.const 0) func $double)
        (func $double (type $unary) local.get 0 i32.const 2 i32.mul)
        (func (export "fixed") (param i32) (result i32)
            i32.const 1 local.get 0 call_indirect $fixed (type $unary))
        (func (export "growable") (param i32) (result i32)
            i32.const 1 local.get 0 call_indirect $growable (type $unary))
        (func (export "shared") (param i32) (result i32)
            i32.const 1 local.get 0 call_indirect $shared (type $unary))
        (func (export "wide") (param i64) (result i32)
            i32.const 1 local.get 0 call_indirect $wide (type $unary)))"#,
    )
    .unwrap()
}

#[test]
fn indirect_call_traps_carry_the_spec_message_in_every_dispatch_shape() {
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let mut world = RuntimeWorld::new();
        let module = Module::new("shapes", &dispatch_shapes()).unwrap();
        let instance = world.instantiate(&engine, module, &[]).unwrap();
        for export in ["fixed", "growable", "shared"] {
            let call = |world: &mut RuntimeWorld, slot: i32| {
                world.invoke(instance, export, &[Value::I32(slot)])
            };
            assert_eq!(i32s(call(&mut world, 0).unwrap()), [2], "{tier:?} {export}");
            let null = call(&mut world, 1).expect_err("null slot");
            assert_eq!(null.message(), "uninitialized element", "{tier:?} {export}");
            let past_end = call(&mut world, 2).expect_err("index past the end");
            assert_eq!(past_end.message(), "undefined element", "{tier:?} {export}");
        }
        // The low 32 bits of each of these name slot 0 or the null slot 1.
        for index in [1i64 << 32, (1i64 << 32) + 1, -1] {
            let err = world
                .invoke(instance, "wide", &[Value::I64(index)])
                .expect_err("64-bit index out of bounds");
            assert_eq!(err.message(), "undefined element", "{tier:?} {index:#x}");
        }
        assert_eq!(
            i32s(world.invoke(instance, "wide", &[Value::I64(0)]).unwrap()),
            [2],
            "{tier:?}"
        );
    }
}
