use sf_nano_core::Module;
use sf_nano_core::{Config, Engine, Import, RuntimeWorld, Tier, Value};

/// Owns an exported table: its own functions, a mistyped one, and a null slot.
fn owner() -> Vec<u8> {
    wat::parse_str(
        r#"(module
        (type $unary (func (param i32) (result i32)))
        (type $unit (func))
        (table (export "table") 4 8 funcref)
        (elem (i32.const 0) $double $square $nop)
        (func $double (type $unary) local.get 0 i32.const 2 i32.mul)
        (func $square (type $unary) local.get 0 local.get 0 i32.mul)
        (func $nop (type $unit))
        (func (export "call") (param i32 i32) (result i32)
            local.get 1 local.get 0 call_indirect (type $unary))
        (func (export "size") (result i32) table.size))"#,
    )
    .unwrap()
}

/// Imports the owner's table and writes its own function into it.
fn peer() -> Vec<u8> {
    wat::parse_str(
        r#"(module
        (type $unary (func (param i32) (result i32)))
        (import "owner" "table" (table 4 8 funcref))
        (func $triple (type $unary) local.get 0 i32.const 3 i32.mul)
        (elem declare func $triple)
        (func (export "install") (param i32)
            local.get 0 ref.func $triple table.set)
        (func (export "grow") (result i32)
            ref.func $triple i32.const 4 table.grow)
        (func (export "call") (param i32 i32) (result i32)
            local.get 1 local.get 0 call_indirect (type $unary)))"#,
    )
    .unwrap()
}

fn i32s(values: Vec<Value>) -> Vec<i32> {
    values
        .into_iter()
        .map(|value| match value {
            Value::I32(v) => v,
            other => panic!("expected i32, got {other:?}"),
        })
        .collect()
}

#[test]
fn indirect_calls_through_a_shared_table_see_every_writer() {
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let mut world = RuntimeWorld::new();
        let owner = world
            .instantiate(&engine, Module::new("owner", &owner()).unwrap(), &[])
            .unwrap();
        let call = |world: &mut RuntimeWorld, instance, slot: i32, arg: i32| {
            world
                .invoke(instance, "call", &[Value::I32(slot), Value::I32(arg)])
                .map(i32s)
        };

        // The owner's own functions, then each guard failure, which every
        // engine reports with the spec's message.
        assert_eq!(call(&mut world, owner, 0, 21).unwrap(), [42], "{tier:?}");
        assert_eq!(call(&mut world, owner, 1, 7).unwrap(), [49], "{tier:?}");
        for (slot, message) in [
            (2, "indirect call type mismatch"),
            (3, "uninitialized element"),
            (4, "undefined element"),
        ] {
            let err = call(&mut world, owner, slot, 1).expect_err(message);
            assert_eq!(err.message(), message, "{tier:?} slot {slot}");
        }

        let imports: Vec<_> = world
            .instance(owner)
            .unwrap()
            .exports()
            .unwrap()
            .into_iter()
            .map(|(name, value)| Import::new("owner", &name, value))
            .collect();
        let peer = world
            .instantiate(&engine, Module::new("peer", &peer()).unwrap(), &imports)
            .unwrap_or_else(|e| panic!("{tier:?}: {e:?}"));

        // The peer reaches the owner's function through the table, and a
        // function the peer writes is visible to both sides.
        assert_eq!(call(&mut world, peer, 0, 21).unwrap(), [42], "{tier:?}");
        world.invoke(peer, "install", &[Value::I32(1)]).unwrap();
        assert_eq!(call(&mut world, owner, 1, 7).unwrap(), [21], "{tier:?}");
        assert_eq!(call(&mut world, peer, 1, 7).unwrap(), [21], "{tier:?}");

        // A grow by the peer moves the elements; the owner sees the new
        // length and slots, and its own functions still resolve.
        let grown = world.invoke(peer, "grow", &[]).map(i32s).unwrap();
        assert_eq!(grown, [4], "{tier:?}");
        let size = world.invoke(owner, "size", &[]).map(i32s).unwrap();
        assert_eq!(size, [8], "{tier:?}");
        for slot in 4..8 {
            assert_eq!(
                call(&mut world, owner, slot, 5).unwrap(),
                [15],
                "{tier:?} slot {slot}"
            );
        }
        assert_eq!(call(&mut world, owner, 0, 21).unwrap(), [42], "{tier:?}");
    }
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
