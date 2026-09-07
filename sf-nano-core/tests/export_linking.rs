use sf_nano_core::Module;
use sf_nano_core::{Config, Engine, Import, Instance, RuntimeWorld, Tier, Value};

fn source() -> Vec<u8> {
    wat::parse_str(
        r#"(module
        (import "host" "plus" (func $plus (param i32) (result i32)))
        (export "forward" (func $plus))
        (func $answer (export "answer") (result i32) i32.const 42)
        (memory (export "memory") 1 3)
        (table (export "table") 1 4 funcref)
        (elem (i32.const 0) $answer)
        (global $g (export "counter") (mut i32) (i32.const 5))
        (tag (export "tag") (param i32))
        (func (export "grow")
            i32.const 1 memory.grow drop
            ref.null func i32.const 1 table.grow drop)
        (func (export "read") (result i32) global.get $g))"#,
    )
    .unwrap()
}

fn target() -> Vec<u8> {
    wat::parse_str(
        r#"(module
        (type $answer_type (func (result i32)))
        (import "p" "forward" (func $forward (param i32) (result i32)))
        (import "p" "memory" (memory 2 3))
        (import "p" "table" (table 2 4 funcref))
        (import "p" "counter" (global $g (mut i32)))
        (import "p" "tag" (tag (param i32)))
        (export "tag" (tag 0))
        (func (export "forward") (param i32) (result i32) local.get 0 call $forward)
        (func (export "indirect") (result i32) i32.const 0 call_indirect (type $answer_type))
        (func (export "write") (param i32) local.get 0 global.set $g)
        (func (export "read") (result i32) global.get $g))"#,
    )
    .unwrap()
}

#[test]
fn opaque_exports_preserve_live_storage_types_and_host_reexports() {
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let mut world = RuntimeWorld::new();
        let host = Import::func("host", "plus", |_, args, out| {
            out[0] = match args[0] {
                Value::I32(v) => Value::I32(v + 1),
                _ => unreachable!(),
            };
            Ok(())
        });
        let src = world
            .instantiate(&engine, Module::new("source", &source()).unwrap(), &[host])
            .unwrap();
        assert!(world
            .instance(src)
            .unwrap()
            .get_export("missing")
            .unwrap()
            .is_none());
        // Capture before grow: linking must read current sizes from the live
        // objects, not stale minimums saved when the export was requested.
        let imports: Vec<_> = world
            .instance(src)
            .unwrap()
            .exports()
            .unwrap()
            .into_iter()
            .map(|(name, value)| Import::new("p", &name, value))
            .collect();
        world.invoke(src, "grow", &[]).unwrap();
        let dst = world
            .instantiate(&engine, Module::new("target", &target()).unwrap(), &imports)
            .unwrap_or_else(|e| panic!("{tier:?}: {e:?}"));
        assert_eq!(
            world.invoke(dst, "forward", &[Value::I32(9)]).unwrap(),
            [Value::I32(10)]
        );
        assert_eq!(
            world.invoke(dst, "indirect", &[]).unwrap(),
            [Value::I32(42)]
        );
        world.invoke(dst, "write", &[Value::I32(17)]).unwrap();
        assert_eq!(world.invoke(src, "read", &[]).unwrap(), [Value::I32(17)]);
        world.instance_mut(dst).unwrap().memory_mut().unwrap()[0] = 23;
        assert_eq!(world.instance(src).unwrap().memory().unwrap()[0], 23);
        assert_eq!(
            world.instance(src).unwrap().tag_identity("tag"),
            world.instance(dst).unwrap().tag_identity("tag")
        );
        world.free(src).unwrap();
        assert_eq!(world.invoke(dst, "read", &[]).unwrap(), [Value::I32(17)]);
        assert!(
            world.invoke(dst, "forward", &[Value::I32(9)]).is_err(),
            "freed function owner must not be called"
        );
    }
}

#[test]
fn exports_cannot_be_imported_into_an_unrelated_world() {
    let memory = wat::parse_str("(module (memory (export \"memory\") 1))").unwrap();
    let consumer = wat::parse_str("(module (import \"p\" \"memory\" (memory 1)))").unwrap();
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let source = Instance::new(&engine, &memory, &[]).unwrap();
        let exported = source.get_export("memory").unwrap().unwrap();
        let imports = [Import::new("p", "memory", exported)];
        assert!(matches!(Instance::new(&engine, &consumer, &imports), Err(e) if e.is_unlinkable()));
    }
}

#[test]
fn reference_container_exports_do_not_confuse_equal_numeric_type_indices() {
    let source = wat::parse_str(
        r#"(module
        (type $f (func (result i32)))
        (table (export "table") 1 (ref null $f))
        (global (export "global") (mut (ref null $f)) (ref.null $f)))"#,
    )
    .unwrap();
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let mut world = RuntimeWorld::new();
        let src = world
            .instantiate(&engine, Module::new("source", &source).unwrap(), &[])
            .unwrap();
        for declaration in ["(table 1 (ref null $f))", "(global (mut (ref null $f)))"] {
            let name = if declaration.starts_with("(table") {
                "table"
            } else {
                "global"
            };
            let wasm = wat::parse_str(format!(
                "(module (type $f (func (result i64))) (import \"p\" \"{name}\" {declaration}))"
            ))
            .unwrap();
            let value = world
                .instance(src)
                .unwrap()
                .get_export(name)
                .unwrap()
                .unwrap();
            let result = world.instantiate(
                &engine,
                Module::new("target", &wasm).unwrap(),
                &[Import::new("p", name, value)],
            );
            assert!(
                matches!(result.map_err(|e| e.into_parts().1), Err(e) if e.is_unlinkable()),
                "{tier:?}: {name} type index 0 means different function types in the two modules"
            );
            let matching = wat::parse_str(format!(
                "(module (type (func (param i64))) (type $f (func (result i32))) (import \"p\" \"{name}\" {declaration}))"
            )).unwrap();
            let value = world
                .instance(src)
                .unwrap()
                .get_export(name)
                .unwrap()
                .unwrap();
            assert!(
                world
                    .instantiate(
                        &engine,
                        Module::new("matching", &matching).unwrap(),
                        &[Import::new("p", name, value)]
                    )
                    .is_ok(),
                "{tier:?}: matching {name} types at different indices must link"
            );
        }
    }
}
