use sf_nano_core::{
    Config, Engine, Import, InstanceId, Module, RuntimeWorld, Tier, Value, WorldAccess,
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

fn instantiate(
    world: &mut RuntimeWorld,
    engine: &Engine,
    text: &str,
    imports: &[Import],
) -> InstanceId {
    let wasm = wat::parse_str(text).unwrap();
    world
        .instantiate(engine, Module::new("memory", &wasm).unwrap(), imports)
        .unwrap()
}

#[test]
fn a_memory_view_blocks_calls_through_a_cloned_world_handle() {
    let wasm = wat::parse_str("(module (memory 1) (func (export \"noop\")))").unwrap();
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let mut world = RuntimeWorld::new();
        let id = world
            .instantiate(&engine, Module::new("memory", &wasm).unwrap(), &[])
            .unwrap();
        let handle = world.handle();
        let memory = world.instance(id).unwrap().memory().unwrap();
        assert!(handle.invoke(id, 0, &[]).is_err(), "{tier:?}");
        assert_eq!(memory.len(), 65536);
    }
}

#[test]
fn shared_readers_block_every_call_path_and_initialization_until_dropped() {
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let mut world = RuntimeWorld::new();
        let source = instantiate(
            &mut world,
            &engine,
            r#"(module
            (memory (export "memory") 1 2)
            (func (export "grow") (result i32) i32.const 1 memory.grow))"#,
            &[],
        );
        let memory_import = Import::new(
            "source",
            "memory",
            world
                .instance(source)
                .unwrap()
                .get_export("memory")
                .unwrap()
                .unwrap(),
        );
        let peer = instantiate(
            &mut world,
            &engine,
            r#"(module
            (import "source" "memory" (memory 1 2))
            (func (export "noop")))"#,
            &[memory_import.clone()],
        );
        let handle = world.handle();
        let func = world.instance(peer).unwrap().get_func("noop").unwrap();
        let first = world.instance(source).unwrap().memory().unwrap();
        let second = world.instance(peer).unwrap().memory().unwrap();
        assert!(world.instance_mut(peer).unwrap().memory_mut().is_err());
        assert!(world
            .instance_mut(peer)
            .unwrap()
            .call(&func, &[], &mut [])
            .is_err());
        assert!(world
            .instance_mut(peer)
            .unwrap()
            .invoke_function_index(0, &[])
            .is_err());
        assert!(world
            .instance_mut(peer)
            .unwrap()
            .invoke("noop", &[])
            .is_err());
        assert!(world.invoke(peer, "noop", &[]).is_err());
        let initializer = wat::parse_str(
            r#"(module
            (import "source" "memory" (memory 1 2))
            (data (i32.const 0) "changed"))"#,
        )
        .unwrap();
        assert!(world
            .instantiate(
                &engine,
                Module::new("initializer", &initializer).unwrap(),
                &[memory_import]
            )
            .is_err());
        assert_eq!(first[0], 0);
        drop(first);
        assert!(handle.invoke(peer, 0, &[]).is_err());
        drop(second);
        handle.invoke(peer, 0, &[]).unwrap();
        assert_eq!(world.invoke(source, "grow", &[]).unwrap(), [Value::I32(1)]);
        assert_eq!(
            world.instance(peer).unwrap().memory().unwrap().len(),
            2 * 65536
        );
    }
}

#[test]
fn mutable_views_are_exclusive_and_keep_the_backing_alive() {
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let mut world = RuntimeWorld::new();
        let id = instantiate(&mut world, &engine, "(module (memory 1))", &[]);
        let mut memory = world.instance_mut(id).unwrap().memory_mut().unwrap();
        memory[0] = 42;
        assert!(world.instance(id).unwrap().memory().is_err());
        assert!(world.instance_mut(id).unwrap().memory_mut().is_err());
        world.free(id).unwrap();
        drop(world);
        assert_eq!(memory[0], 42);
        memory[0] = 43;
        assert_eq!(memory[0], 43);
    }
}

#[test]
fn views_cannot_escape_from_a_callback_but_caller_memory_remains_usable() {
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let world = Rc::new(RefCell::new(RuntimeWorld::new()));
        let weak = Rc::downgrade(&world);
        let identity = Rc::new(Cell::new(None));
        let callback_id = Rc::clone(&identity);
        let import = Import::func("host", "check", move |caller, _, _| {
            let world = weak.upgrade().unwrap();
            let id = callback_id.get().unwrap();
            assert!(world.borrow().instance(id).unwrap().memory().is_err());
            assert!(world
                .borrow_mut()
                .instance_mut(id)
                .unwrap()
                .memory_mut()
                .is_err());
            caller.memory_mut().unwrap()[0] = 55;
            Ok(())
        });
        let id = instantiate(
            &mut world.borrow_mut(),
            &engine,
            r#"(module
            (import "host" "check" (func $check))
            (memory 1) (export "run" (func $check)))"#,
            &[import],
        );
        identity.set(Some(id));
        let access = world.borrow().handle();
        access.invoke(id, 0, &[]).unwrap();
        assert_eq!(
            world.borrow().instance(id).unwrap().memory().unwrap()[0],
            55
        );
    }
}

#[test]
fn caller_memory_borrow_rejects_reentry_to_an_imported_host_function() {
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let mut world = RuntimeWorld::new();
        let access = world.handle();
        let identity = Rc::new(Cell::new(None));
        let callback_id = Rc::clone(&identity);
        let outer = Import::func("host", "outer", move |caller, _, _| {
            let memory = caller.memory_mut().unwrap();
            assert!(access.invoke(callback_id.get().unwrap(), 1, &[]).is_err());
            memory[0] = 77;
            Ok(())
        });
        let calls = Rc::new(Cell::new(0));
        let called = Rc::clone(&calls);
        let inner = Import::func("host", "inner", move |_, _, _| {
            called.set(called.get() + 1);
            Ok(())
        });
        let id = instantiate(
            &mut world,
            &engine,
            r#"(module
            (import "host" "outer" (func $outer))
            (import "host" "inner" (func))
            (memory 1) (export "run" (func $outer)))"#,
            &[outer, inner],
        );
        identity.set(Some(id));
        world.invoke(id, "run", &[]).unwrap();
        assert_eq!(calls.get(), 0);
        assert_eq!(world.instance(id).unwrap().memory().unwrap()[0], 77);
    }
}

#[test]
fn a_callback_cannot_initialize_an_alias_of_its_borrowed_memory() {
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let world = Rc::new(RefCell::new(RuntimeWorld::new()));
        let weak = Rc::downgrade(&world);
        let imports = Rc::new(RefCell::new(Vec::new()));
        let memory_imports = Rc::clone(&imports);
        let initializer = wat::parse_str(
            r#"(module
            (import "source" "memory" (memory 1))
            (data (i32.const 0) "changed"))"#,
        )
        .unwrap();
        let host = Import::func("host", "init", move |caller, _, _| {
            let memory = caller.memory_mut().unwrap();
            memory[0] = 88;
            let world = weak.upgrade().unwrap();
            assert!(world
                .borrow_mut()
                .instantiate(
                    &engine,
                    Module::new("initializer", &initializer).unwrap(),
                    &memory_imports.borrow()
                )
                .is_err());
            assert_eq!(memory[0], 88);
            Ok(())
        });
        let id = instantiate(
            &mut world.borrow_mut(),
            &engine,
            r#"(module
            (import "host" "init" (func $init))
            (memory (export "memory") 1) (export "run" (func $init)))"#,
            &[host],
        );
        imports.borrow_mut().push(Import::new(
            "source",
            "memory",
            world
                .borrow()
                .instance(id)
                .unwrap()
                .get_export("memory")
                .unwrap()
                .unwrap(),
        ));
        let handle: WorldAccess = world.borrow().handle();
        handle.invoke(id, 0, &[]).unwrap();
        assert_eq!(
            world.borrow().instance(id).unwrap().memory().unwrap()[0],
            88
        );
    }
}

#[test]
fn execution_errors_release_the_world_for_later_memory_access() {
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let mut world = RuntimeWorld::new();
        let id = instantiate(
            &mut world,
            &engine,
            r#"(module
            (memory 1) (func (export "fail") unreachable))"#,
            &[],
        );
        assert!(world.invoke(id, "fail", &[]).is_err());
        assert!(world.instance_mut(id).unwrap().memory_mut().is_ok());
    }
}

#[test]
fn caller_borrow_blocks_a_linked_call_through_a_module_without_memory() {
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let mut world = RuntimeWorld::new();
        let access = world.handle();
        let peer_id = Rc::new(Cell::new(None));
        let callback_peer = Rc::clone(&peer_id);
        let host = Import::func("host", "check", move |caller, _, _| {
            let memory = caller.memory_mut().unwrap();
            assert!(access.invoke(callback_peer.get().unwrap(), 1, &[]).is_err());
            assert_eq!(memory[0], 0);
            Ok(())
        });
        let owner = instantiate(
            &mut world,
            &engine,
            r#"(module
            (import "host" "check" (func $check))
            (memory 1)
            (func (export "noop"))
            (export "run" (func $check)))"#,
            &[host],
        );
        let linked = Import::new(
            "owner",
            "noop",
            world
                .instance(owner)
                .unwrap()
                .get_export("noop")
                .unwrap()
                .unwrap(),
        );
        let peer = instantiate(
            &mut world,
            &engine,
            r#"(module
            (import "owner" "noop" (func $noop))
            (func (export "forward") call $noop))"#,
            &[linked],
        );
        peer_id.set(Some(peer));
        world.invoke(owner, "run", &[]).unwrap();
        world.invoke(peer, "forward", &[]).unwrap();
    }
}
