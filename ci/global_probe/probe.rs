use sf_nano_core::{Config, Engine, Instance, Tier, Value};
use std::time::Instant;
fn main() {
    let engine = Engine::new(Config::new().tier(Tier::Interp)).unwrap();
    let mut instance = Instance::new(&engine, include_bytes!("counter-global.wasm"), &[]).unwrap();
    let func = instance.get_func("run").unwrap();
    let mut result = [Value::I32(-1)];
    for _ in 0..10 { instance.call(&func, &[Value::I32(500000)], &mut result).unwrap(); }
    let start = Instant::now();
    for _ in 0..200 {
        instance.call(&func, &[Value::I32(500000)], &mut result).unwrap();
        assert_eq!(result, [Value::I32(0)]);
    }
    println!("{}", start.elapsed().as_secs_f64());
}
