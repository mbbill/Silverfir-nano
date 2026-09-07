//! One WAST runner for every enabled execution engine.
mod discovery;
mod driver;
mod summary;
mod types;
mod wast_test_runner;

fn main() {
    driver::run()
}
