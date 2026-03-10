use alloc::string::String;

#[cfg(any(feature = "std", feature = "wasi", test))]
use std::{
    collections::HashSet,
    env,
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};

#[cfg(any(feature = "std", feature = "wasi", test))]
pub(crate) fn sanitize_token(raw: &str) -> String {
    raw.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' | '.' => c,
            _ => '_',
        })
        .collect()
}

#[cfg(any(feature = "std", feature = "wasi", test))]
fn dump_root() -> Option<PathBuf> {
    env::var_os("SF_DUMP_DIR").map(PathBuf::from)
}

#[cfg(any(feature = "std", feature = "wasi", test))]
pub(crate) fn enabled() -> bool {
    dump_root().is_some()
}

#[cfg(any(feature = "std", feature = "wasi", test))]
fn func_filter() -> &'static Option<HashSet<u32>> {
    static FILTER: OnceLock<Option<HashSet<u32>>> = OnceLock::new();
    FILTER.get_or_init(|| {
        env::var("SF_TRACE_FUNC").ok().map(|val| {
            val.split(',')
                .filter_map(|s| s.trim().parse::<u32>().ok())
                .collect()
        })
    })
}

#[cfg(any(feature = "std", feature = "wasi", test))]
pub(crate) fn should_dump_func(func_idx: u32) -> bool {
    match func_filter() {
        None => true,
        Some(set) => set.contains(&func_idx),
    }
}

#[cfg(any(feature = "std", feature = "wasi", test))]
pub(crate) fn function_dir(module_name: &str, func_idx: u32) -> Option<PathBuf> {
    let root = dump_root()?;
    let dir = root
        .join(sanitize_token(module_name))
        .join(alloc::format!("func_{func_idx:04}"));
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

#[cfg(any(feature = "std", feature = "wasi", test))]
pub(crate) fn subdir(module_name: &str, func_idx: u32, name: &str) -> Option<PathBuf> {
    let dir = function_dir(module_name, func_idx)?.join(name);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

#[cfg(any(feature = "std", feature = "wasi", test))]
pub(crate) fn write_text(path: &Path, text: &str) -> Option<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok()?;
    }
    fs::write(path, text).ok()?;
    Some(())
}

#[cfg(any(feature = "std", feature = "wasi", test))]
pub(crate) fn write_bytes(path: &Path, bytes: &[u8]) -> Option<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok()?;
    }
    fs::write(path, bytes).ok()?;
    Some(())
}
