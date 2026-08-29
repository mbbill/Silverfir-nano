use sf_nano_core::module::Module;
use sf_nano_eager_tier_census::{analyze_module, render_markdown, ModuleCensus};
use std::{env, fs, path::PathBuf, process::ExitCode};

fn usage() -> &'static str {
    "usage: sf-nano-eager-tier-census --out-dir DIR name=module.wasm [...]"
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    if args.next().as_deref() != Some("--out-dir") {
        return Err(usage().to_owned());
    }
    let out_dir = PathBuf::from(args.next().ok_or_else(|| usage().to_owned())?);
    let inputs: Vec<String> = args.collect();
    if inputs.is_empty() {
        return Err(usage().to_owned());
    }
    fs::create_dir_all(&out_dir)
        .map_err(|error| format!("create {}: {error}", out_dir.display()))?;

    let mut reports: Vec<ModuleCensus> = Vec::with_capacity(inputs.len());
    for input in inputs {
        let (name, path) = input
            .split_once('=')
            .ok_or_else(|| format!("input must be name=path: {input}"))?;
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
        {
            return Err(format!("unsafe or empty module name: {name:?}"));
        }
        let bytes = fs::read(path).map_err(|error| format!("read {path}: {error}"))?;
        let module = Module::new(name, &bytes).map_err(|error| format!("parse {path}: {error}"))?;
        let report = analyze_module(&module).map_err(|error| format!("analyze {path}: {error}"))?;
        let json_path = out_dir.join(format!("{name}.json"));
        let json = serde_json::to_vec_pretty(&report)
            .map_err(|error| format!("serialize {name}: {error}"))?;
        fs::write(&json_path, json)
            .map_err(|error| format!("write {}: {error}", json_path.display()))?;
        reports.push(report);
    }

    let summary_json = serde_json::to_vec_pretty(&reports)
        .map_err(|error| format!("serialize summary: {error}"))?;
    fs::write(out_dir.join("summary.json"), summary_json)
        .map_err(|error| format!("write summary.json: {error}"))?;
    let markdown = render_markdown(&reports);
    fs::write(out_dir.join("summary.md"), &markdown)
        .map_err(|error| format!("write summary.md: {error}"))?;
    print!("{markdown}");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
