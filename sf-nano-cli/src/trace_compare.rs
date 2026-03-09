use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::process;

#[derive(Debug, Clone, PartialEq, Eq)]
struct TraceHeader {
    version: String,
    backend: String,
    memory_enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TraceEvent {
    seq: u64,
    kind: String,
    depth: u32,
    func_idx: u32,
    results: String,
    globals_hash: String,
    memory_hash: String,
    error: String,
}

#[derive(Debug)]
struct ParsedTrace {
    header: TraceHeader,
    events: Vec<TraceEvent>,
}

#[derive(Debug, Clone, Copy, Default)]
struct CompareOptions {
    ignore_results: bool,
}

pub fn run_from_args(args: &[String]) -> ! {
    let mut ignore_results = false;
    let mut paths = Vec::new();
    for arg in args {
        if arg == "--ignore-results" {
            ignore_results = true;
        } else {
            paths.push(arg.clone());
        }
    }

    if paths.len() != 2 {
        eprintln!("USAGE:");
        eprintln!("  sf-nano-cli trace-compare [--ignore-results] <left.trace> <right.trace>");
        process::exit(1);
    }

    let left = load_trace(&paths[0]).unwrap_or_else(|err| {
        eprintln!("Error reading '{}': {}", paths[0], err);
        process::exit(1);
    });
    let right = load_trace(&paths[1]).unwrap_or_else(|err| {
        eprintln!("Error reading '{}': {}", paths[1], err);
        process::exit(1);
    });

    let options = CompareOptions { ignore_results };

    match compare_traces(&left, &right, &options) {
        Ok(()) => {
            println!(
                "match: {} events (left backend={}, right backend={}, ignore_results={})",
                left.events.len(),
                left.header.backend,
                right.header.backend,
                if ignore_results { 1 } else { 0 }
            );
            process::exit(0);
        }
        Err(report) => {
            eprintln!("{}", report);
            process::exit(2);
        }
    }
}

fn load_trace(path: &str) -> Result<ParsedTrace, String> {
    let content = fs::read_to_string(Path::new(path))
        .map_err(|err| err.to_string())?;
    parse_trace(&content)
}

fn parse_trace(content: &str) -> Result<ParsedTrace, String> {
    let mut lines = content.lines();
    let header_line = lines.next().ok_or_else(|| "empty trace".to_string())?;
    let header = parse_header(header_line)?;
    let mut events = Vec::new();
    for (line_no, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        events.push(parse_event(line, line_no + 2)?);
    }
    Ok(ParsedTrace { header, events })
}

fn parse_header(line: &str) -> Result<TraceHeader, String> {
    let mut parts = line.split_whitespace();
    let marker = parts.next().ok_or_else(|| "missing header marker".to_string())?;
    if marker != "#" {
        return Err(format!("invalid header marker '{}'", marker));
    }
    let version = parts
        .next()
        .ok_or_else(|| "missing trace version".to_string())?
        .to_string();
    let mut backend = None;
    let mut memory_enabled = None;
    for part in parts {
        if let Some(value) = part.strip_prefix("backend=") {
            backend = Some(value.to_string());
        } else if let Some(value) = part.strip_prefix("memory=") {
            memory_enabled = Some(match value {
                "0" => false,
                "1" => true,
                other => return Err(format!("invalid memory flag '{}'", other)),
            });
        }
    }
    Ok(TraceHeader {
        version,
        backend: backend.ok_or_else(|| "missing backend in header".to_string())?,
        memory_enabled: memory_enabled.ok_or_else(|| "missing memory flag in header".to_string())?,
    })
}

fn parse_event(line: &str, line_no: usize) -> Result<TraceEvent, String> {
    let parts: Vec<&str> = line.split('\t').collect();
    if parts.len() != 10 {
        return Err(format!(
            "line {}: expected 10 tab-separated fields, got {}",
            line_no,
            parts.len()
        ));
    }
    if parts[0] != "T1" {
        return Err(format!("line {}: invalid event marker '{}'", line_no, parts[0]));
    }
    Ok(TraceEvent {
        seq: parts[1]
            .parse()
            .map_err(|_| format!("line {}: invalid sequence '{}'", line_no, parts[1]))?,
        kind: parts[3].to_string(),
        depth: parts[4]
            .parse()
            .map_err(|_| format!("line {}: invalid depth '{}'", line_no, parts[4]))?,
        func_idx: parts[5]
            .parse()
            .map_err(|_| format!("line {}: invalid func idx '{}'", line_no, parts[5]))?,
        results: parts[6].to_string(),
        globals_hash: parts[7].to_string(),
        memory_hash: parts[8].to_string(),
        error: parts[9].to_string(),
    })
}

fn compare_traces(
    left: &ParsedTrace,
    right: &ParsedTrace,
    options: &CompareOptions,
) -> Result<(), String> {
    if left.header.version != right.header.version {
        return Err(format!(
            "trace version mismatch: left={}, right={}",
            left.header.version, right.header.version
        ));
    }
    if left.header.memory_enabled != right.header.memory_enabled {
        return Err(format!(
            "trace memory mode mismatch: left={}, right={}",
            left.header.memory_enabled as u8,
            right.header.memory_enabled as u8
        ));
    }

    let min_len = left.events.len().min(right.events.len());
    for idx in 0..min_len {
        let lhs = &left.events[idx];
        let rhs = &right.events[idx];
        if !events_equivalent(lhs, rhs, options) {
            return Err(render_mismatch(
                idx,
                Some(lhs),
                Some(rhs),
                &left.header.backend,
                &right.header.backend,
            ));
        }
    }

    if left.events.len() != right.events.len() {
        return Err(render_mismatch(
            min_len,
            left.events.get(min_len),
            right.events.get(min_len),
            &left.header.backend,
            &right.header.backend,
        ));
    }

    Ok(())
}

fn events_equivalent(left: &TraceEvent, right: &TraceEvent, options: &CompareOptions) -> bool {
    left.kind == right.kind
        && left.depth == right.depth
        && left.func_idx == right.func_idx
        && (options.ignore_results || left.results == right.results)
        && left.globals_hash == right.globals_hash
        && left.memory_hash == right.memory_hash
        && left.error == right.error
}

fn render_mismatch(
    idx: usize,
    left: Option<&TraceEvent>,
    right: Option<&TraceEvent>,
    left_backend: &str,
    right_backend: &str,
) -> String {
    let mut out = String::new();
    let _ = writeln!(&mut out, "trace mismatch at event {}", idx);
    let _ = writeln!(&mut out, "left backend: {}", left_backend);
    render_side(&mut out, "left", left);
    let _ = writeln!(&mut out, "right backend: {}", right_backend);
    render_side(&mut out, "right", right);
    out
}

fn render_side(out: &mut String, label: &str, event: Option<&TraceEvent>) {
    match event {
        Some(event) => {
            let _ = writeln!(
                out,
                "  {}: seq={} kind={} depth={} func={} results={} globals={} memory={} error={}",
                label,
                event.seq,
                event.kind,
                event.depth,
                event.func_idx,
                event.results,
                event.globals_hash,
                event.memory_hash,
                event.error
            );
        }
        None => {
            let _ = writeln!(out, "  {}: <end-of-trace>", label);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{compare_traces, parse_trace, CompareOptions};

    #[test]
    fn compare_ignores_backend_and_sequence() {
        let left = parse_trace(
            "# sf-nano-function-trace-v1 backend=base memory=0\nT1\t0\tfast\tentry\t0\t7\t-\t0000000000000001\t-\t-\n",
        )
        .unwrap();
        let right = parse_trace(
            "# sf-nano-function-trace-v1 backend=native memory=0\nT1\t99\tnative\tentry\t0\t7\t-\t0000000000000001\t-\t-\n",
        )
        .unwrap();
        assert!(compare_traces(&left, &right, &CompareOptions::default()).is_ok());
    }

    #[test]
    fn compare_reports_first_mismatch() {
        let left = parse_trace(
            "# sf-nano-function-trace-v1 backend=base memory=1\nT1\t0\tfast\texit\t1\t3\t000000000000000a\t0000000000000001\t0000000000000002\t-\n",
        )
        .unwrap();
        let right = parse_trace(
            "# sf-nano-function-trace-v1 backend=native memory=1\nT1\t0\tnative\texit\t1\t3\t000000000000000b\t0000000000000001\t0000000000000002\t-\n",
        )
        .unwrap();
        let err = compare_traces(&left, &right, &CompareOptions::default()).unwrap_err();
        assert!(err.contains("trace mismatch at event 0"));
        assert!(err.contains("results=000000000000000a"));
        assert!(err.contains("results=000000000000000b"));
    }

    #[test]
    fn compare_can_ignore_result_mismatches() {
        let left = parse_trace(
            "# sf-nano-function-trace-v1 backend=base memory=0\nT1\t0\tfast\texit\t1\t3\t000000000000000a\t0000000000000001\t-\t-\n",
        )
        .unwrap();
        let right = parse_trace(
            "# sf-nano-function-trace-v1 backend=native memory=0\nT1\t0\tnative\texit\t1\t3\t000000000000000b\t0000000000000001\t-\t-\n",
        )
        .unwrap();
        let options = CompareOptions {
            ignore_results: true,
        };
        assert!(compare_traces(&left, &right, &options).is_ok());
    }
}
