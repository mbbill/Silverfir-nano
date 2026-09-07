//! Minimal report for the internal allocation profiler.
use std::fmt::Write;
use std::path::PathBuf;
use tracked_alloc::AllocationProfile;

pub struct Session {
    output_path: PathBuf,
    command_line: Vec<String>,
}
impl Session {
    pub fn new(output_path: Option<PathBuf>, command_line: &[String]) -> Self {
        let output_path = output_path.unwrap_or_else(|| {
            std::env::temp_dir().join(format!("sf-nano-memprof-{}.html", std::process::id()))
        });
        let command_line = command_line.to_vec();
        tracked_alloc::reset_tracking();
        tracked_alloc::set_tracking_enabled(true);
        Self {
            output_path,
            command_line,
        }
    }
    pub fn finish(self) -> Result<PathBuf, String> {
        tracked_alloc::set_tracking_enabled(false);
        let report = render(&tracked_alloc::profile(), &self.command_line);
        std::fs::write(&self.output_path, report).map_err(|e| e.to_string())?;
        Ok(self.output_path.clone())
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        tracked_alloc::set_tracking_enabled(false);
    }
}
fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
fn render(profile: &AllocationProfile, command: &[String]) -> String {
    let s = &profile.snapshot;
    let mut html = format!("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>Silverfir-nano memory profile</title><style>body{{font:16px system-ui;max-width:1100px;margin:3em auto;padding:0 1em}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;border-bottom:1px solid #ccc;padding:.5em}}code{{overflow-wrap:anywhere}}</style><h1>Memory profile</h1><code>{}</code><p>Process-wide heap allocations recorded after measurement began; profiler bookkeeping is excluded. These are allocator byte counts, not RSS. Allocation traffic counts new blocks and positive realloc growth. Rust container types and allocation backtraces are not collected.</p><table><tr><th>Metric</th><th>Value</th></tr>", escape(&command.join(" ")));
    for (label, value) in [
        ("Live heap bytes", s.total_bytes as u64),
        ("Peak heap bytes", s.peak_bytes as u64),
        ("Allocated bytes", s.allocated_bytes),
        ("Allocations", s.allocations),
        ("Reallocations", s.reallocations),
        ("Deallocations", s.deallocations),
        ("Live allocations", s.live_allocations as u64),
        ("Code buffer capacity bytes", s.code_buffer_bytes as u64),
        (
            "Peak code buffer capacity bytes",
            s.peak_code_buffer_bytes as u64,
        ),
        ("Guard memory committed bytes", s.guard_page_bytes as u64),
        (
            "Peak guard memory committed bytes",
            s.peak_guard_page_bytes as u64,
        ),
        ("Other explicit runtime bytes", s.other_runtime_bytes as u64),
        (
            "Peak other explicit runtime bytes",
            s.peak_other_runtime_bytes as u64,
        ),
    ] {
        write!(html, "<tr><td>{label}</td><td>{value}</td></tr>").unwrap();
    }
    html.push_str("</table><h2>Compilation phases</h2><p>Traffic is process-wide during each span. Nested and overlapping spans must not be summed.</p><table><tr><th>Phase</th><th>Function</th><th>Duration (ms)</th><th>Allocated bytes</th></tr>");
    for phase in &profile.phases {
        write!(
            html,
            "<tr><td>{}</td><td>{}</td><td>{:.3}</td><td>{}</td></tr>",
            escape(phase.name),
            phase
                .function_index
                .map(|v| v.to_string())
                .unwrap_or_default(),
            phase.end_time_ns.saturating_sub(phase.start_time_ns) as f64 / 1_000_000.0,
            phase.allocated_bytes
        )
        .unwrap();
    }
    write!(
        html,
        "</table><p>Omitted phase records: {}</p></html>",
        profile.omitted_phases
    )
    .unwrap();
    html
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn report_escapes_untrusted_command_and_phase_names() {
        let mut profile = AllocationProfile::default();
        profile.phases.push(tracked_alloc::ProfilePhase {
            name: "<script>",
            function_index: None,
            start_time_ns: 0,
            end_time_ns: 1_000_000,
            allocated_bytes: 42,
        });
        let html = render(&profile, &["<&\"'".into()]);
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;&amp;&quot;&#39;"));
        assert!(html.contains("1.000"));
    }
}
