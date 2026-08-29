use serde::Serialize;
use sf_nano_core::{
    module::{
        entities::{ConstExpr, ElementInit, FunctionDef},
        Module,
    },
    op_decoder::{Decoder, Immediate, OpStream, OpcodeHandler},
    opcodes::{Opcode, WasmOpcode},
    WasmError,
};
use std::fmt::Write as _;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct DirectCallSite {
    pub caller: usize,
    pub callee: usize,
    pub in_loop: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct IndirectCallSite {
    pub caller: usize,
    pub type_index: u32,
    pub table_index: u32,
    pub in_loop: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct FunctionCensus {
    pub index: usize,
    pub imported: bool,
    pub code_bytes: usize,
    pub opcode_count: usize,
    pub contains_loop: bool,
    pub direct_call_sites: usize,
    pub loop_direct_call_sites: usize,
    pub indirect_call_sites: usize,
    pub loop_indirect_call_sites: usize,
    pub ref_func_count: usize,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Coverage {
    pub members: Vec<usize>,
    pub local_function_count: usize,
    pub opcode_count: usize,
    pub code_bytes: usize,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct SizeBucket {
    pub name: &'static str,
    pub min_opcodes: usize,
    pub max_opcodes: Option<usize>,
    pub function_count: usize,
    pub opcode_count: usize,
    pub code_bytes: usize,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct RecursiveScc {
    pub members: Vec<usize>,
    pub opcode_count: usize,
    pub code_bytes: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModuleCensus {
    pub name: String,
    pub total_function_count: usize,
    pub imported_function_count: usize,
    pub local_function_count: usize,
    pub total_opcode_count: usize,
    pub total_code_bytes: usize,
    pub loop_function_count: usize,
    pub direct_call_site_count: usize,
    pub loop_direct_call_site_count: usize,
    pub call_indirect_site_count: usize,
    pub loop_call_indirect_site_count: usize,
    pub call_ref_site_count: usize,
    pub open_world_table_count: usize,
    pub functions: Vec<FunctionCensus>,
    pub direct_calls: Vec<DirectCallSite>,
    pub indirect_calls: Vec<IndirectCallSite>,
    pub size_buckets: Vec<SizeBucket>,
    pub recursive_sccs: Vec<RecursiveScc>,
    pub export_roots: Coverage,
    pub export_root_closure: Coverage,
    pub start_roots: Coverage,
    pub start_root_closure: Coverage,
    pub element_roots: Coverage,
    pub element_root_closure: Coverage,
    pub roots_closure: Coverage,
    pub loop_functions: Coverage,
    pub loop_function_closure: Coverage,
    pub loop_call_target_closure: Coverage,
    pub recursive_function_closure: Coverage,
    pub declared_ref_targets: Coverage,
    pub conservative_indirect_targets: Coverage,
    pub conservative_indirect_closure: Coverage,
    pub static_hot_closure: Coverage,
    pub loop_policy_skippable_opcodes: usize,
    pub loop_policy_skippable_opcode_percent: f64,
    pub heavy_predecode_skippable_opcodes: usize,
    pub heavy_predecode_skippable_opcode_percent: f64,
}

#[derive(Default)]
struct FunctionScanner {
    caller: usize,
    opcode_count: usize,
    contains_loop: bool,
    loop_depth: usize,
    control_stack: Vec<bool>,
    direct_calls: Vec<DirectCallSite>,
    indirect_calls: Vec<IndirectCallSite>,
    ref_funcs: Vec<usize>,
    call_ref_sites: usize,
}

impl FunctionScanner {
    fn new(caller: usize) -> Self {
        Self {
            caller,
            ..Self::default()
        }
    }
}

impl OpcodeHandler for FunctionScanner {
    fn on_decode_begin(&mut self) -> Result<(), WasmError> {
        Ok(())
    }

    fn on_stream<'x, 'y, 'z>(
        &mut self,
        stream: &mut OpStream<'x, 'y, 'z>,
    ) -> Result<(), WasmError> {
        while let Some(decoded) = stream.next()? {
            self.opcode_count += 1;
            let WasmOpcode::OP(opcode) = decoded.wasm_op else {
                continue;
            };
            let in_loop = self.loop_depth != 0;
            match opcode {
                Opcode::CALL | Opcode::RETURN_CALL => {
                    let Immediate::FunctionIndex(callee) = decoded.imm else {
                        return Err(WasmError::internal("direct call immediate mismatch"));
                    };
                    self.direct_calls.push(DirectCallSite {
                        caller: self.caller,
                        callee: callee as usize,
                        in_loop,
                    });
                }
                Opcode::CALL_INDIRECT | Opcode::RETURN_CALL_INDIRECT => {
                    let Immediate::CallIndirectArgs { typeidx, tableidx } = decoded.imm else {
                        return Err(WasmError::internal("indirect call immediate mismatch"));
                    };
                    self.indirect_calls.push(IndirectCallSite {
                        caller: self.caller,
                        type_index: typeidx,
                        table_index: tableidx,
                        in_loop,
                    });
                }
                Opcode::CALL_REF | Opcode::RETURN_CALL_REF => self.call_ref_sites += 1,
                Opcode::REF_FUNC => {
                    let Immediate::FunctionIndex(index) = decoded.imm else {
                        return Err(WasmError::internal("ref.func immediate mismatch"));
                    };
                    self.ref_funcs.push(index as usize);
                }
                Opcode::LOOP => {
                    self.contains_loop = true;
                    self.control_stack.push(true);
                    self.loop_depth += 1;
                }
                Opcode::BLOCK | Opcode::IF | Opcode::TRY_TABLE => {
                    self.control_stack.push(false);
                }
                Opcode::END => {
                    if self.control_stack.pop().unwrap_or(false) {
                        self.loop_depth -= 1;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn on_decode_end(&mut self) -> Result<(), WasmError> {
        if !self.control_stack.is_empty() || self.loop_depth != 0 {
            return Err(WasmError::invalid("census control stack did not close"));
        }
        Ok(())
    }
}

#[derive(Default)]
struct RefFuncScanner {
    targets: Vec<usize>,
}

impl OpcodeHandler for RefFuncScanner {
    fn on_decode_begin(&mut self) -> Result<(), WasmError> {
        Ok(())
    }

    fn on_stream<'x, 'y, 'z>(
        &mut self,
        stream: &mut OpStream<'x, 'y, 'z>,
    ) -> Result<(), WasmError> {
        while let Some(decoded) = stream.next()? {
            if decoded.wasm_op == WasmOpcode::OP(Opcode::REF_FUNC) {
                let Immediate::FunctionIndex(index) = decoded.imm else {
                    return Err(WasmError::internal("ref.func immediate mismatch"));
                };
                self.targets.push(index as usize);
            }
        }
        Ok(())
    }

    fn on_decode_end(&mut self) -> Result<(), WasmError> {
        Ok(())
    }
}

fn scan_ref_funcs(expr: &ConstExpr, targets: &mut Vec<usize>) -> Result<(), WasmError> {
    let mut scanner = RefFuncScanner::default();
    let mut decoder = Decoder::new(expr);
    decoder.add_handler(&mut scanner);
    decoder.decode_function()?;
    drop(decoder);
    targets.extend(scanner.targets);
    Ok(())
}

fn normalize(indices: &mut Vec<usize>, bound: usize) {
    indices.retain(|&index| index < bound);
    indices.sort_unstable();
    indices.dedup();
}

fn adjacency(function_count: usize, calls: &[DirectCallSite], local: &[bool]) -> Vec<Vec<usize>> {
    let mut graph = vec![Vec::new(); function_count];
    for call in calls {
        if call.caller < function_count && call.callee < function_count && local[call.callee] {
            graph[call.caller].push(call.callee);
        }
    }
    for edges in &mut graph {
        edges.sort_unstable();
        edges.dedup();
    }
    graph
}

fn transitive_closure(seeds: &[usize], graph: &[Vec<usize>], local: &[bool]) -> Vec<usize> {
    let mut seen = vec![false; graph.len()];
    let mut stack = Vec::new();
    for &seed in seeds {
        if seed < graph.len() && local[seed] && !seen[seed] {
            seen[seed] = true;
            stack.push(seed);
        }
    }
    while let Some(function) = stack.pop() {
        for &callee in &graph[function] {
            if !seen[callee] {
                seen[callee] = true;
                stack.push(callee);
            }
        }
    }
    seen.into_iter()
        .enumerate()
        .filter_map(|(index, present)| present.then_some(index))
        .collect()
}

fn strongly_connected_components(graph: &[Vec<usize>], local: &[bool]) -> Vec<Vec<usize>> {
    let mut visited = vec![false; graph.len()];
    let mut order = Vec::new();
    for start in 0..graph.len() {
        if !local[start] || visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![(start, 0usize)];
        while let Some((node, next_edge)) = stack.last_mut() {
            if *next_edge < graph[*node].len() {
                let target = graph[*node][*next_edge];
                *next_edge += 1;
                if !visited[target] {
                    visited[target] = true;
                    stack.push((target, 0));
                }
            } else {
                let (node, _) = stack.pop().expect("non-empty DFS stack");
                order.push(node);
            }
        }
    }

    let mut reverse = vec![Vec::new(); graph.len()];
    for (caller, callees) in graph.iter().enumerate() {
        for &callee in callees {
            reverse[callee].push(caller);
        }
    }
    let mut assigned = vec![false; graph.len()];
    let mut components = Vec::new();
    for &start in order.iter().rev() {
        if assigned[start] {
            continue;
        }
        assigned[start] = true;
        let mut members = Vec::new();
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            members.push(node);
            for &caller in &reverse[node] {
                if !assigned[caller] {
                    assigned[caller] = true;
                    stack.push(caller);
                }
            }
        }
        members.sort_unstable();
        components.push(members);
    }
    components.sort_by_key(|members| members[0]);
    components
}

fn coverage(indices: Vec<usize>, functions: &[FunctionCensus]) -> Coverage {
    let mut members = indices;
    normalize(&mut members, functions.len());
    let mut opcode_count = 0usize;
    let mut code_bytes = 0usize;
    let mut local_function_count = 0usize;
    for &index in &members {
        let function = &functions[index];
        if !function.imported {
            local_function_count += 1;
            opcode_count += function.opcode_count;
            code_bytes += function.code_bytes;
        }
    }
    Coverage {
        members,
        local_function_count,
        opcode_count,
        code_bytes,
    }
}

fn size_buckets(functions: &[FunctionCensus]) -> Vec<SizeBucket> {
    let bounds = [
        ("0-8", 0, Some(8)),
        ("9-32", 9, Some(32)),
        ("33-128", 33, Some(128)),
        ("129-512", 129, Some(512)),
        ("513-2048", 513, Some(2048)),
        ("2049+", 2049, None),
    ];
    bounds
        .into_iter()
        .map(|(name, min_opcodes, max_opcodes)| {
            let mut bucket = SizeBucket {
                name,
                min_opcodes,
                max_opcodes,
                function_count: 0,
                opcode_count: 0,
                code_bytes: 0,
            };
            for function in functions.iter().filter(|function| !function.imported) {
                if function.opcode_count >= min_opcodes
                    && max_opcodes.is_none_or(|max| function.opcode_count <= max)
                {
                    bucket.function_count += 1;
                    bucket.opcode_count += function.opcode_count;
                    bucket.code_bytes += function.code_bytes;
                }
            }
            bucket
        })
        .collect()
}

pub fn analyze_module(module: &Module) -> Result<ModuleCensus, WasmError> {
    let function_count = module.functions().len();
    let mut functions = Vec::with_capacity(function_count);
    let mut direct_calls = Vec::new();
    let mut indirect_calls = Vec::new();
    let mut body_ref_funcs = Vec::new();
    let mut call_ref_site_count = 0usize;

    for (index, function) in module.functions().iter().enumerate() {
        let FunctionDef::Local(spec) = function.def() else {
            functions.push(FunctionCensus {
                index,
                imported: true,
                code_bytes: 0,
                opcode_count: 0,
                contains_loop: false,
                direct_call_sites: 0,
                loop_direct_call_sites: 0,
                indirect_call_sites: 0,
                loop_indirect_call_sites: 0,
                ref_func_count: 0,
            });
            continue;
        };
        let mut scanner = FunctionScanner::new(index);
        let mut decoder = Decoder::new(spec.code());
        decoder.add_handler(&mut scanner);
        decoder.decode_function()?;
        drop(decoder);
        let direct_count = scanner.direct_calls.len();
        let loop_direct_count = scanner
            .direct_calls
            .iter()
            .filter(|call| call.in_loop)
            .count();
        let indirect_count = scanner.indirect_calls.len();
        let loop_indirect_count = scanner
            .indirect_calls
            .iter()
            .filter(|call| call.in_loop)
            .count();
        call_ref_site_count += scanner.call_ref_sites;
        body_ref_funcs.extend(scanner.ref_funcs.iter().copied());
        direct_calls.extend(scanner.direct_calls);
        indirect_calls.extend(scanner.indirect_calls);
        functions.push(FunctionCensus {
            index,
            imported: false,
            code_bytes: spec.code().len(),
            opcode_count: scanner.opcode_count,
            contains_loop: scanner.contains_loop,
            direct_call_sites: direct_count,
            loop_direct_call_sites: loop_direct_count,
            indirect_call_sites: indirect_count,
            loop_indirect_call_sites: loop_indirect_count,
            ref_func_count: scanner.ref_funcs.len(),
        });
    }

    let local: Vec<bool> = functions
        .iter()
        .map(|function| !function.imported)
        .collect();
    let graph = adjacency(function_count, &direct_calls, &local);

    let mut export_roots: Vec<usize> = module
        .functions()
        .iter()
        .enumerate()
        .filter_map(|(index, function)| (!function.export_names().is_empty()).then_some(index))
        .collect();
    normalize(&mut export_roots, function_count);
    let mut start_roots = module
        .start_function_index()
        .into_iter()
        .collect::<Vec<_>>();
    normalize(&mut start_roots, function_count);

    let mut element_roots = Vec::new();
    let mut declared_refs = body_ref_funcs;
    for element in module.elements() {
        match element.get_init() {
            ElementInit::FunctionIndexes(indices) => {
                element_roots.extend(indices.iter().copied());
                declared_refs.extend(indices.iter().copied());
            }
            ElementInit::InitExprs { exprs, .. } => {
                for expr in exprs {
                    let before = declared_refs.len();
                    scan_ref_funcs(expr, &mut declared_refs)?;
                    element_roots.extend_from_slice(&declared_refs[before..]);
                }
            }
        }
    }
    for table in module.tables() {
        if let Some(expr) = table.spec().init_expr() {
            scan_ref_funcs(expr, &mut declared_refs)?;
        }
    }
    for global in module.globals() {
        if let Some(spec) = global.spec() {
            scan_ref_funcs(spec.init_expr(), &mut declared_refs)?;
        }
    }
    declared_refs.extend(export_roots.iter().copied());
    normalize(&mut element_roots, function_count);
    normalize(&mut declared_refs, function_count);

    let open_world_table_count = module
        .tables()
        .iter()
        .filter(|table| table.is_import() || !table.export_names().is_empty())
        .count();
    let mut indirect_targets = Vec::new();
    for site in &indirect_calls {
        let open_world = module
            .tables()
            .get(site.table_index as usize)
            .is_none_or(|table| table.is_import() || !table.export_names().is_empty());
        for (index, function) in module.functions().iter().enumerate() {
            if !local[index]
                || (!open_world && declared_refs.binary_search(&index).is_err())
                || !module
                    .types()
                    .types_equivalent(site.type_index, function.type_index())
            {
                continue;
            }
            indirect_targets.push(index);
        }
    }
    normalize(&mut indirect_targets, function_count);

    let loop_functions: Vec<usize> = functions
        .iter()
        .filter_map(|function| function.contains_loop.then_some(function.index))
        .collect();
    let mut loop_call_targets: Vec<usize> = direct_calls
        .iter()
        .filter_map(|call| call.in_loop.then_some(call.callee))
        .collect();
    normalize(&mut loop_call_targets, function_count);

    let components = strongly_connected_components(&graph, &local);
    let mut recursive_members = Vec::new();
    let mut recursive_sccs = Vec::new();
    for members in components {
        let recursive = members.len() > 1
            || members
                .first()
                .is_some_and(|&only| graph[only].binary_search(&only).is_ok());
        if !recursive {
            continue;
        }
        recursive_members.extend(members.iter().copied());
        let report = coverage(members, &functions);
        recursive_sccs.push(RecursiveScc {
            members: report.members,
            opcode_count: report.opcode_count,
            code_bytes: report.code_bytes,
        });
    }
    normalize(&mut recursive_members, function_count);

    let mut root_union = export_roots.clone();
    root_union.extend(start_roots.iter().copied());
    root_union.extend(element_roots.iter().copied());
    normalize(&mut root_union, function_count);
    let roots_closure_indices = transitive_closure(&root_union, &graph, &local);
    let export_closure_indices = transitive_closure(&export_roots, &graph, &local);
    let start_closure_indices = transitive_closure(&start_roots, &graph, &local);
    let element_closure_indices = transitive_closure(&element_roots, &graph, &local);
    let loop_closure_indices = transitive_closure(&loop_functions, &graph, &local);
    let loop_call_closure_indices = transitive_closure(&loop_call_targets, &graph, &local);
    let recursive_closure_indices = transitive_closure(&recursive_members, &graph, &local);
    let indirect_closure_indices = transitive_closure(&indirect_targets, &graph, &local);

    let mut hot_seeds = root_union;
    hot_seeds.extend(loop_functions.iter().copied());
    hot_seeds.extend(loop_call_targets.iter().copied());
    hot_seeds.extend(recursive_members.iter().copied());
    hot_seeds.extend(indirect_targets.iter().copied());
    if call_ref_site_count != 0 {
        hot_seeds.extend(declared_refs.iter().copied());
    }
    normalize(&mut hot_seeds, function_count);
    let hot_indices = transitive_closure(&hot_seeds, &graph, &local);

    let total_opcode_count: usize = functions.iter().map(|function| function.opcode_count).sum();
    let total_code_bytes: usize = functions.iter().map(|function| function.code_bytes).sum();
    let hot_coverage = coverage(hot_indices, &functions);
    let skippable = total_opcode_count.saturating_sub(hot_coverage.opcode_count);
    let skippable_percent = if total_opcode_count == 0 {
        0.0
    } else {
        skippable as f64 * 100.0 / total_opcode_count as f64
    };
    let imported_function_count = functions
        .iter()
        .filter(|function| function.imported)
        .count();
    let local_function_count = functions.len() - imported_function_count;
    let bucket_reports = size_buckets(&functions);
    let export_coverage = coverage(export_roots, &functions);
    let export_closure = coverage(export_closure_indices, &functions);
    let start_coverage = coverage(start_roots, &functions);
    let start_closure = coverage(start_closure_indices, &functions);
    let element_coverage = coverage(element_roots, &functions);
    let element_closure = coverage(element_closure_indices, &functions);
    let roots_coverage = coverage(roots_closure_indices, &functions);
    let loop_coverage = coverage(loop_functions.clone(), &functions);
    let loop_closure = coverage(loop_closure_indices, &functions);
    let loop_call_closure = coverage(loop_call_closure_indices, &functions);
    let recursive_closure = coverage(recursive_closure_indices, &functions);
    let declared_ref_coverage = coverage(declared_refs, &functions);
    let indirect_target_coverage = coverage(indirect_targets, &functions);
    let indirect_closure = coverage(indirect_closure_indices, &functions);
    let loop_skippable = total_opcode_count.saturating_sub(loop_closure.opcode_count);
    let loop_skippable_percent = if total_opcode_count == 0 {
        0.0
    } else {
        loop_skippable as f64 * 100.0 / total_opcode_count as f64
    };

    Ok(ModuleCensus {
        name: module.name().to_owned(),
        total_function_count: function_count,
        imported_function_count,
        local_function_count,
        total_opcode_count,
        total_code_bytes,
        loop_function_count: loop_functions.len(),
        direct_call_site_count: direct_calls.len(),
        loop_direct_call_site_count: direct_calls.iter().filter(|call| call.in_loop).count(),
        call_indirect_site_count: indirect_calls.len(),
        loop_call_indirect_site_count: indirect_calls.iter().filter(|call| call.in_loop).count(),
        call_ref_site_count,
        open_world_table_count,
        size_buckets: bucket_reports,
        recursive_sccs,
        export_roots: export_coverage,
        export_root_closure: export_closure,
        start_roots: start_coverage,
        start_root_closure: start_closure,
        element_roots: element_coverage,
        element_root_closure: element_closure,
        roots_closure: roots_coverage,
        loop_functions: loop_coverage,
        loop_function_closure: loop_closure,
        loop_call_target_closure: loop_call_closure,
        recursive_function_closure: recursive_closure,
        declared_ref_targets: declared_ref_coverage,
        conservative_indirect_targets: indirect_target_coverage,
        conservative_indirect_closure: indirect_closure,
        static_hot_closure: hot_coverage,
        loop_policy_skippable_opcodes: loop_skippable,
        loop_policy_skippable_opcode_percent: loop_skippable_percent,
        heavy_predecode_skippable_opcodes: skippable,
        heavy_predecode_skippable_opcode_percent: skippable_percent,
        functions,
        direct_calls,
        indirect_calls,
    })
}

fn percent(part: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 * 100.0 / total as f64
    }
}

pub fn render_markdown(reports: &[ModuleCensus]) -> String {
    let mut out = String::from(
        "# Eager-tier structural census\n\n\
         No wall-clock measurements are collected. `skippable` is `(all local Wasm opcodes - static-hot closure opcodes) / all local Wasm opcodes`. Static-hot seeds are exports/start/elements, loop functions, recursive SCCs, conservative type-compatible indirect targets, and declared ref targets when `call_ref` exists.\n\n\
         | Module | Local funcs | Wasm ops | Code bytes | Loop funcs | Loop closure ops | Root closure ops | Indirect closure ops | Static-hot ops | Loop-only skip | Conservative skip |\n\
         |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    for report in reports {
        writeln!(
            out,
            "| `{}` | {} | {} | {} | {:.2}% | {:.2}% | {:.2}% | {:.2}% | {:.2}% | {:.2}% | {:.2}% |",
            report.name,
            report.local_function_count,
            report.total_opcode_count,
            report.total_code_bytes,
            percent(report.loop_function_count, report.local_function_count),
            percent(
                report.loop_function_closure.opcode_count,
                report.total_opcode_count
            ),
            percent(report.roots_closure.opcode_count, report.total_opcode_count),
            percent(
                report.conservative_indirect_closure.opcode_count,
                report.total_opcode_count
            ),
            percent(
                report.static_hot_closure.opcode_count,
                report.total_opcode_count
            ),
            report.loop_policy_skippable_opcode_percent,
            report.heavy_predecode_skippable_opcode_percent,
        )
        .expect("write markdown");
    }

    for report in reports {
        writeln!(out, "\n## {}\n", report.name).expect("write markdown");
        writeln!(
            out,
            "- Direct call sites: {} ({} inside loops)",
            report.direct_call_site_count, report.loop_direct_call_site_count
        )
        .expect("write markdown");
        writeln!(
            out,
            "- `call_indirect` sites: {} ({} inside loops); open-world tables: {}",
            report.call_indirect_site_count,
            report.loop_call_indirect_site_count,
            report.open_world_table_count
        )
        .expect("write markdown");
        writeln!(
            out,
            "- Recursive SCCs: {}; declared ref targets: {}; conservative indirect targets: {}",
            report.recursive_sccs.len(),
            report.declared_ref_targets.local_function_count,
            report.conservative_indirect_targets.local_function_count
        )
        .expect("write markdown");
        out.push_str(
            "\n| Function size | Functions | Opcodes | Code bytes |\n|---|---:|---:|---:|\n",
        );
        for bucket in &report.size_buckets {
            writeln!(
                out,
                "| {} | {} | {} | {} |",
                bucket.name, bucket.function_count, bucket.opcode_count, bucket.code_bytes
            )
            .expect("write markdown");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn census(wat: &str) -> ModuleCensus {
        let bytes = wat::parse_str(wat).expect("wat");
        let module = Module::new("fixture", &bytes).expect("module");
        analyze_module(&module).expect("census")
    }

    #[test]
    fn nested_loops_mark_calls_and_transitive_callees() {
        let report = census(
            r#"(module
                (func $leaf)
                (func $middle call $leaf)
                (func $loop
                    (loop
                        call $middle
                        (block (loop call $leaf))))
                (export "run" (func $loop)))"#,
        );
        assert_eq!(report.loop_functions.members, vec![2]);
        assert_eq!(report.loop_direct_call_site_count, 2);
        assert_eq!(report.loop_function_closure.members, vec![0, 1, 2]);
        assert_eq!(report.loop_call_target_closure.members, vec![0, 1]);
        assert_eq!(report.export_roots.members, vec![2]);
        assert_eq!(report.roots_closure.members, vec![0, 1, 2]);
    }

    #[test]
    fn recursive_sccs_cover_mutual_and_self_recursion() {
        let report = census(
            r#"(module
                (func $a call $b)
                (func $b call $a)
                (func $self call $self)
                (func))"#,
        );
        let members: Vec<Vec<usize>> = report
            .recursive_sccs
            .iter()
            .map(|scc| scc.members.clone())
            .collect();
        assert_eq!(members, vec![vec![0, 1], vec![2]]);
        assert_eq!(report.recursive_function_closure.members, vec![0, 1, 2]);
    }

    #[test]
    fn private_indirect_targets_use_declared_refs_and_type_equivalence() {
        let report = census(
            r#"(module
                (type $t (func))
                (type $other (func (param i32)))
                (table 2 funcref)
                (func $elem (type $t))
                (func $ref (type $t))
                (func $undeclared (type $t))
                (func $wrong (type $other))
                (elem (i32.const 0) $elem)
                (func $caller
                    ref.func $ref drop
                    i32.const 0 call_indirect (type $t)))"#,
        );
        assert_eq!(report.call_indirect_site_count, 1);
        assert_eq!(report.declared_ref_targets.members, vec![0, 1]);
        assert_eq!(report.conservative_indirect_targets.members, vec![0, 1]);
        assert!(!report.conservative_indirect_targets.members.contains(&2));
        assert!(!report.conservative_indirect_targets.members.contains(&3));
    }

    #[test]
    fn exported_table_makes_indirect_target_set_open_world() {
        let report = census(
            r#"(module
                (type $t (func))
                (type $other (func (param i32)))
                (table (export "table") 1 funcref)
                (func $declared (type $t))
                (func $undeclared (type $t))
                (func $wrong (type $other))
                (elem (i32.const 0) $declared)
                (func (param i32) local.get 0 call_indirect (type $t)))"#,
        );
        assert_eq!(report.open_world_table_count, 1);
        assert_eq!(report.conservative_indirect_targets.members, vec![0, 1]);
        assert!(!report.conservative_indirect_targets.members.contains(&2));
    }

    #[test]
    fn roots_and_size_buckets_are_deterministic() {
        let report = census(
            r#"(module
                (func $start call $leaf)
                (func $leaf)
                (func $element)
                (start $start)
                (elem declare func $element)
                (export "leaf" (func $leaf)))"#,
        );
        assert_eq!(report.start_roots.members, vec![0]);
        assert_eq!(report.start_root_closure.members, vec![0, 1]);
        assert_eq!(report.export_roots.members, vec![1]);
        assert_eq!(report.export_root_closure.members, vec![1]);
        assert_eq!(report.element_roots.members, vec![2]);
        assert_eq!(report.element_root_closure.members, vec![2]);
        assert_eq!(report.roots_closure.members, vec![0, 1, 2]);
        assert_eq!(
            report
                .size_buckets
                .iter()
                .map(|bucket| bucket.function_count)
                .sum::<usize>(),
            report.local_function_count
        );
    }

    #[test]
    fn report_serializes_to_json_and_markdown() {
        let report = census("(module (func (export \"run\") (loop)))");
        let json = serde_json::to_string_pretty(&report).expect("json");
        assert!(json.contains("\"heavy_predecode_skippable_opcodes\""));
        assert!(json.contains("\"loop_function_closure\""));
        let markdown = render_markdown(&[report]);
        assert!(markdown.contains("# Eager-tier structural census"));
        assert!(markdown.contains("| `fixture` |"));
        assert!(markdown.contains("Loop-only skip"));
        assert!(markdown.contains("Conservative skip"));
    }
}
