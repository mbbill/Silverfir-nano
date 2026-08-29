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
    pub loop_depth: usize,
    pub in_loop: bool,
    pub reachable: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct IndirectCallSite {
    pub caller: usize,
    pub type_index: u32,
    pub table_index: u32,
    pub loop_depth: usize,
    pub in_loop: bool,
    pub reachable: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct FunctionCensus {
    pub index: usize,
    pub imported: bool,
    pub code_bytes: usize,
    pub opcode_count: usize,
    pub contains_loop: bool,
    pub loop_structure_opcode_count: usize,
    pub loop_structure_code_bytes: usize,
    pub in_loop_opcode_count: usize,
    pub in_loop_code_bytes: usize,
    pub reachable_in_loop_opcode_count: usize,
    pub reachable_in_loop_code_bytes: usize,
    pub loop_body_opcode_count: usize,
    pub loop_body_code_bytes: usize,
    pub reachable_loop_body_opcode_count: usize,
    pub reachable_loop_body_code_bytes: usize,
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
pub struct BlockTierCensus {
    pub loop_structure_opcode_count: usize,
    pub loop_structure_code_bytes: usize,
    pub syntactic_in_loop_opcode_count: usize,
    pub syntactic_in_loop_code_bytes: usize,
    pub reachable_in_loop_opcode_count: usize,
    pub reachable_in_loop_code_bytes: usize,
    pub syntactic_loop_body_opcode_count: usize,
    pub syntactic_loop_body_code_bytes: usize,
    pub reachable_loop_body_opcode_count: usize,
    pub reachable_loop_body_code_bytes: usize,
    pub full_callee_closure: Coverage,
    pub native_opcode_count: usize,
    pub native_code_bytes: usize,
    pub native_opcode_percent: f64,
    pub native_code_byte_percent: f64,
    pub baseline_opcode_count: usize,
    pub baseline_code_bytes: usize,
    pub baseline_opcode_percent: f64,
    pub baseline_code_byte_percent: f64,
    pub body_only_native_opcode_lower_bound: usize,
    pub body_only_native_opcode_upper_bound: usize,
    pub body_only_baseline_opcode_percent_lower_bound: f64,
    pub body_only_baseline_opcode_percent_upper_bound: f64,
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
    pub block_tier: BlockTierCensus,
    pub loop_policy_skippable_opcodes: usize,
    pub loop_policy_skippable_opcode_percent: f64,
    pub heavy_predecode_skippable_opcodes: usize,
    pub heavy_predecode_skippable_opcode_percent: f64,
}

#[derive(Clone, Copy, Default)]
struct ControlFrame {
    is_loop: bool,
    is_if: bool,
    entry_reachable: bool,
    then_reachable: bool,
    saw_else: bool,
    end_targeted: bool,
}

#[derive(Default)]
struct FunctionScanner {
    caller: usize,
    opcode_count: usize,
    contains_loop: bool,
    loop_depth: usize,
    control_stack: Vec<ControlFrame>,
    reachable: bool,
    loop_structure_opcode_count: usize,
    loop_structure_code_bytes: usize,
    in_loop_opcode_count: usize,
    in_loop_code_bytes: usize,
    reachable_in_loop_opcode_count: usize,
    reachable_in_loop_code_bytes: usize,
    loop_body_opcode_count: usize,
    loop_body_code_bytes: usize,
    reachable_loop_body_opcode_count: usize,
    reachable_loop_body_code_bytes: usize,
    direct_calls: Vec<DirectCallSite>,
    indirect_calls: Vec<IndirectCallSite>,
    ref_funcs: Vec<usize>,
    call_ref_sites: usize,
}

impl FunctionScanner {
    fn new(caller: usize) -> Self {
        Self {
            caller,
            reachable: true,
            ..Self::default()
        }
    }

    fn mark_branch_target(&mut self, depth: u32) {
        let Some(frame_index) = self.control_stack.len().checked_sub(depth as usize + 1) else {
            return;
        };
        let frame = &mut self.control_stack[frame_index];
        if !frame.is_loop {
            frame.end_targeted = true;
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
            let in_loop = self.loop_depth != 0;
            let reachable = self.reachable;
            let encoded_bytes = decoded.next_op_offset - decoded.op_offset;
            let opcode = match decoded.wasm_op {
                WasmOpcode::OP(opcode) => Some(opcode),
                _ => None,
            };
            let loop_open = opcode == Some(Opcode::LOOP);
            let loop_close = opcode == Some(Opcode::END)
                && self.control_stack.last().is_some_and(|frame| frame.is_loop);
            let loop_structure = loop_open || loop_close;
            if loop_structure {
                self.loop_structure_opcode_count += 1;
                self.loop_structure_code_bytes += encoded_bytes;
            }
            if in_loop {
                self.in_loop_opcode_count += 1;
                self.in_loop_code_bytes += encoded_bytes;
                if reachable {
                    self.reachable_in_loop_opcode_count += 1;
                    self.reachable_in_loop_code_bytes += encoded_bytes;
                }
                if !loop_structure {
                    self.loop_body_opcode_count += 1;
                    self.loop_body_code_bytes += encoded_bytes;
                    if reachable {
                        self.reachable_loop_body_opcode_count += 1;
                        self.reachable_loop_body_code_bytes += encoded_bytes;
                    }
                }
            }

            let Some(opcode) = opcode else {
                if reachable {
                    if let Immediate::BrOnCast { label_idx, .. } = &decoded.imm {
                        self.mark_branch_target(*label_idx);
                    }
                }
                continue;
            };
            match opcode {
                Opcode::CALL | Opcode::RETURN_CALL => {
                    let Immediate::FunctionIndex(callee) = &decoded.imm else {
                        return Err(WasmError::internal("direct call immediate mismatch"));
                    };
                    self.direct_calls.push(DirectCallSite {
                        caller: self.caller,
                        callee: *callee as usize,
                        loop_depth: self.loop_depth,
                        in_loop,
                        reachable,
                    });
                    if opcode == Opcode::RETURN_CALL && reachable {
                        self.reachable = false;
                    }
                }
                Opcode::CALL_INDIRECT | Opcode::RETURN_CALL_INDIRECT => {
                    let Immediate::CallIndirectArgs { typeidx, tableidx } = &decoded.imm else {
                        return Err(WasmError::internal("indirect call immediate mismatch"));
                    };
                    self.indirect_calls.push(IndirectCallSite {
                        caller: self.caller,
                        type_index: *typeidx,
                        table_index: *tableidx,
                        loop_depth: self.loop_depth,
                        in_loop,
                        reachable,
                    });
                    if opcode == Opcode::RETURN_CALL_INDIRECT && reachable {
                        self.reachable = false;
                    }
                }
                Opcode::CALL_REF | Opcode::RETURN_CALL_REF => {
                    self.call_ref_sites += 1;
                    if opcode == Opcode::RETURN_CALL_REF && reachable {
                        self.reachable = false;
                    }
                }
                Opcode::REF_FUNC => {
                    let Immediate::FunctionIndex(index) = &decoded.imm else {
                        return Err(WasmError::internal("ref.func immediate mismatch"));
                    };
                    self.ref_funcs.push(*index as usize);
                }
                Opcode::LOOP => {
                    self.contains_loop = true;
                    self.control_stack.push(ControlFrame {
                        is_loop: true,
                        entry_reachable: reachable,
                        ..ControlFrame::default()
                    });
                    self.loop_depth += 1;
                }
                Opcode::BLOCK => self.control_stack.push(ControlFrame {
                    entry_reachable: reachable,
                    ..ControlFrame::default()
                }),
                Opcode::IF => self.control_stack.push(ControlFrame {
                    is_if: true,
                    entry_reachable: reachable,
                    ..ControlFrame::default()
                }),
                Opcode::TRY_TABLE => {
                    if reachable {
                        if let Immediate::TryTable { catches, .. } = &decoded.imm {
                            for catch in catches {
                                self.mark_branch_target(catch.label_idx);
                            }
                        }
                    }
                    self.control_stack.push(ControlFrame {
                        entry_reachable: reachable,
                        ..ControlFrame::default()
                    });
                }
                Opcode::ELSE => {
                    if let Some(frame) = self.control_stack.last_mut() {
                        frame.then_reachable = self.reachable;
                        frame.saw_else = true;
                        self.reachable = frame.entry_reachable;
                    }
                }
                Opcode::END => {
                    if let Some(frame) = self.control_stack.pop() {
                        if frame.is_loop {
                            self.loop_depth -= 1;
                        }
                        self.reachable = if frame.is_if {
                            let false_path = if frame.saw_else {
                                frame.then_reachable
                            } else {
                                frame.entry_reachable
                            };
                            self.reachable || false_path || frame.end_targeted
                        } else {
                            self.reachable || frame.end_targeted
                        };
                    }
                }
                Opcode::BR => {
                    if reachable {
                        if let Immediate::LabelIndex(depth) = &decoded.imm {
                            self.mark_branch_target(*depth);
                        }
                        self.reachable = false;
                    }
                }
                Opcode::BR_IF => {
                    if reachable {
                        if let Immediate::LabelIndex(depth) = &decoded.imm {
                            self.mark_branch_target(*depth);
                        }
                    }
                }
                Opcode::BR_TABLE => {
                    if reachable {
                        if let Immediate::BrLabels(labels, default) = &decoded.imm {
                            for &depth in labels {
                                self.mark_branch_target(depth);
                            }
                            self.mark_branch_target(*default);
                        }
                        self.reachable = false;
                    }
                }
                Opcode::BR_ON_NULL | Opcode::BR_ON_NON_NULL => {
                    if reachable {
                        if let Immediate::LabelIndex(depth) = &decoded.imm {
                            self.mark_branch_target(*depth);
                        }
                    }
                }
                Opcode::UNREACHABLE | Opcode::RETURN | Opcode::THROW | Opcode::THROW_REF => {
                    if reachable {
                        self.reachable = false;
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

fn reachable_adjacency(
    function_count: usize,
    calls: &[DirectCallSite],
    local: &[bool],
) -> Vec<Vec<usize>> {
    let mut graph = vec![Vec::new(); function_count];
    for call in calls.iter().filter(|call| call.reachable) {
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

fn conservative_indirect_targets<'a>(
    module: &Module,
    sites: impl IntoIterator<Item = &'a IndirectCallSite>,
    declared_refs: &[usize],
    local: &[bool],
) -> Vec<usize> {
    let function_count = module.functions().len();
    let mut targets = Vec::new();
    for site in sites {
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
            targets.push(index);
        }
    }
    normalize(&mut targets, function_count);
    targets
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
                loop_structure_opcode_count: 0,
                loop_structure_code_bytes: 0,
                in_loop_opcode_count: 0,
                in_loop_code_bytes: 0,
                reachable_in_loop_opcode_count: 0,
                reachable_in_loop_code_bytes: 0,
                loop_body_opcode_count: 0,
                loop_body_code_bytes: 0,
                reachable_loop_body_opcode_count: 0,
                reachable_loop_body_code_bytes: 0,
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
            loop_structure_opcode_count: scanner.loop_structure_opcode_count,
            loop_structure_code_bytes: scanner.loop_structure_code_bytes,
            in_loop_opcode_count: scanner.in_loop_opcode_count,
            in_loop_code_bytes: scanner.in_loop_code_bytes,
            reachable_in_loop_opcode_count: scanner.reachable_in_loop_opcode_count,
            reachable_in_loop_code_bytes: scanner.reachable_in_loop_code_bytes,
            loop_body_opcode_count: scanner.loop_body_opcode_count,
            loop_body_code_bytes: scanner.loop_body_code_bytes,
            reachable_loop_body_opcode_count: scanner.reachable_loop_body_opcode_count,
            reachable_loop_body_code_bytes: scanner.reachable_loop_body_code_bytes,
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
    let reachable_graph = reachable_adjacency(function_count, &direct_calls, &local);

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
    let indirect_targets =
        conservative_indirect_targets(module, &indirect_calls, &declared_refs, &local);
    let loop_indirect_targets = conservative_indirect_targets(
        module,
        indirect_calls
            .iter()
            .filter(|site| site.in_loop && site.reachable),
        &declared_refs,
        &local,
    );

    let loop_functions: Vec<usize> = functions
        .iter()
        .filter_map(|function| function.contains_loop.then_some(function.index))
        .collect();
    let mut loop_call_targets: Vec<usize> = direct_calls
        .iter()
        .filter_map(|call| call.in_loop.then_some(call.callee))
        .collect();
    normalize(&mut loop_call_targets, function_count);
    let mut reachable_loop_call_targets: Vec<usize> = direct_calls
        .iter()
        .filter_map(|call| (call.in_loop && call.reachable).then_some(call.callee))
        .collect();
    normalize(&mut reachable_loop_call_targets, function_count);

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

    let mut block_full_callee_seeds = reachable_loop_call_targets;
    block_full_callee_seeds.extend(loop_indirect_targets);
    normalize(&mut block_full_callee_seeds, function_count);
    let block_full_callee_indices =
        transitive_closure(&block_full_callee_seeds, &reachable_graph, &local);

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
    let loop_structure_opcode_count = functions
        .iter()
        .map(|function| function.loop_structure_opcode_count)
        .sum();
    let loop_structure_code_bytes = functions
        .iter()
        .map(|function| function.loop_structure_code_bytes)
        .sum();
    let syntactic_in_loop_opcode_count = functions
        .iter()
        .map(|function| function.in_loop_opcode_count)
        .sum();
    let syntactic_in_loop_code_bytes = functions
        .iter()
        .map(|function| function.in_loop_code_bytes)
        .sum();
    let reachable_in_loop_opcode_count = functions
        .iter()
        .map(|function| function.reachable_in_loop_opcode_count)
        .sum();
    let reachable_in_loop_code_bytes = functions
        .iter()
        .map(|function| function.reachable_in_loop_code_bytes)
        .sum();
    let syntactic_loop_body_opcode_count = functions
        .iter()
        .map(|function| function.loop_body_opcode_count)
        .sum();
    let syntactic_loop_body_code_bytes = functions
        .iter()
        .map(|function| function.loop_body_code_bytes)
        .sum();
    let reachable_loop_body_opcode_count = functions
        .iter()
        .map(|function| function.reachable_loop_body_opcode_count)
        .sum();
    let reachable_loop_body_code_bytes = functions
        .iter()
        .map(|function| function.reachable_loop_body_code_bytes)
        .sum();

    let block_full_callee_coverage = coverage(block_full_callee_indices, &functions);
    let mut block_full_function = vec![false; function_count];
    for &index in &block_full_callee_coverage.members {
        block_full_function[index] = true;
    }
    let mut block_native_opcode_count = 0usize;
    let mut block_native_code_bytes = 0usize;
    for function in functions.iter().filter(|function| !function.imported) {
        if block_full_function[function.index] {
            block_native_opcode_count += function.opcode_count;
            block_native_code_bytes += function.code_bytes;
        } else {
            block_native_opcode_count += function.reachable_in_loop_opcode_count;
            block_native_code_bytes += function.reachable_in_loop_code_bytes;
        }
    }
    let block_baseline_opcode_count = total_opcode_count.saturating_sub(block_native_opcode_count);
    let block_baseline_code_bytes = total_code_bytes.saturating_sub(block_native_code_bytes);
    let block_tier = BlockTierCensus {
        loop_structure_opcode_count,
        loop_structure_code_bytes,
        syntactic_in_loop_opcode_count,
        syntactic_in_loop_code_bytes,
        reachable_in_loop_opcode_count,
        reachable_in_loop_code_bytes,
        syntactic_loop_body_opcode_count,
        syntactic_loop_body_code_bytes,
        reachable_loop_body_opcode_count,
        reachable_loop_body_code_bytes,
        full_callee_closure: block_full_callee_coverage,
        native_opcode_count: block_native_opcode_count,
        native_code_bytes: block_native_code_bytes,
        native_opcode_percent: percent(block_native_opcode_count, total_opcode_count),
        native_code_byte_percent: percent(block_native_code_bytes, total_code_bytes),
        baseline_opcode_count: block_baseline_opcode_count,
        baseline_code_bytes: block_baseline_code_bytes,
        baseline_opcode_percent: percent(block_baseline_opcode_count, total_opcode_count),
        baseline_code_byte_percent: percent(block_baseline_code_bytes, total_code_bytes),
        body_only_native_opcode_lower_bound: reachable_loop_body_opcode_count,
        body_only_native_opcode_upper_bound: syntactic_loop_body_opcode_count,
        body_only_baseline_opcode_percent_lower_bound: percent(
            total_opcode_count.saturating_sub(syntactic_loop_body_opcode_count),
            total_opcode_count,
        ),
        body_only_baseline_opcode_percent_upper_bound: percent(
            total_opcode_count.saturating_sub(reachable_loop_body_opcode_count),
            total_opcode_count,
        ),
    };
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
        block_tier,
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
         No wall-clock measurements are collected. Function-level `skippable` is `(all local Wasm opcodes - static-hot closure opcodes) / all local Wasm opcodes`. Static-hot seeds are exports/start/elements, loop functions, recursive SCCs, conservative type-compatible indirect targets, and declared ref targets when `call_ref` exists.\n\n\
         Block-level native coverage is the de-duplicated union of reachable opcodes decoded at `loop_depth > 0`, complete reachable direct-call closures rooted at loop call sites, and conservative type-compatible targets of reachable `call_indirect` sites in loops. An outer `loop` opener has depth zero; loop-structure counts list every `loop` opener and its matching `end` separately. Loop-body counts exclude both structural opcodes. The body-only baseline range is bounded by syntactically present loop-body opcodes on the low side and reachability-filtered loop-body opcodes on the high side.\n\n\
         ## Function-level policies\n\n\
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

    out.push_str(
        "\n## Block-level structural coverage\n\n\
         | Module | Loop structure ops | In-loop ops (syntax / reachable) | Loop-body ops (syntax / reachable) | Full callee closure ops | Block native ops | Block native bytes | Baseline ops | Baseline bytes | Body-only baseline ops | Body-only baseline bytes |\n\
         |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    for report in reports {
        let block = &report.block_tier;
        let body_byte_baseline_lower = percent(
            report
                .total_code_bytes
                .saturating_sub(block.syntactic_loop_body_code_bytes),
            report.total_code_bytes,
        );
        let body_byte_baseline_upper = percent(
            report
                .total_code_bytes
                .saturating_sub(block.reachable_loop_body_code_bytes),
            report.total_code_bytes,
        );
        writeln!(
            out,
            "| `{}` | {} ({:.2}%) | {} / {} | {} / {} | {} ({:.2}%) | {} ({:.2}%) | {} ({:.2}%) | {} ({:.2}%) | {} ({:.2}%) | {:.2}%–{:.2}% | {:.2}%–{:.2}% |",
            report.name,
            block.loop_structure_opcode_count,
            percent(block.loop_structure_opcode_count, report.total_opcode_count),
            block.syntactic_in_loop_opcode_count,
            block.reachable_in_loop_opcode_count,
            block.syntactic_loop_body_opcode_count,
            block.reachable_loop_body_opcode_count,
            block.full_callee_closure.opcode_count,
            percent(
                block.full_callee_closure.opcode_count,
                report.total_opcode_count
            ),
            block.native_opcode_count,
            block.native_opcode_percent,
            block.native_code_bytes,
            block.native_code_byte_percent,
            block.baseline_opcode_count,
            block.baseline_opcode_percent,
            block.baseline_code_bytes,
            block.baseline_code_byte_percent,
            block.body_only_baseline_opcode_percent_lower_bound,
            block.body_only_baseline_opcode_percent_upper_bound,
            body_byte_baseline_lower,
            body_byte_baseline_upper,
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
        writeln!(
            out,
            "- Loop-depth spans: syntactic {}/{} opcodes/bytes; reachable {}/{}; loop structure {}/{}; loop body syntactic {}/{} and reachable {}/{}",
            report.block_tier.syntactic_in_loop_opcode_count,
            report.block_tier.syntactic_in_loop_code_bytes,
            report.block_tier.reachable_in_loop_opcode_count,
            report.block_tier.reachable_in_loop_code_bytes,
            report.block_tier.loop_structure_opcode_count,
            report.block_tier.loop_structure_code_bytes,
            report.block_tier.syntactic_loop_body_opcode_count,
            report.block_tier.syntactic_loop_body_code_bytes,
            report.block_tier.reachable_loop_body_opcode_count,
            report.block_tier.reachable_loop_body_code_bytes,
        )
        .expect("write markdown");
        writeln!(
            out,
            "- Block tier: {} full callees; native {}/{} opcodes ({:.2}%) and {}/{} bytes ({:.2}%); baseline {:.2}% of opcodes",
            report.block_tier.full_callee_closure.local_function_count,
            report.block_tier.native_opcode_count,
            report.total_opcode_count,
            report.block_tier.native_opcode_percent,
            report.block_tier.native_code_bytes,
            report.total_code_bytes,
            report.block_tier.native_code_byte_percent,
            report.block_tier.baseline_opcode_percent,
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
        assert_eq!(
            report
                .direct_calls
                .iter()
                .filter(|call| call.caller == 2)
                .map(|call| (call.callee, call.loop_depth, call.reachable))
                .collect::<Vec<_>>(),
            vec![(1, 1, true), (0, 2, true)]
        );
        let loop_function = &report.functions[2];
        assert_eq!(loop_function.loop_structure_opcode_count, 4);
        assert_eq!(loop_function.in_loop_opcode_count, 7);
        assert_eq!(loop_function.reachable_in_loop_opcode_count, 7);
        assert_eq!(loop_function.loop_body_opcode_count, 4);
        assert_eq!(loop_function.reachable_loop_body_opcode_count, 4);
        assert!(loop_function.in_loop_code_bytes > loop_function.loop_body_code_bytes);
        assert_eq!(report.block_tier.full_callee_closure.members, vec![0, 1]);
        assert_eq!(report.block_tier.native_opcode_count, 10);
        assert_eq!(
            report.block_tier.native_code_bytes,
            report.functions[0].code_bytes
                + report.functions[1].code_bytes
                + loop_function.reachable_in_loop_code_bytes
        );
        assert_eq!(report.export_roots.members, vec![2]);
        assert_eq!(report.roots_closure.members, vec![0, 1, 2]);
    }

    #[test]
    fn unreachable_loop_calls_do_not_seed_block_hot_closure() {
        let report = census(
            r#"(module
                (func $dead_target call $leaf)
                (func $leaf)
                (func $live_target)
                (func $caller
                    (loop
                        (block
                            br 0
                            call $dead_target)
                        call $live_target
                        unreachable
                        call $dead_target
                        nop)))"#,
        );
        assert_eq!(
            report
                .direct_calls
                .iter()
                .filter(|call| call.caller == 3)
                .map(|call| (call.callee, call.loop_depth, call.reachable))
                .collect::<Vec<_>>(),
            vec![(0, 1, false), (2, 1, true), (0, 1, false)]
        );
        let caller = &report.functions[3];
        assert_eq!(caller.loop_structure_opcode_count, 2);
        assert_eq!(caller.in_loop_opcode_count, 9);
        assert_eq!(caller.reachable_in_loop_opcode_count, 4);
        assert_eq!(caller.loop_body_opcode_count, 8);
        assert_eq!(caller.reachable_loop_body_opcode_count, 4);
        assert_eq!(report.block_tier.full_callee_closure.members, vec![2]);
        assert_eq!(report.block_tier.native_opcode_count, 5);
        assert_eq!(report.block_tier.body_only_native_opcode_lower_bound, 4);
        assert_eq!(report.block_tier.body_only_native_opcode_upper_bound, 8);
        assert_eq!(
            report.block_tier.baseline_opcode_count,
            report.total_opcode_count - report.block_tier.native_opcode_count
        );
    }

    #[test]
    fn loop_indirect_targets_add_complete_reachable_callee_closure() {
        let report = census(
            r#"(module
                (type $t (func))
                (table 1 funcref)
                (func $leaf)
                (func $target (type $t) call $leaf)
                (func $undeclared (type $t))
                (elem (i32.const 0) $target)
                (func $caller
                    (loop
                        i32.const 0
                        call_indirect (type $t))))"#,
        );
        assert_eq!(report.indirect_calls.len(), 1);
        assert_eq!(report.indirect_calls[0].caller, 3);
        assert_eq!(report.indirect_calls[0].loop_depth, 1);
        assert!(report.indirect_calls[0].reachable);
        assert_eq!(report.conservative_indirect_targets.members, vec![1]);
        assert_eq!(report.block_tier.full_callee_closure.members, vec![0, 1]);
        assert!(!report.block_tier.full_callee_closure.members.contains(&2));
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
        assert!(json.contains("\"block_tier\""));
        let markdown = render_markdown(&[report]);
        assert!(markdown.contains("# Eager-tier structural census"));
        assert!(markdown.contains("| `fixture` |"));
        assert!(markdown.contains("Loop-only skip"));
        assert!(markdown.contains("Conservative skip"));
        assert!(markdown.contains("Block-level structural coverage"));
        assert!(markdown.contains("Body-only baseline ops"));
    }
}
