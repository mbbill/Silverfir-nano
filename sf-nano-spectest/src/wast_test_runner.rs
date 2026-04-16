//! WAST test runner adapted for sf-nano (single-module WebAssembly 2.0 interpreter)

use log::debug;
use sf_nano_core::module::entities::{FunctionDef, GlobalDef};
use sf_nano_core::module::type_context::TypeContext;
use sf_nano_core::module::Module;
use sf_nano_core::value_type::{AbstractHeapType, HeapType, RefType};
use sf_nano_core::{
    Caller, HostFn, Import, Instance, Limitable, LinkRegistry, RefHandle, Value, WasmError,
};
use std::{cell::RefCell, collections::HashMap, fmt, fs, path::Path};
use wast::{
    core::{WastArgCore, WastRetCore},
    QuoteWat, Wast, WastArg, WastDirective, WastExecute, WastInvoke, WastRet,
};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum TestError {
    Runtime { context: String, error: WasmError },
    Infrastructure(String),
}

impl fmt::Display for TestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TestError::Runtime { context, error } => write!(f, "{}, Actual: {}", context, error),
            TestError::Infrastructure(msg) => write!(f, "{}", msg),
        }
    }
}

impl TestError {
    pub fn runtime(context: String, error: WasmError) -> Self {
        TestError::Runtime { context, error }
    }

    pub fn infrastructure(msg: String) -> Self {
        TestError::Infrastructure(msg)
    }

    pub fn wasm_error(&self) -> Option<&WasmError> {
        match self {
            TestError::Runtime { error, .. } => Some(error),
            TestError::Infrastructure(_) => None,
        }
    }

    pub fn context(&self) -> Option<&str> {
        match self {
            TestError::Runtime { context, .. } => Some(context.as_str()),
            TestError::Infrastructure(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Test result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum TestResult {
    Pass,
    Fail(TestError),
    Skip(String),
    Error(String),
}

impl fmt::Display for TestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TestResult::Pass => write!(f, "PASS"),
            TestResult::Fail(err) => write!(f, "FAIL: {}", err),
            TestResult::Skip(msg) => write!(f, "SKIP: {}", msg),
            TestResult::Error(msg) => write!(f, "ERROR: {}", msg),
        }
    }
}

// ---------------------------------------------------------------------------
// Compiled module
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct CompiledModule {
    pub name: Option<String>,
    pub wasm_bytes: Vec<u8>,
}

// ---------------------------------------------------------------------------
// WastValue - simplified for WASM 2.0
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum WastValue {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    NullRef(RefType),
    FuncRef(Option<u32>),
    ExternRef(Option<u32>),
    AnyFuncRef,
    AnyExternRef,
    AnyI31Ref(RefType),
    AnyStructRef(RefType),
    AnyArrayRef(RefType),
    Ref(Option<u32>, RefType),
}

impl From<WastValue> for Value {
    fn from(wv: WastValue) -> Self {
        match wv {
            WastValue::I32(v) => Value::I32(v),
            WastValue::I64(v) => Value::I64(v),
            WastValue::F32(v) => Value::F32(v),
            WastValue::F64(v) => Value::F64(v),
            WastValue::NullRef(ref_type) => Value::Ref(RefHandle::null(), ref_type),
            WastValue::FuncRef(Some(idx)) => {
                Value::Ref(RefHandle::new(idx as usize), RefType::funcref())
            }
            WastValue::FuncRef(None) => Value::Ref(RefHandle::null(), RefType::funcref()),
            WastValue::ExternRef(Some(idx)) => {
                let externref_type = RefType::new(false, AbstractHeapType::Extern.into());
                Value::Ref(RefHandle::externref(idx as usize), externref_type)
            }
            WastValue::ExternRef(None) => Value::Ref(RefHandle::null(), RefType::externref()),
            WastValue::AnyFuncRef => {
                panic!("AnyFuncRef should not be converted to Value")
            }
            WastValue::AnyExternRef => {
                panic!("AnyExternRef should not be converted to Value")
            }
            WastValue::AnyI31Ref(_) => {
                panic!("AnyI31Ref should not be converted to Value")
            }
            WastValue::AnyStructRef(_) => {
                panic!("AnyStructRef should not be converted to Value")
            }
            WastValue::AnyArrayRef(_) => {
                panic!("AnyArrayRef should not be converted to Value")
            }
            WastValue::Ref(Some(idx), ref_type) => {
                let handle = match ref_type.heap_type {
                    HeapType::Abstract(AbstractHeapType::Any)
                    | HeapType::Abstract(AbstractHeapType::Eq) => RefHandle::hostref(idx as usize),
                    HeapType::Abstract(AbstractHeapType::Extern) => {
                        RefHandle::externref(idx as usize)
                    }
                    _ => RefHandle::new(idx as usize),
                };
                Value::Ref(handle, ref_type)
            }
            WastValue::Ref(None, ref_type) => Value::Ref(RefHandle::null(), ref_type),
        }
    }
}

fn convert_abstract_null_ref(ty: wast::core::AbstractHeapType) -> WastValue {
    use wast::core::AbstractHeapType as AHT;
    match ty {
        AHT::Func => WastValue::FuncRef(None),
        AHT::Extern => WastValue::ExternRef(None),
        AHT::NoFunc => WastValue::NullRef(RefType::nullfuncref()),
        AHT::NoExtern => WastValue::NullRef(RefType::nullexternref()),
        AHT::NoExn => WastValue::NullRef(RefType::nullexnref()),
        AHT::Any => WastValue::NullRef(RefType::anyref()),
        AHT::Eq => WastValue::NullRef(RefType::eqref()),
        AHT::I31 => WastValue::NullRef(RefType::i31ref()),
        AHT::Struct => WastValue::NullRef(RefType::structref()),
        AHT::Array => WastValue::NullRef(RefType::arrayref()),
        AHT::Exn => WastValue::NullRef(RefType::exnref()),
        AHT::None => WastValue::NullRef(RefType::nullref()),
        _ => WastValue::FuncRef(None),
    }
}

// ---------------------------------------------------------------------------
// Spectest imports
// ---------------------------------------------------------------------------

fn noop_print(_: &mut Caller, _: &[Value], _: &mut [Value]) -> Result<(), WasmError> {
    Ok(())
}
fn noop_print_i32(_: &mut Caller, _: &[Value], _: &mut [Value]) -> Result<(), WasmError> {
    Ok(())
}
fn noop_print_i64(_: &mut Caller, _: &[Value], _: &mut [Value]) -> Result<(), WasmError> {
    Ok(())
}
fn noop_print_f32(_: &mut Caller, _: &[Value], _: &mut [Value]) -> Result<(), WasmError> {
    Ok(())
}
fn noop_print_f64(_: &mut Caller, _: &[Value], _: &mut [Value]) -> Result<(), WasmError> {
    Ok(())
}
fn noop_print_i32_f32(_: &mut Caller, _: &[Value], _: &mut [Value]) -> Result<(), WasmError> {
    Ok(())
}
fn noop_print_f64_f64(_: &mut Caller, _: &[Value], _: &mut [Value]) -> Result<(), WasmError> {
    Ok(())
}

fn spectest_imports() -> Vec<Import> {
    vec![
        Import::func("spectest", "print", noop_print as HostFn),
        Import::func("spectest", "print_i32", noop_print_i32 as HostFn),
        Import::func("spectest", "print_i64", noop_print_i64 as HostFn),
        Import::func("spectest", "print_f32", noop_print_f32 as HostFn),
        Import::func("spectest", "print_f64", noop_print_f64 as HostFn),
        Import::func("spectest", "print_i32_f32", noop_print_i32_f32 as HostFn),
        Import::func("spectest", "print_f64_f64", noop_print_f64_f64 as HostFn),
        Import::global("spectest", "global_i32", Value::I32(666), false),
        Import::global("spectest", "global_i64", Value::I64(666), false),
        Import::global("spectest", "global_f32", Value::F32(666.6_f32), false),
        Import::global("spectest", "global_f64", Value::F64(666.6_f64), false),
        Import::table("spectest", "table", 10, Some(20)),
        Import::table64("spectest", "table64", 10, Some(20)),
        Import::memory("spectest", "memory", 1, Some(2)),
    ]
}

// ---------------------------------------------------------------------------
// Cross-module function forwarding (spectest-only, lives in std code)
// ---------------------------------------------------------------------------
//
// HostFn is a plain fn pointer — no closures. To forward calls to
// registered module exports, we use a thread-local slot table:
//   1. Before instantiation, allocate a slot per cross-module function import.
//   2. Each slot stores (instance_name, export_name).
//   3. Macro-generated fn pointers (fwd_00..fwd_31) each call forward_call(N).
//   4. forward_call reads the slot, finds the instance, and invokes the export.

enum ForwardingTarget {
    FunctionIndex(usize),
}

struct ForwardingSlot {
    instance_name: String,
    target: ForwardingTarget,
}

thread_local! {
    static FORWARDING_SLOTS: RefCell<Vec<Option<ForwardingSlot>>> =
        RefCell::new(Vec::new());
    // Raw pointers to instances — valid only during single-threaded test execution.
    static FORWARDING_INSTANCES: RefCell<HashMap<String, *mut Instance>> =
        RefCell::new(HashMap::new());
}

fn register_forwarding_instances(
    instances: &mut HashMap<String, Instance>,
    registered_as: &HashMap<String, String>,
) {
    FORWARDING_INSTANCES.with(|cell| {
        let mut map = cell.borrow_mut();
        map.clear();
        for (internal_name, inst) in instances.iter_mut() {
            map.insert(internal_name.clone(), inst as *mut Instance);
        }
        for (reg_name, internal_name) in registered_as {
            if let Some(inst_ptr) = map.get(internal_name).copied() {
                map.insert(reg_name.clone(), inst_ptr);
            }
        }
    });
}

fn find_exported_global_index(module: &Module, export_name: &str) -> Option<usize> {
    module
        .globals()
        .iter()
        .enumerate()
        .find_map(|(idx, global)| {
            global
                .export_names()
                .iter()
                .any(|name| name == export_name)
                .then_some(idx)
        })
}

fn find_exported_function_index(module: &Module, export_name: &str) -> Option<usize> {
    module
        .functions()
        .iter()
        .enumerate()
        .find_map(|(idx, func)| {
            func.export_names()
                .iter()
                .any(|name| name == export_name)
                .then_some(idx)
        })
}

fn find_exported_table_index(module: &Module, export_name: &str) -> Option<usize> {
    module.tables().iter().enumerate().find_map(|(idx, table)| {
        table
            .export_names()
            .iter()
            .any(|name| name == export_name)
            .then_some(idx)
    })
}

fn find_exported_memory_index(module: &Module, export_name: &str) -> Option<usize> {
    module
        .memories()
        .iter()
        .enumerate()
        .find_map(|(idx, memory)| {
            memory
                .export_names()
                .iter()
                .any(|name| name == export_name)
                .then_some(idx)
        })
}

fn alloc_forwarding_function_slot(instance_name: &str, function_index: usize) -> usize {
    alloc_forwarding_slot_target(
        instance_name,
        ForwardingTarget::FunctionIndex(function_index),
    )
}

fn alloc_forwarding_slot_target(instance_name: &str, target: ForwardingTarget) -> usize {
    FORWARDING_SLOTS.with(|cell| {
        let mut slots = cell.borrow_mut();
        let idx = slots.len();
        slots.push(Some(ForwardingSlot {
            instance_name: instance_name.to_string(),
            target,
        }));
        idx
    })
}

fn clear_forwarding() {
    FORWARDING_SLOTS.with(|cell| cell.borrow_mut().clear());
    FORWARDING_INSTANCES.with(|cell| cell.borrow_mut().clear());
}

fn forward_call(
    slot: usize,
    _caller: &mut Caller,
    args: &[Value],
    results: &mut [Value],
) -> Result<(), WasmError> {
    let (inst_name, target) = FORWARDING_SLOTS.with(|cell| {
        let slots = cell.borrow();
        match slots.get(slot).and_then(|s| s.as_ref()) {
            Some(s) => Ok((
                s.instance_name.clone(),
                match &s.target {
                    ForwardingTarget::FunctionIndex(idx) => ForwardingTarget::FunctionIndex(*idx),
                },
            )),
            None => Err(WasmError::internal("forwarding slot empty")),
        }
    })?;
    FORWARDING_INSTANCES.with(|cell| {
        let map = cell.borrow();
        match map.get(&inst_name) {
            Some(&inst_ptr) => {
                // Safety: single-threaded spectest, instance outlives the call
                let inst = unsafe { &mut *inst_ptr };
                let ret = match target {
                    ForwardingTarget::FunctionIndex(function_index) => {
                        inst.invoke_function_index(function_index, args)?
                    }
                };
                for (i, v) in ret.iter().enumerate() {
                    if i < results.len() {
                        results[i] = *v;
                    }
                }
                Ok(())
            }
            None => Err(WasmError::internal("forwarding instance not found")),
        }
    })
}

macro_rules! make_forwarder {
    ($name:ident, $n:expr) => {
        fn $name(
            caller: &mut Caller,
            args: &[Value],
            results: &mut [Value],
        ) -> Result<(), WasmError> {
            forward_call($n, caller, args, results)
        }
    };
}

make_forwarder!(fwd_00, 0);
make_forwarder!(fwd_01, 1);
make_forwarder!(fwd_02, 2);
make_forwarder!(fwd_03, 3);
make_forwarder!(fwd_04, 4);
make_forwarder!(fwd_05, 5);
make_forwarder!(fwd_06, 6);
make_forwarder!(fwd_07, 7);
make_forwarder!(fwd_08, 8);
make_forwarder!(fwd_09, 9);
make_forwarder!(fwd_10, 10);
make_forwarder!(fwd_11, 11);
make_forwarder!(fwd_12, 12);
make_forwarder!(fwd_13, 13);
make_forwarder!(fwd_14, 14);
make_forwarder!(fwd_15, 15);
make_forwarder!(fwd_16, 16);
make_forwarder!(fwd_17, 17);
make_forwarder!(fwd_18, 18);
make_forwarder!(fwd_19, 19);
make_forwarder!(fwd_20, 20);
make_forwarder!(fwd_21, 21);
make_forwarder!(fwd_22, 22);
make_forwarder!(fwd_23, 23);
make_forwarder!(fwd_24, 24);
make_forwarder!(fwd_25, 25);
make_forwarder!(fwd_26, 26);
make_forwarder!(fwd_27, 27);
make_forwarder!(fwd_28, 28);
make_forwarder!(fwd_29, 29);
make_forwarder!(fwd_30, 30);
make_forwarder!(fwd_31, 31);
make_forwarder!(fwd_32, 32);
make_forwarder!(fwd_33, 33);
make_forwarder!(fwd_34, 34);
make_forwarder!(fwd_35, 35);
make_forwarder!(fwd_36, 36);
make_forwarder!(fwd_37, 37);
make_forwarder!(fwd_38, 38);
make_forwarder!(fwd_39, 39);
make_forwarder!(fwd_40, 40);
make_forwarder!(fwd_41, 41);
make_forwarder!(fwd_42, 42);
make_forwarder!(fwd_43, 43);
make_forwarder!(fwd_44, 44);
make_forwarder!(fwd_45, 45);
make_forwarder!(fwd_46, 46);
make_forwarder!(fwd_47, 47);
make_forwarder!(fwd_48, 48);
make_forwarder!(fwd_49, 49);
make_forwarder!(fwd_50, 50);
make_forwarder!(fwd_51, 51);
make_forwarder!(fwd_52, 52);
make_forwarder!(fwd_53, 53);
make_forwarder!(fwd_54, 54);
make_forwarder!(fwd_55, 55);
make_forwarder!(fwd_56, 56);
make_forwarder!(fwd_57, 57);
make_forwarder!(fwd_58, 58);
make_forwarder!(fwd_59, 59);
make_forwarder!(fwd_60, 60);
make_forwarder!(fwd_61, 61);
make_forwarder!(fwd_62, 62);
make_forwarder!(fwd_63, 63);
make_forwarder!(fwd_64, 64);
make_forwarder!(fwd_65, 65);
make_forwarder!(fwd_66, 66);
make_forwarder!(fwd_67, 67);
make_forwarder!(fwd_68, 68);
make_forwarder!(fwd_69, 69);
make_forwarder!(fwd_70, 70);
make_forwarder!(fwd_71, 71);
make_forwarder!(fwd_72, 72);
make_forwarder!(fwd_73, 73);
make_forwarder!(fwd_74, 74);
make_forwarder!(fwd_75, 75);
make_forwarder!(fwd_76, 76);
make_forwarder!(fwd_77, 77);
make_forwarder!(fwd_78, 78);
make_forwarder!(fwd_79, 79);
make_forwarder!(fwd_80, 80);
make_forwarder!(fwd_81, 81);
make_forwarder!(fwd_82, 82);
make_forwarder!(fwd_83, 83);
make_forwarder!(fwd_84, 84);
make_forwarder!(fwd_85, 85);
make_forwarder!(fwd_86, 86);
make_forwarder!(fwd_87, 87);
make_forwarder!(fwd_88, 88);
make_forwarder!(fwd_89, 89);
make_forwarder!(fwd_90, 90);
make_forwarder!(fwd_91, 91);
make_forwarder!(fwd_92, 92);
make_forwarder!(fwd_93, 93);
make_forwarder!(fwd_94, 94);
make_forwarder!(fwd_95, 95);
make_forwarder!(fwd_96, 96);
make_forwarder!(fwd_97, 97);
make_forwarder!(fwd_98, 98);
make_forwarder!(fwd_99, 99);
make_forwarder!(fwd_100, 100);
make_forwarder!(fwd_101, 101);
make_forwarder!(fwd_102, 102);
make_forwarder!(fwd_103, 103);
make_forwarder!(fwd_104, 104);
make_forwarder!(fwd_105, 105);
make_forwarder!(fwd_106, 106);
make_forwarder!(fwd_107, 107);
make_forwarder!(fwd_108, 108);
make_forwarder!(fwd_109, 109);
make_forwarder!(fwd_110, 110);
make_forwarder!(fwd_111, 111);
make_forwarder!(fwd_112, 112);
make_forwarder!(fwd_113, 113);
make_forwarder!(fwd_114, 114);
make_forwarder!(fwd_115, 115);
make_forwarder!(fwd_116, 116);
make_forwarder!(fwd_117, 117);
make_forwarder!(fwd_118, 118);
make_forwarder!(fwd_119, 119);
make_forwarder!(fwd_120, 120);
make_forwarder!(fwd_121, 121);
make_forwarder!(fwd_122, 122);
make_forwarder!(fwd_123, 123);
make_forwarder!(fwd_124, 124);
make_forwarder!(fwd_125, 125);
make_forwarder!(fwd_126, 126);
make_forwarder!(fwd_127, 127);

const FORWARDER_TABLE: [HostFn; 128] = [
    fwd_00, fwd_01, fwd_02, fwd_03, fwd_04, fwd_05, fwd_06, fwd_07, fwd_08, fwd_09, fwd_10, fwd_11,
    fwd_12, fwd_13, fwd_14, fwd_15, fwd_16, fwd_17, fwd_18, fwd_19, fwd_20, fwd_21, fwd_22, fwd_23,
    fwd_24, fwd_25, fwd_26, fwd_27, fwd_28, fwd_29, fwd_30, fwd_31, fwd_32, fwd_33, fwd_34, fwd_35,
    fwd_36, fwd_37, fwd_38, fwd_39, fwd_40, fwd_41, fwd_42, fwd_43, fwd_44, fwd_45, fwd_46, fwd_47,
    fwd_48, fwd_49, fwd_50, fwd_51, fwd_52, fwd_53, fwd_54, fwd_55, fwd_56, fwd_57, fwd_58, fwd_59,
    fwd_60, fwd_61, fwd_62, fwd_63, fwd_64, fwd_65, fwd_66, fwd_67, fwd_68, fwd_69, fwd_70, fwd_71,
    fwd_72, fwd_73, fwd_74, fwd_75, fwd_76, fwd_77, fwd_78, fwd_79, fwd_80, fwd_81, fwd_82, fwd_83,
    fwd_84, fwd_85, fwd_86, fwd_87, fwd_88, fwd_89, fwd_90, fwd_91, fwd_92, fwd_93, fwd_94, fwd_95,
    fwd_96, fwd_97, fwd_98, fwd_99, fwd_100, fwd_101, fwd_102, fwd_103, fwd_104, fwd_105, fwd_106,
    fwd_107, fwd_108, fwd_109, fwd_110, fwd_111, fwd_112, fwd_113, fwd_114, fwd_115, fwd_116,
    fwd_117, fwd_118, fwd_119, fwd_120, fwd_121, fwd_122, fwd_123, fwd_124, fwd_125, fwd_126,
    fwd_127,
];

// ---------------------------------------------------------------------------
// WastTestRunner
// ---------------------------------------------------------------------------

pub struct WastTestRunner {
    instances: HashMap<String, Instance>,
    module_bytes: HashMap<String, Vec<u8>>,
    module_counter: u32,
    named_modules: HashMap<String, String>,
    registered_as: HashMap<String, String>,
    module_definitions: HashMap<String, Vec<u8>>,
    linked_function_refs: HashMap<(String, String, usize), usize>,
    function_registry: LinkRegistry,
    retained_failed_instances: Vec<Instance>,
}

impl WastTestRunner {
    pub fn new() -> Self {
        clear_forwarding();
        WastTestRunner {
            instances: HashMap::new(),
            module_bytes: HashMap::new(),
            module_counter: 0,
            named_modules: HashMap::new(),
            registered_as: HashMap::new(),
            module_definitions: HashMap::new(),
            linked_function_refs: HashMap::new(),
            function_registry: LinkRegistry::new(),
            retained_failed_instances: Vec::new(),
        }
    }

    /// Parse and execute a WAST file
    pub fn run_wast_file(&mut self, file_path: &Path) -> TestResult {
        let content = match fs::read_to_string(file_path) {
            Ok(content) => content,
            Err(e) => return TestResult::Error(format!("Failed to read file: {}", e)),
        };

        match self.execute_wast_content(&content) {
            Ok(()) => TestResult::Pass,
            Err(e) => TestResult::Fail(e),
        }
    }

    /// Execute WAST content as sequence of directives
    fn execute_wast_content(&mut self, content: &str) -> Result<(), TestError> {
        let mut lexer = wast::lexer::Lexer::new(content);
        lexer.allow_confusing_unicode(true);

        let buf = wast::parser::ParseBuffer::new_with_lexer(lexer)
            .map_err(|e| TestError::infrastructure(format!("Parse buffer error: {}", e)))?;
        let mut wast = wast::parser::parse::<Wast>(&buf)
            .map_err(|e| TestError::infrastructure(format!("WAST parse error: {}", e)))?;

        for (index, directive) in wast.directives.iter_mut().enumerate() {
            debug!("Executing directive {}", index);
            let span = directive.span();
            match self.execute_wast_directive(directive, index) {
                Ok(()) => {}
                Err(err) => {
                    let (line0, col0) = span.linecol_in(content);
                    let line = line0 + 1;
                    let col = col0 + 1;
                    let augmented = match err {
                        TestError::Runtime { context, error } => TestError::Runtime {
                            context: format!(
                                "{} (at line {}, col {}, directive #{})",
                                context, line, col, index
                            ),
                            error,
                        },
                        TestError::Infrastructure(msg) => TestError::Infrastructure(format!(
                            "{} (at line {}, col {}, directive #{})",
                            msg, line, col, index
                        )),
                    };
                    return Err(augmented);
                }
            }
        }

        Ok(())
    }

    /// Execute a single WAST directive
    fn execute_wast_directive(
        &mut self,
        directive: &mut WastDirective,
        index: usize,
    ) -> Result<(), TestError> {
        match directive {
            WastDirective::Module(quote_wat) => self.execute_wast_module(quote_wat, index),
            WastDirective::Invoke(invoke) => {
                debug!(
                    "Directive {} action: invoke '{}' in module '{}'",
                    index,
                    invoke.name,
                    invoke
                        .module
                        .as_ref()
                        .map(|id| id.name())
                        .unwrap_or("$last")
                );
                let _result = self.execute_wast_invoke(invoke)?;
                Ok(())
            }
            WastDirective::AssertReturn { exec, results, .. } => {
                debug!(
                    "Directive {} action: {}",
                    index,
                    self.describe_wast_action(exec)
                );
                self.execute_wast_assert_return(exec, results)
            }
            WastDirective::AssertTrap { exec, message, .. } => {
                debug!(
                    "Directive {} action: {} (expect trap: {})",
                    index,
                    self.describe_wast_action(exec),
                    message
                );
                self.execute_wast_assert_trap(exec, message)
            }
            WastDirective::AssertInvalid {
                module, message, ..
            } => self.execute_wast_assert_invalid(module, message),
            WastDirective::AssertMalformed {
                module, message, ..
            } => self.execute_wast_assert_malformed(module, message),
            WastDirective::AssertUnlinkable {
                module, message, ..
            } => self.execute_wast_assert_unlinkable(module, message),
            WastDirective::AssertExhaustion { call, message, .. } => {
                self.execute_wast_assert_exhaustion(call, message)
            }
            WastDirective::Register { name, module, .. } => {
                self.execute_wast_register(name, module.as_ref())
            }
            WastDirective::ModuleDefinition(quote_wat) => {
                self.execute_wast_module_definition(quote_wat, index)
            }
            WastDirective::ModuleInstance {
                instance, module, ..
            } => {
                let instance_name = instance.as_ref().map(|id| id.name());
                let module_name = module.as_ref().map(|id| id.name());
                self.execute_wast_module_instance(instance_name, module_name, index)
            }
            _ => Ok(()),
        }
    }

    // -----------------------------------------------------------------------
    // Module loading
    // -----------------------------------------------------------------------

    fn execute_wast_module(
        &mut self,
        quote_wat: &mut QuoteWat,
        _index: usize,
    ) -> Result<(), TestError> {
        let compiled = self.compile_quote_wat(quote_wat).map_err(|e| {
            TestError::infrastructure(format!(
                "Expected: successful module compilation, Actual: {}",
                e
            ))
        })?;
        self.load_and_instantiate_module(compiled).map_err(|e| {
            TestError::runtime("successful load and instantiation of module".to_string(), e)
        })?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Invoke
    // -----------------------------------------------------------------------

    fn execute_wast_invoke(&mut self, invoke: &WastInvoke) -> Result<Vec<Value>, TestError> {
        // Refresh forwarding pointers (HashMap may have been modified since last registration)
        register_forwarding_instances(&mut self.instances, &self.registered_as);
        self.sync_registered_imports_from_sources();

        let internal_name = self
            .resolve_module_name(invoke.module.as_ref())
            .map_err(TestError::infrastructure)?;

        let args: Vec<Value> = self
            .convert_wast_args(&invoke.args)
            .into_iter()
            .map(|arg| arg.into())
            .collect();

        let result = {
            let instance = self.instances.get_mut(&internal_name).ok_or_else(|| {
                TestError::infrastructure(format!("Instance '{}' not found", internal_name))
            })?;
            instance
                .invoke(invoke.name, &args)
                .map(|v| v.into_iter().collect())
        };

        self.sync_registered_imports_back_to_sources(&internal_name);
        self.sync_registered_imports_from_sources();

        result.map_err(|e| {
            TestError::runtime(
                format!("successful invocation of function '{}'", invoke.name),
                e,
            )
        })
    }

    // -----------------------------------------------------------------------
    // assert_return
    // -----------------------------------------------------------------------

    fn execute_wast_assert_return(
        &mut self,
        exec: &mut WastExecute,
        expected: &[WastRet],
    ) -> Result<(), TestError> {
        let action_description = self.describe_wast_action(exec);
        let actual = self.execute_wast_action(exec)?;
        let expected_values = self.convert_wast_returns(expected);

        if actual.len() != expected_values.len() {
            return Err(TestError::infrastructure(format!(
                "Expected: {} results for {}, Actual: {} results {:?}",
                expected_values.len(),
                action_description,
                actual.len(),
                actual
            )));
        }

        for (i, (actual_val, expected_val)) in actual.iter().zip(expected_values.iter()).enumerate()
        {
            if !values_equal_with_nan(actual_val, expected_val) {
                return Err(TestError::infrastructure(format!(
                    "Expected: {:?} for {} result {}, Actual: {:?}",
                    expected_val, action_description, i, actual_val
                )));
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // assert_trap
    // -----------------------------------------------------------------------

    fn execute_wast_assert_trap(
        &mut self,
        exec: &mut WastExecute,
        expected_message: &str,
    ) -> Result<(), TestError> {
        let action_description = self.describe_wast_action(exec);
        match self.execute_wast_action(exec) {
            Ok(results) => Err(TestError::infrastructure(format!(
                "Expected: trap with error '{}' for {}, Actual: execution succeeded with results {:?}",
                expected_message, action_description, results
            ))),
            Err(_) => Ok(()),
        }
    }

    // -----------------------------------------------------------------------
    // assert_exhaustion
    // -----------------------------------------------------------------------

    fn execute_wast_assert_exhaustion(
        &mut self,
        invoke: &mut WastInvoke,
        expected_message: &str,
    ) -> Result<(), TestError> {
        let module_name = invoke
            .module
            .as_ref()
            .map(|id| id.name())
            .unwrap_or("<current>");
        let action_description = format!("invoke '{}' in module '{}'", invoke.name, module_name);

        match self.execute_wast_invoke(invoke) {
            Ok(results) => Err(TestError::infrastructure(format!(
                "Expected: {} for {}, Actual: execution succeeded with results {:?}",
                expected_message, action_description, results
            ))),
            Err(_) => Ok(()),
        }
    }

    // -----------------------------------------------------------------------
    // assert_invalid
    // -----------------------------------------------------------------------

    fn execute_wast_assert_invalid(
        &mut self,
        quote_wat: &mut QuoteWat,
        expected_message: &str,
    ) -> Result<(), TestError> {
        match self.compile_quote_wat(quote_wat) {
            Ok(compiled) => {
                match self.try_instantiate_temp(&compiled.wasm_bytes) {
                    Ok(_) => Err(TestError::infrastructure(format!(
                        "Expected: invalid module with error '{}', Actual: validation and instantiation succeeded",
                        expected_message
                    ))),
                    Err(_) => Ok(()),
                }
            }
            Err(_) => Ok(()),
        }
    }

    // -----------------------------------------------------------------------
    // assert_malformed
    // -----------------------------------------------------------------------

    fn execute_wast_assert_malformed(
        &mut self,
        quote_wat: &mut QuoteWat,
        expected_message: &str,
    ) -> Result<(), TestError> {
        match self.compile_quote_wat(quote_wat) {
            Ok(compiled) => {
                let bytes = compiled.wasm_bytes.clone();
                match Module::new("test_malformed", &bytes) {
                    Ok(module) => {
                        let imports = self.build_imports(&bytes);
                        match Instance::from_module(module, &imports) {
                            Ok(_) => Err(TestError::infrastructure(format!(
                                "Expected: malformed module with error '{}', Actual: WASM parsing succeeded ({} bytes)",
                                expected_message, compiled.wasm_bytes.len()
                            ))),
                            Err(_) => Ok(()),
                        }
                    }
                    Err(_) => Ok(()),
                }
            }
            Err(_) => Ok(()),
        }
    }

    // -----------------------------------------------------------------------
    // assert_unlinkable
    // -----------------------------------------------------------------------

    fn execute_wast_assert_unlinkable(
        &mut self,
        wat: &mut wast::Wat,
        expected_message: &str,
    ) -> Result<(), TestError> {
        match wat {
            wast::Wat::Module(ref mut module) => {
                match module.encode() {
                    Ok(wasm_bytes) => {
                        match self.try_instantiate_temp(&wasm_bytes) {
                            Ok(_) => Err(TestError::infrastructure(format!(
                                "Expected: unlinkable module with error '{}', Actual: instantiation succeeded",
                                expected_message
                            ))),
                            Err(_) => Ok(()),
                        }
                    }
                    Err(_) => Ok(()),
                }
            }
            _ => Err(TestError::infrastructure(
                "Component unlinkable tests not supported yet".to_string(),
            )),
        }
    }

    // -----------------------------------------------------------------------
    // register
    // -----------------------------------------------------------------------

    fn execute_wast_register(
        &mut self,
        name: &str,
        module: Option<&wast::token::Id>,
    ) -> Result<(), TestError> {
        let internal_name = match module {
            Some(id) => {
                let named = id.name();
                self.named_modules
                    .get(named)
                    .ok_or_else(|| {
                        TestError::infrastructure(format!("Named module '{}' not found", named))
                    })?
                    .clone()
            }
            None => self.last_module_name(),
        };

        self.registered_as.insert(name.to_string(), internal_name);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Module definition / instance (module linking)
    // -----------------------------------------------------------------------

    fn execute_wast_module_definition(
        &mut self,
        quote_wat: &mut QuoteWat,
        index: usize,
    ) -> Result<(), TestError> {
        let compiled = self
            .compile_quote_wat(quote_wat)
            .map_err(TestError::infrastructure)?;

        let temp_name = compiled
            .name
            .clone()
            .unwrap_or_else(|| format!("_temp_def_{}", index));
        Module::new(&temp_name, &compiled.wasm_bytes).map_err(|e| {
            TestError::infrastructure(format!("Module definition validation failed: {}", e))
        })?;

        if let Some(module_name) = compiled.name {
            self.module_definitions
                .insert(module_name, compiled.wasm_bytes);
        }

        Ok(())
    }

    fn execute_wast_module_instance(
        &mut self,
        instance_name: Option<&str>,
        module_name: Option<&str>,
        _index: usize,
    ) -> Result<(), TestError> {
        let instance_name = instance_name.ok_or_else(|| {
            TestError::infrastructure("Module instance must have a name".to_string())
        })?;

        let module_name = module_name.ok_or_else(|| {
            TestError::infrastructure(
                "Module instance must reference a module definition".to_string(),
            )
        })?;

        let wasm_bytes = self
            .module_definitions
            .get(module_name)
            .ok_or_else(|| {
                TestError::infrastructure(format!("Module definition '{}' not found", module_name))
            })?
            .clone();

        let compiled = CompiledModule {
            wasm_bytes,
            name: Some(instance_name.to_string()),
        };

        let internal_name = self.load_and_instantiate_module(compiled).map_err(|e| {
            TestError::infrastructure(format!("Failed to instantiate module: {}", e))
        })?;

        self.named_modules
            .insert(instance_name.to_string(), internal_name);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Action execution
    // -----------------------------------------------------------------------

    fn execute_wast_action(&mut self, exec: &mut WastExecute) -> Result<Vec<Value>, TestError> {
        match exec {
            WastExecute::Invoke(invoke) => self.execute_wast_invoke(invoke),
            WastExecute::Get { module, global, .. } => {
                self.sync_registered_imports_from_sources();
                let internal_name = self
                    .resolve_module_name(module.as_ref())
                    .map_err(TestError::infrastructure)?;
                let instance = self.instances.get(&internal_name).ok_or_else(|| {
                    TestError::infrastructure(format!("Instance '{}' not found", internal_name))
                })?;
                let value = instance.get_global(global).ok_or_else(|| {
                    TestError::infrastructure(format!(
                        "Global '{}' not found in instance '{}'",
                        global, internal_name
                    ))
                })?;
                Ok(vec![value])
            }
            WastExecute::Wat(wat) => match wat {
                wast::Wat::Module(module) => match module.encode() {
                    Ok(wasm_bytes) => {
                        register_forwarding_instances(&mut self.instances, &self.registered_as);
                        match self.instantiate_with_registry(&wasm_bytes, true) {
                            Ok(_instance) => Ok(vec![]),
                            Err(e) => Err(TestError::runtime(
                                "successful module instantiation".to_string(),
                                e,
                            )),
                        }
                    }
                    Err(e) => Err(TestError::infrastructure(format!(
                        "Module encoding failed: {}",
                        e
                    ))),
                },
                _ => Err(TestError::infrastructure(
                    "Component execution not supported yet".to_string(),
                )),
            },
        }
    }

    // -----------------------------------------------------------------------
    // Compilation and instantiation helpers
    // -----------------------------------------------------------------------

    fn compile_quote_wat(&self, quote_wat: &mut QuoteWat) -> Result<CompiledModule, String> {
        match quote_wat {
            QuoteWat::Wat(wast::Wat::Module(ref mut module)) => {
                let name = module.id.as_ref().map(|id| id.name().to_string());
                match module.encode() {
                    Ok(wasm_bytes) => Ok(CompiledModule { name, wasm_bytes }),
                    Err(e) => Err(format!("Failed to encode module: {}", e)),
                }
            }
            QuoteWat::Wat(wast::Wat::Component(_)) => {
                Err("WebAssembly components not supported yet".to_string())
            }
            QuoteWat::QuoteModule(_source, data) => {
                if data.is_empty() {
                    return Err("Empty quote module data".to_string());
                }

                let mut wat_source = String::new();
                for (_span, bytes) in data {
                    wat_source.push_str(
                        std::str::from_utf8(bytes)
                            .map_err(|e| format!("Invalid UTF-8 in quoted module: {}", e))?,
                    );
                }

                debug!("Compiling quoted WAT source: {}", wat_source.trim());

                match wat::parse_str(&wat_source) {
                    Ok(wasm_bytes) => Ok(CompiledModule {
                        name: None,
                        wasm_bytes,
                    }),
                    Err(e) => Err(format!("Failed to compile quoted WAT: {}", e)),
                }
            }
            QuoteWat::QuoteComponent(_, _) => {
                Err("WebAssembly components not supported yet".to_string())
            }
        }
    }

    fn load_and_instantiate_module(
        &mut self,
        compiled: CompiledModule,
    ) -> Result<String, WasmError> {
        let internal_name = format!("module_{}", self.module_counter);
        self.module_counter += 1;

        register_forwarding_instances(&mut self.instances, &self.registered_as);
        let instance = self.instantiate_with_registry(&compiled.wasm_bytes, false)?;

        self.instances.insert(internal_name.clone(), instance);
        self.module_bytes
            .insert(internal_name.clone(), compiled.wasm_bytes);

        if let Some(name) = compiled.name {
            self.named_modules.insert(name, internal_name.clone());
        }

        self.sync_registered_imports_back_to_sources(&internal_name);
        self.sync_registered_imports_from_sources();

        Ok(internal_name)
    }

    /// Try to instantiate a module temporarily (for assert_invalid/assert_unlinkable).
    fn try_instantiate_temp(&mut self, wasm_bytes: &[u8]) -> Result<Instance, WasmError> {
        register_forwarding_instances(&mut self.instances, &self.registered_as);
        self.instantiate_with_registry(wasm_bytes, true)
    }

    fn instantiate_with_registry(
        &mut self,
        wasm_bytes: &[u8],
        retain_partial: bool,
    ) -> Result<Instance, WasmError> {
        let imports = self.build_imports(wasm_bytes);
        let module = Module::new("main", wasm_bytes)?;
        match Instance::from_module_with_registry(module, &imports, &self.function_registry) {
            Ok(instance) => Ok(instance),
            Err(err) => {
                let (partial, error) = err.into_parts();
                if retain_partial {
                    if let Some(instance) = partial {
                        self.retained_failed_instances.push(instance);
                    }
                }
                Err(error)
            }
        }
    }

    /// Build imports for a module by providing spectest imports plus exports
    /// Build imports for instantiation, forwarding cross-module function calls
    /// via thread-local slot table.
    fn build_imports(&self, wasm_bytes: &[u8]) -> Vec<Import> {
        let mut imports = spectest_imports();

        // For each registered module, provide its exports as imports.
        for (registered_name, internal_name) in &self.registered_as {
            if let Some(instance) = self.instances.get(internal_name) {
                if let Some(bytes) = self.module_bytes.get(internal_name) {
                    if let Ok(module) = Module::new("_export_scan", bytes) {
                        // Global exports — read current value from live instance
                        for global in module.globals() {
                            for export_name in global.export_names() {
                                if let Some(value) = instance.get_global(export_name) {
                                    imports.push(Import::global_with_linked_function(
                                        registered_name,
                                        export_name,
                                        value,
                                        global.mutable(),
                                        None,
                                    ));
                                }
                            }
                        }

                        // Function exports — use shared linked handles
                        for func in module.functions() {
                            for export_name in func.export_names() {
                                let ft = func.func_type().clone();
                                let type_ctx = module.types().clone();
                                if let Some(func_idx) =
                                    find_exported_function_index(&module, export_name)
                                {
                                    if let Some(handle) = instance.function_handle_at(func_idx) {
                                        imports.push(Import::linked_func_typed_with_context(
                                            registered_name,
                                            export_name,
                                            handle,
                                            ft,
                                            type_ctx.clone(),
                                        ));
                                    }
                                }
                            }
                        }

                        // Tag exports — link-time only, no runtime handle yet
                        for tag in module.tags() {
                            for export_name in tag.export_names() {
                                imports.push(Import::tag_typed_with_context(
                                    registered_name,
                                    export_name,
                                    tag.func_type().clone(),
                                    module.types().clone(),
                                ));
                            }
                        }

                        // Table exports — use live instance sizes
                        for table in module.tables() {
                            for export_name in table.export_names() {
                                let current_size = instance
                                    .table_size(export_name)
                                    .unwrap_or(table.limits().min());
                                let state = find_exported_table_index(&module, export_name)
                                    .and_then(|table_idx| {
                                        instance.shared_table_state_at(table_idx)
                                    });
                                let import = if table.limits().is64 {
                                    Import::table_with_state(
                                        registered_name,
                                        export_name,
                                        sf_nano_core::Limits::new_64(
                                            current_size,
                                            table.limits().max(),
                                        )
                                        .expect("registered table export limits should stay valid"),
                                        state,
                                    )
                                } else {
                                    Import::table_with_state(
                                        registered_name,
                                        export_name,
                                        sf_nano_core::Limits::new(
                                            current_size,
                                            table.limits().max(),
                                        )
                                        .expect("registered table export limits should stay valid"),
                                        state,
                                    )
                                };
                                imports.push(import);
                            }
                        }

                        // Memory exports — use live instance sizes
                        for memory in module.memories() {
                            for export_name in memory.export_names() {
                                let current_pages = instance
                                    .memory_pages(export_name)
                                    .unwrap_or(memory.limits().min());
                                let shared_memory =
                                    find_exported_memory_index(&module, export_name)
                                        .and_then(|mem_idx| instance.shared_memory_at(mem_idx));
                                let import = if memory.limits().is64 {
                                    Import::memory_with_state(
                                        registered_name,
                                        export_name,
                                        sf_nano_core::Limits::new_64(
                                            current_pages,
                                            memory.limits().max(),
                                        )
                                        .expect(
                                            "registered memory export limits should stay valid",
                                        ),
                                        shared_memory,
                                    )
                                } else {
                                    Import::memory_with_state(
                                        registered_name,
                                        export_name,
                                        sf_nano_core::Limits::new(
                                            current_pages,
                                            memory.limits().max(),
                                        )
                                        .expect(
                                            "registered memory export limits should stay valid",
                                        ),
                                        shared_memory,
                                    )
                                };
                                imports.push(import);
                            }
                        }
                    }
                }
            }
        }

        // Provide stubs/forwarders for imports from non-registered named modules
        if let Ok(module) = Module::new("_import_scan", wasm_bytes) {
            for func in module.functions() {
                if let FunctionDef::Import {
                    module: ref mod_name,
                    ref name,
                    ..
                } = *func.def()
                {
                    let import_name = name.as_str();
                    let mod_name = mod_name.as_str();
                    if mod_name == "spectest" || self.registered_as.contains_key(mod_name) {
                        continue;
                    }
                    if let Some(internal) = self.named_modules.get(mod_name) {
                        if let Some(inst) = self.instances.get(internal) {
                            if let Some(value) = inst.get_global(import_name) {
                                imports.push(Import::global(mod_name, import_name, value, false));
                            } else {
                                fn fallback_stub(
                                    _: &mut Caller,
                                    _: &[Value],
                                    _: &mut [Value],
                                ) -> Result<(), WasmError> {
                                    Ok(())
                                }
                                imports.push(Import::func(
                                    mod_name,
                                    import_name,
                                    fallback_stub as HostFn,
                                ));
                            }
                        }
                    }
                }
            }
        }

        imports
    }

    fn sync_registered_imports_from_sources(&mut self) {
        let mut global_ops = Vec::new();

        let module_entries: Vec<_> = self
            .module_bytes
            .iter()
            .map(|(name, bytes)| (name.clone(), bytes.clone()))
            .collect();

        for (dst_internal, bytes) in module_entries {
            let Ok(module) = Module::new("_sync_imports_dst", &bytes) else {
                continue;
            };

            for (dst_idx, global) in module.globals().iter().enumerate() {
                let GlobalDef::Import { module, name, .. } = global.def() else {
                    continue;
                };
                let Some(src_internal) = self.registered_as.get(module.as_str()) else {
                    continue;
                };
                let src_internal = src_internal.clone();
                let Some(src_bytes) = self.module_bytes.get(&src_internal) else {
                    continue;
                };
                let Ok(src_module) = Module::new("_sync_imports_src", src_bytes) else {
                    continue;
                };
                let Some(src_idx) = find_exported_global_index(&src_module, name.as_str()) else {
                    continue;
                };
                let Some(value) = self
                    .instances
                    .get(&src_internal)
                    .and_then(|instance| instance.global_at(src_idx))
                else {
                    continue;
                };
                let value = self.remap_global_value(&src_internal, &dst_internal, value);
                global_ops.push((dst_internal.clone(), dst_idx, value));
            }
        }

        for (dst_internal, dst_idx, value) in global_ops {
            if let Some(instance) = self.instances.get_mut(&dst_internal) {
                let _ = instance.replace_global_at(dst_idx, value);
            }
        }
    }

    fn sync_registered_imports_back_to_sources(&mut self, src_internal: &str) {
        let mut global_ops = Vec::new();

        let Some(bytes) = self.module_bytes.get(src_internal) else {
            return;
        };
        let Ok(module) = Module::new("_sync_exports_src", bytes) else {
            return;
        };

        for (src_idx, global) in module.globals().iter().enumerate() {
            let GlobalDef::Import {
                module,
                name,
                mutable,
                ..
            } = global.def()
            else {
                continue;
            };
            if !*mutable {
                continue;
            }
            let Some(dst_internal) = self.registered_as.get(module.as_str()) else {
                continue;
            };
            let dst_internal = dst_internal.clone();
            let Some(dst_bytes) = self.module_bytes.get(&dst_internal) else {
                continue;
            };
            let Ok(dst_module) = Module::new("_sync_exports_dst", dst_bytes) else {
                continue;
            };
            let Some(dst_idx) = find_exported_global_index(&dst_module, name.as_str()) else {
                continue;
            };
            let Some(value) = self
                .instances
                .get(src_internal)
                .and_then(|instance| instance.global_at(src_idx))
            else {
                continue;
            };
            let value = self.remap_global_value(src_internal, &dst_internal, value);
            global_ops.push((dst_internal, dst_idx, value));
        }

        for (dst_internal, dst_idx, value) in global_ops {
            if let Some(instance) = self.instances.get_mut(&dst_internal) {
                let _ = instance.replace_global_at(dst_idx, value);
            }
        }
    }

    fn remap_table_ref(
        &mut self,
        src_internal: &str,
        dst_internal: &str,
        handle: RefHandle,
    ) -> RefHandle {
        if handle.is_null() || handle.is_extern() {
            return handle;
        }

        let src_func_idx = handle.payload();
        let key = (
            dst_internal.to_string(),
            src_internal.to_string(),
            src_func_idx,
        );
        if let Some(&dst_func_idx) = self.linked_function_refs.get(&key) {
            return RefHandle::new(dst_func_idx);
        }

        let Some(func_type) = self
            .instances
            .get(src_internal)
            .and_then(|instance| instance.function_type_at(src_func_idx))
        else {
            return handle;
        };

        let slot = alloc_forwarding_function_slot(src_internal, src_func_idx);
        let Some(&callback) = FORWARDER_TABLE.get(slot) else {
            return handle;
        };

        let Some(dst_instance) = self.instances.get_mut(dst_internal) else {
            return handle;
        };
        let dst_func_idx = dst_instance.append_host_function(func_type, callback);
        self.linked_function_refs.insert(key, dst_func_idx);
        RefHandle::new(dst_func_idx)
    }

    fn remap_global_value(
        &mut self,
        src_internal: &str,
        dst_internal: &str,
        value: Value,
    ) -> Value {
        match value {
            Value::Ref(handle, ref_type) if !handle.is_null() && !handle.is_extern() => Value::Ref(
                self.remap_table_ref(src_internal, dst_internal, handle),
                ref_type,
            ),
            other => other,
        }
    }

    // -----------------------------------------------------------------------
    // Name resolution
    // -----------------------------------------------------------------------

    fn resolve_module_name(&self, module: Option<&wast::token::Id>) -> Result<String, String> {
        match module {
            Some(id) => {
                let name = id.name();
                self.named_modules
                    .get(name)
                    .cloned()
                    .or_else(|| self.instances.get(name).map(|_| name.to_string()))
                    .ok_or_else(|| format!("Module '{}' not found", name))
            }
            None => Ok(self.last_module_name()),
        }
    }

    fn last_module_name(&self) -> String {
        format!("module_{}", self.module_counter.saturating_sub(1))
    }

    // -----------------------------------------------------------------------
    // WAST arg/ret conversion (WASM 2.0 only)
    // -----------------------------------------------------------------------

    fn convert_wast_args(&self, args: &[WastArg]) -> Vec<WastValue> {
        args.iter()
            .filter_map(|arg| self.convert_wast_arg(arg))
            .collect()
    }

    fn convert_wast_arg(&self, arg: &WastArg) -> Option<WastValue> {
        match arg {
            WastArg::Core(core_arg) => self.convert_core_arg(core_arg),
            _ => None,
        }
    }

    fn convert_core_arg(&self, arg: &WastArgCore) -> Option<WastValue> {
        match arg {
            WastArgCore::I32(val) => Some(WastValue::I32(*val)),
            WastArgCore::I64(val) => Some(WastValue::I64(*val)),
            WastArgCore::F32(f32_val) => Some(WastValue::F32(f32::from_bits(f32_val.bits))),
            WastArgCore::F64(f64_val) => Some(WastValue::F64(f64::from_bits(f64_val.bits))),
            WastArgCore::RefNull(ref_type) => match ref_type {
                wast::core::HeapType::Abstract { ty, .. } => Some(convert_abstract_null_ref(*ty)),
                _ => Some(WastValue::FuncRef(None)),
            },
            WastArgCore::RefExtern(idx) => Some(WastValue::ExternRef(Some(*idx))),
            WastArgCore::RefHost(idx) => Some(WastValue::Ref(Some(*idx), RefType::anyref())),
            _ => None,
        }
    }

    fn convert_wast_returns(&self, returns: &[WastRet]) -> Vec<WastValue> {
        returns
            .iter()
            .filter_map(|ret| self.convert_wast_ret(ret))
            .collect()
    }

    fn convert_wast_ret(&self, ret: &WastRet) -> Option<WastValue> {
        match ret {
            WastRet::Core(core_ret) => self.convert_core_ret(core_ret),
            _ => None,
        }
    }

    fn convert_core_ret(&self, ret: &WastRetCore) -> Option<WastValue> {
        match ret {
            WastRetCore::I32(val) => Some(WastValue::I32(*val)),
            WastRetCore::I64(val) => Some(WastValue::I64(*val)),
            WastRetCore::F32(nan_pattern) => match nan_pattern {
                wast::core::NanPattern::Value(f32_val) => {
                    Some(WastValue::F32(f32::from_bits(f32_val.bits)))
                }
                wast::core::NanPattern::CanonicalNan => Some(WastValue::F32(f32::NAN)),
                wast::core::NanPattern::ArithmeticNan => Some(WastValue::F32(f32::NAN)),
            },
            WastRetCore::F64(nan_pattern) => match nan_pattern {
                wast::core::NanPattern::Value(f64_val) => {
                    Some(WastValue::F64(f64::from_bits(f64_val.bits)))
                }
                wast::core::NanPattern::CanonicalNan => Some(WastValue::F64(f64::NAN)),
                wast::core::NanPattern::ArithmeticNan => Some(WastValue::F64(f64::NAN)),
            },
            WastRetCore::RefNull(opt_ref_type) => match opt_ref_type {
                Some(wast::core::HeapType::Abstract { ty, .. }) => {
                    Some(convert_abstract_null_ref(*ty))
                }
                _ => Some(WastValue::FuncRef(None)),
            },
            WastRetCore::RefExtern(opt_idx) => match opt_idx {
                Some(idx) => Some(WastValue::ExternRef(Some(*idx))),
                None => Some(WastValue::AnyExternRef),
            },
            WastRetCore::RefHost(idx) => Some(WastValue::Ref(
                Some(*idx),
                RefType::new(true, AbstractHeapType::Any.into()),
            )),
            WastRetCore::RefFunc(opt_idx) => match opt_idx {
                Some(idx) => match idx {
                    wast::token::Index::Num(n, _) => Some(WastValue::FuncRef(Some(*n))),
                    _ => None,
                },
                None => Some(WastValue::AnyFuncRef),
            },
            WastRetCore::RefI31 => Some(WastValue::AnyI31Ref(RefType::new(
                false,
                AbstractHeapType::I31.into(),
            ))),
            WastRetCore::RefStruct => Some(WastValue::AnyStructRef(RefType::new(
                false,
                AbstractHeapType::Struct.into(),
            ))),
            WastRetCore::RefArray => Some(WastValue::AnyArrayRef(RefType::new(
                false,
                AbstractHeapType::Array.into(),
            ))),
            WastRetCore::RefAny => Some(WastValue::Ref(
                None,
                RefType::new(false, AbstractHeapType::Any.into()),
            )),
            WastRetCore::RefEq => Some(WastValue::Ref(
                None,
                RefType::new(false, AbstractHeapType::Eq.into()),
            )),
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Description helper
    // -----------------------------------------------------------------------

    fn describe_wast_action(&self, exec: &WastExecute) -> String {
        match exec {
            WastExecute::Invoke(invoke) => {
                let module_name = invoke
                    .module
                    .as_ref()
                    .map(|id| id.name())
                    .unwrap_or("<current>");
                format!("invoke '{}' in module '{}'", invoke.name, module_name)
            }
            WastExecute::Get { module, global, .. } => {
                let module_name = module.as_ref().map(|id| id.name()).unwrap_or("<current>");
                format!("get global '{}' from module '{}'", global, module_name)
            }
            _ => "unsupported action".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// NaN-aware value comparison
// ---------------------------------------------------------------------------

fn values_equal_with_nan(actual: &Value, expected: &WastValue) -> bool {
    match (actual, expected) {
        (Value::I32(a), WastValue::I32(e)) => a == e,
        (Value::I64(a), WastValue::I64(e)) => a == e,
        (Value::F32(a), WastValue::F32(e)) => {
            if a.is_nan() && e.is_nan() {
                true
            } else {
                a == e
            }
        }
        (Value::F64(a), WastValue::F64(e)) => {
            if a.is_nan() && e.is_nan() {
                true
            } else {
                a == e
            }
        }
        (Value::Ref(actual_ref, ref_type), WastValue::FuncRef(expected_ref))
            if ref_type.is_funcref() =>
        {
            match (actual_ref, expected_ref) {
                (ref_val, Some(expected_idx)) => {
                    !ref_val.is_null() && ref_val.payload() == *expected_idx as usize
                }
                (ref_val, None) => ref_val.is_null(),
            }
        }
        (Value::Ref(actual_ref, _), WastValue::FuncRef(None)) => actual_ref.is_null(),
        (Value::Ref(actual_ref, ref_type), WastValue::NullRef(expected_type)) => {
            actual_ref.is_null()
                && (ref_type.is_subtype_of(expected_type, &TypeContext::empty())
                    || expected_type.is_subtype_of(&ref_type, &TypeContext::empty()))
                || (actual_ref.is_null()
                    && ((ref_type.is_funcref() && expected_type.is_funcref())
                        || (ref_type.is_externref() && expected_type.is_externref())))
        }
        (Value::Ref(actual_ref, _), WastValue::AnyFuncRef) => !actual_ref.is_null(),
        (Value::Ref(actual_ref, ref_type), WastValue::AnyExternRef) if ref_type.is_externref() => {
            !actual_ref.is_null()
        }
        (Value::Ref(actual_ref, actual_rt), WastValue::AnyI31Ref(expected_rt)) => {
            if actual_ref.is_null() {
                return false;
            }
            actual_rt.heap_type == expected_rt.heap_type
                || matches!(
                    actual_rt.heap_type,
                    HeapType::Abstract(AbstractHeapType::Any)
                        | HeapType::Abstract(AbstractHeapType::Eq)
                )
        }
        (Value::Ref(actual_ref, actual_rt), WastValue::AnyStructRef(_)) => {
            if actual_ref.is_null() {
                return false;
            }
            match actual_rt.heap_type {
                HeapType::Abstract(AbstractHeapType::Struct)
                | HeapType::Abstract(AbstractHeapType::Any)
                | HeapType::Abstract(AbstractHeapType::Eq)
                | HeapType::Concrete(_) => true,
                _ => false,
            }
        }
        (Value::Ref(actual_ref, actual_rt), WastValue::AnyArrayRef(_)) => {
            if actual_ref.is_null() {
                return false;
            }
            match actual_rt.heap_type {
                HeapType::Abstract(AbstractHeapType::Array)
                | HeapType::Abstract(AbstractHeapType::Any)
                | HeapType::Abstract(AbstractHeapType::Eq)
                | HeapType::Concrete(_) => true,
                _ => false,
            }
        }
        (Value::Ref(actual_ref, ref_type), WastValue::ExternRef(expected_ref))
            if ref_type.is_externref() =>
        {
            match (actual_ref, expected_ref) {
                (ref_val, Some(expected_idx)) => {
                    !ref_val.is_null() && ref_val.payload() == *expected_idx as usize
                }
                (ref_val, None) => ref_val.is_null(),
            }
        }
        (Value::Ref(actual_ref, actual_rt), WastValue::Ref(expected_ref, expected_rt)) => {
            match (actual_ref, expected_ref) {
                (ref_val, Some(expected_idx)) => {
                    if ref_val.is_null() || *actual_rt != *expected_rt {
                        false
                    } else {
                        ref_val.payload() == *expected_idx as usize
                    }
                }
                (ref_val, None) => ref_val.is_null(),
            }
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sf_nano_core::module::Module;
    use sf_nano_core::set_reference_backend;
    use sf_nano_core::{set_backend_mode, BackendMode, FunctionInst, Value};
    use std::path::PathBuf;

    fn instantiate_first_module_with_backend(path: &str, backend: BackendMode) -> WastTestRunner {
        set_backend_mode(backend);
        let full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("target")
            .join("webassembly-testsuite")
            .join(path);
        let content = fs::read_to_string(&full_path).expect("read wast");
        let mut lexer = wast::lexer::Lexer::new(&content);
        lexer.allow_confusing_unicode(true);
        let buf = wast::parser::ParseBuffer::new_with_lexer(lexer).expect("parse buffer");
        let mut wast = wast::parser::parse::<Wast>(&buf).expect("parse wast");

        let mut runner = WastTestRunner::new();
        let directive = wast.directives.first_mut().expect("module directive");
        match directive {
            WastDirective::Module(quote_wat) => {
                runner
                    .execute_wast_module(quote_wat, 0)
                    .expect("instantiate module");
            }
            _ => panic!("expected first directive to be a module"),
        }

        runner
    }

    fn instantiate_first_module(path: &str) -> WastTestRunner {
        instantiate_first_module_with_backend(path, BackendMode::Auto)
    }

    fn run_wast_fixture(path: &str) -> TestResult {
        set_backend_mode(BackendMode::Auto);
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("target")
            .join("webassembly-testsuite")
            .join(path);
        let mut runner = WastTestRunner::new();
        runner.run_wast_file(&path)
    }

    #[test]
    fn regress_if_as_br_table_last_true() {
        let mut runner = instantiate_first_module("if.wast");
        let instance = runner.instances.values_mut().next().expect("instance");
        let ret = instance
            .invoke("as-br_table-last", &[Value::I32(1)])
            .expect("invoke export");
        assert_eq!(ret, vec![Value::I32(2)]);
    }

    #[test]
    fn regress_br_table_as_if_else_false() {
        let mut runner = instantiate_first_module("br_table.wast");
        let instance = runner.instances.values_mut().next().expect("instance");
        let ret = instance
            .invoke("as-if-else", &[Value::I32(0), Value::I32(6)])
            .expect("invoke export");
        assert_eq!(ret, vec![Value::I32(4)]);
    }

    #[test]
    fn regress_memory_redundancy_malloc_aliasing() {
        let mut runner = instantiate_first_module("memory_redundancy.wast");
        let instance = runner.instances.values_mut().next().expect("instance");
        let ret = instance
            .invoke("malloc_aliasing", &[])
            .expect("invoke export");
        assert_eq!(ret, vec![Value::I32(43)]);
    }

    #[test]
    fn native_regress_repeated_local_calls_and_aliasing() {
        set_reference_backend(true).expect("enable emulator backend");
        set_backend_mode(BackendMode::Native);

        let wasm_bytes = wat::parse_str(
            r#"
            (module
              (memory 1 1)
              (func $malloc (param $size i32) (result i32)
                (i32.const 16)
              )
              (func (export "malloc") (param i32) (result i32)
                (call $malloc (local.get 0))
              )
              (func (export "two_calls_second") (result i32)
                (local $x i32)
                (local $y i32)
                (local.set $x (call $malloc (i32.const 4)))
                (local.set $y (call $malloc (i32.const 4)))
                (local.get $y)
              )
              (func (export "two_calls_diff") (result i32)
                (local $x i32)
                (local $y i32)
                (local.set $x (call $malloc (i32.const 4)))
                (local.set $y (call $malloc (i32.const 4)))
                (i32.sub (local.get $y) (local.get $x))
              )
              (func (export "store_y_load_x") (result i32)
                (local $x i32)
                (local $y i32)
                (local.set $x (call $malloc (i32.const 4)))
                (local.set $y (call $malloc (i32.const 4)))
                (i32.store (local.get $x) (i32.const 42))
                (i32.store (local.get $y) (i32.const 43))
                (i32.load (local.get $x))
              )
            )
            "#,
        )
        .expect("compile wat");

        let mut runner = WastTestRunner::new();
        let mut instance = runner
            .try_instantiate_temp(&wasm_bytes)
            .expect("instantiate temp module");

        let malloc = instance
            .invoke("malloc", &[Value::I32(4)])
            .expect("invoke malloc");
        assert_eq!(malloc, vec![Value::I32(16)]);

        let second = instance
            .invoke("two_calls_second", &[])
            .expect("invoke two_calls_second");
        assert_eq!(second, vec![Value::I32(16)]);

        let diff = instance
            .invoke("two_calls_diff", &[])
            .expect("invoke two_calls_diff");
        assert_eq!(diff, vec![Value::I32(0)]);

        let alias = instance
            .invoke("store_y_load_x", &[])
            .expect("invoke store_y_load_x");
        assert_eq!(alias, vec![Value::I32(43)]);
    }

    #[test]
    fn spectest_if_wast_passes() {
        match run_wast_fixture("if.wast") {
            TestResult::Pass => {}
            other => panic!("expected if.wast to pass, got {:?}", other),
        }
    }

    #[test]
    fn spectest_br_table_wast_passes() {
        match run_wast_fixture("br_table.wast") {
            TestResult::Pass => {}
            other => panic!("expected br_table.wast to pass, got {:?}", other),
        }
    }

    #[test]
    fn spectest_memory_redundancy_wast_passes() {
        match run_wast_fixture("memory_redundancy.wast") {
            TestResult::Pass => {}
            other => panic!("expected memory_redundancy.wast to pass, got {:?}", other),
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn native_if_mixed_operands_uses_direct_arm64() {
        let mut runner = instantiate_first_module_with_backend("if.wast", BackendMode::Native);
        let wasm_bytes = runner
            .module_bytes
            .values()
            .next()
            .expect("module bytes")
            .clone();
        let module = Module::new("debug", &wasm_bytes).expect("parse module");
        let func_index = module
            .functions()
            .iter()
            .enumerate()
            .find(|(_, func)| {
                func.export_names()
                    .iter()
                    .any(|name| name == "as-mixed-operands")
            })
            .map(|(index, _)| index)
            .expect("exported function index");
        let instance = runner.instances.values_mut().next().expect("instance");
        let ret = instance
            .invoke("as-mixed-operands", &[Value::I32(0)])
            .expect("invoke export");
        assert_eq!(ret, vec![Value::I32(-3)]);

        match &instance.store().module().functions[func_index] {
            FunctionInst::Local { spec, .. } => {
                assert!(
                    spec.has_native_code(),
                    "expected native code to be compiled for as-mixed-operands"
                );
            }
            FunctionInst::Host { .. } => panic!("expected local export"),
        }
    }

    #[test]
    fn native_if_params_id_break_uses_emulator_join_payload() {
        set_reference_backend(true).expect("enable emulator backend");
        let mut runner = instantiate_first_module_with_backend("if.wast", BackendMode::Native);
        let instance = runner.instances.values_mut().next().expect("instance");

        let ret_false = instance
            .invoke("params-id-break", &[Value::I32(0)])
            .expect("invoke export");
        assert_eq!(ret_false, vec![Value::I32(3)]);

        let ret_true = instance
            .invoke("params-id-break", &[Value::I32(1)])
            .expect("invoke export");
        assert_eq!(ret_true, vec![Value::I32(3)]);

        set_reference_backend(false).expect("restore default backend");
    }
}
