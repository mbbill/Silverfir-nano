//! WAST test runner adapted for sf-nano (single-module WebAssembly 2.0 interpreter)

use log::debug;
use sf_nano_core::module::entities::FunctionDef;
use sf_nano_core::module::Module;
use sf_nano_core::value_type::{AbstractHeapType, RefType};
use sf_nano_core::vm::entities::Caller;
use sf_nano_core::vm::value::RefHandle;
use sf_nano_core::{ExternalFn, Import, Instance, Limitable, Value, WasmError};
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
    FuncRef(Option<u32>),
    ExternRef(Option<u32>),
    AnyFuncRef,
    AnyExternRef,
}

impl From<WastValue> for Value {
    fn from(wv: WastValue) -> Self {
        match wv {
            WastValue::I32(v) => Value::I32(v),
            WastValue::I64(v) => Value::I64(v),
            WastValue::F32(v) => Value::F32(v),
            WastValue::F64(v) => Value::F64(v),
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
        }
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
        Import::func("spectest", "print", noop_print as ExternalFn),
        Import::func("spectest", "print_i32", noop_print_i32 as ExternalFn),
        Import::func("spectest", "print_i64", noop_print_i64 as ExternalFn),
        Import::func("spectest", "print_f32", noop_print_f32 as ExternalFn),
        Import::func("spectest", "print_f64", noop_print_f64 as ExternalFn),
        Import::func(
            "spectest",
            "print_i32_f32",
            noop_print_i32_f32 as ExternalFn,
        ),
        Import::func(
            "spectest",
            "print_f64_f64",
            noop_print_f64_f64 as ExternalFn,
        ),
        Import::global("spectest", "global_i32", Value::I32(666), false),
        Import::global("spectest", "global_i64", Value::I64(666), false),
        Import::global("spectest", "global_f32", Value::F32(666.6_f32), false),
        Import::global("spectest", "global_f64", Value::F64(666.6_f64), false),
        Import::table("spectest", "table", 10, Some(20)),
        Import::memory("spectest", "memory", 1, Some(2)),
    ]
}

// ---------------------------------------------------------------------------
// Cross-module function forwarding (spectest-only, lives in std code)
// ---------------------------------------------------------------------------
//
// ExternalFn is a plain fn pointer — no closures. To forward calls to
// registered module exports, we use a thread-local slot table:
//   1. Before instantiation, allocate a slot per cross-module function import.
//   2. Each slot stores (instance_name, export_name).
//   3. Macro-generated fn pointers (fwd_00..fwd_31) each call forward_call(N).
//   4. forward_call reads the slot, finds the instance, and invokes the export.

struct ForwardingSlot {
    instance_name: String,
    export_name: String,
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
        for (reg_name, internal_name) in registered_as {
            if let Some(inst) = instances.get_mut(internal_name) {
                map.insert(reg_name.clone(), inst as *mut Instance);
            }
        }
    });
}

fn alloc_forwarding_slot(instance_name: &str, export_name: &str) -> usize {
    FORWARDING_SLOTS.with(|cell| {
        let mut slots = cell.borrow_mut();
        let idx = slots.len();
        slots.push(Some(ForwardingSlot {
            instance_name: instance_name.to_string(),
            export_name: export_name.to_string(),
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
    let (inst_name, export_name) = FORWARDING_SLOTS.with(|cell| {
        let slots = cell.borrow();
        match slots.get(slot).and_then(|s| s.as_ref()) {
            Some(s) => Ok((s.instance_name.clone(), s.export_name.clone())),
            None => Err(WasmError::internal("forwarding slot empty".into())),
        }
    })?;
    FORWARDING_INSTANCES.with(|cell| {
        let map = cell.borrow();
        match map.get(&inst_name) {
            Some(&inst_ptr) => {
                // Safety: single-threaded spectest, instance outlives the call
                let inst = unsafe { &mut *inst_ptr };
                let ret = inst.invoke(&export_name, args)?;
                for (i, v) in ret.iter().enumerate() {
                    if i < results.len() {
                        results[i] = *v;
                    }
                }
                Ok(())
            }
            None => Err(WasmError::internal(format!(
                "forwarding instance '{}' not found",
                inst_name
            ))),
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

const FORWARDER_TABLE: [ExternalFn; 32] = [
    fwd_00, fwd_01, fwd_02, fwd_03, fwd_04, fwd_05, fwd_06, fwd_07, fwd_08, fwd_09, fwd_10, fwd_11,
    fwd_12, fwd_13, fwd_14, fwd_15, fwd_16, fwd_17, fwd_18, fwd_19, fwd_20, fwd_21, fwd_22, fwd_23,
    fwd_24, fwd_25, fwd_26, fwd_27, fwd_28, fwd_29, fwd_30, fwd_31,
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
}

impl WastTestRunner {
    pub fn new() -> Self {
        WastTestRunner {
            instances: HashMap::new(),
            module_bytes: HashMap::new(),
            module_counter: 0,
            named_modules: HashMap::new(),
            registered_as: HashMap::new(),
            module_definitions: HashMap::new(),
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

        let internal_name = self
            .resolve_module_name(invoke.module.as_ref())
            .map_err(TestError::infrastructure)?;

        let args: Vec<Value> = self
            .convert_wast_args(&invoke.args)
            .into_iter()
            .map(|arg| arg.into())
            .collect();

        let instance = self.instances.get_mut(&internal_name).ok_or_else(|| {
            TestError::infrastructure(format!("Instance '{}' not found", internal_name))
        })?;

        instance.invoke(invoke.name, &args).map_err(|e| {
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
                        let imports = self.build_imports(&wasm_bytes);
                        match Instance::new(&wasm_bytes, &imports) {
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
        let imports = self.build_imports(&compiled.wasm_bytes);
        let instance = Instance::new(&compiled.wasm_bytes, &imports)?;

        self.instances.insert(internal_name.clone(), instance);
        self.module_bytes
            .insert(internal_name.clone(), compiled.wasm_bytes);

        if let Some(name) = compiled.name {
            self.named_modules.insert(name, internal_name.clone());
        }

        Ok(internal_name)
    }

    /// Try to instantiate a module temporarily (for assert_invalid/assert_unlinkable).
    fn try_instantiate_temp(&mut self, wasm_bytes: &[u8]) -> Result<Instance, WasmError> {
        register_forwarding_instances(&mut self.instances, &self.registered_as);
        let imports = self.build_imports(wasm_bytes);
        Instance::new(wasm_bytes, &imports)
    }

    /// Build imports for a module by providing spectest imports plus exports
    /// Build imports for instantiation, forwarding cross-module function calls
    /// via thread-local slot table.
    fn build_imports(&self, wasm_bytes: &[u8]) -> Vec<Import> {
        let mut imports = spectest_imports();
        clear_forwarding();

        // For each registered module, provide its exports as imports.
        for (registered_name, internal_name) in &self.registered_as {
            if let Some(instance) = self.instances.get(internal_name) {
                if let Some(bytes) = self.module_bytes.get(internal_name) {
                    if let Ok(module) = Module::new("_export_scan", bytes) {
                        // Global exports — read current value from live instance
                        for global in module.globals() {
                            for export_name in global.export_names() {
                                if let Some(value) = instance.get_global(export_name) {
                                    imports.push(Import::global(
                                        registered_name,
                                        export_name,
                                        value,
                                        global.mutable(),
                                    ));
                                }
                            }
                        }

                        // Function exports — use forwarding fn pointers
                        for func in module.functions() {
                            for export_name in func.export_names() {
                                let ft = func.func_type().clone();
                                let slot = alloc_forwarding_slot(registered_name, export_name);
                                if slot < FORWARDER_TABLE.len() {
                                    imports.push(Import::func_typed(
                                        registered_name,
                                        export_name,
                                        FORWARDER_TABLE[slot],
                                        ft,
                                    ));
                                } else {
                                    fn stub_fn(
                                        _: &mut Caller,
                                        _: &[Value],
                                        _: &mut [Value],
                                    ) -> Result<(), WasmError> {
                                        Ok(())
                                    }
                                    imports.push(Import::func_typed(
                                        registered_name,
                                        export_name,
                                        stub_fn as ExternalFn,
                                        ft,
                                    ));
                                }
                            }
                        }

                        // Table exports — use live instance sizes
                        for table in module.tables() {
                            for export_name in table.export_names() {
                                let current_size = instance
                                    .table_size(export_name)
                                    .unwrap_or(table.limits().min());
                                imports.push(Import::table(
                                    registered_name,
                                    export_name,
                                    current_size,
                                    table.limits().max(),
                                ));
                            }
                        }

                        // Memory exports — use live instance sizes
                        for memory in module.memories() {
                            for export_name in memory.export_names() {
                                let current_pages = instance
                                    .memory_pages(export_name)
                                    .unwrap_or(memory.limits().min());
                                imports.push(Import::memory(
                                    registered_name,
                                    export_name,
                                    current_pages,
                                    memory.limits().max(),
                                ));
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
                                    fallback_stub as ExternalFn,
                                ));
                            }
                        }
                    }
                }
            }
        }

        imports
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
            WastArgCore::RefNull(ref_type) => {
                match ref_type {
                    wast::core::HeapType::Abstract { ty, .. } => {
                        use wast::core::AbstractHeapType as AHT;
                        match ty {
                            AHT::Func => Some(WastValue::FuncRef(None)),
                            AHT::Extern => Some(WastValue::ExternRef(None)),
                            _ => Some(WastValue::FuncRef(None)), // fallback
                        }
                    }
                    _ => Some(WastValue::FuncRef(None)),
                }
            }
            WastArgCore::RefExtern(idx) => Some(WastValue::ExternRef(Some(*idx))),
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
                    use wast::core::AbstractHeapType as AHT;
                    match ty {
                        AHT::Func => Some(WastValue::FuncRef(None)),
                        AHT::Extern => Some(WastValue::ExternRef(None)),
                        _ => Some(WastValue::FuncRef(None)),
                    }
                }
                _ => Some(WastValue::FuncRef(None)),
            },
            WastRetCore::RefExtern(opt_idx) => match opt_idx {
                Some(idx) => Some(WastValue::ExternRef(Some(*idx))),
                None => Some(WastValue::AnyExternRef),
            },
            WastRetCore::RefFunc(opt_idx) => match opt_idx {
                Some(idx) => match idx {
                    wast::token::Index::Num(n, _) => Some(WastValue::FuncRef(Some(*n))),
                    _ => None,
                },
                None => Some(WastValue::AnyFuncRef),
            },
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
                    !ref_val.is_null() && ref_val.raw_value() == *expected_idx as usize
                }
                (ref_val, None) => ref_val.is_null(),
            }
        }
        (Value::Ref(actual_ref, _), WastValue::AnyFuncRef) => !actual_ref.is_null(),
        (Value::Ref(actual_ref, ref_type), WastValue::AnyExternRef) if ref_type.is_externref() => {
            !actual_ref.is_null()
        }
        (Value::Ref(actual_ref, ref_type), WastValue::ExternRef(expected_ref))
            if ref_type.is_externref() =>
        {
            match (actual_ref, expected_ref) {
                (ref_val, Some(expected_idx)) => {
                    !ref_val.is_null() && ref_val.raw_value() == *expected_idx as usize
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
    use sf_nano_core::vm::backend::{set_backend_mode, BackendMode};
    use sf_nano_core::vm::entities::FunctionInst;
    use sf_nano_core::vm::value::Value;
    use std::path::PathBuf;

    fn instantiate_first_module_with_backend(path: &str, backend: BackendMode) -> WastTestRunner {
        set_backend_mode(backend);
        let full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("target")
            .join("webassembly-testsuite-2.0")
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
        instantiate_first_module_with_backend(path, BackendMode::Base)
    }

    fn run_wast_fixture(path: &str) -> TestResult {
        set_backend_mode(BackendMode::Base);
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("target")
            .join("webassembly-testsuite-2.0")
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
            .find(|(_, func)| func.export_names().iter().any(|name| name == "as-mixed-operands"))
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
                    spec.native_cache().internal_entry().is_some(),
                    "expected direct ARM64 internal entry for as-mixed-operands"
                );
            }
            FunctionInst::External { .. } => panic!("expected local export"),
        }
    }
}
