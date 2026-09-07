//! WAST test runner adapted for sf-nano (single-module WebAssembly 2.0 interpreter)

use log::debug;
use sf_nano_core::value_type::{AbstractHeapType, HeapType, RefType};
use sf_nano_core::Module;
use sf_nano_core::{
    Caller, Engine, HostFn, Import, InstanceId, RefValue, RuntimeWorld, Value, WasmError,
};
use std::{collections::HashMap, fmt, fs, path::Path};
use wast::{
    core::{NanPattern, V128Pattern, WastArgCore, WastRetCore},
    QuoteWat, Wast, WastArg, WastDirective, WastExecute, WastInvoke, WastRet,
};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
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
pub enum TestResult {
    Pass,
    Fail(TestError),
    Error(String),
}

impl fmt::Display for TestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TestResult::Pass => write!(f, "PASS"),
            TestResult::Fail(err) => write!(f, "FAIL: {}", err),
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

#[derive(Debug, Clone)]
pub enum WastValue {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    /// `nan:canonical` / `nan:arithmetic` result expectation: any NaN
    /// matches, mirroring `f32_matches_nan_pattern` on the V128 lane path.
    /// Literal float expectations stay in `F32`/`F64` and compare bit-exact.
    F32AnyNan,
    F64AnyNan,
    V128([u8; 16]),
    V128Pattern(V128Pattern),
    Either(Vec<WastValue>),
    NullRef(RefType),
    FuncRef(Option<u32>),
    ExternRef(Option<u32>),
    AnyFuncRef,
    AnyExternRef,
    AnyI31Ref(RefType),
    AnyStructRef(RefType),
    AnyArrayRef(RefType),
    AnyEqRef(RefType),
    AnyAnyRef(RefType),
    Ref(Option<u32>, RefType),
}

impl From<WastValue> for Value {
    fn from(wv: WastValue) -> Self {
        match wv {
            WastValue::I32(v) => Value::I32(v),
            WastValue::I64(v) => Value::I64(v),
            WastValue::F32(v) => Value::F32(v),
            WastValue::F64(v) => Value::F64(v),
            WastValue::V128(v) => Value::from_v128_bytes(v),
            WastValue::V128Pattern(_) => {
                panic!("V128Pattern should not be converted to Value")
            }
            WastValue::F32AnyNan | WastValue::F64AnyNan => {
                panic!("NaN result patterns should not be converted to Value")
            }
            WastValue::Either(_) => {
                panic!("Either should not be converted to Value")
            }
            WastValue::NullRef(ref_type) => Value::Ref(RefValue::null(), ref_type),
            WastValue::FuncRef(Some(_)) => panic!("indexed function references need an instance"),
            WastValue::FuncRef(None) => Value::Ref(RefValue::null(), RefType::funcref()),
            WastValue::ExternRef(Some(idx)) => {
                let externref_type = RefType::new(false, AbstractHeapType::Extern.into());
                Value::Ref(RefValue::externref(idx as usize), externref_type)
            }
            WastValue::ExternRef(None) => Value::Ref(RefValue::null(), RefType::externref()),
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
            WastValue::AnyEqRef(_) => {
                panic!("AnyEqRef should not be converted to Value")
            }
            WastValue::AnyAnyRef(_) => {
                panic!("AnyAnyRef should not be converted to Value")
            }
            WastValue::Ref(Some(idx), ref_type) => {
                let handle = match ref_type.heap_type {
                    HeapType::Abstract(AbstractHeapType::Any)
                    | HeapType::Abstract(AbstractHeapType::Eq) => RefValue::hostref(idx as usize),
                    HeapType::Abstract(AbstractHeapType::Extern) => {
                        RefValue::externref(idx as usize)
                    }
                    _ => panic!("indexed engine references need an instance"),
                };
                Value::Ref(handle, ref_type)
            }
            WastValue::Ref(None, ref_type) => Value::Ref(RefValue::null(), ref_type),
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
// WastTestRunner
// ---------------------------------------------------------------------------

pub struct WastTestRunner {
    engine: Engine,
    world: RuntimeWorld,
    instances: HashMap<String, InstanceId>,
    module_counter: u32,
    current_module: Option<String>,
    named_modules: HashMap<String, String>,
    registered_as: HashMap<String, String>,
    module_definitions: HashMap<String, Vec<u8>>,
}

impl WastTestRunner {
    pub fn new(engine: Engine) -> Self {
        let world = RuntimeWorld::new();
        WastTestRunner {
            engine,
            world,
            instances: HashMap::new(),
            module_counter: 0,
            current_module: None,
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
        self.run_wast_content(&content)
    }

    /// Parse and execute WAST content embedded in the test binary.
    pub fn run_wast_content(&mut self, content: &str) -> TestResult {
        match self.execute_wast_content(content) {
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
            WastDirective::AssertException { exec, .. } => {
                debug!(
                    "Directive {} action: {} (expect uncaught exception)",
                    index,
                    self.describe_wast_action(exec)
                );
                self.execute_wast_assert_exception(exec)
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
            // No silent skipping: a directive kind this runner does not
            // drive (AssertSuspension / Thread / Wait today) is a failure,
            // not a vacuous pass.
            other => {
                let mut kind = format!("{other:?}");
                kind.truncate(80);
                Err(TestError::infrastructure(format!(
                    "Directive {index}: no handler for {kind}; the runner does not skip directives"
                )))
            }
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
        let internal_name = self
            .resolve_module_name(invoke.module.as_ref())
            .map_err(TestError::infrastructure)?;

        let id =
            self.instances.get(&internal_name).copied().ok_or_else(|| {
                TestError::infrastructure("missing invocation instance".to_string())
            })?;
        let instance = self
            .world
            .instance(id)
            .ok_or_else(|| TestError::infrastructure("freed invocation instance".to_string()))?;
        let args = self
            .convert_wast_args(&invoke.args)
            .into_iter()
            .map(|arg| {
                let indexed = match &arg {
                    WastValue::FuncRef(Some(index)) => Some((*index, RefType::funcref())),
                    WastValue::Ref(Some(index), ty) if ty.is_funcref() => Some((*index, *ty)),
                    _ => None,
                };
                if let Some((index, ty)) = indexed {
                    let function = instance.get_func_by_index(index as usize).ok_or_else(|| {
                        TestError::infrastructure(format!(
                            "unknown function reference index {index}"
                        ))
                    })?;
                    Ok(Value::Ref(function.to_value().into(), ty))
                } else {
                    Ok(arg.into())
                }
            })
            .collect::<Result<Vec<Value>, TestError>>()?;

        let result = self
            .instances
            .get(&internal_name)
            .copied()
            .ok_or_else(|| {
                TestError::infrastructure(format!("Instance '{}' not found", internal_name))
            })
            .and_then(|id| {
                self.world
                    .invoke(id, invoke.name, &args)
                    .map(|values| values.into_iter().collect())
                    .map_err(|error| {
                        TestError::runtime(
                            format!("successful invocation of function '{}'", invoke.name),
                            error,
                        )
                    })
            });

        result
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

        let module = match exec {
            WastExecute::Invoke(invoke) => invoke.module.as_ref(),
            WastExecute::Get { module, .. } => module.as_ref(),
            WastExecute::Wat(_) => None,
        };
        let instance = self
            .resolve_module_name(module)
            .ok()
            .and_then(|name| self.instances.get(&name).copied())
            .and_then(|id| self.world.instance(id));
        let function_ref = |index: u32| {
            instance?
                .get_func_by_index(index as usize)
                .map(|function| RefValue::from(function.to_value()))
        };

        for (i, (actual_val, expected_val)) in actual.iter().zip(expected_values.iter()).enumerate()
        {
            if !values_equal_with_nan(actual_val, expected_val, &function_ref) {
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
            // An uncaught wasm exception is *not* a trap. Reject the
            // assertion so mixed-directive EH tests cannot accidentally
            // mask a real bug behind `assert_trap`.
            Err(err)
                if err
                    .wasm_error()
                    .is_some_and(|w| w.is_exception()) =>
            {
                Err(TestError::infrastructure(format!(
                    "Expected: trap with error '{}' for {}, Actual: uncaught wasm exception ({})",
                    expected_message, action_description, err
                )))
            }
            Err(_) => Ok(()),
        }
    }

    // -----------------------------------------------------------------------
    // assert_exception
    // -----------------------------------------------------------------------

    fn execute_wast_assert_exception(&mut self, exec: &mut WastExecute) -> Result<(), TestError> {
        let action_description = self.describe_wast_action(exec);
        match self.execute_wast_action(exec) {
            Ok(results) => Err(TestError::infrastructure(format!(
                "Expected: uncaught exception for {}, Actual: execution succeeded with results {:?}",
                action_description, results
            ))),
            Err(err)
                if err
                    .wasm_error()
                    .is_some_and(|w| w.is_exception()) =>
            {
                Ok(())
            }
            Err(other) => Err(TestError::infrastructure(format!(
                "Expected: uncaught exception for {}, Actual: {}",
                action_description, other
            ))),
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
            Ok(compiled) => match self.try_instantiate_temp(&compiled.wasm_bytes) {
                Ok(id) => {
                    self.discard_temp(id)?;
                    Err(TestError::infrastructure(format!(
                            "Expected: invalid module with error '{}', Actual: validation and instantiation succeeded",
                            expected_message
                        )))
                }
                Err(_) => Ok(()),
            },
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
                        let imports = self
                            .build_imports()
                            .map_err(|error| TestError::infrastructure(error.to_string()))?;
                        match self.world.instantiate(&self.engine, module, &imports) {
                            Ok(id) => {
                                self.discard_temp(id)?;
                                Err(TestError::infrastructure(format!(
                                    "Expected: malformed module with error '{}', Actual: WASM parsing succeeded ({} bytes)",
                                    expected_message, compiled.wasm_bytes.len()
                                )))
                            }
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
            wast::Wat::Module(ref mut module) => match module.encode() {
                Ok(wasm_bytes) => match self.try_instantiate_temp(&wasm_bytes) {
                    Ok(id) => {
                        self.discard_temp(id)?;
                        Err(TestError::infrastructure(format!(
                                    "Expected: unlinkable module with error '{}', Actual: instantiation succeeded",
                                    expected_message
                                )))
                    }
                    Err(_) => Ok(()),
                },
                Err(_) => Ok(()),
            },
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
            None => self
                .current_module
                .clone()
                .ok_or_else(|| TestError::infrastructure("No current module".to_string()))?,
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
                let id = self.instances.get(&internal_name).copied().ok_or_else(|| {
                    TestError::infrastructure(format!("Instance '{}' not found", internal_name))
                })?;
                let value = self
                    .world
                    .instance(id)
                    .ok_or_else(|| {
                        TestError::infrastructure(format!(
                            "Instance '{}' is no longer available",
                            internal_name
                        ))
                    })?
                    .get_global(global)
                    .map_err(|error| {
                        TestError::runtime(format!("reading global '{}'", global), error)
                    })?
                    .ok_or_else(|| {
                        TestError::infrastructure(format!(
                            "Global '{}' not found in instance '{}'",
                            global, internal_name
                        ))
                    })?;
                Ok(vec![value])
            }
            WastExecute::Wat(wat) => match wat {
                wast::Wat::Module(module) => match module.encode() {
                    Ok(wasm_bytes) => match self.instantiate_with_registry(&wasm_bytes) {
                        Ok(id) => {
                            self.discard_temp(id)?;
                            Ok(vec![])
                        }
                        Err(e) => Err(TestError::runtime(
                            "successful module instantiation".to_string(),
                            e,
                        )),
                    },
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

        let instance_id = self.instantiate_named(&compiled.wasm_bytes)?;
        let previous_current = self.current_module.replace(internal_name.clone());

        self.instances.insert(internal_name.clone(), instance_id);

        if let Some(name) = compiled.name {
            self.named_modules.insert(name, internal_name.clone());
        }

        if let Some(previous_current) = previous_current {
            self.drop_unreachable_module(&previous_current)?;
        }

        Ok(internal_name)
    }

    /// Try to instantiate a module temporarily (for assert_invalid/assert_unlinkable).
    fn try_instantiate_temp(&mut self, wasm_bytes: &[u8]) -> Result<InstanceId, WasmError> {
        self.instantiate_with_registry(wasm_bytes)
    }

    fn discard_temp(&mut self, id: InstanceId) -> Result<(), TestError> {
        self.world.free(id).map_err(|error| {
            TestError::runtime("dropping temporary module instance".to_string(), error)
        })
    }

    fn drop_unreachable_module(&mut self, internal_name: &str) -> Result<(), WasmError> {
        if self.current_module.as_deref() == Some(internal_name) {
            return Ok(());
        }
        if self
            .named_modules
            .values()
            .any(|name| name.as_str() == internal_name)
        {
            return Ok(());
        }
        if self
            .registered_as
            .values()
            .any(|name| name.as_str() == internal_name)
        {
            return Ok(());
        }

        if let Some(id) = self.instances.get(internal_name).copied() {
            self.world.free(id)?;
        }
        self.instances.remove(internal_name);
        Ok(())
    }

    fn instantiate_with_registry(&mut self, wasm_bytes: &[u8]) -> Result<InstanceId, WasmError> {
        self.instantiate_named(wasm_bytes)
    }

    fn instantiate_named(&mut self, wasm_bytes: &[u8]) -> Result<InstanceId, WasmError> {
        let imports = self.build_imports()?;
        let module = Module::new("main", wasm_bytes)?;
        self.world
            .instantiate(&self.engine, module, &imports)
            .map_err(|error| error.into_parts().1)
    }

    /// Bind spectest host imports and registered modules' opaque exports.
    fn build_imports(&self) -> Result<Vec<Import>, WasmError> {
        let mut imports = spectest_imports();

        // The runtime preserves shared identity, current sizes and private
        // type-context metadata for every kind of registered export.
        for (registered_name, internal_name) in &self.registered_as {
            if let Some(instance) = self
                .instances
                .get(internal_name)
                .and_then(|id| self.world.instance(*id))
            {
                for (name, value) in instance.exports()? {
                    imports.push(Import::new(registered_name, &name, value));
                }
            }
        }

        Ok(imports)
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
            None => self
                .current_module
                .clone()
                .ok_or_else(|| "No current module".to_string()),
        }
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
            WastArgCore::V128(v128) => Some(WastValue::V128(v128.to_le_bytes())),
            WastArgCore::RefNull(ref_type) => match ref_type {
                wast::core::HeapType::Abstract { ty, .. } => Some(convert_abstract_null_ref(*ty)),
                _ => Some(WastValue::FuncRef(None)),
            },
            WastArgCore::RefExtern(idx) => Some(WastValue::ExternRef(Some(*idx))),
            WastArgCore::RefHost(idx) => Some(WastValue::Ref(Some(*idx), RefType::anyref())),
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
                wast::core::NanPattern::CanonicalNan => Some(WastValue::F32AnyNan),
                wast::core::NanPattern::ArithmeticNan => Some(WastValue::F32AnyNan),
            },
            WastRetCore::F64(nan_pattern) => match nan_pattern {
                wast::core::NanPattern::Value(f64_val) => {
                    Some(WastValue::F64(f64::from_bits(f64_val.bits)))
                }
                wast::core::NanPattern::CanonicalNan => Some(WastValue::F64AnyNan),
                wast::core::NanPattern::ArithmeticNan => Some(WastValue::F64AnyNan),
            },
            WastRetCore::V128(pattern) => Some(WastValue::V128Pattern(pattern.clone())),
            WastRetCore::Either(cases) => Some(WastValue::Either(
                cases
                    .iter()
                    .filter_map(|case| self.convert_core_ret(case))
                    .collect(),
            )),
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
            WastRetCore::RefAny => Some(WastValue::AnyAnyRef(RefType::new(
                false,
                AbstractHeapType::Any.into(),
            ))),
            WastRetCore::RefEq => Some(WastValue::AnyEqRef(RefType::new(
                false,
                AbstractHeapType::Eq.into(),
            ))),
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

// Null expectations carry abstract heap types or module-local indices. No
// runtime type context is available to this value-only comparison.
fn null_type_is_subtype(actual: RefType, expected: RefType) -> bool {
    if actual.nullable && !expected.nullable {
        return false;
    }
    match (actual.heap_type, expected.heap_type) {
        (HeapType::Abstract(actual), HeapType::Abstract(expected)) => {
            actual.is_subtype_of(&expected)
        }
        (HeapType::Concrete(actual), HeapType::Concrete(expected)) => actual == expected,
        _ => false,
    }
}

fn values_equal_with_nan(
    actual: &Value,
    expected: &WastValue,
    function_ref: &impl Fn(u32) -> Option<RefValue>,
) -> bool {
    if let WastValue::Either(cases) = expected {
        return cases
            .iter()
            .any(|candidate| values_equal_with_nan(actual, candidate, function_ref));
    }

    if let Some(actual_v128) = actual.as_v128_bytes() {
        return match expected {
            WastValue::V128(expected_v128) => actual_v128 == *expected_v128,
            WastValue::V128Pattern(pattern) => v128_matches_pattern(&actual_v128, pattern),
            _ => false,
        };
    }

    match (actual, expected) {
        (Value::I32(a), WastValue::I32(e)) => a == e,
        (Value::I64(a), WastValue::I64(e)) => a == e,
        // Bit-exact: `==` would accept -0.0 for +0.0 (and vice versa),
        // leaving every signed-zero assertion in the suite vacuous.
        (Value::F32(a), WastValue::F32(e)) => a.to_bits() == e.to_bits(),
        (Value::F64(a), WastValue::F64(e)) => a.to_bits() == e.to_bits(),
        (Value::F32(a), WastValue::F32AnyNan) => a.is_nan(),
        (Value::F64(a), WastValue::F64AnyNan) => a.is_nan(),
        (Value::Ref(actual_ref, ref_type), WastValue::FuncRef(expected_ref))
            if ref_type.is_funcref() =>
        {
            match (actual_ref, expected_ref) {
                (ref_val, Some(expected_idx)) => {
                    function_ref(*expected_idx).is_some_and(|expected| *ref_val == expected)
                }
                (ref_val, None) => ref_val.is_null(),
            }
        }
        (Value::Ref(actual_ref, _), WastValue::FuncRef(None)) => actual_ref.is_null(),
        (Value::Ref(actual_ref, ref_type), WastValue::NullRef(expected_type)) => {
            actual_ref.is_null()
                && (null_type_is_subtype(*ref_type, *expected_type)
                    || null_type_is_subtype(*expected_type, *ref_type))
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
        (Value::Ref(actual_ref, actual_rt), WastValue::AnyEqRef(_)) => {
            if actual_ref.is_null() {
                return false;
            }
            match actual_rt.heap_type {
                HeapType::Abstract(
                    AbstractHeapType::Eq
                    | AbstractHeapType::Any
                    | AbstractHeapType::I31
                    | AbstractHeapType::Struct
                    | AbstractHeapType::Array,
                )
                | HeapType::Concrete(_) => true,
                _ => false,
            }
        }
        (Value::Ref(actual_ref, actual_rt), WastValue::AnyAnyRef(_)) => {
            if actual_ref.is_null() {
                return false;
            }
            match actual_rt.heap_type {
                HeapType::Abstract(
                    AbstractHeapType::Any
                    | AbstractHeapType::Eq
                    | AbstractHeapType::I31
                    | AbstractHeapType::Struct
                    | AbstractHeapType::Array,
                )
                | HeapType::Concrete(_) => true,
                _ => false,
            }
        }
        (Value::Ref(actual_ref, ref_type), WastValue::ExternRef(expected_ref))
            if ref_type.is_externref() =>
        {
            match (actual_ref, expected_ref) {
                (ref_val, Some(expected_idx)) => ref_val.host_id() == Some(*expected_idx as usize),
                (ref_val, None) => ref_val.is_null(),
            }
        }
        (Value::Ref(actual_ref, actual_rt), WastValue::Ref(expected_ref, expected_rt)) => {
            match (actual_ref, expected_ref) {
                (ref_val, Some(expected_idx)) => {
                    if ref_val.is_null() || *actual_rt != *expected_rt {
                        false
                    } else {
                        if expected_rt.is_funcref() {
                            function_ref(*expected_idx).is_some_and(|expected| *ref_val == expected)
                        } else {
                            ref_val.host_id() == Some(*expected_idx as usize)
                        }
                    }
                }
                (ref_val, None) => ref_val.is_null(),
            }
        }
        _ => false,
    }
}

fn v128_matches_pattern(actual: &[u8; 16], pattern: &V128Pattern) -> bool {
    match pattern {
        V128Pattern::I8x16(expected) => actual
            .iter()
            .copied()
            .map(|lane| lane as i8)
            .zip(expected.iter().copied())
            .all(|(a, e)| a == e),
        V128Pattern::I16x8(expected) => actual_i16x8(actual)
            .into_iter()
            .zip(expected.iter().copied())
            .all(|(a, e)| a == e),
        V128Pattern::I32x4(expected) => actual_i32x4(actual)
            .into_iter()
            .zip(expected.iter().copied())
            .all(|(a, e)| a == e),
        V128Pattern::I64x2(expected) => actual_i64x2(actual)
            .into_iter()
            .zip(expected.iter().copied())
            .all(|(a, e)| a == e),
        V128Pattern::F32x4(expected) => actual_f32x4(actual)
            .into_iter()
            .zip(expected.iter())
            .all(|(a, e)| f32_matches_nan_pattern(a, e)),
        V128Pattern::F64x2(expected) => actual_f64x2(actual)
            .into_iter()
            .zip(expected.iter())
            .all(|(a, e)| f64_matches_nan_pattern(a, e)),
    }
}

fn actual_i16x8(actual: &[u8; 16]) -> [i16; 8] {
    core::array::from_fn(|i| {
        let base = i * 2;
        i16::from_le_bytes([actual[base], actual[base + 1]])
    })
}

fn actual_i32x4(actual: &[u8; 16]) -> [i32; 4] {
    core::array::from_fn(|i| {
        let base = i * 4;
        i32::from_le_bytes([
            actual[base],
            actual[base + 1],
            actual[base + 2],
            actual[base + 3],
        ])
    })
}

fn actual_i64x2(actual: &[u8; 16]) -> [i64; 2] {
    core::array::from_fn(|i| {
        let base = i * 8;
        i64::from_le_bytes([
            actual[base],
            actual[base + 1],
            actual[base + 2],
            actual[base + 3],
            actual[base + 4],
            actual[base + 5],
            actual[base + 6],
            actual[base + 7],
        ])
    })
}

fn actual_f32x4(actual: &[u8; 16]) -> [f32; 4] {
    core::array::from_fn(|i| {
        let base = i * 4;
        f32::from_bits(u32::from_le_bytes([
            actual[base],
            actual[base + 1],
            actual[base + 2],
            actual[base + 3],
        ]))
    })
}

fn actual_f64x2(actual: &[u8; 16]) -> [f64; 2] {
    core::array::from_fn(|i| {
        let base = i * 8;
        f64::from_bits(u64::from_le_bytes([
            actual[base],
            actual[base + 1],
            actual[base + 2],
            actual[base + 3],
            actual[base + 4],
            actual[base + 5],
            actual[base + 6],
            actual[base + 7],
        ]))
    })
}

fn f32_matches_nan_pattern(actual: f32, expected: &NanPattern<wast::token::F32>) -> bool {
    match expected {
        NanPattern::Value(bits) => {
            let expected = f32::from_bits(bits.bits);
            if actual.is_nan() && expected.is_nan() {
                true
            } else {
                actual == expected
            }
        }
        NanPattern::CanonicalNan | NanPattern::ArithmeticNan => actual.is_nan(),
    }
}

fn f64_matches_nan_pattern(actual: f64, expected: &NanPattern<wast::token::F64>) -> bool {
    match expected {
        NanPattern::Value(bits) => {
            let expected = f64::from_bits(bits.bits);
            if actual.is_nan() && expected.is_nan() {
                true
            } else {
                actual == expected
            }
        }
        NanPattern::CanonicalNan | NanPattern::ArithmeticNan => actual.is_nan(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sf_nano_core::{Config, Engine, Tier, Value};
    use std::path::PathBuf;

    // Native-code inspection and GC fixtures require the JIT engine.
    #[cfg(feature = "jit")]
    mod jit;

    fn expect_values(values: impl AsRef<[Value]>, expected: &[Value]) {
        assert_eq!(values.as_ref(), expected);
    }

    fn only_instance_id(runner: &WastTestRunner) -> InstanceId {
        runner.instances.values().copied().next().expect("instance")
    }

    /// One engine per test, on the tier the test names.
    fn engine_for(tier: Tier) -> Engine {
        Engine::new(Config::new().tier(tier)).expect("engine")
    }

    fn test_engine() -> Engine {
        engine_for(Tier::DEFAULT)
    }

    #[test]
    fn module_names_do_not_register_imports() {
        for &tier in Tier::ALL {
            let mut runner = WastTestRunner::new(engine_for(tier));
            runner
                .execute_wast_content(
                    r#"
                (module $source (func (export "f")))
                (assert_unlinkable
                    (module (import "source" "f" (func))) "unknown import")
                (register "source" $source)
                (module (import "source" "f" (func $f))
                    (func (export "run") call $f))
                (invoke "run")
                "#,
                )
                .unwrap();
        }
    }

    #[test]
    fn registered_globals_share_start_and_trapping_call_writes() {
        for &tier in Tier::ALL {
            let mut runner = WastTestRunner::new(engine_for(tier));
            runner
                .execute_wast_content(
                    r#"
                (module $source
                    (global $g (export "g") (mut i32) (i32.const 1))
                    (func (export "read") (result i32) global.get $g))
                (register "source" $source)
                (module $target
                    (import "source" "g" (global $g (mut i32)))
                    (import "source" "read" (func $read (result i32)))
                    (func $start i32.const 2 global.set $g)
                    (start $start)
                    (func (export "read_after_write") (result i32)
                        i32.const 3 global.set $g call $read)
                    (func (export "write_then_trap")
                        i32.const 4 global.set $g unreachable))
                (assert_return (get $source "g") (i32.const 2))
                (assert_return (invoke $target "read_after_write") (i32.const 3))
                (assert_trap (invoke $target "write_then_trap") "unreachable")
                (assert_return (get $source "g") (i32.const 4))
                "#,
                )
                .unwrap();
        }
    }

    fn instantiate_first_module_with_backend(path: &str, tier: Tier) -> WastTestRunner {
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

        let mut runner = WastTestRunner::new(engine_for(tier));
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
        instantiate_first_module_with_backend(path, Tier::DEFAULT)
    }

    fn run_wast_fixture(path: &str) -> TestResult {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("target")
            .join("webassembly-testsuite")
            .join(path);
        let mut runner = WastTestRunner::new(test_engine());
        runner.run_wast_file(&path)
    }

    #[test]
    fn regress_if_as_br_table_last_true() {
        let mut runner = instantiate_first_module("if.wast");
        let id = only_instance_id(&runner);
        let ret = runner
            .world
            .invoke(id, "as-br_table-last", &[Value::I32(1)])
            .expect("invoke export");
        expect_values(ret, &[Value::I32(2)]);
    }

    #[test]
    fn regress_br_table_as_if_else_false() {
        let mut runner = instantiate_first_module("br_table.wast");
        let id = only_instance_id(&runner);
        let ret = runner
            .world
            .invoke(id, "as-if-else", &[Value::I32(0), Value::I32(6)])
            .expect("invoke export");
        expect_values(ret, &[Value::I32(4)]);
    }

    #[test]
    fn regress_memory_redundancy_malloc_aliasing() {
        let mut runner = instantiate_first_module("memory_redundancy.wast");
        let id = only_instance_id(&runner);
        let ret = runner
            .world
            .invoke(id, "malloc_aliasing", &[])
            .expect("invoke export");
        expect_values(ret, &[Value::I32(43)]);
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

    #[test]
    fn regress_repeated_local_calls_and_aliasing() {
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

        let mut runner = WastTestRunner::new(test_engine());
        let id = runner
            .try_instantiate_temp(&wasm_bytes)
            .expect("instantiate temp module");

        let malloc = runner
            .world
            .invoke(id, "malloc", &[Value::I32(4)])
            .expect("invoke malloc");
        assert_eq!(malloc.as_slice(), &[Value::I32(16)]);

        let second = runner
            .world
            .invoke(id, "two_calls_second", &[])
            .expect("invoke two_calls_second");
        assert_eq!(second.as_slice(), &[Value::I32(16)]);

        let diff = runner
            .world
            .invoke(id, "two_calls_diff", &[])
            .expect("invoke two_calls_diff");
        assert_eq!(diff.as_slice(), &[Value::I32(0)]);

        let alias = runner
            .world
            .invoke(id, "store_y_load_x", &[])
            .expect("invoke store_y_load_x");
        assert_eq!(alias.as_slice(), &[Value::I32(43)]);
    }

    #[test]
    fn regress_if_params_id_break_uses_join_payload() {
        let mut runner = instantiate_first_module("if.wast");
        let id = only_instance_id(&runner);

        let ret_false = runner
            .world
            .invoke(id, "params-id-break", &[Value::I32(0)])
            .expect("invoke export");
        assert_eq!(ret_false.as_slice(), &[Value::I32(3)]);

        let ret_true = runner
            .world
            .invoke(id, "params-id-break", &[Value::I32(1)])
            .expect("invoke export");
        assert_eq!(ret_true.as_slice(), &[Value::I32(3)]);
    }

    #[cfg(feature = "interp")]
    #[test]
    fn interp_exception_funcref_payload_keeps_cross_instance_identity() {
        let mut runner = WastTestRunner::new(engine_for(Tier::Interp));
        runner
            .execute_wast_content(
                r#"
                (module
                  (type $ft (func (result i32)))
                  (type $pair (func (result i32 i64)))
                  (tag $e (export "e") (param (ref $ft)))
                  (tag $epair (export "epair") (param (ref $pair)))
                  (func $dummy (type $ft) (result i32) i32.const 99)
                  (func $pair (type $pair) (result i32 i64)
                    i32.const 41
                    i64.const 42)
                  (elem declare func $dummy)
                  (elem declare func $pair)
                  (func (export "throw")
                    (throw $e (ref.func $dummy)))
                  (func (export "throw_pair")
                    (throw $epair (ref.func $pair)))
                  (func (export "caught_local") (result i32)
                    (block $h (result (ref $ft))
                      (try_table (catch $e $h)
                        (throw $e (ref.func $dummy)))
                      unreachable)
                    (call_ref $ft)))
                (register "src")
                (assert_return (invoke "caught_local") (i32.const 99))
                (module
                  (type $ft (func (result i32)))
                  (type $pair (func (result i32 i64)))
                  (tag $e (import "src" "e") (param (ref $ft)))
                  (tag $epair (import "src" "epair") (param (ref $pair)))
                  (func $throw (import "src" "throw"))
                  (func $throw_pair (import "src" "throw_pair"))
                  (table 1 (ref null $ft))
                  (func (export "via_table") (result i32)
                    (table.set 0 (i32.const 0)
                      (block $h (result (ref $ft))
                        (try_table (catch $e $h)
                          (call $throw))
                        unreachable))
                    (call_indirect 0 (type $ft) (i32.const 0)))
                  (func (export "via_ref") (result i32)
                    (block $h (result (ref $ft))
                      (try_table (catch $e $h)
                        (call $throw))
                      unreachable)
                    (call_ref $ft))
                  (func (export "via_table_acc") (result i32)
                    (table.set 0 (i32.const 0)
                      (block $h (result (ref $ft))
                        (try_table (catch $e $h)
                          (call $throw))
                        unreachable))
                    (i32.add
                      (call_indirect 0 (type $ft) (i32.const 0))
                      (i32.const 1)))
                  (func (export "via_ref_acc") (result i32)
                    (i32.add
                      (call_ref $ft
                        (block $h (result (ref $ft))
                          (try_table (catch $e $h)
                            (call $throw))
                          unreachable))
                      (i32.const 1)))
                  (func (export "tail_via_table") (param i32) (result i32)
                    (table.set 0 (i32.const 0)
                      (block $h (result (ref $ft))
                        (try_table (catch $e $h)
                          (call $throw))
                        unreachable))
                    (return_call_indirect 0 (type $ft) (i32.const 0)))
                  (func (export "tail_via_ref") (param i32) (result i32)
                    (local i32)
                    (block $h (result (ref $ft))
                      (try_table (catch $e $h)
                        (call $throw))
                      unreachable)
                    (return_call_ref $ft))
                  (func (export "tail_pair_via_ref") (param i32) (result i32 i64)
                    (local i64)
                    (block $h (result (ref $pair))
                      (try_table (catch $epair $h)
                        (call $throw_pair))
                      unreachable)
                    (return_call_ref $pair)))
                (assert_return (invoke "via_table") (i32.const 99))
                (assert_return (invoke "via_ref") (i32.const 99))
                (assert_return (invoke "via_table_acc") (i32.const 100))
                (assert_return (invoke "via_ref_acc") (i32.const 100))
                (assert_return (invoke "tail_via_table" (i32.const 7)) (i32.const 99))
                (assert_return (invoke "tail_via_ref" (i32.const 7)) (i32.const 99))
                (assert_return
                  (invoke "tail_pair_via_ref" (i32.const 7))
                  (i32.const 41)
                  (i64.const 42))
                "#,
            )
            .expect("cross-instance exception payload");
    }
}
