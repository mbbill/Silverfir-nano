//! Streaming template JIT for functions that are too large for the full
//! semantic/SSA/MachineIR pipeline.
//!
//! This path emits through a streaming template backend interface. It does not
//! materialize per-op semantic, SSA, or MachineIR vectors.

use crate::{
    error::WasmError,
    module::entities::FunctionSpec,
    opcodes::Opcode,
    utils::payload::Payload,
    value_type::ValueType,
    vm::{
        jit::arch,
        jit::arch::common::{
            pipeline,
            template::{TemplateBackend, TemplateBranchSense, TEMPLATE_PATCH_CHAIN_END},
            types::FunctionArtifact,
        },
        jit::backend::BackendConfig,
        jit::machine::machine_ir::{
            MachineAddr, MachineBranchCond, MachineCompareKind, MachineConvertOp, MachineFuncId,
            MachineIntBinaryOp, MachineIntUnaryOp, MachineIntWidth, MachineLoadExtension,
            MachineMemWidth, MachineReg, MachineRegOwner, MachineSign, MachineStorageType,
            MachineTrapKind, MachineValue, MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
        },
        jit::runtime::{code::CodegenModuleView, code_buf::CodeBuffer},
    },
};

const TEMPLATE_UNSUPPORTED: &str = "template jit unsupported opcode";
const TEMPLATE_EMPTY_BLOCK_TYPE: u8 = 0x40;
const TEMPLATE_MAX_CONTROL_DEPTH: usize = 16;

#[derive(Clone, Copy, Debug)]
pub(crate) struct TemplateFunctionSummary {
    pub local_count: u16,
    pub max_stack_height: u16,
    pub result_count: u16,
}

#[derive(Clone, Copy, Debug)]
struct TemplateShape {
    local_count: u16,
    operand_base: u16,
    result_count: u16,
    result_type: Option<ValueType>,
    gp_unit_bytes: u8,
    has_memory: bool,
}

#[inline]
fn unsupported() -> WasmError {
    WasmError::internal(TEMPLATE_UNSUPPORTED)
}

pub(crate) fn is_template_unsupported(err: &WasmError) -> bool {
    matches!(err, WasmError::Internal(message) if *message == TEMPLATE_UNSUPPORTED)
}

pub(crate) fn scan_function(
    config: BackendConfig,
    spec: &FunctionSpec,
    has_memory: bool,
) -> Result<TemplateFunctionSummary, WasmError> {
    let shape = template_shape(config, spec, has_memory)?;
    let mut scanner = TemplateScanner {
        spec,
        shape,
        stack_height: 0,
        stack_type: None,
        max_stack_height: 0,
        control: [TemplateScanControlFrame::EMPTY; TEMPLATE_MAX_CONTROL_DEPTH],
        control_depth: 0,
    };
    scanner.scan()
}

pub(crate) fn compile_function_into_buffer(
    active_backend: arch::NativeBackend,
    compiled: &dyn CodegenModuleView,
    spec: &FunctionSpec,
    func_id: MachineFuncId,
    executable: &mut CodeBuffer,
    has_memory: bool,
) -> Result<FunctionArtifact, WasmError> {
    arch::dispatch_compile_template_function_into_buffer(
        active_backend,
        compiled,
        spec,
        func_id,
        executable,
        has_memory,
    )
}

pub(crate) fn compile_template_for_backend<'a, A: TemplateBackend<'a>>(
    compiled: &'a dyn CodegenModuleView,
    spec: &'a FunctionSpec,
    func_id: MachineFuncId,
    executable: &mut CodeBuffer,
    has_memory: bool,
) -> Result<FunctionArtifact, WasmError> {
    let mut emitter = TemplateEmitter::new(compiled.backend(), spec, has_memory)?;
    pipeline::compile_template_function_into_buffer::<A, _>(
        compiled,
        func_id,
        executable,
        move |backend| emitter.emit(backend),
    )
}

fn template_shape(
    config: BackendConfig,
    spec: &FunctionSpec,
    has_memory: bool,
) -> Result<TemplateShape, WasmError> {
    if config.gp_unit_bytes != 4 && config.gp_unit_bytes != 8 {
        return Err(unsupported());
    }
    if config.gp_dynamic_budget < 2 {
        return Err(unsupported());
    }

    for &ty in spec.func_type().params() {
        ensure_template_type(config.gp_unit_bytes, ty)?;
    }
    for &ty in spec.locals() {
        ensure_template_type(config.gp_unit_bytes, ty)?;
    }
    for &ty in spec.func_type().results() {
        ensure_template_type(config.gp_unit_bytes, ty)?;
    }

    let params = spec.func_type().params().len();
    let local_count = params
        .checked_add(spec.locals().len())
        .and_then(|count| u16::try_from(count).ok())
        .ok_or_else(|| WasmError::internal("template jit local count overflow"))?;
    let operand_base = local_count
        .checked_add(config.call_scratch_slots)
        .ok_or_else(|| WasmError::internal("template jit frame layout overflow"))?;
    let result_count = u16::try_from(spec.func_type().results().len())
        .map_err(|_| WasmError::internal("template jit result count overflow"))?;
    let result_type = homogeneous_result_type(spec.func_type().results())?;

    Ok(TemplateShape {
        local_count,
        operand_base,
        result_count,
        result_type,
        gp_unit_bytes: config.gp_unit_bytes,
        has_memory,
    })
}

fn homogeneous_result_type(results: &[ValueType]) -> Result<Option<ValueType>, WasmError> {
    let Some((&first, rest)) = results.split_first() else {
        return Ok(None);
    };
    if rest.iter().any(|&ty| ty != first) {
        return Err(unsupported());
    }
    Ok(Some(first))
}

fn binary_result_type(op: Opcode) -> ValueType {
    match op {
        Opcode::I32_EQ
        | Opcode::I32_NE
        | Opcode::I32_LT_S
        | Opcode::I32_LT_U
        | Opcode::I32_GT_S
        | Opcode::I32_GT_U
        | Opcode::I32_LE_S
        | Opcode::I32_LE_U
        | Opcode::I32_GE_S
        | Opcode::I32_GE_U
        | Opcode::I64_EQ
        | Opcode::I64_NE
        | Opcode::I64_LT_S
        | Opcode::I64_LT_U
        | Opcode::I64_GT_S
        | Opcode::I64_GT_U
        | Opcode::I64_LE_S
        | Opcode::I64_LE_U
        | Opcode::I64_GE_S
        | Opcode::I64_GE_U => ValueType::I32,
        Opcode::I64_ADD
        | Opcode::I64_SUB
        | Opcode::I64_MUL
        | Opcode::I64_DIV_S
        | Opcode::I64_DIV_U
        | Opcode::I64_REM_S
        | Opcode::I64_REM_U
        | Opcode::I64_AND
        | Opcode::I64_OR
        | Opcode::I64_XOR
        | Opcode::I64_SHL
        | Opcode::I64_SHR_S
        | Opcode::I64_SHR_U
        | Opcode::I64_ROTL
        | Opcode::I64_ROTR => ValueType::I64,
        _ => ValueType::I32,
    }
}

fn unary_result_type(op: Opcode) -> ValueType {
    match op {
        Opcode::I64_EQZ => ValueType::I32,
        Opcode::I64_CLZ
        | Opcode::I64_CTZ
        | Opcode::I64_POPCNT
        | Opcode::I64_EXTEND8_S
        | Opcode::I64_EXTEND16_S
        | Opcode::I64_EXTEND32_S => ValueType::I64,
        _ => ValueType::I32,
    }
}

struct TemplateScanner<'a> {
    spec: &'a FunctionSpec,
    shape: TemplateShape,
    stack_height: u16,
    stack_type: Option<ValueType>,
    max_stack_height: u16,
    control: [TemplateScanControlFrame; TEMPLATE_MAX_CONTROL_DEPTH],
    control_depth: usize,
}

#[derive(Clone, Copy)]
struct TemplateScanControlFrame {
    entry_stack_height: u16,
    entry_stack_type: Option<ValueType>,
    seen_else: bool,
}

impl TemplateScanControlFrame {
    const EMPTY: Self = Self {
        entry_stack_height: 0,
        entry_stack_type: None,
        seen_else: false,
    };
}

impl<'a> TemplateScanner<'a> {
    fn scan(&mut self) -> Result<TemplateFunctionSummary, WasmError> {
        let mut code = Payload::from(&**self.spec.code());
        let mut terminated = false;
        while !code.is_empty() {
            let op: Opcode = code.read_u8()?.try_into()?;
            if terminated && op != Opcode::END {
                return Err(unsupported());
            }
            match op {
                Opcode::END => {
                    if self.control_depth != 0 {
                        self.end_control()?;
                        continue;
                    }
                    if !terminated {
                        self.validate_function_results()?;
                    }
                    return Ok(TemplateFunctionSummary {
                        local_count: self.shape.local_count,
                        max_stack_height: self.max_stack_height,
                        result_count: self.shape.result_count,
                    });
                }
                Opcode::RETURN => {
                    self.validate_function_results()?;
                    terminated = true;
                }
                Opcode::NOP => {}
                Opcode::UNREACHABLE => return Err(unsupported()),
                Opcode::IF => {
                    let block_type = code.read_u8()?;
                    if block_type != TEMPLATE_EMPTY_BLOCK_TYPE {
                        return Err(unsupported());
                    }
                    self.pop(ValueType::I32)?;
                    if self.control_depth == TEMPLATE_MAX_CONTROL_DEPTH {
                        return Err(unsupported());
                    }
                    self.control[self.control_depth] = TemplateScanControlFrame {
                        entry_stack_height: self.stack_height,
                        entry_stack_type: self.stack_type,
                        seen_else: false,
                    };
                    self.control_depth += 1;
                }
                Opcode::ELSE => {
                    self.else_control()?;
                }
                Opcode::DROP => {
                    self.pop_any()?;
                }
                Opcode::LOCAL_GET => {
                    let index = code.read_leb128_u32()?;
                    self.push(local_type(self.spec, index)?)?;
                }
                Opcode::LOCAL_SET => {
                    let index = code.read_leb128_u32()?;
                    self.pop(local_type(self.spec, index)?)?;
                }
                Opcode::LOCAL_TEE => {
                    let index = code.read_leb128_u32()?;
                    self.peek(local_type(self.spec, index)?)?;
                }
                Opcode::I32_CONST => {
                    let _ = code.read_leb128_i32()?;
                    self.push(ValueType::I32)?;
                }
                Opcode::I64_CONST => {
                    let _ = code.read_leb128_i64()?;
                    self.push(ValueType::I64)?;
                }
                Opcode::I32_LOAD => {
                    self.validate_memory_access(code.read_leb128_u32()?)?;
                    let offset = code.read_leb128_u32()?;
                    if offset != 0 {
                        return Err(unsupported());
                    }
                    self.pop(ValueType::I32)?;
                    self.push(ValueType::I32)?;
                }
                Opcode::I32_STORE => {
                    self.validate_memory_access(code.read_leb128_u32()?)?;
                    let offset = code.read_leb128_u32()?;
                    if offset != 0 {
                        return Err(unsupported());
                    }
                    self.pop(ValueType::I32)?;
                    self.pop(ValueType::I32)?;
                }
                Opcode::I32_ADD
                | Opcode::I32_SUB
                | Opcode::I32_MUL
                | Opcode::I32_DIV_S
                | Opcode::I32_DIV_U
                | Opcode::I32_REM_S
                | Opcode::I32_REM_U
                | Opcode::I32_AND
                | Opcode::I32_OR
                | Opcode::I32_XOR
                | Opcode::I32_SHL
                | Opcode::I32_SHR_S
                | Opcode::I32_SHR_U
                | Opcode::I32_ROTL
                | Opcode::I32_ROTR
                | Opcode::I64_ADD
                | Opcode::I64_SUB
                | Opcode::I64_MUL
                | Opcode::I64_DIV_S
                | Opcode::I64_DIV_U
                | Opcode::I64_REM_S
                | Opcode::I64_REM_U
                | Opcode::I64_AND
                | Opcode::I64_OR
                | Opcode::I64_XOR
                | Opcode::I64_SHL
                | Opcode::I64_SHR_S
                | Opcode::I64_SHR_U
                | Opcode::I64_ROTL
                | Opcode::I64_ROTR
                | Opcode::I32_EQ
                | Opcode::I32_NE
                | Opcode::I32_LT_S
                | Opcode::I32_LT_U
                | Opcode::I32_GT_S
                | Opcode::I32_GT_U
                | Opcode::I32_LE_S
                | Opcode::I32_LE_U
                | Opcode::I32_GE_S
                | Opcode::I32_GE_U
                | Opcode::I64_EQ
                | Opcode::I64_NE
                | Opcode::I64_LT_S
                | Opcode::I64_LT_U
                | Opcode::I64_GT_S
                | Opcode::I64_GT_U
                | Opcode::I64_LE_S
                | Opcode::I64_LE_U
                | Opcode::I64_GE_S
                | Opcode::I64_GE_U => {
                    if matches!(
                        op,
                        Opcode::I64_ADD
                            | Opcode::I64_SUB
                            | Opcode::I64_MUL
                            | Opcode::I64_DIV_S
                            | Opcode::I64_DIV_U
                            | Opcode::I64_REM_S
                            | Opcode::I64_REM_U
                            | Opcode::I64_AND
                            | Opcode::I64_OR
                            | Opcode::I64_XOR
                            | Opcode::I64_SHL
                            | Opcode::I64_SHR_S
                            | Opcode::I64_SHR_U
                            | Opcode::I64_ROTL
                            | Opcode::I64_ROTR
                            | Opcode::I64_EQ
                            | Opcode::I64_NE
                            | Opcode::I64_LT_S
                            | Opcode::I64_LT_U
                            | Opcode::I64_GT_S
                            | Opcode::I64_GT_U
                            | Opcode::I64_LE_S
                            | Opcode::I64_LE_U
                            | Opcode::I64_GE_S
                            | Opcode::I64_GE_U
                    ) {
                        self.binary(ValueType::I64, binary_result_type(op))?;
                    } else {
                        self.binary(ValueType::I32, binary_result_type(op))?;
                    }
                }
                Opcode::I32_EQZ
                | Opcode::I64_EQZ
                | Opcode::I32_CLZ
                | Opcode::I32_CTZ
                | Opcode::I32_POPCNT
                | Opcode::I32_EXTEND8_S
                | Opcode::I32_EXTEND16_S
                | Opcode::I64_CLZ
                | Opcode::I64_CTZ
                | Opcode::I64_POPCNT
                | Opcode::I64_EXTEND8_S
                | Opcode::I64_EXTEND16_S
                | Opcode::I64_EXTEND32_S => {
                    if matches!(
                        op,
                        Opcode::I64_EQZ
                            | Opcode::I64_CLZ
                            | Opcode::I64_CTZ
                            | Opcode::I64_POPCNT
                            | Opcode::I64_EXTEND8_S
                            | Opcode::I64_EXTEND16_S
                            | Opcode::I64_EXTEND32_S
                    ) {
                        self.unary(ValueType::I64, unary_result_type(op))?;
                    } else {
                        self.unary(ValueType::I32, unary_result_type(op))?;
                    }
                }
                _ => return Err(unsupported()),
            }
        }

        Err(WasmError::malformed(
            "template jit reached unexpected end of code",
        ))
    }

    fn push(&mut self, ty: ValueType) -> Result<(), WasmError> {
        self.ensure_type(ty)?;
        if let Some(stack_type) = self.stack_type {
            if stack_type != ty {
                return Err(unsupported());
            }
        } else {
            self.stack_type = Some(ty);
        }
        self.stack_height = self
            .stack_height
            .checked_add(1)
            .ok_or_else(|| WasmError::internal("template jit stack height overflow"))?;
        if self.stack_height > self.max_stack_height {
            self.max_stack_height = self.stack_height;
        }
        Ok(())
    }

    fn pop(&mut self, ty: ValueType) -> Result<(), WasmError> {
        self.ensure_type(ty)?;
        self.peek(ty)?;
        self.stack_height = self
            .stack_height
            .checked_sub(1)
            .ok_or_else(|| WasmError::invalid("template jit stack underflow"))?;
        if self.stack_height == 0 {
            self.stack_type = None;
        }
        Ok(())
    }

    fn pop_any(&mut self) -> Result<(), WasmError> {
        if self.stack_height == 0 {
            return Err(WasmError::invalid("template jit stack underflow"));
        }
        self.stack_height -= 1;
        if self.stack_height == 0 {
            self.stack_type = None;
        }
        Ok(())
    }

    fn peek(&self, ty: ValueType) -> Result<(), WasmError> {
        self.ensure_type(ty)?;
        if self.stack_height == 0 || self.stack_type != Some(ty) {
            return Err(WasmError::invalid("template jit type mismatch"));
        }
        Ok(())
    }

    fn binary(&mut self, input: ValueType, output: ValueType) -> Result<(), WasmError> {
        self.pop(input)?;
        self.pop(input)?;
        self.push(output)
    }

    fn unary(&mut self, input: ValueType, output: ValueType) -> Result<(), WasmError> {
        self.pop(input)?;
        self.push(output)
    }

    fn else_control(&mut self) -> Result<(), WasmError> {
        let index = self.control_depth.checked_sub(1).ok_or_else(unsupported)?;
        let mut frame = self.control[index];
        if frame.seen_else
            || self.stack_height != frame.entry_stack_height
            || self.stack_type != frame.entry_stack_type
        {
            return Err(WasmError::invalid("template jit control stack mismatch"));
        }
        frame.seen_else = true;
        self.control[index] = frame;
        Ok(())
    }

    fn end_control(&mut self) -> Result<(), WasmError> {
        let index = self.control_depth.checked_sub(1).ok_or_else(unsupported)?;
        let frame = self.control[index];
        if self.stack_height != frame.entry_stack_height
            || self.stack_type != frame.entry_stack_type
        {
            return Err(WasmError::invalid("template jit control stack mismatch"));
        }
        self.control[index] = TemplateScanControlFrame::EMPTY;
        self.control_depth = index;
        Ok(())
    }

    fn validate_function_results(&self) -> Result<(), WasmError> {
        if self.stack_height != self.shape.result_count {
            return Err(WasmError::invalid("template jit result arity mismatch"));
        }
        if match self.shape.result_type {
            Some(ty) => self.stack_type != Some(ty),
            None => self.stack_type.is_some(),
        } {
            return Err(WasmError::invalid("template jit result type mismatch"));
        }
        Ok(())
    }

    fn validate_memory_access(&self, align: u32) -> Result<(), WasmError> {
        if !self.shape.has_memory {
            return Err(WasmError::invalid("unknown memory"));
        }
        if align > 2 {
            return Err(WasmError::invalid("memory access alignment too large"));
        }
        Ok(())
    }

    fn ensure_type(&self, ty: ValueType) -> Result<(), WasmError> {
        ensure_template_type(self.shape.gp_unit_bytes, ty)
    }
}

struct TemplateEmitter<'a> {
    spec: &'a FunctionSpec,
    local_count: u16,
    operand_base: u16,
    result_count: u16,
    gp_unit_bytes: u8,
    stack_height: u16,
    control: [TemplateControlFrame; TEMPLATE_MAX_CONTROL_DEPTH],
    control_depth: usize,
    memory_oob_label: TemplateLabel,
}

#[derive(Clone, Copy)]
struct TemplateControlFrame {
    else_label: TemplateLabel,
    end_label: TemplateLabel,
    entry_stack_height: u16,
    seen_else: bool,
}

impl TemplateControlFrame {
    const EMPTY: Self = Self {
        else_label: TemplateLabel::UNBOUND,
        end_label: TemplateLabel::UNBOUND,
        entry_stack_height: 0,
        seen_else: false,
    };
}

#[derive(Clone, Copy)]
struct TemplateLabel {
    bound_offset: usize,
    unresolved_head: usize,
}

impl TemplateLabel {
    const UNBOUND: Self = Self {
        bound_offset: usize::MAX,
        unresolved_head: TEMPLATE_PATCH_CHAIN_END,
    };

    fn has_unresolved(&self) -> bool {
        self.unresolved_head != TEMPLATE_PATCH_CHAIN_END
    }

    fn emit_jump<'a, A: TemplateBackend<'a>>(&mut self, backend: &mut A) -> Result<(), WasmError> {
        if self.bound_offset != usize::MAX {
            return backend.emit_template_jump_to_offset(self.bound_offset);
        }
        self.unresolved_head = backend.emit_template_jump_placeholder(self.unresolved_head)?;
        Ok(())
    }

    fn bind<'a, A: TemplateBackend<'a>>(&mut self, backend: &mut A) -> Result<(), WasmError> {
        if self.bound_offset != usize::MAX {
            return Err(WasmError::internal("template jit label rebound"));
        }
        let target = backend.core().text.len();
        let mut site = self.unresolved_head;
        while site != TEMPLATE_PATCH_CHAIN_END {
            let next = backend.read_template_jump_next(site)?;
            backend.patch_template_jump(site, target)?;
            site = next;
        }
        self.bound_offset = target;
        self.unresolved_head = TEMPLATE_PATCH_CHAIN_END;
        Ok(())
    }
}

impl<'a> TemplateEmitter<'a> {
    fn new(
        config: BackendConfig,
        spec: &'a FunctionSpec,
        has_memory: bool,
    ) -> Result<Self, WasmError> {
        let shape = template_shape(config, spec, has_memory)?;

        Ok(Self {
            spec,
            local_count: shape.local_count,
            operand_base: shape.operand_base,
            result_count: shape.result_count,
            gp_unit_bytes: shape.gp_unit_bytes,
            stack_height: 0,
            control: [TemplateControlFrame::EMPTY; TEMPLATE_MAX_CONTROL_DEPTH],
            control_depth: 0,
            memory_oob_label: TemplateLabel::UNBOUND,
        })
    }

    fn emit<A: TemplateBackend<'a>>(&mut self, backend: &mut A) -> Result<(), WasmError> {
        self.emit_zero_locals(backend)?;

        let mut code = Payload::from(&**self.spec.code());
        let mut terminated = false;
        while !code.is_empty() {
            let op: Opcode = code.read_u8()?.try_into()?;
            match op {
                Opcode::END => {
                    if self.control_depth != 0 {
                        self.emit_end_control(backend)?;
                        continue;
                    }
                    if !terminated {
                        self.emit_return(backend)?;
                    }
                    self.emit_template_tails(backend)?;
                    return Ok(());
                }
                Opcode::RETURN => {
                    self.emit_return(backend)?;
                    terminated = true;
                }
                Opcode::NOP => {}
                Opcode::UNREACHABLE => return Err(unsupported()),
                Opcode::IF => self.emit_if(backend, &mut code)?,
                Opcode::ELSE => self.emit_else(backend)?,
                Opcode::DROP => {
                    self.pop_slot()?;
                }
                Opcode::LOCAL_GET => {
                    let index = code.read_leb128_u32()?;
                    self.emit_local_get(backend, index)?;
                }
                Opcode::LOCAL_SET => {
                    let index = code.read_leb128_u32()?;
                    self.emit_local_set(backend, index)?;
                }
                Opcode::LOCAL_TEE => {
                    let index = code.read_leb128_u32()?;
                    self.emit_local_tee(backend, index)?;
                }
                Opcode::I32_CONST => {
                    let value = code.read_leb128_i32()? as u32 as u64;
                    self.emit_push_const(backend, ValueType::I32, value)?;
                }
                Opcode::I64_CONST => {
                    let value = code.read_leb128_i64()? as u64;
                    self.emit_push_const(backend, ValueType::I64, value)?;
                }
                Opcode::I32_LOAD => self.emit_i32_load(backend, &mut code)?,
                Opcode::I32_STORE => self.emit_i32_store(backend, &mut code)?,
                Opcode::I32_ADD => {
                    self.emit_int_binary(backend, ValueType::I32, MachineIntBinaryOp::Add)?
                }
                Opcode::I32_SUB => {
                    self.emit_int_binary(backend, ValueType::I32, MachineIntBinaryOp::Sub)?
                }
                Opcode::I32_MUL => {
                    self.emit_int_binary(backend, ValueType::I32, MachineIntBinaryOp::Mul)?
                }
                Opcode::I32_DIV_S => {
                    self.emit_int_binary(backend, ValueType::I32, MachineIntBinaryOp::DivS)?
                }
                Opcode::I32_DIV_U => {
                    self.emit_int_binary(backend, ValueType::I32, MachineIntBinaryOp::DivU)?
                }
                Opcode::I32_REM_S => {
                    self.emit_int_binary(backend, ValueType::I32, MachineIntBinaryOp::RemS)?
                }
                Opcode::I32_REM_U => {
                    self.emit_int_binary(backend, ValueType::I32, MachineIntBinaryOp::RemU)?
                }
                Opcode::I32_AND => {
                    self.emit_int_binary(backend, ValueType::I32, MachineIntBinaryOp::And)?
                }
                Opcode::I32_OR => {
                    self.emit_int_binary(backend, ValueType::I32, MachineIntBinaryOp::Or)?
                }
                Opcode::I32_XOR => {
                    self.emit_int_binary(backend, ValueType::I32, MachineIntBinaryOp::Xor)?
                }
                Opcode::I32_SHL => {
                    self.emit_int_binary(backend, ValueType::I32, MachineIntBinaryOp::Shl)?
                }
                Opcode::I32_SHR_S => {
                    self.emit_int_binary(backend, ValueType::I32, MachineIntBinaryOp::ShrS)?
                }
                Opcode::I32_SHR_U => {
                    self.emit_int_binary(backend, ValueType::I32, MachineIntBinaryOp::ShrU)?
                }
                Opcode::I32_ROTL => {
                    self.emit_int_binary(backend, ValueType::I32, MachineIntBinaryOp::Rotl)?
                }
                Opcode::I32_ROTR => {
                    self.emit_int_binary(backend, ValueType::I32, MachineIntBinaryOp::Rotr)?
                }
                Opcode::I64_ADD => {
                    self.emit_int_binary(backend, ValueType::I64, MachineIntBinaryOp::Add)?
                }
                Opcode::I64_SUB => {
                    self.emit_int_binary(backend, ValueType::I64, MachineIntBinaryOp::Sub)?
                }
                Opcode::I64_MUL => {
                    self.emit_int_binary(backend, ValueType::I64, MachineIntBinaryOp::Mul)?
                }
                Opcode::I64_DIV_S => {
                    self.emit_int_binary(backend, ValueType::I64, MachineIntBinaryOp::DivS)?
                }
                Opcode::I64_DIV_U => {
                    self.emit_int_binary(backend, ValueType::I64, MachineIntBinaryOp::DivU)?
                }
                Opcode::I64_REM_S => {
                    self.emit_int_binary(backend, ValueType::I64, MachineIntBinaryOp::RemS)?
                }
                Opcode::I64_REM_U => {
                    self.emit_int_binary(backend, ValueType::I64, MachineIntBinaryOp::RemU)?
                }
                Opcode::I64_AND => {
                    self.emit_int_binary(backend, ValueType::I64, MachineIntBinaryOp::And)?
                }
                Opcode::I64_OR => {
                    self.emit_int_binary(backend, ValueType::I64, MachineIntBinaryOp::Or)?
                }
                Opcode::I64_XOR => {
                    self.emit_int_binary(backend, ValueType::I64, MachineIntBinaryOp::Xor)?
                }
                Opcode::I64_SHL => {
                    self.emit_int_binary(backend, ValueType::I64, MachineIntBinaryOp::Shl)?
                }
                Opcode::I64_SHR_S => {
                    self.emit_int_binary(backend, ValueType::I64, MachineIntBinaryOp::ShrS)?
                }
                Opcode::I64_SHR_U => {
                    self.emit_int_binary(backend, ValueType::I64, MachineIntBinaryOp::ShrU)?
                }
                Opcode::I64_ROTL => {
                    self.emit_int_binary(backend, ValueType::I64, MachineIntBinaryOp::Rotl)?
                }
                Opcode::I64_ROTR => {
                    self.emit_int_binary(backend, ValueType::I64, MachineIntBinaryOp::Rotr)?
                }
                Opcode::I32_EQZ => self.emit_eqz(backend, ValueType::I32)?,
                Opcode::I64_EQZ => self.emit_eqz(backend, ValueType::I64)?,
                Opcode::I32_EQ => self.emit_int_compare(
                    backend,
                    ValueType::I32,
                    MachineCompareKind::Eq,
                    MachineSign::Unsigned,
                )?,
                Opcode::I32_NE => self.emit_int_compare(
                    backend,
                    ValueType::I32,
                    MachineCompareKind::Ne,
                    MachineSign::Unsigned,
                )?,
                Opcode::I32_LT_S => self.emit_int_compare(
                    backend,
                    ValueType::I32,
                    MachineCompareKind::Lt,
                    MachineSign::Signed,
                )?,
                Opcode::I32_LT_U => self.emit_int_compare(
                    backend,
                    ValueType::I32,
                    MachineCompareKind::Lt,
                    MachineSign::Unsigned,
                )?,
                Opcode::I32_GT_S => self.emit_int_compare(
                    backend,
                    ValueType::I32,
                    MachineCompareKind::Gt,
                    MachineSign::Signed,
                )?,
                Opcode::I32_GT_U => self.emit_int_compare(
                    backend,
                    ValueType::I32,
                    MachineCompareKind::Gt,
                    MachineSign::Unsigned,
                )?,
                Opcode::I32_LE_S => self.emit_int_compare(
                    backend,
                    ValueType::I32,
                    MachineCompareKind::Le,
                    MachineSign::Signed,
                )?,
                Opcode::I32_LE_U => self.emit_int_compare(
                    backend,
                    ValueType::I32,
                    MachineCompareKind::Le,
                    MachineSign::Unsigned,
                )?,
                Opcode::I32_GE_S => self.emit_int_compare(
                    backend,
                    ValueType::I32,
                    MachineCompareKind::Ge,
                    MachineSign::Signed,
                )?,
                Opcode::I32_GE_U => self.emit_int_compare(
                    backend,
                    ValueType::I32,
                    MachineCompareKind::Ge,
                    MachineSign::Unsigned,
                )?,
                Opcode::I64_EQ => self.emit_int_compare(
                    backend,
                    ValueType::I64,
                    MachineCompareKind::Eq,
                    MachineSign::Unsigned,
                )?,
                Opcode::I64_NE => self.emit_int_compare(
                    backend,
                    ValueType::I64,
                    MachineCompareKind::Ne,
                    MachineSign::Unsigned,
                )?,
                Opcode::I64_LT_S => self.emit_int_compare(
                    backend,
                    ValueType::I64,
                    MachineCompareKind::Lt,
                    MachineSign::Signed,
                )?,
                Opcode::I64_LT_U => self.emit_int_compare(
                    backend,
                    ValueType::I64,
                    MachineCompareKind::Lt,
                    MachineSign::Unsigned,
                )?,
                Opcode::I64_GT_S => self.emit_int_compare(
                    backend,
                    ValueType::I64,
                    MachineCompareKind::Gt,
                    MachineSign::Signed,
                )?,
                Opcode::I64_GT_U => self.emit_int_compare(
                    backend,
                    ValueType::I64,
                    MachineCompareKind::Gt,
                    MachineSign::Unsigned,
                )?,
                Opcode::I64_LE_S => self.emit_int_compare(
                    backend,
                    ValueType::I64,
                    MachineCompareKind::Le,
                    MachineSign::Signed,
                )?,
                Opcode::I64_LE_U => self.emit_int_compare(
                    backend,
                    ValueType::I64,
                    MachineCompareKind::Le,
                    MachineSign::Unsigned,
                )?,
                Opcode::I64_GE_S => self.emit_int_compare(
                    backend,
                    ValueType::I64,
                    MachineCompareKind::Ge,
                    MachineSign::Signed,
                )?,
                Opcode::I64_GE_U => self.emit_int_compare(
                    backend,
                    ValueType::I64,
                    MachineCompareKind::Ge,
                    MachineSign::Unsigned,
                )?,
                Opcode::I32_CLZ => {
                    self.emit_int_unary(backend, ValueType::I32, MachineIntUnaryOp::Clz)?
                }
                Opcode::I32_CTZ => {
                    self.emit_int_unary(backend, ValueType::I32, MachineIntUnaryOp::Ctz)?
                }
                Opcode::I32_POPCNT => {
                    self.emit_int_unary(backend, ValueType::I32, MachineIntUnaryOp::Popcnt)?
                }
                Opcode::I32_EXTEND8_S => {
                    self.emit_int_unary(backend, ValueType::I32, MachineIntUnaryOp::Extend8S)?
                }
                Opcode::I32_EXTEND16_S => {
                    self.emit_int_unary(backend, ValueType::I32, MachineIntUnaryOp::Extend16S)?
                }
                Opcode::I64_CLZ => {
                    self.emit_int_unary(backend, ValueType::I64, MachineIntUnaryOp::Clz)?
                }
                Opcode::I64_CTZ => {
                    self.emit_int_unary(backend, ValueType::I64, MachineIntUnaryOp::Ctz)?
                }
                Opcode::I64_POPCNT => {
                    self.emit_int_unary(backend, ValueType::I64, MachineIntUnaryOp::Popcnt)?
                }
                Opcode::I64_EXTEND8_S => {
                    self.emit_int_unary(backend, ValueType::I64, MachineIntUnaryOp::Extend8S)?
                }
                Opcode::I64_EXTEND16_S => {
                    self.emit_int_unary(backend, ValueType::I64, MachineIntUnaryOp::Extend16S)?
                }
                Opcode::I64_EXTEND32_S => {
                    self.emit_int_unary(backend, ValueType::I64, MachineIntUnaryOp::Extend32S)?
                }
                _ => return Err(WasmError::internal(TEMPLATE_UNSUPPORTED)),
            }
        }

        Err(WasmError::malformed(
            "template jit reached unexpected end of code",
        ))
    }

    fn emit_if<A: TemplateBackend<'a>>(
        &mut self,
        backend: &mut A,
        code: &mut Payload<'_>,
    ) -> Result<(), WasmError> {
        let block_type = code.read_u8()?;
        if block_type != TEMPLATE_EMPTY_BLOCK_TYPE {
            return Err(unsupported());
        }
        if self.control_depth == TEMPLATE_MAX_CONTROL_DEPTH {
            return Err(unsupported());
        }
        let cond_slot = self.pop_slot()?;
        self.load_slot(backend, ValueType::I32, cond_slot, self.gp0())?;
        let mut else_label = TemplateLabel::UNBOUND;
        let end_label = TemplateLabel::UNBOUND;
        self.emit_branch_to_label(
            backend,
            &MachineBranchCond::IntCompare {
                width: MachineIntWidth::I32,
                kind: MachineCompareKind::Eq,
                sign: MachineSign::Unsigned,
                lhs: MachineValue::Reg(self.gp0()),
                rhs: MachineValue::Imm64(0),
            },
            TemplateBranchSense::IfTrue,
            &mut else_label,
        )?;
        self.control[self.control_depth] = TemplateControlFrame {
            else_label,
            end_label,
            entry_stack_height: self.stack_height,
            seen_else: false,
        };
        self.control_depth += 1;
        Ok(())
    }

    fn emit_else<A: TemplateBackend<'a>>(&mut self, backend: &mut A) -> Result<(), WasmError> {
        if self.control_depth == 0 {
            return Err(unsupported());
        }
        let index = self.control_depth - 1;
        let mut frame = self.control[index];
        if frame.seen_else || self.stack_height != frame.entry_stack_height {
            return Err(unsupported());
        }
        frame.end_label.emit_jump(backend)?;
        frame.else_label.bind(backend)?;
        frame.seen_else = true;
        self.control[index] = frame;
        Ok(())
    }

    fn emit_end_control<A: TemplateBackend<'a>>(
        &mut self,
        backend: &mut A,
    ) -> Result<(), WasmError> {
        let index = self.control_depth.checked_sub(1).ok_or_else(unsupported)?;
        let mut frame = self.control[index];
        if self.stack_height != frame.entry_stack_height {
            return Err(unsupported());
        }
        if !frame.seen_else {
            frame.else_label.bind(backend)?;
        }
        frame.end_label.bind(backend)?;
        self.control[index] = TemplateControlFrame::EMPTY;
        self.control_depth = index;
        Ok(())
    }

    fn emit_zero_locals<A: TemplateBackend<'a>>(&self, backend: &mut A) -> Result<(), WasmError> {
        let params = self.spec.func_type().params().len() as u16;
        for slot in params..self.local_count {
            backend.emit_template_store(
                MachineStorageType::GpI64,
                self.frame_addr(slot)?,
                MachineMemWidth::U64,
                MachineValue::Imm64(0),
            )?;
        }
        Ok(())
    }

    fn emit_return<A: TemplateBackend<'a>>(&mut self, backend: &mut A) -> Result<(), WasmError> {
        if self.stack_height != self.result_count {
            return Err(unsupported());
        }
        backend.emit_template_return()
    }

    fn emit_template_tails<A: TemplateBackend<'a>>(
        &mut self,
        backend: &mut A,
    ) -> Result<(), WasmError> {
        if self.memory_oob_label.has_unresolved() {
            self.memory_oob_label.bind(backend)?;
            backend.emit_template_trap(MachineTrapKind::MemoryOutOfBounds)?;
        }
        Ok(())
    }

    fn emit_branch_to_label<A: TemplateBackend<'a>>(
        &self,
        backend: &mut A,
        cond: &MachineBranchCond,
        jump_when: TemplateBranchSense,
        label: &mut TemplateLabel,
    ) -> Result<(), WasmError> {
        backend.emit_template_skip_unless(cond, jump_when, A::TEMPLATE_JUMP_BYTES)?;
        label.emit_jump(backend)
    }

    fn emit_local_get<A: TemplateBackend<'a>>(
        &mut self,
        backend: &mut A,
        index: u32,
    ) -> Result<(), WasmError> {
        let ty = local_type(self.spec, index)?;
        self.ensure_type(ty)?;
        let reg = self.gp0();
        backend.emit_template_load(
            MachineRegOwner::LinearValue,
            self.storage_type(ty),
            reg,
            self.frame_addr(
                u16::try_from(index)
                    .map_err(|_| WasmError::invalid("template jit local index overflow"))?,
            )?,
            MachineMemWidth::U64,
            MachineLoadExtension::None,
        )?;
        self.push_reg(backend, ty, reg)
    }

    fn emit_local_set<A: TemplateBackend<'a>>(
        &mut self,
        backend: &mut A,
        index: u32,
    ) -> Result<(), WasmError> {
        let ty = local_type(self.spec, index)?;
        self.ensure_type(ty)?;
        let reg = self.gp0();
        let slot = self.pop_slot()?;
        self.load_slot(backend, ty, slot, reg)?;
        let local_slot = u16::try_from(index)
            .map_err(|_| WasmError::invalid("template jit local index overflow"))?;
        self.store_reg(backend, ty, local_slot, reg)
    }

    fn emit_local_tee<A: TemplateBackend<'a>>(
        &mut self,
        backend: &mut A,
        index: u32,
    ) -> Result<(), WasmError> {
        let ty = local_type(self.spec, index)?;
        self.ensure_type(ty)?;
        let reg = self.gp0();
        let slot = self.peek_slot()?;
        self.load_slot(backend, ty, slot, reg)?;
        let local_slot = u16::try_from(index)
            .map_err(|_| WasmError::invalid("template jit local index overflow"))?;
        self.store_reg(backend, ty, local_slot, reg)
    }

    fn emit_push_const<A: TemplateBackend<'a>>(
        &mut self,
        backend: &mut A,
        ty: ValueType,
        value: u64,
    ) -> Result<(), WasmError> {
        self.ensure_type(ty)?;
        let reg = self.gp0();
        backend.emit_template_move(
            MachineRegOwner::LinearValue,
            self.storage_type(ty),
            reg,
            MachineValue::Imm64(value),
        )?;
        self.push_reg(backend, ty, reg)
    }

    fn emit_i32_load<A: TemplateBackend<'a>>(
        &mut self,
        backend: &mut A,
        code: &mut Payload<'_>,
    ) -> Result<(), WasmError> {
        let _align = code.read_leb128_u32()?;
        let offset = code.read_leb128_u32()?;
        if offset != 0 {
            return Err(unsupported());
        }
        let addr_slot = self.pop_slot()?;
        self.load_slot(backend, ValueType::I32, addr_slot, self.gp0())?;
        self.emit_mem0_i32_access_addr(backend)?;
        backend.emit_template_load(
            MachineRegOwner::LinearValue,
            MachineStorageType::GpWord,
            self.gp0(),
            MachineAddr {
                base: self.gp0(),
                offset: 0,
            },
            MachineMemWidth::U32,
            MachineLoadExtension::None,
        )?;
        self.push_reg(backend, ValueType::I32, self.gp0())
    }

    fn emit_i32_store<A: TemplateBackend<'a>>(
        &mut self,
        backend: &mut A,
        code: &mut Payload<'_>,
    ) -> Result<(), WasmError> {
        let _align = code.read_leb128_u32()?;
        let offset = code.read_leb128_u32()?;
        if offset != 0 {
            return Err(unsupported());
        }
        let value_slot = self.pop_slot()?;
        let addr_slot = self.pop_slot()?;
        self.load_slot(backend, ValueType::I32, addr_slot, self.gp0())?;
        self.emit_mem0_i32_access_addr(backend)?;
        self.load_slot(backend, ValueType::I32, value_slot, self.gp1())?;
        backend.emit_template_store(
            MachineStorageType::GpWord,
            MachineAddr {
                base: self.gp0(),
                offset: 0,
            },
            MachineMemWidth::U32,
            MachineValue::Reg(self.gp1()),
        )
    }

    fn emit_mem0_i32_access_addr<A: TemplateBackend<'a>>(
        &mut self,
        backend: &mut A,
    ) -> Result<(), WasmError> {
        if self.gp_unit_bytes == 8 {
            backend.emit_template_convert(
                MachineConvertOp::I64ExtendI32U,
                self.gp0(),
                MachineValue::Reg(self.gp0()),
            )?;
        }
        backend.emit_template_int_binary(
            self.word_int_width(),
            MachineIntBinaryOp::Add,
            self.gp1(),
            MachineValue::Reg(self.gp0()),
            MachineValue::Imm64(4),
        )?;
        if self.gp_unit_bytes == 4 {
            let mut oob_label = self.memory_oob_label;
            self.emit_branch_to_label(
                backend,
                &MachineBranchCond::IntCompare {
                    width: MachineIntWidth::I32,
                    kind: MachineCompareKind::Lt,
                    sign: MachineSign::Unsigned,
                    lhs: MachineValue::Reg(self.gp1()),
                    rhs: MachineValue::Reg(self.gp0()),
                },
                TemplateBranchSense::IfTrue,
                &mut oob_label,
            )?;
            self.memory_oob_label = oob_label;
        }
        let mut oob_label = self.memory_oob_label;
        self.emit_branch_to_label(
            backend,
            &MachineBranchCond::IntCompare {
                width: self.word_int_width(),
                kind: MachineCompareKind::Le,
                sign: MachineSign::Unsigned,
                lhs: MachineValue::Reg(self.gp1()),
                rhs: MachineValue::Reg(MACHINE_MEM0_SIZE_REG),
            },
            TemplateBranchSense::IfFalse,
            &mut oob_label,
        )?;
        self.memory_oob_label = oob_label;
        backend.emit_template_int_binary(
            self.word_int_width(),
            MachineIntBinaryOp::Add,
            self.gp0(),
            MachineValue::Reg(MACHINE_MEM0_BASE_REG),
            MachineValue::Reg(self.gp0()),
        )
    }

    fn emit_int_binary<A: TemplateBackend<'a>>(
        &mut self,
        backend: &mut A,
        ty: ValueType,
        op: MachineIntBinaryOp,
    ) -> Result<(), WasmError> {
        self.ensure_type(ty)?;
        let rhs_slot = self.pop_slot()?;
        let lhs_slot = self.pop_slot()?;
        self.load_slot(backend, ty, lhs_slot, self.gp0())?;
        self.load_slot(backend, ty, rhs_slot, self.gp1())?;
        backend.emit_template_int_binary(
            self.int_width(ty),
            op,
            self.gp0(),
            MachineValue::Reg(self.gp0()),
            MachineValue::Reg(self.gp1()),
        )?;
        self.push_reg(backend, ty, self.gp0())
    }

    fn emit_int_unary<A: TemplateBackend<'a>>(
        &mut self,
        backend: &mut A,
        ty: ValueType,
        op: MachineIntUnaryOp,
    ) -> Result<(), WasmError> {
        self.ensure_type(ty)?;
        let slot = self.pop_slot()?;
        self.load_slot(backend, ty, slot, self.gp0())?;
        backend.emit_template_int_unary(
            self.int_width(ty),
            op,
            self.gp0(),
            MachineValue::Reg(self.gp0()),
        )?;
        self.push_reg(backend, ty, self.gp0())
    }

    fn emit_eqz<A: TemplateBackend<'a>>(
        &mut self,
        backend: &mut A,
        ty: ValueType,
    ) -> Result<(), WasmError> {
        self.ensure_type(ty)?;
        let slot = self.pop_slot()?;
        self.load_slot(backend, ty, slot, self.gp0())?;
        backend.emit_template_int_compare(
            self.int_width(ty),
            MachineCompareKind::Eq,
            MachineSign::Unsigned,
            self.gp0(),
            MachineValue::Reg(self.gp0()),
            MachineValue::Imm64(0),
        )?;
        self.push_reg(backend, ValueType::I32, self.gp0())
    }

    fn emit_int_compare<A: TemplateBackend<'a>>(
        &mut self,
        backend: &mut A,
        ty: ValueType,
        kind: MachineCompareKind,
        sign: MachineSign,
    ) -> Result<(), WasmError> {
        self.ensure_type(ty)?;
        let rhs_slot = self.pop_slot()?;
        let lhs_slot = self.pop_slot()?;
        self.load_slot(backend, ty, lhs_slot, self.gp0())?;
        self.load_slot(backend, ty, rhs_slot, self.gp1())?;
        backend.emit_template_int_compare(
            self.int_width(ty),
            kind,
            sign,
            self.gp0(),
            MachineValue::Reg(self.gp0()),
            MachineValue::Reg(self.gp1()),
        )?;
        self.push_reg(backend, ValueType::I32, self.gp0())
    }

    fn load_slot<A: TemplateBackend<'a>>(
        &self,
        backend: &mut A,
        ty: ValueType,
        slot: u16,
        reg: MachineReg,
    ) -> Result<(), WasmError> {
        backend.emit_template_load(
            MachineRegOwner::LinearValue,
            self.storage_type(ty),
            reg,
            self.frame_addr(slot)?,
            MachineMemWidth::U64,
            MachineLoadExtension::None,
        )
    }

    fn store_reg<A: TemplateBackend<'a>>(
        &self,
        backend: &mut A,
        ty: ValueType,
        slot: u16,
        reg: MachineReg,
    ) -> Result<(), WasmError> {
        backend.emit_template_store(
            self.storage_type(ty),
            self.frame_addr(slot)?,
            MachineMemWidth::U64,
            MachineValue::Reg(reg),
        )
    }

    fn push_reg<A: TemplateBackend<'a>>(
        &mut self,
        backend: &mut A,
        ty: ValueType,
        reg: MachineReg,
    ) -> Result<(), WasmError> {
        let slot = self.operand_slot(self.stack_height)?;
        self.store_reg(backend, ty, slot, reg)?;
        self.stack_height = self
            .stack_height
            .checked_add(1)
            .ok_or_else(|| WasmError::internal("template jit stack height overflow"))?;
        Ok(())
    }

    fn pop_slot(&mut self) -> Result<u16, WasmError> {
        self.stack_height = self.stack_height.checked_sub(1).ok_or_else(unsupported)?;
        self.operand_slot(self.stack_height)
    }

    fn peek_slot(&self) -> Result<u16, WasmError> {
        let index = self.stack_height.checked_sub(1).ok_or_else(unsupported)?;
        self.operand_slot(index)
    }

    fn operand_slot(&self, index: u16) -> Result<u16, WasmError> {
        self.operand_base
            .checked_add(index)
            .ok_or_else(|| WasmError::internal("template jit operand slot overflow"))
    }

    fn frame_addr(&self, slot: u16) -> Result<MachineAddr, WasmError> {
        let offset = i32::from(slot)
            .checked_mul(8)
            .ok_or_else(|| WasmError::internal("template jit frame offset overflow"))?;
        Ok(MachineAddr {
            base: self.frame_base(),
            offset,
        })
    }

    fn storage_type(&self, ty: ValueType) -> MachineStorageType {
        match ty {
            ValueType::I64 => MachineStorageType::GpI64,
            _ => MachineStorageType::GpWord,
        }
    }

    fn int_width(&self, ty: ValueType) -> MachineIntWidth {
        match ty {
            ValueType::I64 => MachineIntWidth::I64,
            _ => MachineIntWidth::I32,
        }
    }

    fn word_int_width(&self) -> MachineIntWidth {
        match self.gp_unit_bytes {
            4 => MachineIntWidth::I32,
            8 => MachineIntWidth::I64,
            _ => MachineIntWidth::I32,
        }
    }

    fn gp0(&self) -> MachineReg {
        MachineReg(4)
    }

    fn gp1(&self) -> MachineReg {
        MachineReg(5)
    }

    fn frame_base(&self) -> MachineReg {
        MachineReg(1)
    }

    fn ensure_type(&self, ty: ValueType) -> Result<(), WasmError> {
        ensure_template_type(self.gp_unit_bytes, ty)
    }
}

fn local_type(spec: &FunctionSpec, index: u32) -> Result<ValueType, WasmError> {
    let index = usize::try_from(index)
        .map_err(|_| WasmError::invalid("template jit local index overflow"))?;
    let params = spec.func_type().params();
    if let Some(ty) = params.get(index) {
        return Ok(*ty);
    }
    spec.locals()
        .get(index.saturating_sub(params.len()))
        .copied()
        .ok_or_else(|| WasmError::invalid("template jit local index out of range"))
}

fn ensure_template_type(gp_unit_bytes: u8, ty: ValueType) -> Result<(), WasmError> {
    match (gp_unit_bytes, ty) {
        (_, ValueType::I32) => Ok(()),
        (8, ValueType::I64) => Ok(()),
        _ => Err(unsupported()),
    }
}

#[cfg(test)]
mod tests {
    use tracked_alloc::rc::Rc;

    use super::scan_function;
    use crate::{
        collections,
        module::{entities::FunctionSpec, type_defs::FunctionType},
        value_type::ValueType,
        vm::jit::backend::BackendConfig,
    };

    fn test_config(gp_unit_bytes: u8) -> BackendConfig {
        BackendConfig::new(gp_unit_bytes, 4, 0, 2)
    }

    fn function(
        params: collections::Vec<ValueType>,
        results: collections::Vec<ValueType>,
        code: &[u8],
    ) -> FunctionSpec {
        let ty = Rc::new(FunctionType::new(params, results));
        let mut spec = FunctionSpec::new(ty, 0);
        spec.set_code(code.into());
        spec
    }

    #[test]
    fn template_scan_rejects_result_type_mismatch() {
        let spec = function(
            collections::vec![],
            collections::vec![ValueType::I64],
            &[0x41, 0x00, 0x0b],
        );
        let err = scan_function(test_config(8), &spec, false).unwrap_err();
        assert!(matches!(err, crate::error::WasmError::Invalid(_)));
    }

    #[test]
    fn template_scan_rejects_memory_without_memory0() {
        let spec = function(
            collections::vec![],
            collections::vec![ValueType::I32],
            &[
                0x41, 0x00, // i32.const 0
                0x28, 0x02, 0x00, // i32.load align=2 offset=0
                0x0b,
            ],
        );
        let err = scan_function(test_config(4), &spec, false).unwrap_err();
        assert!(matches!(err, crate::error::WasmError::Invalid(_)));
    }

    #[test]
    fn template_scan_accepts_valid_i32_memory_access() {
        let spec = function(
            collections::vec![],
            collections::vec![ValueType::I32],
            &[
                0x41, 0x00, // i32.const 0
                0x28, 0x02, 0x00, // i32.load align=2 offset=0
                0x0b,
            ],
        );
        let summary = scan_function(test_config(4), &spec, true).unwrap();
        assert_eq!(summary.result_count, 1);
        assert_eq!(summary.max_stack_height, 1);
    }
}
