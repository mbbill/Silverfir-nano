use alloc::{string::String, vec::Vec};
use core::hash::{Hash, Hasher};

use crate::error::WasmError;
use crate::module::entities::FunctionSpec;
use crate::vm::entities::{FunctionInst, ModuleInst};
use crate::vm::interp::fast::instruction::Instruction;
use crate::vm::lockstep::{
    ProgramCheckpointPlan,
    CheckpointSite,
    CheckpointSnapshot,
    StepOutcome,
    Stepper,
};
use crate::vm::store::Store;
use crate::vm::value::Value;

use super::context::Context;
use super::handlers::{run_trampoline, NextHandler};

pub struct FastStepper<'a> {
    checkpoint_plan: ProgramCheckpointPlan,
    func_idx: u32,
    _code: &'a super::fast_code::FastCode,
    stack: Vec<u64>,
    fp_index: usize,
    results_len: usize,
    ctx: Context,
    next_pc: *mut Instruction,
    l0: u64,
    l1: u64,
    l2: u64,
    t0: u64,
    t1: u64,
    t2: u64,
    t3: u64,
    started: bool,
    finished: bool,
}

impl<'a> FastStepper<'a> {
    pub fn eval(
        func_inst: &'a FunctionInst,
        store: &mut Store,
        args: &[Value],
        checkpoint_plan: ProgramCheckpointPlan,
    ) -> Result<Self, WasmError> {
        let FunctionInst::Local { spec, .. } = func_inst else {
            return Err(WasmError::invalid(
                "lockstep fast stepper only supports local functions".into(),
            ));
        };

        let func_idx = {
            let module = store.module();
            module
                .functions
                .iter()
                .enumerate()
                .find_map(|(idx, candidate)| match candidate {
                    FunctionInst::Local { spec: candidate_spec, .. }
                        if core::ptr::eq(candidate_spec, spec) => Some(idx as u32),
                    _ => None,
                })
                .ok_or_else(|| WasmError::internal("local function missing from module".into()))?
        };

        super::compiler::build_module_with_checkpoints(store, &checkpoint_plan)?;
        let fast_code = spec
            .get_fast_code()
            .ok_or_else(|| WasmError::internal("missing checkpointed fast code".into()))?;

        let ft = spec.func_type();
        let params_len = ft.params().len();
        if args.len() != params_len {
            return Err(WasmError::invalid(alloc::format!(
                "invalid argument count: got {}, expected {}",
                args.len(),
                params_len
            )));
        }

        let mut stack = alloc::vec![0u64; crate::constants::MAX_STACK_SIZE / core::mem::size_of::<u64>()];
        let stack_base = stack.as_mut_ptr();
        let stack_end = unsafe { stack_base.add(stack.len()) };
        for (i, arg) in args.iter().enumerate() {
            unsafe { core::ptr::write(stack_base.add(i), arg.to_raw()) };
        }

        let fp_index = 0;
        let fp = stack_base;
        let locals_len = spec.locals().len();
        if locals_len > 0 {
            unsafe { core::ptr::write_bytes(fp.add(params_len), 0, locals_len) };
        }

        let (module_ptr, heap_base, heap_size) = {
            let module = store.module();
            let heap = if !module.memories.is_empty() {
                let m = &module.memories[0];
                (m.data.as_ptr() as *mut u8, m.data.len())
            } else {
                (core::ptr::null_mut(), 0usize)
            };
            (module as *const ModuleInst, heap.0, heap.1)
        };
        let mut ctx = Context::new(
            store as *mut Store,
            module_ptr,
            stack_end,
            heap_base,
            heap_size as u64,
        );
        ctx.hot.term_inst = super::handlers::term();

        Ok(Self {
            checkpoint_plan,
            func_idx,
            _code: fast_code,
            stack,
            fp_index,
            results_len: ft.results().len(),
            ctx,
            next_pc: fast_code.entry_ptr(),
            l0: 0,
            l1: 0,
            l2: 0,
            t0: 0,
            t1: 0,
            t2: 0,
            t3: 0,
            started: false,
            finished: false,
        })
    }

    fn fp(&self) -> *mut u64 {
        unsafe { (self.stack.as_ptr() as *mut u64).add(self.fp_index) }
    }

    fn next_nh(&self) -> NextHandler {
        unsafe { core::mem::transmute((*self.next_pc.add(1)).handler) }
    }

    fn site_by_ordinal(&self, ordinal: u32) -> CheckpointSite {
        self.checkpoint_plan
            .site_by_ordinal(ordinal)
            .expect("checkpoint ordinal must exist in plan")
    }

    fn capture_snapshot(&self) -> CheckpointSnapshot {
        let globals = unsafe {
            self.ctx
                .current_module()
                .map(|module| module.globals.iter().map(|g| g.value.to_raw()).collect())
                .unwrap_or_else(Vec::new)
        };
        let memory_page_hashes = unsafe {
            self.ctx
                .current_module()
                .map(|module| {
                    let mut hashes = Vec::new();
                    for mem in &module.memories {
                        for page in mem.data.chunks(crate::constants::WASM_PAGE_SIZE) {
                            hashes.push(hash_page(page));
                        }
                    }
                    hashes
                })
                .unwrap_or_else(Vec::new)
        };
        let trap = if let Some(error) = &self.ctx.error {
            Some(alloc::format!("{:?}", error))
        } else if !self.ctx.hot.trap_message.is_null() {
            Some(unsafe {
                core::ffi::CStr::from_ptr(self.ctx.hot.trap_message)
                    .to_string_lossy()
                    .into_owned()
            })
        } else {
            None
        };

        CheckpointSnapshot {
            call_depth: canonical_call_depth(self.ctx.hot.call_depth),
            result_values: self.stack[..self.results_len].to_vec(),
            globals,
            memory_page_hashes,
            trap,
        }
    }
}

impl Stepper for FastStepper<'_> {
    fn checkpoint_plan(&self) -> &ProgramCheckpointPlan {
        &self.checkpoint_plan
    }

    fn step_until_checkpoint(&mut self) -> StepOutcome {
        if !self.started {
            self.started = true;
            let site = self
                .checkpoint_plan
                .function_plan(self.func_idx)
                .and_then(|plan| plan.entry_site())
                .expect("entry checkpoint must exist for function plan");
            return StepOutcome::Checkpoint {
                site,
                snapshot: self.capture_snapshot(),
            };
        }

        if self.finished {
            return StepOutcome::Finished {
                snapshot: self.capture_snapshot(),
            };
        }

        self.ctx.checkpoint_pc = core::ptr::null_mut();
        self.ctx.checkpoint_fp = core::ptr::null_mut();
        self.ctx.checkpoint_ordinal = 0;

        let fp = self.fp();
        let nh = self.next_nh();
        unsafe {
            run_trampoline(
                &mut self.ctx,
                self.next_pc,
                fp,
                self.l0,
                self.l1,
                self.l2,
                self.t0,
                self.t1,
                self.t2,
                self.t3,
                nh,
            );
        }

        if !self.ctx.checkpoint_pc.is_null() {
            self.next_pc = self.ctx.checkpoint_pc;
            self.l0 = self.ctx.checkpoint_l0;
            self.l1 = self.ctx.checkpoint_l1;
            self.l2 = self.ctx.checkpoint_l2;
            self.t0 = self.ctx.checkpoint_t0;
            self.t1 = self.ctx.checkpoint_t1;
            self.t2 = self.ctx.checkpoint_t2;
            self.t3 = self.ctx.checkpoint_t3;
            if !self.ctx.checkpoint_fp.is_null() {
                let base = self.stack.as_ptr() as usize;
                let fp = self.ctx.checkpoint_fp as usize;
                self.fp_index = (fp - base) / core::mem::size_of::<u64>();
            }
            let site = self.site_by_ordinal(self.ctx.checkpoint_ordinal as u32);
            return StepOutcome::Checkpoint {
                site,
                snapshot: self.capture_snapshot(),
            };
        }

        self.finished = true;
        StepOutcome::Finished {
            snapshot: self.capture_snapshot(),
        }
    }
}

fn hash_page(bytes: &[u8]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

fn canonical_call_depth(raw: u64) -> u32 {
    if raw == u64::MAX {
        0
    } else {
        raw as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::rc::Rc;
    use crate::module::type_context::TypeContext;
    use crate::module::type_defs::FunctionType;
    use crate::value_type::ValueType;
    use crate::vm::lockstep::{CheckpointKind, CheckpointMode};

    #[test]
    fn fast_stepper_function_mode_finishes_cleanly() {
        let sig = Rc::new(FunctionType::new(vec![], vec![ValueType::I32]));
        let types = TypeContext::new(vec![sig.clone()]);
        let mut module = ModuleInst::new("test".into(), types);
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(sig, 0),
            type_index: 0,
        });
        let FunctionInst::Local { spec, .. } = &mut module.functions[0] else {
            unreachable!();
        };
        spec.set_code(crate::module::entities::Bytecode::from(&[
            0x41, 0x01,
            0x0f,
            0x0b,
        ][..]));

        let mut store = Store::new(module);
        let func_inst_ptr = store.function(0) as *const FunctionInst;
        let func_inst = unsafe { &*func_inst_ptr };
        let FunctionInst::Local { spec, .. } = func_inst else {
            unreachable!();
        };
        let plan = crate::vm::lockstep::ProgramCheckpointPlan::for_module(
            store.module(),
            &store,
            CheckpointMode::Function,
        )
        .expect("checkpoint plan");

        let mut stepper = FastStepper::eval(func_inst, &mut store, &[], plan).expect("fast stepper");
        let first = stepper.step_until_checkpoint();
        let StepOutcome::Checkpoint { site, .. } = first else {
            panic!("expected function entry checkpoint");
        };
        assert_eq!(site.id.kind, CheckpointKind::FunctionEntry);

        let second = stepper.step_until_checkpoint();
        let StepOutcome::Finished { snapshot } = second else {
            panic!("expected finished outcome");
        };
        assert_eq!(snapshot.result_values, vec![1]);
    }
}
