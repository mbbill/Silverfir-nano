use alloc::{string::String, vec::Vec};
use core::hash::{Hash, Hasher};

use crate::module::entities::FunctionSpec;
use crate::vm::entities::{FunctionInst, ModuleInst};
use crate::vm::lockstep::{
    ProgramCheckpointPlan,
    CheckpointSite,
    CheckpointSnapshot,
    StepOutcome,
    Stepper,
};
use crate::vm::native::instruction::NativeInst;
use crate::vm::store::Store;
use crate::vm::value::Value;
use crate::error::WasmError;

use super::context::Context;
use super::runtime::{run_from_state, term_entry};

pub struct NativeStepper<'a> {
    checkpoint_plan: ProgramCheckpointPlan,
    func_idx: u32,
    _code: &'a super::code::NativeCode,
    stack: Vec<u64>,
    fp_index: usize,
    results_len: usize,
    ctx: Context,
    next_inst: *mut NativeInst,
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

impl<'a> NativeStepper<'a> {
    pub fn eval(
        func_inst: &'a FunctionInst,
        store: &mut Store,
        args: &[Value],
        checkpoint_plan: ProgramCheckpointPlan,
    ) -> Result<Self, WasmError> {
        let FunctionInst::Local { spec, .. } = func_inst else {
            return Err(WasmError::invalid(
                "lockstep native stepper only supports local functions".into(),
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
        let native_code = spec
            .get_native_code()
            .ok_or_else(|| WasmError::internal("missing checkpointed native code".into()))?;

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
        ctx.hot.term_entry = term_entry();

        let frame_size = params_len + locals_len;
        unsafe {
            *fp.add(frame_size) = 0;
            *fp.add(frame_size + 1) = 0;
            *fp.add(frame_size + 2) = 0;
        }

        let next_inst = native_code.entry_ptr();

        Ok(Self {
            checkpoint_plan,
            func_idx,
            _code: native_code,
            stack,
            fp_index,
            results_len: ft.results().len(),
            ctx,
            next_inst,
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

    fn stack_base(&self) -> *mut u64 {
        self.stack.as_ptr() as *mut u64
    }

    fn fp(&self) -> *mut u64 {
        unsafe { self.stack_base().add(self.fp_index) }
    }

    fn site_by_ordinal(&self, ordinal: u32) -> CheckpointSite {
        self.checkpoint_plan
            .site_by_ordinal(ordinal)
            .expect("checkpoint ordinal must exist in plan")
    }

    fn capture_snapshot(&self) -> CheckpointSnapshot {
        let globals = unsafe {
            self.ctx
                .current_module
                .as_ref()
                .map(|module| module.globals.iter().map(|g| g.value.to_raw()).collect())
                .unwrap_or_else(Vec::new)
        };
        let memory_page_hashes = unsafe {
            self.ctx
                .current_module
                .as_ref()
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

    pub fn result_values(&self) -> &[u64] {
        &self.stack[..self.results_len]
    }
}

impl Stepper for NativeStepper<'_> {
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

        self.ctx.hot.checkpoint_next_inst = core::ptr::null_mut();
        self.ctx.hot.checkpoint_ordinal = 0;
        self.ctx.hot.resume_fp = core::ptr::null_mut();

        let fp = self.fp();
        unsafe {
            run_from_state(
                &mut self.ctx,
                self.next_inst,
                fp,
                self.l0,
                self.l1,
                self.l2,
                self.t0,
                self.t1,
                self.t2,
                self.t3,
            )
        };

        if !self.ctx.hot.checkpoint_next_inst.is_null() {
            self.next_inst = self.ctx.hot.checkpoint_next_inst;
            let resume_fp = self.ctx.hot.resume_fp;
            if !resume_fp.is_null() {
                let base = self.stack_base() as usize;
                let fp = resume_fp as usize;
                self.fp_index = (fp - base) / core::mem::size_of::<u64>();
            }
            self.l0 = self.ctx.checkpoint_l0;
            self.l1 = self.ctx.checkpoint_l1;
            self.l2 = self.ctx.checkpoint_l2;
            self.t0 = self.ctx.checkpoint_t0;
            self.t1 = self.ctx.checkpoint_t1;
            self.t2 = self.ctx.checkpoint_t2;
            self.t3 = self.ctx.checkpoint_t3;
            let site = self.site_by_ordinal(self.ctx.hot.checkpoint_ordinal as u32);
            self.ctx.hot.checkpoint_next_inst = core::ptr::null_mut();
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

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use alloc::rc::Rc;
    use alloc::vec;
    use crate::module::type_context::TypeContext;
    use crate::module::type_defs::FunctionType;
    use crate::value_type::ValueType;
    use crate::vm::entities::{FunctionInst, ModuleInst};
    use crate::vm::lockstep::{CheckpointKind, CheckpointMode};

    #[test]
    fn native_stepper_function_mode_finishes_cleanly() {
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
            0x41, 0x01, // i32.const 1
            0x0f,       // return
            0x0b,       // end
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

        let mut stepper = NativeStepper::eval(func_inst, &mut store, &[], plan).expect("native stepper");
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
