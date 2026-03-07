use super::codegen::JitEmitter;

macro_rules! define_semantics_emitter {
    ($(fn $name:ident($($arg:ident : $ty:ty),*);)*) => {
        pub trait SemanticsEmitter {
            $(fn $name(&mut self, $($arg: $ty),*);)*
        }

        impl SemanticsEmitter for JitEmitter<'_> {
            $(fn $name(&mut self, $($arg: $ty),*) {
                JitEmitter::$name(self, $($arg),*)
            })*
        }
    };
}

define_semantics_emitter! {
    fn i32_add();
    fn i32_sub();
    fn i32_mul();
    fn i32_and();
    fn i32_or();
    fn i32_xor();
    fn i32_shl();
    fn i32_shr_u();
    fn i32_shr_s();
    fn i32_rotl();
    fn i32_rotr();
    fn i64_add();
    fn i64_sub();
    fn i64_mul();
    fn i64_and();
    fn i64_or();
    fn i64_xor();
    fn i64_shl();
    fn i64_shr_u();
    fn i64_shr_s();
    fn i64_rotl();
    fn i64_rotr();
    fn i32_eq();
    fn i32_ne();
    fn i32_lt_s();
    fn i32_lt_u();
    fn i32_gt_s();
    fn i32_gt_u();
    fn i32_le_s();
    fn i32_le_u();
    fn i32_ge_s();
    fn i32_ge_u();
    fn i64_eq();
    fn i64_ne();
    fn i64_lt_s();
    fn i64_lt_u();
    fn i64_gt_s();
    fn i64_gt_u();
    fn i64_le_s();
    fn i64_le_u();
    fn i64_ge_s();
    fn i64_ge_u();
    fn i32_eqz();
    fn i32_clz();
    fn i32_ctz();
    fn i64_eqz();
    fn i64_clz();
    fn i64_ctz();
    fn i32_const(value: u32);
    fn i64_const(value: u64);
    fn f32_const(value: u32);
    fn f64_const(value: u64);
    fn local_get_ln(n: u8);
    fn local_get(idx: u16);
    fn local_set_ln(n: u8);
    fn local_set(idx: u16);
    fn local_tee_ln(n: u8);
    fn local_tee(idx: u16);
    fn drop_val();
    fn select();
    fn f64_add();
    fn f64_sub();
    fn f64_mul();
    fn f64_div();
    fn f64_min();
    fn f64_max();
    fn f32_add();
    fn f32_sub();
    fn f32_mul();
    fn f32_div();
    fn f32_min();
    fn f32_max();
    fn f64_eq();
    fn f64_ne();
    fn f64_lt();
    fn f64_gt();
    fn f64_le();
    fn f64_ge();
    fn f32_eq();
    fn f32_ne();
    fn f32_lt();
    fn f32_gt();
    fn f32_le();
    fn f32_ge();
    fn f64_abs();
    fn f64_neg();
    fn f64_sqrt();
    fn f64_ceil();
    fn f64_floor();
    fn f64_trunc();
    fn f64_nearest();
    fn f32_abs();
    fn f32_neg();
    fn f32_sqrt();
    fn f32_ceil();
    fn f32_floor();
    fn f32_trunc();
    fn f32_nearest();
    fn i32_wrap_i64();
    fn i64_extend_i32_s();
    fn i64_extend_i32_u();
    fn f64_promote_f32();
    fn f32_demote_f64();
    fn f32_convert_i32_s();
    fn f32_convert_i32_u();
    fn f32_convert_i64_s();
    fn f32_convert_i64_u();
    fn f64_convert_i32_s();
    fn f64_convert_i32_u();
    fn f64_convert_i64_s();
    fn f64_convert_i64_u();
    fn i32_trunc_sat_f32_s();
    fn i32_trunc_sat_f32_u();
    fn i32_trunc_sat_f64_s();
    fn i32_trunc_sat_f64_u();
    fn i64_trunc_sat_f32_s();
    fn i64_trunc_sat_f32_u();
    fn i64_trunc_sat_f64_s();
    fn i64_trunc_sat_f64_u();
    fn i32_extend8_s();
    fn i32_extend16_s();
    fn i64_extend8_s();
    fn i64_extend16_s();
    fn i64_extend32_s();
    fn i32_load(offset: u32);
    fn f32_load(offset: u32);
    fn i32_load8_s(offset: u32);
    fn i32_load8_u(offset: u32);
    fn i32_load16_s(offset: u32);
    fn i32_load16_u(offset: u32);
    fn i64_load(offset: u32);
    fn f64_load(offset: u32);
    fn i64_load8_s(offset: u32);
    fn i64_load8_u(offset: u32);
    fn i64_load16_s(offset: u32);
    fn i64_load16_u(offset: u32);
    fn i64_load32_s(offset: u32);
    fn i64_load32_u(offset: u32);
    fn i32_store(offset: u32);
    fn f32_store(offset: u32);
    fn i32_store8(offset: u32);
    fn i32_store16(offset: u32);
    fn i64_store(offset: u32);
    fn f64_store(offset: u32);
    fn i64_store8(offset: u32);
    fn i64_store16(offset: u32);
    fn i64_store32(offset: u32);
    fn init_locals(k0: u16, k1: u16, k2: u16);
    fn spill(slot: u16, count: u8, variant: u8);
    fn fill(slot: u16, count: u8, variant: u8);
}
