;; With AVX, x86_64 folds a scalar FP load or a constant into the memory
;; source of vaddsd/vsubsd/vmulsd/vdivsd (and the ss forms). Operand order
;; must survive for the non-commutative ops whichever side the load or the
;; constant is on, a loaded value read twice must stay in a register, and a
;; folded load must still trap out of bounds.

(module
  (memory 1)

  (func (export "setup")
    (f64.store (i32.const 8) (f64.const 1.5))
    (f64.store (i32.const 16) (f64.const -2.25))
    (f32.store (i32.const 32) (f32.const 3.5))
    (f32.store (i32.const 36) (f32.const 0.75))
    (f64.store (i32.const 65528) (f64.const 10.0)))

  (func (export "add_r") (param f64 i32) (result f64)
    (f64.add (local.get 0) (f64.load offset=8 (local.get 1))))
  (func (export "sub_r") (param f64 i32) (result f64)
    (f64.sub (local.get 0) (f64.load offset=8 (local.get 1))))
  (func (export "sub_l") (param f64 i32) (result f64)
    (f64.sub (f64.load offset=8 (local.get 1)) (local.get 0)))
  (func (export "div_r") (param f64 i32) (result f64)
    (f64.div (local.get 0) (f64.load offset=8 (local.get 1))))
  (func (export "div_l") (param f64 i32) (result f64)
    (f64.div (f64.load offset=8 (local.get 1)) (local.get 0)))
  (func (export "mul_l") (param f64 i32) (result f64)
    (f64.mul (f64.load offset=8 (local.get 1)) (local.get 0)))

  (func (export "add_r32") (param f32 i32) (result f32)
    (f32.add (local.get 0) (f32.load offset=32 (local.get 1))))
  (func (export "sub_l32") (param f32 i32) (result f32)
    (f32.sub (f32.load offset=32 (local.get 1)) (local.get 0)))
  (func (export "div_r32") (param f32 i32) (result f32)
    (f32.div (local.get 0) (f32.load offset=32 (local.get 1))))
  (func (export "mul_l32") (param f32 i32) (result f32)
    (f32.mul (f32.load offset=32 (local.get 1)) (local.get 0)))

  ;; The loaded value feeds two operations.
  (func (export "square") (param f64 i32) (result f64)
    (local $t f64)
    (local.set $t (f64.load offset=8 (local.get 1)))
    (f64.add (local.get $t) (f64.mul (local.get $t) (local.get 0))))

  ;; A loop-carried accumulator reading one element per iteration.
  (func (export "sum") (param $n i32) (result f64)
    (local $i i32) (local $acc f64)
    (block $done
      (loop $l
        (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
        (local.set $acc
          (f64.add (local.get $acc)
            (f64.load offset=8 (i32.shl (local.get $i) (i32.const 3)))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $l)))
    (local.get $acc))

  (func (export "mulc") (param f64) (result f64)
    (f64.mul (local.get 0) (f64.const 2.5)))
  (func (export "divc") (param f64) (result f64)
    (f64.div (local.get 0) (f64.const 4)))
  (func (export "subc_l") (param f64) (result f64)
    (f64.sub (f64.const 1) (local.get 0)))
  (func (export "addc_l") (param f64) (result f64)
    (f64.add (f64.const 0.5) (local.get 0)))
  (func (export "divc_l") (param f64) (result f64)
    (f64.div (f64.const 3) (local.get 0)))
  (func (export "mul_negzero") (param f64) (result f64)
    (f64.mul (local.get 0) (f64.const -0.0)))
  (func (export "add_nan") (param f64) (result f64)
    (f64.add (local.get 0) (f64.const nan)))
  (func (export "mulc32") (param f32) (result f32)
    (f32.mul (local.get 0) (f32.const 2.5)))
  (func (export "subc32") (param f32) (result f32)
    (f32.sub (local.get 0) (f32.const 0.5)))
  (func (export "divc32_l") (param f32) (result f32)
    (f32.div (f32.const 3) (local.get 0)))
)

(invoke "setup")
(assert_return (invoke "add_r" (f64.const 1) (i32.const 0)) (f64.const 2.5))
(assert_return (invoke "sub_r" (f64.const 1) (i32.const 0)) (f64.const -0.5))
(assert_return (invoke "sub_l" (f64.const 1) (i32.const 0)) (f64.const 0.5))
(assert_return (invoke "div_r" (f64.const 3) (i32.const 0)) (f64.const 2))
(assert_return (invoke "div_l" (f64.const 3) (i32.const 0)) (f64.const 0.5))
(assert_return (invoke "mul_l" (f64.const -2) (i32.const 8)) (f64.const 4.5))
(assert_return (invoke "add_r" (f64.const 1) (i32.const 65520)) (f64.const 11))
(assert_trap (invoke "add_r" (f64.const 1) (i32.const 65521)) "out of bounds memory access")
(assert_trap (invoke "add_r" (f64.const 1) (i32.const -8)) "out of bounds memory access")

(assert_return (invoke "add_r32" (f32.const 1) (i32.const 0)) (f32.const 4.5))
(assert_return (invoke "sub_l32" (f32.const 1) (i32.const 0)) (f32.const 2.5))
(assert_return (invoke "div_r32" (f32.const 1.5) (i32.const 4)) (f32.const 2))
(assert_return (invoke "mul_l32" (f32.const 4) (i32.const 4)) (f32.const 3))

(assert_return (invoke "square" (f64.const 3) (i32.const 0)) (f64.const 6))
(assert_return (invoke "sum" (i32.const 2)) (f64.const -0.75))

(assert_return (invoke "mulc" (f64.const 2)) (f64.const 5))
(assert_return (invoke "divc" (f64.const 10)) (f64.const 2.5))
(assert_return (invoke "subc_l" (f64.const 0.25)) (f64.const 0.75))
(assert_return (invoke "addc_l" (f64.const 1)) (f64.const 1.5))
(assert_return (invoke "divc_l" (f64.const 1.5)) (f64.const 2))
(assert_return (invoke "mul_negzero" (f64.const 2)) (f64.const -0.0))
(assert_return (invoke "add_nan" (f64.const 1)) (f64.const nan:arithmetic))
(assert_return (invoke "mulc32" (f32.const 2)) (f32.const 5))
(assert_return (invoke "subc32" (f32.const 2)) (f32.const 1.5))
(assert_return (invoke "divc32_l" (f32.const 1.5)) (f32.const 2))
