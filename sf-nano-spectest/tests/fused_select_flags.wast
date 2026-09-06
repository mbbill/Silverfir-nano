;; Regression: a fused compare+select must not clobber EFLAGS between the
;; compare and the CMOV. Materializing the select's zero immediate with the
;; xor zero idiom did exactly that (ZF forced to 1), so the CMOV always took
;; the true value regardless of the compare.
(module
  (func (export "sel") (param i32) (result i32)
    (select (i32.const 5) (i32.const 0) (i32.eq (local.get 0) (i32.const 5))))
  (func (export "sel_rev") (param i32) (result i32)
    (select (i32.const 0) (i32.const 7) (i32.lt_u (local.get 0) (i32.const 4))))
)
(assert_return (invoke "sel" (i32.const 5)) (i32.const 5))
(assert_return (invoke "sel" (i32.const 3)) (i32.const 0))
(assert_return (invoke "sel_rev" (i32.const 2)) (i32.const 0))
(assert_return (invoke "sel_rev" (i32.const 9)) (i32.const 7))

;; A select may reuse the ZF from its i32 producer. Operand materialization
;; (especially a zero literal) or a new basic block invalidates that proof.
(module
  (func (export "and_regs") (param i32 i32 i32) (result i32)
    (select (local.get 1) (local.get 2) (i32.and (local.get 0) (i32.const 1))))
  (func (export "and_zero") (param i32) (result i32)
    (select (i32.const 5) (i32.const 0) (i32.and (local.get 0) (i32.const 1))))
  (func (export "and_zero_rev") (param i32) (result i32)
    (select (i32.const 0) (i32.const 7) (i32.and (local.get 0) (i32.const 1))))
  (func (export "and_fp") (param i32 f64 f64) (result f64)
    (select (local.get 1) (local.get 2) (i32.and (local.get 0) (i32.const 1))))
  (func (export "join") (param i32 i32 i32 i32) (result i32) (local $cond i32)
    (local.set $cond
      (if (result i32) (local.get 0)
        (then (i32.and (local.get 1) (i32.const 1)))
        (else (local.get 1))))
    (select (local.get 2) (local.get 3) (local.get $cond)))
)
(assert_return (invoke "and_regs" (i32.const 3) (i32.const 41) (i32.const 73)) (i32.const 41))
(assert_return (invoke "and_regs" (i32.const 2) (i32.const 41) (i32.const 73)) (i32.const 73))
(assert_return (invoke "and_zero" (i32.const 3)) (i32.const 5))
(assert_return (invoke "and_zero" (i32.const 2)) (i32.const 0))
(assert_return (invoke "and_zero_rev" (i32.const 3)) (i32.const 0))
(assert_return (invoke "and_zero_rev" (i32.const 2)) (i32.const 7))
(assert_return (invoke "and_fp" (i32.const 3) (f64.const 1.25) (f64.const 2.5)) (f64.const 1.25))
(assert_return (invoke "and_fp" (i32.const 2) (f64.const 1.25) (f64.const 2.5)) (f64.const 2.5))
(assert_return (invoke "join" (i32.const 1) (i32.const 2) (i32.const 41) (i32.const 73)) (i32.const 73))
(assert_return (invoke "join" (i32.const 0) (i32.const 2) (i32.const 41) (i32.const 73)) (i32.const 41))
(assert_return (invoke "join" (i32.const 1) (i32.const 3) (i32.const 41) (i32.const 73)) (i32.const 41))
(assert_return (invoke "join" (i32.const 0) (i32.const 0) (i32.const 41) (i32.const 73)) (i32.const 73))
