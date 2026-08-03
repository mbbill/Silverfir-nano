;; Regression: lowering-scratch borrows must stay satisfiable when linear
;; values and cached cells saturate the allocatable dynamic-register budget.
;;
;; Each block below leaves a ref on the operand stack, so the live window
;; grows by one lane per block with no call boundary to publish it. On a
;; small GP bank (x86_64: 7 allocatable + 1 reserved scratch lane) the
;; window reaches the full allocatable budget and the next table.get's
;; borrowed scratch register must come from the reserved tail. The borrow
;; scanner once capped its scan at the allocatable prefix, making the
;; reserve unreachable exactly when it was needed, and compilation failed
;; with "native lowering requires free GP dynamic registers".
(module
  (type $t0 (func))
  (func $f0)
  (table funcref (elem $f0 $f0 $f0))

  (func (export "run")
    (block (result (ref null func)) (table.get (i32.const 0)))
    (block (result (ref null func)) (table.get (i32.const 1)))
    (block (result (ref null func)) (table.get (i32.const 2)))
    (block (result (ref null func)) (table.get (i32.const 0)))
    (block (result (ref null func)) (table.get (i32.const 1)))
    (block (result (ref null func)) (table.get (i32.const 2)))
    (block (result (ref null func)) (table.get (i32.const 0)))
    (block (result (ref null func)) (table.get (i32.const 1)))
    (block (result (ref null func)) (table.get (i32.const 2)))
    (br 0)
  )
)
(assert_return (invoke "run"))
