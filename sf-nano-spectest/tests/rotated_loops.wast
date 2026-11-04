;; Loops whose header holds nothing but its exit test. The latch binds the
;; header's parameters itself and repeats the test, so the loop closes on
;; one conditional branch. The binding is a parallel copy: a parameter that
;; another one still reads must move last. FP parameters keep the jump.

(module
  ;; Self tail call turned loop: n, a, b <- n - 1, b, a + b.
  (func $fib (param $n i64) (param $a i64) (param $b i64) (result i64)
    (if (i64.eqz (local.get $n)) (then (return (local.get $a))))
    (return_call $fib
      (i64.sub (local.get $n) (i64.const 1))
      (local.get $b)
      (i64.add (local.get $a) (local.get $b))))
  (func (export "fib") (param $n i64) (result i64)
    (return_call $fib (local.get $n) (i64.const 0) (i64.const 1)))

  ;; b takes the old a, so it must be bound before a is rebound.
  (func $order (param $n i64) (param $a i64) (param $b i64) (result i64)
    (if (i64.eqz (local.get $n))
      (then
        (return
          (i64.add
            (i64.mul (local.get $a) (i64.const 1000))
            (local.get $b)))))
    (return_call $order
      (i64.sub (local.get $n) (i64.const 1))
      (i64.add (local.get $b) (i64.const 1))
      (local.get $a)))
  (func (export "order") (param $n i64) (result i64)
    (return_call $order (local.get $n) (i64.const 1) (i64.const 2)))

  ;; x and y swap every iteration through a temporary inside the latch,
  ;; which leaves the latch's jump nothing to bind.
  (func (export "swap") (param $n i32) (result i32)
    (local $x i32) (local $y i32) (local $t i32)
    (local.set $x (i32.const 1))
    (local.set $y (i32.const 2))
    (block $done
      (loop $top
        (br_if $done (i32.eqz (local.get $n)))
        (local.set $t (local.get $x))
        (local.set $x (local.get $y))
        (local.set $y (local.get $t))
        (local.set $n (i32.sub (local.get $n) (i32.const 1)))
        (br $top)))
    (local.get $x))

  ;; The latch binds a constant.
  (func (export "imm") (param $n i32) (param $k i32) (result i32)
    (block $done
      (loop $top
        (br_if $done (i32.eqz (local.get $n)))
        (local.set $k (i32.const 7))
        (local.set $n (i32.sub (local.get $n) (i32.const 1)))
        (br $top)))
    (local.get $k))

  ;; A `while` loop written with block, loop and br_if.
  (func (export "sum") (param $n i32) (result i32) (local $acc i32)
    (block $done
      (loop $top
        (br_if $done (i32.eqz (local.get $n)))
        (local.set $acc (i32.add (local.get $acc) (local.get $n)))
        (local.set $n (i32.sub (local.get $n) (i32.const 1)))
        (br $top)))
    (local.get $acc))

  ;; An FP parameter keeps the jump.
  (func $halve (param $n i64) (param $x f64) (result f64)
    (if (i64.eqz (local.get $n)) (then (return (local.get $x))))
    (return_call $halve
      (i64.sub (local.get $n) (i64.const 1))
      (f64.mul (local.get $x) (f64.const 0.5))))
  (func (export "halve") (param $n i64) (param $x f64) (result f64)
    (return_call $halve (local.get $n) (local.get $x)))
)

(assert_return (invoke "fib" (i64.const 0)) (i64.const 0))
(assert_return (invoke "fib" (i64.const 1)) (i64.const 1))
(assert_return (invoke "fib" (i64.const 10)) (i64.const 55))
(assert_return (invoke "fib" (i64.const 90)) (i64.const 2880067194370816120))
(assert_return (invoke "fib" (i64.const 100)) (i64.const 3736710778780434371))

(assert_return (invoke "order" (i64.const 0)) (i64.const 1002))
(assert_return (invoke "order" (i64.const 1)) (i64.const 3001))
(assert_return (invoke "order" (i64.const 2)) (i64.const 2003))
(assert_return (invoke "order" (i64.const 5)) (i64.const 5003))

(assert_return (invoke "swap" (i32.const 5)) (i32.const 2))
(assert_return (invoke "swap" (i32.const 6)) (i32.const 1))

(assert_return (invoke "imm" (i32.const 0) (i32.const 3)) (i32.const 3))
(assert_return (invoke "imm" (i32.const 5) (i32.const 3)) (i32.const 7))

(assert_return (invoke "sum" (i32.const 0)) (i32.const 0))
(assert_return (invoke "sum" (i32.const 100)) (i32.const 5050))

(assert_return (invoke "halve" (i64.const 0) (f64.const 8)) (f64.const 8))
(assert_return (invoke "halve" (i64.const 3) (f64.const 8)) (f64.const 1))
