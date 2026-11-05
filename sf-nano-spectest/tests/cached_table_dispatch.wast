;; `call_indirect` through a table its module never mutates reads a cached
;; per-instance entry array. Every other writer must invalidate it: an
;; importing instance's `table.set` (between invocations, and in the
;; middle of one through a cross-instance call made via the cached table
;; itself), and `table.grow` past the cached length. Elements may name
;; another instance's functions. Misses must still trap exactly like the
;; generic path: null element, type mismatch, out of bounds.

(module $owner
  (type $ret (func (result i32)))
  (type $other (func (param i32) (result i32)))
  (type $poke (func))
  (func $one (type $ret) (i32.const 1))
  (func $two (type $ret) (i32.const 2))
  (func $wide (type $other) (local.get 0))
  (table $t (export "table") 8 16 funcref)
  (elem (table $t) (i32.const 0) func $one $two $wide)
  (func (export "call") (param i32) (result i32)
    (call_indirect $t (type $ret) (local.get 0)))
  ;; Calls slot $i, then calls slot $k (another instance's function, which
  ;; rewrites the table), then calls slot $i again in the same invocation.
  (func (export "call_poke_call") (param $i i32) (param $k i32) (result i32)
    (i32.add
      (i32.mul (call_indirect $t (type $ret) (local.get $i)) (i32.const 10))
      (block (result i32)
        (call_indirect $t (type $poke) (local.get $k))
        (call_indirect $t (type $ret) (local.get $i)))))
  (func (export "one") (type $ret) (i32.const 1))
)
(register "owner" $owner)

(module $mutator
  (type $ret (func (result i32)))
  (type $poke (func))
  (import "owner" "table" (table $t 8 funcref))
  (import "owner" "one" (func $owner_one (type $ret)))
  (func $seven (type $ret) (i32.const 7))
  (func $eight (type $ret) (i32.const 8))
  (func $poke_eight_into_1 (type $poke)
    (table.set $t (i32.const 1) (ref.func $eight)))
  (func $poke_null_into_1 (type $poke)
    (table.set $t (i32.const 1) (ref.null func)))
  (func (export "mutate") (param $i i32) (param $which i32)
    (table.set $t (local.get $i)
      (if (result funcref) (i32.eqz (local.get $which))
        (then (ref.func $seven))
        (else (if (result funcref) (i32.eq (local.get $which) (i32.const 1))
          (then (ref.func $eight))
          (else (if (result funcref) (i32.eq (local.get $which) (i32.const 2))
            (then (ref.func $owner_one))
            (else (ref.null func)))))))))
  (func (export "install_pokers")
    (table.set $t (i32.const 6) (ref.func $poke_eight_into_1))
    (table.set $t (i32.const 7) (ref.func $poke_null_into_1)))
  (func (export "grow_with_seven") (result i32)
    (table.grow $t (ref.func $seven) (i32.const 2)))
  (elem declare func $owner_one $seven $eight $poke_eight_into_1 $poke_null_into_1)
)

;; Initial contents through the cached fast path.
(assert_return (invoke $owner "call" (i32.const 0)) (i32.const 1))
(assert_return (invoke $owner "call" (i32.const 1)) (i32.const 2))
;; Type mismatch, null element and out of bounds through the miss path.
(assert_trap (invoke $owner "call" (i32.const 2)) "indirect call type mismatch")
(assert_trap (invoke $owner "call" (i32.const 3)) "uninitialized element")
(assert_trap (invoke $owner "call" (i32.const 8)) "undefined element")

;; Another instance rewrites a cached slot between invocations: with its
;; own function, with an owner function again, then with null.
(invoke $mutator "mutate" (i32.const 0) (i32.const 0))
(assert_return (invoke $owner "call" (i32.const 0)) (i32.const 7))
(invoke $mutator "mutate" (i32.const 0) (i32.const 2))
(assert_return (invoke $owner "call" (i32.const 0)) (i32.const 1))
(invoke $mutator "mutate" (i32.const 0) (i32.const 3))
(assert_trap (invoke $owner "call" (i32.const 0)) "uninitialized element")

;; The table now holds the mutator's pokers; calling one through the cached
;; table rewrites slot 1 in the middle of the owner's invocation.
(invoke $mutator "install_pokers")
(assert_trap (invoke $owner "call" (i32.const 6)) "indirect call type mismatch")
(assert_return (invoke $owner "call_poke_call" (i32.const 1) (i32.const 6)) (i32.const 28))
(assert_return (invoke $owner "call" (i32.const 1)) (i32.const 8))
(assert_trap (invoke $owner "call_poke_call" (i32.const 1) (i32.const 7)) "uninitialized element")
(assert_trap (invoke $owner "call" (i32.const 1)) "uninitialized element")

;; Growth past the cached length: the new slots dispatch, the old bound no
;; longer traps, the new bound does.
(assert_return (invoke $mutator "grow_with_seven") (i32.const 8))
(assert_return (invoke $owner "call" (i32.const 8)) (i32.const 7))
(assert_return (invoke $owner "call" (i32.const 9)) (i32.const 7))
(assert_trap (invoke $owner "call" (i32.const 10)) "undefined element")

;; A second module imports the same table and never mutates it, so it
;; caches the table too, and sees every rewrite.
(module $reader
  (type $ret (func (result i32)))
  (import "owner" "table" (table $t 8 funcref))
  (func (export "call") (param i32) (result i32)
    (call_indirect $t (type $ret) (local.get 0)))
)
(assert_return (invoke $reader "call" (i32.const 9)) (i32.const 7))
(invoke $mutator "mutate" (i32.const 9) (i32.const 1))
(assert_return (invoke $reader "call" (i32.const 9)) (i32.const 8))
(assert_return (invoke $owner "call" (i32.const 9)) (i32.const 8))
(assert_trap (invoke $reader "call" (i32.const 1)) "uninitialized element")
(assert_trap (invoke $reader "call" (i32.const 2)) "indirect call type mismatch")
(assert_trap (invoke $reader "call" (i32.const 10)) "undefined element")
