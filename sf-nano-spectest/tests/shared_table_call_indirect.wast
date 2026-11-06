;; call_indirect through a table another instance can reach. Such a table
;; holds world identities rather than local indices: its owner's own
;; functions must still resolve, a guard failure must still trap, and a
;; peer's writes, including a grow that moves the elements, must be visible
;; to both sides.
(module $owner
  (type $unary (func (param i32) (result i32)))
  (type $unit (func))
  (table (export "table") 4 8 funcref)
  (elem (i32.const 0) $double $square $nop)
  (func $double (type $unary) local.get 0 i32.const 2 i32.mul)
  (func $square (type $unary) local.get 0 local.get 0 i32.mul)
  (func $nop (type $unit))
  (func (export "call") (param i32 i32) (result i32)
    local.get 1 local.get 0 call_indirect (type $unary))
  (func (export "size") (result i32) table.size))
(register "owner" $owner)

(assert_return (invoke $owner "call" (i32.const 0) (i32.const 21)) (i32.const 42))
(assert_return (invoke $owner "call" (i32.const 1) (i32.const 7)) (i32.const 49))
(assert_trap (invoke $owner "call" (i32.const 2) (i32.const 1)) "indirect call type mismatch")
(assert_trap (invoke $owner "call" (i32.const 3) (i32.const 1)) "uninitialized element")
(assert_trap (invoke $owner "call" (i32.const 4) (i32.const 1)) "undefined element")

(module $peer
  (type $unary (func (param i32) (result i32)))
  (import "owner" "table" (table 4 8 funcref))
  (func $triple (type $unary) local.get 0 i32.const 3 i32.mul)
  (elem declare func $triple)
  (func (export "install") (param i32)
    local.get 0 ref.func $triple table.set)
  (func (export "grow") (result i32)
    ref.func $triple i32.const 4 table.grow)
  (func (export "call") (param i32 i32) (result i32)
    local.get 1 local.get 0 call_indirect (type $unary)))

(assert_return (invoke $peer "call" (i32.const 0) (i32.const 21)) (i32.const 42))
(invoke $peer "install" (i32.const 1))
(assert_return (invoke $owner "call" (i32.const 1) (i32.const 7)) (i32.const 21))
(assert_return (invoke $peer "call" (i32.const 1) (i32.const 7)) (i32.const 21))
(assert_return (invoke $peer "grow") (i32.const 4))
(assert_return (invoke $owner "size") (i32.const 8))
(assert_return (invoke $owner "call" (i32.const 7) (i32.const 5)) (i32.const 15))
(assert_return (invoke $owner "call" (i32.const 0) (i32.const 21)) (i32.const 42))
(assert_trap (invoke $owner "call" (i32.const 8) (i32.const 1)) "undefined element")
