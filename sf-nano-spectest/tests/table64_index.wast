;; A 64-bit table's index keeps all 64 bits: an index whose low 32 bits name a
;; valid slot but whose high bits are set is out of bounds and traps, for
;; call_indirect and its tail form and for table.get and table.set, instead
;; of reaching that slot. The first module has no table writes, so the JIT
;; gives its fixed table a fixed-local dispatch and the others a cached one;
;; the second writes its table, which leaves its dispatch generic and its
;; accesses inline.
(module
  (type $unary (func (param i32) (result i32)))
  (table $fixed i64 2 2 funcref)
  (table $shared (export "table") i64 2 funcref)
  (table $growable i64 2 funcref)
  (elem (table $fixed) (i64.const 0) func $double $square)
  (elem (table $shared) (i64.const 0) func $double $square)
  (elem (table $growable) (i64.const 0) func $double $square)
  (func $double (type $unary) local.get 0 i32.const 2 i32.mul)
  (func $square (type $unary) local.get 0 local.get 0 i32.mul)
  (func (export "fixed") (param i64 i32) (result i32)
    local.get 1 local.get 0 call_indirect $fixed (type $unary))
  (func (export "shared") (param i64 i32) (result i32)
    local.get 1 local.get 0 call_indirect $shared (type $unary))
  (func (export "growable") (param i64 i32) (result i32)
    local.get 1 local.get 0 call_indirect $growable (type $unary))
  (func (export "tail") (param i64 i32) (result i32)
    local.get 1 local.get 0 return_call_indirect $fixed (type $unary))
  (func (export "is_null") (param i64) (result i32)
    local.get 0 table.get $growable ref.is_null))

(assert_return (invoke "fixed" (i64.const 0) (i32.const 21)) (i32.const 42))
(assert_return (invoke "fixed" (i64.const 1) (i32.const 7)) (i32.const 49))
(assert_trap (invoke "fixed" (i64.const 2) (i32.const 1)) "undefined element")
(assert_trap (invoke "fixed" (i64.const 0x1_0000_0000) (i32.const 1)) "undefined element")
(assert_trap (invoke "fixed" (i64.const 0x1_0000_0001) (i32.const 1)) "undefined element")
(assert_trap (invoke "fixed" (i64.const -1) (i32.const 1)) "undefined element")

(assert_return (invoke "shared" (i64.const 1) (i32.const 7)) (i32.const 49))
(assert_trap (invoke "shared" (i64.const 0x1_0000_0000) (i32.const 1)) "undefined element")
(assert_trap (invoke "shared" (i64.const 0x1_0000_0001) (i32.const 1)) "undefined element")
(assert_trap (invoke "shared" (i64.const -1) (i32.const 1)) "undefined element")

(assert_return (invoke "growable" (i64.const 1) (i32.const 7)) (i32.const 49))
(assert_trap (invoke "growable" (i64.const 0x1_0000_0000) (i32.const 1)) "undefined element")
(assert_trap (invoke "growable" (i64.const 0x1_0000_0001) (i32.const 1)) "undefined element")

(assert_return (invoke "tail" (i64.const 0) (i32.const 21)) (i32.const 42))
(assert_trap (invoke "tail" (i64.const 0x1_0000_0000) (i32.const 1)) "undefined element")
(assert_trap (invoke "tail" (i64.const 0x1_0000_0001) (i32.const 1)) "undefined element")

(assert_return (invoke "is_null" (i64.const 0)) (i32.const 0))
(assert_trap (invoke "is_null" (i64.const 0x1_0000_0000)) "out of bounds table access")
(assert_trap (invoke "is_null" (i64.const -1)) "out of bounds table access")

(module
  (type $unary (func (param i32) (result i32)))
  (table $t i64 2 funcref)
  (elem (table $t) (i64.const 0) func $double $square)
  (elem declare func $double)
  (func $double (type $unary) local.get 0 i32.const 2 i32.mul)
  (func $square (type $unary) local.get 0 local.get 0 i32.mul)
  (func (export "set") (param i64)
    local.get 0 ref.func $double table.set $t)
  (func (export "is_null") (param i64) (result i32)
    local.get 0 table.get $t ref.is_null)
  (func (export "call") (param i64 i32) (result i32)
    local.get 1 local.get 0 call_indirect $t (type $unary))
  (func (export "tail") (param i64 i32) (result i32)
    local.get 1 local.get 0 return_call_indirect $t (type $unary)))

(assert_return (invoke "call" (i64.const 1) (i32.const 7)) (i32.const 49))
(assert_trap (invoke "call" (i64.const 0x1_0000_0000) (i32.const 1)) "undefined element")
(assert_trap (invoke "call" (i64.const 0x1_0000_0001) (i32.const 1)) "undefined element")
(assert_trap (invoke "call" (i64.const -1) (i32.const 1)) "undefined element")
(invoke "set" (i64.const 1))
(assert_return (invoke "call" (i64.const 1) (i32.const 7)) (i32.const 14))
(assert_trap (invoke "set" (i64.const 0x1_0000_0001)) "out of bounds table access")
(assert_return (invoke "tail" (i64.const 0) (i32.const 21)) (i32.const 42))
(assert_trap (invoke "tail" (i64.const 0x1_0000_0001) (i32.const 1)) "undefined element")
(assert_return (invoke "is_null" (i64.const 1)) (i32.const 0))
(assert_trap (invoke "is_null" (i64.const 0x1_0000_0001)) "out of bounds table access")
(assert_return (invoke "call" (i64.const 1) (i32.const 7)) (i32.const 14))
