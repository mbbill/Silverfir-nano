- GC objects (structs and arrays) live in a simple arena: a growable vector
  of boxed objects, with `GcRef` as a plain index into it.
- Objects carry their type-table index plus their field/element values.
- The heap is allocate-only: no collector runs; actual collection
  (refcounting or mark-sweep) is an acknowledged future step.
