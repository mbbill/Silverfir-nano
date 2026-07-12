- Building the fast IR leaks the instruction array, blob, and entry pointer
  (Box::into_raw) and stores them as three raw-pointer cells on the function spec,
  owned for the process lifetime with no reclamation.

- The fast entry pointer is cached as a separate raw-pointer cell distinct from
  the code and blob pointers.

## Moves

- 2025-08-14 (fec5a3aa) replaced by [[code-storage]]: the raw-pointer triple
  leaked the IR boxes to obtain a process-lifetime pointer with no ownership; an
  owned FastCode ties the compiled code and blob to the function spec's lifetime
  instead (code).
