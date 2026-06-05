- SSA lowers directly to the executable instruction stream in a single pass;
  storage homes in the per-frame register file are assigned during SSA
  construction by linear scan, with no intermediate representation between
  SSA and the emitted instructions.
- Expression-tree evaluation is ordered Sethi–Ullman style to minimize
  simultaneously-live values within the register window.
