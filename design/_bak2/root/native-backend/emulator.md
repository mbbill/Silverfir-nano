- A reference backend interprets MachineIR directly, serving as the
  debugging oracle for everything above the ISA layer.
- It compiles only in debug builds and never participates in production
  execution; compiled native code may never jump into it.
