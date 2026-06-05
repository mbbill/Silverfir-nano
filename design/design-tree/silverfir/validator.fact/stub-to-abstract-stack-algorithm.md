commit: 80306a9a

The validator was introduced (4336bd63) as a stub `OpcodeHandler` that did no
semantic work — it only maintained an indent counter so a disassembly printer
could nest blocks. No node was created for it then; the design tree deferred it
until it embodied a real validation decision. That happens here: 80306a9a
replaces the stub body with the WebAssembly spec's abstract type-checking
algorithm (value-type stack, control-frame stack, push/pop against expected
types, stack-polymorphism via an unreachable flag). The two later commits in the
same batch (9b917622 "Implemented validator", and the predicate refactor in
f41bfdb3) fill out the per-opcode rules and replace exact-type pops with
type-predicate pops; the structural shape — validate by abstract-stack
simulation riding the decode walk — is set here. This is the point the deferral
from the stub resolves into the `validator` node.
