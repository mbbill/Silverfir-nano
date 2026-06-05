- The middle layer plans the whole function jointly: cached-local selection
  and transient-register assignment are decided together, not in separate
  passes.
- Planning guarantees the register budget by construction: peak simultaneous
  residency is computed and kept within the target's transient banks, with
  register-pressure reports as the diagnostic.
- Values sink toward their use sites (sink planning) to shorten residency.
- Inlining is conservative by policy: only straight-line leaf functions
  inline.
