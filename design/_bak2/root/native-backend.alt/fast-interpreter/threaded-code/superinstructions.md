- Hot opcode sequences fuse into single superinstruction handlers — 2- to 5-gram patterns (local+load, arithmetic chains, compare+branch,
  shift+add address math), each executing in one dispatch.
- Pattern selection is measurement-driven: a lightweight profiler counts
  executed opcode n-gram sequences and dumps the hottest; top sequences
  become fusion candidates, added one pattern at a time.
- In-repo the fusion pass is named "quickening" — but all rewriting happens
  at IR-build time; nothing self-modifies at runtime.
