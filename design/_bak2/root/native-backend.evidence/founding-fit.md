---
commit: c58d3a9
---
Author (2026-06-04): the reason I started the native backend is that
everything transfers: the rotating TOS transfers directly to linear SSA,
the local cache transfers to plain register usage, and then the
compilation pipeline just emits native code instead of handler pointers.
Everything fits so well.
