---
commit: 8136fd44
---
Author (2026-06-04): profiling showed most functions don't need 8 TOS
registers. Because the TOS cache is refreshed at block boundaries, the
effective register-resident depth between boundaries stays small — 8 is
simply too many; 4 suffices. (The block-boundary refresh itself was noted
as improvable.) A 5-register
experiment late in -rs (branches nh / rebuild-from-4tos, "5 tos, score
4343") did not stick: nano's tos_config fixes TOS_REGISTER_COUNT at 4 as
the single source of truth.
