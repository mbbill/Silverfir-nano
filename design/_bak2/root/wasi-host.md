- WASI lives outside the core: a separate crate implements it against the
  core's external-function hooks; the core contains no WASI code.
- Guest filesystem access is capability-style: directories are preopened
  explicitly; nothing else is reachable.
- WASI behavior is validated against the official wasi-testsuite, fetched
  and run by a dedicated test crate — the same gate discipline as spectest.
