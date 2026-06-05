Author (2026-06-04, recalling the pre-2024 C interpreter):

Writing it in C was quite painful — a lot of ownership/memory issues could not
be easily identified. A wasm runtime's design tends to have a lot of cross
references; Rust allows a clean data model without worrying about dangling
objects, leaking memory, reentrance, use-after-free, and so on.

(Predates the repository; no commit to pin.)
