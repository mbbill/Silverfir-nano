---
commit: ba01d633
---
Data-ownership went through three generations. Day 1 the module borrowed
the binary (Cow<'a, [u8]>, zero-copy, lifetime-threaded). Two weeks in,
lifetimes lost: "function no longer need to hold to a cow reference.
Instead it holds a Rc, therefore the lifetime of the function becomes much
easier to handle" (ba01d633) — Module dropped its lifetime parameter and
became Clone. Two years later nano's single-module model removed the Rc
too: owned structures, plain borrows.
