- Each module entity (function, table, memory, global) embeds a shared linkage
  struct (`Linkage`) recording its optional import path and optional export
  name; whether it is imported is read from that struct via a shared trait.

- An imported and a local entity are the same Rust type; local-only data
  (function code and locals, global init expr) is held in `Option` fields that
  are `None` for imports and unwrapped with an `expect` at use.

## Moves

- 2024-01-25 (49da4692) replaced [[embedded-linkage-field.alt/per-entity-kind-sum]]:
  the per-entity Kind<LocalType> sum type made import-and-local mutually
  exclusive by construction so an imported entity could not hold local data and
  vice versa; flattening to a shared Kind{import_path,export_path} plus
  per-entity Option locals/code/init_expr makes that illegal state
  representable and trades the type-enforced exclusivity for
  import_path()/export_path()/locals()/code() accessors that panic on misuse
  (diff).

- 2024-02-15 (3906283c) replaced by [[external-linkable-wrapper]]: a single
  embedded linkage struct forced every entity to carry optional local-only
  fields (code, locals, init-expr) that imports never use; splitting import and
  local into a sum type with separate property types lets an imported function
  hold only its type and a local hold non-optional code, removing the
  Option/expect panics (diff).
