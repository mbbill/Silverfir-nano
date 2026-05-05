// Mirror of the slice of `tools/sf-lens/src/model.rs` consumed by the viewer.
// Only the fields we actually read are declared here. The Rust extractor emits
// a superset; TypeScript ignores the rest.
//
// When the viewer starts consuming new fields, add them here first and keep
// this file the single source of truth on the JS side.

export type TypeKind = 'struct' | 'enum' | 'union' | 'trait' | 'type_alias';

export type Ownership =
  | 'owned'
  | 'borrow_immut'
  | 'borrow_mut'
  | 'indirection'
  | 'primitive'
  | 'other';

export interface FieldFacts {
  readonly name: string;
  readonly ty_text: string;
  readonly ownership: Ownership;
}

export interface TypeFacts {
  readonly name: string;
  readonly full_path: string;
  readonly kind: TypeKind;
  readonly fields: readonly FieldFacts[];
}

export interface ModuleFacts {
  readonly path: string;
  readonly file: string;
  readonly types: readonly TypeFacts[];
}

export interface CrateFacts {
  readonly name: string;
  readonly modules: Readonly<Record<string, ModuleFacts>>;
}

export type EdgeKind =
  | 'owns'
  | 'borrows_immut'
  | 'borrows_mut'
  | 'indirection'
  | 'trait_impl';

export type ViaKind =
  | 'struct_field'
  | 'union_field'
  | 'enum_variant_payload'
  | 'fn_param'
  | 'fn_return'
  | 'trait_impl_block';

export interface Edge {
  readonly from: string;
  readonly to: string;
  readonly kind: EdgeKind;
  readonly via: ViaKind;
  /**
   * Free-text descriptor of where the edge was declared.
   * For `struct_field`/`union_field` vias: `field {fieldName}`.
   * For `enum_variant_payload`: `field {Variant}::{payloadName}` (or just `field {Variant}` for unit-like).
   * For `fn_param`/`fn_return`: `fn {fnName} param {paramName}` / `fn {fnName} return`.
   */
  readonly origin: string;
}

export interface Facts {
  readonly crates: Readonly<Record<string, CrateFacts>>;
  readonly edges: readonly Edge[];
}
