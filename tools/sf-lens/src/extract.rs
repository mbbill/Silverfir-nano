//! Workspace traversal and per-file parsing with `syn`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use syn::visit::Visit;
use syn::{
    File, ImplItem, Item, ItemEnum, ItemImpl, ItemMod, ItemStruct, ItemTrait, ItemType, ItemUnion,
    ReturnType,
};
use walkdir::WalkDir;

use crate::model::{
    CrateFacts, Edge, EdgeKind, EdgeProfile, FieldFacts, FnFacts, ModuleFacts, Ownership,
    ParamFacts, SelfKind, TypeFacts, TypeKind, ViaKind, WorkspaceFacts,
};
use crate::resolve::{classify, type_text};

pub fn extract_workspace(root: &Path) -> Result<WorkspaceFacts> {
    let crates = discover_crates(root)?;
    let mut workspace = WorkspaceFacts {
        crates: BTreeMap::new(),
        edges: Vec::new(),
        edge_profiles: BTreeMap::new(),
    };

    for (name, crate_root) in crates {
        let cf = extract_crate(&name, &crate_root)?;
        workspace.crates.insert(name, cf);
    }

    // Build the global type registry: short-name -> set of canonical paths.
    let registry = build_type_registry(&workspace);

    // Build edges by re-scanning every type, function, and impl across all
    // crates. We carry the source full-path so edges have a `from` anchor.
    let mut edges = Vec::new();
    for cf in workspace.crates.values() {
        for module in cf.modules.values() {
            for ty in &module.types {
                emit_edges_from_type(ty, &registry, &mut edges);
            }
            for f in &module.functions {
                let from = format!("{}::{}::{}", cf.name, module.path, f.name);
                let from = from.replace("::::", "::");
                emit_edges_from_fn(&from, f, &registry, &mut edges);
            }
        }
    }

    workspace.edge_profiles = build_profiles(&edges);
    workspace.edges = edges;

    Ok(workspace)
}

/// Walk the workspace tree and collect (crate-name, src_root) pairs by
/// looking at every Cargo.toml.
fn discover_crates(root: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut out = Vec::new();
    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !matches!(name.as_ref(), "target" | "tmp" | ".git" | "node_modules")
        })
        .filter_map(|e| e.ok())
    {
        if entry.file_name() == "Cargo.toml" {
            let path = entry.path();
            // Skip the workspace-only root Cargo.toml.
            let txt = std::fs::read_to_string(path).unwrap_or_default();
            if !txt.contains("[package]") {
                continue;
            }
            let crate_name = parse_crate_name(&txt).unwrap_or_else(|| {
                path.parent()
                    .and_then(|p| p.file_name())
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default()
            });
            let src = path.parent().unwrap().join("src");
            if src.is_dir() {
                out.push((crate_name, src));
            }
        }
    }
    out.sort();
    Ok(out)
}

fn parse_crate_name(toml: &str) -> Option<String> {
    let mut in_pkg = false;
    for line in toml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_pkg = trimmed == "[package]";
            continue;
        }
        if in_pkg {
            if let Some(rest) = trimmed.strip_prefix("name") {
                let rest = rest.trim_start_matches(|c: char| c == ' ' || c == '=' || c == '\t');
                let rest = rest.trim();
                let rest = rest.trim_matches('"');
                if !rest.is_empty() {
                    return Some(rest.to_string());
                }
            }
        }
    }
    None
}

fn extract_crate(name: &str, src_root: &Path) -> Result<CrateFacts> {
    let mut crate_facts = CrateFacts {
        name: name.to_string(),
        root: src_root.display().to_string(),
        modules: BTreeMap::new(),
    };
    for entry in WalkDir::new(src_root).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file()
            && entry.path().extension().map(|e| e == "rs").unwrap_or(false)
        {
            let module_path = derive_module_path(src_root, entry.path());
            extract_file(name, src_root, entry.path(), &module_path, &mut crate_facts)
                .with_context(|| format!("parsing {}", entry.path().display()))?;
        }
    }
    Ok(crate_facts)
}

/// Translate a file path under src/ to a "::"-delimited module path.
/// `src/lib.rs` and `src/main.rs` -> "" (crate root).
fn derive_module_path(src_root: &Path, file: &Path) -> String {
    let rel = file.strip_prefix(src_root).unwrap_or(file);
    let mut parts: Vec<String> = rel
        .with_extension("")
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();
    if let Some(last) = parts.last() {
        if last == "mod" || last == "lib" || last == "main" {
            parts.pop();
        }
    }
    parts.join("::")
}

fn extract_file(
    crate_name: &str,
    src_root: &Path,
    file: &Path,
    module_path: &str,
    crate_facts: &mut CrateFacts,
) -> Result<()> {
    let _ = src_root;
    let src = std::fs::read_to_string(file)?;
    let ast: File = match syn::parse_file(&src) {
        Ok(f) => f,
        Err(_) => return Ok(()), // Tolerate unparseable files (macros, etc.).
    };

    let mut ctx = Ctx {
        crate_name: crate_name.to_string(),
        file: file.display().to_string(),
        module_stack: vec![module_path.to_string()],
        modules: BTreeMap::new(),
    };
    ctx.ensure_module(module_path);
    for item in &ast.items {
        ctx.visit_item(item);
    }

    for (path, m) in ctx.modules {
        crate_facts
            .modules
            .entry(path.clone())
            .and_modify(|existing| merge_module(existing, &m))
            .or_insert(m);
    }
    Ok(())
}

fn merge_module(into: &mut ModuleFacts, from: &ModuleFacts) {
    into.types.extend(from.types.iter().cloned());
    into.functions.extend(from.functions.iter().cloned());
    into.unsafe_blocks += from.unsafe_blocks;
    if into.file.is_empty() {
        into.file = from.file.clone();
    }
}

struct Ctx {
    crate_name: String,
    file: String,
    module_stack: Vec<String>,
    modules: BTreeMap<String, ModuleFacts>,
}

impl Ctx {
    fn current_module_path(&self) -> String {
        self.module_stack.last().cloned().unwrap_or_default()
    }

    fn ensure_module(&mut self, path: &str) -> &mut ModuleFacts {
        let key = path.to_string();
        let file = self.file.clone();
        self.modules
            .entry(key.clone())
            .or_insert_with(|| ModuleFacts {
                path: path.to_string(),
                file,
                ..Default::default()
            })
    }

    fn full_path(&self, name: &str) -> String {
        let module = self.current_module_path();
        if module.is_empty() {
            format!("{}::{}", self.crate_name, name)
        } else {
            format!("{}::{}::{}", self.crate_name, module, name)
        }
    }

    fn visit_item(&mut self, item: &Item) {
        match item {
            Item::Mod(m) => self.visit_mod(m),
            Item::Struct(s) => self.visit_struct(s),
            Item::Enum(e) => self.visit_enum(e),
            Item::Union(u) => self.visit_union(u),
            Item::Trait(t) => self.visit_trait(t),
            Item::Type(t) => self.visit_type_alias(t),
            Item::Impl(i) => self.visit_impl(i),
            Item::Fn(f) => {
                let name = f.sig.ident.to_string();
                let visibility = vis_text(&f.vis);
                let facts = build_fn_facts(&name, visibility, &f.sig, Some(&f.block), &f.attrs);
                let module_path = self.current_module_path();
                self.ensure_module(&module_path).functions.push(facts);
            }
            _ => {}
        }
    }

    fn visit_mod(&mut self, m: &ItemMod) {
        if let Some((_, items)) = &m.content {
            let parent = self.current_module_path();
            let new_path = if parent.is_empty() {
                m.ident.to_string()
            } else {
                format!("{}::{}", parent, m.ident)
            };
            self.module_stack.push(new_path);
            for item in items {
                self.visit_item(item);
            }
            self.module_stack.pop();
        }
        // For `mod foo;` (file-based), the other file gets walked separately.
    }

    fn visit_struct(&mut self, s: &ItemStruct) {
        let name = s.ident.to_string();
        let full_path = self.full_path(&name);
        let derives = collect_derives(&s.attrs);
        let lifetime_params = lifetime_params_of(&s.generics);
        let type_params = type_params_of(&s.generics);
        let visibility = vis_text(&s.vis);
        let doc = doc_first_line(&s.attrs);
        let fields = fields_from_struct(&s.fields);
        let facts = TypeFacts {
            name,
            full_path,
            kind: TypeKind::Struct,
            visibility,
            lifetime_params,
            type_params,
            derives,
            fields,
            methods: Vec::new(),
            trait_impls: Vec::new(),
            unsafe_blocks: 0,
            doc_first_line: doc,
        };
        let module_path = self.current_module_path();
        self.ensure_module(&module_path).types.push(facts);
    }

    fn visit_enum(&mut self, e: &ItemEnum) {
        let name = e.ident.to_string();
        let full_path = self.full_path(&name);
        let derives = collect_derives(&e.attrs);
        let lifetime_params = lifetime_params_of(&e.generics);
        let type_params = type_params_of(&e.generics);
        let visibility = vis_text(&e.vis);
        let doc = doc_first_line(&e.attrs);

        let mut fields = Vec::new();
        for variant in &e.variants {
            let var_name = variant.ident.to_string();
            let inner = fields_from_struct(&variant.fields);
            if inner.is_empty() {
                fields.push(FieldFacts {
                    name: var_name,
                    ty_text: "()".into(),
                    ownership: Ownership::Primitive,
                    referenced: vec![],
                    cardinality: vec![],
                    lifetimes: vec![],
                });
            } else {
                for f in inner {
                    fields.push(FieldFacts {
                        name: format!("{}::{}", var_name, f.name),
                        ..f
                    });
                }
            }
        }

        let facts = TypeFacts {
            name,
            full_path,
            kind: TypeKind::Enum,
            visibility,
            lifetime_params,
            type_params,
            derives,
            fields,
            methods: Vec::new(),
            trait_impls: Vec::new(),
            unsafe_blocks: 0,
            doc_first_line: doc,
        };
        let module_path = self.current_module_path();
        self.ensure_module(&module_path).types.push(facts);
    }

    fn visit_union(&mut self, u: &ItemUnion) {
        let name = u.ident.to_string();
        let full_path = self.full_path(&name);
        let derives = collect_derives(&u.attrs);
        let lifetime_params = lifetime_params_of(&u.generics);
        let type_params = type_params_of(&u.generics);
        let visibility = vis_text(&u.vis);
        let doc = doc_first_line(&u.attrs);
        let fields: Vec<FieldFacts> = u.fields.named.iter().map(|f| field_from_named(f)).collect();
        let facts = TypeFacts {
            name,
            full_path,
            kind: TypeKind::Union,
            visibility,
            lifetime_params,
            type_params,
            derives,
            fields,
            methods: Vec::new(),
            trait_impls: Vec::new(),
            unsafe_blocks: 0,
            doc_first_line: doc,
        };
        let module_path = self.current_module_path();
        self.ensure_module(&module_path).types.push(facts);
    }

    fn visit_trait(&mut self, t: &ItemTrait) {
        let name = t.ident.to_string();
        let full_path = self.full_path(&name);
        let lifetime_params = lifetime_params_of(&t.generics);
        let type_params = type_params_of(&t.generics);
        let visibility = vis_text(&t.vis);
        let doc = doc_first_line(&t.attrs);

        // Trait method signatures count as method facts on the trait itself.
        let mut methods = Vec::new();
        for item in &t.items {
            if let syn::TraitItem::Fn(f) = item {
                let name = f.sig.ident.to_string();
                let visibility = "pub".to_string();
                let block_ref = f.default.as_ref();
                let facts = build_fn_facts(&name, visibility, &f.sig, block_ref, &f.attrs);
                methods.push(facts);
            }
        }
        let facts = TypeFacts {
            name,
            full_path,
            kind: TypeKind::Trait,
            visibility,
            lifetime_params,
            type_params,
            derives: vec![],
            fields: Vec::new(),
            methods,
            trait_impls: Vec::new(),
            unsafe_blocks: 0,
            doc_first_line: doc,
        };
        let module_path = self.current_module_path();
        self.ensure_module(&module_path).types.push(facts);
    }

    fn visit_type_alias(&mut self, t: &ItemType) {
        let name = t.ident.to_string();
        let full_path = self.full_path(&name);
        let lifetime_params = lifetime_params_of(&t.generics);
        let type_params = type_params_of(&t.generics);
        let visibility = vis_text(&t.vis);
        let doc = doc_first_line(&t.attrs);
        let (ownership, refs, lifetimes) = classify(&t.ty);
        let (referenced, cardinality) = split_refs(refs);
        let fields = vec![FieldFacts {
            name: "<alias>".to_string(),
            ty_text: type_text(&t.ty),
            ownership,
            referenced,
            cardinality,
            lifetimes,
        }];
        let facts = TypeFacts {
            name,
            full_path,
            kind: TypeKind::TypeAlias,
            visibility,
            lifetime_params,
            type_params,
            derives: vec![],
            fields,
            methods: Vec::new(),
            trait_impls: Vec::new(),
            unsafe_blocks: 0,
            doc_first_line: doc,
        };
        let module_path = self.current_module_path();
        self.ensure_module(&module_path).types.push(facts);
    }

    fn visit_impl(&mut self, i: &ItemImpl) {
        // Identify the Self type by name (last path segment) — heuristic.
        let self_name = match &*i.self_ty {
            syn::Type::Path(tp) => tp.path.segments.last().map(|s| s.ident.to_string()),
            _ => None,
        };
        let Some(self_name) = self_name else {
            return;
        };

        let trait_name = i
            .trait_
            .as_ref()
            .and_then(|(_, p, _)| p.segments.last())
            .map(|s| s.ident.to_string());

        let mut unsafe_blocks: u32 = 0;
        let mut methods = Vec::new();
        for item in &i.items {
            if let ImplItem::Fn(f) = item {
                let name = f.sig.ident.to_string();
                let vis = vis_text(&f.vis);
                let facts = build_fn_facts(&name, vis, &f.sig, Some(&f.block), &f.attrs);
                unsafe_blocks += facts.unsafe_blocks;
                methods.push(facts);
            }
        }

        // Attach to the matching type in the current module by name. If not
        // found in current module, scan the whole crate's collected modules.
        let module_path = self.current_module_path();
        let trait_clone = trait_name.clone();
        if let Some(m) = self.modules.get_mut(&module_path) {
            if let Some(ty) = m.types.iter_mut().find(|t| t.name == self_name) {
                ty.methods.extend(methods.clone());
                ty.unsafe_blocks += unsafe_blocks;
                if let Some(t) = trait_clone.clone() {
                    ty.trait_impls.push(t);
                }
                return;
            }
        }
        // Search anywhere in the crate's already-built modules.
        for m in self.modules.values_mut() {
            if let Some(ty) = m.types.iter_mut().find(|t| t.name == self_name) {
                ty.methods.extend(methods.clone());
                ty.unsafe_blocks += unsafe_blocks;
                if let Some(t) = trait_name.clone() {
                    ty.trait_impls.push(t);
                }
                return;
            }
        }
        // Otherwise: orphan impl block (impl for a type defined in another
        // file). Stash a stub-type entry so the methods are not lost.
        let stub = TypeFacts {
            name: self_name.clone(),
            full_path: self.full_path(&self_name),
            kind: TypeKind::Struct,
            visibility: "<orphan-impl>".into(),
            lifetime_params: vec![],
            type_params: vec![],
            derives: vec![],
            fields: vec![],
            methods,
            trait_impls: trait_name.clone().into_iter().collect(),
            unsafe_blocks,
            doc_first_line: None,
        };
        let module_path_owned = module_path;
        self.ensure_module(&module_path_owned).types.push(stub);
    }
}

fn collect_derives(attrs: &[syn::Attribute]) -> Vec<String> {
    let mut out = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("derive") {
            continue;
        }
        let _ = attr.parse_nested_meta(|meta| {
            if let Some(seg) = meta.path.segments.last() {
                out.push(seg.ident.to_string());
            }
            Ok(())
        });
    }
    out
}

fn lifetime_params_of(g: &syn::Generics) -> Vec<String> {
    g.lifetimes()
        .map(|lp| lp.lifetime.ident.to_string())
        .collect()
}

fn type_params_of(g: &syn::Generics) -> Vec<String> {
    g.type_params().map(|tp| tp.ident.to_string()).collect()
}

fn vis_text(v: &syn::Visibility) -> String {
    match v {
        syn::Visibility::Public(_) => "pub".to_string(),
        syn::Visibility::Restricted(r) => {
            let path: String = r
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            format!("pub({path})")
        }
        syn::Visibility::Inherited => "priv".to_string(),
    }
}

fn doc_first_line(attrs: &[syn::Attribute]) -> Option<String> {
    for a in attrs {
        if a.path().is_ident("doc") {
            if let syn::Meta::NameValue(nv) = &a.meta {
                if let syn::Expr::Lit(lit) = &nv.value {
                    if let syn::Lit::Str(s) = &lit.lit {
                        let v = s.value();
                        let line = v.trim().lines().next().unwrap_or("").trim().to_string();
                        if !line.is_empty() {
                            return Some(line);
                        }
                    }
                }
            }
        }
    }
    None
}

fn fields_from_struct(fields: &syn::Fields) -> Vec<FieldFacts> {
    match fields {
        syn::Fields::Named(named) => named.named.iter().map(field_from_named).collect(),
        syn::Fields::Unnamed(unn) => unn
            .unnamed
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let (ownership, refs, lifetimes) = classify(&f.ty);
                let (referenced, cardinality) = split_refs(refs);
                FieldFacts {
                    name: format!(".{i}"),
                    ty_text: type_text(&f.ty),
                    ownership,
                    referenced,
                    cardinality,
                    lifetimes,
                }
            })
            .collect(),
        syn::Fields::Unit => vec![],
    }
}

fn field_from_named(f: &syn::Field) -> FieldFacts {
    let name = f
        .ident
        .as_ref()
        .map(|i| i.to_string())
        .unwrap_or_else(|| "_".into());
    let (ownership, refs, lifetimes) = classify(&f.ty);
    let (referenced, cardinality) = split_refs(refs);
    FieldFacts {
        name,
        ty_text: type_text(&f.ty),
        ownership,
        referenced,
        cardinality,
        lifetimes,
    }
}

/// Helper: split the (name, cardinality) pairs returned by `classify` into
/// the parallel `Vec<String>` and `Vec<Cardinality>` we store on facts.
fn split_refs(
    refs: Vec<(String, crate::model::Cardinality)>,
) -> (Vec<String>, Vec<crate::model::Cardinality>) {
    let mut names = Vec::with_capacity(refs.len());
    let mut cards = Vec::with_capacity(refs.len());
    for (n, c) in refs {
        names.push(n);
        cards.push(c);
    }
    (names, cards)
}

fn build_fn_facts(
    name: &str,
    visibility: String,
    sig: &syn::Signature,
    body: Option<&syn::Block>,
    attrs: &[syn::Attribute],
) -> FnFacts {
    let mut self_kind = SelfKind::None;
    let mut params = Vec::new();
    let mut input_lifetimes: BTreeSet<String> = BTreeSet::new();
    for input in &sig.inputs {
        match input {
            syn::FnArg::Receiver(r) => {
                if r.reference.is_none() {
                    self_kind = SelfKind::ByValue;
                } else if r.mutability.is_some() {
                    self_kind = SelfKind::RefMut;
                } else {
                    self_kind = SelfKind::Ref;
                }
                if let Some((_, Some(lt))) = &r.reference {
                    input_lifetimes.insert(lt.ident.to_string());
                }
            }
            syn::FnArg::Typed(pt) => {
                let pname = match &*pt.pat {
                    syn::Pat::Ident(pi) => pi.ident.to_string(),
                    _ => "_".into(),
                };
                let (ownership, refs, lifetimes) = classify(&pt.ty);
                let (referenced, cardinality) = split_refs(refs);
                for lt in &lifetimes {
                    input_lifetimes.insert(lt.clone());
                }
                params.push(ParamFacts {
                    name: pname,
                    ty_text: type_text(&pt.ty),
                    ownership,
                    referenced,
                    cardinality,
                    lifetimes,
                });
            }
        }
    }

    let (ret_ownership, ret_referenced, ret_cardinality, ret_lifetimes, ret_text) =
        match &sig.output {
            ReturnType::Default => (
                Ownership::Primitive,
                vec![],
                vec![],
                vec![],
                "()".to_string(),
            ),
            ReturnType::Type(_, ty) => {
                let (ownership, refs, lifetimes) = classify(ty);
                let (referenced, cardinality) = split_refs(refs);
                (ownership, referenced, cardinality, lifetimes, type_text(ty))
            }
        };

    let lifetime_flows_through = ret_lifetimes.iter().any(|lt| input_lifetimes.contains(lt));

    let mut unsafe_blocks: u32 = 0;
    if let Some(b) = body {
        let mut counter = UnsafeCounter { count: 0 };
        counter.visit_block(b);
        unsafe_blocks = counter.count;
    }

    FnFacts {
        name: name.to_string(),
        visibility,
        self_kind,
        is_unsafe: sig.unsafety.is_some(),
        is_const: sig.constness.is_some(),
        is_async: sig.asyncness.is_some(),
        lifetime_params: lifetime_params_of(&sig.generics),
        params,
        return_ty_text: ret_text,
        return_ownership: ret_ownership,
        return_referenced: ret_referenced,
        return_cardinality: ret_cardinality,
        lifetime_flows_through,
        unsafe_blocks,
        doc_first_line: doc_first_line(attrs),
    }
}

struct UnsafeCounter {
    count: u32,
}
impl<'ast> Visit<'ast> for UnsafeCounter {
    fn visit_expr_unsafe(&mut self, _: &'ast syn::ExprUnsafe) {
        self.count += 1;
        // Don't recurse — we're only counting top-level unsafe blocks per body.
    }
}

// ── Edge graph ────────────────────────────────────────────────────────────

/// Map a short type name (e.g. "TypeContext") to the canonical full path of
/// the type definition. If a name is ambiguous across crates, we keep all
/// candidates and emit one edge per candidate so we don't drop information.
type Registry = BTreeMap<String, Vec<String>>;

fn build_type_registry(ws: &WorkspaceFacts) -> Registry {
    let mut r: Registry = BTreeMap::new();
    for cf in ws.crates.values() {
        for m in cf.modules.values() {
            for ty in &m.types {
                r.entry(ty.name.clone())
                    .or_default()
                    .push(ty.full_path.clone());
            }
        }
    }
    r
}

/// Resolve a simple type name to one or more full paths, preferring the
/// candidate whose module-path shares the longest prefix with `source`
/// (the full path of the type or fn that holds the reference).
///
/// When multiple candidates tie at the best prefix score, all tied paths
/// are returned — we lack `use`-statement information, so we don't try to
/// pick one arbitrarily. When no candidate shares any prefix at all, we
/// fall back to returning every candidate, since that may legitimately
/// be a cross-module reference brought in via `use`.
fn resolve_name(name: &str, reg: &Registry, source: &str) -> Vec<String> {
    let candidates = reg.get(name).cloned().unwrap_or_default();
    if candidates.len() <= 1 {
        return candidates;
    }
    let source_segs: Vec<&str> = source.split("::").collect();
    let scored: Vec<(usize, String)> = candidates
        .into_iter()
        .map(|c| {
            let cand_segs: Vec<&str> = c.split("::").collect();
            // The candidate's *module* prefix is everything except the last
            // segment (which is the type name itself).
            let module_len = cand_segs.len().saturating_sub(1);
            let score = source_segs
                .iter()
                .zip(cand_segs.iter().take(module_len))
                .take_while(|(a, b)| a == b)
                .count();
            (score, c)
        })
        .collect();
    let best = scored.iter().map(|(s, _)| *s).max().unwrap_or(0);
    if best == 0 {
        return scored.into_iter().map(|(_, c)| c).collect();
    }
    scored
        .into_iter()
        .filter(|(s, _)| *s == best)
        .map(|(_, c)| c)
        .collect()
}

/// Emit edges originating in `ty`. The `via` for field-derived edges is
/// determined by the kind of `ty` (struct/union vs enum). Type aliases do
/// not emit edges — an alias is a name, not a containment relation.
fn emit_edges_from_type(ty: &TypeFacts, reg: &Registry, out: &mut Vec<Edge>) {
    let from = ty.full_path.clone();

    let field_via = match ty.kind {
        TypeKind::Struct => Some(ViaKind::StructField),
        TypeKind::Union => Some(ViaKind::UnionField),
        TypeKind::Enum => Some(ViaKind::EnumVariantPayload),
        TypeKind::Trait | TypeKind::TypeAlias => None,
    };

    if let Some(via) = field_via {
        for f in &ty.fields {
            let kind = match f.ownership {
                Ownership::Owned => EdgeKind::Owns,
                Ownership::BorrowImmut => EdgeKind::BorrowsImmut,
                Ownership::BorrowMut => EdgeKind::BorrowsMut,
                Ownership::Indirection => EdgeKind::Indirection,
                _ => continue,
            };
            for (i, refname) in f.referenced.iter().enumerate() {
                let cardinality = f
                    .cardinality
                    .get(i)
                    .copied()
                    .unwrap_or(crate::model::Cardinality::One);
                for to in resolve_name(refname, reg, &from) {
                    if to == from {
                        continue;
                    }
                    out.push(Edge {
                        from: from.clone(),
                        to,
                        kind,
                        via,
                        cardinality,
                        origin: format!("field {}", f.name),
                    });
                }
            }
        }
    }

    for tr in &ty.trait_impls {
        for to in resolve_name(tr, reg, &from) {
            out.push(Edge {
                from: from.clone(),
                to,
                kind: EdgeKind::TraitImpl,
                via: ViaKind::TraitImplBlock,
                cardinality: crate::model::Cardinality::One,
                origin: "impl".into(),
            });
        }
    }

    for m in &ty.methods {
        emit_edges_from_fn(&from, m, reg, out);
    }
}

fn emit_edges_from_fn(from: &str, f: &FnFacts, reg: &Registry, out: &mut Vec<Edge>) {
    for p in &f.params {
        let kind = match p.ownership {
            Ownership::Owned => EdgeKind::Owns,
            Ownership::BorrowImmut => EdgeKind::BorrowsImmut,
            Ownership::BorrowMut => EdgeKind::BorrowsMut,
            Ownership::Indirection => EdgeKind::Indirection,
            _ => continue,
        };
        for (i, refname) in p.referenced.iter().enumerate() {
            let cardinality = p
                .cardinality
                .get(i)
                .copied()
                .unwrap_or(crate::model::Cardinality::One);
            for to in resolve_name(refname, reg, from) {
                if to == from {
                    continue;
                }
                out.push(Edge {
                    from: from.to_string(),
                    to,
                    kind,
                    via: ViaKind::FnParam,
                    cardinality,
                    origin: format!("fn {} param {}", f.name, p.name),
                });
            }
        }
    }
    let ret_kind = match f.return_ownership {
        Ownership::Owned => Some(EdgeKind::Owns),
        Ownership::BorrowImmut => Some(EdgeKind::BorrowsImmut),
        Ownership::BorrowMut => Some(EdgeKind::BorrowsMut),
        Ownership::Indirection => Some(EdgeKind::Indirection),
        _ => None,
    };
    if let Some(kind) = ret_kind {
        for (i, refname) in f.return_referenced.iter().enumerate() {
            let cardinality = f
                .return_cardinality
                .get(i)
                .copied()
                .unwrap_or(crate::model::Cardinality::One);
            for to in resolve_name(refname, reg, from) {
                if to == from {
                    continue;
                }
                out.push(Edge {
                    from: from.to_string(),
                    to,
                    kind,
                    via: ViaKind::FnReturn,
                    cardinality,
                    origin: format!("fn {} -> ret", f.name),
                });
            }
        }
    }
}

fn build_profiles(edges: &[Edge]) -> BTreeMap<String, EdgeProfile> {
    let mut out: BTreeMap<String, EdgeProfile> = BTreeMap::new();
    let mut inbound_sources: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut outbound_targets: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for e in edges {
        let kind_key = format!("{:?}", e.kind);
        let via_key = format!("{:?}", e.via);
        let entry_from = out.entry(e.from.clone()).or_default();
        *entry_from.outbound.entry(kind_key.clone()).or_insert(0) += 1;
        *entry_from.outbound_via.entry(via_key.clone()).or_insert(0) += 1;
        outbound_targets
            .entry(e.from.clone())
            .or_default()
            .insert(e.to.clone());

        let entry_to = out.entry(e.to.clone()).or_default();
        *entry_to.inbound.entry(kind_key).or_insert(0) += 1;
        *entry_to.inbound_via.entry(via_key).or_insert(0) += 1;
        inbound_sources
            .entry(e.to.clone())
            .or_default()
            .insert(e.from.clone());
    }
    for (name, p) in out.iter_mut() {
        p.inbound_distinct_sources = inbound_sources
            .get(name)
            .map(|s| s.len() as u32)
            .unwrap_or(0);
        p.outbound_distinct_targets = outbound_targets
            .get(name)
            .map(|s| s.len() as u32)
            .unwrap_or(0);
    }
    out
}
