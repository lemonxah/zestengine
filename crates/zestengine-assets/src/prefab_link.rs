//! Linked prefab instances.
//!
//! A scene stores a prefab instance as its root entry plus a [`PrefabLink`]:
//! the asset path and only what the instance changed (overrides, removed
//! objects, removed components) plus the objects added to it. Loading
//! *expands* the link into ordinary entries; saving *compacts* them back by
//! diffing against the pristine asset (decision D1: diff at save, nothing is
//! recorded while editing).
//!
//! Identity: every object inside an instance has the id `live_id(instance,
//! source id)`, so untouched objects need no storage, their ids survive
//! reloads and merges, and references to them keep working. Nesting composes:
//! an object inside a nested instance has the source id path `S/T` (S = the
//! nested root's id in the outer asset, T = its id in the nested asset) and
//! the live id folds the hash over the path, which equals the id the nested
//! instance would give it on its own.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::scene_file::{ComponentData, ObjectSource, SceneEntry};

/// Nested prefab depth after which a link is treated as a cycle.
const MAX_DEPTH: usize = 8;

/// A `parent` override value starting with this names an object by its scene
/// id (outside the instance, or in another instance); what follows may be empty
/// (the scene root). Any other value is a source id inside the instance.
const EXT_PARENT: &str = "ext:";

/// Reads a prefab's raw entries (nested links still unexpanded).
pub type Loader<'a> = &'a dyn Fn(&str) -> Option<Vec<SceneEntry>>;

/// Which id space expansion produces: real object ids (a scene) or the source
/// id paths used inside an asset image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdMode {
    Scene,
    Image,
}

/// A prefab instance's link to its asset, stored on the instance root.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PrefabLink {
    /// Project-relative path of the `.prefab`.
    pub path: String,
    /// Property changes, keyed by source object id and property path.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overrides: Vec<PrefabOverride>,
    /// Source ids of objects deleted from the instance.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub removed: Vec<String>,
    /// `(source id, component key)` of components deleted from an object.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub removed_components: Vec<(String, String)>,
    /// Set once the instance's objects exist (not stored).
    #[serde(skip)]
    pub expanded: bool,
}

impl PrefabLink {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into(), ..Default::default() }
    }

    /// Overrides whose target or property no longer exists in the asset.
    pub fn unused(&self) -> impl Iterator<Item = &PrefabOverride> {
        self.overrides.iter().filter(|o| o.unused)
    }

    /// True when the instance differs from the asset in any way.
    pub fn has_changes(&self) -> bool {
        !self.overrides.is_empty() || !self.removed.is_empty() || !self.removed_components.is_empty()
    }
}

/// One changed property. `path` is a [`SceneEntry`] field (`enabled`, `layer`,
/// `material`, `scale`...), `components[Kind]` (a whole component) or
/// `components[Kind].field`; `value` is its RON text.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PrefabOverride {
    pub target: String,
    pub path: String,
    pub value: String,
    /// The target or property is gone from the asset (set on expand).
    #[serde(skip)]
    pub unused: bool,
}

// ---------------------------------------------------------------- ids

/// Id of the object whose id is `src` inside the instance `parent`.
fn derive_id(parent: &str, src: &str) -> String {
    let mut h1 = 0xcbf2_9ce4_8422_2325u64;
    let mut h2 = 0x8422_2325_cbf2_9ce4u64;
    for b in parent.bytes().chain(std::iter::once(0xffu8)).chain(src.bytes()) {
        h1 = (h1 ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        h2 = (h2 ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(5);
    }
    let b = (((h1 as u128) << 64) | h2 as u128).to_be_bytes();
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
    )
}

/// The id, in the world of `mode`, of the object that has source id `x` in an
/// instance rooted at `root` (whose asset root has source id `root_src`).
fn mk_id(mode: IdMode, root_src: &str, root: &str, x: &str) -> String {
    if x == root_src {
        return root.to_owned();
    }
    match mode {
        IdMode::Scene => x.split('/').fold(root.to_owned(), |acc, c| derive_id(&acc, c)),
        IdMode::Image => format!("{root}/{x}"),
    }
}

/// The scene object id of source id `x` inside the instance `root` of an
/// asset whose root has source id `root_src`.
pub fn live_id(root_src: &str, root: &str, x: &str) -> String {
    mk_id(IdMode::Scene, root_src, root, x)
}

// ---------------------------------------------------------------- RON text helpers

fn ron_text<T: Serialize>(v: &T) -> String {
    ron::to_string(v).unwrap_or_default()
}

fn put<T: serde::de::DeserializeOwned>(slot: &mut T, text: &str) -> bool {
    match ron::from_str::<T>(text) {
        Ok(v) => {
            *slot = v;
            true
        }
        Err(_) => false,
    }
}

/// Walk `s`, calling `f(index, char, in_string_or_char_literal)`.
fn scan(s: &str, mut f: impl FnMut(usize, char, bool)) {
    let mut lit: Option<char> = None;
    let mut esc = false;
    for (i, c) in s.char_indices() {
        match lit {
            Some(q) => {
                f(i, c, true);
                if esc {
                    esc = false;
                } else if c == '\\' {
                    esc = true;
                } else if c == q {
                    lit = None;
                }
            }
            None => {
                if c == '"' || c == '\'' {
                    lit = Some(c);
                    f(i, c, true);
                } else {
                    f(i, c, false);
                }
            }
        }
    }
}

/// `s` without whitespace outside string literals (compares RON text).
fn norm(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    scan(s, |_, c, lit| {
        if lit || !c.is_whitespace() {
            out.push(c);
        }
    });
    out
}

/// A struct-like component state split into its top-level fields.
struct Fields {
    prefix: String,
    items: Vec<(String, String)>,
}

impl Fields {
    fn parse(data: &str) -> Option<Fields> {
        let s = data.trim();
        let open = s.find('(')?;
        let prefix = &s[..open];
        if !prefix.chars().all(|c| c.is_alphanumeric() || c == '_') || !s.ends_with(')') {
            return None;
        }
        let inner = &s[open + 1..s.len() - 1];
        let mut depth = 0i32;
        let mut cuts = vec![0usize];
        scan(inner, |i, c, lit| {
            if lit {
                return;
            }
            match c {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                ',' if depth == 0 => cuts.push(i + 1),
                _ => {}
            }
        });
        cuts.push(inner.len() + 1);
        let mut items = Vec::new();
        for w in cuts.windows(2) {
            let piece = inner[w[0]..(w[1] - 1).min(inner.len())].trim();
            if piece.is_empty() {
                continue;
            }
            let mut colon = None;
            let mut d = 0i32;
            scan(piece, |i, c, lit| {
                if lit || colon.is_some() {
                    return;
                }
                match c {
                    '(' | '[' | '{' => d += 1,
                    ')' | ']' | '}' => d -= 1,
                    ':' if d == 0 => colon = Some(i),
                    _ => {}
                }
            });
            let colon = colon?;
            items.push((piece[..colon].trim().to_owned(), piece[colon + 1..].trim().to_owned()));
        }
        Some(Fields { prefix: prefix.to_owned(), items })
    }

    fn render(&self) -> String {
        let body: Vec<String> = self.items.iter().map(|(k, v)| format!("{k}: {v}")).collect();
        format!("{}({})", self.prefix, body.join(", "))
    }
}

// ---------------------------------------------------------------- entry fields

const FIELDS: [&str; 14] = [
    "name",
    "source",
    "enabled",
    "static_geometry",
    "lightmap_scale",
    "cast_shadows",
    "layer",
    "material",
    "extra_materials",
    "translation",
    "rotation",
    "scale",
    "net_role",
    "component_roles",
];

/// Properties of an instance root that are never overrides: the instance is
/// placed, named, turned and parented by its own entry.
fn is_default_override(root: bool, field: &str) -> bool {
    root && matches!(field, "name" | "translation" | "rotation")
}

fn field_text(e: &SceneEntry, f: &str) -> String {
    match f {
        "name" => ron_text(&e.name),
        "source" => ron_text(&e.source),
        "enabled" => ron_text(&e.enabled),
        "static_geometry" => ron_text(&e.static_geometry),
        "lightmap_scale" => ron_text(&e.lightmap_scale),
        "cast_shadows" => ron_text(&e.cast_shadows),
        "layer" => ron_text(&e.layer),
        "material" => ron_text(&e.material),
        "extra_materials" => ron_text(&e.extra_materials),
        "translation" => ron_text(&e.translation),
        "rotation" => ron_text(&e.rotation),
        "scale" => ron_text(&e.scale),
        "net_role" => ron_text(&e.net_role),
        "component_roles" => ron_text(&e.component_roles),
        _ => String::new(),
    }
}

/// Whether two object sources show the same thing. A model reference may be
/// stored by flattened index, by sub-asset id or both (loading migrates one
/// into the other), so references are compared by id when both sides have one.
fn sources_match(a: &ObjectSource, b: &ObjectSource) -> bool {
    match (a, b) {
        (
            ObjectSource::Model { path: pa, mesh: ma, extra_meshes: ea, mesh_id: ia, extra_mesh_ids: xa, .. },
            ObjectSource::Model { path: pb, mesh: mb, extra_meshes: eb, mesh_id: ib, extra_mesh_ids: xb, .. },
        ) => {
            if pa != pb || ea.len() != eb.len() {
                return false;
            }
            let ids = matches!((ia, ib), (Some(_), Some(_))) && xa.len() == xb.len();
            if ids { ia == ib && xa == xb } else { ma == mb && ea == eb }
        }
        _ => ron_text(a) == ron_text(b),
    }
}

fn floats_close(a: &SceneEntry, b: &SceneEntry, f: &str) -> bool {
    if f == "source" {
        return sources_match(&a.source, &b.source);
    }
    let (x, y): (&[f32], &[f32]) = match f {
        "translation" => (&a.translation, &b.translation),
        "rotation" => (&a.rotation, &b.rotation),
        "scale" => (&a.scale, &b.scale),
        _ => return false,
    };
    x.iter().zip(y).all(|(p, q)| (p - q).abs() < 1e-5)
}

fn set_field(e: &mut SceneEntry, f: &str, text: &str) -> bool {
    match f {
        "name" => put(&mut e.name, text),
        "source" => put(&mut e.source, text),
        "enabled" => put(&mut e.enabled, text),
        "static_geometry" => put(&mut e.static_geometry, text),
        "lightmap_scale" => put(&mut e.lightmap_scale, text),
        "cast_shadows" => put(&mut e.cast_shadows, text),
        "layer" => put(&mut e.layer, text),
        "material" => put(&mut e.material, text),
        "extra_materials" => put(&mut e.extra_materials, text),
        "translation" => put(&mut e.translation, text),
        "rotation" => put(&mut e.rotation, text),
        "scale" => put(&mut e.scale, text),
        "net_role" => put(&mut e.net_role, text),
        "component_roles" => put(&mut e.component_roles, text),
        _ => false,
    }
}

/// Keys that tell apart components of the same kind (`Kind`, `Kind#1`...).
pub fn component_keys(e: &SceneEntry) -> Vec<String> {
    let mut seen: HashMap<&str, usize> = HashMap::new();
    e.components
        .iter()
        .map(|c| {
            let n = seen.entry(c.kind.as_str()).or_insert(0);
            let key = if *n == 0 { c.kind.clone() } else { format!("{}#{}", c.kind, n) };
            *n += 1;
            key
        })
        .collect()
}

fn kind_of(key: &str) -> &str {
    key.split('#').next().unwrap_or(key)
}

fn comp_path(key: &str, field: Option<&str>) -> String {
    match field {
        Some(f) => format!("components[{key}].{f}"),
        None => format!("components[{key}]"),
    }
}

/// `materials[3]` -> (`materials`, 3).
fn parse_index(field: &str) -> Option<(&str, usize)> {
    let open = field.find('[')?;
    let inner = field[open + 1..].strip_suffix(']')?;
    Some((&field[..open], inner.parse().ok()?))
}

/// The top-level elements of a RON list `[a, b(c, d), "x,y"]`.
fn split_list(text: &str) -> Option<Vec<String>> {
    let inner = text.trim().strip_prefix('[')?.strip_suffix(']')?;
    let mut depth = 0i32;
    let mut cuts = vec![0usize];
    scan(inner, |i, c, lit| {
        if lit {
            return;
        }
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => cuts.push(i + 1),
            _ => {}
        }
    });
    cuts.push(inner.len() + 1);
    Some(cuts.windows(2).map(|w| inner[w[0]..(w[1] - 1).min(inner.len())].trim().to_owned()).filter(|p| !p.is_empty()).collect())
}

/// `components[Kind].field` -> (`Kind`, Some(`field`)).
fn parse_comp_path(path: &str) -> Option<(&str, Option<&str>)> {
    let rest = path.strip_prefix("components[")?;
    let close = rest.find(']')?;
    let key = &rest[..close];
    let tail = &rest[close + 1..];
    match tail {
        "" => Some((key, None)),
        _ => tail.strip_prefix('.').map(|f| (key, Some(f))),
    }
}

/// What differs between the pristine and the live version of one object.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldChange {
    pub path: String,
    /// Pristine value (RON; empty for an added component).
    pub pristine: String,
    pub live: String,
}

fn diff_entry(p: &SceneEntry, l: &SceneEntry, root: bool) -> (Vec<FieldChange>, Vec<String>) {
    let mut out = Vec::new();
    for f in FIELDS {
        if is_default_override(root, f) {
            continue;
        }
        if floats_close(p, l, f) {
            continue;
        }
        let (a, b) = (field_text(p, f), field_text(l, f));
        if a != b {
            out.push(FieldChange { path: f.to_owned(), pristine: a, live: b });
        }
    }
    let (pk, lk) = (component_keys(p), component_keys(l));
    for (i, key) in lk.iter().enumerate() {
        let lc = &l.components[i];
        let Some(j) = pk.iter().position(|k| k == key) else {
            out.push(FieldChange { path: comp_path(key, None), pristine: String::new(), live: lc.data.clone() });
            if !lc.enabled {
                out.push(FieldChange { path: comp_path(key, Some("@enabled")), pristine: "true".into(), live: "false".into() });
            }
            continue;
        };
        let pc = &p.components[j];
        if norm(&pc.data) != norm(&lc.data) {
            match (Fields::parse(&pc.data), Fields::parse(&lc.data)) {
                (Some(a), Some(b)) if a.prefix == b.prefix && a.items.len() == b.items.len() && a.items.iter().zip(&b.items).all(|(x, y)| x.0 == y.0) => {
                    for (x, y) in a.items.iter().zip(&b.items) {
                        if norm(&x.1) == norm(&y.1) {
                            continue;
                        }
                        // A renderer's slot list differs per slot, so two variants that change
                        // different slots merge instead of conflicting on "the whole list".
                        if x.0 == "materials"
                            && zestengine_core::is_renderer_kind(kind_of(key))
                            && let (Some(pl), Some(ll)) = (split_list(&x.1), split_list(&y.1))
                            && pl.len() == ll.len()
                        {
                            for (k, (p, l)) in pl.iter().zip(&ll).enumerate() {
                                if norm(p) != norm(l) {
                                    out.push(FieldChange { path: comp_path(key, Some(&format!("materials[{k}]"))), pristine: p.clone(), live: l.clone() });
                                }
                            }
                            continue;
                        }
                        out.push(FieldChange { path: comp_path(key, Some(&x.0)), pristine: x.1.clone(), live: y.1.clone() });
                    }
                }
                _ => out.push(FieldChange { path: comp_path(key, None), pristine: pc.data.clone(), live: lc.data.clone() }),
            }
        }
        if pc.enabled != lc.enabled {
            out.push(FieldChange { path: comp_path(key, Some("@enabled")), pristine: pc.enabled.to_string(), live: lc.enabled.to_string() });
        }
    }
    let removed = pk.iter().filter(|k| !lk.contains(k)).cloned().collect();
    (out, removed)
}

/// Apply one override to an entry. False when the property or component it
/// names doesn't exist (the override is then "unused").
fn apply_override(e: &mut SceneEntry, path: &str, value: &str) -> bool {
    // An override saved against a loose renderer field means its component path now.
    if let Some(list) = crate::renderer_migration::legacy_override(path, value) {
        return list.iter().fold(true, |ok, (p, v)| apply_override(e, p, v) && ok);
    }
    let Some((key, field)) = parse_comp_path(path) else {
        return set_field(e, path, value);
    };
    let keys = component_keys(e);
    let at = keys.iter().position(|k| k == key);
    match (at, field) {
        (None, None) => {
            e.components.push(ComponentData::new(kind_of(key), value));
            true
        }
        (None, Some(_)) => false,
        (Some(i), None) => {
            e.components[i].data = value.to_owned();
            true
        }
        (Some(i), Some("@enabled")) => {
            e.components[i].enabled = value.trim() != "false";
            true
        }
        (Some(i), Some(f)) => {
            let Some(mut fields) = Fields::parse(&e.components[i].data) else { return false };
            // `materials[2]`: one element of a list field.
            let (name, index) = match parse_index(f) {
                Some((n, k)) => (n, Some(k)),
                None => (f, None),
            };
            match (fields.items.iter_mut().find(|(k, _)| k == name), index) {
                (Some(slot), None) => slot.1 = value.to_owned(),
                (Some(slot), Some(k)) => {
                    let Some(mut list) = split_list(&slot.1) else { return false };
                    if list.len() <= k {
                        list.resize(k + 1, "Model".to_owned());
                    }
                    list[k] = value.to_owned();
                    slot.1 = format!("[{}]", list.join(", "));
                }
                (None, _) => return false,
            }
            e.components[i].data = fields.render();
            true
        }
    }
}

fn remove_component(e: &mut SceneEntry, key: &str) -> bool {
    match component_keys(e).iter().position(|k| k == key) {
        Some(i) => {
            e.components.remove(i);
            true
        }
        None => false,
    }
}

// ---------------------------------------------------------------- images and instances

/// An instance built from an image: `entries[0]` is the root; ids are live.
struct Built {
    entries: Vec<SceneEntry>,
    image_ids: Vec<String>,
}

fn build_instance(image: &[SceneEntry], root: &str, mode: IdMode) -> Built {
    let root_src = image[0].id.clone();
    let image_ids: Vec<String> = image.iter().map(|e| e.id.clone()).collect();
    let live: Vec<String> = image_ids.iter().map(|x| mk_id(mode, &root_src, root, x)).collect();
    // Longest source ids first, so "S/T" is replaced before "S".
    let mut order: Vec<usize> = (0..image.len()).filter(|&i| image_ids[i] != live[i]).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(image_ids[i].len()));
    let mut entries = image.to_vec();
    for (i, e) in entries.iter_mut().enumerate() {
        e.id = live[i].clone();
        e.prefab_parent = None;
        for c in &mut e.components {
            for &k in &order {
                if c.data.contains(&image_ids[k]) {
                    c.data = c.data.replace(&image_ids[k], &live[k]);
                }
            }
        }
    }
    Built { entries, image_ids }
}

/// Rewrite live ids in `text` back to the asset's source ids.
fn to_image_space(text: &str, members: &[(String, String)]) -> String {
    let mut out = text.to_owned();
    for (img, live) in members {
        if img != live && out.contains(live.as_str()) {
            out = out.replace(live.as_str(), img);
        }
    }
    out
}

fn descendants(entries: &[SceneEntry], root: usize) -> Vec<usize> {
    let mut out = vec![root];
    let mut i = 0;
    while i < out.len() {
        let p = out[i];
        for (c, e) in entries.iter().enumerate() {
            if e.parent == Some(p) && !out.contains(&c) {
                out.push(c);
            }
        }
        i += 1;
    }
    out
}

/// `entries` without the flagged ones; parents follow the survivors.
fn drop_entries(entries: &mut Vec<SceneEntry>, drop: &[bool]) {
    if !drop.iter().any(|d| *d) {
        return;
    }
    let mut new_index = vec![None; entries.len()];
    let mut next = 0;
    for i in 0..entries.len() {
        if !drop[i] {
            new_index[i] = Some(next);
            next += 1;
        }
    }
    let old = std::mem::take(entries);
    *entries = old
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !drop[*i])
        .map(|(_, mut e)| {
            e.parent = e.parent.and_then(|p| new_index.get(p).copied().flatten());
            e
        })
        .collect();
}

/// The asset's pristine objects: its first root and everything below it,
/// root first, nested prefabs expanded, ids as source id paths. None when the
/// asset can't be read (or nests itself).
pub fn asset_image(path: &str, load: Loader, depth: usize) -> Option<Vec<SceneEntry>> {
    if depth > MAX_DEPTH {
        tracing::warn!("prefab {path}: nested too deep (a prefab contains itself?)");
        return None;
    }
    let mut raw = load(path)?;
    for (i, e) in raw.iter_mut().enumerate() {
        if e.id.is_empty() {
            e.id = format!("auto{i}");
        }
    }
    let root = raw.iter().position(|e| e.parent.is_none())?;
    let order = descendants(&raw, root);
    let mut at = vec![None; raw.len()];
    for (new, &old) in order.iter().enumerate() {
        at[old] = Some(new);
    }
    let mut out: Vec<SceneEntry> = order
        .iter()
        .map(|&old| {
            let mut e = raw[old].clone();
            e.parent = e.parent.and_then(|p| at[p]);
            e
        })
        .collect();
    expand_with(&mut out, load, depth, IdMode::Image);
    Some(out)
}

/// A reparented object waiting for its parent (named by scene id) to exist.
struct PendingParent {
    /// Index of the object in `entries`.
    at: usize,
    /// Scene id of the new parent.
    parent: String,
    /// Instance root index and the override that asked for it.
    root: usize,
    target: String,
}

/// True when making `parent` the parent of `child` would close a loop.
fn would_cycle(parents: &dyn Fn(usize) -> Option<usize>, child: usize, parent: usize) -> bool {
    let mut cur = Some(parent);
    for _ in 0..100_000 {
        match cur {
            Some(c) if c == child => return true,
            Some(c) => cur = parents(c),
            None => return false,
        }
    }
    true
}

fn expand_with(entries: &mut Vec<SceneEntry>, load: Loader, depth: usize, mode: IdMode) {
    let roots: Vec<usize> = (0..entries.len()).filter(|&i| entries[i].prefab.as_ref().is_some_and(|l| !l.expanded)).collect();
    let mut pending = Vec::new();
    for r in roots {
        pending.extend(expand_one(entries, r, load, depth, mode));
    }
    if pending.is_empty() || mode != IdMode::Scene {
        return;
    }
    // Reparented onto objects outside their instance: those exist now.
    let by_id: HashMap<String, usize> = entries.iter().enumerate().map(|(i, e)| (e.id.clone(), i)).collect();
    for p in pending {
        let ok = match by_id.get(&p.parent) {
            Some(&to) if to != p.at && !would_cycle(&|c| entries[c].parent, p.at, to) => {
                entries[p.at].parent = Some(to);
                true
            }
            _ => false,
        };
        if !ok && let Some(link) = entries[p.root].prefab.as_mut() {
            for o in link.overrides.iter_mut().filter(|o| o.path == "parent" && o.target == p.target) {
                o.unused = true;
            }
        }
    }
}

fn expand_one(entries: &mut Vec<SceneEntry>, r: usize, load: Loader, depth: usize, mode: IdMode) -> Vec<PendingParent> {
    let n0 = entries.len();
    let mut pending = Vec::new();
    let Some(mut link) = entries[r].prefab.take() else { return pending };
    link.expanded = true;
    let Some(image) = asset_image(&link.path, load, depth + 1) else {
        tracing::warn!("{}: prefab {} is missing; the instance keeps its stored data", entries[r].name, link.path);
        entries[r].prefab = Some(link);
        return pending;
    };
    let root_id = entries[r].id.clone();
    let mut built = build_instance(&image, &root_id, mode);
    let n = built.entries.len();
    let at: HashMap<String, usize> = built.image_ids.iter().cloned().enumerate().map(|(k, id)| (id, k)).collect();

    // Reparenting first, so what hangs below a removed object but was moved
    // out of it survives the removal.
    let mut outside: Vec<(usize, String, String)> = Vec::new();
    for o in link.overrides.iter_mut().filter(|o| o.path == "parent") {
        o.unused = true;
        let Some(&k) = at.get(&o.target) else { continue };
        if k == 0 {
            continue;
        }
        let Ok(value) = ron::from_str::<String>(&o.value) else { continue };
        match value.strip_prefix(EXT_PARENT) {
            Some(ext) => {
                o.unused = false;
                built.entries[k].parent = None;
                if !ext.is_empty() {
                    outside.push((k, ext.to_owned(), o.target.clone()));
                }
            }
            None => {
                if let Some(&p) = at.get(&value)
                    && p != k
                    && !would_cycle(&|c| built.entries[c].parent, k, p)
                {
                    built.entries[k].parent = Some(p);
                    o.unused = false;
                }
            }
        }
    }

    let mut gone = vec![false; n];
    // Parents may now come after their children, so settle by passes.
    for _ in 0..n {
        let mut changed = false;
        for k in 1..n {
            let g = link.removed.contains(&built.image_ids[k]) || built.entries[k].parent.is_some_and(|p| gone[p]);
            if g != gone[k] {
                gone[k] = g;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for (target, key) in &link.removed_components {
        if let Some(&k) = at.get(target) {
            remove_component(&mut built.entries[k], key);
        }
    }
    for o in &mut link.overrides {
        if o.path == "parent" {
            continue;
        }
        o.unused = true;
        let Some(&k) = at.get(&o.target) else { continue };
        if gone[k] {
            continue;
        }
        o.unused = if is_default_override(k == 0, &o.path) { false } else { !apply_override(&mut built.entries[k], &o.path, &o.value) };
    }

    // The root keeps its own id, name, placement and parent; the rest is the
    // asset's (with the overrides on top).
    let b0 = built.entries[0].clone();
    let stored = &mut entries[r];
    stored.source = b0.source;
    stored.enabled = b0.enabled;
    stored.static_geometry = b0.static_geometry;
    stored.lightmap_scale = b0.lightmap_scale;
    stored.cast_shadows = b0.cast_shadows;
    stored.layer = b0.layer;
    stored.net_role = b0.net_role;
    stored.component_roles = b0.component_roles;
    stored.material = b0.material;
    stored.extra_materials = b0.extra_materials;
    stored.scale = b0.scale;
    stored.components = b0.components;
    stored.prefab = Some(link);

    let mut abs = vec![r; n];
    let mut next = entries.len();
    for k in 1..n {
        if !gone[k] {
            abs[k] = next;
            next += 1;
        }
    }
    for (k, built_entry) in built.entries.iter().enumerate().skip(1) {
        if gone[k] {
            continue;
        }
        let mut e = built_entry.clone();
        e.parent = e.parent.map(|p| abs[p]);
        entries.push(e);
    }
    for (k, parent, target) in outside {
        if !gone[k] {
            pending.push(PendingParent { at: abs[k], parent, root: r, target });
        }
    }
    // Objects added to the instance under one of its own objects.
    for (j, entry) in entries.iter_mut().enumerate().take(n0) {
        if j != r
            && entry.parent == Some(r)
            && let Some(x) = entry.prefab_parent.take()
            && let Some(&k) = at.get(&x)
            && !gone[k]
        {
            entry.parent = Some(abs[k]);
        }
    }
    pending
}

/// Build every unexpanded prefab link in `entries` into objects: clone the
/// asset with derived ids, drop what the instance removed, apply its
/// overrides (kept, flagged unused, when their target is gone) and splice the
/// objects it added.
pub fn expand_prefabs(entries: &mut Vec<SceneEntry>, load: Loader) {
    expand_with(entries, load, 0, IdMode::Scene);
}

// ---------------------------------------------------------------- compaction

/// What one object of an instance has changed.
#[derive(Clone, Debug, PartialEq)]
pub enum Change {
    Modified { fields: Vec<FieldChange>, removed_components: Vec<String> },
    RemovedObject,
    AddedObject,
}

/// One row of an instance's override list.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectDiff {
    /// Source id of the object in the asset (empty for an added object).
    pub image_id: String,
    pub live_id: String,
    pub name: String,
    pub change: Change,
}

/// An instance compared with its asset.
#[derive(Clone, Debug, Default)]
pub struct InstanceInfo {
    pub root: String,
    pub path: String,
    /// `(source id, live id)` for every object the asset gives the instance.
    pub members: Vec<(String, String)>,
    pub diffs: Vec<ObjectDiff>,
}

fn linked_depth(entries: &[SceneEntry], i: usize) -> usize {
    let mut d = 0;
    let mut cur = entries[i].parent;
    let mut steps = 0;
    while let Some(p) = cur {
        if p >= entries.len() || steps > entries.len() {
            break;
        }
        if entries[p].prefab.is_some() {
            d += 1;
        }
        cur = entries[p].parent;
        steps += 1;
    }
    d
}

fn is_inside(entries: &[SceneEntry], mut j: usize, root: usize) -> bool {
    for _ in 0..=entries.len() {
        match entries[j].parent {
            Some(p) if p == root => return true,
            Some(p) if p < entries.len() => j = p,
            _ => return false,
        }
    }
    false
}

/// Store every instance as its change list: pristine objects (and the nested
/// instances among them) are dropped, differences become overrides, deleted
/// objects and components are listed, and added objects stay. Outermost
/// instances first. Instances whose asset can't be read are left as they are.
pub fn compact_prefabs(entries: &mut Vec<SceneEntry>, load: Loader) -> Vec<InstanceInfo> {
    compact_with(entries, load, IdMode::Scene)
}

/// Who owns what, taken before any instance is compacted (compacting drops
/// objects and renumbers the rest): which instances each object belongs to,
/// and the id of every object's parent.
struct Ownership {
    /// Object id -> ids of the instance roots it is a member of.
    claims: HashMap<String, Vec<String>>,
    /// Object id -> id of its parent, as the scene had it.
    parent_of: HashMap<String, Option<String>>,
}

impl Ownership {
    fn of(entries: &[SceneEntry], load: Loader, mode: IdMode) -> Ownership {
        let mut claims: HashMap<String, Vec<String>> = HashMap::new();
        for e in entries.iter().filter(|e| e.prefab.as_ref().is_some_and(|l| l.expanded)) {
            let Some(link) = &e.prefab else { continue };
            let Some(image) = asset_image(&link.path, load, 1) else { continue };
            let built = build_instance(&image, &e.id, mode);
            for m in built.entries.iter().skip(1) {
                claims.entry(m.id.clone()).or_default().push(e.id.clone());
            }
        }
        let parent_of = entries.iter().map(|e| (e.id.clone(), e.parent.and_then(|p| entries.get(p)).map(|p| p.id.clone()))).collect();
        Ownership { claims, parent_of }
    }

    /// `entries[j]` belongs to the instance rooted at `r`: it hangs below the
    /// root, or below one of the root's members, without passing through an
    /// object another instance owns.
    fn added_to(&self, entries: &[SceneEntry], j: usize, r: usize) -> bool {
        if self.claims.get(&entries[j].id).is_some_and(|c| !c.contains(&entries[r].id)) {
            return false;
        }
        let mut cur = entries[j].parent;
        for _ in 0..=entries.len() {
            match cur {
                Some(p) if p == r => return true,
                Some(p) if p < entries.len() => {
                    if let Some(c) = self.claims.get(&entries[p].id) {
                        return c.contains(&entries[r].id);
                    }
                    cur = entries[p].parent;
                }
                _ => return false,
            }
        }
        false
    }
}

fn compact_with(entries: &mut Vec<SceneEntry>, load: Loader, mode: IdMode) -> Vec<InstanceInfo> {
    let mut infos = Vec::new();
    let mut done: HashSet<String> = HashSet::new();
    let own = Ownership::of(entries, load, mode);
    loop {
        let next = (0..entries.len())
            .filter(|&i| entries[i].prefab.as_ref().is_some_and(|l| l.expanded) && !done.contains(&entries[i].id))
            .min_by_key(|&i| linked_depth(entries, i));
        let Some(r) = next else { break };
        done.insert(entries[r].id.clone());
        if let Some(info) = compact_one(entries, r, load, mode, &own) {
            infos.push(info);
        }
    }
    infos
}

fn compact_one(entries: &mut Vec<SceneEntry>, r: usize, load: Loader, mode: IdMode, own: &Ownership) -> Option<InstanceInfo> {
    let old = entries[r].prefab.clone()?;
    let image = asset_image(&old.path, load, 1)?;
    let root_id = entries[r].id.clone();
    let built = build_instance(&image, &root_id, mode);
    let n = built.entries.len();
    let by_id: HashMap<&str, usize> = entries.iter().enumerate().map(|(i, e)| (e.id.as_str(), i)).collect();
    let live_at: Vec<Option<usize>> = built.entries.iter().map(|e| by_id.get(e.id.as_str()).copied()).collect();

    let mut link = PrefabLink { path: old.path.clone(), expanded: false, ..Default::default() };
    let mut diffs = Vec::new();
    for k in 0..n {
        match live_at[k] {
            None => {
                if k > 0 && built.entries[k].parent.is_some_and(|p| live_at[p].is_some()) {
                    link.removed.push(built.image_ids[k].clone());
                    diffs.push(ObjectDiff {
                        image_id: built.image_ids[k].clone(),
                        live_id: built.entries[k].id.clone(),
                        name: built.entries[k].name.clone(),
                        change: Change::RemovedObject,
                    });
                }
            }
            Some(li) => {
                let (mut fields, removed) = diff_entry(&built.entries[k], &entries[li], k == 0);
                if k > 0 {
                    let pristine = built.entries[k].parent.map(|p| built.entries[p].id.clone());
                    let live = own.parent_of.get(&entries[li].id).cloned().flatten();
                    if pristine != live {
                        // Below another member of this instance: its source id.
                        // Anywhere else: its scene id (or none for the scene root).
                        let inside = live.as_ref().and_then(|id| built.entries.iter().position(|e| &e.id == id).filter(|p| live_at[*p].is_some()));
                        let value = match (&live, inside) {
                            (_, Some(p)) => built.image_ids[p].clone(),
                            (Some(id), None) => format!("{EXT_PARENT}{id}"),
                            (None, None) => EXT_PARENT.to_owned(),
                        };
                        let was = built.entries[k].parent.map(|p| ron_text(&built.image_ids[p])).unwrap_or_default();
                        fields.push(FieldChange { path: "parent".to_owned(), pristine: was, live: ron_text(&value) });
                    }
                }
                for f in &fields {
                    link.overrides.push(PrefabOverride {
                        target: built.image_ids[k].clone(),
                        path: f.path.clone(),
                        value: f.live.clone(),
                        unused: false,
                    });
                }
                for key in &removed {
                    link.removed_components.push((built.image_ids[k].clone(), key.clone()));
                }
                if !fields.is_empty() || !removed.is_empty() {
                    diffs.push(ObjectDiff {
                        image_id: built.image_ids[k].clone(),
                        live_id: built.entries[k].id.clone(),
                        name: entries[li].name.clone(),
                        change: Change::Modified { fields, removed_components: removed },
                    });
                }
            }
        }
    }
    // What the asset can't show any more stays, flagged unused.
    for o in old.overrides.iter().filter(|o| o.unused) {
        if !link.overrides.iter().any(|x| x.target == o.target && x.path == o.path) {
            link.overrides.push(o.clone());
        }
    }
    for id in &old.removed {
        if !built.image_ids.contains(id) {
            link.removed.push(id.clone());
        }
    }
    for (t, key) in &old.removed_components {
        if !built.image_ids.contains(t) {
            link.removed_components.push((t.clone(), key.clone()));
        }
    }

    let members: HashSet<usize> = live_at.iter().flatten().copied().collect();
    let member_image: HashMap<usize, &str> = live_at.iter().enumerate().filter_map(|(k, li)| li.map(|li| (li, built.image_ids[k].as_str()))).collect();
    let added: Vec<usize> = (0..entries.len()).filter(|&j| j != r && !members.contains(&j) && own.added_to(entries, j, r)).collect();
    for &j in &added {
        diffs.push(ObjectDiff { image_id: String::new(), live_id: entries[j].id.clone(), name: entries[j].name.clone(), change: Change::AddedObject });
    }
    // Added objects under a pristine object point at it by source id.
    let mut reparent: Vec<(usize, String)> = Vec::new();
    for &j in &added {
        if let Some(p) = entries[j].parent
            && p != r
            && members.contains(&p)
            && let Some(img) = member_image.get(&p)
        {
            reparent.push((j, (*img).to_owned()));
        }
    }
    let info_members: Vec<(String, String)> =
        built.image_ids.iter().zip(&built.entries).filter(|(_, e)| by_id.contains_key(e.id.as_str())).map(|(i, e)| (i.clone(), e.id.clone())).collect();
    let info = InstanceInfo { root: root_id, path: old.path.clone(), members: info_members, diffs };

    for (j, img) in reparent {
        entries[j].parent = Some(r);
        entries[j].prefab_parent = Some(img);
    }
    let b0 = &built.entries[0];
    let stored = &mut entries[r];
    stored.source = b0.source.clone();
    stored.enabled = b0.enabled;
    stored.static_geometry = b0.static_geometry;
    stored.lightmap_scale = b0.lightmap_scale;
    stored.cast_shadows = b0.cast_shadows;
    stored.layer = b0.layer;
    stored.net_role = b0.net_role;
    stored.component_roles = b0.component_roles.clone();
    stored.material = b0.material.clone();
    stored.extra_materials = b0.extra_materials.clone();
    stored.scale = b0.scale;
    stored.components = Vec::new();
    stored.prefab = Some(link);
    let drop: Vec<bool> = (0..entries.len()).map(|j| j != r && members.contains(&j)).collect();
    drop_entries(entries, &drop);
    Some(info)
}

/// Re-merge after a prefab asset changed: compact every instance against the
/// asset as it was, then expand against the new one. Object ids come from the
/// source ids, so identity and incoming references survive; overrides whose
/// target is gone stay flagged unused.
pub fn remerge_prefabs(entries: &mut Vec<SceneEntry>, old: Loader, new: Loader) {
    compact_prefabs(entries, old);
    expand_prefabs(entries, new);
}

/// The diff of one instance, without touching `entries`.
pub fn instance_info(entries: &[SceneEntry], root_id: &str, load: Loader) -> Option<InstanceInfo> {
    let mut copy = entries.to_vec();
    compact_prefabs(&mut copy, load).into_iter().find(|i| i.root == root_id)
}

// ---------------------------------------------------------------- revert and apply

/// Which part of an instance's changes an operation covers.
#[derive(Clone, Debug, PartialEq)]
pub enum PrefabScope {
    All,
    /// Every override on one object (by source id).
    Object(String),
    Property { target: String, path: String },
    /// An added object (live id) and what is below it.
    Added(String),
    /// A removed object (by source id).
    RemovedObject(String),
    RemovedComponent { target: String, key: String },
}

impl PrefabScope {
    fn covers_override(&self, o: &PrefabOverride) -> bool {
        match self {
            PrefabScope::All => true,
            PrefabScope::Object(t) => &o.target == t,
            PrefabScope::Property { target, path } => &o.target == target && &o.path == path,
            _ => false,
        }
    }
}

/// Take the covered changes out of the instance's stored link (and the added
/// objects it covers out of `entries`). Returns the removed pieces.
struct Taken {
    overrides: Vec<PrefabOverride>,
    removed: Vec<String>,
    removed_components: Vec<(String, String)>,
    /// Stored added entries (the covered subtrees), parents local to this list.
    added: Vec<SceneEntry>,
}

fn take_changes(entries: &mut Vec<SceneEntry>, r: usize, scope: &PrefabScope) -> Taken {
    let mut taken = Taken { overrides: Vec::new(), removed: Vec::new(), removed_components: Vec::new(), added: Vec::new() };
    if let Some(link) = entries[r].prefab.as_mut() {
        let (hit, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut link.overrides).into_iter().partition(|o| !o.unused && scope.covers_override(o));
        link.overrides = keep;
        taken.overrides = hit;
        match scope {
            PrefabScope::All => {
                taken.removed = std::mem::take(&mut link.removed);
                taken.removed_components = std::mem::take(&mut link.removed_components);
            }
            PrefabScope::Object(t) => {
                let (hit, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut link.removed_components).into_iter().partition(|(x, _)| x == t);
                link.removed_components = keep;
                taken.removed_components = hit;
            }
            PrefabScope::RemovedObject(t) => {
                if let Some(i) = link.removed.iter().position(|x| x == t) {
                    taken.removed.push(link.removed.remove(i));
                }
            }
            PrefabScope::RemovedComponent { target, key } => {
                if let Some(i) = link.removed_components.iter().position(|(x, k)| x == target && k == key) {
                    taken.removed_components.push(link.removed_components.remove(i));
                }
            }
            _ => {}
        }
    }
    let roots: Vec<usize> = match scope {
        PrefabScope::All => (0..entries.len()).filter(|&j| j != r && entries[j].parent == Some(r)).collect(),
        PrefabScope::Added(id) => (0..entries.len()).filter(|&j| entries[j].id == *id && is_inside(entries, j, r)).collect(),
        _ => Vec::new(),
    };
    if !roots.is_empty() {
        let mut gone = vec![false; entries.len()];
        let mut order = Vec::new();
        for &a in &roots {
            for d in descendants(entries, a) {
                if !gone[d] {
                    gone[d] = true;
                    order.push(d);
                }
            }
        }
        let local: HashMap<usize, usize> = order.iter().enumerate().map(|(k, &o)| (o, k)).collect();
        taken.added = order
            .iter()
            .map(|&o| {
                let mut e = entries[o].clone();
                e.parent = e.parent.and_then(|p| local.get(&p).copied());
                if e.parent.is_none() && e.prefab_parent.is_none() {
                    // Directly under the instance root: keep that fact.
                    e.prefab_parent = Some(String::new());
                }
                e
            })
            .collect();
        drop_entries(entries, &gone);
    }
    taken
}

fn root_index(entries: &[SceneEntry], id: &str) -> Option<usize> {
    entries.iter().position(|e| e.id == id && e.prefab.is_some())
}

/// Throw the covered changes away and rebuild the instance from its asset.
/// Every instance in `entries` is compacted and expanded on the way, so
/// object ids and incoming references stay the same. False when `root_id`
/// isn't an instance.
pub fn revert_prefab(entries: &mut Vec<SceneEntry>, root_id: &str, scope: &PrefabScope, load: Loader) -> bool {
    compact_prefabs(entries, load);
    let Some(r) = root_index(entries, root_id) else {
        expand_prefabs(entries, load);
        return false;
    };
    take_changes(entries, r, scope);
    expand_prefabs(entries, load);
    true
}

fn is_outside_parent(o: &PrefabOverride) -> bool {
    o.path == "parent" && ron::from_str::<String>(&o.value).is_ok_and(|v| v.starts_with(EXT_PARENT))
}

/// Where an Apply writes: the instance's outermost asset, or a prefab nested
/// inside it (`nested` = the nested root's source id in the outer asset).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApplyLevel {
    pub nested: Option<String>,
}

impl ApplyLevel {
    pub fn outermost() -> Self {
        Self { nested: None }
    }
}

/// Move the covered changes into the prefab asset at `level`. Returns the
/// asset's path and its new entries to be written; the instance (and every
/// other instance, which have all been compacted) is left compact. The caller
/// saves the asset, refreshes its loader and calls [`expand_prefabs`].
/// The rebase mapping of a prefab apply: (member pairs, live-id resolver).
type Rebase = (Vec<(String, String)>, Box<dyn Fn(&str) -> Option<String>>);

pub fn apply_prefab(entries: &mut Vec<SceneEntry>, root_id: &str, scope: &PrefabScope, level: &ApplyLevel, load: Loader) -> Option<(String, Vec<SceneEntry>)> {
    let infos = compact_prefabs(entries, load);
    let info = infos.iter().find(|i| i.root == root_id)?;
    let r = root_index(entries, root_id)?;
    let outer_image = asset_image(&info.path, load, 1)?;
    let root_src = outer_image[0].id.clone();

    // The asset to change and where its objects live.
    let (path, prefix) = match &level.nested {
        None => (info.path.clone(), String::new()),
        Some(p) => {
            let e = outer_image.iter().find(|e| e.id == *p)?;
            (e.prefab.as_ref()?.path.clone(), format!("{p}/"))
        }
    };
    if level.nested.is_some() && !matches!(scope, PrefabScope::Property { .. } | PrefabScope::Object(_) | PrefabScope::All) {
        return None;
    }
    let mut image = asset_image(&path, load, 1)?;
    let (members, rebase): Rebase = match &level.nested {
        None => (info.members.clone(), Box::new(|t: &str| Some(t.to_owned()))),
        Some(p) => {
            let nested_live = live_id(&root_src, root_id, p);
            let b = build_instance(&image, &nested_live, IdMode::Scene);
            let prefix = prefix.clone();
            (
                b.image_ids.iter().cloned().zip(b.entries.iter().map(|e| e.id.clone())).collect(),
                Box::new(move |t: &str| t.strip_prefix(prefix.as_str()).map(str::to_owned)),
            )
        }
    };

    // Only the pieces that belong to this level are taken from the instance.
    let mut taken = take_changes(entries, r, scope);
    let mut put_back = Taken { overrides: Vec::new(), removed: Vec::new(), removed_components: Vec::new(), added: Vec::new() };
    taken.overrides.retain(|o| {
        // A parent outside the instance can't be written into the asset.
        let mine = rebase(&o.target).is_some() && !is_outside_parent(o);
        if !mine {
            put_back.overrides.push(o.clone());
        }
        mine
    });
    if level.nested.is_some() {
        // Nested levels take only the changes that target the nested asset.
        let (mine, other): (Vec<_>, Vec<_>) = std::mem::take(&mut taken.removed).into_iter().partition(|t| rebase(t).is_some());
        taken.removed = mine;
        put_back.removed = other;
        let (mine, other): (Vec<_>, Vec<_>) = std::mem::take(&mut taken.removed_components).into_iter().partition(|(t, _)| rebase(t).is_some());
        taken.removed_components = mine;
        put_back.removed_components = other;
        put_back.added = std::mem::take(&mut taken.added);
    }
    if let Some(link) = entries[r].prefab.as_mut() {
        link.overrides.extend(put_back.overrides);
        link.removed.extend(put_back.removed);
        link.removed_components.extend(put_back.removed_components);
    }
    if !put_back.added.is_empty() {
        let base = entries.len();
        for mut e in put_back.added {
            e.parent = match (e.parent, e.prefab_parent.as_deref()) {
                (Some(p), _) => Some(base + p),
                (None, _) => Some(r),
            };
            if e.prefab_parent.as_deref() == Some("") {
                e.prefab_parent = None;
            }
            entries.push(e);
        }
    }

    let at = |image: &Vec<SceneEntry>, id: &str| image.iter().position(|e| e.id == id);
    for o in &taken.overrides {
        let Some(t) = rebase(&o.target) else { continue };
        if let Some(i) = at(&image, &t) {
            if o.path == "parent" {
                let new = ron::from_str::<String>(&o.value).ok().and_then(|v| rebase(&v)).and_then(|v| at(&image, &v));
                if let Some(p) = new
                    && i > 0
                    && p != i
                    && !would_cycle(&|c| image[c].parent, i, p)
                {
                    image[i].parent = Some(p);
                }
                continue;
            }
            let value = to_image_space(&o.value, &members);
            apply_override(&mut image[i], &o.path, &value);
        }
    }
    for (t, key) in &taken.removed_components {
        if let Some(t) = rebase(t)
            && let Some(i) = at(&image, &t)
        {
            remove_component(&mut image[i], key);
        }
    }
    for t in &taken.removed {
        if let Some(t) = rebase(t)
            && let Some(i) = at(&image, &t)
            && i > 0
        {
            let gone = descendants(&image, i);
            let drop: Vec<bool> = (0..image.len()).map(|k| gone.contains(&k)).collect();
            drop_entries(&mut image, &drop);
        }
    }
    if level.nested.is_none() && !taken.added.is_empty() {
        let base = image.len();
        for mut e in taken.added.drain(..) {
            e.parent = match (e.parent, e.prefab_parent.as_deref()) {
                (Some(p), _) => Some(base + p),
                (None, Some("")) | (None, None) => Some(0),
                (None, Some(x)) => Some(at(&image, x).unwrap_or(0)),
            };
            e.prefab_parent = None;
            for c in &mut e.components {
                c.data = to_image_space(&c.data, &members);
            }
            image.push(e);
        }
        // Added nested instances need their objects before they can be compacted.
        expand_with(&mut image, load, 1, IdMode::Image);
    }
    compact_with(&mut image, load, IdMode::Image);
    for e in &mut image {
        e.prefab_parent = e.prefab_parent.take().filter(|p| !p.is_empty());
    }
    Some((path, image))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_file::{MaterialRef, ObjectSource};
    use std::cell::RefCell;

    fn entry(id: &str, name: &str, parent: Option<usize>) -> SceneEntry {
        SceneEntry {
            id: id.into(),
            name: name.into(),
            source: ObjectSource::Empty,
            enabled: true,
            static_geometry: false,
            lightmap_scale: 1.0,
            cast_shadows: zestengine_core::ShadowCasting::On,
            layer: 0,
            material: MaterialRef::Model,
            extra_materials: Vec::new(),
            renderer_split: false,
            parent,
            components: Vec::new(),
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            model: None,
            model_part: None,
            prefab: None,
            prefab_parent: None,
            net_role: Default::default(),
            component_roles: Default::default(),
        }
    }

    fn turret() -> Vec<SceneEntry> {
        let mut base = entry("base", "Base", None);
        base.components.push(ComponentData::new("Turret", "(range: 5.0, barrel: Some(\"gun\"), name: \"t\")"));
        let mut gun = entry("gun", "Gun", Some(0));
        gun.translation = [0.0, 1.0, 0.0];
        let mut cap = entry("cap", "Cap", Some(1));
        cap.layer = 3;
        vec![base, gun, cap]
    }

    struct Assets(RefCell<HashMap<String, Vec<SceneEntry>>>);
    impl Assets {
        fn new(list: &[(&str, Vec<SceneEntry>)]) -> Self {
            Assets(RefCell::new(list.iter().map(|(p, e)| (p.to_string(), e.clone())).collect()))
        }
        fn set(&self, p: &str, e: Vec<SceneEntry>) {
            self.0.borrow_mut().insert(p.to_string(), e);
        }
        fn get(&self, p: &str) -> Option<Vec<SceneEntry>> {
            self.0.borrow().get(p).cloned()
        }
    }

    fn placed(path: &str, id: &str, assets: &Assets) -> Vec<SceneEntry> {
        let mut root = entry(id, "Turret", None);
        root.prefab = Some(PrefabLink::new(path));
        let mut v = vec![root];
        expand_prefabs(&mut v, &|p| assets.get(p));
        v
    }

    fn by_name<'a>(v: &'a [SceneEntry], n: &str) -> &'a SceneEntry {
        v.iter().find(|e| e.name == n).unwrap_or_else(|| panic!("no {n}"))
    }

    fn idx(v: &[SceneEntry], n: &str) -> usize {
        v.iter().position(|e| e.name == n).unwrap()
    }

    #[test]
    fn expanding_derives_ids_and_remaps_references() {
        let a = Assets::new(&[("t.prefab", turret())]);
        let v = placed("t.prefab", "I1", &a);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0].id, "I1");
        let gun = by_name(&v, "Gun");
        assert_eq!(gun.id, live_id("base", "I1", "gun"));
        assert_ne!(gun.id, "gun");
        assert_eq!(gun.parent, Some(0));
        assert!(v[0].components[0].data.contains(&gun.id) && !v[0].components[0].data.contains("\"gun\""));
        assert_eq!(by_name(&v, "Cap").parent, Some(idx(&v, "Gun")));
        let again = placed("t.prefab", "I2", &a);
        assert_ne!(by_name(&again, "Gun").id, gun.id, "instances never share ids");
    }

    #[test]
    fn pristine_instance_compacts_to_just_the_root_and_round_trips() {
        let a = Assets::new(&[("t.prefab", turret())]);
        let full = placed("t.prefab", "I1", &a);
        let mut v = full.clone();
        let infos = compact_prefabs(&mut v, &|p| a.get(p));
        assert_eq!(v.len(), 1, "only the root is stored");
        assert!(v[0].components.is_empty());
        assert!(infos[0].diffs.is_empty());
        let link = v[0].prefab.as_ref().unwrap();
        assert!(!link.has_changes());
        expand_prefabs(&mut v, &|p| a.get(p));
        let key = |e: &SceneEntry| (e.id.clone(), e.name.clone(), e.parent, e.layer, e.components.iter().map(|c| c.data.clone()).collect::<Vec<_>>());
        let mut x: Vec<_> = full.iter().map(key).collect();
        let mut y: Vec<_> = v.iter().map(key).collect();
        x.sort();
        y.sort();
        assert_eq!(x, y, "compact then expand gives the same objects");
    }

    #[test]
    fn child_rotation_and_scale_overrides_hold_local_values() {
        // SceneEntry transforms are local to the parent entry, so a child's
        // override stores its local rotation / scale (not world values) and
        // survives a save / load round trip unchanged.
        let a = Assets::new(&[("t.prefab", turret())]);
        let mut v = placed("t.prefab", "I1", &a);
        v[0].scale = [2.0, 1.0, 0.5]; // non-uniform parent
        v[0].rotation = [0.0, 0.258_819, 0.0, 0.965_926];
        let gun = idx(&v, "Gun");
        v[gun].rotation = [0.0, 0.0, 0.382_683, 0.923_880];
        v[gun].scale = [1.5, 1.5, 3.0];
        compact_prefabs(&mut v, &|p| a.get(p));
        let link = v[0].prefab.clone().unwrap();
        assert!(link.overrides.iter().any(|o| o.target == "gun" && o.path == "rotation"), "{:?}", link.overrides);
        assert!(link.overrides.iter().any(|o| o.target == "gun" && o.path == "scale"), "{:?}", link.overrides);
        let text = ron::to_string(&v).unwrap();
        let mut back: Vec<SceneEntry> = ron::from_str(&text).unwrap();
        expand_prefabs(&mut back, &|p| a.get(p));
        let g = by_name(&back, "Gun");
        assert_eq!(g.rotation, [0.0, 0.0, 0.382_683, 0.923_880]);
        assert_eq!(g.scale, [1.5, 1.5, 3.0]);
    }

    #[test]
    fn default_overrides_are_not_overrides_but_scale_is() {
        let a = Assets::new(&[("t.prefab", turret())]);
        let mut v = placed("t.prefab", "I1", &a);
        v[0].name = "Renamed".into();
        v[0].translation = [4.0, 0.0, 1.0];
        v[0].rotation = [0.0, 1.0, 0.0, 0.0];
        compact_prefabs(&mut v, &|p| a.get(p));
        assert!(!v[0].prefab.as_ref().unwrap().has_changes(), "name and placement are stored on the root, not as overrides");
        let mut v = placed("t.prefab", "I1", &a);
        v[0].scale = [2.0; 3];
        v[0].enabled = false;
        compact_prefabs(&mut v, &|p| a.get(p));
        let paths: Vec<&str> = v[0].prefab.as_ref().unwrap().overrides.iter().map(|o| o.path.as_str()).collect();
        assert!(paths.contains(&"scale") && paths.contains(&"enabled"), "{paths:?}");
        expand_prefabs(&mut v, &|p| a.get(p));
        assert_eq!(v[0].scale, [2.0; 3]);
        assert!(!v[0].enabled);
    }

    #[test]
    fn overrides_removed_and_added_survive_a_round_trip() {
        let a = Assets::new(&[("t.prefab", turret())]);
        let mut v = placed("t.prefab", "I1", &a);
        let gun = idx(&v, "Gun");
        let cap = idx(&v, "Cap");
        v[gun].translation = [0.0, 2.0, 0.0];
        v[0].components[0].data = "(range: 9.0, barrel: Some(\"x\"), name: \"t\")".replace("\"x\"", &format!("\"{}\"", v[gun].id));
        // Remove Cap, add a Flag under Gun and a Light under the root.
        v.remove(cap);
        let gun = idx(&v, "Gun");
        let mut flag = entry("flag-id", "Flag", Some(gun));
        flag.layer = 7;
        v.push(flag);
        v.push(entry("light-id", "Light", Some(0)));
        compact_prefabs(&mut v, &|p| a.get(p));
        let link = v[0].prefab.clone().unwrap();
        assert_eq!(link.removed, vec!["cap".to_string()]);
        assert!(link.overrides.iter().any(|o| o.target == "gun" && o.path == "translation"));
        assert!(link.overrides.iter().any(|o| o.target == "base" && o.path == "components[Turret].range"), "{:?}", link.overrides);
        assert_eq!(v.len(), 3, "root plus two added objects");
        assert_eq!(by_name(&v, "Flag").prefab_parent.as_deref(), Some("gun"));
        // Save and load through RON.
        let text = ron::to_string(&v).unwrap();
        let mut back: Vec<SceneEntry> = ron::from_str(&text).unwrap();
        expand_prefabs(&mut back, &|p| a.get(p));
        assert!(back.iter().all(|e| e.name != "Cap"));
        let g = by_name(&back, "Gun");
        assert_eq!(g.translation, [0.0, 2.0, 0.0]);
        assert!(back[0].components[0].data.contains("range: 9.0"));
        let flag = by_name(&back, "Flag");
        assert_eq!(flag.parent, Some(idx(&back, "Gun")), "the added object is back under Gun");
        assert_eq!(flag.layer, 7);
        assert_eq!(by_name(&back, "Light").parent, Some(0));
    }

    #[test]
    fn asset_change_remerges_keeping_ids_and_unused_overrides() {
        let a = Assets::new(&[("t.prefab", turret())]);
        let mut v = placed("t.prefab", "I1", &a);
        let gun_id = by_name(&v, "Gun").id.clone();
        let cap = idx(&v, "Cap");
        v[cap].layer = 9;
        let gi = idx(&v, "Gun");
        v[gi].translation = [0.0, 3.0, 0.0];
        let old = turret();
        // New asset: Gun moves (the instance keeps its override), Cap is deleted, Barrel is new.
        let mut new = turret();
        new[1].scale = [2.0; 3];
        new.remove(2);
        new.push(entry("barrel", "Barrel", Some(1)));
        a.set("t.prefab", new);
        let old_assets = Assets::new(&[("t.prefab", old)]);
        remerge_prefabs(&mut v, &|p| old_assets.get(p), &|p| a.get(p));
        let g = by_name(&v, "Gun");
        assert_eq!(g.id, gun_id, "identity kept");
        assert_eq!(g.translation, [0.0, 3.0, 0.0], "override kept");
        assert_eq!(g.scale, [2.0; 3], "asset change arrives");
        assert!(v.iter().any(|e| e.name == "Barrel") && v.iter().all(|e| e.name != "Cap"));
        let link = v[0].prefab.as_ref().unwrap();
        assert!(link.overrides.iter().any(|o| o.target == "cap" && o.unused), "override on a removed target is kept as unused");
        // And it survives another save.
        compact_prefabs(&mut v, &|p| a.get(p));
        assert!(v[0].prefab.as_ref().unwrap().overrides.iter().any(|o| o.target == "cap"));
    }

    #[test]
    fn nested_ids_compose_and_match_the_nested_instance_alone() {
        let gunmount = vec![entry("mount", "Mount", None), entry("pin", "Pin", Some(0))];
        let mut outer_root = entry("hull", "Hull", None);
        outer_root.layer = 1;
        let mut nested = entry("slot", "Slot", Some(0));
        nested.prefab = Some(PrefabLink::new("mount.prefab"));
        let a = Assets::new(&[("mount.prefab", gunmount), ("ship.prefab", vec![outer_root, nested])]);
        let v = placed("ship.prefab", "I9", &a);
        assert_eq!(v.len(), 3);
        let slot = by_name(&v, "Slot");
        let pin = by_name(&v, "Pin");
        assert_eq!(slot.id, live_id("hull", "I9", "slot"));
        assert_eq!(pin.id, live_id("mount", &slot.id, "pin"), "same id as the nested instance would give it alone");
        assert_eq!(pin.parent, Some(idx(&v, "Slot")));
        // An edit to the nested object is an override on the outer instance.
        let mut w = v.clone();
        let p = idx(&w, "Pin");
        w[p].layer = 5;
        compact_prefabs(&mut w, &|p| a.get(p));
        assert_eq!(w.len(), 1);
        let o = &w[0].prefab.as_ref().unwrap().overrides;
        assert_eq!(o.len(), 1);
        assert_eq!(o[0].target, "slot/pin");
        expand_prefabs(&mut w, &|p| a.get(p));
        assert_eq!(by_name(&w, "Pin").layer, 5);
        assert_eq!(by_name(&w, "Pin").id, pin.id);
    }

    #[test]
    fn legacy_entries_without_links_are_left_alone() {
        let a = Assets::new(&[]);
        let mut v = turret();
        let before: Vec<String> = v.iter().map(|e| e.id.clone()).collect();
        expand_prefabs(&mut v, &|p| a.get(p));
        compact_prefabs(&mut v, &|p| a.get(p));
        assert_eq!(v.iter().map(|e| e.id.clone()).collect::<Vec<_>>(), before);
        // An old RON entry without the new fields still loads.
        let text = "(name:\"x\",source:Empty,material:Model,translation:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),scale:(1.0,1.0,1.0))";
        let e: SceneEntry = ron::from_str(text).unwrap();
        assert!(e.prefab.is_none() && e.prefab_parent.is_none());
    }

    #[test]
    fn missing_asset_keeps_the_stored_instance() {
        let a = Assets::new(&[]);
        let mut root = entry("I1", "Lost", None);
        let mut l = PrefabLink::new("gone.prefab");
        l.overrides.push(PrefabOverride { target: "x".into(), path: "layer".into(), value: "4".into(), unused: false });
        root.prefab = Some(l);
        let mut v = vec![root, entry("added", "Extra", Some(0))];
        expand_prefabs(&mut v, &|p| a.get(p));
        compact_prefabs(&mut v, &|p| a.get(p));
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].prefab.as_ref().unwrap().overrides.len(), 1);
    }

    #[test]
    fn revert_restores_the_asset_and_keeps_ids() {
        let a = Assets::new(&[("t.prefab", turret())]);
        let mut v = placed("t.prefab", "I1", &a);
        let gun_id = by_name(&v, "Gun").id.clone();
        let gi = idx(&v, "Gun");
        v[gi].translation = [9.0; 3];
        v[gi].layer = 4;
        v.push(entry("extra", "Extra", Some(0)));
        assert!(revert_prefab(&mut v, "I1", &PrefabScope::Property { target: "gun".into(), path: "layer".into() }, &|p| a.get(p)));
        assert_eq!(by_name(&v, "Gun").layer, 0);
        assert_eq!(by_name(&v, "Gun").translation, [9.0; 3], "other overrides stay");
        assert!(v.iter().any(|e| e.name == "Extra"));
        assert!(revert_prefab(&mut v, "I1", &PrefabScope::All, &|p| a.get(p)));
        assert_eq!(by_name(&v, "Gun").translation, [0.0, 1.0, 0.0]);
        assert_eq!(by_name(&v, "Gun").id, gun_id);
        assert!(v.iter().all(|e| e.name != "Extra"));
    }

    #[test]
    fn apply_writes_into_the_asset_and_other_instances_follow() {
        let a = Assets::new(&[("t.prefab", turret())]);
        let mut v = placed("t.prefab", "I1", &a);
        // Instance 1: change Gun's layer, add a Flag under Gun, delete Cap.
        let gi = idx(&v, "Gun");
        v[gi].layer = 6;
        let g_id = v[gi].id.clone();
        let mut flag = entry("flag-id", "Flag", Some(gi));
        flag.layer = 2;
        v.push(flag);
        let first_cap = v.iter().position(|e| e.name == "Cap").unwrap();
        v.remove(first_cap);
        let (path, raw) = apply_prefab(&mut v, "I1", &PrefabScope::All, &ApplyLevel::outermost(), &|p| a.get(p)).unwrap();
        assert_eq!(path, "t.prefab");
        let by = |n: &str| raw.iter().find(|e| e.name == n).unwrap_or_else(|| panic!("asset has no {n}"));
        assert_eq!(by("Gun").layer, 6);
        assert_eq!(by("Gun").id, "gun", "source ids stay");
        assert!(raw.iter().all(|e| e.name != "Cap"));
        assert_eq!(by("Flag").parent, Some(raw.iter().position(|e| e.name == "Gun").unwrap()));
        assert!(raw.iter().all(|e| e.prefab.is_none()));
        a.set(&path, raw);
        expand_prefabs(&mut v, &|p| a.get(p));
        // Instance 1 matches the asset exactly: nothing left to apply.
        let infos = compact_prefabs(&mut v.clone(), &|p| a.get(p));
        let i1 = infos.iter().find(|i| i.root == "I1").unwrap();
        assert!(i1.diffs.is_empty(), "{:?}", i1.diffs);
        assert_eq!(by_name(&v, "Gun").id, g_id, "the existing objects keep their ids");
        // Instance 2 (compacted with the old asset first) now has the change too.
        let mut v2 = placed("t.prefab", "I2", &Assets::new(&[("t.prefab", turret())]));
        compact_prefabs(&mut v2, &|p| Assets::new(&[("t.prefab", turret())]).get(p));
        expand_prefabs(&mut v2, &|p| a.get(p));
        assert_eq!(by_name(&v2, "Gun").layer, 6);
        assert!(v2.iter().any(|e| e.name == "Flag") && v2.iter().all(|e| e.name != "Cap"));
    }

    #[test]
    fn apply_to_a_single_property_only_moves_that_property() {
        let a = Assets::new(&[("t.prefab", turret())]);
        let mut v = placed("t.prefab", "I1", &a);
        let gi = idx(&v, "Gun");
        v[gi].layer = 6;
        v[gi].translation = [7.0, 0.0, 0.0];
        let (_, raw) = apply_prefab(&mut v, "I1", &PrefabScope::Property { target: "gun".into(), path: "layer".into() }, &ApplyLevel::outermost(), &|p| a.get(p)).unwrap();
        let gun = raw.iter().find(|e| e.name == "Gun").unwrap();
        assert_eq!(gun.layer, 6);
        assert_eq!(gun.translation, [0.0, 1.0, 0.0]);
        assert!(v[0].prefab.as_ref().unwrap().overrides.iter().any(|o| o.path == "translation"), "the translation override stays on the instance");
        assert!(!v[0].prefab.as_ref().unwrap().overrides.iter().any(|o| o.path == "layer"));
    }

    #[test]
    fn apply_to_a_nested_prefab_writes_the_inner_asset() {
        let mount = vec![entry("mount", "Mount", None), entry("pin", "Pin", Some(0))];
        let mut nested = entry("slot", "Slot", Some(0));
        nested.prefab = Some(PrefabLink::new("mount.prefab"));
        let a = Assets::new(&[("mount.prefab", mount), ("ship.prefab", vec![entry("hull", "Hull", None), nested])]);
        let mut v = placed("ship.prefab", "I9", &a);
        let p = idx(&v, "Pin");
        v[p].layer = 5;
        let level = ApplyLevel { nested: Some("slot".into()) };
        let (path, raw) = apply_prefab(&mut v, "I9", &PrefabScope::All, &level, &|p| a.get(p)).unwrap();
        assert_eq!(path, "mount.prefab");
        assert_eq!(raw.iter().find(|e| e.name == "Pin").unwrap().layer, 5);
        assert!(v[0].prefab.as_ref().unwrap().overrides.is_empty());
    }

    #[test]
    fn net_roles_are_overridable_properties_and_compact_back() {
        let a = Assets::new(&[("t.prefab", turret())]);
        // An instance whose Gun is Client-only with one component pinned to the server.
        let mut v = placed("t.prefab", "I1", &a);
        let gi = idx(&v, "Gun");
        v[gi].net_role = zestengine_core::NetRole::Client;
        v[gi].component_roles.insert("Light".into(), zestengine_core::NetRole::Both);
        compact_prefabs(&mut v, &|p| a.get(p));
        let link = v[0].prefab.as_ref().unwrap();
        let paths: Vec<&str> = link.overrides.iter().map(|o| o.path.as_str()).collect();
        assert!(paths.contains(&"net_role") && paths.contains(&"component_roles"), "{paths:?}");
        // Reloading rebuilds the roles from the override; the rest stays the asset's.
        expand_prefabs(&mut v, &|p| a.get(p));
        assert_eq!(by_name(&v, "Gun").net_role, zestengine_core::NetRole::Client);
        assert_eq!(by_name(&v, "Gun").component_roles.get("Light"), Some(&zestengine_core::NetRole::Both));
        assert_eq!(by_name(&v, "Turret").net_role, zestengine_core::NetRole::Both);
        // Reverting the property restores the asset's role.
        assert!(revert_prefab(&mut v, "I1", &PrefabScope::Property { target: "gun".into(), path: "net_role".into() }, &|p| a.get(p)));
        assert_eq!(by_name(&v, "Gun").net_role, zestengine_core::NetRole::Both);
    }

    #[test]
    fn variants_inherit_and_override() {
        let a = Assets::new(&[("t.prefab", turret())]);
        let mut vr = entry("var", "RedTurret", None);
        let mut l = PrefabLink::new("t.prefab");
        l.overrides.push(PrefabOverride { target: "cap".into(), path: "layer".into(), value: "8".into(), unused: false });
        vr.prefab = Some(l);
        a.set("red.prefab", vec![vr]);
        let v = placed("red.prefab", "I5", &a);
        assert_eq!(v.len(), 3);
        assert_eq!(by_name(&v, "Cap").layer, 8);
        assert_eq!(by_name(&v, "Gun").layer, 0);
    }

    #[test]
    fn a_mesh_swap_is_a_source_override_that_applies_and_reverts() {
        let mut gun = entry("gun", "Gun", Some(0));
        gun.source = ObjectSource::Model { path: "a.glb".into(), mesh: 0, extra_meshes: vec![], model: None, mesh_id: None, extra_mesh_ids: vec![] };
        let a = Assets::new(&[("t.prefab", vec![entry("base", "Base", None), gun])]);
        let mut v = placed("t.prefab", "I1", &a);
        let gi = idx(&v, "Gun");
        // Loading may have filled the id of the same mesh: not a change.
        if let ObjectSource::Model { mesh_id, .. } = &mut v[gi].source {
            *mesh_id = Some(crate::sub_asset::SubAssetId(7));
        }
        let infos = compact_prefabs(&mut v.clone(), &|p| a.get(p));
        assert!(infos[0].diffs.is_empty(), "same mesh with an id filled in is no override: {:?}", infos[0].diffs);
        // A different mesh is.
        if let ObjectSource::Model { mesh, mesh_id, .. } = &mut v[gi].source {
            *mesh = 3;
            *mesh_id = None;
        }
        compact_prefabs(&mut v, &|p| a.get(p));
        let link = v[0].prefab.clone().unwrap();
        assert!(link.overrides.iter().any(|o| o.target == "gun" && o.path == "source"), "{:?}", link.overrides);
        expand_prefabs(&mut v, &|p| a.get(p));
        assert!(matches!(&by_name(&v, "Gun").source, ObjectSource::Model { mesh: 3, .. }));
        // Apply writes it into the asset.
        let (_, raw) = apply_prefab(&mut v, "I1", &PrefabScope::All, &ApplyLevel::outermost(), &|p| a.get(p)).unwrap();
        assert!(matches!(&raw.iter().find(|e| e.name == "Gun").unwrap().source, ObjectSource::Model { mesh: 3, .. }));
        // Revert puts the asset's mesh back.
        assert!(revert_prefab(&mut v, "I1", &PrefabScope::All, &|p| a.get(p)));
        assert!(matches!(&by_name(&v, "Gun").source, ObjectSource::Model { mesh: 0, .. }));
    }

    #[test]
    fn reparenting_inside_an_instance_is_a_parent_override() {
        let a = Assets::new(&[("t.prefab", turret())]);
        let mut v = placed("t.prefab", "I1", &a);
        let (cap, base) = (idx(&v, "Cap"), 0);
        v[cap].parent = Some(base); // Cap moves from under Gun to under Base
        let mut w = v.clone();
        let infos = compact_prefabs(&mut w, &|p| a.get(p));
        assert_eq!(w.len(), 1, "still just the root");
        let o = &w[0].prefab.as_ref().unwrap().overrides;
        assert_eq!(o.len(), 1, "{o:?}");
        assert_eq!((o[0].target.as_str(), o[0].path.as_str(), o[0].value.as_str()), ("cap", "parent", "\"base\""));
        assert!(matches!(&infos[0].diffs[0].change, Change::Modified { .. }));
        expand_prefabs(&mut w, &|p| a.get(p));
        assert_eq!(by_name(&w, "Cap").parent, Some(0));
        // Deleting the old parent keeps the moved child.
        let gun = idx(&v, "Gun");
        v.remove(gun);
        let cap = idx(&v, "Cap");
        v[cap].parent = Some(0);
        compact_prefabs(&mut v, &|p| a.get(p));
        let link = v[0].prefab.clone().unwrap();
        assert_eq!(link.removed, vec!["gun".to_string()]);
        expand_prefabs(&mut v, &|p| a.get(p));
        assert!(v.iter().any(|e| e.name == "Cap") && v.iter().all(|e| e.name != "Gun"), "Cap survives its old parent");
        // Apply moves it in the asset; Revert undoes it.
        let (_, raw) = apply_prefab(&mut v, "I1", &PrefabScope::All, &ApplyLevel::outermost(), &|p| a.get(p)).unwrap();
        let cap_at = raw.iter().position(|e| e.name == "Cap").unwrap();
        assert_eq!(raw[cap_at].parent, raw.iter().position(|e| e.name == "Base"));
    }

    #[test]
    fn a_parent_that_would_make_a_loop_is_unused() {
        let a = Assets::new(&[("t.prefab", turret())]);
        let mut root = entry("I1", "Turret", None);
        let mut l = PrefabLink::new("t.prefab");
        l.overrides.push(PrefabOverride { target: "gun".into(), path: "parent".into(), value: "\"cap\"".into(), unused: false });
        root.prefab = Some(l);
        let mut v = vec![root];
        expand_prefabs(&mut v, &|p| a.get(p));
        assert_eq!(by_name(&v, "Gun").parent, Some(0), "unchanged");
        assert!(v[0].prefab.as_ref().unwrap().overrides[0].unused);
    }

    #[test]
    fn objects_move_between_instances_and_survive_save_and_load() {
        let a = Assets::new(&[("t.prefab", turret())]);
        let mut v = placed("t.prefab", "I1", &a);
        let mut second = placed("t.prefab", "I2", &a);
        let off = v.len();
        for e in &mut second {
            e.parent = e.parent.map(|p| p + off);
        }
        v.extend(second);
        // I1's Gun (with its Cap below) moves under I2's Base.
        let gun1 = v.iter().position(|e| e.name == "Gun").unwrap();
        let base2 = v.iter().position(|e| e.id == "I2").unwrap();
        let gun1_id = v[gun1].id.clone();
        let cap1_id = v.iter().find(|e| e.name == "Cap").unwrap().id.clone();
        v[gun1].parent = Some(base2);
        let mut flag = entry("flag", "Flag", Some(gun1));
        flag.layer = 4;
        v.push(flag);
        let mut saved = v.clone();
        compact_prefabs(&mut saved, &|p| a.get(p));
        let gun_ref = saved.iter().find(|e| e.id == "I1").unwrap().prefab.clone().unwrap();
        assert!(gun_ref.overrides.iter().any(|o| o.target == "gun" && o.path == "parent" && o.value.contains("ext:I2")), "{:?}", gun_ref.overrides);
        assert!(saved.iter().any(|e| e.name == "Flag" && e.prefab_parent.as_deref() == Some("gun")), "the Flag stays with the moved Gun");
        assert!(saved.iter().all(|e| e.name != "Gun"), "members are not stored");
        // Through RON and back: either instance may expand first.
        let text = ron::to_string(&saved).unwrap();
        for flip in [false, true] {
            let mut back: Vec<SceneEntry> = ron::from_str(&text).unwrap();
            if flip {
                back.reverse();
                for e in &mut back {
                    e.parent = e.parent.map(|p| saved.len() - 1 - p);
                }
            }
            expand_prefabs(&mut back, &|p| a.get(p));
            let g = back.iter().find(|e| e.id == gun1_id).expect("gun");
            assert_eq!(back[g.parent.unwrap()].id, "I2", "Gun now hangs under the second instance (flip {flip})");
            assert_eq!(back[back.iter().find(|e| e.id == cap1_id).unwrap().parent.unwrap()].id, gun1_id);
            let f = back.iter().find(|e| e.name == "Flag").unwrap();
            assert_eq!(back[f.parent.unwrap()].id, gun1_id);
            assert!(back.iter().filter(|e| e.name == "Gun").count() == 2);
        }
        // Reverting the move puts Gun back under its own instance.
        assert!(revert_prefab(&mut v, "I1", &PrefabScope::Property { target: "gun".into(), path: "parent".into() }, &|p| a.get(p)));
        let g = v.iter().find(|e| e.id == gun1_id).unwrap();
        assert_eq!(v[g.parent.unwrap()].id, "I1");
    }

    #[test]
    fn a_parent_outside_the_instance_may_be_a_plain_object_or_the_scene_root() {
        let a = Assets::new(&[("t.prefab", turret())]);
        let mut v = placed("t.prefab", "I1", &a);
        v.push(entry("plain-id", "Plain", None));
        let plain = v.len() - 1;
        let gi = idx(&v, "Gun");
        v[gi].parent = Some(plain);
        let ci = idx(&v, "Cap");
        v[ci].parent = None;
        compact_prefabs(&mut v, &|p| a.get(p));
        assert_eq!(v.iter().filter(|e| e.name != "Plain").count(), 1, "only the root remains beside Plain");
        expand_prefabs(&mut v, &|p| a.get(p));
        let (plain, gun, cap) = (idx(&v, "Plain"), idx(&v, "Gun"), idx(&v, "Cap"));
        assert_eq!(v[gun].parent, Some(plain));
        assert_eq!(v[cap].parent, None);
    }
    // ---- renderer slots

    fn mesh_asset() -> Vec<SceneEntry> {
        let mut body = entry("body", "Body", None);
        body.source = ObjectSource::Model { path: "m.glb".into(), mesh: 0, extra_meshes: vec![1, 2], model: None, mesh_id: None, extra_mesh_ids: vec![] };
        body.components.push(ComponentData::new(
            zestengine_core::MESH_RENDERER,
            "(cast_shadows:On,lightmap_scale:1.0,materials:[Model,Model,Model])",
        ));
        body.renderer_split = true;
        vec![body]
    }

    fn renderer_data(e: &SceneEntry) -> zestengine_core::MeshRenderer {
        ron::from_str(&e.components.iter().find(|c| zestengine_core::is_renderer_kind(&c.kind)).unwrap().data).unwrap()
    }

    fn set_slot(e: &mut SceneEntry, k: usize, slot: zestengine_core::MaterialSlot) {
        let mut r = renderer_data(e);
        zestengine_core::RendererData::set_slot(&mut r, k, slot);
        e.components.iter_mut().find(|c| zestengine_core::is_renderer_kind(&c.kind)).unwrap().data = ron::to_string(&r).unwrap();
    }

    #[test]
    fn a_material_slot_override_is_per_slot_and_round_trips() {
        use zestengine_core::MaterialSlot as S;
        let a = Assets::new(&[("m.prefab", mesh_asset())]);
        let mut v = placed("m.prefab", "I1", &a);
        set_slot(&mut v[0], 1, S::File("red.material".into()));
        compact_prefabs(&mut v, &|p| a.get(p));
        let link = v[0].prefab.clone().unwrap();
        let paths: Vec<&str> = link.overrides.iter().map(|o| o.path.as_str()).collect();
        assert_eq!(paths, vec!["components[Mesh Renderer].materials[1]"], "one slot, not the whole list");
        assert_eq!(link.overrides[0].value, "File(\"red.material\")");
        let mut back: Vec<SceneEntry> = ron::from_str(&ron::to_string(&v).unwrap()).unwrap();
        expand_prefabs(&mut back, &|p| a.get(p));
        let r = renderer_data(&back[0]);
        assert_eq!(r.materials, vec![S::Model, S::File("red.material".into()), S::Model]);
        assert!(!link.overrides.iter().any(|o| o.unused));
    }

    #[test]
    fn variants_overriding_other_slots_merge_with_a_base_edit() {
        use zestengine_core::MaterialSlot as S;
        let a = Assets::new(&[("m.prefab", mesh_asset())]);
        let mut v = placed("m.prefab", "I1", &a);
        set_slot(&mut v[0], 0, S::File("skin.material".into()));
        compact_prefabs(&mut v, &|p| a.get(p));
        // The asset later changes a DIFFERENT slot; the instance keeps its own and gains the edit.
        let mut asset = mesh_asset();
        set_slot(&mut asset[0], 2, S::File("hair.material".into()));
        let new = Assets::new(&[("m.prefab", asset)]);
        remerge_prefabs(&mut v, &|p| a.get(p), &|p| new.get(p));
        expand_prefabs(&mut v, &|p| new.get(p));
        let r = renderer_data(&v[0]);
        assert_eq!(r.materials, vec![S::File("skin.material".into()), S::Model, S::File("hair.material".into())]);
    }

    #[test]
    fn revert_and_apply_work_on_one_slot() {
        use zestengine_core::MaterialSlot as S;
        let a = Assets::new(&[("m.prefab", mesh_asset())]);
        let mut v = placed("m.prefab", "I1", &a);
        set_slot(&mut v[0], 1, S::File("red.material".into()));
        set_slot(&mut v[0], 2, S::File("blue.material".into()));
        // Apply only slot 1 to the asset.
        let scope = PrefabScope::Property { target: "body".into(), path: "components[Mesh Renderer].materials[1]".into() };
        let (path, raw) = apply_prefab(&mut v, "I1", &scope, &ApplyLevel::outermost(), &|p| a.get(p)).unwrap();
        assert_eq!(path, "m.prefab");
        assert_eq!(renderer_data(&raw[0]).materials, vec![S::Model, S::File("red.material".into()), S::Model], "only the chosen slot moved");
        a.set(&path, raw);
        expand_prefabs(&mut v, &|p| a.get(p));
        // The other slot is still an override on the instance.
        assert_eq!(renderer_data(&v[0]).materials, vec![S::Model, S::File("red.material".into()), S::File("blue.material".into())]);
        let left = v[0].prefab.clone().unwrap().overrides;
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].path, "components[Mesh Renderer].materials[2]");
        assert!(revert_prefab(&mut v, "I1", &PrefabScope::All, &|p| a.get(p)));
        assert_eq!(renderer_data(&v[0]).materials, vec![S::Model, S::File("red.material".into()), S::Model]);
    }

    #[test]
    fn overrides_written_against_the_old_entry_fields_still_apply() {
        use zestengine_core::MaterialSlot as S;
        let a = Assets::new(&[("m.prefab", mesh_asset())]);
        let mut v = placed("m.prefab", "I1", &a);
        v.truncate(1);
        let mut link = PrefabLink::new("m.prefab");
        link.overrides = vec![
            PrefabOverride { target: "body".into(), path: "material".into(), value: "File(\"red.material\")".into(), unused: false },
            PrefabOverride { target: "body".into(), path: "cast_shadows".into(), value: "Off".into(), unused: false },
        ];
        v[0].prefab = Some(link);
        expand_prefabs(&mut v, &|p| a.get(p));
        let r = renderer_data(&v[0]);
        assert_eq!(r.materials[0], S::File("red.material".into()));
        assert_eq!(r.cast_shadows, zestengine_core::ShadowCasting::Off);
        assert!(!v[0].prefab.as_ref().unwrap().overrides.iter().any(|o| o.unused));
    }
}
