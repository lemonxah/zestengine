//! Content roles at load and cook time (NET-3): drop the objects and components
//! that do not belong to a side of a networked game, and lint what the rest
//! still points at.
//!
//! [`strip_entries`] runs on scene entries (a loaded scene file, a prefab spawn,
//! the server cook). Rules:
//!
//! - an object whose [`NetRole`] misses the side is removed with its whole
//!   subtree (a child cannot be more permissive than a missing parent);
//! - a component is removed when its effective role misses the side: the
//!   object's per-component override (`component_roles`, key `Kind` or `Kind#1`)
//!   if there is one, else the type default from the registry;
//! - side `Both` (listen host, offline, editor) keeps everything.
//!
//! Kept components may still hold object ids of removed objects (a trigger
//! pointing at a client-only effect). Those resolve to nothing at run time;
//! [`StripReport::dangling`] lists every such reference once so the loader and
//! the build log can name both objects.

use std::collections::{HashMap, HashSet};

use zestengine_core::{ComponentRegistry, NetRole};

use crate::package::{normalize_rel, visit_ron_strings};
use crate::prefab_link::component_keys;
use crate::scene_file::SceneEntry;

/// A kept component pointing at content that does not exist on this side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DanglingRef {
    /// Hierarchy path of the object holding the reference (`Root/Child`).
    pub holder: String,
    pub holder_id: String,
    /// Registry name of the component holding it.
    pub component: String,
    /// What it points at: an object path, or a project-relative asset path.
    pub target: String,
    pub kind: DanglingKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DanglingKind {
    /// An object id whose object was stripped.
    Object,
    /// An asset the server build does not ship (texture, sound, shader ...).
    Asset,
}

impl DanglingRef {
    /// One log / report line.
    pub fn describe(&self) -> String {
        match self.kind {
            DanglingKind::Object => format!("{} [{}] references stripped object {}", self.holder, self.component, self.target),
            DanglingKind::Asset => format!("{} [{}] references client-only asset {}", self.holder, self.component, self.target),
        }
    }
}

/// What a strip removed.
#[derive(Clone, Debug, Default)]
pub struct StripReport {
    /// Removed objects (path, role of the object or of its first stripped ancestor).
    pub objects: Vec<(String, NetRole)>,
    /// Removed components: object path and registry name.
    pub components: Vec<(String, String)>,
    pub dangling: Vec<DanglingRef>,
}

impl StripReport {
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty() && self.components.is_empty()
    }
}

/// Hierarchy path of entry `i` (names joined by `/`).
pub fn entry_path(entries: &[SceneEntry], i: usize) -> String {
    let mut parts = vec![entries[i].name.as_str()];
    let mut cur = entries[i].parent;
    let mut guard = 0;
    while let Some(p) = cur.filter(|&p| p < entries.len() && guard < 64) {
        parts.push(entries[p].name.as_str());
        cur = entries[p].parent;
        guard += 1;
    }
    parts.reverse();
    parts.join("/")
}

/// The effective role of each of `e`'s components: the explicit override, else the type default.
pub fn effective_component_roles(e: &SceneEntry, registry: &ComponentRegistry) -> Vec<NetRole> {
    component_keys(e)
        .iter()
        .zip(&e.components)
        .map(|(key, c)| e.component_roles.get(key).copied().unwrap_or_else(|| registry.default_role(&c.kind)))
        .collect()
}

/// UUID-shaped tokens of `text` (object ids inside component data).
fn id_tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !(c.is_ascii_hexdigit() || c == '-')).filter(|t| t.len() == 36 && t.as_bytes()[8] == b'-')
}

/// Remove everything that does not exist on `side` from `entries` and fix the
/// parent indices. Returns what was removed and which kept components still
/// reference removed objects.
pub fn strip_entries(entries: &mut Vec<SceneEntry>, side: NetRole, registry: &ComponentRegistry) -> StripReport {
    let mut report = StripReport::default();
    if side == NetRole::Both || entries.is_empty() {
        return report;
    }
    // Removed flags with inheritance down the hierarchy (memoised; cycle-safe).
    let n = entries.len();
    let mut removed: Vec<Option<bool>> = vec![None; n];
    fn resolve(i: usize, entries: &[SceneEntry], side: NetRole, memo: &mut [Option<bool>], depth: usize) -> bool {
        if let Some(v) = memo[i] {
            return v;
        }
        let own = !entries[i].net_role.kept_on(side);
        let v = own || (depth < 256 && entries[i].parent.is_some_and(|p| p < entries.len() && p != i && resolve(p, entries, side, memo, depth + 1)));
        memo[i] = Some(v);
        v
    }
    for i in 0..n {
        resolve(i, entries, side, &mut removed, 0);
    }
    let removed: Vec<bool> = removed.into_iter().map(|v| v.unwrap_or(false)).collect();
    if !removed.iter().any(|&r| r) && !entries.iter().any(|e| !e.components.is_empty()) {
        return report;
    }

    let removed_ids: HashMap<&str, usize> = (0..n).filter(|&i| removed[i] && !entries[i].id.is_empty()).map(|i| (entries[i].id.as_str(), i)).collect();
    // Components first (indices still valid).
    let mut drop_components: Vec<Vec<bool>> = Vec::with_capacity(n);
    let mut dangling = Vec::new();
    for i in 0..n {
        let e = &entries[i];
        if removed[i] {
            drop_components.push(Vec::new());
            continue;
        }
        let roles = effective_component_roles(e, registry);
        let mut drops = Vec::with_capacity(roles.len());
        for (c, role) in e.components.iter().zip(&roles) {
            let drop = !role.kept_on(side);
            drops.push(drop);
            if drop {
                report.components.push((entry_path(entries, i), c.kind.clone()));
            } else if !removed_ids.is_empty() {
                let mut seen = HashSet::new();
                for tok in id_tokens(&c.data) {
                    if let Some(&t) = removed_ids.get(tok)
                        && seen.insert(t)
                    {
                        dangling.push(DanglingRef {
                            holder: entry_path(entries, i),
                            holder_id: e.id.clone(),
                            component: c.kind.clone(),
                            target: entry_path(entries, t),
                            kind: DanglingKind::Object,
                        });
                    }
                }
            }
        }
        drop_components.push(drops);
    }
    // Report branch roots only: the first removed object on each branch.
    for i in 0..n {
        if removed[i] && entries[i].parent.is_none_or(|p| p >= n || !removed[p]) {
            report.objects.push((entry_path(entries, i), entries[i].net_role));
        }
    }
    report.dangling = dangling;

    // Rebuild the list with remapped parents.
    let mut remap = vec![usize::MAX; n];
    let mut next = 0;
    for i in 0..n {
        if !removed[i] {
            remap[i] = next;
            next += 1;
        }
    }
    let old = std::mem::take(entries);
    for (i, mut e) in old.into_iter().enumerate() {
        if removed[i] {
            continue;
        }
        e.parent = e.parent.and_then(|p| remap.get(p).copied().filter(|&m| m != usize::MAX));
        let keys = component_keys(&e);
        let mut k = 0;
        let drops = &drop_components[i];
        let mut dropped_keys = Vec::new();
        e.components.retain(|_| {
            let d = drops.get(k).copied().unwrap_or(false);
            if d {
                dropped_keys.push(keys[k].clone());
            }
            k += 1;
            !d
        });
        // Keys of the survivors can shift (`Kind#1` becomes `Kind`); re-key the overrides.
        if !e.component_roles.is_empty() {
            let before: Vec<String> = keys.iter().filter(|k| !dropped_keys.contains(k)).cloned().collect();
            let after = component_keys(&e);
            let mut moved = std::collections::BTreeMap::new();
            for (old_key, new_key) in before.iter().zip(&after) {
                if let Some(r) = e.component_roles.get(old_key) {
                    moved.insert(new_key.clone(), *r);
                }
            }
            e.component_roles = moved;
        }
        entries.push(e);
    }
    report
}

/// Extensions of files a dedicated server never loads (the server cook drops
/// them, the extraction of a content package skips them).
const CLIENT_ONLY_EXT: &[&str] = &[
    // Images, skyboxes, sprite sheets.
    "png", "jpg", "jpeg", "tga", "bmp", "gif", "webp", "exr", "hdr", "dds", "ktx", "ktx2", "psd", "tif", "tiff", "astc",
    // Sound.
    "wav", "ogg", "mp3", "flac", "opus", "aac", "m4a", "aif", "aiff",
    // Shaders and materials.
    "frag", "vert", "comp", "glsl", "hlsl", "wgsl", "spv", "material", "shadergraph", "matcap",
    // Fonts, video, baked lighting.
    "ttf", "otf", "woff", "woff2", "mp4", "webm", "lightmap", "lightdata", "occlusion", "reflprobe", "audiobake", "lmuv", "postfx",
    // Sprite atlas products.
    "atlas",
];

/// Whether a project-relative file is dropped from a server build / extraction.
/// Models are not dropped: the cook rewrites them to collision-only blobs.
pub fn is_client_only_file(rel: &str) -> bool {
    let lower = rel.to_ascii_lowercase();
    if lower.split('/').any(|p| matches!(p, ".zestengine_geocache" | ".zestengine_texcache" | ".zestengine_atlascache" | ".zestengine_modelcache")) {
        return true;
    }
    // `x.png.meta` etc.: the sidecar of a dropped file goes with it.
    let base = lower.strip_suffix(".meta").unwrap_or(&lower);
    if base.len() != lower.len() && base.rsplit_once('.').is_some_and(|(_, e)| CLIENT_ONLY_EXT.contains(&e)) {
        return true;
    }
    lower.rsplit_once('.').is_some_and(|(_, e)| CLIENT_ONLY_EXT.contains(&e))
}

/// Kept components that name a file a server build drops (a gameplay script
/// pointing at a sound or a texture). `entries` are the entries that remain
/// after [`strip_entries`].
pub fn lint_asset_refs(entries: &[SceneEntry]) -> Vec<DanglingRef> {
    let mut out = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        for c in &e.components {
            let mut seen = HashSet::new();
            visit_ron_strings(&c.data, &mut |s| {
                if let Some(rel) = normalize_rel(s)
                    && is_client_only_file(&rel)
                    && seen.insert(rel.clone())
                {
                    out.push(DanglingRef {
                        holder: entry_path(entries, i),
                        holder_id: e.id.clone(),
                        component: c.kind.clone(),
                        target: rel,
                        kind: DanglingKind::Asset,
                    });
                }
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_file::{ComponentData, MaterialRef, ObjectSource};

    fn entry(name: &str, parent: Option<usize>, role: NetRole) -> SceneEntry {
        SceneEntry {
            id: zestengine_core::ObjectId::new().to_string(),
            name: name.into(),
            source: ObjectSource::Empty,
            enabled: true,
            static_geometry: false,
            lightmap_scale: 1.0,
            cast_shadows: Default::default(),
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
            net_role: role,
            component_roles: Default::default(),
        }
    }

    fn names(v: &[SceneEntry]) -> Vec<&str> {
        v.iter().map(|e| e.name.as_str()).collect()
    }

    #[test]
    fn subtree_follows_a_stripped_parent_even_if_the_child_is_both() {
        let reg = ComponentRegistry::with_builtins();
        let mk = || {
            vec![
                entry("World", None, NetRole::Both),
                entry("Fx", Some(0), NetRole::Client),
                entry("FxChild", Some(1), NetRole::Both),
                entry("FxServerChild", Some(1), NetRole::Server),
                entry("Spawner", Some(0), NetRole::Server),
                entry("Floor", Some(0), NetRole::Both),
            ]
        };
        let mut v = mk();
        let r = strip_entries(&mut v, NetRole::Server, &reg);
        assert_eq!(names(&v), ["World", "Spawner", "Floor"]);
        assert_eq!(v[2].parent, Some(0), "parents are remapped");
        assert_eq!(r.objects.len(), 1, "{:?}", r.objects); // Fx and its two kids: one branch
        let mut v = mk();
        strip_entries(&mut v, NetRole::Client, &reg);
        assert_eq!(names(&v), ["World", "Fx", "FxChild", "Floor"]);
        assert_eq!(v[2].parent, Some(1));
        let mut v = mk();
        assert!(strip_entries(&mut v, NetRole::Both, &reg).is_empty());
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn component_roles_default_override_and_rekey() {
        let reg = ComponentRegistry::with_builtins();
        let mut e = entry("Lamp", None, NetRole::Both);
        e.components = vec![
            ComponentData::new("Box Collider", "()"),
            ComponentData::new("Audio Source", "()"),
            ComponentData::new("Light", "()"),
            ComponentData::new("Light", "()"),
        ];
        // The second Light is kept on the server by an explicit override; the first Light follows the default.
        e.component_roles.insert("Light#1".into(), NetRole::Both);
        assert_eq!(effective_component_roles(&e, &reg), [NetRole::Both, NetRole::Client, NetRole::Client, NetRole::Both]);
        let mut v = vec![e.clone()];
        let r = strip_entries(&mut v, NetRole::Server, &reg);
        let kinds: Vec<_> = v[0].components.iter().map(|c| c.kind.as_str()).collect();
        assert_eq!(kinds, ["Box Collider", "Light"]);
        assert_eq!(v[0].component_roles.get("Light"), Some(&NetRole::Both), "the override follows its component");
        assert_eq!(r.components.len(), 2);
        // A Server-only component goes from clients.
        let mut e2 = entry("Score", None, NetRole::Both);
        e2.components = vec![ComponentData::new("Box Collider", "()")];
        e2.component_roles.insert("Box Collider".into(), NetRole::Server);
        let mut v = vec![e2];
        strip_entries(&mut v, NetRole::Client, &reg);
        assert!(v[0].components.is_empty());
    }

    #[test]
    fn dangling_references_to_stripped_objects_are_listed_once() {
        let reg = ComponentRegistry::with_builtins();
        let fx = entry("Smoke", Some(0), NetRole::Client);
        let mut zone = entry("Zone", Some(0), NetRole::Both);
        zone.components = vec![ComponentData::new("Trigger Events", format!("(target: \"{}\", again: \"{}\")", fx.id, fx.id))];
        let mut v = vec![entry("Root", None, NetRole::Both), fx, zone];
        let r = strip_entries(&mut v, NetRole::Server, &reg);
        assert_eq!(r.dangling.len(), 1, "{:?}", r.dangling);
        assert_eq!(r.dangling[0].holder, "Root/Zone");
        assert_eq!(r.dangling[0].target, "Root/Smoke");
        assert!(r.dangling[0].describe().contains("Root/Zone") && r.dangling[0].describe().contains("Root/Smoke"));
    }

    #[test]
    fn asset_lint_and_file_classes() {
        let mut e = entry("Gun", None, NetRole::Both);
        e.components = vec![ComponentData::new("Script", "(fire_sound: \"audio/shot.wav\", muzzle: \"textures/flash.png\", data: \"data/loot.ron\")")];
        let l = lint_asset_refs(&[e]);
        let targets: Vec<_> = l.iter().map(|d| d.target.as_str()).collect();
        assert_eq!(targets, ["audio/shot.wav", "textures/flash.png"]);
        for f in ["textures/a.png", "audio/a.ogg", "shaders/x.frag", "shaders/x.frag.spv", "materials/m.material", "fonts/f.ttf", "scenes/a.lightmap", "textures/a.png.meta", "models/.zestengine_geocache/x.bin"] {
            assert!(is_client_only_file(f), "{f}");
        }
        for f in ["scenes/a.scene", "models/a.glb", "models/a.glb.meta", "prefabs/p.prefab", "data/x.ron", "navmesh/a.navmesh", "scripts/a.wasm", "animations/a.anim"] {
            assert!(!is_client_only_file(f), "{f}");
        }
    }
}
