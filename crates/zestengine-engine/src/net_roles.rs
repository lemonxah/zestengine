//! Editor-side helpers for content roles (NET-3): per-component role overrides on
//! scene objects and the "what a dedicated server would lose" lint shown in the
//! object inspector. The load-time strip itself is `zestengine_assets::net_strip`.

use std::collections::{BTreeMap, HashSet};

use zestengine_assets::net_strip::is_client_only_file;
use zestengine_core::{Component, ComponentRegistry, NetRole};

use crate::scene::SceneObject;

/// Keys that tell apart components of one kind (`Kind`, `Kind#1`), as saved scenes
/// and prefab overrides name them.
pub fn component_key_list(components: &[Box<dyn Component>]) -> Vec<String> {
    let mut seen: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    components
        .iter()
        .map(|c| {
            let n = seen.entry(c.type_name()).or_insert(0);
            *n += 1;
            if *n == 1 { c.type_name().to_owned() } else { format!("{}#{}", c.type_name(), *n - 1) }
        })
        .collect()
}

/// Per component: the explicit override (if any) and the type default.
pub fn component_role_views(obj: &SceneObject, registry: &ComponentRegistry) -> Vec<(Option<NetRole>, NetRole)> {
    let keys = component_key_list(&obj.components);
    obj.components
        .iter()
        .zip(&keys)
        .map(|(c, key)| (obj.component_roles.get(key).copied(), registry.default_role(c.type_name())))
        .collect()
}

/// Set (`Some`) or clear (`None` = the type's default) the role of component `index`.
pub fn set_component_role(obj: &mut SceneObject, index: usize, role: Option<NetRole>) {
    let keys = component_key_list(&obj.components);
    let Some(key) = keys.get(index) else { return };
    match role {
        Some(r) => {
            obj.component_roles.insert(key.clone(), r);
        }
        None => {
            obj.component_roles.remove(key);
        }
    }
}

/// Remove component `index`, keeping the other components' role overrides attached.
pub fn remove_component(obj: &mut SceneObject, index: usize) {
    let before = component_key_list(&obj.components);
    if index >= before.len() {
        return;
    }
    obj.components.remove(index);
    let after = component_key_list(&obj.components);
    let survivors = before.iter().enumerate().filter(|(i, _)| *i != index).map(|(_, k)| k);
    let mut moved = BTreeMap::new();
    for (old, new) in survivors.zip(&after) {
        if let Some(r) = obj.component_roles.get(old) {
            moved.insert(new.clone(), *r);
        }
    }
    obj.component_roles = moved;
}

/// Set the project-wide role of component type `name` (NET-3). Choosing the type's own default
/// removes the entry (an override that changes nothing is noise in `project.zestengine`);
/// `None` removes it too. Returns whether the map changed.
pub fn set_type_override(overrides: &mut BTreeMap<String, NetRole>, registry: &ComponentRegistry, name: &str, role: Option<NetRole>) -> bool {
    let default = registry.type_role(name);
    match role.filter(|r| *r != default) {
        Some(r) => overrides.insert(name.to_owned(), r) != Some(r),
        None => overrides.remove(name).is_some(),
    }
}

/// Every registered component type with its own default role and the project override, for the
/// Project Settings list: overridden types first, then the rest by name.
pub fn type_role_rows(registry: &ComponentRegistry, overrides: &BTreeMap<String, NetRole>) -> Vec<(String, NetRole, Option<NetRole>)> {
    let mut rows: Vec<(String, NetRole, Option<NetRole>)> =
        registry.names().map(|n| (n.to_owned(), registry.type_role(n), overrides.get(n).copied())).collect();
    // Overrides for types no longer registered (a plugin that is not loaded) stay visible.
    for (n, r) in overrides {
        if !rows.iter().any(|(x, _, _)| x == n) {
            rows.push((n.clone(), NetRole::Both, Some(*r)));
        }
    }
    rows.sort_by(|a, b| b.2.is_some().cmp(&a.2.is_some()).then_with(|| a.0.cmp(&b.0)));
    rows
}

/// Whether object `i` is removed from a process playing `side`: its own role, or an ancestor's.
pub fn stripped_on(objects: &[SceneObject], i: usize, side: NetRole) -> bool {
    let (mut cur, mut guard) = (Some(i), 0);
    while let Some(c) = cur.filter(|&c| c < objects.len() && guard < 128) {
        if !objects[c].net_role.kept_on(side) {
            return true;
        }
        cur = objects[c].parent;
        guard += 1;
    }
    false
}

/// What would break for the object at `index` on a dedicated server: its server-kept
/// components that point at objects the server strips, or at files a server build drops.
pub fn server_view_warnings(objects: &[SceneObject], registry: &ComponentRegistry, index: usize) -> Vec<String> {
    let mut out = Vec::new();
    let Some(obj) = objects.get(index) else { return out };
    if stripped_on(objects, index, NetRole::Server) {
        return out;
    }
    let removed: HashSet<String> = if objects.iter().any(|o| !o.net_role.is_both()) {
        (0..objects.len()).filter(|&i| stripped_on(objects, i, NetRole::Server)).map(|i| objects[i].id.to_string()).collect()
    } else {
        HashSet::new()
    };
    let keys = component_key_list(&obj.components);
    for (c, key) in obj.components.iter().zip(&keys) {
        let role = obj.component_roles.get(key).copied().unwrap_or_else(|| registry.default_role(c.type_name()));
        if !role.kept_on(NetRole::Server) {
            continue;
        }
        let data = c.save();
        // Reuse the cook's lints on a one-component entry.
        let mut entry_like = Vec::new();
        if !removed.is_empty() {
            for tok in data.split(|ch: char| !(ch.is_ascii_hexdigit() || ch == '-')).filter(|t| t.len() == 36 && removed.contains(*t)) {
                if let Some(t) = objects.iter().find(|o| o.id.to_string() == tok) {
                    entry_like.push(format!("{} references {}, which a dedicated server strips", c.type_name(), t.name));
                }
            }
        }
        zestengine_assets::package::visit_ron_strings(&data, &mut |s| {
            if let Some(rel) = zestengine_assets::package::normalize_rel(s)
                && is_client_only_file(&rel)
            {
                entry_like.push(format!("{} references {rel}, which a dedicated server build does not ship", c.type_name()));
            }
        });
        entry_like.sort();
        entry_like.dedup();
        out.extend(entry_like);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_overrides_move_a_type_and_drop_when_they_match_the_default() {
        let mut reg = ComponentRegistry::with_builtins();
        let mut map = BTreeMap::new();
        // "Audio Source" is Client by default; make it kept everywhere, then back.
        assert_eq!(reg.default_role("Audio Source"), NetRole::Client);
        assert!(set_type_override(&mut map, &reg, "Audio Source", Some(NetRole::Both)));
        assert!(!set_type_override(&mut map, &reg, "Audio Source", Some(NetRole::Both)), "no change the second time");
        reg.set_role_overrides(map.clone());
        assert_eq!(reg.default_role("Audio Source"), NetRole::Both);
        assert_eq!(reg.type_role("Audio Source"), NetRole::Client, "the type's own default is still known");
        assert!(set_type_override(&mut map, &reg, "Audio Source", Some(NetRole::Client)), "the default removes the entry");
        assert!(map.is_empty());
        // A server-only gameplay type.
        assert!(set_type_override(&mut map, &reg, "Rigid Body", Some(NetRole::Server)));
        assert!(set_type_override(&mut map, &reg, "Rigid Body", None) && map.is_empty());
        // The listing: overridden types first, an unknown (unloaded plugin) override stays visible.
        map.insert("Zebra Plugin Thing".into(), NetRole::Server);
        map.insert("Wind Zone".into(), NetRole::Both);
        let rows = type_role_rows(&reg, &map);
        assert_eq!(rows[0].0, "Wind Zone");
        assert_eq!(rows[1].0, "Zebra Plugin Thing");
        assert_eq!(rows[0].1, NetRole::Client);
        assert!(rows.iter().filter(|r| r.2.is_none()).all(|r| r.0.as_str() != "Wind Zone"));
    }

    /// The project override list flows through the registry into the load-time strip and the
    /// server cook: a type moved to `Server` disappears from a client and stays on the server.
    #[test]
    fn project_overrides_decide_what_the_strip_removes() {
        use zestengine_assets::{ComponentData, ProjectFile};
        let mut project = ProjectFile::default();
        project.net_role_overrides.insert("Audio Source".into(), NetRole::Both);
        project.net_role_overrides.insert("Rigid Body".into(), NetRole::Server);
        let text = ron::to_string(&project).unwrap();
        let back: ProjectFile = ron::from_str(&text).unwrap();
        assert_eq!(back.net_role_overrides, project.net_role_overrides, "saved in project.zestengine");
        let mut reg = ComponentRegistry::with_builtins();
        back.apply_role_overrides(&mut reg);
        let _ = ComponentData::new("x", "()");
        assert_eq!(reg.default_role("Audio Source"), NetRole::Both);
        assert_eq!(reg.default_role("Rigid Body"), NetRole::Server);
        assert_eq!(reg.default_role("Light"), NetRole::Client, "untouched types keep their default");
        // An old project file without the list loads and writes nothing for it.
        let old = ron::to_string(&ProjectFile::default()).unwrap();
        assert!(!old.contains("net_role_overrides"));
        let _: ProjectFile = ron::from_str(&old).unwrap();
    }
}
