//! Legacy renderer fields -> `Mesh Renderer` component (see `docs/design/mesh-renderer-split.md`).
//!
//! Scenes and prefabs saved before the split keep a renderer's settings as loose fields of the
//! entry (`material`, `extra_materials`, `cast_shadows`, `lightmap_scale`). Migration moves them
//! into a `Mesh Renderer` component and rewrites prefab overrides that name those fields. It is
//! pure CPU, idempotent and gated by `format_version` so a renderer that was deliberately
//! removed from a mesh object of a current file is not brought back.

use zestengine_core::{MESH_RENDERER, MaterialSlot, MeshRenderer, ShadowCasting, SKINNED_MESH_RENDERER, is_renderer_kind};

use crate::prefab_link::PrefabLink;
use crate::scene_file::{ComponentData, MaterialRef, ObjectSource, SceneEntry, SceneFile};

/// `format_version` written by the renderer split: mesh objects carry a renderer component.
pub const RENDERER_FORMAT: u32 = 2;

/// Number of material slots an object with this source has: one per submesh.
pub fn slot_count_of(source: &ObjectSource) -> usize {
    match source {
        ObjectSource::Model { extra_meshes, .. } => 1 + extra_meshes.len(),
        ObjectSource::Builtin { .. } | ObjectSource::Primitive { .. } => 1,
        ObjectSource::Empty | ObjectSource::Camera | ObjectSource::Light => 0,
    }
}

/// The renderer slot for a saved material reference.
pub fn material_slot_of(m: &MaterialRef) -> MaterialSlot {
    match m {
        MaterialRef::Model => MaterialSlot::Model,
        MaterialRef::File(p) => MaterialSlot::File(p.clone()),
        MaterialRef::Inline { .. } => MaterialSlot::Inline(ron::to_string(m).unwrap_or_default()),
    }
}

/// The saved material reference for a renderer slot (an undecodable `Inline` inherits).
pub fn material_ref_of(s: &MaterialSlot) -> MaterialRef {
    match s {
        MaterialSlot::Model => MaterialRef::Model,
        MaterialSlot::File(p) => MaterialRef::File(p.clone()),
        MaterialSlot::Inline(text) => ron::from_str(text).unwrap_or(MaterialRef::Model),
    }
}

/// Every material an entry names, in slot order: its renderer component's slots, or (an entry
/// built in memory in the old style, with no renderer component) the legacy loose fields.
pub fn material_refs(e: &SceneEntry) -> Vec<MaterialRef> {
    let mut out = Vec::new();
    for c in e.components.iter().filter(|c| is_renderer_kind(&c.kind)) {
        // Both renderer types share one data shape.
        if let Ok(r) = ron::from_str::<MeshRenderer>(&c.data) {
            out.extend(r.materials.iter().map(material_ref_of));
        }
    }
    if out.is_empty() {
        out = std::iter::once(&e.material).chain(e.extra_materials.iter()).cloned().collect();
    }
    out
}

/// Whether a component of this kind is a renderer.
pub fn has_renderer(e: &SceneEntry) -> bool {
    e.components.iter().any(|c| is_renderer_kind(&c.kind))
}

fn has_legacy_content(e: &SceneEntry) -> bool {
    !matches!(e.material, MaterialRef::Model)
        || !e.extra_materials.is_empty()
        || !e.cast_shadows.is_on()
        || (e.lightmap_scale - 1.0).abs() > f32::EPSILON
}

impl SceneEntry {
    /// The renderer component this entry's legacy fields describe, padded to the source's slot
    /// count. `None` for sources that never drew (empties, cameras, lights).
    pub fn legacy_renderer(&self) -> Option<MeshRenderer> {
        let slots = slot_count_of(&self.source);
        if slots == 0 {
            return None;
        }
        let mut materials: Vec<MaterialSlot> = std::iter::once(&self.material).chain(self.extra_materials.iter()).map(material_slot_of).collect();
        if materials.len() < slots {
            materials.resize(slots, MaterialSlot::Model);
        }
        Some(MeshRenderer { cast_shadows: self.cast_shadows, lightmap_scale: self.lightmap_scale, materials, ..Default::default() })
    }

    /// Move the legacy renderer fields into a `Mesh Renderer` component (first in the list).
    /// Does nothing to an entry that already has a renderer, or has no mesh. Clears the legacy
    /// fields either way once a renderer exists. `force` creates the component even when the
    /// legacy fields are all default (a file older than [`RENDERER_FORMAT`]).
    pub fn migrate_renderer(&mut self, force: bool) -> bool {
        self.renderer_split = true;
        if has_renderer(self) {
            return false;
        }
        if !(force || has_legacy_content(self)) {
            return false;
        }
        let Some(renderer) = self.legacy_renderer() else {
            // Never drew (camera, light, empty): the leftover material meant nothing.
            let had = has_legacy_content(self);
            self.material = MaterialRef::Model;
            self.extra_materials.clear();
            self.cast_shadows = ShadowCasting::On;
            self.lightmap_scale = 1.0;
            return had;
        };
        self.components.insert(0, ComponentData::new(MESH_RENDERER, ron::to_string(&renderer).unwrap_or_default()));
        self.material = MaterialRef::Model;
        self.extra_materials.clear();
        self.cast_shadows = ShadowCasting::On;
        self.lightmap_scale = 1.0;
        true
    }
}

impl SceneEntry {
    /// Set the material of every slot of this entry's renderer (creating the component when it
    /// has none). Only meaningful for mesh sources.
    pub fn set_renderer_materials(&mut self, refs: &[MaterialRef]) {
        let slots: Vec<MaterialSlot> = refs.iter().map(material_slot_of).collect();
        if let Some(c) = self.components.iter_mut().find(|c| is_renderer_kind(&c.kind)) {
            let mut r: MeshRenderer = ron::from_str(&c.data).unwrap_or_default();
            r.materials = slots;
            c.data = ron::to_string(&r).unwrap_or_default();
        } else {
            let r = MeshRenderer { materials: slots, ..Default::default() };
            self.components.insert(0, ComponentData::new(MESH_RENDERER, ron::to_string(&r).unwrap_or_default()));
        }
        self.renderer_split = true;
    }
}

const OLD_FIELDS: [&str; 4] = ["material", "extra_materials", "cast_shadows", "lightmap_scale"];

fn path_of(field: &str) -> String {
    format!("components[{MESH_RENDERER}].{field}")
}

/// What an override of a legacy renderer field means in component paths: `material` ->
/// `materials[0]`, `extra_materials` -> `materials[1..]`, the settings by name, with values
/// converted from `MaterialRef` to `MaterialSlot` text. `None` for any other path (or a value
/// that cannot be read, which stays an unused override).
pub(crate) fn legacy_override(path: &str, value: &str) -> Option<Vec<(String, String)>> {
    let slot_text = |m: &MaterialRef| ron::to_string(&material_slot_of(m)).unwrap_or_default();
    match path {
        "cast_shadows" | "lightmap_scale" => Some(vec![(path_of(path), value.to_owned())]),
        "material" => ron::from_str::<MaterialRef>(value).ok().map(|m| vec![(path_of("materials[0]"), slot_text(&m))]),
        "extra_materials" => ron::from_str::<Vec<MaterialRef>>(value)
            .ok()
            .map(|list| list.iter().enumerate().map(|(k, m)| (path_of(&format!("materials[{}]", k + 1)), slot_text(m))).collect()),
        _ => None,
    }
}

/// Rewrite overrides that name a legacy renderer field to the component paths (see
/// [`legacy_override`]). Idempotent.
pub fn migrate_link(link: &mut PrefabLink) -> bool {
    if !link.overrides.iter().any(|o| OLD_FIELDS.contains(&o.path.as_str())) {
        return false;
    }
    let old = std::mem::take(&mut link.overrides);
    for o in old {
        match legacy_override(&o.path, &o.value) {
            Some(list) => {
                for (path, value) in list {
                    link.overrides.push(crate::prefab_link::PrefabOverride { target: o.target.clone(), path, value, unused: false });
                }
            }
            None => link.overrides.push(o),
        }
    }
    true
}

/// Migrate a list of entries (a prefab, or the entries of a scene). `legacy_file` = the list
/// comes from a file older than [`RENDERER_FORMAT`], so every mesh object gets a renderer.
/// Returns true when anything changed.
pub fn migrate_entries(entries: &mut [SceneEntry], legacy_file: bool) -> bool {
    let mut changed = false;
    for e in entries {
        changed |= e.migrate_renderer(legacy_file);
        if let Some(link) = &mut e.prefab {
            changed |= migrate_link(link);
        }
    }
    changed
}

impl SceneFile {
    /// Bring a scene saved before the renderer split to the current form (see the module docs).
    /// Run once after loading; a second call changes nothing. Returns true when the file changed
    /// (the caller may mark it dirty so the next save writes the new form).
    pub fn migrate_renderers(&mut self) -> bool {
        let legacy = self.format_version < RENDERER_FORMAT;
        let mut changed = migrate_entries(&mut self.entries, legacy);
        if legacy {
            self.format_version = RENDERER_FORMAT;
            changed = true;
        }
        changed
    }
}

/// Whether `name` is a renderer component kind (re-exported for callers that only have this module).
#[allow(dead_code)] // renderer kind probe used by the migration tests
pub fn is_renderer(name: &str) -> bool {
    name == MESH_RENDERER || name == SKINNED_MESH_RENDERER
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefab_link::PrefabOverride;
    use zestengine_core::{ComponentRegistry, renderer_of};

    const MAIN_SCENE: &str = include_str!("../../../example-game/scenes/main.scene");

    fn model_source(extra: usize) -> ObjectSource {
        ObjectSource::Model { path: "models/hero.glb".into(), mesh: 0, extra_meshes: (1..=extra).collect(), model: None, mesh_id: None, extra_mesh_ids: Vec::new() }
    }

    fn entry(name: &str, source: ObjectSource) -> SceneEntry {
        ron::from_str(&format!(
            "(name: {name:?}, source: {}, material: Model, translation: (0.0, 0.0, 0.0), rotation: (0.0, 0.0, 0.0, 1.0), scale: (1.0, 1.0, 1.0))",
            ron::to_string(&source).unwrap()
        ))
        .unwrap()
    }

    fn inline_material() -> MaterialRef {
        ron::from_str(
            "Inline(params: (base_color: (0.5, 0.25, 1.0, 1.0), emission_color: (0.0, 0.0, 0.0), emission_intensity: 1.0, metallic: 0.5, roughness: 0.3, toon_steps: 3.0, pbr_toon_blend: 1.0, alpha_cutoff: 0.5, normal_strength: 1.0, occlusion_strength: 1.0), features: (toon: false, normal_map: false, emission: false, alpha_mode: Opaque, double_sided: false), shader: \"standard\")",
        )
        .unwrap()
    }

    /// What an entry draws with, per slot, plus its shadow / lightmap settings: the "looks
    /// identical" golden. Read from the legacy fields before migration and from the renderer
    /// component after it.
    type Look = Vec<(usize, String, ShadowCasting, f32)>;

    fn look_before(entries: &[SceneEntry]) -> Look {
        entries
            .iter()
            .enumerate()
            .filter(|(_, e)| slot_count_of(&e.source) > 0)
            .flat_map(|(i, e)| {
                let slots = slot_count_of(&e.source);
                let all: Vec<&MaterialRef> = std::iter::once(&e.material).chain(e.extra_materials.iter()).collect();
                (0..slots).map(move |k| (i * 100 + k, ron::to_string(all.get(k).copied().unwrap_or(&MaterialRef::Model)).unwrap(), e.cast_shadows, e.lightmap_scale)).collect::<Vec<_>>()
            })
            .collect()
    }

    fn look_after(entries: &[SceneEntry]) -> Look {
        let reg = ComponentRegistry::with_builtins();
        entries
            .iter()
            .enumerate()
            .filter(|(_, e)| slot_count_of(&e.source) > 0)
            .flat_map(|(i, e)| {
                let c = e.components.iter().find(|c| is_renderer(&c.kind)).unwrap_or_else(|| panic!("{} has no renderer", e.name)).load(&reg).unwrap();
                let r = renderer_of(c.as_ref()).unwrap();
                let slots = slot_count_of(&e.source);
                (0..slots).map(|k| (i * 100 + k, ron::to_string(&material_ref_of(&r.slot(k))).unwrap(), r.cast_shadows(), r.lightmap_scale())).collect::<Vec<_>>()
            })
            .collect()
    }

    fn assert_migrates_identically(file: &mut SceneFile) {
        let before = look_before(&file.entries);
        assert!(file.migrate_renderers());
        assert_eq!(file.format_version, RENDERER_FORMAT);
        assert_eq!(look_after(&file.entries), before, "every slot draws with the same material and settings");
        for e in &file.entries {
            assert!(matches!(e.material, MaterialRef::Model) && e.extra_materials.is_empty() && e.cast_shadows.is_on() && e.lightmap_scale == 1.0, "{}: legacy fields cleared", e.name);
        }
    }

    #[test]
    fn the_example_scene_migrates_and_is_idempotent() {
        let mut file = SceneFile::parse_unmigrated(MAIN_SCENE);
        assert_eq!(file.format_version, 0);
        let meshes = file.entries.iter().filter(|e| slot_count_of(&e.source) > 0).count();
        assert!(meshes >= 2, "the fixture has mesh objects");
        assert_migrates_identically(&mut file);
        let renderers = file.entries.iter().filter(|e| has_renderer(e)).count();
        assert_eq!(renderers, meshes, "exactly one renderer per mesh object; the camera gets none");
        let once = ron::to_string(&file).unwrap();
        assert!(!file.migrate_renderers(), "a second run changes nothing");
        assert_eq!(ron::to_string(&file).unwrap(), once);
        // save -> load -> migrate again: nothing duplicated, nothing dropped.
        let mut again: SceneFile = ron::from_str(&once).unwrap();
        assert_eq!(again.format_version, RENDERER_FORMAT);
        assert!(!again.migrate_renderers());
        assert_eq!(ron::to_string(&again).unwrap(), once);
        assert_eq!(again.entries.iter().filter(|e| has_renderer(e)).count(), meshes);
    }

    #[test]
    fn synthetic_scene_with_slots_shadows_and_inline_materials() {
        let mut hero = entry("Hero", model_source(2));
        hero.material = MaterialRef::File("materials/skin.material".into());
        hero.extra_materials = vec![MaterialRef::Model, inline_material()];
        hero.cast_shadows = ShadowCasting::TwoSided;
        hero.lightmap_scale = 2.0;
        let mut short = entry("Short list", model_source(3));
        short.extra_materials = vec![MaterialRef::File("materials/a.material".into())];
        let mut off = entry("Off", ObjectSource::Primitive { shape: crate::scene_file::PrimitiveShape::Cube });
        off.cast_shadows = ShadowCasting::Off;
        off.enabled = false;
        let plain = entry("Plain", ObjectSource::Builtin { mesh: 1 });
        let camera = entry("Camera", ObjectSource::Camera);
        let mut file = SceneFile { entries: vec![hero, short, off, plain, camera], ..Default::default() };
        assert_migrates_identically(&mut file);
        let reg = ComponentRegistry::with_builtins();
        let load = |i: usize| file.entries[i].components[0].load(&reg).unwrap();
        let (c0, c1) = (load(0), load(1));
        assert_eq!(renderer_of(c0.as_ref()).unwrap().materials().len(), 3);
        let short_r = renderer_of(c1.as_ref()).unwrap();
        assert_eq!(short_r.materials().len(), 4, "a shorter list is padded with Model up to the slot count");
        assert_eq!(short_r.slot(1), MaterialSlot::File("materials/a.material".into()));
        assert!(short_r.slot(3).is_model());
        assert!(file.entries[4].components.is_empty(), "a camera never draws, so it gets no renderer");
        assert!(!file.entries[2].enabled, "other fields are untouched");
    }

    #[test]
    fn a_renderer_that_was_removed_from_a_current_file_stays_removed() {
        let mut file = SceneFile { entries: vec![entry("Cube", ObjectSource::Primitive { shape: crate::scene_file::PrimitiveShape::Cube })], format_version: RENDERER_FORMAT, ..Default::default() };
        assert!(!file.migrate_renderers());
        assert!(file.entries[0].components.is_empty(), "version 2 + default legacy fields = no renderer on purpose");
        // ...but a legacy-form entry that slipped into a current file (a builder) is still upgraded.
        file.entries[0].material = MaterialRef::File("materials/x.material".into());
        assert!(file.migrate_renderers());
        assert_eq!(file.entries[0].components.len(), 1);
    }

    #[test]
    fn an_existing_renderer_is_left_alone() {
        let mut e = entry("Cube", ObjectSource::Primitive { shape: crate::scene_file::PrimitiveShape::Cube });
        e.components.push(ComponentData::new(MESH_RENDERER, "(materials:[File(\"keep.material\")])"));
        e.material = MaterialRef::File("legacy.material".into());
        assert!(!e.migrate_renderer(true));
        assert_eq!(e.components.len(), 1);
        assert!(e.components[0].data.contains("keep.material"));
    }

    #[test]
    fn prefab_overrides_of_the_old_fields_move_to_component_paths() {
        let ov = |path: &str, value: &str| PrefabOverride { target: "t".into(), path: path.into(), value: value.into(), unused: false };
        let mut link = PrefabLink::new("prefabs/p.prefab");
        link.overrides = vec![
            ov("material", "File(\"materials/red.material\")"),
            ov("extra_materials", "[File(\"materials/a.material\"), Model]"),
            ov("cast_shadows", "ShadowsOnly"),
            ov("lightmap_scale", "2.0"),
            ov("layer", "3"),
        ];
        assert!(migrate_link(&mut link));
        let got: Vec<(&str, &str)> = link.overrides.iter().map(|o| (o.path.as_str(), o.value.as_str())).collect();
        assert_eq!(
            got,
            vec![
                ("components[Mesh Renderer].materials[0]", "File(\"materials/red.material\")"),
                ("components[Mesh Renderer].materials[1]", "File(\"materials/a.material\")"),
                ("components[Mesh Renderer].materials[2]", "Model"),
                ("components[Mesh Renderer].cast_shadows", "ShadowsOnly"),
                ("components[Mesh Renderer].lightmap_scale", "2.0"),
                ("layer", "3"),
            ]
        );
        assert!(link.overrides.iter().all(|o| o.target == "t"));
        assert!(!migrate_link(&mut link), "idempotent");
    }

    #[test]
    fn model_parts_stay_small_when_untouched() {
        let r = MeshRenderer::inheriting(3);
        assert!(r.is_untouched());
        let text = ron::to_string(&r).unwrap();
        assert!(text.len() < 80, "a default renderer is tiny: {text}");
    }
}
