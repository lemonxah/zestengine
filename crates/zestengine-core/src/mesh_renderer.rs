//! `Mesh Renderer` / `Skinned Mesh Renderer`: the components that own how an object's mesh is
//! drawn (shadow casting, lightmap scale and ONE material per submesh slot).
//!
//! The mesh itself is not here: it stays in the object's `ObjectSource` (the "Mesh Filter" the
//! inspector shows as a view). Slot `k` draws submesh `k` of that mesh. See
//! `docs/design/mesh-renderer-split.md`.
//!
//! Core has no dependency on assets or render, and an inline material snapshot carries render
//! types, so [`MaterialSlot::Inline`] keeps the snapshot as opaque RON text that the engine
//! decodes.

use serde::{Deserialize, Serialize};

use crate::{Component, ShadowCasting, TypedComponent, component_inner, component_inner_mut};

/// Registry name of [`MeshRenderer`].
pub const MESH_RENDERER: &str = "Mesh Renderer";
/// Registry name of [`SkinnedMeshRenderer`].
pub const SKINNED_MESH_RENDERER: &str = "Skinned Mesh Renderer";

/// One material slot of a renderer. Slot `k` draws submesh `k` of the object's mesh.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MaterialSlot {
    /// Inherit: the model's own material for this submesh (after the importer's material
    /// remap table); the default material for a primitive. Costs nothing on disk.
    #[default]
    Model,
    /// A `.material` asset, project-relative. Shared by every user of the path.
    File(String),
    /// An edited snapshot of the model's material (RON text of `assets::MaterialRef::Inline`).
    /// Opaque to core; the engine converts it.
    Inline(String),
}

impl MaterialSlot {
    /// True for the `Model` (inherit) slot.
    pub fn is_model(&self) -> bool {
        matches!(self, Self::Model)
    }
}

/// What the engine remembers about the slots it last resolved. Never saved, never compared.
#[derive(Clone, Debug, Default)]
pub struct RendererRuntime {
    /// The normalised slot list the object's resolved render cache was built from
    /// (`None` = not resolved yet, e.g. right after a component reload).
    pub resolved: Option<Vec<MaterialSlot>>,
}

impl PartialEq for RendererRuntime {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

fn default_lightmap_scale() -> f32 {
    1.0
}

fn is_default_scale(v: &f32) -> bool {
    (*v - 1.0).abs() < f32::EPSILON
}

/// The data both renderer types share.
pub trait RendererData {
    /// How the renderer casts shadows.
    fn cast_shadows(&self) -> ShadowCasting;
    /// Set how the renderer casts shadows.
    fn set_cast_shadows(&mut self, value: ShadowCasting);
    /// Lightmap-resolution multiplier.
    fn lightmap_scale(&self) -> f32;
    /// Set the lightmap-resolution multiplier.
    fn set_lightmap_scale(&mut self, value: f32);
    /// The material slots.
    fn materials(&self) -> &[MaterialSlot];
    /// Mutable material slots.
    fn materials_mut(&mut self) -> &mut Vec<MaterialSlot>;
    /// The runtime state.
    fn runtime(&self) -> &RendererRuntime;
    /// Mutable runtime state.
    fn runtime_mut(&mut self) -> &mut RendererRuntime;
    /// Registry name of the concrete type.
    fn kind(&self) -> &'static str;

    /// Slot `k`, `Model` past the end of the list (a shorter list inherits from the model).
    fn slot(&self, k: usize) -> MaterialSlot {
        self.materials().get(k).cloned().unwrap_or_default()
    }

    /// The first `count` slots, padded with `Model`. Extra entries (a longer list) are kept in
    /// the data but not part of this view.
    fn normalised(&self, count: usize) -> Vec<MaterialSlot> {
        (0..count).map(|k| self.slot(k)).collect()
    }

    /// Set slot `k`, padding a short list with `Model`.
    fn set_slot(&mut self, k: usize, value: MaterialSlot) {
        let list = self.materials_mut();
        if list.len() <= k {
            list.resize(k + 1, MaterialSlot::Model);
        }
        list[k] = value;
    }
}

macro_rules! renderer_type {
    ($(#[$doc:meta])* $name:ident, $label:expr) => {
        $(#[$doc])*
        #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
        #[serde(default)]
        pub struct $name {
            /// "Cast Shadows": On / Off / Two Sided / Shadows Only.
            pub cast_shadows: ShadowCasting,
            /// "Scale In Lightmap": lightmap-resolution multiplier (needs Contribute GI).
            pub lightmap_scale: f32,
            /// One entry per submesh slot, in the object's mesh order. Shorter = the rest inherit
            /// from the model; longer = the extras are kept but not drawn.
            pub materials: Vec<MaterialSlot>,
            /// Engine-filled state of the last resolved slots (not saved).
            #[serde(skip)]
            pub runtime: RendererRuntime,
        }

        impl Default for $name {
            fn default() -> Self {
                Self { cast_shadows: ShadowCasting::On, lightmap_scale: default_lightmap_scale(), materials: Vec::new(), runtime: RendererRuntime::default() }
            }
        }

        impl TypedComponent for $name {
            const NAME: &'static str = $label;
        }

        impl RendererData for $name {
            fn cast_shadows(&self) -> ShadowCasting {
                self.cast_shadows
            }
            fn set_cast_shadows(&mut self, value: ShadowCasting) {
                self.cast_shadows = value;
            }
            fn lightmap_scale(&self) -> f32 {
                self.lightmap_scale
            }
            fn set_lightmap_scale(&mut self, value: f32) {
                self.lightmap_scale = value;
            }
            fn materials(&self) -> &[MaterialSlot] {
                &self.materials
            }
            fn materials_mut(&mut self) -> &mut Vec<MaterialSlot> {
                &mut self.materials
            }
            fn runtime(&self) -> &RendererRuntime {
                &self.runtime
            }
            fn runtime_mut(&mut self) -> &mut RendererRuntime {
                &mut self.runtime
            }
            fn kind(&self) -> &'static str {
                $label
            }
        }
    };
}

renderer_type!(
    /// Draws the object's mesh with one material per submesh slot.
    MeshRenderer,
    MESH_RENDERER
);
renderer_type!(
    /// The renderer of a skinned mesh: the same settings and slots as [`MeshRenderer`]; the
    /// inspector adds the skeleton foldout and the Blendshapes section under it.
    SkinnedMeshRenderer,
    SKINNED_MESH_RENDERER
);

impl MeshRenderer {
    /// A renderer whose first `slots` slots all inherit from the model (what the importer's
    /// parts and a fresh primitive get).
    pub fn inheriting(slots: usize) -> Self {
        Self { materials: vec![MaterialSlot::Model; slots], ..Self::default() }
    }

    /// Settings are the defaults and every slot inherits: nothing worth storing.
    pub fn is_untouched(&self) -> bool {
        is_untouched(self)
    }
}

/// Every setting is the default and every slot inherits.
pub fn is_untouched(r: &dyn RendererData) -> bool {
    r.cast_shadows().is_on() && is_default_scale(&r.lightmap_scale()) && r.materials().iter().all(MaterialSlot::is_model)
}

/// A renderer component, whether or not it is switched off, as its shared view.
pub fn renderer_of(c: &dyn Component) -> Option<&dyn RendererData> {
    let c = component_inner(c);
    let any = c.as_any();
    if let Some(r) = any.downcast_ref::<MeshRenderer>() {
        return Some(r);
    }
    any.downcast_ref::<SkinnedMeshRenderer>().map(|r| r as &dyn RendererData)
}

/// Mutable form of [`renderer_of`].
pub fn renderer_of_mut(c: &mut Box<dyn Component>) -> Option<&mut dyn RendererData> {
    let c = component_inner_mut(c);
    let is_mesh = c.as_any().is::<MeshRenderer>();
    let any = c.as_any_mut();
    if is_mesh {
        return any.downcast_mut::<MeshRenderer>().map(|r| r as &mut dyn RendererData);
    }
    any.downcast_mut::<SkinnedMeshRenderer>().map(|r| r as &mut dyn RendererData)
}

/// Whether `name` is one of the two renderer component names.
pub fn is_renderer_kind(name: &str) -> bool {
    name == MESH_RENDERER || name == SKINNED_MESH_RENDERER
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ComponentRegistry;

    #[test]
    fn defaults_are_written_flat_so_prefab_overrides_can_name_fields_and_round_trip() {
        let r = MeshRenderer::default();
        let text = ron::to_string(&r).unwrap();
        assert_eq!(text, "(cast_shadows:On,lightmap_scale:1.0,materials:[])", "fields are always written: {text}");
        assert_eq!(ron::from_str::<MeshRenderer>(&text).unwrap(), r);
        let r = MeshRenderer {
            cast_shadows: ShadowCasting::ShadowsOnly,
            lightmap_scale: 2.0,
            materials: vec![MaterialSlot::Model, MaterialSlot::File("m/a.material".into()), MaterialSlot::Inline("(x)".into())],
            ..Default::default()
        };
        let back: MeshRenderer = ron::from_str(&ron::to_string(&r).unwrap()).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn missing_fields_take_defaults() {
        let r: MeshRenderer = ron::from_str("()").unwrap();
        assert_eq!(r, MeshRenderer::default());
        let r: SkinnedMeshRenderer = ron::from_str("(lightmap_scale: 3.0)").unwrap();
        assert_eq!(r.lightmap_scale, 3.0);
        assert!(r.cast_shadows.is_on() && r.materials.is_empty());
    }

    #[test]
    fn slots_pad_and_trim_without_losing_extras() {
        let mut r = MeshRenderer::default();
        r.set_slot(2, MaterialSlot::File("x.material".into()));
        assert_eq!(r.materials, vec![MaterialSlot::Model, MaterialSlot::Model, MaterialSlot::File("x.material".into())]);
        assert_eq!(r.normalised(2), vec![MaterialSlot::Model; 2], "a shorter mesh sees only its slots");
        assert_eq!(r.normalised(4)[3], MaterialSlot::Model, "a longer mesh inherits the tail");
        assert_eq!(r.materials.len(), 3, "the extras stay in the data");
        assert!(!r.is_untouched());
        assert!(MeshRenderer::inheriting(5).is_untouched());
    }

    #[test]
    fn registry_creates_both_and_they_are_client_side() {
        let reg = ComponentRegistry::with_builtins();
        for name in [MESH_RENDERER, SKINNED_MESH_RENDERER] {
            let c = reg.create(name).unwrap_or_else(|| panic!("{name} is registered"));
            assert_eq!(c.type_name(), name);
            assert!(renderer_of(c.as_ref()).is_some());
            assert_eq!(reg.default_role(name), crate::NetRole::Client, "{name} is stripped from a dedicated server");
            let loaded = reg.load(name, "(materials:[File(\"a.material\")])").unwrap();
            assert_eq!(renderer_of(loaded.as_ref()).unwrap().slot(0), MaterialSlot::File("a.material".into()));
        }
    }

    #[test]
    fn a_disabled_renderer_is_still_readable_as_data() {
        let mut c: Box<dyn Component> = Box::new(MeshRenderer::inheriting(2));
        crate::set_component_enabled(&mut c, false);
        assert!(!crate::component_enabled(c.as_ref()));
        assert_eq!(renderer_of(c.as_ref()).unwrap().materials().len(), 2);
        renderer_of_mut(&mut c).unwrap().set_cast_shadows(ShadowCasting::Off);
        assert_eq!(renderer_of(c.as_ref()).unwrap().cast_shadows(), ShadowCasting::Off);
    }
}
