//! Object-property animation tracks: object active, component enabled,
//! blendshape weight and material colour / float / texture offset.
//!
//! Clips address these by object path + property name (+ material slot), see
//! `zestengine_assets::anim_clip`. At runtime a [`Rig`] resolves each distinct
//! (object, property, slot, name) to a small channel id **once** ([`PropTable`]),
//! and keeps the per-clip list of `track -> channel` bindings ([`PropBinding`])
//! in a cache keyed by the clip's address and a content fingerprint. Sampling a
//! clip then costs one cache probe per clip plus one key evaluation per bound
//! track: no path or name lookups per frame.
//!
//! Poses carry property values as a sparse, channel-sorted list of
//! [`PropSample`]s. Each sample has a *coverage*: how much of the final value
//! the animation accounts for. Cross-fades and layer weights scale coverage
//! instead of needing a rest value, and applying a sample blends the object's
//! current value toward the animated one by its coverage. Channels the
//! animation does not touch are never written (they keep their last value,
//! like a Write Defaults off state); "off" values come from a defaults layer.
//!
//! Applying writes straight to the scene:
//! * active -> `SceneObject::enabled`
//! * component enabled -> `set_component_enabled` on the component found by
//!   type name (index remembered per channel)
//! * blendshape -> the object's `BlendShapes` weights (authored < animation <
//!   face tracking, since tracking overrides at drive time); the GPU morph
//!   pass picks the weights up through the existing face-drive path
//! * material -> queued as [`MaterialWrite`]s on `AnimAssets` and flushed into
//!   the renderer's live `MaterialParams` (push constants, no UBO upload, no
//!   allocation after the queue's first growth). Materials are shared by every
//!   object using them, so a material track animates all of its users.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use zestengine_assets::{AnimClip, TrackProperty};
use zestengine_core::{BlendShapes, Component, CustomProperties, PropSample, SPRITE_SWAP_HASH_MASK, SpriteResolver, component_enabled, component_inner_mut, set_component_enabled};
use zestengine_render::{MaterialHandle, MaterialParams, Renderer};

use crate::anim_graph::{AnimAssets, Pose, Rig};
use crate::scene::{LoadedScene, MaterialEntry, SceneObject};

// ------------------------------------------------------------ material props

/// A material property a track can animate, by the name stored in the clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialProp {
    BaseColor,
    EmissionColor,
    Metallic,
    Roughness,
    EmissionIntensity,
    NormalStrength,
    OcclusionStrength,
    AlphaCutoff,
    AlbedoOffset,
    NormalOffset,
    OrmOffset,
    EmissionOffset,
}

impl MaterialProp {
    pub const ALL: [MaterialProp; 12] = [
        Self::BaseColor,
        Self::EmissionColor,
        Self::Metallic,
        Self::Roughness,
        Self::EmissionIntensity,
        Self::NormalStrength,
        Self::OcclusionStrength,
        Self::AlphaCutoff,
        Self::AlbedoOffset,
        Self::NormalOffset,
        Self::OrmOffset,
        Self::EmissionOffset,
    ];

    /// The name stored in `AnimTrack::name`.
    pub fn name(self) -> &'static str {
        match self {
            Self::BaseColor => "base_color",
            Self::EmissionColor => "emission_color",
            Self::Metallic => "metallic",
            Self::Roughness => "roughness",
            Self::EmissionIntensity => "emission_intensity",
            Self::NormalStrength => "normal_strength",
            Self::OcclusionStrength => "occlusion_strength",
            Self::AlphaCutoff => "alpha_cutoff",
            Self::AlbedoOffset => "albedo_offset",
            Self::NormalOffset => "normal_offset",
            Self::OrmOffset => "orm_offset",
            Self::EmissionOffset => "emission_offset",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::BaseColor => "Base Colour",
            Self::EmissionColor => "Emission Colour",
            Self::Metallic => "Metallic",
            Self::Roughness => "Roughness",
            Self::EmissionIntensity => "Emission Intensity",
            Self::NormalStrength => "Normal Strength",
            Self::OcclusionStrength => "Occlusion Strength",
            Self::AlphaCutoff => "Alpha Cutoff",
            Self::AlbedoOffset => "Albedo Offset",
            Self::NormalOffset => "Normal Offset",
            Self::OrmOffset => "ORM Offset",
            Self::EmissionOffset => "Emission Offset",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == name)
    }

    /// The track kind that animates this property.
    pub fn track(self) -> TrackProperty {
        match self {
            Self::BaseColor | Self::EmissionColor => TrackProperty::MaterialColor,
            Self::AlbedoOffset | Self::NormalOffset | Self::OrmOffset | Self::EmissionOffset => TrackProperty::MaterialOffset,
            _ => TrackProperty::MaterialFloat,
        }
    }

    /// Current (authored) value from the material model.
    pub fn read(self, m: &zestengine_core::MaterialModel) -> [f32; 4] {
        match self {
            Self::BaseColor => m.base_color,
            Self::EmissionColor => [m.emission_color[0], m.emission_color[1], m.emission_color[2], 1.0],
            Self::Metallic => [m.metallic, 0.0, 0.0, 0.0],
            Self::Roughness => [m.roughness, 0.0, 0.0, 0.0],
            Self::EmissionIntensity => [m.emission_intensity, 0.0, 0.0, 0.0],
            Self::NormalStrength => [m.normal_strength, 0.0, 0.0, 0.0],
            Self::OcclusionStrength => [m.occlusion_strength, 0.0, 0.0, 0.0],
            Self::AlphaCutoff => [m.alpha_cutoff, 0.0, 0.0, 0.0],
            Self::AlbedoOffset => [m.albedo_offset[0], m.albedo_offset[1], 0.0, 0.0],
            Self::NormalOffset => [m.normal_offset[0], m.normal_offset[1], 0.0, 0.0],
            Self::OrmOffset => [m.orm_offset[0], m.orm_offset[1], 0.0, 0.0],
            Self::EmissionOffset => [m.emission_offset[0], m.emission_offset[1], 0.0, 0.0],
        }
    }

    /// Write into the renderer's live parameters (all of these are delivered
    /// as push constants, so no uniform upload follows).
    pub fn write(self, p: &mut MaterialParams, v: [f32; 4]) {
        match self {
            Self::BaseColor => p.base_color = v,
            Self::EmissionColor => p.emission_color = [v[0], v[1], v[2]],
            Self::Metallic => p.metallic = v[0],
            Self::Roughness => p.roughness = v[0],
            Self::EmissionIntensity => p.emission_intensity = v[0],
            Self::NormalStrength => p.normal_strength = v[0],
            Self::OcclusionStrength => p.occlusion_strength = v[0],
            Self::AlphaCutoff => p.alpha_cutoff = v[0],
            Self::AlbedoOffset => p.albedo_offset = [v[0], v[1]],
            Self::NormalOffset => p.normal_offset = [v[0], v[1]],
            Self::OrmOffset => p.orm_offset = [v[0], v[1]],
            Self::EmissionOffset => p.emission_offset = [v[0], v[1]],
        }
    }
}

/// A material value waiting to go into the renderer.
#[derive(Clone, Copy, Debug)]
pub struct MaterialWrite {
    pub material: usize,
    pub prop: MaterialProp,
    pub value: [f32; 4],
}

/// More queued writes than this mean nothing is flushing them (no renderer):
/// further ones are dropped instead of growing without bound.
const MAX_QUEUED_WRITES: usize = 16384;

/// Push this frame's queued material writes into the renderer.
pub fn flush_material_writes(assets: &mut AnimAssets, materials: &[MaterialEntry], renderer: &mut Renderer) {
    for w in assets.material_writes.drain(..) {
        let Some(entry) = materials.get(w.material) else {
            continue;
        };
        if entry.handle == MaterialHandle::NONE {
            continue;
        }
        w.prop.write(renderer.material_params_mut(entry.handle), w.value);
    }
}

// ------------------------------------------------------------- channel table

/// One resolved animated property.
#[derive(Clone, Debug)]
pub struct PropTarget {
    /// Rig node (index into the rig's nodes / the editor's `nodes` list).
    pub node: u32,
    pub kind: TrackProperty,
    pub slot: u8,
    pub name: Box<str>,
    /// Resolved once for material tracks (`None` = unknown property name).
    pub material: Option<MaterialProp>,
    /// Remembered component index / blendshape index, verified before use.
    comp_hint: usize,
    shape_hint: usize,
}

impl PropTarget {
    pub fn new(node: u32, kind: TrackProperty, slot: u8, name: &str) -> Self {
        Self {
            node,
            kind,
            slot: if kind.uses_slot() { slot } else { 0 },
            name: name.into(),
            material: kind.uses_slot().then(|| MaterialProp::from_name(name)).flatten(),
            comp_hint: 0,
            shape_hint: 0,
        }
    }
}

/// Which channel a clip track drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PropBinding {
    pub track: u32,
    pub chan: u32,
    pub node: u32,
}

type PropKey = (u32, TrackProperty, u8, String);

#[derive(Default)]
struct Inner {
    targets: Vec<PropTarget>,
    lookup: HashMap<PropKey, u32>,
    clips: HashMap<(usize, u64), Arc<[PropBinding]>>,
}

/// A rig's property channels and the per-clip bindings onto them. Derived
/// data: cloning a rig starts with an empty table.
#[derive(Default)]
pub struct PropTable(Mutex<Inner>);

impl Clone for PropTable {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for PropTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PropTable")
    }
}

impl PropTable {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Forget every binding (after the clip's tracks were edited).
    pub fn clear(&self) {
        let mut g = self.lock();
        g.clips.clear();
    }

    pub fn len(&self) -> usize {
        self.lock().targets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The channel's target (a copy).
    pub fn target(&self, chan: u32) -> Option<PropTarget> {
        self.lock().targets.get(chan as usize).cloned()
    }
}

/// Hash of everything the bindings depend on, over the object-property tracks.
/// `None` when the clip has none. FNV-1a over integers and name bytes: no
/// allocation, and a stale cache entry would need a different clip with the
/// same address and the same hash.
fn fingerprint(tracks: &[zestengine_assets::AnimTrack]) -> Option<u64> {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut any = false;
    let mut mix = |x: u64| h = (h ^ x).wrapping_mul(0x0000_0100_0000_01b3);
    for (i, t) in tracks.iter().enumerate() {
        if !t.property.is_object_prop() {
            continue;
        }
        any = true;
        mix(i as u64);
        mix(t.property as u64);
        mix(t.slot as u64);
        mix(t.keys.is_empty() as u64);
        for b in t.target.bytes() {
            mix(b as u64);
        }
        mix(0x100);
        for b in t.name.bytes() {
            mix(b as u64);
        }
        mix(0x101);
    }
    any.then_some(h)
}

impl Rig {
    /// The channels the clip's object-property tracks drive on this rig,
    /// sorted by channel. Tracks whose object is not in the rig are left
    /// out. `None` when the clip has no such tracks. Cached per clip.
    pub fn prop_bindings(&self, clip: &AnimClip) -> Option<Arc<[PropBinding]>> {
        self.bind_tracks(&clip.tracks, clip as *const AnimClip as usize, false)
    }

    /// [`Rig::prop_bindings`] over a bare track list keyed by `addr` (the list's
    /// own address). `anchored` tracks drive the animated object itself (the
    /// blendshape curves of an imported clip on a skinned mesh): they bind to
    /// node 0 without looking the target path up.
    pub fn bind_tracks(&self, tracks: &[zestengine_assets::AnimTrack], addr: usize, anchored: bool) -> Option<Arc<[PropBinding]>> {
        let fp = fingerprint(tracks)?;
        let key = (addr, fp);
        let mut g = self.props.lock();
        if let Some(b) = g.clips.get(&key) {
            return Some(b.clone());
        }
        if g.clips.len() > 256 {
            g.clips.clear();
        }
        let mut out: Vec<PropBinding> = Vec::new();
        for (i, t) in tracks.iter().enumerate() {
            if !t.property.is_object_prop() || t.keys.is_empty() {
                continue;
            }
            let Some(node) = (if anchored { Some(0) } else { self.index(&t.target) }) else {
                continue;
            };
            let slot = if t.property.uses_slot() { t.slot } else { 0 };
            let k: PropKey = (node as u32, t.property, slot, t.name.clone());
            let chan = match g.lookup.get(&k) {
                Some(&c) => c,
                None => {
                    let c = g.targets.len() as u32;
                    g.targets.push(PropTarget::new(node as u32, t.property, slot, &t.name));
                    g.lookup.insert(k, c);
                    c
                }
            };
            if out.iter().any(|b| b.chan == chan) {
                continue;
            }
            out.push(PropBinding { track: i as u32, chan, node: node as u32 });
        }
        out.sort_by_key(|b| b.chan);
        let arc: Arc<[PropBinding]> = out.into();
        g.clips.insert(key, arc.clone());
        Some(arc)
    }
}

// ------------------------------------------------------------ pose algebra

impl Pose {
    /// Insert (or replace) the property sample for its channel.
    pub fn set_prop(&mut self, s: PropSample) {
        match self.props.binary_search_by_key(&s.chan, |p| p.chan) {
            Ok(i) => self.props[i] = s,
            Err(i) => self.props.insert(i, s),
        }
    }
}

/// Coverage-weighted mix of several property lists (`weight` per list; the
/// weights are used as given). A channel only some lists drive gets the
/// summed coverage of those, so a fade out of a driven channel lowers its
/// coverage rather than its value.
pub fn mix_props(parts: &[(&[PropSample], f32)]) -> Vec<PropSample> {
    if parts.iter().all(|(l, _)| l.is_empty()) {
        return Vec::new();
    }
    struct Acc {
        chan: u32,
        node: u32,
        v: [f32; 4],
        c: f32,
    }
    let mut acc: Vec<Acc> = Vec::new();
    for (list, w) in parts {
        if *w <= 0.0 {
            continue;
        }
        for s in *list {
            let wc = w * s.cover;
            let i = match acc.binary_search_by_key(&s.chan, |a| a.chan) {
                Ok(i) => i,
                Err(i) => {
                    acc.insert(i, Acc { chan: s.chan, node: s.node, v: [0.0; 4], c: 0.0 });
                    i
                }
            };
            for k in 0..4 {
                acc[i].v[k] += s.value[k] * wc;
            }
            acc[i].c += wc;
        }
    }
    acc.into_iter()
        .filter(|a| a.c > 1e-6)
        .map(|a| PropSample { chan: a.chan, node: a.node, value: a.v.map(|x| x / a.c), cover: a.c.min(1.0) })
        .collect()
}

/// Lay a layer's property samples over the base ("over" compositing: the
/// layer's share is `weight * cover`, the base keeps the rest). Additive
/// layers composite the same way: a property has no rest value to add to.
/// `masked(node)` is true for nodes the layer's mask excludes.
pub fn compose_props(base: &mut Vec<PropSample>, layer: &[PropSample], weight: f32, masked: impl Fn(u32) -> bool) {
    for s in layer {
        if masked(s.node) {
            continue;
        }
        let wl = (weight * s.cover).clamp(0.0, 1.0);
        if wl <= 0.0 {
            continue;
        }
        match base.binary_search_by_key(&s.chan, |p| p.chan) {
            Ok(i) => {
                let b = base[i];
                let c = wl + b.cover * (1.0 - wl);
                let value = std::array::from_fn(|k| (s.value[k] * wl + b.value[k] * b.cover * (1.0 - wl)) / c);
                base[i] = PropSample { chan: s.chan, node: s.node, value, cover: c.min(1.0) };
            }
            Err(i) => base.insert(i, PropSample { cover: wl, ..*s }),
        }
    }
}

/// Write Defaults off: channels `held` drove last frame that `props` does not
/// drive keep their last value.
pub fn hold_props(props: &mut Vec<PropSample>, held: &[PropSample]) {
    for h in held {
        if let Err(i) = props.binary_search_by_key(&h.chan, |p| p.chan) {
            props.insert(i, *h);
        }
    }
}

// ------------------------------------------------------------ scene access

fn blend_toward(cur: [f32; 4], target: [f32; 4], cover: f32, lanes: usize) -> [f32; 4] {
    let mut out = target;
    for k in 0..lanes.min(4) {
        out[k] = cur[k] + (target[k] - cur[k]) * cover;
    }
    out
}

fn find_component(comps: &[Box<dyn Component>], name: &str, hint: &mut usize) -> Option<usize> {
    if comps.get(*hint).is_some_and(|c| c.type_name() == name) {
        return Some(*hint);
    }
    let i = comps.iter().position(|c| c.type_name() == name)?;
    *hint = i;
    Some(i)
}

/// Index of the object's switched-on `BlendShapes`; adds one when the object
/// has none (not when a switched-off one exists).
fn blend_shapes_index(comps: &mut Vec<Box<dyn Component>>, hint: &mut usize, create: bool) -> Option<usize> {
    if comps.get(*hint).is_some_and(|c| c.as_any().is::<BlendShapes>()) {
        return Some(*hint);
    }
    if let Some(i) = comps.iter().position(|c| c.as_any().is::<BlendShapes>()) {
        *hint = i;
        return Some(i);
    }
    if !create || comps.iter().any(|c| c.type_name() == <BlendShapes as zestengine_core::TypedComponent>::NAME) {
        return None;
    }
    comps.push(Box::new(BlendShapes::default()));
    *hint = comps.len() - 1;
    Some(*hint)
}

/// The value now, for properties that need no scene access beyond the object.
/// Blendshapes the object has no weight for read as NaN in lane 0.
fn read_target(objects: &[SceneObject], materials: &[MaterialEntry], obj: usize, t: &mut PropTarget) -> Option<[f32; 4]> {
    let o = objects.get(obj)?;
    match t.kind {
        TrackProperty::Active => Some([o.enabled as u8 as f32, 0.0, 0.0, 0.0]),
        TrackProperty::ComponentEnabled => {
            let i = find_component(&o.components, &t.name, &mut t.comp_hint)?;
            Some([component_enabled(o.components[i].as_ref()) as u8 as f32, 0.0, 0.0, 0.0])
        }
        TrackProperty::BlendShape => {
            let w = o
                .components
                .iter()
                .find_map(|c| c.as_any().downcast_ref::<BlendShapes>())
                .and_then(|b| b.weights.iter().find(|(n, _)| **n == *t.name))
                .map(|(_, w)| *w)
                .unwrap_or(f32::NAN);
            Some([w, 0.0, 0.0, 0.0])
        }
        TrackProperty::CustomProperty => {
            let v = o.components.iter().find_map(|c| c.as_any().downcast_ref::<CustomProperties>()).and_then(|p| p.read_lane(&t.name)).unwrap_or(f32::NAN);
            Some([v, 0.0, 0.0, 0.0])
        }
        TrackProperty::SpriteSwap => {
            let r = o.components.iter().find_map(|c| zestengine_core::component_inner(c.as_ref()).as_any().downcast_ref::<SpriteResolver>())?;
            Some([r.hash as f32, 0.0, 0.0, 0.0])
        }
        _ => {
            let ri = o.render_slots().nth(t.slot as usize)?;
            Some(t.material?.read(&materials.get(ri.material)?.model))
        }
    }
}

/// Current value of an object property (for keying and previews); `None` when
/// the object lacks it (no such component, slot or property). Blendshapes
/// without a weight yet read NaN in lane 0.
pub fn read_prop(scene: &LoadedScene, obj: usize, kind: TrackProperty, slot: u8, name: &str) -> Option<[f32; 4]> {
    let mut t = PropTarget::new(0, kind, slot, name);
    read_target(&scene.objects, &scene.materials, obj, &mut t)
}

/// Write one property: blend the current value toward `value` by `cover`.
fn write_target(
    objects: &mut [SceneObject],
    materials: &[MaterialEntry],
    writes: &mut Vec<MaterialWrite>,
    obj: usize,
    t: &mut PropTarget,
    value: [f32; 4],
    cover: f32,
) {
    let Some(o) = objects.get_mut(obj) else {
        return;
    };
    match t.kind {
        TrackProperty::Active => {
            let cur = o.enabled as u8 as f32;
            let want = cur + (value[0] - cur) * cover >= 0.5;
            if o.enabled != want {
                o.enabled = want;
            }
        }
        TrackProperty::ComponentEnabled => {
            // An Animator switching itself off would end the animation.
            if &*t.name == "Animator" {
                return;
            }
            let Some(i) = find_component(&o.components, &t.name, &mut t.comp_hint) else {
                return;
            };
            let cur = component_enabled(o.components[i].as_ref()) as u8 as f32;
            let want = cur + (value[0] - cur) * cover >= 0.5;
            if component_enabled(o.components[i].as_ref()) != want {
                set_component_enabled(&mut o.components[i], want);
            }
        }
        TrackProperty::BlendShape => {
            if value[0].is_nan() {
                // Restore of "no weight": drop the entry.
                if let Some(i) = blend_shapes_index(&mut o.components, &mut t.comp_hint, false)
                    && let Some(b) = o.components[i].as_any_mut().downcast_mut::<BlendShapes>()
                {
                    b.weights.retain(|(n, _)| **n != *t.name);
                }
                return;
            }
            let Some(i) = blend_shapes_index(&mut o.components, &mut t.comp_hint, true) else {
                return;
            };
            let Some(b) = o.components[i].as_any_mut().downcast_mut::<BlendShapes>() else {
                return;
            };
            let at = if b.weights.get(t.shape_hint).is_some_and(|(n, _)| **n == *t.name) {
                Some(t.shape_hint)
            } else {
                b.weights.iter().position(|(n, _)| **n == *t.name)
            };
            match at {
                Some(j) => {
                    t.shape_hint = j;
                    let cur = b.weights[j].1;
                    b.weights[j].1 = cur + (value[0] - cur) * cover;
                }
                None => {
                    b.weights.push((t.name.to_string(), value[0] * cover));
                    t.shape_hint = b.weights.len() - 1;
                }
            }
        }
        TrackProperty::CustomProperty => {
            let idx = o.components.iter().position(|c| c.as_any().is::<CustomProperties>());
            if value[0].is_nan() {
                // Restore of "no value": drop the property again.
                if let Some(i) = idx
                    && let Some(p) = o.components[i].as_any_mut().downcast_mut::<CustomProperties>()
                {
                    let base = t.name.rsplit_once('.').map(|(b, _)| b.to_owned());
                    p.values.retain(|(n, _)| **n != *t.name && Some(n) != base.as_ref());
                }
                return;
            }
            let i = match idx {
                Some(i) => i,
                None => {
                    if o.components.iter().any(|c| c.type_name() == <CustomProperties as zestengine_core::TypedComponent>::NAME) {
                        return; // present but switched off
                    }
                    o.components.push(Box::new(CustomProperties::default()));
                    o.components.len() - 1
                }
            };
            let Some(p) = o.components[i].as_any_mut().downcast_mut::<CustomProperties>() else { return };
            let cur = p.read_lane(&t.name);
            p.write_lane(&t.name, cur.map_or(value[0], |c| c + (value[0] - c) * cover));
        }
        TrackProperty::SpriteSwap => {
            // A discrete id: crossfading hashes would average to nonsense, so only
            // a clear majority writes, and the resolver ignores hashes it does not know.
            if cover < 0.5 || !value[0].is_finite() {
                return;
            }
            let h = value[0].round();
            if h < 1.0 || h > SPRITE_SWAP_HASH_MASK as f32 {
                return;
            }
            for c in o.components.iter_mut() {
                if let Some(r) = component_inner_mut(c).as_any_mut().downcast_mut::<SpriteResolver>() {
                    if r.hash != h as u32 {
                        r.hash = h as u32;
                    }
                    break;
                }
            }
        }
        _ => {
            let (Some(prop), Some(ri)) = (t.material, o.render_slots().nth(t.slot as usize)) else {
                return;
            };
            let Some(entry) = materials.get(ri.material) else {
                return;
            };
            if writes.len() >= MAX_QUEUED_WRITES {
                return;
            }
            let v = if cover >= 0.999 { value } else { blend_toward(prop.read(&entry.model), value, cover, t.kind.lanes()) };
            writes.push(MaterialWrite { material: ri.material, prop, value: v });
        }
    }
}

/// Write the pose's property samples onto the rig's objects. `nodes` maps rig
/// nodes to scene objects. Material values are queued; call
/// [`flush_material_writes`] (done by `LoadedScene::update_skinning`).
pub fn apply_pose_props(scene: &mut LoadedScene, rig: &Rig, nodes: &[usize], pose: &Pose) {
    if pose.props.is_empty() {
        return;
    }
    let LoadedScene { objects, materials, anim_assets, .. } = scene;
    let mut table = rig.props.lock();
    for s in &pose.props {
        if s.cover < 1e-3 {
            continue;
        }
        let (Some(&obj), Some(t)) = (nodes.get(s.node as usize), table.targets.get_mut(s.chan as usize)) else {
            continue;
        };
        write_target(objects, materials, &mut anim_assets.material_writes, obj, t, s.value, s.cover);
    }
}

/// Write the blendshape samples of a skeletal pose (bound with
/// [`Rig::bind_tracks`], anchored) onto the skinned object `obj`.
pub fn apply_object_props(objects: &mut [SceneObject], materials: &[MaterialEntry], writes: &mut Vec<MaterialWrite>, rig: &Rig, obj: usize, props: &[PropSample]) {
    if props.is_empty() {
        return;
    }
    let mut table = rig.props.lock();
    for s in props {
        if s.cover < 1e-3 {
            continue;
        }
        let Some(t) = table.targets.get_mut(s.chan as usize) else { continue };
        if matches!(t.kind, TrackProperty::BlendShape | TrackProperty::CustomProperty) {
            write_target(objects, materials, writes, obj, t, s.value, s.cover);
        }
    }
}

/// Write `tracks` (an imported clip's blendshape curves) sampled at `t` onto
/// object `obj` at full cover: the classic, controller-less Animator path.
pub fn apply_clip_morphs(objects: &mut [SceneObject], obj: usize, tracks: &[zestengine_assets::AnimTrack], t: f32, length: f32, looping: bool) {
    let mut writes = Vec::new();
    for track in tracks {
        if !matches!(track.property, TrackProperty::BlendShape | TrackProperty::CustomProperty) || track.keys.is_empty() {
            continue;
        }
        let tt = if looping && length > 1e-6 { t.rem_euclid(length) } else { t.clamp(0.0, length.max(0.0)) };
        let mut target = PropTarget::new(0, track.property, 0, &track.name);
        write_target(objects, &[], &mut writes, obj, &mut target, track.sample(tt), 1.0);
    }
}

/// Set one property outright (restoring a preview, applying a snapshot).
pub fn write_prop(scene: &mut LoadedScene, obj: usize, kind: TrackProperty, slot: u8, name: &str, value: [f32; 4]) {
    let LoadedScene { objects, materials, anim_assets, .. } = scene;
    let mut t = PropTarget::new(0, kind, slot, name);
    write_target(objects, materials, &mut anim_assets.material_writes, obj, &mut t, value, 1.0);
}

// ------------------------------------------------------------ editor helpers

/// What the editor can offer to animate on one object: component names,
/// material slots and (given the names) blendshapes.
pub fn component_names(scene: &LoadedScene, obj: usize) -> Vec<&'static str> {
    scene.objects[obj].components.iter().map(|c| c.type_name()).filter(|n| *n != "Animator").collect()
}

/// Number of material slots on an object.
pub fn material_slots(scene: &LoadedScene, obj: usize) -> usize {
    scene.objects[obj].render_slots().count()
}

/// Every property of an object the recorder watches, as (kind, slot, name).
pub fn watched_props(scene: &LoadedScene, obj: usize, shapes: &[String]) -> Vec<(TrackProperty, u8, String)> {
    let mut out = vec![(TrackProperty::Active, 0, String::new())];
    out.extend(component_names(scene, obj).into_iter().map(|n| (TrackProperty::ComponentEnabled, 0, n.to_owned())));
    out.extend(shapes.iter().map(|n| (TrackProperty::BlendShape, 0, n.clone())));
    for slot in 0..material_slots(scene, obj) {
        out.extend(MaterialProp::ALL.into_iter().map(|p| (p.track(), slot as u8, p.name().to_owned())));
    }
    out
}

/// Whether two property values differ enough to count as an edit.
pub fn prop_changed(a: [f32; 4], b: [f32; 4]) -> bool {
    let (a, b) = (a.map(|x| if x.is_nan() { 0.0 } else { x }), b.map(|x| if x.is_nan() { 0.0 } else { x }));
    a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};
    use zestengine_assets::{AnimTrack, KeyInterp, Keyframe};
    use zestengine_core::ObjectId;

    use crate::anim_graph::{ClipRef, Trs, compose_layer};
    use zestengine_assets::ObjectSource;

    fn object(name: &str, parent: Option<usize>, components: Vec<Box<dyn Component>>) -> SceneObject {
        SceneObject {
            net_role: Default::default(),
            component_roles: Default::default(),
            id: ObjectId::new(),
            name: name.into(),
            render: None,
            extra_render: Vec::new(),
            source: ObjectSource::Empty,
            enabled: true,
            static_geometry: false,
            layer: 0,
            parent,
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            components,
        }
    }

    /// Root "Avatar" with children "Hat" and "Body".
    fn scene() -> LoadedScene {
        let mut s = LoadedScene::empty();
        s.objects.push(object("Avatar", None, vec![]));
        s.objects.push(object("Hat", Some(0), vec![]));
        s.objects.push(object("Body", Some(0), vec![Box::new(BlendShapes::default())]));
        s
    }

    fn rig_of(scene: &LoadedScene) -> (Rig, Vec<usize>) {
        crate::anim_graph::object_rig(scene, 0)
    }

    fn key(time: f32, v: f32) -> Keyframe {
        Keyframe { time, value: [v, 0.0, 0.0, 0.0], interp: KeyInterp::Linear, ..Default::default() }
    }

    fn clip(tracks: Vec<AnimTrack>) -> AnimClip {
        AnimClip { length: 1.0, looping: false, tracks, ..Default::default() }
    }

    fn track(target: &str, kind: TrackProperty, name: &str, keys: &[(f32, f32)]) -> AnimTrack {
        let mut t = AnimTrack::prop(target, kind, name, 0);
        for &(time, v) in keys {
            t.keys.push(key(time, v));
        }
        t
    }

    fn sample(rig: &Rig, c: &AnimClip, t: f32) -> Pose {
        let mut p = Pose::rest(rig);
        ClipRef::Asset(c).sample(rig, t, false, &mut p);
        p
    }

    #[test]
    fn bindings_resolve_once_and_survive_a_second_call() {
        let s = scene();
        let (rig, _) = rig_of(&s);
        let c = clip(vec![
            track("Hat", TrackProperty::Active, "", &[(0.0, 1.0)]),
            track("Body", TrackProperty::BlendShape, "Smile", &[(0.0, 0.0), (1.0, 1.0)]),
            track("Missing", TrackProperty::Active, "", &[(0.0, 1.0)]),
        ]);
        let a = rig.prop_bindings(&c).unwrap();
        assert_eq!(a.len(), 2, "the track on a missing object is left out");
        assert_eq!(rig.props.len(), 2);
        let b = rig.prop_bindings(&c).unwrap();
        assert!(Arc::ptr_eq(&a, &b), "second call hits the cache");
        assert!(rig.prop_bindings(&clip(vec![])).is_none());
        // A different clip reuses the same channels.
        let c2 = clip(vec![track("Hat", TrackProperty::Active, "", &[(0.0, 0.0)])]);
        let d = rig.prop_bindings(&c2).unwrap();
        assert_eq!(d[0].chan, a[0].chan);
        assert_eq!(rig.props.len(), 2);
    }

    #[test]
    fn imported_blendshape_curves_drive_the_skinned_object() {
        use zestengine_assets::{AnimationClip, Joint, Skeleton};
        let mut s = scene();
        // The classic Animator path writes the curve at the playback time.
        let tracks = vec![track("Body", TrackProperty::BlendShape, "Smile", &[(0.0, 0.0), (1.0, 1.0)])];
        apply_clip_morphs(&mut s.objects, 2, &tracks, 0.25, 1.0, true);
        assert!((read_prop(&s, 2, TrackProperty::BlendShape, 0, "Smile").unwrap()[0] - 0.25).abs() < 1e-5);
        // A looping clip wraps, a non-looping one holds its last value.
        apply_clip_morphs(&mut s.objects, 2, &tracks, 1.5, 1.0, true);
        assert!((read_prop(&s, 2, TrackProperty::BlendShape, 0, "Smile").unwrap()[0] - 0.5).abs() < 1e-5);
        apply_clip_morphs(&mut s.objects, 2, &tracks, 5.0, 1.0, false);
        assert!((read_prop(&s, 2, TrackProperty::BlendShape, 0, "Smile").unwrap()[0] - 1.0).abs() < 1e-5);

        // The graph path: a skeletal clip carrying the curves puts them in the pose, anchored on the object.
        let skel = Skeleton {
            joints: vec![Joint { name: "Root".into(), parent: None, inverse_bind: glam::Mat4::IDENTITY, rest_translation: Vec3::ZERO, rest_rotation: Quat::IDENTITY, rest_scale: Vec3::ONE }],
            rig: None,
            ..Default::default()
        };
        let rig = Rig::from_skeleton(&skel);
        let imported = AnimationClip { name: "Talk".into(), duration: 1.0, channels: vec![], morph_tracks: tracks, playback: None, };
        let mut pose = Pose::rest(&rig);
        ClipRef::Imported(&imported).sample(&rig, 0.5, false, &mut pose);
        assert_eq!(pose.props.len(), 1, "one blendshape sample");
        write_prop(&mut s, 2, TrackProperty::BlendShape, 0, "Smile", [0.0; 4]);
        let LoadedScene { objects, materials, anim_assets, .. } = &mut s;
        apply_object_props(objects, materials, &mut anim_assets.material_writes, &rig, 2, &pose.props);
        assert!((read_prop(&s, 2, TrackProperty::BlendShape, 0, "Smile").unwrap()[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn editing_a_clip_in_place_rebinds() {
        let s = scene();
        let (rig, _) = rig_of(&s);
        let mut c = clip(vec![track("Hat", TrackProperty::Active, "", &[(0.0, 1.0)])]);
        assert_eq!(rig.prop_bindings(&c).unwrap().len(), 1);
        c.tracks[0].target = "Body".into();
        let b = rig.prop_bindings(&c).unwrap();
        assert_eq!(b.len(), 1);
        assert_eq!(rig.props.target(b[0].chan).unwrap().node, rig.index("Body").unwrap() as u32);
        c.tracks.push(track("Hat", TrackProperty::Active, "", &[(0.0, 0.0)]));
        assert_eq!(rig.prop_bindings(&c).unwrap().len(), 2);
    }

    #[test]
    fn active_and_component_tracks_step() {
        let mut s = scene();
        let (rig, nodes) = rig_of(&s);
        let c = clip(vec![
            track("Hat", TrackProperty::Active, "", &[(0.0, 1.0), (0.5, 0.0)]),
            track("Body", TrackProperty::ComponentEnabled, "Blend Shapes", &[(0.0, 1.0), (0.5, 0.0)]),
        ]);
        let p = sample(&rig, &c, 0.25);
        assert_eq!(p.props.len(), 2);
        apply_pose_props(&mut s, &rig, &nodes, &p);
        assert!(s.objects[1].enabled, "still on before the key");
        assert!(component_enabled(s.objects[2].components[0].as_ref()));
        let p = sample(&rig, &c, 0.75);
        apply_pose_props(&mut s, &rig, &nodes, &p);
        assert!(!s.objects[1].enabled);
        assert!(!component_enabled(s.objects[2].components[0].as_ref()));
        let p = sample(&rig, &c, 0.0);
        apply_pose_props(&mut s, &rig, &nodes, &p);
        assert!(s.objects[1].enabled);
        assert!(component_enabled(s.objects[2].components[0].as_ref()));
    }

    #[test]
    fn an_animator_cannot_switch_itself_off() {
        let mut s = scene();
        s.objects[0].components.push(Box::new(zestengine_core::Animator::default()));
        let (rig, nodes) = rig_of(&s);
        let c = clip(vec![track("", TrackProperty::ComponentEnabled, "Animator", &[(0.0, 0.0)])]);
        apply_pose_props(&mut s, &rig, &nodes, &sample(&rig, &c, 0.0));
        assert!(component_enabled(s.objects[0].components[0].as_ref()));
    }

    #[test]
    fn blendshape_tracks_write_weights_and_create_the_component() {
        let mut s = scene();
        let (rig, nodes) = rig_of(&s);
        let c = clip(vec![track("Body", TrackProperty::BlendShape, "Smile", &[(0.0, 0.0), (1.0, 1.0)])]);
        apply_pose_props(&mut s, &rig, &nodes, &sample(&rig, &c, 0.5));
        let w = |s: &LoadedScene| s.objects[2].components[0].as_any().downcast_ref::<BlendShapes>().unwrap().get("Smile");
        assert!((w(&s) - 0.5).abs() < 1e-5);
        apply_pose_props(&mut s, &rig, &nodes, &sample(&rig, &c, 1.0));
        assert!((w(&s) - 1.0).abs() < 1e-3, "a one-shot clip ends a hair before its length");
        // An object without the component gets one.
        let c = clip(vec![track("Hat", TrackProperty::BlendShape, "Blink", &[(0.0, 0.25)])]);
        apply_pose_props(&mut s, &rig, &nodes, &sample(&rig, &c, 0.0));
        let b = s.objects[1].components.iter().find_map(|c| c.as_any().downcast_ref::<BlendShapes>()).unwrap();
        assert_eq!(b.get("Blink"), 0.25);
    }

    #[test]
    fn sprite_swap_tracks_write_the_resolver_hash_and_ignore_unusable_values() {
        use zestengine_core::{SpriteResolver, sprite_swap_hash};
        let mut s = scene();
        s.objects[1].components.push(Box::new(SpriteResolver::default()));
        let (rig, nodes) = rig_of(&s);
        let (happy, sad) = (sprite_swap_hash("Face", "Happy"), sprite_swap_hash("Face", "Sad"));
        let mut t = AnimTrack::prop("Hat", TrackProperty::SpriteSwap, "", 0);
        t.keys.push(Keyframe { time: 0.0, value: [happy as f32, 0.0, 0.0, 0.0], interp: KeyInterp::Step, ..Default::default() });
        t.keys.push(Keyframe { time: 0.5, value: [sad as f32, 0.0, 0.0, 0.0], interp: KeyInterp::Step, ..Default::default() });
        let c = clip(vec![t]);
        let hash = |s: &LoadedScene| s.objects[1].components.iter().find_map(|c| c.as_any().downcast_ref::<SpriteResolver>()).unwrap().hash;
        apply_pose_props(&mut s, &rig, &nodes, &sample(&rig, &c, 0.25));
        assert_eq!(hash(&s), happy);
        apply_pose_props(&mut s, &rig, &nodes, &sample(&rig, &c, 0.75));
        assert_eq!(hash(&s), sad);
        assert_eq!(read_prop(&s, 1, TrackProperty::SpriteSwap, 0, "").unwrap()[0] as u32, sad);
        // A zero or out-of-range key leaves the hash alone; so does an object without a resolver.
        write_prop(&mut s, 1, TrackProperty::SpriteSwap, 0, "", [0.0; 4]);
        assert_eq!(hash(&s), sad);
        write_prop(&mut s, 1, TrackProperty::SpriteSwap, 0, "", [1.0e9, 0.0, 0.0, 0.0]);
        assert_eq!(hash(&s), sad);
        assert!(read_prop(&s, 2, TrackProperty::SpriteSwap, 0, "").is_none());
        write_prop(&mut s, 2, TrackProperty::SpriteSwap, 0, "", [happy as f32, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn bone_tracks_drive_2d_bone_objects_through_the_ordinary_transform_path() {
        // 2D bones are scene objects, so position / rotation / scale tracks by path need no new track kind.
        let mut s = scene();
        let (rig, nodes) = rig_of(&s);
        let rot = |a: f32| Keyframe { time: 0.0, value: Quat::from_rotation_z(a).to_array(), interp: KeyInterp::Linear, ..Default::default() };
        let mut r = AnimTrack::new("Hat", TrackProperty::Rotation);
        r.keys.push(rot(0.0));
        r.keys.push(Keyframe { time: 1.0, ..rot(1.0) });
        let mut p = AnimTrack::new("Hat", TrackProperty::Translation);
        p.keys.push(Keyframe { time: 0.0, value: [0.0; 4], interp: KeyInterp::Linear, ..Default::default() });
        p.keys.push(Keyframe { time: 1.0, value: [2.0, 4.0, 0.0, 0.0], interp: KeyInterp::Linear, ..Default::default() });
        let c = clip(vec![r, p]);
        let pose = sample(&rig, &c, 0.5);
        crate::anim_graph::apply_pose_to_objects(&mut s, &nodes, &pose);
        let z = s.objects[1].rotation.to_euler(glam::EulerRot::ZYX).0;
        assert!((z - 0.5).abs() < 0.02, "half way through a 1 radian turn: {z}");
        assert!((s.objects[1].translation - Vec3::new(1.0, 2.0, 0.0)).length() < 1e-3);
    }

    #[test]
    fn restoring_an_absent_weight_removes_it() {
        let mut s = scene();
        assert!(read_prop(&s, 2, TrackProperty::BlendShape, 0, "Smile").unwrap()[0].is_nan());
        write_prop(&mut s, 2, TrackProperty::BlendShape, 0, "Smile", [0.5, 0.0, 0.0, 0.0]);
        assert_eq!(read_prop(&s, 2, TrackProperty::BlendShape, 0, "Smile").unwrap()[0], 0.5);
        write_prop(&mut s, 2, TrackProperty::BlendShape, 0, "Smile", [f32::NAN, 0.0, 0.0, 0.0]);
        assert!(read_prop(&s, 2, TrackProperty::BlendShape, 0, "Smile").unwrap()[0].is_nan());
        assert_eq!(read_prop(&s, 1, TrackProperty::Active, 0, "").unwrap()[0], 1.0);
        assert!(read_prop(&s, 1, TrackProperty::MaterialFloat, 0, "metallic").is_none(), "no material on the object");
    }

    #[test]
    fn fades_scale_coverage_and_keep_the_value() {
        let s = scene();
        let (rig, _) = rig_of(&s);
        let a = clip(vec![track("Body", TrackProperty::BlendShape, "Smile", &[(0.0, 1.0)])]);
        let b = clip(vec![]);
        let pa = sample(&rig, &a, 0.0);
        let pb = sample(&rig, &b, 0.0);
        assert!(pb.props.is_empty());
        let mid = crate::anim_graph::blend_pose(&pa, &pb, 0.25);
        assert_eq!(mid.props.len(), 1);
        assert!((mid.props[0].cover - 0.75).abs() < 1e-5);
        assert_eq!(mid.props[0].value[0], 1.0);
        // Both driven: ordinary weighted average, full coverage.
        let a2 = clip(vec![track("Body", TrackProperty::BlendShape, "Smile", &[(0.0, 0.0)])]);
        let both = crate::anim_graph::blend_pose(&pa, &sample(&rig, &a2, 0.0), 0.25);
        assert!((both.props[0].value[0] - 0.75).abs() < 1e-5);
        assert!((both.props[0].cover - 1.0).abs() < 1e-5);
        // Fully faded out: nothing left.
        assert!(crate::anim_graph::blend_pose(&pa, &pb, 1.0).props.is_empty());
    }

    #[test]
    fn partial_coverage_blends_toward_the_current_value() {
        let mut s = scene();
        let (rig, nodes) = rig_of(&s);
        write_prop(&mut s, 2, TrackProperty::BlendShape, 0, "Smile", [0.0; 4]);
        let c = clip(vec![track("Body", TrackProperty::BlendShape, "Smile", &[(0.0, 1.0)])]);
        let mut p = sample(&rig, &c, 0.0);
        p.props[0].cover = 0.5;
        apply_pose_props(&mut s, &rig, &nodes, &p);
        assert!((read_prop(&s, 2, TrackProperty::BlendShape, 0, "Smile").unwrap()[0] - 0.5).abs() < 1e-5);
        // A switch with half coverage flips at the half-way point.
        let c = clip(vec![track("Hat", TrackProperty::Active, "", &[(0.0, 0.0)])]);
        let mut p = sample(&rig, &c, 0.0);
        p.props[0].cover = 0.4;
        apply_pose_props(&mut s, &rig, &nodes, &p);
        assert!(s.objects[1].enabled);
        p.props[0].cover = 0.6;
        apply_pose_props(&mut s, &rig, &nodes, &p);
        assert!(!s.objects[1].enabled);
    }

    #[test]
    fn layers_composite_by_weight_and_mask() {
        let s = scene();
        let (rig, _) = rig_of(&s);
        let base_clip = clip(vec![
            track("Body", TrackProperty::BlendShape, "Smile", &[(0.0, 0.0)]),
            track("Hat", TrackProperty::Active, "", &[(0.0, 1.0)]),
        ]);
        let top_clip = clip(vec![
            track("Body", TrackProperty::BlendShape, "Smile", &[(0.0, 1.0)]),
            track("Hat", TrackProperty::Active, "", &[(0.0, 0.0)]),
        ]);
        let base = sample(&rig, &base_clip, 0.0);
        let top = sample(&rig, &top_clip, 0.0);
        let smile = rig.prop_bindings(&base_clip).unwrap().iter().find(|b| b.node == rig.index("Body").unwrap() as u32).unwrap().chan;
        let at = |p: &Pose, chan: u32| p.props.iter().find(|s| s.chan == chan).copied().unwrap();

        let half = compose_layer(&rig, base.clone(), &top, 0.5, zestengine_assets::LayerBlend::Override, &Default::default());
        assert!((at(&half, smile).value[0] - 0.5).abs() < 1e-5);
        let full = compose_layer(&rig, base.clone(), &top, 1.0, zestengine_assets::LayerBlend::Override, &Default::default());
        assert_eq!(at(&full, smile).value[0], 1.0);
        let none = compose_layer(&rig, base.clone(), &top, 0.0, zestengine_assets::LayerBlend::Override, &Default::default());
        assert_eq!(at(&none, smile).value[0], 0.0);

        // A mask on "Hat" lets only the hat through.
        let masked = compose_layer(&rig, base.clone(), &top, 1.0, zestengine_assets::LayerBlend::Override, &zestengine_assets::AvatarMask::from_legacy(&["Hat".to_string()]));
        assert_eq!(at(&masked, smile).value[0], 0.0, "Body is outside the mask");
        let hat = rig.prop_bindings(&base_clip).unwrap().iter().find(|b| b.node == rig.index("Hat").unwrap() as u32).unwrap().chan;
        assert_eq!(at(&masked, hat).value[0], 0.0, "Hat is inside the mask");

        // A channel only the layer drives is added at the layer's share.
        let empty = Pose::rest(&rig);
        let added = compose_layer(&rig, empty, &top, 0.5, zestengine_assets::LayerBlend::Additive, &Default::default());
        assert!((at(&added, smile).cover - 0.5).abs() < 1e-5);
    }

    #[test]
    fn write_defaults_off_holds_undriven_channels() {
        let mut props = vec![PropSample { chan: 2, node: 1, value: [1.0, 0.0, 0.0, 0.0], cover: 1.0 }];
        let held = [
            PropSample { chan: 0, node: 1, value: [0.5, 0.0, 0.0, 0.0], cover: 1.0 },
            PropSample { chan: 2, node: 1, value: [9.0, 0.0, 0.0, 0.0], cover: 1.0 },
        ];
        hold_props(&mut props, &held);
        assert_eq!(props.len(), 2);
        assert_eq!(props[0].chan, 0);
        assert_eq!(props[1].value[0], 1.0, "driven channels keep the new value");
    }

    #[test]
    fn material_names_round_trip_and_write_live_params() {
        for p in MaterialProp::ALL {
            assert_eq!(MaterialProp::from_name(p.name()), Some(p));
            assert!(p.track().is_object_prop());
        }
        assert!(MaterialProp::from_name("nope").is_none());
        let mut params = MaterialParams::default();
        MaterialProp::BaseColor.write(&mut params, [0.1, 0.2, 0.3, 0.4]);
        MaterialProp::Roughness.write(&mut params, [0.9, 0.0, 0.0, 0.0]);
        MaterialProp::AlbedoOffset.write(&mut params, [0.25, 0.5, 0.0, 0.0]);
        assert_eq!(params.base_color, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(params.roughness, 0.9);
        assert_eq!(params.albedo_offset, [0.25, 0.5]);
        assert_eq!(MaterialProp::EmissionColor.track().lanes(), 4);
        assert_eq!(MaterialProp::OrmOffset.track().lanes(), 2);
    }

    #[test]
    fn material_tracks_without_a_material_write_nothing() {
        let mut s = scene();
        let (rig, nodes) = rig_of(&s);
        let mut t = AnimTrack::prop("Body", TrackProperty::MaterialFloat, "roughness", 0);
        t.keys.push(key(0.0, 0.2));
        let c = clip(vec![t]);
        apply_pose_props(&mut s, &rig, &nodes, &sample(&rig, &c, 0.0));
        assert!(s.anim_assets.material_writes.is_empty());
    }

    #[test]
    fn transform_and_property_tracks_share_a_clip() {
        let mut s = scene();
        let (rig, nodes) = rig_of(&s);
        let mut mv = AnimTrack::new("Hat", TrackProperty::Translation);
        mv.keys.push(Keyframe { time: 0.0, value: [2.0, 0.0, 0.0, 0.0], interp: KeyInterp::Linear, ..Default::default() });
        let c = clip(vec![mv, track("Hat", TrackProperty::Active, "", &[(0.0, 0.0)])]);
        let p = sample(&rig, &c, 0.0);
        assert_eq!(p.trs[1].t.x, 2.0);
        assert_eq!(p.props.len(), 1);
        crate::anim_graph::apply_pose_to_objects(&mut s, &nodes, &p);
        apply_pose_props(&mut s, &rig, &nodes, &p);
        assert_eq!(s.objects[1].translation.x, 2.0);
        assert!(!s.objects[1].enabled);
        let _ = Trs::default();
    }

    #[test]
    fn changed_ignores_noise_and_absent_weights() {
        assert!(!prop_changed([1.0; 4], [1.00001; 4]));
        assert!(prop_changed([1.0, 0.0, 0.0, 0.0], [0.0; 4]));
        assert!(!prop_changed([f32::NAN, 0.0, 0.0, 0.0], [0.0; 4]));
    }

    #[test]
    fn watched_props_cover_object_components_shapes_and_materials() {
        let s = scene();
        let w = watched_props(&s, 2, &["Smile".to_string()]);
        assert!(w.contains(&(TrackProperty::Active, 0, String::new())));
        assert!(w.contains(&(TrackProperty::ComponentEnabled, 0, "Blend Shapes".to_string())));
        assert!(w.contains(&(TrackProperty::BlendShape, 0, "Smile".to_string())));
        assert!(!w.iter().any(|(k, ..)| k.uses_slot()), "no render slots, no material props");
    }

    #[test]
    fn custom_property_tracks_write_and_create_the_component() {
        use zestengine_core::{CustomProperties, CustomValue};
        let mut s = scene();
        // The classic Animator path: imported property tracks sampled at t onto object 0.
        let speed = track("Avatar", TrackProperty::CustomProperty, "Speed", &[(0.0, 1.0), (1.0, 5.0)]);
        let mut count = track("Avatar", TrackProperty::CustomProperty, "Count", &[(0.0, 2.0), (1.0, 4.0)]);
        count.keys.iter_mut().for_each(|k| k.interp = KeyInterp::Step);
        let dir_y = track("Avatar", TrackProperty::CustomProperty, "Dir.y", &[(0.0, 0.0), (1.0, 2.0)]);
        let tracks = vec![speed, count, dir_y];
        // No component yet: the first write creates it.
        apply_clip_morphs(&mut s.objects, 0, &tracks, 0.5, 1.0, false);
        let get = |s: &LoadedScene| s.objects[0].components.iter().find_map(|c| c.as_any().downcast_ref::<CustomProperties>()).cloned().unwrap();
        let p = get(&s);
        assert!((p.float("Speed") - 3.0).abs() < 1e-5);
        assert_eq!(p.get("Dir"), Some(CustomValue::Vec3([0.0, 1.0, 0.0])));
        // Existing typed values keep their type: the int rounds, the step holds the first key until the next.
        s.objects[0].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<CustomProperties>()).unwrap().set("Count", CustomValue::Int(0));
        apply_clip_morphs(&mut s.objects, 0, &tracks, 0.5, 1.0, false);
        assert_eq!(get(&s).get("Count"), Some(CustomValue::Int(2)));
        apply_clip_morphs(&mut s.objects, 0, &tracks, 1.0, 1.0, false);
        assert_eq!(get(&s).get("Count"), Some(CustomValue::Int(4)));
        assert_eq!(read_prop(&s, 0, TrackProperty::CustomProperty, 0, "Speed").unwrap()[0], 5.0);
        // Restoring a property that did not exist drops it again.
        write_prop(&mut s, 0, TrackProperty::CustomProperty, 0, "Speed", [f32::NAN, 0.0, 0.0, 0.0]);
        assert!(get(&s).get("Speed").is_none());
    }

    #[test]
    fn custom_property_tracks_blend_by_coverage_through_a_pose() {
        use zestengine_core::CustomProperties;
        let mut s = scene();
        s.objects[2].components.push(Box::new(CustomProperties::default()));
        write_prop(&mut s, 2, TrackProperty::CustomProperty, 0, "Speed", [0.0; 4]);
        let (rig, nodes) = rig_of(&s);
        let c = clip(vec![track("Body", TrackProperty::CustomProperty, "Speed", &[(0.0, 10.0)])]);
        let mut p = sample(&rig, &c, 0.0);
        for q in &mut p.props {
            q.cover = 0.5;
        }
        apply_pose_props(&mut s, &rig, &nodes, &p);
        let v = s.objects[2].components.iter().find_map(|c| c.as_any().downcast_ref::<CustomProperties>()).unwrap().float("Speed");
        assert!((v - 5.0).abs() < 1e-5, "half coverage blends toward the animated value: {v}");
    }
}
