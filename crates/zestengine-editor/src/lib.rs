//! zestengine-editor: in-engine editor panels (egui-only, renderer-agnostic).
//!
//! Panels render into plain `Ui`s so the engine can host them as dock tabs:
//! - Inspector: selected object (transform/mesh/material slots) or file
//! - Scene: object list
//! - Files: project browser with drag & drop and asset creation
//!
//! The full world/avatar editor (gizmo toolbox, VRM tooling) keeps growing
//! here through M5.

mod audio_ui;
mod blendshape_ui;
mod renderer_ui;
pub use renderer_ui::{ModelChild, ModelRootInfo, MaterialSectionModel, MaterialSource, MeshFilterInfo, RendererEvent, RendererInfo, SlotInfo, group_slots, section_open, section_title, set_section_open};
pub use blendshape_ui::blendshapes_section;
pub mod actions_ui;
pub mod avatar_rig;
pub mod avatar_ui;
mod cloth_ui;
pub use cloth_ui::{CLOTH_PAINT_HINT, ClothPaintTool};
mod terrain_ui;
pub use terrain_ui::{TerrainBrushMode, TerrainPaintTool};
mod ik_ui;
mod face_ui;
mod constraint_ui;
mod water_ui;
pub use water_ui::{RIVER_PATH_HINT, RiverHandle, RiverPathEdit, river_handle};
mod physics_ui;
mod input_ui;
pub use input_ui::{BindingsEditor, RebindTarget, bindings_editor_ui};
mod waterfall_ui;
mod water_container_ui;
pub use physics_ui::{physics_settings_ui, physmat_editor_ui};
mod selection_ui;
pub use selection_ui::selection_outline_settings_ui;
mod code_editor;
mod components;
mod kit_ui;
mod wasm_ui;
pub mod icons;
pub mod theme;
pub mod widgets;
pub mod node_canvas;
pub mod shot;
mod vim;
mod file_browser;
mod inspector;
mod scene_panel;
pub mod prefab_ui;
mod sections;
pub use sections::material_editor_ui;
mod texture_slots;
pub use texture_slots::{SLOT_FIX_KEY, SLOT_WARNINGS_KEY, SlotFix, SlotWarning, slot_warning_ui};
mod shader_ui;
mod rig2d_ui;
mod sprite2d_ui;
mod sprite_shape_ui;
mod tilemap_ui;
mod tile_palette_ui;
pub use tilemap_ui::{delete_rule, duplicate_rule, grid_lines, move_rule, preview_match, rule_slot_layout, rule_tile_ui, tile_asset_ui, tile_file_ui};
pub use tile_palette_ui::{MapList, PaletteKey, TilePaletteState, apply_key, brush_from_palette, palette_from_sprites, palette_key, tile_palette_window, write_palette};
pub use sprite2d_ui::{SORTING_LAYERS_KEY, publish_sorting_layers, sorting_layers_ui};
pub use sprite_shape_ui::{SHAPE_HINT, ShapeEditState, sprite_shape_profile_ui};

pub use code_editor::{
    CodeActionItem, CodeEditor, CodeEditorResponse, CompletionItem, CompletionState, HoverState, ReferenceItem,
};
// Editor-only component UI. The runtime component types (Component,
// ComponentRegistry, the structs, ComponentCtx, Transform, …) come from
// zestengine-core. Import those directly, not through the editor.
pub use components::{
    ASSET_TEMPLATES_KEY, AUTHORED_COLLISION_KEY, AssetEditor, AssetTemplate, ComponentsResponse, DRAG_FILE_KEY, DRAG_OBJECT_KEY, DRAG_OBJECT_NAME_KEY, EditorComponents,
    Gizmo, GizmoEdit, GizmoSpec, Inspect, InspectCtx, COMPONENT_ROLES_KEY, MIXED_COMPONENTS_KEY, OPEN_ANIM_CONTROLLER_KEY,
    OPEN_FILE_KEY, OPEN_POSE_EDITOR_KEY, PING_FILE_KEY, PING_OBJECT_KEY, PLAY_MODE_KEY, PROJECT_ROOT_KEY, SELECT_OBJECT_KEY, asset_browse_button, asset_chip, asset_new_button, components_ui,
    edit_bt_node, resolve_slot,
};
pub use audio_ui::{AudioSettingsResponse, audio_settings_ui};
pub use file_browser::{DRAG_SPRITE_KEY, FileBrowser, FileBrowserResponse, SELECTED_SPRITE_KEY, SpriteChild, SpriteDrag, set_atlas_probe, set_sprite_children_probe, set_sprite_probe};
/// Re-export the icon font so the engine's editor UI can use the same glyphs
/// (no emojis) the file browser does.
pub use egui_phosphor;

/// Install the editor fonts: IBM Plex Sans / Mono first, egui's defaults
/// behind them, and the Phosphor icon font. Call once at startup.
pub fn install_icon_font(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    theme::install_fonts(&mut fonts);
    egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
    ctx.set_fonts(fonts);
}
// The egui-free material/shader data models live in zestengine-core now; re-export
// them so existing `zestengine_editor::MaterialModel`-style paths keep working.
pub use zestengine_core::{
    AlphaModeModel, MaterialModel, ShaderCondUi, ShaderGroupUi, ShaderPresetUi, ShaderPropKindUi,
    ShaderPropUi, ShaderUiInfo,
};
pub use inspector::{
    CodeDiagnostic, InspectorContent, InspectorPanel, InspectorResponse, MixedInfo, ObjectHeaderResponse,
    ObjectInfoModel, TransformModel, space_tooltip,
};
pub use scene_panel::{PrefabMark, PrefabMenu, PrefabRole, SceneObjectRow, ScenePanel, ScenePanelResponse, SpawnKind, UiPreset, ViewKeyAction, ViewRequest, toggle_hide_request, view_flags, view_key_action};

// The editor application moved here from zestengine-engine (the engine keeps
// only the runtime).
mod mask_silhouette;
mod mask_ui;
mod ron_fields;
mod editor_water;
mod placement_ui;
mod log_select;
mod camera;
mod sky_probe;
mod crash;
mod gizmo;
mod icon;
mod lsp;
mod splash;
mod tasks;
mod plugin_gate;
mod bake_store;
mod undo;
mod zestsurface;
mod command_palette;
pub mod editor_app;
mod editor_prefs;
mod sprite_editor;
mod sprite_atlas_ui;
mod skinning_editor;
mod sprite_inspector;
mod texture_inspector;
mod net_window;
mod nvim_embed;
mod perf_window;
mod new_project_ui;
mod river_edit;
mod cloth_paint;
mod terrain_paint;
mod tile_edit;
mod shape_edit;
mod ui_edit;
mod ui_editor;
mod anim_canvas;
mod anim_controller_editor;
mod asset_preview;
mod model_preview;
mod rig_editor;
mod rig_extras;
mod rig_muscles;
mod geometry_cache_ui;
mod anim_clip_editor;
mod sequencer_editor;
mod shader_graph_editor;
pub mod unity_convert_ui;
pub mod pack_import_ui;
pub mod model_import_editor;
pub mod anim_behaviour_ui;

pub use editor_app::{AppConfig, run};

pub mod content_build;

#[cfg(test)]
#[path = "shots/materials.rs"]
pub mod shots_materials;
#[cfg(test)]
#[path = "shots/inspector.rs"]
pub mod shots_inspector;

#[cfg(test)]
pub mod test_support {
    use glam::{Mat4, Quat, Vec3};
    use zestengine_assets::human_rig::{AnimationType, HumanBone, RigImport};
    use zestengine_assets::{Joint, Skeleton};

    /// Hips, spine, head, arms with shoulders and legs: a T-pose humanoid facing +Z.
    pub fn skeleton() -> Skeleton {
        let mut joints: Vec<Joint> = Vec::new();
        let mut human: Vec<(HumanBone, String)> = Vec::new();
        let mut add = |b: HumanBone, parent: Option<&str>, t: Vec3| {
            let p = parent.map(|n| joints.iter().position(|j| j.name == n).unwrap());
            joints.push(Joint { name: b.name().into(), parent: p, inverse_bind: Mat4::IDENTITY, rest_translation: t, rest_rotation: Quat::IDENTITY, rest_scale: Vec3::ONE });
            human.push((b, b.name().into()));
        };
        add(HumanBone::Hips, None, Vec3::new(0.0, 0.98, 0.0));
        add(HumanBone::Spine, Some("Hips"), Vec3::new(0.0, 0.12, 0.0));
        add(HumanBone::Chest, Some("Spine"), Vec3::new(0.0, 0.3, 0.0));
        add(HumanBone::Head, Some("Chest"), Vec3::new(0.0, 0.3, 0.0));
        for (side, x) in [("Left", 1.0f32), ("Right", -1.0)] {
            let bone = |n: &str| HumanBone::from_name(&format!("{side}{n}")).unwrap();
            let nm = |n: &str| format!("{side}{n}");
            add(bone("Shoulder"), Some("Chest"), Vec3::new(0.04 * x, 0.1, 0.0));
            add(bone("UpperArm"), Some(&nm("Shoulder")), Vec3::new(0.08 * x, 0.0, 0.0));
            add(bone("LowerArm"), Some(&nm("UpperArm")), Vec3::new(0.28 * x, 0.0, 0.0));
            add(bone("Hand"), Some(&nm("LowerArm")), Vec3::new(0.26 * x, 0.0, 0.0));
            add(bone("UpperLeg"), Some("Hips"), Vec3::new(0.1 * x, 0.0, 0.0));
            add(bone("LowerLeg"), Some(&nm("UpperLeg")), Vec3::new(0.0, -0.45, 0.0));
            add(bone("Foot"), Some(&nm("LowerLeg")), Vec3::new(0.0, -0.45, 0.0));
        }
        let rig = RigImport { animation_type: AnimationType::Humanoid, human, ..Default::default() };
        Skeleton { joints, rig: Some(rig), ..Default::default() }
    }
}

pub mod avatar_edit;
