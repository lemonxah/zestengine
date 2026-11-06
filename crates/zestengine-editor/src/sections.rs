//! Reusable material editor: shader picker + collapsible feature sections.
//! Headers toggle on title click as well as the arrow; sections with a
//! master enable checkbox follow the Poiyomi pattern (off = feature compiled
//! out of the shader variant).

use std::path::PathBuf;

use egui::collapsing_header::{CollapsingState, paint_default_icon};
use egui::{ComboBox, DragValue, Label, RichText, Sense, Ui};

use zestengine_core::{AlphaModeModel, BlendModeModel, MatcapBlend, MaterialModel, ShaderUiInfo, StencilCompareModel, StencilFaceModel, StencilOpModel};
use egui_phosphor::regular as ph;

/// Section metadata for search: (title, keywords).
const SECTIONS: [(&str, &[&str]); 8] = [
    (
        "Base",
        &[
            "color",
            "albedo",
            "metallic",
            "roughness",
            "occlusion",
            "ao",
        ],
    ),
    (
        "Toon Shading",
        &["toon", "steps", "bands", "blend", "ramp", "cel"],
    ),
    ("Emission", &["emission", "glow", "emissive", "intensity"]),
    (
        "Transparency",
        &["alpha", "cutout", "blend", "opacity", "cutoff"],
    ),
    (
        "Geometry",
        &["normal", "double", "sided", "culling", "bump"],
    ),
    ("Matcaps", &["matcap", "matcaps", "sphere", "reflection"]),
    ("Detail", &["detail", "second", "overlay", "tiling", "mask"]),
    (
        "Translucency & Transmission",
        &["translucency", "transmission", "thickness", "glass", "subsurface", "backlight", "leaf", "refraction"],
    ),
];

const MATCAP_TITLES: [&str; 3] = ["Matcap 1", "Matcap 2", "Matcap 3"];


/// One texture-slot row: an asset chip (drag an image from the file
/// browser, browse, click to show, double-click to open) holding an `@<id>`
/// reference. Returns true when the slot changed (in place).
pub(crate) fn texture_row(
    ui: &mut Ui,
    label: &str,
    idx: usize,
    slot: &mut Option<PathBuf>,
    _tex_dropped: &mut Option<(usize, PathBuf)>,
) -> bool {
    // What the engine can load as a texture.
    const IMAGES: [&str; 12] = ["png", "jpg", "jpeg", "tga", "bmp", "dds", "hdr", "exr", "ktx", "ktx2", "webp", "gif"];
    let mut value = slot.as_ref().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
    let salt = format!("texture-slot-{idx}");
    let changed = crate::widgets::property_row(ui, label, |ui| crate::components::asset_chip(ui, &salt, &mut value, "drop an image", &IMAGES));
    if changed {
        *slot = (!value.is_empty()).then(|| PathBuf::from(value));
    }
    // Import-settings mismatch for this slot's texture, if the engine found one.
    crate::texture_slots::show_for_slot(ui, idx);
    changed
}

/// Per-texture UV transform: tiling (scale) + offset, two compact rows. Returns
/// true if either changed.
pub(crate) fn uv_transform_row(ui: &mut Ui, tiling: &mut [f32; 2], offset: &mut [f32; 2]) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new("Tiling").small().weak());
        changed |= ui
            .add(DragValue::new(&mut tiling[0]).speed(0.01).prefix("x "))
            .changed();
        changed |= ui
            .add(DragValue::new(&mut tiling[1]).speed(0.01).prefix("y "))
            .changed();
        ui.separator();
        ui.label(RichText::new("Offset").small().weak());
        changed |= ui
            .add(DragValue::new(&mut offset[0]).speed(0.005).prefix("x "))
            .changed();
        changed |= ui
            .add(DragValue::new(&mut offset[1]).speed(0.005).prefix("y "))
            .changed();
    });
    changed
}

fn section_matches(search: &str, index: usize) -> bool {
    if search.is_empty() {
        return true;
    }
    let (title, keywords) = SECTIONS[index];
    title.to_lowercase().contains(search) || keywords.iter().any(|k| k.contains(search))
}

/// Full material editor. Returns true if anything changed. Texture-slot drops
/// are reported via `tex_dropped` ((slot 0..12, absolute image path)); the
/// engine converts the path to project-relative and assigns it. `salt` must
/// be unique per visible editor instance: several materials can be on screen
/// at once (one section per material), and egui stateful widgets (dropdown
/// popups, foldouts) need distinct ids or they fight over shared state.
pub fn material_editor_ui(
    ui: &mut Ui,
    search: &mut String,
    m: &mut MaterialModel,
    shaders: &[&str],
    shader_info: Option<&ShaderUiInfo>,
    tex_dropped: &mut Option<(usize, PathBuf)>,
    salt: &str,
) -> bool {
    ui.push_id(salt, |ui| {
    let mut changed = false;

    // Shader picker. Only registered shaders are selectable; an unknown
    // shader (e.g. from a hand-edited .material) shows as an error entry.
    ui.horizontal(|ui| {
        ui.label("Shader");
        let known = shaders.contains(&m.shader.as_str());
        let display = if known {
            m.shader.clone()
        } else {
            format!("{} {}", ph::WARNING, m.shader)
        };
        ComboBox::from_id_salt("shader-select")
            .selected_text(display)
            .show_ui(ui, |ui| {
                for &shader in shaders {
                    if ui
                        .selectable_value(&mut m.shader, shader.to_owned(), shader)
                        .changed()
                    {
                        // New shader = new property layout; defaults refill.
                        m.custom_values.clear();
                        changed = true;
                    }
                }
            });
        if !known {
            ui.label(RichText::new("unknown shader").color(ui.visuals().error_fg_color));
        }
    });

    if m.shader != "standard" {
        changed |= crate::shader_ui::custom_shader_ui(ui, search, m, shader_info, tex_dropped);
        // Fixed slots every custom shader can sample (t_albedo, t_normal, t_orm, t_emission).
        section(ui, "Material Textures", None, &mut changed, |ui, changed| {
            for (slot, label) in [(0, "Albedo"), (1, "Normal Map"), (2, "ORM"), (3, "Emission")] {
                *changed |= texture_row(ui, label, slot, m.textures.slot_mut(slot).unwrap(), tex_dropped);
            }
        });
        // Pipeline state still comes from the standard feature set.
        section(ui, "Transparency", None, &mut changed, |ui, changed| {
            alpha_mode_ui(ui, m, changed);
        });
        section(ui, "Geometry", None, &mut changed, |ui, changed| {
            cull_ui(ui, m, changed);
        });
        section(ui, "Stencil", None, &mut changed, |ui, changed| {
            stencil_ui(ui, m, changed);
        });
        // Render queue is the last entry for every shader (draw-order priority).
        ui.separator();
        render_queue_ui(ui, m, &mut changed);
        return changed;
    }

    ui.horizontal(|ui| {
        ui.label(ph::MAGNIFYING_GLASS);
        ui.text_edit_singleline(search);
    });
    ui.separator();

    let search_lower = search.to_lowercase();

    if section_matches(&search_lower, 0) {
        section(ui, "Base", None, &mut changed, |ui, changed| {
            property_row(ui, "Base Color", changed, |ui| {
                ui.color_edit_button_rgba_unmultiplied(&mut m.base_color)
            });
            *changed |= texture_row(ui, "Albedo", 0, m.textures.slot_mut(0).unwrap(), tex_dropped);
            *changed |= uv_transform_row(ui, &mut m.albedo_tiling, &mut m.albedo_offset);
            property_row(ui, "Metallic", changed, |ui| {
                crate::widgets::slider(ui, egui::Slider::new(&mut m.metallic, 0.0..=1.0))
            });
            property_row(ui, "Roughness", changed, |ui| {
                crate::widgets::slider(ui, egui::Slider::new(&mut m.roughness, 0.0..=1.0))
            });
            property_row(ui, "Occlusion", changed, |ui| {
                crate::widgets::slider(ui, egui::Slider::new(&mut m.occlusion_strength, 0.0..=1.0))
            });
            property_row(ui, "Reflection", changed, |ui| {
                crate::widgets::slider(ui, egui::Slider::new(&mut m.reflection_intensity, 0.0..=2.0))
                    .on_hover_text(
                        "Per-material reflection strength (1 = default, 0 = matte).",
                    )
            });
            property_row(ui, "Technique", changed, |ui| {
                let opts: Vec<(zestengine_core::SurfaceReflection, &str)> = [
                    (zestengine_core::SurfaceReflection::Environment, "Environment"),
                    (zestengine_core::SurfaceReflection::Probes, "Probes"),
                    (zestengine_core::SurfaceReflection::Off, "Off"),
                ]
                .into();
                crate::widgets::segmented(ui, &mut m.reflection_mode, &opts).on_hover_text(
                    "Environment = the scene's global reflection technique (Environment -> Reflections). \
                     Probes = baked reflection probes only: the SSR / ray-traced resolve skips this material. \
                     Off = no reflections at all. Planar mirrors come from the object's Planar Reflection component.",
                )
            });
            *changed |= texture_row(
                ui,
                "ORM (Occl/Rough/Metal)",
                2,
                m.textures.slot_mut(2).unwrap(),
                tex_dropped,
            );
            *changed |= uv_transform_row(ui, &mut m.orm_tiling, &mut m.orm_offset);
            ui.label(
                RichText::new(
                    "Or split maps below — each multiplies its packed-ORM channel \
                     (leave ORM empty to use them alone). They share the ORM tiling.",
                )
                .small()
                .weak(),
            );
            *changed |= texture_row(ui, "Ambient Occlusion", 12, m.textures.slot_mut(12).unwrap(), tex_dropped);
            let has_ao = m.textures.ao.is_some();
            ui.add_enabled_ui(has_ao, |ui| {
                *changed |= ui.checkbox(&mut m.ao_invert, "Invert").changed();
            });
            *changed |= texture_row(ui, "Roughness", 13, m.textures.slot_mut(13).unwrap(), tex_dropped);
            let has_rough = m.textures.roughness.is_some();
            ui.add_enabled_ui(has_rough, |ui| {
                *changed |= ui
                    .checkbox(&mut m.roughness_invert, "Invert (smoothness -> roughness)")
                    .changed();
            });
            *changed |= texture_row(ui, "Metallic", 14, m.textures.slot_mut(14).unwrap(), tex_dropped);
            let has_metal = m.textures.metallic.is_some();
            ui.add_enabled_ui(has_metal, |ui| {
                *changed |= ui.checkbox(&mut m.metallic_invert, "Invert").changed();
            });
            ui.separator();
            *changed |=
                texture_row(ui, "Normal Map", 1, m.textures.slot_mut(1).unwrap(), tex_dropped);
            *changed |= uv_transform_row(ui, &mut m.normal_tiling, &mut m.normal_offset);
            ui.add_enabled_ui(m.has_normal_texture, |ui| {
                property_row(ui, "Use Normal Map", changed, |ui| {
                    ui.checkbox(&mut m.normal_map_enabled, "")
                });
                property_row(ui, "Normal Strength", changed, |ui| {
                    crate::widgets::slider(ui, egui::Slider::new(&mut m.normal_strength, 0.0..=2.0))
                });
            });
            ui.separator();
            *changed |=
                texture_row(ui, "Opacity", 4, m.textures.slot_mut(4).unwrap(), tex_dropped);
            ui.separator();
            *changed |= texture_row(
                ui,
                "Displacement (height)",
                15,
                m.textures.slot_mut(15).unwrap(),
                tex_dropped,
            );
            let has_disp = m.textures.displacement.is_some();
            ui.add_enabled_ui(has_disp, |ui| {
                property_row(ui, "Displacement Scale", changed, |ui| {
                    crate::widgets::slider(ui, egui::Slider::new(&mut m.displacement_scale, 0.0..=0.1))
                        .on_hover_text(
                            "Parallax occlusion depth (0 = off). Height samples in the \
                             albedo UV tiling; all maps shift together.",
                        )
                });
            });
            // Real geometric displacement (hardware tessellation): the height map above moves
            // the vertices of a screen-space-subdivided mesh, so silhouettes, depth and shadows
            // follow it. Replaces the parallax illusion while on.
            ui.add_enabled_ui(has_disp, |ui| {
                property_row(ui, "Tessellate", changed, |ui| {
                    ui.checkbox(&mut m.tess.enabled, "").on_hover_text(
                        "Displace the mesh geometry by the height map (needs tessellation hardware; a device \
                         without it draws the surface flat). Costs nothing for materials that leave it off. \
                         Silhouettes, the depth prepass and shadows all follow the displaced surface.",
                    )
                });
                if m.tess.enabled {
                    property_row(ui, "Height Scale (m)", changed, |ui| {
                        ui.add(egui::DragValue::new(&mut m.tess.scale).speed(0.002).range(-2.0..=2.0))
                            .on_hover_text("Displacement of a full-white texel above the mid level, in world units along the normal.")
                    });
                    property_row(ui, "Mid Level", changed, |ui| {
                        crate::widgets::slider(ui, egui::Slider::new(&mut m.tess.midlevel, 0.0..=1.0))
                            .on_hover_text("Height value that stays on the original surface (0.5: half the map raises, half sinks).")
                    });
                    property_row(ui, "Target Edge (px)", changed, |ui| {
                        crate::widgets::slider(ui, egui::Slider::new(&mut m.tess.edge_px, 4.0..=64.0))
                            .on_hover_text("Triangles are subdivided until their edges are about this long on screen.")
                    });
                    property_row(ui, "Max Factor", changed, |ui| {
                        crate::widgets::slider(ui, egui::Slider::new(&mut m.tess.max_factor, 1.0..=64.0))
                            .on_hover_text("Largest subdivision level of one triangle edge.")
                    });
                    property_row(ui, "Phong Smoothing", changed, |ui| {
                        crate::widgets::slider(ui, egui::Slider::new(&mut m.tess.phong, 0.0..=1.0))
                            .on_hover_text("Rounds the subdivided surface along the vertex normals (0 = flat triangles).")
                    });
                }
            });
        });
    }

    if section_matches(&search_lower, 1) {
        section(
            ui,
            "Toon Shading",
            Some(&mut m.toon_enabled),
            &mut changed,
            |ui, changed| {
                ui.label(
                    egui::RichText::new(
                        "Poiyomi-style cel shading (PBR base + ramp + rim). Shares GI \
                         + baked lightmaps with the Standard shader.",
                    )
                    .small()
                    .weak(),
                );
                property_row(ui, "Ramp Steps", changed, |ui| {
                    crate::widgets::slider(ui, egui::Slider::new(&mut m.toon_steps, 2.0..=8.0).step_by(1.0))
                });
                property_row(ui, "Toon Strength", changed, |ui| {
                    crate::widgets::slider(ui, egui::Slider::new(&mut m.pbr_toon_blend, 0.0..=1.0))
                        .on_hover_text("Blend between smooth PBR (0) and full cel ramp (1)")
                });
                property_row(ui, "Ramp Smoothness", changed, |ui| {
                    crate::widgets::slider(ui, egui::Slider::new(&mut m.ramp_smoothness, 0.01..=0.49))
                        .on_hover_text("Softness of each cel-band terminator")
                });
                ui.separator();
                property_row(ui, "Rim Strength", changed, |ui| {
                    crate::widgets::slider(ui, egui::Slider::new(&mut m.rim_strength, 0.0..=3.0))
                        .on_hover_text("Fresnel edge glow (0 = off)")
                });
                property_row(ui, "Rim Color", changed, |ui| {
                    ui.color_edit_button_rgb(&mut m.rim_color)
                });
                property_row(ui, "Rim Power", changed, |ui| {
                    crate::widgets::slider(ui, egui::Slider::new(&mut m.rim_power, 0.5..=8.0))
                        .on_hover_text("Higher = tighter edge")
                });
            },
        );
    }

    if section_matches(&search_lower, 2) {
        section(
            ui,
            "Emission",
            Some(&mut m.emission_enabled),
            &mut changed,
            |ui, changed| {
                property_row(ui, "Color", changed, |ui| {
                    ui.color_edit_button_rgb(&mut m.emission_color)
                });
                *changed |= texture_row(
                    ui,
                    "Emission Map",
                    3,
                    m.textures.slot_mut(3).unwrap(),
                    tex_dropped,
                );
                *changed |= uv_transform_row(ui, &mut m.emission_tiling, &mut m.emission_offset);
                property_row(ui, "Intensity", changed, |ui| {
                    ui.add(
                        DragValue::new(&mut m.emission_intensity)
                            .speed(0.05)
                            .range(0.0..=100.0),
                    )
                });
                *changed |= texture_row(
                    ui,
                    "Emission Mask",
                    5,
                    m.textures.slot_mut(5).unwrap(),
                    tex_dropped,
                );
                property_row(ui, "Pulse Speed", changed, |ui| {
                    crate::widgets::slider(ui, egui::Slider::new(&mut m.emission_pulse, 0.0..=10.0))
                        .on_hover_text("Brightness oscillation rate (0 = steady)")
                });
                ui.horizontal(|ui| {
                    ui.label("UV Scroll");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        *changed |= ui
                            .add(DragValue::new(&mut m.emission_scroll[1]).speed(0.01).prefix("y "))
                            .changed();
                        *changed |= ui
                            .add(DragValue::new(&mut m.emission_scroll[0]).speed(0.01).prefix("x "))
                            .changed();
                    });
                });
            },
        );
    }

    if section_matches(&search_lower, 3) {
        section(ui, "Transparency", None, &mut changed, |ui, changed| {
            alpha_mode_ui(ui, m, changed);
        });
    }

    if section_matches(&search_lower, 4) {
        section(ui, "Geometry", None, &mut changed, |ui, changed| {
            cull_ui(ui, m, changed);
            ui.label(
                RichText::new("Normal map + strength live in the Base section.")
                    .small()
                    .weak(),
            );
        });
    }

    if section_matches(&search_lower, 5) {
        section(ui, "Matcaps", None, &mut changed, |ui, changed| {
            ui.label(
                RichText::new(
                    "Sphere-mapped reflection layers. Drop a matcap texture, pick how it \
                     blends with the colour, and (optionally) a mask. Strength 0 = off.",
                )
                .small()
                .weak(),
            );
            // Per-layer slot indices into the 12-slot texture set:
            // matcap tex = 6/8/10, matcap mask = 7/9/11.
            for (i, title) in MATCAP_TITLES.iter().enumerate() {
                let tex_slot = 6 + i * 2;
                let mask_slot = 7 + i * 2;
                section(ui, title, None, changed, |ui, changed| {
                    *changed |= texture_row(
                        ui,
                        "Matcap",
                        tex_slot,
                        m.textures.slot_mut(tex_slot).unwrap(),
                        tex_dropped,
                    );
                    let before = m.matcap_blend[i];
                    ui.horizontal(|ui| {
                        ui.label("Blend");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ComboBox::from_id_salt(("matcap-blend", i))
                                .selected_text(m.matcap_blend[i].label())
                                .show_ui(ui, |ui| {
                                    for mode in MatcapBlend::ALL {
                                        ui.selectable_value(
                                            &mut m.matcap_blend[i],
                                            mode,
                                            mode.label(),
                                        );
                                    }
                                });
                        });
                    });
                    if m.matcap_blend[i] != before {
                        *changed = true;
                    }
                    property_row(ui, "Strength", changed, |ui| {
                        crate::widgets::slider(ui, egui::Slider::new(&mut m.matcap_strength[i], 0.0..=2.0))
                    });
                    *changed |= texture_row(
                        ui,
                        "Mask",
                        mask_slot,
                        m.textures.slot_mut(mask_slot).unwrap(),
                        tex_dropped,
                    );
                });
            }
        });
    }

    if section_matches(&search_lower, 6) {
        section(ui, "Detail", None, &mut changed, |ui, changed| detail_ui(ui, m, changed, tex_dropped));
    }

    if section_matches(&search_lower, 7) {
        section(ui, "Translucency & Transmission", None, &mut changed, |ui, changed| {
            transmission_ui(ui, m, changed, tex_dropped)
        });
    }

    section(ui, "Stencil", None, &mut changed, |ui, changed| {
        stencil_ui(ui, m, changed);
    });

    // Render queue is the last entry for every shader (draw-order priority).
    ui.separator();
    render_queue_ui(ui, m, &mut changed);

    changed
    })
    .inner
}

/// Detail layer rows: albedo / normal / ORM maps tiled over the base, a mask, and
/// the albedo blend. A strength of 0 switches that map off.
fn detail_ui(ui: &mut Ui, m: &mut MaterialModel, changed: &mut bool, tex_dropped: &mut Option<(usize, PathBuf)>) {
    use zestengine_core::{DETAIL_SLOT_BASE as B, DetailModel};
    ui.label(
        RichText::new("A second set of maps tiled finer over the base. Strength 0 = that map off.")
            .small()
            .weak(),
    );
    *changed |= texture_row(ui, "Detail Albedo", B, m.textures.slot_mut(B).unwrap(), tex_dropped);
    property_row(ui, "Albedo Strength", changed, |ui| {
        crate::widgets::slider(ui, egui::Slider::new(&mut m.detail.albedo_strength, 0.0..=1.0))
    });
    ui.horizontal(|ui| {
        ui.label("Albedo Blend");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let before = m.detail.blend;
            ComboBox::from_id_salt("detail-blend")
                .selected_text(DetailModel::BLEND_LABELS[(m.detail.blend as usize).min(3)])
                .show_ui(ui, |ui| {
                    for (i, label) in DetailModel::BLEND_LABELS.iter().enumerate() {
                        ui.selectable_value(&mut m.detail.blend, i as u32, *label);
                    }
                });
            *changed |= m.detail.blend != before;
        });
    });
    *changed |= texture_row(ui, "Detail Normal", B + 1, m.textures.slot_mut(B + 1).unwrap(), tex_dropped);
    property_row(ui, "Normal Strength", changed, |ui| {
        crate::widgets::slider(ui, egui::Slider::new(&mut m.detail.normal_strength, 0.0..=2.0))
    });
    *changed |= texture_row(ui, "Detail ORM", B + 2, m.textures.slot_mut(B + 2).unwrap(), tex_dropped);
    property_row(ui, "ORM Strength", changed, |ui| {
        crate::widgets::slider(ui, egui::Slider::new(&mut m.detail.orm_strength, 0.0..=1.0))
            .on_hover_text("Multiplies occlusion / roughness / metallic by the detail map")
    });
    *changed |= texture_row(ui, "Detail Mask", B + 3, m.textures.slot_mut(B + 3).unwrap(), tex_dropped);
    *changed |= uv_transform_row(ui, &mut m.detail.tiling, &mut m.detail.offset);
}

/// Thin-surface translucency (light through leaves, paper, cloth) and see-through
/// transmission (glass, thin plastic).
fn transmission_ui(ui: &mut Ui, m: &mut MaterialModel, changed: &mut bool, tex_dropped: &mut Option<(usize, PathBuf)>) {
    use zestengine_core::DETAIL_SLOT_BASE as B;
    ui.label(RichText::new("Translucency: light coming through from behind.").small().weak());
    let t = &mut m.transmission;
    property_row(ui, "Translucency", changed, |ui| {
        crate::widgets::slider(ui, egui::Slider::new(&mut t.translucency, 0.0..=4.0))
            .on_hover_text("Back-lit glow strength (0 = off)")
    });
    property_row(ui, "Colour", changed, |ui| ui.color_edit_button_rgb(&mut t.translucency_color));
    property_row(ui, "Power", changed, |ui| {
        crate::widgets::slider(ui, egui::Slider::new(&mut t.power, 0.5..=16.0)).on_hover_text("Higher = tighter around the light")
    });
    property_row(ui, "Distortion", changed, |ui| {
        crate::widgets::slider(ui, egui::Slider::new(&mut t.distortion, 0.0..=1.0)).on_hover_text("How much the normal bends the light")
    });
    *changed |= texture_row(ui, "Translucency Map", B + 4, m.textures.slot_mut(B + 4).unwrap(), tex_dropped);
    let t = &mut m.transmission;
    property_row(ui, "Map Is Thickness", changed, |ui| {
        ui.checkbox(&mut t.map_is_thickness, "").on_hover_text("On: the map holds thickness (white = thick, passes less light), as glTF volume maps do")
    });
    ui.separator();
    ui.label(RichText::new("Transmission: see the scene behind (needs Scene Colour Read, turned on automatically).").small().weak());
    property_row(ui, "Transmission", changed, |ui| {
        crate::widgets::slider(ui, egui::Slider::new(&mut t.transmission, 0.0..=1.0))
    });
    property_row(ui, "Tint", changed, |ui| ui.color_edit_button_rgb(&mut t.tint));
    property_row(ui, "Refraction", changed, |ui| {
        crate::widgets::slider(ui, egui::Slider::new(&mut t.refraction, 0.0..=0.5)).on_hover_text("Screen-space offset by the surface normal")
    });
    property_row(ui, "Blur", changed, |ui| {
        crate::widgets::slider(ui, egui::Slider::new(&mut t.blur, 0.0..=1.0))
            .on_hover_text("Rough transmission: how much the surface roughness blurs the scene behind (0 = always sharp)")
    });
    property_row(ui, "Thickness", changed, |ui| {
        crate::widgets::slider(ui, egui::Slider::new(&mut t.thickness, 0.0..=1.0))
            .on_hover_text("Volume thickness; thin volumes blur less (times the map's red channel when it holds thickness)")
    });
    property_row(ui, "Map Green = Transmission", changed, |ui| {
        ui.checkbox(&mut t.map_has_transmission, "").on_hover_text("The Translucency Map's green channel scales the transmission (glTF transmission texture)")
    });
}

fn alpha_mode_ui(ui: &mut Ui, m: &mut MaterialModel, changed: &mut bool) {
    let before = m.alpha_mode;
    ui.horizontal(|ui| {
        ui.label("Mode");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ComboBox::from_id_salt("alpha-mode")
                .selected_text(m.alpha_mode.label())
                .show_ui(ui, |ui| {
                    for mode in [
                        AlphaModeModel::Opaque,
                        AlphaModeModel::Cutout,
                        AlphaModeModel::Transclipping,
                        AlphaModeModel::Blend,
                    ] {
                        ui.selectable_value(&mut m.alpha_mode, mode, mode.label());
                    }
                });
        });
    });
    if m.alpha_mode != before {
        *changed = true;
        // Follow the new mode's default queue, unless the user had customized
        // it away from the old mode's default.
        if m.render_queue == before.default_render_queue() {
            m.render_queue = m.alpha_mode.default_render_queue();
        }
    }
    if matches!(m.alpha_mode, AlphaModeModel::Cutout | AlphaModeModel::Transclipping) {
        property_row(ui, "Cutoff", changed, |ui| {
            crate::widgets::slider(ui, egui::Slider::new(&mut m.alpha_cutoff, 0.0..=1.0))
        });
    }
    if matches!(m.alpha_mode, AlphaModeModel::Blend | AlphaModeModel::Transclipping) {
        property_row(ui, "Opacity", changed, |ui| {
            crate::widgets::slider(ui, egui::Slider::new(&mut m.base_color[3], 0.0..=1.0))
        });
        ui.horizontal(|ui| {
            ui.label("Blend");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let before = m.blend_mode;
                ComboBox::from_id_salt("blend-mode")
                    .selected_text(m.blend_mode.label())
                    .show_ui(ui, |ui| {
                        for mode in BlendModeModel::ALL {
                            ui.selectable_value(&mut m.blend_mode, mode, mode.label());
                        }
                    })
                    .response
                    .on_hover_text("Additive and Multiply suit glow and tint films");
                if m.blend_mode != before {
                    *changed = true;
                }
            });
        });
        property_row(ui, "Scene Colour Read", changed, |ui| {
            ui.checkbox(&mut m.grab, "").on_hover_text("Grab pass: the shader may sample the opaque scene behind it (refraction, blur, distortion)")
        });
    }
}

/// Double sided / front-face culling rows.
fn cull_ui(ui: &mut Ui, m: &mut MaterialModel, changed: &mut bool) {
    property_row(ui, "Double Sided", changed, |ui| ui.checkbox(&mut m.double_sided, ""));
    if !m.double_sided {
        property_row(ui, "Cull Front Faces", changed, |ui| {
            ui.checkbox(&mut m.cull_front, "").on_hover_text("Show only the inside of the mesh (skyboxes, panospheres, inverted hulls)")
        });
    }
}

/// Stencil rows (Unity `Stencil { Ref .. Comp .. Pass .. }`): reference, masks,
/// compare and the three outcome ops, optionally split per face. A writer
/// (Replace / Increment / Invert) must have a LOWER render queue than the
/// materials that read it.
fn stencil_ui(ui: &mut Ui, m: &mut MaterialModel, changed: &mut bool) {
    let st = &mut m.stencil;
    property_row(ui, "Reference", changed, |ui| ui.add(DragValue::new(&mut st.reference).range(0..=255)));
    property_row(ui, "Read Mask", changed, |ui| ui.add(DragValue::new(&mut st.read_mask).range(0..=255)));
    property_row(ui, "Write Mask", changed, |ui| ui.add(DragValue::new(&mut st.write_mask).range(0..=255)));
    property_row(ui, "Two Sided State", changed, |ui| {
        ui.checkbox(&mut st.two_sided, "").on_hover_text("Separate compare / ops for back faces (Unity CompBack, PassBack, ...)")
    });
    let two = st.two_sided;
    stencil_face_ui(ui, if two { "Front" } else { "Faces" }, "stencil-front", &mut st.front, changed);
    if two {
        stencil_face_ui(ui, "Back", "stencil-back", &mut st.back, changed);
    }
    property_row(ui, "Write Depth", changed, |ui| {
        ui.checkbox(&mut st.depth_write, "").on_hover_text("Unity ZWrite. Off for a stencil writer that must not hide what is drawn after it")
    });
    let mut colour = st.color_mask & 15 != 0;
    property_row(ui, "Write Colour", changed, |ui| {
        
        ui.checkbox(&mut colour, "").on_hover_text("Off = a pure stencil writer (Unity ColorMask 0): marks the stencil without drawing colour")
    });
    if colour != (st.color_mask & 15 != 0) {
        st.color_mask = if colour { 15 } else { 0 };
        *changed = true;
    }
    ui.label(RichText::new("Writers draw before readers: give the writer a lower Render Queue.").weak().small());
}

fn stencil_face_ui(ui: &mut Ui, title: &str, salt: &str, f: &mut StencilFaceModel, changed: &mut bool) {
    ui.label(RichText::new(title).strong());
    ui.horizontal(|ui| {
        ui.label("Compare");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let before = f.compare;
            ComboBox::from_id_salt((salt, "cmp")).selected_text(f.compare.label()).show_ui(ui, |ui| {
                for c in StencilCompareModel::ALL {
                    ui.selectable_value(&mut f.compare, c, c.label());
                }
            });
            *changed |= f.compare != before;
        });
    });
    for (label, id, op) in [("Pass", "pass", &mut f.pass_op), ("Fail", "fail", &mut f.fail_op), ("Z Fail", "zfail", &mut f.zfail_op)] {
        ui.horizontal(|ui| {
            ui.label(label);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let before = *op;
                ComboBox::from_id_salt((salt, id)).selected_text(op.label()).show_ui(ui, |ui| {
                    for o in StencilOpModel::ALL {
                        ui.selectable_value(op, o, o.label());
                    }
                });
                *changed |= *op != before;
            });
        });
    }
}

/// Render-queue (draw-order priority) control: a value plus Unity-style
/// preset buttons. Higher = drawn later; transparent (≥3000) sorts
/// back-to-front. The gaps let you fine-tune layered transparency ordering.
fn render_queue_ui(ui: &mut Ui, m: &mut MaterialModel, changed: &mut bool) {
    property_row(ui, "Render Queue", changed, |ui| {
        ui.add(
            DragValue::new(&mut m.render_queue)
                .range(0..=5000)
                .speed(1.0),
        )
    });
    ui.horizontal(|ui| {
        for (label, q) in [
            ("Geometry", 2000),
            ("AlphaTest", 2450),
            ("Transparent", 3000),
            ("Overlay", 4000),
        ] {
            if ui.selectable_label(m.render_queue == q, label).clicked() {
                m.render_queue = q;
                *changed = true;
            }
        }
    });
}

/// Collapsible section. The whole header (arrow *and* title) toggles
/// open/closed; `enabled` adds a master feature checkbox to the header.
fn section(
    ui: &mut Ui,
    title: &str,
    enabled: Option<&mut bool>,
    changed: &mut bool,
    body: impl FnOnce(&mut Ui, &mut bool),
) {
    let id = ui.make_persistent_id(("zestengine-section", title));
    let mut state = CollapsingState::load_with_default_open(ui.ctx(), id, true);
    let mut active = true;
    let header = ui.horizontal(|ui| {
        state.show_toggle_button(ui, paint_default_icon);
        if let Some(enabled) = enabled {
            if ui.checkbox(enabled, "").changed() {
                *changed = true;
            }
            active = *enabled;
        }
        let label = ui.add(Label::new(RichText::new(title).strong()).sense(Sense::click()));
        if label.clicked() {
            state.toggle(ui);
        }
        if label.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
    });
    state.show_body_indented(&header.response, ui, |ui| {
        ui.add_enabled_ui(active, |ui| body(ui, changed));
    });
    state.store(ui.ctx());
}

fn property_row(
    ui: &mut Ui,
    label: &str,
    changed: &mut bool,
    widget: impl FnOnce(&mut Ui) -> egui::Response,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if widget(ui).changed() {
                *changed = true;
            }
        });
    });
}
