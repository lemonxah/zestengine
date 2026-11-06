//! Reflected material UI for custom shaders: presets, foldout groups with
//! feature toggles, conditional visibility, and one widget per property type.

use std::path::PathBuf;

use egui::collapsing_header::CollapsingState;
use egui::{ComboBox, DragValue, RichText, Sense, Ui};

use zestengine_core::{MaterialModel, ShaderPropKindUi, ShaderPropUi, ShaderUiInfo};

use crate::sections::{texture_row, uv_transform_row};
use crate::{theme, widgets};

/// Draw a custom shader's properties. Returns true on change.
pub(crate) fn custom_shader_ui(
    ui: &mut Ui,
    search: &mut String,
    m: &mut MaterialModel,
    info: Option<&ShaderUiInfo>,
    tex_dropped: &mut Option<(usize, PathBuf)>,
) -> bool {
    ui.separator();
    let Some(info) = info else {
        ui.label(RichText::new("Loading shader...").color(theme::TEXT_DIM));
        return false;
    };
    if let Some(error) = &info.error {
        ui.label(
            RichText::new(format!("{} shader failed to compile", info.language))
                .color(theme::STATUS_ERROR)
                .strong(),
        );
        egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
            ui.add(egui::Label::new(RichText::new(error).font(theme::mono(theme::SIZE_META)).color(theme::STATUS_ERROR)).wrap());
        });
        return false;
    }
    if m.custom_values.len() < 16 {
        ui.label(RichText::new("Initializing properties...").color(theme::TEXT_DIM));
        return false;
    }

    let before_values = m.custom_values.clone();
    let before_textures = m.textures.shader.clone();

    ui.horizontal(|ui| {
        ui.label(RichText::new(&info.display_name).font(theme::semibold(theme::SIZE_BODY)).color(theme::TEXT_STRONG));
        widgets::badge(ui, info.language);
        let keywords = info.props.iter().filter(|p| p.keyword).count();
        if keywords > 0 {
            let on = info.active_keywords(&m.custom_values);
            widgets::badge(ui, &format!("variant {on}/{keywords}")).on_hover_text(
                "Enabled feature keywords. Each keyword set compiles its own pipeline \
                 with disabled features stripped, so unused features cost nothing.",
            );
        }
    });

    if !info.presets.is_empty() {
        let current = info.matching_preset(&m.custom_values);
        widgets::property_row(ui, "Preset", |ui| {
            ComboBox::from_id_salt("shader-preset")
                .selected_text(current.map_or("Custom", |i| info.presets[i].name.as_str()))
                .show_ui(ui, |ui| {
                    for (i, p) in info.presets.iter().enumerate() {
                        if ui.selectable_label(current == Some(i), &p.name).clicked() {
                            info.apply_preset(i, &mut m.custom_values);
                        }
                    }
                });
        });
    }

    widgets::search_field(ui, search, "Filter properties", ui.available_width());
    ui.add_space(4.0);
    let needle = search.to_lowercase();
    let matches = |p: &ShaderPropUi| {
        needle.is_empty() || p.label.to_lowercase().contains(&needle) || p.name.to_lowercase().contains(&needle)
    };

    // Ungrouped properties first.
    let loose: Vec<usize> = (0..info.props.len())
        .filter(|&i| info.props[i].group.is_none() && matches(&info.props[i]))
        .collect();
    if !loose.is_empty() {
        let id = ui.make_persistent_id(("zestengine-shader-main", &info.display_name));
        shader_section(ui, id, "Main", None, true, m, |ui: &mut Ui, values: &mut Values<'_>| {
            for &i in &loose {
                prop_ui(ui, &info.props[i], values, tex_dropped);
            }
        });
    }

    for (gi, group) in info.groups.iter().enumerate() {
        if !group.show_if.eval(&m.custom_values) {
            continue;
        }
        let members: Vec<usize> = (0..info.props.len())
            .filter(|&i| info.props[i].group == Some(gi) && Some(i) != group.toggle)
            .filter(|&i| matches(&info.props[i]) || (!needle.is_empty() && group.label.to_lowercase().contains(&needle)))
            .collect();
        if members.is_empty() && !needle.is_empty() {
            continue;
        }
        let toggle_offset = group.toggle.map(|t| info.props[t].offset);
        let id = ui.make_persistent_id(("zestengine-shader-group", &info.display_name, &group.label));
        let keyword = group.toggle.is_some_and(|t| info.props[t].keyword);
        shader_section(ui, id, &group.label, toggle_offset, keyword, m, |ui: &mut Ui, values: &mut Values<'_>| {
            for &i in &members {
                prop_ui(ui, &info.props[i], values, tex_dropped);
            }
        });
    }

    m.custom_values != before_values || m.textures.shader != before_textures
}

/// Mutable view handed to property widgets: values + texture slots.
struct Values<'a> {
    values: &'a mut Vec<f32>,
    textures: &'a mut zestengine_core::MaterialTexturePaths,
}

/// Foldout with an optional header checkbox bound to value `toggle`. A toggled
/// off group stays visible but dimmed (discoverability).
fn shader_section(
    ui: &mut Ui,
    id: egui::Id,
    title: &str,
    toggle: Option<usize>,
    keyword: bool,
    m: &mut MaterialModel,
    body: impl FnOnce(&mut Ui, &mut Values<'_>),
) {
    let mut on = toggle.is_none_or(|t| m.custom_values.get(t).is_some_and(|v| *v != 0.0));
    let mut state = CollapsingState::load_with_default_open(ui.ctx(), id, on);
    let header = ui.horizontal(|ui| {
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), Sense::click());
        widgets::paint_caret(ui.painter(), rect.shrink(2.0), state.is_open(), theme::TEXT_DIM);
        if resp.clicked() {
            state.toggle(ui);
        }
        if let Some(t) = toggle {
            let r = ui.checkbox(&mut on, "");
            if keyword {
                r.on_hover_text("Feature keyword: off compiles this feature out of the material's shader variant.");
            }
            if let Some(v) = m.custom_values.get_mut(t) {
                *v = on as u32 as f32;
            }
        }
        let label = ui.add(
            egui::Label::new(RichText::new(title).font(theme::semibold(theme::SIZE_BODY)).color(if on {
                theme::TEXT_STRONG
            } else {
                theme::TEXT_DIM
            }))
            .sense(Sense::click()),
        );
        if label.clicked() {
            state.toggle(ui);
        }
    });
    let mut view = Values { values: &mut m.custom_values, textures: &mut m.textures };
    state.show_body_indented(&header.response, ui, |ui| {
        ui.add_enabled_ui(on, |ui| body(ui, &mut view));
    });
    state.store(ui.ctx());
}

fn prop_ui(
    ui: &mut Ui,
    p: &ShaderPropUi,
    v: &mut Values<'_>,
    tex_dropped: &mut Option<(usize, PathBuf)>,
) {
    if !p.show_if.eval(v.values) {
        return;
    }
    if let Some(h) = &p.header {
        widgets::eyebrow(ui, h);
    }
    let at = p.offset;
    let n = match &p.kind {
        ShaderPropKindUi::Texture { .. } => 4,
        ShaderPropKindUi::Color { .. } => 4,
        ShaderPropKindUi::Color3 { .. } => 3,
        ShaderPropKindUi::Vector(n) => *n as usize,
        _ => 1,
    };
    if at + n > v.values.len() {
        return;
    }
    let label = if p.keyword { format!("{} *", p.label) } else { p.label.clone() };
    let tip = p.tooltip.clone().or_else(|| p.keyword.then(|| "Feature keyword (compiled variant)".to_owned()));
    let resp = match &p.kind {
        ShaderPropKindUi::Float { min, max } => {
            widgets::property_row(ui, &label, |ui| crate::widgets::slider(ui, egui::Slider::new(&mut v.values[at], *min..=*max)))
        }
        ShaderPropKindUi::Drag => {
            widgets::property_row(ui, &label, |ui| ui.add(DragValue::new(&mut v.values[at]).speed(0.01)))
        }
        ShaderPropKindUi::Int { min, max } => {
            let mut i = v.values[at].round() as i32;
            let r = widgets::property_row(ui, &label, |ui| {
                if min < max {
                    crate::widgets::slider(ui, egui::Slider::new(&mut i, *min..=*max))
                } else {
                    ui.add(DragValue::new(&mut i))
                }
            });
            v.values[at] = i as f32;
            r
        }
        ShaderPropKindUi::Toggle => {
            let mut on = v.values[at] != 0.0;
            let r = widgets::property_row(ui, &label, |ui| ui.checkbox(&mut on, ""));
            v.values[at] = on as u32 as f32;
            r
        }
        ShaderPropKindUi::Enum(opts) => {
            let mut sel = (v.values[at].round().max(0.0) as usize).min(opts.len().saturating_sub(1));
            let r = widgets::property_row(ui, &label, |ui| {
                ComboBox::from_id_salt(("shader-enum", &p.name))
                    .selected_text(opts.get(sel).map(String::as_str).unwrap_or(""))
                    .show_ui(ui, |ui| {
                        for (i, o) in opts.iter().enumerate() {
                            ui.selectable_value(&mut sel, i, o);
                        }
                    })
                    .response
            });
            v.values[at] = sel as f32;
            r
        }
        ShaderPropKindUi::Color { hdr } => {
            let hdr = *hdr;
            widgets::property_row(ui, &label, |ui| {
                let c: &mut [f32; 4] = (&mut v.values[at..at + 4]).try_into().unwrap();
                color_edit(ui, c, hdr)
            })
        }
        ShaderPropKindUi::Color3 { hdr } => {
            let hdr = *hdr;
            widgets::property_row(ui, &label, |ui| {
                let mut c = [v.values[at], v.values[at + 1], v.values[at + 2], 1.0];
                let r = color_edit(ui, &mut c, hdr);
                v.values[at..at + 3].copy_from_slice(&c[..3]);
                r
            })
        }
        ShaderPropKindUi::Vector(n) => widgets::property_row(ui, &label, |ui| {
            ui.horizontal(|ui| {
                for (i, axis) in ["x ", "y ", "z ", "w "].iter().take(*n as usize).enumerate() {
                    ui.add(DragValue::new(&mut v.values[at + i]).speed(0.01).prefix(*axis));
                }
            })
            .response
        }),
        ShaderPropKindUi::Texture { slot, default } => {
            match v.textures.slot_mut(*slot) {
                Some(path) => {
                    texture_row(ui, &label, *slot, path, tex_dropped);
                }
                None => {
                    widgets::property_row(ui, &label, |ui| ui.label(RichText::new(*default).color(theme::TEXT_DIM)));
                }
            }
            let (tiling, offset) = v.values[at..at + 4].split_at_mut(2);
            let tiling: &mut [f32; 2] = tiling.try_into().unwrap();
            let offset: &mut [f32; 2] = offset.try_into().unwrap();
            uv_transform_row(ui, tiling, offset);
            return;
        }
    };
    if let Some(tip) = tip {
        resp.on_hover_text(tip);
    }
}

/// Colour swatch; HDR colours get an intensity multiplier next to it (the
/// picker edits the normalized colour).
fn color_edit(ui: &mut Ui, c: &mut [f32; 4], hdr: bool) -> egui::Response {
    if !hdr {
        return widgets::color_hex(ui, c);
    }
    let mut k = c[0].max(c[1]).max(c[2]).max(1.0);
    let mut base = [c[0] / k, c[1] / k, c[2] / k, c[3]];
    let r = ui
        .horizontal(|ui| {
            let a = ui.color_edit_button_rgba_unmultiplied(&mut base);
            let b = ui.add(DragValue::new(&mut k).speed(0.05).range(1.0..=64.0).prefix("x "));
            a.union(b)
        })
        .inner;
    *c = [base[0] * k, base[1] * k, base[2] * k, base[3]];
    r
}
