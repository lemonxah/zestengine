//! Audio asset inspectors: Steam Audio quality, reflections, HRTF,
//! output layout, ray tracer and baking, on the editor widget kit.

use zestengine_core::{OutputLayout, RayTracerKind, ReflectionMode, SpatialAudioSettings};
use egui::{DragValue, Ui};

use crate::widgets;

/// What the panel asked for this frame.
#[derive(Default)]
pub struct AudioSettingsResponse {
    pub changed: bool,
}

fn combo<T: Copy + PartialEq>(ui: &mut Ui, salt: &str, value: &mut T, all: &[T], label: fn(T) -> &'static str) -> bool {
    let mut changed = false;
    egui::ComboBox::from_id_salt(salt)
        .width(ui.available_width())
        .selected_text(label(*value))
        .show_ui(ui, |ui| {
            for &v in all {
                changed |= ui.selectable_value(value, v, label(v)).changed();
            }
        });
    changed
}

pub fn audio_settings_ui(ui: &mut Ui, s: &mut SpatialAudioSettings) -> AudioSettingsResponse {
    let mut r = AudioSettingsResponse::default();
    let mut c = false;

    widgets::eyebrow(ui, "Output");
    widgets::property_row(ui, "Layout", |ui| c |= combo(ui, "zestengine-sa-output", &mut s.output, &OutputLayout::ALL, OutputLayout::label));
    widgets::property_row(ui, "HRTF (.sofa)", |ui| {
        c |= ui
            .add(egui::TextEdit::singleline(&mut s.hrtf_sofa).hint_text("Steam Audio default").desired_width(ui.available_width()))
            .changed();
    });

    widgets::eyebrow(ui, "Reflections");
    widgets::property_row(ui, "Type", |ui| {
        c |= combo(ui, "zestengine-sa-reflections", &mut s.reflections, &ReflectionMode::ALL, ReflectionMode::label);
    });
    ui.add_enabled_ui(s.reflections != ReflectionMode::Off, |ui| {
        widgets::property_row(ui, "Listener reverb", |ui| {
            c |= ui
                .checkbox(&mut s.listener_reverb, "")
                .on_hover_text("Reverb around the listener for sources without their own reflections")
                .changed();
        });
        widgets::property_row(ui, "Rays", |ui| c |= ui.add(DragValue::new(&mut s.rays).speed(16).range(64..=65536)).changed());
        widgets::property_row(ui, "Bounces", |ui| c |= ui.add(DragValue::new(&mut s.bounces).range(1..=64)).changed());
        widgets::property_row(ui, "Duration", |ui| {
            c |= ui.add(DragValue::new(&mut s.duration).speed(0.05).range(0.1..=10.0).suffix(" s")).changed();
        });
        widgets::property_row(ui, "Ambisonic order", |ui| c |= crate::widgets::slider(ui, egui::Slider::new(&mut s.order, 1..=3)).changed());
        widgets::property_row(ui, "Diffuse samples", |ui| {
            c |= ui.add(DragValue::new(&mut s.diffuse_samples).speed(4).range(1..=1024)).changed();
        });
        widgets::property_row(ui, "Update rate", |ui| {
            c |= ui.add(DragValue::new(&mut s.update_rate).speed(0.5).range(1.0..=60.0).suffix(" Hz")).changed();
        });
        widgets::property_row(ui, "Max sources", |ui| c |= ui.add(DragValue::new(&mut s.max_sources).range(1..=256)).changed());
    });

    widgets::eyebrow(ui, "Simulation");
    widgets::property_row(ui, "Ray tracer", |ui| {
        c |= combo(ui, "zestengine-sa-tracer", &mut s.ray_tracer, &RayTracerKind::ALL, RayTracerKind::label);
    });

    widgets::eyebrow(ui, "Baking");
    widgets::property_row(ui, "Rays", |ui| c |= ui.add(DragValue::new(&mut s.bake_rays).speed(64).range(256..=262144)).changed());
    widgets::property_row(ui, "Bounces", |ui| c |= ui.add(DragValue::new(&mut s.bake_bounces).range(1..=256)).changed());
    widgets::property_row(ui, "Duration", |ui| {
        c |= ui.add(DragValue::new(&mut s.bake_duration).speed(0.05).range(0.1..=10.0).suffix(" s")).changed();
    });
    ui.add_space(4.0);
    ui.label("Tools → Bake Audio bakes the scene's Audio Probe Volumes.");
    ui.add_space(4.0);
    widgets::info_note(
        ui,
        "Steam Audio settings apply the next time Play starts. Embree and Radeon Rays fall back to the default tracer when unavailable.",
    );
    r.changed = c;
    r
}

/// One asset editor shared by the file inspector and inline component slots.
pub fn audio_asset_ui(ui: &mut Ui, path: &std::path::Path) -> Option<bool> {
    let settings = match path.extension()?.to_str()? {
        "audiosettings" => true,
        "audiofx" => false,
        _ => return None,
    };
    let result = (|| -> anyhow::Result<bool> {
        if settings {
            let mut profile = zestengine_assets::load_audio_settings(path)?;
            let response = audio_settings_ui(ui, &mut profile);
            if response.changed { zestengine_assets::save_audio_settings(path, &profile)?; }
            Ok(response.changed)
        } else {
            let mut profile = zestengine_assets::load_audiofx(path)?;
            let changed = environment_ui(ui, &mut profile);
            if changed { zestengine_assets::save_audiofx(path, &profile)?; }
            Ok(changed)
        }
    })();
    Some(match result {
        Ok(changed) => changed,
        Err(e) => { ui.colored_label(egui::Color32::LIGHT_RED, format!("Audio asset: {e:#}")); false }
    })
}

fn environment_ui(ui: &mut Ui, p: &mut zestengine_core::AudioEnvironmentProfile) -> bool {
    let mut changed = false;
    widgets::eyebrow(ui, "Listener environment");
    widgets::property_row(ui, "Reverb level", |ui| changed |= widgets::slider(ui, egui::Slider::new(&mut p.reverb_gain, 0.0..=2.0)).changed());
    widgets::property_row(ui, "Fade time", |ui| changed |= ui.add(DragValue::new(&mut p.fade_time).speed(0.05).range(0.0..=30.0).suffix(" s")).changed());
    ui.label("Reverb level scales Steam Audio's simulated reflections. Geometry and acoustic materials determine the room response.");
    for (label, gain, cutoff) in [
        ("Interior sounds heard from outside", &mut p.interior_gain, &mut p.interior_lowpass_hz),
        ("Exterior sounds heard from inside", &mut p.exterior_gain, &mut p.exterior_lowpass_hz),
    ] {
        widgets::eyebrow(ui, label);
        widgets::property_row(ui, "Volume", |ui| changed |= widgets::slider(ui, egui::Slider::new(gain, 0.0..=1.0)).changed());
        widgets::property_row(ui, "Low-pass cutoff", |ui| changed |= widgets::slider(ui, egui::Slider::new(cutoff, 20.0..=22_000.0).logarithmic(true).suffix(" Hz")).changed());
    }
    widgets::info_note(ui, "22,000 Hz bypasses filtering. Only spatial sources participate; ordinary 2D music and UI are unaffected. Changes fade during Play.");
    changed
}
