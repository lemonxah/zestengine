//! Small renderer helpers shared by client-only drawing code and the always-on
//! simulation modules (particles, water FX), kept out of `game_ui` (egui).

/// An unlit, alpha-blended, double-sided material (world UI, particles).
pub fn unlit_material(
    renderer: &mut zestengine_render::Renderer,
    texture: Option<zestengine_render::TextureHandle>,
    name: &str,
) -> anyhow::Result<zestengine_render::MaterialHandle> {
    let features = zestengine_render::MaterialFeatures {
        alpha_mode: zestengine_render::AlphaMode::Blend,
        double_sided: true,
        ..Default::default()
    };
    let desc = zestengine_render::MaterialDesc {
        name: name.to_owned(),
        albedo: texture,
        features,
        ..Default::default()
    };
    let material = renderer.create_material(&desc)?;
    let shader = renderer.unlit_shader()?;
    renderer.set_material_shader(material, Some(shader));
    renderer.set_material_features(material, features);
    Ok(material)
}
