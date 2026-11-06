//! `.material` overrides for the 2D sprite pass (2D-3 / 2D-5): `SpriteRenderer.material` and
//! `TilemapRenderer.material`.
//!
//! The sprite pass has one fixed unlit fragment shader, so a material maps onto the few things that
//! shader varies on instead of running its own:
//!
//! | material | sprite pass |
//! |---|---|
//! | `alpha_mode` Opaque | straight alpha (the texture's own alpha still shapes the sprite) |
//! | `alpha_mode` Cutout | alpha test at `alpha_cutoff` |
//! | `alpha_mode` Blend + `blend` Alpha / Additive / Multiply / Premultiplied | that blend equation |
//! | `base_color` | a tint multiplied into the sprite's / map's colour |
//!
//! Not honoured, with the reason: a custom `shader` (the sprite pass cannot run an arbitrary fragment
//! shader: it draws instanced quads from one pipeline per blend, which is what keeps a tilemap one
//! draw), `stencil` (sprite masks own the stencil buffer), textures and every PBR parameter. A
//! material that asks for a custom shader still gives its blend and tint, with a warning.

use zestengine_assets::MaterialFile;
use zestengine_render::{AlphaMode, BlendMode, SpriteBlend, SpriteMaterialDesc};

/// The sprite material and the tint a `.material` stands for.
pub fn sprite_material_from(file: &MaterialFile) -> (SpriteMaterialDesc, [f32; 4]) {
    let f = &file.features;
    let desc = match f.alpha_mode {
        AlphaMode::Opaque => SpriteMaterialDesc { blend: SpriteBlend::Alpha, alpha_cutoff: 0.0 },
        AlphaMode::Cutout | AlphaMode::Transclipping => SpriteMaterialDesc { blend: SpriteBlend::Alpha, alpha_cutoff: file.params.alpha_cutoff.clamp(0.0, 1.0) },
        AlphaMode::Blend => SpriteMaterialDesc {
            blend: match f.blend {
                BlendMode::Alpha => SpriteBlend::Alpha,
                BlendMode::Additive => SpriteBlend::Additive,
                BlendMode::Multiply => SpriteBlend::Multiply,
                BlendMode::Premultiplied => SpriteBlend::Premultiplied,
            },
            alpha_cutoff: 0.0,
        },
    };
    (desc, file.params.base_color)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(alpha: AlphaMode, blend: BlendMode) -> MaterialFile {
        let mut f = MaterialFile {
            name: "m".into(),
            shader: "standard".into(),
            params: Default::default(),
            features: Default::default(),
            render_queue: None,
            textures: Default::default(),
            custom: Default::default(),
        };
        f.features.alpha_mode = alpha;
        f.features.blend = blend;
        f
    }

    #[test]
    fn alpha_modes_and_blends_map_onto_sprite_materials() {
        let (d, t) = sprite_material_from(&file(AlphaMode::Opaque, BlendMode::Additive));
        assert_eq!((d.blend, d.alpha_cutoff), (SpriteBlend::Alpha, 0.0), "the blend only applies to Blend materials");
        assert_eq!(t, file(AlphaMode::Opaque, BlendMode::Alpha).params.base_color);
        let mut cut = file(AlphaMode::Cutout, BlendMode::Alpha);
        cut.params.alpha_cutoff = 0.35;
        let (d, _) = sprite_material_from(&cut);
        assert_eq!((d.blend, d.alpha_cutoff), (SpriteBlend::Alpha, 0.35));
        for (b, want) in [(BlendMode::Alpha, SpriteBlend::Alpha), (BlendMode::Additive, SpriteBlend::Additive), (BlendMode::Multiply, SpriteBlend::Multiply), (BlendMode::Premultiplied, SpriteBlend::Premultiplied)] {
            assert_eq!(sprite_material_from(&file(AlphaMode::Blend, b)).0.blend, want);
        }
        let mut tinted = file(AlphaMode::Blend, BlendMode::Alpha);
        tinted.params.base_color = [1.0, 0.5, 0.25, 0.8];
        assert_eq!(sprite_material_from(&tinted).1, [1.0, 0.5, 0.25, 0.8]);
    }
}
