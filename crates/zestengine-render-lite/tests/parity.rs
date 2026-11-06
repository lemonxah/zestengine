//! The mirrored helpers must behave exactly like the originals in `zestengine-render`.

use zestengine_render as real;
use zestengine_render_lite as lite;

#[test]
fn mirrored_helpers_match_the_renderer() {
    for s in [1u32, 2, 3, 64, 100, 1024] {
        assert_eq!(real::cube_mip_count(s), lite::cube_mip_count(s));
    }
    assert_eq!(real::lightmap_array_size([4, 9000, 7].into_iter()), lite::lightmap_array_size([4, 9000, 7].into_iter()));
    let src: Vec<f32> = (0..4 * 4 * 4).map(|i| (i as f32 * 0.37).sin()).collect();
    assert_eq!(real::resample_rgba32f(&src, 4, 7), lite::resample_rgba32f(&src, 4, 7));
    assert_eq!(real::rgba32f_to_rgba16f(&src), lite::rgba32f_to_rgba16f(&src));
    let half = real::rgba32f_to_rgba16f(&src);
    assert_eq!(real::rgba16f_to_rgba32f(&half), lite::rgba16f_to_rgba32f(&half));
    for (r, l) in [(real::HdrTextureFormat::Bc6h, lite::HdrTextureFormat::Bc6h), (real::HdrTextureFormat::Rgb9e5, lite::HdrTextureFormat::Rgb9e5)] {
        assert_eq!((r.code(), r.label()), (l.code(), l.label()));
    }
    assert_eq!(real::MAX_PLANAR, lite::MAX_PLANAR);
    assert_eq!(real::CUSTOM_KEYWORDS, lite::CUSTOM_KEYWORDS);
    assert_eq!(real::CUSTOM_TEXTURES, lite::CUSTOM_TEXTURES);
    assert_eq!(real::MAX_WATER_DISPLACERS, lite::MAX_WATER_DISPLACERS);
    assert_eq!(real::RENDER_QUEUE_TRANSPARENT, lite::RENDER_QUEUE_TRANSPARENT);
    assert_eq!(real::ClusterMode::default().label(), lite::ClusterMode::default().label());
    assert_eq!(real::SamplerKey::default().aniso, lite::SamplerKey::default().aniso);
    assert_eq!(real::bc_supported(), lite::bc_supported());
}
