use std::env;
use std::path::Path;
use std::process::Command;

/// Strips bindgen's `# [doc = "..."]` attributes. The FFX headers are Doxygen
/// commented, and rustdoc tries to compile those doc strings as doctests (the
/// `@param jitterPhaseCount ...` lines are not valid Rust). Removing them keeps
/// `cargo test --doc` green and shrinks the generated source.
fn strip_bindgen_docs(mut src: &str) -> String {
    const MARK: &str = "# [doc = \"";
    let mut out = String::with_capacity(src.len());
    while let Some(pos) = src.find(MARK) {
        out.push_str(&src[..pos]);
        let after = &src[pos + MARK.len()..];
        let bytes = after.as_bytes();
        let mut i = 0;
        // Walk the Rust string literal, honoring backslash escapes.
        while i < bytes.len() {
            match bytes[i] {
                b'\\' => i += 2,
                b'"' => {
                    i += 1;
                    break;
                }
                _ => i += 1,
            }
        }
        // Skip the closing `]` that terminates the attribute.
        if i < bytes.len() && bytes[i] == b']' {
            i += 1;
        }
        src = &after[i..];
    }
    out.push_str(src);
    out
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    println!("cargo:rerun-if-changed=shaders");
    // Android (Quest) devices may only offer Vulkan 1.1, which loads SPIR-V
    // up to 1.3; desktop keeps the 1.3 environment.
    let target_env = if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("android") {
        "--target-env=vulkan1.1"
    } else {
        "--target-env=vulkan1.3"
    };

    for name in [
        "standard.vert",
        "standard.frag",
        "error.frag",
        "outline.vert",
        // Screen-space selection outline (mask pass + fullscreen composite).
        "select_mask.vert",
        "select_mask.frag",
        "select_outline.frag",
        // Editor ground grid (Scene view only, fullscreen over the tonemapped viewport).
        "editor_grid.frag",
        "outline.frag",
        "fur.vert",
        "fur.frag",
        "skybox.vert",
        "skybox.frag",
        "shadow.vert",
        "fullscreen.vert",
        "post.frag",
        "exposure_histogram.comp",
        "exposure_adapt.comp",
        "area_atlas.comp",
        // Mip-chain bloom (prefilter, downsample, upsample).
        "bloom.comp",
        // Screen-space ambient occlusion (ao.rs): depth prep / pyramid, SAO or
        // GTAO, denoise (+ temporal), depth-aware upsample.
        "ao_prep.frag",
        "ao_main.frag",
        "ao_denoise.frag",
        "ao_upsample.frag",
        // Temporal anti-aliasing resolve (aa.rs).
        "taa.frag",
        // Deferred SSR resolve (fullscreen, current-frame screen-space reflections).
        "ssr_resolve.frag",
        "flux_depth_reduce.comp",
        "flux_radiance_manage.comp",
        "flux_screen_capture.comp",
        "flux_screen_source_test.frag",
        "flux_screen_trace_test.comp",
        "flux_math_test.comp",
        "flux_radiance_blend_test.comp",
        // Deferred projected decals (fullscreen box projectors over opaque geometry).
        "decal.frag",
        // Underwater view fog / caustics over the HDR scene.
        "underwater.frag",
        // VR overlay quad (left-hand UI panel + pointer markers in the eye images).
        "vr_quad.vert",
        "vr_quad.frag",
        // Ray-traced reflections (1 bounce, ray query against the scene TLAS).
        "rt_reflect.comp",
        "reflection_denoise.comp",
        // Displaced water positions for the RT reflections' water BLAS refit.
        "water_displace.comp",
        // Shallow-water simulation of the interactive water layer.
        "water_sim.comp",
        // FFT line passes of the water simulation's dispersive layer.
        "water_fft.comp",
        // Device readback of the production water optical equations.
        "water_optics_test.comp",
        // Lighting bake (compiled only used when ray query is available).
        "bake_gbuffer.vert",
        "bake_gbuffer.frag",
        "bake_lightmap.comp",
        "bake_probe.comp",
        // Software (SDF) GI probe march. GPU compute over the Global Distance Field.
        "sw_gi.comp",
        // Screen-space GI final gather. Per-pixel GDF trace (screen-probe final gather).
        "screen_gi.comp",
        "flux_surface.vert",
        "flux_surface.frag",
        "flux_surface_filter_test.comp",
        // Probe-space GI denoise (à-trous, variance-driven probe-space spatial filter).
        "flux_denoise.comp",
        // Full-res screen-probe integrate (per-pixel-normal SH resolve of the gather).
        "flux_integrate.comp",
        "flux_history.comp",
        // Hardware ray-query screen-probe gather (RT-core GI backend; selected when
        // the device exposes ray-query, SDF screen_gi.comp is the fallback).
        "screen_gi_rt.comp",
        // GPU skinning for Animator-driven meshes.
        "skin.comp",
        // GPU blendshapes: sparse deltas applied before skinning.
        "morph.comp",
        // Mesh Cloth overrides applied over the skinned vertices.
        "cloth_override.comp",
        // Built-in unlit material (world-space UI, particles).
        "unlit.frag",
        // GPU-driven culling + its Hi-Z depth pyramid.
        "gpu_cull.comp",
        "hiz_build.comp",
        // GPU cluster culling + LOD selection (GEO-4) and the decode test harness.
        "cluster_cull.comp",
        "cluster_skin.comp",
        "cluster_decode_test.comp",
    ] {
        let src = format!("shaders/{name}");
        let dst = Path::new(&out_dir).join(format!("{name}.spv"));
        let output = Command::new("glslc")
            .arg(target_env)
            .arg("-O")
            .arg(&src)
            .arg("-o")
            .arg(&dst)
            .output()
            .expect("failed to run glslc — is shaderc/glslc installed?");
        if !output.status.success() {
            panic!(
                "glslc failed for {src}:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    // Surface lighting uses the same backend geometry/visibility code as the
    // final gather, with a world-space card update entry point.
    for name in ["screen_gi.comp", "screen_gi_rt.comp"] {
        let output = Command::new("glslc")
            .args([target_env, "-O", "-DFLUX_SURFACE_LIGHTING"])
            .arg(format!("shaders/{name}")).arg("-o")
            .arg(Path::new(&out_dir).join(format!("{name}.surface.spv")))
            .output().expect("run glslc for Flux surface lighting");
        assert!(output.status.success(), "Flux surface lighting {name}: {}", String::from_utf8_lossy(&output.stderr));
    }

    for (define,suffix) in [("-DFLUX_RECEIVER_SETUP","receiver"),("-DFLUX_DIRECTIONAL_TRACE","detail"),("-DFLUX_ADAPTIVE_PLACEMENT","placement"),("-DFLUX_ADAPTIVE_RESOLVE","reduce")] {
        let output = Command::new("glslc")
            .args([target_env, "-O", define, "shaders/screen_gi.comp", "-o"])
            .arg(Path::new(&out_dir).join(format!("screen_gi.comp.{suffix}.spv")))
            .output().expect("run glslc for Flux mesh gather");
        assert!(output.status.success(), "Flux mesh gather: {}", String::from_utf8_lossy(&output.stderr));
    }

    for name in ["screen_gi.comp", "screen_gi_rt.comp"] {
        let output=Command::new("glslc").args([target_env,"-O","-DFLUX_RADIANCE_UPDATE"])
            .arg(format!("shaders/{name}")).arg("-o").arg(Path::new(&out_dir).join(format!("{name}.radiance.spv")))
            .output().expect("run glslc for Flux radiance cache");
        assert!(output.status.success(),"Flux radiance cache {name}: {}",String::from_utf8_lossy(&output.stderr));
    }

    // FluxRT's per-pixel emitter visibility traces the TLAS too. Keep this
    // separate so the software pipeline works on devices without ray query.
    let output = Command::new("glslc")
        .args([target_env, "-O", "-DFLUX_RT", "shaders/flux_integrate.comp", "-o"])
        .arg(Path::new(&out_dir).join("flux_integrate.rt.comp.spv"))
        .output().expect("run glslc for FluxRT integrate");
    assert!(output.status.success(), "FluxRT integrate: {}", String::from_utf8_lossy(&output.stderr));

    // Multiview variants (`<name>.mv.spv`): both XR eyes in one pass, the
    // eye picked by gl_ViewIndex.
    for name in ["standard.vert", "standard.frag", "skybox.vert"] {
        let src = format!("shaders/{name}");
        let dst = Path::new(&out_dir).join(format!("{name}.mv.spv"));
        let output = Command::new("glslc")
            .arg(target_env)
            .arg("-DMULTIVIEW")
            .arg("-O")
            .arg(&src)
            .arg("-o")
            .arg(&dst)
            .output()
            .expect("failed to run glslc — is shaderc/glslc installed?");
        if !output.status.success() {
            panic!("glslc failed for {src} (multiview):\n{}", String::from_utf8_lossy(&output.stderr));
        }
    }

    // Clustered vertex-pulling variants of the standard vertex shader (GEO-4),
    // single view and multiview (GEO-5: both eyes from one visible list).
    for (out, defs) in [("standard.vert.cluster.spv", &["-DCLUSTER"][..]), ("standard.vert.cluster.mv.spv", &["-DCLUSTER", "-DMULTIVIEW"][..])] {
        let dst = Path::new(&out_dir).join(out);
        let output = Command::new("glslc")
            .arg(target_env)
            .args(defs)
            .arg("-O")
            .arg("shaders/standard.vert")
            .arg("-o")
            .arg(&dst)
            .output()
            .expect("failed to run glslc — is shaderc/glslc installed?");
        if !output.status.success() {
            panic!("glslc failed for standard.vert ({out}):\n{}", String::from_utf8_lossy(&output.stderr));
        }
    }

    // The shadow pass drawing from the cluster pool (cluster shadows, GEO-5).
    {
        let dst = Path::new(&out_dir).join("shadow.vert.cluster.spv");
        let output = Command::new("glslc")
            .arg(target_env)
            .arg("-DCLUSTER")
            .arg("-O")
            .arg("shaders/shadow.vert")
            .arg("-o")
            .arg(&dst)
            .output()
            .expect("failed to run glslc — is shaderc/glslc installed?");
        if !output.status.success() {
            panic!("glslc failed for shadow.vert (cluster):\n{}", String::from_utf8_lossy(&output.stderr));
        }
    }

    // Tessellated (displaced) materials (UNITY-12): the control / evaluation stages for the
    // camera passes (the evaluation stage also under multiview) and the shadow pass, and the
    // shadow vertex stage that feeds them.
    for (out, src, defs) in [
        ("tess.tesc.spv", "tess.tesc", &[][..]),
        ("tess.tese.spv", "tess.tese", &[][..]),
        ("tess.tese.mv.spv", "tess.tese", &["-DMULTIVIEW"][..]),
        ("tess.tesc.shadow.spv", "tess.tesc", &["-DSHADOW"][..]),
        ("tess.tese.shadow.spv", "tess.tese", &["-DSHADOW"][..]),
        ("shadow.vert.tess.spv", "shadow.vert", &["-DZE_TESS"][..]),
    ] {
        let src = format!("shaders/{src}");
        let dst = Path::new(&out_dir).join(out);
        let output = Command::new("glslc")
            .arg(target_env)
            .args(defs)
            .arg("-O")
            .arg(&src)
            .arg("-o")
            .arg(&dst)
            .output()
            .expect("failed to run glslc — is shaderc/glslc installed?");
        if !output.status.success() {
            panic!("glslc failed for {src} ({out}):\n{}", String::from_utf8_lossy(&output.stderr));
        }
    }

    // Octahedral impostor billboards (GEO-5): scene pass, and the vertex stage under multiview.
    for (out, src, defs) in [
        ("impostor.vert.spv", "impostor.vert", &[][..]),
        ("impostor.vert.mv.spv", "impostor.vert", &["-DMULTIVIEW"][..]),
        ("impostor.frag.spv", "impostor.frag", &[][..]),
    ] {
        let src = format!("shaders/{src}");
        let dst = Path::new(&out_dir).join(out);
        let output = Command::new("glslc")
            .arg(target_env)
            .args(defs)
            .arg("-O")
            .arg(&src)
            .arg("-o")
            .arg(&dst)
            .output()
            .expect("failed to run glslc — is shaderc/glslc installed?");
        if !output.status.success() {
            panic!("glslc failed for {src} ({out}):\n{}", String::from_utf8_lossy(&output.stderr));
        }
    }

    // Text pipeline (SDF/MTSDF glyph quads): screen space, scene pass, and the
    // scene pass under multiview. One source pair, selected by defines.
    for (out, src, defs) in [
        ("text_screen.vert.spv", "text.vert", &[][..]),
        ("text_screen.frag.spv", "text.frag", &[][..]),
        ("text_world.vert.spv", "text.vert", &["-DWORLD"][..]),
        ("text_world.frag.spv", "text.frag", &["-DWORLD"][..]),
        ("text_world.vert.mv.spv", "text.vert", &["-DWORLD", "-DMULTIVIEW"][..]),
    ] {
        let src = format!("shaders/{src}");
        let dst = Path::new(&out_dir).join(out);
        let output = Command::new("glslc")
            .arg(target_env)
            .args(defs)
            .arg("-O")
            .arg(&src)
            .arg("-o")
            .arg(&dst)
            .output()
            .expect("failed to run glslc — is shaderc/glslc installed?");
        if !output.status.success() {
            panic!("glslc failed for {src} ({out}):\n{}", String::from_utf8_lossy(&output.stderr));
        }
    }

    // 2D sprite pass (2D-3): one vertex source (plain / multiview) and one
    // fragment source (plain / texture-array atlas / stencil mask shape, and the
    // mask shape over an atlas), selected by defines.
    for (out, src, defs) in [
        ("sprite.vert.spv", "sprite.vert", &[][..]),
        ("sprite.vert.mv.spv", "sprite.vert", &["-DMULTIVIEW"][..]),
        ("sprite.frag.spv", "sprite.frag", &[][..]),
        ("sprite_atlas.frag.spv", "sprite.frag", &["-DATLAS_ARRAY"][..]),
        ("sprite_mask.frag.spv", "sprite.frag", &["-DMASK"][..]),
        ("sprite_mask_atlas.frag.spv", "sprite.frag", &["-DMASK", "-DATLAS_ARRAY"][..]),
    ] {
        let src = format!("shaders/{src}");
        let dst = Path::new(&out_dir).join(out);
        let output = Command::new("glslc")
            .arg(target_env)
            .args(defs)
            .arg("-O")
            .arg(&src)
            .arg("-o")
            .arg(&dst)
            .output()
            .expect("failed to run glslc — is shaderc/glslc installed?");
        if !output.status.success() {
            panic!("glslc failed for {src} ({out}):\n{}", String::from_utf8_lossy(&output.stderr));
        }
    }

    // FidelityFX FSR 3.0.4: link the prebuilt static libs and generate the FFX
    // FFI bindings from the vendored public headers.
    let ffx = format!("{}/third_party/ffx", env::var("CARGO_MANIFEST_DIR").unwrap());
    println!("cargo:rerun-if-changed={ffx}");
    println!("cargo:rustc-link-search=native={ffx}/lib");
    for lib in [
        "ffx_backend_vk_x64",
        "ffx_fsr3upscaler_x64",
        "ffx_fsr3_x64",
        "ffx_frameinterpolation_x64",
        "ffx_opticalflow_x64",
    ] {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    // The FSR libs are C++; pull in the C++ runtime (and the math/pthread it needs).
    println!("cargo:rustc-link-lib=dylib=stdc++");
    // The FFX Vulkan backend references loader entry points (vkGetPhysicalDeviceProperties2,
    // vkGetDeviceProcAddr, ...) directly rather than through ash's runtime-loaded pointers,
    // so the final binary must link the Vulkan loader.
    println!("cargo:rustc-link-lib=dylib=vulkan");

    let bindings = bindgen::Builder::default()
        .header(format!("{ffx}/include/FidelityFX/host/ffx_fsr3upscaler.h"))
        .header(format!("{ffx}/include/FidelityFX/host/ffx_fsr3.h"))
        .clang_args(["-x", "c++", "-std=c++17", "-DFFX_GCC"])
        .clang_arg(format!("-I{ffx}/include"))
        .allowlist_type("Ffx.*")
        // `FfxInterface` is a self-referential function-pointer table (its callbacks
        // take `FfxInterface*`), which bindgen cannot model as a struct; it becomes an
        // opaque blob. `.opaque_type` lets clang still report the correct size/alignment
        // (152 bytes, 8-byte aligned), so every context description embedding it by value
        // stays ABI-correct.
        .opaque_type("FfxInterface")
        // These context types are just opaque `uint32_t data[N]` buffers the app
        // allocates; bindgen turns them into sized blobs (clang reports the size).
        .opaque_type("FfxFrameInterpolationContext")
        .opaque_type("FfxOpticalflowContext")
        .allowlist_function("ffx.*")
        .allowlist_var("FFX_.*")
        .generate()
        .expect("bindgen failed on the FFX headers");
    let bindings_src = strip_bindgen_docs(
        &bindings
            .to_string()
            // Rust 2024 requires `unsafe` on extern blocks; bindgen's function-pointer
            // typedefs already carry it, so only touch the block-open form.
            .replace("extern \"C\" {", "unsafe extern \"C\" {"),
    );
    std::fs::write(Path::new(&out_dir).join("ffx_bindings.rs"), bindings_src)
        .expect("write ffx bindings");
}
