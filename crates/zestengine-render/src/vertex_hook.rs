//! Vertex hooks for custom shaders: `zestengine_vertex(inout VertexData v)`.
//!
//! A custom shader file may carry a `// @vertex` ... `// @endvertex` section
//! (docs/shaders.md, "Vertex hook"). The assets crate turns it into a GLSL chunk
//! (the `VertexData` struct, the shader's material block / textures / keyword
//! constants and the hook body) and hands it to
//! [`crate::Renderer::register_shader_with_vertex`]. This module splices that chunk, as
//! `zestengine_hook.glsl`, into the engine's own vertex shaders
//! (`standard.vert`, `shadow.vert`) and compiles them with `glslc`, so the very same
//! hook code runs in every pass that draws the material: main, depth prepass, XR
//! multiview, planar mirrors (all `standard.vert`) and the shadow pass
//! (`shadow.vert`). One module per pass shape, so the depth prepass and the main pass
//! share the module and their positions stay bit-identical.
//!
//! Like the assets crate's fragment compile this needs `glslc` at run time; a
//! game that ships without it has to keep the fragment-only shaders (see the
//! docs for the leftover).

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Context as _, Result, bail};

const STANDARD_VERT: &str = include_str!("../shaders/standard.vert");
const SHADOW_VERT: &str = include_str!("../shaders/shadow.vert");
const STANDARD_COMMON: &str = include_str!("../shaders/standard_common.glsl");

/// Specialization-constant ids a hooked `standard.vert` moves `FEAT_WATER` /
/// `FEAT_PER_INSTANCE` to, out of the custom keyword range (0..31).
pub(crate) const HOOK_ID_WATER: u32 = 60;
pub(crate) const HOOK_ID_PER_INSTANCE: u32 = 61;

/// Which engine vertex shader a hook is spliced into.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum HookStage {
    /// `standard.vert`: main, depth prepass, planar.
    Main,
    /// `standard.vert -DMULTIVIEW`: both XR eyes.
    MainMultiview,
    /// `shadow.vert`.
    Shadow,
    /// `shadow.vert -DZE_TESS`: the shadow pass of a tessellated material.
    ShadowTess,
}

impl HookStage {
    pub const COUNT: usize = 4;

    pub fn index(self) -> usize {
        self as usize
    }
}

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// Compile one hooked vertex module (SPIR-V bytes). `hook` is the chunk the assets crate
/// produced; the error carries `glslc`'s message (it names the user's lines).
pub(crate) fn compile(stage: HookStage, hook: &str) -> Result<Vec<u8>> {
    let dir: PathBuf = std::env::temp_dir().join(format!(
        "zestengine-hook-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let result = compile_in(&dir, stage, hook);
    let _ = std::fs::remove_dir_all(&dir);
    result
}

fn compile_in(dir: &std::path::Path, stage: HookStage, hook: &str) -> Result<Vec<u8>> {
    for (name, text) in [
        ("standard.vert", STANDARD_VERT),
        ("shadow.vert", SHADOW_VERT),
        ("standard_common.glsl", STANDARD_COMMON),
        ("zestengine_hook.glsl", hook),
    ] {
        std::fs::write(dir.join(name), text).with_context(|| format!("writing {name}"))?;
    }
    let (file, defs): (&str, &[&str]) = match stage {
        HookStage::Main => ("standard.vert", &["-DZE_HOOK"]),
        HookStage::MainMultiview => ("standard.vert", &["-DZE_HOOK", "-DMULTIVIEW"]),
        HookStage::Shadow => ("shadow.vert", &["-DZE_HOOK"]),
        HookStage::ShadowTess => ("shadow.vert", &["-DZE_HOOK", "-DZE_TESS"]),
    };
    let mut cmd = Command::new("glslc");
    cmd.arg(if cfg!(target_os = "android") { "--target-env=vulkan1.1" } else { "--target-env=vulkan1.3" })
        .arg("-I")
        .arg(dir)
        .args(defs)
        .arg(format!("-DZE_ID_WATER={HOOK_ID_WATER}"))
        .arg(format!("-DZE_ID_PERINST={HOOK_ID_PER_INSTANCE}"))
        .args(["-O", "-fshader-stage=vertex"])
        .arg(dir.join(file))
        .args(["-o", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let child = cmd.spawn().context("running glslc for a vertex hook (is shaderc installed?)")?;
    let output = child.wait_with_output().context("waiting for glslc")?;
    if !output.status.success() {
        // Name the user's section instead of the temp path.
        let stderr = String::from_utf8_lossy(&output.stderr).replace(&dir.display().to_string(), "vertex hook").replace("zestengine_hook.glsl", "vertex hook");
        bail!("vertex hook ({stage:?}) failed to compile:\n{}", stderr.trim());
    }
    Ok(output.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wind-sway example's chunk shape, without the assets crate: every stage compiles.
    #[test]
    fn all_stages_compile() {
        let hook = r#"
struct VertexData {
    vec3 position; vec3 normal; vec4 tangent; vec2 uv; vec2 uv1; vec4 color;
    mat4 model; vec3 world_position; vec3 world_offset; float time; int vertex_index;
};
void zestengine_vertex(inout VertexData v) {
    v.world_offset.x += sin(v.time + v.world_position.y) * 0.1;
}
"#;
        for stage in [HookStage::Main, HookStage::MainMultiview, HookStage::Shadow, HookStage::ShadowTess] {
            let spv = compile(stage, hook).unwrap_or_else(|e| panic!("{stage:?}: {e:#}"));
            assert!(spv.len() > 100 && spv.len().is_multiple_of(4), "{stage:?}");
        }
    }

    #[test]
    fn errors_name_the_hook() {
        let e = compile(HookStage::Main, "void zestengine_vertex(inout VertexData v) { nope }").unwrap_err();
        assert!(format!("{e:#}").contains("vertex hook"), "{e:#}");
    }
}
