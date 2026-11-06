//! The inert `Renderer`. It can never be constructed (its field is uninhabited), so
//! every method body is `match self.never {}`: engine code written against the real
//! renderer compiles unchanged, and at run time there is simply no renderer
//! (`Option<&mut Renderer>` is `None`). The method list is generated from the real
//! signatures (`renderer_methods.rs`); add a method there when render-free code
//! starts calling a new one.

pub enum Never {}

pub struct Renderer {
    pub(crate) never: Never,
}

/// Uploaded-resource bundle of the background loader (opaque; never built here).
pub struct PreparedScene {
    #[allow(dead_code)]
    never: Never,
}

/// Background uploader (transfer queue). Never built without a renderer.
pub struct Uploader {
    never: Never,
}

impl Uploader {
    pub fn prepare(&self, _meshes: &[crate::MeshData], _textures: &[crate::CompressedTexture], _has_skeleton: bool) -> anyhow::Result<PreparedScene> {
        match self.never {}
    }

    pub fn prepare_compressed(&self, _textures: &[crate::CompressedTexture]) -> anyhow::Result<PreparedScene> {
        match self.never {}
    }
}

/// Cluster-DAG upload records. The real type holds GPU record arrays; render-free
/// code never builds one (`ClusterDag::to_gpu_upload` is client-only).
#[derive(Clone, Debug, Default)]
pub struct ClusterMeshUpload;

/// Material custom-keyword / texture slot counts (mirror of `pipeline.rs`).
pub const CUSTOM_KEYWORDS: usize = 32;
pub const CUSTOM_TEXTURES: usize = 24;
