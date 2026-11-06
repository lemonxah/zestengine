//! GPU side of the 2D sprite pass (2D-3): the pooled sprite mesh buffers, the
//! per-frame instance buffers, texture descriptor sets, the pipeline variants
//! and the recording of a built `SpriteFrame` (see `sprite2d.rs` for the CPU
//! half: gather, sort, batch, mask planning).
//!
//! # Where the pass sits, and how it interleaves with 3D
//!
//! The sprites draw inside the main scene pass, after the opaque queues, the
//! 3D transparent queue, impostors and world text (`record_world_text`). So by
//! default **2D sprites composite over every 3D object, transparent ones
//! included**; between sprites the order is the sprite key (sorting layer,
//! order, group, batch, depth), between 3D transparents it stays by distance.
//! The pass has depth testing and depth writes off, so sprites never occlude or
//! are occluded by 3D geometry; they are an overlay layer that can still sit at
//! any world position (the camera's view-projection places them).
//!
//! The sorting-layer rule that overrides this: `BuildConfig::behind_layer_max`
//! marks sprite layers at or below a value as "behind the 3D transparents"; the
//! builder forces a batch break at that boundary and reports
//! `SpriteFrame::behind_items`. This phase records both parts at the single slot
//! above; drawing the leading part before the transparent queue needs a split of
//! the scene draw order (the same place `water_split` cuts) and lands with the
//! engine integration.
//!
//! # Stencil
//!
//! Sprite masks use the stencil attachment of the scene pass. The scene depth
//! format is a depth-stencil one (`D32_SFLOAT_S8_UINT`, else `D24`/`D16` with
//! S8; see `context::pick_depth_format`), so mask pipelines are built and mask
//! items recorded whenever the pass's depth format has stencil (`has_stencil`).
//! With the plain-depth fast path (`ZESTENGINE_STENCIL=0`, or a device with no
//! stencil format) masked sprites draw unmasked and the pass logs once.
//!
//! The pass owns stencil bit 7 only (`sprite2d::MASK_BIT`): mask writes and
//! sprite tests use compare / write mask 0x80, so a 3D material's stencil (bits
//! 0..6) is never read or changed and cannot leak into the masks. The pass
//! erases its still-written mask shapes when it ends (`record` replays them
//! with the ZERO op), so bit 7 is 0 before and after it.

use std::collections::HashMap;

use anyhow::{Context as _, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use crate::alloc::Buffer;
use crate::frame::FRAMES_IN_FLIGHT;
use crate::pipeline::{PassFormats, REACTIVE_FORMAT, VELOCITY_FORMAT};
use crate::sprite2d::{BuildConfig, MaskMode, MaskSource, PassItem, SPRITE_MATERIAL_XZ, SpriteInstance, SpriteMeshPool, SpriteSource, SpriteStats, SpriteVertex, SpritePassBuilder};

const VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sprite.vert.spv"));
const VERT_MV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sprite.vert.mv.spv"));
const FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sprite.frag.spv"));
const FRAG_ATLAS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sprite_atlas.frag.spv"));
const FRAG_MASK: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sprite_mask.frag.spv"));
const FRAG_MASK_ATLAS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sprite_mask_atlas.frag.spv"));

/// Instances one frame can hold (112 bytes each: 14 MiB per frame in flight).
pub const INSTANCE_CAP: usize = 131_072;
/// Pooled sprite mesh capacity (16 MiB of vertices, 6 MiB of `u16` indices).
pub const POOL_VERTICES: usize = 1 << 20;
pub const POOL_INDICES: usize = 3 << 20;
/// Texture descriptor sets (one per registered sprite texture / atlas).
const MAX_TEXTURES: u32 = 1024;
const PUSH_SIZE: u32 = 16;
/// Frame-table entries (16 bytes each, 1 MiB per frame in flight): the tile
/// animation headers and frames of one frame (see `sprite2d::push_animation`).
pub const ANIM_CAP: usize = 65_536;

/// Blend mode of a sprite material.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum SpriteBlend {
    /// Straight alpha: `src*a + dst*(1-a)`.
    #[default]
    Alpha,
    /// The texture is premultiplied: `src + dst*(1-a)`; a tint's alpha still fades it.
    Premultiplied,
    /// `src.rgb * a + dst` (glow, light): the shader premultiplies by alpha, the blend adds.
    Additive,
    /// `src.rgb * dst` (shadows, tint films): the shader fades the tint to white by alpha.
    Multiply,
}

impl SpriteBlend {
    /// The fragment shader's mode value (push constant x).
    pub fn shader_mode(self) -> f32 {
        match self {
            Self::Alpha => 0.0,
            Self::Premultiplied => 1.0,
            Self::Additive => 2.0,
            Self::Multiply => 3.0,
        }
    }
}

/// What a sprite material id (`SpriteSource::material`) means to the GPU.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct SpriteMaterialDesc {
    pub blend: SpriteBlend,
    /// Fragments with final alpha below this are discarded (0 = off).
    pub alpha_cutoff: f32,
}

/// The id of the built-in default sprite material (straight alpha, no cutoff).
pub const DEFAULT_SPRITE_MATERIAL: u32 = 0;

/// What the engine fills each frame (pooled: clear and refill, no allocation
/// once the vectors have grown). Consumed by the frame that renders it.
#[derive(Default)]
pub struct SpriteFrameInput {
    pub sprites: Vec<SpriteSource>,
    pub masks: Vec<MaskSource>,
    pub config: BuildConfig,
    /// Seconds animated tiles run on (one clock for the whole frame).
    pub time: f32,
    /// The tile animation frame table (empty when nothing animates).
    pub anim: Vec<[f32; 4]>,
}

impl SpriteFrameInput {
    pub fn clear(&mut self) {
        self.sprites.clear();
        self.masks.clear();
        self.anim.clear();
    }
    pub fn is_empty(&self) -> bool {
        self.sprites.is_empty()
    }
}

/// Which part of the built frame to record: the scene draw order is cut at the
/// 3D transparent queue when sprite layers are marked "behind 3D transparents".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpritePart {
    /// Every item (nothing is behind, or the caller cannot cut the draw order).
    All,
    /// Items of layers at or below `BuildConfig::behind_layer_max`.
    Behind,
    /// The rest.
    Front,
}

/// Whether `format` has a stencil aspect.
pub(crate) fn has_stencil(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT | vk::Format::S8_UINT
    )
}

/// What a pipeline variant draws.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Role {
    /// Sprites with this stencil test.
    Draw(MaskMode),
    /// Mask shapes: colour off, stencil REPLACE 1.
    MaskWrite,
    /// Mask shapes replayed: colour off, stencil ZERO.
    MaskErase,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PipeKey {
    blend: SpriteBlend,
    /// Depth-tested against the scene (ground tilemaps in the XZ plane); overlay sprites ignore depth.
    depth_test: bool,
    array: bool,
    role: Role,
    color: vk::Format,
    depth: vk::Format,
    gbuf: bool,
    gbuf_format: vk::Format,
    velocity: bool,
    multiview: bool,
    samples: u8,
}

struct SpriteTexture {
    set: vk::DescriptorSet,
    array: bool,
}

pub(crate) struct SpriteGpu {
    builder: SpritePassBuilder,
    vertex: Buffer,
    index: Buffer,
    inst: Vec<Buffer>,
    tex_layout: vk::DescriptorSetLayout,
    tex_pool: vk::DescriptorPool,
    /// Set 2: the animation frame table (one storage buffer per frame in flight).
    anim_layout: vk::DescriptorSetLayout,
    anim_pool: vk::DescriptorPool,
    anim_sets: Vec<vk::DescriptorSet>,
    anim_bufs: Vec<Buffer>,
    time: f32,
    textures: Vec<Option<SpriteTexture>>,
    materials: Vec<SpriteMaterialDesc>,
    layout: vk::PipelineLayout,
    modules: [vk::ShaderModule; 6],
    pipelines: HashMap<PipeKey, vk::Pipeline>,
    warned_stencil: bool,
    warned_overflow: bool,
    warned_anim: bool,
}

fn stencil_op(role: Role) -> Option<vk::StencilOpState> {
    use crate::sprite2d::{MASK_ERASE_STATE, MASK_WRITE_STATE, StencilCompare, StencilOp};
    let state = match role {
        Role::Draw(m) => m.test_state()?,
        Role::MaskWrite => MASK_WRITE_STATE,
        Role::MaskErase => MASK_ERASE_STATE,
    };
    let compare = match state.compare {
        StencilCompare::Always => vk::CompareOp::ALWAYS,
        StencilCompare::Equal => vk::CompareOp::EQUAL,
        StencilCompare::NotEqual => vk::CompareOp::NOT_EQUAL,
    };
    let pass = match state.pass_op {
        StencilOp::Keep => vk::StencilOp::KEEP,
        StencilOp::Replace => vk::StencilOp::REPLACE,
        StencilOp::Zero => vk::StencilOp::ZERO,
    };
    Some(
        vk::StencilOpState::default()
            .fail_op(vk::StencilOp::KEEP)
            .pass_op(pass)
            .depth_fail_op(vk::StencilOp::KEEP)
            .compare_op(compare)
            .compare_mask(state.compare_mask as u32)
            .write_mask(state.write_mask as u32)
            .reference(state.reference as u32),
    )
}

impl SpriteGpu {
    pub fn new(device: &ash::Device, allocator: &mut Allocator, frame_set_layout: vk::DescriptorSetLayout) -> Result<Self> {
        let vertex = Buffer::new(
            device, allocator, (POOL_VERTICES * std::mem::size_of::<SpriteVertex>()) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER, MemoryLocation::CpuToGpu, "sprite vertices",
        )?;
        let index = Buffer::new(
            device, allocator, (POOL_INDICES * 2) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER, MemoryLocation::CpuToGpu, "sprite indices",
        )?;
        let mut inst = Vec::new();
        for _ in 0..FRAMES_IN_FLIGHT {
            inst.push(Buffer::new(
                device, allocator, (INSTANCE_CAP * std::mem::size_of::<SpriteInstance>()) as u64,
                vk::BufferUsageFlags::VERTEX_BUFFER, MemoryLocation::CpuToGpu, "sprite instances",
            )?);
        }
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let tex_layout =
            unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)? };
        let sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(MAX_TEXTURES)];
        let tex_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(MAX_TEXTURES)
                    .pool_sizes(&sizes),
                None,
            )?
        };
        // Set 2: the tile animation frame table, read by the vertex stage.
        let anim_binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];
        let anim_layout =
            unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&anim_binding), None)? };
        let anim_sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(FRAMES_IN_FLIGHT as u32)];
        let anim_pool = unsafe {
            device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(FRAMES_IN_FLIGHT as u32).pool_sizes(&anim_sizes), None)?
        };
        let mut anim_bufs = Vec::new();
        for _ in 0..FRAMES_IN_FLIGHT {
            let mut b = Buffer::new(
                device, allocator, (ANIM_CAP * 16) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "sprite animation table",
            )?;
            // Entry 0 is the "no animation" dummy; never read as uninitialised memory.
            b.write(0, bytemuck::cast_slice(&[[0.0f32; 4]]));
            anim_bufs.push(b);
        }
        let anim_layouts = vec![anim_layout; FRAMES_IN_FLIGHT];
        let anim_sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(anim_pool).set_layouts(&anim_layouts))? };
        for (set, buf) in anim_sets.iter().zip(&anim_bufs) {
            let info = [vk::DescriptorBufferInfo::default().buffer(buf.handle).offset(0).range(vk::WHOLE_SIZE)];
            let write = vk::WriteDescriptorSet::default().dst_set(*set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info);
            unsafe { device.update_descriptor_sets(&[write], &[]) };
        }
        let set_layouts = [frame_set_layout, tex_layout, anim_layout];
        // Fragment: blend mode and cutoff. Vertex: the animation clock (z).
        let push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT).offset(0).size(PUSH_SIZE)];
        let layout = unsafe {
            device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts).push_constant_ranges(&push), None)?
        };
        let module = |spv: &[u8]| -> Result<vk::ShaderModule> {
            let code = ash::util::read_spv(&mut std::io::Cursor::new(spv))?;
            Ok(unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)? })
        };
        let modules = [module(VERT)?, module(VERT_MV)?, module(FRAG)?, module(FRAG_ATLAS)?, module(FRAG_MASK)?, module(FRAG_MASK_ATLAS)?];
        Ok(Self {
            builder: SpritePassBuilder::new(),
            vertex,
            index,
            inst,
            tex_layout,
            tex_pool,
            anim_layout,
            anim_pool,
            anim_sets,
            anim_bufs,
            time: 0.0,
            textures: Vec::new(),
            materials: vec![SpriteMaterialDesc::default()],
            layout,
            modules,
            pipelines: HashMap::new(),
            warned_stencil: false,
            warned_overflow: false,
            warned_anim: false,
        })
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            for (_, p) in self.pipelines.drain() {
                device.destroy_pipeline(p, None);
            }
            for m in self.modules {
                device.destroy_shader_module(m, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_pool(self.tex_pool, None);
            device.destroy_descriptor_set_layout(self.tex_layout, None);
            device.destroy_descriptor_pool(self.anim_pool, None);
            device.destroy_descriptor_set_layout(self.anim_layout, None);
        }
        for b in self.inst.iter_mut().chain(self.anim_bufs.iter_mut()).chain([&mut self.vertex, &mut self.index]) {
            b.destroy(device, allocator);
        }
    }

    /// Bind a texture (2D, or a 2D-array view when `array`) as a sprite texture;
    /// returns the id sprites put in `SpriteSource::texture`. The caller keeps
    /// the view alive until [`Self::unregister_texture`].
    pub fn register_texture(&mut self, device: &ash::Device, view: vk::ImageView, sampler: vk::Sampler, array: bool) -> Result<u32> {
        let layouts = [self.tex_layout];
        let set = unsafe {
            device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.tex_pool).set_layouts(&layouts))
        }
        .context("allocating a sprite texture set")?[0];
        let info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&info);
        unsafe { device.update_descriptor_sets(&[write], &[]) };
        let entry = Some(SpriteTexture { set, array });
        if let Some(slot) = self.textures.iter().position(|t| t.is_none()) {
            self.textures[slot] = entry;
            Ok(slot as u32)
        } else {
            self.textures.push(entry);
            Ok(self.textures.len() as u32 - 1)
        }
    }

    /// Free a sprite texture id. The caller guarantees no in-flight frame still
    /// draws with it (wait for the frames, as for any descriptor free).
    pub fn unregister_texture(&mut self, device: &ash::Device, id: u32) {
        if let Some(t) = self.textures.get_mut(id as usize).and_then(|t| t.take()) {
            unsafe {
                let _ = device.free_descriptor_sets(self.tex_pool, &[t.set]);
            }
        }
    }

    pub fn register_material(&mut self, desc: SpriteMaterialDesc) -> u32 {
        self.materials.push(desc);
        self.materials.len() as u32 - 1
    }

    pub fn stats(&self) -> SpriteStats {
        self.builder.frame().stats
    }

    /// Outside any render pass: upload the new pooled meshes, build the frame
    /// (gather, sort, batch) and write its instances into this frame's buffer.
    pub fn prepare(&mut self, input: &SpriteFrameInput, pool: &mut SpriteMeshPool, frame: usize) {
        let (v0, verts, i0, idx) = pool.take_pending();
        if !verts.is_empty() {
            self.vertex.write(v0 * std::mem::size_of::<SpriteVertex>(), bytemuck::cast_slice(verts));
        }
        if !idx.is_empty() {
            self.index.write(i0 * 2, bytemuck::cast_slice(idx));
        }
        // Skinned sprite meshes rewritten since the last frame (usually none).
        let dirty = pool.take_dirty();
        for &(first, n) in &dirty {
            self.vertex.write(first * std::mem::size_of::<SpriteVertex>(), bytemuck::cast_slice(&pool.vertices()[first..first + n]));
        }
        pool.recycle_dirty(dirty);
        // Sprite-shape index lists rewritten since the last frame.
        let dirty_idx = pool.take_dirty_indices();
        for &(first, n) in &dirty_idx {
            self.index.write(first * 2, bytemuck::cast_slice(&pool.indices()[first..first + n]));
        }
        pool.recycle_dirty_indices(dirty_idx);
        self.time = input.time;
        if !input.anim.is_empty() {
            let n = input.anim.len().min(ANIM_CAP);
            if n < input.anim.len() && !self.warned_anim {
                self.warned_anim = true;
                tracing::warn!("sprites: the tile animation table is full ({ANIM_CAP} entries)");
            }
            self.anim_bufs[frame].write(0, bytemuck::cast_slice(&input.anim[..n]));
        }
        let mut cfg = input.config;
        cfg.max_instances = cfg.max_instances.min(INSTANCE_CAP.saturating_sub(input.masks.len() * 2 + 1));
        let built = self.builder.build(&input.sprites, &input.masks, &cfg);
        if built.stats.dropped > 0 && !self.warned_overflow {
            self.warned_overflow = true;
            tracing::warn!("sprites: more than {} instances in one frame; {} sprites are not drawn", cfg.max_instances, built.stats.dropped);
        }
        let n = built.instances.len().min(INSTANCE_CAP);
        if n > 0 {
            self.inst[frame].write(0, bytemuck::cast_slice(&built.instances[..n]));
        }
    }

    /// Items drawn before the 3D transparents (0 unless layers are marked behind).
    pub fn behind_items(&self) -> usize {
        self.builder.frame().behind_items
    }

    pub fn has_work(&self) -> bool {
        !self.builder.frame().is_empty()
    }

    fn pipeline(&mut self, device: &ash::Device, key: PipeKey) -> Result<vk::Pipeline> {
        if let Some(p) = self.pipelines.get(&key) {
            return Ok(*p);
        }
        let mask = matches!(key.role, Role::MaskWrite | Role::MaskErase);
        let vert = if key.multiview { self.modules[1] } else { self.modules[0] };
        let frag = match (mask, key.array) {
            (false, false) => self.modules[2],
            (false, true) => self.modules[3],
            (true, false) => self.modules[4],
            (true, true) => self.modules[5],
        };
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert).name(c"main"),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(frag).name(c"main"),
        ];
        let f = std::mem::size_of::<f32>() as u32;
        let bindings = [
            vk::VertexInputBindingDescription::default().binding(0).stride(std::mem::size_of::<SpriteVertex>() as u32).input_rate(vk::VertexInputRate::VERTEX),
            vk::VertexInputBindingDescription::default().binding(1).stride(std::mem::size_of::<SpriteInstance>() as u32).input_rate(vk::VertexInputRate::INSTANCE),
        ];
        let attr = |location: u32, binding: u32, format: vk::Format, offset: u32| {
            vk::VertexInputAttributeDescription::default().location(location).binding(binding).format(format).offset(offset)
        };
        use vk::Format as F;
        // Offsets mirror `SpriteInstance` (asserted by the layout test in sprite2d.rs).
        let attrs = [
            attr(0, 0, F::R32G32_SFLOAT, 0),
            attr(1, 0, F::R32G32_SFLOAT, 2 * f),
            attr(2, 1, F::R32G32B32A32_SFLOAT, 0),
            attr(3, 1, F::R32G32B32_SFLOAT, 16),
            attr(4, 1, F::R32_UINT, 28),
            attr(5, 1, F::R32G32B32A32_SFLOAT, 32),
            attr(6, 1, F::R32G32B32A32_SFLOAT, 48),
            attr(7, 1, F::R32G32B32A32_SFLOAT, 64),
            attr(8, 1, F::R32G32B32A32_SFLOAT, 80),
            attr(9, 1, F::R32_UINT, 96),
            attr(10, 1, F::R32_UINT, 100),
            attr(11, 1, F::R32_SFLOAT, 104),
            attr(12, 1, F::R32_UINT, 108),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default().vertex_binding_descriptions(&bindings).vertex_attribute_descriptions(&attrs);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        // Flips mirror geometry and the viewport is y-flipped: no face culling.
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let samples = match key.samples {
            4 => vk::SampleCountFlags::TYPE_4,
            2 => vk::SampleCountFlags::TYPE_2,
            _ => vk::SampleCountFlags::TYPE_1,
        };
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(samples);
        // Depth off: sprites are an overlay ordered by the sprite key alone.
        let mut depth = vk::PipelineDepthStencilStateCreateInfo::default().depth_test_enable(false).depth_write_enable(false);
        if key.depth_test && matches!(key.role, Role::Draw(_)) {
            // The scene's depth convention (0..1, nearer is less): occluded by geometry, never writes.
            depth = depth.depth_test_enable(true).depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
        }
        let stencil_fmt = has_stencil(key.depth);
        if stencil_fmt && let Some(op) = stencil_op(key.role) {
            depth = depth.stencil_test_enable(true).front(op).back(op);
        }
        let color_state = match (key.role, key.blend) {
            (Role::Draw(_), SpriteBlend::Alpha) => vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA),
            (Role::Draw(_), SpriteBlend::Premultiplied) => vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA),
            (Role::Draw(_), SpriteBlend::Additive) => vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ZERO)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA),
            (Role::Draw(_), SpriteBlend::Multiply) => vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::DST_COLOR)
                .dst_color_blend_factor(vk::BlendFactor::ZERO)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ZERO)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA),
            _ => vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::empty()),
        };
        // The scene pass's G-buffer / velocity / reactive attachments are left untouched.
        let untouched = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::empty());
        let mut blends = vec![color_state];
        let mut formats = vec![key.color];
        if key.gbuf {
            blends.push(untouched);
            formats.push(key.gbuf_format);
        }
        if key.velocity {
            blends.push(untouched);
            blends.push(untouched);
            formats.push(VELOCITY_FORMAT);
            formats.push(REACTIVE_FORMAT);
        }
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats).depth_attachment_format(key.depth);
        if stencil_fmt {
            rendering = rendering.stencil_attachment_format(key.depth);
        }
        if key.multiview {
            rendering = rendering.view_mask(0b11);
        }
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(self.layout)
            .push_next(&mut rendering);
        let pipeline = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|(_, e)| e) }
            .context("creating sprite pipeline")?[0];
        self.pipelines.insert(key, pipeline);
        Ok(pipeline)
    }

    /// Record the built frame into the scene pass that is recording, whose
    /// attachment formats are `pass`. `frame_set` is the pass's frame uniform
    /// set (set 0).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        frame: usize,
        frame_set: vk::DescriptorSet,
        pass: PassFormats,
        gbuf: bool,
        velocity: bool,
        part: SpritePart,
    ) {
        if !self.has_work() {
            return;
        }
        let total = self.builder.frame().items.len();
        let (lo, hi) = match part {
            SpritePart::All => (0, total),
            SpritePart::Behind => (0, self.builder.frame().behind_items.min(total)),
            SpritePart::Front => (self.builder.frame().behind_items.min(total), total),
        };
        if lo >= hi {
            return;
        }
        let stencil_ok = has_stencil(pass.depth);
        let base = PipeKey {
            blend: SpriteBlend::Alpha,
            depth_test: false,
            array: false,
            role: Role::Draw(MaskMode::None),
            color: pass.color,
            depth: pass.depth,
            gbuf,
            gbuf_format: pass.gbuf,
            velocity,
            multiview: pass.multiview,
            samples: pass.samples,
        };
        unsafe {
            device.cmd_bind_vertex_buffers(cb, 0, &[self.vertex.handle, self.inst[frame].handle], &[0, 0]);
            device.cmd_bind_index_buffer(cb, self.index.handle, 0, vk::IndexType::UINT16);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[frame_set], &[]);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.layout, 2, &[self.anim_sets[frame]], &[]);
        }
        let mut bound_pipe = vk::Pipeline::null();
        let mut bound_tex = u32::MAX;
        let mut bound_push = [f32::NAN; 2];
        let count = hi - lo;
        // Mask shapes still written when this part's items end: replayed with
        // ZERO so the stencil bit the pass owns is 0 again afterwards (a mask
        // never spans the behind / front boundary).
        let tail: Vec<PassItem> = if stencil_ok && self.builder.frame().stats.mask_writes > 0 {
            let mut live: Vec<(u32, u32, u32)> = Vec::new();
            for it in &self.builder.frame().items[lo..hi] {
                match *it {
                    PassItem::MaskWrite { material, texture, draw } => live.push((material, texture, draw)),
                    PassItem::MaskErase { draw, .. } => live.retain(|l| l.2 != draw),
                    PassItem::Batch(_) => {}
                }
            }
            live.into_iter().map(|(material, texture, draw)| PassItem::MaskErase { material, texture, draw }).collect()
        } else {
            Vec::new()
        };
        for n in 0..count + tail.len() {
            let item = if n < count { self.builder.frame().items[lo + n] } else { tail[n - count] };
            let (material, texture, role, first, draws) = match item {
                PassItem::Batch(b) => {
                    let role = if stencil_ok { Role::Draw(b.stencil) } else { Role::Draw(MaskMode::None) };
                    if !stencil_ok && b.stencil != MaskMode::None && !self.warned_stencil {
                        self.warned_stencil = true;
                        tracing::warn!("sprite masks need a depth format with stencil; masked sprites draw unmasked");
                    }
                    (b.material, b.texture, role, b.first_draw, b.draw_count)
                }
                PassItem::MaskWrite { material, texture, draw } if stencil_ok => (material, texture, Role::MaskWrite, draw, 1),
                PassItem::MaskErase { material, texture, draw } if stencil_ok => (material, texture, Role::MaskErase, draw, 1),
                _ => continue,
            };
            let Some(Some(tex)) = self.textures.get(texture as usize) else { continue };
            let (set, array) = (tex.set, tex.array);
            let xz = material & SPRITE_MATERIAL_XZ != 0;
            let mat = self.materials.get((material & !SPRITE_MATERIAL_XZ) as usize).copied().unwrap_or_default();
            let pipeline = match self.pipeline(device, PipeKey { blend: mat.blend, depth_test: xz, array, role, ..base }) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("sprite pipeline: {e:#}");
                    continue;
                }
            };
            let params = [mat.blend.shader_mode(), mat.alpha_cutoff];
            unsafe {
                if pipeline != bound_pipe {
                    device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    bound_pipe = pipeline;
                }
                if texture != bound_tex {
                    device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.layout, 1, &[set], &[]);
                    bound_tex = texture;
                }
                if params != bound_push {
                    let push = [params[0], params[1], self.time, 0.0f32];
                    device.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, bytemuck::cast_slice(&push));
                    bound_push = params;
                }
                // The command list matches VkDrawIndexedIndirectCommand; recorded as
                // direct draws because drawIndirectFirstInstance is not enabled.
                for k in first..first + draws {
                    let d = self.builder.frame().draws[k as usize];
                    device.cmd_draw_indexed(cb, d.index_count, d.instance_count, d.first_index, d.vertex_offset, d.first_instance);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stencil_formats() {
        assert!(!has_stencil(vk::Format::D32_SFLOAT));
        assert!(has_stencil(vk::Format::D32_SFLOAT_S8_UINT));
        assert!(has_stencil(vk::Format::D24_UNORM_S8_UINT));
        assert!(!has_stencil(vk::Format::UNDEFINED));
    }

    #[test]
    fn stencil_states_map_to_vulkan() {
        assert!(stencil_op(Role::Draw(MaskMode::None)).is_none());
        let w = stencil_op(Role::MaskWrite).unwrap();
        assert_eq!((w.compare_op, w.pass_op, w.write_mask, w.reference), (vk::CompareOp::ALWAYS, vk::StencilOp::REPLACE, 0x80, 0x80));
        let e = stencil_op(Role::MaskErase).unwrap();
        assert_eq!((e.pass_op, e.reference, e.write_mask), (vk::StencilOp::ZERO, 0, 0x80));
        let i = stencil_op(Role::Draw(MaskMode::VisibleInside)).unwrap();
        assert_eq!((i.compare_op, i.write_mask, i.pass_op, i.compare_mask), (vk::CompareOp::EQUAL, 0, vk::StencilOp::KEEP, 0x80));
        let o = stencil_op(Role::Draw(MaskMode::VisibleOutside)).unwrap();
        assert_eq!(o.compare_op, vk::CompareOp::NOT_EQUAL);
    }

    #[test]
    fn default_input_is_empty_and_clears() {
        let mut i = SpriteFrameInput::default();
        assert!(i.is_empty());
        i.sprites.push(SpriteSource::default());
        assert!(!i.is_empty());
        i.clear();
        assert!(i.is_empty());
    }
}
