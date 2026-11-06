//! Persistent, camera-independent material cards for realtime Flux.
//!
//! These are runtime captures of scene geometry, not FluxBaker lightmaps.
//! Geometry-derived interior cards supplement six exterior captures. Individual
//! cards are updated under a budget, retain their slots
//! across camera movement/reordering, and never include displayed GI intensity.
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};

use anyhow::Result;
use ash::{vk, vk::Handle};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use gpu_allocator::{MemoryLocation, vulkan::{Allocation, Allocator}};

use crate::{alloc::Buffer, frame::FRAMES_IN_FLIGHT, screen_pass::ScreenImage,
    AlphaMode, DrawCmd, GpuMesh, Material, MaterialFx, Vertex};

pub(crate) const CARD_SIZE: u32 = 64;
const MAX_CARDS: usize = 1024;
const CAPTURE_BUDGET: usize = 12;
const LIGHTING_DIV: u32 = 4;
// Direct illumination converges more frequently than radiosity. Keep the
// existing 12-page indirect budget at speed 1; do not tie tracing to captures.
const LIGHTING_BUDGETS: [usize;2] = [24,12];

/// View demand only changes priority, never residency or offscreen eligibility.
pub(crate) struct LightingView { position: Vec3, planes: [glam::Vec4;6] }
impl From<crate::CameraData> for LightingView {
    fn from(camera: crate::CameraData) -> Self {
        Self {position:camera.position,planes:crate::frustum_planes(camera.proj*camera.view)}
    }
}
fn lighting_weight(bounds: [Vec3;2], views: &[LightingView], used: [Option<u64>;2], serial: u64) -> f32 {
    let center=(bounds[0]+bounds[1])*0.5;
    let half=(bounds[1]-bounds[0])*0.5;
    let mut distance=f32::INFINITY;
    let mut near_frustum=false;
    for view in views {
        distance=distance.min((view.position-view.position.clamp(bounds[0],bounds[1])).length());
        near_frustum |= view.planes.iter().all(|p|
            p.truncate().dot(center)+p.w+half.dot(p.truncate().abs()) >= -5.0);
    }
    // UE scales this by the first radiance-cache clipmap extent. Flux has no
    // clipmaps yet: use 50 metres, and retain age-based offscreen updates.
    let mut weight=if views.is_empty() {1.0} else {(if near_frustum {2.0} else {1.0})/(1.0+distance/50.0)};
    // Feedback is consumed only after the originating frame fence retires.
    for (kind,last) in used.into_iter().enumerate() {
        if last.is_some_and(|last|serial.saturating_sub(last)<=FRAMES_IN_FLIGHT as u64+2) {
            weight=weight.max(if kind==1 {4.0} else {2.0});
        }
    }
    weight
}
fn lighting_priority(light: PageLighting, weight: f32, serial: u64, max_wait: u64, hardware: bool) -> (u8,u64,u64) {
    let age=serial.saturating_sub(light.last);
    // A continuous stream of new/moving objects must not starve old pages.
    // Overdue pages are oldest-first; otherwise use UE's logarithmic age and
    // view-demand buckets. Give new pages a finite boost: unconditional
    // priority would let a moving mesh consume the whole radiosity budget.
    if age>=max_wait {return (2,age,age);}
    let weight=if light.valid {weight} else {weight.max(4.0)};
    // Software transport is currently more sensitive to variable page cadence
    // than triangle tracing. Preserve its oldest-first cadence; view demand
    // orders equally old pages until its reconstruction can tolerate weighted
    // update frequencies (LongDark camera-motion regression).
    if !hardware {return (0,age,(weight*1024.0) as u64);}
    let bucket=(4.0*age as f32*weight).max(1.0).log2().floor() as u64;
    (0,bucket,age)
}

#[derive(Clone,Copy,PartialEq,Eq,Hash)]
pub(crate) struct LightingSignature { transport: u64, illumination: u64, hardware: bool }

#[derive(Clone,Copy,Default)]
struct PageLighting { valid: bool, reset: bool, last: u64, updates: u32 }

fn lighting_speed(value: Option<f32>) -> f32 {
    value.filter(|v|v.is_finite()).unwrap_or(1.0).clamp(0.5,16.0)
}

pub(crate) fn shared_lighting_speed(scene: Option<Option<f32>>, camera: Option<Option<f32>>) -> f32 {
    // Surface lighting is shared. Use the highest participating view's demand,
    // independent of which view happens to record its gather first.
    [scene,camera].into_iter().flatten().map(lighting_speed).reduce(f32::max).unwrap_or(1.0)
}

const FORMATS: [vk::Format; 4] = [vk::Format::R32G32B32A32_SFLOAT,
    vk::Format::R16G16B16A16_SFLOAT, vk::Format::R8G8B8A8_UNORM, vk::Format::R16G16B16A16_SFLOAT];
const DEPTH: vk::Format = vk::Format::D32_SFLOAT;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Card {
    pub view_proj: [[f32; 4]; 4],
    /// Outward card direction, world units per texel.
    pub normal_texel: [f32; 4],
    /// Pixel origin xy, edge length; w: material resident (0), direct previous/
    /// current (1/2), indirect previous/current (4/5), capture transform (6).
    pub rect: [u32; 4],
    /// Rigid/uniform-scale delta from the material capture to the current
    /// object transform. Bit 6 of rect.w enables it; static lookups skip it.
    pub capture_to_world: [[f32;4];4],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Push {
    model: [[f32; 4]; 4], base: [f32; 4], emission: [f32; 4],
    material: [f32; 4], info: [u32; 4],
}
const _: () = assert!(size_of::<Push>() == 128 && size_of::<Card>() == 160);

// Captured tangent-space normal maps can rotate and scale uniformly with the
// geometry. A changed nonuniform stretch/shear needs a new tangent frame.
fn reusable_transform(delta: Mat4) -> bool {
    let axes=[delta.x_axis.truncate(),delta.y_axis.truncate(),delta.z_axis.truncate()];
    let lengths=axes.map(|a|a.length_squared());
    let scale=lengths[0];
    delta.is_finite() && delta.determinant()>0.0 && scale>1e-12
        && lengths.iter().all(|s|(*s-scale).abs()<=scale*1e-4)
        && axes[0].dot(axes[1]).abs()<=scale*1e-4
        && axes[0].dot(axes[2]).abs()<=scale*1e-4
        && axes[1].dot(axes[2]).abs()<=scale*1e-4
}

pub(crate) fn vertex_bounds(vertices: &[Vertex]) -> [Vec3; 2] {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for v in vertices {
        let p = Vec3::from(v.position);
        if p.is_finite() { lo = lo.min(p); hi = hi.max(p); }
    }
    if !lo.is_finite() { return [Vec3::ZERO; 2]; }
    [lo, hi]
}
fn world_bounds(bounds: [Vec3; 2], model: Mat4) -> [Vec3; 2] {
    let center = (bounds[0] + bounds[1]) * 0.5;
    let half = (bounds[1] - bounds[0]) * 0.5;
    let c = model.transform_point3(center);
    let h = model.x_axis.truncate().abs() * half.x
        + model.y_axis.truncate().abs() * half.y + model.z_axis.truncate().abs() * half.z;
    [c-h, c+h]
}
fn card_for(capture: crate::flux_cards::Capture, model: Mat4, slot: usize, side: u32) -> Card {
    let bounds = capture.bounds;
    let center = (bounds[0]+bounds[1])*0.5;
    let half = ((bounds[1]-bounds[0])*0.5).max(Vec3::splat(1e-6));
    let n = crate::flux_cards::AXES[capture.face];
    let up = if n.y.abs() > 0.5 { Vec3::Z } else { Vec3::Y };
    let right = up.cross(n);
    // Two-texel border; the depth bounds deliberately preserve the selected
    // interior near plane so a nearer layer cannot obscure this capture.
    let hw = half.dot(right.abs()).max(0.005) * CARD_SIZE as f32 / (CARD_SIZE-4) as f32;
    let hh = half.dot(up.abs()).max(0.005) * CARD_SIZE as f32 / (CARD_SIZE-4) as f32;
    let depth = half.dot(n.abs());
    // The basis is known. Deriving it from eye-center loses the direction for
    // a thin plane far from the mesh origin (both positions round identically).
    let mut view = Mat4::from_cols(right.extend(0.0),up.extend(0.0),n.extend(0.0),Vec3::ZERO.extend(1.0)).transpose();
    view.w_axis = Vec3::new(-center.dot(right),-center.dot(up),-center.dot(n)-depth).extend(1.0);
    let proj = Mat4::orthographic_rh(-hw,hw,-hh,hh,0.0,2.0*depth);
    let inverse = model.inverse();
    let normal = inverse.transpose().transform_vector3(n).normalize();
    let texel = 2.0*(hw*model.transform_vector3(right).length())
        .max(hh*model.transform_vector3(up).length())/CARD_SIZE as f32;
    let tile = slot as u32;
    Card { view_proj: (proj*view*inverse).to_cols_array_2d(),
        normal_texel: normal.extend(texel).to_array(),
        rect: [(tile%side)*CARD_SIZE,(tile/side)*CARD_SIZE,CARD_SIZE,1],
        capture_to_world:Mat4::IDENTITY.to_cols_array_2d() }
}
fn capture_push(d: &DrawCmd, m: &Material, time: f32) -> Push {
    let p = &m.params;
    let base = if m.features.per_instance { d.instance_color.unwrap_or(p.base_color) } else { p.base_color };
    let param = if m.features.per_instance { d.instance_param.unwrap_or([p.metallic,p.roughness,0.0,0.0]) }
        else { [p.metallic,p.roughness,0.0,0.0] };
    let flags = u32::from(m.features.normal_map && m.has_normal_texture)
        | u32::from(m.features.emission)*2
        | u32::from(matches!(m.features.alpha_mode,AlphaMode::Cutout|AlphaMode::Transclipping))*4
        | u32::from(m.features.double_sided)*8 | u32::from(m.features.cull_front)*16;
    Push { model: d.transform.to_cols_array_2d(), base,
        emission: [p.emission_color[0]*p.emission_intensity,p.emission_color[1]*p.emission_intensity,
            p.emission_color[2]*p.emission_intensity,0.0],
        material: [param[0],param[1],p.alpha_cutoff,p.normal_strength],
        info: [0,flags,time.to_bits(),u32::from(d.static_geometry)] }
}

struct Atlas {
    images: [ScreenImage; 4],
    lighting: [[ScreenImage; 2]; 2], light_index: usize, light_initialized: bool,
    depth: vk::Image, depth_view: vk::ImageView,
    depth_alloc: Option<Allocation>, initialized: bool, side: u32,
}
impl Atlas {
    fn new(device: &ash::Device, alloc: &mut Allocator, side: u32) -> Result<Self> {
        let extent = vk::Extent2D { width: side*CARD_SIZE, height: side*CARD_SIZE };
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC;
        let mut images = Vec::new();
        for (i,format) in FORMATS.into_iter().enumerate() {
            // Mip2 stores the exact geometry receiver used by each lighting
            // probe. Its compact footprint avoids high-resolution atlas reads
            // at every radiance lookup; it is never averaged across geometry.
            let mips=if i==0 {3} else {1};
            images.push(ScreenImage::new(device,alloc,format,usage|vk::ImageUsageFlags::TRANSFER_DST,extent,mips,"Flux surface material atlas")?);
        }
        let light_extent = vk::Extent2D { width: extent.width/LIGHTING_DIV, height: extent.height/LIGHTING_DIV };
        let mut lighting = Vec::new();
        for _ in 0..2 {
            let mut pair = Vec::new();
            for name in ["Flux cached direct radiance","Flux cached indirect radiance"] {
                pair.push(ScreenImage::new(device,alloc,vk::Format::R16G16B16A16_SFLOAT,
                    vk::ImageUsageFlags::SAMPLED|vk::ImageUsageFlags::STORAGE|vk::ImageUsageFlags::TRANSFER_SRC|vk::ImageUsageFlags::TRANSFER_DST,
                    light_extent,1,name)?);
            }
            lighting.push(pair.try_into().ok().unwrap());
        }
        let (depth,allocation) = crate::create_image(device,alloc,DEPTH,vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,extent,"Flux card depth")?;
        let depth_view = crate::create_view(device,depth,DEPTH,vk::ImageAspectFlags::DEPTH)?;
        Ok(Self { images: images.try_into().ok().unwrap(), lighting: lighting.try_into().ok().unwrap(), light_index: 0, light_initialized: false, depth, depth_view,
            depth_alloc: Some(allocation), initialized: false, side })
    }
    fn destroy(&mut self, device: &ash::Device, alloc: &mut Allocator) {
        for image in self.images.iter_mut().chain(self.lighting.iter_mut().flatten()) { image.destroy(device,alloc); }
        unsafe { device.destroy_image_view(self.depth_view,None); device.destroy_image(self.depth,None); }
        if let Some(a) = self.depth_alloc.take() { let _ = alloc.free(a); }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Node { min: [f32;3], left: u32, max: [f32;3], right: u32 }
fn build_nodes(items: &mut [([Vec3;2],usize)], nodes: &mut Vec<Node>) -> u32 {
    let index = nodes.len() as u32;
    let mut lo = Vec3::splat(f32::INFINITY); let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for (b,_) in items.iter() { lo = lo.min(b[0]); hi = hi.max(b[1]); }
    nodes.push(Node { min: lo.to_array(), max: hi.to_array(), left: 0, right: 0 });
    if items.len() == 1 {
        nodes[index as usize].left = items[0].1 as u32|0x80000000;
    } else {
        let size = hi-lo;
        let axis = if size.x>=size.y && size.x>=size.z {0} else if size.y>=size.z {1} else {2};
        items.sort_by(|a,b| (a.0[0][axis]+a.0[1][axis]).total_cmp(&(b.0[0][axis]+b.0[1][axis])));
        let mid = items.len()/2;
        let (a,b) = items.split_at_mut(mid);
        build_nodes(a,nodes); build_nodes(b,nodes);
    }
    // Preorder escape index: traversal skips a rejected subtree without a
    // per-ray stack, reducing register pressure around hardware ray queries.
    nodes[index as usize].right = nodes.len() as u32;
    index
}
#[derive(Clone,Copy)]
pub(crate) struct Resources {
    pub views: [vk::ImageView;4], pub cards: vk::Buffer, pub nodes: vk::Buffer, pub node_count: u32,
    pub lighting: [vk::ImageView;2],
}
/// Order view hit-feedback writes before later compute accesses and make them
/// visible to the host once this frame slot's submission fence has retired.
pub(crate) fn finish_feedback(device: &ash::Device, cb: vk::CommandBuffer, resources: Option<Resources>) {
    if let Some(resources)=resources {
        let barrier=vk::BufferMemoryBarrier::default().buffer(resources.nodes).size(vk::WHOLE_SIZE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED).dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(vk::AccessFlags::SHADER_READ|vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ|vk::AccessFlags::SHADER_WRITE|vk::AccessFlags::HOST_READ);
        unsafe {device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER|vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),&[],&[barrier],&[]);}
    }
}
#[derive(Clone,Copy)]
pub(crate) struct LightingUpdate {
    pub previous: [vk::ImageView;2], pub output: [vk::ImageView;2],
    pub images: [vk::Image;2], pub page_extent: vk::Extent2D,
    pub card_count: u32,
}
struct Slot { buffer: Buffer, nodes: Buffer, nodes_capacity: usize, node_count: u32, update_base: u32,
    feedback_serial: Option<u64>, generations: Vec<u64>, pool: vk::DescriptorPool, set: vk::DescriptorSet }
#[derive(Clone,Copy)]
struct Refresh { stamp: Option<u64>, response: u64 }
struct Entry { slot: usize, signature: Option<u64>, refresh: Option<u64>, response: Option<u64>, last_capture: u64, card: Card, bounds: Option<[Vec3;2]>,
    lighting: [PageLighting;2], generation: u64, last_used: [Option<u64>;2],
    model:Mat4,capture_model:Mat4 }
fn apply_feedback(used: &mut [Option<u64>;2], generation: u64, submitted_generation: u64, flags: u32, stamp: u64) {
    if generation!=submitted_generation {return;}
    for kind in 0..2 {if flags&(1<<kind)!=0 {
        used[kind]=Some(used[kind].map_or(stamp,|old|old.max(stamp)));
    }}
}

pub(crate) struct SurfaceCache {
    atlas: Atlas,
    retired: Vec<(u64,Atlas)>,
    entries: BTreeMap<Vec<u32>, Entry>,
    slots: Vec<Slot>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    last_frame: Option<u64>, light_frame: Option<u64>,
    material_signature: u64,
    occlusion_signature: u64, lighting_signature: Option<(u64,LightingSignature)>,
    cards: Vec<Card>,
    next_generation: u64,
    pub captures: usize,
    pub resident: usize,
}
impl SurfaceCache {
    pub fn radiance_key(&self)->u64 {
        let mut hash=std::collections::hash_map::DefaultHasher::new();
        (self.material_signature,self.occlusion_signature,self.lighting_signature).hash(&mut hash);hash.finish()
    }

    pub fn new(device: &ash::Device, alloc: &mut Allocator, material_layout: vk::DescriptorSetLayout) -> Result<Self> {
        let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let binding = [vk::DescriptorSetLayoutBinding::default().binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(stages)];
        let set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),None)? };
        let push = [vk::PushConstantRange::default().stage_flags(stages).size(size_of::<Push>() as u32)];
        let layouts = [set_layout,material_layout];
        let layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&push),None)? };
        let pipeline = capture_pipeline(device,layout)?;
        let mut slots = Vec::new();
        for _ in 0..FRAMES_IN_FLIGHT {
            let buffer = Buffer::new(device,alloc,(MAX_CARDS*size_of::<Card>()) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::CpuToGpu,"Flux card metadata")?;
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 }];
            let pool = unsafe { device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes),None)? };
            let ls = [set_layout];
            let set = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&ls))?[0] };
            let info = [vk::DescriptorBufferInfo::default().buffer(buffer.handle).range(vk::WHOLE_SIZE)];
            let write = vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info);
            unsafe { device.update_descriptor_sets(&[write],&[]); }
            let nodes_capacity = MAX_CARDS*4+128;
            let nodes = Buffer::new(device,alloc,(16+nodes_capacity*size_of::<Node>()) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::CpuToGpu,"Flux surface BVH")?;
            slots.push(Slot { buffer,nodes,nodes_capacity,node_count: 0,update_base: 0,
                feedback_serial:None,generations:vec![0;MAX_CARDS],pool,set });
        }
        Ok(Self { atlas: Atlas::new(device,alloc,8)?, retired: Vec::new(), entries: BTreeMap::new(),
            slots,set_layout,layout,pipeline,last_frame: None,light_frame: None,
            material_signature: 0,occlusion_signature: 0,lighting_signature: None,cards: vec![Card::zeroed();MAX_CARDS],next_generation:1,captures: 0,resident: 0 })
    }

    /// Called only after this frame slot's fence retired, at most once per frame
    /// for every view combined. Captures include off-screen objects as well.
    #[allow(clippy::too_many_arguments)]
    pub fn record(&mut self, device: &ash::Device, alloc: &mut Allocator, cb: vk::CommandBuffer,
        frame: usize, serial: u64, time: f32, draws: &[DrawCmd], meshes: &[GpuMesh],
        materials: &[Material], versions: &[u32], skinner: Option<&crate::skin::GpuSkinner>,
        dynamic_textures: &BTreeSet<usize>, mesh_sources: &[Option<crate::GiSource>], gdf_sources: &[Option<crate::GiSource>]) -> Result<()> {
        if self.last_frame == Some(serial) { return Ok(()); }
        self.last_frame = Some(serial);
        self.captures = 0;
        // This frame slot's fence has retired. Never apply feedback to a page
        // that reused the physical slot after its originating GPU submission.
        let slot=&self.slots[frame];
        if let (Some(stamp),Some(bytes))=(slot.feedback_serial,slot.nodes.mapped()) {
            for entry in self.entries.values_mut() {
                if entry.generation!=slot.generations[entry.slot] {continue;}
                let offset=16+(slot.update_base as usize+MAX_CARDS+entry.slot)*size_of::<Node>()+12;
                let flags=u32::from_ne_bytes(bytes[offset..offset+4].try_into().unwrap());
                apply_feedback(&mut entry.last_used,entry.generation,slot.generations[entry.slot],flags,stamp);
            }
        }
        for i in (0..self.retired.len()).rev() {
            if self.retired[i].0 <= serial { self.retired.swap_remove(i).1.destroy(device,alloc); }
        }
        let mut wanted = Vec::new();
        let mut occluders = BTreeSet::new();
        for (i,d) in draws.iter().enumerate() {
            let (Some(m),Some(mesh)) = (materials.get(d.material.0),meshes.get(d.mesh.0)) else { continue; };
            let mut occluder = vec![d.mesh.0 as u32,d.material.0 as u32,d.shadow_casting as u32,
                versions.get(d.mesh.0).copied().unwrap_or(0),m.features.alpha_mode as u32,
                u32::from(m.features.double_sided) | u32::from(m.features.cull_front)*2 | u32::from(m.features.water)*4];
            occluder.extend(d.transform.to_cols_array().map(f32::to_bits));
            // Mask coverage is geometry for transport: changing a hole must
            // refresh incident lighting on other pages, not just this card.
            if matches!(m.features.alpha_mode,AlphaMode::Cutout|AlphaMode::Transclipping) {
                let p=&m.params;
                let alpha=if m.features.per_instance {d.instance_color.unwrap_or(p.base_color)[3]} else {p.base_color[3]};
                occluder.extend([alpha,p.alpha_cutoff,p.albedo_tiling[0],p.albedo_tiling[1],
                    p.albedo_offset[0],p.albedo_offset[1],p.base_scroll[0],p.base_scroll[1]].map(f32::to_bits));
                let descriptor=m.set.as_raw();occluder.extend([descriptor as u32,(descriptor>>32) as u32]);
                for i in [0,4] {
                    occluder.push(m.slots[i].map_or(u32::MAX,|t|t.0 as u32));
                    if m.slots[i].is_some_and(|t|dynamic_textures.contains(&t.0)) {occluder.extend([serial as u32,(serial>>32) as u32]);}
                }
                if p.base_scroll!=[0.0;2] {occluder.push(time.to_bits());}
            }
            occluders.insert(occluder);
            // Custom displacement, blended/transmissive surfaces and skinned
            // capture specialization remain separate parity work.
            if !d.shadow_casting.visible() || m.features.water || m.custom_shader.is_some()
                || skinner.is_some_and(|s| s.prev_vertex_buffer(d.mesh.0).is_some() || s.pose_of(d.mesh.0).is_some())
                || m.features.alpha_mode == AlphaMode::Blend || !d.transform.is_finite()
                || d.transform.determinant().abs() < 1e-10 || mesh.index_count == 0
                || mesh.vertex_buffer.handle == vk::Buffer::null() { continue; }
            let mut push = capture_push(d,m,0.0);
            let mut key = vec![d.mesh.0 as u32,d.material.0 as u32];
            if let Some(source)=d.gi_source {
                key.push(1);
                key.extend((0..4).map(|i|(source.object>>(i*32)) as u32));key.push(source.slot);
                // Stable object/slot identity follows motion and per-instance
                // material edits. Their values remain capture contents.
                push.model=Mat4::IDENTITY.to_cols_array_2d();
            } else {
                key.push(0);
                key.extend(d.transform.to_cols_array().map(f32::to_bits));
                // Without source IDs, explicit overrides distinguish
                // coincident draws. Material defaults remain capture contents.
                for value in [d.instance_color,d.instance_param] {
                    let value=if m.features.per_instance {value} else {None};
                    key.push(u32::from(value.is_some()));
                    if let Some(value)=value { key.extend(value.map(f32::to_bits)); }
                }
            }
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            bytemuck::bytes_of(&push).hash(&mut hash);
            bytemuck::bytes_of(&MaterialFx::from_params(&m.params)).hash(&mut hash);
            versions.get(d.mesh.0).copied().unwrap_or(0).hash(&mut hash);
            // A descriptor replacement reflects texture/UV/material edits.
            m.set.as_raw().hash(&mut hash);
            for t in m.slots { t.map(|v| v.0).hash(&mut hash); }
            // Only mesh/transform/facing changes invalidate captured geometry.
            // Other changes refresh material contents under the capture budget.
            let mut geometry=std::collections::hash_map::DefaultHasher::new();
            versions.get(d.mesh.0).copied().unwrap_or(0).hash(&mut geometry);
            (push.info[1] & (8|16)).hash(&mut geometry);
            let geometry=geometry.finish();
            let dynamic=m.slots.iter().flatten().any(|t|dynamic_textures.contains(&t.0));
            let animated=m.params.base_scroll != [0.0;2] || m.params.emission_scroll != [0.0;2] || m.params.emission_pulse>0.0;
            if dynamic { serial.hash(&mut hash); }
            if animated { time.to_bits().hash(&mut hash); }
            let mut response=std::collections::hash_map::DefaultHasher::new();
            let mut reflected=push;reflected.emission=[0.0;4];reflected.info[1]&=!2;
            bytemuck::bytes_of(&reflected).hash(&mut response);
            let mut fx=MaterialFx::from_params(&m.params);
            fx.toon[2]=0.0;fx.scroll[2..].fill(0.0);fx.emission_st=[0.0;4];
            bytemuck::bytes_of(&fx).hash(&mut response);
            // Rebinding may also change samplers. Conservatively reset reflected
            // lighting for descriptor edits, but never invalidate geometry.
            m.set.as_raw().hash(&mut response);
            for (i,t) in m.slots.iter().enumerate().filter(|(i,_)| ![3,5].contains(i)) {
                i.hash(&mut response);t.map(|v|v.0).hash(&mut response);
                if t.is_some_and(|t|dynamic_textures.contains(&t.0)) {serial.hash(&mut response);}
            }
            if m.params.base_scroll != [0.0;2] {time.to_bits().hash(&mut response);}
            let refresh=Refresh {stamp:Some(hash.finish()),response:response.finish()};
            for (card_index, capture) in mesh.surface_geometry.captures.iter().enumerate() {
                let mut card_key = key.clone(); card_key.push(card_index as u32);
                wanted.push((i,card_key,geometry,*capture,refresh));
            }
        }
        let mut scene_hash = std::collections::hash_map::DefaultHasher::new();
        occluders.hash(&mut scene_hash); self.occlusion_signature = scene_hash.finish();
        let mut material_hash=std::collections::hash_map::DefaultHasher::new();
        wanted.iter().map(|(_,key,_,_,r)|(key,r.stamp)).collect::<BTreeSet<_>>().hash(&mut material_hash);
        self.material_signature=material_hash.finish();
        let keys: BTreeSet<_> = wanted.iter().map(|(_,k,_,_,_)| k.clone()).collect();
        let needed = keys.len().min(MAX_CARDS);
        let mut side = self.atlas.side;
        while (side*side) < needed as u32 { side *= 2; }
        if side != self.atlas.side {
            let next = Atlas::new(device,alloc,side)?;
            self.retired.push((serial+FRAMES_IN_FLIGHT as u64+1,std::mem::replace(&mut self.atlas,next)));
            // Tile coordinates change on resize. Pending entries cannot sample old data.
            self.entries.clear();
            self.lighting_signature = None;
        }
        self.entries.retain(|k,_| keys.contains(k));
        let mut used: BTreeSet<_> = self.entries.values().map(|e| e.slot).collect();
        let capacity = (self.atlas.side*self.atlas.side) as usize;
        for (i,key,_,capture,_) in &wanted {
            if !self.entries.contains_key(key) {
                let Some(slot) = (0..capacity).find(|s| !used.contains(s)) else { continue; };
                used.insert(slot);
                let d = &draws[*i];
                let m=&materials[d.material.0];
                let bounds = capture.visible_bounds(m.features.double_sided,m.features.cull_front).map(|b|world_bounds(b,d.transform));
                self.entries.insert(key.clone(),Entry { slot,signature: None,refresh: None,response: None,last_capture: 0,
                    card: card_for(*capture,d.transform,slot,self.atlas.side), bounds, lighting: [PageLighting {last:serial.saturating_sub(1),..Default::default()};2],
                    generation:self.next_generation,last_used:[None;2],model:d.transform,capture_model:d.transform });
                self.next_generation+=1;
            }
        }
        for (i,key,_,capture,_) in &wanted {
            let Some(e)=self.entries.get_mut(key) else {continue;};
            let d=&draws[*i];
            if e.model==d.transform {continue;}
            let delta=d.transform*e.capture_model.inverse();
            e.model=d.transform;
            e.card=card_for(*capture,d.transform,e.slot,self.atlas.side);
            let m=&materials[d.material.0];
            e.bounds=capture.visible_bounds(m.features.double_sided,m.features.cull_front).map(|b|world_bounds(b,d.transform));
            if e.signature.is_some() && reusable_transform(delta) {
                e.card.capture_to_world=delta.to_cols_array_2d();e.card.rect[3]|=64;
                // Retain the last result until this page's turn, then replace
                // lighting evaluated at the old position/normal. Keep its age
                // so moving objects cannot monopolize the lighting budget.
                for light in &mut e.lighting {light.reset=true;}
            } else {
                e.signature=None;
                for light in &mut e.lighting {*light=PageLighting {last:light.last,..Default::default()};}
            }
        }
        // Oldest dirty card first: animated objects cannot starve other pages.
        // Hard-invalidated pages wait for capture; animation refresh keeps the
        // previous material available while other pages use the frame budget.
        let mut dirty: Vec<_> = wanted.iter().filter_map(|(i,k,s,c,refresh)| {
            let e = self.entries.get(k)?;
            (e.signature != Some(*s) || (refresh.stamp.is_some() && e.refresh!=refresh.stamp))
                .then_some((*i,k.clone(),*s,e.last_capture,*c,*refresh))
        }).collect();
        dirty.sort_by(|a,b| a.3.cmp(&b.3).then(a.1.cmp(&b.1)));
        dirty.dedup_by(|a,b| a.1==b.1);
        dirty.truncate(CAPTURE_BUDGET);
        for (i,key,sig,_,capture,refresh) in &dirty {
            let e = self.entries.get_mut(key).unwrap();
            let d = &draws[*i];
            e.card = card_for(*capture,d.transform,e.slot,self.atlas.side);
            e.capture_model=d.transform;
            let m=&materials[d.material.0];
            e.bounds = capture.visible_bounds(m.features.double_sided,m.features.cull_front).map(|b|world_bounds(b,d.transform));
            if e.signature!=Some(*sig) {
                // New receiver geometry rejects its own history, while keeping
                // its scheduling age. It must not starve unaffected receivers.
                for light in &mut e.lighting {
                    *light=PageLighting {last:light.last,..Default::default()};
                }
            } else if e.response!=Some(refresh.response) {
                // Keep scheduling age/readiness: making every animated page
                // "new" would monopolize lighting and starve static receivers.
                // Reset its accumulation only when it next gets relit.
                for light in &mut e.lighting { light.reset=true; }
            }
            e.signature=Some(*sig);e.refresh=refresh.stamp;e.response=Some(refresh.response);e.last_capture=serial;
        }
        let mut cards = vec![Card::zeroed();MAX_CARDS];
        let mut leaves = Vec::new();
        let mut instance_leaves: BTreeMap<usize,Vec<([Vec3;2],usize)>> = BTreeMap::new();
        for (draw,key,sig,_,_) in &wanted {
            if let Some(e) = self.entries.get(key) {
                if e.signature == Some(*sig) {
                    cards[e.slot] = e.card;
                    if let Some(bounds)=e.bounds {
                        // Lookup accepts a captured position within two card
                        // texels plus ray-hit uncertainty. Keep the BVH
                        // conservative for that SAME acceptance region; thin
                        // plane bounds must not reject otherwise valid hits.
                        let footprint=Vec3::splat(e.card.normal_texel[3]*2.0);
                        let leaf=([bounds[0]-footprint,bounds[1]+footprint],e.slot);
                        leaves.push(leaf);
                        instance_leaves.entry(*draw).or_default().push(leaf);
                    }
                }
            }
        }
        leaves.sort_by_key(|(_,slot)| *slot); leaves.dedup_by_key(|(_,slot)| *slot);
        let mut nodes = Vec::new();
        if !leaves.is_empty() { build_nodes(&mut leaves,&mut nodes); }
        self.slots[frame].node_count = nodes.len() as u32;
        // Exact ray hits address only their object's subtree. Keep the global
        // tree for GDF hits without instance identity. Map the original draw
        // indices, since TLAS construction can skip instances without a BLAS.
        let mut roots=vec![0u32;draws.len()];
        let mut shared_roots=BTreeMap::new();
        for (draw,mut items) in instance_leaves {
            items.sort_by_key(|(_,slot)|*slot); items.dedup_by_key(|(_,slot)|*slot);
            let key: Vec<_>=items.iter().map(|(_,slot)|*slot).collect();
            roots[draw]=*shared_roots.entry(key).or_insert_with(|| build_nodes(&mut items,&mut nodes)+1);
        }
        let mapping_start=nodes.len() as u32;
        nodes.extend(roots.iter().map(|root| Node {left:*root,..Node::zeroed()}));
        // Software mesh indices belong to the immutable SDF snapshot, not
        // the raster draw order. Resolve exact entity/slot IDs on the CPU.
        // Metadata follows the RT draw map, keeping its header/ABI unchanged.
        let source_roots: BTreeMap<_,_> = draws.iter().zip(&roots)
            .filter_map(|(draw,root)|draw.gi_source.map(|source|(source,*root))).collect();
        // Only a complete identified mesh snapshot may refine/reject a coarse
        // GDF hit. Legacy/procedural fields without mesh identity remain valid.
        let mesh_ids:BTreeSet<_>=mesh_sources.iter().flatten().copied().collect();
        let complete_gdf=!gdf_sources.is_empty() && gdf_sources.iter()
            .all(|source|source.is_some_and(|id|mesh_ids.contains(&id)));
        let mesh_mapping=nodes.len() as u32+1;
        nodes.push(Node {left:mesh_mapping,right:mesh_sources.len() as u32,
            min:[if complete_gdf {1.0} else {0.0},0.0,0.0],..Node::zeroed()});
        nodes.extend(mesh_sources.iter().map(|source|Node {
            left:source.and_then(|s|source_roots.get(&s).copied()).unwrap_or(0),
            right:u32::from(source.is_some()),..Node::zeroed()}));
        let gdf_mapping=nodes.len() as u32+1;
        nodes.push(Node {left:gdf_mapping,right:gdf_sources.len() as u32,..Node::zeroed()});
        nodes.extend(gdf_sources.iter().map(|source|Node {
            left:source.and_then(|s|source_roots.get(&s).copied()).unwrap_or(0),
            right:u32::from(source.is_some()),..Node::zeroed()}));
        let update_base=nodes.len() as u32;
        // Fixed-size update queue, followed by per-card atomic hit feedback.
        nodes.resize(nodes.len()+MAX_CARDS*2,Node::zeroed());
        // Independent of view history resets and the recycling frame-slot
        // index: every spatial feedback class must eventually be sampled.
        nodes[update_base as usize+MAX_CARDS].right=serial as u32;
        let slot=&mut self.slots[frame]; slot.update_base=update_base;
        slot.feedback_serial=Some(serial);
        slot.generations.fill(0);
        for entry in self.entries.values() {slot.generations[entry.slot]=entry.generation;}
        if nodes.len()>slot.nodes_capacity {
            let capacity=nodes.len().next_power_of_two();
            let next=Buffer::new(device,alloc,(16+capacity*size_of::<Node>()) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::CpuToGpu,"Flux instance surface BVHs")?;
            // record() runs after this frame slot's fence retired.
            slot.nodes.destroy(device,alloc); slot.nodes=next; slot.nodes_capacity=capacity;
        }
        slot.nodes.write(0,bytemuck::cast_slice(&[mapping_start,draws.len() as u32,update_base,0]));
        slot.nodes.write(16,bytemuck::cast_slice(&nodes));
        self.slots[frame].buffer.write(0,bytemuck::cast_slice(&cards));
        self.cards = cards;
        self.resident = self.cards.iter().filter(|c| c.rect[3]!=0).count();
        if dirty.is_empty() { return Ok(()); }
        self.begin_capture(device,cb);
        for (i,key,_,_,_,_) in dirty {
            let e = &self.entries[&key]; let d = &draws[i];
            let m = &materials[d.material.0]; let mesh = &meshes[d.mesh.0];
            let mut push = capture_push(d,m,time);
            push.info[0] = e.slot as u32;
            self.draw_card(device,cb,frame,mesh,m,&e.card,&push);
            self.captures += 1;
        }
        self.end_capture(device,cb);
        prepare_lighting_geometry(device,cb,self.atlas.images[0].image,self.atlas.images[0].extent);
        Ok(())
    }

    pub fn resources(&self, frame: usize) -> Option<Resources> {
        self.atlas.initialized.then(|| Resources { views: self.atlas.images.each_ref().map(|im| im.view),
            cards: self.slots[frame].buffer.handle, nodes: self.slots[frame].nodes.handle,
            node_count: self.slots[frame].node_count, lighting: self.atlas.lighting[self.atlas.light_index].each_ref().map(|im| im.view) })
    }

    /// Reflections may run without GI. Do not bind stale or uninitialized
    /// lighting from an earlier frame when the gather did not update the cache.
    pub fn reflection_resources(&self, frame: usize, serial: u64) -> Option<Resources> {
        (self.light_frame == Some(serial) && self.atlas.light_initialized)
            .then(|| self.resources(frame)).flatten()
    }

    /// Update a bounded set of surface pages, independent of either camera.
    /// Reads the previous lighting atlas and writes a separate atlas, so no ray
    /// can observe another workgroup's partially updated bounce lighting.
    pub fn prepare_lighting(&mut self,device: &ash::Device,cb: vk::CommandBuffer,
        frame: usize,serial: u64,scene_signature: LightingSignature,speed: f32,views: &[LightingView]) -> Option<LightingUpdate> {
        if self.light_frame == Some(serial) || !self.atlas.initialized { return None; }
        self.light_frame = Some(serial);
        let signature = (self.occlusion_signature,scene_signature);
        if self.lighting_signature.is_none_or(|(_,lighting)|
            lighting.transport!=scene_signature.transport) {
            for e in self.entries.values_mut() { e.lighting=[PageLighting::default();2]; }
        } else if self.lighting_signature!=Some(signature) {
            // Moving lights/occluders and updated distance fields change
            // incident lighting, not the geometry of every cached receiver.
            // Retain page ages, validity and sampling sequences so changes on
            // every frame cannot starve all but the first budget's pages.
            // Evaluate fresh direct light when each page is scheduled. Keep the
            // short radiosity history: resetting it on every light animation
            // discards its variance reduction and interrupts bounce transport.
            for e in self.entries.values_mut() { e.lighting[0].reset=true; }
        }
        self.lighting_signature=Some(signature);
        let selected: [BTreeSet<usize>;2]=std::array::from_fn(|kind| {
            let budget=((LIGHTING_BUDGETS[kind] as f32*lighting_speed(Some(speed))).round() as usize).min(MAX_CARDS);
            let max_wait=(self.resident.div_ceil(budget)*4).max(4) as u64;
            let mut order: Vec<_>=self.entries.values()
                .filter(|e| e.bounds.is_some() && self.cards[e.slot].rect[3]!=0)
                .map(|e| (std::cmp::Reverse(lighting_priority(e.lighting[kind],
                    lighting_weight(e.bounds.unwrap(),views,e.last_used,serial),serial,max_wait,scene_signature.hardware)),e.slot)).collect();
            order.sort_unstable();
            order.truncate(budget);
            order.iter().map(|v|v.1).collect()
        });
        let mut updates=Vec::new();
        for e in self.entries.values_mut() {
            if e.bounds.is_none() || self.cards[e.slot].rect[3]==0 { continue; }
            let update=selected.each_ref().map(|set|set.contains(&e.slot));
            // Card bits: material 1; direct previous/current 2/4;
            // indirect previous/current 16/32. Queue entries carry independent
            // sample indices and history resets; no camera/frame based seeds.
            self.cards[e.slot].rect[3]=1 | (e.card.rect[3]&64)
                | u32::from(e.lighting[0].valid)*2 | u32::from(e.lighting[0].valid||update[0])*4
                | u32::from(e.lighting[1].valid)*16 | u32::from(e.lighting[1].valid||update[1])*32;
            if update.iter().any(|v|*v) {
                let mut flags=0;
                for kind in 0..2 {
                    flags |= u32::from(update[kind])<<kind;
                    flags |= u32::from(e.lighting[kind].valid && !e.lighting[kind].reset)<<(kind+2);
                }
                updates.push(Node {left:e.slot as u32,right:flags,
                    min:[(e.lighting[0].updates&0x000f_ffff) as f32,(e.lighting[1].updates&0x000f_ffff) as f32,0.0],
                    ..Node::zeroed()});
            }
            for kind in 0..2 { if update[kind] {
                let light=&mut e.lighting[kind];
                light.valid=true; light.reset=false; light.last=serial; light.updates=light.updates.wrapping_add(1);
            }}
        }
        self.slots[frame].buffer.write(0,bytemuck::cast_slice(&self.cards));
        if updates.is_empty() { return None; }
        // Only dispatch the union of the two scheduled page sets; the shader
        // evaluates direct light and/or radiosity according to each queue mask.
        let slot=&mut self.slots[frame];
        slot.nodes.write(12,bytemuck::bytes_of(&(updates.len() as u32)));
        slot.nodes.write(16+slot.update_base as usize*size_of::<Node>(),bytemuck::cast_slice(&updates));
        if !self.atlas.light_initialized {
            for im in self.atlas.lighting.iter().flatten() {
                light_barrier(device,cb,im.image,vk::ImageLayout::UNDEFINED,vk::ImageLayout::TRANSFER_DST_OPTIMAL);
                unsafe { device.cmd_clear_color_image(cb,im.image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue {float32:[0.0;4]},&[range(vk::ImageAspectFlags::COLOR)]); }
                light_barrier(device,cb,im.image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            }
            self.atlas.light_initialized=true;
        }
        let previous=self.atlas.light_index; let next=1-previous;
        let extent=self.atlas.lighting[next][0].extent;
        for i in 0..2 {
            let src=self.atlas.lighting[previous][i].image;
            let dst=self.atlas.lighting[next][i].image;
            light_barrier(device,cb,src,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            light_barrier(device,cb,dst,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            let sub=vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1);
            let copy=vk::ImageCopy::default().src_subresource(sub).dst_subresource(sub)
                .extent(vk::Extent3D {width:extent.width,height:extent.height,depth:1});
            unsafe { device.cmd_copy_image(cb,src,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,dst,vk::ImageLayout::TRANSFER_DST_OPTIMAL,&[copy]); }
            light_barrier(device,cb,src,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            light_barrier(device,cb,dst,vk::ImageLayout::TRANSFER_DST_OPTIMAL,vk::ImageLayout::GENERAL);
        }
        self.atlas.light_index=next;
        Some(LightingUpdate { previous:self.atlas.lighting[previous].each_ref().map(|im| im.view),
            output:self.atlas.lighting[next].each_ref().map(|im| im.view),
            images:self.atlas.lighting[next].each_ref().map(|im| im.image),
            page_extent:vk::Extent2D {width:CARD_SIZE/LIGHTING_DIV,height:CARD_SIZE/LIGHTING_DIV},card_count:updates.len() as u32 })
    }

    fn begin_capture(&mut self, device: &ash::Device, cb: vk::CommandBuffer) {
        let old = if self.atlas.initialized { vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL } else { vk::ImageLayout::UNDEFINED };
        let mut barriers: Vec<_> = self.atlas.images.iter().map(|im| vk::ImageMemoryBarrier::default()
            .image(im.image).old_layout(old).new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_access_mask(vk::AccessFlags::SHADER_READ).dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .subresource_range(range(vk::ImageAspectFlags::COLOR).level_count(im.mips))).collect();
        barriers.push(vk::ImageMemoryBarrier::default().image(self.atlas.depth)
            .old_layout(if self.atlas.initialized { vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL } else { vk::ImageLayout::UNDEFINED })
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ|vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .subresource_range(range(vk::ImageAspectFlags::DEPTH)));
        unsafe { device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT|vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS|vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::DependencyFlags::empty(),&[],&[],&barriers); }
        self.atlas.initialized = true;
    }
    fn end_capture(&self, device: &ash::Device, cb: vk::CommandBuffer) {
        let barriers: Vec<_> = self.atlas.images.iter().map(|im| vk::ImageMemoryBarrier::default()
            .image(im.image).old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL).new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ)
            .subresource_range(range(vk::ImageAspectFlags::COLOR).level_count(im.mips))).collect();
        unsafe { device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::COMPUTE_SHADER|vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),&[],&[],&barriers); }
    }
    #[allow(clippy::too_many_arguments)]
    fn draw_card(&self,device: &ash::Device,cb: vk::CommandBuffer,frame: usize,mesh: &GpuMesh,m: &Material,card: &Card,push: &Push) {
        let colors: Vec<_> = self.atlas.images.iter().map(|im| vk::RenderingAttachmentInfo::default()
            .image_view(im.level(0)).image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR).store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0;4] } })).collect();
        let depth = vk::RenderingAttachmentInfo::default().image_view(self.atlas.depth_view)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL).load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE).clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0,stencil: 0 } });
        let area = vk::Rect2D { offset: vk::Offset2D { x: card.rect[0] as i32,y: card.rect[1] as i32 }, extent: vk::Extent2D { width: CARD_SIZE,height: CARD_SIZE } };
        let render = vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&colors).depth_attachment(&depth);
        let viewport = vk::Viewport { x: area.offset.x as f32,y: area.offset.y as f32,width: CARD_SIZE as f32,height: CARD_SIZE as f32,min_depth: 0.0,max_depth: 1.0 };
        unsafe {
            device.cmd_begin_rendering(cb,&render);
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::GRAPHICS,self.pipeline);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::GRAPHICS,self.layout,0,&[self.slots[frame].set,m.set],&[]);
            device.cmd_set_viewport(cb,0,&[viewport]); device.cmd_set_scissor(cb,0,&[area]);
            device.cmd_bind_vertex_buffers(cb,0,&[mesh.vertex_buffer.handle],&[0]);
            device.cmd_bind_index_buffer(cb,mesh.index_buffer.handle,0,mesh.index_type);
            device.cmd_push_constants(cb,self.layout,vk::ShaderStageFlags::VERTEX|vk::ShaderStageFlags::FRAGMENT,0,bytemuck::bytes_of(push));
            device.cmd_draw_indexed(cb,mesh.index_count,1,0,0,0);
            device.cmd_end_rendering(cb);
        }
    }
    #[cfg(test)]
    pub fn debug_lighting_coverage(&self) -> [usize;2] {
        std::array::from_fn(|kind|self.entries.values().filter(|e|e.bounds.is_some()
            && self.cards[e.slot].rect[3]!=0 && e.lighting[kind].valid).count())
    }

    #[cfg(test)]
    pub fn debug_feedback(&self) -> [usize;2] {
        std::array::from_fn(|kind|self.entries.values().filter(|e|e.last_used[kind].is_some()).count())
    }
    #[cfg(test)]
    pub fn debug_card(&self,material:usize,face:usize)->(u64,u64,Card) {
        let e=self.entries.iter().find(|(k,_)|k[1] as usize==material && k.last()==Some(&(face as u32))).unwrap().1;
        (e.generation,e.last_capture,e.card)
    }
    #[cfg(test)]
    pub fn debug_lighting_updates(&self,material: usize,face: usize) -> ([u64;2],[u32;2]) {
        let total=std::array::from_fn(|i|self.entries.values().map(|e|e.lighting[i].updates as u64).sum());
        let entry=self.entries.iter().find(|(k,_)|k[1] as usize==material && k.last()==Some(&(face as u32))).unwrap().1;
        (total,entry.lighting.map(|v|v.updates))
    }
    #[cfg(test)]
    pub fn debug_lighting(&self,device: &ash::Device,alloc: &mut Allocator,
        pool: vk::CommandPool,queue: vk::Queue,material: usize,face: usize) -> Result<[[f32;4];2]> {
        let e=self.entries.iter().find(|(k,e)| k[1] as usize==material && k.last()==Some(&(face as u32)) && e.lighting.iter().all(|l|l.valid)).unwrap().1;
        let card=e.card;
        let mut buffer=Buffer::new(device,alloc,16,vk::BufferUsageFlags::TRANSFER_DST,MemoryLocation::GpuToCpu,"Flux radiance test")?;
        crate::alloc::one_time_submit(device,pool,queue,|cb| {
            for (i,im) in self.atlas.lighting[self.atlas.light_index].iter().enumerate() {
                light_barrier(device,cb,im.image,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                let copy=vk::BufferImageCopy::default().buffer_offset(i as u64*8)
                    .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                    .image_offset(vk::Offset3D { x:((card.rect[0]+CARD_SIZE/2)/LIGHTING_DIV) as i32,
                        y:((card.rect[1]+CARD_SIZE/2)/LIGHTING_DIV) as i32,z:0 })
                    .image_extent(vk::Extent3D {width:1,height:1,depth:1});
                unsafe {device.cmd_copy_image_to_buffer(cb,im.image,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,buffer.handle,&[copy]);}
                light_barrier(device,cb,im.image,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            }
        })?;
        let bytes=buffer.read(); let mut result=[[0.0;4];2];
        for i in 0..2 {for c in 0..4 {
            let start=i*8+c*2;
            result[i][c]=half::f16::from_bits(u16::from_ne_bytes(bytes[start..start+2].try_into().unwrap())).to_f32();
        }}
        buffer.destroy(device,alloc); Ok(result)
    }

    #[cfg(test)]
    pub fn debug_sample(&self, device: &ash::Device, alloc: &mut Allocator,
        pool: vk::CommandPool, queue: vk::Queue, face: usize) -> Result<[[f32;4];4]> {
        self.debug_sample_at(device,alloc,pool,queue,None,face,0)
    }
    #[cfg(test)]
    pub fn debug_lighting_receiver(&self,device:&ash::Device,alloc:&mut Allocator,
        pool:vk::CommandPool,queue:vk::Queue,material:usize,face:usize)->Result<[[f32;4];4]> {
        self.debug_sample_at(device,alloc,pool,queue,Some(material),face,LIGHTING_DIV/2)
    }
    #[cfg(test)]
    fn debug_sample_at(&self,device:&ash::Device,alloc:&mut Allocator,pool:vk::CommandPool,
        queue:vk::Queue,material:Option<usize>,face:usize,offset:u32)->Result<[[f32;4];4]> {
        let e=self.entries.iter().find(|(k,e)|k.last()==Some(&(face as u32)) && e.signature.is_some()
            && material.is_none_or(|m|k[1] as usize==m)).expect("captured surface").1;
        let card = e.card;
        let mut buffer = Buffer::new(device,alloc,64,vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,"Flux capture test readback")?;
        crate::alloc::one_time_submit(device,pool,queue,|cb| {
            let barrier = |from,to| {
                let barriers: Vec<_> = self.atlas.images.iter().map(|im| vk::ImageMemoryBarrier::default()
                    .image(im.image).old_layout(from).new_layout(to)
                    .src_access_mask(vk::AccessFlags::MEMORY_READ|vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ|vk::AccessFlags::MEMORY_WRITE)
                    .subresource_range(range(vk::ImageAspectFlags::COLOR).level_count(im.mips))).collect();
                unsafe { device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::ALL_COMMANDS,vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),&[],&[],&barriers); }
            };
            barrier(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            for (i,im) in self.atlas.images.iter().enumerate() {
                let copy = vk::BufferImageCopy::default().buffer_offset(i as u64*16)
                    .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                    .image_offset(vk::Offset3D { x: (card.rect[0]+CARD_SIZE/2+offset) as i32,y: (card.rect[1]+CARD_SIZE/2+offset) as i32,z: 0 })
                    .image_extent(vk::Extent3D { width: 1,height: 1,depth: 1 });
                unsafe { device.cmd_copy_image_to_buffer(cb,im.image,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,buffer.handle,&[copy]); }
            }
            barrier(vk::ImageLayout::TRANSFER_SRC_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        })?;
        let bytes = buffer.read();
        let mut result = [[0.0;4];4];
        for i in 0..4 { for c in 0..4 {
            result[i][c] = match i {
                0 => f32::from_ne_bytes(bytes[c*4..c*4+4].try_into().unwrap()),
                2 => bytes[i*16+c] as f32/255.0,
                _ => half::f16::from_bits(u16::from_ne_bytes(bytes[i*16+c*2..i*16+c*2+2].try_into().unwrap())).to_f32(),
            };
        } }
        buffer.destroy(device,alloc);
        Ok(result)
    }

    pub fn destroy(&mut self, device: &ash::Device, alloc: &mut Allocator) {
        self.atlas.destroy(device,alloc);
        for (_,a) in &mut self.retired { a.destroy(device,alloc); }
        self.retired.clear();
        for s in &mut self.slots {
            s.buffer.destroy(device,alloc);
            s.nodes.destroy(device,alloc);
            unsafe { device.destroy_descriptor_pool(s.pool,None); }
        }
        unsafe { device.destroy_pipeline(self.pipeline,None); device.destroy_pipeline_layout(self.layout,None);
            device.destroy_descriptor_set_layout(self.set_layout,None); }
    }
}
fn range(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default().aspect_mask(aspect).level_count(1).layer_count(1)
}
fn capture_pipeline(device: &ash::Device,layout: vk::PipelineLayout) -> Result<vk::Pipeline> {
    let vert = crate::screen_pass::shader_module(device,include_bytes!(concat!(env!("OUT_DIR"),"/flux_surface.vert.spv")))?;
    let frag = crate::screen_pass::shader_module(device,include_bytes!(concat!(env!("OUT_DIR"),"/flux_surface.frag.spv")))?;
    let stages = [vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert).name(c"main"),
        vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(frag).name(c"main")];
    let bindings = [vk::VertexInputBindingDescription { binding: 0,stride: size_of::<Vertex>() as u32,input_rate: vk::VertexInputRate::VERTEX }];
    let attrs = [(0,0,vk::Format::R32G32B32_SFLOAT),(1,12,vk::Format::R32G32B32_SFLOAT),(2,24,vk::Format::R32G32_SFLOAT),
        (3,32,vk::Format::R32G32B32A32_SFLOAT),(4,48,vk::Format::R32G32B32A32_SFLOAT)]
        .map(|(location,offset,format)| vk::VertexInputAttributeDescription { location,binding: 0,format,offset });
    let vi = vk::PipelineVertexInputStateCreateInfo::default().vertex_binding_descriptions(&bindings).vertex_attribute_descriptions(&attrs);
    let ia = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let vs = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
    let rs = vk::PipelineRasterizationStateCreateInfo::default().polygon_mode(vk::PolygonMode::FILL).cull_mode(vk::CullModeFlags::NONE).line_width(1.0);
    let ms = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let ds = vk::PipelineDepthStencilStateCreateInfo::default().depth_test_enable(true).depth_write_enable(true).depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
    let blends = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA);4];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);
    let dynamic = [vk::DynamicState::VIEWPORT,vk::DynamicState::SCISSOR];
    let dyns = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
    let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&FORMATS).depth_attachment_format(DEPTH);
    let info = vk::GraphicsPipelineCreateInfo::default().stages(&stages).vertex_input_state(&vi).input_assembly_state(&ia)
        .viewport_state(&vs).rasterization_state(&rs).multisample_state(&ms).depth_stencil_state(&ds)
        .color_blend_state(&blend).dynamic_state(&dyns).layout(layout).push_next(&mut rendering);
    let result = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(),&[info],None) };
    unsafe { device.destroy_shader_module(vert,None); device.destroy_shader_module(frag,None); }
    Ok(result.map_err(|(_,e)| e)?[0])
}

fn light_barrier(device: &ash::Device,cb: vk::CommandBuffer,image: vk::Image,from: vk::ImageLayout,to: vk::ImageLayout) {
    let b=vk::ImageMemoryBarrier::default().image(image).old_layout(from).new_layout(to)
        .src_access_mask(if from==vk::ImageLayout::UNDEFINED {vk::AccessFlags::empty()} else {vk::AccessFlags::MEMORY_READ|vk::AccessFlags::MEMORY_WRITE})
        .dst_access_mask(vk::AccessFlags::MEMORY_READ|vk::AccessFlags::MEMORY_WRITE)
        .subresource_range(range(vk::ImageAspectFlags::COLOR));
    unsafe { device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::ALL_COMMANDS,vk::PipelineStageFlags::ALL_COMMANDS,
        vk::DependencyFlags::empty(),&[],&[],&[b]); }
}

pub(crate) fn lighting_signature(lights: &[crate::BakeLight], emitters: &[crate::GpuGiEmitter],
    sky: [f32;3], rt: bool, bounces: u32, mixed: bool, gdf_revision: u64, emissive_boost:f32,
    area_lights:&[crate::GpuLight],area_sources:&[(u32,vk::ImageView)]) -> LightingSignature {
    let mut h=std::collections::hash_map::DefaultHasher::new();
    rt.hash(&mut h); (bounces>1).hash(&mut h); mixed.hash(&mut h); emissive_boost.to_bits().hash(&mut h);
    let transport=h.finish();
    let mut h=std::collections::hash_map::DefaultHasher::new();
    sky.map(f32::to_bits).hash(&mut h); gdf_revision.hash(&mut h);
    bytemuck::cast_slice::<_,u8>(area_lights).hash(&mut h);
    area_sources.hash(&mut h);
    for l in lights {
        (l.kind as u32).hash(&mut h); l.position.to_array().map(f32::to_bits).hash(&mut h);
        l.direction.to_array().map(f32::to_bits).hash(&mut h); l.color.map(f32::to_bits).hash(&mut h);
        [l.range,l.spot_inner_deg,l.spot_outer_deg,l.radius].map(f32::to_bits).hash(&mut h);
    }
    for e in emitters {
        e.center.map(f32::to_bits).hash(&mut h); e.radius.to_bits().hash(&mut h); e.emission.map(f32::to_bits).hash(&mut h);
    }
    LightingSignature {transport,illumination:h.finish(),hardware:rt}
}

/// Downsample by selecting the exact material receiver 4*q+2, preserving the
/// packed geometric normal. Averaging positions/normals would invent surfaces.
pub(crate) fn prepare_lighting_geometry(device:&ash::Device,cb:vk::CommandBuffer,image:vk::Image,extent:vk::Extent2D) {
    let read=vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    let src=vk::ImageLayout::TRANSFER_SRC_OPTIMAL;let dst=vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    let transition=|level,from,to| vk::ImageMemoryBarrier::default().image(image).old_layout(from).new_layout(to)
        .src_access_mask(vk::AccessFlags::MEMORY_READ|vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ|vk::AccessFlags::MEMORY_WRITE)
        .subresource_range(range(vk::ImageAspectFlags::COLOR).base_mip_level(level));
    unsafe {
        device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::ALL_COMMANDS,vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),&[],&[],&[transition(0,read,src),transition(2,read,dst)]);
        let sub=|level| vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(level).layer_count(1);
        let blit=vk::ImageBlit::default().src_subresource(sub(0)).dst_subresource(sub(2))
            .src_offsets([vk::Offset3D::default(),vk::Offset3D{x:extent.width as i32,y:extent.height as i32,z:1}])
            .dst_offsets([vk::Offset3D::default(),vk::Offset3D{x:(extent.width/4) as i32,y:(extent.height/4) as i32,z:1}]);
        device.cmd_blit_image(cb,image,src,image,dst,&[blit],vk::Filter::NEAREST);
        device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::TRANSFER,vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),&[],&[],&[transition(0,src,read),transition(2,dst,read)]);
    }
}

#[cfg(test)]
mod projection_tests {
    use super::*;
    #[test]
    fn flux_capture_transform_reuse_preserves_tangent_frames() {
        let rotation=glam::Quat::from_rotation_y(0.73);
        let capture=Mat4::from_scale_rotation_translation(Vec3::new(2.0,3.0,0.1),rotation,Vec3::new(10.0,1.0,-3.0));
        let delta=Mat4::from_scale_rotation_translation(Vec3::splat(1.4),glam::Quat::from_rotation_x(1.2),Vec3::new(4.0,2.0,1.0));
        let moved=delta*capture;
        assert!(reusable_transform(moved*capture.inverse()),"rigid motion of a stretched mesh must reuse captures");
        assert!(!reusable_transform(Mat4::from_scale(Vec3::new(1.1,1.0,1.0))));
        assert!(!reusable_transform(Mat4::from_scale(Vec3::new(-1.0,1.0,1.0))));
        let mut shear=Mat4::IDENTITY;shear.y_axis.x=0.1;
        assert!(!reusable_transform(shear));
        assert!(!reusable_transform(Mat4::ZERO));
    }
    #[test]
    fn flux_lighting_view_and_hit_priority() {
        let camera=crate::CameraData {position:Vec3::ZERO,view:Mat4::IDENTITY,
            proj:Mat4::perspective_rh(1.0,1.0,0.1,100.0)};
        let views=[LightingView::from(camera)];
        let front=[Vec3::new(-1.0,-1.0,-31.0),Vec3::new(1.0,1.0,-29.0)];
        let rear=[-front[1],-front[0]];
        let near=lighting_weight(front,&views,[None;2],10);
        let behind=lighting_weight(rear,&views,[None;2],10);
        assert!((near-2.0*behind).abs()<1e-5,"frustum demand was ignored");
        let opposite=crate::CameraData {view:Mat4::look_at_rh(Vec3::ZERO,Vec3::Z,Vec3::Y),..camera};
        let both=[LightingView::from(camera),LightingView::from(opposite)];
        assert!((lighting_weight(rear,&both,[None;2],10)-near).abs()<1e-5,"second view was ignored");
        let far=front.map(|p|p*10.0);
        assert!(lighting_weight(far,&views,[None;2],10)<near);
        assert_eq!(lighting_weight(far,&views,[None,Some(10)],10),4.0);
        assert_eq!(lighting_weight(far,&views,[Some(10),None],10),2.0);
        assert_eq!(lighting_weight(far,&views,[None,Some(0)],10),lighting_weight(far,&views,[None;2],10));
        let mut used=[None;2];
        apply_feedback(&mut used,2,1,3,10);
        assert_eq!(used,[None;2],"retired feedback leaked to a reused atlas slot");
        apply_feedback(&mut used,2,2,2,10);
        apply_feedback(&mut used,2,2,3,8);
        assert_eq!(used,[Some(8),Some(10)],"feedback kinds/timestamps were conflated");
    }
    #[test]
    fn flux_lighting_priority_cannot_starve_offscreen_pages() {
        let mut lights=[PageLighting {valid:true,..Default::default()};32];
        let budget=4;
        let max_wait=(lights.len().div_ceil(budget)*4) as u64;
        let mut worst=0;
        for serial in 1..500 {
            // New moving pages and continuously hit visible pages compete
            // with twelve very distant, unseen but still relevant receivers.
            for light in &mut lights[24..] {*light=PageLighting {last:serial-1,..Default::default()};}
            let mut order:Vec<_>=lights.iter().enumerate().map(|(i,light)|
                (std::cmp::Reverse(lighting_priority(*light,if i<12 {0.001} else {4.0},serial,max_wait,true)),i)).collect();
            order.sort_unstable();
            for (_,i) in order.into_iter().take(budget) {
                if i<12 {worst=worst.max(serial-lights[i].last);}
                lights[i].last=serial;lights[i].valid=true;lights[i].updates+=1;
            }
        }
        assert!(lights[..12].iter().all(|light|light.updates>=10),"offscreen receiver starved");
        assert!(worst<=max_wait+lights.len().div_ceil(budget) as u64,"wait bound exceeded: {worst}");
        let recent=PageLighting {valid:true,last:99,..Default::default()};
        let older=PageLighting {last:98,..recent};
        assert!(lighting_priority(older,0.01,100,32,false)>lighting_priority(recent,4.0,100,32,false),
            "software demand disrupted oldest-first cadence");
        assert!(lighting_priority(recent,4.0,100,32,false)>lighting_priority(recent,0.01,100,32,false),
            "software demand did not prioritize equally old pages");
    }
    #[test]
    fn flux_cards_thin_offset_plane_has_finite_projection() {
        let mut mesh=crate::flux_cards::layered_test_mesh(1);
        for v in &mut mesh.vertices { v.position[2]=100.0; }
        let geometry=crate::flux_cards::MeshCards::new(&mesh,false);
        for capture in geometry.captures {
            let card=super::card_for(capture,glam::Mat4::IDENTITY,0,8);
            let projection=glam::Mat4::from_cols_array_2d(&card.view_proj);
            assert!(projection.is_finite(),"thin plane generated a nonfinite capture");
            if capture.face==4 {
                let p=projection.project_point3(glam::Vec3::new(0.0,0.0,100.0));
                assert!(p.is_finite() && p.z>=-0.001 && p.z<=1.001,"plane is outside its capture: {p:?}");
            }
        }
    }
}
