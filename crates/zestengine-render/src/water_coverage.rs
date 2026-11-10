//! Renderer packing of the authoritative water outline/river samples. A
//! stackless BVH keeps queries local without rasterizing narrow channels into
//! a fixed-resolution mask. No Vulkan types: also used by render-lite.
use glam::Vec2;
use zestengine_core::{Water, WaterBody};
use crate::GpuWater;

#[derive(Clone, Debug, Default)]
pub struct WaterCoverage {
    /// 1 rectangle/ocean, 2 polygon, 3 river. Zero is legacy standalone data.
    pub kind: u32,
    /// Preorder nodes: bounds (min XZ, max XZ), then escape offset / leaf
    /// flag / original segment index / reserved (integer lanes are bitcast).
    /// Polygon leaves append one endpoint pair. River leaves append two
    /// samples, each position+width, direction+speed, depth+padding.
    pub data: Vec<[f32; 4]>,
}

struct Leaf { bounds: [f32; 4], index: u32, data: Vec<[f32; 4]> }

impl WaterCoverage {
    /// The exact sampled geometry used by `Water::column`, without a point
    /// count cap or a texture-resolution-dependent loss of coverage.
    pub fn new(water: &Water) -> Self {
        if water.body==WaterBody::Custom {
            let mut data=Vec::new();if let Some(mesh)=&water.custom_mesh {
                let mut offsets=Vec::with_capacity(mesh.nodes.len()+1);let mut offset=0;for n in &mesh.nodes{offsets.push(offset);offset+=if n.triangle.is_some(){5}else{2};}offsets.push(offset);
                for n in &mesh.nodes{data.push(n.bounds);data.push([f32::from_bits(offsets[n.escape]),f32::from_bits(u32::from(n.triangle.is_some())),f32::from_bits(n.triangle.unwrap_or(0)as u32),0.0]);
                    if let Some(i)=n.triangle{for p in mesh.triangles[i]{data.push([p.x,p.y,p.z,0.0]);}}
                }
            }return Self{kind:4,data};
        }
        let mut leaves = Vec::new();
        let kind = match water.body {
            WaterBody::River => {
                let path = water.river_path();
                for (index, pair) in path.windows(2).enumerate() {
                    let (a, b) = (pair[0], pair[1]);
                    let r = a.width.max(b.width).max(0.01) * 0.5;
                    let lo = Vec2::new(a.pos.x.min(b.pos.x), a.pos.z.min(b.pos.z)) - Vec2::splat(r);
                    let hi = Vec2::new(a.pos.x.max(b.pos.x), a.pos.z.max(b.pos.z)) + Vec2::splat(r);
                    let data = [a, b].into_iter().flat_map(|p| [
                        [p.pos.x, p.pos.y, p.pos.z, p.width],
                        [p.dir.x, p.dir.y, p.dir.z, p.speed],
                        [p.depth, p.flow_offset.x, p.flow_offset.z, p.dist],
                    ]).collect();
                    leaves.push(Leaf { bounds: [lo.x, lo.y, hi.x, hi.y], index: index as u32, data });
                }
                3
            }
            WaterBody::Lake | WaterBody::Pool | WaterBody::Ocean if water.shape.len() >= 3 => {
                let outline = water.outline();
                for (index, a) in outline.iter().enumerate() {
                    let b = outline[(index + outline.len() - 1) % outline.len()];
                    leaves.push(Leaf { bounds: [a[0].min(b[0]), a[1].min(b[1]), a[0].max(b[0]), a[1].max(b[1])],
                        index: index as u32, data: vec![[a[0], a[1], b[0], b[1]]] });
                }
                if water.body==WaterBody::Ocean{5}else{2}
            }
            _ => 1,
        };
        let mut result = Self { kind, data: Vec::new() };
        if !leaves.is_empty() { build(&mut leaves, &mut result.data); }
        result
    }

    /// Attach this coverage to a body and append its nodes to the frame's
    /// shared storage. Offsets are vec4 indices, relative to this body's root.
    pub fn append(&self, water: &Water, body: &mut GpuWater, data: &mut Vec<[f32; 4]>) {
        body.coverage = [self.kind, data.len() as u32, self.data.len() as u32, water.depth.max(0.0).to_bits()];
        data.extend_from_slice(&self.data);
        body.exclusions = [data.len() as u32, water.exclusions.len() as u32, 0, 0];
        for region in &water.exclusions { data.extend_from_slice(&region.to_unit.to_cols_array_2d()); }
        body.shore_field=[0;4];
        if let Some(shore)=water.shore.as_ref().filter(|s|water.shore_depth>0.0&&s.n>0&&s.half>0.0&&s.depth.len()==s.n*s.n){
            body.shore_field=[data.len()as u32,shore.n as u32,shore.half.to_bits(),water.shore_depth.to_bits()];
            for values in shore.depth.chunks(4){let mut packed=[0.0;4];packed[..values.len()].copy_from_slice(values);data.push(packed);}
        }
    }
}

fn build(leaves: &mut [Leaf], out: &mut Vec<[f32; 4]>) {
    let start = out.len();
    let mut lo = Vec2::splat(f32::INFINITY);
    let mut hi = Vec2::splat(f32::NEG_INFINITY);
    for leaf in leaves.iter() {
        lo = lo.min(Vec2::new(leaf.bounds[0], leaf.bounds[1]));
        hi = hi.max(Vec2::new(leaf.bounds[2], leaf.bounds[3]));
    }
    out.push([lo.x, lo.y, hi.x, hi.y]);
    out.push([0.0; 4]);
    if leaves.len() == 1 {
        out.extend_from_slice(&leaves[0].data);
        out[start + 1][1] = f32::from_bits(1);
        out[start + 1][2] = f32::from_bits(leaves[0].index);
    } else {
        let axis = usize::from(hi.y - lo.y > hi.x - lo.x);
        let mid = leaves.len() / 2;
        leaves.select_nth_unstable_by(mid, |a, b| {
            (a.bounds[axis] + a.bounds[axis + 2]).total_cmp(&(b.bounds[axis] + b.bounds[axis + 2]))
                .then(a.index.cmp(&b.index))
        });
        let (left, right) = leaves.split_at_mut(mid);
        build(left, out);
        build(right, out);
    }
    out[start + 1][0] = f32::from_bits(out.len() as u32);
}

/// Changes only when the authored footprint or sampled river attributes
/// change, not when waves, the camera, transforms or simulation move.
pub fn water_coverage_key(water: &Water) -> u64 {
    water.geometry_key()
}
