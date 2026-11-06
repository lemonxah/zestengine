//! Geometry-derived local-space capture volumes. Exterior cards are supplemented
//! by interior cards whose near planes lie between overlapping surface layers.
//! Like Lumen's surfel clustering, placement depends on directional visibility,
//! rather than splitting the object's empty bounding volume into more boxes.
use glam::Vec3;
use crate::{MeshData, Vertex};

pub(crate) const AXES: [Vec3; 6] = [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z];
const GRID: usize = 16;
const MAX_LAYERS: usize = 16;
const MAX_CARDS: usize = 32;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Capture {
    pub bounds: [Vec3; 2], pub face: usize,
    front_bounds: Option<[Vec3; 2]>, back_bounds: Option<[Vec3; 2]>,
}
impl Capture {
    pub fn visible_bounds(self, double_sided: bool, cull_front: bool) -> Option<[Vec3; 2]> {
        if double_sided { union(self.front_bounds,self.back_bounds) }
        else if cull_front { self.back_bounds } else { self.front_bounds }
    }
}
fn union(a: Option<[Vec3;2]>, b: Option<[Vec3;2]>) -> Option<[Vec3;2]> {
    match (a,b) { (Some(a),Some(b))=>Some([a[0].min(b[0]),a[1].max(b[1])]), (a,None)=>a, (None,b)=>b }
}
pub(crate) struct MeshCards {
    pub captures: Vec<Capture>,
    // Only CPU-deformed meshes retain topology. Their capture volumes must be
    // rebuilt from the updated vertices, not left at the bind pose.
    topology: Option<Vec<u32>>,
}
impl MeshCards {
    pub fn new(data: &MeshData, dynamic: bool) -> Self {
        Self { captures: generate(&data.vertices, &data.indices),
            topology: dynamic.then(|| data.indices.clone()) }
    }
    pub fn update(&mut self, vertices: &[Vertex]) {
        if let Some(indices) = &self.topology { self.captures = generate(vertices, indices); }
    }
}

#[derive(Clone, Copy)]
struct Surfel { p: Vec3, lower: f32, upper: f32 }

fn generate(vertices: &[Vertex], indices: &[u32]) -> Vec<Capture> {
    let bounds = crate::flux_surface::vertex_bounds(vertices);
    let padding = (bounds[1]-bounds[0]).max(Vec3::splat(0.01))*1e-4;
    let mut cards: Vec<_> = (0..6).map(|face| Capture { bounds: [bounds[0]-padding,bounds[1]+padding], face,
        front_bounds: None, back_bounds: None }).collect();
    let size = (bounds[1]-bounds[0]).max(Vec3::splat(1e-5));
    let mut directions = Vec::new();
    for (face,n) in AXES.into_iter().enumerate() {
        let up = if n.y.abs()>0.5 { Vec3::Z } else { Vec3::Y };
        let right = up.cross(n);
        let origin = Vec3::new(bounds[0].dot(right).min(bounds[1].dot(right)),
            bounds[0].dot(up).min(bounds[1].dot(up)), 0.0);
        let step = Vec3::new(size.dot(right.abs())/GRID as f32, size.dot(up.abs())/GRID as f32, 0.0);
        let epsilon = (size.dot(n.abs())*1e-5).max(1e-6);
        let mut rays: Vec<Vec<f32>> = vec![Vec::new(); GRID*GRID];
        for tri in indices.chunks_exact(3) {
            let v = [vertices.get(tri[0] as usize),vertices.get(tri[1] as usize),vertices.get(tri[2] as usize)];
            let [Some(a),Some(b),Some(c)] = v else { continue; };
            let v = [a,b,c];
            let positions = v.map(|v|Vec3::from(v.position));
            if positions.iter().any(|p|!p.is_finite()) { continue; }
            // Conservative bounds include every potentially front-facing
            // triangle, not just the sampled surfels. Small triangles and
            // alpha holes therefore cannot disappear from the lookup BVH.
            if v.iter().any(|v|Vec3::from(v.normal).dot(n)>0.0) {
                let lo=positions[0].min(positions[1]).min(positions[2]);
                let hi=positions[0].max(positions[1]).max(positions[2]);
                cards[face].front_bounds=union(cards[face].front_bounds,Some([lo,hi]));
            }
            let normal = v.iter().map(|v| Vec3::from(v.normal)).sum::<Vec3>().normalize_or_zero();
            if normal.dot(n)<0.15 { continue; }
            let p: [Vec3; 3] = std::array::from_fn(|i| {
                let p = Vec3::from(v[i].position);
                Vec3::new((p.dot(right)-origin.x)/step.x, (p.dot(up)-origin.y)/step.y, p.dot(n))
            });
            if p.iter().any(|p| !p.is_finite()) { continue; }
            let a = p[1]-p[0]; let b = p[2]-p[0];
            let det = a.x*b.y-a.y*b.x;
            if det.abs()<1e-8 { continue; }
            let lo = p[0].min(p[1]).min(p[2]); let hi = p[0].max(p[1]).max(p[2]);
            let x0 = (lo.x.floor() as i32).clamp(0,GRID as i32-1) as usize;
            let x1 = (hi.x.ceil() as i32).clamp(0,GRID as i32-1) as usize;
            let y0 = (lo.y.floor() as i32).clamp(0,GRID as i32-1) as usize;
            let y1 = (hi.y.ceil() as i32).clamp(0,GRID as i32-1) as usize;
            for y in y0..=y1 { for x in x0..=x1 {
                let dx = x as f32+0.5-p[0].x; let dy = y as f32+0.5-p[0].y;
                let u = (dx*b.y-dy*b.x)/det; let w = (a.x*dy-a.y*dx)/det;
                if u < -1e-5 || w < -1e-5 || u+w > 1.00001 { continue; }
                let z = p[0].z+u*a.z+w*b.z;
                let ray = &mut rays[y*GRID+x];
                if ray.iter().any(|d| (d-z).abs()<epsilon*2.0) { continue; }
                ray.push(z); ray.sort_by(|a,b| b.total_cmp(a)); ray.truncate(MAX_LAYERS);
            }}
        }
        let mut surfels = Vec::new();
        for (i, ray) in rays.iter().enumerate() {
            for pair in ray.windows(2) {
                let lower = pair[1]+epsilon; let upper = pair[0]-epsilon;
                if lower>=upper { continue; }
                let p = right*(origin.x+(i%GRID) as f32*step.x+step.x*0.5)
                    +up*(origin.y+(i/GRID) as f32*step.y+step.y*0.5)+n*pair[1];
                surfels.push(Surfel { p, lower, upper });
            }
        }
        directions.push((surfels, right.abs()*step.x+up.abs()*step.y, epsilon));
    }
    for face in 0..6 { cards[face].back_bounds=cards[face^1].front_bounds; }
    // An interval sweep chooses the near plane covering the most remaining
    // surfels. A surfel is eligible only between itself and its nearer blocker.
    // This preserves actual visibility through concavities and layered meshes.
    while cards.len()<MAX_CARDS {
        let mut best = (0usize, 0usize, 0.0f32);
        for (face, (surfels, _, _)) in directions.iter().enumerate() {
            let mut events = Vec::with_capacity(surfels.len()*2);
            for s in surfels { events.push((s.lower,1i32)); events.push((s.upper,-1)); }
            events.sort_by(|a,b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let mut active = 0i32;
            for pair in events.windows(2) {
                active += pair[0].1;
                if pair[1].0>pair[0].0 && active>best.0 as i32 {
                    best = (active as usize, face, (pair[0].0+pair[1].0)*0.5);
                }
            }
        }
        if best.0<2 { break; }
        let (_, face, plane) = best;
        let (surfels, footprint, epsilon) = &mut directions[face];
        let mut lo = Vec3::splat(f32::INFINITY); let mut hi = Vec3::splat(f32::NEG_INFINITY);
        surfels.retain(|s| {
            if s.lower<plane && plane<s.upper { lo=lo.min(s.p); hi=hi.max(s.p); false } else { true }
        });
        lo = (lo-*footprint).max(bounds[0]); hi = (hi+*footprint).min(bounds[1]);
        let axis = face/2;
        if face%2==0 { hi[axis]=plane; lo[axis]-=*epsilon; }
        else { lo[axis]=-plane; hi[axis]+=*epsilon; }
        cards.push(Capture { bounds: [lo,hi], face, front_bounds: Some([lo,hi]), back_bounds: Some([lo,hi]) });
    }
    cards
}

#[cfg(test)]
pub(crate) fn layered_test_mesh(layers: usize) -> MeshData {
    let mut vertices = Vec::new(); let mut indices = Vec::new();
    for z in 0..layers {
        let base = vertices.len() as u32;
        for [x,y] in [[-1.0,-1.0],[1.0,-1.0],[1.0,1.0],[-1.0,1.0]] {
            vertices.push(Vertex { position: [x,y,z as f32], normal: [0.0,0.0,1.0], ..Default::default() });
        }
        indices.extend([base,base+1,base+2,base,base+2,base+3]);
    }
    MeshData { vertices, indices, has_lightmap_uv: false }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn flux_cards_clip_hidden_layers_without_subdividing_empty_space() {
        let plane = layered_test_mesh(1);
        assert_eq!(generate(&plane.vertices,&plane.indices).len(),6);
        let mesh = layered_test_mesh(3);
        let cards = generate(&mesh.vertices,&mesh.indices);
        assert_eq!(cards.len(),8);
        for depth in [0.0,1.0] {
            assert!(cards[6..].iter().any(|c| c.face==4 && c.bounds[0].z<depth
                && c.bounds[1].z>depth && c.bounds[1].z<depth+1.0), "hidden layer has no unoccluded capture");
        }
        // Disjoint projected geometry needs no additional near-plane captures.
        let mut separated = layered_test_mesh(2);
        for v in &mut separated.vertices[4..] { v.position[0]+=3.0; }
        assert_eq!(generate(&separated.vertices,&separated.indices).len(),6);
    }
    #[test]
    fn flux_cards_follow_cpu_deformation_and_bound_memory() {
        let mut mesh = layered_test_mesh(3);
        let mut cards = MeshCards::new(&mesh,true);
        for v in &mut mesh.vertices { v.position[2] *= 2.0; }
        cards.update(&mesh.vertices);
        assert!(cards.captures[6..].iter().any(|c| c.bounds[0].z<2.0 && c.bounds[1].z>2.0));
        let large = layered_test_mesh(100);
        assert!(generate(&large.vertices,&large.indices).len()<=MAX_CARDS);
    }
}
