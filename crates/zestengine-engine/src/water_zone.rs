//! Shared, balanced quadtree water topology. Adjacent tiles share all edge
//! vertices and displacement footprints, including coarse/fine boundaries.
use std::collections::{HashMap,BTreeSet};
use std::hash::{Hash,Hasher};
use glam::{Vec2,Vec3,Mat4};
use zestengine_core::{Water,WaterZone};
use zestengine_render::{MeshData,Vertex};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
struct Tile {x:i32,z:i32,size:i32,level:u32}
impl Tile {
    fn children(self)->[Self;4] {let h=self.size/2;std::array::from_fn(|i|Self{x:self.x+(i as i32&1)*h,z:self.z+((i as i32>>1)&1)*h,size:h,level:self.level-1})}

}
#[derive(Clone)]
pub struct ZoneTopology {pub points:Vec<(Vec2,f32)>,pub triangles:Vec<[u32;3]>,pub key:u64}

pub fn topology(zone:&WaterZone,camera:Vec3)->ZoneTopology {
    let tile_size=if zone.tile_size.is_finite(){zone.tile_size.clamp(0.25,4096.0)}else{16.0};
    let quads=zone.tile_quads.clamp(2,32).next_power_of_two() as i32;
    let extent=(Vec2::from(zone.half_extents).abs().clamp(Vec2::splat(tile_size),Vec2::splat(tile_size*512.0))/tile_size).ceil()*tile_size;
    let root_tiles=((extent.max_element()*2.0/tile_size).ceil() as u32).next_power_of_two().clamp(2,1024);
    let root_level=root_tiles.ilog2();let root_cells=root_tiles as i32*quads;
    let cell=tile_size/quads as f32;
    let mut stack=vec![Tile{x:-root_cells/2,z:-root_cells/2,size:root_cells,level:root_level}];let mut tiles=Vec::new();
    let lod_scale=if zone.lod_scale.is_finite(){zone.lod_scale.clamp(2.0,8.0)}else{2.0};
    while let Some(tile)=stack.pop(){
        let lo=Vec2::new(tile.x as f32,tile.z as f32)*cell;let hi=lo+Vec2::splat(tile.size as f32*cell);
        if lo.x>=extent.x || lo.y>=extent.y || hi.x<=-extent.x || hi.y<=-extent.y {continue;}
        let horizontal=(Vec2::new(camera.x,camera.z)-Vec2::new(camera.x,camera.z).clamp(lo,hi)).length();
        let distance=horizontal.hypot(camera.y);
        if tile.level>0 && (tile.level>zone.max_lod.min(10).max(root_level.saturating_sub(5)) || distance<tile.size as f32*cell*lod_scale || lo.x < -extent.x || lo.y < -extent.y || hi.x > extent.x || hi.y > extent.y){stack.extend(tile.children());}
        else{tiles.push(tile);}
    }
    // Enforce a 2:1 neighbour ratio. This bounds the edge stitching to one
    // midpoint per coarse cell, rather than skirts that hide cracks.
    loop {
        let mut split=BTreeSet::new();
        // Sweep matching boundary intervals instead of testing all pairs.
        // This stays O(n log n) for large zones and low authored max LODs.
        type Sides=(Vec<(i32,i32,usize)>,Vec<(i32,i32,usize)>);
        let mut vertical:HashMap<i32,Sides>=HashMap::new();let mut horizontal:HashMap<i32,Sides>=HashMap::new();
        for(i,t)in tiles.iter().enumerate(){
            vertical.entry(t.x).or_default().0.push((t.z,t.z+t.size,i));
            vertical.entry(t.x+t.size).or_default().1.push((t.z,t.z+t.size,i));
            horizontal.entry(t.z).or_default().0.push((t.x,t.x+t.size,i));
            horizontal.entry(t.z+t.size).or_default().1.push((t.x,t.x+t.size,i));
        }
        for(a,b)in vertical.values_mut().chain(horizontal.values_mut()) {
            a.sort_unstable();b.sort_unstable();let(mut i,mut j)=(0,0);
            while i<a.len() && j<b.len(){let(x,y)=(a[i],b[j]);
                if x.0<y.1 && y.0<x.1 {
                    if tiles[x.2].level>tiles[y.2].level+1 {split.insert(x.2);}
                    if tiles[y.2].level>tiles[x.2].level+1 {split.insert(y.2);}
                }
                if x.1<=y.1{i+=1;}else{j+=1;}
            }
        }
        if split.is_empty(){break;}
        let mut balanced=Vec::new();for(i,tile)in tiles.into_iter().enumerate(){if split.contains(&i){balanced.extend(tile.children());}else{balanced.push(tile);}}
        tiles=balanced;
    }
    tiles.sort_by_key(|t|(t.z,t.x,t.level));
    // Collect all tile-grid edge coordinates. Shared integer coordinates
    // remain identical when the camera changes LOD or bodies meet.
    // A shared corner inherits the coarsest footprint: its fine neighbours
    // finish morphing before that coarse tile replaces its children.
    let mut rows:HashMap<i32,BTreeSet<i32>>=HashMap::new();let mut columns:HashMap<i32,BTreeSet<i32>>=HashMap::new();
    let mut spacing:HashMap<(i32,i32),f32>=HashMap::new();
    for tile in &tiles {
        let step=tile.size/quads;
        for z in 0..=quads {for x in 0..=quads {
            let p=(tile.x+x*step,tile.z+z*step);
            rows.entry(p.1).or_default().insert(p.0);columns.entry(p.0).or_default().insert(p.1);
            spacing.entry(p).and_modify(|v|*v=v.max(step as f32*cell)).or_insert(step as f32*cell);
        }}
    }
    let mut points=Vec::new();let mut indices=HashMap::new();let mut triangles=Vec::new();
    let mut vertex=|x:i32,z:i32,foot:f32|->u32 {
        // Coordinates doubled to represent cell centres without rounding.
        *indices.entry((x,z)).or_insert_with(||{let index=points.len() as u32;
            let foot=if x%2==0 && z%2==0 {spacing.get(&(x/2,z/2)).copied().unwrap_or(foot)}else{foot};points.push((Vec2::new(x as f32,z as f32)*cell*0.5,foot));index})
    };
    for tile in &tiles {let step=tile.size/quads;
        for z in 0..quads {for x in 0..quads {
            let x0=tile.x+x*step;let z0=tile.z+z*step;let x1=x0+step;let z1=z0+step;
            let mut edge=Vec::new();
            for &x in rows[&z0].range(x0..x1){edge.push(vertex(x*2,z0*2,step as f32*cell));}
            for &z in columns[&x1].range(z0..z1){edge.push(vertex(x1*2,z*2,step as f32*cell));}
            for &x in rows[&z1].range((x0+1)..=x1).rev(){edge.push(vertex(x*2,z1*2,step as f32*cell));}
            for &z in columns[&x0].range((z0+1)..=z1).rev(){edge.push(vertex(x0*2,z*2,step as f32*cell));}
            let center=vertex(x0+x1,z0+z1,step as f32*cell);
            for i in 0..edge.len(){triangles.push([center,edge[(i+1)%edge.len()],edge[i]]);}
        }}
    }
    let mut h=std::collections::hash_map::DefaultHasher::new();for tile in tiles{(tile.x,tile.z,tile.size,tile.level).hash(&mut h);}quads.hash(&mut h);cell.to_bits().hash(&mut h);
    ZoneTopology{points,triangles,key:h.finish()}
}

/// Extend only open ocean. Every inner ring vertex is shared with the zone;
/// the extension cannot introduce a skirt, overlap, or a T junction.
fn ocean_extension(source:&ZoneTopology,distance:f32)->ZoneTopology {
    let mut result=source.clone();
    let extent=source.points.iter().map(|(p,_)|p.abs().max_element()).fold(0.0f32,f32::max);
    if !distance.is_finite() || distance<=extent || extent<=0.0{return result;}
    let mut edges:HashMap<(u32,u32),(u32,u32,u32)>=HashMap::new();
    for tri in &source.triangles {for i in 0..3 {let(a,b)=(tri[i],tri[(i+1)%3]);let e=edges.entry((a.min(b),a.max(b))).or_insert((a,b,0));e.2+=1;}}
    let mut boundary:Vec<_>=edges.values().filter(|e|e.2==1).map(|e|(e.0,e.1)).collect();boundary.sort_unstable();
    let mut ring=boundary.iter().flat_map(|&(a,b)|[a,b]).collect::<BTreeSet<_>>().into_iter().map(|i|(i,i)).collect::<HashMap<_,_>>();
    let rings=((distance/extent).log2().ceil() as u32).clamp(1,16);
    for level in 1..=rings {
        let scale=(distance/extent).powf(level as f32/rings as f32);
        let mut next=HashMap::new();
        for &(a,b) in &boundary {
            for index in [a,b] {next.entry(index).or_insert_with(||{let(p,foot)=source.points[index as usize];let id=result.points.len() as u32;
                result.points.push((p*scale,(foot*scale).max(extent*(scale-1.0)/rings as f32)));id});}
            // Reverse the original boundary edge to preserve upward winding.
            result.triangles.push([ring[&b],ring[&a],next[&a]]);
            result.triangles.push([ring[&b],next[&a],next[&b]]);
        }
        ring=next;
    }
    result
}

/// Project shared zone vertices into a body's local surface. Conservative
/// triangle coverage retains thin channels; raster/RT apply the exact mask.
pub fn surface(topology:&ZoneTopology,zone_world:Mat4,water_world:Mat4,water:&Water,far_distance:f32)->MeshData {
    let near_points=topology.points.len();
    let extended=(water.body==zestengine_core::WaterBody::Ocean && far_distance>0.0).then(||ocean_extension(topology,far_distance));
    let topology=extended.as_ref().unwrap_or(topology);
    let to_local=water_world.inverse()*zone_world;
    let mut vertices=Vec::new();let mut indices=Vec::new();let mut remap=HashMap::new();
    let local_scale=to_local.transform_vector3(Vec3::X).length().max(to_local.transform_vector3(Vec3::Z).length());
    for tri in &topology.triangles {
        let p=tri.map(|i|{let p=topology.points[i as usize].0;to_local.transform_point3(Vec3::new(p.x,0.0,p.y))});
        let center=(p[0]+p[1]+p[2])/3.0;
        let radius=p.iter().map(|p|Vec2::new(p.x-center.x,p.z-center.z).length()).fold(0.0f32,f32::max);
        if water.shore_sample(Vec2::new(center.x,center.z)).is_none_or(|(_,bank)|bank < -radius){continue;}
        for &index in tri {
            let next=*remap.entry(index).or_insert_with(||{
                let(p,foot)=topology.points[index as usize];let p=to_local.transform_point3(Vec3::new(p.x,0.0,p.y));
                let(column,_)=water.shore_sample(Vec2::new(p.x,p.z)).unwrap();let id=vertices.len() as u32;
                vertices.push(Vertex {position:[p.x,column.surface,p.z],normal:[0.0,1.0,0.0],tangent:[1.0,0.0,0.0,1.0],uv:[p.x,p.z],
                    uv1:[foot*local_scale,if (index as usize)<near_points {foot}else{0.0}],color:[column.flow.x/10.0+0.5,column.flow.z/10.0+0.5,column.edge.clamp(0.0,1.0),1.0],..Default::default()});id
            });indices.push(next);
        }
    }
    MeshData{vertices,indices,has_lightmap_uv:false}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adaptive_zone_has_shared_edges_and_refines_near_the_camera(){
        let zone=WaterZone {half_extents:[128.0;2],tile_size:8.0,tile_quads:4,lod_scale:1.0,..Default::default()};
        let near=topology(&zone,Vec3::new(0.0,2.0,0.0));let far=topology(&zone,Vec3::new(0.0,500.0,0.0));
        assert!(near.triangles.len()>far.triangles.len()*2);
        let mut edges:HashMap<(u32,u32),usize>=HashMap::new();
        for tri in &near.triangles {for i in 0..3 {let(a,b)=(tri[i],tri[(i+1)%3]);*edges.entry((a.min(b),a.max(b))).or_default()+=1;}}
        for((a,b),uses)in edges {assert!(uses<=2);if uses==1{let a=near.points[a as usize].0;let b=near.points[b as usize].0;
            assert!((a.x.abs()==128.0 && b.x==a.x)||(a.y.abs()==128.0 && b.y==a.y),"interior crack {a:?} -> {b:?}");}}
        let water=Water {size:[12.0;2],waves:Vec::new(),..Default::default()};
        let mesh=surface(&near,Mat4::IDENTITY,Mat4::IDENTITY,&water,0.0);assert!(!mesh.indices.is_empty());
        assert!(mesh.vertices.len()<near.points.len()/4);
    }

    #[test]
    fn far_ocean_is_closed_at_zone_seams_and_reaches_authored_extent(){
        let zone=WaterZone{half_extents:[64.0,32.0],tile_size:8.0,tile_quads:4,..Default::default()};
        let near=topology(&zone,Vec3::new(3.0,2.0,7.0));let far=ocean_extension(&near,20000.0);
        let mut edges:HashMap<(u32,u32),usize>=HashMap::new();
        for tri in &far.triangles {
            let p=tri.map(|i|far.points[i as usize].0);
            assert!((p[1]-p[0]).perp_dot(p[2]-p[0])<0.0,"upward nondegenerate triangles");
            for i in 0..3 {let(a,b)=(tri[i],tri[(i+1)%3]);*edges.entry((a.min(b),a.max(b))).or_default()+=1;}
        }
        for((a,b),uses)in edges {assert!(uses<=2);if uses==1 {let a=far.points[a as usize].0;let b=far.points[b as usize].0;
            assert!((a.x.abs()-20000.0).abs()<0.01 && a.x==b.x || (a.y.abs()-10000.0).abs()<0.01 && a.y==b.y,"open seam {a:?} {b:?}");}}
        let ocean=Water{body:zestengine_core::WaterBody::Ocean,..Default::default()};
        let mesh=surface(&near,Mat4::IDENTITY,Mat4::IDENTITY,&ocean,20000.0);
        assert_eq!(mesh.indices.len(),far.triangles.len()*3);
        let lake=Water{size:[128.0,64.0],..Default::default()};
        let lake=surface(&near,Mat4::IDENTITY,Mat4::IDENTITY,&lake,20000.0);
        assert!(lake.vertices.iter().all(|p|p.position[0].abs()<=64.0 && p.position[2].abs()<=32.0));
    }
    #[test]
    fn zone_lod_changes_converge_to_the_same_surface_before_replacement(){
        for scale in [2.0,4.0,8.0]{
        let zone=WaterZone{half_extents:[32.0;2],tile_size:4.0,tile_quads:4,lod_scale:scale,..Default::default()};
        let height=|p:Vec2,foot:f32|((p.x*0.73).sin()*0.4+(p.y*0.51).cos()*0.2)/(1.0+foot*0.2);
        let vertex_height=|p:Vec2,foot:f32,camera:Vec3|{
            let parent=foot*2.0;let tile=parent*4.0;let tile_at=((p+Vec2::splat(tile*1e-5))/tile).floor()*tile;
            let mut distance=0.0f32;
            for z in 0..2{for x in 0..2{let lo=tile_at-Vec2::new(x as f32,z as f32)*tile;
                if p.cmpge(lo-Vec2::splat(tile*2e-5)).all()&&p.cmple(lo+Vec2::splat(tile*(1.0+2e-5))).all(){let delta=Vec2::new(camera.x,camera.z)-Vec2::new(camera.x,camera.z).clamp(lo,lo+Vec2::splat(tile));distance=distance.max(delta.length().hypot(camera.y));}
            }}
            let t=((distance-tile*scale*0.9)/(tile*scale-0.88-tile*scale*0.9)).clamp(0.0,1.0);let weight=t*t*(3.0-2.0*t);
            let lo=(p/parent).floor()*parent;let center=lo+Vec2::splat(parent*0.5);let corners=[lo,lo+Vec2::new(parent,0.0),lo+Vec2::splat(parent),lo+Vec2::new(0.0,parent)];
            let coarse=(0..4).find_map(|i|{let a=corners[i]-center;let b=corners[(i+1)%4]-center;let q=p-center;let u=q.perp_dot(b)/a.perp_dot(b);let v=a.perp_dot(q)/a.perp_dot(b);
                (u>=-1e-5&&v>=-1e-5&&u+v<=1.00001).then(||height(center,parent)*(1.0-u-v)+height(center+a,parent)*u+height(center+b,parent)*v)}).unwrap();
            height(p,foot)+(coarse-height(p,foot))*weight
        };
        let samples:Vec<_>=[-29.3,-22.7,-14.3,-5.7,0.3,7.7,16.3,23.7,29.3].into_iter().flat_map(|x|[-29.3,-22.7,-14.3,-5.7,0.3,7.7,16.3,23.7,29.3].map(|z|Vec2::new(x,z))).collect();
        let render=|mesh:&ZoneTopology,camera:Vec3|{
            let heights:Vec<_>=mesh.points.iter().map(|(p,h)|vertex_height(*p,*h,camera)).collect();
            samples.iter().map(|p|mesh.triangles.iter().find_map(|tri|{let [a,b,c]=tri.map(|i|mesh.points[i as usize].0);let ab=b-a;let ac=c-a;let q=*p-a;
                let u=q.perp_dot(ac)/ab.perp_dot(ac);let v=ab.perp_dot(q)/ab.perp_dot(ac);
                (u>=-1e-5&&v>=-1e-5&&u+v<=1.00001).then(||heights[tri[0]as usize]*(1.0-u-v)+heights[tri[1]as usize]*u+heights[tri[2]as usize]*v)
            }).unwrap()).collect::<Vec<_>>()
        };
        let mut changes=0;let mut max_jump=0.0f32;
        for y in [0.0,6.0*scale,14.0*scale]{for x in -8..8{for diagonal in [false,true]{
            let before=Vec3::new(x as f32+0.5-1e-4,y,if diagonal{x as f32+0.5-1e-4}else{0.0});let after=before+Vec3::new(2e-4,0.0,if diagonal{2e-4}else{0.0});
            let a=topology(&zone,before.round());let b=topology(&zone,after.round());if a.key==b.key{continue;}changes+=1;
            for(a,b)in render(&a,before).into_iter().zip(render(&b,after)){max_jump=max_jump.max((a-b).abs());}
        }}}
        eprintln!("[water LOD motion] scale {scale}: {changes} tile changes; maximum jump {max_jump:.7} m");
        assert!(changes>0);assert!(max_jump<0.001,"zone topology caused a {max_jump} m jump across {changes} camera boundaries at scale {scale}");
        }
    }

}
