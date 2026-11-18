//! History-free recovery of existing intersections. Cutting intersected edges
//! exposes connected cloth regions; all contacts between a pair of regions
//! agree on which side should move. This prevents independent endpoint choices
//! from repeatedly undoing one another along a tessellated intersection.
use super::*;
use std::collections::BTreeMap;

struct Crossing {edge:usize,face:usize,normal:Vec3,d:[f32;2],bary:[f32;3],gap:f32}
fn face(top:&ClothContactTopology,index:usize)->[usize;3]{let mut t=top.triangles[index];t.sort_unstable();t}
struct Regions {parent:Vec<usize>}
impl Regions {
    fn new(n:usize)->Self{Self{parent:(0..n).collect()}}
    fn root(&mut self,mut i:usize)->usize{
        while self.parent[i]!=i{self.parent[i]=self.parent[self.parent[i]];i=self.parent[i];}i
    }
    fn join(&mut self,a:usize,b:usize){let a=self.root(a);let b=self.root(b);self.parent[a.max(b)]=a.min(b);}
}

// Choose the same symbolic side for exactly-on-surface vertices even when
// imported triangles have inconsistent winding. Nonzero crossings are unaffected.
fn normal(t:[glam::DVec3;3])->glam::DVec3 {
    let n=(t[1]-t[0]).cross(t[2]-t[0]).normalize_or_zero();
    let a=n.abs();let axis=if a.x>=a.y&&a.x>=a.z{0}else if a.y>=a.z{1}else{2};
    if n[axis]<0.0{-n}else{n}
}
fn crossing(e:[Vec3;2],t:[Vec3;3])->Option<(Vec3,[f32;2],[f32;3],f64)>{
    // Thin triangles are common while an intersection is being removed.
    // Double precision and oriented-area barycentrics avoid cancellation in
    // the Gram determinant used by a generic closest-point query.
    let e=e.map(|v|v.as_dvec3());let t=t.map(|v|v.as_dvec3());
    let n=normal(t);if n==glam::DVec3::ZERO||!n.is_finite(){return None;}
    let extent=(t[1]-t[0]).length().max((t[2]-t[0]).length()).max((e[1]-e[0]).length());
    let coordinate=e.iter().chain(t.iter()).map(|p|p.abs().max_element()).fold(0.0,f64::max);
    let eps=extent*1e-6+coordinate*f32::EPSILON as f64*4.0;
    let d=e.map(|p|{let d=(p-t[0]).dot(n);if d.abs()<=eps{0.0}else{d}});
    if (d[0]>=0.0)==(d[1]>=0.0){return None;}
    let at=e[0].lerp(e[1],d[0]/(d[0]-d[1]));let cross=(t[1]-t[0]).cross(t[2]-t[0]);let area=cross.length_squared();
    if area<1e-30{return None;}
    let a=(t[1]-at).cross(t[2]-at).dot(cross)/area;let b=(t[2]-at).cross(t[0]-at).dot(cross)/area;let c=1.0-a-b;
    if [a,b,c].iter().any(|v|*v< -1e-6||*v>1.0+1e-6){return None;}
    let bary=[a.max(0.0),b.max(0.0),c.max(0.0)];let sum=bary.iter().sum::<f64>();
    Some((n.as_vec3(),d.map(|d|d as f32),bary.map(|v|(v/sum)as f32),eps))
}
fn separation_tangent(surface:Vec3,normal:Vec3)->Vec3{
    let tangent=surface-normal*surface.dot(normal);
    if tangent.length_squared()>1e-6{tangent.normalize()}else{Vec3::ZERO}
}

pub(super) fn recover(top:&ClothContactTopology,p:&mut[Vec3],weights:&[f32],thickness:f32,owners:Option<&[usize]>,radii:Option<&[f32]>)->usize {
    let mut resolved=recover_pass(top,p,weights,thickness,owners,radii,false);
    if resolved>0&&weights.iter().any(|w|*w<=0.0){
        // A self-intersection repair must not leave cloth inside a fixed
        // garment that was resolved earlier in the same pass.
        for _ in 0..2{let n=recover_pass(top,p,weights,thickness,owners,radii,true);resolved+=n;if n==0{break;}}
    }
    if resolved>0{project_fixed(top,p,weights,thickness,owners,radii);}
    resolved
}
fn recover_pass(top:&ClothContactTopology,p:&mut[Vec3],weights:&[f32],thickness:f32,owners:Option<&[usize]>,radii:Option<&[f32]>,fixed_only:bool)->usize {
    if p.len()!=weights.len()||!thickness.is_finite()||thickness<=0.0||p.iter().any(|p|!p.is_finite())
        ||weights.iter().any(|w|!w.is_finite())||top.triangles.iter().flatten().chain(top.edges.iter().flatten()).any(|&i|i>=p.len()){return 0;}
    let bounds:Vec<_>=top.triangles.iter().map(|t|Bounds::points(&t.map(|i|p[i])).expanded(thickness)).collect();
    let tree=Tree::new(&bounds);let mut candidates=Vec::new();let mut crossings=Vec::new();let mut cut=vec![false;top.edges.len()];
    let mut neighbours=vec![Vec::new();p.len()];for e in &top.edges{neighbours[e[0]].push(e[1]);neighbours[e[1]].push(e[0]);}
    let mut visited=vec![0usize;p.len()];let mut stamp=0;let mut stack=Vec::new();
    for (ei,e) in top.edges.iter().enumerate(){
        candidates.clear();tree.query(Bounds::points(&e.map(|i|p[i])).expanded(thickness),&mut candidates);
        for &ti in &candidates {
            let t=face(top,ti);
            if fixed_only&&t.iter().any(|&i|weights[i]>0.0){continue;}
            // Proximity repulsion skips the one-ring to avoid swelling a rest
            // mesh. True intersections in that ring still require recovery.
            if owners.is_some_and(|o|o[e[0]]==o[t[0]])||e.iter().any(|&i|t.iter().any(|&j|i==j||top.excluded.contains(&(i.min(j),i.max(j))))){continue;}
            let Some((normal,d,bary,epsilon))=crossing(e.map(|i|p[i]),t.map(|i|p[i]))else{continue};
            // An endpoint merely touching a face is not an intersection to
            // reverse. A contour through an exact mesh vertex is real only
            // when its incident cloth continues onto the other side too.
            let mut touches=false;
            for k in 0..2{if d[k]==0.0{
                stamp+=1;stack.clear();stack.push(e[k]);visited[e[k]]=stamp;let mut opposite=false;
                while let Some(i)=stack.pop(){
                    for &j in &neighbours[i]{
                        if visited[j]==stamp{continue;}visited[j]=stamp;
                        let distance=(p[j].as_dvec3()-p[t[0]].as_dvec3()).dot(normal.as_dvec3());
                        if distance>epsilon{opposite=true;break;}
                        if distance>=-epsilon{stack.push(j);}
                    }
                    if opposite{break;}
                }
                touches|=!opposite;
            }}
            if touches{continue;}
            let gap=radii.map_or(thickness,|r|r[e[0]].max(r[e[1]])+t.iter().map(|&j|r[j]).fold(0.0,f32::max));
            cut[ei]=true;crossings.push(Crossing{edge:ei,face:ti,normal,d,bary,gap});
        }
    }
    if crossings.is_empty(){return 0;}
    let mut regions=Regions::new(p.len());
    for (ei,e) in top.edges.iter().enumerate(){if !cut[ei]{regions.join(e[0],e[1]);}}
    let region:Vec<_>=(0..p.len()).map(|i|regions.root(i)).collect();
    let mut mass=vec![0.0f64;p.len()];
    for (i,&w) in weights.iter().enumerate(){mass[region[i]]+=if w>0.0{1.0/w as f64}else{f64::INFINITY};}
    // Pick the smaller movable region, not the currently cheapest endpoint.
    // Boundary depth would prefer a freshly repaired row over the next trapped
    // row and immediately reverse the recovery on the following pass.
    let mut choices:BTreeMap<(usize,usize),usize>=BTreeMap::new();
    for c in &crossings {
        let r=top.edges[c.edge].map(|i|region[i]);if r[0]==r[1]{continue;}
        let pair=(r[0].min(r[1]),r[0].max(r[1]));
        if mass[pair.0].is_finite()||mass[pair.1].is_finite(){choices.insert(pair,if mass[pair.0]<=mass[pair.1]{pair.0}else{pair.1});}
    }
    let mut handled=vec![false;p.len()];for &r in choices.values(){handled[r]=true;}
    let original=p.to_vec();
    let mut surface=vec![Vec3::ZERO;p.len()];
    for i in 0..top.triangles.len(){let t=face(top,i);
        let n=normal(t.map(|i|original[i].as_dvec3())).as_vec3();
        for i in t{let oriented=if surface[i].dot(n)<0.0{-n}else{n};surface[i]+=oriented;}
    }
    for n in &mut surface{*n=n.normalize_or_zero();}
    let mut resolved=0;let mut members=vec![Vec::new();p.len()];let mut constraints=vec![Vec::new();p.len()];
    for i in 0..p.len(){if handled[region[i]]{members[region[i]].push(i);}}
    for (ci,c) in crossings.iter().enumerate() {
        let e=top.edges[c.edge];let r=e.map(|i|region[i]);let pair=(r[0].min(r[1]),r[0].max(r[1]));
        let Some(&chosen)=choices.get(&pair)else{continue};
        let k=usize::from(r[1]==chosen);let side=if c.d[1-k]>=0.0{1.0}else{-1.0};
        constraints[chosen].push((ci,c.normal*side));
    }
    // Move a trapped region coherently. Repairing one boundary row at a time
    // collapses successive rows onto one another and creates new self-crossings.
    let mut order:Vec<_>=(0..p.len()).filter(|&r|handled[r]).collect();
    order.sort_by(|&a,&b|mass[b].total_cmp(&mass[a]).then(b.cmp(&a)));
    let mut offsets=vec![Vec3::ZERO;p.len()];
    for r in order{
        let mut planes=Vec::new();let mut tangent=Vec3::ZERO;let mut gap=0.0f32;
        let source=members[r].iter().map(|&i|surface[i]).sum::<Vec3>().normalize_or_zero();
        for &(ci,n) in &constraints[r]{
            let c=&crossings[ci];let t=face(top,c.face);
            let own=(0..3).filter(|&j|region[t[j]]==r).map(|j|c.bary[j]).sum::<f32>();
            if own>0.9999{continue;}
            let target=(0..3).map(|j|(original[t[j]]+offsets[region[t[j]]])*c.bary[j]).sum::<Vec3>();
            let depth=members[r].iter().map(|&i|(original[i]-target).dot(n)).fold(f32::INFINITY,f32::min);
            planes.push((n,(c.gap-depth)/(1.0-own)));tangent+=separation_tangent(source,n);gap=gap.max(c.gap);
        }
        let mut offset=tangent.normalize_or_zero()*gap;
        for _ in 0..8{for &(n,depth) in &planes{let missing=depth-n.dot(offset);if missing>0.0{offset+=n*missing;}}}
        let points:Vec<_>=members[r].iter().map(|&i|original[i]).collect();let bound=Bounds::points(&points);
        offset=offset.clamp_length_max((bound.hi-bound.lo).length()*2.0+gap*4.0);
        if offset.is_finite()&&offset.length_squared()>1e-20{
            offsets[r]=offset;for &i in &members[r]{p[i]+=offset;resolved+=1;}
        }else{handled[r]=false;}
    }
    // A contour ending inside a sheet does not partition the mesh. Extend the
    // local face plane through its connected source patch for a consistent
    // side decision. Excluding the opposing triangle keeps its own vertices
    // from becoming a bridge between those source regions.
    // Process one target face at a time: storing a full partition for every
    // intersected face would require quadratic memory on dense tangled cloth.
    let mut fallback:Vec<_>=crossings.iter().filter(|c|{
        let e=top.edges[c.edge];let t=face(top,c.face);
        !e.iter().chain(t.iter()).any(|&i|handled[region[i]])
    }).collect();
    fallback.sort_by_key(|c|c.face);
    let mut extended=None;
    for c in fallback {
        let e=top.edges[c.edge];let t=face(top,c.face);
        if extended.as_ref().is_none_or(|(id,_,_,_)|*id!=c.face){
            let side:Vec<_>=original.iter().map(|p|(*p-original[t[0]]).dot(c.normal)>=0.0).collect();
            let mut parts=Regions::new(p.len());
            for edge in &top.edges{if !edge.iter().any(|i|t.contains(i))&&side[edge[0]]==side[edge[1]]{parts.join(edge[0],edge[1]);}}
            let labels:Vec<_>=(0..p.len()).map(|i|parts.root(i)).collect();let mut masses=vec![0.0;p.len()];let mut pinned=vec![false;p.len()];
            for i in 0..p.len(){if weights[i]<=0.0{pinned[labels[i]]=true;}else{masses[labels[i]]+=1.0/weights[i]as f64;}}
            extended=Some((c.face,labels,masses,pinned));
        }
        let (_,labels,masses,pinned)=extended.as_ref().unwrap();
        let r=e.map(|i|labels[i]);
        let cost=|k:usize|if weights[e[k]]<=0.0||(pinned[r[k]]&&!pinned[r[1-k]]){f64::INFINITY}else{masses[r[k]]};
        let k=usize::from(cost(1)<cost(0));if !cost(k).is_finite(){continue;}
        let side=if c.d[1-k]>=0.0{1.0}else{-1.0};let n=c.normal*side;
        let target=(0..3).map(|j|original[t[j]]*c.bary[j]).sum::<Vec3>();let gap=(p[e[k]]-target).dot(n);
        if gap<c.gap{
            let tangent=separation_tangent(surface[e[k]],n);
            p[e[k]]+=n*(c.gap-gap)+tangent*c.gap;resolved+=1;
        }
    }
    if resolved>0{
        // Remove net translation from each entirely free interacting group.
        // A uniform shift preserves the repaired relative geometry and the
        // mass-weighted centre. Anchored groups transmit repair to their pins;
        // those pins, including unrelated fixed objects, must never be moved.
        let mut groups=Regions::new(p.len());
        for e in &top.edges{groups.join(e[0],e[1]);}
        for c in &crossings{groups.join(top.edges[c.edge][0],top.triangles[c.face][0]);}
        let mut total=vec![0.0f64;p.len()];let mut delta=vec![glam::DVec3::ZERO;p.len()];
        for i in 0..p.len(){let root=groups.root(i);if weights[i]<=0.0{total[root]=f64::INFINITY;}else{
            let m=1.0/weights[i]as f64;total[root]+=m;delta[root]+=(p[i]-original[i]).as_dvec3()*m;
        }}
        for (i, point) in p.iter_mut().enumerate(){let root=groups.root(i);if total[root].is_finite()&&total[root]>0.0{*point-=(delta[root]/total[root]).as_vec3();}}
    }
    resolved
}

/// True geometric crossings, including nearby topology that proximity
/// repulsion deliberately excludes. Coplanar contact is not a crossing.
fn intersects(top:&ClothContactTopology,p:&[Vec3],owners:Option<&[usize]>)->bool{
    let bounds:Vec<_>=top.triangles.iter().map(|t|Bounds::points(&t.map(|i|p[i]))).collect();
    let tree=Tree::new(&bounds);let mut candidates=Vec::new();
    for e in &top.edges{
        candidates.clear();tree.query(Bounds::points(&e.map(|i|p[i])),&mut candidates);
        for &ti in &candidates{
            let t=face(top,ti);
            if owners.is_some_and(|o|o[e[0]]==o[t[0]])||e.iter().any(|&i|t.iter().any(|&j|i==j||top.excluded.contains(&(i.min(j),i.max(j))))){continue;}
            if let Some((_,d,_,_))=crossing(e.map(|i|p[i]),t.map(|i|p[i]))&& d[0]!=0.0&&d[1]!=0.0{return true;}
        }
    }
    false
}

/// Constraint projection can introduce a new crossing even after swept
/// contacts were solved. Backtrack such a correction toward the previous
/// valid pose. Pins retain their current animation targets throughout. If
/// those targets already make the old pose intersect, geometric recovery
/// remains responsible; we must not restore an invalid previous pose.
pub(super) fn preserve_valid_pose(top:&ClothContactTopology,p:&mut[Vec3],previous:&[Vec3],weights:&[f32],owners:Option<&[usize]>)->bool{
    if !intersects(top,p,owners){return false;}
    let origin:Vec<_>=(0..p.len()).map(|i|if weights[i]>0.0{previous[i]}else{p[i]}).collect();
    if intersects(top,&origin,owners){return false;}
    let proposed=p.to_vec();let mut fraction=0.5;
    for _ in 0..12{
        for i in 0..p.len(){p[i]=origin[i].lerp(proposed[i],fraction);}
        if !intersects(top,p,owners){return true;}
        fraction*=0.5;
    }
    p.copy_from_slice(&origin);true
}

/// Fixed faces take priority over corrections between movable cloth layers.
/// Their final thickness pass cannot displace another part of the fixed mesh.
pub(super) fn project_fixed(top:&ClothContactTopology,p:&mut[Vec3],weights:&[f32],thickness:f32,owners:Option<&[usize]>,radii:Option<&[f32]>){
    let fixed:Vec<_>=(0..top.triangles.len()).map(|i|face(top,i)).filter(|t|t.iter().all(|&i|weights[i]<=0.0)).collect();
    if fixed.is_empty(){return;}
    let bounds:Vec<_>=fixed.iter().map(|t|Bounds::points(&t.map(|i|p[i]))).collect();let tree=Tree::new(&bounds);
    let mut neighbours=vec![Vec::new();p.len()];for e in &top.edges{neighbours[e[0]].push(e[1]);neighbours[e[1]].push(e[0]);}
    let mut candidates=Vec::new();let mut visited=vec![0usize;p.len()];let mut stamp=0;let mut stack=Vec::new();
    for i in 0..p.len(){if weights[i]<=0.0{continue;}
        candidates.clear();tree.query(Bounds::points(&[p[i]]).expanded(thickness),&mut candidates);
        for &face in &candidates{
            let t=fixed[face];if owners.is_some_and(|o|o[i]==o[t[0]])||t.iter().any(|&j|top.neighbours(i,j)){continue;}
            let gap=radii.map_or(thickness,|r|r[i]+t.iter().map(|&j|r[j]).fold(0.0,f32::max));
            let(q,_)=closest_triangle(p[i],t.map(|i|p[i]));let delta=p[i]-q;let distance=delta.length();
            if distance>=gap{continue;}
            let n=if distance>1e-6{delta/distance}else{
                let n=normal(t.map(|j|p[j].as_dvec3())).as_vec3();
                stamp+=1;stack.clear();stack.push(i);visited[i]=stamp;let mut side=1.0;
                'search:while let Some(j)=stack.pop(){for &k in &neighbours[j]{
                    if visited[k]==stamp{continue;}visited[k]=stamp;
                    let d=(p[k]-q).dot(n);if d.abs()>1e-6{side=d.signum();break 'search;}stack.push(k);
                }}
                n*side
            };
            p[i]=q+n*gap;
        }
    }
}

#[cfg(test)]
#[path="cloth_untangle_tests.rs"]
mod tests;
