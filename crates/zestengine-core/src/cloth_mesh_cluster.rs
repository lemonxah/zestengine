//! Resolution reduction respects connectivity inside each spatial cell. Nearby
//! layers of a fold must not become one rigid driver merely because the full
//! garment is connected elsewhere.
use super::*;
struct Regions{parent:Vec<usize>}
impl Regions{
    fn root(&mut self,mut i:usize)->usize{while self.parent[i]!=i{self.parent[i]=self.parent[self.parent[i]];i=self.parent[i];}i}
    fn join(&mut self,a:usize,b:usize){let a=self.root(a);let b=self.root(b);self.parent[a.max(b)]=a.min(b);}
}
pub(super) fn build(positions:&[Vec3],indices:&[u32],pins:&[f32],spacing:f32)->(Vec<u32>,Vec<Vec<u32>>){
    let cell=if spacing>0.0{spacing}else{1e-4};
    let key=|p:Vec3,cell:f32|((p.x/cell).round()as i64,(p.y/cell).round()as i64,(p.z/cell).round()as i64);
    let keys=positions.iter().enumerate().map(|(i,&p)|(key(p,cell),pins.get(i).copied().unwrap_or(0.0).clamp(0.0,1.0).to_bits())).collect::<Vec<_>>();
    let mut regions=Regions{parent:(0..positions.len()).collect()};let mut welds=HashMap::new();
    for (i,&p)in positions.iter().enumerate(){
        // Preserve UV-seam welding, but never weld different pin classes or
        // use a coarse cell as evidence that two disconnected layers touch.
        let weld=(key(p,1e-4),keys[i]);if let Some(&other)=welds.get(&weld){regions.join(i,other);}else{welds.insert(weld,i);}
    }
    if spacing>0.0{for triangle in indices.chunks_exact(3){for k in 0..3{let(a,b)=(triangle[k]as usize,triangle[(k+1)%3]as usize);if keys[a]==keys[b]{regions.join(a,b);}}}}
    let mut roots=HashMap::new();let mut owner=Vec::with_capacity(positions.len());let mut members:Vec<Vec<u32>>=Vec::new();
    for i in 0..positions.len(){let root=regions.root(i);let next=members.len()as u32;let group=*roots.entry(root).or_insert(next);
        if group==next{members.push(Vec::new());}members[group as usize].push(i as u32);owner.push(group);
    }
    (owner,members)
}
