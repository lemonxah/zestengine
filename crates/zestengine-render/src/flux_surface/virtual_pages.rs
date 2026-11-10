//! Logical chart refinement; coarse captures remain locked while fine tiles
//! compete for the remaining physical-page budget.
use super::*;
pub(super) const MAX_RESOLUTION:u32=512;
/// Publish topology only from captured pages in this frame's immutable table.
/// Readiness is component-specific and checked by the sampling shader, so a
/// newly captured but unlit parent can still lead to a valid coarse ancestor.
pub(super) fn link(cards:&mut [Card]) {
    let table:BTreeMap<_,_>=cards.iter().enumerate().filter(|(_,c)|c.rect[3]!=0 && c.virtual_page[0]!=0)
        .map(|(slot,c)|((c.virtual_page[0],c.virtual_page[1],c.virtual_page[2]&65535,c.virtual_page[2]>>16),slot as u32+1)).collect();
    for card in cards {
        card.neighbours=[0;4];card.virtual_page[3]=0;
        let [root,level,xy,_]=card.virtual_page;
        if card.rect[3]==0 || root==0 || level==0 {continue;}
        let x=xy&65535;let y=xy>>16;
        for parent in (0..level).rev() {
            let delta=level-parent;
            if let Some(&slot)=table.get(&(root,parent,x>>delta,y>>delta)) {card.virtual_page[3]=slot;break;}
        }
        for (edge,(dx,dy)) in [(-1,0),(1,0),(0,-1),(0,1)].into_iter().enumerate() {
            let nx=x as i32+dx;let ny=y as i32+dy;
            if nx<0 || ny<0 || nx>=1<<level || ny>=1<<level {continue;}
            card.neighbours[edge]=table.get(&(root,level,nx as u32,ny as u32)).copied().unwrap_or(0);
        }
    }
}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(super) struct Tile {pub level:u32,pub x:u32,pub y:u32}
impl Tile {
    pub fn key(self,parent:&[u32])->Vec<u32> {
        let mut key=parent.to_vec();key.extend([u32::MAX,self.level,self.x,self.y]);key
    }
    pub fn parent(key:&[u32])->&[u32] {&key[..key.len()-4]}
    /// Equal border ratio on parent and child charts, including the cropped
    /// chart's gutter. Push constants stay within Vulkan's128-byte minimum.
    pub fn parent_uv(self)->[f32;4] {
        let scale=1.0/(1u32<<self.level) as f32;let border=CARD_SIZE as f32/(CARD_SIZE-4) as f32;
        [scale,scale,0.5+((self.x as f32+0.5)*scale-0.5)/border-scale*0.5,
            0.5+((self.y as f32+0.5)*scale-0.5)/border-scale*0.5]
    }
}
pub(super) fn level(bounds:[Vec3;2],views:&[LightingView],current:u32)->u32 {
    let scale=views.iter().map(|v|v.surface_resolution).reduce(f32::max).unwrap_or(1.0);
    let span=(bounds[1]-bounds[0]).max_element();
    let distance=views.iter().map(|v|(v.position-v.position.clamp(bounds[0],bounds[1])).length().max(1.0))
        .reduce(f32::min).unwrap_or(2.0);
    let projected=(span*32.0/distance).min(span*64.0)*scale;
    let maximum=((MAX_RESOLUTION as f32*scale) as u32).next_power_of_two();
    let size=pages::MAX<<current;
    let target=if size<=maximum && projected>=size as f32*0.4 && projected<=size as f32*1.25 {size}
        else {(projected.ceil().max(1.0) as u32).min(maximum).next_power_of_two().max(pages::MAX)};
    (target/pages::MAX).ilog2()
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn child_projection_matches_parent_uv_and_keeps_depth() {
        for extent in [[4.0,3.0,1.0],[16.0,0.001,0.1]] {
        let vertices=[Vertex {position:extent.map(|v|-v),..Vertex::default()},Vertex {position:extent,..Vertex::default()}];
        let data=crate::MeshData {vertices:vertices.to_vec(),indices:vec![0,1,0],has_lightmap_uv:false};
        let captures=crate::flux_cards::MeshCards::new(&data,false);
        for capture in &captures.captures[..6] {for level in [1,2] {for y in 0..1<<level {for x in 0..1<<level {
            let tile=Tile {level,x,y};let page=pages::Page {x:0,y:0,size:128};
            let parent=card_for(*capture,Mat4::IDENTITY,page);let child=card_for(capture.crop(level,x,y),Mat4::IDENTITY,page);
            let parent_proj=Mat4::from_cols_array_2d(&parent.view_proj);let child_inv=Mat4::from_cols_array_2d(&child.view_proj).inverse();
            let transform=tile.parent_uv();
            for uv in [[0.0,0.0],[0.5,0.5],[1.0,1.0]] {
                let p=child_inv.project_point3(Vec3::new(uv[0]*2.0-1.0,uv[1]*2.0-1.0,0.3));
                let q=parent_proj.project_point3(p);
                assert!((q.z-0.3).abs()<1e-5);
                for c in 0..2 {assert!(((q[c]*0.5+0.5)-(uv[c]*transform[c]+transform[c+2])).abs()<1e-5);}
            }
        }}}}
        }
    }
}
