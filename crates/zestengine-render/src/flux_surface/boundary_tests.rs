//! Actual production filtering across virtual pages with unequal histories.
use super::*;
use crate::texture::GpuTexture;

#[derive(Clone,Copy)]
struct Page {card:Card,value:f32,kind:u32}
struct Check {query:[f32;4],value:f32,previous:f32,black:bool,valid:bool,label:String}
fn index(level:u32,x:u32,y:u32)->usize {match level {0=>0,1=>1+(y*2+x) as usize,_=>5+(y*4+x) as usize}}

#[test]
fn virtual_surface_boundaries_preserve_continuity_and_valid_dark_history() {
    for spacing in [4usize,2] {check_boundaries(spacing);}
}
fn check_boundaries(spacing:usize) {
    let left=0.5+0.5/spacing as f32;let right=1.0-left;
    let Some(g)=crate::gpu_tests::gpu() else {return;};
    let validation_before=crate::cluster_headless::VALIDATION_ERRORS.load(std::sync::atomic::Ordering::Relaxed);
    let device=&g.device;let mut alloc=g.allocator().unwrap();
    let mesh=crate::MeshData {vertices:vec![Vertex {position:[-1.0,-1.0,-0.1],..Default::default()},
        Vertex {position:[1.0,1.0,0.1],..Default::default()}],indices:vec![0,1,0],has_lightmap_uv:false};
    let capture=crate::flux_cards::MeshCards::new(&mesh,false).captures[4];
    let lo=capture.bounds[0];let span=capture.bounds[1]-lo;
    let point=|u:f32,v:f32|Vec3::new(lo.x+span.x*u,lo.y+span.y*v,0.0);
    let mut physical=Vec::new();
    let make=|slot:usize,level:u32,x:u32,y:u32,value:f32,kind:u32| {
        let c=if level==0 {capture} else {capture.crop(level,x,y)};
        let mut card=card_for(c,Mat4::IDENTITY,pages::Page {x:(slot as u32%8)*128,y:(slot as u32/8)*128,size:128});
        card.rect[3]=55;card.virtual_page=[1,level,x|(y<<16),0];
        // Neighbour pages were captured in different rigid poses. The physical
        // receivers still share one world plane after their capture deltas.
        if slot%2==1 {card.rect[3]|=64;card.capture_to_world=(Mat4::from_translation(Vec3::new(1.0,2.0,3.0))
            *Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2)).to_cols_array_2d();}
        Page {card,value,kind}
    };
    physical.push(make(0,0,0,0,10.0,0));
    for level in 1..=2 {for y in 0..1<<level {for x in 0..1<<level {
        physical.push(make(physical.len(),level,x,y,1.0+x as f32+(y*(1<<level)) as f32,0));
    }}}
    let special=physical.len();
    // Valid black, unwritten, nonfinite, wrong normal, wrong depth.
    for kind in 1..=5 {physical.push(make(physical.len(),2,0,0,7.0,kind));}
    let bad_root=physical.len();physical.push(make(bad_root,0,0,0,10.0,5));
    let base:Vec<_>=physical[..21].iter().map(|p|p.card).collect();
    let mut cards=base.clone();let mut checks=Vec::new();
    let add=|checks:&mut Vec<Check>,card:usize,p:Vec3,value:f32,previous:f32,black:bool,valid:bool,label:String| {
        for kind in 0..2 {checks.push(Check {query:p.extend((card+kind*65536) as f32).to_array(),value,previous,black,valid,label:format!("{label}, indirect={kind}")});}
    };
    // Every horizontal/vertical edge and interior four-page corner at both
    // fine levels. Poisoned gutters must never replace adjacent core probes.
    for level in 1..=2 {
        let dim=1u32<<level;let value=|x,y|1.0+(y*dim+x) as f32;
        for y in 0..dim {for x in 1..dim {
            let p=point(x as f32/dim as f32,(y as f32+0.5)/dim as f32);
            let expected=left*value(x-1,y)+right*value(x,y);
            for (tile,dx) in [(x-1,-1e-7),(x,1e-7)] {add(&mut checks,index(level,tile,y),p+Vec3::X*dx,expected,expected,false,true,format!("mip{level} vertical{x},{y}"));}
        }}
        for y in 1..dim {for x in 0..dim {
            let p=point((x as f32+0.5)/dim as f32,y as f32/dim as f32);
            let expected=left*value(x,y-1)+right*value(x,y);
            for (tile,dy) in [(y-1,-1e-7),(y,1e-7)] {add(&mut checks,index(level,x,tile),p+Vec3::Y*dy,expected,expected,false,true,format!("mip{level} horizontal{x},{y}"));}
        }}
        for y in 1..dim {for x in 1..dim {
            let p=point(x as f32/dim as f32,y as f32/dim as f32);
            let expected=left*left*value(x-1,y-1)+right*left*value(x,y-1)
                +left*right*value(x-1,y)+right*right*value(x,y);
            for dy in 0..2 {for dx in 0..2 {add(&mut checks,index(level,x-1+dx,y-1+dy),
                p+Vec3::new(if dx==0 {-1e-7} else {1e-7},if dy==0 {-1e-7} else {1e-7},0.0),expected,expected,false,true,format!("mip{level} corner{x},{y}"));}}
        }}
    }
    let variant=|cards:&mut Vec<Card>,edit:&dyn Fn(&mut [Card])| {
        let offset=cards.len();let mut part=base.clone();
        for c in &mut part {c.virtual_page[0]=offset as u32+1;}
        edit(&mut part);cards.extend(part);offset
    };
    // Missing adjacent fine tile, first within a parent and then across parent
    // boundaries. Both sides must evaluate the same coarser lighting at the edge.
    for boundary in [1,2] {
        let offset=variant(&mut cards,&|c|c[index(2,boundary,0)].rect[3]=0);
        let p=point(boundary as f32/4.0,0.125);let expected=if boundary==1 {1.0} else {1.0+right};
        add(&mut checks,offset+index(2,boundary-1,0),p-Vec3::X*1e-7,expected,expected,false,true,format!("missing fine{boundary} left"));
        add(&mut checks,offset+index(1,boundary/2,0),p+Vec3::X*1e-7,expected,expected,false,true,format!("missing fine{boundary} parent"));
    }
    // Skip an absent intermediate mip, preserving root lighting and UVs.
    let offset=variant(&mut cards,&|c| {c[index(1,0,0)].rect[3]=0;c[index(2,1,0)].rect[3]=0;});
    let p=point(0.25,0.125);
    add(&mut checks,offset+index(2,0,0),p-Vec3::X*1e-7,10.0,10.0,false,true,"missing parent edge".into());
    add(&mut checks,offset,p+Vec3::X*1e-7,10.0,10.0,false,true,"root opposite missing parent".into());
    for kind in 1..=5 {
        let offset=variant(&mut cards,&|c| {
            let source=physical[special+kind-1].card;let target=&mut c[index(2,0,0)];
            target.rect=source.rect;target.capture_to_world=source.capture_to_world;
        });
        add(&mut checks,offset+index(2,0,0),point(0.125,0.125),if kind==1 {0.0} else {1.0},if kind==1 {0.0} else {1.0},kind==1,true,format!("fine validity{kind}"));
    }
    // Captured intermediate pages without light still lead to the root.
    for invalid_root in [false,true] {
        let offset=variant(&mut cards,&|c| {
            let source=physical[special+1].card;
            c[index(2,0,0)].rect=source.rect;c[index(2,0,0)].capture_to_world=source.capture_to_world;
            c[index(1,0,0)].rect[3]=1;
            if invalid_root {c[0].rect=physical[bad_root].card.rect;c[0].capture_to_world=physical[bad_root].card.capture_to_world;}
        });
        add(&mut checks,offset+index(2,0,0),point(0.125,0.125),10.0,10.0,false,!invalid_root,format!("unlit ancestor, invalid root={invalid_root}"));
    }
    // A radiosity dispatch may use previous lighting only. The view gather
    // can consume the newly lit fine page in the same snapshot.
    let offset=variant(&mut cards,&|c|c[index(2,1,0)].rect[3]&=!(2|16));
    add(&mut checks,offset+index(2,1,0),point(0.375,0.125),2.0,1.0,false,true,"current-only fine page".into());
    virtual_pages::link(&mut cards);
    for (i,c) in cards.iter().enumerate() {
        if c.rect[3]==0 {continue;}
        for &link in &c.neighbours {if link!=0 {let n=cards[link as usize-1];assert_eq!(n.virtual_page[0],c.virtual_page[0]);assert_eq!(n.virtual_page[1],c.virtual_page[1]);}}
        if c.virtual_page[3]!=0 {let p=cards[c.virtual_page[3] as usize-1];assert!(p.virtual_page[1]<c.virtual_page[1],"ancestor cycle at{i}");}
    }
    const W:usize=1024;const H:usize=512;
    let mut position=vec![[0.0f32;4];W*H];let mut normal=position.clone();let albedo=vec![[0.5f32,0.5,0.5,0.0];W*H];
    let emission=vec![[0.0f32;4];W*H];
    for page in &physical {
        let c=page.card;let projection=Mat4::from_cols_array_2d(&c.view_proj).inverse();let to_capture=Mat4::from_cols_array_2d(&c.capture_to_world).inverse();
        for y in 0..128 {for x in 0..128 {
            let mut p=projection.project_point3(Vec3::new((x as f32+0.5)/64.0-1.0,(y as f32+0.5)/64.0-1.0,0.5));
            if page.kind==5 {p.z=1.0;}
            let n=if page.kind==4 {Vec3::Y} else {Vec3::Z};
            let at=(c.rect[1] as usize+y)*W+c.rect[0] as usize+x;
            position[at]=to_capture.transform_point3(p).extend(1.0).to_array();
            normal[at]=to_capture.transform_vector3(n).extend(1.0).to_array();
        }}
    }
    let mut textures=Vec::new();
    for data in [&position,&normal,&albedo,&emission] {textures.push(GpuTexture::upload_pixels(device,&mut alloc,g.pool,g.queue,W as u32,H as u32,
        vk::Format::R32G32B32A32_SFLOAT,bytemuck::cast_slice(data)).unwrap());}
    let queries:Vec<_>=checks.iter().map(|c|c.query).collect();
    for phase in 0..2 {
        let mut direct=vec![[0.0f32;4];W*H/(spacing*spacing)];let mut indirect=direct.clone();
        for page in &physical {
            let value=if page.kind==1 {0.0} else if phase==0 {page.value} else {18.0-page.value};
            for y in 0..128/spacing {for x in 0..128/spacing {
                let at=(page.card.rect[1] as usize/spacing+y)*(W/spacing)+page.card.rect[0] as usize/spacing+x;
                let gutter=(x<4/spacing || y<4/spacing || x>=124/spacing || y>=124/spacing) && page.kind==0;
                let value=if page.kind==3 {f32::NAN} else if gutter {300.0} else {value};
                let count=if page.kind==2 {0.0} else if phase==0 {32.0} else {1.0};
                direct[at]=[value,value,value,count];indirect[at]=[value*0.1,value*0.1,value*0.1,count];
            }}
        }
        for data in [&direct,&indirect] {textures.push(GpuTexture::upload_pixels(device,&mut alloc,g.pool,g.queue,(W/spacing) as u32,(H/spacing) as u32,
            vk::Format::R32G32B32A32_SFLOAT,bytemuck::cast_slice(data)).unwrap());}
        for (previous,shader) in [(false,include_bytes!(concat!(env!("OUT_DIR"),"/flux_surface_boundary_test.comp.spv")).as_slice()),
            (true,include_bytes!(concat!(env!("OUT_DIR"),"/flux_surface_boundary_previous_test.comp.spv")).as_slice())] {
            let views:Vec<_>=textures.iter().map(|t|t.view).collect();
            let values=lookup_tests::sample(device,&mut alloc,g.pool,g.queue,&cards,&[],&views,&queries,shader,12);
            for (check,actual) in checks.iter().zip(&values) {
                assert!(actual.iter().all(|v|v.is_finite()),"{} nonfinite result {actual:?}",check.label);
                assert_eq!(actual[3]>0.5,check.valid,"{} validity {actual:?}",check.label);
                if !check.valid {continue;}
                let value=if previous {check.previous} else {check.value};
                let value=if check.black {0.0} else if phase==0 {value} else {18.0-value};
                let expected=value*if check.query[3]<65536.0 {1.1} else {0.05};
                for c in 0..3 {assert!((actual[c]-expected).abs()<0.001,"{} phase{phase} previous={previous}: {actual:?}, expected{expected}",check.label);}
            }
            eprintln!("[Flux boundaries]spacing={spacing} {} GPU queries passed, phase{phase}, previous={previous}: every256/512 seam/corner, poisoned gutters, unequal histories, mixed/missing mips, rigid captures, validity and dark lighting",checks.len());
        }
        for _ in 0..2 {textures.pop().unwrap().destroy(device,&mut alloc);}
    }
    for texture in &mut textures {texture.destroy(device,&mut alloc);}
    assert_eq!(crate::cluster_headless::VALIDATION_ERRORS.load(std::sync::atomic::Ordering::Relaxed),validation_before);
}
