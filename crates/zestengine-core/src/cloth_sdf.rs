//! CPU level sets for cloth collision. Fields are independent of rendering/GI,
//! persist as compact samples, and can be shared by many moving colliders.
use std::{collections::HashMap,sync::{Arc,OnceLock}};
use glam::{Vec3,Mat3,Mat4,Quat,DVec3};
use serde::{Serialize,Deserialize,Serializer,Deserializer};
use crate::PackedBytes;

/// Authoring data for a level-set cloth collider. Baking is performed by the
/// editor/importer; games only need the embedded validated field.
#[derive(Clone,Debug,Serialize,Deserialize)]
#[serde(default)]
pub struct ClothLevelSet{pub source:String,pub mesh:usize,pub resolution:u32,pub field:Option<ClothSdf>}
impl Default for ClothLevelSet{fn default()->Self{Self{source:String::new(),mesh:0,resolution:32,field:None}}}

#[derive(Clone,Debug)]
pub struct ClothSdf(Arc<Field>);
#[derive(Debug)]
struct Field{dims:[usize;3],origin:Vec3,cell:Vec3,values:Vec<f32>,lipschitz:f32,encoded:OnceLock<PackedBytes>}
#[derive(Serialize,Deserialize)]
struct SavedField{dims:[u32;3],origin:[f32;3],cell:[f32;3],samples:PackedBytes}
impl PartialEq for ClothSdf{fn eq(&self,other:&Self)->bool{Arc::ptr_eq(&self.0,&other.0)}}
impl Serialize for ClothSdf{
    fn serialize<S:Serializer>(&self,s:S)->Result<S::Ok,S::Error>{
        #[derive(Serialize)]
        struct Borrowed<'a>{dims:[u32;3],origin:[f32;3],cell:[f32;3],samples:&'a PackedBytes}
        let samples=self.0.encoded.get_or_init(||PackedBytes::new(self.0.values.iter().flat_map(|v|v.to_le_bytes()).collect()));
        Borrowed{dims:self.0.dims.map(|n|n as u32),origin:self.0.origin.to_array(),cell:self.0.cell.to_array(),samples}.serialize(s)
    }
}
impl<'de> Deserialize<'de> for ClothSdf{
    fn deserialize<D:Deserializer<'de>>(d:D)->Result<Self,D::Error>{
        let f=SavedField::deserialize(d)?;let values=f.samples.as_slice().as_chunks::<4>().0.iter().map(|b|f32::from_le_bytes(*b)).collect();
        if f.samples.len()%4!=0{return Err(serde::de::Error::custom("Invalid level-set samples"));}
        Self::from_samples(f.dims,Vec3::from(f.origin),Vec3::from(f.cell),values).map_err(serde::de::Error::custom)
    }
}
impl ClothSdf{
    pub fn from_samples(dims:[u32;3],origin:Vec3,cell:Vec3,values:Vec<f32>)->Result<Self,String>{
        if dims.iter().any(|&n|!(3..=256).contains(&n))||!origin.is_finite()||!cell.is_finite()||cell.min_element()<=1e-8{return Err("Invalid level-set dimensions or cell size".into());}
        let hi=origin+cell*Vec3::from_array(dims.map(|n|(n-1)as f32));
        if !hi.is_finite()||!(origin+cell).cmpgt(origin).all(){return Err("Level-set bounds or cell spacing exceed coordinate precision".into());}
        let dims=dims.map(|n|n as usize);let count=dims.iter().product::<usize>();
        if count>4_000_000||values.len()!=count||values.iter().any(|v|!v.is_finite()){return Err("Invalid level-set samples (limit: four million)".into());}
        if !values.iter().any(|v|*v<0.0){return Err("Level set contains no interior samples; increase resolution or use a thicker collision proxy".into());}
        let index=|x,y,z|x+dims[0]*(y+dims[1]*z);let mut slope=Vec3::ZERO;
        for z in 0..dims[2]{for y in 0..dims[1]{for x in 0..dims[0]{let i=index(x,y,z);
            if (x==0||y==0||z==0||x+1==dims[0]||y+1==dims[1]||z+1==dims[2])&&values[i]<=0.0{return Err("Level-set boundary must be outside the body; add padding".into());}
            for (axis,stride) in [1,dims[0],dims[0]*dims[1]].into_iter().enumerate(){if [x,y,z][axis]+1<dims[axis]{slope[axis]=slope[axis].max((values[i+stride]-values[i]).abs()/cell[axis]);}}
        }}}
        // Outside the grid the clamped sample is extended by distance to the
        // box. This also bounds its gradient, including at box edges.
        let lipschitz=(slope.length()+1.0).max(1.0);
        if !lipschitz.is_finite(){return Err("Level-set sample gradient exceeds the supported range".into());}
        Ok(Self(Arc::new(Field{dims,origin,cell,values,lipschitz,encoded:OnceLock::new()})))
    }
    pub fn dimensions(&self)->[usize;3]{self.0.dims}
    pub fn cell_size(&self)->Vec3{self.0.cell}
    pub fn bounds(&self)->(Vec3,Vec3){let f=&self.0;(f.origin,f.origin+f.cell*Vec3::from_array(f.dims.map(|n|(n-1)as f32)))}
    pub fn bytes(&self)->usize{self.0.values.len()*4}
    /// Interpolated signed scalar and its derivative in field coordinates.
    /// The derivative is not normalized; it is needed for scaled projection.
    pub fn sample(&self,p:Vec3)->(f32,Vec3){
        let f=&self.0;let(lo,hi)=self.bounds();let q=p.clamp(lo,hi);let grid=(q-lo)/f.cell;
        let base:[usize;3]=std::array::from_fn(|i|(grid[i].floor()as usize).min(f.dims[i]-2));let u=grid-Vec3::from_array(base.map(|n|n as f32));
        let mut value=0.0;let mut gradient=Vec3::ZERO;
        for z in 0..2{for y in 0..2{for x in 0..2{
            let bits=[x,y,z];let w: [f32;3]=std::array::from_fn(|i|if bits[i]==0{1.0-u[i]}else{u[i]});
            let v=f.values[base[0]+x+f.dims[0]*(base[1]+y+f.dims[1]*(base[2]+z))];value+=v*w[0]*w[1]*w[2];
            for i in 0..3{gradient[i]+=v*(if bits[i]==0{-1.0}else{1.0})*w[(i+1)%3]*w[(i+2)%3]/f.cell[i];}
        }}}
        let outside=p-q;let distance=outside.length();if distance>0.0{
            value+=distance;for i in 0..3{if p[i]<lo[i]||p[i]>hi[i]{gradient[i]=0.0;}}gradient+=outside/distance;
        }
        (value,gradient)
    }
    /// Generate from a closed triangle surface. UV/normal splits are welded
    /// by exact position for manifold validation. Winding may be reversed.
    pub fn from_mesh(vertices:&[Vec3],indices:&[u32],resolution:u32)->Result<Self,String>{
        if !(8..=128).contains(&resolution)||vertices.len()<4||vertices.len()>1_000_000||indices.len()<12||!indices.len().is_multiple_of(3)||indices.len()>6_000_000||vertices.iter().any(|p|!p.is_finite())||indices.iter().any(|&i|i as usize>=vertices.len()){return Err("Invalid closed level-set mesh or resolution (8–128)".into());}
        let mut welded=HashMap::new();let ids:Vec<_>=vertices.iter().map(|p|{let key=p.to_array().map(|x|if x==0.0{0}else{x.to_bits()});let next=welded.len();*welded.entry(key).or_insert(next)}).collect();
        let mut edges=HashMap::new();let mut triangles=Vec::new();
        for t in indices.as_chunks::<3>().0{let points:[Vec3;3]=std::array::from_fn(|k|vertices[t[k]as usize]);
            if (points[1]-points[0]).cross(points[2]-points[0]).length_squared()<1e-20{return Err("Level-set mesh contains a degenerate triangle".into());}
            for k in 0..3{let(a,b)=(ids[t[k]as usize],ids[t[(k+1)%3]as usize]);*edges.entry((a.min(b),a.max(b))).or_insert(0)+=1;}
            triangles.push(points);
        }
        if edges.values().any(|&n|n!=2){return Err("Level-set collision needs a closed manifold mesh; close holes or use a closed collision proxy".into());}
        let tree=Node::build(&triangles,(0..triangles.len()).collect());let span=tree.hi-tree.lo;let longest=span.max_element();
        if span.min_element()<=1e-6{return Err("Level-set mesh has no volume".into());}
        let target_cell=longest/(resolution-1)as f32;let intervals=(span/target_cell).ceil().max(Vec3::splat(4.0));
        let cell=span/intervals;let origin=tree.lo-cell*2.0;let dims=intervals.to_array().map(|v|v as u32+5);
        let directions=[DVec3::new(1.0,0.3713906764,0.52999894).normalize(),DVec3::new(-0.213199,1.0,0.419132).normalize(),DVec3::new(0.319133,-0.271199,1.0).normalize()];
        let mut values=Vec::with_capacity(dims.iter().map(|&v|v as usize).product());let mut hits=Vec::new();
        for z in 0..dims[2]{for y in 0..dims[1]{for x in 0..dims[0]{
            let p=origin+Vec3::new(x as f32,y as f32,z as f32)*cell;let mut distance=f32::INFINITY;tree.nearest(&triangles,p,&mut distance);let distance=distance.sqrt();let mut inside=0;
            if p.cmpge(tree.lo).all()&&p.cmple(tree.hi).all(){for direction in directions{
                hits.clear();tree.ray(&triangles,p.as_dvec3(),direction,&mut hits);hits.sort_by(f64::total_cmp);
                // Shared triangle edges contribute one surface crossing.
                hits.dedup_by(|a,b|(*a-*b).abs()<longest as f64*1e-9);inside+=usize::from(hits.len()%2==1);
            }}
            values.push(if inside>=2{-distance}else{distance});
        }}}
        Self::from_samples(dims,origin,cell,values)
    }
}

struct Node{lo:Vec3,hi:Vec3,children:Option<(Box<Node>,Box<Node>)>,triangles:Vec<usize>}
impl Node{
    fn build(triangles:&[[Vec3;3]],mut ids:Vec<usize>)->Self{
        let mut lo=Vec3::splat(f32::INFINITY);let mut hi=Vec3::splat(f32::NEG_INFINITY);for &i in &ids{for p in triangles[i]{lo=lo.min(p);hi=hi.max(p);}}
        let children=if ids.len()>8{let span=hi-lo;let axis=if span.x>=span.y&&span.x>=span.z{0}else if span.y>=span.z{1}else{2};let center=|i:usize|triangles[i].iter().map(|p|p[axis]).sum::<f32>();ids.sort_by(|&a,&b|center(a).total_cmp(&center(b)));let right=ids.split_off(ids.len()/2);Some((Box::new(Self::build(triangles,std::mem::take(&mut ids))),Box::new(Self::build(triangles,right))))}else{None};Self{lo,hi,children,triangles:ids}
    }
    fn distance(&self,p:Vec3)->f32{p.distance_squared(p.clamp(self.lo,self.hi))}
    fn nearest(&self,triangles:&[[Vec3;3]],p:Vec3,best:&mut f32){
        if self.distance(p)>*best{return;}
        if let Some((a,b))=&self.children{let(a,b)=if a.distance(p)<b.distance(p){(a,b)}else{(b,a)};a.nearest(triangles,p,best);b.nearest(triangles,p,best);}else{for &i in &self.triangles{*best=best.min(p.distance_squared(crate::cloth_contact::closest_triangle(p,triangles[i]).0));}}
    }
    fn ray(&self,triangles:&[[Vec3;3]],p:DVec3,d:DVec3,hits:&mut Vec<f64>){
        let a=(self.lo.as_dvec3()-p)/d;let b=(self.hi.as_dvec3()-p)/d;
        if a.min(b).max_element()>a.max(b).min_element()||a.max(b).min_element()<0.0{return;}
        if let Some((a,b))=&self.children{a.ray(triangles,p,d,hits);b.ray(triangles,p,d,hits);}else{for &i in &self.triangles{
            let [a,b,c]=triangles[i].map(|v|v.as_dvec3());let e=b-a;let f=c-a;let h=d.cross(f);let det=e.dot(h);if det.abs()<1e-20{continue;}
            let u=(p-a).dot(h)/det;let q=(p-a).cross(e);let v=d.dot(q)/det;let t=f.dot(q)/det;
            if u>=-1e-12&&v>=-1e-12&&u+v<=1.0+1e-12&&t>1e-12{hits.push(t);}
        }}
    }
}

/// A field in world space. Nonuniform scales and affine transforms use the
/// inverse-transpose gradient, never an unscaled object-space normal.
#[derive(Clone,Debug,PartialEq)]
pub struct ClothSdfShape{pub field:ClothSdf,world:Mat4,inverse:Mat4,normal:Mat3,frame:Frame}
impl ClothSdfShape{
    pub fn new(field:ClothSdf,world:Mat4)->Result<Self,String>{
        if !world.is_finite()||!world.determinant().is_finite()||world.determinant().abs()<1e-10||world.x_axis.w!=0.0||world.y_axis.w!=0.0||world.z_axis.w!=0.0||world.w_axis.w!=1.0{return Err("Level-set collider needs a finite, invertible affine transform".into());}
        let inverse=world.inverse();Ok(Self{field,world,inverse,normal:Mat3::from_mat4(inverse).transpose(),frame:Frame::new(world)?})
    }
    pub fn transform(&self)->Mat4{self.world}
    pub fn sample(&self,p:Vec3)->(f32,Vec3){
        let (phi,gradient)=self.field.sample(self.inverse.transform_point3(p));let g=self.normal*gradient;
        let len=g.length();if len>1e-8{(phi/len,g/len)}else{(phi,self.normal.x_axis.normalize_or(Vec3::X))}
    }
    pub fn project(&self,p:Vec3,radius:f32)->Option<(Vec3,Vec3)>{
        if !p.is_finite()||!radius.is_finite(){return None;}let radius=radius.max(0.0);let mut out=p;let mut normal=Vec3::X;let mut hit=false;
        for _ in 0..24{let(distance,n)=self.sample(out);if distance>=radius-1e-6{break;}hit=true;normal=n;out+=n*(radius-distance);}
        hit.then_some((out,normal))
    }
}

/// Affine motion uses a polar decomposition: rotate the orthogonal frame and
/// interpolate its positive-definite stretch. This supports parent-induced
/// shear without interpolating a rotation matrix through a singular midpoint.
#[derive(Clone,Copy,Debug,PartialEq)]
struct Frame{rotation:Quat,stretch:Mat3,reflection:Mat3,translation:Vec3}
fn operator_bound(m:Mat3)->f32{
    let abs=Mat3::from_cols(m.x_axis.abs(),m.y_axis.abs(),m.z_axis.abs());
    let columns=abs.x_axis.element_sum().max(abs.y_axis.element_sum()).max(abs.z_axis.element_sum());
    let rows=(abs.x_axis+abs.y_axis+abs.z_axis).max_element();(columns*rows).sqrt().min(matrix_norm(m))
}
fn matrix_norm(m:Mat3)->f32{(m.x_axis.length_squared()+m.y_axis.length_squared()+m.z_axis.length_squared()).sqrt()}
impl Frame{
    fn new(world:Mat4)->Result<Self,String>{
        let linear=Mat3::from_mat4(world);let reflection=if linear.determinant()<0.0{Mat3::from_diagonal(Vec3::new(-1.0,1.0,1.0))}else{Mat3::IDENTITY};
        let positive=linear*reflection;let mut r=positive/matrix_norm(positive);
        for _ in 0..48{let next=(r+r.inverse().transpose())*0.5;let change=matrix_norm(next-r);r=next;if change<1e-6{break;}}
        if !r.is_finite(){return Err("Unsupported level-set transform conditioning".into());}
        let rotation=Quat::from_mat3(&r).normalize();let rotation_matrix=Mat3::from_quat(rotation);let raw=rotation_matrix.transpose()*positive;let stretch=(raw+raw.transpose())*0.5;
        let minor=stretch.x_axis.x*stretch.y_axis.y-stretch.y_axis.x*stretch.x_axis.y;
        if stretch.x_axis.x<=0.0||minor<=0.0||stretch.determinant()<=0.0||!stretch.inverse().is_finite()
            ||matrix_norm(rotation_matrix*stretch-positive)>1e-4*matrix_norm(positive).max(1.0){return Err("Invalid level-set stretch".into());}
        Ok(Self{rotation,stretch,reflection,translation:world.w_axis.truncate()})
    }
}
/// Continuous query includes translation, rotation, independent scale and shear.
/// A field-space Lipschitz bound prevents fixed-sample tunnelling.
pub struct ClothSdfMotion{start:Arc<ClothSdfShape>,end:Arc<ClothSdfShape>,a:Frame,b:Frame,min_scale:f32,stationary:bool}
impl ClothSdfMotion{
    pub fn new(start:impl Into<Arc<ClothSdfShape>>,end:impl Into<Arc<ClothSdfShape>>)->Result<Self,String>{
        let mut start=start.into();let end=end.into();
        if !Arc::ptr_eq(&start.field.0,&end.field.0){start=end.clone();}
        let a=start.frame;let b=end.frame;
        if a.reflection!=b.reflection{return Err("A moving level-set transform cannot change handedness through zero scale".into());}
        // The smallest eigenvalue of a convex combination of SPD stretches is
        // at least the smaller endpoint eigenvalue. Inverse matrix norm bounds
        // give a conservative bound without an iterative eigensolver.
        let min_scale=1.0/operator_bound(a.stretch.inverse()).max(operator_bound(b.stretch.inverse()));
        let stationary=start.world==end.world;Ok(Self{start,end,a,b,min_scale,stationary})
    }
    pub fn inverse_at(&self,t:f32)->Mat4{if t<=0.0{self.start.inverse}else if self.stationary||t>=1.0{self.end.inverse}else{self.at(t).inverse()}}
    pub fn at(&self,t:f32)->Mat4{
        if t<=0.0{return self.start.world;}
        if self.stationary||t>=1.0{return self.end.world;}
        let linear=Mat3::from_quat(self.a.rotation.slerp(self.b.rotation,t))*(self.a.stretch*(1.0-t)+self.b.stretch*t)*self.a.reflection;
        Mat4::from_cols(linear.x_axis.extend(0.0),linear.y_axis.extend(0.0),linear.z_axis.extend(0.0),self.a.translation.lerp(self.b.translation,t).extend(1.0))
    }
    /// Certify a swept triangle clear of a moving field.
    /// Trilinear interpolation reproduces affine functions: the smallest grid
    /// residual from a tangent plane bounds that residual everywhere inside
    /// these cells. This avoids subdividing a broad, nearly planar safe region.
    pub(crate) fn triangle_separated(&self,start:[Vec3;3],end:[Vec3;3],radius:f32)->bool{
        let p=[start.map(|p|self.start.inverse.transform_point3(p)),end.map(|p|self.end.inverse.transform_point3(p))];
        if p.iter().flatten().any(|p|!p.is_finite()){return false;}
        let f=&self.end.field.0;let lo=p.iter().flatten().copied().fold(Vec3::splat(f32::INFINITY),Vec3::min);let hi=p.iter().flatten().copied().fold(Vec3::splat(f32::NEG_INFINITY),Vec3::max);
        let extent=Vec3::from_array(f.dims.map(|n|(n-1)as f32));
        let grid_lo=((lo-f.origin)/f.cell).clamp(Vec3::ZERO,extent);let grid_hi=((hi-f.origin)/f.cell).clamp(Vec3::ZERO,extent);
        if !grid_lo.is_finite()||!grid_hi.is_finite(){return false;}
        let a=grid_lo.floor().to_array().map(|x|x as usize);let b=grid_hi.ceil().to_array().map(|x|x as usize);
        if (0..3).map(|i|b[i]-a[i]+1).product::<usize>()>4096{return false;}
        let center=((lo+hi)*0.5).clamp(f.origin,f.origin+extent*f.cell);let(value,gradient)=self.end.field.sample(center);
        // Outside the grid the field adds distance from its clamped boundary.
        // A plane with |gradient| <= 1 grows no faster than that extension,
        // so the same residual certificate remains valid outside the grid.
        let gradient=gradient.clamp_length_max(1.0);
        let plane=|point:Vec3|value+(point-center).dot(gradient);
        let minimum=p.iter().flatten().map(|&p|plane(p)).fold(f32::INFINITY,f32::min);
        // In field space a rotating/stretching frame makes the path curved.
        // Bound its second derivative and the deviation from the endpoint
        // chord (at most |p''|/8 on [0,1]). For A=S^-1, A'=-A S' A and
        // A''=2 A S' A S' A. Multiplying the deviation by the field's global
        // Lipschitz bound makes the endpoint-hull certificate conservative.
        let inverse_scale=1.0/self.min_scale;let stretch=matrix_norm(self.b.stretch-self.a.stretch);
        let angle=self.a.rotation.angle_between(self.b.rotation).abs();
        let curvature=(0..3).map(|i|{
            let a=start[i]-self.a.translation;let b=end[i]-self.b.translation;let reach=a.length().max(b.length());let speed=(b-a).length();
            2.0*inverse_scale.powi(3)*stretch*stretch*reach
                +2.0*inverse_scale*inverse_scale*stretch*(angle*reach+speed)
                +inverse_scale*(angle*angle*reach+2.0*angle*speed)
        }).fold(0.0,f32::max);
        let margin=radius.max(0.0)*inverse_scale+curvature*0.125*self.end.field.0.lipschitz;
        if !margin.is_finite()||!minimum.is_finite(){return false;}
        let error=32.0*f32::EPSILON*(1.0+value.abs()+gradient.abs().dot((hi-lo).abs()));
        for z in a[2]..=b[2]{for y in a[1]..=b[1]{for x in a[0]..=b[0]{
            let point=f.origin+Vec3::new(x as f32,y as f32,z as f32)*f.cell;
            let residual=f.values[x+f.dims[0]*(y+f.dims[1]*z)]-plane(point);
            if minimum+residual<=margin+error{return false;}
        }}}
        true
    }
    /// First conservative proximity, including time and the impact normal.
    /// Nonuniform scale uses an enclosing field-space radius; callers must
    /// retain the conservative flag when resolving multiple/concave contacts.
    pub fn first_contact(&self,p0:Vec3,p1:Vec3,radius:f32)->Option<crate::cloth_contact::ClothSweep>{
        if !p0.is_finite()||!p1.is_finite()||!radius.is_finite(){return None;}
        let radius=radius.max(0.0);let scale=self.min_scale;
        let reach=(p0-self.a.translation).length().max((p1-self.b.translation).length());let angle=self.a.rotation.angle_between(self.b.rotation).abs();let ds=matrix_norm(self.b.stretch-self.a.stretch);
        let speed=(((p1-p0)-(self.b.translation-self.a.translation)).length()+angle*reach)/scale+reach*ds/(scale*scale);
        let speed=speed*self.end.field.0.lipschitz;let margin=radius/scale;let mut t=0.0;let mut exhausted=true;
        for _ in 0..256{
            let local=self.inverse_at(t).transform_point3(p0.lerp(p1,t));let gap=self.end.field.sample(local).0-margin;
            if !gap.is_finite(){return None;}
            if gap<=1e-6{exhausted=false;break;}
            if speed<1e-12||gap/speed>1.0-t{return None;}
            let next=t+0.9*gap/speed;if next<=t{break;}t=next;
        }
        let inverse=self.inverse_at(t);let (_,gradient)=self.end.field.sample(inverse.transform_point3(p0.lerp(p1,t)));
        let normal=(Mat3::from_mat4(inverse).transpose()*gradient).normalize_or(Vec3::X);
        Some(crate::cloth_contact::ClothSweep{time:t,normal,conservative:exhausted||radius>0.0})
    }
    /// Cover the complete segment by adaptively subdivided swept spheres.
    /// Inflating each midpoint by its half-segment length bounds the interior
    /// even for nonconvex fields. The spatial budget is conservative, never a
    /// fixed sample gap through which a thin body can pass.
    pub fn resolve_segment(&self,start:[Vec3;2],end:[Vec3;2],radii:[f32;2])->Option<(f32,Vec3,Vec3,Vec3)>{
        if start.iter().chain(&end).any(|p|!p.is_finite())||radii.iter().any(|r|!r.is_finite()){return None;}
        let mut stack=vec![(0.0f32,1.0f32,0u32)];let mut visited=0;let mut best:Option<(f32,f32,Vec3,Vec3,Vec3)>=None;
        let tolerance=(self.end.field.cell_size().min_element()*self.min_scale*0.25).max(0.0001);
        while let Some((a,b,depth))=stack.pop(){
            visited+=1;let u=(a+b)*0.5;let half=(start[1]-start[0]).length().max((end[1]-end[0]).length())*(b-a)*0.5;
            let r=(radii[0]+(radii[1]-radii[0])*u).max(0.0)+(radii[1]-radii[0]).abs()*(b-a)*0.5+half;
            let p0=start[0].lerp(start[1],u);let p1=end[0].lerp(end[1],u);
            let Some(hit)=self.first_contact(p0,p1,r)else{continue};
            if best.as_ref().is_some_and(|best|hit.time>best.0){continue;}
            if half>tolerance&&depth<12&&visited<256{stack.push((u,b,depth+1));stack.push((a,u,depth+1));continue;}
            if let Some((out,n,carried))=self.respond(p0,p1,r,hit.time){best=Some((hit.time,u,out-p1,n,carried));}
        }
        best.map(|(_,u,push,n,carried)|(u,push,n,carried))
    }
    /// A single-body response, iterated by the cloth solver alongside material
    /// constraints and other bodies. Later zero-surface crossings remain
    /// authoritative when an enclosing radius reports an early proximity.
    pub fn resolve(&self,p0:Vec3,p1:Vec3,radius:f32)->Option<(Vec3,Vec3,Vec3)>{
        if !radius.is_finite(){return None;}let radius=radius.max(0.0);let hit=self.first_contact(p0,p1,radius)?;
        let response=self.respond(p0,p1,radius,hit.time);
        // An enclosing radius under nonuniform scale can touch a departing
        // lobe before a different lobe actually crosses the trajectory. A
        // separating response from that early proximity is not proof that the
        // rest of the path is clear. Check the proposed path against the zero
        // surface and keep its first crossing authoritative.
        if radius>0.0&&self.start.field.sample(self.start.inverse.transform_point3(p0)).0>1e-6{
            let candidate=response.as_ref().map_or(p1,|r|r.0);
            if let Some(crossing)=self.first_contact(p0,candidate,0.0)
                && let Some(blocked)=self.respond(p0,candidate,radius,crossing.time){return Some(blocked);}
        }
        response
    }
    fn respond(&self,p0:Vec3,p1:Vec3,radius:f32,t:f32)->Option<(Vec3,Vec3,Vec3)>{
        // Retain the impact side even when the endpoint has crossed right
        // through the body. Transport that surface point with the collider.
        let mut local=self.inverse_at(t).transform_point3(p0.lerp(p1,t));let mut gradient=Vec3::X;
        for _ in 0..24{let(phi,g)=self.end.field.sample(local);gradient=g.normalize_or(Vec3::X);if phi.abs()<1e-6{break;}local-=gradient*(phi/g.length().max(1.0));}
        let normal=(self.end.normal*gradient).normalize_or(Vec3::X);
        let support=self.end.world.transform_point3(local)+normal*radius;
        let start_normal=(self.start.normal*gradient).normalize_or(normal);
        let carried=support-(self.start.world.transform_point3(local)+start_normal*radius);
        let depth=(p1-support).dot(normal);let mut out=p1;if depth<0.0{out-=normal*depth;}
        if let Some((projected,_))=self.end.project(out,radius){out=projected;}
        if out.distance_squared(p1)<1e-18&&(p1-p0-carried).dot(normal)>=0.0{return None;}
        Some((out,normal,carried))
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    fn cube(half:Vec3)->(Vec<Vec3>,Vec<u32>){
        let p=vec![Vec3::new(-1.0,-1.0,-1.0),Vec3::new(1.0,-1.0,-1.0),Vec3::new(1.0,1.0,-1.0),Vec3::new(-1.0,1.0,-1.0),Vec3::new(-1.0,-1.0,1.0),Vec3::new(1.0,-1.0,1.0),Vec3::new(1.0,1.0,1.0),Vec3::new(-1.0,1.0,1.0)].into_iter().map(|p|p*half).collect();
        (p,vec![0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,3,7,6,3,6,2,0,4,7,0,7,3,1,2,6,1,6,5])
    }
    fn field(half:Vec3)->ClothSdf{let(p,t)=cube(half);ClothSdf::from_mesh(&p,&t,17).unwrap()}
    #[test]
    fn cloth_sdf_closed_mesh_sign_gradient_and_persistence(){
        let field=field(Vec3::ONE);assert!((field.sample(Vec3::ZERO).0+1.0).abs()<1e-5);
        for p in [Vec3::X*1.1,Vec3::Y*1.2,Vec3::NEG_Z*1.3,Vec3::X*5.0]{let(d,g)=field.sample(p);assert!((d-(p.length()-1.0)).abs()<1e-5,"{p:?}: {d}");assert!(g.normalize().distance(p.normalize())<1e-4);}
        let encoded=ron::to_string(&field).unwrap();let decoded:ClothSdf=ron::from_str(&encoded).unwrap();assert_eq!(decoded.dimensions(),field.dimensions());assert_eq!(decoded.sample(Vec3::new(0.123,0.231,0.779)),field.sample(Vec3::new(0.123,0.231,0.779)));
        let(p,mut t)=cube(Vec3::ONE);t.truncate(t.len()-3);assert!(ClothSdf::from_mesh(&p,&t,17).unwrap_err().contains("closed manifold"));
        assert!(ClothSdf::from_samples([3;3],Vec3::ZERO,Vec3::ONE,vec![-1.0;27]).is_err());
        assert!(ClothSdf::from_samples([3;3],Vec3::splat(f32::MAX),Vec3::splat(f32::MAX),vec![0.0;27]).is_err());
        // A typical imported mesh splits vertices at every face for normals/UVs.
        let(p,t)=cube(Vec3::ONE);let split=t.iter().map(|&i|p[i as usize]).collect::<Vec<_>>();let ids=(0..t.len()as u32).collect::<Vec<_>>();let split=ClothSdf::from_mesh(&split,&ids,8).unwrap();assert!(split.sample(Vec3::ZERO).0<0.0);
    }
    #[test]
    fn cloth_sdf_triangle_plane_certificate_covers_the_whole_sweep(){
        let shape=ClothSdfShape::new(field(Vec3::ONE),Mat4::IDENTITY).unwrap();let motion=ClothSdfMotion::new(shape.clone(),shape.clone()).unwrap();
        let a=[Vec3::new(-0.5,-0.5,1.02),Vec3::new(0.5,-0.5,1.02),Vec3::new(0.0,0.5,1.02)];
        assert!(motion.triangle_separated(a,a.map(|p|p+Vec3::X*0.01),0.01));
        let outside=[Vec3::new(-2.0,-2.0,1.3),Vec3::new(2.0,-2.0,1.3),Vec3::new(0.0,2.0,1.3)];
        assert!(motion.triangle_separated(outside,outside,0.01),"a safe plane extending beyond the field grid can still be certified");
        assert!(!motion.triangle_separated(outside,outside.map(|p|p-Vec3::Z*0.5),0.01));
        assert!(!motion.triangle_separated(a,a.map(|p|p-Vec3::Z*0.1),0.01),"a clear origin cannot certify a crossing endpoint");
        let b=a.map(|p|p+Vec3::Z*0.2);let translated=ClothSdfShape::new(shape.field.clone(),Mat4::from_translation(Vec3::Z*0.2)).unwrap();
        assert!(ClothSdfMotion::new(shape.clone(),translated).unwrap().triangle_separated(a,b,0.01));
        let rotated=ClothSdfShape::new(shape.field.clone(),Mat4::from_rotation_x(0.2)).unwrap();
        assert!(!ClothSdfMotion::new(shape.clone(),rotated).unwrap().triangle_separated(a,b,0.01),"an uncertain rotating path still uses the covering search");
        let rotated=ClothSdfShape::new(shape.field.clone(),Mat4::from_rotation_x(0.005)).unwrap();let rotating=ClothSdfMotion::new(shape.clone(),rotated).unwrap();
        assert!(rotating.triangle_separated(a,a,0.01),"small safe rotations can be certified with a curvature bound");
        for time in [0.0,0.25,0.5,0.75,1.0]{for i in 0..=8{for j in 0..=8-i{
            let u=i as f32/8.0;let v=j as f32/8.0;let q=a[0]*(1.0-u-v)+a[1]*u+a[2]*v;
            assert!(shape.field.sample(rotating.inverse_at(time).transform_point3(q)).0>0.01);
        }}}
        // Probe certificates near a box corner, where the grid no longer
        // represents one plane. Check dense points over the complete sweep.
        let mut accepted=0;
        for k in 0..32{
            let z=0.95+k as f32*0.008;let a=[Vec3::new(0.8,0.6,z),Vec3::new(1.2,0.7,z),Vec3::new(0.9,1.2,z)];let b=a.map(|p|p+Vec3::new(-0.03,0.02,-0.005));
            if !motion.triangle_separated(a,b,0.01){continue;}accepted+=1;
            for time in [0.0,0.25,0.5,0.75,1.0]{let p=std::array::from_fn::<_,3,_>(|i|a[i].lerp(b[i],time));
                for i in 0..=16{for j in 0..=16-i{let u=i as f32/16.0;let v=j as f32/16.0;let q=p[0]*(1.0-u-v)+p[1]*u+p[2]*v;
                    assert!(shape.field.sample(q).0>0.01,"certificate omitted an interior contact");
                }}
            }
        }
        assert!(accepted>0,"the certificate must do useful work near corners");
    }
    #[test]
    fn cloth_sdf_scaled_rotated_projection_uses_world_units(){
        let rotation=Quat::from_rotation_z(0.7);let world=Mat4::from_scale_rotation_translation(Vec3::new(2.0,0.5,3.0),rotation,Vec3::new(4.0,2.0,-1.0));let shape=ClothSdfShape::new(field(Vec3::ONE),world).unwrap();
        let p=world.transform_point3(Vec3::new(0.9,0.0,0.0));let(out,n)=shape.project(p,0.02).unwrap();let want=world.transform_point3(Vec3::X)+rotation*Vec3::X*0.02;
        assert!(out.distance(want)<1e-4,"{out:?} vs {want:?}");assert!(n.distance(rotation*Vec3::X)<1e-5);assert!(shape.project(want+n*0.01,0.02).is_none());
        let center=shape.project(world.transform_point3(Vec3::ZERO),0.01).unwrap().0;assert!(shape.sample(center).0>=0.0099);
        assert!(ClothSdfShape::new(shape.field.clone(),Mat4::from_scale(Vec3::new(0.0,1.0,1.0))).is_err());
    }
    #[test]
    fn cloth_sdf_sweeps_fast_thin_moving_and_rotating_bodies(){
        let field=field(Vec3::new(1.0,1.0,0.005));let fixed=ClothSdfShape::new(field.clone(),Mat4::IDENTITY).unwrap();let motion=ClothSdfMotion::new(fixed.clone(),fixed.clone()).unwrap();
        let hit=motion.first_contact(Vec3::Z*3.0,Vec3::NEG_Z*3.0,0.0).unwrap();assert!((hit.time-(3.0-0.005)/6.0).abs()<1e-5);assert!(hit.normal.z>0.99&&!hit.conservative);
        let(out,n,_)=motion.resolve(Vec3::Z*3.0,Vec3::NEG_Z*3.0,0.01).unwrap();assert!(out.z>=0.0149&&n.z>0.99,"fast cloth crossed thin level set: {out:?}");
        assert!(motion.resolve(Vec3::new(3.0,0.0,3.0),Vec3::new(3.0,0.0,-3.0),0.01).is_none());
        let a=ClothSdfShape::new(field.clone(),Mat4::from_translation(Vec3::NEG_Z)).unwrap();let b=ClothSdfShape::new(field.clone(),Mat4::from_translation(Vec3::Z)).unwrap();let moving=ClothSdfMotion::new(a,b).unwrap();
        let(out,_,carried)=moving.resolve(Vec3::ZERO,Vec3::ZERO,0.01).unwrap();assert!(out.z>=1.0149);assert!(carried.distance(Vec3::Z*2.0)<1e-4);
        let b=ClothSdfShape::new(field,Mat4::from_scale_rotation_translation(Vec3::new(1.5,1.0,2.0),Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),Vec3::ZERO)).unwrap();let rotating=ClothSdfMotion::new(fixed,b).unwrap();
        let p=Vec3::new(0.5,0.0,-0.5);let(out,n,_)=rotating.resolve(p,p,0.01).unwrap();assert!(out.is_finite()&&n.is_finite());assert!(out.distance(p)>0.01,"rotation swept through the stationary particle");
    }
    #[test]
    fn cloth_sdf_affine_motion_handles_shear_and_mirrored_parent_scale(){
        let field=field(Vec3::new(1.0,1.0,0.01));
        for mirrored in [false,true]{
            let mut shear=Mat4::IDENTITY;shear.y_axis.x=0.45;shear.z_axis.y=0.3;if mirrored{shear.x_axis.x = -1.0;}
            let a=Mat4::from_translation(Vec3::NEG_Z)*shear;let b=Mat4::from_translation(Vec3::Z)*shear;
            let motion=ClothSdfMotion::new(ClothSdfShape::new(field.clone(),a).unwrap(),ClothSdfShape::new(field.clone(),b).unwrap()).unwrap();
            for(t,want)in [(0.0,a),(1.0,b)]{for(got,want)in motion.at(t).to_cols_array().into_iter().zip(want.to_cols_array()){assert!((got-want).abs()<1e-5);}}
            let(out,normal,_)=motion.resolve(Vec3::ZERO,Vec3::ZERO,0.01).unwrap();assert!(out.z>=1.0199,"{out:?}");assert!(normal.z>0.99);assert!(motion.end.sample(out).0>=0.0099);
        }
    }
    #[test]
    fn cloth_sdf_runs_in_mesh_constraints_and_sweeps_segment_interiors(){
        use crate::{MeshCloth,MeshClothSim,ClothShapeWorld};
        let sdf=ClothSdfShape::new(field(Vec3::new(1.0,1.0,0.01)),Mat4::IDENTITY).unwrap();
        let p=vec![Vec3::new(-0.5,-0.5,1.0),Vec3::new(0.5,-0.5,1.0),Vec3::new(0.0,0.5,1.0)];
        let mut sim=MeshClothSim::build(&p,&[0,1,2],&[0.0;3],0.0,Mat4::IDENTITY).unwrap();
        for p in &mut sim.particles{p.vel=Vec3::NEG_Z*25.0;}
        let settings=MeshCloth{gravity:[0.0;3],damping:0.0,frequency:10.0,iterations:4,radius:0.01,..Default::default()};
        sim.step(&settings,&p,Mat4::IDENTITY,&[ClothShapeWorld::LevelSet(Arc::new(sdf.clone()))],&[],0.0,0.1);
        assert!(sim.particles.iter().all(|p|p.pos.z>=0.0199),"mesh cloth tunneled through its level set");
        let motion=ClothSdfMotion::new(sdf.clone(),sdf).unwrap();let start=[Vec3::new(-2.0,0.0,1.0),Vec3::new(2.0,0.0,1.0)];let end=start.map(|p|p-Vec3::Z*2.0);
        assert!(motion.first_contact(start[0],end[0],0.01).is_none()&&motion.first_contact(start[1],end[1],0.01).is_none());
        let(u,push,n,_)=motion.resolve_segment(start,end,[0.01;2]).expect("the segment interior crosses the field");
        assert!((end[0].lerp(end[1],u)+push).z>=0.0199&&n.z>0.9,"u={u}, push={push:?}, normal={n:?}");
    }
    #[test]
    fn cloth_sdf_components_roundtrip_and_unbaked_fields_do_not_collide(){
        use crate::{ClothCollider,ClothShape,ClothShapeWorld};
        let mut collider=ClothCollider{shape:ClothShape::LevelSet,center:[1.0,0.0,0.0],..Default::default()};
        assert!(matches!(collider.world_shape(Mat4::IDENTITY,Mat4::IDENTITY),ClothShapeWorld::Disabled));
        collider.level_set.field=Some(field(Vec3::ONE));let collider:ClothCollider=ron::from_str(&ron::to_string(&collider).unwrap()).unwrap();
        let shape=collider.world_shape(Mat4::IDENTITY,Mat4::from_scale(Vec3::splat(2.0)));let ClothShapeWorld::LevelSet(sdf)=shape else{panic!("field lost on reload")};
        assert!(sdf.sample(Vec3::X*2.0).0<0.0);assert!(sdf.sample(Vec3::X*4.1).0>0.09);
    }
    #[test]
    fn cloth_sdf_departing_one_lobe_does_not_hide_a_later_crossing(){
        let(mut p,mut t)=cube(Vec3::ONE);let(q,r)=cube(Vec3::ONE);let offset=p.len()as u32;p.extend(q.into_iter().map(|p|p+Vec3::X*4.0));t.extend(r.into_iter().map(|i|i+offset));
        let field=ClothSdf::from_mesh(&p,&t,32).unwrap();let shape=ClothSdfShape::new(field,Mat4::from_scale(Vec3::new(1.0,1.0,0.1))).unwrap();let motion=ClothSdfMotion::new(shape.clone(),shape).unwrap();
        let p0=Vec3::X*1.05;let p1=Vec3::X*6.0;let(out,n,_)=motion.resolve(p0,p1,0.01).expect("later lobe must block cloth");
        assert!(out.x<=2.991&&n.x< -0.9,"first lobe masked the later crossing: {out:?}, {n:?}");
    }
    #[test]
    fn cloth_sdf_bone_chain_sweeps_bodies_between_particles(){
        use crate::{BoneCloth,BoneClothSim,ClothShapeWorld};use crate::cloth_contact::ClothColliderMotion;
        let models=[Mat4::from_translation(Vec3::Y),Mat4::from_translation(Vec3::NEG_Y)];
        let settings=BoneCloth{roots:vec!["root".into()],gravity:[0.0;3],stiffness:[0.0;2],end_tail:false,frequency:10.0,damping:0.0,radius:[0.01;2],..Default::default()};
        let mut sim=BoneClothSim::build(&settings,&[None,Some(0)],&["root".into(),"tip".into()],&models,Mat4::IDENTITY).unwrap();
        let field=field(Vec3::splat(0.05));let shape=|z|ClothShapeWorld::LevelSet(Arc::new(ClothSdfShape::new(field.clone(),Mat4::from_translation(Vec3::Z*z)).unwrap()));
        let body=ClothColliderMotion{start:shape(-1.0),end:shape(1.0)};
        sim.step_moving(&settings,&models,Mat4::IDENTITY,&[body],&[],0.0,0.1);
        assert_eq!(sim.particles[0].pos,Vec3::Y);assert!(sim.particles[1].pos.z>1.0,"whole-segment contacts must move the free tip");
        assert!(sim.particles.iter().all(|p|p.pos.is_finite()&&p.vel.is_finite()));
    }
    #[test]
    #[ignore = "cloth level-set runtime benchmark"]
    fn cloth_sdf_contact_profile(){
        use crate::{MeshCloth,MeshClothSim,ClothShapeWorld};
        let mut positions=Vec::new();let mut indices=Vec::new();let n=24u32;
        for y in 0..=n{for x in 0..=n{positions.push(Vec3::new(x as f32/n as f32-0.5,y as f32/n as f32-0.5,0.12));}}
        for y in 0..n{for x in 0..n{let a=y*(n+1)+x;indices.extend([a,a+1,a+n+1,a+1,a+n+2,a+n+1]);}}
        let field=field(Vec3::new(2.0,2.0,0.1));let shape=ClothShapeWorld::LevelSet(Arc::new(ClothSdfShape::new(field,Mat4::IDENTITY).unwrap()));
        let settings=MeshCloth{gravity:[0.0,0.0,-9.81],frequency:90.0,iterations:8,..Default::default()};
        let mut sim=MeshClothSim::build(&positions,&indices,&vec![0.0;positions.len()],0.0,Mat4::IDENTITY).unwrap();
        for _ in 0..30{sim.step(&settings,&positions,Mat4::IDENTITY,std::slice::from_ref(&shape),&[],0.0,1.0/60.0);}
        let mut baseline=sim.clone();let baseline_start=std::time::Instant::now();for _ in 0..60{baseline.step(&settings,&positions,Mat4::IDENTITY,&[],&[],0.0,1.0/60.0);}
        eprintln!("[cloth without body contacts] {:.3} ms/step",baseline_start.elapsed().as_secs_f64()*1000.0/60.0);
        let start=std::time::Instant::now();for _ in 0..60{sim.step(&settings,&positions,Mat4::IDENTITY,std::slice::from_ref(&shape),&[],0.0,1.0/60.0);}
        eprintln!("[cloth SDF] {} particles, two substeps, eight iterations: {:.3} ms/step",sim.particles.len(),start.elapsed().as_secs_f64()*1000.0/60.0);
        assert!(sim.particles.iter().all(|p|p.pos.z>=0.1099&&p.pos.is_finite()));
    }
}
