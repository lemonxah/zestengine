//! Mesh cloth follows the pose owner but retains its own component, world
//! transform and render slot. Simulate only after every collider pose is ready.
use super::*;
use crate::cloth_contacts::ClothRenderWork;

pub(super) struct ClothPoseRoutes {
    pub members:HashMap<usize,Vec<usize>>,
    pub targets:HashMap<usize,HashMap<usize,(usize,usize)>>,
    pub collider_drivers:std::collections::HashSet<usize>,
}

/// Pose evaluation and render submission are separate phases: no cloth may
/// sample another object's collider until its animation/IK/overrides are ready.
pub(super) struct PendingPose {
    pub i:usize,
    pub skeleton:usize,
    pub locals:Vec<Mat4>,
    pub gpu:bool,
    pub skinned_slots:Vec<usize>,
    pub drawn:Vec<usize>,
    pub has_face:bool,
    pub morph_weights:Vec<Vec<f32>>,
    pub plan:crate::anim_cull::PosePlan,
    pub interval:u32,
    pub revealed:bool,
    pub dt:f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_assets::{Joint,Skeleton};
    use zestengine_core::{ClothCollider,ClothColliders,ClothShape,MeshCloth,ObjectRef};
    use zestengine_render::Vertex;

    fn scene()->LoadedScene {
        let mut s=LoadedScene::empty();
        s.skeletons.push(Skeleton{joints:vec![Joint{name:"root".into(),parent:None,inverse_bind:Mat4::IDENTITY,
            rest_translation:Vec3::ZERO,rest_rotation:Quat::IDENTITY,rest_scale:Vec3::ONE}],..Default::default()});
        s
    }
    fn mesh(s:&mut LoadedScene,name:&str,z:f32,pin:f32)->usize{
        let vertices=vec![[0.0,0.0,z],[1.0,0.0,z],[0.0,1.0,z]].into_iter().map(|position|Vertex{position,color:[pin;4],normal:[0.0,0.0,1.0],weights:[1.0,0.0,0.0,0.0],..Default::default()}).collect::<Vec<_>>();
        let mesh=s.mesh_geometry.len();
        s.mesh_geometry.push((vertices.iter().map(|v|Vec3::from(v.position)).collect(),vec![0,1,2]));
        s.mesh_bounds.push((Vec3::new(0.0,0.0,z),Vec3::new(1.0,1.0,z)));
        s.mesh_handles.push(MeshHandle::NONE);
        s.skinned_meshes.push(SkinnedMesh::new(mesh,vertices,0));
        let mut o=SceneObject::new_empty(ObjectId::new(),name);*o.render_mut()=Some(RenderInfo{mesh,material:0});
        s.objects.push(o);s.objects.len()-1
    }
    fn add_cloth(s:&mut LoadedScene,i:usize){
        s.objects[i].components.push(Box::new(MeshCloth{gravity:[0.0;3],thickness:0.04,..Default::default()}));
    }
    fn cloth(s:&LoadedScene,i:usize)->zestengine_core::ecs::ComponentRef<'_, MeshCloth>{s.objects[i].components.get_typed::<MeshCloth>().unwrap()}
    fn mount(s:&mut LoadedScene,avatar:usize,outfit:usize){
        let (a,b)=(s.objects[avatar].id(),s.objects[outfit].id());
        s.skin_owner.insert(b,a);s.skin_followers.entry(a).or_default().push(b);
    }

    #[test]
    fn mounted_cloth_uses_its_own_component_and_shares_contacts_without_a_renderer(){
        let mut s=scene();let avatar=mesh(&mut s,"avatar",-1.0,1.0);
        let skirt=mesh(&mut s,"skirt",0.0,1.0);let coat=mesh(&mut s,"coat",0.01,0.0);
        add_cloth(&mut s,skirt);add_cloth(&mut s,coat);mount(&mut s,avatar,skirt);mount(&mut s,avatar,coat);
        // No Animator or MeshCloth on the avatar itself: its mounted garments
        // must still make the avatar a pose driver for this pass.
        for o in &mut s.objects{o.transform_mut().translation=Vec3::new(8.0,3.0,5.0);}
        s.pose_skeletons(1.0/60.0);
        let skirt_view=cloth(&s,skirt);let skirt_sim=skirt_view.runtime.sim.as_ref().expect("mounted skirt simulated");
        let coat_view=cloth(&s,coat);let coat_sim=coat_view.runtime.sim.as_ref().expect("mounted coat simulated");
        assert!(skirt_sim.particles.iter().all(|p|(p.pos.z-5.0).abs()<1e-5));
        assert!(coat_sim.particles.iter().all(|p|p.pos.z>=5.0399),"mounted garments share contacts");
        assert!(coat_sim.particles.iter().all(|p|p.pos.x>=7.99&&p.pos.y>=2.99),"outfit world space");drop(skirt_view);drop(coat_view);
        *s.objects[coat].enabled_mut()=false;s.pose_skeletons(1.0/60.0);
        assert!(cloth(&s,coat).runtime.idle);
        s.objects[coat].transform_mut().translation.z=8.0;*s.objects[coat].enabled_mut()=true;s.pose_skeletons(1.0/60.0);
        assert!(!cloth(&s,coat).runtime.idle);
        assert!(cloth(&s,coat).runtime.sim.as_ref().unwrap().particles.iter().all(|p|(p.pos.z-8.01).abs()<1e-4),"reactivation discards stale collision history");
    }

    #[test]
    fn cloth_uses_external_collider_final_pose_even_when_body_is_scanned_later(){
        let mut s=scene();let sheet=mesh(&mut s,"sheet",0.0,0.0);add_cloth(&mut s,sheet);
        let body=mesh(&mut s,"body",-1.0,1.0);
        s.objects[body].components.push(Box::new(ClothColliders{colliders:vec![ClothCollider{shape:ClothShape::Plane,bone:"root".into(),normal:[0.0,0.0,1.0],..Default::default()}],..Default::default()}));
        let body_id=s.objects[body].id();
        s.objects[sheet].components.get_typed_mut::<MeshCloth>().unwrap().collider_objects.push(ObjectRef(Some(body_id)));
        s.pose_override=Some((body,vec![Mat4::from_translation(Vec3::Z*0.5)]));
        s.pose_skeletons(1.0/60.0);
        assert!(cloth(&s,sheet).runtime.sim.as_ref().unwrap().particles.iter().all(|p|p.pos.z>=0.5099),"use the final, current-frame body pose");
        s.pose_override=Some((body,vec![Mat4::from_translation(Vec3::Z)]));
        s.pose_skeletons(1.0/60.0);
        assert!(cloth(&s,sheet).runtime.sim.as_ref().unwrap().particles.iter().all(|p|p.pos.z>=1.0099));
    }

    #[test]
    fn mounted_colliders_follow_avatar_bones_and_extra_cloth_slots_route_correctly(){
        let mut s=scene();let avatar=mesh(&mut s,"avatar",-1.0,1.0);
        let outfit=mesh(&mut s,"outfit",0.0,0.0);let extra=mesh(&mut s,"extra",0.1,0.0);
        let extra_mesh=s.objects[extra].render().unwrap();*s.objects[extra].enabled_mut()=false;
        (*s.objects[outfit].extra_render_mut()).push(extra_mesh);add_cloth(&mut s,outfit);
        s.objects[outfit].components.get_typed_mut::<MeshCloth>().unwrap().slot=1;
        s.objects[outfit].components.push(Box::new(ClothColliders{colliders:vec![ClothCollider{shape:ClothShape::Plane,bone:"root".into(),normal:[0.0,0.0,1.0],..Default::default()}],..Default::default()}));
        mount(&mut s,avatar,outfit);
        s.pose_override=Some((avatar,vec![Mat4::from_translation(Vec3::Z*0.5)]));s.pose_skeletons(1.0/60.0);
        let c=cloth(&s,outfit);assert!(c.runtime.sim.is_some());
        assert!((c.runtime.bind[0].z-0.1).abs()<1e-6,"the selected extra slot supplies the rest mesh");
        let cc=s.objects[outfit].components.get_typed::<ClothColliders>().unwrap();
        assert!((cc.runtime.frame("root").unwrap().w_axis.z-0.5).abs()<1e-6); drop(cc);
        assert!(c.runtime.sim.as_ref().unwrap().particles.iter().all(|p|p.pos.z>=0.5999)); drop(c);
    }

    #[test]
    fn bone_cloth_reads_current_external_poses_in_both_pose_paths_and_object_orders(){
        use zestengine_core::{Animator,BoneCloth};
        for animated_cloth in [false,true]{for animated_body in [false,true]{for body_first in [false,true]{
            let mut s=scene();
            s.skeletons[0].joints.push(Joint{name:"tip".into(),parent:Some(0),inverse_bind:Mat4::from_translation(-Vec3::Y),
                rest_translation:Vec3::Y,rest_rotation:Quat::IDENTITY,rest_scale:Vec3::ONE});
            let (body,cloth)=if body_first{(mesh(&mut s,"body",-1.0,1.0),mesh(&mut s,"cloth",0.0,0.0))}else{
                let cloth=mesh(&mut s,"cloth",0.0,0.0);(mesh(&mut s,"body",-1.0,1.0),cloth)
            };
            let body_id=s.objects[body].id();
            s.objects[cloth].components.push(Box::new(BoneCloth{roots:vec!["root".into()],end_tail:false,
                stiffness:[0.0;2],gravity:[0.0;3],collider_objects:vec![ObjectRef(Some(body_id))],..Default::default()}));
            if animated_cloth{s.objects[cloth].components.push(Box::new(Animator::default()));}
            if animated_body{s.objects[body].components.push(Box::new(Animator::default()));}
            s.objects[body].components.push(Box::new(ClothColliders{colliders:vec![ClothCollider{shape:ClothShape::Plane,bone:"root".into(),normal:[0.0,0.0,1.0],..Default::default()}],..Default::default()}));
            for height in [0.2,0.4]{
                s.pose_override=Some((body,vec![Mat4::from_translation(Vec3::Z*height),Mat4::from_translation(Vec3::Y)]));
                s.pose_skeletons(1.0/60.0);
                let c=s.objects[cloth].components.get_typed::<BoneCloth>().unwrap();
                let tip=&c.runtime.sim.as_ref().expect("bone cloth simulated").particles[1];
                assert!(tip.pos.z>=height+0.019,"current pose: animated cloth={animated_cloth}, body={animated_body}, body first={body_first}, height={height}, tip={:?}",tip.pos);
                assert!(c.runtime.collider_history.values().any(|shape|matches!(shape,zestengine_core::ClothShapeWorld::Plane{p,..} if (p.z-height).abs()<1e-5)),"history stores the current pose"); drop(c);
            }
        }}}
    }

    #[test]
    fn mounted_collider_dependencies_keep_the_avatar_pose_live(){
        let mut s=scene();let avatar=mesh(&mut s,"avatar",-1.0,1.0);
        let body=mesh(&mut s,"body proxy",-1.0,1.0);mount(&mut s,avatar,body);
        let sheet=mesh(&mut s,"cloth",0.0,0.0);add_cloth(&mut s,sheet);
        let id=s.objects[body].id();
        s.objects[sheet].components.get_typed_mut::<MeshCloth>().unwrap().collider_objects.push(ObjectRef(Some(id)));
        s.objects[body].components.push(Box::new(ClothColliders{colliders:vec![ClothCollider{bone:"root".into(),..Default::default()}],..Default::default()}));
        let routes=s.cloth_pose_routes(&s.active_flags(),&HashMap::new());
        assert!(routes.collider_drivers.contains(&avatar));
        assert!(!routes.collider_drivers.contains(&body));
        *s.objects[body].enabled_mut()=false;
        assert!(!s.cloth_pose_routes(&s.active_flags(),&HashMap::new()).collider_drivers.contains(&avatar));
    }

    #[test]
    fn mounted_colliders_on_a_shared_avatar_feed_other_cloth(){
        let mut s=scene();let sheet=mesh(&mut s,"cloth",0.0,0.0);add_cloth(&mut s,sheet);
        let avatar=mesh(&mut s,"avatar",-1.0,1.0);let proxy=mesh(&mut s,"mounted proxy",-1.0,1.0);mount(&mut s,avatar,proxy);
        let id=s.objects[proxy].id();
        s.objects[sheet].components.get_typed_mut::<MeshCloth>().unwrap().collider_objects.push(ObjectRef(Some(id)));
        s.objects[proxy].components.push(Box::new(ClothColliders{colliders:vec![ClothCollider{bone:"root".into(),shape:ClothShape::Plane,normal:[0.0,0.0,1.0],..Default::default()}],..Default::default()}));
        s.pose_override=Some((avatar,vec![Mat4::from_translation(Vec3::Z*0.3)]));s.pose_skeletons(1.0/60.0);
        assert!(cloth(&s,sheet).runtime.sim.as_ref().unwrap().particles.iter().all(|p|p.pos.z>=0.3099));
    }

    #[test]
    fn bone_cloth_samples_external_animation_once_per_frame(){
        use zestengine_assets::{AnimChannel,AnimationClip,ChannelPath};
        use zestengine_core::{Animator,BoneCloth};
        for animated_body in [false,true]{
            let mut s=scene();
            s.skeletons.push(s.skeletons[0].clone());
            s.skeletons[0].joints.push(Joint{name:"tip".into(),parent:Some(0),inverse_bind:Mat4::from_translation(-Vec3::Y),
                rest_translation:Vec3::Y,rest_rotation:Quat::IDENTITY,rest_scale:Vec3::ONE});
            let cloth=mesh(&mut s,"cloth",0.0,0.0);let body=mesh(&mut s,"animated body",-1.0,1.0);
            s.skinned_meshes[1].skeleton=1;
            s.animations.push(AnimationClip{name:"push".into(),duration:1.0,channels:vec![AnimChannel{
                joint:0,path:ChannelPath::Translation,times:vec![0.0,1.0],vec_values:vec![Vec3::ZERO,Vec3::Z*4.0],..Default::default()
            }],morph_tracks:vec![],playback:None});s.clip_skeleton.push(1);
            let id=s.objects[body].id();
            s.objects[cloth].components.push(Box::new(BoneCloth{roots:vec!["root".into()],end_tail:false,
                stiffness:[0.0;2],gravity:[0.0;3],collider_objects:vec![ObjectRef(Some(id))],..Default::default()}));
            s.objects[body].components.push(Box::new(ClothColliders{colliders:vec![ClothCollider{shape:ClothShape::Plane,bone:"root".into(),normal:[0.0,0.0,1.0],..Default::default()}],..Default::default()}));
            if animated_body{s.objects[body].components.push(Box::new(Animator::default()));}
            for frame in 1..=3 {
                s.pose_skeletons(0.05);let height=frame as f32*0.2;
                let c=s.objects[cloth].components.get_typed::<BoneCloth>().unwrap();
                let particles=&c.runtime.sim.as_ref().unwrap().particles;
                assert!(particles[0].pos.length()<1e-6,"a clip belonging to another skeleton cannot animate the cloth root");
                assert!(particles[1].pos.z>=height+0.019,"animated body pushes cloth on the same frame"); drop(c);
                let cc=s.objects[body].components.get_typed::<ClothColliders>().unwrap();
                assert!((cc.runtime.frame("root").unwrap().w_axis.z-height).abs()<1e-5,"body animation advances exactly once"); drop(cc);
            }
        }
    }
}

impl LoadedScene {
    pub(super) fn refresh_cloth_pose_members(&mut self,driver:usize,skeleton:usize,locals:&[Mat4],routes:&ClothPoseRoutes,skinned:&HashMap<usize,usize>){
        let skel=&self.skeletons[skeleton];
        let models=skel.world_matrices(locals);
        if let Some(members)=routes.members.get(&driver){
            for &object in members {
                let verts:Vec<_>=self.objects[object].render_slots().filter_map(|r|skinned.get(&r.mesh))
                    .map(|&k|self.skinned_meshes[k].base_vertices.as_slice()).collect();
                crate::cloth::refresh_colliders(&mut self.objects[object].components,skel,&models,&verts);
            }
        }
    }

    /// Shared skeletons also supply colliders to animated objects, so their
    /// non-cloth pose belongs in the first phase, before any body snapshot.
    pub(super) fn prepare_shared_cloth_poses(&mut self,users:&std::collections::HashSet<usize>,owners:&HashMap<usize,usize>,routes:&ClothPoseRoutes,skinned:&HashMap<usize,usize>,dt:f32)->(bool,HashMap<usize,Vec<Mat4>>){
        let ik=self.ik_targets;
        let ik_active=ik.is_some_and(|t|t.head.is_some()||t.left_hand.is_some()||t.right_hand.is_some()||t.hips.is_some()||t.left_foot.is_some()||t.right_foot.is_some());
        let mut poses=HashMap::new();
        if !ik_active&&self.animations.is_empty()&&owners.is_empty(){return (ik_active,poses);}
        self.anim_time+=dt;
        for sm in &self.skinned_meshes {
            if !users.contains(&sm.mesh_index){continue;}
            let owner=owners.get(&sm.mesh_index).copied();
            if !ik_active&&self.animations.is_empty()&&owner.is_none(){continue;}
            let Some(skel)=self.skeletons.get(sm.skeleton)else{continue};
            let clip=(0..self.animations.len()).find(|&c|self.clip_skeleton.get(c)==Some(&sm.skeleton)).and_then(|c|self.animations.get(c));
            let mut locals=match ik {
                Some(targets) if ik_active&&crate::humanoid::HumanoidRig::resolve(skel).is_humanoid()=>crate::humanoid::pose_from_trackers(skel,&targets),
                _=>match clip{Some(clip)=>clip.sample(skel,self.anim_time),None=>skel.rest_locals()},
            };
            if let Some(params)=self.foot_ik&&crate::humanoid::HumanoidRig::resolve(skel).is_humanoid(){
                let max_lift=params.max_lift;
                crate::humanoid::apply_foot_ik(skel,&mut locals,&params,|p|self.ground_height(p.x,p.z,p.y+max_lift));
            }
            if let Some((o,forced))=&self.pose_override&&forced.len()==locals.len()
                &&self.objects.get(*o).is_some_and(|ob|ob.render_slots().any(|r|r.mesh==sm.mesh_index)){
                locals.clone_from(forced);
            }
            poses.insert(sm.mesh_index,locals);
        }
        for (&mesh,&owner) in owners {
            if let Some(locals)=poses.get(&mesh)&&let Some(&index)=skinned.get(&mesh){
                self.refresh_cloth_pose_members(owner,self.skinned_meshes[index].skeleton,locals,routes,skinned);
            }
        }
        (ik_active,poses)
    }

    pub(super) fn cloth_pose_routes(&self,active:&[bool],followers:&HashMap<usize,usize>)->ClothPoseRoutes {
        let by_id:HashMap<_,_>=self.objects.iter().enumerate().map(|(i,o)|(o.id(),i)).collect();
        let mut sources=std::collections::HashSet::new();
        for (i,o) in self.objects.iter().enumerate().filter(|(i,_)|active[*i]){
            for component in &o.components {
                let references=if let Some(c)=component.as_any().downcast_ref::<zestengine_core::BoneCloth>(){
                    c.collide.then_some(&c.collider_objects)
                }else if let Some(c)=component.as_any().downcast_ref::<zestengine_core::MeshCloth>(){
                    c.collide.then_some(&c.collider_objects)
                }else{None};
                if let Some(references)=references{
                    sources.insert(i);
                    sources.extend(references.iter().filter_map(|r|r.id()).filter_map(|id|by_id.get(&id).copied()));
                }
            }
        }
        let mut routes=ClothPoseRoutes{members:HashMap::new(),targets:HashMap::new(),collider_drivers:Default::default()};
        for (object,o) in self.objects.iter().enumerate().filter(|(i,_)|active[*i]){
            let driver=self.skin_owner.get(&o.id()).and_then(|id|by_id.get(id)).copied()
                .or_else(||followers.get(&object).copied()).unwrap_or(object);
            if !active[driver]{continue;}
            if sources.contains(&object)&&o.components.iter().any(|c|c.as_any().downcast_ref::<zestengine_core::ClothColliders>()
                .is_some_and(|c|c.colliders.iter().any(|c|c.enabled&&!c.bone.is_empty()))){
                routes.collider_drivers.insert(driver);
            }
            routes.members.entry(driver).or_default().push(object);
            for (slot,r) in o.render_slots().enumerate(){
                if crate::cloth::has_mesh_cloth_slot(&o.components,slot){
                    routes.targets.entry(driver).or_default().insert(r.mesh,(object,slot));
                }
            }
        }
        routes
    }

    pub(super) fn simulate_queued_cloth(&mut self,work:&[ClothRenderWork],active:&[bool],world:&[Mat4],winds:&[zestengine_core::WindSample],viewer:Option<Vec3>,dt:f32){
        let shapes=crate::cloth::shapes_by_object(self.objects.iter().enumerate().filter(|(i,_)|active[*i]).map(|(i,o)|(o.id(),&o.components,world[i])));
        let queued:std::collections::HashSet<_>=work.iter().map(|w|(w.object,w.slot)).collect();
        for (i,o) in self.objects.iter_mut().enumerate(){
            for mut c in &mut o.components{
                if let Some(c)=c.as_any_mut().downcast_mut::<zestengine_core::MeshCloth>(){
                    if !queued.contains(&(i,c.slot)){c.runtime.idle=true;}
                }
            }
        }
        for w in work {
            let mut vertices=w.animated.clone();
            if let Some(sm)=self.skinned_meshes.iter().find(|sm|sm.mesh_index==w.mesh){
                crate::cloth::step_mesh(&mut self.objects[w.object].components,w.slot,&self.skeletons[sm.skeleton],&sm.base_vertices,
                    &mut vertices,&self.mesh_geometry[w.mesh].1,w.world,&shapes,winds,viewer,self.cloth_time,dt);
            }else if let Some(data)=self.static_cloth_source(w.mesh){
                crate::cloth::step_mesh(&mut self.objects[w.object].components,w.slot,&zestengine_assets::Skeleton::default(),&data.vertices,
                    &mut vertices,&data.indices,w.world,&shapes,winds,viewer,self.cloth_time,dt);
            }
        }
    }
}
