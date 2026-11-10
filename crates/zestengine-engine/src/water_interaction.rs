//! Convert posed humanoid limbs to the same water inputs used by physics.
//! Works with imported/overridden bone mappings and with animation or VR IK.
use std::collections::{HashMap, HashSet};
use std::hash::{Hash,Hasher};
use glam::{Mat4, Vec2, Vec3};
use zestengine_core::{ObjectId, SkeletalWaterInteraction, Water, WaterDisplacer, WaterPush, WaterSplash};
use crate::{physics::{WaterStep, Wake}, scene::LoadedScene};

#[derive(Clone, Copy)]
struct Sample { center: Vec3, velocity: Vec3, wet: bool, geometry:u64 }
#[derive(Default)]
pub struct WaterInteractionSim { previous: HashMap<(ObjectId, ObjectId, usize), Sample> }

impl LoadedScene {
    /// Queue an impact through the same ripple and spray paths as collisions.
    /// Returns the receiving water ID, or None when the impact is dry.
    pub fn impact_water(&mut self,impact:zestengine_core::WaterImpact,time:f32)->Option<ObjectId> {
        if !impact.position.is_finite() || !impact.velocity.is_finite() || !impact.radius.is_finite() || impact.radius<=0.0 || !impact.strength.is_finite() || impact.strength<=0.0{return None;}
        let world=self.world_transforms();let radius=impact.radius.clamp(0.005,20.0);
        let selected=self.objects.iter().enumerate().filter(|(i,o)|self.is_active(*i)&&impact.water.id().is_none_or(|id|id==o.id())).filter_map(|(i,o)|{
            let w=o.components.get_typed::<Water>()?;
            if !w.interactive{return None;}
            let(h,_)=w.height_at(world[i],impact.position,time)?;
            ((impact.position.y-h).abs()<=radius*2.0).then_some((i,h,o.id()))
        }).max_by(|a,b|a.1.total_cmp(&b.1).then_with(||b.2.raw().cmp(&a.2.raw())))?;
        let(i,height,id)=selected;let inverse=world[i].inverse();let local=inverse.transform_point3(impact.position);
        let velocity=inverse.transform_vector3(impact.velocity);let local_radius=radius*inverse.to_scale_rotation_translation().0.abs().max_element();
        self.water_fx_pushes.push((i,WaterPush{at:Vec2::new(local.x,local.z),radius:local_radius,strength:impact.strength.clamp(0.0,32.0)*(-velocity.y).max(0.1)}));
        self.water_splashes.push(WaterSplash{at:impact.position.with_y(height),velocity:impact.velocity,size:radius});Some(id)
    }
}

impl WaterInteractionSim {
    pub fn step(&mut self, scene: &LoadedScene, world: &[Mat4], t: f32, dt: f32, out: &mut WaterStep) {
        if !dt.is_finite() || dt <= 0.0 { return; }
        let mut alive = HashSet::new();
        let mut waters:Vec<_>=scene.objects.iter().enumerate().filter_map(|(i,o)|{
            let w=o.components.get_typed::<Water>()?;
            (scene.is_active(i)&&w.interactive).then_some((i,o,w))
        }).collect();
        waters.sort_by_key(|(_,o,w)|std::cmp::Reverse((w.ownership_key(),std::cmp::Reverse(o.id().raw()))));
        for (i, object) in scene.objects.iter().enumerate() {
            if !scene.is_active(i) { continue; }
            let scale=world[i].to_scale_rotation_translation().0.abs().max_element();
            let mut segments=Vec::new();
            for (component_index,component) in object.components.iter().enumerate().filter(|(_,c)|zestengine_core::component_enabled(c.as_ref())) {
                if let Some(proxy)=component.as_any().downcast_ref::<zestengine_core::WaterInteractionProxy>() {
                    let mut h=std::collections::hash_map::DefaultHasher::new();
                    for v in proxy.start.into_iter().chain(proxy.end).chain([proxy.radius]){v.to_bits().hash(&mut h);}
                    segments.push((32+component_index,world[i].transform_point3(proxy.start.into()),world[i].transform_point3(proxy.end.into()),proxy.radius,proxy.momentum,h.finish()));
                }
                if let Some(settings)=component.as_any().downcast_ref::<SkeletalWaterInteraction>() {
                    let Some((skeleton,rig))=scene.humanoid_of(i)else{continue;};
                    let Some((_,joints))=scene.bone_poses.get(&object.id())else{continue;};
                    if joints.len()!=skeleton.joints.len(){continue;}
            let limbs = [
                (settings.hands, rig.l_hand, rig.l_hand, settings.hand_radius),
                (settings.hands, rig.r_hand, rig.r_hand, settings.hand_radius),
                (settings.feet, rig.l_foot, rig.l_foot, settings.foot_radius),
                (settings.feet, rig.r_foot, rig.r_foot, settings.foot_radius),
                (settings.arms, rig.l_upper_arm, rig.l_lower_arm, settings.arm_radius),
                (settings.arms, rig.l_lower_arm, rig.l_hand, settings.arm_radius),
                (settings.arms, rig.r_upper_arm, rig.r_lower_arm, settings.arm_radius),
                (settings.arms, rig.r_lower_arm, rig.r_hand, settings.arm_radius),
                (settings.legs, rig.l_upper_leg, rig.l_lower_leg, settings.leg_radius),
                (settings.legs, rig.l_lower_leg, rig.l_foot, settings.leg_radius),
                (settings.legs, rig.r_upper_leg, rig.r_lower_leg, settings.leg_radius),
                (settings.legs, rig.r_lower_leg, rig.r_foot, settings.leg_radius),
            ];
                    for(slot,(enabled,a,b,radius))in limbs.into_iter().enumerate(){
                        if let(true,Some(a),Some(b))=(enabled,a,b){
                            let mut h=std::collections::hash_map::DefaultHasher::new();
                            for r in object.render_slots(){r.mesh.hash(&mut h);}(a,b).hash(&mut h);
                            for index in [a,b]{let joint=&skeleton.joints[index];joint.name.hash(&mut h);
                                for v in joint.inverse_bind.to_cols_array(){v.to_bits().hash(&mut h);}}
                            segments.push((slot,world[i].transform_point3(joints[a].w_axis.truncate()),world[i].transform_point3(joints[b].w_axis.truncate()),radius,settings.momentum,h.finish()));
                        }
                    }
                }
            }
            for(slot,a,b,radius,momentum,geometry)in segments {
                if !radius.is_finite() || radius<=0.0{continue;}
                let radius=(radius*scale).clamp(0.005,2.0);
                let center = (a + b) * 0.5;
                if !center.is_finite() { continue; }
                // A conservative rounded projected capsule. Both endpoints
                // matter when a leg/arm lies nearly horizontal in the water.
                let lo = a.min(b) - Vec3::splat(radius);
                let hi = a.max(b) + Vec3::splat(radius);
                for &(wi,water_object,ref water) in &waters {
                    if !crate::physics::layers_meet(scene,i,wi){continue;}
                    let key = (object.id(), water_object.id(), slot);
                    alive.insert(key);
                    let old = self.previous.get(&key).copied().filter(|s|s.geometry==geometry);
                    let velocity = old.map_or(Vec3::ZERO, |s| {
                        let delta = center - s.center;
                        if delta.length() > 2.0 { Vec3::ZERO } else {
                            s.velocity.lerp((delta / dt).clamp_length_max(20.0), 1.0 - (-20.0 * dt).exp())
                        }
                    });
                    let inv = world[wi].inverse();
                    let local = inv.transform_point3(center);
                    let xz = Vec2::new(local.x, local.z);
                    let mut wet = false;
                    if water.excluded(local) {
                        self.previous.insert(key, Sample { center, velocity, wet: false, geometry });
                        continue;
                    }
                    if let Some(column) = water.column(xz) {
                        let surface = column.surface + water.surface(xz, t).0;
                        let corners = std::array::from_fn(|k| inv.transform_point3(Vec3::new(
                            if k & 1 == 0 { lo.x } else { hi.x }, if k & 2 == 0 { lo.y } else { hi.y }, if k & 4 == 0 { lo.z } else { hi.z })));
                        let flow = inv.transform_vector3(velocity) - column.flow;
                        let axis = inv.transform_vector3(b - a).normalize_or(Vec3::X);
                        if let Some(mut displacer) = WaterDisplacer::from_corners(&corners, axis, surface, column.depth, Vec2::new(flow.x, flow.z), true) {
                            wet = true;
                            displacer.momentum = if momentum.is_finite() { momentum.clamp(0.0, 4.0) } else { 1.0 };
                            out.displacers.push((wi, displacer));
                            let local_radius = radius * inv.to_scale_rotation_translation().0.abs().max_element();
                            out.pushes.push((wi, WaterPush { at: xz, radius: local_radius, strength: -flow.y * 0.5 + Vec2::new(flow.x, flow.z).length() * 0.08 }));
                            let at = world[wi].transform_point3(Vec3::new(local.x, surface, local.z));
                            // Spawn/teleport into water seeds occupancy without a fake impact.
                            if old.is_some_and(|s| !s.wet && center.distance(s.center) <= 2.0) && velocity.y < -1.0 {
                                out.splashes.push(WaterSplash { at, velocity, size: radius });
                            }
                            if velocity.with_y(0.0).length() > 0.3 && (center.y - at.y).abs() < radius + (b.y - a.y).abs() * 0.5 {
                                out.wakes.push(Wake { water: wi, water_world: world[wi], at, flow: world[wi].transform_vector3(column.flow) - velocity, radius });
                            }
                        }
                    }
                    self.previous.insert(key, Sample { center, velocity, wet, geometry });
                    if wet{break;}
                }
            }
        }
        self.previous.retain(|key, _| alive.contains(key));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneObject;
    use zestengine_assets::{Joint, Skeleton};
    use zestengine_core::WaterBody;

    #[test]
    fn generic_proxy_and_scripted_impacts_share_water_without_rigid_bodies(){
        let mut scene=setup();scene.objects[1].components.clear();scene.objects[1].components.push(Box::new(zestengine_core::WaterInteractionProxy{start:[-0.5,0.0,0.0],end:[0.5,0.0,0.0],radius:0.1,..Default::default()}));
        let mut sim=WaterInteractionSim::default();let mut out=WaterStep::default();sim.step(&scene,&scene.world_transforms(),0.0,0.1,&mut out);
        assert_eq!(out.displacers.len(),1);assert!(out.splashes.is_empty());
        let id=scene.objects[0].id();
        assert_eq!(scene.impact_water(zestengine_core::WaterImpact::default(),0.0),Some(id));assert_eq!(scene.water_fx_pushes.len(),1);assert_eq!(scene.water_splashes.len(),1);
        assert!(scene.impact_water(zestengine_core::WaterImpact{position:Vec3::new(30.0,0.0,0.0),..Default::default()},0.0).is_none());
        *scene.objects[1].enabled_mut()=false;let mut out=WaterStep::default();sim.step(&scene,&scene.world_transforms(),0.1,0.1,&mut out);assert!(out.displacers.is_empty() && sim.previous.is_empty());
    }

    fn setup() -> LoadedScene {
        let mut scene = LoadedScene::empty();
        let mut water = SceneObject::new_empty(ObjectId::new(), "Pool");
        water.components.push(Box::new(Water { body: WaterBody::Pool, size: [4.0; 2], waves: vec![], ..Default::default() }));
        let mut avatar = SceneObject::new_empty(ObjectId::new(), "Custom avatar");
        avatar.components.push(Box::new(SkeletalWaterInteraction { hands: false, arms: false, legs: false, ..Default::default() }));
        let joints = [("LeftFoot", Vec3::new(-0.3, 0.02, 0.0)), ("RightFoot", Vec3::new(0.3, 0.02, 0.0))].into_iter().map(|(name, rest_translation)| Joint {
            name: name.into(), parent: None, inverse_bind: Mat4::IDENTITY, rest_translation,
            rest_rotation: glam::Quat::IDENTITY, rest_scale: Vec3::ONE,
        }).collect();
        let skeleton = Skeleton { joints, ..Default::default() };
        let pose = skeleton.world_matrices(&skeleton.rest_locals());
        scene.objects.extend([water, avatar]);
        scene.add_test_skeleton(1, skeleton);
        scene.bone_poses.insert(scene.objects[1].id(), (0, pose));
        scene
    }

    #[test]
    fn replacing_avatar_remaps_limbs_without_inheriting_the_old_avatars_velocity(){
        let mut scene=setup();let id=scene.objects[1].id();let mut sim=WaterInteractionSim::default();
        for joint in &mut scene.bone_poses.get_mut(&id).unwrap().1{joint.w_axis.y=0.5;}
        sim.step(&scene,&scene.world_transforms(),0.0,0.1,&mut WaterStep::default());
        let (skeleton,_)=scene.humanoid_of(1).unwrap();let mut replacement=skeleton.clone();
        replacement.joints.reverse();for j in &mut replacement.joints{j.rest_translation.y=0.0;j.rest_translation.x*=1.5;}
        let pose=replacement.world_matrices(&replacement.rest_locals());scene.add_test_skeleton(1,replacement);scene.bone_poses.insert(id,(1,pose));
        let mut out=WaterStep::default();sim.step(&scene,&scene.world_transforms(),0.1,0.1,&mut out);
        assert_eq!(out.displacers.len(),2);assert!(out.splashes.is_empty(),"wearing a new avatar is not an impact");
        assert!(out.displacers.iter().all(|(_,d)|d.velocity.length()<1e-6));
        assert!((out.displacers[0].1.center.x+0.45).abs()<0.001);assert!((out.displacers[1].1.center.x-0.45).abs()<0.001);
        scene.bone_poses.get_mut(&id).unwrap().1[1].w_axis.x-=0.15;
        let mut out=WaterStep::default();sim.step(&scene,&scene.world_transforms(),0.2,0.1,&mut out);
        assert!(out.displacers[0].1.velocity.length()>0.1,"new rig resumes normal interactions");
    }

    #[test]
    fn animated_feet_make_separate_footprints_without_rigid_bodies() {
        let mut scene = setup();
        let mut sim = WaterInteractionSim::default();
        let mut out = WaterStep::default();
        sim.step(&scene, &scene.world_transforms(), 0.0, 1.0 / 60.0, &mut out);
        assert_eq!(out.displacers.len(), 2);
        assert!(out.displacers[0].1.center.distance(out.displacers[1].1.center) > 0.5);
        assert!(out.splashes.is_empty(), "spawning is not an impact");
        // Lift only the left foot: the right footprint remains at rest.
        scene.bone_poses.get_mut(&scene.objects[1].id()).unwrap().1[0].w_axis.y = 0.5;
        let mut out = WaterStep::default();
        sim.step(&scene, &scene.world_transforms(), 0.1, 0.1, &mut out);
        assert_eq!(out.displacers.len(), 1);
        assert!(out.displacers[0].1.velocity.length() < 1e-6);
        *scene.objects[1].enabled_mut() = false;
        let mut out = WaterStep::default();
        sim.step(&scene, &scene.world_transforms(), 0.2, 0.1, &mut out);
        assert!(out.displacers.is_empty() && sim.previous.is_empty());
    }

    #[test]
    fn bone_pose_after_ik_drives_impacts_and_teleports_do_not() {
        let mut scene = setup();
        let id = scene.objects[1].id();
        for joint in &mut scene.bone_poses.get_mut(&id).unwrap().1 { joint.w_axis.y = 0.5; }
        let mut sim = WaterInteractionSim::default();
        sim.step(&scene, &scene.world_transforms(), 0.0, 0.1, &mut WaterStep::default());
        scene.bone_poses.get_mut(&id).unwrap().1[0].w_axis.y = 0.0;
        let mut out = WaterStep::default();
        sim.step(&scene, &scene.world_transforms(), 0.1, 0.1, &mut out);
        assert_eq!(out.splashes.len(), 1);
        assert!(out.splashes[0].at.x < 0.0);
        scene.bone_poses.get_mut(&id).unwrap().1[1].w_axis.y = 5.0;
        sim.step(&scene, &scene.world_transforms(), 0.2, 0.1, &mut WaterStep::default());
        scene.bone_poses.get_mut(&id).unwrap().1[1].w_axis.y = 0.0;
        let mut out = WaterStep::default();
        sim.step(&scene, &scene.world_transforms(), 0.3, 0.1, &mut out);
        assert!(out.splashes.is_empty());
    }
}
