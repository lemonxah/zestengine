//! Translational degrees of freedom for collision vertices attached to a
//! reduced simulation mesh. A contact must move every vertex of its driver.
use super::*;

#[derive(Clone,Debug)]
pub struct ClothContactMapping{groups:Vec<Vec<usize>>,owner:Vec<usize>}
impl ClothContactMapping {
    pub fn new(drivers:&[usize])->Self{
        let mut ids=std::collections::HashMap::new();let mut groups:Vec<Vec<usize>>=Vec::new();let mut owner=Vec::with_capacity(drivers.len());
        for (v,&driver) in drivers.iter().enumerate(){let next=ids.len();let group=*ids.entry(driver).or_insert(next);if group==groups.len(){groups.push(Vec::new());}groups[group].push(v);owner.push(group);}
        Self{groups,owner}
    }
    pub(super) fn trivial(&self)->bool{self.groups.len()==self.owner.len()}
    pub(super) fn valid_weights(&self,weights:&[f32])->bool{
        self.owner.len()==weights.len()&&self.groups.iter().all(|g|g.iter().all(|&i|weights[i]==weights[g[0]]))
    }
    pub(super) fn valid(&self,weights:&[f32],owners:&[usize])->bool{
        self.owner.len()==weights.len()&&owners.len()==weights.len()&&self.groups.iter().all(|g|g.iter().all(|&i|weights[i]==weights[g[0]]&&owners[i]==owners[g[0]]))
    }
    pub(super) fn project(&self,p:&mut[Vec3],w:&[f32],ids:[usize;4],contact:ClothFeatureContact,thickness:f32)->f32{
        let n=contact.sweep.normal;let gap=(0..4).map(|k|p[ids[k]]*contact.coefficients[k]).sum::<Vec3>().dot(n);
        if gap>=thickness{return 0.0;}
        let roots=ids.map(|i|self.owner[i]);let mut coefficients=[0.0;4];
        for k in 0..4{let first=(0..=k).find(|&j|roots[j]==roots[k]).unwrap();coefficients[first]+=contact.coefficients[k];}
        let denominator=(0..4).map(|k|w[ids[k]].max(0.0)*coefficients[k]*coefficients[k]).sum::<f32>();
        if denominator<1e-20{return 0.0;}
        for k in 0..4{if coefficients[k]==0.0{continue;}
            let delta=n*((thickness-gap)*w[ids[k]].max(0.0)*coefficients[k]/denominator);
            for &i in &self.groups[roots[k]]{p[i]+=delta;}
        }
        thickness-gap
    }
    /// Geometric recovery and fixed-face cleanup can propose individual vertex
    /// motion. Fit it back to the available translations without deforming a
    /// cluster. Subsequent mapped contact passes resolve remaining constraints.
    pub(super) fn cohere(&self,before:&[Vec3],after:&mut[Vec3]){
        for group in &self.groups{
            let delta=(group.iter().map(|&i|(after[i]-before[i]).as_dvec3()).sum::<glam::DVec3>()/group.len()as f64).as_vec3();
            for &i in group{after[i]=before[i]+delta;}
        }
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    fn layers()->Vec<Vec3>{[0.0,0.01].into_iter().flat_map(|z|[Vec3::new(0.0,0.0,z),Vec3::new(1.0,0.0,z),Vec3::new(0.0,1.0,z)]).collect()}
    #[test]
    fn mapped_contacts_keep_clusters_rigid_and_use_driver_mass_once(){
        let topology=ClothContactTopology::new(vec![[0,1,2],[3,4,5]]);let map=ClothContactMapping::new(&[17,17,17,usize::MAX,usize::MAX,usize::MAX]);
        for pinned in [false,true]{
            let mut p=layers();let before=p.clone();let weights=[if pinned{0.0}else{1.0},if pinned{0.0}else{1.0},if pinned{0.0}else{1.0},2.0,2.0,2.0];
            let center=|p:&[Vec3]|(p[..3].iter().copied().sum::<Vec3>()+p[3..].iter().copied().sum::<Vec3>()*0.5)/3.0;
            let mass_center=center(&p);
            let report=topology.solve_between_mapped(&mut p,&before,&weights,&[0,0,0,1,1,1],&[0.01;6],&map,true,true);
            assert!(report.vertex_face+report.edge_edge>0);assert!(p[3].z-p[0].z>=0.01999);
            for base in [0,3]{for i in 1..3{assert!((p[base+i]-p[base]).distance(before[base+i]-before[base])<1e-6,"contacts cannot deform a rigid simulation cluster");}}
            if pinned{assert_eq!(&p[..3],&before[..3]);}else{assert!(center(&p).distance(mass_center)<1e-6,"driver masses must not be counted separately for every face vertex");}
        }
    }
    #[test]
    fn incompatible_driver_layouts_do_not_partially_change_geometry(){
        let topology=ClothContactTopology::new(vec![[0,1,2],[3,4,5]]);let mut p=layers();let before=p.clone();
        for drivers in [vec![0],vec![0;6]]{
            let map=ClothContactMapping::new(&drivers);
            assert_eq!(topology.solve_between_mapped(&mut p,&before,&[1.0;6],&[0,0,0,1,1,1],&[0.01;6],&map,true,true).vertex_face,0);assert_eq!(p,before);
        }
        let map=ClothContactMapping::new(&[0,0,0,1,1,1]);
        assert_eq!(topology.solve_between_mapped(&mut p,&before,&[1.0,2.0,1.0,1.0,1.0,1.0],&[0,0,0,1,1,1],&[0.01;6],&map,true,true).vertex_face,0);assert_eq!(p,before);
    }
}
