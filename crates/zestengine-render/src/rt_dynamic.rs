//! Budget optional BLAS quality rebuilds; changed vertices always get an update.
//! Keep this budget across all views sharing a rendered frame.
use std::collections::HashSet;

pub(crate) const MIN_REFITS: u32 = 120;
pub(crate) const REBUILD_TRIANGLES: u32 = 128 * 1024;

#[derive(Clone, Copy)]
pub(crate) struct Candidate {
    pub mesh: usize,
    pub refits: u32,
    pub triangles: u32,
}

#[derive(Default)]
pub(crate) struct Rebuilds {
    candidates: Vec<Candidate>,
    selected: HashSet<usize>,
    pub triangles: u64,
    pub count: u32,
}

impl Rebuilds {
    pub fn begin_frame(&mut self) { self.triangles=0; self.count=0; self.selected.clear(); }

    pub fn select(&mut self, candidates: impl Iterator<Item=Candidate>) {
        self.selected.clear();
        self.candidates.clear();
        self.candidates.extend(candidates.filter(|c|c.refits>=MIN_REFITS));
        // Oldest first prevents a stream of small/new requests starving large
        // meshes. Draw aliases must not consume the budget multiple times.
        self.candidates.sort_unstable_by_key(|c|(std::cmp::Reverse(c.refits),c.mesh));
        self.candidates.dedup_by_key(|c|c.mesh);
        for c in &self.candidates {
            let cost=u64::from(c.triangles);
            // One oversized mesh can make progress, exclusively. All other
            // meshes continue refitting their current vertices this frame.
            if self.count>0 && self.triangles+cost>u64::from(REBUILD_TRIANGLES) { break; }
            self.selected.insert(c.mesh);
            self.triangles+=cost;
            self.count+=1;
            if self.triangles>=u64::from(REBUILD_TRIANGLES) { break; }
        }
    }

    pub fn contains(&self,mesh:usize)->bool { self.selected.contains(&mesh) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebuilds_spread_crowds_deduplicate_aliases_and_preserve_cross_view_budget() {
        let mut s=Rebuilds::default();let mut age=[MIN_REFITS;8];let mut rebuilt=[false;8];
        for _ in 0..4 {
            s.begin_frame();
            s.select((0..8).rev().flat_map(|i|[Candidate {mesh:i,refits:age[i],triangles:REBUILD_TRIANGLES/2};2]));
            assert_eq!((s.count,s.triangles),(2,u64::from(REBUILD_TRIANGLES)));
            for i in 0..8 {
                if s.contains(i) {assert!(!rebuilt[i]);rebuilt[i]=true;age[i]=0;} else {age[i]+=1;}
            }
            s.select(std::iter::once(Candidate {mesh:99,refits:999,triangles:1}));
            assert!(!s.contains(99),"second view exceeded this frame's budget");
        }
        assert!(rebuilt.into_iter().all(|v|v));
    }

    #[test]
    fn overdue_large_mesh_gets_exclusive_progress_and_fresh_meshes_do_not_rebuild() {
        let mut s=Rebuilds::default();
        let large=Candidate {mesh:8,refits:MIN_REFITS+1,triangles:REBUILD_TRIANGLES*3};
        let small=Candidate {mesh:0,refits:MIN_REFITS,triangles:1};
        s.select([small,large].into_iter());
        assert!(s.contains(8));assert!(!s.contains(0));assert_eq!(s.count,1);
        s.select(std::iter::once(small));assert!(!s.contains(0));
        s.begin_frame();s.select([Candidate {refits:0,..large},small].into_iter());
        assert!(s.contains(0));assert!(!s.contains(8));
    }
}
