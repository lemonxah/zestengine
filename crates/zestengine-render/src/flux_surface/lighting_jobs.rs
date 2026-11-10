//! Compact surface lighting into workgroup-sized tiles without changing rays.
use super::*;

pub(super) fn pack(pages:Vec<Node>,cards:&[Card],legacy:bool,spacing:u32)->(Vec<Node>,[u32;3],u32) {
    let max_groups=pages.iter().map(|p|cards[p.left as usize].rect[2]/(spacing*8)).max().unwrap_or(0);
    let old_workgroups=max_groups*max_groups*pages.len() as u32;
    if legacy {let count=pages.len() as u32;return (pages,[max_groups,max_groups,count],old_workgroups);}
    let mut jobs=Vec::new();
    for page in pages {
        let size=cards[page.left as usize].rect[2]/spacing;
        for y in (0..size).step_by(8) {for x in (0..size).step_by(8) {
            let mut job=page;job.max=[x as f32,y as f32,page.max[2]];jobs.push(job);
        }}
    }
    assert!(jobs.len()<=LIGHTING_QUEUE_CAPACITY,"surface lighting queue exceeds physical atlas capacity");
    let count=jobs.len() as u32;
    (jobs,[1,1,count],old_workgroups)
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn mixed_resolutions_cover_each_receiver_once_with_identical_history() {
        let mut cards=vec![Card::zeroed();4];
        for (c,size) in cards.iter_mut().zip([128,32,64,32]) {c.rect[2]=size;}
        let pages:Vec<_>=(0..4).map(|i|Node {left:i,right:3|(i<<2),min:[11.0,23.0,5.0],max:[0.0,0.0,7.0]}).collect();
        for spacing in [LIGHTING_DIV,MIN_LIGHTING_DIV] {
        let scale=(LIGHTING_DIV/spacing).pow(2);
        let (jobs,dispatch,old)=pack(pages.clone(),&cards,false,spacing);
        assert_eq!(dispatch,[1,1,22*scale]);assert_eq!(old,64*scale);
        let mut visited=BTreeSet::new();
        for job in jobs {
            let original=pages[job.left as usize];assert_eq!(job.right,original.right);assert_eq!(job.min,original.min);assert_eq!(job.max[2],original.max[2]);
            for y in 0..8 {for x in 0..8 {
                let x=x+job.max[0] as u32;let y=y+job.max[1] as u32;
                assert!(x<cards[job.left as usize].rect[2]/spacing && y<cards[job.left as usize].rect[2]/spacing);
                assert!(visited.insert((job.left,x,y)),"duplicate receiver job");
            }}
        }
        assert_eq!(visited.len(),(22*64*scale) as usize);
        }
        // The full physical atlas is the upper bound even at maximum update speed.
        let cards=vec![Card {rect:[0,0,128,1],..Card::zeroed()};256];
        let pages=(0..256).map(|left|Node {left,..Node::zeroed()}).collect();
        assert_eq!(pack(pages,&cards,false,MIN_LIGHTING_DIV).0.len(),LIGHTING_QUEUE_CAPACITY);
    }
}
