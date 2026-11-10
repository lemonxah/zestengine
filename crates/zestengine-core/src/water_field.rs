//! Shared mip construction for displacement/normal/foam fields.
pub(crate) fn append_mips(fields:&mut Vec<[f32;4]>,n:usize,stride:usize) {
    let(mut side,mut base)=(n,0);
    while side>1 {let next=side/2;let new_base=fields.len();for z in 0..next{for x in 0..next{for lane in 0..stride{
        let mut value=[0.0;4];for oz in 0..2{for ox in 0..2{let v=fields[base+((z*2+oz)*side+x*2+ox)*stride+lane];for c in 0..4{value[c]+=v[c]*0.25;}}}fields.push(value);
    }}}base=new_base;side=next;}
}
