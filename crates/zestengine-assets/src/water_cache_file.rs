//! Versioned, bounded, losslessly compressed `.watercache` recordings.
use std::{io::{Read,Write},path::Path};
use anyhow::{Result,ensure};
use zestengine_core::{WaterSimulationCache,water_cache::WATER_CACHE_MAX_BYTES};
pub const WATER_CACHE_EXTENSION:&str="watercache";
const MAGIC:&[u8;8]=b"ZWATER01";
pub fn save_water_cache(path:impl AsRef<Path>,cache:&WaterSimulationCache)->Result<()> {
    cache.validate().map_err(anyhow::Error::msg)?;let path=path.as_ref();
    if let Some(parent)=path.parent(){std::fs::create_dir_all(parent)?;}
    let temp=path.with_extension(format!("watercache.{}.tmp",zestengine_core::ObjectId::new()));
    let result=(||->Result<()>{
        let mut out=zstd::stream::write::Encoder::new(std::fs::File::create(&temp)?,3)?;
        out.write_all(MAGIC)?;out.write_all(&cache.n.to_le_bytes())?;out.write_all(&(cache.frames.len()as u32).to_le_bytes())?;
        for value in [cache.origin[0],cache.origin[1],cache.cell,cache.frame_rate]{out.write_all(&value.to_le_bytes())?;}
        for frame in &cache.frames {for sample in frame{for value in sample{out.write_all(&value.to_le_bytes())?;}}}
        out.finish()?.sync_all()?;std::fs::rename(&temp,path)?;Ok(())
    })();if result.is_err(){let _=std::fs::remove_file(temp);}result
}
pub fn load_water_cache(path:impl AsRef<Path>)->Result<WaterSimulationCache> {
    let decoder=zstd::stream::read::Decoder::new(std::fs::File::open(path)?)?;
    let mut input=decoder.take((WATER_CACHE_MAX_BYTES+33)as u64);let mut header=[0u8;32];input.read_exact(&mut header)?;
    ensure!(&header[..8]==MAGIC,"unsupported water cache version");
    let u32_at=|i|u32::from_le_bytes(header[i..i+4].try_into().unwrap());
    let(n,count)=(u32_at(8),u32_at(12));
    ensure!((8..=512).contains(&n) && (1..=7200).contains(&count),"invalid water cache dimensions");
    let bytes=n as u64*n as u64*count as u64*16;ensure!(bytes<=WATER_CACHE_MAX_BYTES as u64,"water cache exceeds playback budget");
    let mut cache=WaterSimulationCache{n,origin:[f32::from_bits(u32_at(16)),f32::from_bits(u32_at(20))],cell:f32::from_bits(u32_at(24)),frame_rate:f32::from_bits(u32_at(28)),frames:Vec::new()};
    let mut data=vec![0u8;n as usize*n as usize*16];
    for _ in 0..count{input.read_exact(&mut data)?;cache.frames.push(data.as_chunks::<16>().0.iter().map(|sample|std::array::from_fn(|i|f32::from_le_bytes(sample[i*4..i*4+4].try_into().unwrap()))).collect());}
    let mut extra=[0u8;1];ensure!(input.read(&mut extra)?==0,"trailing water cache data");cache.validate().map_err(anyhow::Error::msg)?;Ok(cache)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recording_roundtrips_and_rejects_truncation(){
        let path=std::env::temp_dir().join(format!("{}.watercache",zestengine_core::ObjectId::new()));
        let cache=WaterSimulationCache{n:8,origin:[-5.0,3.0],cell:0.25,frame_rate:20.0,frames:vec![vec![[0.1,0.2,-0.3,0.4];64];3]};
        save_water_cache(&path,&cache).unwrap();let read=load_water_cache(&path).unwrap();assert_eq!(read.frames,cache.frames);assert_eq!(read.origin,cache.origin);
        let data=std::fs::read(&path).unwrap();std::fs::write(&path,&data[..data.len()/2]).unwrap();assert!(load_water_cache(&path).is_err());std::fs::remove_file(path).unwrap();
    }
}
