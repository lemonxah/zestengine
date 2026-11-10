//! Versioned, bounded and atomically saved `.clothcache` recordings.
use std::{io::{Read,Write},path::Path};
use anyhow::{Result,ensure};
use zestengine_core::{ClothCacheFrame,ClothSimulationCache,cloth_cache::{CLOTH_CACHE_MAX_BYTES,validate_dimensions}};
pub const CLOTH_CACHE_EXTENSION:&str="clothcache";
const MAGIC:&[u8;8]=b"ZCLOTH01";

pub fn save_cloth_cache(path:impl AsRef<Path>,cache:&ClothSimulationCache)->Result<()>{
    cache.validate().map_err(anyhow::Error::msg)?;let path=path.as_ref();
    if let Some(parent)=path.parent(){std::fs::create_dir_all(parent)?;}
    let temp=path.with_extension(format!("clothcache.{}.tmp",zestengine_core::ObjectId::new()));
    let result=(||->Result<()>{
        let mut out=zstd::stream::write::Encoder::new(std::fs::File::create(&temp)?,3)?;
        out.write_all(MAGIC)?;out.write_all(&cache.mesh_signature.to_le_bytes())?;
        out.write_all(&cache.vertex_count.to_le_bytes())?;out.write_all(&(cache.frames.len()as u32).to_le_bytes())?;
        out.write_all(&cache.frame_rate.to_le_bytes())?;out.write_all(&0u32.to_le_bytes())?;
        for frame in &cache.frames{for(p,n)in frame.positions.iter().zip(&frame.normals){for v in p.iter().chain(n){out.write_all(&v.to_le_bytes())?;}}}
        out.finish()?.sync_all()?;std::fs::rename(&temp,path)?;Ok(())
    })();if result.is_err(){let _=std::fs::remove_file(temp);}result
}

pub fn load_cloth_cache(path:impl AsRef<Path>)->Result<ClothSimulationCache>{
    let decoder=zstd::stream::read::Decoder::new(std::fs::File::open(path)?)?;
    let mut input=decoder.take((CLOTH_CACHE_MAX_BYTES+33)as u64);let mut header=[0u8;32];input.read_exact(&mut header)?;
    ensure!(&header[..8]==MAGIC,"unsupported cloth cache version");
    let u32_at=|i|u32::from_le_bytes(header[i..i+4].try_into().unwrap());
    let vertices=u32_at(16);let count=u32_at(20);let rate=f32::from_bits(u32_at(24));
    ensure!(u32_at(28)==0,"unsupported cloth cache flags");
    validate_dimensions(vertices as usize,count as usize,rate).map_err(anyhow::Error::msg)?;
    let mut cache=ClothSimulationCache{mesh_signature:u64::from_le_bytes(header[8..16].try_into().unwrap()),vertex_count:vertices,frame_rate:rate,frames:Vec::with_capacity(count as usize)};
    let mut buffer=vec![0u8;vertices as usize*24];
    for _ in 0..count{
        input.read_exact(&mut buffer)?;
        let mut frame=ClothCacheFrame{positions:Vec::with_capacity(vertices as usize),normals:Vec::with_capacity(vertices as usize)};
        for vertex in buffer.chunks_exact(24){
            let value=|i|f32::from_le_bytes(vertex[i..i+4].try_into().unwrap());
            frame.positions.push([value(0),value(4),value(8)]);frame.normals.push([value(12),value(16),value(20)]);
        }
        frame.validate(vertices as usize).map_err(anyhow::Error::msg)?;cache.frames.push(frame);
    }
    let mut extra=[0u8;1];ensure!(input.read(&mut extra)?==0,"trailing cloth cache data");Ok(cache)
}

#[cfg(test)]
mod tests{
    use super::*;
    struct Temp(std::path::PathBuf);
    impl Drop for Temp{fn drop(&mut self){let _=std::fs::remove_dir_all(&self.0);}}
    fn temp()->Temp{let p=std::env::temp_dir().join(format!("cloth-cache-{}",zestengine_core::ObjectId::new()));std::fs::create_dir_all(&p).unwrap();Temp(p)}
    fn sample()->ClothSimulationCache{ClothSimulationCache{mesh_signature:0x123456789abcdef0,vertex_count:3,frame_rate:30.0,frames:vec![ClothCacheFrame{positions:vec![[0.5,-0.2,4.0];3],normals:vec![[0.0,1.0,0.0];3]};2]}}
    #[test]
    fn cloth_recording_roundtrips_and_an_invalid_save_preserves_the_file(){
        let dir=temp();let path=dir.0.join("take.clothcache");let cache=sample();save_cloth_cache(&path,&cache).unwrap();assert_eq!(load_cloth_cache(&path).unwrap(),cache);
        let mut invalid=cache.clone();invalid.frames[0].positions[0][0]=f32::NAN;
        assert!(save_cloth_cache(&path,&invalid).is_err());assert_eq!(load_cloth_cache(&path).unwrap(),cache);
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(),1);
    }
    #[test]
    fn cloth_recording_rejects_corrupt_truncated_and_oversized_payloads(){
        let dir=temp();let path=dir.0.join("take.clothcache");save_cloth_cache(&path,&sample()).unwrap();
        let data=std::fs::read(&path).unwrap();let original=zstd::stream::decode_all(&data[..]).unwrap();
        let write=|bytes:&[u8]|std::fs::write(&path,zstd::stream::encode_all(bytes,1).unwrap()).unwrap();
        write(&original[..original.len()-1]);assert!(load_cloth_cache(&path).is_err());
        let mut trailing=original.clone();trailing.push(0);write(&trailing);assert!(load_cloth_cache(&path).is_err());
        let mut oversized=original.clone();oversized[16..20].copy_from_slice(&u32::MAX.to_le_bytes());write(&oversized);assert!(load_cloth_cache(&path).is_err());
        let mut nonfinite=original.clone();nonfinite[32..36].copy_from_slice(&f32::NAN.to_le_bytes());write(&nonfinite);assert!(load_cloth_cache(&path).is_err());
        let mut version=original;version[7]=b'2';write(&version);assert!(load_cloth_cache(&path).is_err());
        std::fs::write(&path,&data[..data.len()/2]).unwrap();assert!(load_cloth_cache(&path).is_err());
    }
}
