//! CPU-only model import timing, stage by stage:
//! `cargo run --release -p zestengine-assets --no-default-features --example fbx_bench -- file.fbx`
//! (`NAMES=1` lists the instances). Works on a temp copy so caches never touch the source folder.
use std::time::Instant;

fn lap<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = f();
    println!("{label:34} {:>9.1} ms", t.elapsed().as_secs_f64() * 1e3);
    r
}

/// FNV hash over every mesh's vertices, indices and blendshape deltas, to compare runs.
fn fingerprint(scene: &zestengine_assets::Scene) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut eat = |bytes: &[u8]| {
        for b in bytes {
            h = (h ^ *b as u64).wrapping_mul(0x100000001b3);
        }
    };
    for (i, m) in scene.meshes.iter().enumerate() {
        eat(bytemuck_cast(&m.vertices));
        eat(bytemuck_cast(&m.indices));
        for t in scene.morphs.get(i).into_iter().flatten() {
            eat(t.name.as_bytes());
            for (v, p, n) in &t.deltas {
                eat(&v.to_le_bytes());
                for x in p.iter().chain(n) {
                    eat(&x.to_bits().to_le_bytes());
                }
            }
        }
    }
    h
}

fn bytemuck_cast<T>(v: &[T]) -> &[u8] {
    // Vertices are plain f32/u32 data.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: fbx_bench <file.fbx>");
    let dir = std::env::temp_dir().join(format!("zest_fbx_bench_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let copy = dir.join(std::path::Path::new(&path).file_name().unwrap());
    std::fs::copy(&path, &copy).unwrap();

    if std::env::var_os("ONLY_LOAD").is_some() { let _ = zestengine_assets::load_fbx(&copy).unwrap(); return; }
    let scene = lap("load_fbx (parse+convert)", || zestengine_assets::load_fbx(&copy).expect("load"));
    println!("  fingerprint {:016x}", fingerprint(&scene));
    let verts: usize = scene.meshes.iter().map(|m| m.vertices.len()).sum();
    let shapes: usize = scene.morphs.iter().map(Vec::len).sum();
    let deltas: usize = scene.morphs.iter().flatten().map(|m| m.deltas.len()).sum();
    println!(
        "  {} instances, {} meshes, {} verts, {} textures, {} morph shapes ({} deltas), {} joints",
        scene.instances.len(), scene.meshes.len(), verts, scene.textures.len(), shapes, deltas,
        scene.skeletons.iter().map(|s| s.joints.len()).sum::<usize>(),
    );
    if std::env::var_os("NAMES").is_some() {
        for i in &scene.instances {
            let m = i.slots.first().map(|s| s.mesh).unwrap_or(0);
            println!("  {:40} verts {:8} morphs {}", i.name, scene.meshes[m].vertices.len(), scene.morphs.get(m).map_or(0, Vec::len));
        }
    }
    let (_s, rep) = lap("load_model_cached (miss + write)", || zestengine_assets::load_model_cached(&copy, false).unwrap());
    println!("  outcome {:?} key {:?} load {:?}", rep.outcome, rep.key_time, rep.load_time);
    let _ = lap("load_model_cached (hit)", || zestengine_assets::load_model_cached(&copy, false).unwrap());
    let (scene, _) = zestengine_assets::load_model_cached(&copy, false).unwrap();
    let geo = lap("geo_cache::load_or_build (miss)", || zestengine_assets::geo_cache::load_or_build(&copy, &scene));
    println!("  {} clusters, {:.1} MB", geo.clusters(), geo.dag_bytes() as f64 / 1e6);
    lap("geo_cache::load_or_build (hit)", || zestengine_assets::geo_cache::load_or_build(&copy, &scene));
    let lods = lap("auto_lod::build_auto_lods_with", || zestengine_assets::auto_lod::build_auto_lods_with(&scene, Some(&geo)));
    println!("  {} lod levels", lods.map_or(0, |l| l.len()));
    let _ = std::fs::remove_dir_all(&dir);
}
