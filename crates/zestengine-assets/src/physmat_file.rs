use std::path::Path;

use zestengine_core::PhysicsMaterial;

pub fn load_physmat(path: impl AsRef<Path>) -> Result<PhysicsMaterial, String> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    ron::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))
}

pub fn save_physmat(path: impl AsRef<Path>, mat: &PhysicsMaterial) -> Result<(), String> {
    let path = path.as_ref();
    let text = ron::ser::to_string_pretty(mat, ron::ser::PrettyConfig::default()).map_err(|e| e.to_string())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    crate::write_atomic(path, text.as_bytes()).map_err(|e| format!("writing {}: {e:#}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_material_round_trips_through_its_file_and_bad_files_say_why() {
        let dir = std::env::temp_dir().join(format!("zest-physmat-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("surfaces").join("ice.physmat");
        let (_, mut ice) = PhysicsMaterial::PRESETS[0];
        ice.friction = 0.02;
        save_physmat(&path, &ice).expect("the folder is created");
        assert_eq!(load_physmat(&path).unwrap(), ice);
        save_physmat(&path, &PhysicsMaterial::default()).unwrap();
        assert_eq!(load_physmat(&path).unwrap(), PhysicsMaterial::default(), "saving replaces the file");
        assert!(load_physmat(dir.join("missing.physmat")).unwrap_err().starts_with("reading"));
        std::fs::write(&path, "not ron").unwrap();
        assert!(load_physmat(&path).unwrap_err().starts_with("parsing"));
        std::fs::write(&path, "(friction: 0.3)").unwrap();
        assert_eq!(load_physmat(&path).unwrap().friction, 0.3, "missing fields take their defaults");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
