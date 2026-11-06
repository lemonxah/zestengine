//! Export the real scaffolded template scene data for offline Hub previews.
//! cargo run -p zestengine-engine --example template_preview_data -- /tmp/zestengine-preview-scenes
use anyhow::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    let root = PathBuf::from(std::env::args().nth(1).expect("output folder"));
    std::fs::create_dir_all(&root)?;
    for template in zestengine_engine::templates::TEMPLATES {
        let project = zestengine_engine::scaffold_project(&root, template.id, template.id)?;
        let settings = zestengine_assets::load_project_file(&project)?;
        let scene = zestengine_assets::load_scene_file(project.join(settings.boot_scene.unwrap()))?;
        std::fs::write(project.join("preview.json"), serde_json::to_vec_pretty(&scene)?)?;
        println!("{}", project.display());
    }
    Ok(())
}
