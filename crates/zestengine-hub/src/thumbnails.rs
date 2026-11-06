//! Project thumbnails: `thumbnail.png` in the project root (or in its `.zestengine`
//! folder), decoded and downscaled on a worker thread so a big screenshot
//! never stalls the list.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

/// Thumbnail size in pixels (16:9).
const THUMB: [u32; 2] = [256, 144];

pub struct Thumbnails {
    textures: HashMap<PathBuf, Option<egui::TextureHandle>>,
    requests: mpsc::Sender<PathBuf>,
    results: mpsc::Receiver<(PathBuf, Option<egui::ColorImage>)>,
}

impl Thumbnails {
    pub fn new(ctx: &egui::Context) -> Self {
        let (req_tx, req_rx) = mpsc::channel::<PathBuf>();
        let (res_tx, res_rx) = mpsc::channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            for root in req_rx {
                let image = thumbnail_file(&root).and_then(|f| decode(&f));
                if res_tx.send((root, image)).is_err() {
                    break;
                }
                ctx.request_repaint();
            }
        });
        Self { textures: HashMap::new(), requests: req_tx, results: res_rx }
    }

    /// The project's thumbnail, once decoded; None while loading or if it has none.
    pub fn get(&mut self, ctx: &egui::Context, root: &Path) -> Option<&egui::TextureHandle> {
        while let Ok((path, image)) = self.results.try_recv() {
            let tex = image.map(|img| ctx.load_texture(format!("thumb:{}", path.display()), img, egui::TextureOptions::LINEAR));
            self.textures.insert(path, tex);
        }
        if !self.textures.contains_key(root) {
            self.textures.insert(root.to_path_buf(), None);
            let _ = self.requests.send(root.to_path_buf());
        }
        self.textures.get(root).and_then(Option::as_ref)
    }

    /// Forget every thumbnail so they reload (the window regained focus).
    pub fn clear(&mut self) {
        self.textures.clear();
    }
}

fn thumbnail_file(root: &Path) -> Option<PathBuf> {
    [root.join("thumbnail.png"), root.join(".zestengine").join("thumbnail.png")].into_iter().find(|p| p.is_file())
}

fn decode(path: &Path) -> Option<egui::ColorImage> {
    let img = image::open(path).map_err(|e| tracing::debug!("thumbnail {}: {e}", path.display())).ok()?;
    let img = img.resize_to_fill(THUMB[0], THUMB[1], image::imageops::FilterType::Triangle).to_rgba8();
    Some(egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw()))
}
