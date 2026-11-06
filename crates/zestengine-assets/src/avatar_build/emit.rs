//! Passes 11-14: the single controller, the finished menu, the budget, and
//! writing everything out.

use super::toggles::{controller_param, strip};
use super::*;

/// Longest side of a menu icon in the finished menu.
pub const ICON_MAX: u32 = 256;

/// `bytes` (any image format) scaled to fit `max` x `max`, as PNG. `None`
/// when it already fits or cannot be decoded.
pub fn downscale_icon(bytes: &[u8], max: u32) -> Option<Vec<u8>> {
    let img = image::load_from_memory(bytes).ok()?;
    if img.width() <= max && img.height() <= max {
        return None;
    }
    let small = img.resize(max, max, image::imageops::FilterType::Lanczos3);
    let mut out = std::io::Cursor::new(Vec::new());
    small.write_to(&mut out, image::ImageFormat::Png).ok()?;
    Some(out.into_inner())
}

/// Merge `src` pages into `dst`: Sub Menus with the same normalised label
/// join, everything else appends.
fn merge_pages(dst: &mut Vec<AvatarMenuNode>, src: Vec<AvatarMenuNode>) {
    for n in src {
        let want = normalise_label(&n.item.label);
        let same = (n.item.kind == MenuItemKind::SubMenu)
            .then(|| dst.iter().position(|d| d.item.kind == MenuItemKind::SubMenu && normalise_label(&d.item.label) == want))
            .flatten();
        match same {
            Some(i) => merge_pages(&mut dst[i].children, n.children),
            None => dst.push(n),
        }
    }
}

impl Builder<'_> {
    // ------------------------------------------------ 11 flatten controller

    pub(super) fn flatten_controller(&mut self) {
        if self.gen_layers.is_empty() && self.merged.is_empty() && self.defaults_layer.is_none() {
            return;
        }
        let mut layers: Vec<AnimLayer> = Vec::new();
        let mut listed: Vec<GeneratedLayer> = Vec::new();
        let append = |layers: &mut Vec<AnimLayer>, set: Vec<AnimLayer>| {
            let start = layers.len();
            for mut l in set {
                l.sync_with = l.sync_with.map(|s| s + start);
                layers.push(l);
            }
        };
        if let Some(d) = self.defaults_layer.take() {
            listed.push(GeneratedLayer { name: d.name.clone(), source: "defaults for every animated property".into() });
            append(&mut layers, vec![d]);
        }
        if !self.replace_base && self.had_base {
            let base = std::mem::take(&mut self.base.layers);
            append(&mut layers, base);
        }
        for set in std::mem::take(&mut self.merged) {
            let _ = set.layer;
            for l in &set.layers {
                listed.push(GeneratedLayer { name: l.name.clone(), source: format!("merged {} controller", set.layer.label()) });
            }
            append(&mut layers, set.layers);
        }
        for (l, source) in std::mem::take(&mut self.gen_layers) {
            listed.push(GeneratedLayer { name: l.name.clone(), source });
            layers.push(l);
        }
        let mut parameters: Vec<AnimParameter> = Vec::new();
        let mut add = |p: AnimParameter| {
            if !parameters.iter().any(|q| q.name == p.name) {
                parameters.push(p);
            }
        };
        if !self.replace_base {
            self.base.parameters.iter().cloned().for_each(&mut add);
        }
        self.ctrl_params.iter().cloned().for_each(&mut add);
        for p in &self.params {
            add(controller_param(&p.name, p.kind, p.default));
        }
        self.out.layers = listed;
        self.controller = Some(AnimatorController { parameters, layers });
    }

    // ------------------------------------------------------- 12 finalize menu

    fn menu_parent(&self, i: usize, items: &BTreeMap<usize, AvatarMenuItem>) -> Option<usize> {
        let mut k = self.scene.entries[i].parent;
        let mut guard = 0;
        while let Some(p) = k {
            if p == self.root {
                return None;
            }
            if items.get(&p).is_some_and(|m| m.kind == MenuItemKind::SubMenu) {
                return Some(p);
            }
            guard += 1;
            if guard > 256 {
                return None;
            }
            k = self.scene.entries[p].parent;
        }
        None
    }

    fn menu_page(&mut self, parent: Option<usize>, items: &BTreeMap<usize, AvatarMenuItem>, depth: usize) -> Vec<AvatarMenuNode> {
        let mut page = Vec::new();
        if depth > 16 {
            return page;
        }
        let members: Vec<usize> = self.sub.iter().copied().filter(|i| items.contains_key(i) && self.menu_parent(*i, items) == parent).collect();
        for i in members {
            let mut item = items[&i].clone();
            if i == self.root && item.kind == MenuItemKind::SubMenu {
                continue;
            }
            if item.label.is_empty() {
                item.label = self.name_of(i).to_owned();
            }
            item.parameter = if item.parameter.is_empty() { self.auto_params.get(&i).cloned().unwrap_or_default() } else { self.scoped(i, &item.parameter) };
            item.sub_parameters = item.sub_parameters.iter().map(|p| self.scoped(i, p)).collect();
            let mut node = AvatarMenuNode::new(item);
            if node.item.kind == MenuItemKind::SubMenu {
                node.children = self.menu_page(Some(i), items, depth + 1);
                if !node.item.sub_menu_asset.is_empty() {
                    let path = zestengine_core::asset_ref::resolve_path(self.env.project_root, &node.item.sub_menu_asset);
                    match AvatarMenu::load(&path) {
                        Ok(m) => merge_pages(&mut node.children, m.root),
                        Err(e) => self.warn(format!("menu asset {}: {e}", node.item.sub_menu_asset)),
                    }
                    node.item.sub_menu_asset.clear();
                }
            }
            page.push(node);
        }
        page
    }

    pub(super) fn finalize_menu(&mut self) {
        let items: BTreeMap<usize, AvatarMenuItem> =
            self.sub.iter().filter_map(|&i| comps::<AvatarMenuItem>(&self.scene.entries[i]).into_iter().next().map(|m| (i, m))).collect();
        // A root Sub Menu that points at an asset contributes that asset's page.
        let mut root_page = self.menu_page(None, &items, 0);
        if let Some(m) = items.get(&self.root).filter(|m| m.kind == MenuItemKind::SubMenu && !m.sub_menu_asset.is_empty()) {
            let path = zestengine_core::asset_ref::resolve_path(self.env.project_root, &m.sub_menu_asset);
            match AvatarMenu::load(&path) {
                Ok(a) => merge_pages(&mut root_page, a.root),
                Err(e) => self.warn(format!("menu asset {}: {e}", m.sub_menu_asset)),
            }
        }
        let generated = std::mem::take(&mut self.menu);
        merge_pages(&mut root_page, generated.root);
        let mut menu = AvatarMenu { root: root_page };
        // Menu Installers: an authored page goes to its install target.
        let installers: Vec<(usize, MenuInstaller)> =
            self.sub.iter().flat_map(|&i| comps::<MenuInstaller>(&self.scene.entries[i]).into_iter().map(move |m| (i, m))).collect();
        for (i, inst) in installers {
            let who = format!("{} (Menu Installer)", if self.paths[i].is_empty() { self.name_of(i).to_owned() } else { self.paths[i].clone() });
            if inst.menu.is_empty() {
                self.warn(format!("{who}: no menu asset"));
                continue;
            }
            let path = zestengine_core::asset_ref::resolve_path(self.env.project_root, &inst.menu);
            match AvatarMenu::load(&path) {
                Ok(m) => {
                    let page = menu.page_at_mut(&zestengine_core::split_menu_path(&inst.install_to));
                    merge_pages(page, m.root);
                }
                Err(e) => self.warn(format!("{who}: {}: {e}", inst.menu)),
            }
        }
        menu.finalize();
        self.downscale_icons(&mut menu);
        let declared: BTreeSet<String> = self.params.iter().map(|p| p.name.clone()).collect();
        let known = |n: &str| declared.contains(n) || BuiltinParam::from_name(n).is_some();
        for msg in menu.problems(&known) {
            self.out.warnings.push(format!("menu: {msg}"));
        }
        self.out.menu = menu;
    }

    /// Menu icons larger than [`ICON_MAX`] px are replaced by a downscaled PNG
    /// in the build folder (the original is untouched).
    fn downscale_icons(&mut self, menu: &mut AvatarMenu) {
        fn walk(page: &mut [AvatarMenuNode], f: &mut dyn FnMut(&mut AvatarMenuItem)) {
            for n in page {
                f(&mut n.item);
                walk(&mut n.children, f);
            }
        }
        let root = self.env.project_root.to_path_buf();
        let out_dir = self.out_dir.clone();
        let mut done: BTreeMap<String, String> = BTreeMap::new();
        let mut files: Vec<(String, Vec<u8>)> = Vec::new();
        walk(&mut menu.root, &mut |item| {
            if item.icon.is_empty() {
                return;
            }
            if let Some(d) = done.get(&item.icon) {
                item.icon = d.clone();
                return;
            }
            let path = zestengine_core::asset_ref::resolve_path(&root, &item.icon);
            let Ok(bytes) = std::fs::read(&path) else { return };
            if let Some(png) = downscale_icon(&bytes, ICON_MAX) {
                let rel = format!("{out_dir}/icons/{}.png", short_hash(&item.icon, 10));
                done.insert(item.icon.clone(), rel.clone());
                files.push((rel.clone(), png));
                item.icon = rel;
            }
        });
        self.files.extend(files);
    }

    // ------------------------------------------------------------- 13 budget

    pub(super) fn budget(&mut self) {
        let max = self.root_budget.min(self.env.budget_cap);
        let b = ParamBudget::of(&self.params, max);
        if b.is_over() {
            self.error(format!("the avatar's parameters need {} synced bits but only {max} are allowed ({} over)", b.used, b.used - max));
        }
        self.out.budget = Some(b);
    }

    // --------------------------------------------------------------- 14 emit

    pub(super) fn emit(&mut self) {
        let out_dir = self.out_dir.clone();
        // Root components.
        let root = self.root;
        let budget_bits = self.root_budget;
        let nonroot: Vec<usize> = self.sub.iter().copied().filter(|&i| i != root).collect();
        for i in nonroot {
            strip(&mut self.scene.entries[i], &[AvatarParameters::NAME]);
        }
        for &i in &self.sub {
            strip(&mut self.scene.entries[i], &super::authoring_kinds());
        }
        let any_params = !self.params.is_empty() || self.scene.entries[root].components.iter().any(|c| c.kind == AvatarParameters::NAME);
        if any_params {
            let ap = AvatarParameters { params: self.params.clone(), budget_bits, ..Default::default() };
            set_comp(&mut self.scene.entries[root], &ap);
        }
        if !self.out.menu.is_empty() {
            set_comp(&mut self.scene.entries[root], &AvatarMenuData { menu: self.out.menu.clone() });
        }

        // Files.
        for (rel, clip) in &self.clips {
            self.files.insert(rel.clone(), clip_bytes(clip));
        }
        if let Some(ctrl) = self.controller.take() {
            let rel = format!("{out_dir}/controller.animctrl");
            self.files.insert(rel.clone(), ron::ser::to_string_pretty(&ctrl, ron::ser::PrettyConfig::default()).unwrap_or_default().into_bytes());
            let target = self.animator.unwrap_or(root);
            let mut anim = comps::<Animator>(&self.scene.entries[target]).into_iter().next().unwrap_or_default();
            anim.controller = rel;
            set_comp(&mut self.scene.entries[target], &anim);
        }
        if any_params {
            let ap = AvatarParameters { params: self.params.clone(), budget_bits, ..Default::default() };
            self.files.insert(format!("{out_dir}/params.avparams"), ron::ser::to_string_pretty(&ap, ron::ser::PrettyConfig::default()).unwrap_or_default().into_bytes());
        }
        if !self.out.menu.is_empty() {
            self.files.insert(format!("{out_dir}/menu.{}", zestengine_core::AVATAR_MENU_EXTENSION), self.out.menu.to_ron().into_bytes());
        }
        self.out.params = self.params.clone();
        self.out.generated_params = self.generated_params.clone();
        self.out.files = self.files.clone();
        if self.env.write {
            self.write_files();
        }
    }

    /// Write changed files, remove ones the previous build left behind.
    fn write_files(&mut self) {
        let root = self.env.project_root.to_path_buf();
        let stamp = root.join(&self.out_dir).join(".inputhash");
        // Same inputs as the last build and every output present: nothing to
        // write (stale files are still removed below).
        self.out.unchanged = !self.out.input_hash.is_empty()
            && std::fs::read_to_string(&stamp).ok().as_deref() == Some(self.out.input_hash.as_str())
            && self.files.iter().all(|(rel, bytes)| std::fs::read(root.join(rel)).ok().as_deref() == Some(bytes.as_slice()));
        for (rel, bytes) in self.files.iter().filter(|_| !self.out.unchanged) {
            let path = root.join(rel);
            if std::fs::read(&path).ok().as_deref() == Some(bytes.as_slice()) {
                continue;
            }
            let res = path.parent().map_or(Ok(()), std::fs::create_dir_all).and_then(|_| std::fs::write(&path, bytes));
            if let Err(e) = res {
                self.out.warnings.push(format!("writing {rel}: {e}"));
            }
        }
        if !self.files.keys().any(|k| k.ends_with(".animctrl") || k.ends_with(".anim")) {
            // Nothing generated (an already-built avatar): its files stay.
            return;
        }
        let stamp_hash = self.out.input_hash.clone();
        let dir = root.join(&self.out_dir);
        let mut stack = vec![dir];
        while let Some(d) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for entry in rd.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    stack.push(p);
                    continue;
                }
                let rel = p.strip_prefix(&root).map(|r| r.to_string_lossy().replace('\\', "/")).unwrap_or_default();
                if !self.files.contains_key(&rel) && p.file_name().is_none_or(|n| n != ".inputhash") {
                    let _ = std::fs::remove_file(&p);
                }
            }
            // An emptied folder goes too (not the avatar's own, which may be reused).
            if d != root.join(&self.out_dir) {
                let _ = std::fs::remove_dir(&d);
            }
        }
        if !stamp_hash.is_empty() {
            let _ = std::fs::write(&stamp, stamp_hash);
        }
    }
}
