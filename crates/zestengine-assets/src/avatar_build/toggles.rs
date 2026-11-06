//! Passes 3-6, 8 and 10: menu parameters, toggle / reactive layers, exclusive
//! tag groups, the resting state and the defaults layer.

use super::*;
use zestengine_core::ObjectToggleEntry;

/// A state at `position`. Write Defaults stays on: a state animates only what
/// it names, and the Defaults layer below supplies the resting values.
fn state(name: &str, position: [f32; 2], motion: Motion) -> AnimState {
    AnimState { name: name.into(), motion, looping: false, write_defaults: true, position, ..Default::default() }
}

fn transition(from: Option<&str>, to: &str, conditions: Vec<AnimCondition>) -> AnimTransition {
    AnimTransition { from: from.map(str::to_owned), to: to.into(), conditions, has_exit_time: false, duration: 0.0, ..Default::default() }
}

/// Attach a parameter driver running when `s` is entered.
fn drive(s: &mut AnimState, entries: Vec<(String, AvatarParamKind, f32)>) {
    if entries.is_empty() {
        return;
    }
    let settings = ParamDriverSettings {
        entries: entries.into_iter().map(|(name, kind, value)| DriverEntry { name, kind, op: DriverOp::Set { value } }).collect(),
        local_only: false,
    };
    if !s.behaviours.iter().any(|b| b == PARAMETER_DRIVER) {
        s.behaviours.push(PARAMETER_DRIVER.into());
    }
    s.behaviour_settings.insert(PARAMETER_DRIVER.into(), settings.to_ron());
}

fn anim_kind(k: AvatarParamKind) -> AnimParamKind {
    match k {
        AvatarParamKind::Bool => AnimParamKind::Bool,
        AvatarParamKind::Int => AnimParamKind::Int,
        AvatarParamKind::Float => AnimParamKind::Float,
    }
}

impl Builder<'_> {
    // ------------------------------------------------ 3 menu item parameters

    pub(super) fn assign_menu_parameters(&mut self) {
        let sub = self.sub.clone();
        for &i in &sub {
            for item in comps::<AvatarMenuItem>(&self.scene.entries[i]) {
                if !matches!(item.kind, MenuItemKind::Toggle | MenuItemKind::Button) || !item.parameter.is_empty() {
                    continue;
                }
                let reactive = sub.iter().any(|&k| self.is_within(k, i) && !comps::<ObjectToggle>(&self.scene.entries[k]).is_empty());
                if !reactive {
                    continue;
                }
                let label = if item.label.is_empty() { self.name_of(i).to_owned() } else { item.label.clone() };
                let name = format!("AutoParam_{}_{}", sanitise(&label), short_hash(&self.paths[i], 6));
                self.add_param(
                    AvatarParam {
                        name: name.clone(),
                        kind: AvatarParamKind::Bool,
                        default: item.is_default as u8 as f32,
                        saved: item.saved,
                        sync: if item.synced { AvatarParamSync::Synced } else { AvatarParamSync::LocalOnly },
                        scope: AvatarParamScope::Global,
                    },
                    true,
                    &self.paths[i].clone(),
                );
                self.auto_params.insert(i, name);
            }
        }
    }

    // ------------------------------------------------------------ clip making

    /// The animator-relative address of a root-relative `target`, when the
    /// object exists and the Animator can reach it.
    fn anim_target(&mut self, target: &str, who: &str) -> Option<String> {
        if self.resolve_path(target).is_none() {
            self.warn(format!("{who}: no object at '{target}'; action skipped"));
            return None;
        }
        let t = self.to_anim(target);
        if t.is_none() {
            self.warn(format!("{who}: '{target}' is outside the Animator's object '{}'; action skipped", self.anim_prefix));
        }
        t
    }

    #[allow(clippy::too_many_arguments)]
    fn put_track(
        &mut self,
        clip: &mut AnimClip,
        target: &str,
        property: TrackProperty,
        name: &str,
        slot: u8,
        on: [f32; 4],
        off: [f32; 4],
        slider: bool,
        explicit_off: Option<[f32; 4]>,
    ) {
        let mut track = AnimTrack::prop(target, property, name, slot);
        let interp = property.default_interp();
        let interp = if property.is_step() { crate::KeyInterp::Step } else if interp == crate::KeyInterp::Cubic { crate::KeyInterp::Linear } else { interp };
        if slider && !property.is_step() {
            track.keys.push(Keyframe { time: 0.0, value: off, interp, ..Default::default() });
            track.keys.push(Keyframe { time: 1.0, value: on, interp, ..Default::default() });
        } else {
            track.keys.push(Keyframe { time: 0.0, value: on, interp, ..Default::default() });
        }
        match clip.tracks.iter().position(|t| t.is(target, property, name, slot)) {
            Some(i) => clip.tracks[i] = track,
            None => clip.tracks.push(track),
        }
        if property.is_object_prop() {
            let key = PropKey { target: target.to_owned(), property: property as u8, name: name.to_owned(), slot: if property.uses_slot() { slot } else { 0 } };
            let e = self.keys.entry(key).or_insert(None);
            if explicit_off.is_some() {
                *e = explicit_off;
            }
        }
    }

    pub(super) fn load_clip(&mut self, rel: &str, who: &str) -> Option<AnimClip> {
        let path = zestengine_core::asset_ref::resolve_path(self.env.project_root, rel);
        match crate::load_anim_clip(&path) {
            Ok(c) => Some(c),
            Err(e) => {
                self.warn(format!("{who}: clip {rel} could not be read ({e:#}); skipped"));
                None
            }
        }
    }

    /// The "on" clip of `actions`. A slider's clip is one second long, from the
    /// resting value (0) to the full effect (1); anything else is one frame.
    fn build_clip(&mut self, actions: &[ToggleAction], slider: bool, who: &str) -> AnimClip {
        let mut clip = AnimClip { length: if slider { 1.0 } else { CONSTANT_CLIP_LEN }, looping: false, frame_rate: 60.0, tracks: Vec::new(), events: Vec::new(), ..Default::default() };
        for a in actions {
            match a {
                ToggleAction::Active { target, on } => {
                    if let Some(t) = self.anim_target(target, who) {
                        let (v, o) = (*on as u8 as f32, !*on as u8 as f32);
                        self.put_track(&mut clip, &t, TrackProperty::Active, "", 0, [v, 0.0, 0.0, 0.0], [o, 0.0, 0.0, 0.0], slider, Some([o, 0.0, 0.0, 0.0]));
                    }
                }
                ToggleAction::ComponentEnabled { target, component, on } => {
                    if let Some(t) = self.anim_target(target, who) {
                        let (v, o) = (*on as u8 as f32, !*on as u8 as f32);
                        self.put_track(&mut clip, &t, TrackProperty::ComponentEnabled, component, 0, [v, 0.0, 0.0, 0.0], [o, 0.0, 0.0, 0.0], slider, Some([o, 0.0, 0.0, 0.0]));
                    }
                }
                ToggleAction::BlendShape { target, shape, value } => {
                    if let Some(t) = self.anim_target(target, who) {
                        let rest = self.blend_shape_rest(target, shape);
                        self.put_track(&mut clip, &t, TrackProperty::BlendShape, shape, 0, [*value, 0.0, 0.0, 0.0], [rest, 0.0, 0.0, 0.0], slider, None);
                    }
                }
                ToggleAction::MaterialFloat { target, slot, prop, value, off } => {
                    if let Some(t) = self.anim_target(target, who) {
                        self.put_track(
                            &mut clip,
                            &t,
                            TrackProperty::MaterialFloat,
                            prop,
                            *slot,
                            [*value, 0.0, 0.0, 0.0],
                            [*off, 0.0, 0.0, 0.0],
                            slider,
                            Some([*off, 0.0, 0.0, 0.0]),
                        );
                    }
                }
                ToggleAction::MaterialColor { target, slot, prop, value, off } => {
                    if let Some(t) = self.anim_target(target, who) {
                        self.put_track(&mut clip, &t, TrackProperty::MaterialColor, prop, *slot, *value, *off, slider, Some(*off));
                    }
                }
                ToggleAction::Clip { path } => {
                    if let Some(src) = self.load_clip(path, who) {
                        clip.length = clip.length.max(src.length);
                        for t in src.tracks {
                            if t.property.is_object_prop() {
                                let key = PropKey {
                                    target: t.target.clone(),
                                    property: t.property as u8,
                                    name: t.name.clone(),
                                    slot: if t.property.uses_slot() { t.slot } else { 0 },
                                };
                                self.keys.entry(key).or_insert(None);
                            }
                            match clip.tracks.iter().position(|o| o.is(&t.target, t.property, &t.name, t.slot)) {
                                Some(i) => clip.tracks[i] = t,
                                None => clip.tracks.push(t),
                            }
                        }
                    }
                }
                ToggleAction::SetParam { .. } => {}
            }
        }
        clip.tracks.sort_by(|a, b| (&a.target, a.property as u8, &a.name, a.slot).cmp(&(&b.target, b.property as u8, &b.name, b.slot)));
        clip
    }

    /// Current blendshape weight of `shape` on the object at `target` (0 when
    /// the object has none stored).
    fn blend_shape_rest(&self, target: &str, shape: &str) -> f32 {
        self.resolve_path(target)
            .and_then(|i| comps::<BlendShapes>(&self.scene.entries[i]).into_iter().next())
            .and_then(|b| b.weights.iter().find(|(n, _)| n == shape).map(|(_, w)| *w))
            .unwrap_or(0.0)
    }

    // ---------------------------------------------------------- layer making

    fn unique_layer_name(&self, base: String, salt: &str) -> String {
        if self.gen_layers.iter().any(|(l, _)| l.name == base) { format!("{base} ~{}", short_hash(salt, 4)) } else { base }
    }

    /// The Off / On layer for spec `si`, with its clip.
    fn spec_layer(&mut self, si: usize) {
        let (entry, label, stem, actions, param, cond, mode, default_on, drive_globals, kind) = {
            let s = &self.specs[si];
            (s.entry, s.label.clone(), s.stem.clone(), s.actions.clone(), s.param.clone(), s.cond, s.mode, s.default_on, s.drive_globals.clone(), s.kind)
        };
        let who = format!("{} ({})", if self.paths[entry].is_empty() { self.name_of(entry).to_owned() } else { self.paths[entry].clone() }, label);
        let slider = mode.is_slider();
        let clip = self.build_clip(&actions, slider, &who);
        let rel = format!("{}/toggles/{stem}.anim", self.out_dir);
        let has_tracks = !clip.tracks.is_empty();
        if has_tracks {
            self.clips.insert(rel.clone(), clip);
        }

        let scoped_sets: Vec<(String, AvatarParamKind, f32)> = actions
            .iter()
            .filter_map(|a| match a {
                ToggleAction::SetParam { name, value } => Some((self.scoped(entry, name), self.kind_of(&self.scoped(entry, name)), *value)),
                _ => None,
            })
            .collect();
        let mut on_drive = scoped_sets;
        let mut off_drive = Vec::new();
        for g in &drive_globals {
            on_drive.push((g.clone(), AvatarParamKind::Bool, 1.0));
            off_drive.push((g.clone(), AvatarParamKind::Bool, 0.0));
        }

        let always_on = matches!(mode, ToggleMode::Slider { inactive_at_zero: false, .. });
        let mut on = state("On", [260.0, 0.0], if has_tracks { Motion::Clip(rel) } else { Motion::default() });
        if slider {
            on.time_parameter = param.clone();
        }
        drive(&mut on, on_drive);
        let mut machine = StateMachine::default();
        if always_on {
            machine.default_state = "On".into();
            machine.states.push(on);
        } else {
            let mut off = state("Off", [0.0, 0.0], Motion::default());
            drive(&mut off, off_drive);
            machine.default_state = if default_on { "On".into() } else { "Off".into() };
            machine.states.push(off);
            machine.states.push(on);
            let (fade_in, fade_out, local) = {
                let s = &self.specs[si];
                (s.fade_in, s.fade_out, s.local)
            };
            let is_local = |on: bool| AnimCondition { parameter: "IsLocal".into(), op: if on { CondOp::If } else { CondOp::IfNot }, threshold: 0.0 };
            let mut up = transition(Some("Off"), "On", vec![cond.on(&param)]);
            up.duration = fade_in.max(0.0);
            let mut down = transition(Some("On"), "Off", vec![cond.off(&param)]);
            down.duration = fade_out.max(0.0);
            if local {
                // Remote copies never run it: gate on IsLocal, and fall back
                // to Off if it is ever false.
                up.conditions.push(is_local(true));
                let mut gone = transition(Some("On"), "Off", vec![is_local(false)]);
                gone.duration = fade_out.max(0.0);
                machine.transitions.push(up);
                machine.transitions.push(down);
                machine.transitions.push(gone);
                if !self.ctrl_params.iter().any(|p| p.name == "IsLocal") {
                    self.ctrl_params.push(AnimParameter { name: "IsLocal".into(), kind: AnimParamKind::Bool, default: 1.0 });
                }
            } else {
                machine.transitions.push(up);
                machine.transitions.push(down);
            }
        }
        let prefix = if kind == SpecKind::Toggle { "[gen] " } else { "[gen] reactive " };
        let name = self.unique_layer_name(format!("{prefix}{label}"), &stem);
        let source = who.to_string();
        if !has_tracks && !actions.iter().any(|a| matches!(a, ToggleAction::SetParam { .. })) && !actions.is_empty() {
            self.warn(format!("lint: {who}: every action was skipped, so the toggle's clip is empty and it changes nothing"));
        }
        self.spec_layers.push((si, self.gen_layers.len()));
        self.gen_layers.push((AnimLayer { name, machine, ..Default::default() }, source));
    }

    // ------------------------------------------------------- 4 menu toggles

    /// Selector groups: the Int parameter and each member's option number,
    /// keyed by (entry, menu path). Options number from 1 in hierarchy order.
    fn selector_groups(&self) -> BTreeMap<(usize, String), (String, u32)> {
        let mut groups: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();
        for &i in &self.sub {
            for t in comps::<MenuToggle>(&self.scene.entries[i]) {
                let sel = t.selector.trim();
                if !sel.is_empty() && !t.mode.is_slider() {
                    groups.entry(sel.to_owned()).or_default().push((i, t.menu_path.clone()));
                }
            }
        }
        let mut out = BTreeMap::new();
        for (sel, members) in groups {
            let param = format!("Sel_{}_{}", sanitise(&sel), short_hash(&sel, 6));
            for (n, m) in members.into_iter().enumerate() {
                out.insert(m, (param.clone(), n as u32 + 1));
            }
        }
        out
    }

    pub(super) fn generate_toggles(&mut self) {
        let sub = self.sub.clone();
        let selectors = self.selector_groups();
        // Group defaults: the first member that is on by default.
        let mut sel_default: BTreeMap<String, u32> = BTreeMap::new();
        for &i in &sub {
            for t in comps::<MenuToggle>(&self.scene.entries[i]) {
                if let Some((p, n)) = selectors.get(&(i, t.menu_path.clone())).filter(|_| t.default_on) {
                    sel_default.entry(p.clone()).or_insert(*n);
                }
            }
        }
        for &i in &sub {
            let toggles = comps::<MenuToggle>(&self.scene.entries[i]);
            for t in toggles {
                let entry_name = self.name_of(i).to_owned();
                let segments = t.path_segments(&entry_name);
                let Some(leaf) = segments.last().cloned() else {
                    continue;
                };
                let key = format!("{}|{}", self.paths[i], t.menu_path);
                let slider = t.mode.is_slider();
                let who = format!("{} (Menu Toggle)", if self.paths[i].is_empty() { entry_name.clone() } else { self.paths[i].clone() });
                if t.actions.is_empty() {
                    self.warn(format!("{who}: the toggle has no actions"));
                }
                let selected = selectors.get(&(i, t.menu_path.clone())).cloned();
                let (param, own) = if let Some((p, _)) = &selected {
                    (p.clone(), true)
                } else if t.global_param.is_empty() {
                    (format!("Tgl_{}_{}", sanitise(&leaf), short_hash(&key, 6)), true)
                } else {
                    (self.scoped(i, &t.global_param), false)
                };
                let builtin = BuiltinParam::from_name(&param).is_some();
                if let Some((p, _)) = &selected {
                    self.add_param(
                        AvatarParam {
                            name: p.clone(),
                            kind: AvatarParamKind::Int,
                            default: sel_default.get(p).copied().unwrap_or(0) as f32,
                            saved: t.saved,
                            sync: if t.synced { AvatarParamSync::Synced } else { AvatarParamSync::LocalOnly },
                            scope: AvatarParamScope::Global,
                        },
                        true,
                        &who,
                    );
                } else if !builtin && (own || self.declared(&param).is_none()) {
                    let default = match t.mode {
                        ToggleMode::Slider { default, .. } => default,
                        _ => t.default_on as u8 as f32,
                    };
                    self.add_param(
                        AvatarParam {
                            name: param.clone(),
                            kind: t.param_kind(),
                            default,
                            saved: t.saved,
                            sync: if t.synced { AvatarParamSync::Synced } else { AvatarParamSync::LocalOnly },
                            scope: AvatarParamScope::Global,
                        },
                        true,
                        &who,
                    );
                }
                let drive_globals: Vec<String> = t.drive_globals.iter().map(|g| self.scoped(i, g)).collect();
                for g in &drive_globals {
                    if self.declared(g).is_none() && BuiltinParam::from_name(g).is_none() {
                        self.add_param(
                            AvatarParam { name: g.clone(), kind: AvatarParamKind::Bool, default: 0.0, saved: false, sync: AvatarParamSync::LocalOnly, scope: AvatarParamScope::Global },
                            true,
                            &who,
                        );
                    }
                }

                let (kind, value) = match t.mode {
                    ToggleMode::Toggle => (MenuItemKind::Toggle, 1.0),
                    ToggleMode::Hold => (MenuItemKind::Button, 1.0),
                    ToggleMode::Slider { .. } => (MenuItemKind::RadialPuppet, 1.0),
                };
                let item = AvatarMenuItem {
                    label: leaf.clone(),
                    icon: t.icon.clone(),
                    kind,
                    parameter: if slider { String::new() } else { param.clone() },
                    value: selected.as_ref().map_or(value, |(_, n)| *n as f32),
                    sub_parameters: if slider { vec![param.clone()] } else { Vec::new() },
                    labels: Vec::new(),
                    sub_menu_asset: String::new(),
                    saved: t.saved,
                    synced: t.synced && !builtin,
                    is_default: t.default_on,
                };
                self.menu.insert_at(&segments, item);

                let cond = match (&selected, t.mode) {
                    (Some((_, n)), _) => Cond::Equals(*n as f32),
                    (None, ToggleMode::Slider { inactive_at_zero: true, .. }) => Cond::Above(SLIDER_ON),
                    _ => Cond::Truthy,
                };
                self.specs.push(Spec {
                    kind: SpecKind::Toggle,
                    entry: i,
                    label: segments.join("/"),
                    stem: format!("{}_{}", sanitise(&segments.join("_")), short_hash(&key, 6)),
                    actions: t.actions.clone(),
                    param,
                    cond,
                    mode: t.mode,
                    default_on: t.default_on,
                    invert_rest: t.invert_rest,
                    drive_globals,
                    tags: t.exclusive_tags.iter().map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()).collect(),
                    exclusive_off: t.exclusive_off_state,
                    local: false,
                    fade_in: t.transition_in,
                    fade_out: t.transition_out,
                });
                let si = self.specs.len() - 1;
                self.spec_layer(si);
                if !t.local_actions.is_empty() {
                    let (label, stem, param, cond) = {
                        let s = &self.specs[si];
                        (format!("{} (local)", s.label), format!("{}_local", s.stem), s.param.clone(), s.cond)
                    };
                    self.specs.push(Spec {
                        kind: SpecKind::Toggle,
                        entry: i,
                        label,
                        stem,
                        actions: t.local_actions.clone(),
                        param,
                        cond,
                        mode: t.mode,
                        default_on: t.default_on,
                        invert_rest: t.invert_rest,
                        drive_globals: Vec::new(),
                        tags: Vec::new(),
                        exclusive_off: false,
                        local: true,
                        fade_in: t.transition_in,
                        fade_out: t.transition_out,
                    });
                    let li = self.specs.len() - 1;
                    self.spec_layer(li);
                }
            }
        }
    }

    // ----------------------------------------------------------- 5 reactive

    pub(super) fn generate_reactive(&mut self) {
        let sub = self.sub.clone();
        let mut target_uses: BTreeMap<String, usize> = BTreeMap::new();
        for &i in &sub {
            if self.scene.entries[i].enabled {
                for t in comps::<ObjectToggle>(&self.scene.entries[i]) {
                    for target in t.objects.iter().map(|o| o.target.clone()).collect::<BTreeSet<_>>() {
                        *target_uses.entry(target).or_default() += 1;
                    }
                }
            }
        }
        for &i in &sub {
            if !self.scene.entries[i].enabled {
                continue;
            }
            for t in comps::<ObjectToggle>(&self.scene.entries[i]) {
                let who = format!("{} (Object Toggle)", if self.paths[i].is_empty() { self.name_of(i).to_owned() } else { self.paths[i].clone() });
                // The nearest Toggle / Button menu item at or above this object.
                let mut k = Some(i);
                let mut found = None;
                let mut guard = 0;
                while let Some(j) = k {
                    if let Some(item) = comps::<AvatarMenuItem>(&self.scene.entries[j]).into_iter().find(|m| matches!(m.kind, MenuItemKind::Toggle | MenuItemKind::Button)) {
                        found = Some((j, item));
                        break;
                    }
                    guard += 1;
                    if guard > 256 {
                        break;
                    }
                    k = self.scene.entries[j].parent.filter(|_| j != self.root);
                }
                let Some((j, item)) = found else {
                    self.warn(format!("{who}: no Avatar Menu Item above it, so nothing can switch it"));
                    continue;
                };
                let param = if item.parameter.is_empty() { self.auto_params.get(&j).cloned().unwrap_or_default() } else { self.scoped(j, &item.parameter) };
                if param.is_empty() {
                    continue;
                }
                let cond = if self.kind_of(&param) == AvatarParamKind::Bool || (item.value - 1.0).abs() < 1e-4 && self.declared(&param).is_none() {
                    Cond::Truthy
                } else {
                    Cond::Equals(item.value)
                };
                let label = if self.paths[i].is_empty() { self.name_of(i).to_owned() } else { self.paths[i].clone() };
                // Property-centric: an object another Object Toggle also
                // switches gets its own layer, so toggles on the same property
                // fold into one layer (`optimise_toggles`) instead of fighting.
                let (shared, solo): (Vec<&ObjectToggleEntry>, Vec<&ObjectToggleEntry>) = t.objects.iter().partition(|o| target_uses.get(&o.target).copied().unwrap_or(0) > 1);
                let action = |o: &ObjectToggleEntry| ToggleAction::Active { target: o.target.clone(), on: o.active != t.inverted };
                let mut groups: Vec<(String, String, Vec<ToggleAction>)> = Vec::new();
                if !solo.is_empty() || t.objects.is_empty() {
                    groups.push((label.clone(), format!("{}|{}", self.paths[i], param), solo.iter().map(|o| action(o)).collect()));
                }
                for o in shared {
                    groups.push((format!("{label} -> {}", o.target), format!("{}|{}|{}", self.paths[i], param, o.target), vec![action(o)]));
                }
                for (label, key, actions) in groups {
                    self.specs.push(Spec {
                        kind: SpecKind::Reactive,
                        entry: i,
                        stem: format!("reactive_{}_{}", sanitise(self.name_of(i)), short_hash(&key, 6)),
                        label,
                        actions,
                        param: param.clone(),
                        cond,
                        mode: ToggleMode::Toggle,
                        default_on: item.is_default,
                        invert_rest: false,
                        drive_globals: Vec::new(),
                        tags: Vec::new(),
                        exclusive_off: false,
                        local: false,
                        fade_in: 0.0,
                        fade_out: 0.0,
                    });
                    let si = self.specs.len() - 1;
                    self.spec_layer(si);
                }
            }
        }
    }

    // ------------------------------------------------------ 6 exclusive tags

    pub(super) fn exclusive_tags(&mut self) {
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (si, s) in self.specs.iter().enumerate() {
            if s.kind != SpecKind::Toggle {
                continue;
            }
            for tag in &s.tags {
                let g = groups.entry(tag.clone()).or_default();
                if !g.contains(&si) {
                    g.push(si);
                }
            }
        }
        let mut seen: BTreeSet<Vec<usize>> = BTreeSet::new();
        let mut real_groups: Vec<(String, Vec<usize>)> = Vec::new();
        for (tag, members) in groups {
            if members.len() < 2 || !seen.insert(members.clone()) {
                continue;
            }
            real_groups.push((tag, members));
        }
        for (tag, members) in &real_groups {
            let mut machine = StateMachine { default_state: "Idle".into(), ..Default::default() };
            machine.states.push(state("Idle", [0.0, 0.0], Motion::default()));
            for (n, &m) in members.iter().enumerate() {
                let s = &self.specs[m];
                let sname = format!("On {}", s.label);
                let mut st = state(&sname, [260.0, n as f32 * 70.0 - 70.0], Motion::default());
                let others: Vec<(String, AvatarParamKind, f32)> =
                    members.iter().filter(|&&o| o != m).map(|&o| (self.specs[o].param.clone(), self.kind_of(&self.specs[o].param), 0.0)).collect();
                drive(&mut st, others);
                machine.states.push(st);
                machine.transitions.push(transition(None, &sname, vec![s.cond.on(&s.param)]));
                machine.transitions.push(transition(Some(&sname), "Idle", vec![s.cond.off(&s.param)]));
            }
            let name = self.unique_layer_name(format!("Exclusive Tag - {tag}"), tag);
            self.gen_layers.push((AnimLayer { name, machine, ..Default::default() }, format!("exclusive tag '{tag}'")));
        }
        // A default member that cannot be deselected.
        let mut forced: Vec<(usize, Vec<usize>)> = Vec::new();
        for (si, s) in self.specs.iter().enumerate() {
            if s.kind != SpecKind::Toggle || !s.exclusive_off {
                continue;
            }
            let mut others: BTreeSet<usize> = BTreeSet::new();
            for (_, members) in real_groups.iter().filter(|(_, m)| m.contains(&si)) {
                others.extend(members.iter().copied().filter(|&o| o != si));
            }
            if !others.is_empty() {
                forced.push((si, others.into_iter().collect()));
            }
        }
        if !forced.is_empty() {
            let mut machine = StateMachine { default_state: "Idle".into(), ..Default::default() };
            machine.states.push(state("Idle", [0.0, 0.0], Motion::default()));
            for (n, (si, others)) in forced.iter().enumerate() {
                let s = &self.specs[*si];
                let sname = format!("Force {}", s.label);
                let mut st = state(&sname, [260.0, n as f32 * 70.0], Motion::default());
                drive(&mut st, vec![(s.param.clone(), self.kind_of(&s.param), 1.0)]);
                machine.states.push(st);
                let mut all_off = vec![s.cond.off(&s.param)];
                all_off.extend(others.iter().map(|&o| self.specs[o].cond.off(&self.specs[o].param)));
                machine.transitions.push(transition(Some("Idle"), &sname, all_off));
                machine.transitions.push(transition(Some(&sname), "Idle", vec![s.cond.on(&s.param)]));
            }
            let name = self.unique_layer_name("Exclusive Tag - Off States".into(), "off");
            self.gen_layers.push((AnimLayer { name, machine, ..Default::default() }, "exclusive off states".into()));
        }
    }

    // ------------------------------------------------------ 8 resting state

    /// Apply `on`-values (or `off`-values) of the given specs to the scene,
    /// reporting writers that disagree within the phase.
    fn apply_rest(&mut self, phase: &str, picks: &[(usize, bool)]) {
        let mut written: BTreeMap<PropKey, (f32, String)> = BTreeMap::new();
        for &(si, on_values) in picks {
            let (entry, label, actions) = {
                let s = &self.specs[si];
                (s.entry, s.label.clone(), s.actions.clone())
            };
            let who = format!("{} ({label})", if self.paths[entry].is_empty() { self.name_of(entry).to_owned() } else { self.paths[entry].clone() });
            for a in &actions {
                match a {
                    ToggleAction::Active { target, on } => {
                        let Some(idx) = self.resolve_path(target) else { continue };
                        let v = if on_values { *on } else { !*on };
                        let key = PropKey { target: target.clone(), property: TrackProperty::Active as u8, name: String::new(), slot: 0 };
                        if let Some((prev, other)) = written.get(&key)
                            && (*prev != 0.0) != v
                        {
                            self.error(format!("{phase}: '{who}' and '{other}' disagree on whether '{target}' starts active"));
                            continue;
                        }
                        written.insert(key, (v as u8 as f32, who.clone()));
                        self.scene.entries[idx].enabled = v;
                    }
                    ToggleAction::ComponentEnabled { target, component, on } => {
                        let Some(idx) = self.resolve_path(target) else { continue };
                        let v = if on_values { *on } else { !*on };
                        let key = PropKey { target: target.clone(), property: TrackProperty::ComponentEnabled as u8, name: component.clone(), slot: 0 };
                        if let Some((prev, other)) = written.get(&key)
                            && (*prev != 0.0) != v
                        {
                            self.error(format!("{phase}: '{who}' and '{other}' disagree on whether {component} on '{target}' starts enabled"));
                            continue;
                        }
                        written.insert(key, (v as u8 as f32, who.clone()));
                        if let Some(c) = self.scene.entries[idx].components.iter_mut().find(|c| c.kind == *component) {
                            c.enabled = v;
                        }
                    }
                    ToggleAction::BlendShape { target, shape, value } if on_values => {
                        let Some(idx) = self.resolve_path(target) else { continue };
                        let key = PropKey { target: target.clone(), property: TrackProperty::BlendShape as u8, name: shape.clone(), slot: 0 };
                        if let Some((prev, other)) = written.get(&key)
                            && (*prev - *value).abs() > 1e-4
                        {
                            self.error(format!("{phase}: '{who}' and '{other}' set blendshape '{shape}' on '{target}' to different resting values"));
                            continue;
                        }
                        written.insert(key, (*value, who.clone()));
                        let e = &mut self.scene.entries[idx];
                        let mut bs = comps::<BlendShapes>(e).into_iter().next().unwrap_or_default();
                        bs.set(shape, *value);
                        set_comp(e, &bs);
                    }
                    ToggleAction::MaterialFloat { target, slot, prop, value, .. } if on_values => {
                        if let Some(t) = self.to_anim(target) {
                            let key = PropKey { target: t, property: TrackProperty::MaterialFloat as u8, name: prop.clone(), slot: *slot };
                            self.keys.insert(key, Some([*value, 0.0, 0.0, 0.0]));
                        }
                    }
                    ToggleAction::MaterialColor { target, slot, prop, value, .. } if on_values => {
                        if let Some(t) = self.to_anim(target) {
                            let key = PropKey { target: t, property: TrackProperty::MaterialColor as u8, name: prop.clone(), slot: *slot };
                            self.keys.insert(key, Some(*value));
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    pub(super) fn resting_state(&mut self) {
        // Everything starts in its off look...
        let all: Vec<(usize, bool)> = (0..self.specs.len()).map(|si| (si, false)).collect();
        self.apply_rest("resting state", &all);
        // ...except toggles that rest on (default on, or the rest logic inverted).
        let rest_on: Vec<(usize, bool)> = (0..self.specs.len()).filter(|&si| self.specs[si].default_on != self.specs[si].invert_rest).map(|si| (si, true)).collect();
        self.apply_rest("default-on state", &rest_on);
    }

    // --------------------------------------------------- 10 defaults layer

    /// The value `key` has in the (resting) scene, or the recorded off value.
    fn rest_value(&self, k: &PropKey, explicit: Option<[f32; 4]>) -> Option<[f32; 4]> {
        let idx = self.resolve_path(&self.from_anim(&k.target))?;
        let e = &self.scene.entries[idx];
        match property_of(k.property) {
            TrackProperty::Active => Some([e.enabled as u8 as f32, 0.0, 0.0, 0.0]),
            TrackProperty::ComponentEnabled => e.components.iter().find(|c| c.kind == k.name).map(|c| [c.enabled as u8 as f32, 0.0, 0.0, 0.0]),
            TrackProperty::BlendShape => Some([self.blend_shape_rest(&self.from_anim(&k.target), &k.name), 0.0, 0.0, 0.0]),
            TrackProperty::MaterialColor | TrackProperty::MaterialFloat | TrackProperty::MaterialOffset => explicit,
            _ => None,
        }
    }

    pub(super) fn defaults_layer(&mut self) {
        let mut clip = AnimClip { length: CONSTANT_CLIP_LEN, looping: false, frame_rate: 60.0, tracks: Vec::new(), events: Vec::new(), ..Default::default() };
        let keys: Vec<(PropKey, Option<[f32; 4]>)> = self.keys.iter().map(|(k, v)| (k.clone(), *v)).collect();
        for (k, explicit) in keys {
            let property = property_of(k.property);
            let Some(v) = self.rest_value(&k, explicit) else {
                if property.uses_slot() {
                    self.warn(format!("no resting value for material property '{}' on '{}'; it keeps the last animated value when its toggle turns off", k.name, k.target));
                }
                continue;
            };
            let interp = if property.is_step() { crate::KeyInterp::Step } else { crate::KeyInterp::Linear };
            let mut t = AnimTrack::prop(&k.target, property, &k.name, k.slot);
            t.keys.push(Keyframe { time: 0.0, value: v, interp, ..Default::default() });
            clip.tracks.push(t);
        }
        if clip.tracks.is_empty() {
            return;
        }
        let rel = format!("{}/defaults.anim", self.out_dir);
        self.clips.insert(rel.clone(), clip);
        let mut machine = StateMachine { default_state: "Defaults".into(), ..Default::default() };
        machine.states.push(AnimState { looping: true, ..state("Defaults", [0.0, 0.0], Motion::Clip(rel)) });
        self.defaults_layer = Some(AnimLayer { name: "Defaults".into(), machine, ..Default::default() });
    }
}

pub(super) fn controller_param(name: &str, kind: AvatarParamKind, default: f32) -> AnimParameter {
    AnimParameter { name: name.into(), kind: anim_kind(kind), default }
}

/// Used by the emit pass to remove authoring components.
pub(super) fn strip(e: &mut SceneEntry, kinds: &[&str]) {
    e.components.retain(|c| !kinds.contains(&c.kind.as_str()));
}
