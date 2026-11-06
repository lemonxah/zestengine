//! Pass 6b `optimise_toggles` (the toggle table) and pass 13a `lints`.

use super::*;

/// What a spec's layer animates, read from its on clip.
type KeySet = BTreeSet<(String, u8, String, u8)>;

impl Builder<'_> {
    /// The properties spec `si`'s on clip writes, when its layer is a plain
    /// Off / On layer that can be folded into a shared one.
    fn foldable_keys(&self, si: usize, layer: &AnimLayer) -> Option<KeySet> {
        let s = &self.specs[si];
        let plain = !s.local
            && !s.mode.is_slider()
            && !s.default_on
            && s.drive_globals.is_empty()
            && !s.actions.iter().any(|a| matches!(a, ToggleAction::SetParam { .. }))
            && layer.machine.states.len() == 2
            && layer.machine.states.iter().all(|st| st.behaviours.is_empty());
        if !plain {
            return None;
        }
        let on = layer.machine.states.iter().find(|st| st.name == "On")?;
        let Motion::Clip(rel) = &on.motion else { return None };
        let clip = self.clips.get(rel)?;
        if clip.tracks.is_empty() {
            return None;
        }
        Some(clip.tracks.iter().map(|t| (t.target.clone(), t.property as u8, t.name.clone(), t.slot)).collect())
    }

    /// The toggle table: Off / On layers that animate exactly the same
    /// properties fold into one layer with an Any State transition per
    /// option, later toggles winning when several are on (as stacked layers
    /// did). The state count grows but the layer count, and so the per-frame
    /// layer evaluation, does not.
    pub(super) fn optimise_toggles(&mut self) {
        let mut groups: BTreeMap<(KeySet, u32, u32), Vec<(usize, usize)>> = BTreeMap::new();
        for &(si, li) in &self.spec_layers {
            let Some(keys) = self.foldable_keys(si, &self.gen_layers[li].0) else { continue };
            let s = &self.specs[si];
            groups.entry((keys, s.fade_in.to_bits(), s.fade_out.to_bits())).or_default().push((si, li));
        }
        let mut drop: BTreeSet<usize> = BTreeSet::new();
        let mut replace: BTreeMap<usize, (AnimLayer, String)> = BTreeMap::new();
        let mut replace_first: Vec<Vec<usize>> = Vec::new();
        for (_, mut members) in groups {
            if members.len() < 2 {
                continue;
            }
            members.sort();
            let (fade_in, fade_out) = (self.specs[members[0].0].fade_in, self.specs[members[0].0].fade_out);
            let mut machine = StateMachine { default_state: "Off".into(), ..Default::default() };
            machine.states.push(state_off());
            let mut names = Vec::new();
            for (n, &(si, li)) in members.iter().enumerate() {
                let on = self.gen_layers[li].0.machine.states.iter().find(|st| st.name == "On").cloned().unwrap_or_default();
                let sname = format!("On {}", self.specs[si].label);
                let mut st = AnimState { name: sname.clone(), position: [260.0, n as f32 * 70.0], ..on };
                st.write_defaults = true;
                machine.states.push(st);
                // Highest-numbered active toggle wins.
                let mut conds = vec![self.specs[si].cond.on(&self.specs[si].param)];
                for &(sj, _) in &members[n + 1..] {
                    conds.push(self.specs[sj].cond.off(&self.specs[sj].param));
                }
                let mut tr = AnimTransition { from: None, to: sname.clone(), conditions: conds, has_exit_time: false, duration: fade_in.max(0.0), ..Default::default() };
                tr.can_transition_to_self = false;
                machine.transitions.push(tr);
                names.push(sname);
            }
            let all_off: Vec<AnimCondition> = members.iter().map(|&(si, _)| self.specs[si].cond.off(&self.specs[si].param)).collect();
            machine.transitions.push(AnimTransition { from: None, to: "Off".into(), conditions: all_off, has_exit_time: false, duration: fade_out.max(0.0), ..Default::default() });
            let first = members.iter().map(|m| m.1).min().unwrap_or(members[0].1);
            let name = format!("{} (+{} merged)", self.gen_layers[first].0.name, members.len() - 1);
            let source = members.iter().map(|&(_, li)| self.gen_layers[li].1.clone()).collect::<Vec<_>>().join(" + ");
            for &(_, li) in &members {
                drop.insert(li);
            }
            replace_first.push(members.iter().map(|&(_, li)| li).collect::<Vec<_>>());
            replace_first.last_mut().unwrap().sort();
            replace.insert(first, (AnimLayer { name, machine, ..Default::default() }, source));
            self.out.layers_merged += members.len() - 1;
        }
        if replace.is_empty() {
            return;
        }
        let old = std::mem::take(&mut self.gen_layers);
        let mut remap: Vec<Option<usize>> = vec![None; old.len()];
        for (li, l) in old.into_iter().enumerate() {
            if let Some(r) = replace.remove(&li) {
                remap[li] = Some(self.gen_layers.len());
                self.gen_layers.push(r);
            } else if !drop.contains(&li) {
                remap[li] = Some(self.gen_layers.len());
                self.gen_layers.push(l);
            }
        }
        // Folded specs now point at the layer that replaced their group.
        let mut home: BTreeMap<usize, usize> = BTreeMap::new();
        for &(_, li) in &self.spec_layers {
            let first = replace_first.iter().find(|g| g.contains(&li)).map_or(li, |g| g[0]);
            home.insert(li, first);
        }
        for e in &mut self.spec_layers {
            e.1 = remap[home[&e.1]].unwrap_or(usize::MAX);
        }
    }

    // ----------------------------------------------------------------- lints

    /// Authoring mistakes the other passes tolerate. They are warnings with a
    /// `lint:` prefix so the inspector can group them.
    pub(super) fn lints(&mut self) {
        // Two toggles on different parameters writing one property.
        let mut writers: BTreeMap<(String, u8, String, u8), Vec<usize>> = BTreeMap::new();
        for si in 0..self.specs.len() {
            let s = &self.specs[si];
            if s.local {
                continue;
            }
            for a in &s.actions {
                let key = match a {
                    ToggleAction::Active { target, .. } => (target.clone(), TrackProperty::Active as u8, String::new(), 0),
                    ToggleAction::ComponentEnabled { target, component, .. } => (target.clone(), TrackProperty::ComponentEnabled as u8, component.clone(), 0),
                    ToggleAction::BlendShape { target, shape, .. } => (target.clone(), TrackProperty::BlendShape as u8, shape.clone(), 0),
                    ToggleAction::MaterialFloat { target, slot, prop, .. } => (target.clone(), TrackProperty::MaterialFloat as u8, prop.clone(), *slot),
                    ToggleAction::MaterialColor { target, slot, prop, .. } => (target.clone(), TrackProperty::MaterialColor as u8, prop.clone(), *slot),
                    _ => continue,
                };
                let w = writers.entry(key).or_default();
                if !w.contains(&si) {
                    w.push(si);
                }
            }
        }
        let mut msgs = Vec::new();
        for ((target, prop, name, _), list) in &writers {
            let params: BTreeSet<&str> = list.iter().map(|&si| self.specs[si].param.as_str()).collect();
            if params.len() > 1 {
                let who: Vec<&str> = list.iter().map(|&si| self.specs[si].label.as_str()).collect();
                let what = if name.is_empty() { format!("{:?}", property_of(*prop)) } else { format!("{:?} '{name}'", property_of(*prop)) };
                msgs.push(format!("lint: {} all write {what} on '{target}'; the last one that is on wins", who.join(", ")));
            }
        }
        // A default of 0 on a Toggle-mode Int selector is fine; a Bool whose
        // only reader is a menu item with value 0 is not.
        for it in self.menu_values_zero() {
            msgs.push(format!("lint: menu item '{it}' sets its parameter to 0, which is also its off value"));
        }
        let max = self.root_budget.min(self.env.budget_cap);
        let used = ParamBudget::of(&self.params, max).used;
        if max > 0 && used * 10 >= max * 9 && used <= max {
            msgs.push(format!("lint: the avatar uses {used} of {max} synced bits; one more toggle may not fit"));
        }
        self.out.warnings.extend(msgs);
    }

    fn menu_values_zero(&self) -> Vec<String> {
        let mut v = Vec::new();
        self.out.menu.visit(&mut |n, _| {
            if matches!(n.item.kind, MenuItemKind::Toggle) && !n.item.parameter.is_empty() && n.item.value == 0.0 {
                v.push(n.item.label.clone());
            }
        });
        v
    }
}

fn state_off() -> AnimState {
    AnimState { name: "Off".into(), motion: Motion::default(), looping: false, write_defaults: true, position: [0.0, 0.0], ..Default::default() }
}

/// Parameters of the avatar's synced budget grouped for the inspector: what
/// each costs, biggest first. Pure, for display and tests.
pub fn budget_breakdown(params: &[AvatarParam]) -> Vec<(String, u32)> {
    let mut v: Vec<(String, u32)> = params.iter().filter(|p| p.is_synced()).map(|p| (p.name.clone(), p.kind.bits())).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v
}
