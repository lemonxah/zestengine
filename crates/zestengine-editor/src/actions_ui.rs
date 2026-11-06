//! Editing event/action bindings (`zestengine_core::actions`): one shared
//! widget for every event source (Interactable, Trigger Events, UI Button,
//! animation events).
//!
//! The widget needs to know what the other objects in the scene expose, so
//! the editor publishes an [`ActionCatalog`] into egui memory each frame it
//! draws an object inspector ([`publish_catalog`]). The choices the dropdowns
//! offer and the checks the rows show are plain functions over the catalog, so
//! they are unit tested without a GUI.

use std::sync::{Arc, Mutex};

use egui::{DragValue, RichText, Ui};
use egui_phosphor::regular as ph;
use zestengine_core::{
    ActionDesc, ActionType, ArgSource, ArgValue, Component, EventActions, EventBinding, InstanceSel, ObjectId, ObjectRef,
    OBJECT_ACTIONS, OBJECT_COMPONENT, OBJECT_PROPERTIES, ParamDesc, PropertyDesc,
};

use crate::components::InspectCtx;
use crate::icons::Icon;
use crate::{theme, widgets};

/// egui-memory key under which the editor publishes the `Arc<ActionCatalog>`.
pub const ACTION_CATALOG_KEY: &str = "zestengine-action-catalog";
/// Optional `bool` in egui memory: only offer actions marked UGC-safe. The
/// editor sets it ([`set_ugc_only`]) for objects that came from a content
/// package and while the Content Builder is open (see `editor_app.rs`).
pub const ACTION_UGC_ONLY_KEY: &str = "zestengine-action-ugc-only";
/// egui-memory key of the `Arc<Mutex<Vec<ActionTestRequest>>>` the Test buttons
/// fill and the Play loop drains ([`take_test_requests`]).
pub const ACTION_TEST_KEY: &str = "zestengine-action-test";

/// A Test button press: fire `binding` as if `owner`'s event source ran it.
pub struct ActionTestRequest {
    /// The inspected object (the event source's owner; an unset target means it).
    pub owner: ObjectId,
    pub binding: EventBinding,
    /// Where the result line for this row is stored ([`post_test_result`]).
    pub row: egui::Id,
}

/// Mark the inspector as editing user-generated content (hides actions that
/// are not `ugc_safe`).
pub fn set_ugc_only(ui: &Ui, on: bool) {
    ui.data_mut(|d| d.insert_temp(egui::Id::new(ACTION_UGC_ONLY_KEY), on));
}

/// Requests queued by Test buttons since the last call.
pub fn take_test_requests(ctx: &egui::Context) -> Vec<ActionTestRequest> {
    let queue = ctx.data(|d| d.get_temp::<Arc<Mutex<Vec<ActionTestRequest>>>>(egui::Id::new(ACTION_TEST_KEY)));
    queue.map(|q| std::mem::take(&mut *q.lock().unwrap_or_else(|e| e.into_inner()))).unwrap_or_default()
}

/// Show `text` under the Test button of the row that asked.
pub fn post_test_result(ctx: &egui::Context, row: egui::Id, ok: bool, text: String) {
    ctx.data_mut(|d| d.insert_temp(row.with("test-result"), (ok, text)));
}

/// What one component on one object exposes.
#[derive(Clone)]
pub struct ComponentApi {
    pub name: &'static str,
    /// Which of the object's components of this type (0 = first).
    pub instance: u32,
    /// How many components of this type the object has.
    pub of: u32,
    /// [`Component::instance_label`] (a Wasm Script's module name).
    pub label: String,
    pub actions: &'static [ActionDesc],
    pub properties: &'static [PropertyDesc],
}

/// One entry of the component dropdown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentChoice {
    pub name: String,
    pub instance: u32,
    pub label: String,
    /// The object has several components of this type, so the entry is numbered.
    pub numbered: bool,
}

impl ComponentChoice {
    /// "Health", or "Wasm Script #2 (door)" when there are several.
    pub fn text(&self) -> String {
        match (self.numbered, self.label.is_empty()) {
            (false, _) => self.name.clone(),
            (true, true) => format!("{} #{}", self.name, self.instance + 1),
            (true, false) => format!("{} #{} ({})", self.name, self.instance + 1, self.label),
        }
    }

    /// Whether this entry is the component `b` points at.
    pub fn is(&self, b: &EventBinding) -> bool {
        self.name == b.component && (!self.numbered || self.instance == b.instance)
    }
}

/// The scene's action surface: per object, the components that expose
/// anything. Built once per inspector frame.
#[derive(Default)]
pub struct ActionCatalog {
    /// The object being inspected (what an unset target or property object means).
    pub owner: ObjectId,
    pub objects: Vec<(ObjectId, Vec<ComponentApi>)>,
    /// The editor is in Play mode (the Test button works).
    pub playing: bool,
}

impl ActionCatalog {
    pub fn build<'a>(owner: ObjectId, scene: impl IntoIterator<Item = (ObjectId, &'a [Box<dyn Component>])>) -> Self {
        let objects = scene
            .into_iter()
            .filter_map(|(id, components)| {
                let apis: Vec<ComponentApi> = components
                    .iter()
                    .filter(|c| !c.actions().is_empty() || !c.properties().is_empty())
                    .map(|c| {
                        #[allow(clippy::borrowed_box)] // filter hands each item by reference
                        let same = |o: &&Box<dyn Component>| o.type_name() == c.type_name();
                        let before = components.iter().take_while(|o| !std::ptr::addr_eq(o.as_ref(), c.as_ref())).filter(same).count();
                        ComponentApi {
                            name: c.type_name(),
                            instance: before as u32,
                            of: components.iter().filter(same).count() as u32,
                            label: c.instance_label().unwrap_or_default(),
                            actions: c.actions(),
                            properties: c.properties(),
                        }
                    })
                    .collect();
                (!apis.is_empty()).then_some((id, apis))
            })
            .collect();
        Self { owner, objects, playing: false }
    }

    /// Mark the editor as playing (enables the Test button).
    pub fn with_playing(mut self, playing: bool) -> Self {
        self.playing = playing;
        self
    }

    fn components_of(&self, object: ObjectId) -> &[ComponentApi] {
        self.objects.iter().find(|(id, _)| *id == object).map(|(_, c)| c.as_slice()).unwrap_or(&[])
    }

    /// The component `sel` means among the target's components named `name`
    /// (the same choice the runtime makes: a matching label wins, then the index).
    fn api(&self, object: ObjectId, name: &str, sel: InstanceSel) -> Option<&ComponentApi> {
        let mut by_index = None;
        for api in self.components_of(object).iter().filter(|a| a.name == name) {
            if !sel.label.is_empty() && api.label == sel.label {
                return Some(api);
            }
            if api.instance as usize == sel.index {
                by_index = Some(api);
            }
        }
        by_index
    }

    /// The object a binding/source with `target` refers to.
    fn resolve(&self, target: ObjectRef) -> ObjectId {
        target.id().unwrap_or(self.owner)
    }

    /// Every component name that has properties anywhere (for instigator
    /// sources, whose object is only known at run time).
    fn property_components(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = Vec::new();
        for (_, apis) in &self.objects {
            for api in apis.iter().filter(|a| !a.properties.is_empty()) {
                if !names.contains(&api.name) {
                    names.push(api.name);
                }
            }
        }
        names.sort_unstable();
        names
    }
}

/// Publish the catalog for the widgets drawn this frame.
pub fn publish_catalog(ui: &Ui, catalog: ActionCatalog) {
    ui.data_mut(|d| d.insert_temp(egui::Id::new(ACTION_CATALOG_KEY), Arc::new(catalog)));
}

// ------------------------------------------------------------------ choices

/// Components of the target that expose at least one action (that the context
/// may use), the built-in "Object" first. Several components of one type are
/// numbered ("Wasm Script #2 (door)").
pub fn component_choices(cat: &ActionCatalog, target: ObjectRef, ugc_only: bool) -> Vec<ComponentChoice> {
    let mut out = vec![ComponentChoice { name: OBJECT_COMPONENT.to_owned(), instance: 0, label: String::new(), numbered: false }];
    out.extend(cat.components_of(cat.resolve(target)).iter().filter(|c| c.actions.iter().any(|a| !ugc_only || a.ugc_safe)).map(|c| {
        ComponentChoice { name: c.name.to_owned(), instance: c.instance, label: c.label.clone(), numbered: c.of > 1 }
    }));
    out
}

/// The actions a component offers on the target (its first instance).
pub fn action_choices(cat: &ActionCatalog, target: ObjectRef, component: &str, ugc_only: bool) -> Vec<&'static ActionDesc> {
    action_choices_in(cat, target, component, InstanceSel::FIRST, ugc_only)
}

/// [`action_choices`] for the component instance `sel` picks.
pub fn action_choices_in(
    cat: &ActionCatalog,
    target: ObjectRef,
    component: &str,
    sel: InstanceSel,
    ugc_only: bool,
) -> Vec<&'static ActionDesc> {
    let all: &'static [ActionDesc] = if component.is_empty() || component == OBJECT_COMPONENT {
        OBJECT_ACTIONS
    } else {
        cat.api(cat.resolve(target), component, sel).map(|c| c.actions).unwrap_or(&[])
    };
    all.iter().filter(|a| !ugc_only || a.ugc_safe).collect()
}

/// Properties readable from a component of `object` ("Object" = the object itself).
pub fn property_choices(cat: &ActionCatalog, object: ObjectRef, component: &str) -> &'static [PropertyDesc] {
    if component.is_empty() || component == OBJECT_COMPONENT {
        OBJECT_PROPERTIES
    } else {
        cat.components_of(cat.resolve(object)).iter().find(|c| c.name == component).map(|c| c.properties).unwrap_or(&[])
    }
}

fn instigator_property_choices(cat: &ActionCatalog, component: &str) -> &'static [PropertyDesc] {
    if component.is_empty() || component == OBJECT_COMPONENT {
        return OBJECT_PROPERTIES;
    }
    cat.objects.iter().flat_map(|(_, apis)| apis).find(|a| a.name == component).map(|a| a.properties).unwrap_or(&[])
}

/// The value a binding's argument source produces, if it can be told.
pub fn source_type(
    src: &ArgSource,
    cat: &ActionCatalog,
    payload: &[(&'static str, ActionType)],
) -> Option<ActionType> {
    let property = |props: &[PropertyDesc], component: &str, name: &str| -> Option<ActionType> {
        if name.is_empty() && (component.is_empty() || component == OBJECT_COMPONENT) {
            return Some(ActionType::Object);
        }
        props.iter().find(|p| p.name == name).map(|p| p.ty)
    };
    match src {
        ArgSource::Const(v) => Some(v.ty()),
        ArgSource::Property { object, component, property: name } => {
            property(property_choices(cat, *object, component), component, name)
        }
        ArgSource::Instigator { component, property: name } => {
            property(instigator_property_choices(cat, component), component, name)
        }
        ArgSource::Payload(field) => payload.iter().find(|(n, _)| n == field).map(|(_, t)| *t),
    }
}

/// Whether a value of type `from` can feed a parameter of type `to`.
pub fn convertible(from: ActionType, to: ActionType) -> bool {
    ArgValue::zero(from).coerce(to).is_ok()
}

/// Everything wrong with a binding as configured (empty = fine). The runtime
/// reports the same problems when it fires; this shows them up front.
pub fn problems(
    b: &EventBinding,
    cat: &ActionCatalog,
    payload: &[(&'static str, ActionType)],
    ugc_only: bool,
) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(id) = b.target.id()
        && !cat.objects.iter().any(|(o, _)| *o == id)
        && b.component != OBJECT_COMPONENT
        && !b.component.is_empty()
    {
        out.push("The target has no components that expose actions.".into());
        return out;
    }
    if b.action.is_empty() {
        out.push("Pick an action.".into());
        return out;
    }
    let choices = action_choices_in(cat, b.target, &b.component, b.sel(), false);
    let Some(desc) = choices.iter().find(|a| a.name == b.action) else {
        out.push(format!("{} has no action '{}' on the target.", b.component, b.action));
        return out;
    };
    if ugc_only && !desc.ugc_safe {
        out.push("Not allowed in user-generated content.".into());
    }
    for (i, p) in desc.params.iter().enumerate() {
        let Some(src) = b.args.get(i) else {
            continue; // the default is used
        };
        match source_type(src, cat, payload) {
            Some(t) if !convertible(t, p.ty) => {
                out.push(format!("'{}' expects {}, but gets {}.", p.name, p.ty.label(), t.label()))
            }
            None => out.push(format!("'{}': the source does not resolve.", p.name)),
            _ => {}
        }
    }
    out
}

/// Point a binding at a component (its first instance): picks its first action.
pub fn select_component(b: &mut EventBinding, component: &str, cat: &ActionCatalog, ugc_only: bool) {
    select_choice(b, &ComponentChoice { name: component.to_owned(), instance: 0, label: String::new(), numbered: false }, cat, ugc_only);
}

/// Point a binding at one dropdown entry (component and instance): picks its first action.
pub fn select_choice(b: &mut EventBinding, choice: &ComponentChoice, cat: &ActionCatalog, ugc_only: bool) {
    b.component = choice.name.clone();
    b.instance = if choice.numbered { choice.instance } else { 0 };
    b.instance_label = if choice.numbered { choice.label.clone() } else { String::new() };
    match action_choices_in(cat, b.target, &choice.name, b.sel(), ugc_only).first() {
        Some(first) => select_action(b, first),
        None => {
            b.action.clear();
            b.args.clear();
        }
    }
}

/// Pick an action and reset its arguments to constants of the declared defaults.
pub fn select_action(b: &mut EventBinding, desc: &ActionDesc) {
    b.action = desc.name.to_owned();
    b.args = desc.params.iter().map(|p| ArgSource::Const(p.default.value())).collect();
}

/// Switch a parameter's source kind, keeping nothing that doesn't apply.
pub fn switch_source(kind: SourceKind, param: &ParamDesc, payload: &[(&'static str, ActionType)]) -> ArgSource {
    match kind {
        SourceKind::Value => ArgSource::Const(param.default.value()),
        SourceKind::Property => ArgSource::Property { object: ObjectRef::NONE, component: String::new(), property: String::new() },
        SourceKind::Instigator => ArgSource::Instigator { component: String::new(), property: String::new() },
        SourceKind::Event => ArgSource::Payload(
            payload.iter().find(|(_, t)| convertible(*t, param.ty)).map(|(n, _)| (*n).to_owned()).unwrap_or_default(),
        ),
    }
}

/// The four ways a parameter gets its value.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    Value,
    Property,
    Instigator,
    Event,
}

impl SourceKind {
    const ALL: [SourceKind; 4] = [Self::Value, Self::Property, Self::Instigator, Self::Event];

    pub fn of(src: &ArgSource) -> Self {
        match src {
            ArgSource::Const(_) => Self::Value,
            ArgSource::Property { .. } => Self::Property,
            ArgSource::Instigator { .. } => Self::Instigator,
            ArgSource::Payload(_) => Self::Event,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Value => "Value",
            Self::Property => "Property",
            Self::Instigator => "Instigator",
            Self::Event => "Event",
        }
    }
}

/// Move entry `i` by `delta` places; false if it would leave the list.
pub fn move_binding(list: &mut EventActions, i: usize, delta: isize) -> bool {
    let j = i as isize + delta;
    if i >= list.len() || j < 0 || j as usize >= list.len() {
        return false;
    }
    list.swap(i, j as usize);
    true
}

// ------------------------------------------------------------------ widget

/// Edit a list of bindings. `payload` names the fields the event offers to
/// "Event" sources. Returns true when anything changed.
pub fn event_actions_ui(
    ui: &mut Ui,
    ctx: &InspectCtx,
    id: &str,
    title: &str,
    list: &mut EventActions,
    payload: &[(&'static str, ActionType)],
) -> bool {
    let cat: Arc<ActionCatalog> =
        ui.data(|d| d.get_temp::<Arc<ActionCatalog>>(egui::Id::new(ACTION_CATALOG_KEY))).unwrap_or_default();
    let ugc_only = ui.data(|d| d.get_temp::<bool>(egui::Id::new(ACTION_UGC_ONLY_KEY))).unwrap_or(false);
    let mut changed = false;
    ui.label(RichText::new(title).strong());
    let mut remove = None;
    let mut swap = None;
    let mut test = None;
    let count = list.len();
    for (k, b) in list.iter_mut().enumerate() {
        egui::Frame::new()
            .fill(theme::BG_APP)
            .corner_radius(egui::CornerRadius::same(theme::R_ROW))
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.push_id((id, k), |ui| {
                    let row = ui.id();
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("{}", k + 1)).font(theme::mono(11.0)).color(theme::TEXT_DIM));
                        changed |= ctx.object_ref(ui, "Target", &mut b.target);
                        if !b.target.is_set() {
                            ui.label(RichText::new("this object").small().weak());
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if widgets::icon_button(ui, Icon::Close, "Remove action", false).clicked() {
                                remove = Some(k);
                            }
                            if k + 1 < count && ui.small_button(ph::ARROW_DOWN).on_hover_text("Move down").clicked() {
                                swap = Some((k, 1));
                            }
                            if k > 0 && widgets::icon_button(ui, Icon::ArrowUp, "Move up", false).clicked() {
                                swap = Some((k, -1));
                            }
                            // Fires this binding in the running game (Play mode only).
                            let tip = if cat.playing { "Test: run this action now" } else { "Enter Play mode to test this action" };
                            if ui.add_enabled_ui(cat.playing, |ui| widgets::icon_button(ui, Icon::Play, tip, false)).inner.clicked() {
                                test = Some(ActionTestRequest { owner: cat.owner, binding: b.clone(), row });
                            }
                        });
                    });
                    changed |= binding_ui(ui, ctx, &cat, b, payload, ugc_only);
                    if let Some((ok, text)) = ui.data(|d| d.get_temp::<(bool, String)>(row.with("test-result"))) {
                        let color = if ok { theme::STATUS_OK } else { theme::STATUS_ERROR };
                        ui.label(RichText::new(text).small().color(color));
                    }
                });
            });
        ui.add_space(4.0);
    }
    if let Some(request) = test {
        let queue = ui.data_mut(|d| {
            d.get_temp_mut_or_insert_with(egui::Id::new(ACTION_TEST_KEY), || Arc::new(Mutex::new(Vec::<ActionTestRequest>::new())))
                .clone()
        });
        queue.lock().unwrap_or_else(|e| e.into_inner()).push(request);
    }
    if let Some(k) = remove {
        list.remove(k);
        changed = true;
    }
    if let Some((k, d)) = swap {
        changed |= move_binding(list, k, d);
    }
    if widgets::dashed_button(ui, Some(Icon::Plus), "Add action").clicked() {
        let mut b = EventBinding::default();
        select_component(&mut b, OBJECT_COMPONENT, &cat, ugc_only);
        list.push(b);
        changed = true;
    }
    changed
}

fn binding_ui(
    ui: &mut Ui,
    ctx: &InspectCtx,
    cat: &ActionCatalog,
    b: &mut EventBinding,
    payload: &[(&'static str, ActionType)],
    ugc_only: bool,
) -> bool {
    let mut changed = false;
    // Component dropdown.
    let components = component_choices(cat, b.target, ugc_only);
    let name = if b.component.is_empty() { OBJECT_COMPONENT } else { b.component.as_str() };
    let current_choice = components.iter().find(|c| c.is(b));
    let shown = match current_choice {
        Some(c) => c.text(),
        None if b.instance > 0 => format!("{name} #{}", b.instance + 1),
        None => name.to_owned(),
    };
    let mut pick = None;
    widgets::property_row(ui, "Component", |ui| {
        egui::ComboBox::from_id_salt("component").selected_text(shown.clone()).width(180.0).show_ui(ui, |ui| {
            widgets::menu_frame(ui);
            for choice in &components {
                if ui.selectable_label(choice.is(b), choice.text()).clicked() {
                    pick = Some(choice.clone());
                }
            }
            if current_choice.is_none() {
                ui.label(RichText::new(format!("{shown} (not on the target)")).weak());
            }
        });
    });
    if let Some(choice) = pick
        && !choice.is(b)
    {
        select_choice(b, &choice, cat, ugc_only);
        changed = true;
    }
    // Action dropdown.
    let actions = action_choices_in(cat, b.target, &b.component, b.sel(), ugc_only);
    let current = actions.iter().find(|a| a.name == b.action).copied();
    let action_text = match current {
        Some(a) => a.label.to_owned(),
        None if b.action.is_empty() => "(pick an action)".to_owned(),
        None => format!("{} (missing)", b.action),
    };
    let mut pick = None;
    widgets::property_row(ui, "Function", |ui| {
        egui::ComboBox::from_id_salt("action").selected_text(action_text).width(180.0).show_ui(ui, |ui| {
            widgets::menu_frame(ui);
            for a in &actions {
                if ui.selectable_label(a.name == b.action, a.label).clicked() {
                    pick = Some(*a);
                }
            }
        });
    });
    if let Some(a) = pick
        && a.name != b.action
    {
        select_action(b, a);
        changed = true;
    }
    // Parameters.
    if let Some(desc) = current {
        // A file saved with fewer arguments than the action has parameters
        // (or an action that gained one): fill in the declared defaults.
        if b.args.len() < desc.params.len() {
            let have = b.args.len();
            b.args.extend(desc.params[have..].iter().map(|p| ArgSource::Const(p.default.value())));
            changed = true;
        }
        for (i, p) in desc.params.iter().enumerate() {
            changed |= param_ui(ui, ctx, cat, i, p, &mut b.args[i], payload);
        }
    }
    for msg in problems(b, cat, payload, ugc_only) {
        ui.label(RichText::new(msg).small().color(theme::STATUS_WARN_FG));
    }
    changed
}

fn param_ui(
    ui: &mut Ui,
    ctx: &InspectCtx,
    cat: &ActionCatalog,
    index: usize,
    p: &ParamDesc,
    src: &mut ArgSource,
    payload: &[(&'static str, ActionType)],
) -> bool {
    let mut changed = false;
    let kind = SourceKind::of(src);
    let mut new_kind = None;
    ui.push_id(("param", index), |ui| {
        widgets::property_row(ui, format!("{} ({})", p.name, p.ty.label()), |ui| {
            egui::ComboBox::from_id_salt("source").selected_text(kind.label()).width(90.0).show_ui(ui, |ui| {
                widgets::menu_frame(ui);
                for k in SourceKind::ALL {
                    if ui.selectable_label(k == kind, k.label()).clicked() && k != kind {
                        new_kind = Some(k);
                    }
                }
            });
            if kind == SourceKind::Value {
                changed |= value_editor(ui, ctx, p.ty, src);
            }
        });
    });
    if let Some(k) = new_kind {
        *src = switch_source(k, p, payload);
        changed = true;
    }
    match src {
        ArgSource::Const(_) => {}
        ArgSource::Property { object, component, property } => {
            ui.indent(("prop", index), |ui| {
                changed |= ctx.object_ref(ui, "Object", object);
                changed |= name_combo(ui, "Component", "pc", component, component_names(cat, *object));
                let props = property_choices(cat, *object, component);
                changed |= property_combo(ui, "Property", "pp", property, props);
            });
        }
        ArgSource::Instigator { component, property } => {
            ui.indent(("inst", index), |ui| {
                let names = std::iter::once(OBJECT_COMPONENT).chain(cat.property_components()).collect();
                changed |= name_combo(ui, "Component", "ic", component, names);
                changed |= property_combo(ui, "Property", "ip", property, instigator_property_choices(cat, component));
            });
        }
        ArgSource::Payload(field) => {
            ui.indent(("payload", index), |ui| {
                let names: Vec<&str> = payload.iter().map(|(n, _)| *n).collect();
                changed |= name_combo(ui, "Field", "pf", field, names);
            });
        }
    }
    changed
}

/// "Object" plus the components of the object that expose properties.
fn component_names(cat: &ActionCatalog, object: ObjectRef) -> Vec<&'static str> {
    std::iter::once(OBJECT_COMPONENT)
        .chain(cat.components_of(cat.resolve(object)).iter().filter(|c| !c.properties.is_empty()).map(|c| c.name))
        .collect()
}

fn name_combo(ui: &mut Ui, label: &str, salt: &str, value: &mut String, names: Vec<&str>) -> bool {
    let mut changed = false;
    let shown = if value.is_empty() && salt != "pf" { OBJECT_COMPONENT.to_owned() } else { value.clone() };
    widgets::property_row(ui, label, |ui| {
        egui::ComboBox::from_id_salt(salt).selected_text(shown.clone()).width(160.0).show_ui(ui, |ui| {
            widgets::menu_frame(ui);
            for n in names {
                if ui.selectable_label(shown == n, n).clicked() {
                    let new = if n == OBJECT_COMPONENT && salt != "pf" { String::new() } else { n.to_owned() };
                    if *value != new {
                        *value = new;
                        changed = true;
                    }
                }
            }
        });
    });
    changed
}

fn property_combo(ui: &mut Ui, label: &str, salt: &str, value: &mut String, props: &[PropertyDesc]) -> bool {
    let mut changed = false;
    let shown = if value.is_empty() { "(the object itself)".to_owned() } else { value.clone() };
    widgets::property_row(ui, label, |ui| {
        egui::ComboBox::from_id_salt(salt).selected_text(shown).width(160.0).show_ui(ui, |ui| {
            widgets::menu_frame(ui);
            if ui.selectable_label(value.is_empty(), "(the object itself)").clicked() && !value.is_empty() {
                value.clear();
                changed = true;
            }
            for p in props {
                if ui.selectable_label(*value == p.name, format!("{} ({})", p.name, p.ty.label())).clicked() && *value != p.name {
                    *value = p.name.to_owned();
                    changed = true;
                }
            }
        });
    });
    changed
}

/// A typed editor for a constant. The constant is first brought to the
/// parameter's type, so a retyped action never shows the wrong editor.
fn value_editor(ui: &mut Ui, ctx: &InspectCtx, ty: ActionType, src: &mut ArgSource) -> bool {
    let ArgSource::Const(v) = src else {
        return false;
    };
    let mut changed = false;
    if v.ty() != ty {
        *v = v.coerce(ty).unwrap_or_else(|_| ArgValue::zero(ty));
        changed = true;
    }
    match v {
        ArgValue::Bool(b) => changed |= ui.checkbox(b, "").changed(),
        ArgValue::Int(i) => changed |= ui.add(DragValue::new(i)).changed(),
        ArgValue::Float(f) => changed |= ui.add(DragValue::new(f).speed(0.05)).changed(),
        ArgValue::String(s) => changed |= ui.add(egui::TextEdit::singleline(s).desired_width(ui.available_width().max(60.0))).changed(),
        ArgValue::Vec3(a) => changed |= widgets::vec3_field(ui, a, 2, 0.05),
        ArgValue::Object(o) => changed |= ctx.object_ref(ui, "", o),
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_core::{Health, LightComponent, TypedComponent, UiText};

    fn world() -> (ActionCatalog, ObjectId, ObjectId) {
        let (owner, other) = (ObjectId::new(), ObjectId::new());
        let on_owner: Vec<Box<dyn Component>> = vec![Box::new(zestengine_core::Interactable::default())];
        let on_other: Vec<Box<dyn Component>> =
            vec![Box::new(Health::default()), Box::new(LightComponent::default()), Box::new(UiText::default())];
        let cat = ActionCatalog::build(owner, [(owner, on_owner.as_slice()), (other, on_other.as_slice())]);
        (cat, owner, other)
    }

    fn texts(cat: &ActionCatalog, target: ObjectRef) -> Vec<String> {
        component_choices(cat, target, false).iter().map(ComponentChoice::text).collect()
    }

    #[test]
    fn several_components_of_a_type_are_numbered_and_addressable() {
        let (a, b_) = (ObjectId::new(), ObjectId::new());
        let on_a: Vec<Box<dyn Component>> = vec![Box::new(Health::default())];
        let on_b: Vec<Box<dyn Component>> = vec![
            Box::new(zestengine_core::WasmScript::new("scripts/door.wasm")),
            Box::new(Health::default()),
            Box::new(zestengine_core::WasmScript::new("scripts/lamp.wasm")),
        ];
        let cat = ActionCatalog::build(a, [(a, on_a.as_slice()), (b_, on_b.as_slice())]);
        // Wasm Scripts here expose nothing (no module loaded) so only Health shows.
        assert_eq!(texts(&cat, b_.into()), ["Object", "Health"]);
        // Same-named components: numbered, with the module name as the label.
        let two: Vec<Box<dyn Component>> = vec![
            Box::new(Health::default()),
            Box::new(Health::default()),
        ];
        let cat = ActionCatalog::build(a, [(a, two.as_slice())]);
        assert_eq!(texts(&cat, ObjectRef::NONE), ["Object", "Health #1", "Health #2"]);
        let mut b = EventBinding::default();
        let second = component_choices(&cat, ObjectRef::NONE, false)[2].clone();
        select_choice(&mut b, &second, &cat, false);
        assert_eq!((b.component.as_str(), b.instance, b.action.as_str()), ("Health", 1, "Damage"));
        assert!(second.is(&b) && !component_choices(&cat, ObjectRef::NONE, false)[1].is(&b));
        assert_eq!(action_choices_in(&cat, ObjectRef::NONE, "Health", b.sel(), false).len(), 4);
        // An instance that is not there offers nothing and is reported.
        b.instance = 4;
        assert!(problems(&b, &cat, &[], false).iter().any(|m| m.contains("no action")));
        // Labels in the choice text.
        let c = ComponentChoice { name: "Wasm Script".into(), instance: 1, label: "door".into(), numbered: true };
        assert_eq!(c.text(), "Wasm Script #2 (door)");
    }

    #[test]
    fn dropdowns_follow_the_target() {
        let (cat, owner, other) = world();
        // An unset target is the owner object.
        assert_eq!(texts(&cat, ObjectRef::NONE), ["Object", "Interactable"]);
        assert_eq!(texts(&cat, other.into()), ["Object", "Health", "Light", "UI Text"]);
        let _ = owner;
        let health: Vec<_> = action_choices(&cat, other.into(), "Health", false).iter().map(|a| a.name).collect();
        assert_eq!(health, ["Damage", "Heal", "Revive", "SetInvulnerable"]);
        // Built-in object actions need no component.
        assert!(action_choices(&cat, other.into(), "Object", false).iter().any(|a| a.name == "SetActive"));
        assert!(action_choices(&cat, other.into(), "Nope", false).is_empty());
    }

    #[test]
    fn wasm_components_appear_in_the_dropdowns_with_their_declared_surface() {
        let manifest: String = r#"(name: "Door", actions: [(name: "Open", label: "Open door", ugc_safe: true, params: [(name: "speed", ty: Float)]), (name: "Smash")], properties: [(name: "state", ty: Int)])"#
            .bytes()
            .map(|b| format!("\\{b:02x}"))
            .collect();
        let wasm = wat::parse_str(format!(
            r#"(module (@custom "zest_manifest" "{manifest}") (memory (export "memory") 1)
               (func (export "zest_alloc") (param i32) (result i32) i32.const 64)
               (func (export "zest_invoke") (param i32 i32 i32) (result i32) i32.const 0)
               (func (export "zest_read_property") (param i32) (result i64) i64.const 0))"#
        ))
        .unwrap();
        let path = std::env::temp_dir().join(format!("zest-door-{}.wasm", ObjectId::new()));
        std::fs::write(&path, wasm).unwrap();
        let (owner, door) = (ObjectId::new(), ObjectId::new());
        let on_owner: Vec<Box<dyn Component>> = vec![Box::new(zestengine_core::Interactable::default())];
        let on_door: Vec<Box<dyn Component>> = vec![Box::new(zestengine_core::WasmScript::new(path.to_string_lossy()))];
        let cat = ActionCatalog::build(owner, [(owner, on_owner.as_slice()), (door, on_door.as_slice())]);
        assert_eq!(texts(&cat, door.into()), ["Object", "Wasm Script"]);
        let all: Vec<_> = action_choices(&cat, door.into(), "Wasm Script", false).iter().map(|a| (a.name, a.label)).collect();
        assert_eq!(all, [("Open", "Open door"), ("Smash", "Smash")]);
        let safe: Vec<_> = action_choices(&cat, door.into(), "Wasm Script", true).iter().map(|a| a.name).collect();
        assert_eq!(safe, ["Open"], "UGC editing offers only ugc_safe module actions");
        assert_eq!(property_choices(&cat, door.into(), "Wasm Script")[0].name, "state");
        // Picking one fills the declared parameters.
        let mut b = EventBinding { target: door.into(), ..Default::default() };
        select_component(&mut b, "Wasm Script", &cat, false);
        assert_eq!((b.component.as_str(), b.action.as_str(), b.args.len()), ("Wasm Script", "Open", 1));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn ugc_context_hides_unsafe_actions() {
        let (cat, _, other) = world();
        let names: Vec<_> = action_choices(&cat, other.into(), "Health", true).iter().map(|a| a.name).collect();
        assert!(!names.contains(&"Damage") && names.contains(&"Heal"));
        assert!(!action_choices(&cat, other.into(), "Object", true).iter().any(|a| a.name == "LoadScene"));
    }

    #[test]
    fn picking_a_component_and_action_fills_default_arguments() {
        let (cat, _, other) = world();
        let mut b = EventBinding { target: other.into(), ..Default::default() };
        select_component(&mut b, "Health", &cat, false);
        assert_eq!((b.component.as_str(), b.action.as_str()), ("Health", "Damage"));
        assert_eq!(b.args, vec![ArgSource::Const(ArgValue::Float(10.0))]);
        let desc = action_choices(&cat, other.into(), "Animator", false);
        assert!(desc.is_empty(), "no Animator on the target");
        select_component(&mut b, "Animator", &cat, false);
        assert!(b.action.is_empty() && b.args.is_empty());
    }

    #[test]
    fn problems_catch_what_the_runtime_would_reject() {
        let (cat, _, other) = world();
        let payload = [("uses", ActionType::Int), ("instigator", ActionType::Object)];
        let mut b = EventBinding { target: other.into(), ..Default::default() };
        select_component(&mut b, "Health", &cat, false);
        assert!(problems(&b, &cat, &payload, false).is_empty());
        // Object into a Float parameter.
        b.args = vec![ArgSource::Payload("instigator".into())];
        assert_eq!(problems(&b, &cat, &payload, false).len(), 1);
        // Int converts to Float; a property of the right component resolves.
        b.args = vec![ArgSource::Payload("uses".into())];
        assert!(problems(&b, &cat, &payload, false).is_empty());
        b.args = vec![ArgSource::Property { object: other.into(), component: "Health".into(), property: "current".into() }];
        assert!(problems(&b, &cat, &payload, false).is_empty());
        b.args = vec![ArgSource::Property { object: other.into(), component: "Health".into(), property: "nope".into() }];
        assert_eq!(problems(&b, &cat, &payload, false).len(), 1);
        // A missing field, a removed action, UGC.
        b.args = vec![ArgSource::Payload("gone".into())];
        assert_eq!(problems(&b, &cat, &payload, false).len(), 1);
        b.args.clear();
        assert_eq!(problems(&b, &cat, &payload, true).len(), 1, "Damage is not UGC-safe");
        b.action = "Explode".into();
        assert_eq!(problems(&b, &cat, &payload, false).len(), 1);
    }

    #[test]
    fn instigator_sources_are_typed_from_the_catalog() {
        let (cat, _, _) = world();
        let src = ArgSource::Instigator { component: "Health".into(), property: "fraction".into() };
        assert_eq!(source_type(&src, &cat, &[]), Some(ActionType::Float));
        let itself = ArgSource::Instigator { component: String::new(), property: String::new() };
        assert_eq!(source_type(&itself, &cat, &[]), Some(ActionType::Object));
    }

    #[test]
    fn switching_source_kind_resets_the_value() {
        let p = ParamDesc::float("amount", 10.0);
        assert_eq!(switch_source(SourceKind::Value, &p, &[]), ArgSource::Const(ArgValue::Float(10.0)));
        let payload = [("instigator", ActionType::Object), ("uses", ActionType::Int)];
        // The first event field that converts to the parameter's type.
        assert_eq!(switch_source(SourceKind::Event, &p, &payload), ArgSource::Payload("uses".into()));
        assert!(matches!(switch_source(SourceKind::Property, &p, &[]), ArgSource::Property { .. }));
    }

    #[test]
    fn reordering_stays_in_bounds() {
        let mut list: EventActions = (0..3).map(|i| EventBinding { action: i.to_string(), ..Default::default() }).collect();
        assert!(move_binding(&mut list, 0, 1));
        assert_eq!(list[1].action, "0");
        assert!(!move_binding(&mut list, 0, -1));
        assert!(!move_binding(&mut list, 2, 1));
        assert!(!move_binding(&mut list, 9, -1));
        let _ = <Health as TypedComponent>::NAME;
    }
}
