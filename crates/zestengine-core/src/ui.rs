//! Retained in-game UI components (HUDs, menus). Data only: the engine lays
//! them out and draws them over the game view each frame.
//!
//! A [`UiCanvas`] object is a root; UI components on it and on its descendants
//! (scene hierarchy) are laid out inside the screen. Each element's
//! [`UiRect`] places it relative to its parent's rect with anchors, a pivot,
//! a position and a size (Unity RectTransform style, but y points DOWN: anchor
//! (0,0) is the top-left corner). An element without a `UiRect` fills its
//! parent. Children draw on top of parents, in hierarchy order.

use serde::{Deserialize, Serialize};

use crate::{
    ActionCtx, ActionDesc, ActionType, ArgValue, ComponentCtx, EventActions, FireSource, ObjectId, ParamDesc, PropertyDesc,
    TypedComponent, fire_all,
};

/// How UI sizes respond to the screen resolution.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum UiScaleMode {
    /// Sizes are pixels, regardless of screen size.
    ConstantPixelSize,
    /// Sizes are in reference-resolution pixels, scaled to the screen.
    ScaleWithScreen,
}

impl UiScaleMode {
    /// Every mode, in declaration order.
    pub const ALL: [UiScaleMode; 2] = [Self::ConstantPixelSize, Self::ScaleWithScreen];
    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::ConstantPixelSize => "Constant pixel size",
            Self::ScaleWithScreen => "Scale with screen",
        }
    }
}

/// Where a UI canvas draws: over the screen or as a panel in the world.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum UiRenderMode {
    /// Overlay covering the game view.
    Screen,
    /// A panel in the 3D scene at the canvas object's transform (VR menus,
    /// diegetic screens). Pointed at with the mouse or VR controllers.
    World,
}

impl UiRenderMode {
    /// Every mode, in declaration order.
    pub const ALL: [UiRenderMode; 2] = [Self::Screen, Self::World];
    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Screen => "2D (screen overlay)",
            Self::World => "3D (world panel)",
        }
    }
}

/// Root of a UI tree: a screen overlay, or a panel in the world.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct UiCanvas {
    /// Where the canvas draws: over the screen or as a world panel.
    pub render_mode: UiRenderMode,
    /// World mode: panel size in meters (local XY plane, facing +Z).
    pub world_size: [f32; 2],
    /// World mode: texture pixels per meter (layout units = pixels).
    pub pixels_per_meter: f32,
    /// How sizes respond to the screen resolution.
    pub scale_mode: UiScaleMode,
    /// Resolution the layout was designed for (ScaleWithScreen).
    pub reference_size: [f32; 2],
    /// 0 = scale by width, 1 = by height, between = blend (ScaleWithScreen).
    pub match_height: f32,
    /// Canvases draw in ascending order.
    pub sort_order: i32,
}

impl Default for UiCanvas {
    fn default() -> Self {
        Self {
            render_mode: UiRenderMode::Screen,
            world_size: [1.6, 0.9],
            pixels_per_meter: 800.0,
            scale_mode: UiScaleMode::ScaleWithScreen,
            reference_size: [1920.0, 1080.0],
            match_height: 1.0,
            sort_order: 0,
        }
    }
}

impl UiCanvas {
    /// World mode: render-target size in pixels.
    pub fn world_pixels(&self) -> [u32; 2] {
        let px = |m: f32| (m.max(0.01) * self.pixels_per_meter.max(1.0)).round().clamp(1.0, 4096.0) as u32;
        [px(self.world_size[0]), px(self.world_size[1])]
    }
}

impl TypedComponent for UiCanvas {
    const NAME: &'static str = "UI Canvas";
}

/// Placement inside the parent rect. With `anchor_min == anchor_max` the
/// element has a fixed `size` and sits at `position` from the anchor point;
/// with anchors apart it stretches and `size` is added to the anchored span.
#[derive(Serialize, Deserialize, Clone, Copy)]
#[serde(default)]
pub struct UiRect {
    /// Anchor point of the element's minimum corner (0..1 of the parent).
    pub anchor_min: [f32; 2],
    /// Anchor point of the element's maximum corner (0..1 of the parent).
    pub anchor_max: [f32; 2],
    /// Point of the element (0..1) that sits at `position`.
    pub pivot: [f32; 2],
    /// Offset of the pivot point from the anchor (reference pixels).
    pub position: [f32; 2],
    /// Size in reference pixels (added to the anchored span).
    pub size: [f32; 2],
}

impl Default for UiRect {
    fn default() -> Self {
        Self {
            anchor_min: [0.5, 0.5],
            anchor_max: [0.5, 0.5],
            pivot: [0.5, 0.5],
            position: [0.0, 0.0],
            size: [200.0, 60.0],
        }
    }
}

impl UiRect {
    /// Resolve against a parent rect `(min, size)`; `scale` converts
    /// reference pixels to screen pixels. Returns `(min, size)`.
    pub fn resolve(&self, parent_min: [f32; 2], parent_size: [f32; 2], scale: f32) -> ([f32; 2], [f32; 2]) {
        let mut min = [0.0; 2];
        let mut size = [0.0; 2];
        for a in 0..2 {
            let a0 = parent_min[a] + self.anchor_min[a] * parent_size[a];
            let a1 = parent_min[a] + self.anchor_max[a] * parent_size[a];
            let s = ((a1 - a0) + self.size[a] * scale).max(0.0);
            let pivot_point = a0 + (a1 - a0) * self.pivot[a] + self.position[a] * scale;
            min[a] = pivot_point - s * self.pivot[a];
            size[a] = s;
        }
        (min, size)
    }

    /// Presets for the editor's anchor picker: (label, anchor_min, anchor_max).
    pub const PRESETS: [(&'static str, [f32; 2], [f32; 2]); 10] = [
        ("Center", [0.5, 0.5], [0.5, 0.5]),
        ("Top left", [0.0, 0.0], [0.0, 0.0]),
        ("Top", [0.5, 0.0], [0.5, 0.0]),
        ("Top right", [1.0, 0.0], [1.0, 0.0]),
        ("Left", [0.0, 0.5], [0.0, 0.5]),
        ("Right", [1.0, 0.5], [1.0, 0.5]),
        ("Bottom left", [0.0, 1.0], [0.0, 1.0]),
        ("Bottom", [0.5, 1.0], [0.5, 1.0]),
        ("Bottom right", [1.0, 1.0], [1.0, 1.0]),
        ("Stretch", [0.0, 0.0], [1.0, 1.0]),
    ];
}

impl TypedComponent for UiRect {
    const NAME: &'static str = "UI Rect";
}

/// A filled (optionally rounded / bordered) rectangle.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct UiPanel {
    /// Fill colour, linear RGBA.
    pub color: [f32; 4],
    /// Rounded-corner radius in pixels (0 = sharp).
    pub corner_radius: f32,
    /// Border colour, linear RGBA.
    pub border_color: [f32; 4],
    /// Border thickness in pixels (0 = none).
    pub border_width: f32,
}

impl Default for UiPanel {
    fn default() -> Self {
        Self {
            color: [0.1, 0.1, 0.12, 0.85],
            corner_radius: 6.0,
            border_color: [1.0, 1.0, 1.0, 0.15],
            border_width: 0.0,
        }
    }
}

impl TypedComponent for UiPanel {
    const NAME: &'static str = "UI Panel";
}

/// A texture drawn in the element's rect.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct UiImage {
    /// Project-relative image path (.png / .jpg / ...).
    pub texture: String,
    /// Colour multiplier on the image (white = unchanged).
    pub tint: [f32; 4],
    /// Fit inside the rect keeping the image's aspect ratio.
    pub preserve_aspect: bool,
}

impl Default for UiImage {
    fn default() -> Self {
        Self {
            texture: String::new(),
            tint: [1.0; 4],
            preserve_aspect: true,
        }
    }
}

impl TypedComponent for UiImage {
    const NAME: &'static str = "UI Image";
}

/// Horizontal / vertical alignment of content in an element's rect.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum UiAlign {
    /// Aligned to the left / top.
    Start,
    /// Centred.
    Center,
    /// Aligned to the right / bottom.
    End,
}

impl UiAlign {
    /// Every alignment, in declaration order.
    pub const ALL: [UiAlign; 3] = [Self::Start, Self::Center, Self::End];
    /// Inspector label of this alignment.
    pub fn label(self) -> &'static str {
        match self {
            Self::Start => "Start",
            Self::Center => "Center",
            Self::End => "End",
        }
    }
    /// 0 / 0.5 / 1.
    pub fn factor(self) -> f32 {
        match self {
            Self::Start => 0.0,
            Self::Center => 0.5,
            Self::End => 1.0,
        }
    }
}

/// What happens to text that does not fit its rect.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum UiTextOverflow {
    /// Draw everything, outside the rect if needed.
    #[default]
    Overflow,
    /// Cut off what does not fit and end with "...".
    Ellipsis,
    /// Cut off what does not fit.
    Truncate,
    /// Draw everything but clip to the rect.
    Masking,
    /// Show page `page` of the lines that fit the rect.
    Page,
}

impl UiTextOverflow {
    /// Every overflow, in declaration order.
    pub const ALL: [UiTextOverflow; 5] = [Self::Overflow, Self::Ellipsis, Self::Truncate, Self::Masking, Self::Page];
    /// Inspector label of this overflow.
    pub fn label(self) -> &'static str {
        match self {
            Self::Overflow => "Overflow",
            Self::Ellipsis => "Ellipsis",
            Self::Truncate => "Truncate",
            Self::Masking => "Masking",
            Self::Page => "Page",
        }
    }
}

/// Which text renderer draws a `UiText`.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum UiTextRenderer {
    /// Instanced SDF/MTSDF text: sharp at any scale, shaping, bidi, rich
    /// text. Falls back to egui text when the GPU path is unavailable (editor
    /// previews, a font that fails to load).
    #[default]
    Sdf,
    /// egui's coverage-bitmap text, the pre-SDF path. Ignores the SDF-only
    /// fields (rich text, overflow, auto-size, spacing, outline, localization
    /// key) except `loc_key`.
    Legacy,
}

impl UiTextRenderer {
    /// Every renderer, in declaration order.
    pub const ALL: [UiTextRenderer; 2] = [Self::Sdf, Self::Legacy];
    /// Inspector label of this renderer.
    pub fn label(self) -> &'static str {
        match self {
            Self::Sdf => "SDF",
            Self::Legacy => "Legacy (egui)",
        }
    }
}

/// Text in the element's rect.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct UiText {
    /// The text to draw (or the localization key when `loc_key` resolves).
    pub text: String,
    /// Font size in reference pixels.
    pub size: f32,
    /// Text colour, linear RGBA.
    pub color: [f32; 4],
    /// Horizontal alignment in the rect.
    pub align_x: UiAlign,
    /// Vertical alignment in the rect.
    pub align_y: UiAlign,
    /// Wrap at the rect width.
    pub wrap: bool,
    /// Project-relative .ttf / .otf / .zfont / .zfontstack (empty = built-in
    /// font). A `.zfontstack` carries fallbacks and per-language fonts.
    pub font: String,
    /// Localization key; when the string tables have it, it replaces `text`
    /// (and re-resolves on language change). Empty = use `text`.
    pub loc_key: String,
    /// Parse `<b> <i> <color=#rrggbb> <size=..>` and the other rich tags.
    pub rich_text: bool,
    /// What happens when the text does not fit the rect.
    pub overflow: UiTextOverflow,
    /// Shrink or grow the size within `auto_size_min..=auto_size_max` so the
    /// text fits the rect.
    pub auto_size: bool,
    /// Lower bound of the auto-size range (reference pixels).
    pub auto_size_min: f32,
    /// Upper bound of the auto-size range (reference pixels).
    pub auto_size_max: f32,
    /// Draw the text bold.
    pub bold: bool,
    /// Draw the text italic.
    pub italic: bool,
    /// Spacing in em/100 (like TextMeshPro).
    pub char_spacing: f32,
    /// Extra space between words, in em/100.
    pub word_spacing: f32,
    /// Extra space between lines, in em/100.
    pub line_spacing: f32,
    /// Extra space between paragraphs, in em/100.
    pub paragraph_spacing: f32,
    /// Fill each wrapped line to the rect width (the last line stays ragged).
    pub justify: bool,
    /// Inset from the rect, reference pixels: left, top, right, bottom.
    pub margins: [f32; 4],
    /// Outline width in em/100 (0 = none).
    pub outline_width: f32,
    /// Outline colour, linear RGBA (only when `outline_width` is above 0).
    pub outline_color: [f32; 4],
    /// Which renderer draws the text.
    pub renderer: UiTextRenderer,
}

impl Default for UiText {
    fn default() -> Self {
        Self {
            text: "Text".into(),
            size: 32.0,
            color: [1.0; 4],
            align_x: UiAlign::Center,
            align_y: UiAlign::Center,
            wrap: true,
            font: String::new(),
            loc_key: String::new(),
            rich_text: false,
            overflow: UiTextOverflow::Overflow,
            auto_size: false,
            auto_size_min: 12.0,
            auto_size_max: 72.0,
            bold: false,
            italic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            line_spacing: 0.0,
            paragraph_spacing: 0.0,
            justify: false,
            margins: [0.0; 4],
            outline_width: 0.0,
            outline_color: [0.0, 0.0, 0.0, 1.0],
            renderer: UiTextRenderer::Sdf,
        }
    }
}

static UI_TEXT_ACTIONS: &[ActionDesc] = &[ActionDesc::new("SetText", "Set Text", &[ParamDesc::string("text")], true)];
static UI_TEXT_PROPERTIES: &[PropertyDesc] = &[PropertyDesc::new("text", ActionType::String)];

impl TypedComponent for UiText {
    const NAME: &'static str = "UI Text";

    fn actions(&self) -> &'static [ActionDesc] {
        UI_TEXT_ACTIONS
    }
    fn invoke_action(&mut self, _ctx: &mut ActionCtx, name: &str, args: &[ArgValue]) -> Result<(), String> {
        match name {
            "SetText" => self.text = args[0].as_str().to_owned(),
            _ => return Err(crate::actions::unknown_action(Self::NAME, name)),
        }
        Ok(())
    }
    fn properties(&self) -> &'static [PropertyDesc] {
        UI_TEXT_PROPERTIES
    }
    fn read_property(&self, name: &str) -> Option<ArgValue> {
        match name {
            "text" => Some(ArgValue::String(self.text.clone())),
            _ => None,
        }
    }
}

/// Which visual state a `UiButton` is in (engine-filled).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum UiButtonState {
    /// Not hovered or pressed.
    #[default]
    Normal,
    /// The pointer is over it.
    Hovered,
    /// Being clicked.
    Pressed,
}

/// Clickable element. Tints the object's `UiPanel` (or draws its own
/// rounded rect) by state. Check clicks with `ctx.ui_clicked(button)`.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct UiButton {
    /// Whether the button responds to clicks.
    pub interactable: bool,
    /// Tint in the normal state.
    pub normal: [f32; 4],
    /// Tint while hovered.
    pub hovered: [f32; 4],
    /// Tint while pressed.
    pub pressed: [f32; 4],
    /// Tint while disabled.
    pub disabled: [f32; 4],
    /// Run in order when clicked (no code needed for menus).
    pub on_click: Vec<UiAction>,
    /// Function calls on any component, run after `on_click` (see [`crate::actions`]).
    pub on_click_actions: EventActions,
    /// The state this frame (engine-filled, not saved).
    #[serde(skip)]
    pub state: UiButtonState,
}

impl Default for UiButton {
    fn default() -> Self {
        Self {
            interactable: true,
            normal: [0.18, 0.18, 0.22, 0.95],
            hovered: [0.26, 0.26, 0.32, 0.95],
            pressed: [0.12, 0.12, 0.15, 0.95],
            disabled: [0.15, 0.15, 0.15, 0.5],
            on_click: Vec::new(),
            on_click_actions: Vec::new(),
            state: UiButtonState::Normal,
        }
    }
}

impl UiButton {
    /// The tint for the current state.
    pub fn color(&self) -> [f32; 4] {
        if !self.interactable {
            return self.disabled;
        }
        match self.state {
            UiButtonState::Normal => self.normal,
            UiButtonState::Hovered => self.hovered,
            UiButtonState::Pressed => self.pressed,
        }
    }
}

impl TypedComponent for UiButton {
    const NAME: &'static str = "UI Button";

    fn update(&mut self, ctx: &mut ComponentCtx) {
        if (!self.on_click.is_empty() || !self.on_click_actions.is_empty()) && ctx.ui_clicked(ctx.self_id()) {
            for action in &self.on_click {
                action.run(ctx);
            }
            let me = ctx.self_id();
            let payload = [("object", ArgValue::Object(me.into()))];
            fire_all(&self.on_click_actions, ctx, &FireSource { source: me, instigator: None, payload: &payload });
        }
    }
}

/// Built-in button behaviour. Targets are object names (first match).
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub enum UiAction {
    /// Load the scene at the given project-relative path.
    LoadScene(String),
    /// Activate the named object.
    Show(String),
    /// Deactivate the named object.
    Hide(String),
    /// Flip the named object's active state.
    Toggle(String),
    /// Quit the game.
    Quit,
}

impl UiAction {
    /// One of each kind, for the inspector's picker.
    pub fn kinds() -> [UiAction; 5] {
        [
            UiAction::LoadScene(String::new()),
            UiAction::Show(String::new()),
            UiAction::Hide(String::new()),
            UiAction::Toggle(String::new()),
            UiAction::Quit,
        ]
    }

    /// Inspector label of this action.
    pub fn label(&self) -> &'static str {
        match self {
            UiAction::LoadScene(_) => "Load scene",
            UiAction::Show(_) => "Show object",
            UiAction::Hide(_) => "Hide object",
            UiAction::Toggle(_) => "Toggle object",
            UiAction::Quit => "Quit game",
        }
    }

    /// The scene path / object name the action uses, if it takes one.
    pub fn target_mut(&mut self) -> Option<&mut String> {
        match self {
            UiAction::LoadScene(t) | UiAction::Show(t) | UiAction::Hide(t) | UiAction::Toggle(t) => Some(t),
            UiAction::Quit => None,
        }
    }

    /// Same kind, keeping the target where both take one.
    pub fn with_kind_of(&self, kind: &UiAction) -> UiAction {
        let mut out = kind.clone();
        if let (Some(dst), Some(src)) = (out.target_mut(), self.clone().target_mut().cloned()) {
            *dst = src;
        }
        out
    }

    /// Apply the action through the component context.
    pub fn run(&self, ctx: &mut ComponentCtx) {
        let named = |ctx: &ComponentCtx, name: &str| -> Option<ObjectId> {
            ctx.find_object(name).and_then(|i| ctx.object_ids.get(i).copied())
        };
        match self {
            UiAction::LoadScene(path) if !path.is_empty() => ctx.load_scene(path.clone()),
            UiAction::LoadScene(_) => {}
            UiAction::Show(name) | UiAction::Hide(name) | UiAction::Toggle(name) => {
                let Some(id) = named(ctx, name) else {
                    return;
                };
                let on = match self {
                    UiAction::Show(_) => true,
                    UiAction::Hide(_) => false,
                    _ => !ctx.is_active(id),
                };
                ctx.set_active(id, on);
            }
            UiAction::Quit => ctx.quit(),
        }
    }
}

/// On/off checkbox, drawn as a box at the left of the rect (label it with a
/// child `UiText`). Clicking flips `on`; it also counts as a `ui_clicked`.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct UiToggle {
    /// The checkbox's current state.
    pub on: bool,
    /// Whether the toggle responds to clicks.
    pub interactable: bool,
    /// Box colour, linear RGBA.
    pub box_color: [f32; 4],
    /// Checkmark colour, linear RGBA.
    pub check_color: [f32; 4],
    /// Input action whose vertical axis `on` inverts in every control scheme
    /// that has it (a settings menu's Invert Look, e.g. "Look"). It starts
    /// out showing the player's saved setting.
    pub invert_y_action: String,
    /// Last value applied to the bound action (engine-filled, not saved).
    #[serde(skip)]
    pub applied: Option<bool>,
}

impl Default for UiToggle {
    fn default() -> Self {
        Self {
            on: false,
            interactable: true,
            box_color: [0.2, 0.2, 0.25, 1.0],
            check_color: [0.95, 0.75, 0.2, 1.0],
            invert_y_action: String::new(),
            applied: None,
        }
    }
}

impl TypedComponent for UiToggle {
    const NAME: &'static str = "UI Toggle";

    fn update(&mut self, ctx: &mut ComponentCtx) {
        let action = self.invert_y_action.as_str();
        if action.is_empty() || self.applied == Some(self.on) {
            return;
        }
        if self.applied.is_none() {
            if let Some(a) = ctx.control_scheme().and_then(|s| s.action(action)) {
                self.on = a.invert_y;
            }
            self.applied = Some(self.on);
            return;
        }
        let schemes: Vec<String> =
            ctx.input_bindings().schemes.iter().filter(|s| s.action(action).is_some()).map(|s| s.name.clone()).collect();
        let tuning = crate::ActionTuning { invert_y: Some(self.on), ..Default::default() };
        for scheme in schemes {
            ctx.tune_action(&scheme, action, tuning);
        }
        self.applied = Some(self.on);
    }
}

/// Horizontal slider; drag to change `value` within `min..=max`.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct UiSlider {
    /// The slider's current value.
    pub value: f32,
    /// Lowest value the slider accepts.
    pub min: f32,
    /// Highest value the slider accepts.
    pub max: f32,
    /// Snap to whole numbers.
    pub whole_numbers: bool,
    /// Whether the slider responds to dragging.
    pub interactable: bool,
    /// Track colour, linear RGBA.
    pub track_color: [f32; 4],
    /// Filled portion colour, linear RGBA.
    pub fill_color: [f32; 4],
    /// Handle colour, linear RGBA.
    pub handle_color: [f32; 4],
    /// Audio Mixer bus whose volume follows `value` (e.g. a settings menu).
    pub audio_bus: String,
    /// Last fraction applied to the audio bus (engine-filled, not saved).
    #[serde(skip)]
    pub applied: Option<f32>,
}

impl Default for UiSlider {
    fn default() -> Self {
        Self {
            value: 0.5,
            min: 0.0,
            max: 1.0,
            whole_numbers: false,
            interactable: true,
            track_color: [0.2, 0.2, 0.25, 1.0],
            fill_color: [0.95, 0.75, 0.2, 1.0],
            handle_color: [1.0, 1.0, 1.0, 1.0],
            audio_bus: String::new(),
            applied: None,
        }
    }
}

impl UiSlider {
    /// `value` as 0..1 along the track.
    pub fn fraction(&self) -> f32 {
        let span = self.max - self.min;
        if span.abs() < 1e-9 {
            0.0
        } else {
            ((self.value - self.min) / span).clamp(0.0, 1.0)
        }
    }

    /// Set `value` from a 0..1 fraction along the track (snaps when
    /// `whole_numbers`).
    pub fn set_fraction(&mut self, f: f32) {
        let mut v = self.min + (self.max - self.min) * f.clamp(0.0, 1.0);
        if self.whole_numbers {
            v = v.round();
        }
        self.value = v;
    }
}

impl TypedComponent for UiSlider {
    const NAME: &'static str = "UI Slider";

    fn update(&mut self, ctx: &mut ComponentCtx) {
        if !self.audio_bus.is_empty() && self.applied != Some(self.value) {
            ctx.set_bus_volume(self.audio_bus.clone(), self.fraction());
            self.applied = Some(self.value);
        }
    }
}

/// Which way a bar fills.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum UiFillDirection {
    /// Fills left to right.
    #[default]
    LeftToRight,
    /// Fills right to left.
    RightToLeft,
    /// Fills bottom to top.
    BottomToTop,
    /// Fills top to bottom.
    TopToBottom,
}

impl UiFillDirection {
    /// Every direction, in declaration order.
    pub const ALL: [UiFillDirection; 4] = [
        UiFillDirection::LeftToRight,
        UiFillDirection::RightToLeft,
        UiFillDirection::BottomToTop,
        UiFillDirection::TopToBottom,
    ];

    /// Inspector label of this direction.
    pub fn label(self) -> &'static str {
        match self {
            UiFillDirection::LeftToRight => "Left to right",
            UiFillDirection::RightToLeft => "Right to left",
            UiFillDirection::BottomToTop => "Bottom to top",
            UiFillDirection::TopToBottom => "Top to bottom",
        }
    }
}

/// Non-interactive bar (health, stamina, loading). Set `value` (0..1) from
/// gameplay with `ctx.modify`.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct UiProgressBar {
    /// Filled amount 0..1.
    pub value: f32,
    /// Which way the bar fills.
    pub direction: UiFillDirection,
    /// Track colour, linear RGBA.
    pub background: [f32; 4],
    /// Filled portion colour, linear RGBA.
    pub fill: [f32; 4],
    /// Rounded-corner radius in pixels.
    pub corner_radius: f32,
}

impl Default for UiProgressBar {
    fn default() -> Self {
        Self {
            value: 0.75,
            direction: UiFillDirection::LeftToRight,
            background: [0.1, 0.1, 0.12, 0.85],
            fill: [0.85, 0.25, 0.25, 1.0],
            corner_radius: 4.0,
        }
    }
}

impl TypedComponent for UiProgressBar {
    const NAME: &'static str = "UI Progress Bar";
}

/// Escape / gamepad back toggles `panel` (an object name), pausing time and
/// freeing the cursor while it's open. Put it on an object that stays active
/// (the canvas), not on the panel itself. A Resume button is `Hide <panel>`.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct UiPauseMenu {
    /// Object name of the panel shown while paused.
    pub panel: String,
    /// Set the time scale to 0 while the menu is open.
    pub pause_time: bool,
    /// Re-lock the cursor when the menu closes (first-person games).
    pub lock_cursor_in_game: bool,
    /// The menu's state the engine last saw (engine-filled, not saved).
    #[serde(skip)]
    pub open: Option<bool>,
}

impl Default for UiPauseMenu {
    fn default() -> Self {
        Self {
            panel: "Pause Panel".into(),
            pause_time: true,
            lock_cursor_in_game: false,
            open: None,
        }
    }
}

impl TypedComponent for UiPauseMenu {
    const NAME: &'static str = "UI Pause Menu";

    fn update(&mut self, ctx: &mut ComponentCtx) {
        let Some(id) = ctx.find_object(&self.panel).and_then(|i| ctx.object_ids.get(i).copied()) else {
            return;
        };
        let open = ctx.is_active(id);
        if ctx.ui_cancel_pressed() {
            ctx.set_active(id, !open);
        }
        if self.open != Some(open) {
            // The first frame only acts when the panel starts open.
            if self.open.is_some() || open {
                if self.pause_time {
                    ctx.set_time_scale(if open { 0.0 } else { 1.0 });
                }
                ctx.set_cursor_locked(!open && self.lock_cursor_in_game);
            }
            self.open = Some(open);
        }
    }
}

/// Single-line text entry. Click (or navigate + submit) to edit; Enter
/// submits (`ctx.ui_submitted(field)`), Escape cancels editing.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct UiTextField {
    /// The text being edited.
    pub text: String,
    /// Greyed text shown while the field is empty.
    pub placeholder: String,
    /// 0 = unlimited.
    pub max_length: u32,
    /// Font size in reference pixels.
    pub size: f32,
    /// Text colour, linear RGBA.
    pub text_color: [f32; 4],
    /// Placeholder colour, linear RGBA.
    pub placeholder_color: [f32; 4],
    /// Show bullets instead of the characters.
    pub password: bool,
    /// Whether the field responds to clicks.
    pub interactable: bool,
    /// The field is being edited (engine-filled, not saved).
    #[serde(skip)]
    pub editing: bool,
}

impl Default for UiTextField {
    fn default() -> Self {
        Self {
            text: String::new(),
            placeholder: "Enter text...".into(),
            max_length: 0,
            size: 28.0,
            text_color: [1.0; 4],
            placeholder_color: [1.0, 1.0, 1.0, 0.4],
            password: false,
            interactable: true,
            editing: false,
        }
    }
}

impl UiTextField {
    /// Apply typed text / backspace, respecting `max_length`.
    pub fn type_text(&mut self, typed: &str, backspaces: u32) {
        for _ in 0..backspaces {
            self.text.pop();
        }
        for c in typed.chars().filter(|c| !c.is_control()) {
            if self.max_length > 0 && self.text.chars().count() >= self.max_length as usize {
                break;
            }
            self.text.push(c);
        }
    }

    /// What is drawn: bullets instead of characters when `password` is on.
    pub fn display_text(&self) -> String {
        if self.password {
            "\u{2022}".repeat(self.text.chars().count())
        } else {
            self.text.clone()
        }
    }
}

impl TypedComponent for UiTextField {
    const NAME: &'static str = "UI Text Field";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ui_text_from_before_sdf_text_still_loads() {
        // A scene saved before the SDF text fields existed.
        let t: UiText = ron::from_str(
            r#"(text: "Hi", size: 40.0, color: (1.0, 0.5, 0.0, 1.0), align_x: Start, align_y: End, wrap: false, font: "fonts/a.ttf")"#,
        )
        .unwrap();
        assert_eq!((t.text.as_str(), t.size, t.wrap, t.font.as_str()), ("Hi", 40.0, false, "fonts/a.ttf"));
        assert_eq!(t.renderer, UiTextRenderer::Sdf);
        assert_eq!(t.overflow, UiTextOverflow::Overflow);
        assert!(t.loc_key.is_empty() && !t.rich_text && !t.auto_size);
        assert_eq!(t.margins, [0.0; 4]);
    }

    #[test]
    fn fixed_rect_is_placed_by_pivot_at_anchor() {
        let r = UiRect {
            anchor_min: [1.0, 0.0],
            anchor_max: [1.0, 0.0],
            pivot: [1.0, 0.0],
            position: [-10.0, 10.0],
            size: [100.0, 50.0],
        };
        let (min, size) = r.resolve([0.0, 0.0], [800.0, 600.0], 1.0);
        assert_eq!(size, [100.0, 50.0]);
        assert_eq!(min, [690.0, 10.0]);
    }

    #[test]
    fn stretch_rect_fills_parent_with_margins() {
        let r = UiRect {
            anchor_min: [0.0, 0.0],
            anchor_max: [1.0, 1.0],
            pivot: [0.5, 0.5],
            position: [0.0, 0.0],
            size: [-20.0, -20.0],
        };
        let (min, size) = r.resolve([100.0, 100.0], [400.0, 200.0], 2.0);
        assert_eq!(size, [360.0, 160.0]);
        assert_eq!(min, [120.0, 120.0]);
    }

    #[test]
    fn text_field_respects_max_length_and_backspace() {
        let mut f = UiTextField {
            max_length: 4,
            ..Default::default()
        };
        f.type_text("hello", 0);
        assert_eq!(f.text, "hell");
        f.type_text("\n", 2);
        assert_eq!(f.text, "he");
        f.password = true;
        assert_eq!(f.display_text().chars().count(), 2);
    }

    #[test]
    fn world_canvas_pixel_size() {
        let c = UiCanvas {
            render_mode: UiRenderMode::World,
            world_size: [1.5, 0.5],
            pixels_per_meter: 1000.0,
            ..Default::default()
        };
        assert_eq!(c.world_pixels(), [1500, 500]);
    }

    #[test]
    fn slider_fraction_round_trips_and_snaps() {
        let mut s = UiSlider {
            min: 0.0,
            max: 10.0,
            whole_numbers: true,
            ..Default::default()
        };
        s.set_fraction(0.33);
        assert_eq!(s.value, 3.0);
        assert!((s.fraction() - 0.3).abs() < 1e-6);
    }
}
