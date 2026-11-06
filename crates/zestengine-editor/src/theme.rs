//! Design tokens (docs/design/editor-ui-spec.md §2) and the egui style built
//! from them. Panel code takes every colour, size and font from here, never a
//! literal, so the look changes in one place and a light palette can be
//! swapped in later. The accent is a runtime setting; its faint and soft
//! variants derive from it.

use std::sync::atomic::{AtomicU32, Ordering};

use egui::{Color32, CornerRadius, FontFamily, FontId, Stroke, TextStyle};

use crate::icons::Icon;

const fn hex(rgb: u32) -> Color32 {
    Color32::from_rgb((rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8)
}

// Surfaces.
pub const BG_CHROME: Color32 = hex(0x0B0C0E);
pub const BG_APP: Color32 = hex(0x111214);
pub const BG_VIEWPORT: Color32 = hex(0x0C0D0F);
pub const BG_PANEL: Color32 = hex(0x17181C);
pub const BG_FIELD: Color32 = hex(0x0F1012);
pub const BG_BUTTON: Color32 = hex(0x26282D);
pub const BG_SEGMENT_ACTIVE: Color32 = hex(0x2E3036);
pub const OVERLAY: Color32 = Color32::from_rgba_premultiplied(16, 17, 18, 235);

// Lines.
pub const LINE_PANEL: Color32 = hex(0x24262B);
pub const LINE_FIELD: Color32 = hex(0x2A2C31);
pub const LINE_BUTTON: Color32 = hex(0x33363C);
pub const LINE_DASHED: Color32 = hex(0x3A3C42);

// Text.
pub const TEXT_STRONG: Color32 = hex(0xF2F3F5);
pub const TEXT_BODY: Color32 = hex(0xE4E5E9);
pub const TEXT_SECONDARY: Color32 = hex(0xC9CBD1);
pub const TEXT_MUTED: Color32 = hex(0xA3A6AF);
pub const TEXT_DIM: Color32 = hex(0x8D909A);
pub const TEXT_HIDDEN: Color32 = hex(0x858892);
pub const TEXT_ON_ACCENT: Color32 = hex(0x141413);
/// Name on a selected hierarchy row.
pub const TEXT_SELECTED: Color32 = hex(0xF5F6F8);
/// Eye toggle on unselected visible rows.
pub const EYE_QUIET: Color32 = hex(0x5D6068);

// Accent (runtime).
pub const ACCENT_DEFAULT: Color32 = hex(0xF5C84B);
pub const ACCENT_HOVER: Color32 = hex(0xFFE08A);
/// Preset accents offered in Preferences.
pub const ACCENT_PRESETS: [(&str, Color32); 4] = [
    ("ZestEngine", hex(0xF5C84B)),
    ("Amber", hex(0xFFB547)),
    ("ZestBay", hex(0xC9E265)),
    ("Sky", hex(0x7CC4FA)),
];

static ACCENT: AtomicU32 = AtomicU32::new(0xF5C84B);

/// The accent colour: primary buttons, active tool, active tab underline,
/// focus, selection. Never decoration.
pub fn accent() -> Color32 {
    hex(ACCENT.load(Ordering::Relaxed))
}

/// Change the accent at runtime; call [`apply`] afterwards so egui's widget
/// visuals pick it up.
pub fn set_accent(c: Color32) {
    ACCENT.store(((c.r() as u32) << 16) | ((c.g() as u32) << 8) | c.b() as u32, Ordering::Relaxed);
}

/// `u32::MAX` = follow the accent.
static SELECTION_OUTLINE: AtomicU32 = AtomicU32::new(u32::MAX);

/// Colour of the scene view's selection outline and its selected-object markers:
/// the accent unless a custom colour was chosen in Project Settings.
pub fn selection_outline() -> Color32 {
    match SELECTION_OUTLINE.load(Ordering::Relaxed) {
        u32::MAX => accent(),
        v => hex(v),
    }
}

static SELECTION_OUTLINE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Whether selected objects are marked in the scene view (Project Settings).
pub fn selection_outline_enabled() -> bool {
    SELECTION_OUTLINE_ON.load(Ordering::Relaxed)
}

pub fn set_selection_outline_enabled(on: bool) {
    SELECTION_OUTLINE_ON.store(on, Ordering::Relaxed);
}

/// Set (or with None clear, back to the accent) the selection outline colour.
pub fn set_selection_outline(c: Option<Color32>) {
    SELECTION_OUTLINE.store(c.map_or(u32::MAX, |c| ((c.r() as u32) << 16) | ((c.g() as u32) << 8) | c.b() as u32), Ordering::Relaxed);
}

/// Accent at 14%: selected rows and tiles, status chips.
pub fn accent_faint() -> Color32 {
    with_alpha(accent(), 0.14)
}

/// Accent at 55%: the dashed UI-canvas frame in the viewport.
pub fn accent_soft() -> Color32 {
    with_alpha(accent(), 0.55)
}

/// `c` at `alpha` (0..1), unmultiplied.
pub fn with_alpha(c: Color32, alpha: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), (alpha.clamp(0.0, 1.0) * 255.0).round() as u8)
}

// Axes and status.
pub const AXIS_X: Color32 = hex(0xF06A6A);
pub const AXIS_Y: Color32 = hex(0x8BD46A);
pub const AXIS_Z: Color32 = hex(0x6AA8F0);
pub const STATUS_OK: Color32 = hex(0x8BD46A);
pub const STATUS_WARN_FG: Color32 = hex(0xFFB366);
pub const STATUS_WARN_BG: Color32 = hex(0x3A2612);
pub const STATUS_INFO: Color32 = hex(0x7CC4FA);
pub const STATUS_ERROR: Color32 = hex(0xFF8A8A);
/// Prefab instances: the name colour of every object in one, and the 2 px
/// margin bar on overridden values.
pub const PREFAB_NAME: Color32 = hex(0x5DB4F5);
pub const PREFAB_OVERRIDE: Color32 = hex(0x0199EB);
/// Network role tags (NET-3): hierarchy badge and inspector header chip.
/// Client-only content is green, server-only content orange.
pub const NET_CLIENT: Color32 = hex(0x6FCF97);
pub const NET_SERVER: Color32 = hex(0xF2994A);
pub const WARN_ROW_TINT: Color32 = Color32::from_rgba_premultiplied(20, 14, 8, 20);

pub fn axis(i: usize) -> Color32 {
    [AXIS_X, AXIS_Y, AXIS_Z][i.min(2)]
}

/// Object type families (spec §2.4): hierarchy icons, inspector tiles,
/// component headers, viewport gizmos.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ObjectKind {
    DirectionalLight,
    SpotLight,
    PointLight,
    Model,
    Mesh,
    Camera,
    UiCanvas,
    UiToggle,
    UiText,
    UiOther,
    ReflectionProbe,
    FluxVolume,
    PostFxVolume,
    Audio,
    Physics,
    Water,
    Group,
}

pub const TYPE_LIGHT: Color32 = hex(0xF0A040);
pub const TYPE_MESH: Color32 = hex(0x7CC4FA);
pub const TYPE_CAMERA: Color32 = hex(0xA78BFA);
pub const TYPE_UI: Color32 = hex(0x39C6B0);
pub const TYPE_GI: Color32 = hex(0xE879B5);
pub const TYPE_GROUP: Color32 = hex(0xA3A6AF);
pub const TYPE_POSTFX: Color32 = hex(0x818CF8);
pub const TYPE_WATER: Color32 = hex(0x50AAE6);

// Node-canvas signal colours (pins and wires, by data type) and node
// category chips (see node_canvas.rs).
pub const WIRE_FLOAT: Color32 = hex(0x7CC4FA);
pub const WIRE_VEC2: Color32 = hex(0x8BD46A);
pub const WIRE_VEC3: Color32 = hex(0xF0A040);
pub const WIRE_VEC4: Color32 = hex(0xE879B5);
pub const WIRE_TEXTURE: Color32 = hex(0xA78BFA);
pub const WIRE_BOOL: Color32 = hex(0xF06A6A);
pub const CAT_INPUT: Color32 = hex(0x6FCF97);
pub const CAT_MATH: Color32 = hex(0x7CC4FA);
pub const CAT_VECTOR: Color32 = hex(0xF0A040);
pub const CAT_TEXTURE: Color32 = hex(0xA78BFA);
pub const CAT_UTILITY: Color32 = hex(0xA3A6AF);
pub const CAT_OUTPUT: Color32 = hex(0xE879B5);

impl ObjectKind {
    pub fn color(self) -> Color32 {
        use ObjectKind::*;
        match self {
            DirectionalLight | SpotLight | PointLight => TYPE_LIGHT,
            Model | Mesh => TYPE_MESH,
            Camera => TYPE_CAMERA,
            UiCanvas | UiToggle | UiText | UiOther => TYPE_UI,
            ReflectionProbe | FluxVolume => TYPE_GI,
            PostFxVolume => TYPE_POSTFX,
            Water => TYPE_WATER,
            Audio | Physics | Group => TYPE_GROUP,
        }
    }

    pub fn icon(self) -> Icon {
        use ObjectKind::*;
        match self {
            DirectionalLight => Icon::LightDirectional,
            SpotLight => Icon::LightSpot,
            PointLight => Icon::LightPoint,
            Model => Icon::Model,
            Mesh => Icon::Mesh,
            Camera => Icon::Camera,
            UiCanvas | UiOther => Icon::UiCanvas,
            UiToggle => Icon::UiToggle,
            UiText => Icon::UiText,
            ReflectionProbe => Icon::ReflectionProbe,
            FluxVolume => Icon::FluxVolume,
            PostFxVolume => Icon::PostFxVolume,
            Audio => Icon::Audio,
            Physics => Icon::FluxVolume,
            Water => Icon::Water,
            Group => Icon::Folder,
        }
    }

    /// Lower-case type line under the inspector name ("UI canvas").
    pub fn label(self) -> &'static str {
        use ObjectKind::*;
        match self {
            DirectionalLight => "Directional light",
            SpotLight => "Spot light",
            PointLight => "Point light",
            Model => "Model",
            Mesh => "Mesh",
            Camera => "Camera",
            UiCanvas => "UI canvas",
            UiToggle => "UI toggle",
            UiText => "UI text",
            UiOther => "UI element",
            ReflectionProbe => "Reflection probe",
            FluxVolume => "Flux volume",
            PostFxVolume => "Post FX volume",
            Audio => "Audio",
            Physics => "Physics",
            Water => "Water",
            Group => "Group",
        }
    }
}

// Shape (spec §2.7).
pub const R_SWATCH: u8 = 3;
pub const R_FIELD: u8 = 5;
pub const R_ROW: u8 = 6;
pub const R_TOOLBAR: u8 = 8;

pub const H_TOGGLE: f32 = 22.0;
pub const H_BUTTON_SMALL: f32 = 24.0;
pub const H_FIELD: f32 = 26.0;
pub const H_TOOLBAR: f32 = 28.0;
pub const H_PRIMARY: f32 = 32.0;
pub const H_TAB_STRIP: f32 = 36.0;
pub const H_MENU_BAR: f32 = 40.0;
/// Dropdown and context menu rows (spec §3: 28 px menu items).
pub const H_MENU_ITEM: f32 = 28.0;
/// Narrowest a dropdown / context menu gets: room for an icon column, the
/// label and a right-aligned shortcut.
pub const MENU_MIN_WIDTH: f32 = 240.0;
/// Width of the icon column at the start of a menu row.
pub const MENU_ICON_W: f32 = 20.0;
pub const H_STATUS_BAR: f32 = 26.0;
pub const H_ROW: f32 = 26.0;
pub const INDENT: f32 = 14.0;
pub const PAD: f32 = 12.0;
pub const LABEL_WIDTH: f32 = 104.0;

pub const W_HIERARCHY: f32 = 264.0;
pub const W_INSPECTOR: f32 = 324.0;
pub const H_BOTTOM_DOCK: f32 = 288.0;

// Type (spec §2.6).
pub const SIZE_TITLE: f32 = 13.0;
pub const SIZE_BODY: f32 = 12.0;
pub const SIZE_META: f32 = 11.0;
pub const SIZE_TINY: f32 = 10.0;

const SANS_MEDIUM: &str = "plex-sans-medium";
const SANS_SEMIBOLD: &str = "plex-sans-semibold";
const MONO_MEDIUM: &str = "plex-mono-medium";

const CJK_FALLBACK: &str = "system-cjk";

/// Where a CJK-capable system font usually lives, most preferred first, with the face index
/// inside a collection (`.ttc`): Noto Sans CJK's first face is the Japanese one, which also
/// covers the Chinese ideographs and Hangul well enough for labels.
const CJK_FONT_CANDIDATES: &[(&str, u32)] = &[
    ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/OTF/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 0),
    ("/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc", 0),
    ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 0),
    ("C:\\Windows\\Fonts\\YuGothR.ttc", 0),
    ("C:\\Windows\\Fonts\\msgothic.ttc", 0),
    ("C:\\Windows\\Fonts\\simsun.ttc", 0),
    ("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", 0),
    ("/System/Library/Fonts/PingFang.ttc", 0),
];

/// The first CJK system font that can be read (`ZESTENGINE_CJK_FONT=<path>[,<index>]` overrides).
fn system_cjk_font() -> Option<(Vec<u8>, u32)> {
    if let Ok(spec) = std::env::var("ZESTENGINE_CJK_FONT") {
        let (path, index) = spec.rsplit_once(',').and_then(|(p, i)| i.parse().ok().map(|i| (p, i))).unwrap_or((spec.as_str(), 0));
        if let Ok(bytes) = std::fs::read(path) {
            return Some((bytes, index));
        }
    }
    CJK_FONT_CANDIDATES.iter().find_map(|(path, index)| std::fs::read(path).ok().map(|b| (b, *index)))
}

/// Register IBM Plex Sans / Mono ahead of egui's defaults (kept behind for
/// glyphs Plex lacks) and the Phosphor icon font.
pub fn install_fonts(fonts: &mut egui::FontDefinitions) {
    let faces: [(&str, &'static [u8]); 5] = [
        ("plex-sans", include_bytes!("../assets/fonts/IBMPlexSans-Regular.ttf")),
        (SANS_MEDIUM, include_bytes!("../assets/fonts/IBMPlexSans-Medium.ttf")),
        (SANS_SEMIBOLD, include_bytes!("../assets/fonts/IBMPlexSans-SemiBold.ttf")),
        ("plex-mono", include_bytes!("../assets/fonts/IBMPlexMono-Regular.ttf")),
        (MONO_MEDIUM, include_bytes!("../assets/fonts/IBMPlexMono-Medium.ttf")),
    ];
    for (name, bytes) in faces {
        fonts
            .font_data
            .insert(name.to_owned(), std::sync::Arc::new(egui::FontData::from_static(bytes)));
    }
    fonts.families.entry(FontFamily::Proportional).or_default().insert(0, "plex-sans".to_owned());
    fonts.families.entry(FontFamily::Monospace).or_default().insert(0, "plex-mono".to_owned());
    // Names from imported files (blendshapes, bones, objects) are often Japanese or Chinese:
    // use the system's CJK font, when there is one, behind everything else.
    if let Some((bytes, index)) = system_cjk_font() {
        let mut data = egui::FontData::from_owned(bytes);
        data.index = index;
        fonts.font_data.insert(CJK_FALLBACK.to_owned(), std::sync::Arc::new(data));
        for family in [FontFamily::Proportional, FontFamily::Monospace] {
            fonts.families.entry(family).or_default().push(CJK_FALLBACK.to_owned());
        }
    }
    let fallback: Vec<String> = fonts.families.get(&FontFamily::Proportional).cloned().unwrap_or_default();
    for (family, face) in [(SANS_MEDIUM, SANS_MEDIUM), (SANS_SEMIBOLD, SANS_SEMIBOLD), (MONO_MEDIUM, MONO_MEDIUM)] {
        let mut list = vec![face.to_owned()];
        list.extend(fallback.iter().cloned());
        fonts.families.insert(FontFamily::Name(family.into()), list);
    }
}

pub fn medium(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(SANS_MEDIUM.into()))
}

pub fn semibold(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(SANS_SEMIBOLD.into()))
}

pub fn mono(size: f32) -> FontId {
    FontId::monospace(size)
}

pub fn mono_medium(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(MONO_MEDIUM.into()))
}

/// Apply the tokens to egui: colours, widget states, radii, spacing and type.
/// Call again after [`set_accent`].
pub fn apply(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.text_styles = [
        (TextStyle::Small, FontId::proportional(SIZE_META)),
        (TextStyle::Body, FontId::proportional(SIZE_BODY)),
        (TextStyle::Button, FontId::proportional(SIZE_BODY)),
        (TextStyle::Monospace, mono(SIZE_META)),
        (TextStyle::Heading, semibold(SIZE_TITLE)),
    ]
    .into();

    let spacing = &mut style.spacing;
    spacing.item_spacing = egui::vec2(6.0, 4.0);
    spacing.button_padding = egui::vec2(9.0, 4.0);
    spacing.interact_size = egui::vec2(40.0, H_BUTTON_SMALL);
    spacing.combo_height = 280.0;
    spacing.slider_width = 140.0;
    spacing.indent = INDENT;
    spacing.window_margin = egui::Margin::same(PAD as i8);
    spacing.menu_margin = egui::Margin::same(4);
    spacing.scroll.floating = true;
    spacing.scroll.bar_width = 8.0;
    spacing.scroll.floating_width = 8.0;
    spacing.scroll.floating_allocated_width = 0.0;

    let accent = accent();
    let v = &mut style.visuals;
    *v = egui::Visuals::dark();
    v.override_text_color = None;
    v.panel_fill = BG_PANEL;
    v.window_fill = BG_PANEL;
    v.extreme_bg_color = BG_FIELD;
    v.faint_bg_color = BG_APP;
    v.code_bg_color = BG_FIELD;
    v.text_edit_bg_color = Some(BG_FIELD);
    v.window_stroke = Stroke::new(1.0_f32, LINE_PANEL);
    v.window_corner_radius = CornerRadius::same(R_TOOLBAR);
    v.menu_corner_radius = CornerRadius::same(R_TOOLBAR);
    v.window_shadow = egui::epaint::Shadow::NONE;
    v.popup_shadow = egui::epaint::Shadow::NONE;
    v.hyperlink_color = ACCENT_HOVER;
    v.warn_fg_color = STATUS_WARN_FG;
    v.error_fg_color = STATUS_ERROR;
    v.selection.bg_fill = with_alpha(accent, 0.35);
    v.selection.stroke = Stroke::new(1.0_f32, TEXT_STRONG);
    v.slider_trailing_fill = true;
    v.handle_shape = egui::style::HandleShape::Circle;
    v.striped = false;

    let radius = CornerRadius::same(R_FIELD);
    let w = &mut v.widgets;
    w.noninteractive.bg_fill = BG_PANEL;
    w.noninteractive.weak_bg_fill = BG_PANEL;
    w.noninteractive.bg_stroke = Stroke::new(1.0_f32, LINE_PANEL);
    w.noninteractive.fg_stroke = Stroke::new(1.0_f32, TEXT_BODY);
    w.noninteractive.corner_radius = radius;

    w.inactive.bg_fill = BG_FIELD;
    w.inactive.weak_bg_fill = BG_BUTTON;
    w.inactive.bg_stroke = Stroke::new(1.0_f32, LINE_BUTTON);
    w.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT_SECONDARY);
    w.inactive.corner_radius = radius;
    w.inactive.expansion = 0.0;

    w.hovered.bg_fill = BG_SEGMENT_ACTIVE;
    w.hovered.weak_bg_fill = BG_SEGMENT_ACTIVE;
    w.hovered.bg_stroke = Stroke::new(1.0_f32, LINE_DASHED);
    w.hovered.fg_stroke = Stroke::new(1.0_f32, TEXT_STRONG);
    w.hovered.corner_radius = radius;
    w.hovered.expansion = 0.0;

    w.active.bg_fill = BG_SEGMENT_ACTIVE;
    w.active.weak_bg_fill = BG_SEGMENT_ACTIVE;
    w.active.bg_stroke = Stroke::new(1.0_f32, accent);
    w.active.fg_stroke = Stroke::new(1.0_f32, TEXT_STRONG);
    w.active.corner_radius = radius;
    w.active.expansion = 0.0;

    w.open.bg_fill = BG_SEGMENT_ACTIVE;
    w.open.weak_bg_fill = BG_SEGMENT_ACTIVE;
    w.open.bg_stroke = Stroke::new(1.0_f32, LINE_BUTTON);
    w.open.fg_stroke = Stroke::new(1.0_f32, TEXT_STRONG);
    w.open.corner_radius = radius;

    // Dark only for now (light theme is spec §12); pin it so egui doesn't
    // follow the system theme.
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.set_style_of(egui::Theme::Light, style.clone());
    ctx.set_style_of(egui::Theme::Dark, style);
}

/// egui_dock style: 36 px tab strips on bg.panel, active tab text.strong with
/// a 2 px accent underline, inactive tabs muted, 1 px line.panel separators.
pub fn dock_style(base: &egui::Style) -> egui_dock::Style {
    let mut s = egui_dock::Style::from_egui(base);
    s.main_surface_border_stroke = Stroke::NONE;
    s.main_surface_border_rounding = CornerRadius::ZERO;
    s.separator.width = 1.0;
    s.separator.extra = 6.0;
    s.separator.color_idle = LINE_PANEL;
    s.separator.color_hovered = LINE_DASHED;
    s.separator.color_dragged = accent();

    let tb = &mut s.tab_bar;
    tb.bg_fill = BG_PANEL;
    tb.height = H_TAB_STRIP;
    tb.hline_color = LINE_PANEL;
    tb.fill_tab_bar = false;
    tb.corner_radius = CornerRadius::ZERO;

    let t = &mut s.tab;
    t.tab_body.bg_fill = BG_PANEL;
    t.tab_body.stroke = Stroke::NONE;
    t.tab_body.corner_radius = CornerRadius::ZERO;
    t.tab_body.inner_margin = egui::Margin::same(0);
    t.hline_below_active_tab_name = true;
    t.minimum_width = None;
    for (state, text) in [
        (&mut t.active, TEXT_STRONG),
        (&mut t.focused, TEXT_STRONG),
        (&mut t.inactive, TEXT_MUTED),
        (&mut t.hovered, TEXT_SECONDARY),
        (&mut t.active_with_kb_focus, TEXT_STRONG),
        (&mut t.inactive_with_kb_focus, TEXT_MUTED),
        (&mut t.focused_with_kb_focus, TEXT_STRONG),
    ] {
        state.bg_fill = BG_PANEL;
        state.outline_color = Color32::TRANSPARENT;
        state.text_color = text;
        state.corner_radius = CornerRadius::ZERO;
    }
    s.buttons.close_tab_color = TEXT_DIM;
    s.buttons.close_tab_active_color = TEXT_BODY;
    s.buttons.close_tab_bg_fill = BG_SEGMENT_ACTIVE;
    s
}

/// Colour of a network role tag; `None` for `Both`, which has no tag.
pub fn net_role_color(role: zestengine_core::NetRole) -> Option<Color32> {
    match role {
        zestengine_core::NetRole::Client => Some(NET_CLIENT),
        zestengine_core::NetRole::Server => Some(NET_SERVER),
        zestengine_core::NetRole::Both => None,
    }
}
