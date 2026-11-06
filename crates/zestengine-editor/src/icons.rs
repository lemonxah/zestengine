//! The editor icon set (docs/design/editor-ui-spec.md §11), drawn from its SVG
//! path data on a 16×16 grid: stroked at 1.4–1.5 with round joins, or filled
//! for the transport glyphs. Paths are parsed once and cached as polylines.

use std::sync::OnceLock;

use egui::{Color32, Painter, Pos2, Rect, Response, Sense, Shape, Stroke, Ui, Vec2};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Icon {
    Camera,
    LightDirectional,
    LightSpot,
    LightPoint,
    Model,
    Mesh,
    Folder,
    ReflectionProbe,
    FluxVolume,
    PostFxVolume,
    UiCanvas,
    UiToggle,
    UiText,
    File,
    FileDoc,
    Gear,
    ToolSelect,
    ToolMove,
    ToolRotate,
    ToolScale,
    ToolRect,
    Eye,
    EyeOff,
    Play,
    Stop,
    Pause,
    Step,
    Caret,
    Reset,
    Snap,
    Audio,
    Plus,
    Search,
    Info,
    ArrowUp,
    Target,
    Check,
    Close,
    Grid,
    Stats,
    Gizmo,
    Warning,
    Trash,
    Duplicate,
    Code,
    Lock,
    LockOpen,
    Kebab,
    Water,
}

impl Icon {
    pub const ALL: [Icon; 49] = [
        Icon::Camera,
        Icon::LightDirectional,
        Icon::LightSpot,
        Icon::LightPoint,
        Icon::Model,
        Icon::Mesh,
        Icon::Folder,
        Icon::ReflectionProbe,
        Icon::FluxVolume,
        Icon::PostFxVolume,
        Icon::UiCanvas,
        Icon::UiToggle,
        Icon::UiText,
        Icon::File,
        Icon::FileDoc,
        Icon::Gear,
        Icon::ToolSelect,
        Icon::ToolMove,
        Icon::ToolRotate,
        Icon::ToolScale,
        Icon::ToolRect,
        Icon::Eye,
        Icon::EyeOff,
        Icon::Play,
        Icon::Stop,
        Icon::Pause,
        Icon::Step,
        Icon::Caret,
        Icon::Reset,
        Icon::Snap,
        Icon::Audio,
        Icon::Plus,
        Icon::Search,
        Icon::Info,
        Icon::ArrowUp,
        Icon::Target,
        Icon::Check,
        Icon::Close,
        Icon::Grid,
        Icon::Stats,
        Icon::Gizmo,
        Icon::Warning,
        Icon::Trash,
        Icon::Duplicate,
        Icon::Code,
        Icon::Lock,
        Icon::LockOpen,
        Icon::Kebab,
        Icon::Water,
    ];

    /// SVG path data on the 16×16 grid (the caret uses a 10×10 grid).
    fn path(self) -> &'static str {
        match self {
            Icon::Camera => "M2 5h8v6H2z M10 7.2l4-2v5.6l-4-2",
            Icon::LightDirectional => {
                "M8 5.2a2.8 2.8 0 1 0 0 5.6a2.8 2.8 0 1 0 0-5.6z M8 1.5v1.6 M8 12.9v1.6 M1.5 8h1.6 M12.9 8h1.6 M3.4 3.4l1.1 1.1 M11.5 11.5l1.1 1.1 M3.4 12.6l1.1-1.1 M11.5 4.5l1.1-1.1"
            }
            Icon::LightSpot => "M6 2h4l3 8.5H3z M5.5 13.5h5",
            Icon::LightPoint => {
                "M6.2 12h3.6 M6.6 14h2.8 M8 2a4 4 0 0 0-2.5 7.1c.5.4.6 1 .6 1.9h3.8c0-.9.1-1.5.6-1.9A4 4 0 0 0 8 2z"
            }
            Icon::Model => "M8 1.8l5.5 3v6.4L8 14.2l-5.5-3V4.8z M2.5 4.8L8 7.8l5.5-3 M8 7.8v6.4",
            Icon::Mesh => "M8 2.5l5.5 10h-11z",
            Icon::Folder => "M1.5 4.5a1 1 0 0 1 1-1h3.4l1.4 1.5h6.2a1 1 0 0 1 1 1v6.5a1 1 0 0 1-1 1h-11a1 1 0 0 1-1-1z",
            Icon::ReflectionProbe => "M8 2.5a5.5 5.5 0 1 0 0 11a5.5 5.5 0 1 0 0-11z M2.5 8c0 1.4 2.5 2.4 5.5 2.4s5.5-1 5.5-2.4",
            Icon::FluxVolume => "M2.5 2.5h11v11h-11z M2.5 8h11 M8 2.5v11",
            // A box with a four-point sparkle: post effects applied inside a volume.
            Icon::PostFxVolume => "M2.5 2.5h11v11h-11z M8 4.6l.9 2.5 2.5.9-2.5.9-.9 2.5-.9-2.5-2.5-.9 2.5-.9z",
            Icon::UiCanvas => "M2 3h12v10H2z M2 6h12",
            Icon::UiToggle => "M5 4.5h6a3.5 3.5 0 0 1 0 7H5a3.5 3.5 0 0 1 0-7z M11 6.3a1.7 1.7 0 1 0 0 3.4a1.7 1.7 0 1 0 0-3.4z",
            Icon::UiText => "M3 3.5h10 M8 3.5v9 M6 12.5h4",
            Icon::File => "M4 1.5h5.5l3 3v10h-8.5z M9.5 1.5v3h3",
            Icon::FileDoc => "M4 1.5h5.5l3 3v10h-8.5z M9.5 1.5v3h3 M6 8h4.5 M6 10.5h4.5",
            // An eight-tooth cog with a hub.
            Icon::Gear => {
                "M12.57 6.9L14.34 7.15L14.34 8.85L12.57 9.1L12.01 10.46L13.09 11.88L11.88 13.09L10.46 12.01L9.1 12.57L8.85 14.34L7.15 14.34L6.9 12.57L5.54 12.01L4.12 13.09L2.91 11.88L3.99 10.46L3.43 9.1L1.66 8.85L1.66 7.15L3.43 6.9L3.99 5.54L2.91 4.12L4.12 2.91L5.54 3.99L6.9 3.43L7.15 1.66L8.85 1.66L9.1 3.43L10.46 3.99L11.88 2.91L13.09 4.12L12.01 5.54z M8 5.9a2.1 2.1 0 1 0 0 4.2a2.1 2.1 0 1 0 0-4.2z"
            }
            Icon::ToolSelect => "M3.5 2l9 5-4 1.2-1.6 4.3z",
            Icon::ToolMove => {
                "M8 1.5v13 M1.5 8h13 M8 1.5L6.2 3.3 M8 1.5l1.8 1.8 M8 14.5l-1.8-1.8 M8 14.5l1.8-1.8 M1.5 8l1.8-1.8 M1.5 8l1.8 1.8 M14.5 8l-1.8-1.8 M14.5 8l-1.8 1.8"
            }
            Icon::ToolRotate => "M13 8a5 5 0 1 1-1.5-3.6 M13 2.5v2.4h-2.4",
            Icon::ToolScale => "M2.5 9.5v4h4 M2.5 13.5L7 9 M9.5 2.5h4v4 M13.5 2.5L9 7",
            Icon::ToolRect => "M3.5 3.5h9v9h-9z M3.5 1.5v2 M1.5 3.5h2 M12.5 14.5v-2 M14.5 12.5h-2",
            Icon::Eye => "M1.5 8s2.4-4.5 6.5-4.5S14.5 8 14.5 8s-2.4 4.5-6.5 4.5S1.5 8 1.5 8z M8 6.2a1.8 1.8 0 1 0 0 3.6a1.8 1.8 0 1 0 0-3.6z",
            Icon::EyeOff => {
                "M2 2l12 12 M6.5 3.7A6.8 6.8 0 0 1 8 3.5c4.1 0 6.5 4.5 6.5 4.5a11 11 0 0 1-1.8 2.3 M4.2 4.9C2.5 6.1 1.5 8 1.5 8s2.4 4.5 6.5 4.5c.9 0 1.7-.2 2.4-.5"
            }
            Icon::Play => "M4.5 2.8v10.4L13 8z",
            Icon::Stop => "M4 4h8v8H4z",
            Icon::Pause => "M4 3h3v10H4zM9 3h3v10H9z",
            Icon::Step => "M3 3l7 5-7 5zM11 3h2v10h-2z",
            Icon::Caret => "M3.5 2l3 3-3 3",
            Icon::Reset => "M3 8a5 5 0 1 0 1.5-3.6 M3 2.5v2.4h2.4",
            Icon::Snap => "M4 2v6a4 4 0 0 0 8 0V2 M4 5h3 M9 5h3",
            Icon::Audio => "M2.5 6h2.5l3.5-3v10l-3.5-3H2.5z M11 5.5a3.5 3.5 0 0 1 0 5 M12.8 3.8a6 6 0 0 1 0 8.4",
            Icon::Plus => "M8 3v10 M3 8h10",
            Icon::Search => "M7 2.5a4.5 4.5 0 1 0 0 9a4.5 4.5 0 1 0 0-9z M10.3 10.3l3.2 3.2",
            Icon::Info => "M8 2a6 6 0 1 0 0 12a6 6 0 1 0 0-12z M8 7.2v4 M8 4.9v.3",
            Icon::ArrowUp => "M8 13V3 M4 7l4-4 4 4",
            Icon::Target => "M8 3.5a4.5 4.5 0 1 0 0 9a4.5 4.5 0 1 0 0-9z M8 1v3 M8 12v3 M1 8h3 M12 8h3",
            Icon::Check => "M3 8.5l3 3 7-7",
            Icon::Close => "M4 4l8 8 M12 4l-8 8",
            Icon::Grid => "M2.5 2.5h11v11h-11z M2.5 6.2h11 M2.5 9.8h11 M6.2 2.5v11 M9.8 2.5v11",
            Icon::Stats => "M3 13V9 M8 13V3 M13 13V6",
            Icon::Gizmo => "M8 8V2.5 M8 8l4.8 2.8 M8 8l-4.8 2.8",
            Icon::Warning => "M8 2l6.5 11.5h-13z M8 6.5v3.5 M8 11.8v.3",
            Icon::Trash => "M3 4.5h10 M6.5 4.5V3h3v1.5 M4.5 4.5l.7 9h5.6l.7-9",
            Icon::Duplicate => "M5.5 5.5h8v8h-8z M10.5 5.5v-3h-8v8h3",
            Icon::Code => "M5.5 4L2 8l3.5 4 M10.5 4L14 8l-3.5 4",
            Icon::Lock => "M4.5 7.5h7v6h-7z M6 7.5V5.5a2 2 0 0 1 4 0v2",
            Icon::LockOpen => "M4.5 7.5h7v6h-7z M6 7.5V5.5a2 2 0 0 1 3.9-.6",
            Icon::Kebab => "",
            // Three rolling wave lines.
            Icon::Water => {
                "M1.5 4.5C3 3.2 4.5 3.2 6 4.5S9 5.8 10.5 4.5 13.5 3.2 14.5 4.2 M1.5 8.5C3 7.2 4.5 7.2 6 8.5S9 9.8 10.5 8.5 13.5 7.2 14.5 8.2 M1.5 12.5C3 11.2 4.5 11.2 6 12.5S9 13.8 10.5 12.5 13.5 11.2 14.5 12.2"
            }
        }
    }

    fn filled(self) -> bool {
        matches!(self, Icon::Play | Icon::Stop | Icon::Pause | Icon::Step)
    }

    fn grid(self) -> f32 {
        if self == Icon::Caret { 10.0 } else { 16.0 }
    }
}

struct Subpath {
    points: Vec<Pos2>,
    closed: bool,
}

fn cache() -> &'static Vec<Vec<Subpath>> {
    static CACHE: OnceLock<Vec<Vec<Subpath>>> = OnceLock::new();
    CACHE.get_or_init(|| Icon::ALL.iter().map(|i| parse(i.path())).collect())
}

fn subpaths(icon: Icon) -> &'static [Subpath] {
    let i = Icon::ALL.iter().position(|&x| x == icon).unwrap_or(0);
    &cache()[i]
}

/// Draw `icon` filling `rect` (square) in `color`. `stroke` is in grid units
/// (the spec uses 1.4–1.5 in lists, 0.9 for 34 px file tiles).
pub fn paint_with(painter: &Painter, rect: Rect, icon: Icon, color: Color32, stroke: f32) {
    let size = rect.width().min(rect.height());
    let scale = size / icon.grid();
    let origin = rect.center() - Vec2::splat(size * 0.5);
    let map = |p: Pos2| origin + p.to_vec2() * scale;
    if icon == Icon::Kebab {
        for k in -1..=1 {
            painter.circle_filled(rect.center() + Vec2::new(0.0, k as f32 * 4.5 * scale), 1.25 * scale, color);
        }
        return;
    }
    for sp in subpaths(icon) {
        let pts: Vec<Pos2> = sp.points.iter().map(|&p| map(p)).collect();
        if pts.len() < 2 {
            continue;
        }
        if icon.filled() {
            painter.add(Shape::convex_polygon(pts, color, Stroke::NONE));
        } else {
            let st = Stroke::new((stroke * scale).max(1.0), color);
            if sp.closed {
                painter.add(Shape::closed_line(pts, st));
            } else {
                painter.add(Shape::line(pts, st));
            }
        }
    }
}

/// Draw `icon` at the standard list stroke (1.45).
pub fn paint(painter: &Painter, rect: Rect, icon: Icon, color: Color32) {
    paint_with(painter, rect, icon, color, 1.45);
}

/// An icon as a non-interactive widget of `size` points.
pub fn icon(ui: &mut Ui, icon: Icon, size: f32, color: Color32) -> Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(size), Sense::hover());
    paint(ui.painter(), rect, icon, color);
    response
}

fn logo_color_image() -> &'static egui::ColorImage {
    static IMG: OnceLock<egui::ColorImage> = OnceLock::new();
    IMG.get_or_init(|| {
        let image = image::load_from_memory(include_bytes!("../assets/zestengine-logo.png"))
            .expect("zestengine logo png")
            .into_rgba8();
        let (w, h) = image.dimensions();
        egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], image.as_raw())
    })
}

fn logo_texture(ctx: &egui::Context) -> egui::TextureHandle {
    let id = egui::Id::new("zestengine-official-logo");
    if let Some(handle) = ctx.data(|data| data.get_temp::<egui::TextureHandle>(id)) {
        return handle;
    }
    let handle = ctx.load_texture("zestengine-logo", logo_color_image().clone(), egui::TextureOptions::LINEAR);
    ctx.data_mut(|data| data.insert_temp(id, handle.clone()));
    handle
}

/// The official ZestEngine mark, from `pkg/zestengine_logo.svg`.
/// `accent`'s alpha dims the mark; its colour does not recolour it.
pub fn paint_logo(painter: &Painter, rect: Rect, accent: Color32, _dark: Color32) {
    let texture = logo_texture(painter.ctx());
    let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
    painter.image(texture.id(), rect, uv, Color32::from_white_alpha(accent.a()));
}

// ------------------------------------------------------------ path parsing

fn parse(d: &str) -> Vec<Subpath> {
    let tokens = tokenize(d);
    let mut out: Vec<Subpath> = Vec::new();
    let mut cur = Pos2::ZERO;
    let mut start = Pos2::ZERO;
    let mut last_ctrl: Option<Pos2> = None;
    let mut cmd = 'M';
    let mut i = 0;
    let num = |i: &mut usize| -> f32 {
        match tokens.get(*i) {
            Some(Tok::Num(v)) => {
                *i += 1;
                *v
            }
            _ => 0.0,
        }
    };
    while i < tokens.len() {
        if let Tok::Cmd(c) = tokens[i] {
            cmd = c;
            i += 1;
            if c == 'Z' || c == 'z' {
                if let Some(sp) = out.last_mut() {
                    sp.closed = true;
                }
                cur = start;
                last_ctrl = None;
                continue;
            }
        } else if !matches!(tokens[i], Tok::Num(_)) {
            i += 1;
            continue;
        }
        let rel = cmd.is_ascii_lowercase();
        let base = if rel { cur.to_vec2() } else { Vec2::ZERO };
        match cmd.to_ascii_uppercase() {
            'M' => {
                let p = Pos2::new(num(&mut i), num(&mut i)) + base;
                out.push(Subpath { points: vec![p], closed: false });
                cur = p;
                start = p;
                // Further pairs after a moveto are implicit linetos.
                cmd = if rel { 'l' } else { 'L' };
                last_ctrl = None;
            }
            'L' => {
                let p = Pos2::new(num(&mut i), num(&mut i)) + base;
                push(&mut out, cur, p);
                cur = p;
                last_ctrl = None;
            }
            'H' => {
                let x = num(&mut i) + base.x;
                let p = Pos2::new(x, cur.y);
                push(&mut out, cur, p);
                cur = p;
                last_ctrl = None;
            }
            'V' => {
                let y = num(&mut i) + base.y;
                let p = Pos2::new(cur.x, y);
                push(&mut out, cur, p);
                cur = p;
                last_ctrl = None;
            }
            'C' => {
                let c1 = Pos2::new(num(&mut i), num(&mut i)) + base;
                let c2 = Pos2::new(num(&mut i), num(&mut i)) + base;
                let p = Pos2::new(num(&mut i), num(&mut i)) + base;
                cubic(&mut out, cur, c1, c2, p);
                last_ctrl = Some(c2);
                cur = p;
            }
            'S' => {
                let c1 = match last_ctrl {
                    Some(c) => cur + (cur - c),
                    None => cur,
                };
                let c2 = Pos2::new(num(&mut i), num(&mut i)) + base;
                let p = Pos2::new(num(&mut i), num(&mut i)) + base;
                cubic(&mut out, cur, c1, c2, p);
                last_ctrl = Some(c2);
                cur = p;
            }
            'A' => {
                let rx = num(&mut i);
                let ry = num(&mut i);
                let rot = num(&mut i);
                let large = num(&mut i) != 0.0;
                let sweep = num(&mut i) != 0.0;
                let p = Pos2::new(num(&mut i), num(&mut i)) + base;
                arc(&mut out, cur, rx, ry, rot, large, sweep, p);
                cur = p;
                last_ctrl = None;
            }
            _ => i += 1,
        }
    }
    out
}

fn push(out: &mut Vec<Subpath>, from: Pos2, p: Pos2) {
    if out.is_empty() {
        out.push(Subpath { points: vec![from], closed: false });
    }
    out.last_mut().unwrap().points.push(p);
}

fn cubic(out: &mut Vec<Subpath>, p0: Pos2, c1: Pos2, c2: Pos2, p3: Pos2) {
    const N: usize = 10;
    for k in 1..=N {
        let t = k as f32 / N as f32;
        let u = 1.0 - t;
        let p = p0.to_vec2() * (u * u * u)
            + c1.to_vec2() * (3.0 * u * u * t)
            + c2.to_vec2() * (3.0 * u * t * t)
            + p3.to_vec2() * (t * t * t);
        push(out, p0, p.to_pos2());
    }
}

/// SVG elliptical arc (endpoint parameterisation, W3C SVG 1.1 F.6.5).
#[allow(clippy::too_many_arguments)]
fn arc(out: &mut Vec<Subpath>, p0: Pos2, rx: f32, ry: f32, rot_deg: f32, large: bool, sweep: bool, p1: Pos2) {
    if p0 == p1 {
        return;
    }
    let (mut rx, mut ry) = (rx.abs(), ry.abs());
    if rx < 1e-6 || ry < 1e-6 {
        push(out, p0, p1);
        return;
    }
    let phi = rot_deg.to_radians();
    let (sin, cos) = phi.sin_cos();
    let dx = (p0.x - p1.x) * 0.5;
    let dy = (p0.y - p1.y) * 0.5;
    let x1 = cos * dx + sin * dy;
    let y1 = -sin * dx + cos * dy;
    let lambda = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry);
    if lambda > 1.0 {
        let s = lambda.sqrt();
        rx *= s;
        ry *= s;
    }
    let num = (rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1).max(0.0);
    let den = rx * rx * y1 * y1 + ry * ry * x1 * x1;
    let mut co = if den > 0.0 { (num / den).sqrt() } else { 0.0 };
    if large == sweep {
        co = -co;
    }
    let cx1 = co * rx * y1 / ry;
    let cy1 = -co * ry * x1 / rx;
    let cx = cos * cx1 - sin * cy1 + (p0.x + p1.x) * 0.5;
    let cy = sin * cx1 + cos * cy1 + (p0.y + p1.y) * 0.5;
    let angle = |ux: f32, uy: f32, vx: f32, vy: f32| {
        
        (ux * vy - uy * vx).atan2(ux * vx + uy * vy)
    };
    let ux = (x1 - cx1) / rx;
    let uy = (y1 - cy1) / ry;
    let vx = (-x1 - cx1) / rx;
    let vy = (-y1 - cy1) / ry;
    let theta1 = angle(1.0, 0.0, ux, uy);
    let mut dtheta = angle(ux, uy, vx, vy);
    if !sweep && dtheta > 0.0 {
        dtheta -= std::f32::consts::TAU;
    } else if sweep && dtheta < 0.0 {
        dtheta += std::f32::consts::TAU;
    }
    let steps = ((dtheta.abs() / std::f32::consts::TAU) * 32.0).ceil().max(2.0) as usize;
    for k in 1..=steps {
        let t = theta1 + dtheta * k as f32 / steps as f32;
        let (st, ct) = t.sin_cos();
        let x = cos * rx * ct - sin * ry * st + cx;
        let y = sin * rx * ct + cos * ry * st + cy;
        push(out, p0, Pos2::new(x, y));
    }
}

#[derive(Clone, Copy, Debug)]
enum Tok {
    Cmd(char),
    Num(f32),
}

fn tokenize(d: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let b = d.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_ascii_alphabetic() && c != 'e' && c != 'E' {
            out.push(Tok::Cmd(c));
            i += 1;
        } else if c == '-' || c == '.' || c.is_ascii_digit() {
            let start = i;
            let mut seen_dot = false;
            if c == '-' {
                i += 1;
            }
            while i < b.len() {
                let ch = b[i] as char;
                if ch.is_ascii_digit() {
                    i += 1;
                } else if ch == '.' && !seen_dot {
                    seen_dot = true;
                    i += 1;
                } else {
                    break;
                }
            }
            if let Ok(v) = d[start..i].parse::<f32>() {
                out.push(Tok::Num(v));
            }
        } else {
            i += 1;
        }
    }
    out
}

const APP_ICON_SIZE: u32 = 256;

/// The official mark, scaled for window and desktop icons.
pub fn app_icon_rgba() -> (u32, u32, Vec<u8>) {
    static ICON: OnceLock<(u32, u32, Vec<u8>)> = OnceLock::new();
    ICON.get_or_init(|| {
        let image = image::load_from_memory(include_bytes!("../assets/zestengine-logo.png"))
            .expect("zestengine logo png")
            .resize(APP_ICON_SIZE, APP_ICON_SIZE, image::imageops::FilterType::Lanczos3)
            .into_rgba8();
        let (width, height) = image.dimensions();
        (width, height, image.into_raw())
    })
    .clone()
}

/// Refresh the user icon for Wayland, including older green ZestEngine icons.
/// Shared by the editor and hub so either entry point updates the cached PNG.
/// When ZestEngine is installed system-wide (package or `just install`, which ship every icon size),
/// the user copy is removed instead: it would shadow the installed 256 px icon with an older one.
pub fn install_app_icon() -> image::ImageResult<()> {
    let Some(home) = std::env::var_os("HOME") else { return Ok(()) };
    let data = std::env::var_os("XDG_DATA_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(home).join(".local/share"));
    let rel = "icons/hicolor/256x256/apps/zestengine.png";
    let path = data.join(rel);
    let system_dirs = std::env::var("XDG_DATA_DIRS").ok().filter(|d| !d.is_empty()).unwrap_or_else(|| "/usr/local/share:/usr/share".into());
    if system_dirs.split(':').filter(|d| !d.is_empty()).any(|d| std::path::Path::new(d).join(rel).is_file()) {
        if path.is_file() {
            let _ = std::fs::remove_file(&path);
        }
        return Ok(());
    }
    let (width, height, pixels) = app_icon_rgba();
    let buffer = image::RgbaImage::from_raw(width, height, pixels).expect("icon buffer size");
    if image::open(&path).is_ok_and(|existing| existing.to_rgba8() == buffer) {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    buffer.save(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_icon_parses_inside_its_grid() {
        for icon in Icon::ALL {
            if icon == Icon::Kebab {
                continue;
            }
            let sps = parse(icon.path());
            assert!(!sps.is_empty(), "{icon:?} has no paths");
            let g = icon.grid();
            for sp in &sps {
                for p in &sp.points {
                    assert!(p.x >= -0.5 && p.x <= g + 0.5 && p.y >= -0.5 && p.y <= g + 0.5, "{icon:?} point {p:?}");
                }
            }
        }
    }

    #[test]
    fn relative_commands_and_arcs() {
        let sps = parse("M2 5h8v6H2z");
        assert_eq!(sps[0].points, vec![Pos2::new(2.0, 5.0), Pos2::new(10.0, 5.0), Pos2::new(10.0, 11.0), Pos2::new(2.0, 11.0)]);
        assert!(sps[0].closed);
        // Two half-circle arcs make a full circle of radius 2.8 around (8, 8).
        let circle = parse("M8 5.2a2.8 2.8 0 1 0 0 5.6a2.8 2.8 0 1 0 0-5.6z");
        for p in &circle[0].points {
            assert!((p.distance(Pos2::new(8.0, 8.0)) - 2.8).abs() < 0.05, "{p:?}");
        }
        // Compact numbers: ".5.4" is 0.5 then 0.4.
        let toks = tokenize("c.5.4.6 1");
        assert_eq!(toks.len(), 5);
    }
}
