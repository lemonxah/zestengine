//! Headless visual check of the node canvas: runs the real egui frame, then
//! rasterises the tessellated meshes with a tiny software triangle filler (no
//! GPU) so the result can be inspected as a PNG.
//!
//! `ZESTENGINE_NODE_CANVAS_PNG=<path> cargo test -p zestengine-editor --test node_canvas_render`
//! writes the image; without the variable the test only checks the frame is
//! not blank.

use std::collections::HashMap;

use egui::epaint::{ImageDelta, Primitive};
use egui::{Color32, Event, Pos2, RawInput, Rect, TextureId, ViewportId, pos2, vec2};
use zestengine_editor::node_canvas::{Link, NodeCanvasModel, NodeCanvasState, NodeStyle, PortDesc, PortRef};
use zestengine_editor::theme;

struct DemoNode {
    title: &'static str,
    chip: Color32,
    glyph: &'static str,
    ins: Vec<(&'static str, Color32, &'static str)>,
    outs: Vec<(&'static str, Color32, &'static str)>,
    pos: Pos2,
    value: Option<f32>,
}

struct Demo {
    nodes: Vec<DemoNode>,
    links: Vec<(usize, usize, usize)>, // from node, to node, to port
}

impl NodeCanvasModel for Demo {
    fn node_count(&self) -> usize {
        self.nodes.len()
    }
    fn position(&self, n: usize) -> Pos2 {
        self.nodes[n].pos
    }
    fn set_position(&mut self, n: usize, p: Pos2) {
        self.nodes[n].pos = p;
    }
    fn style(&self, n: usize) -> NodeStyle {
        let d = &self.nodes[n];
        NodeStyle { title: d.title.into(), chip: d.chip, glyph: d.glyph }
    }
    fn input_count(&self, n: usize) -> usize {
        self.nodes[n].ins.len()
    }
    fn output_count(&self, n: usize) -> usize {
        self.nodes[n].outs.len()
    }
    fn input(&self, n: usize, p: usize) -> PortDesc {
        let (l, c, t) = self.nodes[n].ins[p];
        PortDesc { label: l.into(), type_label: t, color: c }
    }
    fn output(&self, n: usize, p: usize) -> PortDesc {
        let (l, c, t) = self.nodes[n].outs[p];
        PortDesc { label: l.into(), type_label: t, color: c }
    }
    fn links(&self, out: &mut Vec<Link>) {
        for &(f, t, p) in &self.links {
            out.push(Link { from: PortRef { node: f, port: 0 }, to: PortRef { node: t, port: p }, color: self.nodes[f].outs[0].1 });
        }
    }
    fn can_connect(&self, f: PortRef, t: PortRef) -> bool {
        f.node != t.node
    }
    fn connect(&mut self, f: PortRef, t: PortRef) -> bool {
        self.links.push((f.node, t.node, t.port));
        true
    }
    fn delete_node(&mut self, _n: usize) {}
    fn matches_filter(&self, n: usize) -> bool {
        n != 5
    }
    fn body_height(&self, n: usize) -> f32 {
        if self.nodes[n].value.is_some() { 32.0 } else { 0.0 }
    }
    fn draw_body(&mut self, ui: &mut egui::Ui, n: usize) -> bool {
        if let Some(v) = &mut self.nodes[n].value {
            let size = vec2(ui.available_width(), ui.available_height());
            return ui.add_sized(size, egui::DragValue::new(v).speed(0.01)).changed();
        }
        false
    }
}

fn demo() -> Demo {
    let f = theme::WIRE_FLOAT;
    let v3 = theme::WIRE_VEC3;
    let v4 = theme::WIRE_VEC4;
    let v2 = theme::WIRE_VEC2;
    let node = |title, chip, glyph, ins: Vec<_>, outs: Vec<_>, x, y, value| DemoNode { title, chip, glyph, ins, outs, pos: pos2(x, y), value };
    Demo {
        nodes: vec![
            node("Texture Sample", theme::CAT_TEXTURE, "I", vec![("UV", v2, "Vec2")], vec![("RGBA", v4, "Vec4")], -300.0, -120.0, None),
            node("Value", theme::CAT_INPUT, "V", vec![], vec![("Value", f, "Float")], -300.0, 70.0, Some(0.5)),
            node("Multiply", theme::CAT_MATH, "M", vec![("A", v4, "Vec4"), ("B", f, "Float")], vec![("Out", v4, "Vec4")], -20.0, -60.0, None),
            node("RGB", theme::CAT_VECTOR, "R", vec![("X", v4, "Vec4")], vec![("Out", v3, "Vec3")], 250.0, -90.0, None),
            node("Output", theme::CAT_OUTPUT, "O", vec![("Color", v3, "Vec3")], vec![], 250.0, 90.0, None),
            node("Sine", theme::CAT_MATH, "S", vec![("X", f, "Float")], vec![("Out", f, "Float")], -20.0, 120.0, None),
        ],
        links: vec![(0, 2, 0), (1, 2, 1), (2, 3, 0), (3, 4, 0), (1, 5, 0)],
    }
}

// ------------------------------------------------------------ rasteriser

struct Tex {
    size: [usize; 2],
    px: Vec<Color32>,
}

fn apply_delta(texs: &mut HashMap<TextureId, Tex>, id: TextureId, d: &ImageDelta) {
    let egui::ImageData::Color(img) = &d.image;
    match d.pos {
        None => {
            texs.insert(id, Tex { size: img.size, px: img.pixels.clone() });
        }
        Some([x, y]) => {
            let t = texs.get_mut(&id).expect("patch before full upload");
            for r in 0..img.size[1] {
                for c in 0..img.size[0] {
                    t.px[(y + r) * t.size[0] + x + c] = img.pixels[r * img.size[0] + c];
                }
            }
        }
    }
}

fn sample(t: &Tex, u: f32, v: f32) -> [f32; 4] {
    let x = (u * t.size[0] as f32 - 0.5).clamp(0.0, t.size[0] as f32 - 1.0);
    let y = (v * t.size[1] as f32 - 0.5).clamp(0.0, t.size[1] as f32 - 1.0);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(t.size[0] - 1), (y0 + 1).min(t.size[1] - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let at = |x: usize, y: usize| {
        let c = t.px[y * t.size[0] + x];
        [c.r() as f32 / 255.0, c.g() as f32 / 255.0, c.b() as f32 / 255.0, c.a() as f32 / 255.0]
    };
    let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
    let mut o = [0.0; 4];
    for i in 0..4 {
        o[i] = (a[i] * (1.0 - fx) + b[i] * fx) * (1.0 - fy) + (c[i] * (1.0 - fx) + d[i] * fx) * fy;
    }
    o
}

fn render(ctx: &egui::Context, w: usize, h: usize, ppp: f32, texs: &HashMap<TextureId, Tex>, prims: &[egui::ClippedPrimitive]) -> Vec<[f32; 4]> {
    let mut fb = vec![[0.0, 0.0, 0.0, 1.0]; w * h];
    let _ = ctx;
    for cp in prims {
        let Primitive::Mesh(mesh) = &cp.primitive else { continue };
        let Some(tex) = texs.get(&mesh.texture_id) else { continue };
        let clip = cp.clip_rect;
        let (cx0, cy0) = ((clip.min.x * ppp).floor().max(0.0) as usize, (clip.min.y * ppp).floor().max(0.0) as usize);
        let (cx1, cy1) = (((clip.max.x * ppp).ceil() as usize).min(w), ((clip.max.y * ppp).ceil() as usize).min(h));
        for tri in mesh.indices.as_chunks::<3>().0 {
            let v = [mesh.vertices[tri[0] as usize], mesh.vertices[tri[1] as usize], mesh.vertices[tri[2] as usize]];
            let p = [v[0].pos * ppp, v[1].pos * ppp, v[2].pos * ppp];
            let area = (p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[2].x - p[0].x) * (p[1].y - p[0].y);
            if area.abs() < 1e-6 {
                continue;
            }
            let minx = (p.iter().map(|q| q.x).fold(f32::MAX, f32::min).floor().max(cx0 as f32)) as usize;
            let maxx = (p.iter().map(|q| q.x).fold(f32::MIN, f32::max).ceil().min(cx1 as f32)) as usize;
            let miny = (p.iter().map(|q| q.y).fold(f32::MAX, f32::min).floor().max(cy0 as f32)) as usize;
            let maxy = (p.iter().map(|q| q.y).fold(f32::MIN, f32::max).ceil().min(cy1 as f32)) as usize;
            for y in miny..maxy {
                for x in minx..maxx {
                    let q = pos2(x as f32 + 0.5, y as f32 + 0.5);
                    let w0 = ((p[1].x - q.x) * (p[2].y - q.y) - (p[2].x - q.x) * (p[1].y - q.y)) / area;
                    let w1 = ((p[2].x - q.x) * (p[0].y - q.y) - (p[0].x - q.x) * (p[2].y - q.y)) / area;
                    let w2 = 1.0 - w0 - w1;
                    if w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4 {
                        continue;
                    }
                    let u = w0 * v[0].uv.x + w1 * v[1].uv.x + w2 * v[2].uv.x;
                    let vv = w0 * v[0].uv.y + w1 * v[1].uv.y + w2 * v[2].uv.y;
                    let col = |f: fn(Color32) -> u8| {
                        (w0 * f(v[0].color) as f32 + w1 * f(v[1].color) as f32 + w2 * f(v[2].color) as f32) / 255.0
                    };
                    let (vr, vg, vb, va) = (col(|c| c.r()), col(|c| c.g()), col(|c| c.b()), col(|c| c.a()));
                    let t = sample(tex, u, vv);
                    let src = [vr * t[0], vg * t[1], vb * t[2], va * t[3]];
                    let dst = &mut fb[y * w + x];
                    for i in 0..3 {
                        dst[i] = src[i] + dst[i] * (1.0 - src[3]);
                    }
                }
            }
        }
    }
    fb
}

#[test]
fn render_node_canvas() {
    let ppp = 2.0;
    let (pw, ph) = (1000.0_f32, 600.0_f32);
    let ctx = egui::Context::default();
    zestengine_editor::install_icon_font(&ctx);
    theme::apply(&ctx);
    let mut model = demo();
    let mut state = NodeCanvasState::default();
    state.selected.insert(2);
    state.primary = Some(2);
    state.request_frame(false);

    let mut texs: HashMap<TextureId, Tex> = HashMap::new();
    let mut last = None;
    for frame in 0..3 {
        let mut input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(pw, ph))),
            viewport_id: ViewportId::ROOT,
            ..Default::default()
        };
        input.viewports.entry(ViewportId::ROOT).or_default().native_pixels_per_point = Some(ppp);
        if frame >= 1 {
            // Park the pointer on Multiply's output pin edge region to show hover.
            input.events.push(Event::PointerMoved(pos2(10.0, 10.0)));
        }
        let out = ctx.run(input, |ctx| {
            egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG_VIEWPORT)).show(ctx, |ui| {
                state.show(ui, &mut model);
            });
        });
        for (id, d) in &out.textures_delta.set {
            apply_delta(&mut texs, *id, d);
        }
        last = Some((out.shapes, out.pixels_per_point));
    }
    let (shapes, ppp) = last.unwrap();
    let prims = ctx.tessellate(shapes, ppp);
    let (w, h) = ((pw * ppp) as usize, (ph * ppp) as usize);
    let fb = render(&ctx, w, h, ppp, &texs, &prims);

    // Not blank: more than the background colour is present.
    let bg = fb[0];
    assert!(fb.iter().filter(|p| (p[0] - bg[0]).abs() > 0.05 || (p[1] - bg[1]).abs() > 0.05 || (p[2] - bg[2]).abs() > 0.05).count() > 5000);

    if let Ok(path) = std::env::var("ZESTENGINE_NODE_CANVAS_PNG") {
        let mut img = image::RgbaImage::new(w as u32, h as u32);
        for (i, p) in fb.iter().enumerate() {
            let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            img.put_pixel((i % w) as u32, (i / w) as u32, image::Rgba([q(p[0]), q(p[1]), q(p[2]), 255]));
        }
        img.save(&path).expect("write png");
    }
}
