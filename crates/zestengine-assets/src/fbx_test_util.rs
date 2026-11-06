//! A small ASCII FBX writer for tests: arbitrary node hierarchies, properties,
//! animation curves, constraints and user properties, so importer features
//! can be checked against ufbx's own evaluation without a binary fixture.

use std::fmt::Write as _;

pub(crate) const TICKS: f64 = 46_186_158_000.0;

pub(crate) fn ticks(sec: f64) -> i64 {
    (sec * TICKS).round() as i64
}

/// One animated channel of a curve node: the `d|` name, keys, interpolation flags.
pub(crate) struct Chan {
    pub name: &'static str,
    pub keys: Vec<(f64, f32)>,
    pub flags: u32,
}

/// Linear keys (`0x4`), cubic auto tangents (`0x2108` as Mixamo writes them).
pub(crate) const LINEAR: u32 = 4;
pub(crate) const CUBIC: u32 = 8456;

impl Chan {
    pub fn linear(name: &'static str, keys: &[(f64, f32)]) -> Self {
        Self { name, keys: keys.to_vec(), flags: LINEAR }
    }
    pub fn cubic(name: &'static str, keys: &[(f64, f32)]) -> Self {
        Self { name, keys: keys.to_vec(), flags: CUBIC }
    }
}

pub(crate) struct Doc {
    objects: String,
    conns: String,
    next: i64,
    pub time_mode: u8,
    /// `UnitScaleFactor` of the file (100 = one unit is a metre, 1 = a centimetre).
    pub unit_scale: f64,
    /// Z-up (front -Y) instead of Y-up.
    pub z_up: bool,
    stacks: Vec<(i64, i64)>,
}

impl Doc {
    pub fn new() -> Self {
        Self { objects: String::new(), conns: String::new(), next: 1000, time_mode: 6, unit_scale: 100.0, z_up: false, stacks: Vec::new() }
    }

    pub fn id(&mut self) -> i64 {
        self.next += 1;
        self.next
    }

    /// A model node. `props` are `P:` lines (without indentation); `parent` 0 = the scene root.
    pub fn model(&mut self, name: &str, kind: &str, props: &str, parent: i64) -> i64 {
        let id = self.id();
        let _ = write!(self.objects, "\tModel: {id}, \"Model::{name}\", \"{kind}\" {{\n\t\tVersion: 232\n\t\tProperties70:  {{\n");
        for line in props.lines().filter(|l| !l.trim().is_empty()) {
            let _ = writeln!(self.objects, "\t\t\t{}", line.trim());
        }
        self.objects.push_str("\t\t}\n\t}\n");
        let _ = writeln!(self.conns, "\tC: \"OO\",{id},{parent}");
        id
    }

    /// A bone: a `LimbNode` with its skeleton attribute.
    pub fn bone(&mut self, name: &str, props: &str, parent: i64) -> i64 {
        let id = self.model(name, "LimbNode", props, parent);
        let attr = self.id();
        let _ = write!(self.objects, "\tNodeAttribute: {attr}, \"NodeAttribute::\", \"LimbNode\" {{\n\t\tTypeFlags: \"Skeleton\"\n\t}}\n");
        let _ = writeln!(self.conns, "\tC: \"OO\",{attr},{id}");
        id
    }

    pub fn raw_object(&mut self, text: &str) {
        self.objects.push_str(text);
    }

    pub fn conn(&mut self, line: &str) {
        let _ = writeln!(self.conns, "\t{line}");
    }

    /// An animation stack with one layer; `curves` = (target model, animated property, channels).
    pub fn stack(&mut self, name: &str, span: (f64, f64), curves: &[(i64, &str, Vec<Chan>)]) {
        let stack = self.id();
        let layer = self.id();
        let (a, b) = (ticks(span.0), ticks(span.1));
        let _ = write!(
            self.objects,
            "\tAnimationStack: {stack}, \"AnimStack::{name}\", \"\" {{\n\t\tProperties70:  {{\n\t\t\tP: \"LocalStart\", \"KTime\", \"Time\", \"\",{a}\n\t\t\tP: \"LocalStop\", \"KTime\", \"Time\", \"\",{b}\n\t\t\tP: \"ReferenceStart\", \"KTime\", \"Time\", \"\",{a}\n\t\t\tP: \"ReferenceStop\", \"KTime\", \"Time\", \"\",{b}\n\t\t}}\n\t}}\n\tAnimationLayer: {layer}, \"AnimLayer::BaseLayer\", \"\" {{\n\t}}\n"
        );
        let _ = writeln!(self.conns, "\tC: \"OO\",{layer},{stack}");
        self.stacks.push((stack, layer));
        for (target, prop, chans) in curves {
            let node = self.id();
            let _ = write!(self.objects, "\tAnimationCurveNode: {node}, \"AnimCurveNode::{prop}\", \"\" {{\n\t\tProperties70:  {{\n");
            for c in chans {
                let _ = writeln!(self.objects, "\t\t\tP: \"{}\", \"Number\", \"\", \"A\",0", c.name);
            }
            self.objects.push_str("\t\t}\n\t}\n");
            let _ = write!(self.conns, "\tC: \"OO\",{node},{layer}\n\tC: \"OP\",{node},{target},\"{prop}\"\n");
            for c in chans {
                let cid = self.id();
                let n = c.keys.len();
                let kt: Vec<String> = c.keys.iter().map(|k| ticks(k.0).to_string()).collect();
                let kv: Vec<String> = c.keys.iter().map(|k| k.1.to_string()).collect();
                let _ = write!(
                    self.objects,
                    "\tAnimationCurve: {cid}, \"AnimCurve::\", \"\" {{\n\t\tDefault: 0\n\t\tKeyVer: 4009\n\t\tKeyTime: *{n} {{\n\t\t\ta: {}\n\t\t}}\n\t\tKeyValueFloat: *{n} {{\n\t\t\ta: {}\n\t\t}}\n\t\tKeyAttrFlags: *1 {{\n\t\t\ta: {}\n\t\t}}\n\t\tKeyAttrDataFloat: *4 {{\n\t\t\ta: 0,0,0,0\n\t\t}}\n\t\tKeyAttrRefCount: *1 {{\n\t\t\ta: {n}\n\t\t}}\n\t}}\n",
                    kt.join(","),
                    kv.join(","),
                    c.flags
                );
                let _ = writeln!(self.conns, "\tC: \"OP\",{cid},{node},\"{}\"", c.name);
            }
        }
    }

    /// A constraint of `kind` ("Position From Positions", "Parent-Child", "Aim", ...) on
    /// `node`, following `sources` (name, id); `props` are extra `P:` lines.
    pub fn constraint(&mut self, kind: &str, node: i64, sources: &[(&str, i64)], props: &str) -> i64 {
        let id = self.id();
        let _ = write!(self.objects, "\tConstraint: {id}, \"Constraint::C{id}\", \"Constraint\" {{\n\t\tType: \"{kind}\"\n\t\tVersion: 100\n\t\tProperties70:  {{\n");
        for line in props.lines().filter(|l| !l.trim().is_empty()) {
            let _ = writeln!(self.objects, "\t\t\t{}", line.trim());
        }
        for (name, _) in sources {
            let _ = writeln!(self.objects, "\t\t\tP: \"{name}.Weight\", \"Number\", \"\", \"A+\",100");
        }
        self.objects.push_str("\t\t}\n\t}\n");
        let _ = writeln!(self.conns, "\tC: \"OP\",{node},{id},\"Constrained Object\"");
        for (_, sid) in sources {
            let _ = writeln!(self.conns, "\tC: \"OP\",{sid},{id},\"Source\"");
        }
        id
    }

    pub fn render(&self) -> String {
        let mut s = String::new();
        let _ = write!(
            s,
            "; FBX 7.4.0 project file\nFBXHeaderExtension:  {{\n\tFBXHeaderVersion: 1003\n\tFBXVersion: 7400\n}}\n\
GlobalSettings:  {{\n\tVersion: 1000\n\tProperties70:  {{\n\
\t\tP: \"UpAxis\", \"int\", \"Integer\", \"\",{}\n\t\tP: \"UpAxisSign\", \"int\", \"Integer\", \"\",1\n\
\t\tP: \"FrontAxis\", \"int\", \"Integer\", \"\",{}\n\t\tP: \"FrontAxisSign\", \"int\", \"Integer\", \"\",{}\n\
\t\tP: \"CoordAxis\", \"int\", \"Integer\", \"\",0\n\t\tP: \"CoordAxisSign\", \"int\", \"Integer\", \"\",1\n\
\t\tP: \"UnitScaleFactor\", \"double\", \"Number\", \"\",{}\n\t\tP: \"TimeMode\", \"enum\", \"\", \"\",{}\n\t}}\n}}\n",
            if self.z_up { 2 } else { 1 },
            if self.z_up { 1 } else { 2 },
            if self.z_up { -1 } else { 1 },
            self.unit_scale,
            self.time_mode
        );
        s += "Objects:  {\n";
        s += &self.objects;
        s += "}\nConnections:  {\n";
        s += &self.conns;
        s += "}\n";
        s
    }

    pub fn write(&self, tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("zest_fbx_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.fbx");
        std::fs::write(&path, self.render()).unwrap();
        path
    }
}

/// `P:` line of a vector property.
pub(crate) fn vec_prop(name: &str, ty: &str, flags: &str, v: [f32; 3]) -> String {
    format!("P: \"{name}\", \"{ty}\", \"\", \"{flags}\",{},{},{}", v[0], v[1], v[2])
}

/// Import a written file with `settings` and remove it again.
pub(crate) fn import(doc: &Doc, tag: &str, settings: &crate::ModelImport) -> anyhow::Result<crate::Scene> {
    let path = doc.write(tag);
    let scene = crate::load_fbx_with(&path, settings);
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
    scene
}
