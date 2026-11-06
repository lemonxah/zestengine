//! The byte-level contract between the engine and a WASM component: the
//! argument / value encoding and the manifest. See [`super`] for the whole ABI.

use serde::{Deserialize, Serialize};

use crate::actions::{ActionType, ArgValue};
use crate::{ObjectId, ObjectRef};

// ------------------------------------------------------------------ encoding

/// Value tags. The numbers are part of the ABI; never reuse one.
pub const TAG_BOOL: u8 = 0;
/// Tag of an Int value.
pub const TAG_INT: u8 = 1;
/// Tag of a Float value.
pub const TAG_FLOAT: u8 = 2;
/// Tag of a String value.
pub const TAG_STRING: u8 = 3;
/// Tag of a Vec3 value.
pub const TAG_VEC3: u8 = 4;
/// Tag of an Object value.
pub const TAG_OBJECT: u8 = 5;

/// Most arguments one call may carry.
pub const MAX_ARGS: usize = 32;
/// Longest string value (bytes).
pub const MAX_STRING_BYTES: usize = 16 * 1024;
/// Largest encoded argument block (bytes).
pub const MAX_ENCODED_BYTES: usize = 64 * 1024;

/// Append one tagged value (see the module docs of [`super`] for the layout).
pub fn encode_value(out: &mut Vec<u8>, v: &ArgValue) {
    match v {
        ArgValue::Bool(b) => {
            out.push(TAG_BOOL);
            out.push(*b as u8);
        }
        ArgValue::Int(i) => {
            out.push(TAG_INT);
            out.extend_from_slice(&i.to_le_bytes());
        }
        ArgValue::Float(f) => {
            out.push(TAG_FLOAT);
            out.extend_from_slice(&f.to_le_bytes());
        }
        ArgValue::String(s) => {
            // Cut on a char boundary so the guest always gets valid UTF-8.
            let mut end = s.len().min(MAX_STRING_BYTES);
            while !s.is_char_boundary(end) {
                end -= 1;
            }
            out.push(TAG_STRING);
            out.extend_from_slice(&(end as u32).to_le_bytes());
            out.extend_from_slice(&s.as_bytes()[..end]);
        }
        ArgValue::Vec3(v) => {
            out.push(TAG_VEC3);
            for c in v {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        ArgValue::Object(o) => {
            out.push(TAG_OBJECT);
            out.extend_from_slice(&object_to_bytes(*o));
        }
    }
}

/// `u32` count, then each value.
pub fn encode_args(args: &[ArgValue]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + args.len() * 9);
    encode_args_into(&mut out, args);
    out
}

/// [`encode_args`] into a reusable buffer (cleared first).
pub fn encode_args_into(out: &mut Vec<u8>, args: &[ArgValue]) {
    out.clear();
    out.extend_from_slice(&(args.len() as u32).to_le_bytes());
    for a in args {
        encode_value(out, a);
    }
}

/// An object reference as 16 bytes: the raw id, little-endian; all zero = none.
pub fn object_to_bytes(o: ObjectRef) -> [u8; 16] {
    o.id().map(|i| i.raw()).unwrap_or(0).to_le_bytes()
}

pub fn object_from_bytes(b: [u8; 16]) -> ObjectRef {
    match u128::from_le_bytes(b) {
        0 => ObjectRef::NONE,
        raw => ObjectRef::new(ObjectId::from_raw(raw)),
    }
}

fn take<'a>(buf: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], String> {
    let end = pos.checked_add(n).filter(|e| *e <= buf.len()).ok_or("value is cut short")?;
    let s = &buf[*pos..end];
    *pos = end;
    Ok(s)
}

/// Read one tagged value at `pos`.
pub fn decode_value(buf: &[u8], pos: &mut usize) -> Result<ArgValue, String> {
    let tag = take(buf, pos, 1)?[0];
    Ok(match tag {
        TAG_BOOL => ArgValue::Bool(take(buf, pos, 1)?[0] != 0),
        TAG_INT => ArgValue::Int(i64::from_le_bytes(take(buf, pos, 8)?.try_into().unwrap())),
        TAG_FLOAT => ArgValue::Float(f32::from_le_bytes(take(buf, pos, 4)?.try_into().unwrap())),
        TAG_STRING => {
            let len = u32::from_le_bytes(take(buf, pos, 4)?.try_into().unwrap()) as usize;
            if len > MAX_STRING_BYTES {
                return Err(format!("string of {len} bytes (limit {MAX_STRING_BYTES})"));
            }
            let bytes = take(buf, pos, len)?;
            ArgValue::String(std::str::from_utf8(bytes).map_err(|_| "string is not UTF-8")?.to_owned())
        }
        TAG_VEC3 => {
            let b = take(buf, pos, 12)?;
            let f = |i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
            ArgValue::Vec3([f(0), f(1), f(2)])
        }
        TAG_OBJECT => ArgValue::Object(object_from_bytes(take(buf, pos, 16)?.try_into().unwrap())),
        t => return Err(format!("unknown value tag {t}")),
    })
}

/// Decode an argument block written by [`encode_args`]. Trailing bytes are an error.
pub fn decode_args(buf: &[u8]) -> Result<Vec<ArgValue>, String> {
    if buf.len() > MAX_ENCODED_BYTES {
        return Err(format!("argument block of {} bytes (limit {MAX_ENCODED_BYTES})", buf.len()));
    }
    let mut pos = 0;
    let count = u32::from_le_bytes(take(buf, &mut pos, 4)?.try_into().unwrap()) as usize;
    if count > MAX_ARGS {
        return Err(format!("{count} arguments (limit {MAX_ARGS})"));
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(decode_value(buf, &mut pos)?);
    }
    if pos != buf.len() {
        return Err("trailing bytes after the arguments".into());
    }
    Ok(out)
}

/// Decode a buffer holding exactly one value (a property result).
pub fn decode_one(buf: &[u8]) -> Result<ArgValue, String> {
    let mut pos = 0;
    let v = decode_value(buf, &mut pos)?;
    if pos != buf.len() {
        return Err("trailing bytes after the value".into());
    }
    Ok(v)
}

// ------------------------------------------------------------------ manifest

/// The manifest format version this engine reads.
/// The manifest format version this engine reads.
pub const MANIFEST_API: u32 = 1;
/// Capabilities a manifest may request (see [`super`]).
pub const CAPABILITIES: &[&str] = &["state", "events", "read", "invoke"];

const MAX_ACTIONS: usize = 64;
const MAX_PARAMS: usize = 16;
const MAX_PROPERTIES: usize = 64;
const MAX_NAME: usize = 64;
/// Longest manifest text (bytes).
pub const MAX_MANIFEST_BYTES: usize = 64 * 1024;

/// One parameter of a declared action.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ManifestParam {
    /// Parameter name.
    pub name: String,
    /// Parameter type.
    pub ty: ActionType,
    /// Used when a binding gives fewer arguments; the zero value of `ty` if unset.
    pub default: Option<ArgValue>,
}

impl Default for ManifestParam {
    fn default() -> Self {
        Self { name: String::new(), ty: ActionType::Float, default: None }
    }
}

/// A callable function (index in the list = `action_index` of `zest_invoke`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct ManifestAction {
    /// Action name (what bindings call).
    pub name: String,
    /// Dropdown text; the name when empty.
    pub label: String,
    /// The action's parameters.
    pub params: Vec<ManifestParam>,
    /// Safe to bind from untrusted content (see `docs/user-content.md`).
    pub ugc_safe: bool,
}


/// A readable value (index in the list = `index` of `zest_read_property`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ManifestProperty {
    /// Property name.
    pub name: String,
    /// Property type.
    pub ty: ActionType,
}

impl Default for ManifestProperty {
    fn default() -> Self {
        Self { name: String::new(), ty: ActionType::Float }
    }
}

/// What a module declares about itself (RON; see [`super`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Manifest {
    /// Shown in the inspector.
    pub name: String,
    /// Manifest format version (`MANIFEST_API`).
    pub api: u32,
    /// Host capabilities the module imports: any of [`CAPABILITIES`].
    pub capabilities: Vec<String>,
    /// The module's declared actions.
    pub actions: Vec<ManifestAction>,
    /// The module's declared properties.
    pub properties: Vec<ManifestProperty>,
}

fn check_name(what: &str, name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > MAX_NAME {
        return Err(format!("{what} name {name:?} must be 1 to {MAX_NAME} bytes"));
    }
    if !name.chars().all(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | ' ' | '.')) {
        return Err(format!("{what} name {name:?} has characters other than letters, digits, space, _ - ."));
    }
    Ok(())
}

impl Manifest {
    /// Parse and validate manifest text.
    pub fn parse(text: &str) -> Result<Manifest, String> {
        if text.len() > MAX_MANIFEST_BYTES {
            return Err(format!("manifest is {} bytes (limit {MAX_MANIFEST_BYTES})", text.len()));
        }
        let m: Manifest = ron::from_str(text).map_err(|e| format!("manifest does not parse: {e}"))?;
        m.validate()?;
        Ok(m)
    }

    /// Check the manifest's shape (sizes, names, duplicates, capabilities).
    pub fn validate(&self) -> Result<(), String> {
        if self.api > MANIFEST_API {
            return Err(format!("manifest api {} is newer than this engine's {MANIFEST_API}", self.api));
        }
        if self.actions.len() > MAX_ACTIONS {
            return Err(format!("{} actions (limit {MAX_ACTIONS})", self.actions.len()));
        }
        if self.properties.len() > MAX_PROPERTIES {
            return Err(format!("{} properties (limit {MAX_PROPERTIES})", self.properties.len()));
        }
        for c in &self.capabilities {
            if !CAPABILITIES.contains(&c.as_str()) {
                return Err(format!("unknown capability {c:?} (known: {})", CAPABILITIES.join(", ")));
            }
        }
        for (i, a) in self.actions.iter().enumerate() {
            check_name("action", &a.name)?;
            if self.actions[..i].iter().any(|b| b.name == a.name) {
                return Err(format!("action {:?} is declared twice", a.name));
            }
            if a.params.len() > MAX_PARAMS {
                return Err(format!("action {:?} has {} parameters (limit {MAX_PARAMS})", a.name, a.params.len()));
            }
            for p in &a.params {
                check_name("parameter", &p.name)?;
                if let Some(d) = &p.default
                    && d.ty() != p.ty
                {
                    return Err(format!("action {:?} parameter {:?}: default is {}, not {}", a.name, p.name, d.ty().label(), p.ty.label()));
                }
            }
        }
        for (i, p) in self.properties.iter().enumerate() {
            check_name("property", &p.name)?;
            if self.properties[..i].iter().any(|b| b.name == p.name) {
                return Err(format!("property {:?} is declared twice", p.name));
            }
        }
        Ok(())
    }

    /// True when the manifest requests the named capability.
    pub fn has_capability(&self, cap: &str) -> bool {
        self.capabilities.iter().any(|c| c == cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every_type() -> Vec<ArgValue> {
        vec![
            ArgValue::Bool(true),
            ArgValue::Bool(false),
            ArgValue::Int(i64::MIN),
            ArgValue::Int(-1),
            ArgValue::Int(i64::MAX),
            ArgValue::Float(-0.5),
            ArgValue::Float(f32::MAX),
            ArgValue::String(String::new()),
            ArgValue::String("héllo wörld \u{1F600}".into()),
            ArgValue::Vec3([1.0, -2.5, 3.25]),
            ArgValue::Object(ObjectRef::NONE),
            ArgValue::Object(ObjectRef::new(ObjectId::new())),
        ]
    }

    #[test]
    fn every_value_type_round_trips() {
        let args = every_type();
        let bytes = encode_args(&args);
        assert_eq!(decode_args(&bytes).unwrap(), args);
        for a in &args {
            let mut one = Vec::new();
            encode_value(&mut one, a);
            assert_eq!(&decode_one(&one).unwrap(), a);
        }
    }

    #[test]
    fn layout_is_the_documented_one() {
        let b = encode_args(&[ArgValue::Bool(true), ArgValue::Int(258), ArgValue::String("ab".into())]);
        assert_eq!(
            b,
            [3, 0, 0, 0, /* bool */ 0, 1, /* int */ 1, 2, 1, 0, 0, 0, 0, 0, 0, /* str */ 3, 2, 0, 0, 0, b'a', b'b']
        );
        let mut f = Vec::new();
        encode_value(&mut f, &ArgValue::Float(1.0));
        assert_eq!(f, [2, 0, 0, 0x80, 0x3f]);
    }

    #[test]
    fn malformed_blocks_are_rejected_not_panicked_on() {
        let good = encode_args(&every_type());
        for cut in 0..good.len() {
            assert!(decode_args(&good[..cut]).is_err(), "cut at {cut}");
        }
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(decode_args(&trailing).is_err());
        assert!(decode_args(&[1, 0, 0, 0, 99]).is_err(), "unknown tag");
        assert!(decode_args(&[255, 255, 255, 255]).is_err(), "huge count");
        assert!(decode_args(&[1, 0, 0, 0, TAG_STRING, 2, 0, 0, 0, 0xff, 0xfe]).is_err(), "bad utf-8");
        assert!(decode_args(&[1, 0, 0, 0, TAG_STRING, 0xff, 0xff, 0xff, 0x7f]).is_err(), "huge string");
    }

    #[test]
    fn long_strings_are_cut_on_a_char_boundary() {
        let s = "é".repeat(MAX_STRING_BYTES);
        let b = encode_args(&[ArgValue::String(s)]);
        match &decode_args(&b).unwrap()[0] {
            ArgValue::String(t) => assert_eq!(t.len(), MAX_STRING_BYTES),
            _ => panic!(),
        }
    }

    const MANIFEST: &str = r#"(
        name: "Door",
        api: 1,
        capabilities: ["state", "events"],
        actions: [
            (name: "Open", label: "Open door", ugc_safe: true, params: [
                (name: "speed", ty: Float, default: Some(Float(2.0))),
                (name: "who", ty: Object),
            ]),
            (name: "Lock"),
        ],
        properties: [(name: "state", ty: Int)],
    )"#;

    #[test]
    fn manifest_parses_with_defaults() {
        let m = Manifest::parse(MANIFEST).unwrap();
        assert_eq!(m.name, "Door");
        assert_eq!(m.actions.len(), 2);
        assert!(m.actions[0].ugc_safe && !m.actions[1].ugc_safe);
        assert_eq!(m.actions[0].params[0].default, Some(ArgValue::Float(2.0)));
        assert_eq!(m.actions[0].params[1].default, None);
        assert_eq!(m.properties[0].ty, ActionType::Int);
        assert!(m.has_capability("events") && !m.has_capability("invoke"));
        assert_eq!(Manifest::parse("()").unwrap(), Manifest::default());
    }

    #[test]
    fn manifest_validation_rejects_bad_files() {
        for bad in [
            "(",
            r#"(api: 9)"#,
            r#"(capabilities: ["root"])"#,
            r#"(actions: [(name: "")])"#,
            r#"(actions: [(name: "A"), (name: "A")])"#,
            r#"(actions: [(name: "A", params: [(name: "x", ty: Int, default: Some(Float(1.0)))])])"#,
            r#"(properties: [(name: "a/b", ty: Int)])"#,
        ] {
            assert!(Manifest::parse(bad).is_err(), "{bad}");
        }
    }
}
