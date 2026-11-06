//! The value / argument-block encoding of the engine's WASM component ABI
//! (see `crates/zestengine-core/src/wasm/mod.rs`). Little-endian, tagged.
//! Self-contained so the engine's tests can check it against the host codec.

/// A value passed to an action or returned from a property.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f32),
    Str(String),
    Vec3([f32; 3]),
    /// An object id as 16 bytes; all zero = none.
    Object([u8; 16]),
}

const MAX_STRING: usize = 16 * 1024;
const MAX_ARGS: usize = 32;

fn take<'a>(buf: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], &'static str> {
    let end = pos.checked_add(n).filter(|e| *e <= buf.len()).ok_or("value is cut short")?;
    let s = &buf[*pos..end];
    *pos = end;
    Ok(s)
}

pub fn encode_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Bool(b) => out.extend_from_slice(&[0, *b as u8]),
        Value::Int(i) => {
            out.push(1);
            out.extend_from_slice(&i.to_le_bytes());
        }
        Value::Float(f) => {
            out.push(2);
            out.extend_from_slice(&f.to_le_bytes());
        }
        Value::Str(s) => {
            let mut end = s.len().min(MAX_STRING);
            while !s.is_char_boundary(end) {
                end -= 1;
            }
            out.push(3);
            out.extend_from_slice(&(end as u32).to_le_bytes());
            out.extend_from_slice(&s.as_bytes()[..end]);
        }
        Value::Vec3(v) => {
            out.push(4);
            for c in v {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        Value::Object(o) => {
            out.push(5);
            out.extend_from_slice(o);
        }
    }
}

/// `u32` count, then the values.
pub fn encode_args(args: &[Value]) -> Vec<u8> {
    let mut out = (args.len() as u32).to_le_bytes().to_vec();
    for a in args {
        encode_value(&mut out, a);
    }
    out
}

pub fn decode_value(buf: &[u8], pos: &mut usize) -> Result<Value, &'static str> {
    Ok(match take(buf, pos, 1)?[0] {
        0 => Value::Bool(take(buf, pos, 1)?[0] != 0),
        1 => Value::Int(i64::from_le_bytes(take(buf, pos, 8)?.try_into().unwrap())),
        2 => Value::Float(f32::from_le_bytes(take(buf, pos, 4)?.try_into().unwrap())),
        3 => {
            let len = u32::from_le_bytes(take(buf, pos, 4)?.try_into().unwrap()) as usize;
            if len > MAX_STRING {
                return Err("string too long");
            }
            Value::Str(String::from_utf8(take(buf, pos, len)?.to_vec()).map_err(|_| "string is not UTF-8")?)
        }
        4 => {
            let b = take(buf, pos, 12)?;
            let f = |i: usize| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
            Value::Vec3([f(0), f(1), f(2)])
        }
        5 => Value::Object(take(buf, pos, 16)?.try_into().unwrap()),
        _ => return Err("unknown value tag"),
    })
}

pub fn decode_args(buf: &[u8]) -> Result<Vec<Value>, &'static str> {
    let mut pos = 0;
    let count = u32::from_le_bytes(take(buf, &mut pos, 4)?.try_into().unwrap()) as usize;
    if count > MAX_ARGS {
        return Err("too many arguments");
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(decode_value(buf, &mut pos)?);
    }
    Ok(out)
}

impl Value {
    pub fn as_bool(&self) -> bool {
        matches!(self, Value::Bool(true))
    }
    pub fn as_i64(&self) -> i64 {
        match self {
            Value::Int(i) => *i,
            Value::Float(f) => *f as i64,
            Value::Bool(b) => *b as i64,
            _ => 0,
        }
    }
    pub fn as_f32(&self) -> f32 {
        match self {
            Value::Float(f) => *f,
            Value::Int(i) => *i as f32,
            _ => 0.0,
        }
    }
    pub fn as_str(&self) -> &str {
        match self {
            Value::Str(s) => s,
            _ => "",
        }
    }
    pub fn as_vec3(&self) -> [f32; 3] {
        match self {
            Value::Vec3(v) => *v,
            _ => [0.0; 3],
        }
    }
}
