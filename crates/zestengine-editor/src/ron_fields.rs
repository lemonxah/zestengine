//! Top-level fields of a component's RON (`(name: value, ...)`), so a
//! multi-object edit can copy just the fields that changed and the
//! inspector can tell which fields differ across a selection.

/// Split `(a: 1, b: (x: 2), c: "s, t")` into its fields as (name, value
/// text). None when the text isn't a named-field struct.
pub fn split(ron: &str) -> Option<Vec<(String, String)>> {
    let body = ron.trim().strip_prefix('(')?.strip_suffix(')')?;
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    let mut start = 0;
    let bytes = body.as_bytes();
    let push = |piece: &str, out: &mut Vec<(String, String)>| -> Option<()> {
        let piece = piece.trim();
        if piece.is_empty() {
            return Some(());
        }
        let (name, value) = piece.split_once(':')?;
        let name = name.trim();
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return None;
        }
        out.push((name.to_owned(), value.trim().to_owned()));
        Some(())
    };
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            match (escaped, b) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => in_str = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b',' if depth == 0 => {
                push(&body[start..i], &mut out)?;
                start = i + 1;
            }
            _ => {}
        }
    }
    push(&body[start..], &mut out)?;
    Some(out)
}

pub fn join(fields: &[(String, String)]) -> String {
    let inner: Vec<String> = fields.iter().map(|(k, v)| format!("{k}:{v}")).collect();
    format!("({})", inner.join(","))
}

/// Fields whose value differs between `before` and `after`.
pub fn changed(before: &str, after: &str) -> Option<Vec<(String, String)>> {
    let (b, a) = (split(before)?, split(after)?);
    Some(a.into_iter().filter(|(k, v)| b.iter().find(|(bk, _)| bk == k).map(|(_, bv)| bv) != Some(v)).collect())
}

/// `target` with the given fields replaced (fields it lacks are added).
pub fn patch(target: &str, fields: &[(String, String)]) -> Option<String> {
    let mut t = split(target)?;
    for (k, v) in fields {
        match t.iter_mut().find(|(tk, _)| tk == k) {
            Some(slot) => slot.1 = v.clone(),
            None => t.push((k.clone(), v.clone())),
        }
    }
    Some(join(&t))
}

/// Names of the fields whose values aren't the same in every one of `rons`.
pub fn mixed(rons: &[String]) -> Vec<String> {
    let parsed: Vec<Vec<(String, String)>> = rons.iter().filter_map(|r| split(r)).collect();
    let Some(first) = parsed.first() else { return Vec::new() };
    first
        .iter()
        .filter(|(k, v)| parsed[1..].iter().any(|p| p.iter().find(|(pk, _)| pk == k).map(|(_, pv)| pv) != Some(v)))
        .map(|(k, _)| k.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_nested_values_and_strings() {
        let f = split(r#"(kind:Spot,color:(1.0,0.5,0.2),name:"a, b (c)",list:[1,2])"#).unwrap();
        let names: Vec<&str> = f.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(names, ["kind", "color", "name", "list"]);
        assert_eq!(f[2].1, r#""a, b (c)""#);
        assert_eq!(join(&f), r#"(kind:Spot,color:(1.0,0.5,0.2),name:"a, b (c)",list:[1,2])"#);
    }

    #[test]
    fn edits_copy_only_changed_fields() {
        let before = "(intensity:1.0,color:(1.0,1.0,1.0),range:5.0)";
        let after = "(intensity:3.0,color:(1.0,1.0,1.0),range:5.0)";
        let d = changed(before, after).unwrap();
        assert_eq!(d, vec![("intensity".to_owned(), "3.0".to_owned())]);
        let other = "(intensity:2.0,color:(0.0,1.0,0.0),range:9.0)";
        assert_eq!(patch(other, &d).unwrap(), "(intensity:3.0,color:(0.0,1.0,0.0),range:9.0)");
        assert_eq!(mixed(&[before.into(), other.into()]), vec!["intensity".to_owned(), "color".to_owned(), "range".to_owned()]);
        assert!(mixed(&[before.into(), before.into()]).is_empty());
    }
}
