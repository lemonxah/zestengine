//! Pure logic for the Log tab's own text selection (egui's label selection
//! drops as soon as an endpoint scrolls out of the virtualised rows, so the
//! Log tracks its selection itself, by ring sequence number).

/// A caret position in the log: (entry seq, line within the message, char
/// within that line). Tuple ordering is document order.
pub type LogPos = (u64, u32, u32);

/// Order a selection's endpoints.
pub fn normalise(a: LogPos, b: LogPos) -> (LogPos, LogPos) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Signed scroll speed (px/s, negative = up) for a drag-selection whose
/// pointer is at `y` relative to a scroll area spanning `top..bottom`. Zero
/// while the pointer is further than `margin` inside the area; grows linearly
/// with how far it is past the margin boundary, up to `max_speed`.
pub fn edge_scroll_speed(y: f32, top: f32, bottom: f32, margin: f32, max_speed: f32) -> f32 {
    const MIN_SPEED: f32 = 40.0;
    const FULL_DIST: f32 = 120.0;
    let dist_up = (top + margin) - y;
    let dist_down = y - (bottom - margin);
    let (dist, sign) = if dist_up > 0.0 {
        (dist_up, -1.0)
    } else if dist_down > 0.0 {
        (dist_down, 1.0)
    } else {
        return 0.0;
    };
    sign * (max_speed * (dist / FULL_DIST).min(1.0)).clamp(MIN_SPEED.min(max_speed), max_speed)
}

/// (line, column) of the char index `idx` in `text` (lines split on '\n').
pub fn idx_to_lc(text: &str, idx: usize) -> (u32, u32) {
    let (mut line, mut col) = (0u32, 0u32);
    for (i, c) in text.chars().enumerate() {
        if i >= idx {
            break;
        }
        if c == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// Char index of (`line`, `col`) in `text`; clamps `col` to the line's end and
/// a too-large `line` to the end of the text.
pub fn lc_to_idx(text: &str, line: u32, col: u32) -> usize {
    let mut idx = 0usize;
    let mut l = 0u32;
    let mut chars = text.chars().peekable();
    while l < line {
        match chars.next() {
            Some('\n') => l += 1,
            Some(_) => {}
            None => return idx,
        }
        idx += 1;
    }
    let mut c = 0u32;
    while c < col {
        match chars.next() {
            Some('\n') | None => break,
            Some(_) => {}
        }
        idx += 1;
        c += 1;
    }
    idx
}

/// The char range (start, end exclusive) of a row's text that the selection
/// covers. `line_base == u32::MAX` means `text` is a whole (multi-line)
/// message; otherwise it is the single physical line `line_base`.
pub fn row_range(sel: (LogPos, LogPos), seq: u64, line_base: u32, text: &str) -> Option<(usize, usize)> {
    let (lo, hi) = sel;
    let whole = line_base == u32::MAX;
    let first_line = if whole { 0 } else { line_base };
    let last_line = if whole { text.matches('\n').count() as u32 } else { line_base };
    let row_start = (seq, first_line, 0);
    let row_end = (seq, last_line, u32::MAX);
    if hi < row_start || lo > row_end {
        return None;
    }
    let n = text.chars().count();
    let idx_of = |p: LogPos| if whole { lc_to_idx(text, p.1, p.2) } else { (p.2 as usize).min(n) };
    let s = if lo <= row_start { 0 } else { idx_of(lo) };
    let e = if hi >= row_end { n } else { idx_of(hi) };
    (s < e).then_some((s, e))
}

/// The selected text, given every shown entry's (seq, message) in order.
/// Lines are joined with '\n'.
pub fn extract<'a>(a: LogPos, b: LogPos, msgs: impl IntoIterator<Item = (u64, &'a str)>) -> String {
    let (lo, hi) = normalise(a, b);
    let mut out: Vec<String> = Vec::new();
    for (seq, msg) in msgs {
        if seq < lo.0 || seq > hi.0 {
            continue;
        }
        let mut lines: Vec<&str> = msg.lines().collect();
        if lines.is_empty() {
            lines.push("");
        }
        for (li, line) in lines.iter().enumerate() {
            let li = li as u32;
            let len = line.chars().count() as u32;
            if (seq, li, len) < lo || (seq, li, 0) > hi {
                continue;
            }
            let s = if lo > (seq, li, 0) { lo.2.min(len) } else { 0 };
            let e = if hi < (seq, li, len) { hi.2.min(len) } else { len };
            out.push(line.chars().skip(s as usize).take(e.saturating_sub(s) as usize).collect());
        }
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_speed_zero_inside_and_signed_outside() {
        assert_eq!(edge_scroll_speed(300.0, 100.0, 500.0, 20.0, 900.0), 0.0);
        assert_eq!(edge_scroll_speed(120.0, 100.0, 500.0, 20.0, 900.0), 0.0);
        assert!(edge_scroll_speed(110.0, 100.0, 500.0, 20.0, 900.0) < 0.0);
        assert!(edge_scroll_speed(490.0, 100.0, 500.0, 20.0, 900.0) > 0.0);
        let near = edge_scroll_speed(40.0, 100.0, 500.0, 20.0, 900.0).abs();
        let far = edge_scroll_speed(-500.0, 100.0, 500.0, 20.0, 900.0).abs();
        assert!(near < far);
        assert_eq!(far, 900.0);
    }

    #[test]
    fn lc_roundtrip() {
        let t = "ab\ncde\n\nf";
        for idx in 0..=t.chars().count() {
            let (l, c) = idx_to_lc(t, idx);
            assert_eq!(lc_to_idx(t, l, c), idx, "idx {idx}");
        }
        assert_eq!(lc_to_idx(t, 99, 0), t.chars().count());
        assert_eq!(lc_to_idx(t, 0, 99), 2);
    }

    #[test]
    fn extract_across_entries() {
        let msgs = [(1u64, "hello world"), (2, "second\nline two"), (3, "third")];
        let it = || msgs.iter().map(|&(s, m)| (s, m));
        // Single line, partial.
        assert_eq!(extract((1, 0, 6), (1, 0, 11), it()), "world");
        // Reversed endpoints give the same text.
        assert_eq!(extract((1, 0, 11), (1, 0, 6), it()), "world");
        // Across entries and a multi-line message.
        assert_eq!(extract((1, 0, 6), (3, 0, 3), it()), "world\nsecond\nline two\nthi");
        // Empty selection.
        assert_eq!(extract((2, 1, 2), (2, 1, 2), it()), "");
    }

    #[test]
    fn row_range_wrap_and_nowrap() {
        let sel = ((1, 0, 6), (3, 0, 3));
        assert_eq!(row_range(sel, 1, u32::MAX, "hello world"), Some((6, 11)));
        assert_eq!(row_range(sel, 2, u32::MAX, "second\nline two"), Some((0, 15)));
        assert_eq!(row_range(sel, 3, u32::MAX, "third"), Some((0, 3)));
        assert_eq!(row_range(sel, 4, u32::MAX, "after"), None);
        // Non-wrap: individual physical lines of entry 2.
        assert_eq!(row_range(sel, 2, 1, "line two"), Some((0, 8)));
        let one = ((2, 1, 2), (2, 1, 5));
        assert_eq!(row_range(one, 2, 0, "second"), None);
        assert_eq!(row_range(one, 2, 1, "line two"), Some((2, 5)));
    }
}
