//! String helpers with Python's notion of whitespace and line breaks.
//!
//! The patch format was defined against Python's `str.strip`, `str.split`
//! and `str.splitlines`. Rust's `char::is_whitespace` omits the four
//! information separators (U+001C to U+001F) that Python strips, and
//! `str::lines` breaks on fewer characters than `splitlines`, so the port
//! uses these helpers wherever the difference would change a result.

/// Python's `str.isspace` for one character.
pub(crate) fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

pub(crate) fn rstrip(text: &str) -> &str {
    text.trim_end_matches(is_space)
}

pub(crate) fn lstrip(text: &str) -> &str {
    text.trim_start_matches(is_space)
}

pub(crate) fn strip(text: &str) -> &str {
    text.trim_matches(is_space)
}

pub(crate) fn is_blank(text: &str) -> bool {
    text.chars().all(is_space)
}

/// The leading whitespace of a line.
pub(crate) fn leading_whitespace(line: &str) -> &str {
    &line[..line.len() - lstrip(line).len()]
}

/// Indentation in characters, which is what tabs-versus-spaces arithmetic
/// works in.
pub(crate) fn indent_width(line: &str) -> usize {
    leading_whitespace(line).chars().count()
}

/// Strip trailing whitespace from every line. Lines end at `\n` only: a form
/// feed inside a string literal is an ordinary character and must survive.
pub(crate) fn strip_trailing_whitespace(text: &str) -> String {
    text.split('\n').map(rstrip).collect::<Vec<_>>().join("\n")
}

fn is_line_break(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r'
            | '\u{0b}'
            | '\u{0c}'
            | '\u{1c}'
            | '\u{1d}'
            | '\u{1e}'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// Python's `str.splitlines(keepends=True)`, with each line's start offset.
pub(crate) fn split_lines_keep_ends(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((index, c)) = chars.next() {
        if !is_line_break(c) {
            continue;
        }
        let mut end = index + c.len_utf8();
        if c == '\r' {
            if let Some((next, '\n')) = chars.peek().copied() {
                chars.next();
                end = next + 1;
            }
        }
        lines.push(&text[start..end]);
        start = end;
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// Python's `str.splitlines()`.
pub(crate) fn split_lines(text: &str) -> Vec<&str> {
    split_lines_keep_ends(text)
        .into_iter()
        .map(|line| line.trim_end_matches(is_line_break))
        .collect()
}

/// Python's `repr` of a string, close enough for a diagnostic.
pub(crate) fn quoted(text: &str) -> String {
    format!("{text:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_trailing_whitespace_keeps_line_structure() {
        // Ported from test_strip_trailing_whitespace.
        assert_eq!(
            strip_trailing_whitespace("line1   \nline2\t\nline3\nline4 \t "),
            "line1\nline2\nline3\nline4"
        );
        assert_eq!(
            strip_trailing_whitespace("a\u{1c} \nb\u{c}c \n"),
            "a\nb\u{c}c\n"
        );
    }

    #[test]
    fn splitlines_matches_python() {
        assert_eq!(
            split_lines("a\nb\r\nc\rd\u{c}e\u{2028}f"),
            ["a", "b", "c", "d", "e", "f"]
        );
        assert_eq!(split_lines("a\n"), ["a"]);
        assert_eq!(split_lines("a\n\n"), ["a", ""]);
        assert!(split_lines("").is_empty());
        assert_eq!(split_lines_keep_ends("a\r\nb"), ["a\r\n", "b"]);
    }

    #[test]
    fn indentation_is_measured_in_characters() {
        assert_eq!(indent_width("\u{3000}\tx"), 2);
        assert_eq!(leading_whitespace("\u{3000}\tx"), "\u{3000}\t");
        assert_eq!(indent_width(""), 0);
        assert!(is_blank(" \t\u{1f}"));
    }
}
