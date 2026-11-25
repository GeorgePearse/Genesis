//! EVOLVE-BLOCK markers: where a program may be edited, and whether a
//! candidate still carries well-formed markers.
//!
//! A marker is the literal `EVOLVE-BLOCK-START` or `EVOLVE-BLOCK-END`,
//! usually inside a comment. The text after a START marker and before the
//! END marker that closes it is mutable; everything else, the marker lines
//! included, is read-only.

use crate::languages::language;
use crate::text::{is_space, quoted, rstrip, split_lines, strip};
use regex::Regex;
use std::sync::LazyLock;

pub(crate) const START_MARKER: &str = "EVOLVE-BLOCK-START";
pub(crate) const END_MARKER: &str = "EVOLVE-BLOCK-END";

/// Python's `\s`: Unicode whitespace plus the information separators.
const WS: &str = r"[\s\x1C-\x1F]";
/// Python's `[^\S\r\n]`: whitespace that does not end a line.
const HWS: &str = r"[\t\x0B\x0C\x1C-\x1F \x{85}\x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}]";

fn marker_regex(marker: &str) -> Regex {
    Regex::new(&format!(
        r"(?:#|//|!|<!--|\(\*)?{HWS}*{marker}{HWS}*(?:-->|\*\))?"
    ))
    .expect("marker pattern")
}

static EVOLVE_START: LazyLock<Regex> = LazyLock::new(|| marker_regex(START_MARKER));
static EVOLVE_END: LazyLock<Regex> = LazyLock::new(|| marker_regex(END_MARKER));

static MARKER_COMMENT_OPENER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(#|//|!|<!--|\(\*){HWS}*EVOLVE-BLOCK-(?:START|END)"
    ))
    .expect("opener pattern")
});

/// Whole lines that are nothing but a marker, in every comment style.
static MARKER_LINES: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    let wrappers = [
        ("#", ""),
        ("//", ""),
        ("!", ""),
        ("<!--", "-->"),
        (r"\(\*", r"\*\)"),
        ("", ""),
    ];
    let mut patterns = Vec::new();
    for marker in [START_MARKER, END_MARKER] {
        for (open, close) in wrappers {
            let open = if open.is_empty() {
                String::new()
            } else {
                format!("{open}{WS}*")
            };
            let close = if close.is_empty() {
                String::new()
            } else {
                format!("{close}{WS}*")
            };
            let pattern = format!(r"(?m)^{WS}*{open}{marker}{WS}*{close}$");
            patterns.push(Regex::new(&pattern).expect("marker line pattern"));
        }
    }
    patterns
});

/// Byte span of every START marker, comment delimiters included.
pub(crate) fn start_markers(text: &str) -> Vec<(usize, usize)> {
    EVOLVE_START
        .find_iter(text)
        .map(|m| (m.start(), m.end()))
        .collect()
}

/// Byte span of every END marker, comment delimiters included.
pub(crate) fn end_markers(text: &str) -> Vec<(usize, usize)> {
    EVOLVE_END
        .find_iter(text)
        .map(|m| (m.start(), m.end()))
        .collect()
}

pub(crate) fn has_start_marker(text: &str) -> bool {
    EVOLVE_START.is_match(text)
}

/// Byte ranges that are legal to edit.
///
/// A range starts right after a START marker and ends where the END marker
/// that closes it begins, so the indentation of the END line belongs to the
/// range. Markers pair like brackets: nested blocks yield the inner range
/// first, and an END without a START, or a START that is never closed, yields
/// nothing. Ranges come in the order their END markers appear.
pub fn mutable_ranges(text: &str) -> Vec<(usize, usize)> {
    let mut markers: Vec<(usize, bool)> = start_markers(text)
        .into_iter()
        .map(|(_, end)| (end, true))
        .collect();
    markers.extend(
        end_markers(text)
            .into_iter()
            .map(|(start, _)| (start, false)),
    );
    markers.sort_by_key(|(position, _)| *position);

    let mut spans = Vec::new();
    let mut open = Vec::new();
    for (position, is_start) in markers {
        if is_start {
            open.push(position);
        } else if let Some(start) = open.pop() {
            spans.push((start, position));
        }
    }
    spans
}

/// The mutable ranges that are not inside another one, in file order.
pub fn outermost_mutable_ranges(text: &str) -> Vec<(usize, usize)> {
    let mut ranges = mutable_ranges(text);
    ranges.sort();
    let mut outer: Vec<(usize, usize)> = Vec::new();
    for (start, end) in ranges {
        match outer.last() {
            Some((_, last_end)) if start < *last_end => {}
            _ => outer.push((start, end)),
        }
    }
    outer
}

/// The read-only text of a program: what lies before, between and after its
/// outermost mutable ranges. A program with `n` regions has `n + 1` parts.
pub fn immutable_parts(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut cursor = 0;
    for (start, end) in outermost_mutable_ranges(text) {
        parts.push(&text[cursor..start]);
        cursor = end;
    }
    parts.push(&text[cursor..]);
    parts
}

/// Why `updated` may not replace `original`, if applying a patch moved the
/// boundary between editable and read-only text.
///
/// The check is on the result, not on how it was produced: the number of
/// markers and every byte outside the outermost editable ranges must be what
/// they were. ShinkaEvolve has no such check. Its diff applier removes marker
/// lines from REPLACE text but lets a marker that shares a line with code
/// through, and a marker written into a region moves where the region ends.
/// Two of them, in different regions, pair up across the read-only code in
/// between and make it editable.
pub(crate) fn boundary_violation(original: &str, updated: &str) -> Option<String> {
    let before = (start_markers(original).len(), end_markers(original).len());
    let after = (start_markers(updated).len(), end_markers(updated).len());
    let consequence = if before != after {
        format!(
            "leave {} {START_MARKER} and {} {END_MARKER} markers in a file that has {} and {}",
            after.0, after.1, before.0, before.1
        )
    } else if immutable_parts(original) != immutable_parts(updated) {
        "change code outside the editable regions".to_string()
    } else {
        return None;
    };
    Some(format!(
        "Patch changes the EVOLVE-BLOCK markers or read-only code

A patch may only change the code between an {START_MARKER} marker and the
{END_MARKER} marker that follows it. Applying this one would {consequence}.

Suggestions:
1. Do not write {START_MARKER} or {END_MARKER} in new code; the markers already in the file stay where they are
2. Keep every line outside the editable regions exactly as it is
3. Put new code between the existing markers"
    ))
}

/// True if `span` lies wholly inside one of `ranges`.
pub(crate) fn inside(span: (usize, usize), ranges: &[(usize, usize)]) -> bool {
    ranges
        .iter()
        .any(|(start, end)| span.0 >= *start && span.1 <= *end)
}

/// Strip trailing whitespace from the lines inside mutable ranges.
///
/// SEARCH text reaches the matcher with its trailing whitespace stripped, so
/// editable lines get the same treatment. Read-only text is returned byte
/// for byte.
pub(crate) fn normalize_mutable_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    for (start, end) in outermost_mutable_ranges(text) {
        out.push_str(&text[cursor..start]);
        let mut lines: Vec<&str> = text[start..end].split('\n').collect();
        // The last piece is the indentation of the END marker's own line.
        let last = lines.pop().unwrap_or("");
        for line in lines {
            out.push_str(rstrip(line));
            out.push('\n');
        }
        out.push_str(last);
        cursor = end;
    }
    out.push_str(&text[cursor..]);
    out
}

/// Remove lines that hold only a marker. A model that repeats the markers
/// around its edit means the code between them.
pub(crate) fn clean_evolve_markers(text: &str) -> String {
    let mut cleaned = text.to_string();
    for pattern in MARKER_LINES.iter() {
        if pattern.is_match(&cleaned) {
            cleaned = pattern.replace_all(&cleaned, "").into_owned();
        }
    }
    cleaned
}

/// How a comment starts in the file's language, read off its own markers.
pub(crate) fn comment_openers(text: &str) -> Vec<String> {
    let mut openers: Vec<String> = Vec::new();
    for captures in MARKER_COMMENT_OPENER.captures_iter(text) {
        let opener = &captures[1];
        if !openers.iter().any(|known| known == opener) {
            openers.push(opener.to_string());
        }
    }
    openers
}

fn is_strict_marker_line(line: &str, open: &str, marker: &str, close: &str) -> bool {
    strip(line)
        .strip_prefix(open)
        .map(|rest| rest.trim_start_matches(is_space))
        .and_then(|rest| rest.strip_prefix(marker))
        .map(|rest| rest.trim_start_matches(is_space))
        .is_some_and(|rest| rest == close)
}

/// Check a candidate's markers. `None` means they are well formed; otherwise
/// the text explains what is wrong, for the model to act on.
///
/// Both markers must be present, or the candidate could not be evolved
/// again. In languages whose comments span lines (Markdown, Wolfram) every
/// marker line must also open and close its comment on that line: a marker
/// left unclosed hides the candidate's body inside one comment. Only marker
/// lines are inspected. An unknown language is not checked.
pub fn validate_evolve_markers(text: &str, language_name: &str) -> Option<String> {
    let spec = language(language_name)?;
    let lines = split_lines(text);
    let lines_with = |marker: &str| -> Vec<(usize, &str)> {
        lines
            .iter()
            .enumerate()
            .filter(|(_, line)| line.contains(marker))
            .map(|(i, line)| (i, *line))
            .collect()
    };
    let start_lines = lines_with(START_MARKER);
    let end_lines = lines_with(END_MARKER);

    let mut errors = Vec::new();
    if start_lines.is_empty() {
        errors.push(format!("No {START_MARKER} marker found in candidate."));
    }
    if end_lines.is_empty() {
        errors.push(format!("No {END_MARKER} marker found in candidate."));
    }
    if spec.has_block_comments() {
        for (marker, marker_lines) in [(START_MARKER, &start_lines), (END_MARKER, &end_lines)] {
            for (index, line) in marker_lines {
                if !is_strict_marker_line(line, spec.comment_prefix, marker, spec.comment_suffix) {
                    errors.push(format!(
                        "Line {}: malformed {marker} marker.\n  Got:      {}\n  Expected: {} (the marker must \
                         occupy its own line and be fully wrapped in {} comment delimiters; nothing else on the \
                         line).",
                        index + 1,
                        quoted(rstrip(line)),
                        quoted(&spec.marker_line(marker)),
                        spec.name,
                    ));
                }
            }
        }
    }
    if errors.is_empty() {
        return None;
    }
    let header = format!(
        "EVOLVE-BLOCK marker validation failed for language '{}':",
        spec.name
    );
    Some(
        std::iter::once(header)
            .chain(errors)
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

/// Make `code` one editable region, with markers in the language's comment
/// style. An unknown language gets `#` comments.
pub fn wrap_in_evolve_block(code: &str, language_name: &str) -> String {
    let (start, end) = match language(language_name) {
        Some(spec) => (spec.marker_line(START_MARKER), spec.marker_line(END_MARKER)),
        None => (format!("# {START_MARKER}"), format!("# {END_MARKER}")),
    };
    format!("{start}\n{code}\n{end}\n")
}

/// The editable text of a program with each read-only stretch replaced by a
/// one-line placeholder.
pub fn redact_immutable(text: &str) -> String {
    let mut out = String::new();
    for (start, end) in mutable_ranges(text) {
        out.push_str("<… non-evolvable code omitted …>");
        out.push_str(&text[start..end]);
    }
    out.push_str("<… non-evolvable tail omitted …>");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const WELL_FORMED: &[(&str, &str)] = &[
        ("python", "# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END"),
        (
            "julia",
            "# EVOLVE-BLOCK-START\nfunction f(x) end\n# EVOLVE-BLOCK-END",
        ),
        ("cpp", "// EVOLVE-BLOCK-START\nint x;\n// EVOLVE-BLOCK-END"),
        (
            "cuda",
            "// EVOLVE-BLOCK-START\n__device__ int f();\n// EVOLVE-BLOCK-END",
        ),
        (
            "rust",
            "// EVOLVE-BLOCK-START\nfn f() {}\n// EVOLVE-BLOCK-END",
        ),
        (
            "swift",
            "// EVOLVE-BLOCK-START\nfunc f() {}\n// EVOLVE-BLOCK-END",
        ),
        (
            "json",
            "// EVOLVE-BLOCK-START\n{\"a\": 1}\n// EVOLVE-BLOCK-END",
        ),
        (
            "json5",
            "// EVOLVE-BLOCK-START\n{a: 1}\n// EVOLVE-BLOCK-END",
        ),
        (
            "go",
            "// EVOLVE-BLOCK-START\nfunc f() {}\n// EVOLVE-BLOCK-END",
        ),
        (
            "fortran",
            "! EVOLVE-BLOCK-START\ninteger :: x\n! EVOLVE-BLOCK-END",
        ),
        (
            "markdown",
            "<!-- EVOLVE-BLOCK-START -->\nhello\n<!-- EVOLVE-BLOCK-END -->",
        ),
        (
            "wolfram",
            "(* EVOLVE-BLOCK-START *)\nf[n_] := n\n(* EVOLVE-BLOCK-END *)",
        ),
    ];

    fn well_formed(name: &str) -> &'static str {
        WELL_FORMED
            .iter()
            .find(|(language, _)| *language == name)
            .unwrap()
            .1
    }

    #[test]
    fn well_formed_passes() {
        for (language, code) in WELL_FORMED {
            assert_eq!(validate_evolve_markers(code, language), None, "{language}");
        }
    }

    #[test]
    fn aliases_normalise_to_canonical_names() {
        for alias in ["py", "python3"] {
            assert_eq!(validate_evolve_markers(well_formed("python"), alias), None);
        }
        for alias in ["wl", "wls", "mathematica", "Mathematica"] {
            assert_eq!(validate_evolve_markers(well_formed("wolfram"), alias), None);
        }
        for alias in ["c++", "cxx", "cc"] {
            assert_eq!(validate_evolve_markers(well_formed("cpp"), alias), None);
        }
        assert_eq!(validate_evolve_markers(well_formed("go"), "golang"), None);
    }

    #[test]
    fn missing_start_fails_and_missing_end_fails() {
        for (language, code) in WELL_FORMED {
            let error =
                validate_evolve_markers(&code.replace(START_MARKER, "REMOVED"), language).unwrap();
            assert!(error.contains(START_MARKER), "{language}: {error}");
            let error =
                validate_evolve_markers(&code.replace(END_MARKER, "REMOVED"), language).unwrap();
            assert!(error.contains(END_MARKER), "{language}: {error}");
        }
    }

    #[test]
    fn wolfram_markers_inside_single_comment_block() {
        let code = "(* EVOLVE-BLOCK-START\nsolve[] := With[{r = Range[n]}, EulerPhi[r] . Quotient[n, r]^2]\n EVOLVE-BLOCK-END *)\n";
        let error = validate_evolve_markers(code, "wolfram").unwrap();
        assert!(error.contains(START_MARKER) && error.contains(END_MARKER));
        assert!(error.starts_with("EVOLVE-BLOCK marker validation failed for language 'wolfram':"));
    }

    #[test]
    fn wolfram_missing_delimiters_and_extra_text() {
        let missing_close = "(* EVOLVE-BLOCK-START\nf[n_] := n\n(* EVOLVE-BLOCK-END *)\n";
        assert!(validate_evolve_markers(missing_close, "wolfram")
            .unwrap()
            .to_lowercase()
            .contains("malformed"));
        let missing_open = "(* EVOLVE-BLOCK-START *)\nf[n_] := n\nEVOLVE-BLOCK-END *)\n";
        assert!(validate_evolve_markers(missing_open, "wolfram")
            .unwrap()
            .to_lowercase()
            .contains("malformed"));
        let extra_text = "(* EVOLVE-BLOCK-START *) f[n_] := n;\n(* EVOLVE-BLOCK-END *)\n";
        assert!(validate_evolve_markers(extra_text, "wolfram").is_some());
    }

    #[test]
    fn wolfram_inner_comments_and_string_delimiters_are_not_inspected() {
        let comments = "(* EVOLVE-BLOCK-START *)\n(* this is a normal Wolfram comment *)\nf[n_] := n\n(* another (* nested *) comment *)\n(* EVOLVE-BLOCK-END *)\n";
        assert_eq!(validate_evolve_markers(comments, "wolfram"), None);
        let string = "(* EVOLVE-BLOCK-START *)\nf[] := \"a literal containing (* which is not a comment\";\n(* EVOLVE-BLOCK-END *)\n";
        assert_eq!(validate_evolve_markers(string, "wolfram"), None);
    }

    #[test]
    fn trailing_markers_are_ok_in_line_comment_languages() {
        let python = "# EVOLVE-BLOCK-START\ndef f():\n    x = 1\n    return x\n    y = 2# EVOLVE-BLOCK-END\n";
        assert_eq!(validate_evolve_markers(python, "python"), None);
        assert_eq!(
            validate_evolve_markers(
                "// EVOLVE-BLOCK-START\nint x = 1;// EVOLVE-BLOCK-END\n",
                "cpp"
            ),
            None
        );
        assert_eq!(
            validate_evolve_markers(
                "// EVOLVE-BLOCK-START\nvar x = 1 // EVOLVE-BLOCK-END\n",
                "go"
            ),
            None
        );
        assert_eq!(
            validate_evolve_markers(
                "! EVOLVE-BLOCK-START\ninteger :: x  ! EVOLVE-BLOCK-END\n",
                "fortran"
            ),
            None
        );
    }

    #[test]
    fn unknown_language_defers() {
        assert_eq!(
            validate_evolve_markers("EVOLVE-BLOCK-START\nx\nEVOLVE-BLOCK-END", "klingon"),
            None
        );
    }

    #[test]
    fn clean_evolve_markers_strips_html_comments() {
        let text = "<!-- EVOLVE-BLOCK-START -->\n<<<<<<< SEARCH\nold text\n=======\nnew text\n>>>>>>> REPLACE\n<!-- EVOLVE-BLOCK-END -->";
        let cleaned = clean_evolve_markers(text);
        assert!(!cleaned.contains(START_MARKER) && !cleaned.contains(END_MARKER));
        assert!(cleaned.contains("new text"));
    }

    #[test]
    fn clean_evolve_markers_keeps_markers_that_share_a_line_with_code() {
        let text = "x = 1  # EVOLVE-BLOCK-END\n    // EVOLVE-BLOCK-START   \n(* EVOLVE-BLOCK-END *)\nEVOLVE-BLOCK-START";
        // `\s*` in the line patterns also takes the blank lines next to a marker.
        assert_eq!(clean_evolve_markers(text), "x = 1  # EVOLVE-BLOCK-END\n");
    }

    #[test]
    fn ranges_start_after_the_start_marker_and_end_at_the_end_marker() {
        let text = "a\n# EVOLVE-BLOCK-START\nb\n    # EVOLVE-BLOCK-END\nc\n";
        let ranges = mutable_ranges(text);
        assert_eq!(ranges.len(), 1);
        assert_eq!(&text[ranges[0].0..ranges[0].1], "\nb\n    ");
        assert_eq!(
            immutable_parts(text),
            ["a\n# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END\nc\n"]
        );
    }

    #[test]
    fn nested_and_unmatched_markers() {
        let text = "// EVOLVE-BLOCK-START\na\n// EVOLVE-BLOCK-START\nb\n// EVOLVE-BLOCK-END\nc\n// EVOLVE-BLOCK-END\n";
        let ranges = mutable_ranges(text);
        assert_eq!(ranges.len(), 2);
        assert_eq!(&text[ranges[0].0..ranges[0].1], "\nb\n");
        assert!(ranges[1].0 < ranges[0].0 && ranges[1].1 > ranges[0].1);
        assert_eq!(outermost_mutable_ranges(text), vec![ranges[1]]);

        assert!(mutable_ranges("x\n# EVOLVE-BLOCK-END\ny\n# EVOLVE-BLOCK-START\nz").is_empty());
        assert_eq!(immutable_parts("no markers"), ["no markers"]);
    }

    #[test]
    fn markers_in_block_comments_include_their_delimiters() {
        let text = "<!-- EVOLVE-BLOCK-START --><p>hi</p>\n<!-- EVOLVE-BLOCK-END -->\n";
        let ranges = mutable_ranges(text);
        assert_eq!(&text[ranges[0].0..ranges[0].1], "<p>hi</p>\n");
        assert_eq!(comment_openers(text), ["<!--"]);
        assert!(comment_openers("EVOLVE-BLOCK-START\nEVOLVE-BLOCK-END").is_empty());
    }

    #[test]
    fn normalisation_touches_only_editable_lines() {
        let text =
            "keep  \n# EVOLVE-BLOCK-START\nx = 1   \ny = 2\t\n    # EVOLVE-BLOCK-END  \ntail  \n";
        assert_eq!(
            normalize_mutable_whitespace(text),
            "keep  \n# EVOLVE-BLOCK-START\nx = 1\ny = 2\n    # EVOLVE-BLOCK-END  \ntail  \n"
        );
    }

    #[test]
    fn boundary_violations_are_named() {
        let original = "a\n# EVOLVE-BLOCK-START\nx\n# EVOLVE-BLOCK-END\nb\n";
        assert_eq!(
            boundary_violation(original, &original.replace("x", "y\nz")),
            None
        );
        let extra =
            boundary_violation(original, &original.replace("x", "x  # EVOLVE-BLOCK-END")).unwrap();
        assert!(extra.contains(
            "leave 1 EVOLVE-BLOCK-START and 2 EVOLVE-BLOCK-END markers in a file that has 1 and 1"
        ));
        let outside = boundary_violation(original, &original.replace("b\n", "c\n")).unwrap();
        assert!(outside.contains("change code outside the editable regions"));
        assert!(outside.contains("Suggestions:"));
    }

    #[test]
    fn wrapping_and_redaction() {
        assert_eq!(
            wrap_in_evolve_block("x = 1", "python"),
            "# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n"
        );
        assert_eq!(
            wrap_in_evolve_block("f[x_] := x", "wl"),
            "(* EVOLVE-BLOCK-START *)\nf[x_] := x\n(* EVOLVE-BLOCK-END *)\n"
        );
        assert_eq!(
            wrap_in_evolve_block("x", "klingon"),
            "# EVOLVE-BLOCK-START\nx\n# EVOLVE-BLOCK-END\n"
        );
        let wrapped = wrap_in_evolve_block("int x;", "c");
        assert_eq!(validate_evolve_markers(&wrapped, "c"), None);
        assert_eq!(
            redact_immutable("head\n# EVOLVE-BLOCK-START\nx\n# EVOLVE-BLOCK-END\ntail"),
            "<… non-evolvable code omitted …>\nx\n<… non-evolvable tail omitted …>"
        );
    }
}
