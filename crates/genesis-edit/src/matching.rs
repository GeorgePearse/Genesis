//! Finding the lines of the original that a SEARCH block names.
//!
//! A block matches a run of whole lines, never part of one, so `x = 1`
//! cannot hit `idx = 10`. Trailing whitespace is ignored. Four readings of
//! the block are tried, most literal first, and only the first reading that
//! matches anywhere is used: a block that names read-only code as written
//! must not fall back to an editable look-alike at another indentation.

use crate::text::{indent_width, is_blank, leading_whitespace, lstrip, rstrip, strip};

/// How literally a block matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Looseness {
    /// As written.
    Exact,
    /// Without the trailing comments of its lines.
    WithoutComments,
    /// After shifting the whole block to another indentation.
    Reindented,
    /// After restoring leading whitespace its first line lost.
    FirstLineIndent,
}

/// Lines of the original that a SEARCH block matches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LineMatch {
    pub pos: usize,
    pub end: usize,
    pub looseness: Looseness,
    /// For a re-indented match: the signed shift in characters and the
    /// leading whitespace of the line the block was matched at.
    pub indent: Option<(isize, String)>,
}

/// Shift every non-blank line by a signed indentation offset.
///
/// Relative indentation between lines is kept, including lines that dedent
/// below the first; only the result is clamped at column zero. Leading
/// whitespace is drawn from `indent_chars`, so tabs stay tabs, and padded
/// with spaces beyond it. Blank lines come out empty.
pub(crate) fn shift_indentation(text: &str, offset: isize, indent_chars: &str) -> String {
    let available = indent_chars.chars().count();
    let shifted: Vec<String> = text
        .split('\n')
        .map(|line| {
            if is_blank(line) {
                return String::new();
            }
            let indent = (indent_width(line) as isize + offset).max(0) as usize;
            let mut out: String = indent_chars.chars().take(indent).collect();
            out.extend(std::iter::repeat_n(' ', indent.saturating_sub(available)));
            out.push_str(strip(line));
            out
        })
        .collect();
    shifted.join("\n")
}

/// Shift REPLACE text by the offset its SEARCH block was matched at, which
/// keeps any intended difference between their first-line indentation.
pub(crate) fn apply_indentation_to_replace(
    replace: &str,
    offset: isize,
    indent_chars: &str,
) -> String {
    if is_blank(replace) {
        return replace.to_string();
    }
    shift_indentation(replace, offset, indent_chars)
}

/// Split into lines with their start offsets. A line ends at a newline and
/// at every offset in `breaks`, which makes code sharing a line with a
/// marker a line of its own.
fn split_at_breaks<'a>(text: &'a str, breaks: &[usize]) -> (Vec<&'a str>, Vec<usize>) {
    let mut breaks = breaks.to_vec();
    breaks.sort_unstable();
    breaks.dedup();
    let (mut lines, mut starts) = (Vec::new(), Vec::new());
    let mut line_start = 0;
    for line in text.split('\n') {
        let line_end = line_start + line.len();
        let mut piece_start = line_start;
        for cut in breaks
            .iter()
            .copied()
            .filter(|cut| line_start < *cut && *cut < line_end)
        {
            lines.push(&text[piece_start..cut]);
            starts.push(piece_start);
            piece_start = cut;
        }
        lines.push(&text[piece_start..line_end]);
        starts.push(piece_start);
        line_start = line_end + 1;
    }
    (lines, starts)
}

/// True if `line` is `code` followed by a comment and nothing else.
fn is_code_then_comment(line: &str, code: &str, comment_openers: &[String]) -> bool {
    if is_blank(code) || !line.starts_with(code) {
        return false;
    }
    let rest = lstrip(&line[code.len()..]);
    comment_openers
        .iter()
        .any(|opener| rest.starts_with(opener.as_str()))
}

/// Every place the block matches under the most literal reading that matches
/// at all, in file order.
///
/// `comment_openers` start a comment in the file's language; without them no
/// line is matched without its comment.
pub(crate) fn line_matches(
    search_text: &str,
    original: &str,
    breaks: &[usize],
    comment_openers: &[String],
) -> Vec<LineMatch> {
    if is_blank(search_text) {
        return Vec::new();
    }
    let search_lines: Vec<&str> = search_text.split('\n').map(rstrip).collect();
    let (lines, starts) = split_at_breaks(original, breaks);
    let stripped: Vec<&str> = lines.iter().map(|line| rstrip(line)).collect();
    let count = search_lines.len();
    if lines.len() < count {
        return Vec::new();
    }
    let candidates = 0..=lines.len() - count;
    // Whitespace after the last line stays: it may separate code from a
    // marker that follows on the same line.
    let span = |index: usize| {
        (
            starts[index],
            starts[index + count - 1] + stripped[index + count - 1].len(),
        )
    };
    let found = |index: usize, looseness: Looseness| {
        let (pos, end) = span(index);
        LineMatch {
            pos,
            end,
            looseness,
            indent: None,
        }
    };

    let exact: Vec<LineMatch> = candidates
        .clone()
        .filter(|index| stripped[*index..index + count] == search_lines[..])
        .map(|index| found(index, Looseness::Exact))
        .collect();
    if !exact.is_empty() {
        return exact;
    }

    // The code is the same, so this comes before any guess about
    // indentation. The comment after the last line is not matched and stays.
    let without_comments: Vec<LineMatch> = candidates
        .clone()
        .filter(|index| {
            stripped[*index..index + count]
                .iter()
                .zip(&search_lines)
                .all(|(line, code)| {
                    line == code || is_code_then_comment(line, code, comment_openers)
                })
        })
        .map(|index| {
            let mut matched = found(index, Looseness::WithoutComments);
            matched.end = starts[index + count - 1] + search_lines[count - 1].len();
            matched
        })
        .collect();
    if !without_comments.is_empty() {
        return without_comments;
    }

    let first_search_line = strip(search_lines[0]);
    if first_search_line.is_empty() {
        return Vec::new();
    }

    // A block copied out of a nested scope carries its own base indentation.
    // Shifting the whole block by the offset to each candidate line keeps
    // lines that dedent below the first dedenting, so they cannot flatten
    // onto a same-indent block in another scope.
    let base_indent = indent_width(search_lines[0]) as isize;
    let mut reindented = Vec::new();
    for index in candidates.clone() {
        let line = lines[index];
        if strip(line) != first_search_line {
            continue;
        }
        let offset = indent_width(line) as isize - base_indent;
        let indent_chars = leading_whitespace(line);
        let shifted = shift_indentation(search_text, offset, indent_chars);
        let shifted_lines: Vec<&str> = shifted.split('\n').collect();
        if shifted_lines == search_lines {
            continue;
        }
        if stripped[index..index + count] == shifted_lines[..] {
            let mut matched = found(index, Looseness::Reindented);
            matched.indent = Some((offset, indent_chars.to_string()));
            reindented.push(matched);
        }
    }
    if !reindented.is_empty() {
        return reindented;
    }

    // The block still names whole lines. It matches from where its text
    // starts inside the indentation, so the replacement keeps the
    // indentation the block left out.
    candidates
        .filter_map(|index| {
            let line = stripped[index];
            let first = search_lines[0];
            let missing = line
                .len()
                .checked_sub(first.len())
                .filter(|missing| *missing > 0)?;
            let lost_only_indentation = line.ends_with(first)
                && is_blank(&line[..missing])
                && stripped[index + 1..index + count] == search_lines[1..];
            lost_only_indentation.then(|| {
                let mut matched = found(index, Looseness::FirstLineIndent);
                matched.pos += missing;
                matched
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first match under any reading, as `(matched text, position)`.
    fn find_indented_match<'a>(search: &str, original: &'a str) -> Option<(&'a str, usize)> {
        line_matches(search, original, &[], &[])
            .first()
            .map(|m| (&original[m.pos..m.end], m.pos))
    }

    #[test]
    fn find_indented_match_exact_match() {
        let original = "def function():\n    x = 1\n    y = 2\n    return x + y";
        let (matched, pos) = find_indented_match("    x = 1", original).unwrap();
        assert_eq!(matched, "    x = 1");
        assert_eq!(&original[pos..pos + matched.len()], matched);
    }

    #[test]
    fn find_indented_match_needs_indentation() {
        let original = "def function():\n    x = 1\n    y = 2\n    return x + y";
        let (matched, _) = find_indented_match("x = 1\ny = 2", original).unwrap();
        assert_eq!(matched, "    x = 1\n    y = 2");
    }

    #[test]
    fn find_indented_match_multiline_with_relative_indentation() {
        let original = "def function():\n    if True:\n        x = 1\n        if nested:\n            y = 2\n    return x + y";
        let search = "if True:\n    x = 1\n    if nested:\n        y = 2";
        let (matched, _) = find_indented_match(search, original).unwrap();
        assert_eq!(
            matched,
            "    if True:\n        x = 1\n        if nested:\n            y = 2"
        );
    }

    #[test]
    fn find_indented_match_not_found_and_empty_search() {
        assert_eq!(
            find_indented_match("z = 3", "def function():\n    x = 1\n    return x"),
            None
        );
        assert_eq!(find_indented_match("", "def function():\n    pass"), None);
    }

    #[test]
    fn find_indented_match_search_block_carries_its_own_indent() {
        let original = "class Outer:\n    class Inner:\n        def score(self, x):\n            total = 0\n            return total";
        let search = "    def score(self, x):\n        total = 0\n        return total";
        let (matched, _) = find_indented_match(search, original).unwrap();
        assert_eq!(
            matched,
            "        def score(self, x):\n            total = 0\n            return total"
        );
    }

    #[test]
    fn find_indented_match_search_deeper_than_target() {
        let original = "class A:\n    def foo(self):\n        x = 1\n        return x";
        let search = "        def foo(self):\n            x = 1\n            return x";
        let (matched, _) = find_indented_match(search, original).unwrap();
        assert_eq!(
            matched,
            "    def foo(self):\n        x = 1\n        return x"
        );
    }

    #[test]
    fn find_indented_match_dedent_below_first_line_does_not_flatten() {
        let original = "def f(a):\n    if a:\n        x()\n        y()\n    return 1";
        assert_eq!(
            find_indented_match("            x()\n        y()", original),
            None
        );
    }

    #[test]
    fn apply_indentation_to_replace_shifts_and_keeps_blank_lines_empty() {
        let replace = "x = 10\nif x > 5:\n    print(\"big\")\nelse:\n    print(\"small\")";
        assert_eq!(
            apply_indentation_to_replace(replace, 4, ""),
            "    x = 10\n    if x > 5:\n        print(\"big\")\n    else:\n        print(\"small\")"
        );
        assert_eq!(
            apply_indentation_to_replace("x = 1\n\ny = 2", 4, ""),
            "    x = 1\n\n    y = 2"
        );
        assert_eq!(apply_indentation_to_replace("  \n", 4, ""), "  \n");
    }

    #[test]
    fn apply_indentation_to_replace_rebases_existing_indent() {
        let replace = "    def foo(self):\n        x = 1\n        return x";
        assert_eq!(
            apply_indentation_to_replace(replace, 4, ""),
            "        def foo(self):\n            x = 1\n            return x"
        );
        assert_eq!(shift_indentation("        a\n    b", -8, ""), "a\nb");
        assert_eq!(shift_indentation("a\n    b", 1, "\t"), "\ta\n\t    b");
    }

    #[test]
    fn breaks_end_a_line_and_tiers_are_exclusive() {
        let original = "x = 1  # EVOLVE-BLOCK-END\n    x = 1\n";
        let matches = line_matches("x = 1", original, &[7], &[]);
        assert_eq!(matches.len(), 1);
        assert_eq!(
            (matches[0].pos, matches[0].end, matches[0].looseness),
            (0, 5, Looseness::Exact)
        );

        let commented = "lr = 0.01  # tuned\n    lr = 0.01\n";
        let matches = line_matches("lr = 0.01", commented, &[], &["#".to_string()]);
        assert_eq!(matches.len(), 1);
        assert_eq!(
            (
                matches[0].looseness,
                &commented[matches[0].pos..matches[0].end]
            ),
            (Looseness::WithoutComments, "lr = 0.01")
        );
        assert_eq!(
            line_matches("lr = 0.01", commented, &[], &[])[0].looseness,
            Looseness::Reindented
        );
    }

    #[test]
    fn a_first_line_that_lost_its_indentation_matches_inside_it() {
        let original = "    for i in range(n):\n        total += i\n";
        let matches = line_matches("for i in range(n):\n        total += i", original, &[], &[]);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].looseness, Looseness::FirstLineIndent);
        assert_eq!(
            &original[matches[0].pos..matches[0].end],
            "for i in range(n):\n        total += i"
        );
        assert!(line_matches(
            "idx = 1\n    done = True",
            "    total = idx = 1\n    done = True\n",
            &[],
            &[]
        )
        .is_empty());
    }
}
