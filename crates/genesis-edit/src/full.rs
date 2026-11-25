//! Applying a full rewrite: the model returns the whole program in a code
//! fence, and only the text of its editable regions is taken from it.

use crate::languages::{fence_tags, language};
use crate::markers::{
    boundary_violation, end_markers, has_start_marker, outermost_mutable_ranges, start_markers,
    validate_evolve_markers,
};
use crate::text::{is_blank, lstrip, rstrip};
use crate::unified::unified_diff;
use crate::{PatchError, PatchOutcome};

/// True for a character that continues a fence tag, so that the tag `c` does
/// not open a `c++` fence and `go` does not open a `golang` one.
fn continues_tag(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '+' | '#' | '-' | '_' | '.')
}

/// Offset just past the first fence opened with exactly `tag`.
fn fence_body_start(response: &str, tag: &str) -> Option<usize> {
    let opener = format!("```{tag}");
    let lowered = response.to_ascii_lowercase();
    let mut from = 0;
    while let Some(offset) = lowered[from..].find(&opener) {
        let body_start = from + offset + opener.len();
        if !response[body_start..]
            .chars()
            .next()
            .is_some_and(continues_tag)
        {
            return Some(body_start);
        }
        from += offset + 3;
    }
    None
}

/// Trim the text of a fence down to its code: no blank lines before or
/// after it, but the indentation of its first line intact.
///
/// ShinkaEvolve strips whitespace on both sides, which also takes the first
/// line's indentation. For a region inside a function that puts the first
/// statement of the rewrite at column zero.
fn trim_code(text: &str) -> &str {
    let text = rstrip(text);
    let code_start = text.len() - lstrip(text).len();
    match text[..code_start].rfind('\n') {
        Some(newline) => &text[newline + 1..],
        // Code that starts on the fence line itself has no indentation to keep.
        None => &text[code_start..],
    }
}

/// The code inside the first fence tagged for `language_name`, trimmed.
///
/// Tags are tried in the order of [`fence_tags`]; for each, the first such
/// fence in the response opens the code. It closes at the first run of three
/// backticks that leaves the code with as many START as END markers, and at
/// least one of each: a program may hold fences of its own (a Markdown
/// document, a prompt in a string), and those must not cut the rewrite
/// short. When no run does, the code closes at the first one, which is where
/// ShinkaEvolve always closes it.
pub(crate) fn extract_code(response: &str, language_name: &str) -> Option<String> {
    for tag in fence_tags(language_name)
        .iter()
        .filter(|tag| !tag.is_empty())
    {
        let Some(body_start) = fence_body_start(response, tag) else {
            continue;
        };
        let body = &response[body_start..];
        let (mut starts, mut ends) = (0, 0);
        let mut first_close = None;
        let mut segment_start = 0;
        while let Some(offset) = body[segment_start..].find("```") {
            let close = segment_start + offset;
            let segment = &body[segment_start..close];
            starts += start_markers(segment).len();
            ends += end_markers(segment).len();
            if starts > 0 && starts == ends {
                return Some(trim_code(&body[..close]).to_string());
            }
            first_close.get_or_insert(close);
            segment_start = close + 3;
        }
        if let Some(close) = first_close {
            return Some(trim_code(&body[..close]).to_string());
        }
    }
    None
}

/// The new content of the only region: `payload` on lines of its own, then
/// the indentation the END marker had.
///
/// The region runs up to the END marker itself, so the marker's indentation
/// is part of what is replaced. ShinkaEvolve does not put it back, which
/// moves an indented END marker to column zero.
fn region_content(payload: &str, old_content: &str) -> String {
    let end_marker_indent = match old_content.rfind('\n') {
        Some(newline) if is_blank(&old_content[newline + 1..]) => &old_content[newline + 1..],
        _ => "",
    };
    // A patch that kept the END marker carries that marker's indentation too.
    let payload = match payload.rfind('\n') {
        Some(newline) if is_blank(&payload[newline + 1..]) => &payload[..=newline],
        _ => payload,
    };
    let mut content = String::with_capacity(payload.len() + end_marker_indent.len() + 2);
    if !payload.starts_with('\n') {
        content.push('\n');
    }
    content.push_str(payload);
    if !content.ends_with('\n') {
        content.push('\n');
    }
    content.push_str(end_marker_indent);
    content
}

/// The text of the patch meant for the single editable region, when the
/// patch carries no marker, one marker, or the file's read-only text around
/// the new code.
fn single_region_payload<'a>(
    code: &'a str,
    original: &str,
    (start, end): (usize, usize),
) -> &'a str {
    let marker_start = start_markers(original)
        .into_iter()
        .find(|(_, marker_end)| *marker_end == start)
        .map_or(start, |m| m.0);
    let marker_end = end_markers(original)
        .into_iter()
        .find(|(begin, _)| *begin == end)
        .map_or(end, |m| m.1);
    // The file's header ends where the START marker's line begins. Counting
    // the marker's own indentation as header, as ShinkaEvolve does, takes
    // that many characters off the first line of a rewrite.
    let marker_line = original[..marker_start]
        .rfind('\n')
        .map_or(0, |newline| newline + 1);
    let header_end = if is_blank(&original[marker_line..marker_start]) {
        marker_line
    } else {
        marker_start
    };
    let header = &original[..header_end];
    let footer = &original[marker_end..];
    // A model often drops the newline that ends the file.
    let footers = [footer, footer.trim_end_matches(['\r', '\n'])];
    let headers = [header, header.trim_end_matches(['\r', '\n'])];

    let patch_start = start_markers(code).first().copied();
    let patch_end = end_markers(code).first().copied();
    match (patch_start, patch_end) {
        (Some((_, after_marker)), None) => {
            let payload = &code[after_marker..];
            let footer = footers
                .into_iter()
                .find(|footer| !footer.is_empty() && payload.ends_with(footer));
            footer.map_or(payload, |footer| &payload[..payload.len() - footer.len()])
        }
        (None, Some((before_marker, _))) => {
            let payload = &code[..before_marker];
            let header = headers
                .into_iter()
                .find(|header| !header.is_empty() && payload.starts_with(header));
            header.map_or(payload, |header| &payload[header.len()..])
        }
        _ => {
            // A whole-file rewrite that left the markers out: keep only what
            // lies between the file's own header and footer.
            let footer = footers.into_iter().find(|footer| code.ends_with(footer));
            match footer {
                Some(footer) if code.starts_with(header) => {
                    let middle_end = code.len() - footer.len();
                    code.get(header.len()..middle_end).unwrap_or("")
                }
                _ => code,
            }
        }
    }
}

/// Apply a full rewrite found in a model's `response` to `original`.
///
/// The code is taken from the first fence tagged with the language. Text
/// outside the original's editable regions is always copied from the
/// original, whatever the rewrite says there:
///
/// * a rewrite with both markers supplies the content of each region in
///   order; regions it leaves out keep their content;
/// * a rewrite with no marker, or only one, replaces the single region, and
///   is refused when the original has several.
///
/// A rewrite that would add or remove a marker is refused.
pub fn apply_full_patch(
    response: &str,
    original: &str,
    language_name: &str,
) -> Result<PatchOutcome, PatchError> {
    let Some(code) = extract_code(response, language_name) else {
        return Err(PatchError(
            "Could not extract code from patch string".to_string(),
        ));
    };
    let regions = outermost_mutable_ranges(original);
    if regions.is_empty() {
        return Err(PatchError(
            "No EVOLVE-BLOCK regions found in original content".to_string(),
        ));
    }
    let patch_has_start = has_start_marker(&code);
    let patch_has_end = !end_markers(&code).is_empty();

    let updated = if patch_has_start && patch_has_end {
        if language(language_name).is_some_and(|spec| spec.has_block_comments()) {
            if let Some(problem) = validate_evolve_markers(&code, language_name) {
                return Err(PatchError(problem));
            }
        }
        let patch_regions = outermost_mutable_ranges(&code);
        let mut updated = String::with_capacity(original.len().max(code.len()));
        let mut cursor = 0;
        for (index, (start, end)) in regions.iter().enumerate() {
            updated.push_str(&original[cursor..*start]);
            match patch_regions.get(index) {
                Some((patch_start, patch_end)) => updated.push_str(&code[*patch_start..*patch_end]),
                None => updated.push_str(&original[*start..*end]),
            }
            cursor = *end;
        }
        updated.push_str(&original[cursor..]);
        updated
    } else {
        if regions.len() != 1 {
            let message = if patch_has_start || patch_has_end {
                format!(
                    "Patch contains only one EVOLVE-BLOCK marker, but the original has {} editable regions; \
                     cannot determine target",
                    regions.len()
                )
            } else {
                "Multiple EVOLVE-BLOCK regions found but patch doesn't specify which to replace"
                    .to_string()
            };
            return Err(PatchError(message));
        }
        let (start, end) = regions[0];
        let content = region_content(
            single_region_payload(&code, original, (start, end)),
            &original[start..end],
        );
        format!("{}{content}{}", &original[..start], &original[end..])
    };

    if let Some(problem) = validate_evolve_markers(&updated, language_name)
        .or_else(|| boundary_violation(original, &updated))
    {
        return Err(PatchError(problem));
    }
    let diff = unified_diff(original, &updated, language_name);
    Ok(PatchOutcome {
        code: updated,
        num_applied: 1,
        diff,
        patch_text: code,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_code_matches_fence_tags_literally() {
        for fence in ["c++", "cpp", "cxx", "cc"] {
            let content = format!(
                "Here is the program:\n```{fence}\nint solve() {{ return 2; }}\n```\nDone."
            );
            assert_eq!(
                extract_code(&content, fence).as_deref(),
                Some("int solve() { return 2; }")
            );
        }
    }

    #[test]
    fn a_tag_does_not_open_a_fence_with_a_longer_tag() {
        // `go` must not take the `go` of ```golang and leave `lang` in the code.
        assert_eq!(
            extract_code("```golang\nfunc f() {}\n```", "go").as_deref(),
            Some("func f() {}")
        );
        assert_eq!(
            extract_code("```python3\nx = 1\n```", "py").as_deref(),
            Some("x = 1")
        );
        assert_eq!(extract_code("```json5\n{a: 1}\n```", "json"), None);
        assert_eq!(
            extract_code("```cpp\nint x;\n```\n```c\nint y;\n```", "c").as_deref(),
            Some("int y;")
        );
        assert_eq!(
            extract_code("```Python\nx = 1\n```", "python").as_deref(),
            Some("x = 1")
        );
    }

    #[test]
    fn the_code_closes_where_its_markers_balance() {
        let inner = "```markdown\n<!-- EVOLVE-BLOCK-START -->\n```bash\nls\n```\n<!-- EVOLVE-BLOCK-END -->\n```\nMore.";
        assert_eq!(
            extract_code(inner, "markdown").as_deref(),
            Some("<!-- EVOLVE-BLOCK-START -->\n```bash\nls\n```\n<!-- EVOLVE-BLOCK-END -->")
        );
        // Without balanced markers the first closing fence still wins.
        assert_eq!(
            extract_code(
                "```python\nx = 1\n```\nand\n```python\ny = 2\n```",
                "python"
            )
            .as_deref(),
            Some("x = 1")
        );
        assert_eq!(
            extract_code(
                "```python\n# EVOLVE-BLOCK-START\nx = 1\n```\ntext\n```",
                "python"
            )
            .as_deref(),
            Some("# EVOLVE-BLOCK-START\nx = 1")
        );
    }

    #[test]
    fn extraction_needs_a_tagged_and_closed_fence() {
        assert_eq!(
            extract_code("def new_function(): return 'no fences'", "python"),
            None
        );
        assert_eq!(extract_code("```\nx = 1\n```", "python"), None);
        assert_eq!(extract_code("```python\nx = 1\n", "python"), None);
        assert_eq!(
            extract_code("```python x = 1```", "python").as_deref(),
            Some("x = 1")
        );
        assert_eq!(extract_code("```rust\nfn f() {}\n```", "klingon"), None);
        assert_eq!(
            extract_code("```klingon\nQapla'\n```", "klingon").as_deref(),
            Some("Qapla'")
        );
    }
}
