//! Pulling structured pieces out of a model's free-text reply: tagged
//! sections, fenced code and JSON documents.

use serde_json::Value;

/// The trimmed text between `<TAG>` and the next `</TAG>`.
pub fn extract_tag(content: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = content.find(&open)? + open.len();
    let end = content[start..].find(&close)? + start;
    Some(content[start..end].trim().to_string())
}

pub(crate) struct Fence<'a> {
    /// Byte offset of the opening backticks.
    pub start: usize,
    pub info: &'a str,
    pub body: &'a str,
}

/// Every closed fenced block in `content`, in order. A fence opens with three
/// backticks and an optional info string, and closes at the next line that
/// starts with three backticks, or failing that at the next three backticks
/// anywhere. An unclosed fence, which is what a truncated reply ends with, is
/// not a block.
pub(crate) fn fences(content: &str) -> Vec<Fence<'_>> {
    let mut blocks = Vec::new();
    let mut cursor = 0;
    while let Some(found) = content[cursor..].find("```") {
        let start = cursor + found;
        let info_start = start + 3;
        let line_end = content[info_start..]
            .find('\n')
            .map_or(content.len(), |offset| info_start + offset);
        let info = &content[info_start..line_end];
        if info.contains("```") {
            // An inline span such as ```x```: not a block.
            cursor = info_start + info.find("```").unwrap_or(0) + 3;
            continue;
        }
        let body_start = (line_end + 1).min(content.len());
        let rest = &content[body_start..];
        let close = if rest.starts_with("```") {
            Some(0)
        } else {
            rest.find("\n```")
                .map(|offset| offset + 1)
                .or_else(|| rest.find("```"))
        };
        let Some(close) = close else { break };
        blocks.push(Fence {
            start,
            info: info.trim(),
            body: &rest[..close],
        });
        cursor = body_start + close + 3;
    }
    blocks
}

fn tidy(body: &str) -> String {
    let without_leading_blank_lines = body.trim_start_matches(['\n', '\r']);
    without_leading_blank_lines.trim_end().to_string()
}

/// The first fenced block whose info string is one of `languages`, trying the
/// languages in the order given; when none matches, the first fenced block of
/// any kind. Indentation inside the block is preserved.
pub fn extract_fenced(content: &str, languages: &[&str]) -> Option<String> {
    let blocks = fences(content);
    for language in languages {
        let hit = blocks.iter().find(|block| {
            let tag = block.info.split_whitespace().next().unwrap_or("");
            tag.eq_ignore_ascii_case(language)
        });
        if let Some(block) = hit {
            return Some(tidy(block.body));
        }
    }
    blocks.first().map(|block| tidy(block.body))
}

/// Drop commas that directly precede a closing bracket, outside strings.
fn strip_trailing_commas(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escaped = false;
    let chars: Vec<char> = text.chars().collect();
    for (index, &ch) in chars.iter().enumerate() {
        if in_string {
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
            out.push(ch);
            continue;
        }
        if ch == ',' {
            let next = chars[index + 1..].iter().find(|c| !c.is_whitespace());
            if matches!(next, Some('}') | Some(']')) {
                continue;
            }
        }
        out.push(ch);
    }
    out
}

fn parse_lenient(text: &str) -> Option<Value> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    serde_json::from_str(text)
        .ok()
        .or_else(|| serde_json::from_str(&strip_trailing_commas(text)).ok())
}

/// End (exclusive) of the bracketed span opening at byte `start`, honouring
/// strings and escapes.
fn balanced_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, &byte) in bytes[start..].iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(start + offset + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// The JSON document in a reply: the contents of a ```` ```json ```` fence if
/// there is one, otherwise the first balanced object or array in the text
/// that parses. Trailing commas are repaired; `NaN` and `Infinity` are not
/// JSON and make a candidate unparseable.
pub fn extract_json(content: &str) -> Option<Value> {
    let blocks = fences(content);
    let tagged = blocks.iter().filter(|block| {
        let tag = block.info.split_whitespace().next().unwrap_or("");
        tag.eq_ignore_ascii_case("json") || tag.eq_ignore_ascii_case("json5")
    });
    for block in tagged {
        if let Some(value) = parse_lenient(block.body) {
            return Some(value);
        }
    }
    let bytes = content.as_bytes();
    let mut cursor = 0;
    while cursor < bytes.len() {
        let Some(offset) = bytes[cursor..]
            .iter()
            .position(|b| *b == b'{' || *b == b'[')
        else {
            break;
        };
        let start = cursor + offset;
        if let Some(end) = balanced_end(bytes, start) {
            if let Some(value) = parse_lenient(&content[start..end]) {
                return Some(value);
            }
        }
        cursor = start + 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tags_are_trimmed_and_missing_tags_are_none() {
        let reply = "<NAME>\n reorder_loops \n</NAME>\n<DESCRIPTION>why</DESCRIPTION>";
        assert_eq!(extract_tag(reply, "NAME").as_deref(), Some("reorder_loops"));
        assert_eq!(extract_tag(reply, "DESCRIPTION").as_deref(), Some("why"));
        assert_eq!(extract_tag(reply, "CODE"), None);
        assert_eq!(extract_tag("<NAME>unterminated", "NAME"), None);
    }

    #[test]
    fn fenced_code_prefers_the_requested_language_and_keeps_indentation() {
        let reply =
            "intro\n```text\nnot code\n```\n```python\n    def f():\n        return 1\n```\ntail";
        assert_eq!(
            extract_fenced(reply, &["rust", "python"]).as_deref(),
            Some("    def f():\n        return 1")
        );
        assert_eq!(extract_fenced(reply, &["go"]).as_deref(), Some("not code"));
        assert_eq!(extract_fenced("no fences here", &["python"]), None);
    }

    #[test]
    fn a_language_tag_must_match_the_whole_info_word() {
        let reply = "```python\nprint(1)\n```";
        assert_eq!(extract_fenced(reply, &["py"]).as_deref(), Some("print(1)"));
        let two = "```pyth\nA\n```\n```py\nB\n```";
        assert_eq!(extract_fenced(two, &["py"]).as_deref(), Some("B"));
    }

    #[test]
    fn an_unclosed_fence_is_not_a_block() {
        assert_eq!(extract_fenced("```rust\nfn main() {", &["rust"]), None);
        assert_eq!(
            extract_fenced("```rust\nfn a() {}\n```\n```rust\nfn b() {", &["rust"]).as_deref(),
            Some("fn a() {}")
        );
    }

    #[test]
    fn json_comes_from_a_fence_or_the_first_parseable_span() {
        let fenced = "Here you go:\n```json\n{\"a\": [1, 2,], \"b\": \"x,}\",}\n```\nthanks";
        assert_eq!(extract_json(fenced), Some(json!({"a": [1, 2], "b": "x,}"})));
        let prose = "The set {x} is small. Result: {\"ok\": true, \"items\": [{\"n\": 1}]} done";
        assert_eq!(
            extract_json(prose),
            Some(json!({"ok": true, "items": [{"n": 1}]}))
        );
        assert_eq!(
            extract_json("values [1, 2, 3] follow"),
            Some(json!([1, 2, 3]))
        );
        assert_eq!(extract_json("```json\n42\n```"), Some(json!(42)));
    }

    #[test]
    fn non_json_tokens_and_truncated_documents_are_rejected() {
        assert_eq!(extract_json("{\"score\": NaN}"), None);
        assert_eq!(extract_json("{\"score\": Infinity}"), None);
        assert_eq!(extract_json("{\"a\": [1, 2"), None);
        assert_eq!(extract_json("nothing here"), None);
        assert_eq!(
            extract_json("a brace in a string {\"k\": \"}\"} ok"),
            Some(json!({"k": "}"}))
        );
    }
}
