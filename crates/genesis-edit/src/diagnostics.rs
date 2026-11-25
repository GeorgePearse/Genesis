//! The text a model reads when its patch is rejected.
//!
//! Each message says what was looked for, what was found instead and
//! whether it is editable, shows the editable regions, and ends with
//! concrete fixes, because the next attempt is written from this text alone.

use crate::markers::inside;
use crate::text::{quoted, split_lines, split_lines_keep_ends, strip};
use similar::{ChangeTag, TextDiff};

pub(crate) const NO_CHANGES_APPLIED: &str = "No changes applied";

pub(crate) const MARKER_ONLY_SEARCH: &str = "SEARCH block contains only EVOLVE-BLOCK markers

The EVOLVE-BLOCK-START and EVOLVE-BLOCK-END lines are immutable and cannot be
used as an anchor.

Suggestions:
1. To change code, copy the lines you want to replace into the SEARCH block
2. To add code at the end of the editable region, leave the SEARCH block empty
3. Do not repeat the EVOLVE-BLOCK markers in SEARCH/REPLACE blocks";

/// Similarity of two strings in `[0, 1]`: twice the matched characters over
/// the total, as `difflib.SequenceMatcher.ratio` defines it.
fn similarity(a: &str, b: &str) -> f64 {
    let (len_a, len_b) = (a.chars().count(), b.chars().count());
    if len_a + len_b == 0 {
        return 1.0;
    }
    f64::from(TextDiff::from_chars(a, b).ratio())
}

/// The best similarity two strings of these lengths could reach.
fn similarity_bound(a: &str, b: &str) -> f64 {
    let (len_a, len_b) = (a.chars().count(), b.chars().count());
    if len_a + len_b == 0 {
        return 1.0;
    }
    2.0 * len_a.min(len_b) as f64 / (len_a + len_b) as f64
}

/// Up to three lines that resemble `search_line`, best first, with their
/// 1-based line numbers.
fn similar_lines<'a>(search_line: &str, original: &'a str) -> Vec<(&'a str, usize)> {
    let wanted = strip(search_line);
    if wanted.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(&str, usize, f64)> = split_lines(original)
        .into_iter()
        .enumerate()
        .filter_map(|(index, line)| {
            let candidate = strip(line);
            if candidate.is_empty() || similarity_bound(wanted, candidate) <= 0.6 {
                return None;
            }
            let ratio = similarity(wanted, candidate);
            (ratio > 0.6).then_some((line, index + 1, ratio))
        })
        .collect();
    scored.sort_by(|a, b| b.2.total_cmp(&a.2));
    scored
        .into_iter()
        .take(3)
        .map(|(line, number, _)| (line, number))
        .collect()
}

struct ClosestBlock<'a> {
    lines: Vec<&'a str>,
    start_line: usize,
    differences: Vec<String>,
}

/// The run of lines, as long as the search block, that resembles it most.
fn closest_block<'a>(search_text: &str, original: &'a str) -> Option<ClosestBlock<'a>> {
    let search_lines = split_lines(strip(search_text));
    if search_lines.is_empty() {
        return None;
    }
    let original_lines = split_lines(original);
    let count = search_lines.len();
    if original_lines.len() < count {
        return None;
    }
    let search_block = search_lines.join("\n");
    let mut best: Option<(usize, f64)> = None;
    for start in 0..=original_lines.len() - count {
        let candidate = original_lines[start..start + count].join("\n");
        let floor = best.map_or(0.7, |(_, ratio)| ratio.max(0.7));
        if similarity_bound(&search_block, &candidate) <= floor {
            continue;
        }
        let ratio = similarity(&search_block, &candidate);
        if ratio > floor {
            best = Some((start, ratio));
        }
    }
    let (start, _) = best?;
    let lines = original_lines[start..start + count].to_vec();
    let diff = TextDiff::from_slices(&search_lines, &lines);
    let differences = diff
        .iter_all_changes()
        .filter_map(|change| match change.tag() {
            ChangeTag::Delete => Some(format!("-  {}", change.value())),
            ChangeTag::Insert => Some(format!("+  {}", change.value())),
            ChangeTag::Equal => None,
        })
        .collect();
    Some(ClosestBlock {
        lines,
        start_line: start + 1,
        differences,
    })
}

/// Byte offset at which a 1-based line starts; 0 when out of range.
fn line_position(text: &str, line_number: usize) -> usize {
    let lines = split_lines_keep_ends(text);
    if line_number < 1 || line_number > lines.len() {
        return 0;
    }
    lines[..line_number - 1].iter().map(|line| line.len()).sum()
}

/// 1-based line number holding a byte offset.
fn line_number_at(text: &str, position: usize) -> usize {
    let lines = split_lines_keep_ends(text);
    let mut consumed = 0;
    for (index, line) in lines.iter().enumerate() {
        if consumed + line.len() > position {
            return index + 1;
        }
        consumed += line.len();
    }
    lines.len().max(1)
}

fn editable(span: (usize, usize), mutable: &[(usize, usize)]) -> &'static str {
    if inside(span, mutable) {
        "✓ editable"
    } else {
        "✗ immutable"
    }
}

/// SEARCH text matched nothing.
pub(crate) fn search_not_found(
    search_text: &str,
    original: &str,
    mutable: &[(usize, usize)],
) -> String {
    let search_lines = split_lines(strip(search_text));
    let Some(first_line) = search_lines.first().map(|line| strip(line)) else {
        return "Empty search text provided".to_string();
    };
    let similar = similar_lines(first_line, original);
    let mut parts: Vec<String> = vec![
        "SEARCH text not found in editable regions".into(),
        String::new(),
    ];

    if search_lines.len() == 1 {
        parts.push(format!("Looking for: {}", quoted(first_line)));
        parts.push(String::new());
    } else {
        parts.push(format!(
            "Looking for {}-line block starting with: {}",
            search_lines.len(),
            quoted(first_line)
        ));
        parts.extend(
            [
                "",
                "Full search pattern:",
                "```",
                strip(search_text),
                "```",
                "",
            ]
            .map(String::from),
        );
    }

    if let Some(block) = closest_block(search_text, original) {
        let start = line_position(original, block.start_line);
        let span = (start, start + block.lines.join("\n").len());
        parts.push(format!(
            "Found similar code block at line {} ({}):",
            block.start_line,
            editable(span, mutable)
        ));
        parts.extend(
            [
                "",
                "Differences between search pattern and actual code:",
                "```diff",
            ]
            .map(String::from),
        );
        parts.extend(block.differences);
        parts.extend(["```", ""].map(String::from));
        if !inside(span, mutable) {
            parts.extend(
                [
                    "⚠️  Note: The similar code is in an immutable region.",
                    "   Look for similar code in the editable regions below.",
                    "",
                ]
                .map(String::from),
            );
        }
    } else if !similar.is_empty() {
        parts.push("Found similar text (but not exact match):".into());
        for (line, number) in &similar {
            let start = line_position(original, *number);
            let status = editable((start, start + line.len()), mutable);
            parts.push(format!("  Line {number}: {} ({status})", strip(line)));
        }
        parts.push(String::new());
    }

    if !mutable.is_empty() {
        parts.push("Editable regions where you can make changes:".into());
        for (index, (start, end)) in mutable.iter().take(2).enumerate() {
            parts.push(format!(
                "  Region {} (lines {}-{}):",
                index + 1,
                line_number_at(original, *start),
                line_number_at(original, *end)
            ));
            let region_lines = split_lines(strip(&original[*start..*end]));
            if region_lines.is_empty() {
                continue;
            }
            if region_lines.len() <= 6 {
                parts.extend(region_lines.iter().map(|line| format!("    {line}")));
            } else {
                parts.extend(region_lines[..3].iter().map(|line| format!("    {line}")));
                parts.push(format!("    ... ({} more lines)", region_lines.len() - 6));
                parts.extend(
                    region_lines[region_lines.len() - 3..]
                        .iter()
                        .map(|line| format!("    {line}")),
                );
            }
            parts.push(String::new());
        }
        if mutable.len() > 2 {
            parts.push(format!("  ... and {} more regions", mutable.len() - 2));
            parts.push(String::new());
        }
    }

    parts.push("Quick fixes:".into());
    if similar.is_empty() {
        parts.extend(
            [
                "• Verify the text exists in the file",
                "• Check that you're searching within EVOLVE-BLOCK regions",
                "• Try a smaller, more specific search pattern",
            ]
            .map(String::from),
        );
    } else {
        parts.extend(
            [
                "• Check indentation - search text must match exactly including spaces/tabs",
                "• Look for typos in the search text",
                "• Try searching for just the first line instead of the full block",
            ]
            .map(String::from),
        );
    }
    parts.join("\n")
}

/// SEARCH text matched, but only in read-only code.
pub(crate) fn outside_evolve_block(
    matched_text: &str,
    position: usize,
    original: &str,
    mutable: &[(usize, usize)],
) -> String {
    let first_line = split_lines(matched_text).first().copied().unwrap_or("");
    let lines = split_lines(original);
    let target = line_number_at(original, position) - 1;
    let from = target.saturating_sub(3);
    let to = (target + 4).min(lines.len());

    let mut parts: Vec<String> = vec![
        "Attempted to edit outside EVOLVE-BLOCK regions".into(),
        String::new(),
        format!("Found text: {}", quoted(first_line)),
        format!("At position: {position}"),
        String::new(),
        "Context around found text:".into(),
    ];
    for (index, line) in lines.iter().enumerate().take(to).skip(from) {
        let pointer = if index == target { " >>> " } else { "     " };
        parts.push(format!("{pointer}Line {:3}: {line}", index + 1));
    }
    parts.extend(
        [
            "",
            "This text was found in an immutable (non-editable) region.",
            "",
            "Available editable regions (EVOLVE-BLOCK content):",
        ]
        .map(String::from),
    );
    if mutable.is_empty() {
        parts.push("  No EVOLVE-BLOCK regions found in the code!".into());
    }
    for (index, (start, end)) in mutable.iter().take(3).enumerate() {
        let region_lines = split_lines(strip(&original[*start..*end]));
        if region_lines.is_empty() {
            continue;
        }
        parts.push(format!("  Region {} (bytes {start}-{end}):", index + 1));
        parts.extend(
            region_lines
                .iter()
                .take(3)
                .map(|line| format!("    {line}")),
        );
        if region_lines.len() > 3 {
            parts.push(format!("    ... ({} more lines)", region_lines.len() - 3));
        }
        parts.push(String::new());
    }
    parts.extend(
        [
            "Suggestions:",
            "1. Move your edit to within an EVOLVE-BLOCK region",
            "2. Check if similar code exists in the editable regions above",
            "3. Ensure your search text targets code within # EVOLVE-BLOCK-START/END markers",
        ]
        .map(String::from),
    );
    parts.join("\n")
}

/// An insertion was asked for, but the file has no editable region.
pub(crate) fn no_evolve_block(original: &str, operation: &str) -> String {
    let lines = split_lines(original);
    let mut parts: Vec<String> = vec![
        format!("Cannot perform {operation}: No EVOLVE-BLOCK regions found"),
        String::new(),
        "The code must contain EVOLVE-BLOCK regions to be editable.".into(),
        String::new(),
        "Current file structure:".into(),
    ];
    parts.extend(
        lines
            .iter()
            .take(10)
            .enumerate()
            .map(|(index, line)| format!("  Line {:2}: {line}", index + 1)),
    );
    if lines.len() > 10 {
        parts.push(format!("  ... ({} more lines)", lines.len() - 10));
    }
    parts.extend(
        [
            "",
            "Expected format:",
            "```",
            "# Your immutable code here",
            "",
            "# EVOLVE-BLOCK-START",
            "# Your editable code goes here",
            "def function():",
            "    pass",
            "# EVOLVE-BLOCK-END",
            "",
            "# More immutable code here",
            "```",
            "",
            "Suggestions:",
            "1. Add EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers around editable code",
            "2. Ensure the markers are properly formatted (with # for Python, // for C/C++)",
            "3. Check that there's at least one EVOLVE-BLOCK region in the file",
        ]
        .map(String::from),
    );
    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markers::mutable_ranges;

    #[test]
    fn search_not_found_shows_the_closest_block_and_whether_it_is_editable() {
        let original = "import os\n# EVOLVE-BLOCK-START\ndef process():\n    data = load_data()\n    result = transform(data)\n    return result\n# EVOLVE-BLOCK-END\n";
        let search = "data = load_data()\nresult = transform_data(data)\nreturn result";
        let message = search_not_found(search, original, &mutable_ranges(original));
        assert!(message
            .starts_with("SEARCH text not found in editable regions\n\nLooking for 3-line block"));
        assert!(
            message.contains("Found similar code block at line 4 (✓ editable):"),
            "{message}"
        );
        assert!(message.contains("-  result = transform_data(data)"));
        assert!(message.contains("+      result = transform(data)"));
        assert!(message.contains("  Region 1 (lines 2-7):"));
        assert!(message.contains("• Check indentation"));
    }

    #[test]
    fn search_not_found_flags_immutable_look_alikes() {
        let original = "value_total = compute()\n# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n";
        let message = search_not_found(
            "value_totals = compute()",
            original,
            &mutable_ranges(original),
        );
        assert!(message.contains("Looking for: \"value_totals = compute()\""));
        assert!(message.contains("(✗ immutable)"));
        assert!(message.contains("The similar code is in an immutable region."));
        let message = search_not_found("zzz", original, &mutable_ranges(original));
        assert!(message.contains("• Verify the text exists in the file"));
        assert_eq!(
            search_not_found(" \n", original, &[]),
            "Empty search text provided"
        );
    }

    #[test]
    fn outside_block_points_at_the_line_that_matched() {
        let original = "import os\nimport sys\n\n# EVOLVE-BLOCK-START\ndef f():\n    return 1\n# EVOLVE-BLOCK-END\n";
        let message = outside_evolve_block("import os", 0, original, &mutable_ranges(original));
        assert!(message.starts_with("Attempted to edit outside EVOLVE-BLOCK regions"));
        assert!(message.contains(" >>> Line   1: import os"), "{message}");
        assert!(message.contains("     Line   2: import sys"));
        assert!(message.contains("Available editable regions"));
        assert!(message.contains("Suggestions:"));
        assert!(outside_evolve_block("x", 0, "x", &[])
            .contains("No EVOLVE-BLOCK regions found in the code!"));
    }

    #[test]
    fn no_evolve_block_lists_the_head_of_the_file() {
        let original = (1..=12)
            .map(|n| format!("line{n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let message = no_evolve_block(&original, "insertion");
        assert!(message.starts_with("Cannot perform insertion: No EVOLVE-BLOCK regions found"));
        assert!(message.contains("  Line 10: line10"));
        assert!(message.contains("  ... (2 more lines)"));
        assert!(message.contains("Expected format:") && message.contains("Suggestions:"));
    }

    #[test]
    fn line_offsets_round_trip() {
        let text = "a\nbb\nccc";
        assert_eq!(
            (
                line_position(text, 1),
                line_position(text, 2),
                line_position(text, 3)
            ),
            (0, 2, 5)
        );
        assert_eq!(line_position(text, 9), 0);
        assert_eq!(
            (
                line_number_at(text, 0),
                line_number_at(text, 2),
                line_number_at(text, 7)
            ),
            (1, 2, 3)
        );
        assert_eq!(line_number_at("", 0), 1);
        assert!((similarity("abcd", "abcf") - 0.75).abs() < 1e-6);
    }
}
