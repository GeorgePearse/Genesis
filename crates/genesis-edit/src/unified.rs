//! Unified diffs of a program before and after a patch, and their summary.

use crate::languages::extension;
use serde::{Deserialize, Serialize};
use similar::{ChangeTag, TextDiff};

/// `start,count` of a hunk that begins at the first line, as `diff -u`
/// writes it: the count is left out when it is one, and an empty side is
/// `0,0`.
fn whole_file_range(lines: usize) -> String {
    match lines {
        0 => "0,0".to_string(),
        1 => "1".to_string(),
        count => format!("1,{count}"),
    }
}

/// Unified diff of `original` against `updated` with the whole file as
/// context, in one hunk, headed `a/original.<ext>` and `b/original.<ext>`.
/// Empty when the two are equal.
///
/// The hunk header is counted from the lines written. `similar` 2.7 can
/// report a stale position for a leading deletion, and its own unified diff
/// then prints a header that disagrees with the hunk under it.
pub fn unified_diff(original: &str, updated: &str, language: &str) -> String {
    if original == updated {
        return String::new();
    }
    let diff = TextDiff::from_lines(original, updated);
    let (mut old_lines, mut new_lines) = (0, 0);
    let mut body = String::with_capacity(original.len() + updated.len());
    for change in diff.iter_all_changes() {
        body.push(match change.tag() {
            ChangeTag::Equal => {
                old_lines += 1;
                new_lines += 1;
                ' '
            }
            ChangeTag::Delete => {
                old_lines += 1;
                '-'
            }
            ChangeTag::Insert => {
                new_lines += 1;
                '+'
            }
        });
        body.push_str(change.value());
        // A last line without a newline must not run into the next one.
        if !change.value().ends_with('\n') {
            body.push('\n');
        }
    }
    let ext = extension(language);
    format!(
        "--- a/original.{ext}\n+++ b/original.{ext}\n@@ -{} +{} @@\n{body}",
        whole_file_range(old_lines),
        whole_file_range(new_lines)
    )
}

/// Line counts of a unified diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct DiffSummary {
    /// Lines added beyond those paired with a removal.
    pub added: usize,
    /// Lines removed beyond those paired with an addition.
    pub deleted: usize,
    /// Removed lines paired with an added line in the same hunk.
    pub modified: usize,
}

/// `@@ -l[,s] +l[,s] @@` gives the old and new line counts of a hunk.
fn hunk_lengths(header: &str) -> Option<(usize, usize)> {
    let mut fields = header.strip_prefix("@@ ")?.split(' ');
    let length = |field: &str, sign: char| -> Option<usize> {
        let range = field.strip_prefix(sign)?;
        match range.split_once(',') {
            Some((_, count)) => count.parse().ok(),
            None => range.parse::<usize>().ok().map(|_| 1),
        }
    };
    Some((length(fields.next()?, '-')?, length(fields.next()?, '+')?))
}

/// Count added, deleted and modified lines over every hunk of `diff`.
/// Within a hunk, removals and additions pair up as modifications.
pub fn summarize_diff(diff: &str) -> DiffSummary {
    let mut summary = DiffSummary::default();
    let mut lines = diff.split('\n');
    while let Some(line) = lines.next() {
        let Some((mut old_left, mut new_left)) = hunk_lengths(line) else {
            continue;
        };
        let (mut added, mut removed) = (0usize, 0usize);
        while old_left > 0 || new_left > 0 {
            let Some(body) = lines.next() else { break };
            match body.as_bytes().first() {
                Some(b'+') => {
                    added += 1;
                    new_left = new_left.saturating_sub(1);
                }
                Some(b'-') => {
                    removed += 1;
                    old_left = old_left.saturating_sub(1);
                }
                Some(b'\\') => {}
                _ => {
                    old_left = old_left.saturating_sub(1);
                    new_left = new_left.saturating_sub(1);
                }
            }
        }
        let modified = added.min(removed);
        summary.modified += modified;
        summary.added += added - modified;
        summary.deleted += removed - modified;
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_has_git_style_headers_and_full_context() {
        let original = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\n";
        let updated = "a\nb\nc\nd\ne\nf\ng\nh\ni\nJ\n";
        let diff = unified_diff(original, updated, "python");
        assert!(
            diff.starts_with("--- a/original.py\n+++ b/original.py\n@@ -1,10 +1,10 @@\n a\n"),
            "{diff}"
        );
        assert!(diff.ends_with(" i\n-j\n+J\n"));
        assert_eq!(unified_diff(original, original, "python"), "");
        assert!(unified_diff("x", "y", "klingon").starts_with("--- a/original.txt\n"));
    }

    #[test]
    fn a_missing_final_newline_does_not_glue_lines() {
        let diff = unified_diff("a\nb", "a\nc", "rust");
        assert!(diff.ends_with(" a\n-b\n+c\n"), "{diff:?}");
    }

    #[test]
    fn summary_pairs_removals_with_additions_per_hunk() {
        let diff = "--- a/x\n+++ b/x\n@@ -1,4 +1,5 @@\n a\n-b\n-c\n+B\n+C\n+D\n d\n@@ -10,2 +11,1 @@\n-x\n y\n";
        assert_eq!(
            summarize_diff(diff),
            DiffSummary {
                added: 1,
                deleted: 1,
                modified: 2
            }
        );
        assert_eq!(summarize_diff(""), DiffSummary::default());
        assert_eq!(
            summarize_diff("not a diff\n+++ nothing\n"),
            DiffSummary::default()
        );
    }

    #[test]
    fn summary_reads_lines_that_look_like_headers_as_content() {
        let diff = unified_diff("--- x\n+++ y\nkeep\n", "++ x\nkeep\nmore\n", "python");
        assert_eq!(
            summarize_diff(&diff),
            DiffSummary {
                added: 0,
                deleted: 0,
                modified: 2
            }
        );
        let diff = unified_diff("one\n", "one\ntwo\nthree\n", "python");
        assert_eq!(
            summarize_diff(&diff),
            DiffSummary {
                added: 2,
                deleted: 0,
                modified: 0
            }
        );
    }
}
