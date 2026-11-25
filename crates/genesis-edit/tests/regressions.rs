//! Plain tests for the failures the property suite found, each with the
//! input ephyra reported and the line that replays it.

use genesis_edit::{
    apply_diff_patch, apply_full_patch, immutable_parts, summarize_diff, unified_diff, DiffSummary,
};

fn patch(search: &str, replace: &str) -> String {
    format!("<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE")
}

/// Found by `a_patch_that_ignores_the_rules_cannot_change_read_only_text_or_markers`
/// (EPHYRA_REPLAY=1:10). ShinkaEvolve removes marker lines from REPLACE text
/// but lets a marker that shares its line with code through, which moves the
/// boundary of the editable region: here `region_2 = 67` became read-only.
#[test]
fn a_replacement_cannot_write_a_marker_next_to_code() {
    let original = "# EVOLVE-BLOCK-START\n    region_2 = 67\n    if idx > 6:\n# EVOLVE-BLOCK-END\n";
    let error = apply_diff_patch(
        &patch("    if idx > 6:", "# EVOLVE-BLOCK-START y = 2"),
        original,
        "python",
    )
    .expect_err("a marker was written into the editable region");
    assert!(error.0.contains("EVOLVE-BLOCK-START"), "{error}");
    assert!(error.0.contains("Suggestions:"), "{error}");

    // An empty SEARCH appends; the same rule holds there.
    let appended = apply_diff_patch(
        &patch("", "done = True  # EVOLVE-BLOCK-END"),
        original,
        "python",
    );
    assert!(appended.is_err());
}

/// The same hole, used on purpose: a START written into the first region and
/// an END written into the second pair up across the read-only code between
/// them, and a third block then edits that code. Each block applied in the
/// original implementation.
#[test]
fn written_markers_cannot_open_read_only_code_between_regions() {
    let original = "\
# EVOLVE-BLOCK-START
a = 1
# EVOLVE-BLOCK-END
SECRET_SEED = 42
# EVOLVE-BLOCK-START
b = 2
# EVOLVE-BLOCK-END
";
    let response = [
        patch("a = 1", "a = 1  # EVOLVE-BLOCK-START"),
        patch("b = 2", "b = 2  # EVOLVE-BLOCK-END"),
        patch("SECRET_SEED = 42", "SECRET_SEED = 0"),
    ]
    .join("\n\n");
    assert!(apply_diff_patch(&response, original, "python").is_err());

    // Each step alone is refused too, so the hole cannot be opened over
    // several generations.
    for step in [
        patch("a = 1", "a = 1  # EVOLVE-BLOCK-START"),
        patch("b = 2", "b = 2  # EVOLVE-BLOCK-END"),
    ] {
        assert!(apply_diff_patch(&step, original, "python").is_err());
    }
    let honest = apply_diff_patch(&patch("b = 2", "b = 3"), original, "python").unwrap();
    assert_eq!(immutable_parts(&honest.code), immutable_parts(original));
}

/// Found by `a_full_rewrite_carrying_a_target_yields_the_target`
/// (EPHYRA_REPLAY=1:4). ShinkaEvolve ends the code at the first run of three
/// backticks, so a fence inside an editable region cut the rewrite short: the
/// END marker was lost and, with one region, the rest of the program with it.
#[test]
fn a_fence_inside_a_region_does_not_end_the_rewrite() {
    let original = "# EVOLVE-BLOCK-START\n    region_1 = 80\n# EVOLVE-BLOCK-END\n# EVOLVE-BLOCK-START\n    region_2 = 64\n# EVOLVE-BLOCK-END\n";
    let target = original.replace("    region_1 = 80", "region_0_new = 46\n    ```");
    let outcome =
        apply_full_patch(&format!("```PYTHON\n{target}\n```"), original, "python").unwrap();
    assert_eq!(outcome.code, target);

    let document = "<!-- EVOLVE-BLOCK-START -->\n# Setup\n<!-- EVOLVE-BLOCK-END -->\n";
    let rewritten = "<!-- EVOLVE-BLOCK-START -->\n# Setup\n\n```bash\nmake install\n```\n\nThen run it.\n<!-- EVOLVE-BLOCK-END -->\n";
    let response = format!(
        "Here is the new guide.\n\n```markdown\n{rewritten}```\n\nIt adds the install step."
    );
    assert_eq!(
        apply_full_patch(&response, document, "markdown")
            .unwrap()
            .code,
        rewritten
    );
}

/// Found by `a_unified_diff_rebuilds_both_files_and_its_summary_counts_its_lines`
/// (EPHYRA_REPLAY=1:51), reduced by hand. `similar` 2.7.0 reports a stale
/// index for the leading deletion, and its unified diff printed the hunk
/// header `@@ -1 +2 @@` for these two-line files.
#[test]
fn the_hunk_header_counts_the_lines_of_both_files() {
    let diff = unified_diff("a\nb\n", "b\nb\n", "python");
    assert_eq!(
        diff,
        "--- a/original.py\n+++ b/original.py\n@@ -1,2 +1,2 @@\n-a\n b\n+b\n"
    );
    assert_eq!(
        summarize_diff(&diff),
        DiffSummary {
            added: 0,
            deleted: 0,
            modified: 1
        }
    );

    let diff = unified_diff(
        "\\ no\na\nb\n",
        "a\né\n+++ y\n\n@@ -1 +1 @@\na\n \nc\na\n--- x\n\n",
        "rust",
    );
    assert!(diff.contains("\n@@ -1,3 +1,11 @@\n"), "{diff}");
    assert_eq!(
        summarize_diff(&diff),
        DiffSummary {
            added: 8,
            deleted: 0,
            modified: 2
        }
    );
    assert!(unified_diff("", "x\n", "python").contains("@@ -0,0 +1 @@"));
    assert!(unified_diff("x\n", "", "python").contains("@@ -1 +0,0 @@"));
}

/// Found by `a_rewrite_without_both_markers_replaces_exactly_the_region`
/// (EPHYRA_REPLAY=1:0). ShinkaEvolve trims the code it takes from the fence
/// on both sides, which removes the indentation of its first line: a region
/// inside a function came back with its first statement at column zero.
#[test]
fn the_first_line_of_a_rewrite_keeps_its_indentation() {
    let original = "import original_header\n<!-- EVOLVE-BLOCK-START -->\nregion_0 = 87\n<!-- EVOLVE-BLOCK-END -->\noriginal_footer()\n";
    let outcome =
        apply_full_patch("```markdown\n    region_new = 91```", original, "markdown").unwrap();
    assert_eq!(
        outcome.code,
        original.replace("region_0 = 87", "    region_new = 91")
    );

    let method = "class Solver:\n    def solve(self):\n        # EVOLVE-BLOCK-START\n        return 1\n        # EVOLVE-BLOCK-END\n";
    let outcome = apply_full_patch(
        "```python\n\n        x = 2\n        return x\n```",
        method,
        "python",
    )
    .unwrap();
    assert_eq!(
        outcome.code,
        method.replace("        return 1\n", "        x = 2\n        return x\n")
    );
}

/// The same property, on markers that are indented (EPHYRA_REPLAY=1:0 and
/// then 1:1, each after the fix before it). ShinkaEvolve put the END marker
/// back at column zero, and it counted the START marker's indentation as
/// part of the file's header, so a whole-file rewrite without markers lost
/// that many characters from its first line or was not recognised at all.
#[test]
fn a_rewrite_without_markers_keeps_indented_markers_where_they_were() {
    let original = "import os\n\ndef solve():\n    # EVOLVE-BLOCK-START\n    return 1\n    # EVOLVE-BLOCK-END\n\nprint(solve())\n";
    let target = original.replace("    return 1\n", "    x = 2\n    return x\n");
    let body = "    x = 2\n    return x";
    let whole = format!("import os\n\ndef solve():\n{body}\n\nprint(solve())\n");
    let start_only =
        format!("import os\n\ndef solve():\n    # EVOLVE-BLOCK-START\n{body}\n\nprint(solve())\n");
    let end_only =
        format!("import os\n\ndef solve():\n{body}\n    # EVOLVE-BLOCK-END\n\nprint(solve())\n");
    for sent in [body.to_string(), whole, start_only, end_only] {
        let outcome =
            apply_full_patch(&format!("```python\n{sent}\n```"), original, "python").unwrap();
        assert_eq!(outcome.code, target, "sent:\n{sent}");
    }
}

/// An empty rewrite empties the region; it does not put both markers on one
/// line, which ShinkaEvolve did.
#[test]
fn an_empty_rewrite_leaves_the_markers_on_their_own_lines() {
    let original = "head\n    # EVOLVE-BLOCK-START\n    x = 1\n    # EVOLVE-BLOCK-END\ntail\n";
    let outcome = apply_full_patch("```python\n```", original, "python").unwrap();
    assert_eq!(
        outcome.code,
        "head\n    # EVOLVE-BLOCK-START\n    # EVOLVE-BLOCK-END\ntail\n"
    );
}
