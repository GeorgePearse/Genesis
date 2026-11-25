//! "SEARCH/REPLACE patches change the lines they name and nothing else":
//! cases ported from ShinkaEvolve's `tests/test_diff_patch_targeting.py`
//! (Apache-2.0, Sakana AI). Case names follow the originals.

use genesis_edit::{apply_diff_patch, apply_search_replace, PatchError, PatchOutcome};

fn patch(search: &str, replace: &str) -> String {
    format!("<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE")
}

fn apply(original: &str, patch: &str) -> Result<PatchOutcome, PatchError> {
    apply_diff_patch(patch, original, "python")
}

fn updated(original: &str, patch: &str) -> String {
    let outcome = apply(original, patch).unwrap_or_else(|error| panic!("patch refused:\n{error}"));
    assert_eq!(outcome.num_applied, 1);
    outcome.code
}

fn refusal(original: &str, patch: &str) -> String {
    apply(original, patch)
        .expect_err("the patch should be refused")
        .0
}

fn block(lines: &[&str], comment: &str) -> String {
    format!(
        "{comment} EVOLVE-BLOCK-START\n{}\n{comment} EVOLVE-BLOCK-END\n",
        lines.join("\n")
    )
}

// -- whole-line matching ------------------------------------------------------

#[test]
fn test_search_does_not_match_inside_a_longer_line() {
    let original = "# EVOLVE-BLOCK-START\nidx = 10\nx = 1\n# EVOLVE-BLOCK-END\n";
    assert_eq!(
        updated(original, &patch("x = 1", "x = 2")),
        "# EVOLVE-BLOCK-START\nidx = 10\nx = 2\n# EVOLVE-BLOCK-END\n"
    );
}

#[test]
fn test_search_that_is_only_part_of_a_line_is_not_found() {
    let original = "# EVOLVE-BLOCK-START\nidx = 10\n# EVOLVE-BLOCK-END\n";
    assert!(refusal(original, &patch("x = 1", "x = 2")).contains("SEARCH text not found"));
}

#[test]
fn test_multiline_search_must_end_at_a_line_end() {
    let original = "# EVOLVE-BLOCK-START\na = 1\nb = 22\na = 1\nb = 2\n# EVOLVE-BLOCK-END\n";
    assert_eq!(
        updated(original, &patch("a = 1\nb = 2", "c = 3")),
        "# EVOLVE-BLOCK-START\na = 1\nb = 22\nc = 3\n# EVOLVE-BLOCK-END\n"
    );
}

#[test]
fn test_reindented_search_replaces_the_line_it_matched() {
    let original = "def helper():\n    value = 1\n# EVOLVE-BLOCK-START\ndef solve():\n    if True:\n        value = 1\n    return value\n# EVOLVE-BLOCK-END\n";
    assert_eq!(
        updated(original, &patch("value = 1", "value = 2")),
        original.replace("        value = 1", "        value = 2")
    );
}

#[test]
fn test_block_whose_first_line_lost_its_indentation_still_applies() {
    let original = "# EVOLVE-BLOCK-START\ndef solve(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total\n# EVOLVE-BLOCK-END\n";
    let edit = patch(
        "for i in range(n):\n        total += i",
        "for i in range(n):\n        total += 2 * i",
    );
    assert_eq!(
        updated(original, &edit),
        original.replace("total += i", "total += 2 * i")
    );
}

#[test]
fn test_first_line_without_indentation_does_not_match_mid_line() {
    let original =
        "# EVOLVE-BLOCK-START\n    total = idx = 1\n    done = True\n# EVOLVE-BLOCK-END\n";
    let edit = patch("idx = 1\n    done = True", "idx = 2\n    done = False");
    assert!(refusal(original, &edit).contains("SEARCH text not found"));
}

// -- matches outside the editable region ---------------------------------------

#[test]
fn test_match_inside_the_block_wins_over_an_earlier_one_outside() {
    let original = "x = 1\n# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n";
    assert_eq!(
        updated(original, &patch("x = 1", "x = 2")),
        "x = 1\n# EVOLVE-BLOCK-START\nx = 2\n# EVOLVE-BLOCK-END\n"
    );
}

#[test]
fn test_match_only_outside_the_block_is_rejected() {
    let original = "x = 1\n# EVOLVE-BLOCK-START\ny = 1\n# EVOLVE-BLOCK-END\n";
    assert!(refusal(original, &patch("x = 1", "x = 2"))
        .contains("Attempted to edit outside EVOLVE-BLOCK regions"));
}

#[test]
fn test_verbatim_immutable_code_is_not_swapped_for_an_indented_look_alike() {
    let original = "def helper():\n    return 1\n# EVOLVE-BLOCK-START\nclass Solver:\n    def helper():\n        return 1\n# EVOLVE-BLOCK-END\n";
    let edit = patch("def helper():\n    return 1", "def helper():\n    return 2");
    assert!(refusal(original, &edit).contains("Attempted to edit outside EVOLVE-BLOCK regions"));
}

// -- code on the same line as a marker -------------------------------------------

#[test]
fn test_code_before_an_end_marker_can_be_replaced() {
    let original = "# EVOLVE-BLOCK-START\ndef f(x):\n    return x  # EVOLVE-BLOCK-END\n";
    assert_eq!(
        updated(original, &patch("    return x", "    return x + 1")),
        original.replace("return x ", "return x + 1 ")
    );
}

#[test]
fn test_code_after_a_start_marker_can_be_replaced() {
    // Marker validation refuses this layout for Markdown, so it is only
    // reachable through the matcher itself.
    let original = "<!-- EVOLVE-BLOCK-START --><p>hi</p>\n<!-- EVOLVE-BLOCK-END -->\n";
    let (code, applied) =
        apply_search_replace(&patch("<p>hi</p>", "<p>bye</p>"), original).unwrap();
    assert_eq!((code, applied), (original.replace("hi", "bye"), 1));
    assert!(
        apply_diff_patch(&patch("<p>hi</p>", "<p>bye</p>"), original, "markdown")
            .unwrap_err()
            .0
            .contains("malformed EVOLVE-BLOCK-START marker")
    );
}

#[test]
fn test_marker_line_itself_stays_out_of_reach() {
    let original = "# EVOLVE-BLOCK-START\nx = 1\ny = 2  # EVOLVE-BLOCK-END\ntail = 0\n";
    assert!(apply(
        original,
        &patch("y = 2  # EVOLVE-BLOCK-END\ntail = 0", "y = 3\ntail = 1")
    )
    .is_err());
}

// -- trailing comments --------------------------------------------------------

#[test]
fn test_search_without_the_trailing_comment_of_its_line_applies() {
    let original = block(&["lr = 0.01  # tuned", "steps = 10"], "#");
    assert_eq!(
        updated(&original, &patch("lr = 0.01", "lr = 0.02")),
        block(&["lr = 0.02  # tuned", "steps = 10"], "#")
    );
}

#[test]
fn test_commented_line_is_preferred_over_a_look_alike_at_another_indentation() {
    let original = block(
        &[
            "def f():",
            "    best = 0.0  # running best",
            "    return best",
            "class G:",
            "    def g(self):",
            "        best = 0.0",
            "        return best",
        ],
        "#",
    );
    assert_eq!(
        updated(&original, &patch("    best = 0.0", "    best = 1.0")),
        original.replace("best = 0.0  # running", "best = 1.0  # running")
    );
}

#[test]
fn test_whole_line_match_is_preferred_over_a_commented_line() {
    let original = block(&["x = 1  # first", "x = 1"], "#");
    assert_eq!(
        updated(&original, &patch("x = 1", "x = 2")),
        block(&["x = 1  # first", "x = 2"], "#")
    );
}

#[test]
fn test_comments_inside_a_block_are_replaced_with_their_lines() {
    let original = block(&["a = 1  # one", "b = 2", "c = 3  # three"], "#");
    assert_eq!(
        updated(&original, &patch("a = 1\nb = 2\nc = 3", "a = 10\nc = 30")),
        block(&["a = 10", "c = 30  # three"], "#")
    );
}

#[test]
fn test_search_line_that_ends_before_code_is_not_found() {
    // Code, not a comment: `//` divides in Python and `#` sits in a string.
    for (line, search) in [
        ("x = a // 2", "x = a"),
        ("x = a + b", "x = a"),
        ("# x = 1", ""),
        ("    # only a comment", "    "),
    ] {
        let original = block(&["y = 0", line, "z = 9"], "#");
        let edit = patch(&format!("y = 0\n{search}\nz = 9"), "y = 1");
        assert!(
            refusal(&original, &edit).contains("SEARCH text not found"),
            "{line:?} {search:?}"
        );
    }
}

#[test]
fn test_trailing_comment_follows_the_comment_style_of_the_file() {
    let original = block(&["int lr = 1;  // tuned", "int q = a # b;"], "//");
    let outcome = apply_diff_patch(&patch("int lr = 1;", "int lr = 2;"), &original, "cpp").unwrap();
    assert_eq!(outcome.code, original.replace("int lr = 1;", "int lr = 2;"));
    let error = apply_diff_patch(&patch("int q = a", "int q = c"), &original, "cpp")
        .unwrap_err()
        .0;
    assert!(error.contains("SEARCH text not found"));
}

// -- marker-only SEARCH -------------------------------------------------------

const TWO_BLOCKS: &str =
    "# EVOLVE-BLOCK-START\na = 1\n# EVOLVE-BLOCK-END\nmid = 0\n# EVOLVE-BLOCK-START\nb = 2\n# EVOLVE-BLOCK-END\n";
const ONE_BLOCK: &str = "import os\n# EVOLVE-BLOCK-START\na = 1\n# EVOLVE-BLOCK-END\nprint(a)\n";

#[test]
fn test_marker_only_search_is_rejected() {
    for search in [
        "# EVOLVE-BLOCK-START",
        "# EVOLVE-BLOCK-END",
        "# EVOLVE-BLOCK-START\n# EVOLVE-BLOCK-END",
        "    # EVOLVE-BLOCK-START   ",
    ] {
        let error = refusal(
            TWO_BLOCKS,
            &patch(search, "# EVOLVE-BLOCK-START\nimport os"),
        );
        assert!(
            error.contains("SEARCH block contains only EVOLVE-BLOCK markers"),
            "{search:?}: {error}"
        );
    }
}

#[test]
fn test_marker_only_search_is_rejected_for_other_comment_styles() {
    let original = "// EVOLVE-BLOCK-START\nint a = 1;\n// EVOLVE-BLOCK-END\n";
    let edit = patch("// EVOLVE-BLOCK-START", "// EVOLVE-BLOCK-START\nint b = 2;");
    assert!(apply_diff_patch(&edit, original, "cpp")
        .unwrap_err()
        .0
        .contains("only EVOLVE-BLOCK markers"));
}

#[test]
fn test_apply_search_replace_rejects_marker_only_search() {
    let error =
        apply_search_replace(&patch("# EVOLVE-BLOCK-START", "import os"), TWO_BLOCKS).unwrap_err();
    assert!(error.0.contains("only EVOLVE-BLOCK markers"));
}

#[test]
fn test_end_marker_search_appends_to_the_only_block() {
    // With one editable region its END marker names the place an append goes.
    for replace in ["b = 2\n# EVOLVE-BLOCK-END", "b = 2"] {
        assert_eq!(
            updated(ONE_BLOCK, &patch("# EVOLVE-BLOCK-END", replace)),
            ONE_BLOCK.replace("a = 1\n", "a = 1\nb = 2\n")
        );
    }
}

#[test]
fn test_start_marker_search_is_rejected_in_a_single_block_file() {
    for search in [
        "# EVOLVE-BLOCK-START",
        "# EVOLVE-BLOCK-START\n# EVOLVE-BLOCK-END",
    ] {
        assert!(refusal(ONE_BLOCK, &patch(search, "b = 2")).contains("only EVOLVE-BLOCK markers"));
    }
}

#[test]
fn test_marker_used_as_context_still_matches_the_code_below_it() {
    let edit = patch("# EVOLVE-BLOCK-START\nb = 2", "# EVOLVE-BLOCK-START\nb = 3");
    assert_eq!(
        updated(TWO_BLOCKS, &edit),
        TWO_BLOCKS.replace("b = 2", "b = 3")
    );
}

#[test]
fn test_empty_search_still_appends_to_the_last_block() {
    assert_eq!(
        updated(TWO_BLOCKS, &patch("", "c = 3")),
        TWO_BLOCKS.replace("b = 2\n", "b = 2\nc = 3\n")
    );
}

// -- immutable regions stay byte for byte --------------------------------------

fn with_immutable(block: &str) -> String {
    let head = "DOC = \"\"\"keep  \ntrailing\u{c}form feed\u{1c}and more\"\"\"\t\n\n";
    let tail = "TAIL = \"also  \"  \n\n\n";
    format!("{head}# EVOLVE-BLOCK-START\n{block}# EVOLVE-BLOCK-END\n{tail}")
}

#[test]
fn test_immutable_text_is_not_rewritten() {
    assert_eq!(
        updated(&with_immutable("x = 1\n"), &patch("x = 1", "x = 2")),
        with_immutable("x = 2\n")
    );
}

#[test]
fn test_rejected_patch_returns_an_error() {
    assert!(apply(&with_immutable("x = 1\n"), &patch("y = 1", "y = 2")).is_err());
}

#[test]
fn test_patch_without_changes_keeps_the_file_identical() {
    let original = with_immutable("x = 1\n");
    assert_eq!(updated(&original, &patch("x = 1", "x = 1")), original);
}

#[test]
fn test_trailing_whitespace_is_ignored_inside_the_block() {
    let original = with_immutable("x = 1   \ny = 2\t\n");
    assert_eq!(
        updated(&original, &patch("x = 1  \ny = 2", "x = 3  \ny = 4")),
        with_immutable("x = 3\ny = 4\n")
    );
}

#[test]
fn test_form_feed_in_editable_string_is_not_a_line_break() {
    let original = with_immutable("a = \"one\u{c}two\"\nb = 1\n");
    assert_eq!(
        updated(&original, &patch("b = 1", "b = \"three\u{c}four\"")),
        with_immutable("a = \"one\u{c}two\"\nb = \"three\u{c}four\"\n")
    );
}

#[test]
fn test_indented_end_marker_keeps_its_indentation() {
    let original = "class Solver:\n    # EVOLVE-BLOCK-START\n    x = 1   \n    # EVOLVE-BLOCK-END\n    y = 2\n";
    assert_eq!(
        updated(original, &patch("    x = 1", "    x = 2")),
        original.replace("    x = 1   \n", "    x = 2\n")
    );
}

#[test]
fn test_result_keeps_the_final_newline_and_diffs_against_the_normalised_original() {
    let original = with_immutable("x = 1  \n");
    let outcome = apply(&original, &patch("x = 1", "x = 2")).unwrap();
    assert_eq!(outcome.code, with_immutable("x = 2\n"));
    assert!(
        outcome.diff.contains("\n-x = 1\n+x = 2\n"),
        "{}",
        outcome.diff
    );
}

// -- empty SEARCH or REPLACE section -------------------------------------------

#[test]
fn test_block_with_an_empty_section() {
    let source = "# EVOLVE-BLOCK-START\ndebug = True\ntotal = 1\n# EVOLVE-BLOCK-END\n";
    let edit = patch("total = 1", "total = 2");
    for gap in ["", "\n"] {
        for kind in ["delete", "insert"] {
            let (block, expected) = if kind == "delete" {
                (
                    format!("<<<<<<< SEARCH\ndebug = True\n=======\n{gap}>>>>>>> REPLACE"),
                    source.replace("debug = True", ""),
                )
            } else {
                (
                    format!("<<<<<<< SEARCH\n{gap}=======\nlast = 9\n>>>>>>> REPLACE"),
                    source.replace("total = 1\n", "total = 1\nlast = 9\n"),
                )
            };
            let cases = [
                (vec![block.clone()], expected.clone()),
                (
                    vec![edit.clone(), block.clone()],
                    expected.replace("total = 1", "total = 2"),
                ),
                (
                    vec![block.clone(), edit.clone()],
                    expected.replace("total = 1", "total = 2"),
                ),
            ];
            for (blocks, expected) in cases {
                let outcome = apply(source, &blocks.join("\n\n")).unwrap();
                assert_eq!(outcome.num_applied, blocks.len(), "{kind} gap={gap:?}");
                assert_eq!(
                    outcome.code,
                    expected,
                    "{kind} gap={gap:?} blocks={}",
                    blocks.len()
                );
            }
        }
    }
}
