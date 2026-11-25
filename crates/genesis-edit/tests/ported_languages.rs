//! Per-language cases ported from ShinkaEvolve's `tests/test_edit_cpp.py`,
//! `test_edit_fortran.py`, `test_edit_go.py`, `test_edit_julia.py`,
//! `test_edit_markdown.py`, `test_edit_verilog.py` and
//! `test_edit_wolfram.py` (Apache-2.0, Sakana AI). The Python cases also
//! check the files written to a patch directory; here the same content is
//! on the returned `PatchOutcome`.

use genesis_edit::{apply_diff_patch, apply_full_patch, extension};

/// A diff patch wrapped in the language's markers, as models often send it.
fn check_diff(language: &str, original: &str, patch: &str, expected_line: &str, ext: &str) {
    let outcome = apply_diff_patch(patch, original, language)
        .unwrap_or_else(|error| panic!("{language}: {error}"));
    assert_eq!(outcome.num_applied, 1, "{language}");
    assert!(
        outcome.code.contains(expected_line),
        "{language}: {}",
        outcome.code
    );
    assert!(
        outcome
            .diff
            .starts_with(&format!("--- a/original.{ext}\n+++ b/original.{ext}\n")),
        "{}",
        outcome.diff
    );
    assert!(outcome.diff.contains(expected_line));
    // Markers are stripped before SEARCH/REPLACE parsing.
    assert!(
        !outcome.patch_text.contains("EVOLVE-BLOCK-START")
            && !outcome.patch_text.contains("EVOLVE-BLOCK-END")
    );
    assert_eq!(extension(language), ext);
}

fn check_full(language: &str, original: &str, patch: &str, expected_line: &str) {
    let outcome = apply_full_patch(patch, original, language)
        .unwrap_or_else(|error| panic!("{language}: {error}"));
    assert_eq!(outcome.num_applied, 1, "{language}");
    assert!(
        outcome.code.contains(expected_line),
        "{language}: {}",
        outcome.code
    );
    assert!(outcome.diff.contains(expected_line));
    assert!(outcome.patch_text.contains(expected_line));
}

#[test]
fn test_apply_diff_patch_supports_fortran() {
    let original = "! EVOLVE-BLOCK-START\ninteger function score(x)\n    integer, intent(in) :: x\n    score = x + 1\nend function score\n! EVOLVE-BLOCK-END";
    let patch = "! EVOLVE-BLOCK-START\n<<<<<<< SEARCH\nscore = x + 1\n=======\nscore = x + 2\n>>>>>>> REPLACE\n! EVOLVE-BLOCK-END";
    check_diff("fortran", original, patch, "    score = x + 2", "f90");
}

#[test]
fn test_apply_full_patch_supports_fortran_fences() {
    let original = "! EVOLVE-BLOCK-START\ninteger function score(x)\n    integer, intent(in) :: x\n    score = x + 1\nend function score\n! EVOLVE-BLOCK-END\n";
    for (language, fence, value) in [("fortran", "f90", "10"), ("f90", "fortran", "20")] {
        let patch = format!(
            "```{fence}\n! EVOLVE-BLOCK-START\ninteger function score(x)\n    integer, intent(in) :: x\n    score = x + {value}\nend function score\n! EVOLVE-BLOCK-END\n```"
        );
        check_full(language, original, &patch, &format!("score = x + {value}"));
    }
}

#[test]
fn test_apply_diff_patch_supports_go() {
    let original =
        "// EVOLVE-BLOCK-START\nfunc score(x int) int {\n    return x + 1\n}\n// EVOLVE-BLOCK-END";
    let patch = "// EVOLVE-BLOCK-START\n<<<<<<< SEARCH\n    return x + 1\n=======\n    return x + 2\n>>>>>>> REPLACE\n// EVOLVE-BLOCK-END";
    check_diff("go", original, patch, "    return x + 2", "go");
}

#[test]
fn test_apply_full_patch_supports_go_and_golang_fence() {
    let original = "// EVOLVE-BLOCK-START\nfunc score(x int) int {\n    return x + 1\n}\n// EVOLVE-BLOCK-END\n";
    let patch = "```golang\n// EVOLVE-BLOCK-START\nfunc score(x int) int {\n    return x + 10\n}\n// EVOLVE-BLOCK-END\n```";
    check_full("go", original, patch, "return x + 10");
    // Without markers the payload is the whole fence: nothing of the tag may leak into it.
    let bare = "```golang\nfunc score(x int) int {\n    return x + 10\n}\n```";
    assert_eq!(
        apply_full_patch(bare, original, "go").unwrap().code,
        original.replace("x + 1", "x + 10")
    );
}

#[test]
fn test_apply_diff_patch_supports_julia() {
    let original =
        "# EVOLVE-BLOCK-START\nfunction score(x)\n    return x + 1\nend\n# EVOLVE-BLOCK-END";
    let patch = "# EVOLVE-BLOCK-START\n<<<<<<< SEARCH\nreturn x + 1\n=======\nreturn x + 2\n>>>>>>> REPLACE\n# EVOLVE-BLOCK-END";
    check_diff("julia", original, patch, "    return x + 2", "jl");
}

#[test]
fn test_apply_full_patch_supports_julia_and_jl_fence() {
    let original =
        "# EVOLVE-BLOCK-START\nfunction score(x)\n    return x + 1\nend\n# EVOLVE-BLOCK-END\n";
    let patch = "```jl\n# EVOLVE-BLOCK-START\nfunction score(x)\n    return x + 10\nend\n# EVOLVE-BLOCK-END\n```";
    check_full("julia", original, patch, "return x + 10");
}

#[test]
fn test_apply_diff_patch_supports_verilog() {
    let original = "// EVOLVE-BLOCK-START\nmodule TopModule (\n  input clk,\n  input reset,\n  output reg [31:0] q\n);\n  always @(posedge clk) begin\n    if (reset)\n      q <= 32'h1;\n    else\n      q <= {q[0], q[31:1]};\n  end\nendmodule\n// EVOLVE-BLOCK-END";
    let patch = "// EVOLVE-BLOCK-START\n<<<<<<< SEARCH\n      q <= {q[0], q[31:1]};\n=======\n      q <= {q[0] ^ q[31], q[31:22], q[21] ^ q[0], q[20:2], q[1] ^ q[0], q[0] ^ q[0]};\n>>>>>>> REPLACE\n// EVOLVE-BLOCK-END";
    check_diff("verilog", original, patch, "q[21] ^ q[0]", "sv");
}

#[test]
fn test_apply_full_patch_supports_verilog_and_sv_fence() {
    let original = "// EVOLVE-BLOCK-START\nmodule TopModule (input a, output b);\n  assign b = a;\nendmodule\n// EVOLVE-BLOCK-END\n";
    let patch = "```sv\n// EVOLVE-BLOCK-START\nmodule TopModule (input a, output b);\n  assign b = ~a;\nendmodule\n// EVOLVE-BLOCK-END\n```";
    check_full("verilog", original, patch, "assign b = ~a;");
    check_full(
        "systemverilog",
        original,
        &patch.replace("```sv", "```verilog"),
        "assign b = ~a;",
    );
}

#[test]
fn test_apply_diff_patch_supports_wolfram() {
    let original = "(* EVOLVE-BLOCK-START *)\nf[n_Integer] := n\n(* EVOLVE-BLOCK-END *)";
    let patch = "(* EVOLVE-BLOCK-START *)\n<<<<<<< SEARCH\nf[n_Integer] := n\n=======\nf[n_Integer] := n (n + 1) / 2\n>>>>>>> REPLACE\n(* EVOLVE-BLOCK-END *)";
    check_diff("wolfram", original, patch, "n (n + 1) / 2", "wl");
}

#[test]
fn test_apply_full_patch_supports_wolfram_fences() {
    let original = "(* EVOLVE-BLOCK-START *)\nf[n_Integer] := n\n(* EVOLVE-BLOCK-END *)\n";
    let patch = "```wl\n(* EVOLVE-BLOCK-START *)\nf[n_Integer] := Fibonacci[n]\n(* EVOLVE-BLOCK-END *)\n```";
    check_full("wolfram", original, patch, "Fibonacci[n]");
    let patch = "```mathematica\n(* EVOLVE-BLOCK-START *)\nf[n_Integer] := 2^n - 1\n(* EVOLVE-BLOCK-END *)\n```";
    check_full("mathematica", original, patch, "2^n - 1");
}

#[test]
fn test_apply_full_patch_supports_wolfram_without_markers() {
    let original = "(* EVOLVE-BLOCK-START *)\nf[n_Integer] := n\n(* EVOLVE-BLOCK-END *)\n";
    let outcome =
        apply_full_patch("```wl\nf[n_Integer] := n + 1\n```", original, "wolfram").unwrap();
    assert_eq!(
        outcome.code,
        "(* EVOLVE-BLOCK-START *)\nf[n_Integer] := n + 1\n(* EVOLVE-BLOCK-END *)\n"
    );
}

#[test]
fn test_apply_diff_patch_supports_markdown() {
    let original =
        "<!-- EVOLVE-BLOCK-START -->\n## Step 1\n\nRun the old command.\n<!-- EVOLVE-BLOCK-END -->";
    let patch = "<!-- EVOLVE-BLOCK-START -->\n<<<<<<< SEARCH\nRun the old command.\n=======\nRun the new command.\n>>>>>>> REPLACE\n<!-- EVOLVE-BLOCK-END -->";
    check_diff("markdown", original, patch, "Run the new command.", "md");
}

#[test]
fn test_diff_patch_keeps_first_replacement_line_with_marker_prefix() {
    let original = "<!-- EVOLVE-BLOCK-START -->\nOld text\n<!-- EVOLVE-BLOCK-END -->\n";
    for quote in [">>>>>>> REPLACEMENT", ">>>>>>> REPLACE this"] {
        let replacement = format!("{quote}\nMore text");
        let patch = format!("<<<<<<< SEARCH\nOld text\n=======\n{replacement}\n>>>>>>> REPLACE");
        let outcome = apply_diff_patch(&patch, original, "markdown").unwrap();
        assert_eq!(outcome.code, original.replace("Old text", &replacement));
    }
}

#[test]
fn test_diff_patch_can_search_for_setext_heading_underline() {
    for indent in ["", "  "] {
        let underline = format!("{indent}=======");
        let replacement = format!("{indent}-------");
        let original = format!(
            "<!-- EVOLVE-BLOCK-START -->\nHeading\n{underline}\n<!-- EVOLVE-BLOCK-END -->\n"
        );
        let patch = format!("<<<<<<< SEARCH\n{underline}\n=======\n{replacement}\n>>>>>>> REPLACE");
        let outcome = apply_diff_patch(&patch, &original, "markdown").unwrap();
        assert_eq!(outcome.code, original.replace(&underline, &replacement));
    }
}

#[test]
fn test_apply_full_patch_supports_markdown_and_md_fence() {
    let original = "<!-- EVOLVE-BLOCK-START -->\n## Step 1\n\nRun the old command.\n<!-- EVOLVE-BLOCK-END -->\n";
    for fence in ["markdown", "md"] {
        let patch = format!(
            "```{fence}\n<!-- EVOLVE-BLOCK-START -->\n## Step 1\n\nRun the new command.\n<!-- EVOLVE-BLOCK-END -->\n```"
        );
        check_full("markdown", original, &patch, "Run the new command.");
    }
}

const CPP_ORIGINAL: &str =
    "#include <cstdio>\n// EVOLVE-BLOCK-START\nint solve() { return 1; }\n// EVOLVE-BLOCK-END\nint main() { return solve(); }\n";

#[test]
fn test_apply_full_patch_cpp_rewrite_is_not_corrupted() {
    // A model that returns just the block payload is the shape where a
    // mis-parsed fence corrupts the candidate.
    let rewritten = CPP_ORIGINAL.replace("return 1;", "return 2;");
    for (language, fence) in [
        ("c++", "cpp"),
        ("c++", "c++"),
        ("cpp", "c++"),
        ("cpp", "cxx"),
        ("c++", "cxx"),
    ] {
        let patch = format!("```{fence}\nint solve() {{ return 2; }}\n```");
        let outcome = apply_full_patch(&patch, CPP_ORIGINAL, language).unwrap();
        assert_eq!(outcome.code, rewritten, "{language} with ```{fence}");
    }
}

#[test]
fn languages_genesis_adds_patch_like_the_rest() {
    let original =
        "// EVOLVE-BLOCK-START\nconst f = (x) => x + 1;\n// EVOLVE-BLOCK-END\nexport default f;\n";
    let patch = "<<<<<<< SEARCH\nconst f = (x) => x + 1;\n=======\nconst f = (x) => x + 2;\n>>>>>>> REPLACE";
    check_diff("typescript", original, patch, "x + 2", "ts");
    check_full(
        "javascript",
        original,
        "```js\nconst f = (x) => x * 2;\n```",
        "x * 2",
    );
    let shell = "#!/bin/sh\n# EVOLVE-BLOCK-START\necho one\n# EVOLVE-BLOCK-END\n";
    check_full("bash", shell, "```sh\necho two\n```", "echo two");
    // An unknown language still patches; its markers are not validated.
    let unknown = "-- EVOLVE-BLOCK-START\nx = 1\n-- EVOLVE-BLOCK-END\n";
    let outcome = apply_diff_patch(
        "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE",
        unknown,
        "lua",
    )
    .unwrap();
    assert_eq!(outcome.code, unknown.replace("x = 1", "x = 2"));
    assert!(outcome.diff.starts_with("--- a/original.txt\n"));
}
