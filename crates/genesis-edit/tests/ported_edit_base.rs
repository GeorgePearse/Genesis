//! Behaviour cases ported from ShinkaEvolve's `tests/test_edit_base.py`,
//! `tests/test_edit_circle.py` and `tests/test_marker_validation.py`
//! (Apache-2.0, Sakana AI). Case names follow the originals.

use genesis_edit::{
    apply_diff_patch, apply_full_patch, apply_patch, PatchError, PatchOutcome, NO_CHANGES_APPLIED,
};

fn diff(patch: &str, original: &str) -> Result<PatchOutcome, PatchError> {
    apply_diff_patch(patch, original, "python")
}

fn full(patch: &str, original: &str) -> Result<PatchOutcome, PatchError> {
    apply_full_patch(patch, original, "python")
}

fn insertion(payload: &str) -> String {
    format!("<<<<<<< SEARCH\n\n=======\n{payload}\n>>>>>>> REPLACE")
}

const MNIST_FILE: &str = "# EVOLVE-BLOCK-START
def run_experiment(train_dataset, device):
    epochs = 5
    batch_size = 64
    learning_rate = 0.01
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
    return model


# EVOLVE-BLOCK-END
";

#[test]
fn test_edit() {
    let patch = "
<<<<<<< SEARCH
def run_experiment(train_dataset, device):
    epochs = 5
    batch_size = 64
    learning_rate = 0.01
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
    return model
=======
THIS IS A TEST
>>>>>>> REPLACE

<<<<<<< SEARCH
THIS IS A TEST
=======
THIS IS A TEST PART 2
>>>>>>> REPLACE
";
    let outcome = diff(patch, MNIST_FILE).unwrap();
    assert_eq!(
        outcome.code,
        "# EVOLVE-BLOCK-START\nTHIS IS A TEST PART 2\n\n\n# EVOLVE-BLOCK-END\n"
    );
    assert_eq!(outcome.num_applied, 2);
    assert!(outcome
        .diff
        .starts_with("--- a/original.py\n+++ b/original.py\n"));
    assert!(outcome.diff.contains("+THIS IS A TEST PART 2\n"));
}

#[test]
fn test_edit_circle() {
    let original = include_str!("fixtures/circle.py");
    let patch_1 = "
<<<<<<< SEARCH
import numpy as np


def construct_packing():
=======
import numpy as np

# Optional LP solver for radii (used if SciPy is available)
try:
    from scipy.optimize import linprog
except Exception:
    linprog = None


def construct_packing():
>>>>>>> REPLACE
</DIFF>


<NAME>dist_matrix_precompute_for_radii</NAME>
<DESCRIPTION>Speed up the radii computation by precomputing the pairwise distance matrix once.</DESCRIPTION>
<DIFF>
<<<<<<< SEARCH
    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)
    return centers, radii
=======
    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)
    return centers, radii
>>>>>>> REPLACE
";
    let outcome = diff(patch_1, original).unwrap();
    assert_eq!(outcome.num_applied, 2);
    assert!(outcome
        .code
        .contains("    linprog = None\n\n\ndef construct_packing():"));

    let patch_2 = "
<<<<<<< SEARCH
    # Then, limit by distance to other circles
    # Each pair of circles with centers at distance d can have
    # sum of radii at most d to avoid overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii
=======
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii
>>>>>>> REPLACE
";
    let outcome = diff(patch_2, original).unwrap();
    assert_eq!(outcome.num_applied, 1);
    assert!(outcome
        .code
        .contains("dist = np.linalg.norm(centers[i] - centers[j])"));
}

#[test]
fn test_apply_full_patch_single_evolve_block() {
    let original = "# Immutable header
import os

# EVOLVE-BLOCK-START
def old_function():
    return \"old\"
# EVOLVE-BLOCK-END

# Immutable footer
if __name__ == \"__main__\":
    pass
";
    let patch = "```python
# Immutable header
import os

# EVOLVE-BLOCK-START
def new_function():
    return \"new\"

def another_function():
    return \"another\"
# EVOLVE-BLOCK-END

# Immutable footer
if __name__ == \"__main__\":
    pass
```";
    let expected = "# Immutable header
import os

# EVOLVE-BLOCK-START
def new_function():
    return \"new\"

def another_function():
    return \"another\"
# EVOLVE-BLOCK-END

# Immutable footer
if __name__ == \"__main__\":
    pass
";
    let outcome = full(patch, original).unwrap();
    assert_eq!(outcome.num_applied, 1);
    assert_eq!(outcome.code, expected);
}

#[test]
fn test_apply_full_patch_with_evolve_blocks_in_patch() {
    let original = "# Header
# EVOLVE-BLOCK-START
def old_func1():
    pass
# EVOLVE-BLOCK-END

# Middle section
# EVOLVE-BLOCK-START
def old_func2():
    pass
# EVOLVE-BLOCK-END
# Footer
";
    let patch = "```python
# Header (the model changed this line; it must not land)
# EVOLVE-BLOCK-START
def new_func1():
    return 1
# EVOLVE-BLOCK-END

# Middle section
# EVOLVE-BLOCK-START
def new_func2():
    return 2
# EVOLVE-BLOCK-END
# Footer
```";
    let outcome = full(patch, original).unwrap();
    assert_eq!(outcome.num_applied, 1);
    assert_eq!(
        outcome.code,
        original
            .replace(
                "def old_func1():\n    pass",
                "def new_func1():\n    return 1"
            )
            .replace(
                "def old_func2():\n    pass",
                "def new_func2():\n    return 2"
            )
    );
}

#[test]
fn test_apply_full_patch_full_file_without_markers_extracts_block_only() {
    let original =
        "# Header line\n# EVOLVE-BLOCK-START\nold_line()\n# EVOLVE-BLOCK-END\n# Footer line\n";
    let outcome = full("```python\nnew_line()\nanother_new_line()\n```", original).unwrap();
    assert_eq!(
        outcome.code,
        "# Header line\n# EVOLVE-BLOCK-START\nnew_line()\nanother_new_line()\n# EVOLVE-BLOCK-END\n# Footer line\n"
    );

    // The same rewrite carrying the read-only header and footer.
    let whole_file = "```python\n# Header line\nnew_line()\nanother_new_line()\n# Footer line\n```";
    assert_eq!(full(whole_file, original).unwrap().code, outcome.code);
}

#[test]
fn test_apply_full_patch_patch_with_start_marker_only() {
    let original =
        "# Header line\n# EVOLVE-BLOCK-START\nold_line()\n# EVOLVE-BLOCK-END\n# Footer line\n";
    let patch = "```python\n# Header line\n# EVOLVE-BLOCK-START\nnew_line()\n# Footer line\n```";
    let outcome = full(patch, original).unwrap();
    assert_eq!(outcome.num_applied, 1);
    assert_eq!(
        outcome.code,
        "# Header line\n# EVOLVE-BLOCK-START\nnew_line()\n# EVOLVE-BLOCK-END\n# Footer line\n"
    );
}

#[test]
fn test_apply_full_patch_patch_with_end_marker_only() {
    let original =
        "# Header line\n# EVOLVE-BLOCK-START\nold_line()\n# EVOLVE-BLOCK-END\n# Footer line\n";
    let patch = "```python\n# Header line\nnew_line()\n# EVOLVE-BLOCK-END\n# Footer line\n```";
    let outcome = full(patch, original).unwrap();
    assert_eq!(
        outcome.code,
        "# Header line\n# EVOLVE-BLOCK-START\nnew_line()\n# EVOLVE-BLOCK-END\n# Footer line\n"
    );
}

#[test]
fn test_apply_full_patch_no_evolve_blocks() {
    let original = "# Just regular code\ndef function():\n    return \"no evolve blocks\"\n";
    let error = full(
        "```python\ndef new_function():\n    return \"new\"\n```",
        original,
    )
    .unwrap_err();
    assert_eq!(error.0, "No EVOLVE-BLOCK regions found in original content");
}

#[test]
fn test_apply_full_patch_multiple_evolve_blocks_ambiguous() {
    let original = "# EVOLVE-BLOCK-START\ndef func1():\n    pass\n# EVOLVE-BLOCK-END\n\n# EVOLVE-BLOCK-START\ndef func2():\n    pass\n# EVOLVE-BLOCK-END\n";
    let patch =
        "```python\ndef new_function():\n    return \"ambiguous which block to replace\"\n```";
    let error = full(patch, original).unwrap_err().0;
    assert!(error.contains("Multiple EVOLVE-BLOCK regions found"));
    assert!(error.contains("doesn't specify which to replace"));
}

#[test]
fn test_apply_full_patch_patch_with_single_marker_ambiguous_multiple_regions() {
    let original = "# Header\n# EVOLVE-BLOCK-START\nfunc1()\n# EVOLVE-BLOCK-END\n\n# EVOLVE-BLOCK-START\nfunc2()\n# EVOLVE-BLOCK-END\n# Footer\n";
    let patch = "```python\n# Header\n# EVOLVE-BLOCK-START\nnew_code()\n# Footer\n```";
    let error = full(patch, original).unwrap_err().0;
    assert!(error.contains("only one EVOLVE-BLOCK marker"), "{error}");
    assert!(error.contains("has 2 editable regions"));
}

#[test]
fn test_apply_full_patch_invalid_extraction() {
    let original = "# EVOLVE-BLOCK-START\ndef old_func():\n    pass\n# EVOLVE-BLOCK-END\n";
    let error = full("def new_function(): return 'no fences'", original).unwrap_err();
    assert_eq!(error.0, "Could not extract code from patch string");
}

#[test]
fn test_apply_full_patch_reports_what_was_applied() {
    // The Python case checks the files written to a patch directory:
    // rewrite.txt, original.py, main.py and edit.diff.
    let original =
        "# EVOLVE-BLOCK-START\ndef old_function():\n    return \"old\"\n# EVOLVE-BLOCK-END\n";
    let outcome = full(
        "```python\ndef new_function():\n    return \"new\"\n```",
        original,
    )
    .unwrap();
    assert_eq!(outcome.num_applied, 1);
    assert_eq!(
        outcome.patch_text,
        "def new_function():\n    return \"new\""
    );
    assert_eq!(
        outcome.code,
        "# EVOLVE-BLOCK-START\ndef new_function():\n    return \"new\"\n# EVOLVE-BLOCK-END\n"
    );
    assert_eq!(
        outcome.diff,
        "--- a/original.py\n+++ b/original.py\n@@ -1,4 +1,4 @@\n # EVOLVE-BLOCK-START\n-def old_function():\n-    return \"old\"\n+def new_function():\n+    return \"new\"\n # EVOLVE-BLOCK-END\n"
    );
}

const CALCULATE: &str = "# EVOLVE-BLOCK-START
def calculate():
    centers = compute_centers()
    radius = get_radius()
    area = math.pi * radius ** 2
    return area
# EVOLVE-BLOCK-END";

#[test]
fn test_indentation_correction_in_patch() {
    let patch = "<<<<<<< SEARCH
centers = compute_centers()
radius = get_radius()
=======
centers = compute_new_centers()
radius = get_new_radius()
>>>>>>> REPLACE";
    let outcome = diff(patch, CALCULATE).unwrap();
    assert_eq!(outcome.num_applied, 1);
    assert!(outcome
        .code
        .contains("    centers = compute_new_centers()\n    radius = get_new_radius()\n"));
}

#[test]
fn test_indentation_correction_multiline_patch() {
    let original = "# EVOLVE-BLOCK-START
def process_data():
    if condition:
        data = load_data()
        result = process(data)
        return result
    return None
# EVOLVE-BLOCK-END";
    let patch = "<<<<<<< SEARCH
if condition:
    data = load_data()
    result = process(data)
    return result
=======
if new_condition:
    data = load_new_data()
    result = new_process(data)
    return enhanced_result
>>>>>>> REPLACE";
    let outcome = diff(patch, original).unwrap();
    assert!(outcome
        .code
        .contains("    if new_condition:\n        data = load_new_data()\n"));
    assert!(outcome
        .code
        .contains("        return enhanced_result\n    return None\n"));
}

#[test]
fn test_indentation_correction_with_trailing_whitespace() {
    let original = "# EVOLVE-BLOCK-START\ndef func():\n    x = 1  \n    y = 2\t\n    return x + y\n# EVOLVE-BLOCK-END";
    let patch = "<<<<<<< SEARCH\nx = 1   \ny = 2\n=======\nx = 10  \ny = 20\n>>>>>>> REPLACE";
    let outcome = diff(patch, original).unwrap();
    assert!(outcome.code.contains("    x = 10\n    y = 20\n"));
    for line in outcome.code.split('\n') {
        assert_eq!(
            line,
            line.trim_end(),
            "line has trailing whitespace: {line:?}"
        );
    }
}

#[test]
fn test_indentation_correction_fails_gracefully() {
    let original = "# EVOLVE-BLOCK-START\ndef func():\n    x = 1\n    y = 2\n    return x + y\n# EVOLVE-BLOCK-END";
    let patch = "<<<<<<< SEARCH\nz = 3\nw = 4\n=======\nz = 30\nw = 40\n>>>>>>> REPLACE";
    assert!(diff(patch, original)
        .unwrap_err()
        .0
        .contains("SEARCH text not found"));
}

#[test]
fn test_mixed_indentation_styles() {
    let original = "# EVOLVE-BLOCK-START\ndef func():\n\tx = 1  # Tab indented\n\ty = 2  # Tab indented\n\treturn x + y\n# EVOLVE-BLOCK-END";
    let patch = "<<<<<<< SEARCH\nx = 1  # Tab indented\ny = 2  # Tab indented\n=======\nx = 10\ny = 20\n>>>>>>> REPLACE";
    let outcome = diff(patch, original).unwrap();
    assert!(outcome.code.contains("\tx = 10\n\ty = 20\n"));
}

#[test]
fn test_indentation_with_empty_lines_in_search() {
    let original = "# EVOLVE-BLOCK-START\ndef func():\n    x = 1\n    \n    y = 2\n    return x + y\n# EVOLVE-BLOCK-END";
    let patch = "<<<<<<< SEARCH\nx = 1\n\ny = 2\n=======\nx = 10\n\ny = 20\n>>>>>>> REPLACE";
    let outcome = diff(patch, original).unwrap();
    assert!(outcome.code.contains("    x = 10\n\n    y = 20\n"));
}

#[test]
fn test_indentation_correction_preserves_mutable_regions() {
    let original = "# Immutable section
def immutable_func():
    x = 1
    return x

# EVOLVE-BLOCK-START
def mutable_func():
    y = 2
    return y
# EVOLVE-BLOCK-END

# Another immutable section
def another_immutable():
    z = 3
    return z";
    let error = diff(
        "<<<<<<< SEARCH\nx = 1\n=======\nx = 100\n>>>>>>> REPLACE",
        original,
    )
    .unwrap_err()
    .0;
    assert!(error.contains("outside EVOLVE-BLOCK"), "{error}");
}

#[test]
fn test_insertion_with_indentation() {
    let original = "# EVOLVE-BLOCK-START\ndef func():\n    x = 1\n    return x\n# EVOLVE-BLOCK-END";
    let outcome = diff(&insertion("    # New comment\n    y = 2"), original).unwrap();
    assert_eq!(outcome.num_applied, 1);
    assert_eq!(
        outcome.code,
        "# EVOLVE-BLOCK-START\ndef func():\n    x = 1\n    return x\n    # New comment\n    y = 2\n# EVOLVE-BLOCK-END"
    );
}

#[test]
fn test_insertion_keeps_end_marker_on_own_line() {
    let original = "# EVOLVE-BLOCK-START\ndef f(x):\n    return x\n# EVOLVE-BLOCK-END";
    let outcome = diff(&insertion("def g(x):\n    return x * 2"), original).unwrap();
    assert!(outcome
        .code
        .contains("    return x * 2\n# EVOLVE-BLOCK-END"));
    assert_eq!(outcome.code.lines().last(), Some("# EVOLVE-BLOCK-END"));
}

#[test]
fn test_insertion_preserves_indented_end_marker() {
    let original =
        "def outer():\n    # EVOLVE-BLOCK-START\n    x = 1\n    # EVOLVE-BLOCK-END\n    return x";
    let outcome = diff(&insertion("    y = 2"), original).unwrap();
    assert!(outcome
        .code
        .contains("    x = 1\n    y = 2\n    # EVOLVE-BLOCK-END\n    return x"));
}

#[test]
fn test_consecutive_insertions_stay_valid_python() {
    let original = "# EVOLVE-BLOCK-START\ndef f(x):\n    return x\n# EVOLVE-BLOCK-END";
    let first = diff(&insertion("y = 2"), original).unwrap();
    let second = diff(&insertion("z = 3"), &first.code).unwrap();
    assert_eq!((first.num_applied, second.num_applied), (1, 1));
    assert_eq!(
        second.code,
        "# EVOLVE-BLOCK-START\ndef f(x):\n    return x\ny = 2\nz = 3\n# EVOLVE-BLOCK-END"
    );
}

#[test]
fn test_insertion_accepted_for_block_comment_language() {
    let original = "(* EVOLVE-BLOCK-START *)\nf[x_] := x\n(* EVOLVE-BLOCK-END *)";
    let outcome = apply_diff_patch(&insertion("g[x_] := x^3"), original, "wolfram").unwrap();
    assert!(outcome
        .code
        .contains("g[x_] := x^3\n(* EVOLVE-BLOCK-END *)"));
}

#[test]
fn test_insertion_appends_after_code_on_marker_line() {
    let original = "def outer():\n    # EVOLVE-BLOCK-START\n    existing = 1  # EVOLVE-BLOCK-END\n    return existing";
    let outcome = diff(&insertion("    inserted = 2"), original).unwrap();
    assert_eq!(
        outcome.code,
        "def outer():\n    # EVOLVE-BLOCK-START\n    existing = 1\n    inserted = 2\n    # EVOLVE-BLOCK-END\n    return existing"
    );
}

#[test]
fn test_insertion_between_markers_that_share_a_line() {
    let original = "head\n# EVOLVE-BLOCK-START # EVOLVE-BLOCK-END\ntail\n";
    let outcome = diff(&insertion("x = 1"), original).unwrap();
    assert_eq!(
        outcome.code,
        "head\n# EVOLVE-BLOCK-START \nx = 1\n# EVOLVE-BLOCK-END\ntail\n"
    );
}

#[test]
fn test_empty_insertion_is_noop() {
    let original = "# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END";
    let outcome = diff("<<<<<<< SEARCH\n\n=======\n\n>>>>>>> REPLACE", original).unwrap();
    assert_eq!((outcome.code.as_str(), outcome.num_applied), (original, 1));
    assert_eq!(outcome.diff, "");
}

#[test]
fn test_enhanced_search_not_found_error() {
    let patch = "<<<<<<< SEARCH\ncenters = compute_center()\n=======\ncenters = compute_new_centers()\n>>>>>>> REPLACE";
    let error = diff(patch, CALCULATE).unwrap_err().0;
    assert!(error.contains("SEARCH text not found"));
    assert!(
        error.contains("Looking for: \"centers = compute_center()\""),
        "{error}"
    );
    assert!(
        error.contains("Found similar code block at line 3 (✓ editable):"),
        "{error}"
    );
    assert!(
        error.contains("-  centers = compute_center()\n+      centers = compute_centers()"),
        "{error}"
    );
    assert!(error.contains("Editable regions where you can make changes:"));
}

#[test]
fn test_enhanced_evolve_block_violation_error() {
    let original = "# Immutable header
import os
import sys

# EVOLVE-BLOCK-START
def mutable_function():
    return \"editable\"
# EVOLVE-BLOCK-END

# Immutable footer
if __name__ == \"__main__\":
    main()";
    let patch = "<<<<<<< SEARCH\nimport os\n=======\nimport os\nimport json\n>>>>>>> REPLACE";
    let error = diff(patch, original).unwrap_err().0;
    for expected in [
        "Attempted to edit outside EVOLVE-BLOCK regions",
        "Context around found text:",
        " >>> Line   2: import os",
        "Available editable regions",
        "Suggestions:",
    ] {
        assert!(
            error.contains(expected),
            "missing {expected:?} in:\n{error}"
        );
    }
}

#[test]
fn test_enhanced_no_evolve_block_error() {
    let original = "def regular_function():\n    return \"no evolve blocks here\"\n\nif __name__ == \"__main__\":\n    print(\"Hello world\")";
    let error = diff(&insertion("# New comment\nnew_var = 42"), original)
        .unwrap_err()
        .0;
    for expected in [
        "Cannot perform insertion: No EVOLVE-BLOCK regions found",
        "Current file structure:",
        "Expected format:",
        "EVOLVE-BLOCK-START",
        "Suggestions:",
    ] {
        assert!(
            error.contains(expected),
            "missing {expected:?} in:\n{error}"
        );
    }
}

#[test]
fn test_enhanced_error_with_multiline_search() {
    let original = "# EVOLVE-BLOCK-START\ndef process():\n    data = load_data()\n    result = transform(data)\n    return result\n# EVOLVE-BLOCK-END";
    let patch = "<<<<<<< SEARCH
data = load_data()
result = transform_data(data)
return result
=======
data = load_new_data()
result = new_transform(data)
return result
>>>>>>> REPLACE";
    let error = diff(patch, original).unwrap_err().0;
    assert!(
        error.contains("Looking for 3-line block starting with: \"data = load_data()\""),
        "{error}"
    );
    assert!(
        error.contains("Found similar code block at line 3 (✓ editable):"),
        "{error}"
    );
}

#[test]
fn test_apply_diff_patch_reindents_nested_block() {
    let original = "# EVOLVE-BLOCK-START
class Solver:
    class Inner:
        def score(self, x):
            total = 0
            return total
# EVOLVE-BLOCK-END
";
    let patch = "<<<<<<< SEARCH
    def score(self, x):
        total = 0
        return total
=======
    def score(self, x):
        total = sum(x)
        return total
>>>>>>> REPLACE";
    let outcome = diff(patch, original).unwrap();
    assert_eq!(
        outcome.code,
        original.replace("total = 0", "total = sum(x)")
    );
}

const DEDENT: &str = "# EVOLVE-BLOCK-START
def f(a):
    if a:
        x()
        y()
    return 1
# EVOLVE-BLOCK-END
";

#[test]
fn test_apply_diff_patch_rejects_flattened_dedent_in_strict_mode() {
    let patch = "<<<<<<< SEARCH\n            x()\n        y()\n=======\n            x()\n        z()\n>>>>>>> REPLACE";
    assert!(diff(patch, DEDENT).is_err());
}

#[test]
fn test_apply_diff_patch_preserves_dedent_in_replacement() {
    let original = DEDENT.replace("        y()\n", "    y()\n");
    let patch = "<<<<<<< SEARCH\n            x()\n        y()\n=======\n            x()\n        z()\n>>>>>>> REPLACE";
    let outcome = diff(patch, &original).unwrap();
    assert!(outcome.code.contains("        x()\n    z()\n    return 1"));
}

#[test]
fn test_apply_diff_patch_keeps_replace_relative_to_search() {
    let original = DEDENT.replace("        y()\n", "");
    let patch = "<<<<<<< SEARCH\n            x()\n=======\n                if a > 1:\n                    x()\n>>>>>>> REPLACE";
    let outcome = diff(patch, &original).unwrap();
    assert!(outcome
        .code
        .contains("    if a:\n            if a > 1:\n                x()\n    return 1"));
}

#[test]
fn test_apply_full_patch_rejects_wolfram_smuggled_markers() {
    let original = "(* EVOLVE-BLOCK-START *)\nf[n_Integer] := n\n(* EVOLVE-BLOCK-END *)\n";
    let bad_patch = "```wolfram\n(* EVOLVE-BLOCK-START\nf[n_Integer] := n (n + 1) / 2\n EVOLVE-BLOCK-END *)\n```\n";
    let error = apply_full_patch(bad_patch, original, "wolfram")
        .unwrap_err()
        .0;
    assert!(error.contains("EVOLVE-BLOCK"));
    assert!(error.contains("malformed"));
}

#[test]
fn test_apply_full_patch_accepts_well_formed_wolfram() {
    let original = "(* EVOLVE-BLOCK-START *)\nf[n_Integer] := n\n(* EVOLVE-BLOCK-END *)\n";
    let good_patch = "```wolfram\n(* EVOLVE-BLOCK-START *)\nf[n_Integer] := n (n + 1) / 2\n(* EVOLVE-BLOCK-END *)\n```\n";
    let outcome = apply_full_patch(good_patch, original, "wolfram").unwrap();
    assert_eq!(
        outcome.code,
        original.replace(":= n\n", ":= n (n + 1) / 2\n")
    );
}

#[test]
fn test_apply_diff_patch_still_works_after_validation_hook() {
    let original = "# EVOLVE-BLOCK-START\ndef f(x):\n    return x + 1\n# EVOLVE-BLOCK-END\n";
    let patch = "<<<<<<< SEARCH\n    return x + 1\n=======\n    return x + 2\n>>>>>>> REPLACE\n";
    let outcome = diff(patch, original).unwrap();
    assert_eq!(outcome.code, original.replace("x + 1", "x + 2"));
}

#[test]
fn a_response_without_blocks_changes_nothing() {
    let original = "# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n";
    assert_eq!(
        diff("I would change x to 2.", original).unwrap_err().0,
        NO_CHANGES_APPLIED
    );
    assert_eq!(diff("", original).unwrap_err().0, "No changes applied");
    // A file that has lost its markers is reported as such first.
    assert!(diff("nothing", "x = 1\n")
        .unwrap_err()
        .0
        .contains("No EVOLVE-BLOCK-START marker found"));
}

#[test]
fn apply_patch_dispatches_on_the_patch_type() {
    let original = "# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n";
    let block = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE";
    let fence = "```python\nx = 2\n```";
    let expected = original.replace("x = 1", "x = 2");
    assert_eq!(
        apply_patch("diff", block, original, "python").unwrap().code,
        expected
    );
    for patch_type in ["full", "cross", "fix"] {
        assert_eq!(
            apply_patch(patch_type, fence, original, "python")
                .unwrap()
                .code,
            expected
        );
    }
    assert_eq!(
        apply_patch("merge", fence, original, "python")
            .unwrap_err()
            .0,
        "Unknown patch type: merge"
    );
}
