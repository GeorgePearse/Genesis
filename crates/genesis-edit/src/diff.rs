//! Applying SEARCH/REPLACE blocks to a program, inside its editable regions
//! only.

use crate::blocks::parse_blocks;
use crate::diagnostics::{self, MARKER_ONLY_SEARCH, NO_CHANGES_APPLIED};
use crate::markers::{
    boundary_violation, clean_evolve_markers, comment_openers, has_start_marker, inside,
    mutable_ranges, normalize_mutable_whitespace, validate_evolve_markers,
};
use crate::matching::{apply_indentation_to_replace, line_matches};
use crate::text::{is_blank, leading_whitespace, rstrip, strip_trailing_whitespace};
use crate::unified::unified_diff;
use crate::{PatchError, PatchOutcome};

/// Reject a SEARCH made only of markers when it names no usable place.
///
/// Removing the markers leaves an empty SEARCH, which appends to the last
/// editable region. That is where an END marker points in a file with one
/// region. A START marker, or any marker in a file with several regions,
/// points somewhere else.
fn reject_marker_only_search(patch_text: &str, original: &str) -> Result<(), PatchError> {
    let single_region = mutable_ranges(original).len() == 1;
    for block in parse_blocks(patch_text) {
        if is_blank(block.search) || !is_blank(&clean_evolve_markers(block.search)) {
            continue;
        }
        if single_region && !has_start_marker(block.search) {
            continue;
        }
        return Err(PatchError(MARKER_ONLY_SEARCH.to_string()));
    }
    Ok(())
}

/// Append `payload` to the editable region that ends at `end`, keeping the
/// END marker on a line of its own.
fn insert_before_end_marker(text: &str, (start, end): (usize, usize), payload: &str) -> String {
    let line_start = text[..end].rfind('\n').map_or(0, |newline| newline + 1);
    if line_start < start {
        // Both markers share a line: open the region up below the START marker.
        let existing = rstrip(&text[start..end]);
        let kept = if is_blank(existing) {
            String::new()
        } else {
            format!("{existing}\n")
        };
        return format!("{}\n{kept}{payload}\n{}", &text[..start], &text[end..]);
    }
    let before_marker = &text[line_start..end];
    if is_blank(before_marker) {
        return format!("{}{payload}\n{}", &text[..line_start], &text[line_start..]);
    }
    // Code precedes the marker on its line: append after that code and move
    // the marker to its own line at the same indentation.
    format!(
        "{}{}\n{payload}\n{}{}",
        &text[..line_start],
        rstrip(before_marker),
        leading_whitespace(before_marker),
        &text[end..]
    )
}

/// Apply every SEARCH/REPLACE block in `patch_text` to `original`, in order,
/// and return the new text with the number of blocks applied.
///
/// Editable ranges are recomputed after each block. An empty SEARCH appends
/// its REPLACE text to the last editable region. This is the matcher itself:
/// it neither normalises whitespace in `original` nor checks the markers of
/// the result, both of which [`apply_diff_patch`] does.
pub fn apply_search_replace(
    patch_text: &str,
    original: &str,
) -> Result<(String, usize), PatchError> {
    reject_marker_only_search(patch_text, original)?;
    let mut text = original.to_string();
    let mut applied = 0;
    for block in parse_blocks(patch_text) {
        let search = strip_trailing_whitespace(&clean_evolve_markers(block.search));
        let replace = strip_trailing_whitespace(&clean_evolve_markers(block.replace));
        let mutable = mutable_ranges(&text);

        if is_blank(&search) {
            let Some(last) = mutable.last() else {
                return Err(PatchError(diagnostics::no_evolve_block(&text, "insertion")));
            };
            if !replace.is_empty() {
                text = insert_before_end_marker(&text, *last, &replace);
            }
            applied += 1;
            continue;
        }

        // Region boundaries end a line, so code sharing a line with a marker
        // can be matched without it.
        let boundaries: Vec<usize> = mutable
            .iter()
            .flat_map(|(start, end)| [*start, *end])
            .collect();
        let matches = line_matches(&search, &text, &boundaries, &comment_openers(&text));
        let Some(chosen) = matches.iter().find(|m| inside((m.pos, m.end), &mutable)) else {
            let message = match matches.first() {
                Some(first) => diagnostics::outside_evolve_block(
                    &text[first.pos..first.end],
                    first.pos,
                    &text,
                    &mutable,
                ),
                None => diagnostics::search_not_found(&search, &text, &mutable),
            };
            return Err(PatchError(message));
        };
        let replace = match &chosen.indent {
            Some((offset, indent_chars)) => {
                apply_indentation_to_replace(&replace, *offset, indent_chars)
            }
            None => replace,
        };
        text = format!("{}{replace}{}", &text[..chosen.pos], &text[chosen.end..]);
        applied += 1;
    }
    Ok((text, applied))
}

/// Apply the SEARCH/REPLACE blocks found anywhere in a model's `response` to
/// `original`.
///
/// Trailing whitespace on editable lines is dropped first, since SEARCH text
/// is compared without it; read-only text is kept byte for byte. The
/// returned diff is against that normalised original. A response without a
/// single block is an error, as is any block that cannot be placed, and any
/// patch that would add or remove a marker or change read-only text.
pub fn apply_diff_patch(
    response: &str,
    original: &str,
    language: &str,
) -> Result<PatchOutcome, PatchError> {
    let original = normalize_mutable_whitespace(original);
    let raw_patch = strip_trailing_whitespace(response);
    let patch_text = clean_evolve_markers(&raw_patch);

    // Checked on the text as written: with the marker lines removed, a
    // marker-only SEARCH can no longer be told from an empty one.
    reject_marker_only_search(&raw_patch, &original)?;
    let (code, num_applied) = apply_search_replace(&patch_text, &original)?;
    if let Some(problem) =
        validate_evolve_markers(&code, language).or_else(|| boundary_violation(&original, &code))
    {
        return Err(PatchError(problem));
    }
    if num_applied == 0 {
        return Err(PatchError(NO_CHANGES_APPLIED.to_string()));
    }
    let diff = unified_diff(&original, &code, language);
    Ok(PatchOutcome {
        code,
        num_applied,
        diff,
        patch_text,
    })
}
