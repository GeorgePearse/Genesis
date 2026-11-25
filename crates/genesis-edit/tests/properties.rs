//! Property tests for genesis-edit, run with ephyra
//! (<https://github.com/datadog-labs/ephyra>).
//!
//! What "correct" means here, and where each expectation comes from:
//!
//! | Property | Oracle | Source |
//! |---|---|---|
//! | A patch that applies leaves every byte outside the editable regions as it was, and the same markers | invariant | the prompt contract: "Everything outside those markers is read-only", "Keep the markers ... in the code" |
//! | A SEARCH/REPLACE patch built from an original and a target that differ only inside regions yields the target | round trip | "Every block's SEARCH section must be copied verbatim from the current file" |
//! | A full rewrite carrying the target yields the target, whatever it says outside the regions | round trip | "Provide the complete new program code"; "Do not change the code outside of these markers" |
//! | A rewrite of a one-region program that leaves out one or both markers still replaces exactly the region | round trip | `apply_full.py`: "treat the middle part as the evolve-block replacement", "Normalize newlines so markers remain on their own lines" |
//! | A SEARCH that names only read-only code is refused | invariant | `apply_diff.py`: "a block that names immutable code as written must not fall back to an editable look-alike" |
//! | When the lines a block names occur several times, the first run of them inside an editable region changes, and only that | reference model | `apply_search_replace`: "take the first match that lies in an editable region"; `test_match_inside_the_block_wins_over_an_earlier_one_outside` |
//! | An empty SEARCH appends to the last editable region | reference model | the refusal text itself: "To add code at the end of the editable region, leave the SEARCH block empty" |
//! | The brace-language metrics equal a recount by splitting the text into words | reference model | `analyze_cpp_complexity`: decision keywords, `?` between word characters, brace depth |
//! | Several edits in one response equal the same edits one response at a time | metamorphic | "the changes you propose will be applied sequentially" |
//! | A block shifted to another indentation applies, and its replacement lands at the file's indentation | round trip | `_iter_line_matches` docstring; `test_apply_diff_patch_reindents_nested_block` |
//! | A program wrapped in a region and then rewritten is the rewrite, wrapped | round trip | the seed-generation path of the runner |
//! | A unified diff rebuilds both files, is empty for equal files, and its summary counts its lines | reference model | definition of a unified diff |
//! | No response, original or language makes an applier or the metrics panic | crash | all three read model-written text |
//! | The read-only invariant also holds for files whose markers are oddly placed: sharing a line with code or with each other, nested, bare, unmatched, with CRLF line ends | invariant | `_mutable_ranges` defines the regions for any text, so the contract covers any text |

use ephyra::prelude::*;
use ephyra::{gens, BoxGen};
use genesis_edit::{
    analyze_code_metrics, apply_diff_patch, apply_full_patch, fence_tags, immutable_parts,
    mutable_ranges, redact_immutable, summarize_diff, unified_diff, validate_evolve_markers,
    wrap_in_evolve_block, PatchOutcome,
};
use std::ops::ControlFlow;

const START: &str = "EVOLVE-BLOCK-START";
const END: &str = "EVOLVE-BLOCK-END";

/// Language, comment opener, comment closer.
const STYLES: &[(&str, &str, &str)] = &[
    ("python", "#", ""),
    ("rust", "//", ""),
    ("fortran", "!", ""),
    ("markdown", "<!--", "-->"),
    ("wolfram", "(*", "*)"),
];

const NAMES: &[&str] = &[
    "x", "y", "total", "best", "idx", "lr", "radius", "état", "值",
];

/// Python's `str.rstrip`: it also strips the information separators.
fn rstrip(line: &str) -> &str {
    line.trim_end_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

fn comment(style: usize, text: &str) -> String {
    let (_, open, close) = STYLES[style];
    if close.is_empty() {
        format!("{open} {text}")
    } else {
        format!("{open} {text} {close}")
    }
}

/// One editable region and the read-only lines that follow it.
#[derive(Debug, Clone, PartialEq)]
struct Region {
    /// Lines between the markers. The first names the region and is unique
    /// in the file; the first and last are never blank.
    body: Vec<String>,
    after: Vec<String>,
}

/// A program that is valid by construction: read-only head, then one to
/// three regions whose markers sit on lines of their own.
#[derive(Debug, Clone, PartialEq)]
struct Program {
    style: usize,
    marker_indent: String,
    head: Vec<String>,
    regions: Vec<Region>,
    final_newline: bool,
}

impl Program {
    fn language(&self) -> &'static str {
        STYLES[self.style].0
    }

    fn marker(&self, which: &str) -> String {
        format!("{}{}", self.marker_indent, comment(self.style, which))
    }

    fn render_bodies(&self, bodies: &[Vec<String>]) -> String {
        let mut lines: Vec<String> = self.head.clone();
        for (region, body) in self.regions.iter().zip(bodies) {
            lines.push(self.marker(START));
            lines.extend(body.iter().cloned());
            lines.push(self.marker(END));
            lines.extend(region.after.iter().cloned());
        }
        let mut text = lines.join("\n");
        if self.final_newline {
            text.push('\n');
        }
        text
    }

    fn bodies(&self) -> Vec<Vec<String>> {
        self.regions
            .iter()
            .map(|region| region.body.clone())
            .collect()
    }

    /// Bodies as the diff applier sees them: no trailing whitespace.
    fn clean_bodies(&self) -> Vec<Vec<String>> {
        self.regions
            .iter()
            .map(|region| {
                region
                    .body
                    .iter()
                    .map(|line| rstrip(line).to_string())
                    .collect()
            })
            .collect()
    }

    fn render(&self) -> String {
        self.render_bodies(&self.bodies())
    }

    /// The read-only text, from how the program was built rather than from
    /// the crate's own reading of the markers.
    fn read_only_parts(&self) -> Vec<String> {
        let start = format!("{}{}", self.marker_indent, comment(self.style, START));
        let end = comment(self.style, END);
        let mut parts = Vec::new();
        let mut current: Vec<String> = self.head.clone();
        for region in &self.regions {
            current.push(start.clone());
            parts.push(current.join("\n"));
            current = vec![end.clone()];
            current.extend(region.after.iter().cloned());
        }
        let mut last = current.join("\n");
        if self.final_newline {
            last.push('\n');
        }
        parts.push(last);
        parts
    }

    fn read_only_lines(&self) -> Vec<&String> {
        self.head
            .iter()
            .chain(self.regions.iter().flat_map(|region| region.after.iter()))
            .collect()
    }
}

fn code_line(g: &mut Source, style: usize) -> String {
    let indent = *g.weighted(&[(4, ""), (3, "    "), (2, "        "), (1, "\t"), (1, "  ")]);
    let (name, other, n) = (*g.pick(NAMES), *g.pick(NAMES), g.int(0u32..=9));
    let body = match g.int(0u8..=11) {
        0..=2 => format!("{name} = {n}"),
        3 => format!("{name} = {other} + {n}"),
        4 => format!("{name}({other})"),
        5 => format!("return {name}"),
        6 => comment(style, &format!("note {n}")),
        7 => format!("{name} = {n}  {}", comment(style, "tuned")),
        8 => format!("s = \"{other}\u{c}{n}\""),
        9 => (*g.pick(&["}", "end", "pass", "else:", "{", "a?b:c", "-------"])).to_string(),
        10 => format!("{name} = [{n}, {other}]"),
        _ => format!("if {name} > {n}:"),
    };
    format!("{indent}{body}")
}

fn trailing_noise(g: &mut Source, line: String) -> String {
    if g.bool_with(0.2) {
        format!("{line}{}", g.pick(&[" ", "\t", "  ", "\u{1c}", " \u{c}"]))
    } else {
        line
    }
}

/// Body lines: a unique first line, optional blank lines in the middle, a
/// non-blank last line. `fences` allows Markdown code fences inside.
fn body_lines(g: &mut Source, style: usize, tag: &str, fences: bool) -> Vec<String> {
    let indent = *g.pick(&["", "    "]);
    let mut lines = vec![format!("{indent}{tag} = {}", g.int(0u32..=99))];
    for _ in 0..g.int(0usize..=4) {
        let line = match g.int(0u8..=9) {
            0 => String::new(),
            1 if fences => (*g.pick(&["```", "```bash", "    ```"])).to_string(),
            _ => code_line(g, style),
        };
        lines.push(line);
    }
    if lines.last().is_some_and(|line| rstrip(line).is_empty()) {
        lines.push(code_line(g, style));
    }
    lines
}

fn program_with(g: &mut Source, fences: bool) -> Program {
    let style = g.index(STYLES.len());
    let region_count = *g.weighted(&[(5, 1usize), (3, 2), (2, 3)]);
    let read_only = |g: &mut Source| -> Vec<String> {
        (0..g.int(0usize..=3))
            .map(|_| {
                let line = if g.bool_with(0.15) {
                    String::new()
                } else {
                    code_line(g, style)
                };
                trailing_noise(g, line)
            })
            .collect()
    };
    let head = read_only(g);
    let mut regions = Vec::new();
    for index in 0..region_count {
        let body = body_lines(g, style, &format!("region_{index}"), fences)
            .into_iter()
            .map(|line| trailing_noise(g, line))
            .collect();
        regions.push(Region {
            body,
            after: read_only(g),
        });
    }
    let mut program = Program {
        style,
        marker_indent: (*g.weighted(&[(3, ""), (1, "    ")])).to_string(),
        head,
        regions,
        final_newline: g.bool_with(0.8),
    };
    // Read-only code that repeats an editable line: the editable one must win.
    if g.bool_with(0.3) {
        let region = g.index(program.regions.len());
        let line = g.pick(&program.regions[region].body).clone();
        program.head.push(line);
    }
    program
}

fn shrink_lines(
    lines: &[String],
    keep: usize,
    visit: &mut dyn FnMut(Vec<String>) -> ControlFlow<()>,
) -> ControlFlow<()> {
    for index in keep..lines.len() {
        let mut fewer = lines.to_vec();
        fewer.remove(index);
        visit(fewer)?;
    }
    ControlFlow::Continue(())
}

/// Smaller programs: fewer regions, fewer lines, no trailing whitespace.
fn shrink_program(
    program: &Program,
    visit: &mut dyn FnMut(Program) -> ControlFlow<()>,
) -> ControlFlow<()> {
    if program.regions.len() > 1 {
        for index in 0..program.regions.len() {
            let mut fewer = program.clone();
            fewer.regions.remove(index);
            visit(fewer)?;
        }
    }
    shrink_lines(&program.head, 0, &mut |head| {
        visit(Program {
            head,
            ..program.clone()
        })
    })?;
    for index in 0..program.regions.len() {
        let with = |change: &dyn Fn(&mut Region)| {
            let mut smaller = program.clone();
            change(&mut smaller.regions[index]);
            smaller
        };
        shrink_lines(&program.regions[index].after, 0, &mut |after| {
            visit(with(&|region| region.after = after.clone()))
        })?;
        // The first body line names the region; a body may not end blank.
        shrink_lines(&program.regions[index].body, 1, &mut |body| {
            if body.last().is_some_and(|line| rstrip(line).is_empty()) {
                return ControlFlow::Continue(());
            }
            visit(with(&|region| region.body = body.clone()))
        })?;
    }
    if !program.marker_indent.is_empty() {
        visit(Program {
            marker_indent: String::new(),
            ..program.clone()
        })?;
    }
    let mut tidy = program.clone();
    for line in tidy.head.iter_mut().chain(
        tidy.regions
            .iter_mut()
            .flat_map(|r| r.body.iter_mut().chain(&mut r.after)),
    ) {
        *line = rstrip(line).to_string();
    }
    if tidy != *program {
        visit(tidy)?;
    }
    ControlFlow::Continue(())
}

/// A new body for one region.
#[derive(Debug, Clone, PartialEq)]
struct Edit {
    region: usize,
    body: Vec<String>,
}

fn edits_for(g: &mut Source, program: &Program, fences: bool) -> Vec<Edit> {
    let mut regions: Vec<usize> = (0..program.regions.len()).collect();
    // A random order: blocks need not follow the file.
    for index in (1..regions.len()).rev() {
        regions.swap(index, g.index(index + 1));
    }
    regions.truncate(g.int(1..=regions.len()));
    regions
        .into_iter()
        .map(|region| Edit {
            region,
            body: body_lines(g, program.style, &format!("region_{region}_new"), fences),
        })
        .collect()
}

fn valid_edits(program: &Program, edits: &[Edit]) -> Vec<Edit> {
    let mut seen = Vec::new();
    edits
        .iter()
        .filter(|edit| edit.region < program.regions.len())
        .filter(|edit| {
            let fresh = !seen.contains(&edit.region);
            seen.push(edit.region);
            fresh
        })
        .cloned()
        .collect()
}

/// How a model dresses up its blocks.
#[derive(Debug, Clone, PartialEq, Default)]
struct Dressing {
    tags: bool,
    markers_around: bool,
    crlf: bool,
    trailing_spaces: bool,
    chatter: bool,
}

fn dressing(g: &mut Source) -> Dressing {
    Dressing {
        tags: g.bool_with(0.3),
        markers_around: g.bool_with(0.2),
        crlf: g.bool_with(0.1),
        trailing_spaces: g.bool_with(0.2),
        chatter: g.bool_with(0.3),
    }
}

fn block(search: &[String], replace: &[String], dressing: &Dressing) -> String {
    let pad = if dressing.trailing_spaces { "  " } else { "" };
    let mut lines = vec![format!("<<<<<<< SEARCH{pad}")];
    lines.extend(search.iter().map(|line| format!("{line}{pad}")));
    lines.push(format!("======={pad}"));
    lines.extend(replace.iter().map(|line| format!("{line}{pad}")));
    lines.push(format!(">>>>>>> REPLACE{pad}"));
    lines.join("\n")
}

fn diff_response(program: &Program, blocks: &[String], dressing: &Dressing) -> String {
    let mut text = blocks.join(if dressing.chatter {
        "\n\nNext change:\n"
    } else {
        "\n\n"
    });
    if dressing.markers_around {
        text = format!(
            "{}\n{text}\n{}",
            comment(program.style, START),
            comment(program.style, END)
        );
    }
    if dressing.tags {
        text = format!("<NAME>\nedit\n</NAME>\n\n<DESCRIPTION>\nwhy\n</DESCRIPTION>\n\n<DIFF>\n{text}\n\n</DIFF>");
    }
    if dressing.chatter {
        text = format!("Here is my proposal.\n\n{text}\n\nThat should score higher.");
    }
    if dressing.crlf {
        text = text.replace('\n', "\r\n");
    }
    text
}

#[derive(Debug, Clone, PartialEq)]
struct EditCase {
    program: Program,
    edits: Vec<Edit>,
    dressing: Dressing,
}

fn shrink_edit_case(
    case: &EditCase,
    visit: &mut dyn FnMut(EditCase) -> ControlFlow<()>,
) -> ControlFlow<()> {
    if case.edits.len() > 1 {
        for index in 0..case.edits.len() {
            let mut edits = case.edits.clone();
            edits.remove(index);
            visit(EditCase {
                edits,
                ..case.clone()
            })?;
        }
    }
    shrink_program(&case.program, &mut |program| {
        // Dropping a region renumbers the ones after it; keep edits that still fit.
        let edits = valid_edits(&program, &case.edits);
        if edits.is_empty() {
            return ControlFlow::Continue(());
        }
        visit(EditCase {
            program,
            edits,
            dressing: case.dressing.clone(),
        })
    })?;
    for index in 0..case.edits.len() {
        shrink_lines(&case.edits[index].body, 1, &mut |body| {
            if body.last().is_some_and(|line| rstrip(line).is_empty()) {
                return ControlFlow::Continue(());
            }
            let mut edits = case.edits.clone();
            edits[index].body = body;
            visit(EditCase {
                edits,
                ..case.clone()
            })
        })?;
    }
    if case.dressing != Dressing::default() {
        visit(EditCase {
            dressing: Dressing::default(),
            ..case.clone()
        })?;
    }
    ControlFlow::Continue(())
}

fn edit_cases(fences: bool) -> BoxGen<EditCase> {
    (move |g: &mut Source| {
        let program = program_with(g, fences);
        let edits = edits_for(g, &program, fences);
        EditCase {
            program,
            edits,
            dressing: dressing(g),
        }
    })
    .with_shrink(shrink_edit_case)
    .boxed()
}

fn marker_counts(text: &str) -> (usize, usize) {
    (text.matches(START).count(), text.matches(END).count())
}

/// The boundary check is a backstop for patches that mention markers. In a
/// program whose markers sit on lines of their own, blocks that mention none
/// cannot reach one, so if the check fires for them the applier itself moved
/// read-only text.
///
/// This does not hold for nested regions whose inner marker shares a line
/// with code: there a block can name that line, or the lines around it, and
/// delete the marker without spelling it (EPHYRA_REPLAY=7:412 against an
/// earlier version of the property below). Refusing such a patch is right.
fn assert_not_refused_for_the_appliers_own_doing(
    result: &Result<PatchOutcome, genesis_edit::PatchError>,
    blocks: &[(Vec<String>, Vec<String>)],
) {
    let mentions_markers = blocks.iter().any(|(search, replace)| {
        search
            .iter()
            .chain(replace)
            .any(|line| line.contains("EVOLVE-BLOCK"))
    });
    if let Err(error) = result {
        assert!(
            mentions_markers
                || !error
                    .0
                    .starts_with("Patch changes the EVOLVE-BLOCK markers or read-only code"),
            "a patch that mentions no marker was refused for moving the boundary:\n{error}"
        );
    }
}

/// What every applied patch must leave alone.
fn assert_read_only_text_is_untouched(program: &Program, original: &str, outcome: &PatchOutcome) {
    assert_eq!(
        immutable_parts(original),
        program.read_only_parts(),
        "the crate reads the original's regions differently from how it was built"
    );
    assert_eq!(
        immutable_parts(&outcome.code),
        program.read_only_parts(),
        "read-only text changed"
    );
    assert_eq!(
        marker_counts(&outcome.code),
        marker_counts(original),
        "markers were added or removed"
    );
    assert_eq!(
        validate_evolve_markers(&outcome.code, program.language()),
        None
    );
}

#[ephyra::test(gen = edit_cases(true))]
fn a_diff_built_from_a_target_yields_the_target(case: EditCase) {
    let EditCase {
        program,
        edits,
        dressing,
    } = case;
    let original = program.render();
    let mut bodies = program.clean_bodies();
    let mut blocks = Vec::new();
    for edit in &edits {
        blocks.push(block(&bodies[edit.region], &edit.body, &dressing));
        bodies[edit.region] = edit.body.clone();
    }
    let response = diff_response(&program, &blocks, &dressing);
    ephyra::cover(edits.len() > 1, "several blocks");
    ephyra::cover(dressing.crlf, "CRLF response");

    let outcome = apply_diff_patch(&response, &original, program.language())
        .unwrap_or_else(|error| panic!("a valid patch was refused:\n{error}"));
    assert_eq!(outcome.code, program.render_bodies(&bodies));
    assert_eq!(outcome.num_applied, edits.len());
    assert_read_only_text_is_untouched(&program, &original, &outcome);
    assert_eq!(
        outcome.diff.is_empty(),
        outcome.code == program.render_bodies(&program.clean_bodies())
    );
    assert!(
        outcome
            .patch_text
            .split('\n')
            .all(|line| line == rstrip(line)),
        "{:?}",
        outcome.patch_text
    );
    assert!(!outcome.patch_text.contains(START) && !outcome.patch_text.contains(END));
}

#[ephyra::test(gen = edit_cases(true))]
fn edits_in_one_response_equal_edits_one_at_a_time(case: EditCase) {
    let EditCase {
        program,
        edits,
        dressing,
    } = case;
    let original = program.render();
    let clean = program.clean_bodies();
    let blocks: Vec<String> = edits
        .iter()
        .map(|edit| block(&clean[edit.region], &edit.body, &dressing))
        .collect();

    let together = apply_diff_patch(
        &diff_response(&program, &blocks, &dressing),
        &original,
        program.language(),
    )
    .unwrap_or_else(|error| panic!("the combined patch was refused:\n{error}"));
    let mut current = original.clone();
    for single in &blocks {
        let response = diff_response(&program, std::slice::from_ref(single), &dressing);
        current = apply_diff_patch(&response, &current, program.language())
            .unwrap_or_else(|error| panic!("a single block was refused:\n{error}"))
            .code;
    }
    assert_eq!(together.code, current);
}

/// A full rewrite: the target program in a fence, with the read-only text
/// optionally rewritten too (which must not land).
#[derive(Debug, Clone, PartialEq)]
struct RewriteCase {
    case: EditCase,
    tag: String,
    rewrite_read_only: bool,
}

fn rewrite_cases() -> BoxGen<RewriteCase> {
    (|g: &mut Source| {
        let program = program_with(g, true);
        let edits = edits_for(g, &program, true);
        let tags = fence_tags(program.language());
        let mut tag = g.pick(&tags).clone();
        if g.bool_with(0.1) {
            tag = tag.to_uppercase();
        }
        RewriteCase {
            case: EditCase {
                program,
                edits,
                dressing: dressing(g),
            },
            tag,
            rewrite_read_only: g.bool_with(0.3),
        }
    })
    .with_shrink(|rewrite: &RewriteCase, visit| {
        shrink_edit_case(&rewrite.case, &mut |case| {
            visit(RewriteCase {
                case,
                ..rewrite.clone()
            })
        })?;
        if rewrite.rewrite_read_only {
            visit(RewriteCase {
                rewrite_read_only: false,
                ..rewrite.clone()
            })?;
        }
        ControlFlow::Continue(())
    })
    .boxed()
}

#[ephyra::test(gen = rewrite_cases())]
fn a_full_rewrite_carrying_a_target_yields_the_target(rewrite: RewriteCase) {
    let RewriteCase {
        case: EditCase {
            program,
            edits,
            dressing,
        },
        tag,
        rewrite_read_only,
    } = rewrite;
    let original = program.render();
    let mut bodies = program.bodies();
    for edit in &edits {
        bodies[edit.region] = edit.body.clone();
    }
    let target = program.render_bodies(&bodies);
    let sent = if rewrite_read_only {
        let mut other = program.clone();
        other.head = vec!["import something_else".to_string()];
        for region in &mut other.regions {
            region.after = vec!["print('changed')".to_string()];
        }
        other.final_newline = false;
        other.render_bodies(&bodies)
    } else {
        target.clone()
    };
    let mut response = format!("```{tag}\n{sent}\n```");
    if dressing.chatter {
        response = format!("<NAME>\nrewrite\n</NAME>\n\n<CODE>\n{response}\n</CODE>\n\nDone.");
    }
    ephyra::cover(
        bodies.iter().flatten().any(|line| line.contains("```")),
        "a code fence inside a region",
    );

    let outcome = apply_full_patch(&response, &original, program.language())
        .unwrap_or_else(|error| panic!("a valid rewrite was refused:\n{error}"));
    assert_eq!(outcome.code, target);
    assert_eq!(outcome.num_applied, 1);
    assert_read_only_text_is_untouched(&program, &original, &outcome);
}

#[ephyra::test(gen = edit_cases(false))]
fn a_search_that_names_only_read_only_code_is_refused(case: EditCase) {
    let EditCase {
        program,
        edits,
        dressing,
    } = case;
    let editable: Vec<String> = program.clean_bodies().concat();
    let candidates: Vec<&String> = program
        .read_only_lines()
        .into_iter()
        .filter(|line| {
            !rstrip(line).is_empty() && !editable.iter().any(|body| body == rstrip(line))
        })
        .collect();
    ephyra::cover(!candidates.is_empty(), "has a read-only line to aim at");
    let Some(target) = candidates.get(edits.len() % candidates.len().max(1)) else {
        return;
    };
    let response = diff_response(
        &program,
        &[block(
            &[rstrip(target).to_string()],
            &edits[0].body,
            &dressing,
        )],
        &dressing,
    );
    let error = apply_diff_patch(&response, &program.render(), program.language())
        .expect_err("an edit to read-only code was applied");
    assert!(
        error
            .0
            .contains("Attempted to edit outside EVOLVE-BLOCK regions"),
        "{error}"
    );
}

/// An indented block, and a patch for it written at another indentation.
#[derive(Debug, Clone, PartialEq)]
struct IndentCase {
    style: usize,
    /// Indentation of the block's first line in the file.
    base: usize,
    /// Each line's depth relative to the first, in units of four spaces.
    old: Vec<(isize, String)>,
    new: Vec<(isize, String)>,
    /// How far the patch is shifted from the file, in spaces.
    shift: isize,
    tabs: bool,
}

fn indent_cases() -> BoxGen<IndentCase> {
    (|g: &mut Source| {
        let style = g.index(STYLES.len());
        let tabs = g.bool_with(0.15);
        let base = if tabs { 1 } else { *g.pick(&[4usize, 8, 12]) };
        let lines = |g: &mut Source, tag: &str| -> Vec<(isize, String)> {
            let mut lines = vec![(0, format!("{tag} = {}", g.int(0u32..=99)))];
            for _ in 0..g.int(0usize..=4) {
                let depth = if tabs { 0 } else { g.int(-1isize..=2) };
                let text = if g.bool_with(0.1) {
                    String::new()
                } else {
                    code_line(g, style).trim_start().to_string()
                };
                lines.push((depth, text));
            }
            if lines.last().is_some_and(|(_, text)| text.is_empty()) {
                lines.push((0, "done = True".to_string()));
            }
            lines
        };
        let (old, new) = (lines(g, "block_old"), lines(g, "block_new"));
        let lowest = old
            .iter()
            .chain(&new)
            .map(|(depth, _)| base as isize + 4 * depth)
            .min()
            .unwrap_or(0);
        let wanted = if tabs {
            -1
        } else {
            *g.pick(&[-12isize, -8, -4, 4, 8])
        };
        // Never shift a line left of column zero, and never by nothing.
        let shift = if wanted.max(-lowest) == 0 {
            4
        } else {
            wanted.max(-lowest)
        };
        IndentCase {
            style,
            base,
            old,
            new,
            shift,
            tabs,
        }
    })
    .boxed()
}

fn indented(lines: &[(isize, String)], base: usize, shift: isize, tabs: bool) -> Vec<String> {
    lines
        .iter()
        .map(|(depth, text)| {
            if text.is_empty() {
                return String::new();
            }
            let width = (base as isize + 4 * depth + shift) as usize;
            format!(
                "{}{text}",
                if tabs {
                    "\t".repeat(width)
                } else {
                    " ".repeat(width)
                }
            )
        })
        .collect()
}

#[ephyra::test(gen = indent_cases())]
fn a_block_at_another_indentation_lands_at_the_files_indentation(case: IndentCase) {
    let file = |body: &[(isize, String)]| {
        let mut lines = vec!["def outer():".to_string(), comment(case.style, START)];
        lines.extend(indented(body, case.base, 0, case.tabs));
        lines.push(comment(case.style, END));
        lines.push("outer()".to_string());
        lines.join("\n")
    };
    let patch = block(
        &indented(&case.old, case.base, case.shift, case.tabs),
        &indented(&case.new, case.base, case.shift, case.tabs),
        &Dressing::default(),
    );
    let outcome = apply_diff_patch(&patch, &file(&case.old), STYLES[case.style].0);
    let outcome =
        outcome.unwrap_or_else(|error| panic!("a re-indented patch was refused:\n{error}"));
    assert_eq!(outcome.code, file(&case.new));
}

fn plain_code() -> BoxGen<Vec<String>> {
    (|g: &mut Source| {
        let style = g.index(STYLES.len());
        let mut lines: Vec<String> = (0..g.int(1usize..=5))
            .map(|_| code_line(g, style))
            .collect();
        lines[0] = lines[0].trim_start().to_string();
        lines
    })
    .boxed()
}

#[ephyra::test(gen = (plain_code(), plain_code(), gens::int(0usize..=4)))]
fn a_wrapped_program_rewritten_is_the_rewrite_wrapped(input: (Vec<String>, Vec<String>, usize)) {
    let (seed, rewrite, style) = input;
    let language = STYLES[style].0;
    let seed = seed.join("\n");
    // A fence body is trimmed, so the comparison is on trimmed code.
    let rewrite = rewrite.join("\n").trim().to_string();
    let wrapped = wrap_in_evolve_block(&seed, language);
    assert_eq!(validate_evolve_markers(&wrapped, language), None);
    let ranges = mutable_ranges(&wrapped);
    assert_eq!(ranges.len(), 1);
    assert_eq!(&wrapped[ranges[0].0..ranges[0].1], format!("\n{seed}\n"));

    let outcome = apply_full_patch(
        &format!("```{language}\n{rewrite}\n```"),
        &wrapped,
        language,
    )
    .unwrap_or_else(|error| panic!("a marker-free rewrite was refused:\n{error}"));
    assert_eq!(outcome.code, wrap_in_evolve_block(&rewrite, language));
    let editable = redact_immutable(&outcome.code);
    assert!(editable.contains(&rewrite) && !editable.contains(START) && !editable.contains(END));
}

fn short_lines() -> BoxGen<Vec<String>> {
    gens::vec(gens::pick_from(vec![
        "a",
        "b",
        "c",
        "",
        " ",
        "--- x",
        "+++ y",
        "@@ -1 +1 @@",
        "\\ no",
        "é",
    ]))
    .len(0..=12)
    .map(|lines| lines.into_iter().map(String::from).collect())
    .boxed()
}

fn longest_common_subsequence(a: &[String], b: &[String]) -> usize {
    let mut table = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for (i, left) in a.iter().enumerate() {
        for (j, right) in b.iter().enumerate() {
            table[i + 1][j + 1] = if left == right {
                table[i][j] + 1
            } else {
                table[i][j + 1].max(table[i + 1][j])
            };
        }
    }
    table[a.len()][b.len()]
}

#[ephyra::test(gen = (short_lines(), short_lines()))]
fn a_unified_diff_rebuilds_both_files_and_its_summary_counts_its_lines(
    pair: (Vec<String>, Vec<String>),
) {
    let (a, b) = pair;
    let join = |lines: &[String]| {
        lines
            .iter()
            .map(|line| format!("{line}\n"))
            .collect::<String>()
    };
    let (original, updated) = (join(&a), join(&b));
    assert_eq!(unified_diff(&original, &original, "python"), "");
    let diff = unified_diff(&original, &updated, "rust");
    if a == b {
        assert_eq!(diff, "");
        return;
    }
    let mut lines = diff.split('\n');
    assert_eq!(lines.next(), Some("--- a/original.rs"));
    assert_eq!(lines.next(), Some("+++ b/original.rs"));
    assert!(
        lines
            .next()
            .is_some_and(|header| header.starts_with("@@ -")),
        "one hunk with the whole file\n{diff}"
    );
    let (mut before, mut after, mut added, mut removed) = (Vec::new(), Vec::new(), 0, 0);
    for line in lines {
        let Some(sign) = line.chars().next() else {
            continue;
        };
        let text = line[sign.len_utf8()..].to_string();
        match sign {
            ' ' => {
                before.push(text.clone());
                after.push(text);
            }
            '-' => {
                removed += 1;
                before.push(text);
            }
            '+' => {
                added += 1;
                after.push(text);
            }
            other => panic!("unexpected diff line {other:?} in\n{diff}"),
        }
    }
    assert_eq!(
        (before, after),
        (a.clone(), b.clone()),
        "the diff does not rebuild its inputs\n{diff}"
    );
    let summary = summarize_diff(&diff);
    assert_eq!(summary.modified, added.min(removed));
    assert_eq!(
        (
            summary.added + summary.modified,
            summary.deleted + summary.modified
        ),
        (added, removed)
    );
    let common = longest_common_subsequence(&a, &b);
    assert_eq!(
        (added, removed),
        (b.len() - common, a.len() - common),
        "the diff is not minimal\n{diff}"
    );
}

const SOUP: &[&str] = &[
    "<<<<<<< SEARCH",
    "=======",
    ">>>>>>> REPLACE",
    "<<<<<<<SEARCH",
    "  =======  ",
    ">>>>>>> REPLACEMENT",
    "========",
    "# EVOLVE-BLOCK-START",
    "# EVOLVE-BLOCK-END",
    "// EVOLVE-BLOCK-START",
    "// EVOLVE-BLOCK-END",
    "<!-- EVOLVE-BLOCK-START -->",
    "<!-- EVOLVE-BLOCK-END -->",
    "(* EVOLVE-BLOCK-START *)",
    "(* EVOLVE-BLOCK-END *)",
    "(* EVOLVE-BLOCK-START",
    " EVOLVE-BLOCK-END *)",
    "EVOLVE-BLOCK-START",
    "EVOLVE-BLOCK-END",
    "x = 1  # EVOLVE-BLOCK-END",
    "# EVOLVE-BLOCK-START x = 1",
    "# EVOLVE-BLOCK-START # EVOLVE-BLOCK-END",
    "```",
    "```python",
    "```rust",
    "```py",
    "```markdown",
    "```wolfram",
    "```c++",
    "````",
    "x = 1",
    "    x = 1",
    "\tx = 1",
    "x = 1  # note",
    "def f(:",
    "if x:",
    "    return x",
    "}",
    "{",
    "\"\"\"",
    "'",
    "",
    "",
    " ",
    "\u{c}",
    "\u{1c}",
    "é = 'ü'",
    "值",
    "\u{2028}",
    "a\rb",
    "<NAME>",
    "</CODE>",
    "\\",
];

fn soup() -> BoxGen<String> {
    gens::vec((
        gens::pick_from(SOUP.to_vec()),
        gens::pick_from(vec!["\n", "\n", "\n", "\r\n", "\n\n", " ", ""]),
    ))
    .len(0..=24)
    .map(|pieces| {
        pieces
            .into_iter()
            .map(|(piece, separator)| format!("{piece}{separator}"))
            .collect()
    })
    .boxed()
}

const LANGUAGES: &[&str] = &[
    "python", "rust", "markdown", "wolfram", "cpp", "julia", "klingon", "",
];

#[ephyra::test(gen = (soup(), soup(), gens::int(0usize..=7)))]
fn nothing_panics_and_whatever_applies_keeps_read_only_text(input: (String, String, usize)) {
    let (response, original, language) = input;
    let language = LANGUAGES[language];

    let metrics = analyze_code_metrics(&original, language);
    assert!(
        (0.0..=1.0).contains(&metrics.complexity_score),
        "{metrics:?}"
    );
    assert!(
        metrics.halstead_volume.is_finite() && metrics.maintainability_index.is_finite(),
        "{metrics:?}"
    );
    assert_eq!(metrics, analyze_code_metrics(&original, language));
    let _ = (
        redact_immutable(&original),
        validate_evolve_markers(&original, language),
    );
    let diff = unified_diff(&original, &response, language);
    let _ = summarize_diff(&diff);
    let _ = summarize_diff(&response);
    for (start, end) in mutable_ranges(&original) {
        assert!(start <= end && original.is_char_boundary(start) && original.is_char_boundary(end));
    }

    for (kind, result) in [
        ("diff", apply_diff_patch(&response, &original, language)),
        ("full", apply_full_patch(&response, &original, language)),
    ] {
        ephyra::classify(result.is_ok(), format!("{kind} patch applied"));
        match result {
            Ok(outcome) => {
                assert_eq!(
                    immutable_parts(&outcome.code),
                    immutable_parts(&original),
                    "{kind}: read-only text changed"
                );
                assert_eq!(
                    marker_counts(&outcome.code),
                    marker_counts(&original),
                    "{kind}: markers changed"
                );
                assert_eq!(
                    validate_evolve_markers(&outcome.code, language),
                    None,
                    "{kind}"
                );
                assert!(outcome.num_applied >= 1);
            }
            Err(error) => assert!(!error.0.is_empty()),
        }
    }
}

/// A patch written against a valid program by a model that does not respect
/// the rules: it aims at read-only lines and writes markers of its own.
#[derive(Debug, Clone, PartialEq)]
struct HostileCase {
    program: Program,
    blocks: Vec<(Vec<String>, Vec<String>)>,
    /// Lines of a full rewrite.
    rewrite: Vec<String>,
}

fn hostile_cases() -> BoxGen<HostileCase> {
    (|g: &mut Source| {
        let program = program_with(g, false);
        let style = program.style;
        let rendered: Vec<String> = program
            .render_bodies(&program.clean_bodies())
            .split('\n')
            .map(String::from)
            .collect();
        let marker_text = |g: &mut Source| -> String {
            let which = *g.pick(&[START, END]);
            match g.int(0u8..=3) {
                0 => comment(style, which),
                1 => format!("x = 1  {}", comment(style, which)),
                2 => format!("{} y = 2", comment(style, which)),
                _ => format!("s = '{which}'"),
            }
        };
        let any_line = |g: &mut Source| -> String {
            match g.int(0u8..=5) {
                0 | 1 => g.pick(&rendered).clone(),
                2 => marker_text(g),
                _ => code_line(g, style),
            }
        };
        let blocks = (0..g.int(1usize..=4))
            .map(|_| {
                let search = match g.int(0u8..=5) {
                    0 => Vec::new(),
                    1 | 2 => {
                        let bodies = program.clean_bodies();
                        let body = &bodies[g.index(bodies.len())];
                        let from = g.index(body.len());
                        body[from..g.int(from + 1..=body.len())].to_vec()
                    }
                    _ => (0..g.int(1usize..=2)).map(|_| any_line(g)).collect(),
                };
                let replace = (0..g.int(0usize..=3)).map(|_| any_line(g)).collect();
                (search, replace)
            })
            .collect();
        let mut rewrite = rendered.clone();
        for _ in 0..g.int(0usize..=4) {
            let at = g.index(rewrite.len().max(1));
            match g.int(0u8..=3) {
                0 if !rewrite.is_empty() => {
                    rewrite.remove(at.min(rewrite.len() - 1));
                }
                1 => rewrite.insert(at.min(rewrite.len()), marker_text(g)),
                2 => rewrite.insert(at.min(rewrite.len()), code_line(g, style)),
                _ => rewrite.truncate(at),
            }
        }
        HostileCase {
            program,
            blocks,
            rewrite,
        }
    })
    .with_shrink(|case: &HostileCase, visit| {
        for index in 0..case.blocks.len() {
            let mut blocks = case.blocks.clone();
            blocks.remove(index);
            visit(HostileCase {
                blocks,
                ..case.clone()
            })?;
        }
        for index in 0..case.blocks.len() {
            let (search, replace) = &case.blocks[index];
            shrink_lines(replace, 0, &mut |replace| {
                let mut blocks = case.blocks.clone();
                blocks[index] = (search.clone(), replace);
                visit(HostileCase {
                    blocks,
                    ..case.clone()
                })
            })?;
        }
        shrink_lines(&case.rewrite, 0, &mut |rewrite| {
            visit(HostileCase {
                rewrite,
                ..case.clone()
            })
        })?;
        shrink_program(&case.program, &mut |program| {
            visit(HostileCase {
                program,
                ..case.clone()
            })
        })
    })
    .boxed()
}

#[ephyra::test(gen = hostile_cases())]
fn a_patch_that_ignores_the_rules_cannot_change_read_only_text_or_markers(case: HostileCase) {
    let HostileCase {
        program,
        blocks,
        rewrite,
    } = case;
    let original = program.render();
    let dressing = Dressing::default();
    let written: Vec<String> = blocks
        .iter()
        .map(|(search, replace)| block(search, replace, &dressing))
        .collect();

    let diff = apply_diff_patch(
        &diff_response(&program, &written, &dressing),
        &original,
        program.language(),
    );
    assert_not_refused_for_the_appliers_own_doing(&diff, &blocks);
    ephyra::cover(diff.is_ok(), "hostile diff applied");
    if let Ok(outcome) = diff {
        assert_read_only_text_is_untouched(&program, &original, &outcome);
    }
    let response = format!("```{}\n{}\n```", program.language(), rewrite.join("\n"));
    let full = apply_full_patch(&response, &original, program.language());
    ephyra::cover(full.is_ok(), "hostile rewrite applied");
    if let Ok(outcome) = full {
        assert_read_only_text_is_untouched(&program, &original, &outcome);
    }
}

/// A file whose markers are placed every way the marker pattern accepts,
/// and patches aimed at its own lines.
#[derive(Debug, Clone, PartialEq)]
struct OddCase {
    lines: Vec<String>,
    crlf: bool,
    final_newline: bool,
    blocks: Vec<(Vec<String>, Vec<String>)>,
    rewrite: Vec<String>,
    language: &'static str,
}

fn odd_cases() -> BoxGen<OddCase> {
    (|g: &mut Source| {
        let style = g.index(STYLES.len());
        let marker = |g: &mut Source, which: &str| -> String {
            let marker_style = if g.bool_with(0.8) {
                style
            } else {
                g.index(STYLES.len())
            };
            match g.int(0u8..=9) {
                0 => which.to_string(),
                1 => format!("    {}", comment(marker_style, which)),
                2 => format!("{}  {}", code_line(g, style), comment(marker_style, which)),
                3 => format!(
                    "{} {}",
                    comment(marker_style, which),
                    code_line(g, style).trim_start()
                ),
                4 => format!("{}  ", comment(marker_style, which)),
                _ => comment(marker_style, which),
            }
        };
        let mut lines: Vec<String> = Vec::new();
        for _ in 0..g.int(1usize..=3) {
            for _ in 0..g.int(0usize..=2) {
                lines.push(code_line(g, style));
            }
            match g.int(0u8..=9) {
                // Both markers on one line.
                0 => lines.push(format!("{} {}", comment(style, START), comment(style, END))),
                1 => lines.push(format!(
                    "{} x = 1 {}",
                    comment(style, START),
                    comment(style, END)
                )),
                // A region inside a region.
                2 => {
                    lines.push(marker(g, START));
                    lines.push(code_line(g, style));
                    lines.push(marker(g, START));
                    lines.push(code_line(g, style));
                    lines.push(marker(g, END));
                    lines.push(code_line(g, style));
                    lines.push(marker(g, END));
                }
                // A marker with no partner.
                3 => {
                    let which = *g.pick(&[START, END]);
                    lines.push(marker(g, which));
                }
                _ => {
                    lines.push(marker(g, START));
                    for _ in 0..g.int(0usize..=3) {
                        let line = if g.bool_with(0.15) {
                            String::new()
                        } else {
                            code_line(g, style)
                        };
                        lines.push(trailing_noise(g, line));
                    }
                    lines.push(marker(g, END));
                }
            }
        }
        for _ in 0..g.int(0usize..=2) {
            lines.push(code_line(g, style));
        }
        let own_line = |g: &mut Source| -> String {
            let line = g.pick(&lines).clone();
            match g.int(0u8..=5) {
                0 => line.trim_start().to_string(),
                // The code that shares a line with a marker, without the marker.
                1 => line
                    .split(STYLES[style].1)
                    .next()
                    .unwrap_or("")
                    .trim_end()
                    .to_string(),
                _ => rstrip(&line).to_string(),
            }
        };
        let blocks = (0..g.int(1usize..=3))
            .map(|_| {
                let search = match g.int(0u8..=5) {
                    0 => Vec::new(),
                    1 => {
                        let from = g.index(lines.len());
                        lines[from..(from + 2).min(lines.len())]
                            .iter()
                            .map(|line| rstrip(line).to_string())
                            .collect()
                    }
                    _ => vec![own_line(g)],
                };
                let replace = (0..g.int(0usize..=2))
                    .map(|_| {
                        if g.bool_with(0.1) {
                            let which = *g.pick(&[START, END]);
                            marker(g, which)
                        } else {
                            code_line(g, style)
                        }
                    })
                    .collect();
                (search, replace)
            })
            .collect();
        let rewrite = match g.int(0u8..=3) {
            0 => (0..g.int(1usize..=3))
                .map(|_| code_line(g, style))
                .collect(),
            1 => lines
                .iter()
                .map(|line| {
                    if g.bool_with(0.3) {
                        code_line(g, style)
                    } else {
                        line.clone()
                    }
                })
                .collect(),
            _ => lines
                .iter()
                .filter(|_| g.bool_with(0.85))
                .cloned()
                .collect(),
        };
        OddCase {
            lines,
            crlf: g.bool_with(0.15),
            final_newline: g.bool_with(0.7),
            blocks,
            rewrite,
            language: STYLES[style].0,
        }
    })
    .with_shrink(|case: &OddCase, visit| {
        for index in 0..case.blocks.len() {
            let mut blocks = case.blocks.clone();
            blocks.remove(index);
            visit(OddCase {
                blocks,
                ..case.clone()
            })?;
        }
        shrink_lines(&case.lines, 0, &mut |lines| {
            visit(OddCase {
                lines,
                ..case.clone()
            })
        })?;
        shrink_lines(&case.rewrite, 0, &mut |rewrite| {
            visit(OddCase {
                rewrite,
                ..case.clone()
            })
        })?;
        if case.crlf {
            visit(OddCase {
                crlf: false,
                ..case.clone()
            })?;
        }
        ControlFlow::Continue(())
    })
    .boxed()
}

#[ephyra::test(gen = odd_cases())]
fn whatever_applies_to_an_oddly_marked_file_keeps_its_read_only_text(case: OddCase) {
    let mut original = case.lines.join(if case.crlf { "\r\n" } else { "\n" });
    if case.final_newline {
        original.push('\n');
    }
    let dressing = Dressing::default();
    let blocks: Vec<String> = case
        .blocks
        .iter()
        .map(|(search, replace)| block(search, replace, &dressing))
        .collect();
    let responses = [
        (
            "diff",
            apply_diff_patch(&blocks.join("\n\n"), &original, case.language),
        ),
        (
            "full",
            apply_full_patch(
                &format!("```{}\n{}\n```", case.language, case.rewrite.join("\n")),
                &original,
                case.language,
            ),
        ),
    ];
    for (kind, result) in responses {
        ephyra::cover(result.is_ok(), format!("{kind} patch applied"));
        let Ok(outcome) = result else { continue };
        assert_eq!(
            immutable_parts(&outcome.code),
            immutable_parts(&original),
            "{kind}: read-only text changed"
        );
        assert_eq!(
            marker_counts(&outcome.code),
            marker_counts(&original),
            "{kind}: markers changed"
        );
        assert_eq!(
            validate_evolve_markers(&outcome.code, case.language),
            None,
            "{kind}"
        );
        assert_eq!(
            outcome.diff.is_empty(),
            !outcome.diff.contains("\n+") && !outcome.diff.contains("\n-")
        );
    }
}

#[ephyra::test(gen = edit_cases(true))]
fn an_empty_search_appends_to_the_last_region(case: EditCase) {
    let EditCase {
        program,
        edits,
        dressing,
    } = case;
    let addition = &edits[0].body;
    let mut bodies = program.clean_bodies();
    bodies
        .last_mut()
        .expect("a program has a region")
        .extend(addition.iter().cloned());
    let response = diff_response(&program, &[block(&[], addition, &dressing)], &dressing);
    let original = program.render();
    let outcome = apply_diff_patch(&response, &original, program.language())
        .unwrap_or_else(|error| panic!("an insertion was refused:\n{error}"));
    assert_eq!(outcome.code, program.render_bodies(&bodies));
    assert_eq!(outcome.num_applied, 1);
    assert_read_only_text_is_untouched(&program, &original, &outcome);
}

/// A program built from a handful of lines, so that the lines a block names
/// occur in several places, editable and read-only.
#[derive(Debug, Clone, PartialEq)]
struct TargetCase {
    program: Program,
    region: usize,
    from: usize,
    to: usize,
    replacement: Vec<String>,
    dressing: Dressing,
}

const REPEATED: &[&str] = &[
    "x = 1",
    "y = 2",
    "    x = 1",
    "    y = 2",
    "pass",
    "",
    "",
    "x = 1  # tuned",
    "\tx = 1",
];

fn target_cases() -> BoxGen<TargetCase> {
    (|g: &mut Source| {
        let lines = |g: &mut Source, most: usize| -> Vec<String> {
            (0..g.int(0..=most))
                .map(|_| {
                    let line = (*g.pick(REPEATED)).to_string();
                    trailing_noise(g, line)
                })
                .collect()
        };
        let regions: Vec<Region> = (0..g.int(1usize..=3))
            .map(|_| Region {
                body: lines(g, 5),
                after: lines(g, 2),
            })
            .collect();
        let program = Program {
            style: g.index(STYLES.len()),
            marker_indent: (*g.pick(&["", "    "])).to_string(),
            head: lines(g, 3),
            regions,
            final_newline: g.bool_with(0.8),
        };
        let region = g.index(program.regions.len());
        let length = program.regions[region].body.len().max(1);
        let from = g.index(length);
        let to = g.int(from..=(from + 2).min(length - 1));
        let mut replacement: Vec<String> = (0..g.int(0usize..=3))
            .map(|_| (*g.pick(REPEATED)).to_string())
            .filter(|line| !line.is_empty())
            .collect();
        if g.bool_with(0.2) {
            replacement.clear();
        }
        TargetCase {
            program,
            region,
            from,
            to,
            replacement,
            dressing: dressing(g),
        }
    })
    .with_shrink(|case: &TargetCase, visit| {
        shrink_lines(&case.program.head, 0, &mut |head| {
            visit(TargetCase {
                program: Program {
                    head,
                    ..case.program.clone()
                },
                ..case.clone()
            })
        })?;
        for index in 0..case.program.regions.len() {
            if index == case.region {
                continue;
            }
            let mut program = case.program.clone();
            program.regions[index].body.clear();
            program.regions[index].after.clear();
            if program != case.program {
                visit(TargetCase {
                    program,
                    ..case.clone()
                })?;
            }
        }
        shrink_lines(&case.replacement, 0, &mut |replacement| {
            visit(TargetCase {
                replacement,
                ..case.clone()
            })
        })?;
        if case.dressing != Dressing::default() {
            visit(TargetCase {
                dressing: Dressing::default(),
                ..case.clone()
            })?;
        }
        ControlFlow::Continue(())
    })
    .boxed()
}

#[ephyra::test(gen = target_cases())]
fn a_block_changes_the_first_editable_run_it_names_and_only_that(case: TargetCase) {
    let TargetCase {
        program,
        region,
        from,
        to,
        replacement,
        dressing,
    } = case;
    let clean = program.clean_bodies();
    // A section cannot begin or end with a blank line: the grammar drops them.
    let nameable = clean[region]
        .get(from..=to)
        .filter(|run| !run[0].is_empty() && !run[run.len() - 1].is_empty());
    ephyra::cover(nameable.is_some(), "a run that can be named");
    let Some(search) = nameable else { return };
    let (target, offset) = clean
        .iter()
        .enumerate()
        .find_map(|(index, body)| {
            body.windows(search.len())
                .position(|run| run == search)
                .map(|offset| (index, offset))
        })
        .expect("the run was taken from a body");
    ephyra::cover(
        (target, offset) != (region, from),
        "an earlier editable copy exists",
    );
    ephyra::cover(
        program
            .read_only_lines()
            .iter()
            .any(|line| rstrip(line) == search[0]),
        "a read-only copy exists",
    );
    let mut expected = clean.clone();
    let written = if replacement.is_empty() {
        vec![String::new()]
    } else {
        replacement.clone()
    };
    expected[target].splice(offset..offset + search.len(), written);

    let response = diff_response(
        &program,
        &[block(search, &replacement, &dressing)],
        &dressing,
    );
    let original = program.render();
    let outcome = apply_diff_patch(&response, &original, program.language())
        .unwrap_or_else(|error| panic!("a block naming editable lines was refused:\n{error}"));
    assert_eq!(outcome.code, program.render_bodies(&expected));
    assert_eq!(outcome.num_applied, 1);
    assert_read_only_text_is_untouched(&program, &original, &outcome);
}

const C_WORDS: &[&str] = &[
    "if",
    "else",
    "while",
    "for",
    "switch",
    "case",
    "catch",
    "IF",
    "Else",
    "iffy",
    "elsewhere",
    "foo_if",
    "if2",
    "x",
    "int",
    "return",
    "0",
];
const C_PUNCTUATION: &[&str] = &[
    "{", "}", "(", ")", ";", "?", " ? ", "//", "/*", "*/", "\n", "\n", " ", " ", "=", "\t",
];

fn c_like_text() -> BoxGen<String> {
    gens::vec(gens::frequency(vec![
        (3, gens::pick_from(C_WORDS.to_vec()).boxed()),
        (4, gens::pick_from(C_PUNCTUATION.to_vec()).boxed()),
    ]))
    .len(0..=60)
    .map(|pieces| pieces.concat())
    .boxed()
}

#[ephyra::test(gen = c_like_text())]
fn brace_language_metrics_equal_a_word_by_word_recount(code: String) {
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let keywords = ["if", "else", "while", "for", "switch", "case", "catch"];
    let words = code
        .split(|c: char| !is_word(c))
        .filter(|word| keywords.contains(&word.to_ascii_lowercase().as_str()));
    let chars: Vec<char> = code.chars().collect();
    let ternaries = (1..chars.len().saturating_sub(1))
        .filter(|index| {
            chars[*index] == '?' && is_word(chars[index - 1]) && is_word(chars[index + 1])
        })
        .count();
    let (mut depth, mut deepest) = (0u64, 0u64);
    for c in &chars {
        match c {
            '{' => {
                depth += 1;
                deepest = deepest.max(depth);
            }
            '}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    let lines: Vec<&str> = code.split('\n').collect();
    let comments = lines
        .iter()
        .map(|line| line.trim())
        .filter(|line| {
            line.starts_with("//")
                || line.starts_with("/*")
                || (line.ends_with("*/") && !line.is_empty())
        })
        .count() as u64;
    let blank = lines.iter().filter(|line| line.trim().is_empty()).count() as u64;

    for language in [
        "cpp",
        "c",
        "cuda",
        "go",
        "rust",
        "javascript",
        "typescript",
        "c++",
        "golang",
        "rs",
    ] {
        let metrics = analyze_code_metrics(&code, language);
        assert_eq!(
            metrics.cyclomatic_complexity,
            (1 + words.clone().count() + ternaries) as f64,
            "{language}"
        );
        assert_eq!(metrics.max_nesting_depth, deepest);
        assert_eq!(metrics.lines_of_code, lines.len() as u64);
        assert_eq!(metrics.comments, comments);
        assert_eq!(
            metrics.logical_lines_of_code,
            lines.len() as u64 - blank - comments
        );
        assert!(metrics.maintainability_index >= 0.0 && metrics.halstead_volume >= 1.0);
    }
}

/// A rewrite of a one-region program that does not carry both markers.
#[derive(Debug, Clone, PartialEq)]
struct PayloadCase {
    program: Program,
    body: Vec<String>,
    /// What the fence holds: `body` alone, or the whole file.
    whole_file: bool,
    /// Markers the whole-file form keeps.
    keep_start: bool,
    keep_end: bool,
    final_newline: bool,
}

fn payload_cases() -> BoxGen<PayloadCase> {
    (|g: &mut Source| {
        let mut program = program_with(g, false);
        program.regions.truncate(1);
        // Read-only lines that no body can be mistaken for.
        program.head.insert(0, "import original_header".to_string());
        program.regions[0]
            .after
            .push("original_footer()".to_string());
        let body = body_lines(g, program.style, "region_new", false);
        let whole_file = g.bool();
        let (keep_start, keep_end) =
            *g.pick(&[(false, false), (false, false), (true, false), (false, true)]);
        PayloadCase {
            program,
            body,
            whole_file,
            keep_start,
            keep_end,
            final_newline: g.bool(),
        }
    })
    .with_shrink(|case: &PayloadCase, visit| {
        shrink_lines(&case.body, 1, &mut |body| {
            if body.last().is_some_and(|line| line.is_empty()) {
                return ControlFlow::Continue(());
            }
            visit(PayloadCase {
                body,
                ..case.clone()
            })
        })?;
        shrink_program(&case.program, &mut |program| {
            let intact = program
                .head
                .first()
                .is_some_and(|line| line == "import original_header")
                && program.regions[0]
                    .after
                    .last()
                    .is_some_and(|line| line == "original_footer()");
            if !intact {
                return ControlFlow::Continue(());
            }
            visit(PayloadCase {
                program,
                ..case.clone()
            })
        })
    })
    .boxed()
}

#[ephyra::test(gen = payload_cases())]
fn a_rewrite_without_both_markers_replaces_exactly_the_region(case: PayloadCase) {
    let PayloadCase {
        program,
        body,
        whole_file,
        keep_start,
        keep_end,
        final_newline,
    } = case;
    let original = program.render();
    let target = program.render_bodies(std::slice::from_ref(&body));
    let sent = if whole_file {
        let mut lines = program.head.clone();
        if keep_start {
            lines.push(program.marker(START));
        }
        lines.extend(body.iter().cloned());
        if keep_end {
            lines.push(program.marker(END));
        }
        lines.extend(program.regions[0].after.iter().cloned());
        lines.join("\n")
    } else {
        body.join("\n")
    };
    ephyra::cover(!program.marker_indent.is_empty(), "indented markers");
    ephyra::cover(body[0].starts_with(' '), "an indented first line");
    let response = format!(
        "```{}\n{sent}{}```",
        program.language(),
        if final_newline { "\n" } else { "" }
    );
    let outcome = apply_full_patch(&response, &original, program.language())
        .unwrap_or_else(|error| panic!("a rewrite of the only region was refused:\n{error}"));
    assert_eq!(outcome.code, target);
    assert_read_only_text_is_untouched(&program, &original, &outcome);
}
