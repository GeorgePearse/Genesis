//! Plain tests for the failures the property suite found, each with the
//! input ephyra reported and the line that replays it.

use genesis_core::stats::{Comparison, Summary};
use genesis_paper::data::PaperData;
use genesis_paper::references::{bib_keys, write_bibtex, Reference};
use genesis_paper::render::available_floats;
use genesis_paper::text::{escape_latex, FloatKind};
use genesis_scientist::artifacts::{AblationSummary, VariantResult, VariantSpec};

/// Brace depth by raw count, the way BibTeX counts.
fn open_braces(text: &str) -> i64 {
    text.chars()
        .map(|c| i64::from(c == '{') - i64::from(c == '}'))
        .sum()
}

/// Found by `escaped_text_is_inert_brace_balanced_and_built_character_by_character`
/// (EPHYRA_REPLAY=1:0, counterexample `"{"`) and by
/// `bibtex_entries_carry_distinct_keys_and_balanced_braces` (EPHYRA_REPLAY=1:2,
/// reduced by hand to a one-character title). A brace in text was escaped as
/// `\{`. TeX reads that as a literal brace, but BibTeX counts braces without
/// regard to the backslash, so one unmatched brace in a retrieved title ended
/// the field early: bibtex reported "I was expecting a `,' or a `}'", wrote a
/// broken `.bbl`, and pdflatex stopped with "Extra }" and no PDF.
#[test]
fn a_brace_in_a_reference_title_does_not_unbalance_the_bibtex_file() {
    assert_eq!(open_braces(&escape_latex("{")), 0);
    assert_eq!(open_braces(&escape_latex("}")), 0);
    let references = [
        Reference {
            citation_key: "a2020opening".into(),
            title: "{".into(),
            ..Reference::default()
        },
        Reference {
            citation_key: "b2021closing".into(),
            title: "A closing } brace".into(),
            authors: vec!["van der {Berg".into()],
            venue: "}j$".into(),
            ..Reference::default()
        },
    ];
    let bibtex = write_bibtex(&references);
    let mut depth = 0;
    for line in bibtex.lines() {
        depth += open_braces(line);
        assert!(depth >= 0, "a closing brace has no opener at {line:?}");
        // Inside an entry every field line closes what it opens.
        assert!(depth <= 1, "a field is left open at {line:?}");
    }
    assert_eq!(depth, 0, "{bibtex}");
    assert_eq!(bib_keys(&bibtex), ["a2020opening", "b2021closing"]);
}

/// Found by `tables_and_figures_are_well_formed_and_never_show_a_non_finite_number`
/// (EPHYRA_REPLAY=1:6), reduced by hand. An interval bound that is not finite
/// reached the ablation figure as the pgfplots coordinate `+- (-inf,0)` or
/// `+- (NaN,0)`, and a p-value that is not a number was printed as `NaN`. A
/// JSON file cannot hold such a value, so the pipeline meets it only when two
/// huge bounds overflow in the subtraction (`wide` below); a caller of
/// `available_floats` can pass one directly.
#[test]
fn a_statistic_that_is_not_finite_is_shown_as_not_available_and_left_out_of_the_figure() {
    let variant = |id: &str, low: f64, high: f64, p_value: Option<f64>| VariantResult {
        spec: VariantSpec {
            variant_id: id.into(),
            kind: "ablation".into(),
            ..VariantSpec::default()
        },
        summary: Some(Summary {
            n: 3,
            mean: 1.0,
            ci95_low: low,
            ci95_high: high,
            ..Summary::default()
        }),
        comparison_to_control: p_value.map(|p| Comparison {
            p_value: Some(p),
            ..Comparison::default()
        }),
        ..VariantResult::default()
    };
    let data = PaperData {
        ablations: Some(AblationSummary {
            anchor_variant_id: "control".into(),
            variants: vec![
                variant("control", 0.5, 1.5, None),
                variant("unbounded", f64::INFINITY, 1.5, Some(f64::NAN)),
                variant("other", 0.0, 2.0, Some(0.25)),
                variant("wide", -1.7e308, 1.7e308, Some(f64::INFINITY)),
            ],
            ..AblationSummary::default()
        }),
        ..PaperData::default()
    };
    let floats = available_floats(&data);
    let ablations: Vec<_> = floats
        .iter()
        .filter(|float| float.name == "ablations")
        .collect();
    assert_eq!(ablations.len(), 2, "a table and a figure");
    for float in &ablations {
        let svg = float.svg.as_ref().map_or("", |(_, svg)| svg.as_str());
        for text in [float.latex.as_str(), float.markdown.as_str(), svg] {
            assert!(!text.contains("NaN") && !text.contains("inf"), "{text}");
        }
    }
    let table = ablations
        .iter()
        .find(|float| float.kind == FloatKind::Table)
        .unwrap();
    assert!(
        table.markdown.contains(
            "| unbounded | ablation | 3 | 1.0000 | 0.0000 | [n/a, 1.5000] | 0.0000 | n/a |"
        ),
        "{}",
        table.markdown
    );
    assert!(table.markdown.contains(
        "| other | ablation | 3 | 1.0000 | 0.0000 | [0.0000, 2.0000] | 0.0000 | 0.250 |"
    ));
    let figure = ablations
        .iter()
        .find(|float| float.kind == FloatKind::Figure)
        .unwrap();
    assert!(
        figure.latex.contains("yticklabels={{control},{other}}"),
        "{}",
        figure.latex
    );
    assert!(
        figure
            .latex
            .contains("coordinates {(1,1) +- (0.5,0) (1,2) +- (1,0)}"),
        "{}",
        figure.latex
    );
    assert!(
        figure
            .markdown
            .contains("2 variants without a finite interval are not drawn"),
        "{}",
        figure.markdown
    );
}
