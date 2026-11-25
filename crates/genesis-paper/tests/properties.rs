//! Property tests for genesis-paper, run with ephyra
//! (<https://github.com/datadog-labs/ephyra>).
//!
//! What "correct" means here, and where each expectation comes from:
//!
//! | Property | Oracle | Source |
//! |---|---|---|
//! | Escaped text has balanced braces by raw count, no active character and no control character, and is the concatenation of its characters' escapes | invariant, metamorphic | `text.rs`: "balanced braces and no active character"; BibTeX counts braces without regard to backslashes |
//! | No character's escape is a prefix of another's, and two different texts never escape to the same TeX | invariant | `text.rs`: "Distinct inputs give distinct outputs" |
//! | A literal supports a value exactly when it is a correct rounding of it, of its magnitude, or (for a percentage) of a hundred times it | reference model in integer arithmetic | `numbers.rs` module docs |
//! | The numbers extracted from a manuscript are its claims: none from citations, placeholders, labels, identifiers, versions, ordinals or heading numbering | reference model (the generator knows which numbers are claims) | `extract_numbers` docs |
//! | Extraction accepts any text, and each literal it returns parses back to itself | crash, round trip | it reads model-written prose |
//! | A correctly rounded grounding value is never an audit finding; a number far from every value always is | metamorphic | `audit.rs`: the deterministic layer |
//! | An audit is printable exactly when the contract can carry it, a printed audit parses back to itself, and a wrong count, verdict or number of findings is rejected | reference model, round trip, invariant | `print_model_audit` and `parse_model_audit` docs |
//! | Any reply parses without panicking, and an accepted one obeys the contract | crash, invariant | the reply is model-written |
//! | Citation keys are distinct lowercase ASCII, and the BibTeX file holds exactly those entry keys, in order, with balanced braces | invariant, round trip | `references.rs`: `assign_keys`, `write_bibtex` |
//! | Grounding ids are never overwritten, every measured value is finite, and every finite statistic of the evidence is grounded | invariant, reference model | `data.rs`: `Grounding`, `ground_evidence` |
//! | Generated tables and figures have rows of one width, balanced braces, and no non-finite number, whatever the evidence holds | invariant | `render.rs`: `available_floats`; evidence files are read from disk |
//! | Averaged review scores are the rounded means and stay in range; a form with a score out of range or a field missing is rejected | reference model, invariant | `review.rs`: `SCORE_RANGES`, `average_scores`, `validate_review` |
//! | Any prose renders to TeX with balanced braces and environments and without a command that reads or writes files, and only keys written in a citation group are reported as cited | invariant, reference model | `render.rs`: `assemble`; `text.rs` module docs |

use ephyra::prelude::*;
use ephyra::{gens, BoxGen};
use genesis_core::stats::{Comparison, Summary};
use genesis_paper::audit::{
    deterministic_audit, parse_model_audit, print_model_audit, AuditFinding, ModelAudit,
    MAX_MODEL_FINDINGS,
};
use genesis_paper::data::{ground_evidence, Grounding, GroundingSource, PaperData};
use genesis_paper::draft::{Draft, SECTIONS};
use genesis_paper::numbers::{extract_numbers, NumberLiteral};
use genesis_paper::references::{assign_keys, bib_keys, write_bibtex, Reference};
use genesis_paper::render::{assemble, available_floats};
use genesis_paper::text::escape_latex;
use genesis_paper::{average_scores, validate_review, SCORE_RANGES};
use genesis_scientist::artifacts::{
    AblationSummary, AnalysisResult, AnalysisSpec, FinalEpisode, FinalManifest, PosthocManifest,
    Review, VariantResult, VariantSpec,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};

const SPECIALS: &str = "\\{}$&#_%~^<>|`\"'-[]@*";
const TYPESET: &str = "éüßÅøŁž–—“”…×−±°µ≤≥≈≠→αβΔΩ\u{a0}";
const AWKWARD: &str = "\n\t\r\u{0}\u{1b}\u{7f}\u{85}\u{2028}\u{feff}\u{200b}\u{1F600}";

fn chars_of(set: &str) -> BoxGen<char> {
    gens::pick_from(set.chars().collect()).boxed()
}

/// A character weighted toward the ones TeX, BibTeX and the markup treat
/// specially.
fn tricky_char() -> BoxGen<char> {
    gens::frequency(vec![
        (4, chars_of(SPECIALS)),
        (4, gens::ascii_char().boxed()),
        (2, chars_of(TYPESET)),
        (1, chars_of(AWKWARD)),
        (2, gens::char().boxed()),
    ])
    .boxed()
}

fn tricky_text(max: usize) -> BoxGen<String> {
    gens::string_len(tricky_char(), 0..=max).boxed()
}

fn one_of_texts(texts: &[&str]) -> BoxGen<String> {
    gens::pick_from(texts.iter().map(|text| text.to_string()).collect()).boxed()
}

/// Brace depth by raw count, the way BibTeX counts: a backslash does not
/// hide a brace.
fn raw_braces(text: &str) -> Result<(), String> {
    let mut depth = 0i64;
    for (at, c) in text.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth < 0 {
                    return Err(format!("the closing brace at byte {at} has no opener"));
                }
            }
            _ => {}
        }
    }
    if depth == 0 {
        Ok(())
    } else {
        Err(format!("{depth} braces are left open"))
    }
}

/// No TeX special does anything: a backslash starts a command word or
/// escapes one of `#$%&_`, and no other special appears bare.
fn inert(tex: &str) -> Result<(), String> {
    let mut chars = tex.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some(next) if "#$%&_".contains(next) => {}
                Some(next) if next.is_ascii_alphabetic() => {
                    while chars.peek().is_some_and(char::is_ascii_alphabetic) {
                        chars.next();
                    }
                }
                other => return Err(format!("a backslash is followed by {other:?}")),
            },
            '#' | '$' | '%' | '&' | '_' | '^' | '~' => return Err(format!("a bare `{c}`")),
            _ => {}
        }
    }
    Ok(())
}

/// The names of the control words in a TeX source.
fn control_words(tex: &str) -> HashSet<String> {
    let mut words = HashSet::new();
    let mut chars = tex.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            continue;
        }
        let mut word = String::new();
        while let Some(letter) = chars.peek().copied().filter(char::is_ascii_alphabetic) {
            word.push(letter);
            chars.next();
        }
        if word.is_empty() {
            chars.next();
        } else {
            words.insert(word);
        }
    }
    words
}

#[ephyra::test(gen = tricky_text(40), tests = 2000)]
fn escaped_text_is_inert_brace_balanced_and_built_character_by_character(text: String) {
    let escaped = escape_latex(&text);
    ephyra::cover(text.contains(['{', '}']), "a brace in the input");
    ephyra::cover(text.contains('\\'), "a backslash in the input");
    ephyra::cover(!text.is_ascii(), "non-ASCII input");
    assert_eq!(raw_braces(&escaped), Ok(()), "in {escaped:?}");
    assert_eq!(inert(&escaped), Ok(()), "in {escaped:?}");
    assert!(
        escaped
            .chars()
            .all(|c| c == '\n' || c == '\t' || !c.is_control()),
        "a control character survived in {escaped:?}"
    );
    let by_character: String = text
        .chars()
        .map(|c| escape_latex(c.encode_utf8(&mut [0; 4])))
        .collect();
    assert_eq!(escaped, by_character);
}

#[ephyra::test(gen = (tricky_char(), tricky_char()), tests = 2000)]
fn no_escape_is_a_prefix_of_another(pair: (char, char)) {
    let (a, b) = pair;
    let (escaped_a, escaped_b) = (escape_latex(&a.to_string()), escape_latex(&b.to_string()));
    ephyra::cover(a != b, "two different characters");
    assert!(!escaped_a.is_empty(), "{a:?} escapes to nothing");
    if a != b {
        assert!(
            !escaped_a.starts_with(&escaped_b),
            "{escaped_b:?} (for {b:?}) begins {escaped_a:?} (for {a:?})"
        );
        assert!(
            !escaped_b.starts_with(&escaped_a),
            "{escaped_a:?} (for {a:?}) begins {escaped_b:?} (for {b:?})"
        );
    }
}

/// Short texts made of specials and of what specials escape to, so that two
/// of them can plausibly collide.
fn lookalike() -> BoxGen<String> {
    let fragment = gens::frequency(vec![
        (3, tricky_char().map(String::from).boxed()),
        (
            2,
            one_of_texts(&[
                "\\textbackslash{}",
                "\\{",
                "\\}",
                "{}",
                "\\%",
                "%",
                "\\textsuperscript{U+2028}",
                "U+2028",
                "\u{2028}",
                "\\ensuremath{\\le}",
                "≤",
                "\\textless{}",
                "<",
                "\\",
            ]),
        ),
    ]);
    gens::vec(fragment)
        .len(0..=4)
        .map(|parts| parts.concat())
        .boxed()
}

#[ephyra::test(gen = (lookalike(), lookalike()), tests = 2000)]
fn two_texts_escape_alike_only_when_they_are_equal(pair: (String, String)) {
    let (a, b) = pair;
    ephyra::cover(a == b, "equal inputs");
    assert_eq!(
        a == b,
        escape_latex(&a) == escape_latex(&b),
        "{a:?} and {b:?}"
    );
}

fn pow10(exponent: u32) -> i128 {
    10i128.pow(exponent)
}

/// Whether `m / 10^k` lies within half a unit in the last place of
/// `l / 10^places`, in exact integer arithmetic. `places` may be negative.
fn rounds_to(m: i128, k: u32, l: i128, places: i32) -> bool {
    let (down, up) = (places.max(0) as u32, (-places).max(0) as u32);
    2 * (m * pow10(down) - l * pow10(up) * pow10(k)).abs() <= pow10(up) * pow10(k)
}

/// A literal as prose writes it, and a measured value placed against it.
///
/// The measured value is `m / 10^k` with `k <= 4` and a magnitude of at most
/// 100, and the literal shows at most three places: within those bounds the
/// narrowest miss is ten times wider than the matcher's float tolerance, so
/// the integer model and the matcher must agree on every case.
#[derive(Debug, Clone)]
struct LiteralCase {
    negative: bool,
    unicode_minus: bool,
    digits: u32,
    decimals: u32,
    exponent: Option<i32>,
    upper_e: bool,
    percent: &'static str,
    commas: bool,
    /// 0 anywhere, 1 at the rounding boundary, 2 at the boundary of the
    /// percentage reading, 3 exactly on the literal.
    placement: u8,
    above: bool,
    /// Units past (or short of) the boundary.
    jitter: i64,
    value_negative: bool,
    free_m: i64,
    free_k: u32,
}

impl LiteralCase {
    fn exponent(&self) -> i32 {
        let decimals = self.decimals as i32;
        self.exponent
            .map_or(0, |exponent| exponent.clamp(decimals - 3, decimals + 2))
    }

    /// Decimal places shown once the exponent is applied.
    fn places(&self) -> i32 {
        self.decimals as i32 - self.exponent()
    }

    fn text(&self) -> String {
        let padded = format!(
            "{:0width$}",
            self.digits,
            width = self.decimals as usize + 1
        );
        let (whole, fraction) = padded.split_at(padded.len() - self.decimals as usize);
        let mut out = String::new();
        if self.negative {
            out.push(if self.unicode_minus { '−' } else { '-' });
        }
        if self.commas && self.exponent.is_none() {
            for (index, digit) in whole.chars().enumerate() {
                if index > 0 && (whole.len() - index).is_multiple_of(3) {
                    out.push(',');
                }
                out.push(digit);
            }
        } else {
            out.push_str(whole);
        }
        if !fraction.is_empty() {
            out.push('.');
            out.push_str(fraction);
        }
        if self.exponent.is_some() {
            out.push(if self.upper_e { 'E' } else { 'e' });
            out.push_str(&self.exponent().to_string());
        }
        out.push_str(self.percent);
        out
    }

    /// The measured value as `(m, k)`: `m / 10^k`.
    fn measured(&self) -> (i128, u32) {
        let places = self.places();
        let shown = i128::from(self.digits);
        let sign = if self.value_negative { -1 } else { 1 };
        let side = if self.above { 1 } else { -1 };
        let jitter = i128::from(self.jitter);
        let placed = match self.placement {
            1 | 3 => {
                let k = (places + 1).max(0) as u32;
                let shift = (k as i32 - places) as u32;
                let offset = if self.placement == 1 {
                    side * (pow10(shift) / 2 + jitter)
                } else {
                    0
                };
                Some((sign * (shown * pow10(shift) + offset), k))
            }
            2 if !self.percent.is_empty() && places <= 1 => Some((
                sign * (shown * 10 + side * (5 + jitter)),
                (places + 3) as u32,
            )),
            _ => None,
        };
        match placed {
            Some((m, k)) if m.abs() <= 100 * pow10(k) => (m, k),
            _ => (
                i128::from(self.free_m) % (100 * pow10(self.free_k) + 1),
                self.free_k,
            ),
        }
    }

    fn supports(&self, m: i128, k: u32) -> bool {
        let (shown, places) = (i128::from(self.digits), self.places());
        let close = |m: i128| {
            if self.negative {
                rounds_to(m, k, -shown, places)
            } else {
                rounds_to(m.abs(), k, shown, places)
            }
        };
        close(m) || (!self.percent.is_empty() && close(100 * m))
    }
}

fn literal_case() -> BoxGen<LiteralCase> {
    let digits = gens::frequency(vec![
        (3, gens::int(0u32..=99).boxed()),
        (3, gens::int(0u32..=9_999).boxed()),
        (1, gens::int(0u32..=2_000_000).boxed()),
    ]);
    let shape = (
        gens::bool_with(0.3),
        gens::bool(),
        digits,
        gens::int(0u32..=3),
        gens::option(gens::int(-3i32..=5)),
        gens::bool(),
        gens::pick_from(vec!["", "", "", "%", " %", " percent"]),
        gens::bool_with(0.3),
    );
    let value = (
        gens::int(0u8..=3),
        gens::bool(),
        gens::int(-2i64..=2),
        gens::bool_with(0.3),
        gens::int(-1_000_000i64..=1_000_000),
        gens::int(0u32..=4),
    );
    (shape, value)
        .map(|(shape, value)| {
            let (negative, unicode_minus, digits, decimals, exponent, upper_e, percent, commas) =
                shape;
            let (placement, above, jitter, value_negative, free_m, free_k) = value;
            LiteralCase {
                negative,
                unicode_minus,
                digits,
                decimals,
                exponent,
                upper_e,
                percent,
                commas,
                placement,
                above,
                jitter,
                value_negative,
                free_m,
                free_k,
            }
        })
        .boxed()
}

#[ephyra::test(gen = literal_case(), tests = 4000)]
fn a_literal_supports_exactly_the_values_it_is_a_correct_rounding_of(case: LiteralCase) {
    let text = case.text();
    let literal = NumberLiteral::parse(&text).unwrap_or_else(|| panic!("`{text}` did not parse"));
    assert_eq!(
        (literal.negative, literal.percent),
        (case.negative, !case.percent.is_empty()),
        "`{text}`"
    );

    let (m, k) = case.measured();
    let value = m as f64 / 10f64.powi(k as i32);
    let expected = case.supports(m, k);
    ephyra::cover(expected, "the value is supported");
    ephyra::cover(!expected, "the value is not supported");
    ephyra::cover(
        expected && !case.percent.is_empty(),
        "a supported percentage",
    );
    ephyra::cover(case.exponent.is_some(), "scientific notation");
    assert_eq!(
        literal.matches_value(value),
        expected,
        "`{text}` against {value} ({m} / 10^{k})"
    );

    let sentence = format!("The value is {text} here.");
    assert_eq!(
        extract_numbers(&sentence),
        vec![literal],
        "extracted from {sentence:?}"
    );
}

#[derive(Debug, Clone, PartialEq)]
enum Piece {
    /// A number the manuscript states.
    Claim(String),
    /// Prose, or a number that is a label, a key or part of a name.
    Other(String),
}

fn claim() -> BoxGen<String> {
    gens::frequency(vec![
        (3, gens::int(0u32..=99_999).map(|n| n.to_string()).boxed()),
        (
            3,
            (
                gens::int(0u32..=999),
                gens::int(0u32..=9_999),
                gens::int(1usize..=4),
            )
                .map(|(whole, fraction, places)| {
                    format!("{whole}.{:0places$}", fraction % 10u32.pow(places as u32))
                })
                .boxed(),
        ),
        (
            2,
            (gens::int(0u32..=999), gens::pick_from(vec!["-", "−"]))
                .map(|(n, sign)| format!("{sign}{n}.5"))
                .boxed(),
        ),
        (
            2,
            (
                gens::int(0u32..=100),
                gens::pick_from(vec!["%", " %", " percent"]),
            )
                .map(|(n, suffix)| format!("{n}{suffix}"))
                .boxed(),
        ),
        (
            1,
            (gens::int(1u32..=999), gens::int(0u32..=999))
                .map(|(a, b)| format!("{a},{b:03}"))
                .boxed(),
        ),
        (
            1,
            (
                gens::int(1u32..=9),
                gens::int(0u32..=9),
                gens::int(-9i32..=9),
            )
                .map(|(a, b, exponent)| format!("{a}.{b}e{exponent}"))
                .boxed(),
        ),
    ])
    .boxed()
}

fn not_a_claim() -> BoxGen<String> {
    let n = || gens::int(0u32..=9_999);
    gens::frequency(vec![
        (
            4,
            one_of_texts(&[
                "the",
                "score",
                "improves",
                "over",
                "a baseline",
                "and",
                "(",
                ")",
                "which",
                "résumé",
                "it rose by",
                "p <",
                "=",
            ]),
        ),
        (
            2,
            (n(), n())
                .map(|(a, b)| format!("[@smith{a}evolving; @lee{b}]"))
                .boxed(),
        ),
        (2, n().map(|a| format!("{{{{REF:TABLE:t{a}}}}}")).boxed()),
        (1, n().map(|a| format!("{{{{FIGURE:fig{a}}}}}")).boxed()),
        (
            3,
            (
                gens::pick_from(vec![
                    "Table",
                    "Figure",
                    "(Table",
                    "Section",
                    "Fig.",
                    "Eq.",
                    "Appendix",
                    "table",
                    "Sec.",
                    "Algorithm",
                    "Listing",
                    "Equation",
                    "Tables",
                    "Figures",
                ]),
                n(),
                gens::pick_from(vec![" ", "~", "\u{a0}"]),
            )
                .map(|(word, a, gap)| format!("{word}{gap}{a}"))
                .boxed(),
        ),
        (
            3,
            (
                gens::pick_from(vec![
                    "v", "gen_", "top-", "step_", "GPT-", "x", "island/", "run.",
                ]),
                n(),
            )
                .map(|(prefix, a)| format!("{prefix}{a}"))
                .boxed(),
        ),
        (
            1,
            (n(), n(), n())
                .map(|(a, b, c)| format!("{a}.{b}.{c}"))
                .boxed(),
        ),
        (
            2,
            (n(), gens::pick_from(vec!["st", "nd", "rd", "th", "D"]))
                .map(|(a, suffix)| format!("{a}{suffix}"))
                .boxed(),
        ),
        (
            1,
            (gens::int(1u32..=9), gens::int(0u32..=9), gens::bool())
                .map(|(a, b, subsection)| {
                    if subsection {
                        format!("\n\n### {a}.{b} Detailed results\n\n")
                    } else {
                        format!("\n\n## {a} Results\n\n")
                    }
                })
                .boxed(),
        ),
    ])
    .boxed()
}

fn manuscript() -> BoxGen<Vec<(Piece, &'static str)>> {
    let piece = gens::frequency(vec![
        (2, claim().map(Piece::Claim).boxed()),
        (5, not_a_claim().map(Piece::Other).boxed()),
    ]);
    let gap = gens::pick_from(vec![" ", " ", " ", ", ", ". ", "; ", "\n\n", " — "]);
    gens::vec((piece, gap)).len(0..=14).boxed()
}

#[ephyra::test(gen = manuscript(), tests = 4000)]
fn the_numbers_extracted_from_a_manuscript_are_its_claims(parts: Vec<(Piece, &'static str)>) {
    let mut text = String::new();
    let mut claims = Vec::new();
    for (piece, gap) in &parts {
        match piece {
            Piece::Claim(number) => {
                claims.push(number.clone());
                text.push_str(number);
            }
            Piece::Other(prose) => text.push_str(prose),
        }
        text.push_str(gap);
    }
    ephyra::cover(!claims.is_empty(), "at least one claim");
    ephyra::cover(claims.len() < parts.len(), "something that is not a claim");
    let found: Vec<String> = extract_numbers(&text)
        .into_iter()
        .map(|literal| literal.text)
        .collect();
    assert_eq!(found, claims, "in {text:?}");
}

fn numeric_soup() -> BoxGen<String> {
    let fragment = gens::frequency(vec![
        (
            6,
            one_of_texts(&[
                "0", "1", "5", "9", "12", "007", "1,234", "1e5", "1e999", "2E-400", "e", "E", "-",
                "−", "+", ".", ",", "%", " %", " percent", " ", "\n", "#", "## ", "[@", "]", "{{",
                "}}", "Table ", "th", "D", "_", "(", ")", "/", "x", "é", "\u{a0}",
            ]),
        ),
        (
            1,
            gens::int(1usize..=400)
                .map(|length| "9".repeat(length))
                .boxed(),
        ),
        (1, tricky_text(6)),
    ]);
    gens::vec(fragment)
        .len(0..=24)
        .map(|parts| parts.concat())
        .boxed()
}

#[ephyra::test(gen = numeric_soup(), tests = 4000)]
fn extraction_accepts_any_text_and_returns_literals_that_parse_back(text: String) {
    let literals = extract_numbers(&text);
    ephyra::cover(!literals.is_empty(), "at least one literal");
    for literal in literals {
        assert!(
            literal.value.is_finite() && literal.half_unit >= 0.0,
            "{literal:?} from {text:?}"
        );
        assert_eq!(
            literal.negative,
            literal.text.starts_with(['-', '−']),
            "{literal:?}"
        );
        assert_eq!(
            NumberLiteral::parse(&literal.text),
            Some(literal.clone()),
            "from {text:?}"
        );
    }
}

#[ephyra::test(
    gen = (
        gens::vec((gens::f64_range(-1000.0..=1000.0), gens::int(0usize..=4))).len(1..=6),
        gens::int(0u32..=999_999),
        gens::bool(),
    ),
    tests = 2000
)]
fn a_correct_rounding_is_never_a_finding_and_a_foreign_number_always_is(
    input: (Vec<(f64, usize)>, u32, bool),
) {
    let (measurements, offset, inject) = input;
    let mut data = PaperData::default();
    let mut results = String::new();
    for (index, (value, places)) in measurements.iter().copied().enumerate() {
        let id = format!("result.value_{index}");
        let source = GroundingSource {
            id: id.clone(),
            kind: "metric".into(),
            value: json!(value),
            description: String::new(),
        };
        data.grounding_sources.insert(id, source);
        results.push_str(&format!("One measurement gave {value:.places$}. "));
    }
    // Further from every value, and from a hundred times every value, than
    // any rounding reaches.
    let foreign = (10_000_000 + offset).to_string();
    if inject {
        results.push_str(&format!("Another gave {foreign}."));
    }
    let draft = Draft {
        sections: BTreeMap::from([("results".to_string(), results.clone())]),
        ..Draft::default()
    };
    let findings = deterministic_audit(&draft, &data, &[]);
    let claims: Vec<&str> = findings
        .iter()
        .map(|finding| finding.claim.as_str())
        .collect();
    if inject {
        assert_eq!(claims, [foreign.as_str()], "in {results:?}");
        assert_eq!(findings[0].id, "number-results-1");
    } else {
        assert!(findings.is_empty(), "{claims:?} flagged in {results:?}");
    }
}

/// Whether the reply contract can carry `text` as one value: a non-empty
/// trimmed line in which no space-separated word is the field separator.
fn carriable(text: &str) -> bool {
    !text.is_empty()
        && text.trim() == text
        && !text.contains(['\n', '\r'])
        && text.split(' ').all(|word| word != "||")
}

fn field_text() -> BoxGen<String> {
    gens::frequency(vec![
        (
            40,
            gens::vec(gens::string_len(gens::alphanumeric(), 1..=8))
                .len(1..=5)
                .map(|words| words.join(" "))
                .boxed(),
        ),
        (
            1,
            one_of_texts(&[
                "a || b",
                "",
                " padded ",
                "two\nlines",
                "||",
                "ends ||",
                "|| starts",
                "a||b",
                "a | | b",
                "tab\t||\tb",
                "AUDIT_FINDING_COUNT: 3",
                "AUDIT_VERDICT: PASS",
                "id=x",
                "```",
            ]),
        ),
        (1, tricky_text(12)),
    ])
    .boxed()
}

fn finding() -> BoxGen<AuditFinding> {
    (
        field_text(),
        field_text(),
        field_text(),
        field_text(),
        field_text(),
        field_text(),
    )
        .map(|(id, section, claim, issue, evidence, fix)| AuditFinding {
            id,
            section,
            claim,
            issue,
            evidence,
            fix,
        })
        .boxed()
}

fn model_audit() -> BoxGen<ModelAudit> {
    let findings = gens::frequency(vec![
        (4, gens::vec(finding()).len(0..=3).boxed()),
        (
            1,
            gens::vec(finding()).len(0..=MAX_MODEL_FINDINGS + 2).boxed(),
        ),
    ]);
    (field_text(), findings, gens::bool_with(0.1))
        .map(|(summary, findings, wrong_verdict)| ModelAudit {
            pass: findings.is_empty() != wrong_verdict,
            summary,
            findings,
        })
        .boxed()
}

#[ephyra::test(gen = (model_audit(), gens::int(1usize..=5)), tests = 3000)]
fn a_printed_model_audit_parses_back_and_a_miscounted_one_is_rejected(input: (ModelAudit, usize)) {
    let (audit, miscount) = input;
    let printable = audit.pass == audit.findings.is_empty()
        && audit.findings.len() <= MAX_MODEL_FINDINGS
        && carriable(&audit.summary)
        && audit.findings.iter().all(|finding| {
            [
                &finding.id,
                &finding.section,
                &finding.claim,
                &finding.issue,
                &finding.evidence,
                &finding.fix,
            ]
            .into_iter()
            .all(|value| carriable(value))
        });
    ephyra::cover(printable, "a printable audit");
    ephyra::cover(
        printable && !audit.findings.is_empty(),
        "a printable audit with findings",
    );
    let printed = print_model_audit(&audit);
    assert_eq!(printed.is_some(), printable, "{audit:?}");
    let Some(text) = printed else { return };
    assert_eq!(
        parse_model_audit(&text).ok(),
        Some(audit.clone()),
        "from {text:?}"
    );

    let lines: Vec<String> = text.lines().map(String::from).collect();
    let declared = audit.findings.len();
    let mut miscounted = lines.clone();
    miscounted[2] = format!("AUDIT_FINDING_COUNT: {}", declared + miscount);
    assert!(
        parse_model_audit(&miscounted.join("\n")).is_err(),
        "{declared} findings passed as {}",
        declared + miscount
    );
    let mut flipped = lines.clone();
    flipped[0] = format!(
        "AUDIT_VERDICT: {}",
        if audit.pass { "FAIL" } else { "PASS" }
    );
    assert!(
        parse_model_audit(&flipped.join("\n")).is_err(),
        "the opposite verdict was accepted for {text:?}"
    );
    if declared > 0 {
        let short = &lines[..lines.len() - 1];
        assert!(
            parse_model_audit(&short.join("\n")).is_err(),
            "a missing finding went unnoticed in {text:?}"
        );
    }
}

fn reply_soup() -> BoxGen<String> {
    let line = gens::frequency(vec![
        (
            8,
            one_of_texts(&[
                "AUDIT_VERDICT: PASS",
                "AUDIT_VERDICT: FAIL",
                "AUDIT_VERDICT: maybe",
                "AUDIT_VERDICT:",
                "AUDIT_SUMMARY: fine",
                "AUDIT_SUMMARY:",
                "AUDIT_SUMMARY: a || b",
                "AUDIT_FINDING_COUNT: 0",
                "AUDIT_FINDING_COUNT: 1",
                "AUDIT_FINDING_COUNT: 2",
                "AUDIT_FINDING_COUNT: -1",
                "AUDIT_FINDING_COUNT: 99999999999999999999999",
                "AUDIT_FINDING: id=a || section=b || claim=c || issue=d || evidence=e || fix=f",
                "AUDIT_FINDING: id=a || section=b || claim=c || issue=d || evidence=e || fix=f ||",
                "AUDIT_FINDING: id=a || section=b || claim=c || issue=d || evidence=e",
                "AUDIT_FINDING: id= || section=b || claim=c || issue=d || evidence=e || fix=f",
                "AUDIT_FINDING: section=b || id=a || claim=c || issue=d || evidence=e || fix=f",
                "AUDIT_FINDING:",
                "```",
                "```text",
                "",
                "  AUDIT_VERDICT: PASS  ",
                "Here is my audit.",
            ]),
        ),
        (1, tricky_text(20)),
    ]);
    let end = gens::pick_from(vec!["\n", "\n", "\r\n", "\n\n"]);
    gens::vec((line, end))
        .len(0..=10)
        .map(|lines| {
            lines
                .into_iter()
                .map(|(line, end)| line + end)
                .collect::<String>()
        })
        .boxed()
}

#[ephyra::test(gen = reply_soup(), tests = 4000)]
fn any_reply_parses_without_panicking_and_an_accepted_one_obeys_the_contract(reply: String) {
    let parsed = parse_model_audit(&reply);
    ephyra::cover(parsed.is_ok(), "an accepted reply");
    let Ok(audit) = parsed else { return };
    assert_eq!(audit.pass, audit.findings.is_empty(), "{audit:?}");
    assert!(audit.findings.len() <= MAX_MODEL_FINDINGS);
    let one_line = |text: &String| !text.is_empty() && text.trim() == text && !text.contains('\n');
    assert!(one_line(&audit.summary), "{audit:?}");
    for finding in &audit.findings {
        let values = [
            &finding.id,
            &finding.section,
            &finding.claim,
            &finding.issue,
            &finding.evidence,
            &finding.fix,
        ];
        assert!(values.into_iter().all(one_line), "{finding:?}");
    }
}

fn reference() -> BoxGen<Reference> {
    let name = || {
        gens::frequency(vec![
            (
                4,
                one_of_texts(&[
                    "Ada Lovelace",
                    "Kurt Gödel",
                    "",
                    "  ",
                    "Æ",
                    "O'Neil, Cathy",
                    "van der {Berg}",
                    "x",
                ]),
            ),
            (1, tricky_text(12)),
        ])
    };
    let title = gens::frequency(vec![
        (
            4,
            one_of_texts(&[
                "Evolving programs",
                "Evolving programs again",
                "a b c",
                "",
                "100% {faster} search_methods & more",
                "An opening { brace",
                "A closing } brace",
                "Line\nbreak\n@article{evil,",
            ]),
        ),
        (2, tricky_text(30)),
    ]);
    let year = gens::pick_from(vec![
        None,
        Some(2021u32),
        Some(2022),
        Some(0),
        Some(u32::MAX),
    ]);
    let venue = gens::frequency(vec![
        (2, gens::constant(String::new()).boxed()),
        (1, gens::constant("NeurIPS".to_string()).boxed()),
        (1, tricky_text(10)),
    ]);
    (title, gens::vec(name()).len(0..=3), year, venue)
        .map(|(title, authors, year, venue)| Reference {
            title,
            authors,
            year,
            venue,
            ..Reference::default()
        })
        .boxed()
}

fn bibliography() -> BoxGen<Vec<Reference>> {
    gens::frequency(vec![
        (3, gens::vec(reference()).len(0..=12).boxed()),
        (
            1,
            (reference(), gens::int(27usize..=60))
                .map(|(reference, copies)| vec![reference; copies])
                .boxed(),
        ),
    ])
    .boxed()
}

#[ephyra::test(gen = bibliography(), tests = 2000)]
fn bibtex_entries_carry_distinct_keys_and_balanced_braces(references: Vec<Reference>) {
    let mut references = references;
    assign_keys(&mut references);
    let keys: Vec<String> = references
        .iter()
        .map(|reference| reference.citation_key.clone())
        .collect();
    ephyra::cover(keys.len() > 26, "more colliding references than letters");
    let distinct: HashSet<&String> = keys.iter().collect();
    assert_eq!(distinct.len(), keys.len(), "duplicate keys in {keys:?}");
    assert!(
        keys.iter().all(|key| !key.is_empty()
            && key
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())),
        "{keys:?}"
    );
    let mut again = references.clone();
    assign_keys(&mut again);
    assert_eq!(again, references, "assigning keys twice changed them");

    let bibtex = write_bibtex(&references);
    assert_eq!(bib_keys(&bibtex), keys, "in {bibtex:?}");
    assert_eq!(raw_braces(&bibtex), Ok(()), "in {bibtex:?}");
    assert!(
        bibtex.chars().all(|c| c == '\n' || !c.is_control()),
        "a control character in {bibtex:?}"
    );
}

fn statistic() -> BoxGen<f64> {
    gens::frequency(vec![
        (8, gens::f64_range(-1e6..=1e6).boxed()),
        (
            2,
            gens::int(0usize..=3)
                .map(|which| [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0][which])
                .boxed(),
        ),
    ])
    .boxed()
}

fn summary_statistics() -> BoxGen<Summary> {
    (
        gens::int(0usize..=50),
        statistic(),
        statistic(),
        statistic(),
        statistic(),
        statistic(),
        statistic(),
    )
        .map(|(n, mean, std, min, max, ci95_low, ci95_high)| Summary {
            n,
            mean,
            std,
            sem: std,
            ci95_low,
            ci95_high,
            min,
            max,
            median: mean,
        })
        .boxed()
}

fn comparison() -> BoxGen<Comparison> {
    let optional = || gens::option(statistic());
    (statistic(), optional(), optional(), optional(), optional())
        .map(
            |(mean_diff, p_value, diff_ci95_low, diff_ci95_high, cohens_d)| Comparison {
                mean_diff,
                p_value,
                diff_ci95_low,
                diff_ci95_high,
                cohens_d,
                ..Comparison::default()
            },
        )
        .boxed()
}

/// Names that collide once slugified, are empty, or are arbitrary text.
fn label() -> BoxGen<String> {
    gens::frequency(vec![
        (
            5,
            one_of_texts(&[
                "control_best",
                "no restarts",
                "no_restarts",
                "No-Restarts",
                "",
                "a.b",
                "Ünï",
                "p value",
            ]),
        ),
        (1, tricky_text(8)),
    ])
    .boxed()
}

/// Names with characters that table and figure syntax gives meaning to,
/// none of which spells a non-finite number.
fn cell_label() -> BoxGen<String> {
    one_of_texts(&[
        "control_best",
        "no restarts",
        "a, b",
        "x]y",
        "a | b",
        "R&D",
        "50% {off}",
        "back\\slash",
        "",
    ])
}

fn ablations(label: fn() -> BoxGen<String>) -> BoxGen<AblationSummary> {
    let variant = (
        label(),
        statistic(),
        gens::option(summary_statistics()),
        gens::option(comparison()),
    )
        .map(
            |(variant_id, correct_rate, summary, comparison_to_control)| VariantResult {
                spec: VariantSpec {
                    variant_id,
                    kind: "ablation".into(),
                    ..VariantSpec::default()
                },
                correct_rate,
                summary,
                comparison_to_control,
                ..VariantResult::default()
            },
        );
    (
        label(),
        gens::vec(variant).len(0..=5),
        gens::int(0usize..=9),
    )
        .map(
            |(anchor_variant_id, variants, num_replications)| AblationSummary {
                anchor_variant_id,
                num_replications,
                total_variants: variants.len(),
                total_completed: variants.len(),
                variants,
                ..AblationSummary::default()
            },
        )
        .boxed()
}

fn final_manifest(label: fn() -> BoxGen<String>) -> BoxGen<FinalManifest> {
    (
        label(),
        gens::bool(),
        gens::option(summary_statistics()),
        gens::option(summary_statistics()),
        gens::option(comparison()),
        gens::vec(gens::bool()).len(0..=4),
    )
        .map(
            |(outcome, correct, candidate, baseline, comparison, episodes)| FinalManifest {
                status: "completed".into(),
                outcome,
                correct,
                candidate,
                baseline,
                comparison,
                episodes: episodes
                    .into_iter()
                    .map(|correct| FinalEpisode {
                        correct,
                        ..FinalEpisode::default()
                    })
                    .collect(),
                ..FinalManifest::default()
            },
        )
        .boxed()
}

fn posthoc(label: fn() -> BoxGen<String>) -> BoxGen<PosthocManifest> {
    gens::vec((
        label(),
        label(),
        gens::vec((label(), statistic())).len(0..=4),
    ))
    .len(0..=3)
    .map(|analyses| PosthocManifest {
        valid_analyses: analyses
            .into_iter()
            .map(|(id, claim, numbers)| AnalysisResult {
                spec: AnalysisSpec {
                    id,
                    ..AnalysisSpec::default()
                },
                claim,
                key_numbers: numbers.into_iter().collect(),
                ..AnalysisResult::default()
            })
            .collect(),
        ..PosthocManifest::default()
    })
    .boxed()
}

fn finite(values: &[f64]) -> usize {
    values.iter().filter(|value| value.is_finite()).count()
}

fn finite_present(values: &[Option<f64>]) -> usize {
    values
        .iter()
        .flatten()
        .filter(|value| value.is_finite())
        .count()
}

fn of_summary(stats: &Summary) -> usize {
    finite(&[
        stats.mean,
        stats.std,
        stats.min,
        stats.max,
        stats.ci95_low,
        stats.ci95_high,
    ])
}

/// How many sources the evidence must give, and how many relative
/// differences it may add on top (one per comparison, when the reference
/// mean allows a ratio).
fn expected_sources(
    ablations: Option<&AblationSummary>,
    final_manifest: Option<&FinalManifest>,
    posthoc: Option<&PosthocManifest>,
) -> (usize, usize) {
    let (mut certain, mut optional) = (0, 0);
    if let Some(summary) = ablations {
        certain += 3;
        for variant in &summary.variants {
            certain += finite(&[variant.correct_rate]);
            certain += variant
                .summary
                .as_ref()
                .map_or(0, |stats| of_summary(stats) + 1);
            if let Some(comparison) = &variant.comparison_to_control {
                certain += finite(&[comparison.mean_diff])
                    + finite_present(&[
                        comparison.p_value,
                        comparison.diff_ci95_low,
                        comparison.diff_ci95_high,
                    ]);
                optional += 1;
            }
        }
    }
    if let Some(manifest) = final_manifest {
        certain += 7;
        certain += manifest.candidate.as_ref().map_or(0, of_summary)
            + manifest.baseline.as_ref().map_or(0, of_summary);
        if let Some(comparison) = &manifest.comparison {
            certain += finite(&[comparison.mean_diff])
                + finite_present(&[
                    comparison.p_value,
                    comparison.diff_ci95_low,
                    comparison.diff_ci95_high,
                    comparison.cohens_d,
                ]);
            optional += 1;
        }
    }
    if let Some(manifest) = posthoc {
        for analysis in &manifest.valid_analyses {
            certain += 1 + analysis
                .key_numbers
                .values()
                .filter(|value| value.is_finite())
                .count();
        }
    }
    (certain, optional)
}

#[ephyra::test(
    gen = (gens::option(ablations(label)), gens::option(final_manifest(label)), gens::option(posthoc(label))),
    tests = 2000
)]
fn every_finite_statistic_is_grounded_under_an_id_of_its_own(
    evidence: (
        Option<AblationSummary>,
        Option<FinalManifest>,
        Option<PosthocManifest>,
    ),
) {
    let (ablations, final_manifest, posthoc) = evidence;
    let mut grounding = Grounding::default();
    ground_evidence(
        &mut grounding,
        ablations.as_ref(),
        final_manifest.as_ref(),
        posthoc.as_ref(),
    );
    let count = grounding.sources().len();
    let (certain, optional) = expected_sources(
        ablations.as_ref(),
        final_manifest.as_ref(),
        posthoc.as_ref(),
    );
    ephyra::cover(count > 0, "some evidence");
    assert_eq!(grounding.inserted(), count, "a source was overwritten");
    assert!(
        (certain..=certain + optional).contains(&count),
        "{count} sources; the evidence holds {certain} to {}",
        certain + optional
    );
    for (id, source) in grounding.sources() {
        assert_eq!(id, &source.id);
        assert!(
            !id.is_empty()
                && id
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '.'),
            "id {id:?}"
        );
        if matches!(
            source.kind.as_str(),
            "metric" | "derived" | "posthoc_data" | "count"
        ) {
            assert!(
                source.value.as_f64().is_some_and(f64::is_finite),
                "{id} = {}",
                source.value
            );
        }
    }

    ground_evidence(
        &mut grounding,
        ablations.as_ref(),
        final_manifest.as_ref(),
        posthoc.as_ref(),
    );
    assert_eq!(
        grounding.sources().len(),
        2 * count,
        "a second pass replaced sources of the first"
    );
    assert_eq!(grounding.inserted(), 2 * count);
}

/// How many times `mark` occurs in `line` without a backslash before it.
fn unescaped(line: &str, mark: char) -> usize {
    let mut previous = ' ';
    let mut count = 0;
    for c in line.chars() {
        if c == mark && previous != '\\' {
            count += 1;
        }
        previous = c;
    }
    count
}

#[ephyra::test(gen = (gens::option(ablations(cell_label)), gens::option(final_manifest(cell_label))), tests = 2000)]
fn tables_and_figures_are_well_formed_and_never_show_a_non_finite_number(
    evidence: (Option<AblationSummary>, Option<FinalManifest>),
) {
    let (ablations, final_evaluation) = evidence;
    let data = PaperData {
        ablations,
        final_evaluation,
        ..PaperData::default()
    };
    for float in available_floats(&data) {
        ephyra::cover(float.svg.is_some(), "a figure");
        ephyra::cover(float.name == "ablations", "an ablation table or figure");
        assert_eq!(raw_braces(&float.latex), Ok(()), "in {}", float.latex);
        let svg = float.svg.as_ref().map_or("", |(_, svg)| svg.as_str());
        for (medium, text) in [
            ("TeX", float.latex.as_str()),
            ("Markdown", float.markdown.as_str()),
            ("SVG", svg),
        ] {
            assert!(
                !text.contains("NaN") && !text.contains("inf"),
                "a non-finite number in the {medium} of `{}`: {text}",
                float.name
            );
        }
        let tex_rows: HashSet<usize> = float
            .latex
            .lines()
            .filter(|line| line.ends_with("\\\\"))
            .map(|line| unescaped(line, '&'))
            .collect();
        assert!(
            tex_rows.len() <= 1,
            "TeX rows of different widths in {}",
            float.latex
        );
        let markdown_rows: HashSet<usize> = float
            .markdown
            .lines()
            .filter(|line| line.starts_with('|'))
            .map(|line| unescaped(line, '|'))
            .collect();
        assert!(
            markdown_rows.len() <= 1,
            "Markdown rows of different widths in {}",
            float.markdown
        );
    }
}

/// A filled-in form. Each raw number picks a half-step within its score's
/// range.
fn form(raw: &[u32], accept: bool) -> Value {
    let mut form = json!({
        "Summary": "The paper reports one automated search.",
        "Strengths": ["a clear protocol"],
        "Weaknesses": ["a single task"],
        "Questions": [],
        "Limitations": [],
        "Ethical Concerns": false,
        "Decision": if accept { "Accept" } else { "Reject" },
    });
    for ((name, low, high), raw) in SCORE_RANGES.iter().zip(raw) {
        let steps = 2 * (*high - *low) as u32 + 1;
        form[*name] = json!(*low + f64::from(raw % steps) / 2.0);
    }
    form
}

fn raw_scores() -> BoxGen<Vec<u32>> {
    gens::vec(gens::int(0u32..=1000))
        .len(SCORE_RANGES.len()..=SCORE_RANGES.len())
        .boxed()
}

#[ephyra::test(gen = gens::vec((raw_scores(), gens::bool())).len(1..=7), tests = 2000)]
fn averaged_scores_are_the_rounded_means_and_stay_in_range(reviewers: Vec<(Vec<u32>, bool)>) {
    let forms: Vec<Value> = reviewers
        .iter()
        .map(|(raw, accept)| form(raw, *accept))
        .collect();
    let mut reviews: Vec<Review> = forms
        .iter()
        .map(|form| validate_review(form).expect("a form within every range"))
        .collect();
    let mut meta = reviews[0].clone();
    average_scores(&mut meta, &reviews);
    let averaged = serde_json::to_value(&meta).unwrap();
    let reviewer_count = forms.len() as u64;
    for (name, low, high) in SCORE_RANGES {
        let halves: u64 = forms
            .iter()
            .map(|form| (form[*name].as_f64().unwrap() * 2.0) as u64)
            .sum();
        let expected = ((halves + reviewer_count) / (2 * reviewer_count)) as f64;
        let score = averaged[*name].as_f64().unwrap();
        ephyra::cover(
            halves % (2 * reviewer_count) == reviewer_count,
            "a mean exactly between two scores",
        );
        assert_eq!(
            score, expected,
            "{name} over {reviewer_count} reviewers ({halves} half points)"
        );
        assert!((*low..=*high).contains(&score), "{name} = {score}");
    }
    assert_eq!(
        (meta.decision.as_str(), &meta.summary),
        (reviews[0].decision.as_str(), &reviews[0].summary)
    );

    reviews.reverse();
    let mut reversed = meta.clone();
    average_scores(&mut reversed, &reviews);
    assert_eq!(
        reversed, meta,
        "the order of the reviewers changed the average"
    );
}

#[ephyra::test(
    gen = (raw_scores(), gens::int(0usize..=8), gens::bool(), gens::f64_range(0.001..=1e6), gens::int(0usize..=15)),
    tests = 2000
)]
fn a_form_with_a_score_out_of_range_or_a_field_missing_is_rejected(
    input: (Vec<u32>, usize, bool, f64, usize),
) {
    let (raw, which, above, by, dropped) = input;
    let valid = form(&raw, true);
    assert!(validate_review(&valid).is_ok(), "{valid}");

    let (name, low, high) = SCORE_RANGES[which];
    let mut out_of_range = valid.clone();
    out_of_range[name] = json!(if above { high + by } else { low - by });
    assert!(
        validate_review(&out_of_range).is_err(),
        "{name} = {} was accepted",
        out_of_range[name]
    );
    let mut worded = valid.clone();
    worded[name] = json!("four");
    assert!(
        validate_review(&worded).is_err(),
        "{name} in words was accepted"
    );

    let keys: Vec<String> = valid.as_object().unwrap().keys().cloned().collect();
    let key = &keys[dropped % keys.len()];
    let mut missing = valid.clone();
    missing.as_object_mut().unwrap().remove(key);
    assert!(
        validate_review(&missing).is_err(),
        "a form without `{key}` was accepted"
    );
}

/// Every key written in a `[@a; @b]` group of `body`. The markup allows a
/// comma for the semicolon, spaces around a key, and a key without its `@`.
fn citation_groups(body: &str) -> impl Iterator<Item = &str> {
    body.split("[@")
        .skip(1)
        .filter_map(|after| after.split_once(']'))
        .flat_map(|(group, _)| group.split([';', ',']))
        .map(|key| key.trim().trim_start_matches('@'))
}

/// Prose in the writer's markup, with broken markup and TeX mixed in.
fn markup() -> BoxGen<String> {
    let fragment = gens::frequency(vec![
        (
            5,
            one_of_texts(&[
                "the search",
                "improves",
                " ",
                "\n\n",
                "\n",
                "*emph*",
                "**strong**",
                "`code {x}`",
                "$x_{1}$",
                "$\\frac{a}{b}$",
                "$\\input{evil}$",
                "[@key1; @ghost]",
                "[@key1]",
                "{{TABLE:results}}",
                "{{FIGURE:nope}}",
                "{{REF:FIGURE:improvement}}",
                "### Heading\n",
                "- item\n",
                "{",
                "}",
                "\\",
                "%",
                "&",
                "_",
                "#",
                "~",
                "^",
                "\\end{document}",
                "\\input{/etc/passwd}",
                "\\begin{verbatim}",
                "\\write18{rm}",
                "$$",
                "$",
                "`",
                "*",
                "**",
                "[@",
                "{{",
                "}}",
            ]),
        ),
        (1, tricky_text(8)),
    ]);
    gens::vec(fragment)
        .len(0..=14)
        .map(|parts| parts.concat())
        .boxed()
}

#[ephyra::test(
    gen = (tricky_text(20), gens::vec(markup()).len(SECTIONS.len()..=SECTIONS.len()), gens::vec(tricky_text(20)).len(0..=2)),
    tests = 1500
)]
fn any_prose_renders_to_tex_with_balanced_braces_and_environments(
    input: (String, Vec<String>, Vec<String>),
) {
    let (title, bodies, limitations) = input;
    let draft = Draft {
        title,
        sections: SECTIONS
            .iter()
            .zip(&bodies)
            .map(|(spec, body)| (spec.key.to_string(), body.clone()))
            .collect(),
        claims: vec![],
    };
    let data = PaperData {
        task_name: "task".into(),
        limitations,
        ..PaperData::default()
    };
    let references = [Reference {
        citation_key: "key1".into(),
        title: "Evolving {programs".into(),
        authors: vec!["Ada Lovelace".into()],
        year: Some(2021),
        ..Reference::default()
    }];
    let assembled = assemble(&draft, &data, &references);
    let tex = &assembled.tex;
    ephyra::cover(!assembled.cite_keys.is_empty(), "a citation");
    ephyra::cover(
        !assembled.unresolved.is_empty(),
        "an unresolved placeholder",
    );

    assert_eq!(raw_braces(tex), Ok(()), "in {tex:?}");
    assert_eq!(
        tex.matches("\\begin{").count(),
        tex.matches("\\end{").count(),
        "in {tex:?}"
    );
    assert_eq!(
        (
            tex.matches("\\begin{document}").count(),
            tex.matches("\\end{document}").count()
        ),
        (1, 1)
    );
    assert!(tex.trim_end().ends_with("\\end{document}"));
    let words = control_words(tex);
    for forbidden in [
        "input",
        "include",
        "write",
        "immediate",
        "openout",
        "openin",
        "read",
        "catcode",
        "def",
        "csname",
    ] {
        assert!(
            !words.contains(forbidden),
            "\\{forbidden} reached the TeX source: {tex:?}"
        );
    }
    assert!(
        !tex.contains("\\begin{verbatim}"),
        "a verbatim environment reached the TeX source: {tex:?}"
    );
    for key in &assembled.cite_keys {
        assert!(
            bodies
                .iter()
                .any(|body| citation_groups(body).any(|written| written == key)),
            "`{key}` is cited by no section"
        );
    }
    assert!(!assembled.markdown.is_empty());
}
