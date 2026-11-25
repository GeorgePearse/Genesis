//! Numeric literals in prose, and whether the evidence supports them.
//!
//! A number written with `d` decimal places is supported by a measured value
//! `v` when it is a correct rounding of it: `|v - x| <= 0.5 * 10^-d`. A
//! percentage may also state a fraction (`0.8` as `80%`), and an unsigned
//! number may state a magnitude ("drops by 0.3" for `-0.3`).

use regex::Regex;
use std::sync::OnceLock;

/// Numbers the template and ordinary statistical wording introduce: the
/// level of a 95% interval, the 0.05 significance level, and the 100 of a
/// percentage scale.
pub const STRUCTURAL: &[f64] = &[95.0, 0.05, 100.0];

/// Words that turn the number after them into a label rather than a claim.
const LABEL_WORDS: &[&str] = &[
    "table",
    "tables",
    "figure",
    "figures",
    "fig",
    "fig.",
    "section",
    "sections",
    "sec",
    "sec.",
    "appendix",
    "equation",
    "equations",
    "eq",
    "eq.",
    "algorithm",
    "listing",
];

#[derive(Debug, Clone, PartialEq)]
pub struct NumberLiteral {
    /// The literal as written, sign and percent sign included.
    pub text: String,
    /// Its value; negative only when the sign was written.
    pub value: f64,
    /// Half of one unit in the last place shown.
    pub half_unit: f64,
    pub negative: bool,
    pub percent: bool,
}

fn literal_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(r"(?P<sign>[-−])?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<frac>\d+))?(?:[eE](?P<exp>[-+]?\d+))?(?P<pct>\s?%|\spercent\b)?")
            .expect("valid regex")
    })
}

fn build(
    sign: bool,
    int: &str,
    frac: Option<&str>,
    exp: Option<&str>,
    percent: bool,
    text: &str,
) -> Option<NumberLiteral> {
    let digits = int.replace(',', "");
    let exponent: i32 = match exp {
        Some(exp) => exp.parse().ok()?,
        None => 0,
    };
    if exponent.abs() > 300 {
        return None;
    }
    let decimals = frac.map_or(0, str::len) as i32;
    let mantissa: f64 = match frac {
        Some(frac) => format!("{digits}.{frac}").parse().ok()?,
        None => digits.parse().ok()?,
    };
    let magnitude = mantissa * 10f64.powi(exponent);
    if !magnitude.is_finite() {
        return None;
    }
    Some(NumberLiteral {
        text: text.to_string(),
        value: if sign { -magnitude } else { magnitude },
        half_unit: 0.5 * 10f64.powi(exponent - decimals),
        negative: sign,
        percent,
    })
}

impl NumberLiteral {
    /// Parse a string that is exactly one literal: `12`, `-0.5`, `1,234.5`,
    /// `12.5%`, `1.5e-3`.
    pub fn parse(text: &str) -> Option<Self> {
        let trimmed = text.trim();
        let found = literal_pattern().captures(trimmed)?;
        let whole = found.get(0)?;
        if whole.start() != 0 || whole.end() != trimmed.len() {
            return None;
        }
        build(
            found.name("sign").is_some(),
            &found["int"],
            found.name("frac").map(|m| m.as_str()),
            found.name("exp").map(|m| m.as_str()),
            found.name("pct").is_some(),
            trimmed,
        )
    }

    /// Whether this literal is a correct rounding of `value`.
    pub fn matches_value(&self, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        let close = |candidate: f64| {
            let (shown, measured) = if self.negative {
                (self.value, candidate)
            } else {
                (self.value, candidate.abs())
            };
            (measured - shown).abs() <= self.half_unit * (1.0 + 1e-9) + 1e-12 * measured.abs()
        };
        close(value) || (self.percent && close(value * 100.0))
    }

    pub fn matches_any(&self, values: &[f64]) -> bool {
        values.iter().any(|value| self.matches_value(*value))
    }
}

/// Blank out, byte for byte, the spans that are not claims: citations,
/// placeholders, and the numbering at the start of a heading line.
fn mask(prose: &str) -> String {
    let mut bytes = prose.as_bytes().to_vec();
    let mut blank = |range: std::ops::Range<usize>| {
        for byte in &mut bytes[range] {
            if *byte != b'\n' {
                *byte = b' ';
            }
        }
    };
    for (open, close) in [("[@", "]"), ("{{", "}}")] {
        let mut cursor = 0;
        while let Some(start) = prose[cursor..].find(open).map(|offset| cursor + offset) {
            match prose[start..].find(close) {
                Some(length) => {
                    blank(start..start + length + close.len());
                    cursor = start + length + close.len();
                }
                None => break,
            }
        }
    }
    let mut offset = 0;
    for line in prose.split_inclusive('\n') {
        let hashes = line.len() - line.trim_start_matches([' ', '#']).len();
        if line.trim_start().starts_with('#') {
            let numbering = line[hashes..]
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .count();
            blank(offset + hashes..offset + hashes + numbering);
        }
        offset += line.len();
    }
    // Every blanked span starts and ends on an ASCII delimiter or digit, so
    // only whole characters were replaced.
    String::from_utf8(bytes)
        .unwrap_or_else(|error| String::from_utf8_lossy(error.as_bytes()).into_owned())
}

/// Every numeric literal in `prose` that makes a claim.
///
/// Skipped: numbers inside citations and placeholders; numbers that are
/// part of an identifier (`v2`, `gen_12`, `top-5`, `1.2.3`); numbers that
/// follow a label word (`Table 1`, `Section 3`); and ordinals and
/// dimensions (`1st`, `2D`).
pub fn extract_numbers(prose: &str) -> Vec<NumberLiteral> {
    let masked = mask(prose);
    let text = masked.as_str();
    let mut out = Vec::new();
    for found in literal_pattern().captures_iter(text) {
        let whole = found.get(0).expect("group 0");
        let digits_start = found.name("int").expect("int group").start();
        let before = &text[..digits_start];
        let mut preceding = before.chars().rev();
        let previous = preceding.next();
        let earlier = preceding.next();

        let glued_to_word = match previous {
            Some(c) if c.is_alphabetic() || c == '_' => true,
            Some('-' | '−' | '/' | '.') => earlier.is_some_and(|c| c.is_alphabetic() || c == '_'),
            _ => false,
        };
        let dotted_before = previous == Some('.') && earlier.is_some_and(|c| c.is_ascii_digit());
        let after = &text[whole.end()..];
        let mut following = after.chars();
        let dotted_after =
            following.next() == Some('.') && following.next().is_some_and(|c| c.is_ascii_digit());
        if glued_to_word || dotted_before || dotted_after {
            continue;
        }
        let suffix: String = after.chars().take_while(|c| c.is_alphabetic()).collect();
        if matches!(suffix.as_str(), "st" | "nd" | "rd" | "th" | "D") {
            continue;
        }
        let label = before
            .trim_end_matches([' ', '~', '\u{a0}'])
            .rsplit(|c: char| c.is_whitespace() || c == '(')
            .next()
            .unwrap_or("")
            .to_lowercase();
        if LABEL_WORDS.contains(&label.as_str()) {
            continue;
        }
        // A hyphen is a minus sign only when it does not join two tokens.
        let sign = found.name("sign").is_some_and(|sign| {
            text[..sign.start()]
                .chars()
                .next_back()
                .is_none_or(|c| !(c.is_alphanumeric() || c == ')' || c == ']'))
        });
        let start = if sign { whole.start() } else { digits_start };
        if let Some(literal) = build(
            sign,
            &found["int"],
            found.name("frac").map(|m| m.as_str()),
            found.name("exp").map(|m| m.as_str()),
            found.name("pct").is_some(),
            text[start..whole.end()].trim(),
        ) {
            out.push(literal);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn literal(text: &str) -> NumberLiteral {
        NumberLiteral::parse(text).unwrap_or_else(|| panic!("{text} should parse"))
    }

    #[test]
    fn literals_parse_with_their_precision() {
        assert_eq!((literal("12").value, literal("12").half_unit), (12.0, 0.5));
        assert_eq!(
            (
                literal("-0.50").value,
                literal("-0.50").half_unit,
                literal("-0.50").negative
            ),
            (-0.5, 0.005, true)
        );
        assert_eq!(literal("1,234.5").value, 1234.5);
        assert!(literal("12.5%").percent && literal("12 percent").percent);
        assert!((literal("1.5e-3").value - 0.0015).abs() < 1e-15);
        assert!((literal("1.5e-3").half_unit - 0.00005).abs() < 1e-15);
        assert_eq!(literal("−3").value, -3.0);
        assert!(NumberLiteral::parse("abc").is_none() && NumberLiteral::parse("1.2.3").is_none());
        assert!(NumberLiteral::parse("1e999").is_none());
    }

    #[test]
    fn a_literal_matches_values_that_round_to_it() {
        assert!(literal("2.64").matches_value(2.6359));
        assert!(literal("2.64").matches_value(2.6449));
        assert!(!literal("2.64").matches_value(2.6451));
        assert!(!literal("2.64").matches_value(2.63));
        // A tie is accepted either way.
        assert!(literal("0.12").matches_value(0.125) && literal("0.13").matches_value(0.125));
        assert!(literal("3").matches_value(2.6) && !literal("3").matches_value(3.6));
        // Percentages state the value or a fraction.
        assert!(literal("80%").matches_value(0.8) && literal("80%").matches_value(80.2));
        assert!(!literal("80").matches_value(0.8));
        // Unsigned literals state a magnitude; signed ones do not.
        assert!(literal("0.3").matches_value(-0.3));
        assert!(literal("-0.3").matches_value(-0.3) && !literal("-0.3").matches_value(0.3));
        assert!(
            literal("1.5e-3").matches_value(0.00151) && !literal("1.5e-3").matches_value(0.0016)
        );
        assert!(!literal("1").matches_value(f64::NAN));
        assert!(literal("95%").matches_any(STRUCTURAL));
    }

    #[test]
    fn extraction_keeps_claims_and_skips_labels_identifiers_and_citations() {
        let prose = "### 2.1 Setup\nThe score rose from 6.0 to 9.84 (a 64% gain) over 12 generations [@smith2020; @lee21] \
                     as {{REF:TABLE:results}} and Table 3 show. Model gpt-4 used top-5 sampling in 2D with v2 of gen_12, \
                     version 1.2.3, the 3rd run, costing 10ms and a change of -0.25 or 2-3 restarts.";
        let found: Vec<String> = extract_numbers(prose).into_iter().map(|n| n.text).collect();
        assert_eq!(found, ["6.0", "9.84", "64%", "12", "10", "-0.25", "2", "3"]);
        assert!(extract_numbers("no numbers here, see Figure 2 and Sec. 4").is_empty());
        let signed = extract_numbers("x = -4 and (−2.5)");
        assert_eq!(
            signed.iter().map(|n| n.value).collect::<Vec<_>>(),
            [-4.0, -2.5]
        );
    }
}
