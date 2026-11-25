//! Checking the numbers of a sentence against the numbers an analysis
//! computed.
//!
//! A model may word a finding, but it may not supply a number. Every number
//! in its text must be one of the analysis' key numbers, rounded to the
//! digits the text shows; a percentage may also be a key number times one
//! hundred. Names in backticks and digits that are part of a word
//! (`gen_12`, `gpt-5`) are not numbers and are not checked. A number too
//! large or too fine for a float cannot be compared with anything and is
//! never accepted.

use regex::Regex;
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// One number as it appears in a text.
#[derive(Debug, Clone, PartialEq)]
pub struct NumberToken {
    pub text: String,
    pub value: f64,
    /// Half a unit of the last digit shown: how far a key number may be
    /// from `value` and still round to it.
    pub tolerance: f64,
    /// Followed by a percent sign.
    pub percent: bool,
}

impl NumberToken {
    /// Whether a float holds both the value and the precision it was shown
    /// with. `1e999` reads as infinity and `8e-987` as zero; neither can be
    /// compared with a key number.
    pub fn representable(&self) -> bool {
        self.value.is_finite() && self.tolerance.is_finite() && self.tolerance > 0.0
    }
}

fn scanner() -> &'static Regex {
    static SCANNER: OnceLock<Regex> = OnceLock::new();
    SCANNER.get_or_init(|| {
        Regex::new(concat!(
            r"(?P<code>`[^`]*`)",
            r"|(?P<word>[\p{L}_][\p{L}\p{N}_]*(?:[-./][\p{L}\p{N}_]+)*)",
            r"|(?P<num>[-+\u{2212}]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)(?P<pct>\s?%)?",
        ))
        .expect("valid regex")
    })
}

/// Every number in `text`, in order.
pub fn numbers_in(text: &str) -> Vec<NumberToken> {
    let mut tokens = Vec::new();
    for found in scanner().captures_iter(text) {
        let Some(number) = found.name("num") else {
            continue;
        };
        let mut raw = number.as_str().replace('\u{2212}', "-");
        // A sign directly after a digit, a letter or a closing bracket is a
        // dash ("3-5", "gen-5"), not a sign.
        let before = text[..number.start()].chars().next_back();
        if before.is_some_and(|c| c.is_alphanumeric() || c == ')' || c == '%') {
            raw = raw.trim_start_matches(['-', '+']).to_string();
        }
        let plain = raw.replace(',', "");
        let Ok(value) = plain.parse::<f64>() else {
            continue;
        };
        let (mantissa, exponent) = match plain.find(['e', 'E']) {
            Some(at) => (&plain[..at], &plain[at + 1..]),
            None => (plain.as_str(), "0"),
        };
        // An exponent beyond an `i32` is far beyond a float either way.
        let exponent = exponent
            .parse::<i32>()
            .unwrap_or(if exponent.starts_with('-') {
                -100_000
            } else {
                100_000
            });
        let decimals = mantissa
            .split_once('.')
            .map_or(0, |(_, fraction)| fraction.len());
        let last_digit = exponent.saturating_sub(i32::try_from(decimals).unwrap_or(i32::MAX));
        tokens.push(NumberToken {
            text: number.as_str().to_string(),
            value,
            tolerance: 0.5 * 10f64.powi(last_digit),
            percent: found.name("pct").is_some(),
        });
    }
    tokens
}

/// Whether `key` is within half a unit of the token's last digit. The extra
/// allowance is a few units of floating-point precision, far below any
/// digit a text can show, so a key that sits exactly on a rounding boundary
/// is accepted and one a whole digit away never is.
fn rounds_to(key: f64, token: &NumberToken) -> bool {
    let precision = 8.0 * f64::EPSILON * key.abs().max(token.value.abs());
    (key - token.value).abs() <= token.tolerance + precision
}

/// The numbers in `text` that no key number rounds to.
pub fn ungrounded_numbers(text: &str, key_numbers: &BTreeMap<String, f64>) -> Vec<String> {
    numbers_in(text)
        .into_iter()
        .filter(|token| {
            !(token.representable()
                && key_numbers
                    .values()
                    .filter(|key| key.is_finite())
                    .any(|key| {
                        rounds_to(*key, token) || (token.percent && rounds_to(key * 100.0, token))
                    }))
        })
        .map(|token| token.text)
        .collect()
}

/// True when every number in `text` is one of `key_numbers` after rounding.
pub fn claim_is_grounded(text: &str, key_numbers: &BTreeMap<String, f64>) -> bool {
    ungrounded_numbers(text, key_numbers).is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), *value))
            .collect()
    }

    #[test]
    fn numbers_are_found_with_their_precision() {
        let found =
            numbers_in("Rose from -1.50 to 2.635 in 12 steps (45.5% of 1,200), about 3e-2 each.");
        let seen: Vec<(f64, bool)> = found
            .iter()
            .map(|token| (token.value, token.percent))
            .collect();
        assert_eq!(
            seen,
            [
                (-1.5, false),
                (2.635, false),
                (12.0, false),
                (45.5, true),
                (1200.0, false),
                (0.03, false)
            ]
        );
        assert_eq!(found[0].tolerance, 0.005);
        assert_eq!(found[2].tolerance, 0.5);
        assert!((found[5].tolerance - 0.005).abs() < 1e-12);
    }

    #[test]
    fn names_and_code_spans_are_not_numbers() {
        let text =
            "`claude-sonnet-5-5` beat gpt-5.6-luna at gen_12 on island 0; v2 used top-3 with x1.5.";
        let found: Vec<f64> = numbers_in(text).iter().map(|token| token.value).collect();
        assert_eq!(found, [0.0]);
        assert_eq!(numbers_in("a range of 3-5 edits")[1].value, 5.0);
        assert_eq!(numbers_in("change of \u{2212}0.25")[0].value, -0.25);
    }

    /// Found by `the_number_scanner_reads_any_text` with the counterexample
    /// `"8e-987"` (EPHYRA_REPLAY=16493861309807371888:1103): a number too
    /// fine for a float read as exactly zero and was grounded by a key of
    /// zero, and one too large read as infinity and was grounded by any key.
    #[test]
    fn a_number_a_float_cannot_hold_is_never_grounded() {
        let numbers = keys(&[("zero", 0.0), ("share", 0.3), ("best", 2.5)]);
        for text in [
            "8e-987",
            "1e999",
            "-1e999",
            "1e99999999999",
            "1e-99999999999",
            "0e-987",
        ] {
            assert_eq!(
                ungrounded_numbers(text, &numbers),
                [text],
                "{text} was accepted"
            );
        }
        assert!(claim_is_grounded("0 and 0.0 and 0e0", &numbers));
        assert!(claim_is_grounded("3e-1", &numbers));
    }

    #[test]
    fn a_claim_is_grounded_only_by_key_numbers_under_rounding() {
        let numbers = keys(&[
            ("best", 2.634_917),
            ("share", 0.4567),
            ("steps", 12.0),
            ("delta", -0.125),
        ]);
        assert!(claim_is_grounded(
            "The best score was 2.635 (2.63 rounded) after 12 steps.",
            &numbers
        ));
        assert!(claim_is_grounded(
            "45.7% of proposals improved; the mean change was -0.13 or -0.12.",
            &numbers
        ));
        assert!(claim_is_grounded(
            "No numbers at all, only `gen_99`.",
            &numbers
        ));
        assert_eq!(
            ungrounded_numbers("The best score was 2.64 after 13 steps.", &numbers),
            ["2.64", "13"]
        );
        assert_eq!(
            ungrounded_numbers("0.4567% improved", &numbers),
            Vec::<String>::new(),
            "a share may be shown raw"
        );
        assert_eq!(ungrounded_numbers("46.7% improved", &numbers), ["46.7"]);
        assert_eq!(
            ungrounded_numbers("45.7 improved", &numbers),
            ["45.7"],
            "without a percent sign it is not a share"
        );
        assert!(!claim_is_grounded(
            "It scored 1.0.",
            &keys(&[("nan", f64::NAN)])
        ));
    }
}
