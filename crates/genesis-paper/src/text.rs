//! LaTeX escaping, and the light markup the writer model uses for prose.
//!
//! The model never writes LaTeX. It writes paragraphs with a small markup
//! (`*emphasis*`, `**strong**`, `` `code` ``, `$math$`, `[@key]` citations,
//! `### ` subsection headings, `- ` lists and `{{…}}` placeholders); this
//! module parses that markup, and everything that is not markup is escaped
//! before it reaches the TeX file.

use std::fmt::Write as _;

/// Latin letters with diacritics that `inputenc` and the T1 encoding typeset
/// directly.
const T1_LETTERS: &str = "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ\
ĂĄĆČĎĚĘĞĹĽŁŃŇŐŔŘŚŠŞŤŢŰŮŸŹŽŻăąćčďěęğĺľłńňőŕřśšşťţűůźžż";

const SYMBOLS: &[(char, &str)] = &[
    ('\u{a0}', "\\nobreakspace{}"),
    ('–', "\\textendash{}"),
    ('—', "\\textemdash{}"),
    ('‘', "\\textquoteleft{}"),
    ('’', "\\textquoteright{}"),
    ('“', "\\textquotedblleft{}"),
    ('”', "\\textquotedblright{}"),
    ('…', "\\ldots{}"),
    ('×', "\\texttimes{}"),
    ('−', "\\textminus{}"),
    ('±', "\\textpm{}"),
    ('°', "\\textdegree{}"),
    ('·', "\\textperiodcentered{}"),
    ('µ', "\\textmu{}"),
    ('≤', "\\ensuremath{\\le}"),
    ('≥', "\\ensuremath{\\ge}"),
    ('≈', "\\ensuremath{\\approx}"),
    ('≠', "\\ensuremath{\\neq}"),
    ('→', "\\ensuremath{\\rightarrow}"),
    ('←', "\\ensuremath{\\leftarrow}"),
    ('∞', "\\ensuremath{\\infty}"),
    ('α', "\\ensuremath{\\alpha}"),
    ('β', "\\ensuremath{\\beta}"),
    ('γ', "\\ensuremath{\\gamma}"),
    ('δ', "\\ensuremath{\\delta}"),
    ('ε', "\\ensuremath{\\epsilon}"),
    ('θ', "\\ensuremath{\\theta}"),
    ('λ', "\\ensuremath{\\lambda}"),
    ('μ', "\\ensuremath{\\mu}"),
    ('π', "\\ensuremath{\\pi}"),
    ('ρ', "\\ensuremath{\\rho}"),
    ('σ', "\\ensuremath{\\sigma}"),
    ('τ', "\\ensuremath{\\tau}"),
    ('φ', "\\ensuremath{\\phi}"),
    ('ω', "\\ensuremath{\\omega}"),
    ('Δ', "\\ensuremath{\\Delta}"),
    ('Σ', "\\ensuremath{\\Sigma}"),
    ('Ω', "\\ensuremath{\\Omega}"),
];

/// Escape arbitrary text for a LaTeX paragraph.
///
/// The result has balanced braces and no active character: every special is
/// replaced by a command, and a character the fonts cannot typeset becomes
/// a visible `U+XXXX` marker instead of a compile error. Distinct inputs
/// give distinct outputs. Braces are balanced by raw count, because the
/// text also goes into BibTeX fields and BibTeX does not treat `\{` as
/// escaped.
pub fn escape_latex(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 16);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\textbackslash{}"),
            '{' => out.push_str("\\textbraceleft{}"),
            '}' => out.push_str("\\textbraceright{}"),
            '$' => out.push_str("\\$"),
            '&' => out.push_str("\\&"),
            '#' => out.push_str("\\#"),
            '_' => out.push_str("\\_"),
            '%' => out.push_str("\\%"),
            '~' => out.push_str("\\textasciitilde{}"),
            '^' => out.push_str("\\textasciicircum{}"),
            '<' => out.push_str("\\textless{}"),
            '>' => out.push_str("\\textgreater{}"),
            '|' => out.push_str("\\textbar{}"),
            '`' => out.push_str("\\textasciigrave{}"),
            '\n' | '\t' => out.push(c),
            c if c.is_ascii() && !c.is_ascii_control() => out.push(c),
            c if T1_LETTERS.contains(c) => out.push(c),
            c => match SYMBOLS.iter().find(|(symbol, _)| *symbol == c) {
                Some((_, command)) => out.push_str(command),
                None => {
                    let _ = write!(out, "\\textsuperscript{{U+{:04X}}}", c as u32);
                }
            },
        }
    }
    out
}

/// Escape text for a place where a paragraph break is an error: a title, a
/// caption, a table cell. Whitespace runs collapse to one space.
pub fn escape_inline(text: &str) -> String {
    escape_latex(&text.split_whitespace().collect::<Vec<_>>().join(" "))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FloatKind {
    Table,
    Figure,
}

impl FloatKind {
    pub fn word(self) -> &'static str {
        match self {
            Self::Table => "Table",
            Self::Figure => "Figure",
        }
    }

    fn parse(tag: &str) -> Option<Self> {
        match tag {
            "TABLE" => Some(Self::Table),
            "FIGURE" => Some(Self::Figure),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Inline {
    Text(String),
    Emph(String),
    Strong(String),
    Code(String),
    /// Inline mathematics that passed [`is_safe_math`].
    Math(String),
    /// `[@a; @b]`.
    Cite(Vec<String>),
    /// `{{REF:TABLE:name}}`: a reference to a generated float.
    Ref(FloatKind, String),
    /// A `{{…}}` token this module does not know, kept verbatim so
    /// validation can report it.
    Unresolved(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    Heading(Vec<Inline>),
    Paragraph(Vec<Inline>),
    List(Vec<Vec<Inline>>),
    /// `{{TABLE:name}}` or `{{FIGURE:name}}` alone on a line.
    Float(FloatKind, String),
}

const MATH_COMMANDS: &[&str] = &[
    "alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda", "mu", "pi", "rho", "sigma",
    "tau", "phi", "omega", "Delta", "Sigma", "Omega", "frac", "sqrt", "sum", "prod", "cdot",
    "times", "le", "ge", "leq", "geq", "neq", "approx", "in", "infty", "mathbb", "mathrm",
    "mathbf", "text", "min", "max", "log", "exp", "left", "right", "ldots", "pm", "to", "bar",
    "hat",
];

/// Whether `content` may go between `$…$` unescaped: a short one-line
/// formula over a fixed alphabet, with balanced braces and only commands
/// from a known list.
pub fn is_safe_math(content: &str) -> bool {
    if content.trim().is_empty() || content.len() > 200 {
        return false;
    }
    let mut depth = 0i32;
    let chars: Vec<char> = content.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            '\\' => {
                let start = i + 1;
                let mut end = start;
                while end < chars.len() && chars[end].is_ascii_alphabetic() {
                    end += 1;
                }
                let name: String = chars[start..end].iter().collect();
                if !MATH_COMMANDS.contains(&name.as_str()) {
                    return false;
                }
                i = end;
                continue;
            }
            c if c.is_ascii_alphanumeric() => {}
            ' ' | '^' | '_' | '(' | ')' | '[' | ']' | '+' | '-' | '*' | '/' | '=' | '<' | '>'
            | '.' | ',' | '|' | '!' | ':' | '\'' => {}
            _ => return false,
        }
        i += 1;
    }
    depth == 0
}

fn is_cite_key(key: &str) -> bool {
    !key.is_empty()
        && key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | ':' | '.' | '-'))
}

fn is_float_name(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `{{TABLE:name}}` or `{{FIGURE:name}}`, when `token` is exactly that.
fn parse_float(token: &str) -> Option<(FloatKind, String)> {
    let inner = token.strip_prefix("{{")?.strip_suffix("}}")?;
    let (tag, name) = inner.split_once(':')?;
    let kind = FloatKind::parse(tag.trim())?;
    let name = name.trim();
    is_float_name(name).then(|| (kind, name.to_string()))
}

fn parse_ref(inner: &str) -> Option<(FloatKind, String)> {
    let rest = inner.strip_prefix("REF:")?;
    let (tag, name) = rest.split_once(':')?;
    let kind = FloatKind::parse(tag.trim())?;
    let name = name.trim();
    is_float_name(name).then(|| (kind, name.to_string()))
}

/// Parse one run of inline markup. Anything that does not close properly is
/// plain text.
pub fn parse_inline(text: &str) -> Vec<Inline> {
    let mut out: Vec<Inline> = Vec::new();
    let mut plain = String::new();
    let flush = |plain: &mut String, out: &mut Vec<Inline>| {
        if !plain.is_empty() {
            out.push(Inline::Text(std::mem::take(plain)));
        }
    };
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        let taken = (|| -> Option<(Inline, usize)> {
            if let Some(after) = rest.strip_prefix("{{") {
                let end = after.find("}}")?;
                let inner = &after[..end];
                let consumed = 2 + end + 2;
                return Some(match parse_ref(inner.trim()) {
                    Some((kind, name)) => (Inline::Ref(kind, name), consumed),
                    None => (Inline::Unresolved(rest[..consumed].to_string()), consumed),
                });
            }
            if let Some(after) = rest.strip_prefix("[@") {
                let end = after.find(']')?;
                let keys: Vec<String> = after[..end]
                    .split([';', ','])
                    .map(|key| key.trim().trim_start_matches('@').to_string())
                    .collect();
                return keys
                    .iter()
                    .all(|key| is_cite_key(key))
                    .then(|| (Inline::Cite(keys), 2 + end + 1));
            }
            if let Some(after) = rest.strip_prefix('`') {
                let end = after.find('`')?;
                return (end > 0).then(|| (Inline::Code(after[..end].to_string()), end + 2));
            }
            if let Some(after) = rest.strip_prefix('$') {
                let end = after.find('$')?;
                let body = &after[..end];
                return is_safe_math(body).then(|| (Inline::Math(body.to_string()), end + 2));
            }
            if let Some(after) = rest.strip_prefix("**") {
                let end = after.find("**")?;
                let body = &after[..end];
                return (!body.trim().is_empty())
                    .then(|| (Inline::Strong(body.to_string()), end + 4));
            }
            if let Some(after) = rest.strip_prefix('*') {
                let end = after.find('*')?;
                let body = &after[..end];
                let tight = !body.is_empty() && !body.starts_with(' ') && !body.ends_with(' ');
                return tight.then(|| (Inline::Emph(body.to_string()), end + 2));
            }
            None
        })();
        match taken {
            Some((inline, consumed)) => {
                flush(&mut plain, &mut out);
                out.push(inline);
                rest = &rest[consumed..];
            }
            None => {
                plain.push(c);
                rest = &rest[c.len_utf8()..];
            }
        }
    }
    flush(&mut plain, &mut out);
    out
}

/// Parse a section's prose into blocks. Blank lines separate blocks.
pub fn parse_prose(text: &str) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut paragraph: Vec<&str> = Vec::new();
    let mut list: Vec<Vec<Inline>> = Vec::new();
    fn close(paragraph: &mut Vec<&str>, list: &mut Vec<Vec<Inline>>, blocks: &mut Vec<Block>) {
        if !paragraph.is_empty() {
            blocks.push(Block::Paragraph(parse_inline(&paragraph.join(" "))));
            paragraph.clear();
        }
        if !list.is_empty() {
            blocks.push(Block::List(std::mem::take(list)));
        }
    }
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            close(&mut paragraph, &mut list, &mut blocks);
        } else if let Some((kind, name)) = parse_float(trimmed) {
            close(&mut paragraph, &mut list, &mut blocks);
            blocks.push(Block::Float(kind, name));
        } else if trimmed.starts_with('#') {
            close(&mut paragraph, &mut list, &mut blocks);
            blocks.push(Block::Heading(parse_inline(
                trimmed.trim_start_matches('#').trim(),
            )));
        } else if let Some(item) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
        {
            if !paragraph.is_empty() {
                close(&mut paragraph, &mut list, &mut blocks);
            }
            list.push(parse_inline(item.trim()));
        } else {
            if !list.is_empty() {
                close(&mut paragraph, &mut list, &mut blocks);
            }
            paragraph.push(trimmed);
        }
    }
    close(&mut paragraph, &mut list, &mut blocks);
    blocks
}

/// Every inline of a block list, in reading order.
pub fn inlines(blocks: &[Block]) -> impl Iterator<Item = &Inline> {
    blocks
        .iter()
        .flat_map(|block| -> Box<dyn Iterator<Item = &Inline> + '_> {
            match block {
                Block::Heading(items) | Block::Paragraph(items) => Box::new(items.iter()),
                Block::List(rows) => Box::new(rows.iter().flatten()),
                Block::Float(..) => Box::new(std::iter::empty()),
            }
        })
}

/// The characters a reader would see, for length checks.
pub fn visible_len(blocks: &[Block]) -> usize {
    inlines(blocks)
        .map(|inline| match inline {
            Inline::Text(text)
            | Inline::Emph(text)
            | Inline::Strong(text)
            | Inline::Code(text)
            | Inline::Math(text) => text.chars().filter(|c| !c.is_whitespace()).count(),
            _ => 0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specials_become_commands_and_unknown_characters_become_markers() {
        assert_eq!(
            escape_latex("50% of $x_1 & {y}"),
            "50\\% of \\$x\\_1 \\& \\textbraceleft{}y\\textbraceright{}"
        );
        assert_eq!(
            escape_latex("a\\b~c^d"),
            "a\\textbackslash{}b\\textasciitilde{}c\\textasciicircum{}d"
        );
        assert_eq!(escape_latex("Müller – “ok” ≤ α"), "Müller \\textendash{} \\textquotedblleft{}ok\\textquotedblright{} \\ensuremath{\\le} \\ensuremath{\\alpha}");
        assert_eq!(
            escape_latex("楼\u{b}"),
            "\\textsuperscript{U+697C}\\textsuperscript{U+000B}"
        );
        assert_eq!(escape_inline("  a\n\n b\t# "), "a b \\#");
    }

    #[test]
    fn inline_markup_is_recognised_and_broken_markup_is_text() {
        let parsed = parse_inline("We *improve* the **best** score with `restart()` [@smith2020; @lee2021] in {{REF:TABLE:results}} where $x^2 \\le 1$.");
        assert_eq!(
            parsed,
            vec![
                Inline::Text("We ".into()),
                Inline::Emph("improve".into()),
                Inline::Text(" the ".into()),
                Inline::Strong("best".into()),
                Inline::Text(" score with ".into()),
                Inline::Code("restart()".into()),
                Inline::Text(" ".into()),
                Inline::Cite(vec!["smith2020".into(), "lee2021".into()]),
                Inline::Text(" in ".into()),
                Inline::Ref(FloatKind::Table, "results".into()),
                Inline::Text(" where ".into()),
                Inline::Math("x^2 \\le 1".into()),
                Inline::Text(".".into()),
            ]
        );
        assert_eq!(
            parse_inline("5 * 3 and a lone ` or $ or [@bad key]"),
            vec![Inline::Text("5 * 3 and a lone ` or $ or [@bad key]".into())]
        );
        assert_eq!(
            parse_inline("{{TABLE:results}} inline"),
            vec![
                Inline::Unresolved("{{TABLE:results}}".into()),
                Inline::Text(" inline".into())
            ]
        );
        assert_eq!(
            parse_inline("$\\input{x}$"),
            vec![Inline::Text("$\\input{x}$".into())]
        );
    }

    #[test]
    fn math_is_only_accepted_from_a_fixed_vocabulary() {
        assert!(is_safe_math("\\frac{a}{b} + \\sqrt{x_1^2}"));
        assert!(!is_safe_math("\\write18{rm}"));
        assert!(!is_safe_math("a } b {"));
        assert!(!is_safe_math("x & y"));
        assert!(!is_safe_math("  "));
    }

    #[test]
    fn blocks_follow_blank_lines_headings_lists_and_floats() {
        let blocks = parse_prose("First line\nsame paragraph.\n\n### Setup\nAfter heading.\n- one\n- two *x*\n\n{{FIGURE:improvement}}\n\nLast.");
        assert_eq!(blocks.len(), 6);
        assert_eq!(
            blocks[0],
            Block::Paragraph(vec![Inline::Text("First line same paragraph.".into())])
        );
        assert_eq!(
            blocks[1],
            Block::Heading(vec![Inline::Text("Setup".into())])
        );
        assert!(matches!(&blocks[3], Block::List(rows) if rows.len() == 2));
        assert_eq!(
            blocks[4],
            Block::Float(FloatKind::Figure, "improvement".into())
        );
        assert_eq!(
            visible_len(&blocks),
            "Firstlinesameparagraph.SetupAfterheading.onetwoxLast.".len()
        );
        assert!(parse_prose(" \n\n").is_empty());
    }
}
