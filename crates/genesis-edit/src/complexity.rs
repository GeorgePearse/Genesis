//! Code metrics that summarise how complex a program is.
//!
//! One number, `complexity_score` in `[0, 1]`, is stored with every program
//! and can be traded off against its fitness. It is a weighted sum of
//! cyclomatic complexity, Halstead volume, size and nesting depth.
//!
//! Three analysers exist, as in ShinkaEvolve's `complexity.py`:
//!
//! * **Python** is tokenised and its block structure read from indentation.
//!   ShinkaEvolve uses the `radon` library on the parsed syntax tree; this is
//!   an approximation of it that needs no parser, and it follows radon's
//!   rules where they are surprising (a nested function or class is not
//!   scored, an `assert` is one decision whatever it tests, a line with a
//!   colon in the middle is two logical lines). Against radon 6.0.1 on 1,400
//!   generated programs it gave the same cyclomatic complexity, line counts,
//!   comment count and nesting depth on every one, a Halstead volume 4% off
//!   on average, and a `complexity_score` within 0.02 on every one. The
//!   volume differs because radon names an operand that is itself an
//!   expression by its node type, which takes operator precedence to know.
//!   Text that does not tokenise as Python is measured by the next analyser,
//!   as code that fails to parse is in the original; code in another
//!   language that happens to tokenise is not detected.
//! * **Brace languages** (C, C++, CUDA, Go, Rust, and here JavaScript and
//!   TypeScript) count decision keywords and brace depth.
//! * **Anything else** is scored by its number of lines.

use crate::languages::language;
use crate::text::split_lines;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::LazyLock;

/// What the analysers report. Field names match ShinkaEvolve's metrics
/// dictionary, which is stored in program metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct CodeMetrics {
    pub cyclomatic_complexity: f64,
    pub average_cyclomatic_complexity: f64,
    pub halstead_volume: f64,
    pub halstead_difficulty: f64,
    pub halstead_effort: f64,
    pub lines_of_code: u64,
    pub logical_lines_of_code: u64,
    pub comments: u64,
    pub maintainability_index: f64,
    pub max_nesting_depth: u64,
    /// Weighted, normalised sum of the above, in `[0, 1]`.
    pub complexity_score: f64,
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

fn log2_or_zero(value: f64) -> f64 {
    if value > 0.0 {
        value.log2()
    } else {
        0.0
    }
}

/// Measure `code` with the analyser for `language_name`.
pub fn analyze_code_metrics(code: &str, language_name: &str) -> CodeMetrics {
    let canonical = language(language_name).map_or_else(
        || language_name.trim().to_lowercase(),
        |spec| spec.name.into(),
    );
    match canonical.as_str() {
        "python" => python_metrics(code).unwrap_or_else(|| brace_metrics(code)),
        "cpp" | "c" | "cuda" | "go" | "rust" | "javascript" | "typescript" => brace_metrics(code),
        _ => generic_metrics(code),
    }
}

static DECISION_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"\bif\b",
        r"\belse\b",
        r"\bwhile\b",
        r"\bfor\b",
        r"\bswitch\b",
        r"\bcase\b",
        r"\bcatch\b",
        r"\b\?\b",
    ]
    .iter()
    .map(|pattern| Regex::new(&format!("(?i){pattern}")).expect("decision pattern"))
    .collect()
});

/// Keyword and brace counting for C-like code.
fn brace_metrics(code: &str) -> CodeMetrics {
    let lines: Vec<&str> = code.split('\n').collect();
    let loc = lines.len() as u64;
    let (mut lloc, mut comments) = (0u64, 0u64);
    for line in &lines {
        let stripped = crate::text::strip(line);
        if stripped.is_empty() {
            continue;
        }
        if stripped.starts_with("//") || stripped.starts_with("/*") || stripped.ends_with("*/") {
            comments += 1;
        } else {
            lloc += 1;
        }
    }
    let total_cc = 1.0
        + DECISION_PATTERNS
            .iter()
            .map(|pattern| pattern.find_iter(code).count())
            .sum::<usize>() as f64;

    let (mut max_nesting, mut nesting) = (0u64, 0u64);
    for c in code.chars() {
        match c {
            '{' => {
                nesting += 1;
                max_nesting = max_nesting.max(nesting);
            }
            '}' => nesting = nesting.saturating_sub(1),
            _ => {}
        }
    }

    let volume = (lloc as f64 * total_cc.max(1.0).log2()).max(1.0);
    let mi = (171.0
        - 5.2 * volume.max(1.0).log2()
        - 0.23 * total_cc
        - 16.2 * (loc as f64).max(1.0).log2())
    .max(0.0);
    let score = 0.4 * (total_cc / 10.0).min(1.0)
        + 0.4 * ((volume + 1.0).log2() / 10.0).min(1.0)
        + 0.1 * ((loc as f64 + 1.0).log2() / 10.0).min(1.0)
        + 0.1 * (max_nesting as f64 / 5.0).min(1.0);
    CodeMetrics {
        cyclomatic_complexity: total_cc,
        average_cyclomatic_complexity: total_cc,
        halstead_volume: volume,
        halstead_difficulty: 1.0,
        halstead_effort: volume,
        lines_of_code: loc,
        logical_lines_of_code: lloc,
        comments,
        maintainability_index: mi,
        max_nesting_depth: max_nesting,
        complexity_score: round3(score.min(1.0)),
    }
}

/// Line counting for languages with no analyser.
fn generic_metrics(code: &str) -> CodeMetrics {
    let loc = code
        .split('\n')
        .filter(|line| !crate::text::is_blank(line))
        .count() as u64;
    let size = (loc as f64).max(1.0);
    CodeMetrics {
        cyclomatic_complexity: 1.0,
        average_cyclomatic_complexity: 1.0,
        halstead_volume: size,
        halstead_difficulty: 1.0,
        halstead_effort: size,
        lines_of_code: loc,
        logical_lines_of_code: loc,
        comments: 0,
        maintainability_index: 100.0,
        max_nesting_depth: 1,
        complexity_score: round3((size.log2() / 10.0).min(1.0)),
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Name(String),
    /// A number or string literal, with its source text.
    Literal(String),
    Op(&'static str),
}

impl Token {
    fn is_op(&self, op: &str) -> bool {
        matches!(self, Token::Op(text) if *text == op)
    }

    fn is_name(&self, name: &str) -> bool {
        matches!(self, Token::Name(text) if text == name)
    }
}

#[derive(Debug)]
struct LogicalLine {
    indent: usize,
    tokens: Vec<Token>,
}

/// Longest first, so that `**=` is not read as `*` `*` `=`.
const OPERATORS: &[&str] = &[
    "**=", "//=", ">>=", "<<=", "...", "!=", "==", "<=", ">=", "->", ":=", "+=", "-=", "*=", "/=",
    "%=", "&=", "|=", "^=", "@=", "**", "//", "<<", ">>", "+", "-", "*", "/", "%", "@", "&", "|",
    "^", "~", "<", ">", "(", ")", "[", "]", "{", "}", ",", ":", ".", ";", "=",
];

/// Split Python source into logical lines of tokens and count its comments.
/// `None` when the text cannot be Python: a bracket that never closes, a
/// string that never ends, a character Python has no use for.
fn tokenize_python(code: &str) -> Option<(Vec<LogicalLine>, u64)> {
    let chars: Vec<char> = code.chars().collect();
    let mut lines: Vec<LogicalLine> = Vec::new();
    let mut current: Option<LogicalLine> = None;
    let mut brackets: Vec<char> = Vec::new();
    let mut comments = 0u64;
    let mut at_line_start = true;
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        if at_line_start && brackets.is_empty() && current.is_none() {
            let mut indent = 0;
            while i < chars.len() && (chars[i] == ' ' || chars[i] == '\t' || chars[i] == '\u{c}') {
                indent = if chars[i] == '\t' {
                    (indent / 8 + 1) * 8
                } else {
                    indent + 1
                };
                i += 1;
            }
            at_line_start = false;
            if i < chars.len() && !matches!(chars[i], '\n' | '\r' | '#') {
                current = Some(LogicalLine {
                    indent,
                    tokens: Vec::new(),
                });
            }
            continue;
        }
        match c {
            '\n' | '\r' => {
                i += 1;
                if brackets.is_empty() {
                    lines.extend(current.take().filter(|line| !line.tokens.is_empty()));
                    at_line_start = true;
                }
            }
            ' ' | '\t' | '\u{c}' => i += 1,
            '\\' if matches!(chars.get(i + 1), Some('\n' | '\r')) => {
                i += 2;
                if chars.get(i - 1) == Some(&'\r') && chars.get(i) == Some(&'\n') {
                    i += 1;
                }
            }
            '#' => {
                comments += 1;
                while i < chars.len() && !matches!(chars[i], '\n' | '\r') {
                    i += 1;
                }
            }
            _ => {
                let line = current.get_or_insert_with(|| LogicalLine {
                    indent: 0,
                    tokens: Vec::new(),
                });
                if let Some(end) = string_literal_end(&chars, i)? {
                    line.tokens
                        .push(Token::Literal(chars[i..end].iter().collect()));
                    i = end;
                } else if c.is_ascii_digit()
                    || (c == '.' && chars.get(i + 1).is_some_and(char::is_ascii_digit))
                {
                    let start = i;
                    while i < chars.len()
                        && (chars[i].is_alphanumeric()
                            || matches!(chars[i], '_' | '.')
                            || (matches!(chars[i], '+' | '-') && matches!(chars[i - 1], 'e' | 'E')))
                    {
                        i += 1;
                    }
                    line.tokens
                        .push(Token::Literal(chars[start..i].iter().collect()));
                } else if c.is_alphabetic() || c == '_' {
                    let start = i;
                    while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                        i += 1;
                    }
                    line.tokens
                        .push(Token::Name(chars[start..i].iter().collect()));
                } else {
                    let op = OPERATORS
                        .iter()
                        .find(|op| op.chars().eq(chars[i..].iter().take(op.len()).copied()))?;
                    match *op {
                        "(" | "[" | "{" => brackets.push(c),
                        ")" | "]" | "}" => {
                            let open = brackets.pop()?;
                            if !matches!((open, c), ('(', ')') | ('[', ']') | ('{', '}')) {
                                return None;
                            }
                        }
                        _ => {}
                    }
                    line.tokens.push(Token::Op(op));
                    i += op.len();
                }
            }
        }
    }
    if !brackets.is_empty() {
        return None;
    }
    lines.extend(current.take().filter(|line| !line.tokens.is_empty()));
    Some((lines, comments))
}

/// If a string literal starts at `start`, the index just past it.
/// `Some(None)` when no string starts there; `None` when one starts but
/// never ends.
fn string_literal_end(chars: &[char], start: usize) -> Option<Option<usize>> {
    let mut quote_at = start;
    while quote_at < chars.len() && quote_at - start < 2 && "rRbBuUfF".contains(chars[quote_at]) {
        quote_at += 1;
    }
    let Some(quote) = chars
        .get(quote_at)
        .copied()
        .filter(|c| matches!(c, '"' | '\''))
    else {
        return Some(None);
    };
    let triple = chars.get(quote_at + 1) == Some(&quote) && chars.get(quote_at + 2) == Some(&quote);
    let mut i = quote_at + if triple { 3 } else { 1 };
    while i < chars.len() {
        match chars[i] {
            '\\' => i += 2,
            c if c == quote => {
                if !triple {
                    return Some(Some(i + 1));
                }
                if chars.get(i + 1) == Some(&quote) && chars.get(i + 2) == Some(&quote) {
                    return Some(Some(i + 3));
                }
                i += 1;
            }
            '\n' | '\r' if !triple => return None,
            _ => i += 1,
        }
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Scope {
    Module,
    Class(usize),
    Function(usize),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Header {
    If,
    Loop,
    Try,
    Other,
}

#[derive(Debug, Clone, Copy)]
struct Frame {
    indent: usize,
    /// Nesting depth of this block, counting the blocks radon's visitor
    /// counts: `if`, loops, `with`, `try` and functions.
    depth: u64,
    scope: Scope,
    header: Header,
}

#[derive(Debug, Default)]
struct Function {
    name: String,
    complexity: f64,
    /// Defined inside another function, or a method of a nested class.
    nested: bool,
}

#[derive(Debug, Default)]
struct Class {
    body_complexity: f64,
    methods: Vec<usize>,
    /// Defined inside a function or another class.
    nested: bool,
}

const COMPOUND_KEYWORDS: &[&str] = &[
    "if", "elif", "else", "for", "while", "try", "except", "finally", "with", "def", "class",
    "match", "case",
];

/// Decision points on one logical line. An `assert` is one, whatever its
/// condition holds: radon does not look inside it.
fn decisions_in(tokens: &[Token]) -> f64 {
    let mut count = 0.0;
    let mut in_assert = false;
    for token in tokens {
        if token.is_op(";") {
            in_assert = false;
        } else if in_assert {
            continue;
        } else if token.is_name("assert") {
            in_assert = true;
            count += 1.0;
        } else if ["if", "elif", "for", "while", "except", "and", "or"]
            .iter()
            .any(|keyword| token.is_name(keyword))
        {
            count += 1.0;
        }
    }
    count
}

/// Logical lines radon counts on one logical line, by its own rule: each
/// `;`-separated part is one, or two when it holds a colon that does not end
/// it. The rule is about colons, not statements, so a dictionary literal, a
/// slice or a lambda also makes a line count twice. radon tests "ends it"
/// one token short on every part but the last, which is reproduced here.
fn statements_in(tokens: &[Token]) -> u64 {
    let parts: Vec<&[Token]> = tokens.split(|token| token.is_op(";")).collect();
    let last_part = parts.len() - 1;
    parts
        .iter()
        .enumerate()
        .map(
            |(index, part)| match part.iter().rposition(|token| token.is_op(":")) {
                _ if part.is_empty() => 0,
                None => 1,
                Some(colon) if index == last_part => 2 - u64::from(colon + 1 == part.len()),
                Some(colon) => 2 - u64::from(colon + 2 == part.len()),
            },
        )
        .sum()
}

/// Index of the colon that ends a compound statement's header.
fn header_colon(tokens: &[Token]) -> Option<usize> {
    let mut depth = 0i32;
    for (index, token) in tokens.iter().enumerate() {
        match token {
            Token::Op("(" | "[" | "{") => depth += 1,
            Token::Op(")" | "]" | "}") => depth -= 1,
            Token::Op(":") if depth == 0 => return Some(index),
            _ => {}
        }
    }
    None
}

/// Keywords after which `-` negates and `*` unpacks.
const PREFIX_KEYWORDS: &[&str] = &[
    "return", "yield", "in", "is", "not", "and", "or", "if", "else", "elif", "while", "assert",
    "lambda", "await", "del", "raise", "from", "import", "for",
];

const BINARY_OPERATORS: &[&str] = &[
    "+", "-", "*", "/", "//", "%", "**", "@", "<<", ">>", "&", "|", "^", "<", ">", "<=", ">=",
    "==", "!=", "+=", "-=", "*=", "/=", "//=", "%=", "**=", "@=", "<<=", ">>=", "&=", "|=", "^=",
];

/// The operand to the left of an operator, named as radon names it.
fn left_operand(tokens: &[Token], index: usize) -> Option<String> {
    match tokens.get(index.checked_sub(1)?)? {
        Token::Name(name) => Some(name.clone()),
        Token::Literal(text) => Some(text.clone()),
        Token::Op(")") => Some("Call".into()),
        Token::Op("]") => Some("Subscript".into()),
        Token::Op("}") => Some("Dict".into()),
        Token::Op(_) => None,
    }
}

/// The operand to the right of an operator.
fn right_operand(tokens: &[Token], index: usize) -> Option<String> {
    match tokens.get(index + 1)? {
        Token::Literal(text) => Some(text.clone()),
        Token::Op("(") => Some("Tuple".into()),
        Token::Op("[") => Some("List".into()),
        Token::Op("{") => Some("Dict".into()),
        Token::Op("-" | "+" | "~") => Some("UnaryOp".into()),
        Token::Op(_) => None,
        Token::Name(name) if name == "not" => Some("UnaryOp".into()),
        Token::Name(name) if name == "lambda" => Some("Lambda".into()),
        Token::Name(name) => {
            let mut last = name.clone();
            let mut at = index + 2;
            while tokens.get(at).is_some_and(|token| token.is_op(".")) {
                match tokens.get(at + 1) {
                    Some(Token::Name(attribute)) => last = attribute.clone(),
                    _ => break,
                }
                at += 2;
            }
            match tokens.get(at) {
                Some(Token::Op("(")) => Some("Call".into()),
                Some(Token::Op("[")) => Some("Subscript".into()),
                _ => Some(last),
            }
        }
    }
}

#[derive(Default)]
struct Halstead {
    operators: f64,
    operands: f64,
    distinct_operators: HashSet<String>,
    distinct_operands: HashSet<(Option<String>, String)>,
}

impl Halstead {
    fn operator(&mut self, name: &str) {
        self.operators += 1.0;
        self.distinct_operators.insert(name.to_string());
    }

    fn operand(&mut self, context: &Option<String>, name: Option<String>) {
        if let Some(name) = name {
            self.operands += 1.0;
            self.distinct_operands.insert((context.clone(), name));
        }
    }

    /// Count the operators of one logical line with the operands next to
    /// them. `context` is the enclosing function: radon tells operands of
    /// different functions apart.
    fn count_line(&mut self, tokens: &[Token], context: &Option<String>) {
        let mut pending_for_in = 0usize;
        for (index, token) in tokens.iter().enumerate() {
            let previous = index.checked_sub(1).map(|i| &tokens[i]);
            // After another operator, an opening bracket or a keyword, `-`
            // negates and `*` unpacks.
            let prefix_position = match previous {
                None => true,
                Some(Token::Op(op)) => !matches!(*op, ")" | "]" | "}"),
                Some(Token::Name(name)) => PREFIX_KEYWORDS.contains(&name.as_str()),
                Some(Token::Literal(_)) => false,
            };
            match token {
                Token::Name(name) if name == "for" => pending_for_in += 1,
                Token::Name(name) if name == "in" => {
                    if previous.is_some_and(|token| token.is_name("not")) {
                        continue;
                    }
                    if pending_for_in > 0 {
                        pending_for_in -= 1;
                        continue;
                    }
                    self.operator("In");
                    self.operand(context, left_operand(tokens, index));
                    self.operand(context, right_operand(tokens, index));
                }
                Token::Name(name) if name == "not" => {
                    if tokens.get(index + 1).is_some_and(|next| next.is_name("in")) {
                        self.operator("NotIn");
                        self.operand(context, left_operand(tokens, index));
                        self.operand(context, right_operand(tokens, index + 1));
                    } else if !previous.is_some_and(|token| token.is_name("is")) {
                        self.operator("Not");
                        self.operand(context, right_operand(tokens, index));
                    }
                }
                Token::Name(name) if name == "is" => {
                    let negated = tokens
                        .get(index + 1)
                        .is_some_and(|next| next.is_name("not"));
                    self.operator(if negated { "IsNot" } else { "Is" });
                    self.operand(context, left_operand(tokens, index));
                    self.operand(context, right_operand(tokens, index + usize::from(negated)));
                }
                Token::Name(name) if name == "and" || name == "or" => {
                    self.operator(name);
                    self.operand(context, left_operand(tokens, index));
                    self.operand(context, right_operand(tokens, index));
                }
                Token::Op("~") => {
                    self.operator("Invert");
                    self.operand(context, right_operand(tokens, index));
                }
                Token::Op(op @ ("-" | "+")) if prefix_position => {
                    self.operator(if *op == "-" { "USub" } else { "UAdd" });
                    self.operand(context, right_operand(tokens, index));
                }
                Token::Op("*" | "**" | "@") if prefix_position => {}
                Token::Op(op) if BINARY_OPERATORS.contains(op) => {
                    // `x += 1` uses the same operator as `x + 1`.
                    let is_comparison = matches!(*op, "<=" | ">=" | "==" | "!=");
                    self.operator(if is_comparison {
                        op
                    } else {
                        op.trim_end_matches('=')
                    });
                    self.operand(context, left_operand(tokens, index));
                    self.operand(context, right_operand(tokens, index));
                }
                _ => {}
            }
        }
    }
}

/// Token-level approximation of radon's metrics for Python source.
fn python_metrics(code: &str) -> Option<CodeMetrics> {
    let (lines, comments) = tokenize_python(code)?;

    let mut functions: Vec<Function> = Vec::new();
    let mut classes: Vec<Class> = Vec::new();
    let mut frames: Vec<Frame> = Vec::new();
    let mut halstead = Halstead::default();
    let mut max_depth = 0u64;
    let mut lloc = 0u64;
    let mut previous_opens_block = false;
    let mut indents: Vec<usize> = vec![0];

    for line in &lines {
        // Indentation may only grow after a line that opens a block, and may
        // only shrink back to a level already in use.
        let current_indent = *indents.last().expect("indent stack");
        if line.indent > current_indent {
            if !previous_opens_block {
                return None;
            }
            indents.push(line.indent);
        } else {
            while indents.last().is_some_and(|indent| *indent > line.indent) {
                indents.pop();
            }
            if indents.last() != Some(&line.indent) {
                return None;
            }
        }
        lloc += statements_in(&line.tokens);

        let mut continued: Option<Frame> = None;
        while frames
            .last()
            .is_some_and(|frame| frame.indent >= line.indent)
        {
            let frame = frames.pop().expect("frame");
            if frame.indent == line.indent {
                continued = Some(frame);
            }
        }
        let enclosing = frames.last().copied();
        let scope = enclosing.map_or(Scope::Module, |frame| frame.scope);
        let depth = enclosing.map_or(0, |frame| frame.depth);

        let keyword_at = usize::from(
            line.tokens
                .first()
                .is_some_and(|token| token.is_name("async")),
        );
        let keyword = match line.tokens.get(keyword_at) {
            Some(Token::Name(name)) => name.as_str(),
            _ => "",
        };
        let name_after = || match line.tokens.get(keyword_at + 1) {
            Some(Token::Name(name)) => name.clone(),
            _ => String::new(),
        };
        let opens_block = line.tokens.last().is_some_and(|token| token.is_op(":"))
            || (COMPOUND_KEYWORDS.contains(&keyword)
                && line.tokens.iter().any(|token| token.is_op(":")));

        let mut line_scope = scope;
        let frame = match keyword {
            "def" => {
                let nested = match scope {
                    Scope::Function(_) => true,
                    Scope::Class(class) => classes[class].nested,
                    Scope::Module => false,
                };
                functions.push(Function {
                    name: name_after(),
                    complexity: 1.0,
                    nested,
                });
                let index = functions.len() - 1;
                if let Scope::Class(class) = scope {
                    classes[class].methods.push(index);
                }
                line_scope = Scope::Function(index);
                Some(Frame {
                    indent: line.indent,
                    depth: depth + 1,
                    scope: line_scope,
                    header: Header::Other,
                })
            }
            "class" => {
                classes.push(Class {
                    body_complexity: 1.0,
                    methods: Vec::new(),
                    nested: scope != Scope::Module,
                });
                line_scope = Scope::Class(classes.len() - 1);
                Some(Frame {
                    indent: line.indent,
                    depth,
                    scope: line_scope,
                    header: Header::Other,
                })
            }
            "if" | "for" | "while" | "with" | "try" => {
                let header = match keyword {
                    "if" => Header::If,
                    "for" | "while" => Header::Loop,
                    "try" => Header::Try,
                    _ => Header::Other,
                };
                // radon's nesting visitor does not visit `async for` or `async with`.
                let counted = u64::from(keyword_at == 0);
                Some(Frame {
                    indent: line.indent,
                    depth: depth + counted,
                    scope,
                    header,
                })
            }
            // `elif` is an `if` nested in the `else` of the one before it.
            "elif" => continued.map(|before| Frame {
                depth: before.depth + 1,
                ..before
            }),
            "else" | "except" | "finally" => continued,
            _ if opens_block => Some(Frame {
                indent: line.indent,
                depth,
                scope,
                header: Header::Other,
            }),
            _ => None,
        };

        // A function's signature is evaluated outside it: only what follows
        // the header counts towards the function.
        let counted_from = if keyword == "def" {
            header_colon(&line.tokens).map_or(line.tokens.len(), |c| c + 1)
        } else {
            0
        };
        let mut decisions = decisions_in(&line.tokens[counted_from..]);
        // `else` after a loop or a `try` is one more path; after `if` it is
        // the path already counted.
        if keyword == "else"
            && continued.is_some_and(|before| matches!(before.header, Header::Loop | Header::Try))
        {
            decisions += 1.0;
        }
        match line_scope {
            Scope::Function(index) => functions[index].complexity += decisions,
            Scope::Class(index) => classes[index].body_complexity += decisions,
            Scope::Module => {}
        }
        let context = match line_scope {
            Scope::Function(index) => Some(functions[index].name.clone()),
            _ => None,
        };
        halstead.count_line(&line.tokens, &context);

        if let Some(frame) = frame {
            max_depth = max_depth.max(frame.depth);
            if opens_block && line.tokens.last().is_some_and(|token| token.is_op(":")) {
                frames.push(frame);
            }
        }
        previous_opens_block = line.tokens.last().is_some_and(|token| token.is_op(":"));
    }

    // radon scores top-level functions, classes and their methods. A
    // function or class defined inside another is neither a block of its own
    // nor added to the one around it.
    let mut blocks: Vec<f64> = functions
        .iter()
        .filter(|function| !function.nested)
        .map(|function| function.complexity)
        .collect();
    for class in classes.iter().filter(|class| !class.nested) {
        let methods: f64 = class
            .methods
            .iter()
            .map(|index| functions[*index].complexity)
            .sum();
        let real = class.body_complexity + methods;
        blocks.push(match class.methods.len() {
            0 => real,
            count => (real / count as f64).trunc() + f64::from(u8::from(count > 1)),
        });
    }
    let total_cc: f64 = blocks.iter().sum();
    let average_cc = if blocks.is_empty() {
        0.0
    } else {
        total_cc / blocks.len() as f64
    };

    let vocabulary = (halstead.distinct_operators.len() + halstead.distinct_operands.len()) as f64;
    let length = halstead.operators + halstead.operands;
    let volume = if vocabulary > 0.0 {
        length * vocabulary.log2()
    } else {
        0.0
    };
    let difficulty = match halstead.distinct_operands.len() {
        0 => 0.0,
        distinct => {
            halstead.distinct_operators.len() as f64 * halstead.operands / (2.0 * distinct as f64)
        }
    };

    let loc = split_lines(code).len() as u64;
    let mi = 171.0 - 5.2 * log2_or_zero(volume) - 0.23 * total_cc - 16.2 * log2_or_zero(loc as f64);
    let score = 0.4 * total_cc / 10.0
        + 0.4 * (volume + 1.0).log2() / 10.0
        + 0.1 * (loc as f64 + 1.0).log2() / 10.0
        + 0.1 * max_depth as f64 / 5.0;
    Some(CodeMetrics {
        cyclomatic_complexity: total_cc,
        average_cyclomatic_complexity: average_cc,
        halstead_volume: volume,
        halstead_difficulty: difficulty,
        halstead_effort: difficulty * volume,
        lines_of_code: loc,
        logical_lines_of_code: lloc,
        comments,
        maintainability_index: mi,
        max_nesting_depth: max_depth,
        complexity_score: round3(score.min(1.0)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PYTHON_PROGRAM: &str =
        "def classify(values):\n    total = 0\n    for value in values:\n        if value > 0:\n            total += value\n    return total\n";

    const GO_PROGRAM: &str = "package main\n\nfunc score(xs []int) int {\n    total := 0\n    for _, x := range xs {\n        if x > 0 {\n            total += x\n        }\n    }\n    return total\n}\n";

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn python_complexity_metrics_contract() {
        let m = analyze_code_metrics(PYTHON_PROGRAM, "python");
        assert_eq!(
            (m.cyclomatic_complexity, m.average_cyclomatic_complexity),
            (3.0, 3.0)
        );
        assert!(close(m.halstead_volume, 13.931568569324174), "{m:?}");
        assert!(close(m.halstead_difficulty, 1.3333333333333333), "{m:?}");
        assert!(close(m.halstead_effort, 18.575424759098897), "{m:?}");
        assert_eq!(
            (m.lines_of_code, m.logical_lines_of_code, m.comments),
            (6, 6, 0)
        );
        assert!(close(m.maintainability_index, 108.67212134673596), "{m:?}");
        assert_eq!(m.max_nesting_depth, 3);
        assert_eq!(m.complexity_score, 0.364);
        assert_eq!(analyze_code_metrics(PYTHON_PROGRAM, "py"), m);
    }

    #[test]
    fn invalid_python_preserves_cpp_fallback_metrics() {
        let m = analyze_code_metrics("def broken(:\n    pass\n", "python");
        assert_eq!(
            (m.cyclomatic_complexity, m.average_cyclomatic_complexity),
            (1.0, 1.0)
        );
        assert_eq!(
            (m.halstead_volume, m.halstead_difficulty, m.halstead_effort),
            (1.0, 1.0, 1.0)
        );
        assert_eq!(
            (m.lines_of_code, m.logical_lines_of_code, m.comments),
            (3, 2, 0)
        );
        assert!(close(m.maintainability_index, 145.09360748831728), "{m:?}");
        assert_eq!((m.max_nesting_depth, m.complexity_score), (0, 0.1));
    }

    #[test]
    fn go_uses_regex_complexity_analysis() {
        let m = analyze_code_metrics(GO_PROGRAM, "go");
        assert_eq!(
            (m.cyclomatic_complexity, m.average_cyclomatic_complexity),
            (3.0, 3.0)
        );
        assert!(close(m.halstead_volume, 15.84962500721156), "{m:?}");
        assert_eq!(
            (m.halstead_difficulty, m.halstead_effort),
            (1.0, m.halstead_volume)
        );
        assert_eq!(
            (m.lines_of_code, m.logical_lines_of_code, m.comments),
            (12, 10, 0)
        );
        assert!(close(m.maintainability_index, 91.50444811614277), "{m:?}");
        assert_eq!((m.max_nesting_depth, m.complexity_score), (3, 0.38));
        assert_eq!(analyze_code_metrics(GO_PROGRAM, "golang"), m);
    }

    #[test]
    fn julia_is_measured_by_line_count_not_as_python() {
        let julia = "function f(x)\n    if x > 0\n        return x\n    end\n    return -x\nend\n";
        let generic = analyze_code_metrics(julia, "julia");
        assert_eq!(
            (
                generic.cyclomatic_complexity,
                generic.lines_of_code,
                generic.max_nesting_depth
            ),
            (1.0, 6, 1)
        );
        assert_eq!(generic.maintainability_index, 100.0);
        assert_eq!(generic.complexity_score, 0.258);
        assert_ne!(analyze_code_metrics(julia, "python"), generic);
        assert_eq!(analyze_code_metrics("", "julia").complexity_score, 0.0);
    }

    #[test]
    fn python_classes_closures_and_branches() {
        let code = "\
class Solver:
    limit = 3 if FAST else 9

    def run(self, xs):
        def inner(x):
            return x and x > 0
        try:
            for x in xs:
                if inner(x):
                    continue
                elif x < 0 or x == 7:
                    break
            else:
                return None
        except ValueError:
            pass
        return [y for y in xs if y]

    def idle(self):
        pass


def top():
    assert True
";
        let m = analyze_code_metrics(code, "python");
        // run: 1 + for + if + elif + or + loop-else + except + comprehension for/if = 9
        //      (the function defined inside it is not counted); idle: 1; top: 1 + assert = 2.
        // The class: (1 + ternary + 9 + 1) / 2 methods, + 1 = 7.
        assert_eq!(m.cyclomatic_complexity, 9.0 + 1.0 + 2.0 + 7.0, "{m:?}");
        assert_eq!(m.average_cyclomatic_complexity, 19.0 / 4.0);
        // def run > try > for > if > elif
        assert_eq!(m.max_nesting_depth, 5);
        // radon 6.0.1 reports the same numbers for this program.
        assert_eq!(
            (m.lines_of_code, m.logical_lines_of_code, m.comments),
            (24, 20, 0)
        );
        assert_eq!(m.complexity_score, 1.0);
    }

    #[test]
    fn python_strings_comments_and_continuations() {
        let code = "# header\nx = \"\"\"a # not a comment\nstill (string\"\"\"  # trailing\ny = (1 +\n     2) \\\n    + 3; z = y if x else 0\nif y: z = 1\n";
        let m = analyze_code_metrics(code, "python");
        assert_eq!(m.comments, 2);
        assert_eq!(m.lines_of_code, 7);
        // x = ...; y = ...; z = ...; `if y:` and its inline body.
        assert_eq!(m.logical_lines_of_code, 5);
        assert_eq!(m.cyclomatic_complexity, 0.0);
        assert_eq!(m.max_nesting_depth, 1);
        // radon 6.0.1 reports the same volume and score.
        assert!(close(m.halstead_volume, 13.931568569324174), "{m:?}");
        assert_eq!(m.complexity_score, 0.206);
    }

    #[test]
    fn python_follows_radon_where_it_is_surprising() {
        // Checked against radon 6.0.1.
        let nested = "class Empty:\n    pass\n\nclass Outer:\n    limit = 3\n    class Inner:\n        def m(self):\n            if self: return 1\n            return 2\n    @property\n    def p(self):\n        return self.limit or 0\n";
        let m = analyze_code_metrics(nested, "python");
        assert_eq!(
            (m.cyclomatic_complexity, m.average_cyclomatic_complexity),
            (6.0, 2.0)
        );
        assert_eq!((m.logical_lines_of_code, m.max_nesting_depth), (12, 2));

        let signature = "\"\"\"Doc.\"\"\"\nimport os\n\ndef f(a, b=1 if os else 2):\n    \"\"\"Doc.\"\"\"\n    g = lambda q: q and q > 1\n    return [g(v) for v in a\n            if v]\n";
        let m = analyze_code_metrics(signature, "python");
        assert_eq!(
            (
                m.cyclomatic_complexity,
                m.lines_of_code,
                m.logical_lines_of_code
            ),
            (4.0, 8, 7)
        );

        let asynchronous = "async def run(xs):\n    async for x in xs:\n        async with lock:\n            await x\n    with open(a) as f:\n        assert x and y or z\n";
        let m = analyze_code_metrics(asynchronous, "python");
        assert_eq!((m.cyclomatic_complexity, m.max_nesting_depth), (3.0, 2));
        assert_eq!(
            statements_in(&tokenize_python("d = {1: 2}; e = d[1:2]; f = 3").unwrap().0[0].tokens),
            5
        );
    }

    #[test]
    fn text_that_cannot_be_python_falls_back() {
        for code in [
            "x = [1, 2",
            "x = 'unterminated\n",
            "x = (1]\n",
            "cost = $5\n",
            "x = 1\n        y = 2\n",
        ] {
            assert_eq!(
                analyze_code_metrics(code, "python"),
                brace_metrics(code),
                "{code:?}"
            );
        }
        let empty = analyze_code_metrics("", "python");
        assert_eq!(
            (
                empty.cyclomatic_complexity,
                empty.maintainability_index,
                empty.complexity_score
            ),
            (0.0, 171.0, 0.0)
        );
    }

    #[test]
    fn brace_analysis_counts_comments_and_ternaries() {
        let code = "// note\nint f(int a) {\n  /* block */\n  return a?1:0; // tail\n}\n";
        let m = analyze_code_metrics(code, "cpp");
        assert_eq!(
            (m.lines_of_code, m.logical_lines_of_code, m.comments),
            (6, 3, 2)
        );
        assert_eq!(m.cyclomatic_complexity, 2.0);
        assert_eq!(m.max_nesting_depth, 1);
        assert_eq!(analyze_code_metrics(code, "c++"), m);
        assert_eq!(analyze_code_metrics(code, "typescript"), m);
    }
}
