//! The SEARCH/REPLACE block grammar.
//!
//! ```text
//! <<<<<<< SEARCH
//! lines to find
//! =======
//! lines to put in their place
//! >>>>>>> REPLACE
//! ```
//!
//! ShinkaEvolve defines the grammar with one backtracking regular
//! expression:
//!
//! ```text
//! <{7}\s*SEARCH\s*\n
//! (.*?)(?:\n|^(?!(?:[^\S\n]*={7}[^\S\n]*\n){2}))
//! \s*={7}\s*\n
//! (.*?)(?:^|\n)\s*>{7}[^\S\n]*REPLACE[^\S\n]*$
//! ```
//!
//! It uses a lookahead, which the `regex` crate does not have, so this
//! module walks the same alternatives in the same order by hand. What that
//! order implies:
//!
//! * Either section may be empty, and blank lines at the start or end of a
//!   section are not part of it.
//! * The first `=======` line ends SEARCH, unless SEARCH opens with two of
//!   them: then the first is SEARCH text (a Markdown Setext underline).
//! * Only a line that is `>>>>>>> REPLACE` and nothing else ends REPLACE.

use crate::text::is_space;

/// One block, as written in the response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Block<'a> {
    pub search: &'a str,
    pub replace: &'a str,
}

struct Parser<'a> {
    text: &'a str,
    bytes: &'a [u8],
    /// No REPLACE marker closes a block from this offset on.
    no_closing_from: usize,
    /// No separator followed by a REPLACE marker exists from this offset on.
    no_separator_from: usize,
}

/// Every block in `text`, in order. Text between blocks is ignored.
pub(crate) fn parse_blocks(text: &str) -> Vec<Block<'_>> {
    let mut parser = Parser {
        text,
        bytes: text.as_bytes(),
        no_closing_from: usize::MAX,
        no_separator_from: usize::MAX,
    };
    let mut blocks = Vec::new();
    let mut cursor = 0;
    while let Some(offset) = text[cursor..].find("<<<<<<<") {
        let start = cursor + offset;
        match parser.block_at(start) {
            Some((block, end)) => {
                blocks.push(block);
                cursor = end;
            }
            None => cursor = start + 1,
        }
    }
    blocks
}

impl<'a> Parser<'a> {
    fn skip(&self, from: usize, keep_going: impl Fn(char) -> bool) -> usize {
        let mut end = from;
        for c in self.text[from..].chars() {
            if !keep_going(c) {
                break;
            }
            end += c.len_utf8();
        }
        end
    }

    /// `\s*`
    fn skip_whitespace(&self, from: usize) -> usize {
        self.skip(from, is_space)
    }

    /// `[^\S\n]*`
    fn skip_inline_whitespace(&self, from: usize) -> usize {
        self.skip(from, |c| c != '\n' && is_space(c))
    }

    /// Offsets just past each newline in `from..to`, last first: the order a
    /// greedy `\s*\n` tries them in.
    fn line_starts_within(&self, from: usize, to: usize) -> impl Iterator<Item = usize> + '_ {
        (from..to)
            .rev()
            .filter(|index| self.bytes[*index] == b'\n')
            .map(|index| index + 1)
    }

    fn next_newline(&self, from: usize) -> Option<usize> {
        self.bytes[from..]
            .iter()
            .position(|byte| *byte == b'\n')
            .map(|offset| from + offset)
    }

    /// `<{7}\s*SEARCH\s*\n` at `start`, then the rest of the block.
    fn block_at(&mut self, start: usize) -> Option<(Block<'a>, usize)> {
        let keyword = self.skip_whitespace(start + 7);
        if !self.text[keyword..].starts_with("SEARCH") {
            return None;
        }
        let after_keyword = keyword + 6;
        let whitespace_end = self.skip_whitespace(after_keyword);
        let candidates: Vec<usize> = self
            .line_starts_within(after_keyword, whitespace_end)
            .collect();
        candidates
            .into_iter()
            .find_map(|search_start| self.sections_from(search_start))
    }

    /// The lazy SEARCH group: end it at the first place a separator and a
    /// closing marker follow.
    fn sections_from(&mut self, search_start: usize) -> Option<(Block<'a>, usize)> {
        let len = self.text.len();
        let mut line_start = search_start;
        while line_start < len && line_start < self.no_separator_from {
            let line_end = self.next_newline(line_start).unwrap_or(len);
            // An empty SEARCH: the separator opens the section, unless two
            // separator lines start it. Later line starts are reached through
            // the newline before them, below.
            if line_start == search_start
                && line_end > line_start
                && !self.two_separator_lines_at(line_start)
            {
                if let Some(found) = self.separator_at(line_start) {
                    return Some(self.block(search_start, line_start, found));
                }
            }
            // After the line's newline.
            if line_end < len {
                if let Some(found) = self.separator_at(line_end + 1) {
                    return Some(self.block(search_start, line_end, found));
                }
            }
            line_start = line_end + 1;
        }
        self.no_separator_from = self.no_separator_from.min(search_start);
        None
    }

    fn block(
        &self,
        search_start: usize,
        search_end: usize,
        (replace_start, replace_end, end): (usize, usize, usize),
    ) -> (Block<'a>, usize) {
        let block = Block {
            search: &self.text[search_start..search_end],
            replace: &self.text[replace_start..replace_end],
        };
        (block, end)
    }

    /// `[^\S\n]*={7}[^\S\n]*\n`, returning the offset after the newline.
    fn separator_line_at(&self, from: usize) -> Option<usize> {
        let equals = self.skip_inline_whitespace(from);
        if !self.text[equals..].starts_with("=======") {
            return None;
        }
        let end = self.skip_inline_whitespace(equals + 7);
        (self.bytes.get(end) == Some(&b'\n')).then_some(end + 1)
    }

    fn two_separator_lines_at(&self, from: usize) -> bool {
        self.separator_line_at(from)
            .and_then(|second| self.separator_line_at(second))
            .is_some()
    }

    /// `\s*={7}\s*\n`, the lazy REPLACE group and the closing marker.
    /// Returns where REPLACE starts and ends and where the block ends.
    fn separator_at(&mut self, from: usize) -> Option<(usize, usize, usize)> {
        let equals = self.skip_whitespace(from);
        if !self.text[equals..].starts_with("=======") {
            return None;
        }
        let after_equals = equals + 7;
        let whitespace_end = self.skip_whitespace(after_equals);
        let candidates: Vec<usize> = self
            .line_starts_within(after_equals, whitespace_end)
            .collect();
        candidates.into_iter().find_map(|replace_start| {
            self.closing_from(replace_start)
                .map(|(replace_end, end)| (replace_start, replace_end, end))
        })
    }

    /// The lazy REPLACE group: end it at the first line start where the
    /// closing marker matches. Returns where REPLACE ends and the block ends.
    fn closing_from(&mut self, replace_start: usize) -> Option<(usize, usize)> {
        if replace_start < self.no_closing_from {
            if let Some(end) = self.closing_at(replace_start) {
                return Some((replace_start, end));
            }
            let mut from = replace_start;
            while let Some(newline) = self.next_newline(from) {
                if newline + 1 >= self.no_closing_from {
                    break;
                }
                if let Some(end) = self.closing_at(newline + 1) {
                    return Some((newline, end));
                }
                from = newline + 1;
            }
        }
        self.no_closing_from = self.no_closing_from.min(replace_start);
        None
    }

    /// `\s*>{7}[^\S\n]*REPLACE[^\S\n]*$`, returning the end of the match.
    fn closing_at(&self, from: usize) -> Option<usize> {
        let marker = self.skip_whitespace(from);
        if !self.text[marker..].starts_with(">>>>>>>") {
            return None;
        }
        let keyword = self.skip_inline_whitespace(marker + 7);
        if !self.text[keyword..].starts_with("REPLACE") {
            return None;
        }
        let end = self.skip_inline_whitespace(keyword + 7);
        (end == self.text.len() || self.bytes[end] == b'\n').then_some(end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(text: &str) -> Vec<(&str, &str)> {
        parse_blocks(text)
            .into_iter()
            .map(|block| (block.search, block.replace))
            .collect()
    }

    #[test]
    fn a_plain_block_and_text_around_it() {
        let text = "Here you go:\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\nDone.";
        assert_eq!(pairs(text), [("x = 1", "x = 2")]);
    }

    #[test]
    fn several_blocks_keep_their_order() {
        let text = "<<<<<<< SEARCH\na\nb\n=======\nc\n>>>>>>> REPLACE\n\n<<<<<<< SEARCH\nd\n=======\ne\nf\n>>>>>>> REPLACE\n";
        assert_eq!(pairs(text), [("a\nb", "c"), ("d", "e\nf")]);
    }

    #[test]
    fn empty_sections_with_and_without_a_blank_line() {
        for gap in ["", "\n"] {
            let delete = format!("<<<<<<< SEARCH\ndebug = True\n=======\n{gap}>>>>>>> REPLACE");
            assert_eq!(pairs(&delete), [("debug = True", "")]);
            let insert = format!("<<<<<<< SEARCH\n{gap}=======\nlast = 9\n>>>>>>> REPLACE");
            assert_eq!(pairs(&insert), [("", "last = 9")]);
        }
        assert_eq!(
            pairs("<<<<<<< SEARCH\n=======\n>>>>>>> REPLACE"),
            [("", "")]
        );
    }

    #[test]
    fn indentation_survives_but_surrounding_blank_lines_do_not() {
        let text = "<<<<<<< SEARCH\n\n    x = 1\n\n=======\n\n    y = 2\n\n>>>>>>> REPLACE";
        assert_eq!(pairs(text), [("    x = 1", "    y = 2")]);
    }

    #[test]
    fn a_setext_underline_can_be_searched_for() {
        for indent in ["", "  "] {
            let text = format!(
                "<<<<<<< SEARCH\n{indent}=======\n=======\n{indent}-------\n>>>>>>> REPLACE"
            );
            let blocks = parse_blocks(&text);
            assert_eq!(blocks[0].search, format!("{indent}======="));
            assert_eq!(blocks[0].replace, format!("{indent}-------"));
        }
        // Three separator lines: the first is SEARCH text, the second ends it.
        assert_eq!(
            pairs("<<<<<<< SEARCH\n=======\n=======\n=======\nx\n>>>>>>> REPLACE"),
            [("=======", "=======\nx")]
        );
    }

    #[test]
    fn only_a_bare_replace_marker_closes_a_block() {
        for quote in [
            ">>>>>>> REPLACEMENT",
            ">>>>>>> REPLACE this",
            ">>>>>>>> REPLACE",
        ] {
            let text =
                format!("<<<<<<< SEARCH\nOld text\n=======\n{quote}\nMore text\n>>>>>>> REPLACE");
            assert_eq!(
                parse_blocks(&text)[0].replace,
                format!("{quote}\nMore text")
            );
        }
        assert_eq!(
            pairs("<<<<<<< SEARCH\na\n=======\nb\n  >>>>>>>\tREPLACE \t\nrest"),
            [("a", "b")]
        );
    }

    #[test]
    fn incomplete_blocks_are_not_blocks() {
        assert!(pairs("<<<<<<< SEARCH\na\n=======\nb\n").is_empty());
        assert!(pairs("<<<<<<< SEARCH\na\n>>>>>>> REPLACE").is_empty());
        assert!(pairs("<<<<<<< SEARCH a\n=======\nb\n>>>>>>> REPLACE").is_empty());
        assert!(pairs("<<<<<<< SEARCH\na\n========\nb\n>>>>>>> REPLACE").is_empty());
        assert!(pairs("").is_empty());
    }

    #[test]
    fn an_unfinished_block_does_not_swallow_the_next_one() {
        let text = "<<<<<<< SEARCH\nlost\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE";
        assert_eq!(pairs(text), [("lost\n<<<<<<< SEARCH\na", "b")]);
        let text = "<<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE";
        assert_eq!(pairs(text), [("a", "b")]);
    }

    #[test]
    fn many_unterminated_openers_parse_in_linear_time() {
        let text = "<<<<<<< SEARCH\nx\n=======\ny\n".repeat(20_000);
        let started = std::time::Instant::now();
        assert!(parse_blocks(&text).is_empty());
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }
}
